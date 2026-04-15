import os
import sys

import numpy as np

from utils.logger import log_training_metrics_console

sys.path.insert(0, '../')
import torch
import torch.nn as nn
import time
from utils.helper import AverageMeter, mAP
from utils.validations import validate, validate_zsl
from utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3, UncertaintyAwareAsymmetricLoss
from torch.cuda.amp import autocast


def train_classic_fc(data_loader, val_loader, model, optim, sched, scaler, args, cfg, epoch):
    """Train with classic fully-connected classifier.

    Args:
        data_loader: Training data loader
        val_loader: Validation data loader
        model: Model to train
        optim: Optimizer
        sched: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        args: Command line arguments
        cfg: Configuration
        epoch: Current epoch number

    Returns:
        Tuple of (batch_time, losses, mAP_batches)
    """
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)

    # Set model to evaluation mode, but enable training for classifier layer
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.fc.train()  # Train only the final classifier
        if cfg.TRAINER.FINETUNE:
            model.train()  # Also finetune the entire model
    else:
        model.module.fc.train()
        if cfg.TRAINER.FINETUNE:
            model.train()

    criterion = AsymmetricLoss(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)

    end = time.time()
    for i, (images, target) in enumerate(data_loader):
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)

        # Compute forward pass with mixed precision
        with autocast():
            output = model(images)
        loss = args.loss_w * criterion(output, target)

        # Backward pass with gradient scaling
        model.zero_grad()
        scaler.scale(loss).backward()

        # Update parameters and learning rate
        scaler.step(optim)
        scaler.update()
        sched.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        pred = Softmax(output.detach())[:, 1, :]  # Extract positive class probabilities
        mAP_value = mAP(target.cpu().numpy(), pred.cpu().numpy())
        mAP_batches.update(mAP_value, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Print training progress
        if i % args.print_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(
                i, len(data_loader), batch_time=batch_time,
                losses=losses, mAP_batches=mAP_batches), flush=True)

        # Perform validation during training if specified
        if args.val_freq_in_epoch != -1 and (i + 1) % args.val_freq_in_epoch == 0:
            p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loader, model, args)
            print('Test: [{}/{}]\t '
                  ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                  .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)

    return batch_time, losses, mAP_batches


def train_coop(data_loader, val_loaders, model, optim, sched, args, cfg, epoch, cls_id=None):
    """Uncertainty-aware CoOp training with uncertainty weighting and teacher memory.

    This training function combines:
    - Positive CoOp training logic
    - Support for external optimizer/scheduler (AdamW, RAdam, SGD, etc.)
    - Gradient clipping for training stability
    - Uncertainty-aware loss with teacher-student memory

    Args:
        data_loader: Training data loader
        val_loaders: List of validation data loaders
        model: Model to train
        optim: Optimizer
        sched: Learning rate scheduler
        args: Command line arguments
        cfg: Configuration
        epoch: Current epoch number
        cls_id: Class ID mapping for ZSL scenarios

    Returns:
        Tuple of (batch_time, losses, mAP_batches, grad_mean, grad_max)
    """

    # Initialize performance meters
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    # Determine number of training classes
    if cls_id is not None:
        num_train_cls = len(cls_id['train'])
    else:
        # Use total dataset classes
        num_train_cls = cfg.DATASET.NUM_CLASSES

    # Set model mode: evaluation for base, training for specific components
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.prompt_learner.train()  # Train prompt learner
        if cfg.TRAINER.FINETUNE_ATTN:
            model.image_encoder.attnpool.train()  # Finetune attention pooling

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.image_encoder.train()  # Finetune image encoder
    else:
        model.module.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.module.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.module.image_encoder.train()

    # Define loss functions
    criterion = AsymmetricLoss(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)
    criterion2 = AsymmetricLoss2(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)
    criterion3 = AsymmetricLoss3(cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG, cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS)

    # Uncertainty-aware loss
    criterion_ua = UncertaintyAwareAsymmetricLoss(
        cfg.TRAINER.COOP_MLC.ASL_GAMMA_NEG,
        cfg.TRAINER.COOP_MLC.ASL_GAMMA_POS
    )

    # AMP support (commented out)
    # scaler = torch.cuda.amp.GradScaler(enabled=(hasattr(cfg.TRAIN, "AMP") and cfg.TRAIN.AMP))

    # ========== Initialize Global Teacher Memory ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize teacher memory, uncertainty memory, and count memory as model attributes
    # This allows them to persist across epochs
    if not hasattr(model, "teacher_memory"):
        model.teacher_memory = torch.zeros(num_train_cls, device=device)
        model.uncert_memory = torch.zeros(num_train_cls, device=device)
        model.count_memory = torch.zeros(num_train_cls, device=device)

    teacher_memory = model.teacher_memory
    uncert_global = model.uncert_memory
    count_memory = model.count_memory
    # ==============================================

    end = time.time()

    # Train
    for i, (images, target) in enumerate(data_loader):
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)
        # Random class sampling (ZSL compatibility)
        if cls_id is not None:
            if num_train_cls > args.num_train_cls:
                batch_cls_id = torch.randperm(num_train_cls).cpu().tolist()[:args.num_train_cls]
                batch_cls_id_input = [cls_id['train'][a] for a in batch_cls_id]
            else:
                batch_cls_id_input = cls_id['train']
        else:
            batch_cls_id_input = None

        # ================= First Forward Pass (Get Pseudo Labels) =================
        with torch.no_grad():
            output_pre = model(images, batch_cls_id_input)
            pseudo_pred = torch.sigmoid(output_pre.detach())[:, 1, :]  # [B, num_classes]

            # === Uncertainty Estimation ===
            # Calculate entropy-based uncertainty
            entropy = - (pseudo_pred * torch.log(pseudo_pred + 1e-8) +
                         (1 - pseudo_pred) * torch.log(1 - pseudo_pred + 1e-8))
            entropy = entropy / torch.log(torch.tensor(2.0, device=device))  # Normalize to [0,1]

            # Calculate confidence-based uncertainty
            conf_uncert = 4 * pseudo_pred * (1 - pseudo_pred)

            # Combine uncertainties with weighting factor
            λ = 0.5
            u_weight = λ * entropy + (1 - λ) * conf_uncert

            # === Teacher Memory Update ===
            batch_mean_pred = pseudo_pred.mean(dim=0)
            uncert_batch_mean = u_weight.mean(dim=0)
            active_classes = (target.sum(dim=0) > 0).nonzero(as_tuple=True)[0]

            rho = 0.8  # Momentum factor
            for c in active_classes:
                # Update global uncertainty with momentum
                uncert_global[c] = rho * uncert_global[c] + (1 - rho) * uncert_batch_mean[c]

                # Calculate teacher weight based on uncertainty ratio
                w_teacher = uncert_batch_mean[c] / (uncert_global[c] + uncert_batch_mean[c] + 1e-8)
                w_teacher = w_teacher.clamp(0.0, 1.0)

                # Update teacher memory with weighted combination
                teacher_memory[c] = (1 - w_teacher) * batch_mean_pred[c] + w_teacher * teacher_memory[c]
                count_memory[c] += 1

        # Expand teacher vector for current batch
        pseudo_teacher = teacher_memory.unsqueeze(0).expand_as(pseudo_pred)

        # Second Forward Pass (with uncertainty fusion)
        with autocast():
            output = model(images, batch_cls_id_input, uncertainty_weight=u_weight)
        # =======================

        if cls_id is not None:
            # output = output[:, :, cls_id['train']]
            # target = target[:, cls_id['train']]
            target = target[:, batch_cls_id_input]
        if output.dim() == 3:
            final_loss, asl_loss, kl_loss = criterion_ua(output, target, u_weight, pseudo_teacher=pseudo_teacher)
            loss = args.loss_w * final_loss
        elif args.single_prompt == 'pos':
            loss = args.loss_w * criterion2(output, target)
        elif args.single_prompt == 'neg':
            loss = args.loss_w * criterion3(output, target)
        else:
            raise ValueError

        # Backward + update the network
        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        grad_norms_after = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms_after.append(grad_norm)

        if grad_norms_after and i % args.print_freq == 0:
            grad_mean_after = np.mean(grad_norms_after)
            grad_max_after = np.max(grad_norms_after)
            print(f"[DEBUG] - mean: {grad_mean_after:.4f}, max: {grad_max_after:.4f}")

        optim.step()

        losses.update(loss.item(), images.size(0))
        if output.dim() == 3:
            pred = Softmax(output.detach())[:, 1, :]
        else:
            pred = Sig(output.detach())
        mAP_value = mAP(target.cpu().numpy(), pred.cpu().numpy())
        mAP_batches.update(mAP_value, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Debug
        if i % args.print_freq == 0:
            mean_u = float(u_weight.mean().item()) if 'u_weight' in locals() else 0
            std_u = float(u_weight.std().item()) if 'u_weight' in locals() else 0
            mean_entropy = float(entropy.mean().item()) if 'entropy' in locals() else 0
            max_entropy = float(entropy.max().item()) if 'entropy' in locals() else 0
            asl_loss_val = asl_loss.item() if 'asl_loss' in locals() else 0.0
            kl_loss_val = kl_loss.item() if 'kl_loss' in locals() else 0.0

            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})\t'
                  'U_mean {mean_u:.3f}({mean_u:.3f})\tU_std {std_u:.3f}({std_u:.3f})\t'
                  'Entropy {mean_entropy:.3f}/{max_entropy:.3f}\t'.format(
                i, len(data_loader),
                batch_time=batch_time, losses=losses, mAP_batches=mAP_batches,
                mean_u=mean_u, std_u=std_u, mean_entropy=mean_entropy, max_entropy=max_entropy),
                flush=True)

        if args.val_freq_in_epoch != -1 and (i + 1) % args.val_freq_in_epoch == 0:
            if len(val_loaders) == 1:
                p_c, r_c, f_c, p_o, r_o, f_o, mAP_score = validate(val_loaders[0], model, args)
                print('Test: [{}/{}]\t '
                      ' P_C {:.2f} \t R_C {:.2f} \t F_C {:.2f} \t P_O {:.2f} \t R_O {:.2f} \t F_O {:.2f} \t mAP {:.2f}'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_c, r_c, f_c, p_o, r_o, f_o, mAP_score), flush=True)
            elif len(val_loaders) == 2:
                p_unseen, r_unseen, f1_unseen, mAP_unseen = validate_zsl(val_loaders[0], model, args,
                                                                         cls_id['val_unseen'])
                p_gzsl, r_gzsl, f1_gzsl, mAP_gzsl = validate_zsl(val_loaders[1], model, args, cls_id['val_gzsi'])
                print('Test: [{}/{}]\t '
                      ' P_unseen {:.2f} \t R_unseen {:.2f} \t F1_unseen {:.2f} \t mAP_unseen {:.2f}\t'
                      ' P_gzsl {:.2f} \t R_gzsl {:.2f} \t F1_gzsl {:.2f} \t mAP_gzsl {:.2f}\t'
                      .format(epoch + 1, cfg.OPTIM.MAX_EPOCH, p_unseen, r_unseen, f1_unseen, mAP_unseen, p_gzsl, r_gzsl,
                              f1_gzsl, mAP_gzsl), flush=True)
            else:
                raise ValueError

    sched.step()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.data.norm(2).item())

    grad_mean = float(torch.tensor(grad_norms).mean().item()) if len(grad_norms) > 0 else 0.0
    grad_max = float(torch.tensor(grad_norms).max().item()) if len(grad_norms) > 0 else 0.0

    return batch_time, losses, mAP_batches, grad_mean, grad_max
