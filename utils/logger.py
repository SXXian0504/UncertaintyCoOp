import numpy as np
import torch
import torch.nn.functional as F
import os


def compute_ece(preds, targets, n_bins=10):
    """Expected Calibration Error (ECE)"""
    preds = torch.sigmoid(preds.detach())
    confs = preds.flatten()
    labels = targets.flatten()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=preds.device)
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.any():
            acc = labels[mask].float().mean()
            conf = confs[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)
    return ece.item()


def log_training_metrics_console(epoch, step, u_weight, loss_asl, loss_kl, pseudo_pred, target, log_path):
    mean_u = float(u_weight.mean().item()) if u_weight is not None else 0
    std_u  = float(u_weight.std().item()) if u_weight is not None else 0
    mean_entropy = float((- (pseudo_pred * torch.log(pseudo_pred + 1e-8)
                             + (1 - pseudo_pred) * torch.log(1 - pseudo_pred + 1e-8))
                          ).mean().item()) if pseudo_pred is not None else 0
    kl_ratio = loss_kl.item() / (loss_asl.item() + 1e-8) if loss_asl is not None and loss_kl is not None else 0

    print(f"[Epoch {epoch:03d} | Step {step:04d}] "
          f"U_mean={mean_u:.3f} U_std={std_u:.3f} Entropy={mean_entropy:.3f} "
          f"KL/ASL={kl_ratio:.3f}")

    with open(log_path, "a") as f:
        f.write(f"{epoch},{step},{mean_u:.4f},{std_u:.4f},{mean_entropy:.4f},{kl_ratio:.4f}\n")


# === Parameter Counting ===
def count_parameters(model,optim, only_trainable=True):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n>>> Total trainable parameters: {total/1e6:.3f}M")
    print(f">>> Frozen parameters: {frozen/1e6:.3f}M")
    print(f">>> Optimizer parameters:", sum(p.numel() for g in optim.param_groups for p in g['params']))

    if hasattr(model, 'prompt_learner'):
        prompt_params = sum(p.numel() for p in model.prompt_learner.parameters() if p.requires_grad)
        print(f">>> Prompt learner parameters: {prompt_params/1e6:.3f}M")

    if hasattr(model, 'txt_prompt_uncertain'):
        print(f">>> Uncertainty prompt vector: {model.txt_prompt_uncertain.numel()/1e6:.3f}M")

    print("============================================\n")
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
            print(f"[TRAINABLE] {name:60s} {tuple(p.shape)}")
        else:
            frozen_params += n

    print(f"Total parameters: {total_params / 1e6:.3f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f}M")
    print(f"Frozen parameters: {frozen_params / 1e6:.3f}M")
    print("============================================\n")
