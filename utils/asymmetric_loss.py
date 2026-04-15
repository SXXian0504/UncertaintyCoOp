import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    This loss function combines asymmetric clipping and asymmetric focusing
    to handle class imbalance in multi-label scenarios.

    Reference:
        Ridnik, T., et al. "Asymmetric loss for multi-label classification."
        Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        Compute asymmetric loss.

        Parameters
        ----------
        x: input logits of shape [B, 2, num_classes] or [B, 2]
        y: targets (multi-label binarized vector)

        Returns
        -------
        loss: scalar tensor
        """

        # Calculate probabilities using softmax
        x_softmax = self.softmax(x)
        xs_pos = x_softmax[:, 1, :]  # Positive class probabilities
        xs_neg = x_softmax[:, 0, :]  # Negative class probabilities

        # Flatten tensors
        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        # Filter out ignored targets (y == -1)
        xs_pos = xs_pos[y != -1]
        xs_neg = xs_neg[y != -1]
        y = y[y != -1]

        # Asymmetric Clipping - reduce false positives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross-Entropy calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing - focal loss component
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLoss2(nn.Module):
    """Asymmetric Loss variant using sigmoid instead of softmax.

    This version uses sigmoid activation for binary classification.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss2, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        Compute asymmetric loss with sigmoid activation.

        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)

        Returns
        -------
        loss: scalar tensor
        """

        # Calculate probabilities using sigmoid
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping - reduce false positives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross-Entropy calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing - focal loss component
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLoss3(nn.Module):
    """Asymmetric Loss variant with inverted sigmoid interpretation.

    This version inverts the interpretation of positive and negative classes.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss3, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        Compute asymmetric loss with inverted sigmoid interpretation.

        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)

        Returns
        -------
        loss: scalar tensor
        """

        # Calculate probabilities using sigmoid with inverted interpretation
        x_sigmoid = torch.sigmoid(x)
        xs_neg = x_sigmoid  # Treat sigmoid output as negative class
        xs_pos = 1 - x_sigmoid  # Treat 1-sigmoid as positive class

        # Asymmetric Clipping - reduce false positives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross-Entropy calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing - focal loss component
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """Optimized Asymmetric Loss implementation.

    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """
        Compute optimized asymmetric loss.

        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)

        Returns
        -------
        loss: scalar tensor
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculate probabilities using sigmoid
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping - reduce false positives
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic Cross-Entropy calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing - focal loss component
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class UncertaintyAwareAsymmetricLoss(nn.Module):
    """Uncertainty-Aware Asymmetric Loss.

    Combines Asymmetric Loss with KL divergence regularization based on uncertainty weights.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6):
        super(UncertaintyAwareAsymmetricLoss, self).__init__()
        self.asl = AsymmetricLoss(gamma_neg, gamma_pos, clip, eps)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, target, u_weight, pseudo_teacher):
        """
        Compute uncertainty-aware asymmetric loss.

        Parameters
        ----------
        pred: [B, 2, num_classes] - model predictions
        target: [B, num_classes] - ground truth targets
        u_weight: [B, num_classes] - uncertainty weights
        pseudo_teacher: [B, num_classes] - teacher model predictions

        Returns
        -------
        final_loss: scalar tensor
        asl_loss: ASL component
        kl_loss: KL divergence component
        """
        # === ASL ===
        asl_loss = self.asl(pred, target)

        # === KL ===
        # Current prediction distribution
        pred_prob = self.softmax(pred)[:, 1, :]  # [B, num_classes]
        # Teacher distribution
        pseudo_prob = pseudo_teacher.clamp(min=1e-8, max=1 - 1e-8)
        # KL divergence loss
        kl_loss = self.kl(
            F.log_softmax(pred_prob, dim=-1),
            F.softmax(pseudo_prob, dim=-1)
        )

        # === Fuse with uncertainty weights ===
        # Control KL loss weight range
        u_mean = u_weight.mean()

        # Smooth upper bound mapping: near linear growth when u_mean < limit,
        # slowly approaches limit when u_mean > limit
        limit = 0.3  # Upper bound
        k = 15.0     # Smooth control coefficient (larger = closer to hard upper bound)
        s = torch.sigmoid(k * (u_mean - limit))
        w_kl = (1 - s) * u_mean + s * limit

        final_loss = (1 - w_kl) * asl_loss + w_kl * kl_loss

        # Debug logging
        if torch.rand(1) < 0.01:
            print(
                f"[DEBUG] ASL: {asl_loss.item():.4f}, KL: {kl_loss.item():.4f}, "
                f"U_mean: {u_mean.item():.4f}, w_kl: {w_kl.item():.4f}")

        return final_loss, asl_loss, kl_loss