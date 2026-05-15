from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    def __init__(self, reconstruction_weight: float, label_smoothing: float):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

    @staticmethod
    def ssim_loss(pred: torch.Tensor, target: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
        mu_p = F.avg_pool2d(pred, 11, 1, 5)
        mu_t = F.avg_pool2d(target, 11, 1, 5)
        sig_p = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_p * mu_p
        sig_t = F.avg_pool2d(target * target, 11, 1, 5) - mu_t * mu_t
        sig_pt = F.avg_pool2d(pred * target, 11, 1, 5) - mu_p * mu_t
        ssim = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / ((mu_p * mu_p + mu_t * mu_t + c1) * (sig_p + sig_t + c2))
        return 1.0 - ssim.mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, recon: torch.Tensor | None = None, clean: torch.Tensor | None = None):
        cls_loss = self.ce(logits, labels)
        rec_loss = torch.zeros((), device=logits.device)
        if self.reconstruction_weight > 0 and recon is not None and clean is not None:
            rec_loss = 0.5 * self.mse(recon, clean) + 0.5 * self.ssim_loss(recon, clean)
        loss = cls_loss + self.reconstruction_weight * rec_loss
        return loss, {"loss_cls": float(cls_loss.detach().cpu()), "loss_rec": float(rec_loss.detach().cpu())}
