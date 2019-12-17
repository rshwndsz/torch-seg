# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_loss(logits, target):
    # See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    probs = torch.sigmoid(logits)
    smooth = 1e-7  # <<<<<<<<<<<<< Note: Hardcoded smoothing
    iflat = probs.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    # See: https://arxiv.org/abs/1708.02002
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))
        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()
        invprobs = F.logsigmoid(-logits * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


# Custom loss function combining Focal loss and Dice loss
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, logits, target):
        loss = self.alpha * self.focal(logits, target) - torch.log(dice_loss(logits, target))
        return loss.mean()
