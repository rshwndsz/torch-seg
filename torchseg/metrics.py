# Python STL
from datetime import datetime
import time
# Data Science
import numpy as np
# PyTorch
import torch

# Local
from torchseg import utils

# TODO: Generalize to multiclass segmentation
# TODO: Add tests to test integrity


def dice_score(probs, targets, threshold=0.5):
    """Calculate Sorenson-Dice coefficient

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities
    targets : torch.Tensor
        Ground truths
    threshold : float
        probs > threshold => 1
        probs <= threshold => 0

    Returns
    -------
    dice : float
        Dice score

    See Also
    --------
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    batch_size = targets.shape[0]
    with torch.no_grad():
        # Shape: [N, C, H, W]targets
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        # Shape: [N, C*H*W]
        if not (probs.shape == targets.shape):
            raise ValueError("Shape of probs: {} must be the same as that of targets: {}."
                             .format(probs.shape, targets.shape))
        # Only 1's and 0's in p & t
        p = utils.predict(probs, threshold)
        t = utils.predict(targets, 0.5)
        # Shape: [N, 1]
        dice = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

    return utils.nanmean(dice).item()


def true_positive(preds, targets, num_classes=2):
    """Compute number of true positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tp : torch.Tensor
        Tensor of number of true positives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets == i)).sum())

    return torch.tensor(out)


def true_negative(preds, targets, num_classes):
    """Computes number of true negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    tn : torch.Tensor
        Tensor of true negatives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets != i)).sum())

    return torch.tensor(out)


def false_positive(preds, targets, num_classes):
    """Computes number of false positive predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fp : torch.Tensor
        Tensor of false positives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((preds == i) & (targets != i)).sum())

    return torch.tensor(out)


def false_negative(preds, targets, num_classes):
    """Computes number of false negative predictions

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    fn : torch.Tensor
        Tensor of false negatives for each class
    """
    out = []
    for i in range(num_classes):
        out.append(((preds != i) & (targets == i)).sum())

    return torch.tensor(out)


def precision_score(preds, targets, num_classes):
    """Computes precision score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)

    Returns
    -------
    precision : (float, float)
        Precision score for class 2
    """
    tp = true_positive(preds, targets, num_classes).to(torch.float)
    fp = false_positive(preds, targets, num_classes).to(torch.float)
    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out[1].item()  # <<< Change: Hardcoded for binary segmentation


def accuracy_score(preds, targets):
    """Compute accuracy score

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths

    Returns
    -------
    acc : float
        Average accuracy score
    """
    valids = (targets >= 0)
    acc_sum = (valids * (preds == targets)).sum().item()
    valid_sum = valids.sum().item()
    return float(acc_sum) / (valid_sum + 1e-10)    # <<< Change: Hardcoded smoothing


def iou_score(preds, targets, num_classes):
    """Computes IoU or Jaccard index

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    num_classes : int
        Number of classes (including background)
    Returns
    -------
    iou : float
        IoU score or Jaccard index
    """
    intersection = torch.sum(targets * preds)
    union = torch.sum(targets) + torch.sum(preds) - intersection + 1e-7
    score = (intersection + 1e-7) / union

    return score.item()


class Meter:
    """A meter to keep track of losses and scores"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.metrics = {
            'dice': [],
            'iou': [],
            'acc': [],
            'prec': []
        }

    def update(self, targets, logits):
        """Calculates metrics for each batch and updates meter

        Parameters
        ----------
        targets : torch.FloatTensor
            [N C H W]
            Ground truths
        logits : torch.FloatTensor
            [N C H W]
            Raw logits
        """
        probs = torch.sigmoid(logits)
        preds = utils.predict(probs, self.base_threshold)

        # Assertion for shapes
        if not (preds.shape == targets.shape):
            raise ValueError("Shape of preds: {} must be the same as that of targets: {}."
                             .format(preds.shape, targets.shape))

        # TODO: Automate
        dice = dice_score(probs, targets, self.base_threshold)
        self.metrics['dice'].append(dice)

        iou = iou_score(preds, targets, num_classes=2)  # <<< TODO: Remove hardcoded num_classes
        self.metrics['iou'].append(iou)

        acc = accuracy_score(preds, targets)
        self.metrics['acc'].append(acc)

        prec = precision_score(preds, targets, num_classes=2)
        self.metrics['prec'].append(prec)

    def get_metrics(self):
        """Compute mean of batchwise metrics

        Returns
        -------
        self.metrics : dict[str, float]
            Mean of all metrics as a dictionary
        """
        self.metrics.update({key: np.nanmean(self.metrics[key])
                             for key in self.metrics.keys()})
        return self.metrics

    @staticmethod
    def epoch_log(phase, epoch, epoch_loss, meter, start_time, fmt):
        """Logs and returns metrics

        Parameters
        ----------
        phase : str
            Phase of training
        epoch : int
            Current epoch number
        epoch_loss : float
            Current average epoch loss
        meter : Meter
            Meter object holding metrics for current epoch
        start_time : str
            Start time as a string
        fmt : str
            Formatting applied to `start_time`

        Returns
        -------
        metrics : dict[str, float]
            Dictionary of metrics
        """
        metrics = meter.get_metrics()
        end_time = time.strftime(fmt, time.localtime())
        delta_t = (datetime.strptime(end_time, fmt) - datetime.strptime(start_time, fmt))

        # TODO: Automate
        print(f"Loss: {epoch_loss:.4f} | dice: {metrics['dice']:.4f} | "
              f"IoU: {metrics['iou']:4f} | Acc: {metrics['acc']:4f} | "
              f"Prec: {metrics['prec']:4f} in {delta_t}")

        return metrics
