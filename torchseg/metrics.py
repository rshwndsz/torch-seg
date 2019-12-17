# Python STL
from datetime import datetime
# Data Science
import numpy as np
import sklearn.metrics as skm
# PyTorch
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add


# TODO: Move into utils
# TODO: Test new nanmean for tensors
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


# TODO: Add tests to test integrity
def dice_coeff(probs, targets, threshold=0.5):
    """
    Calculate Sorenson-Dice coefficient
    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    :param probs: Predicted outputs in probabilites [0..1]
    :param targets: Ground truth [0 or 1]
    :param threshold: Threshold to convert probabilities to binary
    :return: Sorenson-Dice coefficient
    """

    batch_size = len(targets)
    with torch.no_grad():
        # Shape: [N, C, H, W]targets
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        # Shape: [N, C*H*W]
        if not (probs.shape == targets.shape):
            raise ValueError("Shape of probs: {} must be the same as that of targets: {}."
                             .format(probs.shape, targets.shape))
        # Only 1's and 0's in p & t
        p = (probs > threshold).float()
        t = (targets > 0.5).float()

        # Shape: [N, 1]
        dice = 2 * (p * t).sum(-1) / ((p + t).sum(-1))
    return dice


# TODO: Add tests to test integrity
def accuracy_score(preds, targets):
    """
    Calculate accuracy of predictions

    # Credits: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
    :param preds: Predicted outputs [0..C-1]
    :param targets: Ground truth [0..C-1]
    :return: Accuracy
    """
    valids = (targets >= 0)
    acc_sum = (valids * (preds == targets)).sum().item()
    valid_sum = valids.sum()
    return float(acc_sum) / (valid_sum + 1e-10)    # <<< Change: Hardcoded smoothing


def computer_acc_batch(outputs, labels):
    #     from IPython.core.debugger import set_trace
    #     set_trace()
    accs = []
    preds = np.copy(outputs)  # copy is important
    labels = np.array(labels)  # Tensor to ndarray
    for pred, label in zip(preds, labels):
        accs.append(skm.accuracy_score(label.flatten(),
                                       pred.flatten(),
                                       normalize=True))
    acc = np.nanmean(accs)
    return acc


# TODO: Find a place for this
def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """Computes IoU for one ground truth mask and predicted mask."""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


# TODO: Find a place for this
def compute_iou_batch(outputs, labels, classes=None):
    """Computes mean IoU for a batch of ground truth masks and predicted masks."""
    ious = []
    preds = np.copy(outputs)  # copy is important
    labels = np.array(labels)  # Tensor to ndarray
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def true_positive(pred, target, num_classes=2):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)


def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)


def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)


def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)


def precision(pred, target, num_classes):
    r"""Computes the precision
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def intersection_and_union(pred, target, num_classes, batch=None):
    r"""Computes intersection and union of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    if batch is None:
        i = (pred & target).sum(dim=0)
        u = (pred | target).sum(dim=0)
    else:
        i = scatter_add(pred & target, batch, dim=0)
        u = scatter_add(pred | target, batch, dim=0)

    return i, u


def mean_iou(pred, target, num_classes, batch=None):
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: :class:`Tensor`
    """
    i, u = intersection_and_union(pred, target, num_classes, batch)
    iou = i.to(torch.float) / u.to(torch.float)
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou


class Meter:
    """A meter to keep track of losses and scores"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.metrics = {
            'dice': [],
            'iou': [],
            'acc': [],
        }

    def update(self, targets, logits):
        """
        Calculate metrics for each batch.

        :param targets: Ground truth
        :type targets: torch.tensor
        :param logits: Raw logits
        :type logits: torch.tensor
        """
        probs = torch.sigmoid(logits)
        preds = Meter._predict(probs, self.base_threshold)

        dice = dice_coeff(probs, targets, self.base_threshold)
        self.metrics['dice'].extend(dice)

        iou = compute_iou_batch(preds, targets, classes=[1])
        self.metrics['iou'].append(iou)

        acc = computer_acc_batch(preds, targets)
        self.metrics['acc'].append(acc)

    def get_metrics(self):
        """
        Compute mean of batchwise metrics

        :return: Dictionary of average metrics
        :rtype: dict[str, float]
        """
        self.metrics.update({key: np.nanmean(self.metrics[key])
                             for key in self.metrics.keys()})
        return self.metrics

    @staticmethod
    def _predict(probs, threshold):
        """
        Thresholding probabilities for binary prediction

        :param probs: Probabilities from predicted output [0..1]
        :type probs: torch.Tensor
        :param threshold: logits > threshold => 1, else 0
        :type threshold: float
        :return: logits [0 or 1]
        :rtype: torch.FloatTensor
        """
        return (probs > threshold).float()

    @staticmethod
    def epoch_log(phase, epoch, epoch_loss, meter, start, fmt):
        """
        Logging metrics at the end of an epoch.

        :param phase: Phase of training ['train' or 'val']
        :type phase: str
        :param epoch: Current epoch
        :type epoch: int
        :param epoch_loss: Current average epoch loss
        :type epoch_loss: float
        :param meter: Meter object containing metrics for the epoch
        :type meter: Meter
        :param start: Time when epoch started
        :type start: str
        :param fmt: Format of the time string `start`
        :type fmt: str
        :return: Dictionary of metrics
        :rtype: dict[str, float]
        """
        metrics = meter.get_metrics()
        delta_t = datetime.strptime(start, fmt) - datetime.strptime(start, fmt)
        print(f"Loss: {epoch_loss} | dice: {metrics['dice']} | "
              f"IoU: {metrics['iou']} | Acc: {metrics['acc']} "
              f"in {delta_t}")
        return metrics
