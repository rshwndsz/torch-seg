# Data Science
import numpy as np
import sklearn.metrics as skm
# PyTorch
import torch


# TODO: Move this into Metric
def predict(logits, threshold):
    """Thresholding X for binary prediction"""
    logits_copy = np.copy(logits)
    preds = (logits_copy > threshold).astype('uint8')
    return preds


# TODO: Move this into Metric
def metric(probability, truth, threshold=0.5, reduction='none'):
    """Calculates dice of positive and negative images seperately
       probability and truth must be torch tensors"""

    batch_size = len(truth)
    with torch.no_grad():
        # Shape [N, C, H, W]
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        # Shape: [N, C*H*W]
        assert (probability.shape == truth.shape), "Shape of prob and truth in fn():metric are not same"

        # Only 1's and 0's in p & t
        p = (probability > threshold).float()
        t = (truth > 0.5).float()  # <<<< Change: Hardcoded threshold for ground-truth

        # Shape: [N, C*H*W]
        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        # Shape: [N, 1]
        pos_index = torch.nonzero(t_sum >= 1)

        # Shape: [N, 1]
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))
    return dice_pos


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


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<< Change: Hardcoded threshold
        self.base_dice_scores = []
        self.iou_scores = []
        self.acc_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)

        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

        acc = computer_acc_batch(preds, targets)  # <<< TODO: Test this
        self.acc_scores.append(acc)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        acc = np.nanmean(self.acc_scores)
        return dice, iou, acc  # <<< TODO: Return metrics as a dict


# TODO: Move this into the trainer
def epoch_log(phase, epoch, epoch_loss, meter, start):
    """Logging the metrics at the end of an epoch."""
    dice, iou, acc = meter.get_metrics()
    print("Loss: %0.4f | dice: %0.4f | IoU: %0.4f | Acc: %0.4f" % (epoch_loss, dice, iou, acc))
    return dice, iou, acc


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
