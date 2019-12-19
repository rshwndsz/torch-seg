# Python STL
from datetime import datetime
import logging
import time
# Data Science
import numpy as np
from scipy.optimize import linear_sum_assignment
# PyTorch
import torch

# Local
from torchseg import utils

logger = logging.getLogger(__name__)

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


def iou_score(preds, targets):
    """Computes IoU or Jaccard index

    Parameters
    ----------
    preds : torch.Tensor
        Predictions
    targets : torch.Tensor
        Ground truths
    Returns
    -------
    iou : float
        IoU score or Jaccard index
    """
    intersection = torch.sum(targets * preds)
    union = torch.sum(targets) + torch.sum(preds) - intersection + 1e-7
    score = (intersection + 1e-7) / union

    return score.item()


def get_fast_dice_2(pred, true):
    """
        Ensemble dice
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total


def get_fast_pq(pred, true, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) - 1,
                             len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        # <<< Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        # <<< extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def get_fast_aji(pred, true):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


def custom_pq_dq_sq(preds, targets, iou):
    tp = true_positive(preds, targets)[1]
    fp = false_positive(preds, targets, num_classes=2)[1]
    fn = false_negative(preds, targets, num_classes=2)[1]

    dq = tp / (tp + 0.5*fp + 0.5*fn)
    sq = iou / tp
    pq = dq * sq
    return pq, dq, sq


class Meter:
    """A meter to keep track of losses and scores"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.metrics = {
            'dice': [],
            'iou': [],
            'acc': [],
            'prec': [],
            'pq': [],
            'sq': [],
            'dq': [],
            'aji': [],
            'dice_2': [],
        }

    def update(self, targets, logits):
        """Calculates metrics for each batch and updates meter

        Parameters
        ----------
        targets : torch.Tensor
            [N C H W]
            Ground truths
        logits : torch.Tensor
            [N C H W]
            Raw logits
        """
        probs = torch.sigmoid(logits)
        preds = utils.predict(probs, self.base_threshold)

        # Assertion for shapes
        if not (preds.shape == targets.shape):
            raise ValueError("Shape of preds: {} must be the same as that of targets: {}."
                             .format(preds.shape, targets.shape))

        # Calculate and add to metric lists
        dice = dice_score(probs, targets, self.base_threshold)
        self.metrics['dice'].append(dice)

        iou = iou_score(preds, targets)
        self.metrics['iou'].append(iou)

        acc = accuracy_score(preds, targets)
        self.metrics['acc'].append(acc)

        prec = precision_score(preds, targets, num_classes=2)  # <<< TODO: Remove hardcoded num_classes
        self.metrics['prec'].append(prec)

        # pq = get_fast_pq(preds.long(), targets.long())
        # self.metrics['pq'].append(pq[0][2])
        # self.metrics['dq'].append(pq[0][0])
        # self.metrics['sq'].append(pq[0][1])

        pq = custom_pq_dq_sq(preds.long(), targets.long(), iou)
        self.metrics['pq'].append(pq[0])
        self.metrics['dq'].append(pq[1])
        self.metrics['sq'].append(pq[2])

        aji = get_fast_aji(preds.long(), targets.long())
        self.metrics['aji'].append(aji)

        dice_2 = get_fast_dice_2(preds.long(), targets.long())
        self.metrics['dice_2'].append(dice_2)

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
    def epoch_log(epoch_loss, meter, start_time, fmt):
        """Logs and returns metrics

        Parameters
        ----------
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

        # Construct string for logging
        metric_string = f"Loss: {epoch_loss:.4f} | "
        for metric_name, metric_value in metrics.items():
            metric_string += f"{metric_name}: {metric_value:.4f} | "
        metric_string += f"in {delta_t}"

        logger.info(f"{metric_string}")
        return {'loss': epoch_loss, **metrics}
