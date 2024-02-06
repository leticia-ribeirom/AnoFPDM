# copied from https://github.com/SoloChe/AnoDDPM/blob/master/evaluation.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from skimage.measure import label, regionprops
import scipy.spatial.distance as sp


def evaluate(
    real_mask: torch.Tensor,
    recon_mask: torch.Tensor,
    source: torch.Tensor,
    ano_map: torch.Tensor,
    label: torch.Tensor = None,
    cc_filter=True,
):
    """
    real_mask: [b, 1, h, w]; recon_mask: [b, 1, h, w];
    ano_map: [b, 1, h, w]; source: [b, n_mod, h, w]
    label: [b] - 0 for normal, 1 for anomalous, image-level label
    if label is not none, only evaluate on anomalous samples
    Rerurn a dict of average metrics (float) for each batch
    """
    if label is not None:
        real_mask = real_mask[torch.where(label == 1)[0], ...]
        recon_mask = recon_mask[torch.where(label == 1)[0], ...]
        source = source[torch.where(label == 1)[0], ...]
        ano_map = ano_map[torch.where(label == 1)[0], ...]

    if cc_filter:
        recon_mask = connected_components_3d(recon_mask, thr=40)

    dice_batch = dice_coeff(real_mask, recon_mask)
    iou_batch = IoU(real_mask, recon_mask)
    precision_batch = precision(real_mask, recon_mask)
    recall_batch = recall(real_mask, recon_mask)
    fpr, tpr, _ = ROC_AUC(source, real_mask, ano_map)
    AUC_score_batch = auc(fpr, tpr)
    PR_AUC_score_batch = PR_AUC_score(real_mask, ano_map)

    return {
        "dice": dice_batch,
        "iou": iou_batch,
        "precision": precision_batch,
        "recall": recall_batch,
        "AUC": AUC_score_batch,
        "PR_AUC": PR_AUC_score_batch,
    }


# for anomalous dataset - metric of crossover
def dice_coeff(real_mask: torch.Tensor, recon_mask: torch.Tensor, smooth=0.000001):
    intersection = torch.logical_and(recon_mask, real_mask).sum(dim=[1, 2, 3])
    union = torch.sum(recon_mask, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2.0 * intersection + smooth) / (union + smooth))
    return dice.item()


def IoU(real_mask, recon_mask, smooth=0.000001):
    intersection = torch.logical_and(real_mask, recon_mask).sum(dim=(1, 2, 3))
    union = torch.logical_or(real_mask, recon_mask).sum(dim=(1, 2, 3))
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou.item()


def precision(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FP = (real_mask == 1) & (recon_mask == 0)
    pr = torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)
    return pr.item()


def recall(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FN = (real_mask == 0) & (recon_mask == 1)
    re = torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)
    return re.item()


def ROC_AUC(source, real_mask, ano_map):
    # note that source is rescaled to [-1, 1]
    # find the region with mean intensity > -0.95 for auc
    foreground = source.mean(dim=1, keepdim=True).reshape(-1) > -0.95
    real_mask = real_mask.reshape(-1)[foreground]
    ano_map = ano_map.reshape(-1)[foreground]
    return roc_curve(
        real_mask.detach().cpu().numpy().flatten(),
        ano_map.detach().cpu().numpy().flatten(),
    )


def PR_AUC_score(real_mask, ano_map):
    return average_precision_score(
        real_mask.detach().cpu().numpy().flatten(),
        ano_map.detach().cpu().numpy().flatten(),
    )


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


# image-level classification metrics
def get_stats(Y, PRED_Y):
    Y = torch.cat(Y, dim=0)
    PRED_Y = torch.cat(PRED_Y, dim=0)
    acc = torch.sum(Y == PRED_Y).float() / Y.shape[0]
    re = recall(Y, PRED_Y)
    pr = precision(Y, PRED_Y)
    num_ano = torch.where(Y == 1)[0].shape[0]
    return {"acc": acc.item(), "recall": re, "precision": pr, "num_ano": num_ano}


def median_pool(ano_map, kernel_size=5, stride=1, padding=2):
    # ano_map: [b, 1, h, w]; source: [b, n_mod, h, w]
    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)
    x = F.pad(ano_map, padding, mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


def connected_components_3d(volume, thr=20):
    device = volume.device
    is_batch = True
    is_torch = torch.is_tensor(volume)
    if is_torch:
        volume = volume.cpu().numpy()
    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=0)
        is_batch = False
    # shape [b, d, h, w], treat every sample in batch independently
    pbar = range(len(volume))
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)

        nonzero_props = []
        for prop in props:
            if prop["filled_area"] <= thr:
                volume[i, cc_volume == prop["label"]] = 0
            else:
                nonzero_props.append(prop)

    if not is_batch:
        volume = volume.squeeze(0)
    if is_torch:
        volume = torch.from_numpy(volume).to(device)
    return volume
