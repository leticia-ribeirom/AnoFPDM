# copied from https://github.com/SoloChe/AnoDDPM/blob/master/evaluation.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import auc, roc_curve
import torchvision

from skimage.measure import label, regionprops
import scipy.spatial.distance as sp 

def connected_components_3d(volume):
    
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
            if prop['filled_area'] <= 20:
                volume[i, cc_volume == prop['label']] = 0
            else:
                nonzero_props.append(prop)
        
    if not is_batch:
        volume = volume.squeeze(0)
    if is_torch:
        volume = torch.from_numpy(volume).to(device)
    return volume


# for anomalous dataset - metric of crossover
def dice_coeff(real_mask: torch.Tensor, pred_mask: torch.Tensor, smooth=0.000001):
    intersection = torch.sum(pred_mask * real_mask, dim=[1, 2, 3])
    union = torch.sum(pred_mask, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice.item()

def IoU(real, recon):
    real = real.cpu().numpy()
    recon = recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)


def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(source, real_mask, square_error):
   
    foreground = source[[0],...].reshape(-1) > -1
    real_mask = real_mask[foreground]
    square_error = square_error[foreground]
    
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), 
                         square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr).item()


