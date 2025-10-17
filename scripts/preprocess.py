# Description: Preprocess the BraTS2021 dataset into numpy arrays
# adpated from https://github.com/AntanasKascenas/DenoisingAE/tree/master

import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import nrrd
import torch.nn.functional as F


def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :, :] /= p_99
    return volume

def center_crop(volume, target_shape):
    h, w, _ = volume.shape
    th, tw = target_shape, target_shape
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    cropped_volume = volume[y1:y1+th, x1:x1+tw, :]
    return cropped_volume
    

def process_patient(name, path, target_path, mod, first=-1, last=-1, downsample=False):
    
    if name == 'brats':
        flair = nib.load(path / f"{path.name}_flair.nii.gz").get_fdata()
        t1 = nib.load(path / f"{path.name}_t1.nii.gz").get_fdata()
        t1ce = nib.load(path / f"{path.name}_t1ce.nii.gz").get_fdata()
        t2 = nib.load(path / f"{path.name}_t2.nii.gz").get_fdata()
        labels = nib.load(path / f"{path.name}_seg.nii.gz").get_fdata()
    elif name == "atlas":
        t1 = nib.load(path / f"{path.name}_T1w.nii.gz").get_fdata()
        labels = nib.load(path / f"{path.name}_mask.nii.gz").get_fdata()
    elif name == 'mmbrain':
        seed = random.randint(1, 5)
        flair = center_crop(nrrd.read(path / f"TrialSeed{seed}_FLAIR.nrrd")[0], 240).astype(np.float64)
        t1 = center_crop(nrrd.read(path / f"TrialSeed{seed}_T1.nrrd")[0], 240).astype(np.float64)
        t1ce = center_crop(nrrd.read(path / f"TrialSeed{seed}_T1Gad.nrrd")[0], 240).astype(np.float64)
        t2 = center_crop(nrrd.read(path / f"TrialSeed{seed}_T2.nrrd")[0], 240).astype(np.float64)
        labels = center_crop(nrrd.read(path / f"TrialSeed{seed}_discrete_truth.nrrd")[0], 240).astype(np.float64)
    elif name == "mslub":
        flair = np.moveaxis(nib.load(path / f"{path.name}_FLAIR.nii.gz").get_fdata(), 0, -1)
        labels = np.moveaxis(nib.load(path / f"{path.name}_consensus_gt.nii.gz").get_fdata(), 0, -1)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    assert mod in ["all", "flair", "t1", "t1ce", "t2"]

    # volume shape: [1, 1, h, w, slices]
    if mod == "all":
        volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
    elif mod == "flair":
        volume = torch.stack([torch.from_numpy(x) for x in [flair]], dim=0).unsqueeze(dim=0)
    elif mod == "t1":
        volume = torch.stack([torch.from_numpy(x) for x in [t1]], dim=0).unsqueeze(dim=0)
    elif mod == "t1ce":
        volume = torch.stack([torch.from_numpy(x) for x in [t1ce]], dim=0).unsqueeze(dim=0)
    elif mod == "t2":
        volume = torch.stack([torch.from_numpy(x) for x in [t2]], dim=0).unsqueeze(dim=0)

    # exclude first n and last m slices
    # 1 4 240 240 155; 240 240 155
    
    if first > 0 and last > 0:
        volume = volume[:, :, :, :, first:-last]
        labels = labels[:, :, first:-last]
    elif first > 0 and last < 0:
        volume = volume[:, :, :, :, first:]
        labels = labels[:, :, first:]
    elif first < 0 and last > 0:
        volume = volume[:, :, :, :, :-last]
        labels = labels[:, :, :-last]
        
    # 1 1 240 240 155
    if name == 'brats' or name == 'mslub' or name == 'atlas':
        labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)
    elif name == 'mmbrain':
        labels = torch.where(torch.from_numpy(labels)==5, 1, 0).float().unsqueeze(dim=0).unsqueeze(dim=0)

    patient_dir = target_path / f"patient_{path.name}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    volume = normalise_percentile(volume)

    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

    print(f"Patient {path.name} has {fs_dim2} to {ls_dim2} slices with brain tissue.", flush=True)
    
    for slice_idx in range(fs_dim2, ls_dim2):
        if downsample:
            if name == 'brats' or name == 'atlas':
                low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
                low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
            elif name == 'mslub':
                low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(256, 256))
                low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(256, 256))
        else:
            low_res_x = volume[:, :, :, :, slice_idx]
            low_res_y = labels[:, :, :, :, slice_idx]
        np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


def preprocess(name: str, datapath: Path, mod: str, first=-1, last=-1, shape=128, downsample=True):

    all_imgs = sorted(list((datapath).iterdir()))

    sub_dir = f"preprocessed_data_{mod}_{first}{last}_{shape}"
    splits_path = datapath.parent / sub_dir / "data_splits"

    if not splits_path.exists():

        indices = list(range(len(all_imgs)))
        random.seed(10)
        random.shuffle(indices)

        if name == 'brats':
            n_train = int(len(indices) * 0.75)
            n_val = int(len(indices) * 0.05)
            n_test = len(indices) - n_train - n_val
        elif name == 'atlas':
            n_train = int(len(indices) * 0.75)
            n_val = int(len(indices) * 0.05)
            n_test = len(indices) - n_train - n_val
        elif name == 'mmbrain':
            n_train = 0
            n_val = 5
            n_test = len(indices) - n_train - n_val
        elif name == 'MSLUB':
            n_train = 20
            n_val = 5
            n_test = 5

        split_indices = {}
        split_indices["train"] = indices[:n_train]
        split_indices["val"] = indices[n_train:n_train + n_val]
        split_indices["test"] = indices[n_train + n_val:]

        for split in ["train", "val", "test"]:
            (splits_path / split).mkdir(parents=True, exist_ok=True)
            with open(splits_path / split / "scans.csv", "w") as f:
                f.write("\n".join([all_imgs[idx].name for idx in split_indices[split]]))

    for split in ["train", "val", "test"]:
        paths = [datapath / x.strip() for x in open(splits_path / split / "scans.csv").readlines()]

        print(f"Patients in {split}]: {len(paths)}")

        for source_path in tqdm(paths):
            target_path = datapath.parent / sub_dir / f"npy_{split}"
            process_patient(name, source_path, target_path, mod, first, last, downsample=downsample)


if __name__ == "__main__":
   
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/BraTS21_training/BraTS21', type=str, help="path to Brats2021 Data directory")
    parser.add_argument("--name", default='brats', type=str, help="dataset name")
    parser.add_argument("-m", "--mod", default='all', type=str, help="modelity to preprocess")
     
    # parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/MSLUB/data', type=str, help="path to data directory")
    # parser.add_argument("--name", default='MSLUB', type=str, help="dataset name")
    # parser.add_argument("--mod", default='all', type=str, help="modelity to preprocess")
    
    # parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/ATLAS/raw2', type=str, help="path to data directory")
    # parser.add_argument("--name", default='atlas', type=str, help="dataset name")
    # parser.add_argument("--mod", default='t1', type=str, help="modelity to preprocess")
     
     
    parser.add_argument("--first", default=0, 
                        type=int, help="skip first n slices")
    parser.add_argument("--last", default=0,
                        type=int, help="skip last n slices")
    
    args = parser.parse_args()

    datapath = Path(args.source)
   
    preprocess(args.name, datapath, args.mod, args.first, args.last, downsample=True)