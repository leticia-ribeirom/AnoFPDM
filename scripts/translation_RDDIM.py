"""
Synthetic domain translation from a source 2D domain to a target.
"""

import argparse
import os
import pathlib

import numpy as np
import torch.distributed as dist
import torch

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)

from data import get_brats_data_iter, check_data

from evaluate import (
    dice_coeff,
    precision,
    recall,
    ROC_AUC,
    AUC_score,
    connected_components_3d,
    get_stats,
    median_pool,
    evaluate
)
from sample import sample

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


torch.random.manual_seed(0)
np.random.seed(0)


def get_mask_batch(
    minibatch_metrics, source, modality, thr_01, abe_min, abe_max, shape, logger=None
):
    """
    mse: batch_size x sample_steps x n_modality x 128 x 128 or 256 x 256
    mse_null: batch_size x sample_steps x n_modality x 128 x 128 or 256 x 256
    """
    mse = (
        minibatch_metrics["xstart"] - source[:, modality, ...].unsqueeze(1)
    ) ** 2  # batch_size x sample_steps x n_modality x 128 x 128
    mse_null = (
        minibatch_metrics["xstart_null"] - source[:, modality, ...].unsqueeze(1)
    ) ** 2

    abe_diff = torch.mean(
        torch.abs((mse - mse_null)), dim=(3, 4) # batch_size x sample_steps x n_modality
    )  
   
    mse_flat = torch.mean(mse, dim=(2, 3, 4))  # batch_size x sample_steps
    mse_null_flat = torch.mean(mse_null, dim=(2, 3, 4))

    batch_mask = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(dist_util.dev())
    batch_map = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(dist_util.dev())
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    quant_range = torch.flip(torch.linspace(0.90, 0.99, 101), dims=(0,)).to(dist_util.dev())

    pred_lab = []
    for sample_num in range(abe_diff.shape[0]):
        # get the cosine similarity between mse and mse_null
        sim = cos(
            mse_flat[sample_num : sample_num + 1, ...],
            mse_null_flat[sample_num : sample_num + 1, ...],
        )

        # get the quantile threshold for predicted mask
        # abe_diff_i = torch.mean(abe_diff[sample_num, ...], dim=1) # sample_steps 
        abe_diff_i = abe_diff[sample_num, ...] # sample_steps x n_modality
        abe_max_i = abe_diff_i.max(dim=0)[0] # n_modality
         
        abe_max_i = torch.clamp((abe_max_i/abe_max), 0, 1)
        abe_max_i = torch.round(abe_max_i, decimals=2) * 100
        index = abe_max_i.to(torch.int64) # n_modality
        quant = quant_range[index] 

        # get the steps for predicted mask
        num1 = 0
        num2 = torch.argmax(abe_diff[sample_num, ...], dim=0) # n_modality

        # mask = torch.zeros(1, 1, shape, shape).to(dist_util.dev())
        mapp = torch.zeros(1, 1, shape, shape).to(dist_util.dev())
        thr = 0
        # for mod in range(mse.shape[2]):
        #     mask_mod = torch.mean(
        #         mse[sample_num, num1:num2, [mod], ...], axis=[0, 1], keepdim=True
        #     )  # 1 x 1 x 128 x 128
        #     mask_mod = (mask_mod - torch.min(mask_mod)) / (
        #         torch.max(mask_mod) - torch.min(mask_mod)
        #     )  # [0, 1]

        #     mapp += mask_mod  #  for auc

        #     thr = torch.quantile(mask_mod.reshape(-1), quant)
        #     mask_mod = mask_mod >= thr
        #     # p = 0.5 if mod == 0 else 0.7
        #     # mask_mod = (F.avg_pool2d(mask_mod.float(), kernel_size=5, stride=1, padding=2) > p)
        #     mask += mask_mod.float()  # union of modalities
        # mask = (mask > 0).float()  # union of modalities
        
        for mod in range(mse.shape[2]):
            mask_mod = torch.mean(
                    mse[sample_num, num1:num2[mod], [mod],...], axis=[0, 1], keepdim=True
                )  # 1 x 1 x 128 x 128   
            thr += torch.quantile(mask_mod.reshape(-1), quant[mod]) 
            mapp += mask_mod    
        
        # collect the predicted mask and map
        mapp /= mse.shape[2]
        mapp = median_pool(mapp, kernel_size=5, stride=1, padding=2)
        batch_map[sample_num] = mapp 
        
        if sim <= thr_01:
            mask = mapp >= (thr/mse.shape[2])
            batch_mask[sample_num] = connected_components_3d(mask, thr=40)
            pred_lab.append(1)
        else:
            pred_lab.append(0)

    return batch_mask, torch.tensor(pred_lab), batch_map


def cal_cos_and_abe_range(mse_flat, mse_null_flat, abe_flat, lab):
    """
    mse_flat: N_val x sample_steps x n_modality
    mse_null_flat: N_val x sample_steps x n_modality
    abe_flat: N_val x sample_steps x n_modality
    lab: N_val
    
    get the cosine similarity threshold to differentiate healthy and tumour slices
    get the abe diff range for tumour slices to determine the quantile threshold for predicted mask
    """
    mse_0_flat = mse_flat[torch.where(lab == 0)[0]]
    mse_1_flat = mse_flat[torch.where(lab == 1)[0]]
    # print(f"mse_0_flat: {mse_0_flat.shape}")
    mse_0_null_flat = mse_null_flat[torch.where(lab == 0)[0]]
    mse_1_null_flat = mse_null_flat[torch.where(lab == 1)[0]]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output_0 = cos(torch.mean(mse_0_flat, dim=2), torch.mean(mse_0_null_flat, dim=2)) # N_val
    output_1 = cos(torch.mean(mse_1_flat, dim=2), torch.mean(mse_1_null_flat, dim=2)) # N_val

    n_min = 1e6
    for q in np.linspace(0.01, 0.08, 100):
        n = torch.sum(output_1 > torch.quantile(output_0, q)) + torch.sum(
            output_0 < torch.quantile(output_0, q)
        )
        if n < n_min:
            n_min = n
            thr_01 = torch.quantile(output_0, q)
          
    # print(f"output_0: {output_0.shape}")
    abe_diff_1 = abe_flat[torch.where(lab == 1)[0]] # N_val_1 x sample_steps x n_modality
    # print(f"abe_diff_1: {abe_diff_1.shape}")
    abe_max = torch.max(abe_diff_1, dim=1)[0] # N_val_1 x n_modality
    print(f"abe_max: {abe_max.shape}")
    return thr_01, abe_max.min(dim=0)[0], abe_max.max(dim=0)[0]


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"args: {args}")
    logger.log("starting to evaluate...")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes > 0:
        args.class_cond = True
    args.multi_class = True if args.num_classes > 2 else False

    model, diffusion = read_model_and_diffusion(
        args, args.source_dir, args.model_num, args.ema
    )

    data_val = get_brats_data_iter(
        args.data_dir,
        args.batch_size_val,
        split="val",
        mixed=True,
        training=False,
        logger=logger,
    )

    data_test = get_brats_data_iter(
        args.data_dir,
        args.batch_size,
        split="test",
        mixed=True,
        training=False,
        logger=logger,
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log(f"starting to get threshold and abe range ...")

    MSE = []
    MSE_NULL = []
    ABE_DIFF = []
    LAB = []

    for i in range(args.num_batches_val):
        source_val, _, lab_val = next(data_val)
        source_val = source_val.to(dist_util.dev())

    #     y0 = torch.ones(source_val.shape[0], dtype=torch.long) *\
    #             torch.arange(start = 0, end = 1).reshape(-1, 1) # 0 for healthy
    #     y0 = y0.reshape(-1,1).squeeze().to(dist_util.dev())

    #     model_kwargs_reverse = {'threshold':-1, 'clf_free':True, 'null':args.null}
    #     model_kwargs = {'y': y0, 'threshold':-1, 'clf_free':True}
    #     minibatch_metrics = diffusion.calc_pred_xstart_loop(
    #                                                         model, source_val, args.w,
    #                                                         modality=args.modality,
    #                                                         d_reverse=args.d_reverse,
    #                                                         sample_steps=args.rev_steps,
    #                                                         model_kwargs=model_kwargs,
    #                                                         model_kwargs_reverse=model_kwargs_reverse)

    #     mse = (minibatch_metrics['xstart'] - source_val[:, args.modality, ...].unsqueeze(1))**2 # batch_size x sample_steps x n_modality x 128 x 128
    #     # mse = torch.mean(mse, dim=2, keepdim=True) # batch_size x sample_steps x 1 x 128 x 128

    #     mse_null = (minibatch_metrics['xstart_null'] - source_val[:, args.modality, ...].unsqueeze(1))**2
    #     # mse_null = torch.mean(mse_null, dim=2, keepdim=True)

    #     abe_diff = torch.abs((mse - mse_null)) # batch_size x sample_steps x n_modality x 128 x 128
        
    #     mse_flat = torch.mean(mse, dim=(3,4)) # batch_size x sample_steps x n_modality
    #     mse_null_flat = torch.mean(mse_null, dim=(3,4))
    #     abe_diff_flat = torch.mean(abe_diff, dim=(3,4))

    #     MSE.append(mse_flat)
    #     MSE_NULL.append(mse_null_flat)
    #     ABE_DIFF.append(abe_diff_flat)
    #     LAB.append(lab_val)

    # MSE = torch.cat(MSE, dim=0)
    # MSE_NULL = torch.cat(MSE_NULL, dim=0)
    # ABE_DIFF = torch.cat(ABE_DIFF, dim=0)
    # LAB = torch.cat(LAB, dim=0)
    # thr_01, abe_min, abe_max = cal_cos_and_abe_range(MSE, MSE_NULL, ABE_DIFF, LAB)
    # logger.log(f'num_val_samples: {MSE.shape[0]}, healthy: {torch.where(LAB == 0)[0].shape[0]}, tumour: {torch.where(LAB == 1)[0].shape[0]}')
    # logger.log(f'abe_min: {abe_min}, abe_max: {abe_max}, thr_01: {thr_01}')
    # # print(f"MSE: {MSE.shape}, MSE_NULL: {MSE_NULL.shape}, ABE_DIFF: {ABE_DIFF.shape}")

    logger.log(f"starting to evaluate ...")
    DICE = []
    IOU = []
    RECALL = []
    PRECISION = []
    AUC = []
    Y = []
    PRED_Y = []
    k = 0
    while k < args.num_batches:
        all_sources = []
        all_masks = []
        all_terms = {"xstart_null": [], "xstart": []}

        k += 1

        source, mask, lab = next(data_test)
        Y.append(lab)

        logger.log(
            f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}..."
        )
        logger.log(f"device: {torch.cuda.current_device()}")

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(
            f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}"
        )

        y0 = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 for healthy
        y0 = y0.reshape(-1, 1).squeeze().to(dist_util.dev())

        model_kwargs_reverse = {"threshold": -1, "clf_free": True, "null": args.null}
        model_kwargs0 = {"y": y0, "threshold": -1, "clf_free": True}

        minibatch_metrics = diffusion.calc_pred_xstart_loop(
            model,
            source,
            args.w,
            modality=args.modality,
            d_reverse=args.d_reverse,
            sample_steps=args.rev_steps,
            model_kwargs=model_kwargs0,
            model_kwargs_reverse=model_kwargs_reverse,
        )

        #### w = 2
        if args.d_reverse:
            thr_01 = 0.9945
            abe_min = torch.tensor([0.0021, 0.0014], device=dist_util.dev()) 
            abe_max = torch.tensor([0.143, 0.145], device=dist_util.dev())
        else:
            ## random
            thr_01 = 0.9840
            abe_min = torch.tensor([0.0023, 0.0018], device=dist_util.dev())
            abe_max = torch.tensor([0.247, 0.174], device=dist_util.dev())

        # collect metrics
        pred_mask, pred_lab, pred_map = get_mask_batch(
            minibatch_metrics,
            source,
            args.modality,
            thr_01,
            abe_min,
            abe_max,
            args.image_size,
            logger=logger,
        )
        PRED_Y.append(pred_lab)

        eval_metrics = evaluate(mask, pred_mask, source, pred_map)
        cls_metrics = get_stats(Y, PRED_Y)
        
        DICE.append(eval_metrics["dice"])
        IOU.append(eval_metrics["iou"])
        RECALL.append(eval_metrics["recall"])
        PRECISION.append(eval_metrics["precision"])
        AUC.append(eval_metrics["AUC"])
        
        logger.log(
            f"-------------------------------------at batch {k}-----------------------------------------"
        )
        logger.log(f"mean dice: {eval_metrics['dice']}")
        logger.log(f"mean iou: {eval_metrics['iou']}")
        logger.log(f"mean precision: {eval_metrics['precision']}")
        logger.log(f"mean recall: {eval_metrics['recall']}")
        logger.log(f"mean auc: {eval_metrics['AUC']}")
        
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"running dice: {np.mean(DICE)}")
        logger.log(f"running iou: {np.mean(IOU)}")
        logger.log(f"running precision: {np.mean(PRECISION)}")
        logger.log(f"running recall: {np.mean(RECALL)}")
        logger.log(f"running auc: {np.mean(AUC)}")
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"running cls acc: {cls_metrics['acc']}")
        logger.log(f"running cls recall: {cls_metrics['recall']}")
        logger.log(f"running cls precision: {cls_metrics['precision']}")
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )

        if args.save_data:
            logger.log("collecting metrics...")
            for key in all_terms.keys():
                terms = minibatch_metrics[key]
                gathered_terms = [
                    torch.zeros_like(terms) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_terms, terms)
                all_terms[key].extend([term.cpu().numpy() for term in gathered_terms])

            gathered_source = [
                torch.zeros_like(source) for _ in range(dist.get_world_size())
            ]
            gathered_mask = [
                torch.zeros_like(mask) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_mask, mask)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])

            all_sources = np.concatenate(all_sources, axis=0)
            all_sources_path = os.path.join(image_subfolder, f"source_{k}.npy")
            np.save(all_sources_path, all_sources)

            all_masks = np.concatenate(all_masks, axis=0)
            all_masks_path = os.path.join(image_subfolder, f"mask_{k}.npy")
            np.save(all_masks_path, all_masks)

            for key in all_terms.keys():
                all_terms_path = os.path.join(logger.get_dir(), f"{key}_terms_{k}.npy")
                np.save(all_terms_path, all_terms[key])

    dist.barrier()

    logger.log(f"evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        source_dir="",
        target_dir="",
        batch_size=32,
        rev_steps=600,
        model_num=None,
        ema=False,
        null=False,
        save_data=False,
        num_batches_val=2,
        batch_size_val=100,
        d_reverse=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=0.0,  # flair as default
    )

    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1,  # disabled in default
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        help="weight for clf-free samples",
        default=1,  # disabled in default
    )

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
