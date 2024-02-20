import argparse
import os
import pathlib
import random
import numpy as np
import torch.distributed as dist
import torch

from common import read_model_and_diffusion, read_classifier
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    classifier_defaults
)

from data import get_brats_data_iter

from evaluate import get_stats, median_pool, evaluate
from sample import sample

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from obtain_hyperpara import get_mask_for_batch, obtain_optimal_threshold


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log(f"args: {args}")
    

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes:
        args.class_cond = True

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )
    model.eval()

    classifier = read_classifier(args, args.clf_dir, args.clf_num)
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1) # 100x2
            selected = log_probs[range(len(logits)), y.view(-1)] # 100
            a = torch.autograd.grad(selected.sum(), x_in)[0] # 100x4x128x128
            return a * args.classifier_scale

    all_sources = []
    all_latents = []
    all_targets = []
    all_masks = []

    data_val = get_brats_data_iter(
        args.data_dir,
        args.batch_size_val,
        split="val",
        mixed=True,
        training=False,
        seed=args.seed,
        logger=logger,
    )

    data_test = get_brats_data_iter(
        args.data_dir,
        args.batch_size,
        split="test",
        mixed=True,
        training=False,
        seed=args.seed,
        logger=logger,
    )
    opt_thr, dice_max_val = obtain_optimal_threshold(
        data_val, diffusion, model, args, dist_util.dev(), ddib=False, cond_fn=cond_fn
    )
    logger.log(f"optimal threshold: {opt_thr}, dice_max_val: {dice_max_val}")
   

    DICE = []
    DICE_ANO = []
    IOU = []
    IOU_ANO = []
    RECALL = []
    RECALL_ANO = []
    PRECISION = []
    PRECISION_ANO = []
    AUC = []
    AUC_ANO = []
    PR_AUC = []
    PR_AUC_ANO = []
    Y = []
    PRED_Y = []

    k = 0
    while k < args.num_batches:
        k += 1

        source, mask, lab = next(data_test)
        logger.log(
            f"translating at batch {k} on rank {dist.get_rank()}, shape {source.shape}..."
        )

        source = source.to(dist_util.dev())
        mask = mask.to(dist_util.dev())

        logger.log(
            f"source with mean {source.mean()} and std {source.std()} on rank {dist.get_rank()}"
        )

        Y.append(lab)

        y = torch.ones(source.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 only for healthy
        y = y.reshape(-1, 1).squeeze().to(dist_util.dev())

        t = torch.tensor(
            [args.sample_steps - 1] * source.shape[0], device=dist_util.dev()
        )
        noise = diffusion.q_sample(source, t=t)

        target, _ = sample(
            model,
            diffusion,
            y=y,
            cond_fn=cond_fn,
            noise=noise,
            w=args.w,
            sample_shape=source.shape,
            sample_steps=args.sample_steps,
            normalize_img=False,
        )

        pred_mask, pred_map, pred_lab = get_mask_for_batch(
            source, target, opt_thr, args.modality, median_filter=True
        )
        PRED_Y.append(pred_lab)

        eval_metrics = evaluate(mask, pred_mask, source, pred_map)
        eval_metrics_ano = evaluate(mask, pred_mask, source, pred_map, lab)
        cls_metrics = get_stats(Y, PRED_Y)

        DICE.append(eval_metrics["dice"])
        DICE_ANO.append(eval_metrics_ano["dice"])

        IOU.append(eval_metrics["iou"])
        IOU_ANO.append(eval_metrics_ano["iou"])

        RECALL.append(eval_metrics["recall"])
        RECALL_ANO.append(eval_metrics_ano["recall"])

        PRECISION.append(eval_metrics["precision"])
        PRECISION_ANO.append(eval_metrics_ano["precision"])

        AUC.append(eval_metrics["AUC"])
        AUC_ANO.append(eval_metrics_ano["AUC"])

        PR_AUC.append(eval_metrics["PR_AUC"])
        PR_AUC_ANO.append(eval_metrics_ano["PR_AUC"])

        logger.log(
            f"-------------------------------------at batch {k}-----------------------------------------"
        )
        logger.log(f"mean dice: {eval_metrics['dice']:0.3f}")
        logger.log(f"mean iou: {eval_metrics['iou']:0.3f}")
        logger.log(f"mean precision: {eval_metrics['precision']:0.3f}")
        logger.log(f"mean recall: {eval_metrics['recall']:0.3f}")
        logger.log(f"mean auc: {eval_metrics['AUC']:0.3f}")
        logger.log(f"mean pr auc: {eval_metrics['PR_AUC']:0.3f}")

        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"running dice: {np.mean(DICE):0.3f}")  # keep 3 decimals
        logger.log(f"running iou: {np.mean(IOU):0.3f}")
        logger.log(f"running precision: {np.mean(PRECISION):0.3f}")
        logger.log(f"running recall: {np.mean(RECALL):0.3f}")
        logger.log(f"running auc: {np.mean(AUC):0.3f}")
        logger.log(f"running pr auc: {np.mean(PR_AUC):0.3f}")
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"running dice ano: {np.mean(DICE_ANO):0.3f}")
        logger.log(f"running iou ano: {np.mean(IOU_ANO):0.3f}")
        logger.log(f"running precision ano: {np.mean(PRECISION_ANO):0.3f}")
        logger.log(f"running recall ano: {np.mean(RECALL_ANO):0.3f}")
        logger.log(f"running auc ano: {np.mean(AUC_ANO):0.3f}")
        logger.log(f"running pr auc ano: {np.mean(PR_AUC_ANO):0.3f}")
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"running cls acc: {cls_metrics['acc']:0.3f}")
        logger.log(f"running cls recall: {cls_metrics['recall']:0.3f}")
        logger.log(f"running cls precision: {cls_metrics['precision']:0.3f}")
        logger.log(f"running cls num_ano: {cls_metrics['num_ano']}")
        logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        logger.log(f"finished translation {target.shape}")

        if args.save_data:
            logger.log("collecting metrics...")
            gathered_source = [
                torch.zeros_like(source) for _ in range(dist.get_world_size())
            ]
            gathered_latent = [
                torch.zeros_like(noise) for _ in range(dist.get_world_size())
            ]
            gathered_target = [
                torch.zeros_like(target) for _ in range(dist.get_world_size())
            ]
            gathered_mask = [
                torch.zeros_like(mask) for _ in range(dist.get_world_size())
            ]

            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_latent, noise)
            dist.all_gather(gathered_target, pred_map)
            dist.all_gather(gathered_mask, mask)
            
    if args.save_data:
        all_sources.extend([source.cpu().numpy() for source in gathered_source])
        all_latents.extend([noise.cpu().numpy() for noise in gathered_latent])
        all_targets.extend([target.cpu().numpy() for target in gathered_target])
        all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])

        all_sources = np.concatenate(all_sources, axis=0)
        all_sources_path = os.path.join(image_subfolder, "source.npy")
        np.save(all_sources_path, all_sources)

        all_latents = np.concatenate(all_latents, axis=0)
        all_latents_path = os.path.join(image_subfolder, "latent.npy")
        np.save(all_latents_path, all_latents)

        all_targets = np.concatenate(all_targets, axis=0)
        all_targets_path = os.path.join(image_subfolder, "target.npy")
        np.save(all_targets_path, all_targets)

        all_masks = np.concatenate(all_masks, axis=0)
        all_masks_path = os.path.join(image_subfolder, "mask.npy")
        np.save(all_masks_path, all_masks)

    dist.barrier()
    logger.log(f"synthetic data translation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        model_dir="",  # model directory,
        clf_dir="", # classifier directory
        seed=0,
        batch_size=32,
        sample_steps=1000,
        model_num=None,
        clf_num=None,
        ema=True,
        save_data=False,
        dynamic_clip=False,
        num_batches_val=2,
        batch_size_val=100,
        classifier_scale=100,
        unet_ver="v1",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
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
        default=-1.0,  # disabled in default
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
