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
    classifier_defaults,
)

from data import get_data_iter

from evaluate import get_stats, evaluate, logging_metrics
from sample import sample

import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from obtain_hyperpara import get_mask_batch, obtain_optimal_threshold


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
            log_probs = F.log_softmax(logits, dim=-1)  # 100x2
            selected = log_probs[range(len(logits)), y.view(-1)]  # 100
            a = torch.autograd.grad(selected.sum(), x_in)[0]  # 100x4x128x128
            return a * args.classifier_scale

    data_test = get_data_iter(
        args.name,
        args.data_dir,
        mixed=True,
        batch_size=args.batch_size,
        split="test",
        seed=args.seed,
        logger=logger,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    if args.num_batches_val != 0:
        data_val = get_data_iter(
            args.name,
            args.data_dir,
            mixed=True,
            batch_size=args.batch_size_val,
            split="val",
            seed=args.seed,
            logger=logger,
            use_weighted_sampler=args.use_weighted_sampler,
        )
        opt_thr, dice_max_val = obtain_optimal_threshold(
            data_val,
            diffusion,
            model,
            args,
            dist_util.dev(),
            ddib=False,
            cond_fn=cond_fn,
        )
        logger.log(f"optimal threshold: {opt_thr}, dice_max_val: {dice_max_val}")
    else:
        opt_thr = 0.63 # atlas 200 500

    logging = logging_metrics(logger) 
    Y = []
    PRED_Y = []

    k = 0
    while k < args.num_batches:
        k += 1

        all_sources = []
        all_latents = []
        all_targets = []
        all_masks = []
        all_pred_maps = []

        source, mask, lab = data_test.__iter__().__next__()
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

        if args.num_batches_val == 0:
            opt_thr = 0.24  # for optimal setup

        pred_mask, pred_map, pred_lab = get_mask_batch(
            source, target, opt_thr, args.modality, median_filter=True
        )
        PRED_Y.append(pred_lab)

        eval_metrics = evaluate(mask, pred_mask, source, pred_map)
        eval_metrics_ano = evaluate(mask, pred_mask, source, pred_map, lab)
        cls_metrics = get_stats(Y, PRED_Y)

        logging.logging(eval_metrics, eval_metrics_ano, cls_metrics, k)

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
            gathered_pred_map = [
                torch.zeros_like(pred_map) for _ in range(dist.get_world_size())
            ]

            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_latent, noise)
            dist.all_gather(gathered_target, target)
            dist.all_gather(gathered_mask, mask)
            dist.all_gather(gathered_pred_map, pred_map)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_latents.extend([noise.cpu().numpy() for noise in gathered_latent])
            all_targets.extend([target.cpu().numpy() for target in gathered_target])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            all_pred_maps.extend(
                [pred_map.cpu().numpy() for pred_map in gathered_pred_map]
            )

            all_sources = np.concatenate(all_sources, axis=0)
            all_sources_path = os.path.join(image_subfolder, f"source_{k}.npy")
            np.save(all_sources_path, all_sources)

            all_latents = np.concatenate(all_latents, axis=0)
            all_latents_path = os.path.join(image_subfolder, f"latent_{k}.npy")
            np.save(all_latents_path, all_latents)

            all_targets = np.concatenate(all_targets, axis=0)
            all_targets_path = os.path.join(image_subfolder, f"target_{k}.npy")
            np.save(all_targets_path, all_targets)

            all_masks = np.concatenate(all_masks, axis=0)
            all_masks_path = os.path.join(image_subfolder, f"mask_{k}.npy")
            np.save(all_masks_path, all_masks)

            all_pred_maps = np.concatenate(all_pred_maps, axis=0)
            all_pred_maps_path = os.path.join(image_subfolder, f"pred_map_{k}.npy")
            np.save(all_pred_maps_path, all_pred_maps)

    dist.barrier()
    logger.log(f"synthetic data translation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        name="",
        image_dir="",
        model_dir="",  # model directory,
        clf_dir="",  # classifier directory
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
        use_weighted_sampler=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=0,  # first modality as default
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
