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

from data import get_brats_data_iter

from evaluate import get_stats, evaluate
from sample import sample

from obtain_hyperpara import obtain_optimal_threshold, get_mask_batch

from torch.nn.parallel.distributed import DistributedDataParallel as DDP


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log(f"args: {args}")
    logger.log("starting to sample.")

    image_subfolder = args.image_dir
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
    if args.num_classes:
        args.class_cond = True

    if args.noise_type == "simplex":
        from noise import generate_simplex_noise
        from simplex import Simplex_CLASS

        simplex = Simplex_CLASS()
        noise_fn = lambda x, t: generate_simplex_noise(
            simplex,
            x,
            t,
            False,
            in_channels=args.in_channels,
            octave=6,
            persistence=0.8,
            frequency=64,
        )
    elif args.noise_type == "gaussian":
        noise_fn = None
    else:
        raise ValueError(f"Unknown noise type: {args.noise_type}")

    model, diffusion = read_model_and_diffusion(
        args, args.model_dir, args.model_num, args.ema
    )

    data_test = get_brats_data_iter(
        args.data_dir,
        args.batch_size,
        split="test",
        mixed=True,
        seed=args.seed,
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

    if args.num_batches_val != 0:
        data_val = get_brats_data_iter(
            args.data_dir,
            args.batch_size_val,
            split="val",
            mixed=True,
            seed=args.seed,
            logger=logger,
        )

        opt_thr, dice_max_val = obtain_optimal_threshold(
            data_val,
            diffusion,
            model,
            args,
            dist_util.dev(),
            guided=False,
            ddib=False,
            noise_fn=noise_fn,
            use_ddpm=args.use_ddpm,
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

        t = torch.tensor(
            [args.sample_steps - 1] * source.shape[0], device=dist_util.dev()
        )
        ep = noise_fn(source, t) if args.noise_type == "simplex" else None
        noise = diffusion.q_sample(source, t=t, noise=ep)

        Y.append(lab)

        target, _ = sample(
            model,
            diffusion,
            noise=noise,
            w=args.w,
            sample_shape=source.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
            noise_fn=noise_fn,
            ddpm=args.use_ddpm,
        )

        pred_mask, pred_map, pred_lab = get_mask_batch(
            source, target, opt_thr, args.modality
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
            gathered_pred_maps = [
                torch.zeros_like(pred_map) for _ in range(dist.get_world_size())
            ]

            dist.all_gather(gathered_source, source)
            dist.all_gather(gathered_latent, noise)
            dist.all_gather(gathered_target, target)
            dist.all_gather(gathered_mask, mask)
            dist.all_gather(gathered_pred_maps, pred_map)

            all_sources.extend([source.cpu().numpy() for source in gathered_source])
            all_latents.extend([noise.cpu().numpy() for noise in gathered_latent])
            all_targets.extend([target.cpu().numpy() for target in gathered_target])
            all_masks.extend([mask.cpu().numpy() for mask in gathered_mask])
            all_pred_maps.extend(
                [pred_map.cpu().numpy() for pred_map in gathered_pred_maps]
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
            all_pred_maps_path = os.path.join(image_subfolder, f"pred_maps_{k}.npy")
            np.save(all_pred_maps_path, all_pred_maps)

    dist.barrier()
    logger.log(f"synthetic data translation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        model_dir="",
        seed=0,
        batch_size=32,
        sample_steps=1000,
        use_ddpm=True,
        model_num=None,
        ema=False,
        dynamic_clip=False,
        save_data=False,
        num_batches_val=2,
        batch_size_val=100,
        noise_type="gaussian",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=int,
        nargs="+",
        help="0:flair, 1:t1, 2:t1ce, 3:t2",
        default=0,  # flair as default
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
