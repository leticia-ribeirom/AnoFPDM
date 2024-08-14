"""
Train a diffusion model on images.
"""

import sys
import os

sys.path.append(os.path.realpath("./"))

import argparse
import pathlib
from guided_diffusion import dist_util, logger
from data import get_data_iter, check_data
from guided_diffusion.resample import create_named_schedule_sampler

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from sample import sample


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    args.w = args.w if isinstance(args.w, list) else [args.w]

    args.num_classes = int(args.num_classes) if int(args.num_classes) > 0 else None
    if args.num_classes:
        args.class_cond = True

    logger.log(f"args: {args}")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # get model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    logger.log("Model params: %.2f M" % (model_size / 1024 / 1024))

    pathlib.Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

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

    logger.log("creating data loader...")

    data = get_data_iter(
        args.name,
        args.data_dir,
        mixed=args.mixed,
        batch_size=args.batch_size, # global batch size, for each device it will be batch_size // num_devices
        split=args.split,
        ret_lab=args.ret_lab,
        logger=logger,
    )

    check_data(data[0], args.image_dir, name=args.name, split=args.split)

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_shape=tuple(args.sample_shape),
        img_dir=args.image_dir,
        threshold=args.threshold,
        w=args.w,
        num_classes=args.num_classes,
        sample_fn=sample,
        noise_fn=noise_fn,
        ddpm_sampling=args.ddpm_sampling,
        total_epochs=args.total_epochs,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        image_dir="",
        name="",
        split="train",
        training=True,
        mixed=True,
        ret_lab=True,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        n_tumour_patients=None,
        n_healthy_patients=None,
        noise_type="gaussian",
        ddpm_sampling=False,
        unet_ver="v2",
        total_epochs=100,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_shape", type=int, nargs="+", help="sample shape")

    parser.add_argument(
        "--w",
        type=float,
        nargs="+",
        help="weight for clf-free samples",
        default=-1.0,  # disabled in default
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold for clf-free training",
        default=-1.0,  # disabled in default
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
