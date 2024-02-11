import argparse
import os
import sys

sys.path.append(os.path.realpath("./"))

import pathlib

import numpy as np
import torch
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from guided_diffusion.gaussian_diffusion import clamp_to_spatial_quantile

from torchvision.utils import make_grid, save_image


def sample(
    model,
    diffusion,
    num_classes=None,
    w=None,
    noise=None,
    y=None,
    cond_fn=None,
    sample_shape=None,
    sample_steps=None,
    clip_denoised=True,
    dynamic_clip=False,
    reverse=False,
    normalize_img=True,
    null=False,
    ddpm=False,
    noise_fn=None,
):
    samples_for_each_cls = 8  # default
    clf_free = False if w == -1 else True
    if num_classes is not None and y is None:  # for clf-free
        samples_for_each_cls = sample_shape[0] // num_classes
        y = torch.ones(samples_for_each_cls, dtype=torch.long) * torch.arange(
            start=0, end=num_classes
        ).reshape(-1, 1)
        y = y.reshape(-1, 1).squeeze().to(dist_util.dev())
        model_kwargs = {"y": y, "threshold": -1, "clf_free": clf_free}
    elif y is not None:
        model_kwargs = {"y": y, "threshold": -1, "clf_free": clf_free}

    else:
        model_kwargs = {}

    if not ddpm:
        if not reverse:
            samples = diffusion.ddim_sample_loop(
                model,
                sample_shape,
                noise=noise,
                cond_fn=cond_fn,
                clip_denoised=clip_denoised,
                w=w,
                denoised_fn=clamp_to_spatial_quantile if dynamic_clip else None,
                sample_steps=sample_steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
            )

        else:
            if null:
                model_kwargs = {"threshold": -1, "clf_free": True, "null": True}
            samples = diffusion.ddim_reverse_sample_loop(
                model,
                image=noise,
                clip_denoised=clip_denoised,
                denoised_fn=clamp_to_spatial_quantile if dynamic_clip else None,
                sample_steps=sample_steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
            )
    else:
        samples = diffusion.p_sample_loop(
            model,
            sample_shape,
            noise=noise,
            cond_fn=cond_fn,
            clip_denoised=clip_denoised,
            noise_fn=noise_fn,
            denoised_fn=clamp_to_spatial_quantile if dynamic_clip else None,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
        )

    if normalize_img:
        samples = (samples + 1) / 2

    return samples, samples_for_each_cls


if __name__ == "__main__":
    pass
