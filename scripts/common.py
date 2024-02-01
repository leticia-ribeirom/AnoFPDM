import sys
import os

sys.path.append(os.path.realpath("./"))

import numpy as np
import torch as th
import torch.distributed as dist
from skimage import io

from guided_diffusion import dist_util, logger

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    args_to_dict,
    classifier_defaults,
    create_classifier,
)


def get_latest_model_path_in_directory(directory, model_number=None, ema=False):
    """Returns the path to the latest model in the given directory."""

    if not ema:
        model_files = [
            file for file in os.listdir(directory) if file.startswith("model")
        ]
    else:
        model_files = [file for file in os.listdir(directory) if file.startswith("ema")]

    if not model_number:
        model_numbers = sorted([int(file[5:-3]) for file in model_files])
        model_number = str(f"{model_numbers[-1]}").zfill(6)
    if not ema:
        model_file = f"model{model_number}.pt"
    else:
        model_file = f"ema_0.9999_{model_number}.pt"
    model_path = os.path.join(directory, model_file)
    return model_path, model_number


def read_model_and_diffusion(args, log_dir, model_number=None, ema=False):
    """Reads the latest model from the given directory."""
    model_path, _ = get_latest_model_path_in_directory(
        log_dir, model_number=model_number, ema=ema
    )
    logger.log(f"Model path: {model_path}")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


def read_classifier(args, log_dir, model_number=None, ema=False):
    """Reads the latest classifier from the given directory."""
    model_path, _ = get_latest_model_path_in_directory(
        log_dir, model_number=model_number, ema=ema
    )
    logger.log(f"Model path: {model_path}")

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))

    classifier.to(dist_util.dev())
    if args.use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    return classifier
