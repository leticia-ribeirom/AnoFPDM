# adapted from https://github.com/SoloChe/ddib/blob/main/scripts/synthetic_sample.py

import argparse
import os
import sys
sys.path.append(os.path.realpath('./'))

import pathlib

import numpy as np
import torch as th
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import  model_and_diffusion_defaults, add_dict_to_argparser

from torchvision.utils import make_grid, save_image



def sample(model, diffusion, num_classes=None, 
           w=None, noise=None, y=None,
           sample_shape=None, sample_steps=None, clip_denoised=True,
           reverse=False, normalize_img=True, null=False):
    
    samples_for_each_cls = 8 # default
    
    if num_classes is not None and y is None: # for clf-free
        samples_for_each_cls = sample_shape[0] // num_classes
        y = th.ones(samples_for_each_cls, dtype=th.long) *\
            th.arange(start = 0, end = num_classes).reshape(-1, 1)
        y = y.reshape(-1,1).squeeze().to(dist_util.dev())
        model_kwargs = {'y': y, 'threshold':-1, 'clf_free':True}
    elif y is not None: 
        model_kwargs = {'y': y, 'threshold':-1, 'clf_free':True}
    else:
        model_kwargs = {}
    
   
    
    if not reverse:
        samples = diffusion.ddim_sample_loop(model, 
                                            sample_shape,
                                            noise=noise,
                                            clip_denoised=clip_denoised, 
                                            w=w,
                                            sample_steps=sample_steps,
                                            model_kwargs=model_kwargs,
                                            device=dist_util.dev())
        
        
    else:
        if null:
            model_kwargs = {'threshold':-1, 'clf_free':True, 'null':True}
        samples = diffusion.ddim_reverse_sample_loop(model, 
                                                    image=noise,
                                                    clip_denoised=clip_denoised, 
                                                    sample_steps=sample_steps,
                                                    model_kwargs=model_kwargs,
                                                    device=dist_util.dev())
        
    if normalize_img:
        samples = (samples + 1) / 2
        
    return samples, samples_for_each_cls

def main():
    args = create_argparser().parse_args()
   
    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample.")
    
    logger.log(f"args: {args}")
    
    

    image_folder = args.image_dir
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    logger.log(f"reading models ...")
    args.num_classes = int(args.num_classes) if args.num_classes else None
    if args.num_classes > 0:
        args.class_cond = True
        
    args.multi_class = True if args.num_classes > 2 else False
    
    model, diffusion = read_model_and_diffusion(args, args.model_path, args.model_num)

    all_samples = []

    
    for n in range(args.n_batches):
        logger.log("sampling in progress.")
        logger.log(f"on batch {n}, device: {dist_util.dev()}")
        
        # save samples
        
        
        samples, samples_for_each_cls = sample(model, diffusion, num_classes=args.num_classes, 
                                     w=args.w, sample_shape=args.sample_shape, name=args.name)
        
        if args.save_image:
            if args.sample_shape[1] == 4:
                samples = samples.reshape(-1, 1, *args.sample_shape[2:])
                samples_for_each_cls = args.sample_shape[1]
                
                logger.log(f"reshaped samples: {samples.shape}")
                logger.log(f"args.samples: {args.sample_shape}")
                            
            grid = make_grid(samples, nrow=samples_for_each_cls)
            
            path = os.path.join(image_folder, f'batch_{n}_rank_{dist.get_rank()}.png')
            save_image(grid, path)
        
        if args.save_data_numpy:
            gathered_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples)
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

    if args.save_data_numpy:
        samples = np.concatenate(all_samples, axis=0)
        points_path = os.path.join(image_folder, f"all_samples.npy")
        np.save(points_path, samples)
    

    dist.barrier()
    logger.log(f"sampling synthetic data complete\n\n")


def create_argparser():
    defaults = dict(
        n_batches=4,
        model_path="",
        image_dir="",
        name="",
        save_data_numpy=True, # save data in numpy format
        save_image=True, # save image as in png format
        dynamical_norm=False,
        model_num=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--sample_shape",
        type=int,
        nargs="+",
        help="sample shape for a batch"
    )
    parser.add_argument(
        "--w",
        type=float,
        help="weight for clf-free samples",
        default=-1. # disabled in default
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
