import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from image_adapt.guided_diffusion import dist_util, logger
from image_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from image_adapt.guided_diffusion.image_datasets import load_data
from torchvision import utils
import math
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from PIL import Image

# added
def load_reference(data_dir, batch_size, image_size, class_cond=False, corruption="shot_noise", 
                   severity=5, train=True, dataset="cifar10"): 
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
        corruption=corruption,
        severity=severity,
        # deg_type=deg_type,
        train=train,
        dataset=dataset,
    )
    for large_batch, model_kwargs, filename in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs, filename


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        corruption=args.corruption,
        severity=args.severity,
        train=args.train,
        dataset=args.dataset,
    )

    assert args.num_samples >= args.batch_size * dist_util.get_world_size(), "The number of the generated samples will be larger than the specified number."
    

    logger.log(f"creating samples for corruption: {args.corruption} and severity: {args.severity}")
    count = 0
    while count * args.batch_size * dist_util.get_world_size() < args.num_samples:
        model_kwargs, filename = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (len(filename), 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            N=args.N,
            D=args.D,
            scale=args.scale, 
        )

        # From guided-diffusion and improved-diffusion. 
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        for i in range(len(filename)): 
            if args.dataset == "imagenetc":
                path = os.path.join(logger.get_dir(), args.corruption, str(args.severity), filename[i].split('/')[0])
                out_path = os.path.join(path, filename[i].split('/')[1])
            elif args.dataset == "imagenet":
                postfix = 'train' if args.train else 'val'
                path = os.path.join(logger.get_dir(), args.corruption, postfix, filename[i].split('/')[-2]) #, filename[i].split('/')[-2])
                out_path = os.path.join(path, filename[i].split('/')[-1])
            elif args.dataset == "cifar10":
                postfix = 'cifar_train' if args.train else 'cifar_test'
                path = os.path.join(logger.get_dir(), args.corruption, postfix)
                out_path = os.path.join(path, filename[i].split('/')[-1])
            elif args.dataset == "cifar10c":
                path = os.path.join(logger.get_dir(), args.corruption, str(args.severity))
                out_path = os.path.join(path, filename[i].split('/')[-1])
            else:
                raise NotImplementedError("Dataset loading not implemented for {}".format(args.dataset))
            
            os.makedirs(path, exist_ok=True)
            
            img = Image.fromarray(sample[i].cpu().numpy())
            img.save(out_path)

        count += 1
        logger.log(f"created {count * args.batch_size * dist_util.get_world_size()} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        D=32, # scaling factor
        N=50,
        scale=1,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
        corruption=None, 
        severity=5,
        train = True,
        dataset = "cifar10",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()