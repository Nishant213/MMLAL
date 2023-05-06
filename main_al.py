import os
import os.path as osp

import torch
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from helpers import BuildDataLoader, BuildDataset
from core_models import SegmenthorSL, SegmenthorSSL

import numpy as np
from PIL import Image
import torch.nn.functional as F

from tqdm import tqdm

if __name__ == "__main__":

    parser = ArgumentParser(description="SegSL")

    ### 0. Config file?
    parser.add_argument(
        "--config-file", default=None, help="Path to configuration file"
    )

    ### 1. Data Loading
    parser.add_argument("--dataset", default="camvid")
    parser.add_argument("--train_l_n", default="train_al_lab")
    parser.add_argument("--train_ul_n", default=None)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--batch_iters", default=100, type=int)

    ### 2. Trainer options
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--sync_batchnorm", action="store_true", default=False)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=20)

    ### 3. Noisy Student
    parser.add_argument("--use_ns", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--pixel_threshold", type=float, default=0.70)
    parser.add_argument("--pseudo_dir", default=None)
    parser.add_argument("--label_grid_size", default=0, type = int)
    parser.add_argument("--label_grid_counts", default=0, type = int)

    ### 4. Model specific args
    parser.add_argument("--pretrained_ckpt", default = None)
    parser = SegmenthorSL.add_model_specific_args(parser)

    args = parser.parse_args()
    print(args)
    
    if args.pseudo_dir is not None:
        os.makedirs(args.pseudo_dir, exist_ok=True)

    if args.exp_name is None:
        args.exp_name = "{}_{}_{}_{}_{}_{}".format(
            args.backbone,
            args.dataset,
            args.train_l_n,
            not not args.train_ul_n,
            args.loss_fx_focal_weight,
            args.use_polyloss,
        )

    checkpoint_callback = ModelCheckpoint(filename="{epoch:03d}", save_last=True)

    trainer = Trainer.from_argparse_args(args, callbacks=checkpoint_callback,)

    data_loader_main = BuildDataLoader(dataset=args.dataset)
    dataset_info = data_loader_main.build_supervised(args)
    
    if args.pretrained_ckpt is not None:
        model = SegmenthorSL.load_from_checkpoint(args.pretrained_ckpt, args = args, dataset_info = dataset_info)
        trainer.validate(model)
    else:
        model = SegmenthorSL(args=args, dataset_info=dataset_info)
        trainer.fit(model)
    
    if args.use_ns:

        args.batch_size = int(args.batch_size / 2)
        args.batch_iters = int(args.batch_iters * 2)

        for gen_idx in range(args.num_generations):

            trainer.predict(model, dataloaders = model.label_dataloader())

            data_loader_main = BuildDataLoader(dataset=args.dataset,
                                               pseudo_dir=args.pseudo_dir
                                               )

            dataset_info = data_loader_main.build_supervised(args)

            checkpoint_callback = ModelCheckpoint(
                filename="{epoch:03d}", save_last=True
            )

            trainer = Trainer.from_argparse_args(args, callbacks=checkpoint_callback,)

            model = SegmenthorSSL(args=args, dataset_info=dataset_info)
            trainer.fit(model)

    torch.cuda.empty_cache()
    
