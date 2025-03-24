import logging
import sys

import torch

from config import SpectrogramConfig
from tqdne import metric, plot
from tqdne.architectures import get_2d_unet_config
from tqdne.dataloader import get_train_and_val_loader
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint, get_device


def run(args):
    name = "EDM-LogSpectrogram"
    config = SpectrogramConfig(args.workdir, args.infile)
    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning

    train_loader, val_loader = get_train_and_val_loader(config, 1, 2)
    for batch in train_loader:
        print(batch['waveform'].shape)
        print(signal['waveform'].shape)
        break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Train a 2D diffusion model"
    )
    parser.add_argument("--workdir", type=str, help="the working directory in which checkpoints and all output are saved to")
    parser.add_argument("--infile", type=str, default=None, help="location of the training file; if not given assumes training data is located as `workdir/data/preprocessed_waveforms.h5`")    
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
