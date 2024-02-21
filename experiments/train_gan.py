import logging
from math import e
from pathlib import Path
from re import S
from diffusers.commands import env
from numpy import test


from torch.utils.data import DataLoader

from tqdne.dataset import StationarySignalDataset, WaveformDataset
from tqdne.gan import WGAN
from tqdne.utils import get_last_checkpoint
from tqdne.training import get_pl_trainer
from tqdne.metric import PowerSpectralDensity, SamplePlot, RepresentationInversion
from tqdne.representations import CenteredMaxEnvelope, CenteredPMeanEnvelope, GlobalMaxEnvelope
from tqdne.conf import Config

def main():
    # Setting up Args
    run_name = "WGAN Dummy Data"
    config = Config()
    resume = False
    conditional = True
    max_epochs = 100
    batch_size = 64
    wfs_expected_size = 512
    latent_dim = 128
    encoding_L = 8
    dim = 32

    logging.info("Loading data...")
    # train_path = config.datasetdir / Path(config.data_upsample_train)
    # test_path = config.datasetdir / Path(config.data_upsample_test)
    # envelope_representation = CenteredMaxEnvelope(config)
    # train_dataset = WaveformDataset(train_path, envelope_representation, reduced=wfs_expected_size)
    # test_dataset = WaveformDataset(test_path, envelope_representation, reduced=wfs_expected_size)
    
    # envelope_representation = GlobalMaxEnvelope(config)
    envelope_representation = CenteredPMeanEnvelope(config, window_length=17, p=7)
    train_dataset = StationarySignalDataset(100000, envelope_representation, wfs_expected_size)
    test_dataset = StationarySignalDataset(10000, envelope_representation, wfs_expected_size)

    channels = train_dataset[0]["high_res"].shape[0]
    logging.info(f"Channels: {channels}")
    num_conditional_vars = train_dataset[0]["cond"].shape[0]
    logging.info(f"Number of conditional variables: {num_conditional_vars}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)
    condv_names = ["dist", "mag"]

    logging.info("Set parameters...")
    plots = [
        SamplePlot(fs=config.fs, channel=0),
        PowerSpectralDensity(config.fs, channel=0),
    ]
    # plots = [
    #     RepresentationInversion(metric, envelope_representation) for metric in plots
    # ]
    metrics = plots

    optimizer_parameters = {
        "lr": 1e-4,
        # "momentum": 0.5,
        "b1": 0.9,
        "b2": 0.99,
    }
    generator_parameters = {
        "latent_dim": latent_dim,
        "wave_size": wfs_expected_size,
        "out_channels": channels,
        "encoding_L": encoding_L,
        "num_cond_vars": num_conditional_vars,
        "dim": dim,
    }
    discriminator_parameters = {
        "wave_size": wfs_expected_size,
        "in_channels": channels,
        "encoding_L": encoding_L,
        "num_cond_vars": num_conditional_vars,
        "dim": dim,
    }
    model_parameters = {
        "reg_lambda": 10.0,
        "n_critics": 3,
        "optimizer_params": optimizer_parameters,
        "generator_params": generator_parameters,
        "discriminator_params": discriminator_parameters,
        "conditional": conditional,
    }
    trainer_parameters = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "log_every_n_steps": 10,
    }

    print("Loading Model")
    model = WGAN(**model_parameters)
    trainer = get_pl_trainer(
        run_name,
        val_loader=test_loader,
        metrics=metrics,
        eval_every=10,
        log_to_wandb=True,
        config=config,
        **trainer_parameters
    )
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    logging.info("Start training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )
    logging.info("Training finished")


if __name__ == "__main__":
    main()
