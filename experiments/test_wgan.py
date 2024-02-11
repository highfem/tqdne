import logging
from pathlib import Path


from torch.utils.data import DataLoader

from tqdne.dataset import WaveformDataset
from tqdne.gan import WGAN
from tqdne.utils import get_last_checkpoint
from tqdne.training import get_pl_trainer
from tqdne.metric import SamplePlot, RepresentationInversion
from tqdne.representations import LogMaxEnvelope
from tqdne.conf import Config

def main():
    # Setting up Args
    config = Config()
    resume = False
    conditional = True
    max_epochs = 800
    batch_size = 64
    wfs_expected_size = 1024
    latent_dim = 128
    encoding_L = 4

    logging.info("Loading data...")
    train_path = config.datasetdir / Path(config.data_upsample_train)
    test_path = config.datasetdir / Path(config.data_upsample_test)
    envelope_representation = LogMaxEnvelope(config)
    train_dataset = WaveformDataset(train_path, envelope_representation, reduced=wfs_expected_size)
    test_dataset = WaveformDataset(test_path, envelope_representation, reduced=wfs_expected_size)

    channels = train_dataset[0]["high_res"].shape[0]
    logging.info(f"Channels: {channels}")
    num_conditional_vars = train_dataset[0]["cond"].shape[0]
    logging.info(f"Number of conditional variables: {num_conditional_vars}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)
    condv_names = ["dist", "mag"]

    logging.info("Set parameters...")
    plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels // 2)]
    plots = [
        RepresentationInversion(metric, envelope_representation) for metric in plots
    ]
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
        "dim": 32,
    }
    discriminator_parameters = {
        "wave_size": wfs_expected_size,
        "in_channels": channels,
        "encoding_L": encoding_L,
        "num_cond_vars": num_conditional_vars,
        "dim": 32,
    }
    model_parameters = {
        "reg_lambda": 10.0,
        "n_critics": 4,
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
        "WGAN",
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
