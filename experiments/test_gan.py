from tqdne.gan_lightning import GAN
from tqdne.callbacks import LogGanCallback

# from tqdne.ganutils.data_utils import SeisData
from tqdne.ganutils.dataset import WFDataModule, WaveformDataset
import pytorch_lightning as L

# from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import WandbLogger
import torch
from tqdne.conf import Config
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    # Setting up Args
    config = Config()
    data_file = config.datasetdir / Path(config.data_waveforms)
    attr_file = config.datasetdir / Path(config.data_attributes)
    condv_names = ["dist", "mag"]
    plot_format = "pdf"

    batch_size = 128
    sample_rate = 100
    gp_lambda = 10.0
    n_critic = 10
    beta_1 = 0.0
    beta_2 = 0.9
    learning_rate = 1e-4
    latent_dim = 100
    time_delta = 0.05
    discriminator_size = 1000
    frac_train = 0.8

    print("Loading data...")
    # sdat_train = SeisData(
    #     data_file=data_file,
    #     attr_file=attr_file,
    #     batch_size=batch_size,
    #     sample_rate=sample_rate,
    #     v_names=condv_names,
    #     cut=frac_train,
    # )
    wfs = np.load(data_file)
    df_attr = pd.read_csv(attr_file)
    dm = WFDataModule(data_file, attr_file, condv_names, batch_size, frac_train)
    # dataset = WaveformDataset(wfs, df_attr, condv_names)
    # waveform_dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True, num_workers=7
    # )

    model = GAN(
        latent_dim=latent_dim,
        reg_lambda=gp_lambda,
        n_critics=n_critic,
        batch_size=batch_size,
        lr=learning_rate,
        b1=beta_1,
        b2=beta_2,
        # time_delta=time_delta,
        # discriminator_size=discriminator_size,
        # plot_format=plot_format,
    )
    # mlf_logger = MLFlowLogger(
    #     experiment_name="lightning_logs", tracking_uri="file:./mlruns"
    # )
    wandb_logger = WandbLogger(project="tqdne")
    print("--------------------------- TRAIN ------------------------------")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=300,
        logger=wandb_logger,
        callbacks=[LogGanCallback(wandb_logger, dataset=dm)],
    )
    trainer.fit(model, dm)


main()
