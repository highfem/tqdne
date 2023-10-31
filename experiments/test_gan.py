from tqdne.gan import GAN
from tqdne.data_utils import SeisData
import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
import torch

import numpy as np

class WaveformDataset(torch.utils.data.Dataset):
    pass

def main():
    # Setting up Args
    data_file = "/home/fcomoreira/projects/tqdne/datasets/japan/waveforms.py"
    attr_file = "/home/fcomoreira/projects/tqdne/datasets/japan/attributes.csv"
    batch_size = 16
    sample_rate = 100
    gp_lambda=10.0
    n_critic = 10
    beta_1 = 0.0
    beta_2 = 0.9
    learning_rate = 1e-4
    latent_dim = 100
    time_delta = 0.05
    discriminator_size = 1000
    frac_train = 0.8


    condv_names = ["dist", "mag"]
    # total number of training samples
    f = np.load(data_file)
    n_samples = len(f)
    del f

    # get all indexes
    ix_all = np.arange(n_samples)
    # get training indexes
    n_train = int(n_samples * frac_train)
    ix_train = np.random.choice(ix_all, size=n_train, replace=False, )
    ix_train.sort()
    # get validation indexes
    ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
    ix_val.sort()

    sdat_train = SeisData(
        data_file=data_file,
        attr_file=attr_file,
        batch_size=batch_size,
        sample_rate=sample_rate,
        v_names=condv_names,
        isel=ix_train,
    )

    sdat_val = SeisData(
        data_file=data_file,
        attr_file=attr_file,
        batch_size=batch_size,
        sample_rate=sample_rate,
        v_names=condv_names,
        isel=ix_val,
    )

    model = GAN(
        latent_dim=latent_dim,
        reg_lambda=gp_lambda, 
        n_critic=n_critic,
        lr=learning_rate,
        b1=beta_1,
        b2=beta_2,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=5,
    )

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    trainer.fit(model, sdat_trian)


main()
