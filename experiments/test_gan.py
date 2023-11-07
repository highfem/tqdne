from tqdne.gan import GAN, LogGanCallback
from tqdne.ganutils.data_utils import SeisData
import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
import torch
import pandas as pd
import logging
from tqdne.conf import Config
from pathlib import Path

import numpy as np


class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, attr_file, v_names, cut=None):
        print("Loading data ...")
        wfs = np.load(data_file)
        if cut:
            wfs = wfs[: int(wfs.shape[0] * cut)]
        print("Loaded samples: ", wfs.shape[0])
        self.ws = wfs.copy()
        print("normalizing data ...")
        wfs_norm = np.max(np.abs(wfs), axis=1)  # 2)
        self.cnorms = wfs_norm.copy()
        wfs_norm = wfs_norm[:, np.newaxis]
        self.wfs = wfs / wfs_norm
        lc_m = np.log10(self.cnorms)
        max_lc_m = np.max(lc_m)
        min_lc_m = np.min(lc_m)
        self.ln_cns = 2.0 * (lc_m - min_lc_m) / (max_lc_m - min_lc_m) - 1.0

        df = pd.read_csv(attr_file)
        # store All attributes
        df_meta_all = df.copy()
        # store pandas dict as attribute
        df_meta = df[v_names]

        self.vc_lst = []
        for v_name in v_names:
            print("---------", v_name, "-----------")
            print("min " + v_name, df_meta[v_name].min())
            print("max " + v_name, df_meta[v_name].max())
            # 1. rescale variables to be between 0,1
            v = df_meta[v_name].to_numpy()
            v = (v - v.min()) / (v.max() - v.min())

            # reshape conditional variables
            vc = np.reshape(v, (v.shape[0], 1))
            print("vc shape", vc.shape)
            # 3. store conditional variable
            self.vc_lst.append(vc)

        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[:, :])
        self.vc_lst = vc_b

    def __len__(self):
        return self.ws.shape[0]

    def __getitem__(self, index):
        vc_b = [v[index, :] for v in self.vc_lst]
        return (self.wfs[index], self.ln_cns[index], vc_b)


def main():
    # Setting up Args
    config = Config()
    data_file = config.datasetdir / Path(config.data_waveforms)
    attr_file = config.datasetdir / Path(config.data_attributes)
    condv_names = ["dist", "mag"]
    plot_format = "pdf"

    batch_size = 4
    sample_rate = 100
    gp_lambda = 10.0
    n_critic = 10
    beta_1 = 0.0
    beta_2 = 0.9
    learning_rate = 1e-4
    latent_dim = 100
    time_delta = 0.05
    discriminator_size = 1000
    frac_train = 0.005

    print("Loading data...")
    sdat_train = SeisData(
        data_file=data_file,
        attr_file=attr_file,
        batch_size=batch_size,
        sample_rate=sample_rate,
        v_names=condv_names,
        cut=frac_train,
    )
    dataset = WaveformDataset(data_file, attr_file, condv_names, cut=frac_train)
    waveform_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=7
    )

    model = GAN(
        latent_dim=latent_dim,
        reg_lambda=gp_lambda,
        n_critics=n_critic,
        batch_size=batch_size,
        lr=learning_rate,
        b1=beta_1,
        b2=beta_2,
        time_delta=time_delta,
        discriminator_size=discriminator_size,
        plot_format=plot_format,
    )
    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri="file:./mlruns"
    )
    print("--------------------------- TRAIN ------------------------------")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=2,
        logger=mlf_logger,
        callbacks=[LogGanCallback(mlf_logger, sdat_train)],
    )
    trainer.fit(model, waveform_dataloader)


main()
