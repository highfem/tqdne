import os
import pytorch_lightning as L
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdne.ganutils.evaluation import evaluate_model


class LogGanCallback(L.callbacks.Callback):
    def __init__(
        self, wandb_logger, dataset, every=1, timedelta=0.05, n_waveforms=72 * 5
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.scores = []
        self.scores_val = []
        self.attr = dataset.get_attr()
        self.wfs = dataset.get_wfs()
        self.cur_epoch = 0
        self.every = every
        self.timedelta = 0.05
        self.n_waveforms = n_waveforms

    def get_cond_var_bins(self, num_bins=10):
        # Add slight padding to boundaries in both directions in order to include
        # values that land on the boundaries of the bins
        dist_min = self.attr["dist"].min() - 1e-5
        dist_max = self.attr["dist"].max() + 1e-5
        dist_step_size = (dist_max - dist_min) / num_bins
        dist_bins = np.arange(
            dist_min, dist_max + dist_step_size / 2.0, step=dist_step_size
        )

        mag_min = self.attr["mag"].min() - 1e-5
        mag_max = self.attr["mag"].max() + 1e-5
        mag_step_size = (mag_max - mag_min) / num_bins
        mag_bins = np.arange(mag_min, mag_max + mag_step_size / 2.0, step=mag_step_size)

        return {"dist_bins": dist_bins, "mag_bins": mag_bins}

    def get_sample_from_conds(self, pl_module, mag, dist):
        dist_min = self.attr["dist"].min()
        dist_max = self.attr["dist"].max()
        dist_mean = self.attr["dist"].mean()

        mag_min = self.attr["mag"].min()
        mag_max = self.attr["mag"].max()
        mag_mean = self.attr["mag"].mean()
        vc_list = [
            dist / dist_max * torch.ones(self.n_waveforms, 1),
            mag / mag_max * torch.ones(self.n_waveforms, 1),
        ]
        vc_list = [i.to(pl_module.device) for i in vc_list]

        syn_data, syn_scaler = pl_module.sample(self.n_waveforms, *vc_list)
        syn_data = syn_data.squeeze().detach().cpu().numpy()
        syn_data = syn_data * syn_scaler.detach().cpu().numpy()

        synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
        sd_mean = np.mean(synthetic_data_log, axis=0)

        y = np.exp(sd_mean)

        nt = synthetic_data_log.shape[1]
        tt = self.timedelta * np.arange(0, nt)
        return tt, y

    def log_sample_image(self, trainer, pl_module):
        cond_var_bins = self.get_cond_var_bins(num_bins=10)

        dist_min = self.attr["dist"].min()
        dist_max = self.attr["dist"].max()
        dist_mean = self.attr["dist"].mean()

        mag_min = self.attr["mag"].min()
        mag_max = self.attr["mag"].max()
        mag_mean = self.attr["mag"].mean()
        dists = [dist_min, dist_mean, dist_max]
        mags = [mag_min, mag_mean, mag_max]
        for dist in dists:
            for mag in mags:
                tt, y = self.get_sample_from_conds(
                    pl_module,
                    mag,
                    dist,
                )
                plt.semilogy(
                    tt,
                    y,
                    "-",
                    label=f"Dist: {dist:.2f}km, Mag: {mag:.2f}",
                    alpha=0.8,
                    lw=0.5,
                )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Log-Amplitude")
        wandb.log({"LogAmplitude-x-Time": plt})
        plt.close("all")
        plt.clf()
        plt.cla()

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.cur_epoch + 1) % self.every != 0:
            return
        self.cur_epoch += 1
        self.log_sample_image(trainer, pl_module)
