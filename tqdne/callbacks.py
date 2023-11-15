import os
import pytorch_lightning as L
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdne.ganutils.evaluation import evaluate_model


class LogGanCallback(L.callbacks.Callback):
    def __init__(
        self, wandb_logger, dataset, every=1, timedelta=0.05, n_waveforms = 5
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.attr = dataset.get_attr()
        self.wfs = dataset.get_wfs()
        self.cur_epoch = 0
        self.every = every
        self.timedelta = 0.05
        self.n_waveforms = n_waveforms
    
        self.datasize = len(self.attr)
        self.attr["norm_dist"] = (self.attr["dist"] - self.attr["dist"].min()) / (self.attr["dist"].max() - self.attr["dist"].min())
        self.attr["norm_mag"] = (self.attr["mag"] - self.attr["mag"].min()) / (self.attr["mag"].max() - self.attr["mag"].min())

    def get_sample_from_conds(self, pl_module, mag, dist):
        vc_list = [
            dist * torch.ones(self.n_waveforms, 1),
            mag * torch.ones(self.n_waveforms, 1),
        ]
        vc_list = [i.to(pl_module.device) for i in vc_list]

        syn_data, syn_scaler = pl_module.sample(self.n_waveforms, *vc_list)
        syn_data = syn_data.squeeze().detach().cpu().numpy()
        syn_data = syn_data * syn_scaler.detach().cpu().numpy()

        # synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
        # sd_mean = np.mean(synthetic_data_log, axis=0)
        sd_mean = np.mean(syn_data, axis=0)

        y = np.exp(sd_mean)

        nt = sd_mean.shape[0]
        tt = self.timedelta * np.arange(0, nt)
        return tt, y

    def log_sample_image(self, trainer, pl_module):
        n = np.random.randint(0, self.datasize, size=(2,))
        fig, axis = plt.subplots(2, 1, figsize=(12, 6))
        for cnt, i in enumerate(n):
            sample_dist = self.attr["dist"].loc[i]
            sample_mag = self.attr["mag"].loc[i]
            sample_norm_dist = self.attr["norm_dist"].loc[i]
            sample_norm_mag = self.attr["norm_mag"].loc[i]
            tt, y = self.get_sample_from_conds(
                pl_module,
                sample_norm_mag,
                sample_norm_dist,
            )
            axis[cnt].legend()
            axis[cnt].set_xlabel("Time [s]")
            axis[cnt].set_ylabel("Log-Amplitude")
            axis[cnt].set_title(f"Dist: {sample_dist:.2f}km, Mag: {sample_mag:.2f}")
            axis[cnt].semilogy(
                tt,
                y,
                "-",
                label="Synthetic",
                alpha=0.8,
                lw=0.5,
            )
            axis[cnt][0].semilogy(
                tt,
                self.wfs[i],
                "-",
                label="Real",
                alpha=0.8,
                lw=0.5,
            )
        wandb.log({"LogAmplitude-x-Time": fig})
        plt.close("all")
        plt.clf()
        plt.cla()

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.cur_epoch + 1) % self.every != 0:
            return
        self.cur_epoch += 1
        self.log_sample_image(trainer, pl_module)

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
