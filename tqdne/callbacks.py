import os
from random import sample
import pytorch_lightning as L
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

class MetricsCallback(L.callbacks.Callback):
    def __init__(self, dataset, every=1, n_samples=500):
        super().__init__()
        self.dataset = dataset
        self.wfs = dataset.get_wfs()
        self.attr = dataset.get_attr()
        self.attr["norm_dist"] = self.attr["dist"]/self.attr["dist"].max()
        self.attr["norm_mag"] = self.attr["mag"]/self.attr["mag"].max()  
        self.every = every
        self.num_samples = n_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (pl_module.cur_epoch + 1) % self.every != 0:
            return
        wfs, lcn, vc_i = batch

        syn_data, syn_scaler = pl_module.sample(wfs.size(0), *vc_i)
        syn_data = syn_data.squeeze().detach().cpu().numpy()
        syn_scaler = syn_scaler.detach().cpu().numpy()
        # syn_data = syn_data * syn_scaler
        syn_data = self.dataset.getSignalFromDecomp(syn_data, syn_scaler)

        wfs = wfs.detach().cpu().numpy()
        lcn = lcn.detach().cpu().numpy().reshape(-1, 1)
        real_wfs = self.dataset.getSignalFromDecomp(wfs, lcn)
        
        metrics = np.mean(np.sum((syn_data - real_wfs) ** 2, axis = 1), axis=0)
        wandb.log({"Mean Squared Difference": metrics})    

class PlotCallback(L.callbacks.Callback):
    def __init__(
        self, dataset, every=1, n_waveforms = 5
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.attr = dataset.get_attr()
        self.wfs = dataset.get_wfs()
        self.cur_epoch = 0
        self.every = every
        self.n_waveforms = n_waveforms
    
        self.datasize = len(self.attr)
        #self.attr["norm_dist"] = (self.attr["dist"] - self.attr["dist"].min()) / (self.attr["dist"].max() - self.attr["dist"].min())
        #self.attr["norm_mag"] = (self.attr["mag"] - self.attr["mag"].min()) / (self.attr["mag"].max() - self.attr["mag"].min())
        self.attr["norm_dist"] = self.attr["dist"]/self.attr["dist"].max()
        self.attr["norm_mag"] = self.attr["mag"]/self.attr["mag"].max()  

    def get_sample_from_conds(self, pl_module, mag, dist):
        vc_list = [
            dist * torch.ones(self.n_waveforms, 1),
            mag * torch.ones(self.n_waveforms, 1),
        ]
        vc_list = [i.to(pl_module.device) for i in vc_list]

        syn_data, syn_scaler = pl_module.sample(self.n_waveforms, *vc_list)
        syn_data = syn_data.squeeze().detach().cpu().numpy()
        syn_scaler = syn_scaler.detach().cpu().numpy()
        syn_data = self.dataset.getSignalFromDecomp(syn_data, syn_scaler)
        # syn_data = syn_data * syn_scaler

        # synthetic_data_log = np.log(np.abs(np.array(syn_data + 1e-10)))
        # sd_mean = np.mean(synthetic_data_log, axis=0)
        sd_mean = np.mean(syn_data, axis=0)

        # y = np.exp(sd_mean)
        y = sd_mean

        nt = sd_mean.shape[0]
        tt = np.arange(0, nt)
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
            axis[cnt].plot(
                tt,
                y,
                "-",
                label="Synthetic",
                alpha=0.8,
                lw=0.5,
            )
            axis[cnt].plot(
                tt,
                self.wfs[i],
                "-",
                label="Real",
                alpha=0.8,
                lw=0.5,
            )
            axis[cnt].set_xlabel("Time [s]")
            axis[cnt].set_ylabel("Log-Amplitude")
            axis[cnt].set_title(f"Dist: {sample_dist:.2f}km, Mag: {sample_mag:.2f}")
            axis[cnt].legend()
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
