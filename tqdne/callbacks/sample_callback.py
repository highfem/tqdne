import torch
from pytorch_lightning.callbacks import Callback
import time
import matplotlib.pyplot as plt
import numpy as np
import wandb 

class SimplePlotCallback(Callback):
    def __init__(
        self, dataset, every=5, n_waveforms = 5, conditional=False
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataloader = dataset.val_dataloader()
        self.every = every
        # self.n_waveforms = n_waveforms
        self.cond = conditional

    def get_sample(self, pl_module, conds=None):
        with torch.no_grad():
            syn_data = pl_module.sample(1, conds)
            # print("syn_data shape", syn_data.shape)
            syn_data = self.dataset.denormalize(syn_data[:, 0], syn_data[:, 1])
            y = np.mean(syn_data, axis=0).reshape(-1)
            nt = y.shape[0]
            tt = np.arange(0, nt)
        return tt, y

    def log_sample_image(self, trainer, pl_module):
        wave, conds = None, None
        with torch.no_grad():
            if self.cond:
                wave, conds = next(iter(self.dataloader))
                # print("old wave shape", wave.shape)
                wave = self.dataset.denormalize(wave[:, 0], wave[:, 1])
                ix = np.random.randint(0, wave.shape[0], 1)
                wave = wave[ix].view(-1)
                # print("new wave shape", wave.shape)
                conds = conds[ix].to(pl_module.device)
                # print(wave.shape, conds.shape)
                # conds = np.random.choice(conds.reshape(-1), (self.n_waveforms, 1))
                # conds = torch.from_numpy(conds).to(pl_module.device)
            tt, y = self.get_sample(pl_module, conds)
            # print("tt shape", tt.shape)
            # print("y shape", y.shape)
            # print("--------------------------")
            fig, axis = plt.subplots(1, 1)
            if self.cond:
                axis.plot(
                    tt,
                    wave,
                    "-",
                    label="Real",
                    alpha=0.8,
                    lw=0.5,
                )
            axis.plot(
                tt,
                y,
                "-",
                label="Synthetic",
                alpha=0.8,
                lw=0.5,
            )
            p = conds.view(-1)[0] if self.cond else -1
            axis.set_title("Synthetic waveforms p = {}".format(p))
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Amplitude")
            axis.legend()
            # fig = wandb.plot.line_series(xs=tt, ys=[y], keys=["synthetic"], title="Synthetic waveforms")
            wandb.log({"sample wave": fig})
            plt.close("all")
            plt.clf()
            plt.cla()

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1)% self.every == 0:
            self.log_sample_image(trainer, pl_module)