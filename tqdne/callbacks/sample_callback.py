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
        self.every = every
        self.n_waveforms = n_waveforms
        self.cond = conditional

    def get_sample(self, pl_module):
        with torch.no_grad():
            if self.cond:
                _, conds = next(iter(self.dataset))
                conds = np.random.choice(conds.reshape(-1), (self.n_waveforms, 1))
                conds = torch.from_numpy(conds).to(pl_module.device)
                syn_data = pl_module.sample(self.n_waveforms, conds)
            else:
                syn_data = pl_module.sample(self.n_waveforms)
            y = np.mean(syn_data, axis=0).reshape(-1)
            nt = y.shape[0]
            tt = np.arange(0, nt)
        return tt, y

    def log_sample_image(self, trainer, pl_module):
        tt, y = self.get_sample(pl_module)
        fig, axis = plt.subplots(1, 1)
        axis.plot(
            tt,
            y,
            "-",
            label="Synthetic",
            alpha=0.8,
            lw=0.5,
        )
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