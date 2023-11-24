from tqdne.gan_lightning import GAN

# from tqdne.ganutils.data_utils import SeisData
from tqdne.ganutils.dataset import WFDataModule
from tqdne.model_utils import get_last_checkpoint
from tqdne.training import get_pl_trainer
from tqdne.callbacks import PlotCallback, MetricsCallback

# from pytorch_lightning.loggers import MLFlowLogger
from tqdne.conf import Config
from pathlib import Path

def main():
    # Setting up Args
    config = Config()
    data_file = config.datasetdir / Path(config.data_waveforms)
    attr_file = config.datasetdir / Path(config.data_attributes)
    condv_names = ["dist", "mag"]
    # plot_format = "pdf"

    resume = True
    max_epochs = 100
    batch_size = 128
    frac_train = 0.8
    wfs_expected_size = 1024

    print("Loading data...")
    dm = WFDataModule(data_file, attr_file, wfs_expected_size, condv_names, batch_size, frac_train)

    model_parameters = {
        "waveform_size": wfs_expected_size,
        "reg_lambda": 10.0,
        "latent_dim": 100,
        "n_critics": 10,
        "batch_size": batch_size,
        "lr": 1e-4,
        "b1": 0.9,
        "b2": 0.999,
    }
    trainer_parameters = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "log_every_n_steps": 10,
    }
    plot_callback_parameters = {
        "dataset": dm,
        "every": 1,
        "n_waveforms": 3,
    }
    metrics_callback_parameters = {
        "dataset": dm,
        "every": 1,
        "n_samples": 500,        
    }
    specific_callbacks = [
        MetricsCallback(**metrics_callback_parameters),
        PlotCallback(**plot_callback_parameters)
    ]

    print("Loading Model")
    model = GAN(**model_parameters)
    trainer = get_pl_trainer("WGAN", dm, specific_callbacks=specific_callbacks, **trainer_parameters)
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    
    trainer.fit(model, dm, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
