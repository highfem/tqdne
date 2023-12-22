from email import generator
from tqdne.gan_lightning import GAN
# from tqdne.ganutils.data_utils import SeisData
from tqdne.wfdataset import WFDataModule
from tqdne.utils.model_utils import get_last_checkpoint
from tqdne.models.gan import Discriminator
from tqdne.training import get_pl_trainer
from tqdne.callbacks.sample_callback import SimplePlotCallback
from tqdne.simple_dataset import StationarySignalDM

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

    resume = False
    max_epochs = 400
    batch_size = 128
    frac_train = 0.8
    wfs_expected_size = 1024
    latent_dim = 128
    channels = 1

    print("Loading data...")
    # dm = WFDataModule(data_file, attr_file, wfs_expected_size, condv_names, batch_size, frac_train)
    dataset_size = 10000
    dm = StationarySignalDM(dataset_size, wfs_expected_size, batch_size, frac_train)

    optimizer_parameters = {
        "lr": 1e-4,
        "momentum": 0.5,
        #"b1": 0.9,
        #"b2": 0.999,
    }
    generator_parameters = {
        "num_variables": 0,
        "latent_dim": latent_dim,
        "encoding_L":4,
        "out_size": wfs_expected_size,
        "out_channels": channels,
        "step_channels":32,
        "batchnorm": False,
    }
    discriminator_parameters = {
        "num_variables": 0,
        "in_size": wfs_expected_size,
        "encoding_L":4,
        "in_channels":channels,
        "step_channels":32,
        "batchnorm": False,
    }
    model_parameters = {
        "waveform_size": wfs_expected_size,
        "reg_lambda": 10.0,
        "n_critics": 3,
        "batch_size": batch_size,
        "optimizer_params": optimizer_parameters,
        "generator_params": generator_parameters,
        "discriminator_params": discriminator_parameters,
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
        "n_samples": 50,        
    }
    # specific_callbacks = [
    #     MetricsCallback(**metrics_callback_parameters),
    #     PlotCallback(**plot_callback_parameters)
    # ]
    specific_callbacks = [
        SimplePlotCallback(**plot_callback_parameters)
    ]

    print("Loading Model")
    model = GAN(**model_parameters)
    trainer = get_pl_trainer("WGAN", dm, project="tqdne-dummydataset", specific_callbacks=specific_callbacks, **trainer_parameters)
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    
    trainer.fit(model, dm, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
