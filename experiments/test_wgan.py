from tqdne.wgan_lightning import WGAN
from tqdne.model_utils import get_last_checkpoint
from tqdne.training import get_pl_trainer
from tqdne.callbacks import SimplePlotCallback
from tqdne.simple_dataset import StationarySignalDM

# from pytorch_lightning.loggers import MLFlowLogger
from tqdne.conf import Config
from pathlib import Path

def main():
    # Setting up Args
    resume = False
    max_epochs = 400
    batch_size = 64
    frac_train = 0.8
    wfs_expected_size = 512
    latent_dim = 100
    channels = 1

    print("Loading data...")
    # dm = WFDataModule(data_file, attr_file, wfs_expected_size, condv_names, batch_size, frac_train)
    dataset_size = 10000
    dm = StationarySignalDM(dataset_size, wfs_expected_size, batch_size, frac_train)

    optimizer_parameters = {
        "lr": 1e-4,
        # "momentum": 0.5,
        "b1": 0.9,
        "b2": 0.99,
    }
    generator_parameters = {
        "latent_dim": latent_dim,
        "wave_size": wfs_expected_size,
        "channels": channels,
        "dim": 16,
    }
    discriminator_parameters = {
        "wave_size": wfs_expected_size,
        "channels": 1,
        "dim": 16,
    }
    model_parameters = {
        "waveform_size": wfs_expected_size,
        "reg_lambda": 10.0,
        "n_critics": 1,
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
    model = WGAN(**model_parameters)
    trainer = get_pl_trainer("WGAN", dm, project="tqdne-dummydataset", specific_callbacks=specific_callbacks, **trainer_parameters)
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    
    trainer.fit(model, dm, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
