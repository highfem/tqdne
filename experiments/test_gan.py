from tqdne.gan_lightning import GAN

# from tqdne.ganutils.data_utils import SeisData
from tqdne.ganutils.dataset import WFDataModule
from tqdne.utils import get_last_checkpoint
from tqdne.training import get_pl_trainer

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
    max_epochs = 50
    batch_size = 256
    frac_train = 0.01

    print("Loading data...")
    dm = WFDataModule(data_file, attr_file, condv_names, batch_size, frac_train)

    model_parameters = {
        "reg_lambda": 10.0,
        "latent_dim": 100,
        "n_critics": 10,
        "batch_size": batch_size,
        "lr": 1e-4,
        "b1": 0.0,
        "b2": 0.9,
    }
    trainer_parameters = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "log_every_n_steps": 10,
    }
    log_callback_parameters = {
        "every": 1,
        "timedelta": 0.05,
        "n_waveforms": 72 * 5,
    }

    print("Loading Model")
    model = GAN(**model_parameters)
    trainer = get_pl_trainer("WGAN", dm, callback_pars=log_callback_parameters, **trainer_parameters)
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    
    trainer.fit(model, dm, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()
