# REF: https://github.com/dirmeier/dmvi

from pathlib import Path
from absl import flags, app
from ml_collections import config_flags
from diffusers import DDIMScheduler, DDPMScheduler
from tqdne.conf import Config
from tqdne.consistency_model import LightningConsistencyModel
from tqdne.dataset import EnvelopeDataset
from tqdne.diffusion import LightningDiffusion
from tqdne.metric import *
from tqdne.plot import *
from tqdne.representations import *

from tqdne.training import get_pl_trainer
from tqdne.unet import UNetModel
from tqdne.utils import get_last_checkpoint
from torch.utils.data import DataLoader
import os
import logging
import multiprocessing

general_config = Config()

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "training configuration", lock_config=False)
flags.DEFINE_string("checkpoint", None, "checkpoint file of the previously trained model")
flags.DEFINE_string("train_datapath", str(general_config.datasetdir / general_config.data_train), "path to the training data")
flags.DEFINE_string("test_datapath", str(general_config.datasetdir / general_config.data_test), "path to the test data")
flags.DEFINE_string("outdir", str(general_config.outputdir) , "out directory, i.e., place where results are written to")
flags.mark_flags_as_required(["config"])

def main(argv):
    del argv

    name = f"{FLAGS.config.name}-{FLAGS.config.model.net_params.dims}D_{FLAGS.config.data_repr.name}-{FLAGS.config.data_repr.params.env_function}-{FLAGS.config.data_repr.params.env_transform}-{FLAGS.config.data_repr.params.env_transform_params}".replace(" ", "").replace("\n", "")

    t = general_config.signal_length
    batch_size = FLAGS.config.model.optimizer_params.batch_size

    data_representation = _get_data_representation(FLAGS.config.data_repr.name, FLAGS.config.data_repr.params, general_config.signal_statistics)

    train_dataset = EnvelopeDataset(h5_path=Path(FLAGS.train_datapath), cut=t, data_repr=data_representation) 
    test_dataset = EnvelopeDataset(h5_path=Path(FLAGS.test_datapath), cut=t, data_repr=data_representation)

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Set the number of workers based on the number of CPU cores
    num_workers = num_cores - 1 if num_cores > 1 else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    repr_channels = train_dataset[0]["representation"].shape[0]

    logging.info("Build network...")
    net = UNetModel(in_channels=repr_channels, out_channels=repr_channels, **FLAGS.config.model.net_params)
    logging.info(net.config)

    if FLAGS.config.name == "consistency-model":
        logging.info("Build consistency model...")
        model = LightningConsistencyModel(net, scheduler, **FLAGS.config.model)
    elif FLAGS.config.name == "ddpm" or FLAGS.config.name == "ddim":
        logging.info("Build diffusion model...")
        scheduler = DDPMScheduler(**FLAGS.config.model.scheduler_params) if FLAGS.config.name == "ddpm" else DDIMScheduler(**FLAGS.config.model.scheduler_params)
        logging.info(scheduler.config)
        logging.info("Build lightning module...")

        FLAGS.model.optimizer_params.update(
        {
            "n_train": len(train_dataset) // batch_size,
            "max_epochs": FLAGS.trainer_params.max_epochs,
        }
        )
        model = LightningDiffusion(
            net,
            scheduler,
            prediction_type=FLAGS.config.model.scheduler_params.prediction_type,
            optimizer_params=FLAGS.model.optimizer_params,
            low_res_input=False,
            cond_input=True,
        )
    else:
        raise ValueError(f"Unknown model name: {FLAGS.config.name}")
    

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        metrics=_get_metrics_list(FLAGS.config.metrics),
        plots=_get_plots_list(FLAGS.config.plots),
        max_steps=FLAGS.trainer_params.max_epochs * len(train_loader),
        eval_every=5,
        limit_eval_batches=-1,
        log_to_wandb=True,
        **FLAGS.trainer_params,
    )

    if FLAGS.checkpoint is not None:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None   
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")

if __name__ == "__main__":
    app.run(main)    


def _get_metrics_list(metrics_config):
    metrics = []
    for metric, v in metrics_config.items():
        if v == -1:
            channels = [c for c in range(general_config.num_channels)]
        else:
            channels = [v]    
        if metric == "psd":
            for c in channels:
                metrics.append(PowerSpectralDensity(fs=general_config.fs, channel=c))
        elif metric == "mse":
            for c in channels:
                metrics.append(MeanSquaredError(channel=c))
        else:
            raise ValueError(f"Unknown metric name: {metric}")
    return metrics    

def _get_plots_list(plots_config, metrics):
    plots = []
    for plot, v_plot in plots_config.items():
        if plot == "bin":
            if v_plot.metrics == "all":
                for m in metrics:
                    plots.append(BinPlot(metric=m, num_mag_bins=v_plot.num_mag_bins, num_dist_bins=v_plot.num_dist_bins))
            elif v_plot.metrics == "channels-avg":
                #for i in range(0, len(metrics), general_config.num_channels):
                #    metric_avg = np.mean(metrics[i:i+general_config.num_channels])
                #    plots.append(BinPlot())
                # TODO: it should be handled by BinPlot itself 
                raise NotImplementedError("channels-avg not implemented yet")
        else:
            if v_plot == -1:
                channels = [c for c in range(general_config.num_channels)]
            else:
                channels = [v_plot]    
            if plot == "psd":
                for c in channels:
                    plots.append(PowerSpectralDensity(fs=general_config.fs, channel=c))
            elif plot == "mse":
                for c in channels:
                    plots.append(MeanSquaredError(channel=c))
            else:
                raise ValueError(f"Unknown metric name: {plot}")
    return plots    

def _get_data_representation(repr_name, repr_params, signal_stats_dict):
    if repr_name == "SignalWithEnvelope":
        return SignalWithEnvelope(repr_params, dataset_stats_dict=signal_stats_dict)
    else:
        raise ValueError(f"Unknown representation name: {repr_name}")

