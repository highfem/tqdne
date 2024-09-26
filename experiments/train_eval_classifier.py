# REF: https://github.com/dirmeier/dmvi

from tqdne.classification import LightningClassifier
from tqdne.utils import *


from pathlib import Path
from absl import flags, app
from ml_collections import config_flags
from tqdne.conf import Config
from tqdne.dataset import ClassifierDataset
from tqdne.training import get_pl_trainer
from tqdne.unet import HalfUNetClassifierModel
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score

import os
import logging
import multiprocessing

general_config = Config()

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "training configuration", lock_config=False)
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("use_last_checkpoint", False, "get the last checkpoint from the output directory")
flags.DEFINE_string("checkpoint_file", None, "checkpoint file of the previously trained model")
flags.DEFINE_string("train_datapath", str(general_config.datasetdir / general_config.data_train), "path to the training data")
flags.DEFINE_string("test_datapath", str(general_config.datasetdir / general_config.data_test), "path to the test data")
flags.DEFINE_integer("downsampling_factor", 1, "downsampling factor")
flags.DEFINE_string("outdir", str(general_config.outputdir) , "out directory, i.e., place where results are written to")
flags.mark_flags_as_required(["config"])


def main(argv):
    del argv

    batch_size = FLAGS.config.optimizer_params.batch_size

    # Update configuration parameters
    general_config.fs = general_config.fs // FLAGS.downsampling_factor
    general_config.signal_length = general_config.signal_length // FLAGS.downsampling_factor

    # Get the data representation 
    data_representation = get_data_representation(FLAGS.config, signal_length=general_config.signal_length) 
    name = f"{FLAGS.config.name}-{FLAGS.config.model.net_params.dims}D-{FLAGS.config.model.net_params.model_channels}Chan-{FLAGS.config.model.net_params.channel_mult}Mult-{FLAGS.config.model.net_params.num_res_blocks}ResBlocks-{FLAGS.config.model.net_params.num_heads}AttHeads"
    name = data_representation.get_name(FLAGS, name)
    data_repr_channels = data_representation.get_num_channels(general_config.num_channels)
    
    # Datastes
    # Bins decided based on the distribution of the data (see dataset_stats.ipynb)
    mag_bins = FLAGS.config.bins.mag
    dist_bins = FLAGS.config.bins.dist
    train_dataset = ClassifierDataset(
        h5_path=Path(FLAGS.train_datapath), 
        downsample=FLAGS.downsampling_factor,
        mag_bins=mag_bins,
        dist_bins=dist_bins,    
        data_repr=data_representation
        ) 
    test_dataset = ClassifierDataset(
        h5_path=Path(FLAGS.test_datapath), 
        downsample=FLAGS.downsampling_factor, 
        mag_bins=mag_bins,
        dist_bins=dist_bins,
        data_repr=data_representation
        )

    # Loss function with class weights
    class_weights = train_dataset.get_class_weights()
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Get the number of conditioning features
    num_classes = train_dataset.get_num_classes()
    
    # DEBUG
    if FLAGS.debug:
        logging.info("DEBUG MODE: Use only a subset of the dataset")
        train_dataset = torch.utils.data.Subset(train_dataset, range(0, 100))
        test_dataset = torch.utils.data.Subset(test_dataset, range(0, 20))
        logging.info(f"Train dataset size: {len(train_dataset)}")
        logging.info(f"Test dataset size: {len(test_dataset)}")
        batch_size = 4
        logging.info(f"Decreasing the Batch Size to: {batch_size}")
        logging.info("DEBUG: Not storing checkpoints")
        FLAGS.config.trainer_params.update({"enable_checkpointing": False})
        logging.info("DEBUG: not logging to wandb")
        FLAGS.config.trainer_params.update({"log_to_wandb": False})


    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Set the number of workers based on the number of CPU cores
    num_workers = num_cores - 1 if num_cores > 1 else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


    logging.info("Build network...")
    net = HalfUNetClassifierModel(in_channels=data_repr_channels, num_classes=num_classes, **FLAGS.config.model.net_params)
    logging.info(FLAGS.config.model.net_params)

    # Metrics
    accuracy_micro = Accuracy(task="multiclass", average='micro', num_classes=num_classes)
    accuracy_macro = Accuracy(task="multiclass", average='macro', num_classes=num_classes)
    precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
    recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
    f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)
    auroc = AUROC(task="multiclass", num_classes=num_classes)
    metrics = {
        "accuracy_micro": accuracy_micro,
        "accuracy_macro": accuracy_macro,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }

    logging.info("Build Pytorch Lightning Trainer...")
    example_input = train_dataset[0]["repr"][None, ...]
    model = LightningClassifier(
        net=net, 
        optimizer_params=FLAGS.config.optimizer_params, 
        loss=loss, 
        metrics=metrics, 
        example_input_array=example_input,
        ml_config=FLAGS.config
    )

    trainer = get_pl_trainer(
        name=name,
        task="classification",
        val_loader=test_loader,
        metrics=metrics,
        plots=[],
        limit_eval_batches=-1,
        flags=flags.FLAGS,
        **FLAGS.config.trainer_params,
    )

    logging.info("Start training...")
    if FLAGS.checkpoint_file is not None:
        checkpoint = Path(FLAGS.checkpoint_file)
    elif FLAGS.use_last_checkpoint:
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


