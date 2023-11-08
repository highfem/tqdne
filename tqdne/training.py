
from pathlib import Path
# from tqdne.lightning import LogCallback
from tqdne.callbacks import LogGanCallback

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

from tqdne.conf import OUTPUTDIR, PROJECT_NAME

import pytorch_lightning as pl




def get_pl_trainer(name, val_loader, project=PROJECT_NAME, callback_pars = {}, **trainer_params):

    # 1. Wandb Logger
    wandb_logger = WandbLogger(project=project) # add project='projectname' to log to a specific project

    # 2. Learning Rate Logger
    lr_logger = LearningRateMonitor()
    # 3. Set Early Stopping
    # early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
    # 4. saves checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(dirpath=OUTPUTDIR / Path(name), filename='{name}_{epoch}-{val_loss:.2f}',
                                        monitor='val_loss', mode='min', save_top_k=5)
    # 5. My custom callback
    log_callback = LogGanCallback(wandb_logger, val_loader, **callback_pars)

    output_dir = (OUTPUTDIR/Path(name))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(**trainer_params, logger=wandb_logger, callbacks=[lr_logger, log_callback, checkpoint_callback], 
                        default_root_dir=output_dir ) 
    
    return trainer
