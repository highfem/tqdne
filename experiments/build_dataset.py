from tqdne.model import MLP 
from torch.utils.data import DataLoader
from tqdne.dataset import load_mnist
from tqdne.lightning import LightningClassifier
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# output directory
from tqdne.conf import OUTPUTDIR
from tqdne.utils import load_model
import logging


resume = True

# hyperparameters
batch_size = 64
num_epochs = 5
learning_rate = 1e-3
hidden_dim = 256
num_classes = 10
n_layers = 3
input_dim = 28*28

name = 'mnist-mlp'


# load data
logging.info('Loading data')
train, val, test = load_mnist()

train_loader, val_loader, test_loader = DataLoader(train, batch_size=batch_size), DataLoader(val, batch_size=batch_size), DataLoader(test, batch_size=batch_size)

# define model
logging.info('Defining model')
net = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes, n_layers=n_layers, use_softmax=True)
model = LightningClassifier(net, lr_rate=learning_rate)

if resume:
    path_last = OUTPUTDIR/Path(name)/'last.ckpt'
    if path_last.exists():
        logging.info('Resuming from last checkpoint: %s', path_last)
        model = load_model(LightningClassifier, path_last) # Load best model
        

logging.info('Building trainer')
# 1. Wandb Logger
WANDB_MODE = "disabled" # "disabled" or "online"
wandb_logger = WandbLogger(offline=True) # add project='projectname' to log to a specific project

# 2. Learning Rate Logger
lr_logger = LearningRateMonitor()
# 3. Set Early Stopping
early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
# 4. saves checkpoints to 'model_path' whenever 'val_loss' has a new min
checkpoint_callback = ModelCheckpoint(dirpath=OUTPUTDIR / Path(name), filename='{name}_{epoch}-{val_loss:.2f}',
                                      monitor='val_loss', mode='min', save_top_k=5)

(OUTPUTDIR/Path(name)).mkdir(parents=True, exist_ok=True)
# Define Trainer
trainer = pl.Trainer(max_epochs=100, logger=wandb_logger, callbacks=[lr_logger, early_stopping, checkpoint_callback], 
                     default_root_dir=OUTPUTDIR/Path(name)) #gpus=1

logging.info('Starting training')
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
