import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from tqdne.representations import to_numpy

class LightningClassifier(pl.LightningModule):
    """A PyTorch Lightning module for training a classification model

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    example_input_array : torch.Tensor, optional
        An example input array for the network.
    ml_config : ml_collections.ConfigDict, optional
        A configuration object for the model.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_params: dict,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        metrics: dict = {},
        example_input_array: torch.Tensor = None,
        ml_config: ml_collections.ConfigDict = None,
    ):
        super().__init__()

        self.net = net
        self.loss = loss
        self.metrics = metrics
        self.optimizer_params = optimizer_params
        self.example_input_array = example_input_array
        self.ml_config = ml_config
        self.save_hyperparameters(ignore=["example_input_array"])

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward(self, x):
        return self.net(x)
    
    def evaluate(self, batch):
        x = batch["repr"]
        return self(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["repr"], batch["classes"] 
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_value(loss, "loss")
        for metric_name, metric in self.metrics.items():
            metric = metric.to(self.device)
            self.log_value(metric(y_hat, y), metric_name)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["repr"], batch["classes"] 
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_value(loss, "loss", train=False)
        for metric_name, metric in self.metrics.items():
            metric = metric.to(self.device)
            self.log_value(metric(y_hat, y), metric_name, train=False)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        return optimizer
    
    def get_embeddings(self, data, data_represenation=None, return_stats=False):
            """
            Get the embeddings for the input data.

            Args:
                data (np.ndarray or torch.utils.data.DataLoader): The input data.
                data_represenation (Representation, optional): The data representation used by the classifier.
                return_stats (bool, optional): Whether to return the mean and standard deviation of the embeddings instead of the embeddings themselves. Defaults to False.

            Returns:
                torch.Tensor: The embeddings for the input data.
            """ 
            print("Getting embeddings")
            if return_stats:
                pass
                # Compute the mean and standard deviation of the embeddings online
                # TODO: Delete
                # from tqdne.utils import OnlineStats
                # stats = OnlineStats()
                # with torch.no_grad():
                #     for batch in x.split(self.ml_config.optimizer_params.batch_size):
                #         batch = batch.to(self.device)
                #         embeddings = self.net.get_embeddings(batch)
                #         stats.update(embeddings)
                # return (stats.mean, stats.std)
            else:    
                embeddings_list = []
                with torch.no_grad():
                    if isinstance(data, torch.utils.data.DataLoader):
                        for batch in tqdm(data):
                            if data_represenation is not None:
                                batch = data_represenation.get_representation(batch['repr'])
                            batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                            embeddings = self.net.get_embeddings(batch) 
                            embeddings_list.append(to_numpy(embeddings))
                    else:  
                        for batch in tqdm(np.array_split(data, self.ml_config.optimizer_params.batch_size)):
                            if data_represenation is not None:
                                batch = data_represenation.get_representation(batch)           
                            batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                            embeddings = self.net.get_embeddings(batch) 
                            embeddings_list.append(to_numpy(embeddings))
                            
                return np.concatenate(embeddings_list, axis=0)

    def get_probabilities(self, data, data_represenation=None):
            """
            Get class probabilities for the input data.

            Args:
                data (np.ndarray or torch.utils.data.DataLoader): The input data.
                data_represenation (Representation, optional): The data representation object. Defaults to None.

            Returns:
                numpy.ndarray: The concatenated class probabilities for the input data.

            """
            probabilities_list = []
            with torch.no_grad():
                if isinstance(data, torch.utils.data.DataLoader):
                    for batch in tqdm(data):
                        if data_represenation is not None:
                            batch = data_represenation.get_representation(batch['repr'])
                        batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                        probabilities = torch.nn.functional.softmax(self.net(batch), dim=1)
                        probabilities_list.append(to_numpy(probabilities))
                else:      
                    for batch in tqdm(np.array_split(data, self.ml_config.optimizer_params.batch_size)):
                        if data_represenation is not None:
                            batch = data_represenation.get_representation(batch)       
                        batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                        probabilities = torch.nn.functional.softmax(self.net(batch), dim=1)
                        probabilities_list.append(to_numpy(probabilities))
                                
            return np.concatenate(probabilities_list, axis=0)
