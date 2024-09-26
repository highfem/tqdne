import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

from tqdne.representations import to_numpy, to_torch

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
    
    def get_embeddings(self, data, data_representation=None):
        """
        Get the embeddings for the input data.

        Args:
            data (np.ndarray or torch.utils.data.DataLoader): The input data.
            data_representation (Representation, optional): The data representation used by the classifier.

        Returns:
            torch.Tensor: The embeddings for the input data.
        """ 

        embeddings_list = []
        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in tqdm(data):
                    if data_representation is not None:
                        batch = data_representation.get_representation(batch['waveform'])
                    else:    
                        batch = to_torch(batch['repr'], device=self.device)
                    batch = to_torch(batch, device=self.device)
                    embeddings = self.net.get_embeddings(batch) 
                    embeddings_list.append(to_numpy(embeddings))
            else:  
                if self.ml_config.optimizer_params.batch_size >= data.shape[0]:
                    if data_representation is not None:
                        data = data_representation.get_representation(data) 
                    embeddings = self.net.get_embeddings(to_torch(data, device=self.device))
                    return to_numpy(embeddings)
                for batch in tqdm(np.array_split(data, self.ml_config.optimizer_params.batch_size)):
                    if data_representation is not None:
                        batch = data_representation.get_representation(batch)           
                    batch = to_torch(batch, device=self.device)
                    embeddings = self.net.get_embeddings(batch) 
                    embeddings_list.append(to_numpy(embeddings))
                    
        return np.concatenate(embeddings_list, axis=0)

    def get_probabilities(self, data, data_representation=None, from_embeddings=False):
        """
        Get class probabilities for the input data.

        Args:
            data (np.ndarray or torch.utils.data.DataLoader): The input data (np.ndarray or torch.utils.data.DataLoader) or the embeddings (np.ndarray).
            data_representation (Representation, optional): The data representation object. Defaults to None.
            from_embeddings (bool, optional): Whether to get the probabilities from the embeddings. Defaults to False.

        Returns:
            numpy.ndarray: The concatenated class probabilities for the input data.

        """
        predictions = to_torch(self.get_predictions(data, data_representation, from_embeddings))
        probabilities = torch.nn.functional.softmax(predictions, dim=1)                 
        return to_numpy(probabilities)

    
    def get_predictions(self, data, data_representation=None, from_embeddings=False):
        """
        Get the class predictions for the input data, before the softmax.

        data (np.ndarray or torch.utils.data.DataLoader): The input data (np.ndarray or torch.utils.data.DataLoader) or the embeddings (np.ndarray).
        data_representation (Representation, optional): The data representation object. Defaults to None.
        from_embeddings (bool, optional): Whether to get the predictions

        Returns:
            numpy.ndarray: The concatenated class predictions for the input data.

        """
        predictions_list = []
        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in tqdm(data):
                    if data_representation is not None:
                        batch = data_representation.get_representation(batch['waveform'])
                    else:    
                        batch = to_torch(batch['repr'], device=self.device)
                    batch = to_torch(batch, device=self.device)
                    predictions = self.net(batch)
                    predictions_list.append(to_numpy(predictions))
            else:      
                if self.ml_config.optimizer_params.batch_size >= data.shape[0]:
                    if data_representation is not None:
                        assert not from_embeddings, "Cannot use data representation with embeddings"
                        data = data_representation.get_representation(data) 
                    if from_embeddings:
                        preds = self.net.get_predictions(to_torch(data, device=self.device), from_embeddings=True)
                    else:
                        preds = self.net(to_torch(data, device=self.device))
                    return to_numpy(preds)
                for batch in tqdm(np.array_split(data, self.ml_config.optimizer_params.batch_size)):
                    if data_representation is not None:
                        assert not from_embeddings, "Cannot use data representation with embeddings"
                        batch = data_representation.get_representation(batch)       
                    if from_embeddings:
                        preds = self.net.get_predictions(to_torch(batch, device=self.device), from_embeddings=True)
                    else:
                        preds = self.net(to_torch(batch, device=self.device))
                    predictions_list.append(to_numpy(preds))

        return np.concatenate(predictions_list, axis=0)


            