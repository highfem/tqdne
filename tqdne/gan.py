import pytorch_lightning as pl

class LightningGAN(pl.LightningModule):
    """A PyTorch Lightning module for training a Generative Adversarial Network.

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    noise_scheduler : DDPMScheduler
        A scheduler for adding noise to the clean images.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    prediction_type : str, optional
        The type of prediction to make. One of "epsilon" or "sample".
    cond_signal_input : bool, optional
        Whether low resolution input is provided.
    cond_input : bool, optional
        Whether conditional input is provided.
    """


