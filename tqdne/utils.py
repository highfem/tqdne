import logging
from pathlib import Path
from typing import Type


import torch
import numpy as np
import PIL
import pytorch_lightning as pl
import torch


def to_numpy(x):
    return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


def load_model(type: Type[pl.LightningModule], path: Path, **kwargs):
    """Load the autoencoder model from the save directory.

    Parameters
    ----------
    type : Type[pl.LightningModule]
        The type of the model to load.
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.

    """
    if not path.exists():
        logging.info("Model not found. Returning None.")
        return None
    model = type.load_from_checkpoint(path, **kwargs)
    return model


def fig2PIL(fig):
    """Convert a matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure.

    Returns
    -------
    PIL.Image
        The PIL Image.

    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    return PIL.Image.frombytes(mode="RGB", size=(w, h), data=buf)



def get_last_checkpoint(dirpath):
    checkpoints = sorted(list(Path(dirpath).glob("*.ckpt")))
    if len(checkpoints) == 0:
        logging.info("No checkpoint found. Returning None.")
        return None
    # Load the checkpoint with the latest epoch
    checkpoint = checkpoints[-1]
    logging.info(f"Last checkpoint is : {checkpoint}")
    return checkpoint

def positional_encoding(x, L, xrange=1.0):
    """Compute the positional encoding for a given vector x

    The positional encoding is a vector of length 2L, where L is the number of frequencies to use.
    The first L entries are the sine of the input vector multiplied by $2^\ell * \pi * x$, where $\ell$ is the index of the frequency.
    The second L entries are the cosine of the input vector multiplied by $2^\ell * \pi * x$, where $\ell$ is the index of the frequency.

    Parameters
    ----------
    x : torch.Tensor
        Input vector
    L : int
        Number of frequencies to use
    
    Returns
    -------
    torch.Tensor
        The positional encoding of x. The shape is the same as x, with an additional dimension of size 2L appended to the end.
    """
    isnumpy = type(x) == np.ndarray
    if isnumpy:
        x = torch.from_numpy(x)
    x /= (xrange * 2.0) 
    shape = x.shape
    ndim = len(shape)
    x = x.unsqueeze(ndim)
    ell = torch.arange(L, requires_grad=False, dtype=x.dtype, device=x.device)
    fac = 2**ell
    fac = fac.view(*([1]*ndim),L)
    inner = fac*np.pi * x
    assert inner.shape == (*shape, L)
    s = torch.sin(inner)
    c = torch.cos(inner)
    out = torch.cat([s,c], dim=ndim)
    assert out.shape == (*shape, 2*L)
    if isnumpy:
        out = out.numpy()
    return out