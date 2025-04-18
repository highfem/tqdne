import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import PIL
import pytorch_lightning as pl
import torch
import torch as th

def get_device():
    """Get the available accelerator device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_numpy(x):
    if isinstance(x, Sequence):
        return x.__class__(to_numpy(v) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_numpy(v)) for k, v in x.items())
    else:
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


class NumpyArgMixin:
    """Mixin for automatic conversion of method arguments to numpy arrays."""

    def __getattribute__(self, name):
        """Return a function wrapper that converts method arguments to numpy arrays."""
        attr = super().__getattribute__(name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            args = to_numpy(args)
            kwargs = to_numpy(kwargs)
            return attr(*args, **kwargs)

        return wrapper


def load_model(type: type[pl.LightningModule], path: Path, **kwargs):
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


def mask_from_indexes(mask_idxs, x, fill_with=th.nan):
    mask = th.arange(x.size(-1), device=x.device).expand(x.size(0), x.size(-1)) >= mask_idxs.unsqueeze(1)
    if x.ndim == 4:
        mask = mask[:, None, None, :]
    else:
        mask = mask[:, None, :]
    x = x.masked_fill(mask, fill_with)
    return x


def get_latent_mask_indexes(mask, dim=2):
    if dim == 2:
        low = (((((mask - 8) / 2) - 8) / 2) - 3).type(torch.int32)
        up = ((((low - 6) * 2) - 6) * 2)
        return low, up
    else:
        raise ValueError("only have dim 2")


