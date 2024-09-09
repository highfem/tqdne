from dataclasses import dataclass
from pathlib import Path

from tqdne import representation

# path processed dataset
PATH_ROOT = Path(__file__).parents[1]


@dataclass
class Config:
    """Configuration class for the project."""

    project_name: str = "tqdne"

    datasetdir: Path = PATH_ROOT / Path("datasets")
    outputdir: Path = PATH_ROOT / Path("outputs")
    original_datapath: Path = datasetdir / Path("raw_waveforms.h5")
    datapath: Path = datasetdir / Path("processed_waveforms.h5")
    channels: int = 3
    fs: int = 100
    t = None

    features_keys: tuple[str, ...] = (
        "hypocentral_distance",
        "is_shallow_crustal",
        "magnitude",
        "vs30",
    )

    representation = representation.Identity()


@dataclass
class SpectrogramConfig(Config):
    """Configuration class for the spectrogram representation."""

    # representation size: 128 x 128
    stft_channels: int = 256
    hop_size: int = 32
    representation = representation.LogSpectrogram(stft_channels=stft_channels, hop_size=hop_size)
    t: int = 4096 - hop_size  # subtract hop_size to make sure spectrogram has even number of frames


@dataclass
class LatentSpectrogramConfig(SpectrogramConfig):
    """Configuration class for latent diffusion on spectrogram representation."""

    latent_channels: int = 4
    kl_weight: float = 1e-6


@dataclass
class SpectrogramClassificationConfig(SpectrogramConfig):
    """Configuration class for the spectrogram representation."""

    mag_bins = [4.5, 4.75, 5, 5.25, 6, 9.1]
    dist_bins = [0, 75, 100, 125, 150, 200]


@dataclass
class MovingAverageEnvelopeConfig(Config):
    """Configuration class for the moving average envelope representation."""

    channels: int = 6  # 3 signal + 3 envelope
    representation = representation.MovingAverageEnvelope()
    t: int = 4096 - 32  # for compatibility with the spectrogram representation


@dataclass
class LatentMovingAverageEnvelopeConfig(MovingAverageEnvelopeConfig):
    """Configuration class for latent diffusion on moving average envelope representation."""

    latent_channels: int = 16
    kl_weight: float = 1e-6
