from dataclasses import dataclass
from pathlib import Path

from tqdne import representation


@dataclass
class Config:
    """Configuration class for the project."""

    workdir: str | Path
    project_name: str = "tqdne"
    channels: int = 3
    fs: int = 100
    t = None
    features_keys: tuple[str, ...] = (
        "hypocentral_distance",
        "magnitude",
        "vs30",
        "hypocentre_depth",
        "azimuthal_gap",
    )
    representation = representation.Identity()

    def __post_init__(self):
        path = self.workdir if isinstance(self.workdir, Path) else Path(self.workdir)
        self.datasetdir: Path = path / Path("data")
        self.outputdir: Path = path / Path("outputs")
        self.original_datapath: Path = self.datasetdir / Path("raw_waveforms.h5")
        self.datapath: Path = self.datasetdir / Path("preprocessed_waveforms.h5")


@dataclass
class SpectrogramConfig(Config):
    """Configuration class for the spectrogram representation."""

    # representation size: 128 x 128
    stft_channels: int = 256
    hop_size: int = 32
    representation = representation.LogSpectrogram(stft_channels=stft_channels, hop_size=hop_size)
    # we need to increase this from earlier version, since now data is bigger
    t: int = 4064


@dataclass
class LatentSpectrogramConfig(SpectrogramConfig):
    """Configuration class for latent diffusion on spectrogram representation."""

    latent_channels: int = 8
    kl_weight: float = 1e-6


@dataclass
class SpectrogramClassificationConfig(SpectrogramConfig):
    """Configuration class for the spectrogram representation."""

    mag_bins = [4, 4.75, 5, 5.5, 6.5, 7.5, 9.1]
    dist_bins = [0, 75, 100, 125, 150, 175, 200]


@dataclass
class MovingAverageEnvelopeConfig(Config):
    """Configuration class for the moving average envelope representation."""

    channels: int = 6  # 3 signal + 3 envelope
    representation = representation.MovingAverageEnvelope()
    t: int = 4064  # for compatibility with the spectrogram representation


@dataclass
class LatentMovingAverageEnvelopeConfig(MovingAverageEnvelopeConfig):
    """Configuration class for latent diffusion on moving average envelope representation."""

    latent_channels: int = 16
    kl_weight: float = 1e-6
