from dataclasses import dataclass
from pathlib import Path

from tqdne import representation


@dataclass
class Config:
    """Configuration class for the project."""

    workdir: str | Path
    infile: str | Path | None = None
    project_name: str = "tqdne"
    channels: int = 3
    fs: int = 100
    t = None
    features_keys: tuple[str, ...] = (
        "hypocentral_distance",        
        "magnitude",
        "vs30",
        "hypocentre_depth",
        "azimuthal_gap"
    )
    representation = representation.Identity()

    def __post_init__(self):
        path = self.workdir if isinstance(self.workdir, Path) else Path(self.workdir)
        if self.infile is not None:
            self.infile = self.infile if isinstance(self.infile, Path) else Path(self.infile)
        self.datasetdir: Path = path / Path("data")
        self.outputdir: Path = path / Path("outputs")
        self.original_datapath: Path = self.datasetdir / Path("raw_waveforms.h5")
        self.datapath: Path = self.infile or self.datasetdir / Path("preprocessed_waveforms.h5")


@dataclass
class SpectrogramConfig(Config):
    """Configuration class for the spectrogram representation."""

    # representation size: 128 x 384
    stft_channels: int = 256
    hop_size: int = 32
    representation = representation.LogSpectrogram(stft_channels=stft_channels, hop_size=hop_size)
     # we need to increase this from earlier version, since now data is bigger
    t: int = 12256


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
    t: int = 12256  # for compatibility with the spectrogram representation


@dataclass
class LatentMovingAverageEnvelopeConfig(MovingAverageEnvelopeConfig):
    """Configuration class for latent diffusion on moving average envelope representation."""

    latent_channels: int = 16
    kl_weight: float = 1e-6
