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


@dataclass
class TransferConfig(LatentSpectrogramConfig):
    """Configuration class for transfer learning."""

    # Source model checkpoints
    source_autoencoder_checkpoint: str | Path | None = None
    source_diffusion_checkpoint: str | Path | None = None

    # Transfer learning strategy
    transfer_strategy: str = "conservative"  # "conservative" or "aggressive"

    # Autoencoder transfer settings
    freeze_encoder: bool = True  # Freeze encoder in conservative mode
    freeze_decoder: bool = False  # Typically fine-tune decoder
    autoencoder_learning_rate: float = 1e-5  # Lower than original (1e-4)
    autoencoder_max_steps: int = 30_000  # Fewer steps than original

    # Diffusion model transfer settings
    diffusion_learning_rate: float = 1e-5  # Lower than original (1e-4)
    diffusion_warmup_steps: int = 500  # Shorter warmup
    diffusion_max_steps: int = 100_000  # Fewer steps than original
    diffusion_end_learning_rate: float = 1e-6

    # Training parameters
    gradient_clipping: float = 1.0
    weight_decay: float = 1e-5
    ema_decay: float = 0.9999

    # Override workdir to prevent conflicts with source
    target_workdir: str | Path | None = None

    def __post_init__(self):
        super().__post_init__()

        # Use target_workdir if specified, otherwise use workdir
        if self.target_workdir is not None:
            path = (
                self.target_workdir
                if isinstance(self.target_workdir, Path)
                else Path(self.target_workdir)
            )
            self.datasetdir = path / Path("data")
            self.outputdir = path / Path("outputs")
            self.original_datapath = self.datasetdir / Path("raw_waveforms.h5")
            self.datapath = self.datasetdir / Path("preprocessed_waveforms.h5")

        # Adjust settings based on transfer strategy
        if self.transfer_strategy == "aggressive":
            self.freeze_encoder = False
            self.freeze_decoder = False
            self.autoencoder_learning_rate = 1e-6  # Even lower for aggressive
            self.diffusion_learning_rate = 1e-6
        elif self.transfer_strategy != "conservative":
            raise ValueError(
                f"Invalid transfer_strategy: {self.transfer_strategy}. "
                "Must be 'conservative' or 'aggressive'."
            )
