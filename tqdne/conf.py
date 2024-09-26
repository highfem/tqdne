import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, Type

from dotenv import load_dotenv
import h5py

# Set up the default logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# Create the logger object
logger = logging.getLogger("Default Logger")

env_path = Path(__file__).parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path.resolve(), override=True, verbose=True)


class LazyEnv:
    """Lazy environment variable."""

    def __init__(
        self,
        env_var: str,
        default=None,
        return_type: Type = str,
        after_eval: Callable = None,
    ):
        """Construct lazy evaluated environment variable."""
        self.env_var = env_var
        self.default = default
        self.return_type = return_type
        self.after_eval = after_eval

    def eval(self):
        """Evaluate environment variable."""
        value = self.return_type(os.environ.get(self.env_var, self.default))

        if self.after_eval:
            self.after_eval(value)

        return value


# path processed dataset
PATH_ROOT = Path(__file__).parents[1]


DATASETDIR = LazyEnv(
    "DATASET_DIR",
    PATH_ROOT / Path("datasets"),
    #"/store/sdsc/sd28/data/GM0-dataset-split",
    return_type=Path,
).eval()

OUTPUTDIR = LazyEnv(
    "OUTPUTDIR",
    PATH_ROOT / Path("outputs"),
    return_type=Path,
).eval()

PROJECT_NAME = "tqdne"


@dataclass
class Config:
    """Configuration class for the project."""

    # Singleton
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    # Dataset
    datasetdir: Path = DATASETDIR
    outputdir: Path = OUTPUTDIR
    project_name: str = PROJECT_NAME

    # dataset_files:
    data_upsample_train: str = "data_upsample_train.h5"
    data_upsample_test: str = "data_upsample_test.h5"

    data_train: str = "data_train.h5"
    data_test: str = "data_test.h5"
    train_ratio: float = 0.9

    # Sampling frequency
    original_fs: int = 100
    fs = original_fs
    # Filter parameters
    params_filter: dict = field(default_factory=lambda: {"N": 2, "Wn": 1, "btype": "lp"})

    # Noise on the input
    sigma_in: float = 0.01

    # Input data parameters
    datapath: Path = Path("/store/sdsc/sd28") / Path("wforms_GAN_input_v20220805.h5")
    features_keys: Tuple[str] = (
        "hypocentral_distance",
        "is_shallow_crustal",
        "magnitude",
        "vs30",
    )
 
    conditional_params_range = {
        "hypocentral_distance": (0, 180.), # in the dataset: (4, 180) [km]
        "is_shallow_crustal": (0., 1.), # 0: False, 1: True
        "magnitude": (4.5, 10), # in the dataset: (4.5, 9.1) [Mw]
        "vs30": (70., 2100.) # in the dataset (when removing vs30<=0): (76, 2100) [m/s]
    }
    
    # Open the h5 file
    with h5py.File(datasetdir / data_test, 'r', locking=False) as file:
        # Get the 10th sample
        example_signal = file["waveforms"][10]
        num_channels: int = example_signal.shape[0]
        original_signal_length: int = example_signal.shape[1] 
        signal_length = original_signal_length

    # Train Dataset statistics
    # TODO: this shouldn't be here. It should be a parameter of the class SignalWithEnvelope, that should be passed to the init method 
    with open(datasetdir / Path("signal_statistics.pkl"), 'rb') as pickle_file:
        signal_statistics = pickle.load(pickle_file)

    with open(datasetdir / Path("trans_env_statistics.pkl"), 'rb') as pickle_file:
        transformed_env_statistics = pickle.load(pickle_file)      

