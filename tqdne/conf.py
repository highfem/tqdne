import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, Type

from dotenv import load_dotenv

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
    #PATH_ROOT / Path("datasets"),
    "/store/sdsc/sd28/data",
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

    # Dataset
    datasetdir: Path = DATASETDIR
    outputdir: Path = OUTPUTDIR
    project_name: str = PROJECT_NAME
    
    # dataset_files:
    data_upsample_train: str = "data_upsample_train.h5"
    data_upsample_test: str = "data_upsample_test.h5"

    data_train: str = "data_train.h5"
    data_test: str = "data_test.h5"


    # Sampling frequency
    fs: int = 100
    # Filter parameters
    params_filter: dict = field(default_factory= lambda: {"N": 2, "Wn": 1, "btype":'lp'})

    # Noise on the input
    sigma_in: float = 0.01

    # Input data parameters
    datapath: Path = DATASETDIR / Path("wforms_GAN_input_v20220805.h5")
    features_keys: Tuple[str] = (
        "hypocentral_distance",
        "is_shallow_crustal",
        "log10snr",
        "magnitude",
        "vs30"
    )

    # Train Dataset statistics
    transformed_env_statistics = pickle.load(datapath / Path("GM0-dataset-split/transformed_env_statistics.pkl"))
