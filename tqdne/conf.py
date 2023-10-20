from pathlib import Path
from typing import Callable, Type
import logging
import os

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
    PATH_ROOT / Path("datasets"),
    return_type=Path,
).eval()

OUTPUTDIR = LazyEnv(
    "OUTPUTDIR",
    PATH_ROOT / Path("outputs"),
    return_type=Path,
).eval()
