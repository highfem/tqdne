__version__ = "0.1.0"

from tqdne.conf import logger


def print_version():
    """Print the version of the package.

    This function is used to print the version of the package.

    Returns
    -------
    None
        This function does not return anything.

    """
    logger.info(__version__)
