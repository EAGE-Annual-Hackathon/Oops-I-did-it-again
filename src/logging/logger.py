import logging
import os
import sys

module_path = "../data"
sys.path.append(os.path.abspath(module_path))
from config import get_config

def get_logger(config_name="config") -> logging.Logger:
    """
    Функция для получения логгера.

    Parameters
    ----------
    config_name : str
        Config file name.

    Returns
    -------
    logger : logging.Logger
        Logger.
    """

    logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s [%(threadName)s]"
    )

    mode = "w"
    # Setup file handler
    fh = logging.FileHandler(
        "log.log", mode=mode, encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Configure stream handler for the cells
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    # Show the handlers
    logger.info(logger.handlers)

    return logger