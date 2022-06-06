import yaml
import logging
import sys
import numpy as np
import os

logger = logging.getLogger(__name__)

def get_config(config_name="config", path="../../data/config/", debug=False) -> dict:
    """
    Возвращает файл с конфигом.

    Parameters
    ----------
    config_name : str
        Config file name.
    path: str
        Путь к конфигу.
    debug: bool
        Если True, то выводит в консоль путь к конфигу.

    Returns
    -------
    config : dict
        Dictionary containing config values.
    """
    try:
        path = path + config_name + ".yml"
        if debug:
            import pathlib
            logger.info(50*"=")
            logger.info(f"get_config")
            logger.info(f"pathlib.Path(path).resolve(): {pathlib.Path(path).resolve()}")
            logger.info(50*"=")

        with open(path, "r", encoding="utf8") as file:
            config = yaml.safe_load(file)
        return config

    except Exception as e:
        logger.error(f"Error reading the config file.")
        logger.error(e)
