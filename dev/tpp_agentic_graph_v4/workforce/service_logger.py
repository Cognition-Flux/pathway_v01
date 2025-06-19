"""
Implements a common logger for all the things in this proyect
"""

# %%
import logging
import os
import sys
from logging.config import dictConfig
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
load_dotenv(override=True)

os.chdir(package_root)


# region Logger configuration
class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "capacity"
    LOG_FORMAT: str = "%(levelprefix)s %(asctime)s - %(message)s"
    LOG_LEVEL: str = "DEBUG"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }


# endregion Logger configuration

# region Instanciate the logger
dictConfig(LogConfig().model_dump(mode="python"))
logger = logging.getLogger("capacity")
logger.setLevel(logging.DEBUG)
# endregion Instanciate the logger
