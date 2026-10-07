"""Bounded UTC logs owned by the facelive logger, without changing root logging."""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LoggingConfig


def close_logging() -> None:
    logger = logging.getLogger("facelive")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def configure_logging(config: LoggingConfig) -> Path:
    config.directory.mkdir(parents=True, exist_ok=True)
    path = config.directory / "facelive.log"
    file_handler = RotatingFileHandler(
        path, maxBytes=config.max_bytes, backupCount=config.backup_count, encoding="utf-8"
    )
    close_logging()
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    logger = logging.getLogger("facelive")
    logger.setLevel(config.level)
    logger.propagate = False
    for handler in (file_handler, logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return path
