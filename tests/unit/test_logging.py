import logging

from app.config import LoggingConfig
from app.diagnostics.logging_setup import close_logging, configure_logging


def test_logging_is_bounded_and_reconfiguration_does_not_duplicate(tmp_path):
    config = LoggingConfig(directory=tmp_path, max_bytes=200, backup_count=2)
    root_handlers = logging.getLogger().handlers[:]
    try:
        path = configure_logging(config)
        configure_logging(config)
        logger = logging.getLogger("facelive.test")
        assert len(logging.getLogger("facelive").handlers) == 2
        for index in range(12):
            logger.info("Sample event %d with enough content to exercise rotation", index)
        logger.warning("Final event")
        contents = path.read_text(encoding="utf-8")
        assert contents.count("Final event") == 1
        assert "Z WARNING facelive.test:" in contents
        assert 1 < len(list(tmp_path.glob("facelive.log*"))) <= 3
        assert logging.getLogger().handlers == root_handlers
    finally:
        close_logging()
