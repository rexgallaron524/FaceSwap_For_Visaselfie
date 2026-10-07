"""Composition root and CLI for the desktop shell."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.config import AppConfig, ConfigError, load_config
from app.diagnostics.logging_setup import close_logging, configure_logging
from app.ui.main_window import MainWindow


def create_application(config: AppConfig) -> tuple[QApplication, MainWindow]:
    """Create the shell without starting the event loop or opening any devices."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(["facelive"])
    if not isinstance(application, QApplication):
        raise RuntimeError("FaceLive requires a QApplication, not a QCoreApplication")
    application.setApplicationName("FaceLive")
    application.setOrganizationName("FaceLive")
    return application, MainWindow(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FaceLive desktop application")
    parser.add_argument("--config", type=Path, help="Explicit TOML configuration file")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Show the shell and close after 250 ms"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        log_path = configure_logging(config.logging)
    except (ConfigError, OSError) as exc:
        print(f"FaceLive startup failed: {exc}", file=sys.stderr)
        return 2

    logger = logging.getLogger("facelive.app")
    try:
        logger.info("Starting FaceLive; logs: %s", log_path)
        application, window = create_application(config)
        window.show()
        logger.info("Application shell ready; camera and output inactive")
        if args.smoke_test:
            QTimer.singleShot(250, window.close)
        exit_code = application.exec()
        logger.info("Application stopped (exit %d)", exit_code)
        return exit_code
    except Exception:
        logger.exception("Application startup/runtime failure")
        return 1
    finally:
        close_logging()


if __name__ == "__main__":
    raise SystemExit(main())
