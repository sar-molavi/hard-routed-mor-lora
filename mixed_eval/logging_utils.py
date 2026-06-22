"""Logging helpers for mixed evaluation scripts."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """Configure a simple timestamped logger for CLI scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

