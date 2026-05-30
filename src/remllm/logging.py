"""Structured logging with structlog — human-readable console output."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import structlog

_initialized = False
_log_dir: Path | None = None


def init_logging() -> None:
    """Initialize structured logging for human-readable console output."""
    global _initialized

    if _initialized:
        return
    _initialized = True

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(pad_event_to=0),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    """Get a bound structured logger.

    Usage:
        log = get_logger(run_id="exp-001", phase="train")
        log.info("Starting training", epochs=3, lr=1.2e-4)
    """
    if not _initialized:
        init_logging()
    return structlog.get_logger().bind(**context)


def set_log_dir(path: str | Path) -> None:
    """Set or change the log directory."""
    global _log_dir
    _log_dir = Path(path)
    _log_dir.mkdir(parents=True, exist_ok=True)


def get_log_dir() -> Path | None:
    """Return the current log directory."""
    return _log_dir
