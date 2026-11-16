"""Structured logging with optional JSON output and chat-content redaction."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict

from chatgangs.config import get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON formatter (no external dependencies)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("request_id", "path", "method", "status", "latency_ms",
                    "client", "module", "event"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Idempotently configure the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.handlers.clear()
    root.addHandler(handler)

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Keep noisy third-party loggers quiet in development.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


class Timer:
    """Context manager measuring elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
