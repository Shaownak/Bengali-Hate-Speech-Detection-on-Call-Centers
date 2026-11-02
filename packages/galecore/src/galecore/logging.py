"""Structured JSON logging with a privacy scrubber (CLAUDE.md hard rule 1, docs/05 §4).

The scrubber:
- drops any key whose name contains ``text``, ``transcript``, ``audio`` or ``pcm``
  (so ``text_redacted``, ``pcm_s16le``, ``audio_b64`` are dropped too: even redacted
  transcripts stay out of logs);
- redacts phone/NID/account/email patterns in every remaining string value, including the
  event message itself.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

from galecore.redact import redact

FORBIDDEN_KEY_PARTS = ("text", "transcript", "audio", "pcm")
DROPPED = "[dropped]"


def _is_forbidden(key: str) -> bool:
    k = key.lower()
    return any(part in k for part in FORBIDDEN_KEY_PARTS)


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value).text
    if isinstance(value, bytes | bytearray | memoryview):
        return DROPPED
    if isinstance(value, Mapping):
        return {
            k: (DROPPED if _is_forbidden(str(k)) else _scrub_value(v)) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_scrub_value(v) for v in value]
    return value


def scrub_event(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: remove content keys and redact PII patterns."""
    for key in list(event_dict):
        if key == "event":
            continue
        if _is_forbidden(key):
            del event_dict[key]
        else:
            event_dict[key] = _scrub_value(event_dict[key])
    if isinstance(event_dict.get("event"), str):
        event_dict["event"] = redact(event_dict["event"]).text
    return event_dict


def configure_logging(
    service: str, level: str = "INFO", json: bool = True, stream: Any = None
) -> None:
    """Configure structlog + stdlib logging for a service. Safe to call more than once."""
    lvl = logging.getLevelName(level.upper())
    if not isinstance(lvl, int):
        lvl = logging.INFO
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            scrub_event,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
