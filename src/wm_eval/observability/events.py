"""Append-only JSONL event recorder plus standard logging.

Every pipeline stage calls `recorder.emit(...)`. Values whose key looks secret are
redacted before anything is written.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_MARKERS = ("secret", "api_key", "apikey", "password", "token_value", "watermark_keys")
SECRET_EXACT = {"keys"}

log = logging.getLogger("wm_eval")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def redact(value: Any) -> Any:
    """Recursively replace values stored under secret-looking keys."""
    if isinstance(value, dict):
        return {k: "***REDACTED***" if _is_secret(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def _is_secret(key: str) -> bool:
    k = key.lower()
    return k in SECRET_EXACT or any(m in k for m in SECRET_MARKERS)


class Recorder:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, data: dict[str, Any] | None = None, **ids: Any) -> dict[str, Any]:
        record = {"ts": utc_now(), "event": event, "run_id": self.run_id, **ids}
        if data:
            record["data"] = data
        record = redact(record)
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
        level = logging.ERROR if event.endswith("failed") else logging.INFO
        log.log(level, line)
        return record


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
