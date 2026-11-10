"""Run directories and write-once artifacts.

results/RUN_ID/
  config.yaml  environment.json  events.jsonl  trials.parquet
  artifacts/{sources,generated,detections,instructions,responses,failures,evaluations}/
  metrics/  figures/  tables/  report.md
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ARTIFACT_KINDS = (
    "sources",
    "generated",
    "detections",
    "instructions",
    "responses",
    "failures",
    "evaluations",
)
TRACKED_PACKAGES = (
    "torch", "transformers", "accelerate", "openai", "pydantic", "numpy", "pandas",
    "pyarrow", "scipy", "scikit-learn", "sentence-transformers", "spacy", "datasets",
    "matplotlib",
)  # fmt: skip


def new_run_id(name: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{name}-{secrets.token_hex(3)}"


class RunDir:
    def __init__(self, root: str | Path, run_id: str):
        self.run_id = run_id
        self.path = Path(root) / run_id

    def create(self) -> RunDir:
        self.path.mkdir(parents=True, exist_ok=False)
        for kind in ARTIFACT_KINDS:
            (self.path / "artifacts" / kind).mkdir(parents=True)
        for sub in ("metrics", "figures", "tables"):
            (self.path / sub).mkdir()
        return self

    @property
    def events(self) -> Path:
        return self.path / "events.jsonl"

    def artifact(self, kind: str, name: str) -> Path:
        assert kind in ARTIFACT_KINDS, kind
        return self.path / "artifacts" / kind / f"{name}.json"

    def has(self, kind: str, name: str) -> bool:
        return self.artifact(kind, name).exists()

    def save(self, kind: str, name: str, obj: BaseModel | dict[str, Any]) -> Path:
        return write_once(self.artifact(kind, name), obj)

    def load(self, kind: str, name: str) -> dict[str, Any]:
        return json.loads(self.artifact(kind, name).read_text())

    def list(self, kind: str) -> list[Path]:
        return sorted((self.path / "artifacts" / kind).glob("*.json"))


def write_once(path: Path, obj: BaseModel | dict[str, Any]) -> Path:
    """Write JSON, refusing to overwrite raw evidence (mode 'x')."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    with open(path, "x") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def write_yaml_once(path: Path, data: dict[str, Any]) -> None:
    with open(path, "x") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def environment_info() -> dict[str, Any]:
    """Software and hardware details recorded once per run."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {},
    }
    for pkg in TRACKED_PACKAGES:
        try:
            info["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info["packages"][pkg] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as e:  # torch missing or broken: record, don't crash
        info["torch_error"] = repr(e)
    return info
