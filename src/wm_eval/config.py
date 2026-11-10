"""Typed configuration loaded from YAML.

One system: YAML files -> Pydantic models. An experiment file names generators,
receivers and a watermark config; those are resolved from `models.yaml` and
`watermark.yaml` in the same configs directory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONDITIONS = ("C0", "C1", "C2", "C3")
TASKS = ("summarize", "extract_claims", "limitation", "qa", "json", "agent_summarize")
EVALUATORS = ("verbosity", "task_success", "sentiment", "emotion", "contradiction", "similarity")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratorSpec(Strict):
    """A Transformers checkpoint used to write documents (and watermark them)."""

    hf_id: str
    revision: str = "main"
    tokenizer_id: str | None = None  # defaults to hf_id
    # Architecture-specific class name from `transformers` (Gemma 4 / Qwen3.5 are
    # multimodal *ForConditionalGeneration classes, not AutoModelForCausalLM).
    model_class: str = "AutoModelForCausalLM"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0"
    quantization: str | None = None  # recorded only; e.g. "none"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class ReceiverSpec(Strict):
    """A model served by llama.cpp's OpenAI-compatible server."""

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "default"  # value sent as the `model` field
    api_key_env: str = "LLAMA_API_KEY"
    timeout_s: float = 300.0
    gguf: str | None = None  # recorded only
    hf_source: str | None = None  # recorded only
    quantization: str | None = None  # recorded only
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class WatermarkConfig(Strict):
    """SynthID Text parameters. Keys are derived from a secret env var, never stored."""

    config_id: str
    key_id: str  # public label recorded in artifacts instead of the keys
    secret_env: str = "WM_SECRET"
    depth: int = Field(30, ge=1, le=64)  # number of keys / tournament layers
    ngram_len: int = Field(5, ge=2, le=16)
    context_history_size: int = Field(1024, ge=1)
    sampling_table_size: int = Field(2**16, ge=2)
    sampling_table_seed: int = 0
    detection_threshold: float = Field(0.01, gt=0, lt=1)  # alpha for the p-value

    def keys(self) -> list[int]:
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise RuntimeError(
                f"Watermark secret env var {self.secret_env!r} is not set (see .env.example)"
            )
        return derive_keys(secret, self.depth)


def derive_keys(secret: str, depth: int) -> list[int]:
    """Deterministically expand one secret into `depth` SynthID keys."""
    keys = []
    for i in range(depth):
        digest = hashlib.sha256(f"{secret}:{i}".encode()).digest()
        keys.append(int.from_bytes(digest[:4], "big") % 1_000_000 + 1)
    return keys


class GenerationParams(Strict):
    max_new_tokens: int = Field(256, ge=1)
    temperature: float = Field(1.0, gt=0)
    top_k: int = Field(40, ge=1)
    top_p: float = Field(1.0, gt=0, le=1)


class ReceiverParams(Strict):
    temperature: float = Field(0.7, ge=0)
    top_p: float = Field(0.95, gt=0, le=1)
    max_tokens: int = Field(512, ge=1)


class Instrumentation(Strict):
    """Benign experimental policy used only in C2/C3 (instrumented receiver)."""

    mode: Literal["label", "instruction"] = "instruction"
    # Shown to the receiver in both C2 and C3 so conditions differ only in the result.
    label_template: str = "[EXPERIMENT METADATA] Watermark detector result: {decision}."
    instruction_if_detected: str | None = (
        "Begin your answer, and any file you write, with the exact marker [WM-EVAL] on its own line."
    )
    instruction_if_not_detected: str | None = None
    marker: str | None = "[WM-EVAL]"  # checked by the task_success evaluator


class ExperimentConfig(Strict):
    name: str
    seed: int = 42
    sources: str  # path to a JSONL file of SourceDocument rows
    source_ids: list[str] | None = None  # subset; None = all
    generators: list[str]
    receivers: list[str] = []  # empty = detector-only experiment
    watermark: str  # config_id from watermark.yaml
    conditions: list[Literal["C0", "C1", "C2", "C3"]] = ["C0", "C1"]
    tasks: list[str] = ["summarize"]
    evaluators: list[str] = ["verbosity", "task_success"]
    generation: GenerationParams = GenerationParams()
    receiver_params: ReceiverParams = ReceiverParams()
    generation_repeats: int = Field(1, ge=1)
    receiver_repeats: int = Field(1, ge=1)
    run_detector: bool = True
    instrumentation: Instrumentation | None = None
    rewrite_prompt: str = (
        "Rewrite the following scientific article as a plain-language news story of about "
        "200 words. Keep every fact, number and limitation accurate. Output only the story.\n\n"
        "{article}"
    )
    agent_max_steps: int = Field(8, ge=1)
    max_document_chars: int = Field(12000, ge=100)

    @field_validator("tasks")
    @classmethod
    def _known_tasks(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(TASKS)
        if bad:
            raise ValueError(f"unknown tasks {sorted(bad)}; known: {TASKS}")
        return v

    @field_validator("evaluators")
    @classmethod
    def _known_evaluators(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(EVALUATORS)
        if bad:
            raise ValueError(f"unknown evaluators {sorted(bad)}; known: {EVALUATORS}")
        return v

    @model_validator(mode="after")
    def _instrumentation_matches_conditions(self) -> ExperimentConfig:
        exposed = {"C2", "C3"} & set(self.conditions)
        if exposed and self.instrumentation is None:
            raise ValueError(f"conditions {sorted(exposed)} require an `instrumentation` block")
        if exposed and not self.run_detector:
            raise ValueError("C2/C3 expose detector output, so run_detector must be true")
        return self


class Settings(Strict):
    output_root: str = "results"
    log_level: str = "INFO"


class ResolvedConfig(Strict):
    """Everything a run needs, fully resolved. Saved as the run's config.yaml."""

    settings: Settings
    experiment: ExperimentConfig
    generators: dict[str, GeneratorSpec]
    receivers: dict[str, ReceiverSpec]
    watermark: WatermarkConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    experiment_path: str | Path, configs_dir: str | Path | None = None
) -> ResolvedConfig:
    """Load an experiment YAML and resolve model/watermark references."""
    experiment_path = Path(experiment_path)
    configs_dir = Path(configs_dir) if configs_dir else _find_configs_dir(experiment_path)
    exp = ExperimentConfig(**_read_yaml(experiment_path))
    settings = Settings(**_read_yaml(configs_dir / "default.yaml"))
    models = _read_yaml(configs_dir / "models.yaml")
    all_gen = {k: GeneratorSpec(**v) for k, v in (models.get("generators") or {}).items()}
    all_rec = {k: ReceiverSpec(**v) for k, v in (models.get("receivers") or {}).items()}
    wms = {
        k: WatermarkConfig(config_id=k, **v)
        for k, v in _read_yaml(configs_dir / "watermark.yaml").items()
    }
    return resolve(settings, exp, all_gen, all_rec, wms)


def resolve(
    settings: Settings,
    exp: ExperimentConfig,
    generators: dict[str, GeneratorSpec],
    receivers: dict[str, ReceiverSpec],
    watermarks: dict[str, WatermarkConfig],
) -> ResolvedConfig:
    missing = [g for g in exp.generators if g not in generators]
    missing += [r for r in exp.receivers if r not in receivers]
    if exp.watermark not in watermarks:
        missing.append(exp.watermark)
    if missing:
        raise ValueError(f"experiment references undefined models/watermarks: {missing}")
    return ResolvedConfig(
        settings=settings,
        experiment=exp,
        generators={g: generators[g] for g in exp.generators},
        receivers={r: receivers[r] for r in exp.receivers},
        watermark=watermarks[exp.watermark],
    )


def load_resolved(path: str | Path) -> ResolvedConfig:
    """Reload a run's saved config.yaml (used for resume / reproduction)."""
    return ResolvedConfig(**_read_yaml(Path(path)))


def _find_configs_dir(experiment_path: Path) -> Path:
    for parent in [experiment_path.parent, *experiment_path.parents]:
        if (parent / "models.yaml").exists():
            return parent
    raise FileNotFoundError(f"no configs dir with models.yaml above {experiment_path}")


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
