"""Typed records that flow between pipeline stages and are persisted as artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReceiverMode = Literal["watermark_only", "detector_aware", "instrumented"]


class SourceDocument(BaseModel):
    """An original (trusted-provenance, untrusted-content) article. Never modified."""

    source_id: str
    title: str
    text: str
    domain: str = "general"
    # Optional ground truth used by deterministic task scorers.
    question: str | None = None
    answer_keywords: list[str] = Field(default_factory=list)
    limitation_keywords: list[str] = Field(default_factory=list)
    sample_size: int | None = None
    # Where the text came from: file, url, licence, extraction notes. Free-form, kept with the run.
    provenance: dict[str, str] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    source_id: str
    prompt: str
    generator: str  # key in models.yaml
    watermark_enabled: bool
    seed: int


class GenerationResult(BaseModel):
    doc_id: str
    run_id: str
    source_id: str
    generator: str
    watermark_enabled: bool
    watermark_config_id: str | None
    watermark_key_id: str | None
    seed: int
    prompt: str
    prompt_token_ids: list[int]
    token_ids: list[int]  # generated tokens only, from the generator's own tokenizer
    text: str
    finish_reason: Literal["eos", "length"]
    duration_s: float
    metadata: dict[str, Any] = Field(default_factory=dict)  # model ids, revisions, params


class DetectionResult(BaseModel):
    doc_id: str
    method: str = "synthid_frequentist_binomial"
    input_kind: Literal["token_ids", "retokenized_text"]
    watermark_config_id: str
    key_id: str
    tokenizer_id: str
    watermark_detected: bool
    # P(score >= observed | text not watermarked with this key). NOT P(watermarked).
    p_value: float
    threshold: float
    mean_g: float  # SynthID mean score; ~0.5 for unwatermarked text
    n_scored: int  # number of unmasked g-values that entered the test
    n_tokens: int
    duration_s: float


class ReceiverResult(BaseModel):
    trial_id: str
    run_id: str
    receiver: str
    status: Literal["ok", "failed"]
    request: dict[str, Any]  # full payload sent to llama.cpp
    response: dict[str, Any] | None  # full raw payload returned
    text: str | None
    reasoning_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_s: float
    error: str | None = None
    tool_trace: list[dict[str, Any]] | None = None  # agent tasks only
    workspace_files: dict[str, str] | None = None  # agent tasks only: final sandbox state


class Trial(BaseModel):
    """One planned receiver call. `trial_id` is a deterministic hash of the fields."""

    trial_id: str
    pair_key: str  # identical for trials that differ only by condition
    source_id: str
    generator: str
    receiver: str
    task: str
    condition: Literal["C0", "C1", "C2", "C3"]
    watermark_enabled: bool
    expose_detection: bool
    receiver_mode: ReceiverMode
    generation_repeat: int
    generation_seed: int
    receiver_repeat: int
    receiver_seed: int
    doc_id: str


class Instruction(BaseModel):
    """What (if anything) the instrumented receiver adapter adds to the prompt."""

    applied: bool
    detection_label: str | None = None
    instruction: str | None = None
    text: str | None = None  # the exact system text injected


class EvaluationResult(BaseModel):
    trial_id: str
    metrics: dict[str, float | int | bool | None]
    evaluator_versions: dict[str, str]
    duration_s: float
    errors: dict[str, str] = Field(default_factory=dict)
