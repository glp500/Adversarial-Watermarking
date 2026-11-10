"""Instrumented-receiver adapter (conditions C2/C3 only).

Maps a *verified detector result* to an explicitly configured, benign experimental
instruction. Kept separate from the detector (which only computes statistics) and from
the model client (which only transports messages). Nothing here runs for C0/C1.
"""

from __future__ import annotations

from wm_eval.config import Instrumentation
from wm_eval.schemas import DetectionResult, Instruction

NOT_APPLIED = Instruction(applied=False)


def instruction_for(
    expose_detection: bool, detection: DetectionResult | None, policy: Instrumentation | None
) -> Instruction:
    if not expose_detection:
        return NOT_APPLIED
    if policy is None or detection is None:
        raise ValueError(
            "an exposed condition needs both an instrumentation policy and a detection"
        )
    decision = "watermark detected" if detection.watermark_detected else "no watermark detected"
    label = policy.label_template.format(decision=decision)
    instruction = None
    if policy.mode == "instruction":
        instruction = (
            policy.instruction_if_detected
            if detection.watermark_detected
            else policy.instruction_if_not_detected
        )
    text = label if not instruction else f"{label}\n{instruction}"
    return Instruction(applied=True, detection_label=label, instruction=instruction, text=text)
