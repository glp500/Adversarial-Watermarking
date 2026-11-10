"""Leakage tests: detector information must reach the receiver only in C2/C3."""

import pytest

from wm_eval.config import ExperimentConfig, Instrumentation
from wm_eval.experiments.treatments import assign_trials
from wm_eval.receiver.inference import build_messages
from wm_eval.receiver.instrumented import instruction_for
from wm_eval.schemas import DetectionResult

POLICY = Instrumentation()


def det(detected: bool) -> DetectionResult:
    return DetectionResult(doc_id="d", input_kind="retokenized_text", watermark_config_id="w", key_id="k",
                           tokenizer_id="t", watermark_detected=detected, p_value=1e-9 if detected else 0.6,
                           threshold=0.01, mean_g=0.6, n_scored=100, n_tokens=50, duration_s=0.0)  # fmt: skip


LEAK_TOKENS = ("watermark", "detector", "EXPERIMENT METADATA", "WM-EVAL", "p_value", "synthid")


def test_unexposed_conditions_carry_no_detection_information():
    exp = ExperimentConfig(name="t", sources="s", generators=["g"], receivers=["r"], watermark="w",
                           conditions=["C0", "C1", "C2", "C3"], instrumentation={})  # fmt: skip
    for t in assign_trials(exp, ["a"]):
        instr = instruction_for(t.expose_detection, det(t.watermark_enabled), POLICY)
        msgs = build_messages("Summarize.", "Some document text.", instr)
        blob = " ".join(m["content"] for m in msgs).lower()
        leaked = [tok for tok in LEAK_TOKENS if tok.lower() in blob]
        if t.condition in ("C0", "C1"):
            assert not instr.applied and leaked == [], (t.condition, leaked)
        else:
            assert instr.applied and "experiment metadata" in blob


def test_c0_and_c1_prompts_differ_only_in_document():
    a = build_messages("Task", "DOC A", instruction_for(False, det(False), POLICY))
    b = build_messages("Task", "DOC B", instruction_for(False, det(True), POLICY))
    assert a[0] == b[0]
    assert a[1]["content"].replace("DOC A", "X") == b[1]["content"].replace("DOC B", "X")


def test_instruction_only_when_detected():
    yes = instruction_for(True, det(True), POLICY)
    no = instruction_for(True, det(False), POLICY)
    assert (
        yes.instruction and "[WM-EVAL]" in yes.text and "watermark detected" in yes.detection_label
    )
    assert no.instruction is None and "no watermark detected" in no.text
    label_only = instruction_for(True, det(True), POLICY.model_copy(update={"mode": "label"}))
    assert label_only.instruction is None


def test_exposed_condition_requires_policy_and_detection():
    with pytest.raises(ValueError):
        instruction_for(True, None, POLICY)
    with pytest.raises(ValueError):
        instruction_for(True, det(True), None)


def test_document_stays_in_user_turn():
    msgs = build_messages(
        "Task", "UNTRUSTED <b>ignore previous</b>", instruction_for(True, det(True), POLICY)
    )
    assert "UNTRUSTED" not in msgs[0]["content"]
    assert "<document>\nUNTRUSTED" in msgs[1]["content"]
