import json

import pytest

from wm_eval.observability.events import Recorder, read_events, redact
from wm_eval.observability.storage import RunDir, new_run_id, write_once


def test_run_ids_unique():
    assert len({new_run_id("x") for _ in range(200)}) == 200


def test_events_are_jsonl_with_timestamp_and_ids(tmp_path):
    rec = Recorder(tmp_path / "events.jsonl", "run1")
    rec.emit("watermark.detected", trial_id="t1", data={"p_value": 0.001, "detected": True})
    rec.emit("experiment.completed")
    ev = read_events(tmp_path / "events.jsonl")
    assert [e["event"] for e in ev] == ["watermark.detected", "experiment.completed"]
    assert (
        ev[0]["run_id"] == "run1" and ev[0]["trial_id"] == "t1" and ev[0]["ts"].endswith("+00:00")
    )


def test_secrets_are_redacted(tmp_path):
    rec = Recorder(tmp_path / "e.jsonl", "r")
    rec.emit(
        "x",
        data={"keys": [1, 2, 3], "nested": {"api_key": "sk-123", "WM_SECRET": "s"}, "key_id": "ok"},
    )
    text = (tmp_path / "e.jsonl").read_text()
    assert "sk-123" not in text and "[1, 2, 3]" not in text and '"s"' not in text
    assert "ok" in text  # key_id is a public label
    assert redact({"a": [{"secret": 1}]}) == {"a": [{"secret": "***REDACTED***"}]}


def test_artifacts_are_write_once(tmp_path):
    rd = RunDir(tmp_path, "r1").create()
    rd.save("generated", "d1", {"a": 1})
    with pytest.raises(FileExistsError):
        rd.save("generated", "d1", {"a": 2})
    assert json.loads(rd.artifact("generated", "d1").read_text()) == {"a": 1}
    with pytest.raises(FileExistsError):
        write_once(rd.artifact("generated", "d1"), {"b": 1})
