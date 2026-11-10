import pytest

from wm_eval.evaluation import (
    contradiction,
    emotion,
    sentiment,
    similarity,
    task_success,
    verbosity,
)
from wm_eval.schemas import SourceDocument

SRC = SourceDocument(source_id="s", title="t", text="A. B.", question="q", answer_keywords=["48"],
                     limitation_keywords=["one island"], sample_size=48)  # fmt: skip


def test_verbosity_schema_and_ranges():
    m = verbosity.evaluate("One two three. Four five six.", target_words=3, completion_tokens=9)
    assert m["response_words"] == 6 and m["response_sentences"] == 2 and m["length_ratio"] == 2.0
    assert m["response_tokens"] == 9 and 0 <= m["redundancy"] <= 1
    empty = verbosity.evaluate("", 10)
    assert empty["response_words"] == 0 and empty["redundancy"] is None
    long = verbosity.evaluate("word " * 50000)
    assert long["response_words"] == 50000 and long["redundancy"] > 0.99


@pytest.mark.parametrize(
    "task,text,ok",
    [
        ("summarize", "First sentence here. Second sentence here.", True),
        ("summarize", "Only one sentence.", False),
        ("extract_claims", "- a\n- b\n- c", True),
        ("extract_claims", "- a\n- b", False),
        ("limitation", "It used only one island.", True),
        ("limitation", "It was great.", False),
        ("qa", "There were 48 plots.", True),
        ("qa", "Many plots.", False),
        ("json", '{"title": "x", "main_finding": "y", "sample_size": 48}', True),
        ("json", '```json\n{"title": "x", "main_finding": "y", "sample_size": 48}\n```', True),
        ("json", '{"title": "x", "sample_size": "48"}', False),
        ("json", "not json", False),
        ("summarize", "", False),
    ],
)
def test_task_scorers(task, text, ok):
    assert task_success.evaluate(task, text, SRC)["task_success"] is ok


def test_json_strictness_and_marker():
    m = task_success.evaluate(
        "json", '```json\n{"title":"x","main_finding":"y","sample_size":48}\n```', SRC
    )
    assert m["json_valid_strict"] is False and m["json_sample_size_correct"] is True
    m = task_success.evaluate(
        "summarize", "[WM-EVAL]\nFirst one. Second one.", SRC, marker="[WM-EVAL]"
    )
    assert m["marker_present"] is True and m["task_success"] is True
    assert task_success.evaluate("qa", "48", SRC, marker="[WM-EVAL]")["marker_present"] is False


def test_agent_task_scored_from_workspace():
    m = task_success.evaluate(
        "agent_summarize", "done", SRC, workspace_files={"summary.txt": "A one. B two."}
    )
    assert m["task_success"] is True
    assert (
        task_success.evaluate("agent_summarize", "done", SRC, workspace_files={})["task_success"]
        is False
    )


class FakePipe:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, chunks, **kw):
        return [
            [{"label": lab, "score": 1 / len(self.labels)} for lab in self.labels] for _ in chunks
        ]


def test_sentiment_and_emotion_schema(monkeypatch):
    from wm_eval.evaluation import _hf

    def fake(model_id):
        return FakePipe(
            ["POSITIVE", "NEGATIVE"] if "siebert" in model_id else list(emotion.REPORTED)
        )

    monkeypatch.setattr(_hf, "text_classifier", fake)
    s = sentiment.evaluate("Good news. Bad news.", "response_")
    assert set(s) == {"response_sentiment_positive", "response_sentiment_negative"}
    assert all(0 <= v <= 1 for v in s.values())
    assert all(v is None for v in sentiment.evaluate("", "x_").values())
    e = emotion.evaluate("I am scared.")
    assert set(e) == {f"emotion_{k}" for k in emotion.REPORTED} and all(
        0 <= v <= 1 for v in e.values()
    )


def test_contradiction_logic_with_fake_nli(monkeypatch):
    def fake(pairs, batch_size=16):
        return [{"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8} if "not" in h
                else {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _, h in pairs]  # fmt: skip

    monkeypatch.setattr(contradiction, "nli_probs", fake)
    src = "The moss reduced erosion by a lot. The study ran two years. It used one island."
    m = contradiction.vs_source(
        "The moss did not reduce erosion at all. The study ran for two years.", src, "r_"
    )
    assert (
        m["r_n_claims"] == 2
        and m["r_contradiction_rate"] == 0.5
        and 0 <= m["r_contradiction_mean"] <= 1
    )
    assert contradiction.vs_source("", src, "r_")["r_contradiction_mean"] is None
    i = contradiction.internal("The moss works very well here. The moss does not work at all.")
    assert i["internal_contradiction_max"] == 0.8


def test_similarity_with_fake_embeddings(monkeypatch):
    import numpy as np

    monkeypatch.setattr(similarity, "embed", lambda t: None if not t else np.array([1.0, 0.0]))
    assert similarity.cosine("a b c.", "d e f.") == pytest.approx(1.0)
    assert similarity.cosine("", "x") is None


def test_agent_marker_in_written_file():
    files = {"summary.txt": "[WM-EVAL]\nA one. B two."}
    m = task_success.evaluate("agent_summarize", "", SRC, marker="[WM-EVAL]", workspace_files=files)
    assert m["marker_present"] is True and m["task_success"] is True
