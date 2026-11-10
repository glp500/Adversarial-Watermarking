import json

import pytest

from wm_eval.data.ingest import load_jsonl, save_jsonl
from wm_eval.data.preprocess import chunk_sentences, normalize, sentences
from wm_eval.schemas import SourceDocument


def test_roundtrip_and_subset(tmp_path):
    docs = [SourceDocument(source_id=f"s{i}", title="t", text=f"Text {i}.") for i in range(3)]
    save_jsonl(docs, tmp_path / "c.jsonl")
    assert load_jsonl(tmp_path / "c.jsonl") == docs
    assert [d.source_id for d in load_jsonl(tmp_path / "c.jsonl", ["s1"])] == ["s1"]
    with pytest.raises(ValueError, match="not found"):
        load_jsonl(tmp_path / "c.jsonl", ["nope"])
    with pytest.raises(FileExistsError):
        save_jsonl(docs, tmp_path / "c.jsonl")


def test_duplicate_ids_rejected(tmp_path):
    row = json.dumps({"source_id": "a", "title": "t", "text": "x"})
    (tmp_path / "d.jsonl").write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(tmp_path / "d.jsonl")


def test_preprocess():
    assert normalize("a  b\r\n\n\n\nc") == "a b\n\nc"
    assert sentences("One here. Two here.\n- bullet") == ["One here.", "Two here.", "- bullet"]
    assert sentences("") == []
    assert chunk_sentences(["a b c", "d e f", "g"], max_words=4) == ["a b c", "d e f g"]
