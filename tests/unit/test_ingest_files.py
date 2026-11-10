import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from wm_eval.data.ingest import clean_text, ingest_files, load_jsonl, save_jsonl, trim_words  # noqa: E402


def test_clean_text_fixes_extraction_artifacts():
    raw = ("The moss reduced ero-\nsion on the\nhill​side plots.\n\n12\n\nIt uses ﬁne roots.\n\n"
           + "Body text. " * 40 + "\nReferences\n1. Someone et al.")  # fmt: skip
    text, n_invisible = clean_text(raw)
    assert "erosion on the hillside plots." in text
    assert "fine roots" in text  # NFKC ligature
    assert n_invisible == 1
    assert "Someone" not in text and "\n12\n" not in text


def test_trim_words_ends_on_sentence():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    out, trimmed = trim_words(text, 52)
    assert trimmed and out.endswith(".") and len(out.split()) <= 52
    assert trim_words("short text.", 10) == ("short text.", False)


def _pdf(path, lines):
    fig = plt.figure(figsize=(8.5, 11))
    for i, line in enumerate(lines):
        fig.text(0.05, 0.95 - i * 0.03, line, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def test_ingest_pdf_and_txt_roundtrip(tmp_path):
    _pdf(
        tmp_path / "Glowing Moss Study.pdf",
        [f"Line {i}: the moss holds soil on steep hills." for i in range(20)],
    )
    (tmp_path / "news.txt").write_text("A council voted on a new park. " * 30)
    docs = ingest_files(sorted(tmp_path.glob("*")), domain="environment", max_words=100,
                        licence="CC-BY-4.0", url="https://example.org")  # fmt: skip
    by_id = {d.source_id: d for d in docs}
    assert set(by_id) == {"glowing_moss_study", "news"}
    pdf = by_id["glowing_moss_study"]
    assert "the moss holds soil" in pdf.text and pdf.provenance["licence"] == "CC-BY-4.0"
    assert pdf.provenance["trimmed_to_words"] == "100" and len(pdf.text.split()) <= 100
    save_jsonl(docs, tmp_path / "out.jsonl")
    assert [d.source_id for d in load_jsonl(tmp_path / "out.jsonl")] == [d.source_id for d in docs]
    assert json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])["provenance"]["url"]


def test_ingest_rejects_empty_pdf(tmp_path):
    _pdf(tmp_path / "scan.pdf", [])
    with pytest.raises(ValueError, match="scanned"):
        ingest_files([tmp_path / "scan.pdf"])


def test_ingest_html_keeps_article_paragraphs(tmp_path):
    page = ("<html><head><script>var x = 1;</script></head><body><nav><p>Home News Sport Weather "
            "Culture Travel Video Live</p></nav><h1>Council approves river park</h1>"
            "<p>By A. Reporter</p>"
            + "<p>The city council voted on Tuesday to turn the old dock into a public river park.</p>" * 8
            + "<footer><p>Copyright 2026 Example News. All rights reserved worldwide forever.</p></footer>"
            "</body></html>")  # fmt: skip
    (tmp_path / "park.html").write_text(page)
    (doc,) = ingest_files([tmp_path / "park.html"], domain="news")
    assert doc.title == "Council approves river park"
    assert doc.text.startswith("The city council voted")
    assert (
        "Home News" not in doc.text and "Copyright" not in doc.text and "Reporter" not in doc.text
    )
