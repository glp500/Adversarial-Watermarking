"""Load source documents. Sources are read-only inputs; runs copy them into artifacts."""

from __future__ import annotations

import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from wm_eval.schemas import SourceDocument


def load_jsonl(path: str | Path, source_ids: list[str] | None = None) -> list[SourceDocument]:
    docs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                docs.append(SourceDocument(**json.loads(line)))
    ids = [d.source_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source_id in {path}")
    if source_ids is not None:
        missing = set(source_ids) - set(ids)
        if missing:
            raise ValueError(f"source_ids not found in {path}: {sorted(missing)}")
        docs = [d for d in docs if d.source_id in source_ids]
    return docs


def load_hf_dataset(
    name: str,
    split: str,
    text_field: str,
    id_field: str | None = None,
    title_field: str | None = None,
    limit: int | None = None,
    domain: str = "general",
) -> list[SourceDocument]:
    """Pull documents from a Hugging Face dataset (e.g. to build a larger corpus)."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        SourceDocument(
            source_id=str(row[id_field]) if id_field else f"{name.replace('/', '_')}_{i}",
            title=str(row[title_field]) if title_field else "",
            text=row[text_field],
            domain=domain,
        )
        for i, row in enumerate(ds)
    ]


def save_jsonl(docs: list[SourceDocument], path: str | Path) -> None:
    with open(path, "x") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")


# --- Real documents (PDF / plain text) -> SourceDocument rows ---------------------------

# Zero-width and bidi-control characters. They are invisible to readers but tokenized by
# models, so a source could smuggle text past a human reviewer; strip them on the way in.
_INVISIBLE = re.compile(
    r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]"
)
_REFS = re.compile(r"^\s*(references|bibliography|works cited|literature cited)\s*$", re.I | re.M)


def extract_pdf_text(path: str | Path) -> tuple[str, str]:
    """Return (title, raw text) from a text-based PDF. Scanned PDFs (images only) give
    empty text; OCR them first."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    title = ((reader.metadata.title if reader.metadata else None) or "").strip()
    if len(title.split()) < 3:  # metadata titles are often file ids like "pone.0013255 1..8"
        title = ""
    return title, raw


class _ParagraphText(HTMLParser):
    """Collect the text of <h1> and <p> elements, skipping scripts, styles, nav and footers."""

    SKIP = {"script", "style", "nav", "footer", "aside", "form", "figure"}

    def __init__(self):
        super().__init__()
        self.title, self.paras, self._buf, self._in, self._skip = "", [], [], None, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "h1"} and not self._skip:
            self._in, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == self._in:
            text = " ".join("".join(self._buf).split())
            if tag == "h1" and not self.title:
                self.title = text
            elif tag == "p" and len(text.split()) >= 8:  # drop bylines, captions, buttons
                self.paras.append(text)
            self._in = None

    def handle_data(self, data):
        if self._in and not self._skip:
            self._buf.append(data)


def extract_html_text(path: str | Path) -> tuple[str, str]:
    """Return (title, paragraph text) from a saved web page (e.g. a news article)."""
    parser = _ParagraphText()
    parser.feed(Path(path).read_text(encoding="utf-8", errors="replace"))
    return parser.title, "\n\n".join(parser.paras)


def clean_text(raw: str) -> tuple[str, int]:
    """Normalise extracted text. Returns (text, number of invisible characters removed).

    - NFKC (ligatures such as "ﬁ" -> "fi"), invisible characters removed
    - everything after a References/Bibliography heading dropped
    - words split across lines re-joined, lines inside a paragraph joined
    - bare page-number lines dropped, whitespace collapsed
    """
    text = unicodedata.normalize("NFKC", raw)
    text, n_invisible = _INVISIBLE.subn("", text)
    m = _REFS.search(text)
    if m and m.start() > len(text) * 0.3:  # only a real trailing reference section
        text = text[: m.start()]
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.M)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p), n_invisible


def trim_words(text: str, max_words: int) -> tuple[str, bool]:
    """Cut to at most max_words, ending at a sentence boundary when one is close."""
    words = text.split(" ")
    if len(text.split()) <= max_words:
        return text, False
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
    return (cut[: end + 1] if end > len(cut) * 0.8 else cut).rstrip(), True


def ingest_files(
    paths: list[str | Path],
    domain: str = "general",
    max_words: int = 2000,
    licence: str = "",
    url: str = "",
) -> list[SourceDocument]:
    """Turn .pdf / .html / .txt / .md files into SourceDocuments with provenance."""
    docs = []
    for path in map(Path, paths):
        if path.suffix.lower() == ".pdf":
            title, raw = extract_pdf_text(path)
        elif path.suffix.lower() in {".html", ".htm"}:
            title, raw = extract_html_text(path)
        elif path.suffix.lower() in {".txt", ".md"}:
            title, raw = "", path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"unsupported file type: {path} (use .pdf, .html, .txt or .md)")
        text, n_invisible = clean_text(raw)
        if len(text.split()) < 50:
            raise ValueError(
                f"{path}: only {len(text.split())} words extracted; is it a scanned PDF?"
            )
        full_words = len(text.split())
        text, trimmed = trim_words(text, max_words)
        if not title:
            title = " ".join(text.split()[:20])
        slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:60]
        docs.append(
            SourceDocument(
                source_id=slug,
                title=title,
                text=text,
                domain=domain,
                provenance={
                    "file": path.name,
                    "url": url,
                    "licence": licence,
                    "words_extracted": str(full_words),
                    "trimmed_to_words": str(max_words) if trimmed else "",
                    "invisible_chars_removed": str(n_invisible),
                },
            )
        )
    ids = [d.source_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"two files map to the same source_id (rename one): {ids}")
    return docs
