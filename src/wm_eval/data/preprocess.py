"""Text normalisation and sentence segmentation (spaCy rule-based, no model download)."""

from __future__ import annotations

import re
from functools import lru_cache

_WS = re.compile(r"[ \t]+")
_WORD = re.compile(r"\b[\w'-]+\b")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0]


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    return nlp


def sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    out = []
    for block in text.split("\n"):  # treat line breaks (bullets, lists) as boundaries
        if block.strip():
            out += [s.text.strip() for s in _nlp()(block).sents if s.text.strip()]
    return out


def words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def chunk_sentences(sents: list[str], max_words: int = 200) -> list[str]:
    """Greedy packing of sentences into chunks of at most ~max_words words."""
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(words(s))
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks
