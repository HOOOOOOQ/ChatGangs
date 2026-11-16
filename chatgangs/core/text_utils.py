"""Shared helpers for the NLP core: caching, text cleaning and token models."""

from __future__ import annotations

import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")

# Zero-width and control characters that commonly arrive from copy/paste.
_INVISIBLE = {
    0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD,
}


def _cache_capacity(default: int) -> int:
    """Scale module caches from ``CHATGANGS_NLP_CACHE_SIZE``.

    The setting is the budget for *one* cache; larger caches (e.g. the stemmer)
    keep their relative proportion but stay bounded by ``capacity`` checks.
    """
    try:
        from chatgangs.config import get_settings

        configured = int(get_settings().nlp_cache_size)
    except Exception:  # pragma: no cover - bootstrap
        configured = 0
    if configured <= 0:
        return default
    # ``nlp_cache_size`` is expressed in the same unit as the 2048-entry
    # reference budget, so a value of 4096 doubles every module cache.
    scaled = int(default * (configured / 2048))
    return max(64, min(scaled, 100_000))


class LRUCache(Generic[T]):
    """Small thread-safe LRU cache used to memoise deterministic NLP calls."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = max(int(capacity), 1)
        self._data: "OrderedDict[str, T]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "capacity": self.capacity,
                "size": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


@dataclass
class Token:
    """A single token produced by the tokenizer."""

    text: str
    normalized: str
    index: int
    start: int
    end: int
    is_word: bool = True
    is_stopword: bool = False
    is_punctuation: bool = False
    is_emoji: bool = False
    is_url: bool = False
    is_email: bool = False
    is_mention: bool = False
    is_hashtag: bool = False
    is_number: bool = False
    pos: Optional[str] = None
    lemma: Optional[str] = None
    stem: Optional[str] = None
    entity_label: Optional[str] = None

    def to_dict(self, include_offsets: bool = True) -> dict:
        data = {
            "text": self.text,
            "normalized": self.normalized,
            "index": self.index,
            "is_word": self.is_word,
            "is_stopword": self.is_stopword,
            "is_punctuation": self.is_punctuation,
        }
        if include_offsets:
            data.update({"start": self.start, "end": self.end})
        for flag in ("is_emoji", "is_url", "is_email", "is_mention", "is_hashtag", "is_number"):
            if getattr(self, flag):
                data[flag] = True
        if self.pos:
            data["pos"] = self.pos
        if self.lemma:
            data["lemma"] = self.lemma
        if self.stem:
            data["stem"] = self.stem
        if self.entity_label:
            data["entity_label"] = self.entity_label
        return data


@dataclass
class Entity:
    """A named entity extracted from the text."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "rule"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Span:
    """A reusable text span (used by phrase-matching modules)."""

    text: str
    start: int
    end: int
    meta: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Remove control/zero-width characters and normalise whitespace."""
    if not text:
        return ""
    out = []
    for ch in text:
        code = ord(ch)
        if code in _INVISIBLE:
            out.append(" ")  # keep the word boundary ("Halo\u200bdunia")
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn") and ch not in "\n\t":
            out.append(" ")
            continue
        out.append(ch)
    normalized = unicodedata.normalize("NFKC", "".join(out))
    return normalized


def truncate(text: str, max_chars: int) -> str:
    """Truncate safely on a codepoint boundary, appending an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 1, 0)].rstrip() + "…"


def safe_ratio(part: float, total: float, default: float = 0.0) -> float:
    return round(part / total, 4) if total else default


def to_plain(value: Any) -> Any:
    """Recursively convert numpy/pydantic-ish values to JSON-serialisable ones."""
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    return value


def split_sentences(text: str) -> List[str]:
    """Lightweight multilingual sentence splitter (no extra dependency)."""
    if not text:
        return []
    sentences: List[str] = []
    buffer = []
    closers = "\"')]»”’"
    abbrev = ("mr.", "mrs.", "ms.", "dr.", "prof.", "no.", "vs.", "etc.", "e.g.", "i.e.", "st.")
    tokens = text.replace("\n", " \n ").split(" ")
    for token in tokens:
        if token == "\n":
            if buffer:
                sentences.append(" ".join(buffer).strip())
                buffer = []
            continue
        buffer.append(token)
        stripped = token.rstrip(closers)
        if not stripped:
            continue
        ends = stripped[-1] in ".!?。！？"
        if ends:
            # Spanish ¿¡ and ellipsis handling
            if stripped.endswith("...") and not token.endswith("..."):
                continue
            if stripped.lower() in abbrev:
                continue
            if stripped[-1] == "." and len(stripped) == 2 and stripped[0].isupper():
                continue  # initials like "J."
            sentences.append(" ".join(buffer).strip())
            buffer = []
    if buffer:
        sentences.append(" ".join(buffer).strip())
    return [s for s in sentences if s]
