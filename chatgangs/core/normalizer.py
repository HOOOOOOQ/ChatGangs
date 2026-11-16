"""Text normalisation, slang expansion, typo correction and dictionary loading."""

from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from chatgangs.config import PROJECT_ROOT
from chatgangs.core.languages import SUPPORTED_LANGUAGES
from chatgangs.core.text_utils import LRUCache, _cache_capacity, clean_text

DATA_DIR = PROJECT_ROOT / "data"

_REPEAT_RE = re.compile(r"(.)\1{2,}")  # runs of 3+ → single character
_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# Emoji → shortcode-ish description used only when the caller wants text-only
# normalisation (NEVER applied by default so sentiment on emoji survives).
DEFAULT_KEEP = ("emoji", "url", "hashtag", "mention")


def _read_pairs(path: Path) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    if not path.exists():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t+", line)
        if len(parts) < 2:
            continue
        key, value = parts[0].strip().lower(), parts[1].strip()
        if key and value:
            pairs[key] = value
    return pairs


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.extend(part for part in re.split(r"[\s,]+", line) if part)
    return items


def _read_sections(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse ``[section]`` formatted lexicon files into weights."""
    sections: Dict[str, Dict[str, float]] = {}
    current = None
    if not path.exists():
        return sections
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower()
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        parts = re.split(r"\t+", line)
        if len(parts) < 2:
            # fall back to last-token-as-weight ("luar biasa 0.9")
            pieces = line.rsplit(None, 1)
            if len(pieces) != 2:
                continue
            parts = pieces
        key = parts[0].strip().lower()
        try:
            weight = float(parts[1])
        except ValueError:
            continue
        if key:
            sections[current][key] = weight
    return sections


class DictionaryStore:
    """Loads every static resource once per process."""

    _instance: Optional["DictionaryStore"] = None

    def __init__(self) -> None:
        self.stopwords: Dict[str, Set[str]] = {}
        for code in SUPPORTED_LANGUAGES:
            path = DATA_DIR / "stopwords" / f"{code}.txt"
            self.stopwords[code] = {w.lower() for w in _read_list(path)}
        # Indonesian and English share a few tokens; guarantee the important
        # negation words are never dropped as pure stopwords at analysis time.
        self.stopwords["id"] -= {"tidak", "bukan", "belum", "jangan", "tak", "kurang"}
        self.stopwords["en"] -= {"not", "no", "never", "without", "hardly", "barely"}

        self.slang: Dict[str, str] = _read_pairs(DATA_DIR / "spelling" / "slang.txt")
        self.corrections: Dict[str, str] = _read_pairs(DATA_DIR / "spelling" / "corrections.txt")
        self.sentiment: Dict[str, Dict[str, float]] = {
            "id": _read_sections(DATA_DIR / "lexicons" / "sentiment_id.txt"),
            "en": _read_sections(DATA_DIR / "lexicons" / "sentiment_en.txt"),
        }
        self.emoticons: Dict[str, float] = {
            token: weight
            for token, weight in _read_pairs_weighted(DATA_DIR / "lexicons" / "emoticons.txt").items()
        }
        self.vocabulary: Set[str] = set()
        for lang, markers in SUPPORTED_LANGUAGES.items():
            self.vocabulary |= self.stopwords.get(lang, set())
        # Intent examples, knowledge-base variants and QA keywords are real,
        # commonly-typed sentences and make the vocabulary far more complete.
        self.vocabulary |= self._load_sentence_vocabulary()
        for section_map in self.sentiment.values():
            for section in section_map.values():
                self.vocabulary |= set(section)
        self.vocabulary |= set(self.slang.values())
        self.vocabulary |= set(self.corrections.values())
        self.vocabulary = {v.lower() for v in self.vocabulary if v and " " not in v}

    @staticmethod
    def _load_sentence_vocabulary() -> Set[str]:
        import json

        words: Set[str] = set()
        intent_path = DATA_DIR / "intents" / "intents.json"
        if intent_path.exists():
            try:
                payload = json.loads(intent_path.read_text(encoding="utf-8"))
                for intent in payload.get("intents", []):
                    for example in intent.get("examples", []):
                        words.update(re.findall(r"[^\W\d_]+", example.lower()))
            except Exception:  # pragma: no cover
                pass
        kb_path = DATA_DIR / "qa" / "knowledge_base.json"
        if kb_path.exists():
            try:
                payload = json.loads(kb_path.read_text(encoding="utf-8"))
                for entry in payload.get("entries", []):
                    for text in [entry.get("question", "")] + entry.get("variants", []):
                        words.update(re.findall(r"[^\W\d_]+", text.lower()))
            except Exception:  # pragma: no cover
                pass
        return words

    @classmethod
    def instance(cls) -> "DictionaryStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def stopword_set(self, language: str) -> Set[str]:
        if language in self.stopwords:
            return self.stopwords[language]
        if language == "zh" or language == "ja":
            return self.stopwords.get("ja", set())
        return self.stopwords["en"]


def _read_pairs_weighted(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t+", line)
        if len(parts) < 2:
            continue
        try:
            out[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


@lru_cache(maxsize=1)
def get_dictionaries() -> DictionaryStore:
    return DictionaryStore.instance()


class TextNormalizer:
    """Normalisation pipeline used before tokenisation / classification.

    Steps (configurable):

    1. Unicode NFKC + removal of invisible characters.
    2. Redaction of e-mail addresses and long digit runs (PII safety).
    3. Case folding.
    4. Repeated-character squeezing (``bagusss`` → ``bagus``).
    5. Slang / informal word expansion (Indonesian + English).
    6. Optional typo correction against the known vocabulary.
    7. Whitespace collapse.
    """

    def __init__(self, store: Optional[DictionaryStore] = None) -> None:
        self.store = store or get_dictionaries()
        self._cache = LRUCache[str](capacity=_cache_capacity(4096))

    # ------------------------------------------------------------------ words
    def normalize_word(self, word: str, *, correct_typos: bool = True) -> str:
        word = word.lower().strip()
        if not word:
            return ""
        if word in self.store.slang:
            return self.store.slang[word]
        # "bagussss" -> "bagus", "bangetttt" -> "banget"
        squeezed = _REPEAT_RE.sub(r"\1", word)
        for variant in (word, squeezed):
            if variant in self.store.slang:
                return self.store.slang[variant]
            if correct_typos and variant in self.store.corrections:
                return self.store.corrections[variant]
        if correct_typos:
            for variant in (squeezed, word):
                if variant in self.store.vocabulary or len(variant) < 5 or not variant.isalpha():
                    continue
                candidate = difflib.get_close_matches(
                    variant, self.store.vocabulary, n=1, cutoff=0.90
                )
                # Require the same first letter and a very high ratio: this
                # prevents valid words from being "corrected" into unrelated
                # vocabulary entries ("tinggal" ≠ "tanggal").
                if candidate and candidate[0][0] == variant[0]:
                    ratio = difflib.SequenceMatcher(None, variant, candidate[0]).ratio()
                    if ratio >= 0.90:
                        return candidate[0]
        if squeezed in self.store.vocabulary or word in self.store.vocabulary:
            return squeezed if squeezed in self.store.vocabulary else word
        return squeezed if len(squeezed) > 2 else word

    # ------------------------------------------------------------------- text
    def normalize(
        self,
        text: str,
        *,
        language: str = "id",
        keep: Iterable[str] = DEFAULT_KEEP,
        correct_typos: bool = True,
        redact_pii: bool = True,
        use_cache: bool = True,
    ) -> str:
        keep = tuple(keep)
        cache_key = f"{language}|{correct_typos}|{redact_pii}|{','.join(keep)}|{text}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        value = clean_text(text or "")
        value = _ZERO_WIDTH_RE.sub("", value)
        value = unicodedata.normalize("NFKC", value)
        if redact_pii:
            value = redact(value)
        value = value.lower()

        def _word(match: re.Match) -> str:
            token = match.group(0)
            if token in keep:
                return token
            return self.normalize_word(token, correct_typos=correct_typos)

        # Protect URLs / mentions / hashtags / emoji while normalising words.
        protected: List[str] = []

        def _protect(match: re.Match) -> str:
            protected.append(match.group(0))
            return f"\x00{len(protected) - 1}\x00"

        working = re.sub(r"https?://\S+|@\w+|#\w+", _protect, value)
        working = re.sub(r"[^\W\d_]+(?:'[^\W\d_]+)?", _word, working, flags=re.UNICODE)
        if "emoji" in keep:
            working = _REPEAT_RE.sub(r"\1\1", working)
        working = _SPACE_RE.sub(" ", working).strip()

        def _restore(match: re.Match) -> str:
            idx = int(match.group(1))
            return protected[idx] if idx < len(protected) else ""

        working = re.sub(r"\x00(\d+)\x00", _restore, working)
        if use_cache:
            self._cache.put(cache_key, working)
        return working

    def cache_stats(self) -> dict:
        return self._cache.stats()


def redact(text: str) -> str:
    """Redact e-mail addresses and phone-like digit runs for privacy."""
    text = re.sub(
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", text, flags=re.IGNORECASE
    )
    # Long digit sequences (>= 9) are treated as account/phone numbers.
    text = re.sub(r"\b\+?\d[\d\s\-()]{7,}\d\b", "[NUMBER]", text)
    return text


def strip_diacritics(text: str) -> str:
    """Remove combining marks (useful for matching Spanish/German/French)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


_normalizer: Optional[TextNormalizer] = None


def get_normalizer() -> TextNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer
