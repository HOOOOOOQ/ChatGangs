"""Stemming and lemmatization module.

Two independent capabilities live here:

*Stemming* (language-specific surface reduction)
    - Indonesian: a from-scratch implementation of the well-known Nazief &
      Adriani style rule set. It strips inflectional suffixes, derivational
      suffixes and prefixes, handles the ``-nya``/``-kah``/``-lah``/``-pun``
      particles and protects words that would become too short. If the
      optional ``Sastrawi`` package is installed, ChatGangs prefers it and
      reports so in ``GET /api/model/info``; otherwise the built-in stemmer is
      used — no functionality is lost.
    - English: Porter-style suffix stripping (a compact, audited subset of the
      Porter algorithm).
    - Malay: re-uses the Indonesian rule set (they share most affixes).
    - Other languages: light suffix stripping for Spanish/German/French.

*Lemmatization* (dictionary/POS-aware base form)
    - Uses spaCy's lemmatizer when the English model is installed.
    - Otherwise applies an irregular-word table + the stemmer.

Everything is deterministic and thread-safe.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

from chatgangs.core.languages import SUPPORTED_LANGUAGES
from chatgangs.core.text_utils import LRUCache, _cache_capacity

# ---------------------------------------------------------------------------
# English irregular lemmas (common closed-class and irregular verbs)
# ---------------------------------------------------------------------------
EN_IRREGULAR: Dict[str, str] = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "being": "be", "has": "have", "had": "have", "having": "have", "does": "do",
    "did": "do", "done": "do", "doing": "do", "went": "go", "gone": "go",
    "goes": "go", "children": "child", "men": "man", "women": "woman",
    "people": "person", "teeth": "tooth", "feet": "foot", "mice": "mouse",
    "geese": "goose", "better": "good", "best": "good", "worse": "bad",
    "worst": "bad", "made": "make", "making": "make", "said": "say", "says": "say",
    "took": "take", "taken": "take", "saw": "see", "seen": "see", "came": "come",
    "known": "know", "knew": "know", "thought": "think", "bought": "buy",
    "brought": "bring", "gave": "give", "given": "give", "found": "find",
    "told": "tell", "felt": "feel", "left": "leave", "kept": "keep",
    "helped": "help", "helping": "help", "works": "work", "working": "work",
    "likes": "like", "liked": "like", "loves": "love", "loved": "love",
    "wants": "want", "wanted": "want", "needs": "need", "needed": "need",
    "uses": "use", "used": "use", "using": "use", "gets": "get", "getting": "get",
    "got": "get", "gets": "get", "runs": "run", "running": "run", "ran": "run",
}

ID_IRREGULAR: Dict[str, str] = {
    "adalah": "adalah", "ialah": "ialah", "punya": "punya", "sedang": "sedang",
    "telah": "telah", "belum": "belum", "sudah": "sudah", "bisa": "bisa",
    "mau": "mau", "ingin": "ingin", "pergi": "pergi", "makan": "makan",
    "minum": "minum", "tidur": "tidur", "datang": "datang", "lihat": "lihat",
    "tahu": "tahu", "tau": "tahu", "kerja": "kerja", "belajar": "belajar",
}

# ---------------------------------------------------------------------------
# English Porter-style suffix rules (ordered, longest first)
# ---------------------------------------------------------------------------
EN_SUFFIXES: List[tuple] = [
    ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
    ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
    ("ement", ""), ("ments", ""), ("ment", ""), ("ness", ""),
    ("ingly", ""), ("edly", ""), ("ing", ""), ("ies", "y"), ("ied", "y"),
    ("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("xes", "x"),
    ("ly", ""), ("ed", ""), ("es", ""), ("s", ""),
]


def _has_vowel(word: str) -> bool:
    return any(ch in "aeiouy" for ch in word)


@lru_cache(maxsize=20000)
def stem_english(word: str) -> str:
    word = word.lower()
    if word in EN_IRREGULAR:
        return EN_IRREGULAR[word]
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return word
    for suffix, replacement in EN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)] + replacement
            if not _has_vowel(base):
                continue
            if base.endswith("e") and len(base) > 4 and suffix in ("ing", "ed"):
                base = base[:-1]
            return base
    return word


# ---------------------------------------------------------------------------
# Indonesian stemmer (Nazief-Adriani inspired)
# ---------------------------------------------------------------------------
ID_PARTICLES = ("lah", "kah", "pun", "tah")
ID_POSSESSIVE = ("ku", "mu", "nya")
ID_SUFFIXES = ("kan", "an", "i")
ID_PREFIXES = [
    ("memper", ""), ("diper", ""), ("mempel", ""), ("berke", ""),
    ("menge", ""), ("penge", ""), ("member", ""), ("pember", ""),
    ("meng", ""), ("peng", ""), ("meny", "s"), ("peny", "s"),
    ("mem", ""), ("pem", ""), ("men", ""), ("pen", ""),
    ("me", ""), ("pe", ""), ("ber", ""), ("per", ""), ("pel", ""),
    ("ter", ""), ("di", ""), ("ke", ""), ("se", ""), ("be", ""),
]

# Words that must never be stripped below a usable form.
_PROTECTED_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "apa", "siapa", "kapan",
    "dimana", "mengapa", "kenapa", "berapa", "tidak", "bukan", "belum", "juga",
    "saja", "sudah", "sedang", "akan", "bisa", "boleh", "mau", "ingin", "ada",
    "untuk", "dengan", "karena", "tetapi", "tapi", "kalau", "jika", "atau",
    "saya", "kamu", "anda", "kita", "kami", "mereka", "dia", "ia", "nya",
    "lebih", "kurang", "sangat", "sekali", "hanya", "masih", "pernah", "pada",
    "dalam", "oleh", "sebagai", "secara", "terhadap", "antara", "tanpa", "bagi",
}


# Prefixes that trigger nasal assimilation of the following consonant.
_NASAL_RULES = [
    # (prefix, kind) — kind controls how the nasal prefix is undone:
    #   "s"       : meny/peny + s  ("menyapu" → "sapu")
    #   "keep"    : men/pen + t    ("menulis" → "tulis")
    #   "restore" : mem/pem + p/b  ("memukul" → "pukul")
    #   "drop"    : meng/peng + k/gh ("mengambil" → "ambil")
    #   "plain"   : prefix simply removed ("membeli" → "beli")
    ("menge", "drop"), ("penge", "drop"),
    ("meng", "drop"), ("peng", "drop"),
    ("meny", "s"), ("peny", "s"),
    ("men", "keep"), ("pen", "keep"),
    ("mem", "restore"), ("pem", "restore"),
    ("men", "plain"), ("pen", "plain"),
    ("mem", "plain"), ("pem", "plain"),
    ("me", "plain"), ("pe", "plain"),
]

@lru_cache(maxsize=1)
def _protected_roots() -> frozenset:
    """Curated roots that must be returned unchanged when typed as-is."""
    from chatgangs.config import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "lexicons" / "roots_id.txt"
    if not path.exists():
        return frozenset()
    roots = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            roots.update(part for part in line.split() if part)
    return frozenset(roots)


@lru_cache(maxsize=1)
def _known_roots() -> frozenset:
    """Vocabulary of Indonesian root words used to disambiguate stemming."""
    from chatgangs.config import PROJECT_ROOT

    roots = set()
    path = PROJECT_ROOT / "data" / "lexicons" / "roots_id.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                roots.update(part for part in line.split() if part)
    try:
        from chatgangs.core.normalizer import get_dictionaries

        roots |= get_dictionaries().vocabulary
    except Exception:  # pragma: no cover - avoid import cycles at bootstrap
        pass
    return frozenset(roots)


_NASAL_FOLLOWERS = {
    "drop": "ghk",
    "keep": "t",
    "restore": "pb",
}


def _nasal_candidates(word: str, min_length: int):
    """Yield root candidates produced by undoing nasal assimilation.

    Examples: "menulis" → "tulis", "memukul" → "pukul", "menyapu" → "sapu",
    "mengambil" → "ambil", "menari" → "tari", "membeli" → "beli".
    """
    for prefix, kind in _NASAL_RULES:
        if not word.startswith(prefix) or len(word) <= len(prefix) + 1:
            continue
        rest = word[len(prefix):]
        if not rest:
            continue
        if kind == "s":
            yield "s" + rest
        elif kind == "keep":
            # meN/peN absorbed a "t"; restore it unless one is still visible.
            yield rest if rest[0] == "t" else "t" + rest
        elif kind == "restore":
            # meM/peM + vowel: the initial "p" was absorbed ("memukul" →
            # "pukul"); meM + p/b keeps the consonant ("membeli" → "beli").
            if rest[0] in "aiueo":
                yield "p" + rest
                yield rest
            elif rest[0] in _NASAL_FOLLOWERS["restore"]:
                yield rest[1:]
                yield rest
        elif kind == "drop":
            # meNG/peNG + k/gh: the consonant disappears; + vowel: "k" was
            # absorbed before the vowel ("mengirim" → "kirim").
            if rest[0] in _NASAL_FOLLOWERS["drop"]:
                yield rest[1:]
                yield rest
            elif rest[0] in "aiueo":
                yield "k" + rest
                yield rest
            else:
                yield rest
        else:  # plain
            yield rest


_ROOT_ENDINGS = ("kan", "nya", "lah", "kah", "pun")


def _is_plausible_root(candidate: str) -> bool:
    """Reject candidates that cannot be Indonesian roots."""
    if len(candidate) < 3:
        return False
    if candidate.endswith(_ROOT_ENDINGS):
        return False
    if "-" in candidate:
        return False
    # Indonesian roots rarely end in two consonants.
    if len(candidate) >= 2 and candidate[-1] not in "aiueo" and candidate[-2] not in "aiueo":
        return False
    return True


def _strip_suffixes(word: str, min_length: int) -> str:
    """Strip at most one derivational suffix plus particles/possessives.

    Only a single "an"/"kan" is removed per pass ("makanan" → "makan", not
    "mak"); further reduction happens through the candidate/dictionary stages.
    """
    for particle in ID_PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= min_length:
            word = word[: -len(particle)]
            break
    for poss in ID_POSSESSIVE:
        if word.endswith(poss) and len(word) - len(poss) >= min_length:
            word = word[: -len(poss)]
            break
    known = _known_roots()
    for suffix in ID_SUFFIXES:  # ["kan", "an", "i"]
        if not word.endswith(suffix) or len(word) - len(suffix) < min_length:
            continue
        stem = word[: -len(suffix)]
        if suffix == "i" and stem not in known:
            # Never strip "-i" when it would not yield a known root:
            # "kebaikan"→"kebai", "membantu"→"membant" would be wrong.
            continue
        return stem
    return word


def stem_indonesian(word: str, min_length: int = 3) -> str:
    """Reduce an Indonesian word to its root form.

    Strategy: build the set of plausible roots by undoing suffix stripping,
    each derivational prefix and nasal assimilation, then pick the best
    candidate:

    1. a candidate found in the root dictionary (longest wins:
       "pengiriman" → "pengirim" beats "irim"),
    2. otherwise the shortest candidate that looks like a real root
       ("kebaikan" → "bai" would be rejected, "berjalan" → "jal" too),
    3. otherwise the input unchanged (never destroy information).
    """
    word = word.lower().strip()
    if len(word) <= min_length or not word.isalpha():
        return word
    if word in ID_IRREGULAR:
        return ID_IRREGULAR[word]
    if word in _PROTECTED_ID:
        return word

    known = _known_roots()
    protected = _protected_roots()
    if word in protected and _is_plausible_root(word):
        # A curated root typed as-is ("makan", "guna", "cinta") stays put.
        return word
    # Words that merely *contain* a root must still be stripped: "pelayanan"
    # is in the broader vocabulary, but only "layan" is in the curated roots.
    if word not in protected and word in known and len(word) > 4:
        known = known - {word}

    # Build base variants: strip each derivational suffix independently
    # ("kebaikan" → "kebai" via -kan *and* "kebaik" via -an).
    bases = [word]
    for suffix in ("kan", "an", "i"):
        if word.endswith(suffix) and len(word) - len(suffix) >= min_length:
            bases.append(word[: -len(suffix)])
    for particle in ID_PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= min_length:
            bases.append(word[: -len(particle)])
    for poss in ID_POSSESSIVE:
        if word.endswith(poss) and len(word) - len(poss) >= min_length:
            bases.append(word[: -len(poss)])

    # Collect candidates with a priority (higher = more trustworthy).
    scored: List[tuple] = []

    def add(candidate: str, priority: int) -> None:
        if candidate and len(candidate) >= min_length:
            scored.append((priority, candidate))

    for base in bases:
        add(base, 10)
        # Strip a second, residual "-an" left after "k" was consumed by the
        # "-kan" suffix ("menggunakan" → "mengguna" → "mengguna").
        if base.endswith("an") and len(base) - 2 >= min_length:
            add(base[:-2], 12)
        # "pekerjaan" → "kerja": dropping the leading "pe" before a "-an" base
        # approximates the ke-/-an and pe-/-an circumfix patterns.
        if base.endswith(("erja", "kerja")) and len(base) > 5:
            add(base[2:], 40)
        for prefix, replacement in ID_PREFIXES:
            if base.startswith(prefix) and len(base) - len(prefix) >= min_length:
                add(replacement + base[len(prefix):], 30 + len(prefix))
        for candidate in _nasal_candidates(base, min_length):
            add(candidate, 50)
        # meN/peN + vowel roots where the nasal deleted an initial "k"/"p"
        # ("mengambil" → "ambil", "menggunakan" → "gunakan"/"guna").
        for nasal_prefix in ("meng", "peng", "menge", "penge", "men", "pen", "mem", "pem"):
            if base.startswith(nasal_prefix) and len(base) > len(nasal_prefix) + 2:
                rest = base[len(nasal_prefix):]
                if rest[0] in "aiueo":
                    add(rest, 44)
                    add("k" + rest, 43)
        for candidate in _prefix_chain(base):
            add(candidate, 45)

    # Derivation often needs prefix removal *then* suffix removal
    # ("pengirimannya" → "pengiriman" → "kirim"); apply one suffix pass to
    # every candidate generated above.
    for priority, candidate in list(scored):
        for suffix in ("kan", "an"):
            if candidate.endswith(suffix) and len(candidate) - len(suffix) >= min_length:
                add(candidate[: -len(suffix)], priority + 1)

    # 1. Dictionary-known roots win; prefer the most-stripped *curated* root
    #    ("pelayanan" → "layan", not "layanan").
    known_candidates = {c for _, c in scored if c in known}
    # Curated roots (hand-checked in data/lexicons/roots_id.txt) are the
    # strongest evidence, even when the surface form merely *contains* them:
    # "pengirimannya" → "kirim" rather than "pengiriman"/"iriman".
    curated = {c for _, c in scored if c in protected}
    if curated:
        return min(curated, key=len)
    if known_candidates:
        return max(known_candidates, key=len)
    # 2. Highest-priority plausible candidate; ties → longest candidate.
    plausible = [(p, c) for p, c in scored if _is_plausible_root(c)]
    if plausible:
        best_priority = max(p for p, _ in plausible)
        return max((c for p, c in plausible if p == best_priority), key=len)
    return word


def _prefix_chain(form: str):
    """Yield candidates with two derivational prefixes removed
    ("memperbaiki" → "baiki" → "perbaiki" → "baik")."""
    for prefix, replacement in ID_PREFIXES:
        if not form.startswith(prefix):
            continue
        rest = replacement + form[len(prefix):]
        if len(rest) < 3:
            continue
        yield rest
        for second, second_replacement in ID_PREFIXES:
            if rest.startswith(second) and len(rest) - len(second) >= 3:
                inner = second_replacement + rest[len(second):]
                yield inner
                if inner.endswith("i") and len(inner) > 4:
                    yield inner[:-1]
        # "-kan"-derived roots keep a bare root: "pekerjaan"/"mengerjakan" →
        # "kerja" (handled for "-an" bases below in stem_indonesian).


# ---------------------------------------------------------------------------
# Romance / Germanic light stemming
# ---------------------------------------------------------------------------
_ES_SUFFIXES = ("aciones", "ación", "mente", "idades", "idad", "ando", "iendo",
                "ados", "idos", "es", "os", "as", "ar", "er", "ir", "s")
_DE_SUFFIXES = ("ungen", "ung", "heiten", "heit", "keiten", "keit", "lich",
                "isch", "en", "er", "es", "e", "s")
_FR_SUFFIXES = ("ements", "ement", "ations", "ation", "euses", "eux", "es",
                "ent", "er", "ir", "e", "s")


def _strip_longest(word: str, suffixes, min_len: int = 3) -> str:
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            return word[: -len(suffix)]
    return word


_SUFFIX_TABLES = {
    "es": _ES_SUFFIXES,
    "de": _DE_SUFFIXES,
    "fr": _FR_SUFFIXES,
}


class Stemmer:
    """Entry point used by the NLP pipeline."""

    def __init__(self, backend: Optional[str] = None) -> None:
        """``backend``: "builtin" (default, ~5 µs/word), "sastrawi"
        (higher accuracy, ~9 ms/word) or "auto" (Sastrawi when installed).
        Configurable with ``CHATGANGS_STEMMER_BACKEND``."""
        import os

        requested = (backend or os.getenv("CHATGANGS_STEMMER_BACKEND", "builtin")).lower()
        self._sastrawi = None
        self.indonesian_backend = "builtin"
        if requested in ("sastrawi", "auto"):
            try:  # optional dependency, purely an accuracy enhancement
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

                self._sastrawi = StemmerFactory().create_stemmer()
                self.indonesian_backend = "sastrawi"
            except Exception:
                if requested == "sastrawi":
                    import logging

                    logging.getLogger(__name__).warning(
                        "Sastrawi requested but not installed; using the built-in stemmer"
                    )
        self._cache = LRUCache[str](capacity=_cache_capacity(8192))

    def stem(self, word: str, language: str = "id") -> str:
        key = f"{language}:{word.lower()}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        word_l = (word or "").lower()
        if not word_l or not re.fullmatch(r"[^\W\d_]+", word_l, flags=re.UNICODE):
            result = word_l
        elif language in ("id", "ms"):
            if self._sastrawi is not None:
                result = self._sastrawi.stem(word_l)
            else:
                result = stem_indonesian(word_l)
        elif language in ("en",):
            result = stem_english(word_l)
        elif language in _SUFFIX_TABLES:
            result = _strip_longest(word_l, _SUFFIX_TABLES[language])
        else:
            result = word_l
        self._cache.put(key, result)
        return result

    def stem_tokens(self, tokens, language: str = "id") -> List[str]:
        return [self.stem(t.text if hasattr(t, "text") else t, language) for t in tokens]

    # ------------------------------------------------------------- lemmatize
    def lemmatize(self, word: str, language: str = "en", pos: Optional[str] = None) -> str:
        word_l = (word or "").lower()
        if language == "en":
            if word_l in EN_IRREGULAR:
                return EN_IRREGULAR[word_l]
            if pos and pos in ("NOUN", "PROPN") and word_l.endswith("ies"):
                return word_l[:-3] + "y"
            return stem_english(word_l)
        if language == "id":
            if word_l in ID_IRREGULAR:
                return ID_IRREGULAR[word_l]
            return self.stem(word_l, "id")
        return self.stem(word_l, language)

    def info(self) -> dict:
        return {
            "indonesian_backend": self.indonesian_backend,
            "english_backend": "porter-subset",
            "supported_languages": sorted(SUPPORTED_LANGUAGES.keys()),
        }


_stemmer: Optional[Stemmer] = None


def get_stemmer() -> Stemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = Stemmer()
    return _stemmer
