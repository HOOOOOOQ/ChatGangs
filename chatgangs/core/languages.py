"""Language registry and detection.

ChatGangs prioritises Indonesian and English; Malay, Japanese, Korean,
Spanish, German and French are supported by the language processor.

Detection strategy (in order, first confident signal wins):

1. Script analysis — Japanese kana, Hangul, Han, Arabic, Cyrillic, Thai.
2. Lexical overlap with per-language function-word dictionaries
   (deterministic, explainable and dependency-free).
3. N-gram statistical detection through ``langdetect`` when installed.

The overlap model is deliberately used for Indonesian/Malay disambiguation
because general-purpose statistical detectors frequently confuse the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "id": {"name": "Indonesian", "native": "Bahasa Indonesia"},
    "en": {"name": "English", "native": "English"},
    "ms": {"name": "Malay", "native": "Bahasa Melayu"},
    "ja": {"name": "Japanese", "native": "日本語"},
    "ko": {"name": "Korean", "native": "한국어"},
    "es": {"name": "Spanish", "native": "Español"},
    "de": {"name": "German", "native": "Deutsch"},
    "fr": {"name": "French", "native": "Français"},
}

PRIORITY_LANGUAGES = ("id", "en")

# Function words / high-frequency markers per language.
LANGUAGE_MARKERS: Dict[str, Tuple[str, ...]] = {
    "id": (
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
        "tidak", "saya", "aku", "kamu", "anda", "kita", "kami", "mereka",
        "adalah", "akan", "sudah", "belum", "bisa", "boleh", "apa", "apakah",
        "bagaimana", "bagaimanakah", "mengapa", "kenapa", "kapan", "dimana",
        "siapa", "berapa", "tolong", "mohon", "terima", "kasih", "halo",
        "hai", "selamat", "pagi", "siang", "sore", "malam", "ingin", "mau",
        "punya", "ada", "juga", "saja", "hanya", "sangat", "sekali", "karena",
        "tetapi", "tapi", "namun", "atau", "jika", "kalau", "supaya", "agar",
        "sedang", "telah", "pernah", "masih", "lebih", "kurang", "paling",
        "sebuah", "para", "nya", "kah", "lah", "sih", "dong", "ya", "bisa",
        "bermanfaat", "membantu", "membuat", "menjadi", "menggunakan",
        "bekerja", "belajar", "makanan", "minuman", "uang", "waktu", "hari",
    ),
    "en": (
        "the", "and", "is", "are", "was", "were", "of", "to", "in", "for",
        "with", "on", "at", "by", "from", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "it", "my", "your", "our",
        "their", "what", "how", "why", "when", "where", "who", "which",
        "do", "does", "did", "can", "could", "would", "should", "will",
        "have", "has", "had", "not", "no", "yes", "please", "thanks",
        "thank", "hello", "hi", "hey", "good", "morning", "evening",
        "want", "need", "like", "love", "help", "about", "there", "here",
        "because", "but", "or", "if", "while", "being", "been", "very",
        "just", "only", "also", "some", "any", "more", "most", "much",
        "many", "would", "should", "could", "might", "must", "am",
    ),
    "ms": (
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
        "tidak", "saya", "awak", "anda", "kita", "kami", "mereka", "ialah",
        "adalah", "akan", "sudah", "belum", "boleh", "apa", "bagaimana",
        "mengapa", "kenapa", "bila", "dimana", "siapa", "berapa", "tolong",
        "minta", "terima", "kasih", "helo", "apa khabar", "selamat",
        "pagi", "petang", "malam", "mahu", "hendak", "punya", "ada",
        "juga", "sahaja", "hanya", "sangat", "kerana", "tetapi", "atau",
        "jikalau", "kalau", "supaya", "agar", "sedang", "telah", "pernah",
        "masih", "lebih", "kurang", "paling", "lah", "kah", "pun", "tak",
        "nak", "pergi", "makan", "minum", "wang", "masa", "hari", "khabar",
    ),
    "es": (
        "el", "la", "los", "las", "un", "una", "de", "del", "que", "y", "en",
        "por", "para", "con", "sin", "es", "son", "era", "fue", "ser", "estar",
        "yo", "tú", "usted", "nosotros", "ellos", "mi", "tu", "su", "nuestro",
        "qué", "cómo", "por qué", "cuándo", "dónde", "quién", "cuál", "cuánto",
        "hola", "buenos", "buenas", "días", "gracias", "por favor", "quiero",
        "necesito", "gusta", "ayuda", "pero", "porque", "también", "muy",
        "más", "menos", "mucho", "poco", "bien", "mal", "sí", "no", "está",
        "estoy", "tiene", "tengo", "hay", "como", "sobre", "entre", "hasta",
    ),
    "de": (
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "und", "oder", "aber", "ist", "sind", "war",
        "waren", "bin", "bist", "sein", "haben", "hat", "hatte", "ich",
        "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "unser",
        "was", "wie", "warum", "wann", "wo", "wer", "welche", "hallo",
        "guten", "morgen", "tag", "abend", "danke", "bitte", "möchte",
        "brauche", "mag", "helfen", "hilfe", "nicht", "kein", "keine",
        "auch", "sehr", "mehr", "weniger", "viel", "mit", "für", "von",
        "bei", "nach", "über", "unter", "zwischen", "kann", "können",
    ),
    "fr": (
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou",
        "mais", "est", "sont", "était", "être", "avoir", "ai", "as", "a",
        "avons", "avez", "ont", "je", "tu", "il", "elle", "nous", "vous",
        "ils", "elles", "mon", "ton", "son", "notre", "votre", "leur",
        "que", "qui", "quoi", "comment", "pourquoi", "quand", "où",
        "quel", "quelle", "bonjour", "salut", "merci", "s'il vous plaît",
        "veux", "voudrais", "besoin", "aime", "aider", "pas", "non", "oui",
        "aussi", "très", "plus", "moins", "beaucoup", "peu", "bien",
        "avec", "pour", "par", "sur", "sous", "entre", "dans", "chez",
    ),
    "ja": (),
    "ko": (),
}

# Japanese / Korean rely on functional particles and endings.
SCRIPT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ja", re.compile(r"[\u3040-\u309f]")),          # Hiragana
    ("ja", re.compile(r"[\u30a0-\u30ff]")),          # Katakana
    ("ko", re.compile(r"[\uac00-\ud7af]")),          # Hangul syllables
    ("ko", re.compile(r"[\u1100-\u11ff]")),          # Hangul jamo
    ("ja", re.compile(r"[\u4e00-\u9fff]")),          # Han
]

JA_MARKERS = ("は", "が", "を", "に", "で", "と", "も", "の", "です", "ます",
              "か", "ね", "よ", "これ", "それ", "あれ", "私", "あなた",
              "こんにちは", "ありがとう", "ください", "お願い", "何", "どう")
KO_MARKERS = ("은", "는", "이", "가", "을", "를", "에", "에서", "와", "과",
              "도", "의", "입니까", "입니다", "해요", "합니다", "안녕하세요",
              "감사합니다", "주세요", "무엇", "어떻게", "왜", "언제", "어디")

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ']+|[\u3040-\u30ff\u4e00-\u9fff]+|[\uac00-\ud7af]+")


@dataclass
class LanguagePrediction:
    language: str
    confidence: float
    method: str
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "language": self.language,
            "name": SUPPORTED_LANGUAGES.get(self.language, {}).get("name", "Unknown"),
            "native_name": SUPPORTED_LANGUAGES.get(self.language, {}).get("native", ""),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "scores": {k: round(v, 4) for k, v in sorted(
                self.scores.items(), key=lambda kv: kv[1], reverse=True)[:8]},
            "supported": self.language in SUPPORTED_LANGUAGES,
        }


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _script_scores(text: str) -> Dict[str, float]:
    total = max(len(text.strip()), 1)
    scores: Dict[str, float] = {}
    for lang, pattern in SCRIPT_PATTERNS:
        matches = len(pattern.findall(text))
        if matches:
            scores[lang] = scores.get(lang, 0.0) + matches / total
    return scores


def _lexical_scores(text: str) -> Dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    joined = " ".join(tokens)
    scores: Dict[str, float] = {}
    for lang, markers in LANGUAGE_MARKERS.items():
        if not markers:
            continue
        hits = 0.0
        for marker in markers:
            if " " in marker:
                if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", joined):
                    hits += 1.5
            elif marker in tokens or marker == "nya" and any(t.endswith("nya") for t in tokens):
                hits += 1.0
            elif marker == "kah" and any(t.endswith("kah") for t in tokens):
                hits += 0.6
        # Frequency weighting: matched marker types / marker types in the
        # language table, normalised by sentence length so long inputs do not
        # overpower short ones.
        scores[lang] = hits / (1.0 + 0.35 * len(tokens))
    # Particle markers for Japanese/Korean.
    ja_hits = sum(1 for m in JA_MARKERS if m in text)
    ko_hits = sum(1 for m in KO_MARKERS if m in text)
    if ja_hits:
        scores["ja"] = scores.get("ja", 0.0) + ja_hits / (1.0 + 0.2 * len(tokens))
    if ko_hits:
        scores["ko"] = scores.get("ko", 0.0) + ko_hits / (1.0 + 0.2 * len(tokens))
    return scores


def _statistical_scores(text: str) -> Optional[Dict[str, float]]:
    try:  # pragma: no cover - optional dependency
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 7
        results = detect_langs(text)
        return {r.lang: float(r.prob) for r in results if r.lang in SUPPORTED_LANGUAGES}
    except Exception:
        return None


def detect_language(text: str, threshold: float = 0.55) -> LanguagePrediction:
    """Detect the language of ``text``.

    Returns a :class:`LanguagePrediction`; low-confidence inputs fall back to
    the configured default language (Indonesian).
    """
    from chatgangs.config import get_settings

    default = get_settings().default_language
    text = (text or "").strip()
    if not text:
        return LanguagePrediction(default, 0.0, "empty", {})

    script = _script_scores(text)
    lexical = _lexical_scores(text)
    # The statistical detector costs ~40 ms; only consult it when the fast
    # lexical/script signal is not already decisive (or the text is long).
    best_lexical = max(lexical.values(), default=0.0)
    statistical = None
    if len(text) >= 12 and (best_lexical < 0.45 or len(text) > 160):
        statistical = _statistical_scores(text)

    combined: Dict[str, float] = {}
    for lang, value in lexical.items():
        combined[lang] = combined.get(lang, 0.0) + value * 1.0
    if statistical:
        for lang, value in statistical.items():
            combined[lang] = combined.get(lang, 0.0) + value * 0.9
    for lang, value in script.items():
        combined[lang] = combined.get(lang, 0.0) + value * 2.5

    if not combined:
        return LanguagePrediction(default, 0.25, "fallback", {})

    best = max(combined, key=combined.__getitem__)
    total = sum(combined.values()) or 1.0
    confidence = combined[best] / total

    # Indonesian vs Malay disambiguation: Malay-specific markers (khabar,
    # awak, hendak, wang, sahaja, kerana, ...) must appear explicitly.
    if best in ("id", "ms"):
        ms_specific = {"awak", "khabar", "hendak", "wang", "sahaja", "kerana",
                       "jikalau", "pun", "nak", "pergi", "petang", "helo",
                       "mahu", "ialah"}
        tokens = set(_tokenize(text))
        if tokens & ms_specific and len(tokens & ms_specific) >= 2:
            best = "ms"
        else:
            best = "id"

    if confidence < threshold:
        # Short/greeting inputs: trust the lexical winner when it is the
        # priority language, otherwise fall back.
        if best in PRIORITY_LANGUAGES and confidence > 0.30:
            return LanguagePrediction(best, confidence, "lexical-low-confidence", combined)
        return LanguagePrediction(default, confidence, "fallback-low-confidence", combined)

    method = "hybrid" if statistical else "lexical"
    if script and script.get(best, 0) > 0.15:
        method = "script"
    return LanguagePrediction(best, confidence, method, combined)


def is_supported(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES


def language_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, {}).get("name", code)
