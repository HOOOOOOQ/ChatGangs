"""Named Entity Recognition (NER) and structured entity extraction.

Two complementary extractors are combined:

1. **Rule/lexical NER** — deterministic pattern matching that works for every
   supported language without downloading models:
   * e-mail, URL, phone, money/currency, percentage, number/quantity, date,
     time, organisation suffixes (Inc., Ltd., PT, CV, Co., GmbH) and known
     place/org gazetteers.

2. **Statistical NER** — used automatically when the spaCy English model is
   installed (PERSON, ORG, GPE, LOC, PRODUCT, DATE ...). Reported separately so
   callers know the provenance of every entity.

Entity extraction also powers intent slot filling (see ``intent.py``) and the
``POST /api/nlp/entities`` endpoint.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chatgangs.core.spacy_pipeline import get_spacy_pipeline
from chatgangs.core.text_utils import Entity, clean_text

# ---------------------------------------------------------------------------
# Gazetteers (small, high-precision; expandable via data/lexicons/gazetteer.txt)
# ---------------------------------------------------------------------------
COUNTRIES = {
    "indonesia", "malaysia", "singapore", "singapura", "japan", "jepang",
    "korea", "south korea", "korea selatan", "china", "tiongkok", "america",
    "amerika", "united states", "usa", "as", "england", "inggris", "germany",
    "jerman", "france", "prancis", "spain", "spanyol", "australia", "india",
    "thailand", "thailand", "vietnam", "philippines", "filipina", "netherlands",
    "belanda", "canada", "kanada", "brazil", "brasil", "mexico", "meksiko",
    "turkey", "turki", "saudi arabia", "arab saudi", "egypt", "mesir",
}

CITIES_ID = {
    "jakarta", "bandung", "surabaya", "medan", "semarang", "makassar",
    "palembang", "tangerang", "depok", "bekasi", "bogor", "yogyakarta",
    "jogja", "malang", "denpasar", "bali", "batam", "pekanbaru", "padang",
    "sukabumi", "cimahi", "tasikmalaya", "cirebon", "garut", "purwakarta",
    "karawang", "subang", "sumedang", "indramayu", "banjarmasin",
    "pontianak", "samarinda", "manado", "jayapura", "ambon", "kupang",
    "mataram", "surakarta", "solo", "salatiga", "magelang", "kediri",
    "jember", "madiun", "probolinggo", "pasuruan", "sidoarjo", "gresik",
}

CITIES_WORLD = {
    "london", "paris", "tokyo", "seoul", "beijing", "shanghai", "singapore",
    "kuala lumpur", "bangkok", "hanoi", "manila", "sydney", "melbourne",
    "new york", "los angeles", "san francisco", "chicago", "toronto",
    "berlin", "munich", "madrid", "barcelona", "rome", "milan", "amsterdam",
    "brussels", "vienna", "zurich", "moscow", "dubai", "doha", "cairo",
    "mumbai", "delhi", "bangkok",
}

ORGANIZATIONS = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook", "netflix",
    "tesla", "nvidia", "openai", "anthropic", "github", "gitlab", "spotify",
    "tokopedia", "bukalapak", "shopee", "lazada", "gojek", "grab", "traveloka",
    "tiket.com", "blibli", "jdid", "tokopedia", "bca", "bri", "bni", "mandiri",
    "telkom", "telkomsel", "indosat", "xl", "smartfren", "pertamina", "garuda",
    "lion air", "airasia", "samsung", "xiaomi", "huawei", "oppo", "vivo",
    "nasa", "spacex", "ibm", "intel", "amd", "oracle", "salesforce", "adobe",
    "uber", "airbnb", "twitter", "tiktok", "bytedance", "baidu", "alibaba",
}

PRODUCTS = {
    "iphone", "ipad", "macbook", "airpods", "galaxy", "pixel", "windows",
    "android", "ios", "linux", "ubuntu", "docker", "kubernetes", "python",
    "javascript", "fastapi", "django", "flask", "react", "vue", "angular",
    "chatgpt", "gemini", "claude", "llama", "bert", "gpt-4", "gpt-3",
    "spacy", "nltk", "tensorflow", "pytorch", "scikit-learn", "postgresql",
    "mysql", "sqlite", "mongodb", "redis", "excel", "word", "powerpoint",
}

FIRST_NAMES = {
    "budi", "andi", "siti", "dewi", "agus", "rina", "dian", "eko", "putri",
    "rahma", "fajar", "nadia", "bayu", "intan", "yoga", "sari", "wawan",
    "tetty", "hendra", "agus", "joko", "susi", "rani", "tono", "made",
    "john", "mary", "james", "robert", "michael", "william", "david",
    "richard", "joseph", "thomas", "charles", "christopher", "daniel",
    "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew",
    "joshua", "kenneth", "kevin", "brian", "george", "edward", "ronald",
    "timothy", "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas",
    "eric", "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "emily", "sarah", "jessica", "ashley", "brittany", "amanda", "jennifer",
    "lisa", "michelle", "kimberly", "melissa", "deborah", "stephanie",
    "rebecca", "laura", "helen", "sandra", "donna", "carol", "ruth",
    "sharon", "anne", "maria", "linda", "patricia", "barbara", "elizabeth",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+|\bwww\.[\w.-]+", re.I)
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+62|62|0)\s?8[\d\s\-().]{7,14}\d|"
    r"(?<![\w.])\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?![\w])"
)
MONEY_RE = re.compile(
    r"(?:Rp\.?|IDR|USD|\$|€|EUR|£|GBP|¥|JPY|MYR|RM|SGD)\s?"
    r"\d[\d.,]*\s?(?:juta|ribu|miliar|milyar|triliun|rb|jt|k|m|billion|million|thousand|k)?",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?\s?(?:%|persen|percent)", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?\s?"
    r"(?:unit|buah|orang|kali|tahun|bulan|minggu|hari|jam|menit|detik|"
    r"items?|people|times|years?|months?|weeks?|days?|hours?|minutes?|seconds?)?",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:,?\s+\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|"
    r"agustus|september|oktober|november|desember|january|february|march|"
    r"april|may|june|july|august|september|october|november|december)?"
    r"(?:\s+\d{4})?)?|"
    r"\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|agustus|"
    r"september|oktober|november|desember|january|february|march|april|may|"
    r"june|july|august|september|october|november|december)"
    r"(?:\s+\d{4})?|"
    r"(?:hari ini|besok|kemarin|minggu depan|bulan depan|tahun depan|"
    r"today|tomorrow|yesterday|next week|next month|next year))",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"\b(?:pukul|jam|at|around|sekitar)?\s?\d{1,2}[:.]\d{2}(?:\s?(?:am|pm|wib|wita|wit))?\b|"
    r"\b(?:pukul|jam)\s+\d{1,2}(?:\s?(?:pagi|siang|sore|malam))?\b",
    re.IGNORECASE,
)
ORG_SUFFIX_RE = re.compile(
    r"\b(?:PT|CV|UD|PD|Inc|Incorporated|Corp|Corporation|Ltd|Limited|LLC|"
    r"GmbH|AG|S\.A|SAS|SARL|Co|Company|Group|Holdings)\b\.?",
    re.IGNORECASE,
)
ORG_NAME_RE = re.compile(
    r"\b(?:(?:PT|CV|UD|PD)\.?\s+)([A-Z][\w&'-]*(?:[ ]+[A-Z][\w&'-]*){0,4})"
    r"|\b([A-Z][\w&'-]*(?:[ ]+[A-Z][\w&'-]*){0,4})[ ]+"
    r"(Inc|Corporation|Corp|Ltd|Limited|LLC|GmbH|AG|Group|Holdings|Company)\b\.?"
)
PERSON_TITLE_RE = re.compile(
    r"\b(?:Bapak|Ibu|Pak|Bu|Mas|Mbak|Kak|Sdr|Sdri|Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam)"
    r"\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)


class EntityRecognizer:
    """Extracts entities using rules, gazetteers and (optionally) spaCy."""

    def __init__(self) -> None:
        self.spacy = get_spacy_pipeline()

    # ------------------------------------------------------------------ public
    def extract(
        self,
        text: str,
        language: str = "id",
        *,
        use_spacy: bool = True,
        include_default_labels: Sequence[str] | None = None,
    ) -> List[Entity]:
        text = clean_text(text or "")
        entities: List[Entity] = []
        entities.extend(self._regex_entities(text))
        entities.extend(self._gazetteer_entities(text))
        if use_spacy:
            entities.extend(self._spacy_entities(text))
        entities = _dedupe(entities)
        if include_default_labels:
            allowed = {label.upper() for label in include_default_labels}
            entities = [e for e in entities if e.label.upper() in allowed]
        return sorted(entities, key=lambda e: (e.start, -e.end))

    def summarize(self, entities: List[Entity]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for entity in entities:
            grouped.setdefault(entity.label, [])
            if entity.text not in grouped[entity.label]:
                grouped[entity.label].append(entity.text)
        return grouped

    # ------------------------------------------------------------------- rules
    def _regex_entities(self, text: str) -> List[Entity]:
        out: List[Entity] = []
        specs: List[Tuple[str, re.Pattern, float]] = [
            ("EMAIL", EMAIL_RE, 0.99),
            ("URL", URL_RE, 0.99),
            ("PHONE", PHONE_RE, 0.9),
            ("MONEY", MONEY_RE, 0.92),
            ("PERCENT", PERCENT_RE, 0.9),
            ("DATE", DATE_RE, 0.8),
            ("TIME", TIME_RE, 0.78),
        ]
        for label, pattern, confidence in specs:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if not value:
                    continue
                out.append(
                    Entity(
                        text=value,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        source="regex",
                    )
                )
        # Numbers are only reported when they carry a unit or currency to
        # avoid flooding results with every digit in the input.
        for match in NUMBER_RE.finditer(text):
            value = match.group(0).strip()
            if re.search(r"(?:unit|buah|orang|kali|tahun|bulan|minggu|hari|jam|"
                         r"menit|detik|item|people|times|year|month|week|day|"
                         r"hour|minute|second)", value, re.IGNORECASE):
                out.append(
                    Entity(value, "QUANTITY", match.start(), match.end(), 0.7, "regex")
                )
        return out

    def _gazetteer_entities(self, text: str) -> List[Entity]:
        out: List[Entity] = []
        lowered = text.lower()
        gazetteers = (
            ("ORGANIZATION", ORGANIZATIONS, 0.85),
            ("PRODUCT", PRODUCTS, 0.8),
            ("GPE", COUNTRIES | CITIES_ID | CITIES_WORLD, 0.82),
            ("LOCATION", CITIES_ID | CITIES_WORLD, 0.72),
        )
        for label, gazetteer, confidence in gazetteers:
            for name in gazetteer:
                for match in re.finditer(rf"(?<![\w]){re.escape(name)}(?![\w])", lowered):
                    out.append(
                        Entity(
                            text=text[match.start(): match.end()],
                            label=label,
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                            source="gazetteer",
                        )
                    )
        for match in ORG_NAME_RE.finditer(text):
            value = match.group(0).strip()
            out.append(Entity(value, "ORGANIZATION", match.start(), match.end(), 0.88, "pattern"))
        for match in PERSON_TITLE_RE.finditer(text):
            value = match.group(0).strip()
            out.append(Entity(value, "PERSON", match.start(), match.end(), 0.85, "pattern"))

        # Capitalised unknown tokens → heuristic PERSON when mid-sentence and
        # surrounded by lowercase words (e.g. "...dengan Budi kemarin").
        for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,15})\b", text):
            word = match.group(1).lower()
            if word in FIRST_NAMES:
                out.append(
                    Entity(match.group(1), "PERSON", match.start(), match.end(), 0.7, "lexicon")
                )
        return out

    # ------------------------------------------------------------------ spaCy
    def _spacy_entities(self, text: str) -> List[Entity]:
        doc = self.spacy.analyze(text)
        if doc is None:
            return []
        out: List[Entity] = []
        for ent in doc.ents:
            label = _SPACY_LABEL_MAP.get(ent.label_, ent.label_)
            out.append(
                Entity(
                    text=ent.text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,
                    source="spacy",
                )
            )
        return out


_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "GPE",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "DATE": "DATE",
    "TIME": "TIME",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
    "CARDINAL": "CARDINAL",
    "ORDINAL": "CARDINAL",
    "QUANTITY": "QUANTITY",
    "NORP": "NORP",
    "LANGUAGE": "LANGUAGE",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
}


def _dedupe(entities: Iterable[Entity]) -> List[Entity]:
    """Drop overlapping duplicates, preferring the higher-confidence entity."""
    result: List[Entity] = []
    for entity in sorted(entities, key=lambda e: (e.start, -(e.end - e.start), -e.confidence)):
        duplicate = False
        for existing in result:
            if entity.start < existing.end and existing.start < entity.end:
                if (
                    entity.text.lower() == existing.text.lower()
                    or existing.confidence >= entity.confidence
                ):
                    duplicate = True
                    break
        if not duplicate:
            result.append(entity)
    return result


_recognizer: Optional[EntityRecognizer] = None


def get_entity_recognizer() -> EntityRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = EntityRecognizer()
    return _recognizer
