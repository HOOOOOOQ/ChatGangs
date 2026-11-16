"""Optional spaCy integration.

spaCy is treated as an *enhancement*: when the model is installed ChatGangs
gains statistical POS tagging, dependency parsing and NER for English. When it
is absent the pipeline automatically falls back to the built-in rule engine,
and :meth:`SpacyPipeline.info` reports ``"available": false`` so the API never
overstates its capabilities.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from chatgangs.config import get_settings
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)


class SpacyPipeline:
    """Lazily loads and caches a spaCy model."""

    def __init__(self) -> None:
        self._nlp = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self._error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._nlp is not None

    def load(self) -> Any:
        if self._nlp is not None or self._load_attempted:
            return self._nlp
        with self._lock:
            if self._nlp is not None or self._load_attempted:
                return self._nlp
            self._load_attempted = True
            settings = get_settings()
            if not settings.enable_spacy:
                self._error = "disabled by configuration (CHATGANGS_ENABLE_SPACY=false)"
                logger.info("spaCy disabled by configuration; rule-based NLP is active")
                return None
            try:
                import spacy

                # A multilingual model id can be supplied instead of the
                # English one (e.g. xx_ent_wiki_sm) for non-English NER.
                self._nlp = spacy.load(settings.spacy_model_id or settings.spacy_model_en)
                self._nlp.max_length = max(self._nlp.max_length, 1_000_000)
                logger.info("spaCy model '%s' loaded", settings.spacy_model_id or settings.spacy_model_en)
            except Exception as exc:  # model not installed
                self._error = (
                    f"spaCy model '{settings.spacy_model_id or settings.spacy_model_en}' unavailable: "
                    f"{type(exc).__name__}. Rule-based NLP is active."
                )
                logger.warning(self._error)
                self._nlp = None
            return self._nlp

    def analyze(self, text: str) -> Any:
        nlp = self.load()
        if nlp is None:
            return None
        try:
            return nlp(text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("spaCy analysis failed: %s", exc)
            return None

    def info(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            "available": self.available,
            "model": (settings.spacy_model_id or settings.spacy_model_en) if self.available else None,
            "requested_model": settings.spacy_model_id or settings.spacy_model_en,
            "enable_spacy": settings.enable_spacy,
            "error": self._error,
            "capabilities": (
                ["pos", "lemma", "dependency", "ner"]
                if self.available
                else ["rule-based pos", "rule-based lemma", "dictionary ner"]
            ),
        }


_pipeline: Optional[SpacyPipeline] = None


def get_spacy_pipeline() -> SpacyPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SpacyPipeline()
    return _pipeline
