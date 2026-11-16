"""Sentiment analysis module.

The engine is **hybrid** and fully deterministic by default:

1. Lexicon scoring with linguistic modifiers — negation windows, amplifiers,
   diminishers, contrastive conjunctions (``tapi``/``but`` shift weight to the
   clause that follows), emoji sentiment and exclamation emphasis.
2. Sentence-level aggregation so a mixed review gets both a document label and
   per-sentence labels plus the contributing evidence terms.

If a trained scikit-learn model is present in ``models/sentiment`` (produced by
``python -m scripts.train``), its probability is blended with the lexicon
score — the blend weights and both raw sources are always reported, so the
output stays auditable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chatgangs.config import get_settings
from chatgangs.core.normalizer import get_dictionaries, get_normalizer, strip_diacritics
from chatgangs.core.tokenizer import get_tokenizer
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)

LABELS = ("negative", "neutral", "positive")

CONTRAST_WORDS = {
    "id": {"tapi", "tetapi", "namun", "meski", "meskipun", "walaupun", "walau", "kecuali"},
    "ms": {"tapi", "tetapi", "namun", "walaupun"},
    "en": {"but", "however", "although", "though", "yet", "except"},
    "es": {"pero", "sin embargo", "aunque"},
    "de": {"aber", "jedoch", "obwohl"},
    "fr": {"mais", "cependant", "bien que"},
    "ja": {"しかし", "でも", "けれど"},
    "ko": {"하지만", "그러나"},
}

NEGATION_SCOPE = 3  # tokens


@dataclass
class SentimentResult:
    label: str
    score: float                     # aggregate polarity in [-1, 1]
    confidence: float                # 0..1
    language: str
    sentences: List[dict] = field(default_factory=list)
    evidence: List[dict] = field(default_factory=list)
    lexicon_score: float = 0.0
    model_score: Optional[float] = None
    model_used: bool = False
    tokens_analyzed: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "language": self.language,
            "tokens_analyzed": self.tokens_analyzed,
            "lexicon_score": round(self.lexicon_score, 4),
            "model_score": None if self.model_score is None else round(self.model_score, 4),
            "model_used": self.model_used,
            "evidence": self.evidence,
            "sentences": self.sentences,
        }


def _label_from_score(score: float, threshold: float = 0.18) -> str:
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


def _confidence(score: float, support: float, model_conf: float = 0.0) -> float:
    """Confidence grows with |score| and with the number of evidence terms."""
    base = 1.0 - math.exp(-abs(score) * 1.9)
    support_factor = 1.0 - math.exp(-support * 0.55)
    return max(0.05, min(0.99, 0.35 + 0.45 * base + 0.2 * support_factor + 0.25 * model_conf))


class SentimentAnalyzer:
    """Lexicon + optional statistical sentiment analyzer."""

    def __init__(self) -> None:
        self.store = get_dictionaries()
        self.normalizer = get_normalizer()
        self.tokenizer = get_tokenizer()
        self._model = None
        self._vectorizer = None
        self._model_meta: Dict = {}
        self._load_model()

    # ------------------------------------------------------------------ model
    def _load_model(self) -> None:
        settings = get_settings()
        model_dir = Path(settings.models_dir) / "sentiment"
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            return
        try:
            import joblib

            bundle = joblib.load(model_path)
            self._model = bundle["model"]
            self._vectorizer = bundle["vectorizer"]
            self._model_meta = bundle.get("metadata", {})
            logger.info("Loaded sentiment model %s", self._model_meta.get("version", "unknown"))
        except Exception as exc:  # pragma: no cover - corrupted artifact
            logger.warning("Could not load sentiment model: %s", exc)
            self._model = None

    @property
    def model_available(self) -> bool:
        return self._model is not None

    def reload(self) -> None:
        self._model = None
        self._vectorizer = None
        self._model_meta = {}
        self._load_model()

    # ------------------------------------------------------------- main entry
    def analyze(
        self,
        text: str,
        language: str = "id",
        *,
        explain: bool = True,
    ) -> SentimentResult:
        text = (text or "").strip()
        if language not in self.store.sentiment:
            language = "en" if language not in ("id", "ms") else "id"
        if language == "ms":
            language = "id"
        result = self._lexicon_analyze(text, language, explain=explain)

        model_score, model_conf = self._model_score(text)
        # For very short inputs without any lexicon evidence the statistical
        # model has little to work with ("halo" is not a sentiment), so we
        # report neutral instead of a spurious polarity.
        token_count = len(result.sentences[0]["text"].split()) if result.sentences else 0
        if model_score is not None and not result.evidence and token_count < 3:
            model_score = None
        if model_score is not None:
            result.model_score = model_score
            result.model_used = True
            # Blend: historical lexicon weight 0.35, trained model 0.65.
            blended = 0.35 * result.lexicon_score + 0.65 * model_score
            result.score = max(-1.0, min(1.0, blended))
            result.label = _label_from_score(result.score)
            support = float(len(result.evidence)) if result.evidence else 1.0
            result.confidence = _confidence(result.score, support, model_conf)
        return result

    # ----------------------------------------------------------- lexicon path
    def _lexicon_analyze(self, text: str, language: str, explain: bool = True) -> SentimentResult:
        lexicon = self.store.sentiment[language]
        positives = lexicon.get("positive", {})
        negatives = lexicon.get("negative", {})
        negations = set(lexicon.get("negation", {})) | {"tidak", "bukan", "not", "no", "tanpa"}
        amplifiers = lexicon.get("amplifiers", {})
        diminishers = lexicon.get("diminishers", {})
        contrast = CONTRAST_WORDS.get(language, set())

        normalized = self.normalizer.normalize(
            text, language=language, keep=("emoji",), redact_pii=False
        )
        tokens = [t for t in self.tokenizer.tokenize(normalized) if not t.is_punctuation]

        # Build lookup of multi-word lexicon entries.
        multiword: Dict[str, float] = {}
        for source, sign in ((positives, 1.0), (negatives, -1.0)):
            for entry, weight in source.items():
                if " " in entry:
                    multiword[entry] = weight if sign > 0 else weight

        sentence_units = self._split_clauses(normalized, contrast)
        total_score = 0.0
        total_weight = 0.0
        evidence: List[dict] = []
        sentence_reports: List[dict] = []
        tokens_analyzed = 0
        global_offset = 0

        for clause_text, clause_weight, in_contrast in sentence_units:
            clause_tokens = [
                t for t in self.tokenizer.tokenize(clause_text) if not t.is_punctuation
            ]
            tokens_analyzed += len(clause_tokens)
            raw_score, weight, clause_evidence = self._score_tokens(
                clause_tokens, positives, negatives, negations, amplifiers,
                diminishers, multiword,
            )
            if clause_tokens:
                emphasis = 1.0 + 0.05 * min(clause_text.count("!"), 4)
                raw_score *= emphasis
            # Squash each clause into [-1, 1] so one strongly amplified word
            # cannot dominate the document-level average.
            clause_polarity = raw_score / max(1.0, weight)
            clause_polarity = max(-1.0, min(1.0, clause_polarity))
            score = clause_polarity
            total_score += clause_polarity * clause_weight
            total_weight += clause_weight
            for item in clause_evidence:
                item["sentence_index"] = len(sentence_reports)
                item["offset"] = global_offset + item.get("offset", 0)
            evidence.extend(clause_evidence)
            sentence_reports.append(
                {
                    "text": clause_text.strip(),
                    "label": _label_from_score(score),
                    "score": round(score, 4),
                    "contrast_clause": in_contrast,
                }
            )
            global_offset += len(clause_text) + 1

        polarity = total_score if total_weight > 0 else 0.0
        # Squash to [-1, 1] while keeping direction; empty text → neutral.
        polarity = max(-1.0, min(1.0, polarity))
        label = _label_from_score(polarity) if text else "neutral"
        confidence = _confidence(polarity, len(evidence)) if text else 0.0
        if not text:
            confidence = 0.0

        return SentimentResult(
            label=label,
            score=polarity,
            confidence=confidence,
            language=language,
            sentences=sentence_reports if explain else [],
            evidence=evidence if explain else [],
            lexicon_score=polarity,
            tokens_analyzed=tokens_analyzed,
        )

    def _split_clauses(self, text: str, contrast: set) -> List[Tuple[str, float, bool]]:
        """Split into sentence-level clauses, then on contrastive conjunctions.

        Each clause carries a weight: sentences are weighted equally, and a
        contrastive clause gets more weight because readers interpret
        "good, but slow" as predominantly negative.
        """
        if not text.strip():
            return []
        from chatgangs.core.text_utils import split_sentences

        sentences = [unit for unit in split_sentences(text) if unit.strip()] or [text]
        pattern = None
        if contrast:
            pattern = re.compile(
                r"\b(" + "|".join(
                    sorted(map(re.escape, contrast), key=len, reverse=True)
                ) + r")\b",
                re.IGNORECASE,
            )

        clauses: List[Tuple[str, bool]] = []
        for sentence in sentences:
            if pattern is None:
                clauses.append((sentence, False))
                continue
            parts = pattern.split(sentence)
            buffer = ""
            first_in_sentence = True
            for index, part in enumerate(parts):
                if index % 2 == 1:  # the separator itself
                    if buffer.strip():
                        clauses.append((buffer, not first_in_sentence))
                        first_in_sentence = False
                    buffer = ""
                else:
                    buffer += part
            if buffer.strip():
                clauses.append((buffer, not first_in_sentence))

        if not clauses:
            return [(text, 1.0, False)]

        weighted: List[Tuple[str, float, bool]] = []
        for position, (clause, in_contrast) in enumerate(clauses):
            weight = 1.0 + (0.6 * position if in_contrast else 0.0)
            weighted.append((clause, weight, in_contrast))
        total_weight = sum(weight for _, weight, _ in weighted) or 1.0
        return [
            (clause, weight / total_weight, flag)
            for clause, weight, flag in weighted
        ]

    def _score_tokens(
        self,
        tokens,
        positives: Dict[str, float],
        negatives: Dict[str, float],
        negations: set,
        amplifiers: Dict[str, float],
        diminishers: Dict[str, float],
        multiword: Dict[str, float],
    ) -> Tuple[float, float, List[dict]]:
        score = 0.0
        weight = 0.0
        evidence: List[dict] = []
        words = [t.normalized for t in tokens]
        lower_text = " ".join(words)

        # Emoji / emoticon sentiment (raw token text is needed for emoji).
        emoji_scores: Dict[int, float] = {}
        for idx, token in enumerate(tokens):
            raw = token.text
            if raw in self.store.emoticons:
                emoji_scores[idx] = self.store.emoticons[raw]
            elif token.normalized in self.store.emoticons:
                emoji_scores[idx] = self.store.emoticons[token.normalized]

        i = 0
        consumed_until = -1
        for i, word in enumerate(words):
            if i <= consumed_until:
                continue
            # Multi-word lexicon entry takes priority.
            matched_entry, matched_weight, span = None, 0.0, 1
            for phrase, phrase_weight in multiword.items():
                p_tokens = phrase.split()
                if words[i: i + len(p_tokens)] == p_tokens:
                    matched_entry, matched_weight, span = phrase, phrase_weight, len(p_tokens)
                    break
            if matched_entry:
                polarity = matched_weight
                source = matched_entry
                consumed_until = i + span - 1
            elif word in positives or word in negatives:
                polarity = positives.get(word, negatives.get(word, 0.0))
                source = word
            else:
                if i in emoji_scores:
                    polarity = emoji_scores[i]
                    source = tokens[i].text
                    boost = 1.0
                    score += polarity * boost
                    weight += abs(polarity)
                    evidence.append({
                        "term": source, "polarity": round(polarity, 3),
                        "kind": "emoji", "negated": False,
                        "offset": tokens[i].start,
                    })
                continue

            negated = False
            boost = 1.0
            window_start = max(0, i - NEGATION_SCOPE)
            for j in range(window_start, i):
                wj = words[j]
                if wj in negations:
                    negated = True
                if wj in amplifiers:
                    boost *= amplifiers[wj]
                elif wj in diminishers:
                    boost *= diminishers[wj]
            # Indonesian/Malay intensifiers follow the word they modify
            # ("bagus banget", "lambat sekali"); inspect the next two tokens.
            for j in range(i + 1, min(i + 3, len(words))):
                wj = words[j]
                if wj in amplifiers:
                    boost *= amplifiers[wj]
                elif wj in diminishers:
                    boost *= diminishers[wj]
                elif wj in negations or wj in positives or wj in negatives:
                    break
            if negated:
                polarity = -polarity * 0.85
            polarity *= boost
            score += polarity
            weight += abs(polarity)
            evidence.append({
                "term": source,
                "polarity": round(polarity, 3),
                "kind": "phrase" if span > 1 else "word",
                "negated": negated,
                "amplified": round(boost, 3),
                "offset": tokens[i].start,
            })
        if not weight:
            weight = 1.0
        return score, weight, evidence

    # ------------------------------------------------------------ model path
    def _model_score(self, text: str) -> Tuple[Optional[float], float]:
        if self._model is None or self._vectorizer is None or not text.strip():
            return None, 0.0
        try:
            features = self._vectorizer.transform([text])
            proba = self._model.predict_proba(features)[0]
            classes = list(self._model.classes_)
            mapping = {label: proba[classes.index(label)] for label in classes}
            score = (
                mapping.get("positive", 0.0)
                - mapping.get("negative", 0.0)
            )
            confidence = max(proba)
            return float(max(-1.0, min(1.0, score))), float(confidence)
        except Exception as exc:  # pragma: no cover
            logger.warning("Sentiment model inference failed: %s", exc)
            return None, 0.0

    def model_info(self) -> dict:
        return {
            "available": self.model_available,
            "path": str(Path(get_settings().models_dir) / "sentiment" / "model.joblib"),
            "metadata": self._model_meta,
            "labels": list(LABELS),
            "lexicon": {
                "positive_terms": len(self.store.sentiment["id"].get("positive", {}))
                + len(self.store.sentiment["en"].get("positive", {})),
                "negative_terms": len(self.store.sentiment["id"].get("negative", {}))
                + len(self.store.sentiment["en"].get("negative", {})),
                "emoji_terms": len(self.store.emoticons),
            },
        }


_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
