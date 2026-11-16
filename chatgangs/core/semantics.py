"""Semantic similarity, text classification and retrieval-based QA.

The representation is a hand-built, dependency-free TF-IDF vectoriser:

* word unigrams + bigrams,
* sublinear term-frequency scaling and smoothed inverse document frequency,
* L2 normalisation so cosine similarity is a plain dot product.

It is deliberately implemented here (rather than imported) so the module can be
audited, cached per corpus and unit-tested deterministically. scikit-learn is
used only where a *trained* model is genuinely needed (intent / sentiment
classifiers trained by ``scripts/train.py``).

Capabilities exposed:

* :meth:`SemanticIndex.similarity` — pairwise semantic similarity.
* :meth:`SemanticIndex.rank` — nearest corpus entries (retrieval).
* :class:`TextClassifier` — zero-shot-ish keyword-centroid classifier for
  arbitrary user-defined labels, plus a real trained TF-IDF + Logistic
  Regression classifier when one is available in ``models/classification``.
* :class:`KnowledgeBase` — QA corpus loader with cached retrieval.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chatgangs.config import get_settings
from chatgangs.core.languages import SUPPORTED_LANGUAGES
from chatgangs.core.normalizer import get_normalizer
from chatgangs.core.stopwords import get_stopword_remover
from chatgangs.core.text_utils import LRUCache, _cache_capacity
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)


_STEM_CACHE = LRUCache[str](capacity=_cache_capacity(16384))


def _stem_for_index(word: str, language: str) -> str:
    key = f"{language}:{word}"
    cached = _STEM_CACHE.get(key)
    if cached is not None:
        return cached
    from chatgangs.core.stemmer import get_stemmer

    stemmed = get_stemmer().stem(word, language)
    if len(stemmed) < 3:
        stemmed = word
    _STEM_CACHE.put(key, stemmed)
    return stemmed


def tokenize_for_index(text: str, language: str = "id") -> List[str]:
    """Tokenise + normalise + stopword-remove + stem for semantic indexing.

    Stemming makes morphologically related forms match ("harganya" ~ "harga",
    "menggunakan" ~ "guna") which is essential for Indonesian retrieval.
    """
    normalizer = get_normalizer()
    remover = get_stopword_remover()
    normalized = normalizer.normalize(text, language=language, keep=())
    words = _WORD_RE.findall(normalized.lower())
    kept = remover.remove(words, language)
    stem_lang = language if language in ("id", "ms", "en", "es", "de", "fr") else "en"
    return [_stem_for_index(w, stem_lang) for w in kept]


@dataclass
class SimilarityResult:
    corpus: List[str]
    matrix: List[List[float]]
    most_similar: Tuple[int, int]
    score: float

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "matrix": [[round(v, 4) for v in row] for row in self.matrix],
            "most_similar": {
                "i": self.most_similar[0],
                "j": self.most_similar[1],
                "text_i": self.corpus[self.most_similar[0]],
                "text_j": self.corpus[self.most_similar[1]],
            },
            "score": round(self.score, 4),
        }


class TFIDFVectorizer:
    """Minimal TF-IDF vectoriser (unigrams + bigrams)."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
        self.fitted = False

    def _ngrams(self, tokens: Sequence[str]) -> Iterable[str]:
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            if n == 1:
                yield from tokens
            else:
                for i in range(len(tokens) - n + 1):
                    yield " ".join(tokens[i: i + n])

    def fit(self, documents: Sequence[Sequence[str]]) -> "TFIDFVectorizer":
        df: Counter = Counter()
        for tokens in documents:
            df.update(set(self._ngrams(tokens)))
        terms = [term for term, count in df.items() if count >= self.min_df]
        terms.sort()
        self.vocabulary = {term: idx for idx, term in enumerate(terms)}
        total = max(len(documents), 1)
        self.idf = [
            math.log((1 + total) / (1 + df[term])) + 1.0 for term in terms
        ]
        self.fitted = True
        return self

    def transform(self, documents: Sequence[Sequence[str]]) -> List[List[float]]:
        if not self.fitted:
            raise RuntimeError("TFIDFVectorizer must be fitted before transform()")
        out: List[List[float]] = []
        for tokens in documents:
            vector = [0.0] * len(self.vocabulary)
            counts = Counter(gram for gram in self._ngrams(tokens) if gram in self.vocabulary)
            for gram, count in counts.items():
                idx = self.vocabulary[gram]
                # sublinear tf
                tf = 1.0 + math.log(count) if count > 1 else 1.0
                vector[idx] = tf * self.idf[idx]
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            out.append(vector)
        return out

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class SemanticIndex:
    """Cached TF-IDF semantic similarity engine."""

    def __init__(self) -> None:
        self._cache = LRUCache[List[Tuple[str, float]]](capacity=_cache_capacity(512))
        self._lock = threading.Lock()

    def similarity(self, texts: Sequence[str], language: str = "id") -> SimilarityResult:
        if not texts:
            raise ValueError("At least one text is required")
        tokenized = [tokenize_for_index(t, language) for t in texts]
        vectorizer = TFIDFVectorizer()
        matrix = vectorizer.fit_transform(tokenized)
        sims = [[cosine(matrix[i], matrix[j]) for j in range(len(matrix))] for i in range(len(matrix))]
        best = (0, 0)
        best_score = -1.0
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if sims[i][j] > best_score:
                    best_score = sims[i][j]
                    best = (i, j)
        if best_score < 0:
            best_score = 1.0 if len(matrix) == 1 else 0.0
        return SimilarityResult(
            corpus=list(texts),
            matrix=sims,
            most_similar=best,
            score=max(best_score, 0.0),
        )

    def rank(
        self,
        query: str,
        candidates: Sequence[str],
        language: str = "id",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        cache_key = f"{language}|{hash(tuple(candidates))}|{top_k}|{query}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        docs = [tokenize_for_index(query, language)] + [
            tokenize_for_index(c, language) for c in candidates
        ]
        vectorizer = TFIDFVectorizer()
        matrix = vectorizer.fit_transform(docs)
        query_vec = matrix[0]
        scores = [(candidates[i], cosine(query_vec, matrix[i + 1])) for i in range(len(candidates))]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        result = scores[:top_k]
        self._cache.put(cache_key, result)
        return result


# --------------------------------------------------------------------------- #
# Knowledge base backed QA
# --------------------------------------------------------------------------- #
@dataclass
class QAMatch:
    entry: dict
    score: float
    matched_variant: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.entry.get("id"),
            "topic": self.entry.get("topic"),
            "language": self.entry.get("language"),
            "question": self.entry.get("question"),
            "score": round(self.score, 4),
            "matched_variant": self.matched_variant,
        }


class KnowledgeBase:
    """Loads ``data/qa/knowledge_base.json`` and retrieves entries by similarity.

    The TF-IDF index is built **once** at load time (not per query), so
    retrieval latency is a few milliseconds even with a large knowledge base.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path(get_settings().data_dir) / "qa" / "knowledge_base.json")
        self.entries: List[dict] = []
        self.version = "0"
        self._documents: List[str] = []
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._matrix: List[List[float]] = []
        self._normalized_variants: List[List[Tuple[int, str]]] = []
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            logger.warning("Knowledge base not found at %s", self.path)
            self.entries = []
            self._documents = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.version = payload.get("version", "0")
        self.entries = payload.get("entries", [])
        self._build_index()
        logger.info(
            "Loaded knowledge base %s with %d entries (%d vocabulary terms)",
            self.version,
            len(self.entries),
            len(self._vectorizer.vocabulary) if self._vectorizer else 0,
        )

    def _build_index(self) -> None:
        self._documents = [
            " | ".join(
                [entry.get("question", "")]
                + entry.get("variants", [])
                + entry.get("keywords", [])
            )
            for entry in self.entries
        ]
        if not self._documents:
            self._vectorizer = None
            self._matrix = []
            self._normalized_variants = []
            return
        tokenized = [tokenize_for_index(doc, "id") for doc in self._documents]
        self._vectorizer = TFIDFVectorizer(ngram_range=(1, 2)).fit(tokenized)
        self._matrix = self._vectorizer.transform(tokenized)
        self._normalized_variants = []
        for entry in self.entries:
            variants = []
            for variant in entry.get("variants", []):
                normalized = self._normalize_variant(variant, entry.get("language", "id"))
                if normalized:
                    variants.append((len(normalized), normalized))
            self._normalized_variants.append(variants)

    @property
    def documents(self) -> List[str]:
        return self._documents

    def retrieve(self, query: str, language: str = "id", top_k: int = 3) -> List[QAMatch]:
        """Hybrid retrieval: TF-IDF semantic ranking + exact variant matching.

        Semantic ranking works on stemmed content words; exact variant matching
        catches short utterances that consist entirely of stopwords such as
        "siapa kamu" or "halo".
        """
        if not self.entries or self._vectorizer is None:
            return []
        with self._lock:
            query_tokens = tokenize_for_index(query, language)
            query_vector = self._vectorizer.transform([query_tokens])[0]
            scores: Dict[int, float] = {}
            for idx, doc_vector in enumerate(self._matrix):
                score = cosine(query_vector, doc_vector)
                if score > 0:
                    scores[idx] = score

        normalized_query = self._normalize_variant(query, language)
        if normalized_query:
            for idx, entry in enumerate(self.entries):
                for _length, candidate in self._normalized_variants[idx]:
                    if candidate == normalized_query:
                        scores[idx] = max(scores.get(idx, 0.0), 1.0)
                        break
                    if len(candidate) >= 4 and (
                        normalized_query.startswith(candidate)
                        or candidate.startswith(normalized_query)
                        or normalized_query == candidate
                    ):
                        coverage = min(len(candidate), len(normalized_query)) / max(
                            len(candidate), len(normalized_query))
                        scores[idx] = max(scores.get(idx, 0.0), 0.9 * coverage)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [QAMatch(entry=self.entries[idx], score=score) for idx, score in ordered]

    @staticmethod
    def _normalize_variant(text: str, language: str) -> str:
        """Lowercase variant with punctuation stripped (stopwords preserved)."""
        normalizer = get_normalizer()
        cleaned = normalizer.normalize(
            text, language=language, keep=(), correct_typos=False, redact_pii=False
        )
        cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def best(self, query: str, language: str = "id", min_score: float = 0.12) -> Optional[QAMatch]:
        matches = self.retrieve(query, language=language, top_k=5)
        if not matches:
            return None
        # Prefer entries written in the user's language when scores are close.
        matches.sort(
            key=lambda m: (m.score + (0.05 if m.entry.get("language") == language else 0.0)),
            reverse=True,
        )
        best = matches[0]
        return best if best.score >= min_score else None

    def get(self, entry_id: str) -> Optional[dict]:
        for entry in self.entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def topics(self) -> List[str]:
        return sorted({e.get("topic", "") for e in self.entries})


# --------------------------------------------------------------------------- #
# Text classification
# --------------------------------------------------------------------------- #
@dataclass
class ClassificationResult:
    """Two complementary views of the same text.

    * ``label`` — the requested task. When a trained topic model is loaded this
      is the predicted topic ("teknologi", "kesehatan", ...); otherwise it is a
      speech-act label from :data:`DEFAULT_TAXONOMY` ("pertanyaan", "keluhan").
    * ``speech_act`` — always computed from the keyword taxonomy, so the
      conversational role of a sentence never disappears behind a topic label.
    """

    label: str
    confidence: float
    method: str
    scores: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)
    speech_act: Optional[str] = None
    speech_act_scores: Dict[str, float] = field(default_factory=dict)
    topic: Optional[str] = None
    topic_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "scores": {k: round(v, 4) for k, v in sorted(
                self.scores.items(), key=lambda kv: kv[1], reverse=True)},
            "rationale": self.rationale,
        }
        # Both views are always present so clients can rely on the shape:
        # ``topic`` is null when no topic model has been trained.
        data["speech_act"] = self.speech_act
        data["speech_act_scores"] = {
            k: round(v, 4) for k, v in sorted(
                self.speech_act_scores.items(), key=lambda kv: kv[1], reverse=True)
        }
        data["topic"] = self.topic
        data["topic_scores"] = {
            k: round(v, 4) for k, v in sorted(
                self.topic_scores.items(), key=lambda kv: kv[1], reverse=True)
        }
        return data


DEFAULT_TAXONOMY: Dict[str, List[str]] = {
    "pertanyaan": ["apa", "apakah", "bagaimana", "mengapa", "kenapa", "kapan",
                   "dimana", "siapa", "berapa", "what", "how", "why", "when",
                   "where", "who", "which", "?"],
    "perintah": ["tolong", "bantu", "coba", "buatkan", "jelaskan", "tampilkan",
                 "hitung", "analisis", "hapus", "kirim", "please", "help",
                 "show", "create", "explain", "analyze", "delete", "send"],
    "keluhan": ["kecewa", "buruk", "rusak", "error", "lambat", "gagal", "tidak puas",
                "jelek", "parah", "complain", "broken", "slow", "failed", "bad"],
    "pujian": ["bagus", "hebat", "keren", "mantap", "puas", "membantu", "suka",
               "great", "excellent", "love", "helpful", "perfect", "amazing"],
    "sapaan": ["halo", "hai", "hello", "hi", "pagi", "siang", "malam",
               "selamat", "morning", "evening"],
    "ucapan_terima_kasih": ["terima kasih", "makasih", "thanks", "thank you", "thx"],
}


class TextClassifier:
    """Classifies text into either a trained model's labels or a keyword taxonomy."""

    def __init__(self) -> None:
        self.normalizer = get_normalizer()
        self._model = None
        self._vectorizer = None
        self.metadata: Dict = {}
        self._load()

    def _load(self) -> None:
        model_path = Path(get_settings().models_dir) / "classification" / "model.joblib"
        if not model_path.exists():
            return
        try:
            import joblib

            bundle = joblib.load(model_path)
            self._model = bundle["model"]
            self._vectorizer = bundle["vectorizer"]
            self.metadata = bundle.get("metadata", {})
            logger.info("Loaded classification model %s", self.metadata.get("version"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load classification model: %s", exc)

    @property
    def model_available(self) -> bool:
        return self._model is not None

    def classify(
        self,
        text: str,
        labels: Optional[Dict[str, List[str]]] = None,
        language: str = "id",
        task: str = "auto",
    ) -> ClassificationResult:
        """Classify ``text``.

        ``task``:
          * ``"auto"`` (default) — use the trained topic model when it exists,
            otherwise the speech-act taxonomy;
          * ``"topic"`` — force the trained topic model (falls back to the
            taxonomy with a clear ``method`` when no model is trained);
          * ``"speech_act"`` — force the keyword taxonomy.
        """
        normalized = self.normalizer.normalize(text, language=language, keep=())
        speech_act, act_scores, act_rationale = self._taxonomy(normalized, DEFAULT_TAXONOMY)
        if labels is not None:
            label, scores, rationale = self._taxonomy(normalized, labels)
            return ClassificationResult(
                label, self._confidence(scores), "custom-taxonomy", scores, rationale,
                speech_act=speech_act, speech_act_scores=act_scores,
            )
        topic, topic_scores = None, {}
        if self._model is not None and task in ("auto", "topic"):
            features = self._vectorizer.transform([normalized])
            proba = self._model.predict_proba(features)[0]
            classes = list(self._model.classes_)
            topic_scores = {str(c): float(p) for c, p in zip(classes, proba)}
            topic = max(topic_scores, key=topic_scores.__getitem__)

        if task == "topic" and topic is not None:
            return ClassificationResult(
                topic, topic_scores[topic], "trained-model", topic_scores,
                speech_act=speech_act, speech_act_scores=act_scores,
                topic=topic, topic_scores=topic_scores,
            )
        # Default: the conversational role is the label; the trained topic (if
        # any) is reported alongside it instead of silently replacing it.
        method = (
            "keyword-taxonomy+trained-topic" if topic is not None else "keyword-taxonomy"
        )
        return ClassificationResult(
            speech_act, self._confidence(act_scores), method, act_scores, act_rationale,
            speech_act=speech_act, speech_act_scores=act_scores,
            topic=topic, topic_scores=topic_scores,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _confidence(scores: Dict[str, float]) -> float:
        if not any(scores.values()):
            return 0.25
        total = sum(scores.values()) or 1.0
        return max(0.35, min(0.97, max(scores.values()) / total + 0.25))

    def _taxonomy(self, normalized: str, taxonomy: Dict[str, List[str]]):
        """Score ``normalized`` against a keyword taxonomy (deterministic)."""
        scores: Dict[str, float] = {}
        rationale: List[str] = []
        words = set(_WORD_RE.findall(normalized.lower()))
        for label, keywords in taxonomy.items():
            hits = [kw for kw in keywords if kw in normalized.lower()]
            hits = [kw for kw in hits if " " in kw or kw in words or kw == "?"]
            if not hits:
                scores[label] = 0.0
                continue
            scores[label] = len(hits) / math.sqrt(len(keywords))
            rationale.append(f"{label}: {', '.join(hits[:4])}")
        if not any(scores.values()):
            return "lainnya", scores, rationale
        best = max(scores, key=scores.__getitem__)
        # A sentence can carry an explicit request ("tolong jelaskan ...") *and*
        # evaluative words; the request framing is the more useful label.
        if best != "perintah" and scores.get("perintah", 0.0) > 0 and "perintah" in scores:
            if any(marker in normalized for marker in ("tolong ", "mohon ", "please ", "bantu ")):
                if scores["perintah"] >= scores[best] * 0.8:
                    best = "perintah"
        if best == "pertanyaan" and scores.get("keluhan", 0) > scores.get("pertanyaan", 0):
            best = "keluhan"
        return best, scores, rationale

    def _classify_legacy(
        self,
        text: str,
        labels: Optional[Dict[str, List[str]]] = None,
        language: str = "id",
    ) -> ClassificationResult:
        normalized = self.normalizer.normalize(text, language=language, keep=())
        if self._model is not None and labels is None:
            features = self._vectorizer.transform([normalized])
            proba = self._model.predict_proba(features)[0]
            classes = list(self._model.classes_)
            mapping = {c: float(p) for c, p in zip(classes, proba)}
            best = max(mapping, key=mapping.__getitem__)
            return ClassificationResult(best, mapping[best], "trained-model", mapping)
        taxonomy = labels or DEFAULT_TAXONOMY
        scores: Dict[str, float] = {}
        rationale: List[str] = []
        words = set(_WORD_RE.findall(normalized.lower()))
        for label, keywords in taxonomy.items():
            hits = [kw for kw in keywords if kw in normalized.lower()]
            hits = [kw for kw in hits if " " in kw or kw in words or kw == "?"]
            if not hits:
                scores[label] = 0.0
                continue
            scores[label] = len(hits) / math.sqrt(len(keywords))
            rationale.append(f"{label}: {', '.join(hits[:4])}")
        if not any(scores.values()):
            return ClassificationResult("lainnya", 0.25, "keyword-taxonomy", scores, rationale)
        best = max(scores, key=scores.__getitem__)
        total = sum(scores.values()) or 1.0
        confidence = max(0.35, min(0.97, scores[best] / total + 0.25))
        # A sentence can carry an explicit request ("tolong jelaskan ...") *and*
        # evaluative words; the request framing is the more useful label.
        if best != "perintah" and scores.get("perintah", 0.0) > 0 and "perintah" in scores:
            if any(marker in normalized for marker in ("tolong ", "mohon ", "please ", "bantu ")):
                if scores["perintah"] >= scores[best] * 0.8:
                    best = "perintah"
        if best == "pertanyaan" and scores.get("keluhan", 0) > scores["pertanyaan"]:
            best = "keluhan"
        return ClassificationResult(best, confidence, "keyword-taxonomy", scores, rationale)

    def labels(self) -> List[str]:
        """Labels produced by default (:meth:`classify` with ``task="auto"``)."""
        return list(DEFAULT_TAXONOMY)

    def topic_labels(self) -> List[str]:
        """Labels of the trained topic model (empty when untrained)."""
        if self._model is not None:
            return [str(c) for c in self._model.classes_]
        return []

    def info(self) -> dict:
        return {
            "available": self.model_available,
            "version": self.metadata.get("version"),
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics", {}),
            "labels": self.labels(),
            "topic_labels": self.topic_labels(),
            "method": "trained-model" if self.model_available else "keyword-taxonomy",
        }


# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #
_semantic_index: Optional[SemanticIndex] = None
_knowledge_base: Optional[KnowledgeBase] = None
_classifier: Optional[TextClassifier] = None
_lock = threading.Lock()


def get_semantic_index() -> SemanticIndex:
    global _semantic_index
    if _semantic_index is None:
        with _lock:
            if _semantic_index is None:
                _semantic_index = SemanticIndex()
    return _semantic_index


def get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        with _lock:
            if _knowledge_base is None:
                _knowledge_base = KnowledgeBase()
    return _knowledge_base


def get_text_classifier() -> TextClassifier:
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = TextClassifier()
    return _classifier


def supported_index_languages() -> List[str]:
    return sorted(SUPPORTED_LANGUAGES.keys())
