"""Intent detection module.

Three complementary strategies are combined:

* **Rule-based detection** — high-precision keyword/regex rules per intent
  (greeting, farewell, gratitude, delete-data requests ...). Rules win when a
  strong pattern matches, because they are explainable and never wrong on the
  obvious cases.
* **Statistical classification** — a TF-IDF + Logistic Regression model trained
  by ``scripts/train.py --task intent``. Handles paraphrases and typos because
  the vectoriser works on character n-grams as well as word n-grams.
* **Fuzzy similarity fallback** — when no trained model exists, the utterance
  is compared with the intent catalogue examples using token Jaccard overlap,
  which still gives reasonable coverage on unseen wording.

Every decision is returned together with its confidence, the strategy used and
(from rules / fuzzy matching) the matched evidence, so the API remains
auditable and never invents certainty.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chatgangs.config import get_settings
from chatgangs.core.normalizer import get_normalizer
from chatgangs.core.stopwords import get_stopword_remover
from chatgangs.core.text_utils import LRUCache, _cache_capacity
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IntentMatch:
    name: str
    confidence: float
    method: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "evidence": self.evidence,
            "candidates": {k: round(v, 4) for k, v in sorted(
                self.all_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]},
        }


# --------------------------------------------------------------------------- #
# Rule patterns: (intent, regex, weight)
# Patterns are matched against the normalised lowercase utterance.
# --------------------------------------------------------------------------- #
RULES: List[Tuple[str, str, float]] = [
    # --- explicit request to delete memory/data (highest precision)
    ("privacy", r"\b(hapus|delete|clear|bersihkan|forget|lupakan)\b.{0,25}\b"
                r"(riwayat|histori|history|memori|memory|data|chat|percakapan|conversation|memories)\b", 0.95),
    ("privacy", r"\b(forget me|lupakan saya|delete my data|hapus data saya)\b", 0.97),
    ("privacy", r"\b(privasi|privacy)\b.{0,30}\b(data|percakapan|chat|simpan|stored)\b", 0.8),
    ("privacy", r"\b(apakah|does|do|is)\b.{0,25}\b(disimpan|menyimpan|stored|save|keep)\b.{0,30}\b(data|chat|percakapan|conversation)\b", 0.8),

    # --- greetings / farewells / thanks
    ("greeting", r"^(halo|hallo|hai|hei|hi|hello|hey|yo|pagi|siang|sore|malam|"
                 r"selamat pagi|selamat siang|selamat sore|selamat malam|"
                 r"good morning|good afternoon|good evening|assalamualaikum|salam)"
                 r"(?:[\s,.!]+(?:bot|chatgangs|semua|there|kak|gan|bro|sis))?[\s,.!?]*$", 0.92),
    ("greeting", r"\b(apa kabar|how are you|how's it going|kabar baik)\b", 0.85),
    ("farewell", r"^(selamat tinggal|sampai jumpa|dadah|bye|goodbye|see you|"
                 r"sampai nanti|good night|selamat jalan|talk later)\b", 0.92),
    ("gratitude", r"\b(terima kasih|terimakasih|makasih|trims|thanks|thank you|"
                  r"thx|tq|appreciate it)\b", 0.9),

    # --- capability / identity / api / pricing / training
    ("identity", r"\b(siapa (kamu|anda|kamu ini)|who are you|what are you|"
                 r"apa itu chatgangs|what is chatgangs|perkenalkan (diri|dirimu)|"
                 r"introduce yourself|kenalan)\b", 0.9),
    ("capabilities", r"\b(kamu bisa apa|apa (saja )?kemampuan(mu)?|"
                     r"fitur (apa|saja)|what can you do|your (features|capabilities)|"
                     r"apa yang bisa kamu lakukan|bisa bantu apa)\b", 0.88),
    ("api_help", r"\b(api|endpoint|rest|curl|integrasi|integrate|webhook|sdk|"
                 r"swagger|dokumentasi api|api docs)\b", 0.72),
    ("pricing", r"\b(harga|harganya|biaya|gratis|free|pricing|cost|langganan|"
                r"subscription|bayar|premium)\b", 0.75),
    ("training_help", r"\b(training|latih|melatih|fine[- ]?tun(e|ing)|dataset|"
                      r"retrain|akurasi model|model accuracy|evaluasi model)\b", 0.75),

    # --- analysis requests
    ("analyze_text", r"\b(analisis|analisa|analize|analy[sz]e|cek sentimen|"
                     r"sentiment of|extract entit|ekstrak entitas|klasifikasikan|"
                     r"classify|tokeni[sz]e|deteksi bahasa|detect language)\b", 0.8),

    # --- translation
    ("translate", r"\b(terjemah\w*|translate|artikan|bahasa inggrisnya|"
                  r"how do you say|in japanese|ke bahasa)\b", 0.8),

    # --- feedback / complaint
    ("complaint", r"\b(tidak puas|kecewa|buruk|jelek|parah|error terus|crash|"
                  r"tidak (bekerja|berfungsi|membantu)|doesn'?t work|not working|"
                  r"useless|terrible|awful|worst|kesal|marah|lambat sekali|bug)\b", 0.8),
    ("feedback_positive", r"\b(bagus sekali|keren|mantap|sangat membantu|"
                          r"memuaskan|suka (sistem|ini|kamu)|great|awesome|"
                          r"love it|good job|nice work|well done|perfect)\b", 0.78),

    # --- help / question
    ("help", r"^(help|tolong|bantu|bantu saya|saya bingung|bingung|i need help)\b", 0.82),
    ("help", r"\b(mulai dari mana|how does this work|gimana cara pakai|"
             r"where do i start|cara pakai)\b", 0.78),
    ("question", r"\b(ingin|mau|hendak) (bertanya|tanya|menanyakan)\b", 0.55),
    ("question", r"\bi (want|would like) to ask\b", 0.55),
    ("question", r"^(apa|apakah|bagaimana|bagaimanakah|mengapa|kenapa|kapan|"
                 r"dimana|di mana|siapa|berapa|what|why|how|when|where|who|which|"
                 r"can you explain|jelaskan|explain)\b", 0.6),
    ("small_talk", r"\b(lagi apa|lagi ngapain|what'?s up|tell me a joke|"
                    r"bercanda|fun fact|how old are you|ceritakan)\b", 0.75),
]

_COMPILED_RULES = [(intent, re.compile(pattern, re.IGNORECASE), weight)
                   for intent, pattern, weight in RULES]

# Social intents describe the *frame* of an utterance ("hello, what can you
# do?"); content intents describe what the user actually wants. When both are
# present in a longer utterance the content intent must win.
SOCIAL_INTENTS = {"greeting", "farewell", "gratitude", "small_talk"}
CONTENT_INTENTS = {
    "identity", "capabilities", "api_help", "pricing", "privacy",
    "training_help", "analyze_text", "translate", "feedback_positive",
    "complaint", "help", "question",
}


class IntentDetector:
    """Hybrid intent detector."""

    def __init__(self, catalogue_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.catalogue_path = catalogue_path or (
            Path(settings.data_dir) / "intents" / "intents.json"
        )
        self.normalizer = get_normalizer()
        self.stopwords = get_stopword_remover()
        self.catalogue: List[dict] = []
        self.examples: List[Tuple[str, str]] = []
        self._load_catalogue()
        self._load_model()
        self._cache = LRUCache[IntentMatch](capacity=_cache_capacity(4096))

    # ------------------------------------------------------------------ loading
    def _load_catalogue(self) -> None:
        if not self.catalogue_path.exists():
            logger.warning("Intent catalogue not found at %s", self.catalogue_path)
            self.catalogue = []
            return
        payload = json.loads(self.catalogue_path.read_text(encoding="utf-8"))
        self.catalogue = payload.get("intents", [])
        self.examples = [
            (example, intent["name"])
            for intent in self.catalogue
            for example in intent.get("examples", [])
        ]
        logger.info(
            "Loaded %d intents / %d examples", len(self.catalogue), len(self.examples)
        )

    def _load_model(self) -> None:
        self.model = None
        self.vectorizer = None
        self.metadata: Dict = {}
        model_dir = Path(get_settings().models_dir) / "intent"
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            return
        try:
            import joblib

            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.vectorizer = bundle["vectorizer"]
            self.metadata = bundle.get("metadata", {})
            logger.info("Loaded intent model %s", self.metadata.get("version", "unknown"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load intent model: %s", exc)

    def reload(self) -> None:
        self._load_catalogue()
        self._load_model()
        self._cache.clear()

    @property
    def model_available(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    @property
    def intents(self) -> List[dict]:
        return self.catalogue

    def response_hint(self, intent_name: str) -> Optional[str]:
        for intent in self.catalogue:
            if intent["name"] == intent_name:
                return intent.get("response_hint")
        return None

    def description(self, intent_name: str) -> Optional[str]:
        for intent in self.catalogue:
            if intent["name"] == intent_name:
                return intent.get("description")
        return None

    # ----------------------------------------------------------------- detect
    def detect(
        self,
        text: str,
        language: str = "id",
        *,
        threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> IntentMatch:
        threshold = threshold if threshold is not None else get_settings().intent_confidence_threshold
        cache_key = f"{language}|{threshold}|{text}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        normalized = self.normalizer.normalize(text, language=language, keep=("emoji",))
        scores: Dict[str, float] = {}
        evidence: List[str] = []

        # 1. Rules -------------------------------------------------------
        rule_intents: set = set()
        for intent, pattern, weight in _COMPILED_RULES:
            match = pattern.search(normalized)
            if match:
                scores[intent] = max(scores.get(intent, 0.0), weight)
                rule_intents.add(intent)
                evidence.append(f"rule:{intent}:'{match.group(0)[:40]}'")

        # 2. Statistical model ------------------------------------------
        model_scores: Dict[str, float] = {}
        if self.model_available and normalized.strip():
            try:
                features = self.vectorizer.transform([normalized])
                proba = self.model.predict_proba(features)[0]
                model_scores = {
                    label: float(prob) for label, prob in zip(self.model.classes_, proba)
                }
                for intent, prob in model_scores.items():
                    scores[intent] = max(scores.get(intent, 0.0), min(0.99, prob * 1.05))
            except Exception as exc:  # pragma: no cover
                logger.warning("Intent model inference failed: %s", exc)

        # 3. Fuzzy example overlap (only when no model) ------------------
        if not self.model_available and normalized.strip():
            for intent, similarity in self._fuzzy_scores(normalized, language):
                scores[intent] = max(scores.get(intent, 0.0), similarity)
                if similarity >= 0.55:
                    evidence.append(f"fuzzy:{intent}")

        if not scores:
            result = IntentMatch("unknown", 0.0, "none", {})
        else:
            # A social frame ("hello, what can you do?") must not mask a real
            # request, but a greeting combined with small talk must stay a
            # greeting. Only a *strong* content signal wins.
            token_count = len(normalized.split())
            rule_content = {
                intent for intent in scores
                if intent in CONTENT_INTENTS and intent in rule_intents and scores[intent] >= 0.6
            }
            content_present = rule_content
            social_only = set(scores) <= SOCIAL_INTENTS
            if content_present and token_count > 3 and not social_only:
                best_social = max(
                    (i for i in scores if i in SOCIAL_INTENTS),
                    key=lambda i: scores[i],
                    default=None,
                )
                best_content = max(
                    content_present, key=lambda i: scores[i]
                )
                if best_social and scores[best_social] >= scores[best_content]:
                    scores[best_content] = min(0.995, scores[best_social] + 0.01)
                    evidence.append(f"social-overridden:{best_social}")
            best = max(scores, key=scores.__getitem__)
            confidence = float(scores[best])
            if confidence < threshold:
                runner_up = sorted(scores.values(), reverse=True)
                gap = runner_up[0] - (runner_up[1] if len(runner_up) > 1 else 0.0)
                result = IntentMatch(
                    "unknown" if confidence < threshold * 0.6 or gap < 0.05 else best,
                    confidence,
                    "low-confidence",
                    scores,
                    evidence,
                )
                if result.name == "unknown":
                    result.confidence = confidence
            else:
                method = "rule" if any(e.startswith("rule:") for e in evidence) and best in {
                    e.split(":")[1] for e in evidence if e.startswith("rule:")
                } else ("model" if best in model_scores else "fuzzy")
                if best in model_scores and best in {e.split(":")[1] for e in evidence if e.startswith("rule:")}:
                    method = "rule+model"
                result = IntentMatch(best, confidence, method, scores, evidence)

        if use_cache:
            self._cache.put(cache_key, result)
        return result

    def _fuzzy_scores(self, normalized: str, language: str) -> List[Tuple[str, float]]:
        tokens = self.stopwords.remove(normalized.split(), language)
        token_set = set(tokens)
        if not token_set:
            return []
        best_per_intent: Dict[str, float] = {}
        for example, intent in self.examples:
            ex_tokens = set(self.stopwords.remove(example.lower().split(), language))
            if not ex_tokens:
                continue
            overlap = len(token_set & ex_tokens)
            if overlap == 0:
                continue
            jaccard = overlap / len(token_set | ex_tokens)
            containment = overlap / min(len(token_set), len(ex_tokens))
            score = max(jaccard, containment * 0.85)
            if score > best_per_intent.get(intent, 0.0):
                best_per_intent[intent] = score
        return sorted(best_per_intent.items(), key=lambda kv: kv[1], reverse=True)

    def training_examples(self) -> List[Tuple[str, str]]:
        return list(self.examples)

    def model_info(self) -> dict:
        return {
            "available": self.model_available,
            "version": self.metadata.get("version"),
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics", {}),
            "intent_count": len(self.catalogue),
            "example_count": len(self.examples),
            "intents": [i["name"] for i in self.catalogue],
            "algorithm": self.metadata.get("algorithm", "tfidf+logistic-regression"),
        }


_detector: Optional[IntentDetector] = None


def get_intent_detector() -> IntentDetector:
    global _detector
    if _detector is None:
        _detector = IntentDetector()
    return _detector
