"""ChatGangs NLP pipeline — the orchestrator that wires every module together.

Two entry points:

``analyze(text)``
    Runs the full understanding stack once and returns a typed
    :class:`AnalysisResult`: language, tokens, stems/lemmas, stopwords,
    entities, intent, sentiment, semantic keywords, classification and
    question-answering candidates.

``chat(message, session_id)``
    Uses the analysis, the conversation context and the memory manager to
    decide *what to say*, retrieves the answer from the knowledge base when the
    intent is a question, then has the NLG module phrase the response.

Both are synchronous and CPU-bound; the API layer runs them in a thread pool
so the event loop is never blocked.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chatgangs.config import get_settings
from chatgangs.core.context import ConversationState, get_context_manager
from chatgangs.core.entities import Entity, get_entity_recognizer
from chatgangs.core.intent import IntentMatch, get_intent_detector
from chatgangs.core.languages import SUPPORTED_LANGUAGES, detect_language
from chatgangs.core.nlg import GenerationResult, get_response_generator
from chatgangs.core.normalizer import get_normalizer
from chatgangs.core.semantics import (
    ClassificationResult,
    QAMatch,
    get_knowledge_base,
    get_semantic_index,
    get_text_classifier,
)
from chatgangs.core.sentiment import SentimentResult, get_sentiment_analyzer
from chatgangs.core.stemmer import get_stemmer
from chatgangs.core.stopwords import get_stopword_remover
from chatgangs.core.text_utils import LRUCache, truncate
from chatgangs.logging_config import Timer
from chatgangs.core.tokenizer import get_tokenizer
from chatgangs.logging_config import get_logger
from chatgangs.memory.manager import get_memory_manager

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class AnalysisResult:
    text: str
    language: str
    language_confidence: float
    language_method: str
    normalized: str
    tokens: List[dict]
    token_stats: Dict[str, Any]
    stems: List[str]
    lemmas: List[str]
    stopwords: List[str]
    content_terms: List[str]
    entities: List[dict]
    entity_summary: Dict[str, List[str]]
    intent: dict
    sentiment: dict
    classification: dict
    keywords: List[Dict[str, Any]]
    qa_matches: List[dict]
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": {
                "code": self.language,
                "name": SUPPORTED_LANGUAGES.get(self.language, {}).get("name", "Unknown"),
                "confidence": round(self.language_confidence, 4),
                "method": self.language_method,
            },
            "normalized": self.normalized,
            "tokens": self.tokens,
            "token_stats": self.token_stats,
            "stems": self.stems,
            "lemmas": self.lemmas,
            "stopwords": self.stopwords,
            "content_terms": self.content_terms,
            "entities": self.entities,
            "entity_summary": self.entity_summary,
            "intent": self.intent,
            "sentiment": self.sentiment,
            "classification": self.classification,
            "keywords": self.keywords,
            "qa_matches": self.qa_matches,
            "stats": self.stats,
        }


@dataclass
class ChatResult:
    session_id: str
    reply: str
    language: str
    intent: dict
    sentiment: dict
    entities: List[dict]
    context: Dict[str, Any]
    generation: Dict[str, Any]
    memory: Dict[str, Any]
    latency_ms: float
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "reply": self.reply,
            "language": self.language,
            "intent": self.intent,
            "sentiment": self.sentiment,
            "entities": self.entities,
            "context": self.context,
            "generation": self.generation,
            "memory": self.memory,
            "latency_ms": round(self.latency_ms, 2),
            "suggestions": self.suggestions,
        }


MAX_TOKENS = 400


class NLPengine:
    """Facade over all ChatGangs NLP modules."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tokenizer = get_tokenizer()
        self.normalizer = get_normalizer()
        self.stemmer = get_stemmer()
        self.stopwords = get_stopword_remover()
        self.sentiment = get_sentiment_analyzer()
        self.intents = get_intent_detector()
        self.entities = get_entity_recognizer()
        self.context = get_context_manager()
        self.memory = get_memory_manager()
        self.nlg = get_response_generator()
        self.kb = get_knowledge_base()
        self.semantic = get_semantic_index()
        self.classifier = get_text_classifier()
        # Optional caches (disabled by default; enabled with
        # CHATGANGS_RESPONSE_CACHE_TTL_SECONDS > 0).
        cache_size = (
            self.settings.response_cache_size
            if self.settings.response_cache_ttl_seconds > 0 else 0
        )
        self._analysis_cache = LRUCache[AnalysisResult](capacity=cache_size)
        self._response_cache = LRUCache[ChatResult](capacity=cache_size)

    @property
    def caches_enabled(self) -> bool:
        return self._analysis_cache.capacity > 1 or self._response_cache.capacity > 1

    def cache_stats(self) -> dict:
        return {
            "enabled": self.caches_enabled,
            "ttl_seconds": self.settings.response_cache_ttl_seconds,
            "analysis": self._analysis_cache.stats(),
            "chat": self._response_cache.stats(),
        }

    def _apply_token_budget(self, text: str) -> str:
        """Enforce ``CHATGANGS_MAX_TOKENS_PER_REQUEST`` (approximate by words)."""
        limit = max(int(self.settings.max_tokens_per_request or 0), 1)
        words = text.split()
        if len(words) <= limit:
            return text
        logger.info("Input truncated to %s tokens", limit)
        return " ".join(words[:limit])

    # ------------------------------------------------------------------ analyze
    def analyze(
        self,
        text: str,
        language: Optional[str] = None,
        *,
        explain: bool = True,
        include_tokens: bool = True,
        include_qa: bool = True,
    ) -> AnalysisResult:
        if self._analysis_cache.capacity > 1:
            key = f"{language}|{explain}|{include_tokens}|{include_qa}|{text}"
            cached = self._analysis_cache.get(key)
            if cached is not None:
                return cached
            result = self._analyze_uncached(
                text,
                language=language,
                explain=explain,
                include_tokens=include_tokens,
                include_qa=include_qa,
            )
            self._analysis_cache.put(key, result)
            return result
        return self._analyze_uncached(
            text,
            language=language,
            explain=explain,
            include_tokens=include_tokens,
            include_qa=include_qa,
        )

    def _analyze_uncached(
        self,
        text: str,
        language: Optional[str] = None,
        *,
        explain: bool = True,
        include_tokens: bool = True,
        include_qa: bool = True,
    ) -> AnalysisResult:
        started = time.perf_counter()
        text = (text or "").strip()
        if len(text) > self.settings.max_input_chars:
            text = truncate(text, self.settings.max_input_chars)
        text = self._apply_token_budget(text)
        if not text:
            raise ValueError("Text must not be empty")

        timer = Timer()
        with timer:
            prediction = detect_language(text, self.settings.language_confidence_threshold)
            lang = language if language in SUPPORTED_LANGUAGES else prediction.language

        with Timer() as t_norm:
            normalized = self.normalizer.normalize(text, language=lang, keep=("emoji",))
        with Timer() as t_tok:
            tokens = self.tokenizer.tokenize(normalized, keep_punctuation=False)
            # enrich tokens with stopword / pos-lite / stem / lemma
            self.stopwords.filter_tokens(tokens, lang)
            for token in tokens:
                if token.is_word and not token.is_number:
                    token.pos = _guess_pos(token.normalized, lang)
                    token.stem = self.stemmer.stem(token.normalized, lang)
                    token.lemma = self.stemmer.lemmatize(token.normalized, lang, token.pos)
            token_stats = self.tokenizer.count(text)
            stems = [t.stem for t in tokens if t.stem]
            lemmas = [t.lemma for t in tokens if t.lemma]
            stopword_list = [t.normalized for t in tokens if t.is_stopword]
            content_terms = [
                t.stem or t.normalized
                for t in tokens
                if not t.is_stopword and t.is_word and not t.is_number
            ]

        with Timer() as t_sent:
            sentiment = self.sentiment.analyze(text, lang, explain=explain)
        with Timer() as t_ent:
            entities = self.entities.extract(text, lang)
            entity_summary = self.entities.summarize(entities)
        with Timer() as t_int:
            intent = self.intents.detect(text, lang)
        with Timer() as t_cls:
            classification = self.classifier.classify(text, language=lang)

        keywords = self._keywords(content_terms, tokens)
        qa_matches: List[QAMatch] = []
        if include_qa:
            with Timer() as t_qa:
                qa_matches = self.kb.retrieve(text, language=lang, top_k=3)
        else:
            t_qa = Timer()

        total_ms = (time.perf_counter() - started) * 1000
        return AnalysisResult(
            text=text,
            language=lang,
            language_confidence=prediction.confidence,
            language_method=prediction.method,
            normalized=normalized,
            tokens=[t.to_dict() for t in tokens] if include_tokens else [],
            token_stats=token_stats,
            stems=stems,
            lemmas=lemmas,
            stopwords=stopword_list,
            content_terms=content_terms,
            entities=[e.to_dict() for e in entities],
            entity_summary=entity_summary,
            intent=intent.to_dict(),
            sentiment=sentiment.to_dict(),
            classification=classification.to_dict(),
            keywords=keywords,
            qa_matches=[m.to_dict() for m in qa_matches],
            stats={
                "detect_language_ms": round(
                    max(total_ms - t_norm.elapsed_ms - t_tok.elapsed_ms - t_sent.elapsed_ms
                        - t_ent.elapsed_ms - t_int.elapsed_ms - t_cls.elapsed_ms
                        - t_qa.elapsed_ms, 0.0), 2),
                "normalize_ms": round(t_norm.elapsed_ms, 2),
                "tokenize_ms": round(t_tok.elapsed_ms, 2),
                "sentiment_ms": round(t_sent.elapsed_ms, 2),
                "entities_ms": round(t_ent.elapsed_ms, 2),
                "intent_ms": round(t_int.elapsed_ms, 2),
                "classification_ms": round(t_cls.elapsed_ms, 2),
                "qa_ms": round(t_qa.elapsed_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        )

    # --------------------------------------------------------------------- chat
    def chat(
        self,
        message: str,
        session_id: str = "default",
        *,
        language: Optional[str] = None,
        store_memory: Optional[bool] = None,
        user_profile: Optional[dict] = None,
    ) -> ChatResult:
        started = time.perf_counter()
        message = (message or "").strip()
        if not message:
            raise ValueError("Message must not be empty")
        if len(message) > self.settings.max_message_chars:
            message = truncate(message, self.settings.max_message_chars)
        message = self._apply_token_budget(message)

        state = self.context.get(session_id)
        prediction = detect_language(message, self.settings.language_confidence_threshold)
        lang = language if language in SUPPORTED_LANGUAGES else prediction.language
        if state.turn_count == 0:
            state.language = lang

        # -------- understanding
        sentiment = self.sentiment.analyze(message, lang, explain=False)
        entities = self.entities.extract(message, lang)
        context_digest = self.context.enrich(message, state, lang)
        intent_input = message
        if context_digest.get("is_followup") and context_digest.get("expanded_query"):
            intent_input = context_digest["expanded_query"]
        intent = self.intents.detect(intent_input, lang)
        topic_intent = state.last_intent if context_digest.get("is_followup") else None

        # -------- decision + generation
        reply, generation, extra = self._respond(
            message=message,
            lang=lang,
            intent=intent,
            sentiment=sentiment,
            entities=entities,
            context_digest=context_digest,
            state=state,
            session_id=session_id,
            topic_intent=topic_intent,
        )

        # -------- persistence
        # A deletion request must leave nothing behind: the reply is returned
        # but not written back into the history it just erased.
        persist = store_memory if store_memory is not None else True
        if extra.get("erased"):
            persist = False
        memory_info: Dict[str, Any] = {"enabled": self.settings.memory_enabled and persist}
        if persist and self.settings.memory_enabled:
            self.memory.ensure_session(
                session_id, language=lang, user_profile=user_profile
            )
            self.memory.add_message(
                session_id,
                "user",
                message,
                intent=intent.name,
                language=lang,
                sentiment=sentiment.label,
                metadata={"entities": [e.to_dict() for e in entities][:10]},
                tokens_used=len(message.split()),
            )
            message_id = self.memory.add_message(
                session_id,
                "assistant",
                reply,
                intent=intent.name,
                language=lang,
                tokens_used=len(reply.split()),
            )
            memory_info.update({"message_id": message_id, "stored": True})
        else:
            memory_info["stored"] = False

        if extra.get("erased"):
            # Recreate a clean context object for subsequent turns.
            self.context.get(session_id)
        self.context.record_user_turn(
            state,
            message,
            intent=intent.name,
            entities=[e.to_dict() for e in entities],
            sentiment=sentiment.label,
            language=lang,
            topic=extra.get("topic"),
        )
        self.context.record_assistant_turn(state, reply, intent=intent.name, language=lang)

        latency_ms = (time.perf_counter() - started) * 1000
        return ChatResult(
            session_id=session_id,
            reply=reply,
            language=lang,
            intent=intent.to_dict(),
            sentiment={
                "label": sentiment.label,
                "score": round(sentiment.score, 4),
                "confidence": round(sentiment.confidence, 4),
            },
            entities=[e.to_dict() for e in entities],
            context={
                "is_followup": context_digest.get("is_followup", False),
                "topic": state.topic,
                "turn_count": state.turn_count,
                "language": lang,
                "language_confidence": round(prediction.confidence, 4),
                "sentiment_trend": self.context.sentiment_trend(state),
            },
            generation=generation.to_dict(),
            memory=memory_info,
            latency_ms=latency_ms,
            suggestions=extra.get("suggestions", DEFAULT_SUGGESTIONS.get(lang, DEFAULT_SUGGESTIONS["en"])),
        )

    # ------------------------------------------------------------------ respond
    def _respond(
        self,
        *,
        message: str,
        lang: str,
        intent: IntentMatch,
        sentiment: SentimentResult,
        entities: List[Entity],
        context_digest: Dict[str, Any],
        state: ConversationState,
        session_id: str,
        topic_intent: Optional[str],
    ):
        name = intent.name
        suggestions = DEFAULT_SUGGESTIONS.get(lang, DEFAULT_SUGGESTIONS["en"])
        extra: Dict[str, Any] = {"topic": None, "suggestions": suggestions}
        wants_detail = _wants_detail(message, lang)

        # 1. Explicit data-deletion request → act, then confirm.
        if name == "privacy" and _is_deletion_request(message):
            self.memory.delete_session(session_id)
            self.context.reset(session_id)
            self.nlg.reset(session_id)
            generation = self.nlg.generate(
                "privacy_deleted", lang, session_id=session_id
            )
            extra["topic"] = "privacy"
            extra["erased"] = True
            return generation.text, generation, extra

        # 2. A follow-up that asks for more depth expands the previous answer
        #    from the knowledge base (its ``details`` field) instead of
        #    repeating the same short template.
        if context_digest.get("is_followup") and wants_detail and topic_intent:
            match = self._topic_match(topic_intent, lang)
            if match is not None and match.entry.get("details"):
                answer = (
                    self._fill(match.entry.get("answer", ""))
                    + " " + self._fill(match.entry["details"])
                )
                generation = self.nlg.compose_answer(
                    lang, answer, intent="question", session_id=session_id
                )
                extra["topic"] = match.entry.get("topic")
                extra["qa"] = match.to_dict()
                extra["depth"] = "detailed"
                return generation.text, generation, extra

        # 3. Conversational intents rendered from templates.
        if name in TEMPLATE_INTENTS:
            generation = self.nlg.generate(
                name,
                lang,
                session_id=session_id,
                slots={"_user_text": message},
                include_followup=name in {"help", "capabilities"},
            )
            extra["topic"] = name
            return generation.text, generation, extra

        # 4. Analysis requests: summarise the analysis we already computed.
        if name == "analyze_text":
            generation = self.nlg.generate(
                "analyze_text",
                lang,
                session_id=session_id,
                slots={
                    "sentiment_label": sentiment.label,
                    "sentiment_score": f"{sentiment.score:+.2f}",
                    "intent": intent.name,
                    "language": SUPPORTED_LANGUAGES.get(lang, {}).get("name", lang),
                },
            )
            extra["topic"] = "analyze_text"
            return generation.text, generation, extra

        # 5. Question-like intents → knowledge base.
        if name in KB_INTENTS or name == "unknown":
            query = context_digest.get("expanded_query") or message
            match = self.kb.best(query, language=lang)
            # A follow-up that is too short to retrieve well re-uses the topic.
            if match is None and context_digest.get("is_followup") and topic_intent:
                match = self._topic_match(topic_intent, lang)
            if match is not None:
                answer = self._fill(match.entry.get("answer", ""))
                # A follow-up that asks for more depth returns the extended
                # explanation stored alongside the short answer.
                if match.entry.get("details") and wants_detail:
                    answer = f"{answer} {self._fill(match.entry['details'])}"
                slot_note = self._slot_note(match, entities, lang)
                generation = self.nlg.compose_answer(
                    lang if match.entry.get("language") in (lang, "en") else match.entry.get("language", lang),
                    answer,
                    intent="question",
                    session_id=session_id,
                    slot_note=slot_note,
                    include_followup=match.score < 0.35,
                )
                extra["topic"] = match.entry.get("topic")
                extra["qa"] = match.to_dict()
                return generation.text, generation, extra

        # 6. Nothing matched → honest fallback that lists real capabilities.
        topics = ", ".join(self.kb.topics()[:8])
        generation = self.nlg.generate(
            "knowledge_miss" if name in KB_INTENTS else "unknown",
            lang,
            session_id=session_id,
            slots={"topics": topics, "_user_text": message},
        )
        return generation.text, generation, extra

    def _topic_match(self, topic: str, lang: str) -> Optional[QAMatch]:
        """Find the knowledge-base entry that backs a conversation topic.

        Topic names mirror knowledge-base topics (``capabilities``, ``api``),
        so prefer an entry whose topic matches exactly and whose language
        matches the conversation; only then fall back to free-text retrieval.
        """
        if not topic:
            return None
        candidates = [
            entry for entry in self.kb.entries
            if entry.get("topic") == topic and entry.get("language") in (lang, "en")
        ]
        if candidates:
            candidates.sort(key=lambda entry: (entry.get("language") != lang,))
            return QAMatch(entry=candidates[0], score=1.0)
        return self.kb.best(topic.replace("_", " "), language=lang)

    def _fill(self, text: str) -> str:
        """Substitute the placeholders used inside knowledge-base entries."""
        return (text or "").replace(
            "{bot_name}", self.settings.app_name
        ).replace("{version}", self.settings.app_version)

    def _slot_note(self, match: QAMatch, entities: List[Entity], lang: str) -> Optional[str]:
        """Add a contextual sentence when entities were mentioned."""
        if not entities:
            return None
        label_priority = ["DATE", "TIME", "MONEY", "ORGANIZATION", "PERSON", "GPE", "LOCATION"]
        by_label: Dict[str, str] = {}
        for entity in entities:
            if entity.label in label_priority and entity.label not in by_label:
                by_label[entity.label] = entity.text
        if not by_label:
            return None
        parts = []
        for label in label_priority:
            if label in by_label and len(parts) < 2:
                parts.append(f"{ENTITY_LABEL_LANG.get(lang, ENTITY_LABEL_LANG['en']).get(label, label)} {by_label[label]}")
        if not parts:
            return None
        if lang == "id":
            return "Saya juga menangkap " + " dan ".join(parts) + " dalam pesan Anda."
        if lang == "ms":
            return "Saya juga mengesan " + " dan ".join(parts) + " dalam mesej anda."
        return "I also picked up " + " and ".join(parts) + " in your message."

    # ------------------------------------------------------------------ extras
    @staticmethod
    def _keywords(content_terms: List[str], tokens) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        positions: Dict[str, int] = {}
        for idx, term in enumerate(content_terms):
            counts[term] = counts.get(term, 0) + 1
            positions.setdefault(term, idx)
        if not counts:
            return []
        max_count = max(counts.values())
        total = len(content_terms)
        scored = []
        for term, count in counts.items():
            tf = count / total
            position_bonus = 1.0 + 0.25 * (1 - positions[term] / max(total, 1))
            scored.append(
                {
                    "term": term,
                    "frequency": count,
                    "score": round(tf * (1 + (count / max_count)) * position_bonus, 4),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:10]

    # -------------------------------------------------------------- convenience
    def tokenize(self, text: str) -> dict:
        tokens = self.tokenizer.tokenize(text or "")
        return {
            "tokens": [t.to_dict() for t in tokens],
            "sentences": self.tokenizer.sentences(text or ""),
            "stats": self.tokenizer.count(text or ""),
        }

    def analyze_sentiment(self, text: str, language: Optional[str] = None) -> dict:
        lang = language or detect_language(text).language
        return self.sentiment.analyze(text, lang).to_dict()

    def detect_intent(self, text: str, language: Optional[str] = None) -> dict:
        lang = language or detect_language(text).language
        result = self.intents.detect(text, lang)
        payload = result.to_dict()
        payload["description"] = self.intents.description(result.name)
        payload["response_hint"] = self.intents.response_hint(result.name)
        return payload

    def extract_entities(self, text: str, language: Optional[str] = None) -> dict:
        lang = language or detect_language(text).language
        entities = self.entities.extract(text, lang)
        return {
            "language": lang,
            "entities": [e.to_dict() for e in entities],
            "summary": self.entities.summarize(entities),
        }

    def similarity(self, texts: List[str], language: Optional[str] = None) -> dict:
        lang = language or detect_language(" ".join(texts[:3]) or "x").language
        return self.semantic.similarity(texts, lang).to_dict()

    def classify(
        self, text: str, language: Optional[str] = None, task: str = "auto"
    ) -> dict:
        lang = language or detect_language(text).language
        return self.classifier.classify(text, language=lang, task=task).to_dict()

    def answer_question(self, question: str, language: Optional[str] = None, top_k: int = 3) -> dict:
        lang = language or detect_language(question).language
        matches = self.kb.retrieve(question, language=lang, top_k=top_k)
        best = self.kb.best(question, language=lang)
        answer = None
        if best is not None:
            answer = best.entry.get("answer", "").replace(
                "{bot_name}", self.settings.app_name
            ).replace("{version}", self.settings.app_version)
        return {
            "question": question,
            "language": lang,
            "answer": answer,
            "answered": answer is not None,
            "matches": [m.to_dict() for m in matches],
            "kb_version": self.kb.version,
        }

    # ------------------------------------------------------------------- status
    def model_info(self) -> dict:
        from chatgangs.core.spacy_pipeline import get_spacy_pipeline

        return {
            "name": self.settings.app_name,
            "version": self.settings.app_version,
            "environment": self.settings.environment,
            "engines": {
                "tokenizer": {"backend": "regex-unicode", "languages": list(SUPPORTED_LANGUAGES)},
                "normalizer": {"backend": "dictionary+diff", "cache": self.normalizer.cache_stats()},
                "stemmer": self.stemmer.info(),
                "lemmatizer": self.stemmer.info(),
                "sentiment": self.sentiment.model_info(),
                "intent": self.intents.model_info(),
                "entities": {
                    "backends": ["regex", "gazetteer", "pattern"] + (
                        ["spacy"] if get_spacy_pipeline().available else []
                    ),
                    "spacy": get_spacy_pipeline().info(),
                },
                "semantics": {"backend": "tfidf-ngram", "ngram_range": [1, 2]},
                "classification": self.classifier.info(),
                "question_answering": {
                    "backend": "tfidf-retrieval+template-nlg",
                    "kb_version": self.kb.version,
                    "kb_entries": len(self.kb.entries),
                    "topics": self.kb.topics(),
                },
                "generation": {"backend": "slot-template-nlg", "languages": ["id", "en", "ms"]},
            },
            "languages": SUPPORTED_LANGUAGES,
            "limits": {
                "max_input_chars": self.settings.max_input_chars,
                "max_message_chars": self.settings.max_message_chars,
                "max_batch_size": self.settings.max_batch_size,
            },
            "training": {
                "pipeline": "scripts/train.py",
                "intent_model_trained": self.intents.model_available,
                "sentiment_model_trained": self.sentiment.model_available,
                "classification_model_trained": self.classifier.model_available,
            },
        }


def _guess_pos(word: str, language: str) -> str:
    """Lightweight POS heuristic used when spaCy is not available."""
    if language in ("id", "ms"):
        if word in {"saya", "aku", "kamu", "anda", "dia", "kita", "kami", "mereka", "ia"}:
            return "PRON"
        if word in {"dan", "atau", "tetapi", "tapi", "karena", "jika", "kalau"}:
            return "CCONJ"
        if word in {"di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "oleh"}:
            return "ADP"
        if word in {"tidak", "bukan", "belum", "jangan"}:
            return "PART"
        if word in {"sangat", "sekali", "lebih", "paling", "sudah", "telah", "akan", "sedang"}:
            return "ADV"
        if word.endswith(("kan", "nya", "lah", "kah")) or word.endswith("i"):
            return "VERB"
        return "NOUN"
    if word in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}:
        return "PRON"
    if word in {"the", "a", "an"}:
        return "DET"
    if word in {"and", "or", "but"}:
        return "CCONJ"
    if word in {"in", "on", "at", "for", "with", "from", "to", "of"}:
        return "ADP"
    if word in {"not", "no", "n't"}:
        return "PART"
    if word.endswith("ing"):
        return "VERB"
    if word.endswith("ly"):
        return "ADV"
    if word.endswith(("ed", "s")):
        return "VERB"
    return "NOUN"


TEMPLATE_INTENTS = {
    "greeting", "farewell", "gratitude", "identity", "capabilities",
    "api_help", "pricing", "privacy", "training_help", "translate",
    "feedback_positive", "complaint", "small_talk", "help",
}

KB_INTENTS = {
    "question", "capabilities", "api_help", "pricing", "privacy",
    "training_help", "identity", "help", "unknown",
}

DELETION_PATTERNS = re.compile(
    r"\b(hapus|delete|clear|bersihkan|forget|lupakan|reset)\b.{0,40}"
    r"\b(riwayat|histori|history|memori|memory|data|chat|percakapan|conversation|semua)\b",
    re.IGNORECASE,
)
DELETION_PATTERNS_EN = re.compile(r"\b(delete|forget|clear)\b.{0,20}\b(me|everything|my data)\b", re.IGNORECASE)


DETAIL_REQUEST_RE = re.compile(
    r"\b(jelaskan|detail|lengkap|rinci|lebih|contoh|explain|example|more|deeper|elaborate)\b",
    re.IGNORECASE,
)


def _wants_detail(text: str, language: str) -> bool:
    return bool(DETAIL_REQUEST_RE.search(text or ""))


def _is_deletion_request(text: str) -> bool:
    return bool(DELETION_PATTERNS.search(text) or DELETION_PATTERNS_EN.search(text))


DEFAULT_SUGGESTIONS: Dict[str, List[str]] = {
    "id": [
        "Apa saja kemampuanmu?",
        "Bagaimana cara memakai API?",
        "Analisis sentimen: pelayanannya bagus tapi lambat",
        "Bagaimana privasi data saya?",
    ],
    "en": [
        "What can you do?",
        "How do I use the API?",
        "Analyse the sentiment: the service is good but slow",
        "How is my data stored?",
    ],
    "ms": ["Apa yang boleh anda lakukan?", "Bagaimana cara guna API?"],
}

ENTITY_LABEL_LANG = {
    "id": {"PERSON": "nama", "ORGANIZATION": "organisasi", "GPE": "lokasi", "LOCATION": "lokasi",
            "DATE": "tanggal", "TIME": "waktu", "MONEY": "jumlah", "EMAIL": "email",
            "PHONE": "nomor", "PERCENT": "persentase", "PRODUCT": "produk", "QUANTITY": "kuantitas"},
    "en": {"PERSON": "person", "ORGANIZATION": "organisation", "GPE": "place", "LOCATION": "location",
           "DATE": "date", "TIME": "time", "MONEY": "amount", "EMAIL": "email",
           "PHONE": "phone number", "PERCENT": "percentage", "PRODUCT": "product", "QUANTITY": "quantity"},
}

_engine: Optional[NLPengine] = None


def get_engine() -> NLPengine:
    global _engine
    if _engine is None:
        _engine = NLPengine()
    return _engine
