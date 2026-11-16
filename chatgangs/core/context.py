"""Context Manager — conversation state and follow-up resolution.

Responsibilities
----------------
* Hold a bounded, TTL-aware in-process view of each conversation.
* Detect *follow-up* utterances ("yang tadi", "itu", "how about that one",
  "jelaskan lagi") and resolve the pronoun/reference to the previous topic,
  intent or entity so the answer stays on-topic.
* Track the active topic, language, last intent, last entities, sentiment
  trend and the number of turns.
* Provide a compact *context digest* that is injected into the NLP pipeline
  (intent detection, QA retrieval) so a bare "dan harganya?" is interpreted
  together with the previous question.

The Context Manager is deliberately separate from the Memory Manager:
context is ephemeral operational state (fast, in-process), memory is the
persisted, user-controllable history.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chatgangs.config import get_settings
from chatgangs.core.languages import detect_language
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)

FOLLOWUP_PATTERNS = [
    # Indonesian
    r"\b(yang|yg) tadi\b", r"\b(itu|tersebut)\b", r"\bsama seperti tadi\b",
    r"\bjelaskan (lagi|lebih)\b", r"\blanjut\b", r"\bterus\b", r"\bkalau (itu|yang itu)\b",
    r"\bcontohnya\b", r"\blalu\b", r"^(dan|nah|jadi)\b",
    # English
    r"\bthe (previous|last) (one|answer|question)\b", r"\bwhat about (that|it|those)\b",
    r"\btell me more\b", r"\bgo on\b", r"\bexplain (that|it|again)\b",
    r"\band then\b", r"^(so|and)\b", r"\bthat one\b",
]
_FOLLOWUP_RE = re.compile("|".join(FOLLOWUP_PATTERNS), re.IGNORECASE)

PRONOUN_REFS = {
    "id": {"itu", "ini", "tersebut", "tadi", "nya", "dia", "mereka"},
    "en": {"it", "that", "this", "those", "they", "them", "he", "she"},
}


@dataclass
class Turn:
    role: str                     # "user" | "assistant"
    text: str
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[str] = None
    language: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "intent": self.intent,
            "entities": self.entities,
            "sentiment": self.sentiment,
            "language": self.language,
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class ConversationState:
    session_id: str
    language: str = "id"
    turns: List[Turn] = field(default_factory=list)
    topic: Optional[str] = None
    last_intent: Optional[str] = None
    last_entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    followups: int = 0

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "topic": self.topic,
            "last_intent": self.last_intent,
            "turn_count": self.turn_count,
            "followups": self.followups,
            "sentiment_history": self.sentiment_history[-10:],
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "turns": [t.to_dict() for t in self.turns],
        }


class ContextManager:
    """Bounded in-process conversation context store."""

    def __init__(self, max_sessions: int = 2000) -> None:
        self._sessions: "OrderedDict[str, ConversationState]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_sessions = max_sessions
        self.settings = get_settings()

    # ---------------------------------------------------------------- sessions
    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            state = self._sessions.get(session_id)
            now = time.time()
            ttl = self.settings.memory_ttl_hours * 3600
            if state is not None and now - state.updated_at > ttl:
                del self._sessions[session_id]
                state = None
            if state is None:
                state = ConversationState(session_id=session_id)
                self._sessions[session_id] = state
                self._evict_locked()
            self._sessions.move_to_end(session_id)
            return state

    def _evict_locked(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[dict]:
        with self._lock:
            return [state.to_dict() for state in self._sessions.values()]

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "max_sessions": self._max_sessions,
                "ttl_hours": self.settings.memory_ttl_hours,
            }

    # ------------------------------------------------------------ understanding
    def is_followup(self, text: str, state: ConversationState) -> bool:
        """True when the utterance is likely a continuation of a previous turn."""
        if state.turn_count == 0:
            return False
        tokens = (text or "").strip().split()
        if len(tokens) <= 6 and _FOLLOWUP_RE.search(text or ""):
            return True
        language = state.language if state.language in PRONOUN_REFS else "en"
        pronouns = PRONOUN_REFS[language]
        if len(tokens) <= 4 and any(t.lower().strip(".,!?") in pronouns for t in tokens):
            return True
        return False

    def enrich(
        self,
        text: str,
        state: ConversationState,
        language: str,
    ) -> Dict[str, Any]:
        """Build a context digest used by intent detection and QA retrieval."""
        followup = self.is_followup(text, state)
        digest: Dict[str, Any] = {
            "is_followup": followup,
            "topic": state.topic,
            "last_intent": state.last_intent,
            "turn_count": state.turn_count,
            "language": language,
        }
        if followup and state.turns:
            previous_user_turns = [t for t in state.turns if t.role == "user"]
            if previous_user_turns:
                digest["previous_user_text"] = previous_user_turns[-1].text
            if state.last_entities:
                digest["last_entities"] = state.last_entities
            digest["expanded_query"] = self._expand_query(text, state)
        return digest

    @staticmethod
    def _expand_query(text: str, state: ConversationState) -> str:
        """Append the previous question so retrieval has enough signal."""
        previous_user_turns = [t for t in state.turns if t.role == "user"]
        if not previous_user_turns:
            return text
        previous = previous_user_turns[-1].text
        return f"{previous} {text}".strip()

    # ---------------------------------------------------------------- recording
    def record_user_turn(
        self,
        state: ConversationState,
        text: str,
        *,
        intent: Optional[str] = None,
        entities: Optional[List[dict]] = None,
        sentiment: Optional[str] = None,
        language: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> None:
        delta = detect_language(text) if language is None else None
        resolved_language = language or (delta.language if delta else state.language)
        state.turns.append(
            Turn(
                role="user",
                text=text,
                intent=intent,
                entities=entities or [],
                sentiment=sentiment,
                language=resolved_language,
            )
        )
        state.language = resolved_language
        if intent and intent != "unknown":
            state.last_intent = intent
        if entities:
            state.last_entities = entities
        if sentiment:
            state.sentiment_history.append(sentiment)
            del state.sentiment_history[:-20]
        if topic:
            state.topic = topic
        elif intent and intent not in {"greeting", "farewell", "gratitude", "small_talk"}:
            state.topic = intent
        if self.is_followup(text, ConversationState(session_id=state.session_id, turns=state.turns[:-1])):
            state.followups += 1
        state.updated_at = time.time()
        self._trim(state)

    def record_assistant_turn(
        self,
        state: ConversationState,
        text: str,
        *,
        intent: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        state.turns.append(
            Turn(role="assistant", text=text, intent=intent, language=language or state.language)
        )
        state.updated_at = time.time()
        self._trim(state)

    def _trim(self, state: ConversationState) -> None:
        limit = max(self.settings.memory_max_turns, 1) * 2
        if len(state.turns) > limit:
            del state.turns[:-limit]

    # ------------------------------------------------------------- sentiment
    @staticmethod
    def sentiment_trend(state: ConversationState) -> Optional[str]:
        history = state.sentiment_history[-3:]
        if len(history) < 2:
            return None
        score = {"positive": 1, "neutral": 0, "negative": -1}
        if all(score[h] > 0 for h in history):
            return "consistently_positive"
        if all(score[h] < 0 for h in history):
            return "consistently_negative"
        if score[history[0]] < 0 < score[history[-1]]:
            return "improving"
        if score[history[0]] > 0 > score[history[-1]]:
            return "deteriorating"
        return "mixed"


_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
