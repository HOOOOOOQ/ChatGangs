"""Memory Manager — persisted, privacy-aware conversation history.

The manager is the *only* component that writes chat content to the database.
It honours the privacy switches in :class:`chatgangs.config.Settings`:

* ``memory_enabled``       – turn persistence on/off globally.
* ``memory_store_user_profile`` – whether non-sensitive preferences may be kept.
* ``memory_ttl_hours``     – expiry window; expired sessions are purged lazily.
* ``log_chat_content``     – content is never logged, only metadata.

Deletion is a first-class operation: :meth:`MemoryManager.delete_session`
removes the session row *and* cascades to its messages, and
:meth:`MemoryManager.purge_expired` enforces the TTL.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from sqlalchemy import delete, func, select

from chatgangs.config import get_settings
from chatgangs.logging_config import get_logger
from chatgangs.memory.database import session_scope
from chatgangs.memory.models import ConversationSession, Feedback, Message, RequestLog

logger = get_logger(__name__)


class MemoryManager:
    """CRUD facade over the conversation tables."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- sessions
    def ensure_session(
        self,
        session_id: str,
        *,
        language: str = "id",
        title: str = "",
        user_profile: Optional[dict] = None,
        consent: bool = True,
    ) -> Optional[dict]:
        if not self.settings.memory_persist or not consent:
            return None
        # Profiles are optional and never stored when the operator disables it.
        if not self.settings.memory_store_user_profile:
            user_profile = None
        ttl = self.settings.memory_ttl_hours * 3600
        with session_scope() as session:
            row = session.scalar(
                select(ConversationSession).where(ConversationSession.session_id == session_id)
            )
            if row is None:
                row = ConversationSession(
                    session_id=session_id,
                    language=language,
                    title=title[:200],
                    user_profile=user_profile or {},
                    consent=consent,
                    expires_at=time.time() + ttl if ttl > 0 else 0.0,
                )
                session.add(row)
            else:
                row.updated_at = time.time()
                if language:
                    row.language = language
                if user_profile and self.settings.memory_store_user_profile:
                    row.user_profile = {**(row.user_profile or {}), **user_profile}
                if ttl > 0:
                    row.expires_at = time.time() + ttl
            session.flush()
            return row.to_dict()

    def get_session(self, session_id: str) -> Optional[dict]:
        with session_scope() as session:
            row = session.scalar(
                select(ConversationSession).where(ConversationSession.session_id == session_id)
            )
            return row.to_dict() if row else None

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[dict]:
        with session_scope() as session:
            rows = session.scalars(
                select(ConversationSession)
                .order_by(ConversationSession.updated_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return [row.to_dict() for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and every message that belongs to it."""
        with session_scope() as session:
            row = session.scalar(
                select(ConversationSession).where(ConversationSession.session_id == session_id)
            )
            if row is None:
                # Messages may exist without a session row if persistence of the
                # session row was disabled; clean them up anyway.
                session.execute(delete(Message).where(Message.session_id == session_id))
                return False
            session.delete(row)
            return True

    # -------------------------------------------------------------- messages
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        intent: str = "",
        language: str = "",
        sentiment: str = "",
        metadata: Optional[dict] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> Optional[int]:
        if not self.settings.memory_persist or not self.settings.memory_enabled:
            return None
        with session_scope() as session:
            row = session.scalar(
                select(ConversationSession).where(ConversationSession.session_id == session_id)
            )
            if row is None:
                ttl = self.settings.memory_ttl_hours * 3600
                row = ConversationSession(
                    session_id=session_id,
                    language=language or self.settings.default_language,
                    expires_at=time.time() + ttl if ttl > 0 else 0.0,
                )
                session.add(row)
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                language=language,
                sentiment=sentiment,
                metadata_json=metadata or {},
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
            session.add(message)
            row.message_count = (row.message_count or 0) + 1
            row.updated_at = time.time()
            session.flush()
            return message.id

    def get_messages(self, session_id: str, limit: int = 50) -> List[dict]:
        with session_scope() as session:
            rows = session.scalars(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.desc())
                .limit(limit)
            ).all()
            return [row.to_dict() for row in reversed(rows)]

    def message_count(self, session_id: str) -> int:
        with session_scope() as session:
            return int(
                session.scalar(
                    select(func.count(Message.id)).where(Message.session_id == session_id)
                )
                or 0
            )

    # ------------------------------------------------------------- retention
    def purge_expired(self) -> int:
        now = time.time()
        removed = 0
        with session_scope() as session:
            rows = session.scalars(
                select(ConversationSession).where(
                    ConversationSession.expires_at > 0,
                    ConversationSession.expires_at < now,
                )
            ).all()
            for row in rows:
                session.delete(row)
                removed += 1
        if removed:
            logger.info("Purged %d expired conversation sessions", removed)
        return removed

    def delete_all(self) -> int:
        """Delete every conversation (GDPR-style erase). Returns row count."""
        with session_scope() as session:
            count = int(session.scalar(select(func.count(ConversationSession.id))) or 0)
            session.execute(delete(Message))
            session.execute(delete(ConversationSession))
        return count

    # ---------------------------------------------------------------- stats
    def stats(self) -> Dict[str, int]:
        with session_scope() as session:
            sessions = int(session.scalar(select(func.count(ConversationSession.id))) or 0)
            messages = int(session.scalar(select(func.count(Message.id))) or 0)
        return {"sessions": sessions, "messages": messages}

    # ------------------------------------------------------------- audit log
    def log_request(
        self,
        request_id: str,
        endpoint: str,
        *,
        method: str = "POST",
        status_code: int = 200,
        latency_ms: float = 0.0,
        input_chars: int = 0,
        client_hash: str = "",
    ) -> None:
        """Persist request metadata only (no message content)."""
        try:
            with session_scope() as session:
                session.add(
                    RequestLog(
                        request_id=request_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        input_chars=input_chars,
                        client_hash=client_hash,
                    )
                )
        except Exception as exc:  # pragma: no cover - logging must never break requests
            logger.debug("Could not persist request log: %s", exc)

    def recent_requests(self, limit: int = 50) -> List[dict]:
        with session_scope() as session:
            rows = session.scalars(
                select(RequestLog).order_by(RequestLog.id.desc()).limit(limit)
            ).all()
            return [
                {
                    "request_id": row.request_id,
                    "endpoint": row.endpoint,
                    "method": row.method,
                    "status_code": row.status_code,
                    "latency_ms": round(row.latency_ms, 2),
                    "input_chars": row.input_chars,
                    "created_at": row.created_at,
                }
                for row in rows
            ]

    def add_feedback(self, session_id: str, rating: int, comment: str = "", message_id: int = 0) -> int:
        with session_scope() as session:
            row = Feedback(
                session_id=session_id,
                rating=rating,
                comment=comment[:2000],
                message_id=message_id,
            )
            session.add(row)
            session.flush()
            return row.id


_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        with threading.Lock():
            if _manager is None:
                _manager = MemoryManager()
    return _manager
