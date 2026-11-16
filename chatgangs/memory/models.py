"""SQLAlchemy ORM models for conversation storage."""

from __future__ import annotations

import time

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> float:
    return time.time()


class Base(DeclarativeBase):
    pass


class ConversationSession(Base):
    """One chat session. Rows can be deleted at any time by the user."""

    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    language: Mapped[str] = mapped_column(String(8), default="id")
    title: Mapped[str] = mapped_column(String(200), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    user_profile: Mapped[dict] = mapped_column(JSON, default=dict)
    consent: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    updated_at: Mapped[float] = mapped_column(Float, default=_now, onupdate=_now)
    expires_at: Mapped[float] = mapped_column(Float, default=0.0)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.id"
    )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "title": self.title,
            "summary": self.summary,
            "message_count": self.message_count,
            "user_profile": self.user_profile or {},
            "consent": bool(self.consent),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


class Message(Base):
    """A single message (user or assistant) in a session."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("conversation_sessions.session_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(64), default="")
    language: Mapped[str] = mapped_column(String(8), default="")
    sentiment: Mapped[str] = mapped_column(String(16), default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[float] = mapped_column(Float, default=_now, index=True)

    session: Mapped[ConversationSession] = relationship(back_populates="messages")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "language": self.language,
            "sentiment": self.sentiment,
            "metadata": self.metadata_json or {},
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 2),
            "created_at": self.created_at,
        }


class RequestLog(Base):
    """Operational request log (metadata only — never message content)."""

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    endpoint: Mapped[str] = mapped_column(String(120))
    method: Mapped[str] = mapped_column(String(8), default="POST")
    status_code: Mapped[int] = mapped_column(Integer, default=200)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_chars: Mapped[int] = mapped_column(Integer, default=0)
    client_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now, index=True)


class Feedback(Base):
    """Optional user feedback on responses (used for future model evaluation)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    message_id: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[int] = mapped_column(Integer, default=0)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=_now)
