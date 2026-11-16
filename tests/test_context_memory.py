"""Context manager and memory manager tests."""

import uuid

import pytest

from chatgangs.core.context import ContextManager, get_context_manager
from chatgangs.memory.database import reset_database
from chatgangs.memory.manager import MemoryManager, get_memory_manager


def unique_id():
    return "test-" + uuid.uuid4().hex[:12]


# --------------------------------------------------------------- context
def test_context_creation_and_trimming():
    manager = ContextManager(max_sessions=2)
    state = manager.get("s1")
    assert state.session_id == "s1"
    assert manager.get("s1") is state
    manager.get("s2")
    manager.get("s3")  # evicts the least-recently used session
    assert len(manager.list_sessions()) <= 2


def test_followup_detection():
    manager = ContextManager()
    state = manager.get("followup")
    assert manager.is_followup("jelaskan lagi", state) is False
    manager.record_user_turn(state, "apa itu NLP dan bagaimana cara kerjanya", language="id")
    assert manager.is_followup("jelaskan lagi", state) is True
    assert manager.is_followup("yang tadi itu apa", state) is True
    assert manager.is_followup("berapa harga mobil listrik", state) is False


def test_enrich_expands_query_for_followups():
    manager = ContextManager()
    state = manager.get("enrich")
    manager.record_user_turn(state, "bagaimana cara memakai api", language="id", intent="api_help")
    digest = manager.enrich("jelaskan lebih detail", state, "id")
    assert digest["is_followup"] is True
    assert "bagaimana cara memakai api" in digest["expanded_query"]


def test_sentiment_trend():
    manager = ContextManager()
    state = manager.get("trend")
    state.sentiment_history = ["negative", "neutral", "positive"]
    assert manager.sentiment_trend(state) == "improving"
    state.sentiment_history = ["positive", "neutral", "negative"]
    assert manager.sentiment_trend(state) == "deteriorating"


def test_singleton_context_manager():
    assert get_context_manager() is get_context_manager()


# ---------------------------------------------------------------- memory
@pytest.fixture()
def memory():
    reset_database()
    return MemoryManager()


def test_session_and_message_roundtrip(memory):
    session_id = unique_id()
    memory.ensure_session(session_id, language="id")
    first = memory.add_message(session_id, "user", "halo")
    second = memory.add_message(session_id, "assistant", "Hai!")
    assert first and second
    messages = memory.get_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert memory.message_count(session_id) == 2


def test_delete_session_removes_messages(memory):
    session_id = unique_id()
    memory.ensure_session(session_id)
    memory.add_message(session_id, "user", "hapus saya")
    assert memory.delete_session(session_id) is True
    assert memory.get_messages(session_id) == []
    assert memory.get_session(session_id) is None


def test_stats_and_delete_all(memory):
    ids = [unique_id() for _ in range(3)]
    for session_id in ids:
        memory.ensure_session(session_id)
        memory.add_message(session_id, "user", "pesan")
    stats = memory.stats()
    assert stats["sessions"] >= 3
    memory.delete_all()
    assert memory.stats()["sessions"] == 0


def test_feedback_and_request_logs(memory):
    session_id = unique_id()
    feedback_id = memory.add_feedback(session_id, rating=1, comment="bagus")
    assert feedback_id > 0
    memory.log_request("req-1", "/api/chat", latency_ms=12.5, input_chars=10)
    logs = memory.recent_requests(1)
    assert logs[0]["request_id"] == "req-1"
    assert logs[0]["latency_ms"] == pytest.approx(12.5)
    # request logs must never contain message content
    assert "content" not in logs[0]


def test_purge_expired(memory):
    session_id = unique_id()
    memory.ensure_session(session_id)
    from chatgangs.memory.database import session_scope
    from chatgangs.memory.models import ConversationSession

    with session_scope() as session:
        from sqlalchemy import select

        row = session.scalar(
            select(ConversationSession).where(ConversationSession.session_id == session_id)
        )
        row.expires_at = 1.0  # long in the past
    purged = memory.purge_expired()
    assert purged >= 1


def test_memory_never_stores_when_disabled(monkeypatch):
    from chatgangs.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "memory_persist", False)
    manager = MemoryManager()
    assert manager.add_message(unique_id(), "user", "tidak tersimpan") is None
    monkeypatch.undo()


def test_singleton_memory_manager():
    assert get_memory_manager() is get_memory_manager()
