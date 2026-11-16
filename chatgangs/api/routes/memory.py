"""Memory & privacy endpoints — users own their conversation data."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from chatgangs.core.context import get_context_manager
from chatgangs.memory.manager import get_memory_manager

router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.get("", summary="List stored sessions")
def list_sessions(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    memory = get_memory_manager()
    return {
        "sessions": memory.list_sessions(limit=limit, offset=offset),
        "stats": memory.stats(),
        "context": get_context_manager().stats(),
    }


@router.get("/{session_id}", summary="Get a session with its messages")
def get_session(session_id: str, limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    memory = get_memory_manager()
    session = memory.get_session(session_id)
    messages = memory.get_messages(session_id, limit=limit)
    if session is None and not messages:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found. It may have expired, been deleted, "
                "or memory persistence may be disabled."
            ),
        )
    # The in-memory context is *not* created as a side effect of reading it:
    # after a deletion we must not resurrect a phantom conversation.
    return {
        "session": session,
        "messages": messages,
        "context": _peek_context(session_id),
    }


def _peek_context(session_id: str):
    from chatgangs.core.context import get_context_manager

    manager = get_context_manager()
    with manager._lock:  # read-only peek, no side effect
        state = manager._sessions.get(session_id)
        return state.to_dict() if state else None


@router.delete("/{session_id}", summary="Delete a session and all its messages")
def delete_session(session_id: str) -> Dict[str, Any]:
    """Erase persisted history *and* the in-memory context for a session."""
    memory = get_memory_manager()
    deleted = memory.delete_session(session_id)
    context_reset = get_context_manager().reset(session_id)
    return {
        "status": "deleted",
        "session_id": session_id,
        "persisted_deleted": deleted,
        "context_reset": context_reset,
    }


@router.delete("", summary="Purge expired sessions")
def purge_expired() -> Dict[str, Any]:
    removed = get_memory_manager().purge_expired()
    return {"status": "ok", "purged_sessions": removed}
