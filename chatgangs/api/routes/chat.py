"""Conversational endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from chatgangs.api.deps import get_engine_dep
from chatgangs.api.schemas import ChatRequest, FeedbackRequest
from chatgangs.api.security import enforce_chat_budget
from chatgangs.core.pipeline import NLPengine
from chatgangs.memory.manager import get_memory_manager

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat", summary="Send a message to ChatGangs")
def chat(
    payload: ChatRequest,
    request: Request,
    engine: NLPengine = Depends(get_engine_dep),
) -> Dict[str, Any]:
    """Context-aware conversation turn.

    The response contains the generated reply plus the full understanding of
    the message (language, intent, sentiment, entities), the context digest
    used to resolve follow-ups and the memory status of the turn.
    """
    enforce_chat_budget(getattr(request.state, "client", "unknown"))
    try:
        result = engine.chat(
            payload.message,
            session_id=payload.session_id,
            language=payload.language,
            store_memory=payload.store_memory,
            user_profile=payload.user_profile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.to_dict()


@router.post("/feedback", summary="Submit feedback for a reply")
def feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    """Store a rating (-1/0/1..5) for model evaluation. Never used for training without review."""
    memory = get_memory_manager()
    feedback_id = memory.add_feedback(
        payload.session_id, payload.rating, payload.comment, payload.message_id
    )
    return {"status": "recorded", "feedback_id": feedback_id}
