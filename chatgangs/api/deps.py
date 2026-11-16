"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Request, status

from chatgangs.api.security import enforce_rate_limit, verify_api_key
from chatgangs.config import get_settings


def get_engine_dep():
    """Return the singleton NLP engine (imported lazily to keep start-up fast)."""
    from chatgangs.core.pipeline import get_engine

    return get_engine()


async def security_dependency(request: Request) -> None:
    """Apply rate limiting and optional API-key authentication to a route."""
    settings = get_settings()
    key = request.headers.get(settings.api_key_header)
    if not verify_api_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or missing API key (header: {settings.api_key_header})",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    enforce_rate_limit(request)
