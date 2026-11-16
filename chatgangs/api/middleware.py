"""HTTP middleware: request IDs, safe structured access logs, security headers."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from chatgangs.api.security import client_identity, new_request_id
from chatgangs.config import get_settings
from chatgangs.logging_config import get_logger
from chatgangs.monitoring.metrics import RequestSample, get_metrics_recorder

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, measure latency and log metadata (never content)."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        request.state.request_id = request_id
        request.state.started_at = time.perf_counter()
        try:
            request.state.client = client_identity(request)
        except Exception:  # pragma: no cover
            request.state.client = "unknown"

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - request.state.started_at) * 1000
            logger.exception(
                "Unhandled error on %s %s",
                request.method,
                request.url.path,
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status": 500,
                    "latency_ms": round(latency_ms, 2),
                    "client": request.state.client,
                },
            )
            raise

        latency_ms = (time.perf_counter() - request.state.started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")

        # Record metadata for /api/metrics (never the body).
        try:
            content_length = request.headers.get("content-length", "0")
            get_metrics_recorder().record(
                RequestSample(
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    latency_ms=round(latency_ms, 2),
                    input_chars=int(content_length or 0) if str(content_length).isdigit() else 0,
                    client=getattr(request.state, "client", "unknown"),
                    timestamp=time.time(),
                )
            )
        except Exception:  # pragma: no cover - metrics must never break a request
            pass

        # Log only metadata. Content length and endpoint are safe.
        content_length = request.headers.get("content-length", "0")
        logger.info(
            "%s %s -> %s (%.1f ms, %s bytes)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            content_length,
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "client": request.state.client,
            },
        )
        return response
