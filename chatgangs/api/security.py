"""API security: key authentication, rate limiting and request identification.

* **Authentication** is opt-in. Set ``CHATGANGS_API_KEYS`` to a comma-separated
  list and ``CHATGANGS_REQUIRE_API_KEY=true`` to enforce the ``X-API-Key``
  header. Keys are compared in constant time and never logged.
* **Rate limiting** is a sliding-window counter per client (keyed by API key
  hash when authenticated, otherwise by IP). Limits are configurable.
* **Request identification** gives every request a UUID that is echoed in the
  ``X-Request-ID`` response header and used in structured logs.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status

from chatgangs.config import get_settings
from chatgangs.logging_config import get_logger

logger = get_logger(__name__)


def hash_client(value: str) -> str:
    """Stable, non-reversible client identifier for logs and rate limits."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _api_keys() -> Tuple[str, ...]:
    return tuple(key for key in get_settings().api_keys if key)


def verify_api_key(provided: Optional[str]) -> bool:
    """Constant-time API-key check. Returns True when auth is disabled."""
    settings = get_settings()
    keys = _api_keys()
    if not keys or not settings.require_api_key:
        return True
    if not provided:
        return False
    return any(hmac.compare_digest(provided, key) for key in keys)


_chat_budgets: Dict[str, Deque[float]] = {}
_chat_budget_lock = threading.Lock()


def enforce_chat_budget(client: str) -> None:
    """Per-client budget for the (heavier) conversational endpoint.

    ``CHATGANGS_BOT_REQUESTS_PER_MINUTE`` protects the CPU-bound chat path in
    addition to the global rate limit. Raises ``HTTPException(429)``.
    """
    limit = get_settings().bot_requests_per_minute
    if limit <= 0:
        return
    now = time.time()
    window = 60
    with _chat_budget_lock:
        bucket = _chat_budgets.setdefault(client, deque())
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(int(window - (now - bucket[0])) + 1, 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Chat budget exceeded. Please retry shortly.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        bucket.append(now)
        if len(_chat_budgets) > 50000:  # keep memory bounded
            for key in list(_chat_budgets)[:10000]:
                if not _chat_budgets[key]:
                    _chat_budgets.pop(key, None)


def chat_budget_stats() -> dict:
    with _chat_budget_lock:
        return {
            "tracked_clients": len(_chat_budgets),
            "per_minute": get_settings().bot_requests_per_minute,
        }


def client_identity(request: Request) -> str:
    """Return a privacy-preserving client identifier."""
    settings = get_settings()
    provided = request.headers.get(settings.api_key_header)
    if provided and _api_keys():
        return "key:" + hash_client(provided)
    forwarded = request.headers.get("x-forwarded-for") if settings.trust_forwarded_for else None
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    elif request.client:
        ip = request.client.host
    else:  # pragma: no cover - test client
        ip = "unknown"
    return "ip:" + hash_client(ip)


@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after: float


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, limit: int, window_seconds: int, max_clients: int = 50000) -> None:
        self.limit = max(limit, 1)
        self.window = max(window_seconds, 1)
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._max_clients = max_clients

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
                if len(self._hits) > self._max_clients:
                    self._evict(cutoff)
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                reset_after = max(0.0, self.window - (now - bucket[0]))
                return RateLimitDecision(False, self.limit, 0, reset_after)
            bucket.append(now)
            return RateLimitDecision(
                True, self.limit, self.limit - len(bucket), float(self.window)
            )

    def _evict(self, cutoff: float) -> None:
        stale = [key for key, bucket in self._hits.items() if not bucket or bucket[-1] < cutoff]
        for key in stale[: len(self._hits) // 2 or 1]:
            self._hits.pop(key, None)

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_clients": len(self._hits),
                "limit": self.limit,
                "window_seconds": self.window,
            }


_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        settings = get_settings()
        _limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
    return _limiter


def new_request_id() -> str:
    return uuid.uuid4().hex


def enforce_rate_limit(request: Request) -> RateLimitDecision:
    decision = get_rate_limiter().check(client_identity(request))
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Rate limit exceeded. "
                f"Try again in {decision.reset_after:.0f} seconds."
            ),
            headers={
                "Retry-After": str(int(decision.reset_after) + 1),
                "X-RateLimit-Limit": str(decision.limit),
                "X-RateLimit-Remaining": "0",
            },
        )
    return decision
