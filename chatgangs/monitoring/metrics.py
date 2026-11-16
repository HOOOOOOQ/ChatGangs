"""Runtime request metrics.

The API middleware records every request here. Two things happen:

1. The sample is appended to an in-process ring buffer so ``GET /api/metrics``
   can report percentiles with zero database queries.
2. The sample is queued to a single background writer thread that persists
   *metadata only* (path, method, status, latency, size) into ``request_logs``.
   Persistence never blocks a request: the queue is bounded and drops samples
   when full, which is the correct trade-off for observability data.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from chatgangs.logging_config import get_logger

logger = get_logger(__name__)

_RING_SIZE = 2000
_QUEUE_SIZE = 10_000


@dataclass
class RequestSample:
    request_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    input_chars: int
    client: str
    timestamp: float


class MetricsRecorder:
    """Thread-safe ring buffer + background persistence."""

    def __init__(self, ring_size: int = _RING_SIZE) -> None:
        self._samples: Deque[RequestSample] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._queue: "queue.Queue[RequestSample]" = queue.Queue(maxsize=_QUEUE_SIZE)
        self._worker: threading.Thread | None = None
        self._started = False
        self.dropped = 0
        self.persist_enabled = True

    # ------------------------------------------------------------------ public
    def record(self, sample: RequestSample) -> None:
        with self._lock:
            self._samples.append(sample)
        if not self.persist_enabled:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(sample)
        except queue.Full:  # pragma: no cover - overload
            self.dropped += 1

    def recent(self, limit: int = 200) -> List[RequestSample]:
        with self._lock:
            items = list(self._samples)
        return items[-limit:][::-1]

    def summary(self, window: int = 200) -> Dict[str, object]:
        samples = self.recent(window)
        latencies = sorted(s.latency_ms for s in samples)
        errors = sum(1 for s in samples if s.status_code >= 400)
        server_errors = sum(1 for s in samples if s.status_code >= 500)
        return {
            "requests_recent": len(samples),
            "error_rate_recent": round(errors / len(samples), 4) if samples else 0.0,
            "server_error_rate_recent": round(server_errors / len(samples), 4) if samples else 0.0,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p90": _percentile(latencies, 0.90),
                "p99": _percentile(latencies, 0.99),
                "max": round(latencies[-1], 2) if latencies else 0.0,
            },
            "by_status": _status_buckets(samples),
            "writer_backlog": self._queue.qsize(),
            "dropped_samples": self.dropped,
        }

    def flush(self, timeout: float = 2.0) -> None:
        """Wait until the persistence queue drains (used by tests/shutdown)."""
        deadline = time.time() + timeout
        while self._queue.qsize() and time.time() < deadline:
            time.sleep(0.01)

    # --------------------------------------------------------------- internals
    def _ensure_worker(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
            self._worker = threading.Thread(
                target=self._writer_loop, name="chatgangs-metrics", daemon=True
            )
            self._worker.start()

    def _writer_loop(self) -> None:  # pragma: no cover - background thread
        while True:
            sample = self._queue.get()
            try:
                batch = [sample]
                while len(batch) < 50:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                from chatgangs.memory.manager import get_memory_manager

                manager = get_memory_manager()
                for item in batch:
                    manager.log_request(
                        item.request_id,
                        item.path,
                        method=item.method,
                        status_code=item.status_code,
                        latency_ms=item.latency_ms,
                        input_chars=item.input_chars,
                        client_hash=item.client,
                    )
            except Exception as exc:  # logging must never crash the worker
                logger.debug("metrics persistence skipped: %s", exc)
            finally:
                self._queue.task_done()


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * percentile), len(sorted_values) - 1)
    return round(float(sorted_values[index]), 2)


def _status_buckets(samples: List[RequestSample]) -> Dict[str, int]:
    buckets = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
    for sample in samples:
        bucket = f"{sample.status_code // 100}xx"
        if bucket in buckets:
            buckets[bucket] += 1
    return buckets


_recorder = MetricsRecorder()


def get_metrics_recorder() -> MetricsRecorder:
    return _recorder
