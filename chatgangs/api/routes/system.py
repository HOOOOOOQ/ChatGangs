"""Operational endpoints: health, model information, metrics, batch processing."""

from __future__ import annotations

import platform
import sys
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from chatgangs.api.deps import get_engine_dep
from chatgangs.api.schemas import BatchRequest
from chatgangs.config import get_settings
from chatgangs.core.pipeline import NLPengine
from chatgangs.core.semantics import get_knowledge_base
from chatgangs.core.spacy_pipeline import get_spacy_pipeline
from chatgangs.memory.database import get_engine
from chatgangs.memory.manager import get_memory_manager

router = APIRouter(tags=["System"])

STARTED_AT = time.time()


@router.get("/api/health", summary="Health check")
def health(engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Liveness/readiness probe with dependency checks."""
    settings = get_settings()
    checks: Dict[str, Any] = {}

    try:
        from sqlalchemy import text

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok",
                              "backend": settings.database_url.split(":")[0]}
    except Exception as exc:  # pragma: no cover - environment dependent
        checks["database"] = {"status": "error", "detail": str(exc)}

    checks["nlp_engine"] = {
        "status": "ok",
        "intents": len(engine.intents.intents),
        "kb_entries": len(engine.kb.entries),
        "spacy": get_spacy_pipeline().available,
    }
    checks["memory"] = {
        "status": "ok" if settings.memory_enabled else "disabled",
        **get_memory_manager().stats(),
        "context": engine.context.stats(),
    }
    healthy = all(
        check.get("status") in ("ok", "disabled") for check in checks.values()
    )
    return {
        "status": "ok" if healthy else "degraded",
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "checks": checks,
    }


@router.get("/api/model/info", summary="Model and engine information")
def model_info(engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Describe every engine, its backend and whether trained artefacts exist.

    The response is deliberately explicit about what is *not* available (for
    example an untrained model or a missing spaCy download) so clients never
    over-trust the system.
    """
    settings = get_settings()
    info = engine.model_info()
    info["runtime"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "database": settings.database_url.split("@")[-1],
    }
    info["artifacts"] = _artifact_summary()
    return info


@router.get("/api/metrics", summary="Runtime metrics snapshot")
def metrics(engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Lightweight, dependency-free metrics suitable for scraping or dashboards."""
    from chatgangs.api.security import chat_budget_stats, get_rate_limiter

    from chatgangs.monitoring.metrics import get_metrics_recorder

    memory = get_memory_manager()
    stats = memory.stats()
    summary = get_metrics_recorder().summary(window=200)

    return {
        **summary,
        "memory": stats,
        "context_sessions": engine.context.stats()["sessions"],
        "rate_limiter": get_rate_limiter().stats(),
        "chat_budget": chat_budget_stats(),
        "caches": {
            "normalizer": engine.normalizer.cache_stats(),
            "analysis": engine.cache_stats(),
            "spacy_available": get_spacy_pipeline().available,
        },
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }


@router.post("/api/nlp/batch", summary="Batch analysis")
def batch(payload: BatchRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Analyse up to 64 texts in one request (bounded for stability)."""
    results = []
    for text in payload.texts:
        item: Dict[str, Any] = {"text": text}
        analysis = None
        needs_full = any(task in payload.tasks for task in ("language", "sentiment", "intent"))
        if needs_full:
            analysis = engine.analyze(
                text, explain=False, include_tokens="tokenize" in payload.tasks
            )
        if "language" in payload.tasks and analysis:
            item["language"] = analysis.language
        if "sentiment" in payload.tasks and analysis:
            item["sentiment"] = analysis.sentiment
        if "intent" in payload.tasks and analysis:
            item["intent"] = analysis.intent
        if "entities" in payload.tasks:
            item["entities"] = engine.extract_entities(text)["entities"]
        if "tokenize" in payload.tasks and analysis:
            item["tokens"] = analysis.tokens
        results.append(item)
    return {"count": len(results), "tasks": payload.tasks, "results": results}


@router.get("/api/languages", summary="Supported languages")
def languages() -> Dict[str, Any]:
    from chatgangs.core.languages import PRIORITY_LANGUAGES, SUPPORTED_LANGUAGES

    return {
        "priority": list(PRIORITY_LANGUAGES),
        "languages": [
            {"code": code, **meta, "priority": code in PRIORITY_LANGUAGES}
            for code, meta in SUPPORTED_LANGUAGES.items()
        ],
    }


@router.get("/api/intents", summary="Intent catalogue")
def intents(engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    return {
        "count": len(engine.intents.intents),
        "intents": [
            {
                "name": intent["name"],
                "description": intent.get("description"),
                "examples": intent.get("examples", [])[:5],
                "example_count": len(intent.get("examples", [])),
            }
            for intent in engine.intents.intents
        ],
    }


@router.get("/api/knowledge-base", summary="Knowledge base topics")
def knowledge_base(engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    kb = get_knowledge_base()
    return {
        "version": kb.version,
        "entries": len(kb.entries),
        "topics": kb.topics(),
        "questions": [entry.get("question") for entry in kb.entries],
    }


def _artifact_summary() -> Dict[str, Any]:
    from chatgangs.config import get_settings as _settings

    models_dir = _settings().models_dir
    summary: Dict[str, Any] = {}
    for task in ("intent", "sentiment", "classification"):
        metrics_file = models_dir / task / "metrics.json"
        summary[task] = {
            "trained": (models_dir / task / "model.joblib").exists(),
            "metrics_available": metrics_file.exists(),
        }
        if metrics_file.exists():
            import json

            try:
                payload = json.loads(metrics_file.read_text(encoding="utf-8"))
                summary[task]["version"] = payload.get("version")
                summary[task]["trained_at"] = payload.get("trained_at")
                summary[task]["validation_accuracy"] = payload.get("metrics", {}).get(
                    "validation_accuracy"
                )
            except Exception:  # pragma: no cover
                pass
    return summary
