"""FastAPI application factory for ChatGangs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from chatgangs.api.deps import security_dependency
from chatgangs.api.middleware import RequestContextMiddleware
from chatgangs.api.routes import chat as chat_routes
from chatgangs.api.routes import memory as memory_routes
from chatgangs.api.routes import nlp as nlp_routes
from chatgangs.api.routes import system as system_routes
from chatgangs.config import PROJECT_ROOT, get_settings
from chatgangs.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

WEB_DIR = PROJECT_ROOT / "web"

DESCRIPTION = """
**ChatGangs** is a modular NLP / conversational-AI engine.

Every capability listed below is implemented in this repository and runs on
real models or deterministic algorithms — no static answer tables:

* tokenisation, normalisation, stemming, lemmatisation, stopword removal
* named entity recognition, intent detection, sentiment analysis
* semantic similarity, text classification, language detection (8 languages)
* context-aware conversation with persistent, user-deletable memory
* retrieval-based question answering + template NLG
* a real training pipeline (`scripts/train.py`) with cross-validation

Interactive docs: `/docs` · ReDoc: `/redoc` · Health: `/api/health`
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up heavy components and prepare the database on start-up."""
    settings = get_settings()
    configure_logging()
    logger.info(
        "Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment
    )
    from chatgangs.memory.database import get_engine

    get_engine()  # creates tables / verifies the connection
    from chatgangs.core.pipeline import get_engine as get_nlp_engine

    engine = get_nlp_engine()
    # Warm the deterministic caches so the first user request is not slow.
    try:
        from chatgangs.core.entities import get_entity_recognizer
        from chatgangs.core.languages import detect_language
        from chatgangs.core.spacy_pipeline import get_spacy_pipeline

        get_spacy_pipeline().load()
        get_entity_recognizer().extract("Warm up: hubungi budi@mail.com di Jakarta.", "id")
        detect_language("Halo, selamat datang di ChatGangs.")
    except Exception as exc:  # pragma: no cover - warm-up must never block start-up
        logger.warning("Warm-up skipped: %s", exc)
    logger.info(
        "NLP engines ready: %d intents, %d knowledge-base entries, "
        "intent-model=%s, sentiment-model=%s",
        len(engine.intents.intents),
        len(engine.kb.entries),
        engine.intents.model_available,
        engine.sentiment.model_available,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
        contact={"name": "ChatGangs", "url": "https://github.com/HOOOOOOQ/ChatGangs"},
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "Chat", "description": "Conversational endpoint with context & memory"},
            {"name": "NLP", "description": "Individual NLP modules"},
            {"name": "Memory", "description": "Conversation history and privacy controls"},
            {"name": "System", "description": "Health, models, metrics and catalogues"},
        ],
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms", "X-RateLimit-Remaining"],
    )

    # Routers ------------------------------------------------------------
    app.include_router(system_routes.router)
    app.include_router(
        chat_routes.router, dependencies=[Depends(security_dependency)]
    )
    app.include_router(
        nlp_routes.router, dependencies=[Depends(security_dependency)]
    )
    app.include_router(
        memory_routes.router, dependencies=[Depends(security_dependency)]
    )

    # Error handling -----------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", []))
            errors.append({"field": location, "message": error.get("msg"), "type": error.get("type")})
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "request_id": getattr(request.state, "request_id", None),
                "details": errors,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):  # pragma: no cover
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": str(exc) if not settings.is_production else "An internal error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Static web client --------------------------------------------------
    if WEB_DIR.exists():
        app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(str(WEB_DIR / "index.html"))

    return app


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Attach the optional API-key security scheme to the OpenAPI document."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": get_settings().api_key_header,
        "description": "Optional. Required only when CHATGANGS_REQUIRE_API_KEY=true.",
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app = create_app()
app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]
