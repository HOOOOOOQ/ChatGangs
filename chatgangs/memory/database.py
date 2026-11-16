"""Persistence layer for ChatGangs.

SQLite by default (``CHATGANGS_DATABASE_URL=sqlite:///./data/chatgangs.db``),
PostgreSQL in deployment (``postgresql+psycopg2://user:pass@host/db``).
The schema is created automatically at start-up.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from chatgangs.config import get_settings
from chatgangs.logging_config import get_logger
from chatgangs.memory.models import Base

logger = get_logger(__name__)

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None
# Guards lazy initialisation: several request threads may hit the API at the
# same moment before the lifespan warm-up finished (or in embedded use).
_engine_lock = threading.RLock()


def _ensure_sqlite_directory(url: str) -> None:
    if not url.startswith("sqlite"):
        return
    # sqlite:///./data/chatgangs.db → ./data/chatgangs.db
    path_part = url.split("sqlite:///")[-1]
    if path_part and path_part != ":memory:":
        path = Path(path_part)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:  # another thread won the race
            return _engine
        _engine, _SessionFactory = _create_engine()
        return _engine


def _create_engine():
    settings = get_settings()
    url = settings.database_url
    _ensure_sqlite_directory(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    # ``checkfirst`` plus the lock makes concurrent first-requests safe.
    Base.metadata.create_all(engine, checkfirst=True)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    logger.info("Database ready (%s)", url.split("@")[-1])
    return engine, factory


def get_session_factory() -> sessionmaker:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope around a series of operations."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_database() -> None:
    """Drop and recreate all tables (used by tests)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
