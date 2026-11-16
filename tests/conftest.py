"""Shared pytest fixtures.

Environment variables are set **before** any ChatGangs module imports the
cached settings object, so tests always run against an isolated SQLite file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="chatgangs-tests-"))

os.environ.update(
    {
        "CHATGANGS_ENVIRONMENT": "test",
        "CHATGANGS_DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "CHATGANGS_MEMORY_PERSIST": "true",
        "CHATGANGS_RATE_LIMIT_REQUESTS": "10000",
        "CHATGANGS_LOG_LEVEL": "WARNING",
        "CHATGANGS_ENABLE_SPACY": "false",
        "CHATGANGS_REQUIRE_API_KEY": "false",
        "CHATGANGS_API_KEYS": "",
    }
)


@pytest.fixture(scope="session")
def app_client():
    from fastapi.testclient import TestClient

    from chatgangs.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def client(app_client):
    return app_client


@pytest.fixture(scope="session")
def engine():
    from chatgangs.core.pipeline import get_engine

    return get_engine()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    yield
    try:
        from chatgangs.api.security import get_rate_limiter

        get_rate_limiter().reset()
    except Exception:
        pass


@pytest.fixture()
def clean_db():
    from chatgangs.memory.database import reset_database

    reset_database()
    yield
