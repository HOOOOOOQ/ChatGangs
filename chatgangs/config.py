"""Central configuration for ChatGangs.

Every tunable value is loaded from environment variables (or a local ``.env``
file) through :class:`pydantic_settings.BaseSettings`, so the same code base
runs in development, test and production without modification.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings (12-factor style)."""

    model_config = SettingsConfigDict(
        env_prefix="CHATGANGS_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "ChatGangs"
    app_version: str = "1.0.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    docs_enabled: bool = True
    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------ security
    api_keys: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description="Optional API keys (comma separated). Empty list disables key authentication.",
    )
    require_api_key: bool = False
    api_key_header: str = "X-API-Key"
    max_input_chars: int = 4000
    max_batch_size: int = 64
    max_message_chars: int = 2000
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    bot_requests_per_minute: int = 300
    trust_forwarded_for: bool = False

    # ----------------------------------------------------------- storage
    database_url: str = "sqlite:///./data/chatgangs.db"
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"

    # ------------------------------------------------------------ memory
    memory_max_turns: int = 20
    memory_ttl_hours: int = 72
    memory_persist: bool = True
    memory_store_user_profile: bool = True
    memory_enabled: bool = True

    # ------------------------------------------------------------- nlp
    spacy_model_en: str = "en_core_web_sm"
    spacy_model_id: str = ""
    enable_spacy: bool = True
    max_tokens_per_request: int = 512
    default_language: str = "id"
    language_confidence_threshold: float = 0.55
    intent_confidence_threshold: float = 0.30

    # ---------------------------------------------------------- logging
    log_level: str = "INFO"
    log_json: bool = False
    log_chat_content: bool = False
    log_file: str = ""

    # --------------------------------------------------------- performance
    response_cache_ttl_seconds: int = 0  # 0 = disabled by default
    response_cache_size: int = 256
    nlp_cache_size: int = 2048
    bot_concurrency: int = 8

    @field_validator("cors_origins", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v):
        """Accept both JSON arrays and simple comma-separated strings."""
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return []
            if text.startswith("["):
                import json

                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return []
            return [item.strip() for item in text.split(",") if item.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
