"""Pydantic request/response models — the API contract."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chatgangs.config import get_settings

SUPPORTED = ["id", "en", "ms", "ja", "ko", "es", "de", "fr"]


def _max_chars() -> int:
    return get_settings().max_input_chars


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TextRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=20000,
                      description="Text to analyse", examples=["Pelayanannya bagus sekali!"])
    language: Optional[Literal["id", "en", "ms", "ja", "ko", "es", "de", "fr"]] = Field(
        None, description="Force a language; auto-detected when omitted"
    )
    explain: bool = Field(True, description="Include evidence terms and per-sentence breakdown")

    @field_validator("text")
    @classmethod
    def _limit(cls, value: str) -> str:
        limit = _max_chars()
        if len(value) > limit:
            raise ValueError(f"text exceeds the maximum of {limit} characters")
        return value


class AnalyzeRequest(TextRequest):
    include_tokens: bool = True
    include_qa: bool = True


class LanguageRequest(StrictModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TokenizeRequest(TextRequest):
    include_offsets: bool = True
    include_stems: bool = True
    lowercase: bool = True


class SentimentRequest(TextRequest):
    pass


class IntentRequest(TextRequest):
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_candidates: bool = True


class EntitiesRequest(TextRequest):
    labels: Optional[List[str]] = Field(
        None, description="Restrict results to these entity labels"
    )
    use_spacy: bool = True


class SimilarityRequest(StrictModel):
    texts: List[str] = Field(..., min_length=2, max_length=25)
    language: Optional[Literal["id", "en", "ms", "ja", "ko", "es", "de", "fr"]] = None

    @field_validator("texts")
    @classmethod
    def _validate(cls, value: List[str]) -> List[str]:
        if any(not item.strip() for item in value):
            raise ValueError("texts must not contain empty strings")
        limit = _max_chars()
        if any(len(item) > limit for item in value):
            raise ValueError(f"each text must be at most {limit} characters")
        return value


class ClassificationRequest(TextRequest):
    labels: Optional[Dict[str, List[str]]] = Field(
        None, description="Custom keyword taxonomy {label: [keywords]}"
    )
    task: Literal["auto", "topic", "speech_act"] = Field(
        "auto",
        description=(
            "auto = speech-act label + trained topic when available; "
            "topic = trained topic model only; speech_act = keyword taxonomy only"
        ),
    )


class QARequest(StrictModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: Optional[Literal["id", "en", "ms", "ja", "ko", "es", "de", "fr"]] = None
    top_k: int = Field(3, ge=1, le=10)


class BatchRequest(StrictModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)
    tasks: List[Literal["language", "sentiment", "intent", "entities", "tokenize"]] = Field(
        default_factory=lambda: ["language", "sentiment", "intent"]
    )

    @field_validator("texts")
    @classmethod
    def _limit(cls, value: List[str]) -> List[str]:
        limit = get_settings().max_input_chars
        max_batch = get_settings().max_batch_size
        if len(value) > max_batch:
            raise ValueError(f"at most {max_batch} texts per batch")
        if any(len(item) > limit for item in value):
            raise ValueError(f"each text must be at most {limit} characters")
        return value


class ChatRequest(StrictModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         examples=["Halo, apa saja kemampuanmu?"])
    session_id: str = Field("default", min_length=1, max_length=128)
    language: Optional[Literal["id", "en", "ms", "ja", "ko", "es", "de", "fr"]] = None
    store_memory: bool = Field(True, description="Persist this turn to conversation history")
    user_profile: Optional[Dict[str, Any]] = Field(
        None, description="Optional non-sensitive profile hints (e.g. {name: 'Budi'})"
    )

    @field_validator("message")
    @classmethod
    def _limit(cls, value: str) -> str:
        limit = get_settings().max_message_chars
        if len(value) > limit:
            raise ValueError(f"message exceeds the maximum of {limit} characters")
        return value

    @field_validator("session_id")
    @classmethod
    def _valid_session(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "session_id may only contain letters, digits, hyphens and underscores"
            )
        return cleaned


class FeedbackRequest(StrictModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(..., ge=-1, le=5)
    comment: str = Field("", max_length=2000)
    message_id: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, Any]
