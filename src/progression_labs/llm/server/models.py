"""Pydantic request/response models for the HTTP gateway."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that serializes to camelCase for API responses."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class UsageResponse(CamelModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionRequest(CamelModel):
    model: str
    messages: list[dict[str, str]]
    fallbacks: list[str] | None = None
    max_retries: int | None = None
    timeout: float | None = None


class CompletionResponse(CamelModel):
    content: str
    usage: UsageResponse | None = None


class ExtractionRequest(CamelModel):
    model: str
    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    response_schema: dict  # Dynamic JSON Schema — structure varies per request
    max_retries: int = 3


class ExtractionResponse(CamelModel):
    data: dict  # Dynamic shape matching user-provided response_schema
    usage: UsageResponse | None = None


class ErrorDetail(CamelModel):
    """Nested error object per data conventions standard."""

    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(CamelModel):
    """Standard error envelope."""

    error: ErrorDetail
