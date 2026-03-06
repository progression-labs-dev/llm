"""API route handlers for the HTTP gateway."""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model

from progression_labs.llm.completion import complete
from progression_labs.llm.server.models import (
    CompletionRequest,
    CompletionResponse,
    ExtractionRequest,
    ExtractionResponse,
    UsageResponse,
)
from progression_labs.llm.structured import extract

router = APIRouter()

# JSON Schema type → Python type mapping
_JSON_SCHEMA_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _json_schema_to_fields(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Convert JSON Schema properties to pydantic create_model field definitions."""
    required = required or []
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        python_type = _resolve_type(prop)
        if name in required:
            fields[name] = (python_type, ...)
        else:
            fields[name] = (python_type | None, None)
    return fields


def _resolve_type(prop: dict[str, Any]) -> type:
    """Resolve a single JSON Schema property to a Python type."""
    schema_type = prop.get("type", "string")

    if schema_type == "array":
        item_type = _resolve_type(prop.get("items", {"type": "string"}))
        return list[item_type]  # type: ignore[valid-type]

    if schema_type == "object":
        nested_props = prop.get("properties", {})
        nested_required = prop.get("required", [])
        nested_fields = _json_schema_to_fields(nested_props, nested_required)
        return create_model("NestedModel", **nested_fields)

    return _JSON_SCHEMA_TYPES.get(schema_type, str)


def _build_response_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Build a dynamic Pydantic model from a JSON Schema dict."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    fields = _json_schema_to_fields(properties, required)
    return create_model("DynamicResponseModel", **fields)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/complete", response_model=CompletionResponse)
async def complete_endpoint(body: CompletionRequest, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    response = await complete(
        model=body.model,
        messages=body.messages,
        fallbacks=body.fallbacks,
        max_retries=body.max_retries,
        timeout=body.timeout,
        request_id=request_id,
    )
    # Access via model_dump() — litellm's ModelResponse uses Pydantic extra='allow'
    # which ty cannot resolve for direct attribute access.
    data = response.model_dump()
    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] or "" if choices else ""
    usage = None
    usage_data = data.get("usage")
    if usage_data:
        usage = UsageResponse(
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
            total_tokens=usage_data["total_tokens"],
        )
    result = CompletionResponse(content=content, usage=usage)
    return JSONResponse(content=result.model_dump(by_alias=True, exclude_none=True))


@router.post("/v1/extract", response_model=ExtractionResponse)
async def extract_endpoint(body: ExtractionRequest, request: Request) -> JSONResponse:
    response_model = _build_response_model(body.response_schema)
    result = await extract(
        response_model=response_model,
        model=body.model,
        messages=body.messages,
        prompt=body.prompt,
        max_retries=body.max_retries,
    )
    data = result.model_dump()
    response = ExtractionResponse(data=data)
    return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))
