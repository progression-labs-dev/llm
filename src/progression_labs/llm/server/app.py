"""FastAPI HTTP gateway for the Progression Labs LLM SDK."""

import importlib
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from progression_labs.llm.server.models import ErrorDetail, ErrorResponse
from progression_labs.llm.server.routes import router

REQUEST_ID_PREFIX = "req_"

logger = logging.getLogger(__name__)


# --- Structured JSON logging ---
class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter with required observability fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "service": os.getenv("SERVICE_NAME", "progression-labs-llm"),
            "environment": os.getenv("SERVICE_ENVIRONMENT", "development"),
            "logger": record.name,
        }
        if hasattr(record, "request_id"):
            log_data["requestId"] = record.request_id
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def _configure_logging() -> None:
    """Configure structured JSON logging for the server."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _resolve_secrets() -> None:
    """Resolve *_SECRET_NAME env vars from GCP Secret Manager into actual values."""
    secret_name_vars = {k: v for k, v in os.environ.items() if k.endswith("_SECRET_NAME")}
    if not secret_name_vars:
        return
    try:
        secretmanager = importlib.import_module("google.cloud.secretmanager")
        client = secretmanager.SecretManagerServiceClient()
        for env_var, secret_resource in secret_name_vars.items():
            target_var = env_var.removesuffix("_SECRET_NAME")
            if os.environ.get(target_var):
                continue  # Already set directly, don't override
            try:
                response = client.access_secret_version(name=f"{secret_resource}/versions/latest")
                os.environ[target_var] = response.payload.data.decode("utf-8")
                logger.info("Resolved secret %s -> %s", env_var, target_var)
            except Exception as exc:
                logger.warning("Failed to resolve secret %s: %s", env_var, exc)
    except ImportError:
        logger.warning("google-cloud-secret-manager not installed, skipping secret resolution")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: configure logging, resolve secrets, init observability. Shutdown: flush traces."""
    _configure_logging()
    _resolve_secrets()
    try:
        from progression_labs.llm.observability import init_observability

        init_observability()
        logger.info("Langfuse observability initialized")
    except (ValueError, Exception) as exc:
        logger.warning("Observability not initialized: %s", exc)
    yield
    try:
        from progression_labs.llm.observability import flush_traces

        flush_traces()
    except Exception:  # noqa: S110
        pass


app = FastAPI(title="Progression Labs LLM Gateway", lifespan=lifespan)

# --- CORS ---
cors_origins = os.getenv("LLM_CORS_ORIGINS", "*").split(",")
# cast() needed because ty cannot match Starlette middleware classes to the
# _MiddlewareFactory[P] ParamSpec protocol used by add_middleware.
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID middleware (raw ASGI for performance) ---
class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = f"{REQUEST_ID_PREFIX}{secrets.token_urlsafe(16)}"
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


app.add_middleware(cast(Any, RequestIDMiddleware))


# --- Exception handlers ---
def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(status_code: int, code: str, message: str, request: Request) -> JSONResponse:
    """Build a standard error response per data conventions."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                request_id=_get_request_id(request),
            )
        ).model_dump(by_alias=True, exclude_none=True),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return _error_response(400, "BAD_REQUEST", str(exc), request)


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return _error_response(422, "VALIDATION_ERROR", str(exc), request)


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return _error_response(500, "INTERNAL_SERVER_ERROR", str(exc), request)


# --- Routes ---
app.include_router(router)
