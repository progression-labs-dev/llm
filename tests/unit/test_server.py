"""Tests for the FastAPI HTTP gateway."""

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from progression_labs.llm.server.app import _JSONFormatter, _resolve_secrets, app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# _JSONFormatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Tests for structured JSON log formatter."""

    def test_basic_format(self):
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "info"
        assert data["message"] == "hello world"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data
        assert "service" in data
        assert "environment" in data

    def test_format_with_request_id(self):
        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        record.request_id = "req_abc123"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["requestId"] == "req_abc123"

    def test_format_with_exception(self):
        formatter = _JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "boom" in data["exception"]


# ---------------------------------------------------------------------------
# _resolve_secrets
# ---------------------------------------------------------------------------


class TestResolveSecrets:
    """Tests for _resolve_secrets()."""

    def test_noop_when_no_secret_name_vars(self):
        """Returns early when no *_SECRET_NAME env vars exist."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ["PATH"] = "/usr/bin"
            _resolve_secrets()  # should not raise

    def test_resolves_secret_from_gcp(self):
        """Fetches secret and sets target env var."""
        mock_response = MagicMock()
        mock_response.payload.data.decode.return_value = "s3cr3t"

        mock_client = MagicMock()
        mock_client.access_secret_version.return_value = mock_response

        mock_module = MagicMock()
        mock_module.SecretManagerServiceClient.return_value = mock_client

        env = {"MY_KEY_SECRET_NAME": "projects/p/secrets/s"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("importlib.import_module", return_value=mock_module),
        ):
            _resolve_secrets()
            # Assert inside the patch context before env is restored
            assert os.environ.get("MY_KEY") == "s3cr3t"
            mock_client.access_secret_version.assert_called_once_with(
                name="projects/p/secrets/s/versions/latest"
            )

    def test_skips_when_target_already_set(self):
        """Does not override existing env var."""
        mock_client = MagicMock()
        mock_module = MagicMock()
        mock_module.SecretManagerServiceClient.return_value = mock_client

        env = {
            "MY_KEY_SECRET_NAME": "projects/p/secrets/s",
            "MY_KEY": "already-set",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("importlib.import_module", return_value=mock_module),
        ):
            _resolve_secrets()
            mock_client.access_secret_version.assert_not_called()
            assert os.environ["MY_KEY"] == "already-set"

    def test_warns_when_google_cloud_not_installed(self):
        """Logs warning when google-cloud-secret-manager is missing."""
        import importlib

        original_import = importlib.import_module

        def selective_import(name, *args, **kwargs):
            if name == "google.cloud.secretmanager":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        env = {"MY_KEY_SECRET_NAME": "projects/p/secrets/s"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("importlib.import_module", side_effect=selective_import),
            patch("progression_labs.llm.server.app.logger") as mock_logger,
        ):
            _resolve_secrets()

        mock_logger.warning.assert_called_once()
        assert "not installed" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_has_request_id():
    resp = client.get("/health")
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# POST /v1/complete
# ---------------------------------------------------------------------------


def _mock_model_response(content: str = "Hello!") -> MagicMock:
    """Build a fake ModelResponse-like object."""
    resp = MagicMock()
    resp.model_dump.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return resp


@patch("progression_labs.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete(mock_complete: AsyncMock):
    mock_complete.return_value = _mock_model_response("Hi there")
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Hi there"
    assert data["usage"]["totalTokens"] == 15


@patch("progression_labs.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete_has_request_id(mock_complete: AsyncMock):
    mock_complete.return_value = _mock_model_response()
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert "x-request-id" in resp.headers


def test_complete_invalid_body():
    """Missing required 'model' field returns 422."""
    resp = client.post("/v1/complete", json={"messages": []})
    assert resp.status_code == 422


@patch("progression_labs.llm.server.routes.complete", new_callable=AsyncMock)
def test_complete_llm_error(mock_complete: AsyncMock):
    """LLM exception returns 500 with structured error."""
    mock_complete.side_effect = RuntimeError("provider down")
    resp = client.post(
        "/v1/complete",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "provider down" in data["error"]["message"]
    assert data["error"]["requestId"] is not None


# ---------------------------------------------------------------------------
# POST /v1/extract
# ---------------------------------------------------------------------------


@patch("progression_labs.llm.server.routes.extract", new_callable=AsyncMock)
def test_extract(mock_extract: AsyncMock):
    fake_result = MagicMock()
    fake_result.model_dump.return_value = {"name": "Alice", "age": 30}
    mock_extract.return_value = fake_result

    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o",
            "prompt": "Alice is 30",
            "response_schema": {
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == {"name": "Alice", "age": 30}


@patch("progression_labs.llm.server.routes.extract", new_callable=AsyncMock)
def test_extract_has_request_id(mock_extract: AsyncMock):
    fake_result = MagicMock()
    fake_result.model_dump.return_value = {"x": 1}
    mock_extract.return_value = fake_result

    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o",
            "prompt": "x is 1",
            "response_schema": {
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    )
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_headers():
    resp = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@patch("progression_labs.llm.server.routes.complete", new_callable=AsyncMock)
def test_value_error_returns_400(mock_complete: AsyncMock):
    mock_complete.side_effect = ValueError("bad model name")
    resp = client.post(
        "/v1/complete",
        json={"model": "bad", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "BAD_REQUEST"
    assert "bad model name" in data["error"]["message"]
