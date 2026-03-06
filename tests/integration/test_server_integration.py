"""Integration tests for the FastAPI HTTP gateway.

These tests call real LLM providers through the HTTP endpoints.
Requires API keys (loaded via conftest.py GCP Secret Manager fixture).
"""

import pytest
from fastapi.testclient import TestClient

from progression_labs.llm.server.app import app

pytestmark = pytest.mark.integration

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert "x-request-id" in resp.headers


# ---------------------------------------------------------------------------
# POST /v1/complete — real LLM calls
# ---------------------------------------------------------------------------


def test_complete_openai():
    resp = client.post(
        "/v1/complete",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "hello" in data["content"].lower()
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["usage"]["total_tokens"] > 0
    assert "x-request-id" in resp.headers


def test_complete_anthropic():
    resp = client.post(
        "/v1/complete",
        json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "hello" in data["content"].lower()
    assert data["usage"]["total_tokens"] > 0


def test_complete_with_system_message():
    resp = client.post(
        "/v1/complete",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a pirate. Respond like a pirate."},
                {"role": "user", "content": "Hello!"},
            ],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["content"].lower()
    assert any(word in content for word in ["ahoy", "arr", "matey", "aye", "pirate"])


# ---------------------------------------------------------------------------
# POST /v1/extract — real LLM calls
# ---------------------------------------------------------------------------


def test_extract_simple():
    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o-mini",
            "prompt": "Alice is 30 years old",
            "response_schema": {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "Alice"
    assert data["age"] == 30
    assert "x-request-id" in resp.headers


def test_extract_with_messages():
    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract user information from the text."},
                {"role": "user", "content": "Bob is 25 years old."},
            ],
            "response_schema": {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "Bob"
    assert data["age"] == 25


def test_extract_nested_object():
    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o-mini",
            "prompt": "Acme Corp is a technology company based in San Francisco, USA.",
            "response_schema": {
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "headquarters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["city", "country"],
                    },
                },
                "required": ["name", "industry", "headquarters"],
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "Acme Corp"
    assert data["headquarters"]["city"] == "San Francisco"


def test_extract_with_array():
    resp = client.post(
        "/v1/extract",
        json={
            "model": "gpt-4o-mini",
            "prompt": "The primary colors are red, blue, and yellow.",
            "response_schema": {
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["colors"],
            },
        },
    )
    assert resp.status_code == 200
    colors = resp.json()["data"]["colors"]
    assert isinstance(colors, list)
    assert len(colors) == 3
    assert "red" in [c.lower() for c in colors]


# ---------------------------------------------------------------------------
# Error cases — real server, no mocks
# ---------------------------------------------------------------------------


def test_invalid_body_returns_422():
    resp = client.post("/v1/complete", json={"messages": []})
    assert resp.status_code == 422
    assert "x-request-id" in resp.headers


def test_extract_missing_schema_returns_422():
    resp = client.post(
        "/v1/extract",
        json={"model": "gpt-4o-mini", "prompt": "hello"},
    )
    assert resp.status_code == 422
