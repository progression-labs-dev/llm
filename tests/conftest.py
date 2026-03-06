"""Shared test configuration and fixtures."""

import os
import warnings

import pytest

GCP_PROJECT = os.environ.get("GCP_PROJECT", "christopher-little-dev")

SECRET_MAP = {
    "OPENAI_API_KEY": os.environ.get(
        "OPENAI_SECRET_NAME", "livekit-agent-openai-api-key-secret-dev"
    ),
    "ANTHROPIC_API_KEY": os.environ.get(
        "ANTHROPIC_SECRET_NAME", "christopher-little-dev-anthropic-api-key-secret-dev"
    ),
}


@pytest.fixture(autouse=True, scope="session")
def _load_secrets_from_gcp():
    """Fetch API keys from Google Secret Manager and set as env vars.

    Skips gracefully if Secret Manager is unavailable so unit tests still pass.
    """
    # Skip if keys are already set (e.g. via CI env vars or .env)
    if all(os.environ.get(k) for k in SECRET_MAP):
        return

    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()

        for env_var, secret_id in SECRET_MAP.items():
            if os.environ.get(env_var):
                continue
            name = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            os.environ[env_var] = response.payload.data.decode("UTF-8").strip()

    except Exception as exc:
        warnings.warn(
            f"Could not load secrets from Google Secret Manager: {exc}. "
            "Integration tests requiring API keys may be skipped.",
            stacklevel=1,
        )
