FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra server --extra secrets --no-install-project

# Copy application source
COPY src/ src/
RUN uv sync --frozen --extra server --extra secrets

# Run as non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "progression_labs.llm.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
