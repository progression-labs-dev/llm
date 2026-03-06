# 01: Project Setup

## Objective

Set up the Python package structure, dependencies, and CI/CD for `@progression-labs/llm`.

## Tasks

- [ ] Initialize Python package with `pyproject.toml`
- [ ] Set up directory structure
- [ ] Configure development tools (ruff, mypy, pytest)
- [ ] Set up GitHub Actions for CI
- [ ] Configure package publishing

## Directory Structure

```
llm/
├── src/
│   └── progression_labs/
│       └── llm/
│           ├── __init__.py
│           ├── client.py          # Main client
│           ├── completion.py      # LiteLLM wrapper
│           ├── structured.py      # Instructor wrapper
│           ├── observability.py   # Langfuse setup
│           ├── rag/
│           │   ├── __init__.py
│           │   ├── embeddings.py
│           │   └── vectorstore.py
│           └── eval/
│               ├── __init__.py
│               └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── test_completion.py
│   ├── test_structured.py
│   └── test_rag.py
├── docs/
│   ├── research/
│   └── plans/
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        ├── ci.yml
        └── publish.yml
```

## pyproject.toml

```toml
[project]
name = "progression-labs-llm"
version = "0.1.0"
description = "Unified Python LLM SDK for Progression Labs"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.55.0",
    "instructor>=1.7.0",
    "chromadb>=0.5.0",
    "deepeval>=2.0.0",
    "langfuse>=2.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/progression_labs"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

## GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy src/
      - run: pytest
```

## Acceptance Criteria

- [ ] `pip install -e .` works
- [ ] `ruff check .` passes
- [ ] `mypy src/` passes
- [ ] `pytest` runs (even with no tests yet)
- [ ] CI runs on push/PR
