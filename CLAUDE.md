# Agent Instructions

## Project Overview

llm is a Python library for LLM interactions. Built with Python.

- **Tier:** internal
- **Package:** `llm`

## Quick Reference

| Task | Command |
|------|---------|
| Install | `uv sync` |
| Test | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |

## Architecture

```
src/
  progression_labs/   # Main library code
tests/               # Test suite
```

See `docs/` for detailed architecture documentation.

## Standards & Guidelines

This project uses [@progression-labs-development/conform](https://github.com/progression-labs-development/standards-kit) for coding standards.

- **Config:** `standards.toml` (extends `python-internal` from the standards registry)
- **Guidelines:** https://chrismlittle123.github.io/standards/

Use the MCP tools to query standards at any time:

| Tool | Purpose |
|------|---------|
| `get_standards` | Get guidelines matching a context (e.g., `python openapi llm`) |
| `list_guidelines` | List all available guidelines |
| `get_guideline` | Get a specific guideline by ID |
| `get_ruleset` | Get a tool configuration ruleset (e.g., `python-internal`) |

## Workflow

- **Branch:** Create feature branches from `main`
- **CI:** GitHub Actions runs test and lint on PRs
- **Deploy:** pip install (library package)
- **Commits:** Use conventional commits (`feat:`, `fix:`, `chore:`, etc.)

## Project-Specific Notes

- Uses `uv` as the Python package manager (not pip/poetry)
- The HTTP gateway has been split into a separate repo (`llm-gateway`)
