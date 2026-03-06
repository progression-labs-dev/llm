---
type: architecture
title: LLM SDK Architecture
owner: chris-little
last_reviewed: 2026-03-06
tracks: src
---

# LLM SDK Architecture

## Overview

`llm` is a Python SDK for multi-provider LLM access with structured output support, observability, RAG, and evaluation capabilities.

## Main Components

- `src/progression_labs/`
  Core SDK implementation.
- `tests/`
  Validation of SDK behavior.

## Integration Model

The library provides a single SDK surface across providers. The HTTP gateway service has been split into a separate repository (`llm-gateway`).
