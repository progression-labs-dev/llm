---
type: architecture
title: LLM SDK Architecture
owner: chris-little
last_reviewed: 2026-03-03
tracks: src,client,scripts
---

# LLM SDK Architecture

## Overview

`llm` is a Python SDK for multi-provider LLM access with structured output support, generated clients, and optional HTTP gateway capabilities.

## Main Components

- `src/progression_labs/`
  Core SDK implementation.
- `client/`
  Generated OpenAPI client assets.
- `scripts/`
  Build and generation support.
- `tests/`
  Validation of SDK behavior.

## Integration Model

The library provides a single SDK surface across providers while keeping generation artifacts and testing separated from the core package.
