"""Thin wrapper around the Anthropic API.

All provider-specific code (the Anthropic SDK, model names, API key handling)
lives here so the rest of the pipeline is provider-agnostic and the provider
can be swapped later without touching extract.py or run.py.

Responses are cached on disk, keyed by a SHA256 hash of the model + prompt,
so re-running the pipeline on unchanged reports makes no API calls.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Lazily construct the Anthropic client from ANTHROPIC_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def _cache_path(cache_dir: Path, model: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def call_llm_json(prompt: str, model: str, cache_dir: Path, max_tokens: int = 8000) -> str:
    """Call Claude with a prompt that must return JSON only.

    Returns the raw response text. Caching and parsing/validation are the
    caller's responsibility to keep this function a pure provider wrapper.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, model, prompt)
    if path.exists():
        return path.read_text(encoding="utf-8")

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    path.write_text(text, encoding="utf-8")
    return text
