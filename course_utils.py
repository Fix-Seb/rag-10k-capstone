"""Shared helpers used by notebooks across all three topics.

Why is this a real module instead of inline cells? So students see a clean
``from course_utils import get_llm_client`` line in every notebook and don't
get distracted by ~40 lines of boilerplate. The full source is here for the
curious.

Two main entry points:

- :func:`get_llm_client` - returns an OpenAI-compatible client + model name,
  auto-selecting Ollama (local) or Groq (hosted, free tier).
- :func:`load_env` - loads ``.env`` if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _auto_enable_hf_offline_if_unreachable() -> None:
    """If HuggingFace is unreachable (SSL/firewall/offline), set
    ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` so ``transformers`` and
    ``sentence-transformers`` use the local cache instead of retrying network
    calls 5x with growing backoff on every single model load.

    Respects any value the user already set. Runs once at import time so it
    applies before notebooks import ``sentence_transformers``/``transformers``.
    """
    if "HF_HUB_OFFLINE" in os.environ or "TRANSFORMERS_OFFLINE" in os.environ:
        return
    try:
        import requests
        requests.head("https://huggingface.co", timeout=3)
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_auto_enable_hf_offline_if_unreachable()


@dataclass
class LLMHandle:
    """Returned by :func:`get_llm_client`.

    Use as a tuple ``(client, model)`` or access attributes.
    """

    client: Any
    model: str
    backend: str

    def __iter__(self):  # so callers can do: client, model = get_llm_client()
        yield self.client
        yield self.model


def load_env() -> None:
    """Load environment variables from a ``.env`` file at the repo root, if any."""
    if load_dotenv is None:
        return
    # Walk up to find .env at repo root
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        candidate = os.path.join(cur, ".env")
        if os.path.exists(candidate):
            load_dotenv(candidate)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent


def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def _ollama_available_with_models() -> bool:
    """Ollama is only useful for us if it has at least one model pulled."""
    try:
        import ollama
        return bool(ollama.list().get("models", []))
    except Exception:
        return False


def get_llm_client(prefer: str | None = None) -> LLMHandle:
    """Return an OpenAI-compatible client + default model.

    Order of preference:

    1. ``prefer="ollama"`` -> raises if Ollama is unreachable.
    2. ``prefer="groq"``   -> raises if ``GROQ_API_KEY`` is unset.
    3. No preference (default): Ollama if reachable, else Groq if key present.
    """
    load_env()
    from openai import OpenAI

    # 0. Local OpenAI-compatible server (e.g. mlx_lm.server, vLLM, llama.cpp).
    #    Activated by setting LOCAL_LLM_BASE_URL (+ optional LOCAL_LLM_MODEL).
    local_url = os.environ.get("LOCAL_LLM_BASE_URL")
    if prefer == "local" or (prefer is None and local_url):
        if not local_url:
            raise RuntimeError("prefer='local' requires LOCAL_LLM_BASE_URL env var")
        client = OpenAI(base_url=local_url, api_key=os.environ.get("LOCAL_LLM_API_KEY", "none"))
        model = os.environ.get("LOCAL_LLM_MODEL", "local")
        return LLMHandle(client=client, model=model, backend="local")

    want_ollama = prefer == "ollama" or (prefer is None and _ollama_available_with_models())
    if want_ollama:
        # Ollama exposes an OpenAI-compatible endpoint on port 11434.
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Find a Llama 3.2 tag the user actually has pulled
        import ollama
        names = [m.get("name", m.get("model", "")) for m in ollama.list().get("models", [])]
        model = next((n for n in names if "llama3.2" in n), "llama3.2:3b-instruct-q4_K_M")
        return LLMHandle(client=client, model=model, backend="ollama")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM backend available. Either start Ollama "
            "(`ollama serve` + `ollama pull llama3.2:3b-instruct-q4_K_M`) "
            "or set GROQ_API_KEY in a .env file, "
            "or set LOCAL_LLM_BASE_URL to any OpenAI-compatible endpoint."
        )
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    return LLMHandle(client=client, model="llama-3.1-8b-instant", backend="groq")


def get_embedder():
    """Return a CPU-friendly sentence-transformer for RAG demos.

    Local, ~80 MB, runs in milliseconds. No API key required.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
