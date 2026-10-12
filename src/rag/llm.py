"""Optional LLM provider client (stdlib only: urllib).

Nothing is sent anywhere unless the user sets LLM_ENABLED=true AND configures a
provider/model (and a key, unless the endpoint is localhost, e.g. Ollama).
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from config import Settings


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMDisabledError(LLMError):
    """The LLM provider is not enabled or not fully configured."""


class LLMTimeoutError(LLMError):
    """The provider did not answer in time."""


def _base_url(s: Settings) -> str:
    if s.llm_base_url:
        return s.llm_base_url.rstrip("/")
    return "https://api.anthropic.com/v1" if s.llm_provider == "anthropic" else "https://api.openai.com/v1"


def is_local_endpoint(s: Settings) -> bool:
    host = (urlparse(_base_url(s)).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def llm_status(s: Settings) -> tuple[bool, str]:
    """Return ``(ready, human_readable_reason)``. Does not make a network call."""
    if not s.llm_enabled:
        return False, "LLM generation is disabled (LLM_ENABLED is not true). Retrieval-only mode is active."
    if s.llm_provider not in {"openai", "anthropic"}:
        return False, f"Unknown LLM_PROVIDER '{s.llm_provider}'. Use 'openai' (OpenAI-compatible) or 'anthropic'."
    if not s.llm_model:
        return False, "LLM_MODEL is not set. Choose a model name from your provider."
    if not s.llm_api_key and not is_local_endpoint(s):
        key = "ANTHROPIC_API_KEY" if s.llm_provider == "anthropic" else "OPENAI_API_KEY"
        return False, f"{key} is missing. Add it to your .env file (never commit it)."
    return True, f"Provider '{s.llm_provider}' with model '{s.llm_model}' at {urlparse(_base_url(s)).netloc}."


def generate(s: Settings, system: str, user: str, max_tokens: int = 700) -> str:
    """Call the configured provider and return the answer text."""
    ready, reason = llm_status(s)
    if not ready:
        raise LLMDisabledError(reason)
    if s.llm_provider == "anthropic":
        url = _base_url(s) + "/messages"
        headers = {"content-type": "application/json", "x-api-key": s.llm_api_key, "anthropic-version": "2023-06-01"}
        body = {"model": s.llm_model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}]}
    else:
        url = _base_url(s) + "/chat/completions"
        headers = {"content-type": "application/json"}
        if s.llm_api_key:
            headers["authorization"] = f"Bearer {s.llm_api_key}"
        body = {"model": s.llm_model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    scheme = urlparse(url).scheme
    if scheme != "https" and not (scheme == "http" and is_local_endpoint(s)):
        raise LLMDisabledError("LLM_BASE_URL must use https (plain http is only allowed for localhost).")
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=s.llm_timeout_s) as resp:  # noqa: S310  # nosec B310 - scheme validated above
            payload = json.loads(resp.read().decode("utf-8"))
    except (socket.timeout, TimeoutError) as exc:
        raise LLMTimeoutError(f"The model provider did not respond within {s.llm_timeout_s}s.") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"Provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise LLMTimeoutError(f"The model provider did not respond within {s.llm_timeout_s}s.") from exc
        raise LLMError(f"Could not reach the provider: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("Provider returned a response that is not valid JSON.") from exc
    try:
        if s.llm_provider == "anthropic":
            return "".join(b.get("text", "") for b in payload["content"] if b.get("type") == "text").strip()
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Unexpected response format from provider.") from exc
