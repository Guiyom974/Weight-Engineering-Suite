"""LLM endpoint settings and live model discovery for any OpenAI-compatible server.

One configuration path for every provider (Poe, OpenRouter, Ollama, LM Studio, llama.cpp server, vLLM, OpenAI):
a base URL, the *name* of the environment variable that holds the API key, and a model id. The key itself is never
stored, logged or returned by any API in this package.

Resolution order for each field: explicit argument (CLI flag or Studio selection) > environment variable >
workspace config (``llm:`` section of ``neuroforge.yaml``) > default. Defaults target Poe, which exposes hundreds of
models behind one key: base URL ``https://api.poe.com/v1``, key variable ``POE_API_KEY``.

Safety rule: the default key variable (``POE_API_KEY``) is only sent to ``api.poe.com``. For any other host the key
variable must be named explicitly (flag ``--key-env``, ``WEIGHTLAB_LLM_KEY_ENV`` or the workspace config), so a Poe
key never leaks to an arbitrary base URL. Keys are never sent over plain HTTP to a non-loopback host.

Main entry points: ``resolve_settings``, ``LLMSettings``, ``fetch_models`` (reasoning models by default),
``list_models``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .llm_backends import OpenAICompatibleBackend

DEFAULT_BASE_URL = "https://api.poe.com/v1"
DEFAULT_KEY_ENV = "POE_API_KEY"
POE_HOST = "api.poe.com"
ENV_BASE_URL = "WEIGHTLAB_LLM_BASE_URL"
ENV_MODEL = "WEIGHTLAB_LLM_MODEL"
ENV_KEY_ENV = "WEIGHTLAB_LLM_KEY_ENV"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_MAX_MODELS_BYTES = 4 * 1024 * 1024


class ModelListError(RuntimeError):
    """The model list could not be fetched; ``reason`` is a short machine-readable code.

    Reasons: ``no-key`` (key variable unset), ``unauthorized`` (HTTP 401/403), ``http-error``, ``unreachable``
    (network or timeout), ``bad-response`` (not an OpenAI-style list), ``insecure`` (key over plain HTTP refused).
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class LLMSettings:
    """Resolved endpoint settings. Holds the key variable *name*, never the key."""

    base_url: str
    api_key_env: str | None
    model: str | None
    timeout: float = 120.0
    note: str = ""

    @property
    def host(self) -> str:
        """Lower-cased host name of the base URL."""
        return (urlparse(self.base_url).hostname or "").lower()

    @property
    def is_poe(self) -> bool:
        """True when the endpoint is Poe's OpenAI-compatible API."""
        return self.host == POE_HOST

    @property
    def key_present(self) -> bool:
        """True when the named key variable is set to a non-empty value."""
        return bool(self.api_key_env and os.environ.get(self.api_key_env))

    def describe(self) -> dict[str, Any]:
        """Public, secret-free summary for the CLI and the Studio."""
        return {
            "base_url": self.base_url,
            "provider": "Poe" if self.is_poe else self.host or "unknown",
            "api_key_env": self.api_key_env,
            "key_present": self.key_present,
            "model": self.model,
            "note": self.note,
        }

    def backend(self, model: str | None = None) -> OpenAICompatibleBackend:
        """Chat backend for ``model`` (or the configured model).

        Raises:
            ValueError: no model selected, or the URL/key combination is unsafe.
        """
        chosen = model or self.model
        if not chosen:
            raise ValueError("no LLM model selected (use --model, WEIGHTLAB_LLM_MODEL or the Studio dropdown)")
        key_env = self.api_key_env if self.key_present else None
        return OpenAICompatibleBackend(self.base_url, chosen, api_key_env=key_env, timeout=self.timeout)


def _check_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"LLM base URL must be an http(s) URL, got {url!r}")
    return url.rstrip("/")


def _check_env_name(name: str) -> str:
    if not _ENV_NAME_RE.fullmatch(name):
        raise ValueError(f"key variable name must look like POE_API_KEY, got {name!r}")
    return name


def resolve_settings(
    base_url: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
    workspace_llm: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> LLMSettings:
    """Combine flags, environment, workspace config and defaults into ``LLMSettings``.

    Args:
        base_url: explicit base URL (highest priority).
        api_key_env: explicit name of the key variable.
        model: explicit model id.
        workspace_llm: the ``llm:`` mapping from ``neuroforge.yaml`` (may hold ``base_url``, ``api_key_env``,
            ``model``; a ``api_key`` entry is rejected so secrets never live in config files).
        env: environment mapping (defaults to ``os.environ``; injectable for tests).

    Raises:
        ValueError: invalid URL, invalid variable name, or an API key stored in the workspace config.
    """
    env = os.environ if env is None else env
    ws = dict(workspace_llm or {})
    if "api_key" in ws:
        raise ValueError("neuroforge.yaml must not contain an API key; set an environment variable instead")
    url = _check_base_url(base_url or env.get(ENV_BASE_URL) or ws.get("base_url") or DEFAULT_BASE_URL)
    chosen_model = model or env.get(ENV_MODEL) or ws.get("model") or None
    explicit_key_env = api_key_env or env.get(ENV_KEY_ENV) or ws.get("api_key_env")
    note = ""
    host = (urlparse(url).hostname or "").lower()
    if explicit_key_env:
        key_env: str | None = _check_env_name(str(explicit_key_env))
    elif host == POE_HOST:
        key_env = DEFAULT_KEY_ENV
    else:
        key_env = None
        note = (
            f"no key variable named for {host}; requests are sent without a key "
            f"(set --key-env or {ENV_KEY_ENV} if this server needs one)"
        )
    if key_env and urlparse(url).scheme == "http" and host not in _LOOPBACK:
        raise ValueError("refusing to send an API key over plain HTTP to a non-loopback host; use https")
    return LLMSettings(base_url=url, api_key_env=key_env, model=str(chosen_model) if chosen_model else None, note=note)


@dataclass(frozen=True)
class ModelListing:
    """Models offered by an endpoint, optionally narrowed to reasoning models.

    ``reasoning_filter`` is ``"applied"`` when the endpoint marks reasoning models (Poe's ``reasoning`` object,
    OpenRouter's ``supported_parameters``) and only those are listed; ``"not-available"`` when the endpoint gives no
    such metadata, so every model is listed rather than guessing; ``"off"`` when the caller asked for all models.
    """

    ids: list[str]
    total: int
    reasoning_filter: str


def _is_reasoning_entry(entry: dict[str, Any]) -> bool | None:
    """True/False when the entry says whether it reasons, None when it carries no reasoning metadata."""
    known = False
    if "reasoning" in entry:
        known = True
        if isinstance(entry["reasoning"], dict) and entry["reasoning"]:
            return True
    params = entry.get("supported_parameters")
    if isinstance(params, list):
        known = True
        if any(p in ("reasoning", "include_reasoning", "reasoning_effort") for p in params):
            return True
    return False if known else None


def _serves_chat_text(entry: dict[str, Any]) -> bool:
    """Exclude image/video/audio-only models and models without chat completions, when the entry says so."""
    arch = entry.get("architecture")
    if isinstance(arch, dict) and isinstance(arch.get("output_modalities"), list):
        if "text" not in arch["output_modalities"]:
            return False
    endpoints = entry.get("supported_endpoints")
    if isinstance(endpoints, list) and endpoints and "/v1/chat/completions" not in endpoints:
        return False
    return True


def fetch_models(settings: LLMSettings, timeout: float = 15.0, reasoning_only: bool = True) -> ModelListing:
    """Fetch ``GET {base_url}/models`` and return the ids (reasoning models only by default), sorted.

    Model ids are used exactly as the endpoint names them.

    Raises:
        ModelListError: with a reason code and a one-line human message; never includes the key.
    """
    if settings.api_key_env and not settings.key_present and settings.is_poe:
        raise ModelListError("no-key", f"{settings.api_key_env} is not set; Poe needs a key to list models")
    headers = {"Accept": "application/json"}
    if settings.key_present and settings.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[settings.api_key_env]}"
    req = urllib.request.Request(f"{settings.base_url}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - scheme validated in resolve_settings
            raw = resp.read(_MAX_MODELS_BYTES + 1)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (401, 403):
            raise ModelListError("unauthorized", f"{settings.host} rejected the key (HTTP {code})") from None
        raise ModelListError("http-error", f"{settings.host} answered HTTP {code} for /models") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ModelListError("unreachable", f"cannot reach {settings.base_url}: {type(exc).__name__}") from None
    if len(raw) > _MAX_MODELS_BYTES:
        raise ModelListError("bad-response", "model list exceeds the size limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
        entries = [e for e in (payload["data"] if isinstance(payload, dict) else payload) if isinstance(e, dict) and e.get("id")]
    except (ValueError, KeyError, TypeError) as exc:
        raise ModelListError("bad-response", "response is not an OpenAI-style model list") from exc
    all_ids = sorted({str(e["id"]) for e in entries})
    if not all_ids:
        raise ModelListError("bad-response", "the endpoint returned no models")
    if not reasoning_only:
        return ModelListing(ids=all_ids, total=len(all_ids), reasoning_filter="off")
    flags = [(e, _is_reasoning_entry(e)) for e in entries]
    if all(flag is None for _, flag in flags):
        return ModelListing(ids=all_ids, total=len(all_ids), reasoning_filter="not-available")
    ids = sorted({str(e["id"]) for e, flag in flags if flag and _serves_chat_text(e)})
    if not ids:
        raise ModelListError("bad-response", "the endpoint lists no reasoning models (use --all to see every model)")
    return ModelListing(ids=ids, total=len(all_ids), reasoning_filter="applied")


def list_models(settings: LLMSettings, timeout: float = 15.0, reasoning_only: bool = False) -> list[str]:
    """Model ids from ``GET {base_url}/models`` (every model unless ``reasoning_only``).

    Raises:
        ModelListError: see ``fetch_models``.
    """
    return fetch_models(settings, timeout=timeout, reasoning_only=reasoning_only).ids
