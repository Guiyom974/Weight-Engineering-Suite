"""Text-completion backends for reasoners and self-editing proposers.

Backends implement ``complete(system, prompt) -> str``:
    - ``OpenAICompatibleBackend``: ``POST {base_url}/chat/completions`` for any OpenAI-compatible server
      (Poe, OpenRouter, Ollama ``/v1``, LM Studio, llama.cpp server, vLLM, OpenAI). Endpoint, key variable and
      model come from ``libweightspace.llm_config``; a key is never sent over plain HTTP to a non-loopback host.
    - ``InProcessModelBackend``: greedy generation with the live in-process model (W-RSI ``--proposer self``).
    - ``CallableBackend``: wraps a Python function (tests, scripted reasoners).

Every network call has a timeout; errors become ``LLMBackendError`` and never include secrets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

import torch
import torch.nn as nn

from .prober import model_logits

logger = logging.getLogger(__name__)
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/@-]{1,128}$")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class LLMBackendError(RuntimeError):
    """A backend call failed (HTTP error, timeout, malformed response)."""


class LLMBackend(Protocol):
    """Minimal completion interface."""

    name: str

    def complete(self, system: str, prompt: str) -> str:
        """Returns the model's text reply."""
        ...


class CallableBackend:
    """Backend backed by a Python callable ``fn(system, prompt) -> str``."""

    name = "callable"

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn

    def complete(self, system: str, prompt: str) -> str:
        """Delegates to the wrapped function."""
        return self.fn(system, prompt)


class InProcessModelBackend:
    """Autoregressive text generation using the live in-process PyTorch model (``M_t``).

    Used by W-RSI when ``proposer="self"`` so that weight updates applied to ``model`` in-place immediately
    govern subsequent forward passes during self-edit generation.
    """

    name = "in-process"

    def __init__(self, model: nn.Module, tokenizer: Any, max_new_tokens: int = 128):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def complete(self, system: str, prompt: str) -> str:
        """Runs greedy autoregressive decoding on ``self.model`` and decodes the generated tokens."""
        full_text = f"{system}\n\n{prompt}"
        encoded = self.tokenizer(full_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        generated: list[int] = []
        cur_ids = input_ids
        for _ in range(self.max_new_tokens):
            logits = model_logits(self.model(input_ids=cur_ids))
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(next_token)
            next_tensor = torch.tensor([[next_token]], dtype=cur_ids.dtype, device=device)
            cur_ids = torch.cat([cur_ids, next_tensor], dim=1)
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(generated))
        return "".join(chr(max(32, min(126, t))) for t in generated)


def _check_model(model: str) -> str:
    if not _MODEL_RE.match(model):
        raise ValueError(f"invalid model name '{model}'")
    return model


class OpenAICompatibleBackend:
    """Chat-completions client for any OpenAI-compatible server (local or remote)."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        max_response_bytes: int = 8 * 1024 * 1024,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        if api_key_env and parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("refusing to send an API key over plain HTTP to a non-loopback host")
        self.base_url = base_url.rstrip("/")
        self.model = _check_model(model)
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_response_bytes = max_response_bytes

    def build_request(self, system: str, prompt: str) -> urllib.request.Request:
        """HTTP request object (exposed for testing; the key is only placed in the header)."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )

    def complete(self, system: str, prompt: str) -> str:
        """POSTs the chat request and returns ``choices[0].message.content``.

        Raises:
            LLMBackendError: the named key variable is empty for a non-local host (checked before any request), or
                the request or its response failed.
        """
        host = urlparse(self.base_url).hostname
        if self.api_key_env and not os.environ.get(self.api_key_env) and host not in _LOOPBACK_HOSTS:
            # Fail with the actionable cause instead of sending an unauthenticated request (a bare HTTP 400/401).
            raise LLMBackendError(f"{self.api_key_env} is not set; set it in the shell that runs this command")
        req = self.build_request(system, prompt)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - http(s) checked in __init__
                raw = resp.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            exc.close()  # HTTPError carries the open response; release its socket
            raise LLMBackendError(f"HTTP {exc.code} from {self.base_url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMBackendError(f"request to {self.base_url} failed: {exc}") from exc
        if len(raw) > self.max_response_bytes:
            raise LLMBackendError("response exceeds size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError("malformed chat-completions response") from exc
        if not isinstance(content, str):
            raise LLMBackendError("chat-completions content is not text")
        return content


def make_backend(
    kind: str,
    model: str | None = None,
    base_url: str | None = None,
    model_instance: nn.Module | None = None,
    tokenizer: Any = None,
    api_key_env: str | None = None,
    workspace_llm: Mapping[str, Any] | None = None,
) -> LLMBackend:
    """Factory used by the CLIs and the Studio.

    Args:
        kind: ``llm`` (any OpenAI-compatible endpoint; ``openai`` is accepted as an alias) or ``self``
            (in-process ``M_t`` for W-RSI).
        model: model id for ``llm`` (falls back to ``WEIGHTLAB_LLM_MODEL`` and the workspace config).
        base_url: endpoint for ``llm`` (falls back to ``WEIGHTLAB_LLM_BASE_URL``, the workspace config, then Poe).
        model_instance: live ``nn.Module`` when ``kind == "self"``.
        tokenizer: tokenizer paired with ``model_instance`` when ``kind == "self"``.
        api_key_env: name of the variable holding the key (see ``llm_config`` for the defaults and safety rule).
        workspace_llm: the ``llm:`` section of ``neuroforge.yaml``.

    Raises:
        ValueError: unknown kind, missing model, or an unsafe URL/key combination.
    """
    if kind == "self":
        if model_instance is None or tokenizer is None:
            raise ValueError("model_instance and tokenizer are required for the 'self' backend")
        return InProcessModelBackend(model_instance, tokenizer)
    if kind in ("llm", "openai"):
        from .llm_config import resolve_settings  # local import: llm_config imports this module

        settings = resolve_settings(base_url=base_url, api_key_env=api_key_env, model=model, workspace_llm=workspace_llm)
        return settings.backend()
    raise ValueError(f"unknown backend '{kind}' (choose llm or self)")
