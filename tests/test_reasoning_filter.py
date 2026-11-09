"""Tests for the reasoning-model filter in fetch_models (owner requirement: list only reasoning models, by live name).

The Poe fixture mirrors the shape of api.poe.com/v1/models on 2026-11-08: every entry has a ``reasoning`` key, which
is an object for reasoning models and null otherwise.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from libweightspace.llm_config import ModelListError, fetch_models, resolve_settings

POE_STYLE = [
    {"id": "gpt-5.4-mini", "reasoning": {"budget": None, "required": False, "supports_reasoning_effort": True},
     "architecture": {"output_modalities": ["text", "image"]}, "supported_endpoints": ["/v1/chat/completions"]},
    {"id": "claude-haiku-4.5", "reasoning": {"budget": None, "required": False, "supports_reasoning_effort": False},
     "architecture": {"output_modalities": ["text"]}, "supported_endpoints": ["/v1/chat/completions"]},
    {"id": "gpt-4o-mini", "reasoning": None, "architecture": {"output_modalities": ["text"]},
     "supported_endpoints": ["/v1/chat/completions"]},
    {"id": "image-maker", "reasoning": {"budget": None, "required": False, "supports_reasoning_effort": False},
     "architecture": {"output_modalities": ["image"]}, "supported_endpoints": ["/v1/chat/completions"]},
]
OPENROUTER_STYLE = [
    {"id": "vendor/thinker", "supported_parameters": ["max_tokens", "reasoning", "include_reasoning"]},
    {"id": "vendor/plain", "supported_parameters": ["max_tokens", "temperature"]},
]
NO_METADATA = [{"id": "llama3.2"}, {"id": "qwen3:8b"}]


class _Stub(BaseHTTPRequestHandler):
    entries: list[dict[str, Any]] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = json.dumps({"object": "list", "data": _Stub.entries}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def stub() -> Iterator[str]:
    """Loopback endpoint whose /models returns ``_Stub.entries``."""
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_poe_style_keeps_only_text_reasoning_models_with_live_names(stub: str) -> None:
    _Stub.entries = POE_STYLE
    listing = fetch_models(resolve_settings(base_url=stub, env={}))
    assert listing.ids == ["claude-haiku-4.5", "gpt-5.4-mini"]
    assert listing.total == 4 and listing.reasoning_filter == "applied"


def test_openrouter_style_uses_supported_parameters(stub: str) -> None:
    _Stub.entries = OPENROUTER_STYLE
    listing = fetch_models(resolve_settings(base_url=stub, env={}))
    assert listing.ids == ["vendor/thinker"] and listing.reasoning_filter == "applied"


def test_endpoint_without_metadata_lists_everything_and_says_so(stub: str) -> None:
    _Stub.entries = NO_METADATA
    listing = fetch_models(resolve_settings(base_url=stub, env={}))
    assert listing.ids == ["llama3.2", "qwen3:8b"] and listing.reasoning_filter == "not-available"


def test_all_mode_lists_every_model(stub: str) -> None:
    _Stub.entries = POE_STYLE
    listing = fetch_models(resolve_settings(base_url=stub, env={}), reasoning_only=False)
    assert len(listing.ids) == 4 and listing.reasoning_filter == "off"


def test_no_reasoning_models_is_an_explicit_error(stub: str) -> None:
    _Stub.entries = [{"id": "plain", "reasoning": None}]
    with pytest.raises(ModelListError, match="no reasoning models"):
        fetch_models(resolve_settings(base_url=stub, env={}))
