"""Tests for endpoint settings resolution and live model listing (libweightspace.llm_config).

A loopback stub server stands in for an OpenAI-compatible endpoint; no external network is used.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from libweightspace.llm_config import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_ENV,
    ModelListError,
    list_models,
    resolve_settings,
)

SECRET = "test-secret-value-123"


class _Stub(BaseHTTPRequestHandler):
    mode = "ok"
    seen_auth: list[str | None] = []

    def log_message(self, *args: object) -> None:  # silence the test output
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        _Stub.seen_auth.append(self.headers.get("Authorization"))
        if self.path != "/v1/models":
            self.send_error(404)
            return
        if _Stub.mode == "unauthorized":
            self.send_error(401)
            return
        body = b"not json" if _Stub.mode == "garbage" else json.dumps(
            {"object": "list", "data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-a"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def stub() -> Iterator[str]:
    """Start the stub on a free loopback port and yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Stub.mode = "ok"
    _Stub.seen_auth = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_defaults_target_poe_with_poe_key_variable() -> None:
    settings = resolve_settings(env={})
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.api_key_env == DEFAULT_KEY_ENV
    assert settings.is_poe


def test_poe_key_is_not_attached_to_another_host_by_default() -> None:
    settings = resolve_settings(base_url="https://openrouter.ai/api/v1", env={DEFAULT_KEY_ENV: SECRET})
    assert settings.api_key_env is None
    assert "no key variable" in settings.note


def test_explicit_key_variable_is_used_for_other_hosts() -> None:
    settings = resolve_settings(base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY", env={})
    assert settings.api_key_env == "OPENROUTER_API_KEY"


def test_flag_beats_environment_beats_workspace() -> None:
    env = {"WEIGHTLAB_LLM_MODEL": "env-model"}
    assert resolve_settings(model="flag-model", env=env, workspace_llm={"model": "ws"}).model == "flag-model"
    assert resolve_settings(env=env, workspace_llm={"model": "ws"}).model == "env-model"
    assert resolve_settings(env={}, workspace_llm={"model": "ws"}).model == "ws"


def test_key_over_plain_http_to_remote_host_is_refused() -> None:
    with pytest.raises(ValueError, match="plain HTTP"):
        resolve_settings(base_url="http://example.com/v1", api_key_env="SOME_KEY", env={})


def test_api_key_in_workspace_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain an API key"):
        resolve_settings(workspace_llm={"api_key": SECRET}, env={})


@pytest.mark.parametrize("bad", ["lower_case", "1STARTS_WITH_DIGIT", "HAS-DASH", "X" * 80])
def test_invalid_key_variable_names_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="key variable name"):
        resolve_settings(api_key_env=bad, env={})


def test_describe_never_contains_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_KEY_ENV, SECRET)
    summary = resolve_settings(env={}).describe()
    assert summary["key_present"] is True
    assert SECRET not in json.dumps(summary)


def test_list_models_returns_sorted_unique_ids_and_sends_key(stub: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUB_KEY", SECRET)
    settings = resolve_settings(base_url=stub, api_key_env="STUB_KEY", env={})
    assert list_models(settings) == ["model-a", "model-b"]
    assert _Stub.seen_auth == [f"Bearer {SECRET}"]


def test_list_models_without_key_variable_sends_no_authorization(stub: str) -> None:
    settings = resolve_settings(base_url=stub, env={})
    assert list_models(settings) == ["model-a", "model-b"]
    assert _Stub.seen_auth == [None]


def test_unauthorized_is_reported_with_reason(stub: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _Stub.mode = "unauthorized"
    monkeypatch.setenv("STUB_KEY", SECRET)
    with pytest.raises(ModelListError) as info:
        list_models(resolve_settings(base_url=stub, api_key_env="STUB_KEY", env={}))
    assert info.value.reason == "unauthorized"
    assert SECRET not in str(info.value)


def test_garbage_response_is_reported(stub: str) -> None:
    _Stub.mode = "garbage"
    with pytest.raises(ModelListError) as info:
        list_models(resolve_settings(base_url=stub, env={}))
    assert info.value.reason == "bad-response"


def test_unreachable_endpoint_is_reported() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ModelListError) as info:
        list_models(resolve_settings(base_url=f"http://127.0.0.1:{port}/v1", env={}), timeout=2.0)
    assert info.value.reason == "unreachable"


def test_poe_without_key_fails_fast_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_KEY_ENV, raising=False)
    with pytest.raises(ModelListError) as info:
        list_models(resolve_settings(env={}))
    assert info.value.reason == "no-key"


def test_backend_requires_a_model() -> None:
    with pytest.raises(ValueError, match="no LLM model selected"):
        resolve_settings(env={}).backend()
