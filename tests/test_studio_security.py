"""Security and end-to-end tests for the hardened Studio (threat model T1-T4, REQ-008/009/011/012).

Every test runs a real StudioServer on an ephemeral loopback port against a demo workspace; requests are sent with
http.client so Host, Origin and Content-Type can be set exactly as a hostile page or a rebinding attack would.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from typer.testing import CliRunner

from libweightspace.loading import load_model
from neuroforge.cli import app as nf_app
from neuroforge.studio import JobManager, StudioError, StudioServer, StudioState

SECRET = "studio-test-secret-456"


@pytest.fixture()
def studio(tmp_path: Path) -> Iterator[tuple[StudioServer, Path]]:
    """A demo workspace served by a background Studio; closed after the test."""
    ws = tmp_path / "demo"
    result = CliRunner().invoke(nf_app, ["init", "-w", str(ws), "--demo"])
    assert result.exit_code == 0, result.output
    state = StudioState.from_workspace(ws, fetch_models_on_start=False)
    srv = StudioServer(state, host="127.0.0.1", port=0)
    srv.start_background()
    try:
        yield srv, ws
    finally:
        srv.close()


def call(
    srv: StudioServer,
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
    token: bool = True,
    raw: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One request with explicit headers; returns (status, lower-cased headers, body)."""
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=120)
    hdrs = {"Host": f"127.0.0.1:{srv.port}"}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
        if token:
            hdrs["X-Studio-Token"] = srv.state.token
    hdrs.update(headers or {})
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    out = resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    conn.close()
    return out


def test_model_list_refresh_needs_the_token(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    srv.state.refresh_models = lambda background=False: None  # type: ignore[method-assign]  # no network in tests
    assert call(srv, "GET", "/api/llm/models?refresh=1")[0] == 403  # e.g. an <img> on another site
    assert call(srv, "GET", "/api/llm/models?refresh=1", headers={"X-Studio-Token": srv.state.token})[0] == 200
    assert call(srv, "GET", "/api/llm/models")[0] == 200  # reading the cached list stays open to the page


def test_index_has_token_nonce_csp_and_no_cors(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    status, headers, body = call(srv, "GET", "/")
    html = body.decode()
    assert status == 200
    assert srv.state.token in html and "__STUDIO_TOKEN__" not in html and "__CSP_NONCE__" not in html
    csp = headers["content-security-policy"]
    assert "script-src 'nonce-" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline' ;" not in csp
    assert headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in headers


def test_foreign_host_is_refused(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    for path in ("/", "/api/status"):
        status, _, body = call(srv, "GET", path, headers={"Host": f"evil.example:{srv.port}"})
        assert status == 421, path
        assert srv.state.token.encode() not in body


def test_foreign_origin_and_preflight_are_refused(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    status, headers, _ = call(srv, "POST", "/api/probe", {"top": 3}, headers={"Origin": "http://evil.example"})
    assert status == 403 and "access-control-allow-origin" not in headers
    status, headers, _ = call(
        srv, "OPTIONS", "/api/export", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert status == 403 and "access-control-allow-origin" not in headers


def test_post_without_token_or_with_text_plain_is_refused(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    status, _, _ = call(srv, "POST", "/api/export", {"format": "safetensors"}, token=False)
    assert status == 403
    status, _, _ = call(
        srv,
        "POST",
        "/api/export",
        raw=json.dumps({"format": "safetensors"}).encode(),
        headers={"Content-Type": "text/plain", "X-Studio-Token": srv.state.token},
    )
    assert status == 415
    assert not (ws / "exports").exists()


def test_wrong_or_non_ascii_token_is_refused_not_crashing(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    for bad in ("x" * 43, srv.state.token[:-1], "tökén-" + srv.state.token[7:]):
        status, _, body = call(
            srv, "POST", "/api/export", {"format": "safetensors"}, token=False,
            headers={"X-Studio-Token": bad},
        )
        assert status == 403, (bad, status)
        assert b"Traceback" not in body
    assert not (ws / "exports").exists()


def test_body_cap_is_enforced_before_parsing(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    big = {"yaml": "x" * (srv.state.max_body_bytes + 10)}
    status, _, body = call(srv, "POST", "/api/intent/save", big)
    assert status == 413 and b"exceeds" in body


def test_clients_cannot_choose_where_files_go(studio: tuple[StudioServer, Path], tmp_path: Path) -> None:
    srv, ws = studio
    evil = tmp_path / "evil_target"
    status, _, body = call(
        srv, "POST", "/api/export", {"format": "safetensors", "output_dir": str(evil), "harness_base_dir": str(evil)}
    )
    assert status == 200, body
    result = json.loads(body)
    assert result["path"].startswith("exports")
    assert (ws / result["path"]).is_file()
    assert not evil.exists()
    status, _, body = call(srv, "POST", "/api/checkout", {"ref": "../../etc"})
    assert status == 400 and b"invalid commit reference" in body and b"Traceback" not in body


def test_status_and_models_never_reveal_the_key(studio: tuple[StudioServer, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    srv, _ = studio
    monkeypatch.setenv("POE_API_KEY", SECRET)
    for path in ("/api/status", "/api/llm/models"):
        status, _, body = call(srv, "GET", path)
        assert status == 200 and SECRET.encode() not in body
    assert json.loads(call(srv, "GET", "/api/status")[2])["llm"]["key_present"] is True


def test_model_selection_is_validated_and_persisted_without_a_key(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    status, _, _ = call(srv, "POST", "/api/llm/select", {"model": "bad model; rm -rf /"})
    assert status == 400
    status, _, body = call(srv, "POST", "/api/llm/select", {"model": "some-model-1"})
    assert status == 200, body
    cfg = yaml.safe_load((ws / "neuroforge.yaml").read_text(encoding="utf-8"))
    assert cfg["llm"] == {"model": "some-model-1"}


def test_llm_actions_without_a_model_explain_what_to_do(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    status, _, body = call(srv, "POST", "/api/intent/draft", {"description": "stop looping"})
    assert status == 409 and b"choose a model" in body
    status, _, body = call(srv, "POST", "/api/loop", {"use_llm": True})
    assert status == 409


def test_invalid_intent_is_rejected_with_a_reason(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    before = (ws / "intent.yaml").read_text(encoding="utf-8")
    status, _, body = call(srv, "POST", "/api/intent/save", {"yaml": "intent_id: 3\n"})
    assert status == 400 and b"intent is invalid" in body
    assert (ws / "intent.yaml").read_text(encoding="utf-8") == before


def test_guided_flow_probe_repair_history_and_exact_restore(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    status, _, body = call(srv, "POST", "/api/probe", {"top": 3, "columns": 4})
    assert status == 200
    layers = json.loads(body)["layers"]
    # The demo plants a defect in layer 1, columns 5, 12 and 19: the diagnosis must find exactly that.
    assert layers[0]["layer"] == 1
    assert {5, 12, 19} <= set(layers[0]["culprit_columns"])

    status, _, body = call(srv, "POST", "/api/loop", {"iterations": 3, "candidates": 3, "use_llm": False})
    assert status == 202
    job_id = json.loads(body)["job_id"]
    deadline = time.time() + 240
    job: dict[str, Any] = {}
    while time.time() < deadline:
        job = json.loads(call(srv, "GET", f"/api/jobs/{job_id}")[2])
        if job["state"] != "running":
            break
        time.sleep(0.5)
    assert job["state"] == "done", job
    result = job["result"]
    assert result["proposer"] == "offline demo templates (toy model, no LLM)"
    committed = [r for r in result["rows"] if r.get("status") == "committed"]
    assert committed and all(r["target_score"] is not None for r in committed)

    history = json.loads(call(srv, "GET", "/api/history")[2])
    assert len(history["commits"]) == len(committed) and history["head"] != "ROOT"

    status, _, body = call(srv, "POST", "/api/checkout", {"ref": "ROOT"})
    assert status == 200 and json.loads(body)["head"] == "ROOT"
    cfg = yaml.safe_load((ws / "neuroforge.yaml").read_text(encoding="utf-8"))
    base = load_model(cfg["model"], cfg["device"], cfg["demo_defect"]).model.state_dict()
    restored = srv.state.loaded.model.state_dict()
    assert all(torch.equal(base[k], restored[k]) for k in base)


def test_only_one_job_runs_at_a_time() -> None:
    jobs = JobManager()
    release = threading.Event()
    first = jobs.start("slow", lambda _job_id: (release.wait(10), {"ok": True})[1])
    with pytest.raises(StudioError) as info:
        jobs.start("second", lambda _job_id: {})
    assert info.value.status == 409
    release.set()
    for _ in range(100):
        if jobs.get(first)["state"] == "done":
            break
        time.sleep(0.05)
    assert jobs.get(first)["result"] == {"ok": True}
