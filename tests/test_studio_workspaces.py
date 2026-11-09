"""Studio tests for the workspace manager (CR2) and the frontier-only proposer policy (REQ-241).

A Studio starts without a workspace, creates and switches workspaces through jobs, refuses anything that looks like a
path, and runs the whole Hugging Face flow (check -> verified download -> workspace -> facts) against a tiny real
Llama checkpoint served by the local fake Hub. Nothing touches the network or the real Studio home.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from neuroforge.studio import StudioServer, StudioState
from neuroforge.templates import DEMO_INTENT
from neuroforge.workspace_manager import HubClient, StudioHome

from .fake_hub import SHA, FakeHub, FakeRepo
from .test_studio_security import call


def wait(srv: StudioServer, job_id: str, limit: float = 240) -> dict[str, Any]:
    deadline = time.time() + limit
    job: dict[str, Any] = {}
    while time.time() < deadline:
        job = json.loads(call(srv, "GET", f"/api/jobs/{job_id}")[2])
        if job["state"] not in ("running", "awaiting_approval"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job still running: {job}")


@pytest.fixture()
def empty_studio(tmp_path: Path) -> Iterator[StudioServer]:
    state = StudioState.without_workspace(fetch_models_on_start=False, home=StudioHome(tmp_path / "home"))
    srv = StudioServer(state, port=0)
    srv.start_background()
    try:
        yield srv
    finally:
        srv.close()


def create(srv: StudioServer, **body: Any) -> dict[str, Any]:
    status, _, raw = call(srv, "POST", "/api/workspaces/create", {"device": "cpu", **body})
    assert status == 202, raw
    return wait(srv, json.loads(raw)["job_id"])


def test_start_page_without_workspace(empty_studio: StudioServer) -> None:
    status = json.loads(call(empty_studio, "GET", "/api/status")[2])
    assert status["workspace"] is None and "llm" in status
    listing = json.loads(call(empty_studio, "GET", "/api/workspaces")[2])
    assert listing["workspaces"] == [] and listing["current"] is None
    assert any(d["device"] == "cpu" for d in listing["devices"])
    code, _, raw = call(empty_studio, "GET", "/api/intent")
    assert code == 409 and b"open or create a workspace" in raw
    page = call(empty_studio, "GET", "/")[2]
    assert b"Weight Engineering Studio" in page


def test_create_demo_then_switch_keeps_the_token(empty_studio: StudioServer) -> None:
    token = empty_studio.state.token
    first = create(empty_studio, kind="demo", name="first")
    assert first["state"] == "done" and first["result"]["facts"]["toy"] is True
    assert create(empty_studio, kind="demo", name="second")["state"] == "done"
    assert json.loads(call(empty_studio, "GET", "/api/status")[2])["workspace"] == "second"
    rows = json.loads(call(empty_studio, "GET", "/api/workspaces")[2])["workspaces"]
    target = next(r for r in rows if r["name"] == "first")
    status, _, raw = call(empty_studio, "POST", "/api/workspaces/open", {"id": target["id"]})
    assert status == 202
    assert wait(empty_studio, json.loads(raw)["job_id"])["state"] == "done"
    assert json.loads(call(empty_studio, "GET", "/api/status")[2])["workspace"] == "first"
    assert empty_studio.state.token == token
    assert json.loads(call(empty_studio, "GET", "/api/intent")[2])["exists"] is True
    status, _, raw = call(empty_studio, "POST", "/api/workspaces/create", {"kind": "demo", "name": "first"})
    assert status == 409 and b"already exists" in raw


def test_a_preview_does_not_survive_a_workspace_switch(empty_studio: StudioServer) -> None:
    assert create(empty_studio, kind="demo", name="first")["state"] == "done"
    body = {"name": "bang", "positives": ["wow!!!", "yes!!!"], "negatives": ["wow...", "yes..."]}
    status, _, raw = call(empty_studio, "POST", "/api/direction/preview", body)
    assert status == 202
    plan = wait(empty_studio, json.loads(raw)["job_id"])["result"]
    kept = empty_studio.state._plans[plan["plan_id"]]
    assert create(empty_studio, kind="demo", name="second")["state"] == "done"  # switches to "second"
    # Switching drops the previews measured on the other workspace...
    assert call(empty_studio, "POST", "/api/plans/apply", {"plan_id": plan["plan_id"]})[0] == 400
    # ...and a preview that is still around is refused all the same: its workspace is checked, not just the head.
    empty_studio.state._plans[plan["plan_id"]] = kept
    status, _, raw = call(empty_studio, "POST", "/api/plans/apply", {"plan_id": plan["plan_id"]})
    assert status == 409 and b"another workspace" in raw
    assert json.loads(call(empty_studio, "GET", "/api/history")[2])["commits"] == []


def test_routes_refuse_paths(empty_studio: StudioServer) -> None:
    for body in ({"id": "../../etc"}, {"id": "C:\\Windows"}, {"id": "0" * 16}):
        assert call(empty_studio, "POST", "/api/workspaces/open", body)[0] == 400
    bad_creates = (
        {"kind": "folder", "name": "w", "folder": "../outside"},
        {"kind": "folder", "name": "w", "folder": "C:\\Users"},
        {"kind": "demo", "name": "../escape"},
        {"kind": "script", "name": "w"},
        {"kind": "hub", "name": "w", "check_id": "nope"},
    )
    for body in bad_creates:
        status, _, raw = call(empty_studio, "POST", "/api/workspaces/create", {"device": "cpu", **body})
        if status == 202:  # name validation happens inside the job: it must fail without creating anything
            assert wait(empty_studio, json.loads(raw)["job_id"])["state"] == "failed"
        else:
            assert status == 400, (body, raw)
    home = empty_studio.state.home
    assert not (home.root.parent / "escape").exists()
    assert [p.name for p in home.workspaces_dir.iterdir()] == []


def _tiny_llama_files(folder: Path) -> dict[str, bytes]:
    """A real (random) two-layer Llama checkpoint with a word-level tokenizer, saved as safetensors."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    words = ["[UNK]", "[PAD]"] + [f"w{i}" for i in range(30)] + list("abcdefghijklmnopqrstuvwxyz") + ["x", "y", "z"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    config = transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(folder, safe_serialization=True)
    fast.save_pretrained(folder)
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.fixture()
def tiny_hub(tmp_path: Path) -> Iterator[FakeHub]:
    files = _tiny_llama_files(tmp_path / "src-model")
    hub = FakeHub({"org/tiny-llama": FakeRepo(files=files, lfs={n for n in files if n.endswith(".safetensors")})})
    try:
        yield hub
    finally:
        hub.close()


def test_hub_check_download_workspace_and_facts(empty_studio: StudioServer, tiny_hub: FakeHub) -> None:
    empty_studio.state._hub = HubClient(tiny_hub.url)
    status, _, raw = call(empty_studio, "POST", "/api/models/check", {"repo": "org/tiny-llama", "revision": "main", "device": "cpu"})
    assert status == 200, raw
    check = json.loads(raw)
    assert check["revision"] == SHA and check["adapter"] == "qwen3_8" and check["license"] == "apache-2.0"
    assert check["download_bytes"] > 0 and check["weights_bytes_bf16"] > 0 and "free_bytes" in check
    job = create(empty_studio, kind="hub", name="tiny", check_id=check["check_id"])
    assert job["state"] == "done", job
    home = empty_studio.state.home
    cfg = yaml.safe_load((home.workspaces_dir / "tiny" / "neuroforge.yaml").read_text(encoding="utf-8"))
    assert cfg["revision"] == SHA and cfg["model"] == f"org/tiny-llama@{SHA}"
    assert Path(cfg["model_path"]).resolve().is_relative_to(home.models_dir.resolve())
    facts = json.loads(call(empty_studio, "GET", "/api/workspace/facts")[2])
    assert facts["revision"] == SHA and facts["layers"] == 2 and facts["toy"] is False
    folders = json.loads(call(empty_studio, "GET", "/api/workspaces")[2])["model_folders"]
    assert [f["revision"] for f in folders] == [SHA]

    # REQ-241: offline templates are refused on a real model; the frontier model is required.
    (home.workspaces_dir / "tiny" / "intent.yaml").write_text(yaml.safe_dump(DEMO_INTENT), encoding="utf-8")
    status, _, raw = call(empty_studio, "POST", "/api/loop", {"use_llm": False})
    assert status == 400 and b"frontier model" in raw
    status, _, raw = call(empty_studio, "POST", "/api/loop", {"use_llm": True})
    assert status == 409 and b"choose a model" in raw


def test_cancelled_download_leaves_no_workspace(empty_studio: StudioServer, tmp_path: Path) -> None:
    files = {"config.json": json.dumps({"model_type": "llama"}).encode(), "model.safetensors": b"\x01" * 200_000}
    hub = FakeHub({"org/slow": FakeRepo(files=files, lfs={"model.safetensors"})}, slow_bytes_per_read=1000, delay_per_chunk=0.02)
    try:
        empty_studio.state._hub = HubClient(hub.url)
        check = json.loads(call(empty_studio, "POST", "/api/models/check", {"repo": "org/slow", "device": "cpu"})[2])
        status, _, raw = call(empty_studio, "POST", "/api/workspaces/create",
                              {"kind": "hub", "name": "slow", "device": "cpu", "check_id": check["check_id"]})
        job_id = json.loads(raw)["job_id"]
        time.sleep(0.5)
        assert call(empty_studio, "POST", f"/api/jobs/{job_id}/cancel", {})[0] == 200
        job = wait(empty_studio, job_id)
    finally:
        hub.close()
    assert job["state"] == "cancelled" and "partial files were removed" in job["error"]
    home = empty_studio.state.home
    assert not (home.workspaces_dir / "slow").exists()
    assert list(home.models_dir.iterdir()) == []
