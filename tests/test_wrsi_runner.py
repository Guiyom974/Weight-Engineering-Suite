"""Tests for the shared W-RSI runner and the Studio self-improvement view (CR10).

Each run uses the toy demo model with the anchor vault as a real child process, so these tests exercise the same path
as ``wrsi start`` and the Studio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from libweightspace.candidates import CandidateSource, TemplateCandidateSource, nullification_script
from libweightspace.loading import LoadedModel, load_model
from neuroforge.studio import StudioServer
from wrsi.runner import Guardrails, SessionInputs, rollback, rollback_preview, run_session, status, timeline
from wrsi.scientist_agent import GymTemplateProposer
from wrsi.workspace import WRSIWorkspace

from .test_studio_controls import studio, wait_job  # noqa: F401 - fixture re-export
from .test_studio_security import call

TENSOR = "model.layers.1.mlp.down_proj.weight"


def templates(_loaded: LoadedModel) -> tuple[CandidateSource, CandidateSource]:
    return TemplateCandidateSource(), GymTemplateProposer()


def run(root: Path, steps: int = 1, **guard: Any) -> dict[str, Any]:
    return run_session(root, SessionInputs(), Guardrails(meta_fraction=0.0, **guard), templates, steps)


def test_first_step_promotes_and_the_timeline_records_every_candidate(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result["generation"] == 1 and result["ended"] == "limit"
    entry = timeline(tmp_path)[-1]
    assert entry["promoted"] is True and entry["anchor_kl"] is not None and entry["anchor_kl"] < 0.08
    assert entry["trials"] and all(t["reason"] or t["promoted"] for t in entry["trials"])
    assert any("Sparsity violation" in (t["reason"] or "") for t in entry["trials"])  # gate and measured value


def test_forgetting_check_blocks_promotion(tmp_path: Path) -> None:
    # The repair makes 'x' after 'x' much less likely: a held-out prompt that expects 'x' must block it.
    result = run(tmp_path, forgetting=[("ab x", "x")], max_forgetting_drop=0.0)
    assert result["generation"] == 0
    reasons = [t["reason"] or "" for t in timeline(tmp_path)[-1]["trials"]]
    assert any(r.startswith("Forgetting check: held-out score dropped") for r in reasons)


def test_protected_tensors_block_every_candidate(tmp_path: Path) -> None:
    result = run(tmp_path, protected_tensors=("layers.1.mlp.down_proj",))
    assert result["generation"] == 0
    assert any("Protected tensor" in (t["reason"] or "") for t in timeline(tmp_path)[-1]["trials"])


def test_pause_stops_after_the_current_generation_with_state_saved(tmp_path: Path) -> None:
    result = run_session(tmp_path, SessionInputs(), Guardrails(meta_fraction=0.0), templates, 3, should_pause=lambda: True)
    assert result["ended"] == "paused" and result["steps_run"] == 1
    saved = status(tmp_path)
    assert saved["steps_run"] == 1 and saved["generation"] == result["generation"]


def test_stop_promotes_nothing_from_the_current_generation(tmp_path: Path) -> None:
    result = run_session(tmp_path, SessionInputs(), Guardrails(meta_fraction=0.0), templates, 3, should_stop=lambda: True)
    assert result["ended"] == "stopped" and result["generation"] == 0


def test_rollback_preview_then_bit_exact_rollback(tmp_path: Path) -> None:
    assert rollback_preview(tmp_path) is None
    run(tmp_path)
    preview = rollback_preview(tmp_path)
    assert preview is not None and preview["generation"] == 0 and TENSOR in preview["tensors"]
    assert rollback(tmp_path)["generation"] == 0
    m0 = load_file(str(WRSIWorkspace(tmp_path).dir / "vault" / "m0.safetensors"))
    reloaded = load_model("toy:qwen3_8", demo_defect=True)
    WRSIWorkspace(tmp_path).apply_overlay(reloaded.model)
    assert torch.equal(dict(reloaded.model.named_parameters())[TENSOR], m0[TENSOR])
    assert timeline(tmp_path)[-1]["kind"] == "rollback"
    with pytest.raises(ValueError, match="nothing to roll back"):
        rollback(tmp_path)


def test_a_rollback_takes_no_step_number_from_the_generations(tmp_path: Path) -> None:
    run(tmp_path)
    rollback(tmp_path)
    run(tmp_path)
    entries = timeline(tmp_path)
    assert [e["kind"] for e in entries][1] == "rollback" and entries[1]["step"] is None
    steps = [e["step"] for e in entries if e["kind"] != "rollback"]
    assert len(steps) == len(set(steps)) == 2


class FakeFrontier:
    """A frontier backend stand-in that proposes the attractor repair and records the system prompt it got."""

    name = "fake-frontier"

    def __init__(self) -> None:
        self.systems: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.systems.append(system)
        return json.dumps({"candidates": [{"rationale": "remove the attractor direction", "python_script": nullification_script(3)}]})


def test_studio_self_improvement_with_a_frontier_model(studio: tuple[StudioServer, Path], monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    srv, _ = studio
    fake = FakeFrontier()
    srv.state.selected_model = "frontier-x"
    monkeypatch.setattr(srv.state, "_backend", lambda: fake)
    status_code, _, raw = call(srv, "POST", "/api/selfimprove/start", {"generations": 1, "use_llm": True})
    assert status_code == 202, raw
    job = wait_job(srv, json.loads(raw)["job_id"])
    assert job["state"] == "done" and job["result"]["proposer"].startswith("frontier model: frontier-x")
    assert job["result"]["generation"] == 1
    assert fake.systems and all("not to you" in s for s in fake.systems)  # never told the weights are its own


def test_studio_self_improvement_view(studio: tuple[StudioServer, Path]) -> None:  # noqa: F811
    srv, ws = studio
    assert json.loads(call(srv, "GET", "/api/selfimprove/status")[2])["exists"] is False
    body = {"generations": 2, "use_llm": False, "forgetting": "hello world =>  ", "max_kl": 0.08}
    assert call(srv, "POST", "/api/selfimprove/start", body)[0] == 400  # a forgetting line without an answer
    body["forgetting"] = ""
    status_code, _, raw = call(srv, "POST", "/api/selfimprove/start", body)
    assert status_code == 202, raw
    job_id = json.loads(raw)["job_id"]
    assert call(srv, "POST", f"/api/jobs/{job_id}/pause", {})[0] in (200, 409)  # 409 when it already finished
    job = wait_job(srv, job_id)
    assert job["state"] == "done" and job["result"]["proposer"].startswith("offline demo templates")
    view = json.loads(call(srv, "GET", "/api/selfimprove/status")[2])
    assert view["exists"] and view["timeline"] and view["guardrails"]["max_kl"] == 0.08
    assert call(srv, "POST", "/api/selfimprove/rollback", {"confirm": False})[0] == 400
    if view["rollback"] is not None:
        assert call(srv, "POST", "/api/selfimprove/rollback", {"confirm": True})[0] == 200
    # The edit history of the Edit view is untouched by self-improvement.
    assert json.loads(call(srv, "GET", "/api/history")[2])["commits"] == []
