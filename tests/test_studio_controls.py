"""Tests for the Studio control features: dry run, protected tensors, approve-before-commit, cancel (CR5, CR7),
original-vs-current answers and persisted results (CR1, CR3, CR4), examples, and the edit bundle (CR8)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from neuroforge.cli import app as nf_app
from neuroforge.schemas import IntentSpec
from neuroforge.studio import StudioServer, StudioState
from neuroforge.weight_git import WeightGitRepository

from .test_studio_security import call


@pytest.fixture()
def studio(tmp_path: Path) -> Iterator[tuple[StudioServer, Path]]:
    """Demo workspace served by a background Studio."""
    ws = tmp_path / "demo"
    assert CliRunner().invoke(nf_app, ["init", "-w", str(ws), "--demo"]).exit_code == 0
    srv = StudioServer(StudioState.from_workspace(ws, fetch_models_on_start=False), port=0)
    srv.start_background()
    try:
        yield srv, ws
    finally:
        srv.close()


def wait_job(srv: StudioServer, job_id: str, until: tuple[str, ...] = ("done", "failed"), limit: float = 240) -> dict[str, Any]:
    """Poll a job until its state is one of ``until``."""
    deadline = time.time() + limit
    job: dict[str, Any] = {}
    while time.time() < deadline:
        job = json.loads(call(srv, "GET", f"/api/jobs/{job_id}")[2])
        if job["state"] in until:
            return job
        time.sleep(0.3)
    raise AssertionError(f"job did not reach {until}: {job}")


def start(srv: StudioServer, **options: Any) -> str:
    body = {"iterations": 3, "candidates": 3, "use_llm": False, **options}
    status, _, raw = call(srv, "POST", "/api/loop", body)
    assert status == 202, raw
    return str(json.loads(raw)["job_id"])


def test_dry_run_leaves_history_and_weights_untouched(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    before = {k: v.clone() for k, v in srv.state.loaded.model.state_dict().items()}
    job = wait_job(srv, start(srv, dry_run=True))
    assert job["state"] == "done" and job["result"]["dry_run"] is True
    assert WeightGitRepository(ws).list_commits() == []
    after = srv.state.loaded.model.state_dict()
    assert all((before[k] == after[k]).all() for k in before)


def test_protected_tensor_rejects_candidates_that_touch_it(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    job = wait_job(srv, start(srv, protected="layers.1.mlp.down_proj"))
    statuses = [r["status"] for r in job["result"]["rows"] if r.get("candidate")]
    # Every candidate edits layer 1's down projection: none may be committed, and the reason is reported.
    assert statuses and not any(s == "committed" for s in statuses)
    assert any("touches protected tensor model.layers.1.mlp.down_proj" in s for s in statuses)
    assert WeightGitRepository(ws).list_commits() == []


def test_invalid_protected_names_are_refused(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    status, _, _ = call(srv, "POST", "/api/loop", {"protected": "../etc/passwd"})
    assert status == 400


def test_rejecting_the_best_candidate_commits_nothing(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    job_id = start(srv, approve_each=True)
    waiting = wait_job(srv, job_id, until=("awaiting_approval", "done", "failed"))
    assert waiting["state"] == "awaiting_approval"
    assert waiting["awaiting"]["modified_params"] > 0 and "def compute_delta" in waiting["awaiting"]["script"]
    assert call(srv, "POST", f"/api/jobs/{job_id}/decision", {"approve": False})[0] == 200
    while True:  # later iterations may offer another candidate: keep rejecting
        state = wait_job(srv, job_id, until=("awaiting_approval", "done", "failed"))
        if state["state"] != "awaiting_approval":
            break
        call(srv, "POST", f"/api/jobs/{job_id}/decision", {"approve": False})
    assert state["state"] == "done"
    assert WeightGitRepository(ws).list_commits() == []


def test_approving_the_best_candidate_commits_it(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    job_id = start(srv, approve_each=True)
    assert wait_job(srv, job_id, until=("awaiting_approval", "done", "failed"))["state"] == "awaiting_approval"
    assert call(srv, "POST", f"/api/jobs/{job_id}/decision", {"approve": True})[0] == 200
    done = wait_job(srv, job_id, until=("done", "failed", "awaiting_approval"))
    while done["state"] == "awaiting_approval":
        call(srv, "POST", f"/api/jobs/{job_id}/decision", {"approve": True})
        done = wait_job(srv, job_id, until=("done", "failed", "awaiting_approval"))
    assert done["state"] == "done" and len(WeightGitRepository(ws).list_commits()) >= 1


def test_cancel_while_waiting_commits_nothing(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    job_id = start(srv, approve_each=True)
    assert wait_job(srv, job_id, until=("awaiting_approval", "done", "failed"))["state"] == "awaiting_approval"
    assert call(srv, "POST", f"/api/jobs/{job_id}/cancel", {})[0] == 200
    done = wait_job(srv, job_id)
    assert done["result"]["cancelled"] is True
    assert WeightGitRepository(ws).list_commits() == []


def test_progress_is_reported(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    job_id = start(srv, approve_each=True)
    waiting = wait_job(srv, job_id, until=("awaiting_approval", "done", "failed"))
    assert "evaluating candidate" in waiting["progress"]["text"] and 0 <= waiting["progress"]["fraction"] <= 1
    call(srv, "POST", f"/api/jobs/{job_id}/cancel", {})
    wait_job(srv, job_id)


def test_compare_and_persisted_results(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    status, _, raw = call(srv, "POST", "/api/compare", {})
    assert status == 200
    rows = json.loads(raw)["rows"]
    assert rows and all(r["original"] == r["current"] for r in rows)  # nothing edited yet
    call(srv, "POST", "/api/probe", {"top": 2, "columns": 4})
    wait_job(srv, start(srv))
    last = json.loads(call(srv, "GET", "/api/last")[2])
    assert last["probe"]["layers"] and last["run"]["rows"] and last["compare"]["rows"]
    assert last["run"]["measured_at_head"] == last["head"]
    rows_after = json.loads(call(srv, "POST", "/api/compare", {})[2])["rows"]
    assert any(r["changed"] for r in rows_after if r["role"] == "failing")


def test_examples_validate(studio: tuple[StudioServer, Path]) -> None:
    srv, _ = studio
    examples = json.loads(call(srv, "GET", "/api/examples")[2])["examples"]
    assert len(examples) >= 2
    for ex in examples:
        IntentSpec.model_validate(ex["intent"])


def test_bundle_export_has_checksums_and_card(studio: tuple[StudioServer, Path]) -> None:
    srv, ws = studio
    status, _, raw = call(srv, "POST", "/api/export", {"format": "bundle"})
    assert status == 400 and b"no edit" in raw  # nothing to bundle at the original state
    wait_job(srv, start(srv))
    status, _, raw = call(srv, "POST", "/api/export", {"format": "bundle"})
    assert status == 200, raw
    folder = ws / json.loads(raw)["path"]
    names = {p.name for p in folder.iterdir()}
    assert {"patch.sparse.safetensors", "manifests.json", "intent.yaml", "MODEL_CARD.md", "checksums.txt"} <= names
    sums = (folder / "checksums.txt").read_text(encoding="utf-8")
    assert "patch.sparse.safetensors" in sums and "base-fingerprint" in sums
    assert yaml.safe_load((folder / "intent.yaml").read_text(encoding="utf-8"))["intent_id"] == "break-x-loop"
