"""Tests for the Model Factory (CR9): suites, logged checks, held-out isolation, verdict, packaging and MCP round trip."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from libweightspace.candidates import TemplateCandidateSource, nullification_script
from neuroforge.factory import Task, TaskSuite, check_task, suite_examples, verdict
from neuroforge.studio import StudioServer

from .test_studio_controls import studio, wait_job  # noqa: F401 - fixture re-export
from .test_studio_security import call

SRC = Path(__file__).resolve().parents[1] / "src"


def _demo_yaml() -> str:
    return yaml.safe_dump(suite_examples()[0]["suite"], sort_keys=False)


def _run(srv: StudioServer, **body: Any) -> dict[str, Any]:
    status, _, raw = call(srv, "POST", "/api/factory/run", {"suite_yaml": _demo_yaml(), "approve_each": False, **body})
    assert status == 202, raw
    return wait_job(srv, json.loads(raw)["job_id"])


def test_suites_are_validated() -> None:
    with pytest.raises(ValueError, match="python_test needs test"):
        Task(id="a", prompt="p", check="python_test")
    with pytest.raises(ValueError, match="expected rows"):
        Task(id="a", prompt="p", check="sql_rows", expected="x")
    base = {"suite_id": "s", "description": "d"}
    with pytest.raises(ValueError, match="unique"):
        TaskSuite(**base, tasks=[Task(id="a", prompt="p", check="starts_with", expected="x")] * 2)
    with pytest.raises(ValueError, match="training task"):
        TaskSuite(**base, tasks=[Task(id="a", prompt="p", check="starts_with", expected="x", split="heldout")])
    for ex in suite_examples():
        TaskSuite.model_validate(ex["suite"])


def test_checks_run_in_the_hardened_runner() -> None:
    py = Task(id="add", prompt="def add(a, b):\n", check="python_test", test="assert add(1, 2) == 3",
              answer_starts_with_prompt=True)
    assert check_task(py, "    return a + b\n").passed
    assert not check_task(py, "    import socket\n    return 3\n").passed
    sql = Task(id="q", prompt="Q:", check="sql_rows", setup="create table t(n int); insert into t values (1),(2);",
               expected=[[2]])
    assert check_task(sql, "select max(n) from t;").passed
    assert not check_task(sql, "select min(n) from t").passed
    text = Task(id="t", prompt="ab x", check="not_starts_with", expected="x")
    assert check_task(text, "yz").passed and not check_task(text, "xx").passed


def test_factory_run_is_backed_by_logged_checks_and_reports_splits(studio: tuple[StudioServer, Path]) -> None:  # noqa: F811
    srv, ws = studio
    job = _run(srv)
    assert job["state"] == "done", job
    rep = job["result"]
    assert rep["baseline"]["train"] == 0.0 and rep["baseline"]["heldout"] == 0.0
    assert rep["after"]["heldout"] > rep["baseline"]["heldout"] and rep["prompt_only"]["configured"] is True
    assert rep["verdict"]["split"] == "heldout" and rep["verdict"]["packaging_allowed"] is True
    log = [json.loads(line) for line in (ws / ".factory" / rep["run_id"] / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    tasks = {t["id"] for t in suite_examples()[0]["suite"]["tasks"]}
    for phase in ("baseline", "after"):
        rows = [r for r in log if r["phase"] == phase]
        assert {r["task"] for r in rows} == tasks and all("passed" in r and r["reason"] for r in rows)
    stored = json.loads(call(srv, "GET", f"/api/factory/report?run={rep['run_id']}")[2])
    assert stored["report"]["verdict"] == rep["verdict"] and len(stored["log"]) == len(log)
    assert [r["run_id"] for r in json.loads(call(srv, "GET", "/api/factory/runs")[2])["runs"]] == [rep["run_id"]]


def test_heldout_tasks_never_reach_the_proposer(studio: tuple[StudioServer, Path], monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    srv, ws = studio
    seen: list[str] = []

    class Recording(TemplateCandidateSource):
        def propose(self, context: Any) -> Any:
            seen.append(json.dumps({"intent": context.intent_description, "history": context.history}, default=str))
            return super().propose(context)

    monkeypatch.setattr(srv.state, "repair_proposer", lambda use_llm: (Recording(), "recording"))
    rep = _run(srv)["result"]
    heldout = [t["prompt"] for t in suite_examples()[0]["suite"]["tasks"] if t["split"] == "heldout"]
    intent_text = (ws / ".factory" / rep["run_id"] / "intent.yaml").read_text(encoding="utf-8")
    assert seen and not any(p in s for p in heldout for s in seen)
    assert not any(p in intent_text for p in heldout)


def test_factory_run_with_a_frontier_model(studio: tuple[StudioServer, Path], monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    srv, _ = studio

    class Frontier:
        name = "fake-frontier"

        def complete(self, system: str, prompt: str) -> str:
            return json.dumps({"candidates": [{"rationale": "remove the attractor", "python_script": nullification_script(3)}]})

    srv.state.selected_model = "frontier-x"
    monkeypatch.setattr(srv.state, "_backend", lambda: Frontier())
    rep = _run(srv, use_llm=True)["result"]
    assert rep["proposer"].startswith("frontier model: frontier-x")
    assert rep["after"]["heldout"] > rep["baseline"]["heldout"]


def test_verdict_says_when_a_system_prompt_is_enough() -> None:
    rows = [{"split": "heldout", "passed": p} for p in (False, False)]
    after = [{"split": "heldout", "passed": p} for p in (True, False)]
    v = verdict(rows, after, {"configured": True, "heldout": 1.0}, committed=True)
    assert v["packaging_allowed"] is False and any("prompt-only baseline" in m for m in v["messages"])
    v = verdict(rows, after, {"configured": True, "heldout": 0.0}, committed=True)
    assert v["packaging_allowed"] is True
    v = verdict(rows, None, {"configured": False}, committed=False)
    assert v["packaging_allowed"] is False and any("No edit was committed" in m for m in v["messages"])


def test_package_contains_no_request_text_in_code_and_answers_mcp(studio: tuple[StudioServer, Path]) -> None:  # noqa: F811
    srv, ws = studio
    rep = _run(srv)["result"]
    dest = json.loads(call(srv, "GET", "/api/factory/destination?kind=mcp")[2])
    assert dest["path"].startswith("packages")
    status, _, raw = call(srv, "POST", "/api/factory/package", {"run_id": rep["run_id"], "kind": "mcp"})
    assert status == 200, raw
    folder = ws / json.loads(raw)["path"]
    suite = suite_examples()[0]["suite"]
    texts = [suite["description"], suite["suite_id"]] + [t["prompt"] for t in suite["tasks"]] + suite["prompt_baselines"]
    code = "".join(p.read_text(encoding="utf-8") for p in folder.glob("*.py"))
    assert code and not any(t in code for t in texts if len(t) > 3)
    config = json.loads((folder / "tool_config.json").read_text(encoding="utf-8"))
    assert config["description"] == suite["description"] and config["patch_id"] == rep["head_after"]

    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": config["tool_name"], "arguments": {"prompt": "ab x"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "nope"},
    ]
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run([sys.executable, str(folder / "server.py")], input="\n".join(json.dumps(m) for m in messages) + "\n",
                          capture_output=True, text=True, timeout=120, env=env, check=False)
    replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4], proc.stderr[-800:]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert replies[1]["result"]["tools"][0]["name"] == config["tool_name"]
    answer = replies[2]["result"]["content"][0]["text"]
    assert replies[2]["result"]["isError"] is False and not answer.startswith("x")  # the packaged edit is applied
    assert replies[3]["error"]["code"] == -32601


def test_packaging_is_refused_without_a_winning_verdict(studio: tuple[StudioServer, Path]) -> None:  # noqa: F811
    srv, _ = studio
    first = _run(srv)["result"]
    # The model is already repaired: a second run has nothing to fix, commits nothing and may not package.
    second = _run(srv)["result"]
    assert second["verdict"]["packaging_allowed"] is False
    status, _, raw = call(srv, "POST", "/api/factory/package", {"run_id": second["run_id"], "kind": "skill"})
    assert status == 409 and b"only offered" in raw
    # The first run's state moved on? No: the second run committed nothing, so the first may still be packaged.
    assert call(srv, "POST", "/api/factory/package", {"run_id": first["run_id"], "kind": "skill"})[0] == 200
    assert call(srv, "POST", "/api/factory/package", {"run_id": "../x", "kind": "mcp"})[0] == 400
    assert call(srv, "POST", "/api/factory/package", {"run_id": first["run_id"], "kind": "exe"})[0] == 400
