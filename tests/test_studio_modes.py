"""Studio route tests for the CR6 edit modes: previews as jobs, apply exactly what was previewed, input limits."""

from __future__ import annotations

import json
from typing import Any

import pytest

from neuroforge.studio import StudioServer

from .test_studio_controls import studio, wait_job  # noqa: F401 - fixture re-export
from .test_studio_security import call


def _job(srv: StudioServer, route: str, body: dict[str, Any]) -> dict[str, Any]:
    status, _, raw = call(srv, "POST", route, body)
    assert status == 202, raw
    return wait_job(srv, json.loads(raw)["job_id"])


def test_mode_info_explains_the_toy_model(studio: tuple[StudioServer, Any]) -> None:  # noqa: F811
    srv, _ = studio
    info = json.loads(call(srv, "GET", "/api/modes/info")[2])
    assert info["toy"] is True and info["editable_layers"] == [0, 1] and "toy model" in info["note"]
    assert info["examples"]["facts"] and info["examples"]["direction"]["positives"]


def test_fact_preview_on_the_toy_refuses_to_apply(studio: tuple[StudioServer, Any]) -> None:  # noqa: F811
    srv, ws = studio
    facts = [{"prompt": "ab x", "new_object": "y", "old_object": "x", "neighbours": ["hello world"]}]
    job = _job(srv, "/api/facts/preview", {"facts": facts, "batch_id": "demo"})
    assert job["state"] == "done", job
    rep = job["result"]
    assert rep["violations"] and rep["facts"][0]["before"]["answer"] == "x"
    status, _, raw = call(srv, "POST", "/api/plans/apply", {"plan_id": rep["plan_id"]})
    assert status == 409 and b"cannot be applied" in raw
    assert json.loads(call(srv, "GET", "/api/history")[2])["commits"] == []


def test_direction_preview_then_apply_commits_one_patch(studio: tuple[StudioServer, Any]) -> None:  # noqa: F811
    srv, _ = studio
    body = {"name": "bang", "positives": "wow!!!\nyes!!!\ngo!!!", "negatives": ["wow...", "yes...", "go..."], "strength": 1.0}
    job = _job(srv, "/api/direction/preview", body)
    assert job["state"] == "done", job
    rep = job["result"]
    assert rep["violations"] == [] and {"kl_general", "shift", "analysis"} <= set(rep)
    assert call(srv, "POST", "/api/plans/apply", {"plan_id": "f" * 16})[0] == 400
    status, _, raw = call(srv, "POST", "/api/plans/apply", {"plan_id": rep["plan_id"]})
    assert status == 200, raw
    commits = json.loads(call(srv, "GET", "/api/history")[2])["commits"]
    assert len(commits) == 1 and commits[0]["intent"] == "direction-bang"
    # The plan is used up; previewing again from the new state is required.
    assert call(srv, "POST", "/api/plans/apply", {"plan_id": rep["plan_id"]})[0] == 400


def test_a_stale_preview_cannot_be_applied(studio: tuple[StudioServer, Any]) -> None:  # noqa: F811
    srv, _ = studio
    body = {"name": "bang", "positives": ["wow!!!", "yes!!!"], "negatives": ["wow...", "yes..."]}
    first = _job(srv, "/api/direction/preview", body)["result"]
    second = _job(srv, "/api/direction/preview", body)["result"]
    assert call(srv, "POST", "/api/plans/apply", {"plan_id": first["plan_id"]})[0] == 200
    status, _, raw = call(srv, "POST", "/api/plans/apply", {"plan_id": second["plan_id"]})
    assert status == 409 and b"changed since this preview" in raw


@pytest.mark.parametrize(
    "route,body",
    [
        ("/api/facts/preview", {"facts": [{"prompt": "a", "new_object": "b"}], "layers": "../1"}),
        ("/api/facts/preview", {"facts": [{"prompt": "a", "new_object": "b"}], "layers": "7"}),
        ("/api/facts/preview", {"facts": [{"prompt": "a", "new_object": "b"}], "budget_percent": 50}),
        ("/api/facts/preview", {"facts": [{"prompt": "", "new_object": "b"}]}),
        ("/api/facts/preview", {"csv": "question,answer\nq,a"}),
        ("/api/facts/preview", {"facts": [{"prompt": "a", "new_object": "b"}] * 40}),
        ("/api/direction/preview", {"positives": ["a"], "negatives": ["b", "c"]}),
        ("/api/direction/preview", {"positives": ["a", "b"], "negatives": ["c", "d"], "strength": 3}),
        ("/api/direction/preview", {"positives": ["a", "b"], "negatives": ["c", "d"], "layer": "x"}),
        ("/api/facts/parse_csv", {"csv": "prompt,new_object,subject\nThe capital is, Rome,France"}),
    ],
)
def test_invalid_mode_inputs_are_refused(studio: tuple[StudioServer, Any], route: str, body: dict[str, Any]) -> None:  # noqa: F811
    srv, _ = studio
    assert call(srv, "POST", route, body)[0] == 400


def test_drafts_need_a_frontier_model_then_return_validated_text(studio: tuple[StudioServer, Any], monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    srv, _ = studio
    assert call(srv, "POST", "/api/direction/draft", {"description": "exclamation marks"})[0] == 409

    class Fake:
        name = "fake"

        def complete(self, system: str, prompt: str) -> str:
            return '{"positives": ["Great!!!", "Wow!!!"], "negatives": ["Great.", "Wow."]}'

    srv.state.selected_model = "frontier-x"
    monkeypatch.setattr(srv.state, "_backend", lambda: Fake())
    status, _, raw = call(srv, "POST", "/api/direction/draft", {"description": "exclamation marks"})
    assert status == 200 and json.loads(raw)["positives"] == ["Great!!!", "Wow!!!"]
