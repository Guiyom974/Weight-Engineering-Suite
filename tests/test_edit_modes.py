"""Tests for the CR6 edit modes (fact batches and behaviour directions) on the toy models.

The toy model's blocks write very little into its residual stream, so fact edits that change an answer exceed the
safety limits there: these tests check that the effect is measured, the budget is held, and the limits are reported
and enforced. The full path within the limits (efficacy, paraphrases, neighbours) is checked on a real checkpoint in
``test_real_lfm3_edit_modes.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from libweightspace.loading import load_model
from neuroforge.edit_modes import (
    DirectionSpec,
    Fact,
    FactBatch,
    analyse_direction,
    commit_plan,
    draft_direction_sets,
    draft_fact_probes,
    key_position,
    parse_facts_csv,
    plan_direction_edit,
    plan_fact_edit,
)
from neuroforge.schemas import EditConstraints
from neuroforge.weight_git import WeightGitRepository
from neuroforge.workspace_ops import checkout

TOY_LIMITS = EditConstraints(max_modified_param_ratio=0.01, max_frobenius_rel_norm=0.15, max_spectral_norm_shift=0.10)
# On the 2-layer toy, protecting the general prompts leaves almost no free direction in 8 budgeted columns, so the fact
# tests below switch nullspace protection off to exercise the edit itself; protection is covered by the real test.
TOY_FACT_LIMITS = TOY_LIMITS.model_copy(update={"enforce_nullspace_projection": False})


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture()
def demo(tmp_path: Path) -> tuple[object, WeightGitRepository, Path]:
    import yaml

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "neuroforge.yaml").write_text(yaml.safe_dump({"model": "toy:qwen3_8", "demo_defect": True}), encoding="utf-8")
    repo = WeightGitRepository.init(ws)
    return load_model("toy:qwen3_8", demo_defect=True), repo, ws


def _batch(constraints: EditConstraints = TOY_FACT_LIMITS) -> FactBatch:
    facts = [Fact(prompt="ab x", new_object="y", old_object="x", paraphrases=["cd x"], neighbours=["hello world"])]
    return FactBatch(batch_id="t", facts=facts, constraints=constraints)


def test_protection_with_no_room_is_reported_not_hidden(demo: tuple) -> None:
    loaded, repo, _ = demo
    plan = plan_fact_edit(loaded, repo, _batch(TOY_LIMITS), "p0")
    rep = plan.report
    # The fact's key only uses directions the protected prompts use: the preview says so and cannot be applied.
    assert rep["nullspace_prompts"] > 0 and rep["summary"]["efficacy_after"] == 0.0
    assert any("lies entirely in directions the protected prompts use" in v for v in rep["violations"])
    with pytest.raises(ValueError, match="breaks the limits"):
        commit_plan(plan, loaded, repo, "toy:qwen3_8")


def test_fact_preview_measures_everything_and_changes_nothing(demo: tuple) -> None:
    loaded, repo, _ = demo
    before = _snapshot(loaded.model)
    plan = plan_fact_edit(loaded, repo, _batch(), "p1")
    assert _same(before, _snapshot(loaded.model))  # a preview never changes the model
    rep = plan.report
    row = rep["facts"][0]
    assert row["before"]["answer"] == "x" and row["after"]["success"] is True and row["after"]["answer"] == "y"
    assert rep["summary"]["efficacy_before"] == 0.0 and rep["summary"]["efficacy_after"] == 1.0
    assert row["paraphrases"]["count"] == 1 and row["neighbourhood"]["count"] == 1
    assert rep["summary"]["kl_general"] >= 0.0 and rep["drift"][0]["before"] == 0.0
    # Budget: only the budgeted columns change, and never more weights than the budget allows.
    delta = plan.deltas[rep["tensors"][0]]
    assert int((delta != 0).any(dim=0).sum()) <= len(rep["columns"][rep["tensors"][0]])
    assert int((delta != 0).sum()) <= rep["weights_targeted"] <= rep["weights_budget"]


def test_toy_fact_edits_report_and_enforce_the_limits(demo: tuple) -> None:
    loaded, repo, _ = demo
    plan = plan_fact_edit(loaded, repo, _batch(), "p1")
    assert any("relative Frobenius" in v for v in plan.report["violations"])
    with pytest.raises(ValueError, match="breaks the limits"):
        commit_plan(plan, loaded, repo, "toy:qwen3_8")
    assert repo.list_commits() == []


def test_a_fact_batch_is_one_patch_and_restores_bit_exactly(demo: tuple) -> None:
    loaded, repo, ws = demo
    original = _snapshot(loaded.model)
    plan = plan_fact_edit(loaded, repo, _batch(), "p1")
    plan.report["violations"] = []  # commit mechanics only: the limits are covered by the test above
    patch_id = commit_plan(plan, loaded, repo, "toy:qwen3_8")
    assert repo.list_commits() == [patch_id]
    manifest = repo.load_manifest(patch_id)
    assert manifest.intent_id == "facts-t" and manifest.target_metrics_after["facts.efficacy"] == 1.0
    assert not _same(original, _snapshot(loaded.model))
    # Sequential batches: the next preview reports the drift already accumulated in the tensor.
    second = plan_fact_edit(loaded, WeightGitRepository(ws), _batch(), "p2")
    assert second.report["drift"][0]["before"] > 0.0
    _, restored = checkout(ws, "ROOT")
    assert _same(original, _snapshot(restored.model))


def test_preview_against_a_moved_model_is_refused(demo: tuple) -> None:
    loaded, repo, _ = demo
    plan = plan_fact_edit(loaded, repo, _batch(), "p1")
    plan.head = "0" * 16
    with pytest.raises(ValueError, match="changed since this preview"):
        commit_plan(plan, loaded, repo, "toy:qwen3_8")


def test_budget_smaller_than_one_column_is_explained(demo: tuple) -> None:
    loaded, repo, _ = demo
    tiny = EditConstraints(max_modified_param_ratio=0.0001)
    with pytest.raises(ValueError, match="smaller than one column"):
        plan_fact_edit(loaded, repo, _batch(tiny), "p")


def test_csv_import_and_validation() -> None:
    text = "prompt,new_object,old_object,subject,paraphrases,neighbours\n" \
           "The capital of Exampleland is, Newtown, Oldtown,Exampleland,Exampleland's capital is|Capital of Exampleland:,The capital of France is\n"
    facts = parse_facts_csv(text)
    assert facts[0].subject == "Exampleland" and len(facts[0].paraphrases) == 2 and facts[0].neighbours
    with pytest.raises(ValueError, match="header"):
        parse_facts_csv("question,answer\nq,a\n")
    with pytest.raises(ValueError, match="row 2"):
        parse_facts_csv("prompt,new_object,subject\nThe capital is, Rome,France\n")
    with pytest.raises(ValueError, match="no fact rows"):
        parse_facts_csv("prompt,new_object\n")
    many = "prompt,new_object\n" + "".join(f"p{i},o\n" for i in range(40))
    with pytest.raises(ValueError, match="at most 32"):
        parse_facts_csv(many)


def test_subject_key_position(tokenizer: object) -> None:
    assert key_position(tokenizer, "The capital of France is", "France") == len("The capital of France") - 1
    assert key_position(tokenizer, "The capital of France is", None) == len("The capital of France is") - 1


def test_direction_layers_kl_shift_and_commit(demo: tuple) -> None:
    loaded, repo, ws = demo
    original = _snapshot(loaded.model)
    spec = DirectionSpec(name="bang", positives=["wow!!!", "yes!!!", "go!!!"], negatives=["wow...", "yes...", "go..."],
                         constraints=TOY_LIMITS)
    analysis = analyse_direction(loaded, spec)
    assert [r["layer"] for r in analysis["layers"]] == [0, 1] and analysis["suggested_layer"] in (0, 1)
    plan = plan_direction_edit(loaded, repo, spec, "d1")
    rep = plan.report
    assert rep["violations"] == [] and rep["kl_general"] > 0.0 and rep["kl_positives"] > 0.0
    shift = rep["shift"]
    assert shift["positives_after"] - shift["negatives_after"] < shift["positives_before"] - shift["negatives_before"]
    assert _same(original, _snapshot(loaded.model))
    patch_id = commit_plan(plan, loaded, repo, "toy:qwen3_8")
    assert repo.load_manifest(patch_id).intent_id == "direction-bang"
    _, restored = checkout(ws, "ROOT")
    assert _same(original, _snapshot(restored.model))


def test_direction_strength_zero_is_refused(demo: tuple) -> None:
    loaded, repo, _ = demo
    spec = DirectionSpec(name="x", positives=["a!", "b!"], negatives=["a.", "b."], strength=0.0, constraints=TOY_LIMITS)
    with pytest.raises(ValueError, match="strength 0"):
        plan_direction_edit(loaded, repo, spec, "d")


class _FakeBackend:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_frontier_drafts_are_validated() -> None:
    fact = Fact(prompt="The capital of Exampleland is", new_object=" Newtown")
    good = _FakeBackend('{"paraphrases": ["Exampleland capital:"], "neighbours": ["The capital of France is"]}')
    assert draft_fact_probes(good, fact) == {"paraphrases": ["Exampleland capital:"], "neighbours": ["The capital of France is"]}
    with pytest.raises(ValueError):
        draft_fact_probes(_FakeBackend("no json here"), fact)
    sets = _FakeBackend('{"positives": ["Great!!!", "Wow!!!"], "negatives": ["Great.", "Wow."]}')
    assert draft_direction_sets(sets, "exclamation marks")["negatives"] == ["Great.", "Wow."]
    with pytest.raises(ValueError):
        draft_direction_sets(_FakeBackend('{"positives": "not a list"}'), "x")
