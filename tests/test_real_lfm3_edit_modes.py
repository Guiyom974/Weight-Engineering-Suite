"""Opt-in validation of the CR6 edit modes on a real checkpoint (LiquidAI LFM 3 1.7B).

CR6 requires a validation run per architecture before the fact mode is relied on. These tests preview (never commit,
the cached model is shared with other real-model tests) and check the measured results against the default limits.
Run with ``WEIGHTLAB_REAL_MODEL_DIR=<folder> pytest -m real_model``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from neuroforge.edit_modes import DirectionSpec, Fact, FactBatch, plan_direction_edit, plan_fact_edit
from neuroforge.weight_git import WeightGitRepository

from .conftest import load_real_lfm3

pytestmark = pytest.mark.real_model


def _checksum(model: torch.nn.Module, names: list[str]) -> list[float]:
    params = dict(model.named_parameters())
    return [float(params[n].detach().double().sum()) for n in names]


def test_fact_edit_within_default_limits_on_lfm3() -> None:
    loaded = load_real_lfm3()
    repo = WeightGitRepository.init(Path(tempfile.mkdtemp()))
    fact = Fact(prompt="The capital of France is", subject="France", new_object=" Rome", old_object=" Paris",
                paraphrases=["France's capital city is"], neighbours=["The capital of Germany is", "The capital of Spain is"])
    plan = plan_fact_edit(loaded, repo, FactBatch(batch_id="real", facts=[fact]), "real")  # type: ignore[arg-type]
    rep = plan.report
    assert rep["layers"] == [4, 5, 6, 7]
    names = rep["tensors"]
    assert _checksum(loaded.model, names) == _checksum(loaded.model, names)  # stable read
    assert rep["violations"] == [], rep["violations"]
    assert rep["summary"]["efficacy_after"] == 1.0 and rep["facts"][0]["after"]["answer"].strip() == "Rome"
    assert rep["summary"]["neighbourhood_unchanged"] == 1.0 and rep["summary"]["kl_general"] < 0.05
    assert rep["weights_targeted"] <= rep["weights_budget"]


def test_direction_preview_on_lfm3_measures_kl_and_shift() -> None:
    loaded = load_real_lfm3()
    repo = WeightGitRepository.init(Path(tempfile.mkdtemp()))
    spec = DirectionSpec(
        name="exclaim",
        positives=["What a wonderful day!!!", "I love this so much!!!", "This is amazing!!!", "We won the game!!!"],
        negatives=["What a wonderful day.", "I love this so much.", "This is amazing.", "We won the game."],
    )
    before = _checksum(loaded.model, [loaded.adapter.get_layer_anatomy(i).mlp_down_proj or "" for i in range(4, 8)])
    rep = plan_direction_edit(loaded, repo, spec, "d").report  # type: ignore[arg-type]
    after = _checksum(loaded.model, [loaded.adapter.get_layer_anatomy(i).mlp_down_proj or "" for i in range(4, 8)])
    assert before == after  # a preview leaves the model unchanged
    assert rep["violations"] == [] and rep["kl_general"] < 0.05
    shift = rep["shift"]
    assert shift["positives_after"] - shift["negatives_after"] < shift["positives_before"] - shift["negatives_before"]
