"""Prober and spectral tests (PRD_01 section 5.3, blueprint step 6)."""

import pytest
import torch

from libweightspace import spectral
from libweightspace.adapters import Qwen3Adapter
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from libweightspace.toy_models import build_tiny_qwen3

from .conftest import FAIL_PROMPTS, PASS_PROMPTS

PLANTED = [5, 12, 19]


def test_planted_attractor_is_localised(planted_qwen, tokenizer):
    model, defect = planted_qwen
    prober = MechanisticProber(model, Qwen3Adapter(model), tokenizer)
    reports = prober.trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS, top_k_cols=64)
    assert reports[0].layer_idx == 1
    assert set(PLANTED) <= set(reports[0].top_culpable_mlp_cols[:5])
    assert reports[0].mlp_down_proj == defect.tensor_name
    # The recovered attractor matches the planted direction.
    cos = torch.dot(reports[0].attractor_direction, defect.direction) / defect.direction.norm()
    assert float(cos) > 0.98  # measured at the down_proj output, so embedding differences do not bias it
    assert len(reports[0].singular_values_top5) == 5


def test_negative_control_without_planted_defect(tiny_qwen, tokenizer):
    """Without the plant, the same columns must not surface; guards against a trivially passing test."""
    prober = MechanisticProber(tiny_qwen, Qwen3Adapter(tiny_qwen), tokenizer)
    top5 = set(prober.trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS)[0].top_culpable_mlp_cols[:5])
    assert not set(PLANTED) <= top5


def test_removing_attractor_reduces_divergence(planted_qwen, tokenizer):
    model, defect = planted_qwen
    adapter = Qwen3Adapter(model)
    prober = MechanisticProber(model, adapter, tokenizer)
    before = {r.layer_idx: r.divergence_score for r in prober.trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS)}
    w = dict(model.named_parameters())[defect.tensor_name]
    with torch.no_grad():
        v = defect.direction
        for c in PLANTED:
            w[:, c] -= torch.dot(w[:, c], v) * v
    after = {r.layer_idx: r.divergence_score for r in prober.trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS)}
    assert after[1] < 0.5 * before[1]
    assert after[0] == pytest.approx(before[0], rel=1e-5)  # layer 0 untouched


def test_indirect_effect_last_layer_fully_restores(tiny_qwen, tokenizer):
    prober = MechanisticProber(tiny_qwen, Qwen3Adapter(tiny_qwen), tokenizer)
    effects = prober.layer_indirect_effects("ab yy", "ab xx", target_token_id=ord("y"))
    assert len(effects) == 2
    assert effects[-1] == pytest.approx(1.0, abs=1e-4)


def test_trace_validates_inputs(tiny_qwen, tokenizer):
    prober = MechanisticProber(tiny_qwen, Qwen3Adapter(tiny_qwen), tokenizer)
    with pytest.raises(ValueError, match="at least one"):
        prober.trace_contrastive([], PASS_PROMPTS)
    with pytest.raises(ValueError, match="top_k_cols"):
        prober.trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS, top_k_cols=0)


def test_capture_linear_inputs_shapes(tiny_qwen, tokenizer):
    name = "model.layers.1.mlp.down_proj.weight"
    acts = capture_linear_inputs(tiny_qwen, tokenizer, ["abc", "defg"], [name, "model.norm.weight", "nope"])
    assert list(acts) == [name]
    assert acts[name].shape == (7, tiny_qwen.config.intermediate_size)


def test_build_diagnostics(planted_qwen, tokenizer):
    model, _ = planted_qwen
    reports = MechanisticProber(model, Qwen3Adapter(model), tokenizer).trace_contrastive(
        FAIL_PROMPTS, PASS_PROMPTS
    )
    diag = build_diagnostics(reports, top_n=2)
    assert diag["layer_idx"] == 1 and diag["target_tensor"].endswith("layers.1.mlp.down_proj.weight")
    assert len(diag["layers"]) == 2 and diag["attractor_direction"].shape == (model.config.hidden_size,)
    with pytest.raises(ValueError):
        build_diagnostics([])


# --- spectral -------------------------------------------------------------------------------------------


def test_singular_values_and_effective_rank():
    d = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    assert torch.allclose(spectral.singular_values(d), torch.tensor([4.0, 2.0, 1.0]))
    assert spectral.effective_rank(torch.eye(6)) == pytest.approx(6.0, rel=1e-5)
    rank1 = torch.outer(torch.randn(8), torch.randn(5))
    assert spectral.effective_rank(rank1) == pytest.approx(1.0, abs=1e-3)
    profile = spectral.profile_matrix(d, "d", top_k=2)
    assert profile.top_singular_values == pytest.approx([4.0, 2.0])
    assert profile.condition_number == pytest.approx(4.0)
    assert profile.spectral_norm == pytest.approx(4.0)
    assert profile.frobenius_norm == pytest.approx(21**0.5)
    assert spectral.effective_rank(torch.zeros(3, 3)) == 0.0


def test_spectral_shift_and_clamp():
    w = torch.diag(torch.tensor([2.0, 1.0]))
    assert spectral.spectral_norm_shift(w, torch.diag(torch.tensor([0.2, 0.0]))) == pytest.approx(0.1)
    clamped = spectral.clamp_spectral_norm(w, 1.0)
    assert spectral.spectral_norm(clamped) == pytest.approx(1.0)
    assert torch.equal(spectral.clamp_spectral_norm(w, 5.0), w)
    with pytest.raises(ValueError):
        spectral.clamp_spectral_norm(w, 0.0)


def test_large_matrix_paths_match_exact(monkeypatch):
    g = torch.Generator().manual_seed(3)
    w = torch.randn(40, 30, generator=g)
    exact_sigma = spectral.spectral_norm(w)
    exact_top = spectral.singular_values(w, 4)
    monkeypatch.setattr(spectral, "EXACT_SVD_MAX_ELEMENTS", 10)
    assert spectral.spectral_norm(w, iters=200) == pytest.approx(exact_sigma, rel=1e-3)
    assert torch.allclose(spectral.singular_values(w, 4), exact_top, rtol=5e-2)


def test_profile_handles_vectors():
    profile = spectral.profile_matrix(torch.tensor([3.0, 4.0]), "bias")
    assert profile.spectral_norm == pytest.approx(5.0)


def test_prober_on_other_families(tokenizer):
    model = build_tiny_qwen3(seed=5, num_hidden_layers=3)
    reports = MechanisticProber(model, Qwen3Adapter(model), tokenizer).trace_contrastive(["aaa"], ["bbb"], 4)
    assert len(reports) == 3 and all(len(r.top_culpable_mlp_cols) == 4 for r in reports)
