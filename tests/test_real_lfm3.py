"""Real integration tests for Liquid Foundation Model 3 (LiquidAI/LFM3-1.7B).

Validates end-to-end model loading, layer anatomy resolution for hybrid conv/attention
blocks, heavy-tailed spectral profiling (HTSR alpha and Marchenko-Pastur bulk edges),
contrastive circuit tracing, and surgical weight intervention inside CoWWeightOverlay.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.prober import CircuitEdge, MechanisticProber
from libweightspace.spectral import profile_matrix
from libweightspace.surgery_ops import norm_preserving_column_edit

from .conftest import load_real_lfm3

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def lfm3_bundle() -> Any:
    """Loads the local LiquidAI/LFM3-1.7B copy named by WEIGHTLAB_REAL_MODEL_DIR (opt-in)."""
    return load_real_lfm3()  # type: ignore[return-value]


def test_real_lfm3_load_and_anatomy(lfm3_bundle: Any) -> None:
    """Verifies that LiquidAI/LFM3-1.7B loads correctly and maps hybrid conv and attention anatomy."""
    model = lfm3_bundle.model
    adapter = lfm3_bundle.adapter
    tokenizer = lfm3_bundle.tokenizer

    assert type(model).__name__ == "Lfm3ForCausalLM"
    assert adapter.family == "lfm3"
    assert adapter.num_layers() == 24
    assert adapter.hidden_size() == 2048

    # Layer 0 is a gated short-convolution operator block
    anat0 = adapter.get_layer_anatomy(0)
    assert anat0.ssm_in_proj == "model.layers.0.conv.in_proj.weight"
    assert anat0.ssm_out_proj == "model.layers.0.conv.out_proj.weight"
    assert anat0.mlp_down_proj == "model.layers.0.feed_forward.w2.weight"
    assert anat0.attn_q_proj is None

    # Layer 2 is a full/sliding attention operator block
    anat2 = adapter.get_layer_anatomy(2)
    assert anat2.attn_q_proj == "model.layers.2.self_attn.q_proj.weight"
    assert anat2.attn_o_proj == "model.layers.2.self_attn.o_proj.weight"
    assert anat2.mlp_down_proj == "model.layers.2.feed_forward.w2.weight"
    assert anat2.ssm_in_proj is None

    # Forward pass sanity check
    inputs = tokenizer("Liquid AI neural weights", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits.ndim == 3
    assert out.logits.shape[-1] == 128000


def test_real_lfm3_spectral_properties(lfm3_bundle: Any) -> None:
    """Evaluates empirical spectral power-law alpha and Marchenko-Pastur SNR on real weights."""
    params = dict(lfm3_bundle.model.named_parameters())

    # Profile feed_forward.w2 (down_proj) from layer 0
    w_down0 = params["model.layers.0.feed_forward.w2.weight"]
    profile_down = profile_matrix(w_down0)
    assert 1.0 <= profile_down.power_law_alpha <= 10.0
    assert profile_down.mp_bulk_edge > 0.0
    assert profile_down.snr_ratio >= 0.0

    # Profile self_attn.o_proj from layer 2
    w_attn2 = params["model.layers.2.self_attn.o_proj.weight"]
    profile_attn = profile_matrix(w_attn2)
    assert 1.0 <= profile_attn.power_law_alpha <= 10.0
    assert profile_attn.mp_bulk_edge > 0.0


def test_real_lfm3_mechanistic_circuit_and_surgery(lfm3_bundle: Any) -> None:
    """Traces causal circuit subcomponents and executes a norm-preserving surgical edit in CoW overlay."""
    model = lfm3_bundle.model
    adapter = lfm3_bundle.adapter
    tokenizer = lfm3_bundle.tokenizer

    prober = MechanisticProber(model, adapter, tokenizer, window=2)
    fail_texts = ["The capital of France is Rome."]
    pass_texts = ["The capital of France is Paris."]

    # Contrastive attribution across layers
    reports = prober.trace_contrastive(fail_texts, pass_texts, top_k_cols=4)
    assert len(reports) == 24
    top_report = reports[0]
    assert top_report.divergence_score >= 0.0
    assert len(top_report.top_culpable_mlp_cols) > 0

    # Subcircuit edge attribution
    edges = prober.trace_circuit(fail_texts, pass_texts, top_k_edges=4)
    assert len(edges) > 0
    for edge in edges:
        assert isinstance(edge, CircuitEdge)
        assert edge.source_layer <= edge.target_layer

    # Surgical intervention using DoRA-style norm-preserving column update in CoW overlay
    culprit_col = top_report.top_culpable_mlp_cols[0]
    target_layer = top_report.layer_idx
    target_param_name = adapter.get_layer_anatomy(target_layer).mlp_down_proj
    assert target_param_name is not None

    orig_w = dict(model.named_parameters())[target_param_name].detach().clone()
    col_norm_before = float(torch.norm(orig_w[:, culprit_col]).item())

    # Apply norm-preserving column edit delta
    perturbation = torch.randn(orig_w.shape[0])
    delta = norm_preserving_column_edit(orig_w, culprit_col, perturbation, scale=0.5)
    edited_w = orig_w + delta
    col_norm_after = float(torch.norm(edited_w[:, culprit_col]).item())

    # Verify exact column norm preservation within float precision
    assert abs(col_norm_before - col_norm_after) < 1e-4

    # Mount in CoW overlay and verify forward pass divergence
    overlay = CoWWeightOverlay(model)

    prompt = pass_texts[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        with overlay.apply_delta({target_param_name: delta}):
            out_edited = model(**inputs).logits.clone()
        out_orig = model(**inputs).logits.clone()

    divergence = float(torch.norm(out_edited.float() - out_orig.float()).item())
    assert divergence > 0.0, "Weight intervention must perturb forward logits"
