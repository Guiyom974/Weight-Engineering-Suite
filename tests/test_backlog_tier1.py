"""Unit and integration tests for Tier 1 backlog items (NF-BL-01, NF-BL-02, NF-BL-03)."""

from __future__ import annotations

import pytest
import torch

from libweightspace.alphaedit import (
    alphaedit_batch_update,
    compute_covariance_nullspace,
    project_matrix_from_activations,
)
from libweightspace.loading import load_model
from libweightspace.sandbox import execute_weight_surgery_script
from libweightspace.spectral import profile_matrix
from libweightspace.surgery_ops import biproject_direction, norm_preserving_column_edit
from neuroforge.convergence_engine import check_constraints
from neuroforge.schemas import EditConstraints


def test_biproject_direction_orthogonal_to_benign_basis() -> None:
    """NF-BL-01: biproject_direction removes components along benign_basis with <1e-6 residual cosine."""
    torch.manual_seed(42)
    d = 64
    benign = torch.randn(d, 4)
    target = benign[:, 0] * 2.5 + benign[:, 1] * (-1.2) + torch.randn(d)
    u_perp = biproject_direction(target, benign)
    assert u_perp.shape == (d,)
    assert abs(float(torch.linalg.vector_norm(u_perp).item()) - 1.0) < 1e-6
    for col in range(benign.shape[1]):
        b_unit = benign[:, col] / torch.linalg.vector_norm(benign[:, col])
        assert abs(float(torch.dot(u_perp, b_unit).item())) < 1e-6


def test_norm_preserving_column_edit_restores_exact_column_norms() -> None:
    """NF-BL-01: DoRA-style column edit preserves exact pre-edit column L2 norms and works inside sandbox."""
    torch.manual_seed(7)
    w = torch.randn(32, 48)
    direction = torch.randn(32)
    benign = torch.randn(32, 3)
    cols = [3, 11, 27]
    delta = norm_preserving_column_edit(w, cols, direction, scale=1.0, benign_basis=benign)
    w_new = w + delta

    # Unedited columns are untouched.
    unedited = [c for c in range(48) if c not in cols]
    assert torch.allclose(delta[:, unedited], torch.zeros_like(delta[:, unedited]))

    # Edited columns preserve exact L2 norms and have zero projection onto biprojected direction.
    u_perp = biproject_direction(direction, benign)
    for c in cols:
        orig_norm = float(torch.linalg.vector_norm(w[:, c]).item())
        new_norm = float(torch.linalg.vector_norm(w_new[:, c]).item())
        assert abs(orig_norm - new_norm) < 1e-5
        assert abs(float(torch.dot(w_new[:, c], u_perp).item())) < 1e-5

    # Also test execution inside the AST sandbox via torch.surgery.
    script = """
def compute_delta(weights, diagnostics):
    W = weights["layer.weight"]
    v = diagnostics["attractor"]
    b = diagnostics["benign"]
    cols = diagnostics["cols"]
    d = torch.surgery.norm_preserving_column_edit(W, cols, v, 1.0, b)
    return {"layer.weight": d}
"""
    out = execute_weight_surgery_script(
        script,
        {"layer.weight": w},
        {"attractor": direction, "benign": benign, "cols": cols},
    )
    assert torch.allclose(out["layer.weight"], delta, atol=1e-6)


def test_alphaedit_nullspace_and_batch_update() -> None:
    """NF-BL-02: AlphaEdit batch update injects target facts while keeping ||delta_W @ K_0||_F < 1e-5."""
    torch.manual_seed(19)
    d_out, d_in = 32, 48
    w = torch.randn(d_out, d_in) * 0.1
    # Reference activations span a 16-dimensional subspace of R^48.
    basis_ref = torch.linalg.qr(torch.randn(d_in, 16), mode="reduced")[0]
    k0_samples = torch.randn(40, 16) @ basis_ref.T  # [40, d_in]
    p_null = project_matrix_from_activations(k0_samples, eig_threshold=1e-4)

    # New fact keys have components in the orthogonal complement of basis_ref.
    k_facts = torch.linalg.qr(p_null @ torch.randn(d_in, 5), mode="reduced")[0][:, :5]  # [d_in, 5]
    v_targets = torch.randn(d_out, 5)
    delta = alphaedit_batch_update(w, k_facts, v_targets, P_null=p_null, ridge=1e-4)

    # 1. Nullspace preservation on reference activations K_0.
    drift = float(torch.linalg.matrix_norm(delta @ k0_samples.T, ord="fro").item())
    assert drift < 1e-5

    # 2. Target facts are learned accurately.
    pred = (w + delta) @ k_facts
    rel_err = float(torch.linalg.matrix_norm(pred - v_targets) / torch.linalg.matrix_norm(v_targets))
    assert rel_err < 1e-3


def test_compute_covariance_nullspace_on_toy_model() -> None:
    """NF-BL-02: compute_covariance_nullspace captures real linear inputs from a ToyCausalLM."""
    bundle = load_model("toy:qwen3_8")
    target_param = bundle.adapter.get_layer_anatomy(1).mlp_down_proj
    assert target_param is not None
    nulls = compute_covariance_nullspace(
        bundle.model,
        bundle.tokenizer,
        ["The capital of France is Paris.", "Water boils at 100 degrees Celsius."],
        [target_param],
        eig_threshold=1e-2,
    )
    assert target_param in nulls
    d_in = bundle.model.get_parameter(target_param).shape[1]
    assert nulls[target_param].shape == (d_in, d_in)


def test_spectral_htsr_alpha_and_constraint_gate() -> None:
    """NF-BL-03: HTSR power-law alpha estimation and EditConstraints.min_post_edit_alpha enforcement."""
    torch.manual_seed(99)
    u, _ = torch.linalg.qr(torch.randn(64, 64))
    v, _ = torch.linalg.qr(torch.randn(64, 64))
    # Construct synthetic singular values following a well-behaved power law vs rank-1 spike collapse.
    ranks = torch.arange(1, 65, dtype=torch.float32)
    s_healthy = ranks.pow(-0.6)
    w_healthy = (u * s_healthy) @ v.T
    prof_healthy = profile_matrix(w_healthy, "layer.mlp.down_proj.weight")
    assert prof_healthy.power_law_alpha > 1.8
    assert prof_healthy.mp_bulk_edge > 0.0
    assert 0.0 <= prof_healthy.snr_ratio <= 1.0

    # Corrupt with a massive rank-1 spike that collapses the tail exponent alpha toward 1.0.
    delta_spike = (u[:, :1] * 50.0) @ v[:, :1].T
    prof_spiked = profile_matrix(w_healthy + delta_spike, "layer.mlp.down_proj.weight")
    assert prof_spiked.power_law_alpha < prof_healthy.power_law_alpha

    constraints = EditConstraints(min_post_edit_alpha=2.2)
    violations = check_constraints(
        {"layer.mlp.down_proj.weight": delta_spike},
        {"layer.mlp.down_proj.weight": w_healthy},
        w_healthy.numel(),
        constraints,
    )
    assert any("post-edit HTSR alpha" in v for v in violations)

    with pytest.raises(ValueError, match="target_dir must be 1-D"):
        biproject_direction(torch.zeros(2, 2))
