"""Nullspace locking tests (PRD_02 section 4.2, blueprint step 7)."""

import pytest
import torch

from libweightspace.nullspace import NullspaceLocker

NAME = "model.layers.0.mlp.down_proj.weight"


def _acts_spanning(v1, v2, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    coeffs = torch.randn(n, 2, generator=g) * torch.tensor([3.0, 1.5])
    return coeffs[:, :1] * v1 + coeffs[:, 1:] * v2


def test_projection_has_zero_interference_with_locked_skills():
    g = torch.Generator().manual_seed(1)
    v1, v2 = torch.randn(32, generator=g), torch.randn(32, generator=g)
    locker = NullspaceLocker()
    assert locker.register_protected_directions(NAME, _acts_spanning(v1, v2), top_k=2) == 2
    assert locker.locked_rank(NAME) == 2
    dense = torch.randn(16, 32, generator=g)
    safe = locker.project_into_nullspace(NAME, dense)
    assert torch.linalg.vector_norm(safe @ v1) < 1e-5
    assert torch.linalg.vector_norm(safe @ v2) < 1e-5
    # Directions orthogonal to the locked span are untouched.
    basis = locker.basis(NAME)
    w = torch.randn(32, generator=g)
    w_perp = w - basis @ (basis.T @ w)
    assert torch.allclose(safe @ w_perp, dense @ w_perp, atol=1e-4)


def test_negative_control_unprojected_delta_interferes():
    g = torch.Generator().manual_seed(2)
    v1 = torch.randn(32, generator=g)
    dense = torch.randn(16, 32, generator=g)
    assert torch.linalg.vector_norm(dense @ v1) > 1e-2
    assert torch.equal(NullspaceLocker().project_into_nullspace(NAME, dense), dense)  # nothing locked yet


def test_redundant_directions_do_not_inflate_rank():
    g = torch.Generator().manual_seed(3)
    v1, v2 = torch.randn(32, generator=g), torch.randn(32, generator=g)
    locker = NullspaceLocker()
    locker.register_protected_directions(NAME, _acts_spanning(v1, v2))
    assert locker.register_protected_directions(NAME, _acts_spanning(v1, v2, seed=7)) == 0
    assert locker.locked_rank(NAME) == 2
    v3 = torch.randn(32, generator=g)
    assert locker.register_protected_directions(NAME, v3[None, :].repeat(10, 1), top_k=2) == 1
    assert locker.locked_rank(NAME) == 3
    basis = locker.basis(NAME)
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5)


def test_max_rank_keeps_oldest_directions():
    locker = NullspaceLocker(max_rank_per_tensor=3)
    eye = torch.eye(8)
    locker.register_protected_directions(NAME, eye[:2] * torch.tensor([[2.0], [1.0]]), top_k=2)
    first = locker.basis(NAME)
    added = locker.register_protected_directions(NAME, eye[4:6] * torch.tensor([[2.0], [1.0]]), top_k=2)
    assert added == 1 and locker.locked_rank(NAME) == 3
    assert torch.allclose(locker.basis(NAME)[:, :2], first)


def test_non_2d_and_shape_errors():
    locker = NullspaceLocker()
    locker.register_protected_directions(NAME, torch.randn(5, 8))
    bias = torch.randn(8)
    assert torch.equal(locker.project_into_nullspace(NAME, bias), bias)
    with pytest.raises(ValueError, match="d_in"):
        locker.project_into_nullspace(NAME, torch.randn(4, 9))
    with pytest.raises(ValueError, match="does not match"):
        locker.register_protected_directions(NAME, torch.randn(5, 9))
    with pytest.raises(ValueError, match="N_tokens"):
        locker.register_protected_directions(NAME, torch.randn(5))
    with pytest.raises(ValueError):
        NullspaceLocker(max_rank_per_tensor=0)


def test_dtype_preserved_and_persistence(tmp_path):
    locker = NullspaceLocker()
    locker.register_protected_directions(NAME, torch.randn(10, 8))
    out = locker.project_into_nullspace(NAME, torch.randn(4, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    path = tmp_path / "ns.safetensors"
    locker.save(path)
    restored = NullspaceLocker.load(path)
    assert restored.locked_rank(NAME) == locker.locked_rank(NAME)
    assert torch.allclose(restored.basis(NAME), locker.basis(NAME))
    assert NullspaceLocker.load(tmp_path / "missing.safetensors").locked_tensors() == []


def test_preserve_sparsity_bounds_active_columns_and_keeps_nullspace():
    g = torch.Generator().manual_seed(11)
    v1, v2 = torch.randn(64, generator=g), torch.randn(64, generator=g)
    locker = NullspaceLocker()
    locker.register_protected_directions(NAME, _acts_spanning(v1, v2, n=64, seed=12), top_k=2)
    basis = locker.basis(NAME)
    assert basis is not None and basis.shape == (64, 2)

    sparse = torch.zeros(16, 64)
    sparse[:, [3, 7, 11]] = torch.randn(16, 3, generator=g)
    safe = locker.project_into_nullspace(NAME, sparse, preserve_sparsity=True)
    # Nullspace invariant holds across the full space.
    assert torch.linalg.matrix_norm(safe @ basis) < 1e-5
    # Active columns on the original defect support are preserved unattenuated.
    assert torch.allclose(safe[:, [3, 7, 11]], sparse[:, [3, 7, 11]], atol=1e-5)
    # Total non-zero columns are bounded to |S| + 2*r = 3 + 4 = 7 (well below 64).
    active_after = int((safe.abs().sum(dim=0) > 0).sum().item())
    assert active_after <= 7


def test_compact_bases_reduces_rank_via_svd():
    g = torch.Generator().manual_seed(19)
    locker = NullspaceLocker(max_rank_per_tensor=8)
    acts = torch.randn(32, 16, generator=g)
    assert locker.register_protected_directions(NAME, acts, top_k=6) == 6
    assert locker.locked_rank(NAME) == 6
    ranks = locker.compact_bases(max_rank=3)
    assert ranks[NAME] == 3 and locker.locked_rank(NAME) == 3
    basis = locker.basis(NAME)
    assert basis is not None and torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5)
    with pytest.raises(ValueError):
        locker.compact_bases(max_rank=0)
