"""Norm-preserving biprojection and DoRA magnitude/direction decoupling operators (NF-BL-01).

Provides closed-form tensor surgery primitives exposed inside the sandbox as ``torch.surgery``:
    - ``biproject_direction``: two-pass Gram-Schmidt orthogonalization of a target direction against a benign
      or capability subspace basis so that surgical projection leaves harmless directions untouched.
    - ``norm_preserving_column_edit``: DoRA-style magnitude/direction decoupling on selected matrix columns
      ``W[:, cols]`` that removes a scaled component along the (optionally biprojected) direction and rescales
      each modified column back to its exact pre-edit L2 norm ``||W[:, c]||_2``, preventing RMSNorm gain
      miscalibration downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def biproject_direction(
    target_dir: torch.Tensor,
    benign_basis: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Orthogonalizes ``target_dir`` against ``benign_basis`` using two-pass Gram-Schmidt and normalizes it.

    Args:
        target_dir: 1-D vector ``[d]`` representing the undesired or target direction.
        benign_basis: Optional 1-D ``[d]`` or 2-D ``[d, m]`` / ``[m, d]`` tensor spanning benign directions
            that must not be perturbed.
        eps: Numerical floor for norm division.

    Returns:
        Unit vector ``u_perp`` of shape ``[d]`` (in ``target_dir``'s dtype/device) orthogonal to ``benign_basis``.

    Raises:
        ValueError: If ``target_dir`` is not 1-D or ``benign_basis`` has incompatible dimensions.
    """
    if target_dir.ndim != 1:
        raise ValueError(f"target_dir must be 1-D, got shape {tuple(target_dir.shape)}")
    d = target_dir.shape[0]
    u = target_dir.detach().float().clone()
    base_norm = torch.linalg.vector_norm(u)
    if float(base_norm.item()) <= eps:
        return torch.zeros_like(target_dir)
    u = u / base_norm

    if benign_basis is not None and benign_basis.numel() > 0:
        b = benign_basis.detach().float().to(u.device)
        if b.ndim == 1:
            if b.shape[0] != d:
                raise ValueError(f"benign_basis length {b.shape[0]} != target_dir dim {d}")
            b = b.unsqueeze(1)
        elif b.ndim == 2:
            if b.shape[0] != d and b.shape[1] == d:
                b = b.T
            elif b.shape[0] != d:
                raise ValueError(
                    f"benign_basis shape {tuple(benign_basis.shape)} incompatible with target_dir dim {d}"
                )
        else:
            raise ValueError(f"benign_basis must be 1-D or 2-D, got {b.ndim}-D")

        # Orthonormalize benign_basis via reduced QR for numerical stability.
        q, _ = torch.linalg.qr(b, mode="reduced")
        # Two-pass Kahan-Parlett Gram-Schmidt to eliminate floating-point loss of orthogonality.
        for _ in range(2):
            u = u - q @ (q.T @ u)

    norm_after = torch.linalg.vector_norm(u)
    if float(norm_after.item()) <= eps:
        return torch.zeros_like(target_dir)
    u = u / norm_after
    return u.to(dtype=target_dir.dtype, device=target_dir.device)


def norm_preserving_column_edit(
    W: torch.Tensor,
    cols: Sequence[int] | torch.Tensor | int,
    direction: torch.Tensor | None = None,
    scale: float = 1.0,
    benign_basis: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Applies a DoRA-style norm-preserving directional edit to ``W[:, cols]`` and returns ``delta_W``.

    Each selected column ``w_c = W[:, c]`` is updated via:
        1. ``u = biproject_direction(direction, benign_basis)``
        2. ``w_c' = w_c - scale * (u^T w_c) * u``
        3. ``w_c'' = (||w_c||_2 / max(||w_c'||_2, eps)) * w_c'``
    so that ``||W[:, c] + delta_W[:, c]||_2 == ||W[:, c]||_2`` up to floating-point precision.

    Args:
        W: 2-D weight matrix ``[d_out, d_in]``.
        cols: Column indices into ``W`` (sequence of ints or 1-D integer/long tensor).
        direction: Optional 1-D target direction ``[d_out]``. Defaults to column mean.
        scale: Projection strength (``1.0`` completely removes component along ``u`` before norm restoration).
        benign_basis: Optional benign subspace basis passed to ``biproject_direction``.
        eps: Numerical floor for norm division.

    Returns:
        Sparse delta tensor ``delta_W`` of the same shape, dtype, and device as ``W``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be a 2-D matrix, got shape {tuple(W.shape)}")
    d_out, d_in = W.shape

    col_list: list[int]
    if isinstance(cols, int):
        col_list = [cols]
    elif isinstance(cols, torch.Tensor):
        col_list = [int(c) for c in cols.detach().cpu().reshape(-1).tolist()]
    else:
        col_list = [int(c) for c in cols]

    valid_cols = [c for c in col_list if 0 <= c < d_in]
    if not valid_cols:
        return torch.zeros_like(W)

    if direction is None:
        sub_mean = torch.mean(W[:, valid_cols].detach().float(), dim=1)
        if float(torch.linalg.vector_norm(sub_mean).item()) <= eps:
            direction = torch.ones(d_out, dtype=W.dtype, device=W.device)
        else:
            direction = sub_mean.to(dtype=W.dtype, device=W.device)

    if direction.ndim != 1 or direction.shape[0] != d_out:
        raise ValueError(
            f"direction must have shape ({d_out},), got {tuple(direction.shape)}"
        )

    u = biproject_direction(direction.to(device=W.device), benign_basis=benign_basis, eps=eps).float()
    if float(torch.linalg.vector_norm(u).item()) <= eps:
        return torch.zeros_like(W)

    w_float = W.detach().float()
    idx = torch.tensor(valid_cols, dtype=torch.long, device=W.device)
    sub = w_float[:, idx]  # [d_out, k]
    orig_norms = torch.linalg.vector_norm(sub, dim=0, keepdim=True)  # [1, k]

    proj_coeffs = u.unsqueeze(0) @ sub  # [1, k]
    edited = sub - float(scale) * (u.unsqueeze(1) @ proj_coeffs)  # [d_out, k]
    new_norms = torch.linalg.vector_norm(edited, dim=0, keepdim=True)  # [1, k]

    # Where new_norms > eps, restore exact pre-edit column L2 norm.
    scale_factors = torch.where(new_norms > eps, orig_norms / new_norms.clamp_min(eps), torch.ones_like(new_norms))
    restored = edited * scale_factors

    delta = torch.zeros_like(w_float)
    delta[:, idx] = restored - sub
    return delta.to(dtype=W.dtype, device=W.device)
