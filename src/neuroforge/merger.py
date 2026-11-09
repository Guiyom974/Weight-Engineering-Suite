"""Multi-branch .wgit merging: DELLA, SLERP, OrthoMerge, and Nullspace-Chained fusion (NF-BL-05).

Combines two or more weight-delta dictionaries ``deltas_list`` (e.g. from ``WeightGitRepository.cumulative_delta``
across branches) into a single conflict-free delta dictionary without destructive parameter interference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

MergeStrategy = Literal["della", "slerp", "orthomerge", "nullspace-chain"]


def _della_prune_and_rescale(t: torch.Tensor, density: float = 0.5) -> torch.Tensor:
    """DELLA magnitude-ranked pruning with inverse keep-probability rescaling and sign preservation."""
    flat = t.detach().float().reshape(-1)
    n = flat.numel()
    if n == 0 or density >= 1.0:
        return t.detach().float().clone()
    k = max(1, int(round(density * n)))
    mags = flat.abs()
    if float(mags.max().item()) <= 1e-12:
        return torch.zeros_like(t, dtype=torch.float32)

    # Rank magnitudes ascending: higher magnitude -> higher keep probability p_i in [density/2, min(1, 1.5*density)].
    order = torch.argsort(mags)
    ranks = torch.zeros_like(flat)
    ranks[order] = torch.linspace(0.0, 1.0, steps=n, device=flat.device)
    keep_prob = (density * (0.5 + ranks)).clamp(min=1e-3, max=1.0)

    # Deterministic top-k selection weighted by magnitude rank so merges are 100% reproducible.
    _, top_idx = torch.topk(mags, k=k)
    mask = torch.zeros_like(flat)
    mask[top_idx] = 1.0 / keep_prob[top_idx]
    return (flat * mask).reshape(t.shape)


def _merge_della(tensors: Sequence[torch.Tensor], density: float = 0.5) -> torch.Tensor:
    """DELLA drop-and-rescale followed by TIES sign-consensus aggregation."""
    pruned = [_della_prune_and_rescale(t, density=density) for t in tensors]
    stacked = torch.stack(pruned, dim=0)  # [B, ...]
    total_sign = torch.sign(stacked.sum(dim=0))
    agree_mask = (torch.sign(stacked) == total_sign.unsqueeze(0)).float()
    agreed = stacked * agree_mask
    counts = agree_mask.sum(dim=0).clamp_min(1.0)
    return agreed.sum(dim=0) / counts


def _merge_slerp(tensors: Sequence[torch.Tensor], t_interp: float = 0.5) -> torch.Tensor:
    """Spherical linear interpolation (folded pairwise when more than 2 branches are provided)."""
    acc = tensors[0].detach().float().clone()
    for nxt_raw in tensors[1:]:
        nxt = nxt_raw.detach().float()
        a_flat = acc.reshape(-1)
        b_flat = nxt.reshape(-1)
        na = float(torch.linalg.vector_norm(a_flat).item())
        nb = float(torch.linalg.vector_norm(b_flat).item())
        if na <= 1e-12 and nb <= 1e-12:
            acc = torch.zeros_like(acc)
            continue
        if na <= 1e-12 or nb <= 1e-12:
            acc = (1.0 - t_interp) * acc + t_interp * nxt
            continue
        u_a = a_flat / na
        u_b = b_flat / nb
        dot = float(torch.dot(u_a, u_b).clamp(-1.0, 1.0).item())
        omega = math.acos(dot)
        sin_omega = math.sin(omega)
        target_norm = (1.0 - t_interp) * na + t_interp * nb
        if abs(sin_omega) < 1e-6:
            interp_dir = (1.0 - t_interp) * u_a + t_interp * u_b
            interp_norm = float(torch.linalg.vector_norm(interp_dir).item())
            if interp_norm > 1e-12:
                interp_dir = interp_dir / interp_norm
        else:
            w_a = math.sin((1.0 - t_interp) * omega) / sin_omega
            w_b = math.sin(t_interp * omega) / sin_omega
            interp_dir = w_a * u_a + w_b * u_b
        acc = (interp_dir * target_norm).reshape(acc.shape)
    return acc


def _merge_orthomerge(tensors: Sequence[torch.Tensor], rank_cap: int = 8) -> torch.Tensor:
    """OrthoMerge: orthogonalizes each branch's delta against the column space of previous branches."""
    first = tensors[0].detach().float().clone()
    if first.ndim != 2:
        return torch.stack([t.detach().float() for t in tensors], dim=0).mean(dim=0)

    merged = first.clone()
    d_out, _ = first.shape
    occupied_cols: list[torch.Tensor] = []

    def _extract_col_basis(mat: torch.Tensor) -> torch.Tensor | None:
        if float(torch.linalg.vector_norm(mat).item()) <= 1e-12:
            return None
        u, s, _ = torch.linalg.svd(mat, full_matrices=False)
        keep = s > (float(s[0].item()) * 1e-3)
        k = min(rank_cap, int(keep.sum().item()))
        return u[:, : max(1, k)]

    b0 = _extract_col_basis(first)
    if b0 is not None:
        occupied_cols.append(b0)

    for t_raw in tensors[1:]:
        cur = t_raw.detach().float()
        if occupied_cols:
            all_u = torch.cat(occupied_cols, dim=1)
            q, _ = torch.linalg.qr(all_u, mode="reduced")
            if q.shape[1] < d_out:
                cur = cur - q @ (q.T @ cur)
        merged = merged + cur
        b_cur = _extract_col_basis(cur)
        if b_cur is not None:
            occupied_cols.append(b_cur)
    return merged


def _merge_nullspace_chain(tensors: Sequence[torch.Tensor], rank_cap: int = 8) -> torch.Tensor:
    """Nullspace-chained merge: projects each subsequent delta onto the right-nullspace of earlier deltas."""
    first = tensors[0].detach().float().clone()
    if first.ndim != 2:
        return torch.stack([t.detach().float() for t in tensors], dim=0).mean(dim=0)

    merged = first.clone()
    _, d_in = first.shape
    occupied_rows: list[torch.Tensor] = []

    def _extract_row_basis(mat: torch.Tensor) -> torch.Tensor | None:
        if float(torch.linalg.vector_norm(mat).item()) <= 1e-12:
            return None
        _, s, vh = torch.linalg.svd(mat.double(), full_matrices=False)
        keep = s > (float(s[0].item()) * 1e-3)
        k = min(rank_cap, int(keep.sum().item()))
        return vh[: max(1, k), :].T.float()  # [d_in, k]

    v0 = _extract_row_basis(first)
    if v0 is not None:
        occupied_rows.append(v0)

    for t_raw in tensors[1:]:
        cur = t_raw.detach().float()
        if occupied_rows:
            all_v = torch.cat(occupied_rows, dim=1).double()
            q, _ = torch.linalg.qr(all_v, mode="reduced")
            if q.shape[1] < d_in:
                cur_d = cur.double()
                cur = (cur_d - (cur_d @ q) @ q.T).float()
        merged = merged + cur
        v_cur = _extract_row_basis(cur)
        if v_cur is not None:
            occupied_rows.append(v_cur)
    return merged


def merge_deltas(
    deltas_list: Sequence[Mapping[str, torch.Tensor]],
    strategy: MergeStrategy = "orthomerge",
    density: float = 0.5,
    t_interp: float = 0.5,
    rank_cap: int = 8,
) -> dict[str, torch.Tensor]:
    """Merges a sequence of weight-delta mappings using ``strategy``.

    Args:
        deltas_list: 2 or more delta dictionaries ``param_name -> delta_tensor``.
        strategy: One of ``"della"``, ``"slerp"``, ``"orthomerge"``, or ``"nullspace-chain"``.
        density: Retention fraction in ``(0, 1]`` for ``"della"``.
        t_interp: Interpolation weight in ``[0, 1]`` for ``"slerp"``.
        rank_cap: Maximum protected singular rank per branch for ``"orthomerge"`` and ``"nullspace-chain"``.

    Returns:
        Combined delta dictionary ``param_name -> merged_delta``.
    """
    if len(deltas_list) < 2:
        raise ValueError("merge_deltas requires at least 2 delta dictionaries")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if not 0.0 <= t_interp <= 1.0:
        raise ValueError("t_interp must be in [0, 1]")

    all_names = sorted({k for d in deltas_list for k in d})
    out: dict[str, torch.Tensor] = {}
    for name in all_names:
        # Determine canonical shape/dtype from the first branch that contains this parameter.
        exemplar = next(d[name] for d in deltas_list if name in d)
        branch_tensors = [
            d[name] if name in d else torch.zeros_like(exemplar) for d in deltas_list
        ]
        if strategy == "della":
            merged = _merge_della(branch_tensors, density=density)
        elif strategy == "slerp":
            merged = _merge_slerp(branch_tensors, t_interp=t_interp)
        elif strategy == "orthomerge":
            merged = _merge_orthomerge(branch_tensors, rank_cap=rank_cap)
        elif strategy == "nullspace-chain":
            merged = _merge_nullspace_chain(branch_tensors, rank_cap=rank_cap)
        else:
            raise ValueError(f"Unknown merge strategy '{strategy}'")
        if bool((merged != 0).any().item()):
            out[name] = merged.to(dtype=exemplar.dtype, device=exemplar.device)
    return out
