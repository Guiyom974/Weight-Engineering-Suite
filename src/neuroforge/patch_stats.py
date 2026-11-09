"""Audit statistics for weight patches and model fingerprints.

``tensor_delta_stats`` fills ``TensorDeltaStats`` (sparsity, Frobenius and spectral norms before/after);
``summarize_patch`` aggregates a whole delta dict; ``state_fingerprint`` hashes a model's parameters so a
``PatchManifest`` can name the exact base weights it applies to.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
import torch.nn as nn

from libweightspace.spectral import spectral_norm

from .schemas import TensorDeltaStats


def tensor_delta_stats(
    name: str,
    base: torch.Tensor,
    delta: torch.Tensor,
    structural_delta: torch.Tensor | None = None,
) -> TensorDeltaStats:
    """Statistics of one delta relative to its base tensor.

    When ``structural_delta`` (the pre-nullspace surgical patch) is provided, ``nnz_count`` and
    ``sparsity_ratio`` measure the structural parameters targeted by the script rather than dense
    floating-point nullspace compensation fill-in, while norms reflect the actual applied ``delta``.
    """
    base_f = base.detach().float()
    delta_f = delta.detach().float().to(base_f.device)
    struct_f = (
        structural_delta.detach().float().to(base_f.device)
        if structural_delta is not None
        else delta_f
    )
    nnz = int((struct_f != 0).sum().item())
    total = base_f.numel()
    frob_delta = float(torch.linalg.vector_norm(delta_f).item())
    frob_base = float(torch.linalg.vector_norm(base_f).item())
    return TensorDeltaStats(
        tensor_name=name,
        shape=list(base_f.shape),
        nnz_count=nnz,
        total_elements=total,
        sparsity_ratio=nnz / total if total else 0.0,
        frobenius_norm_delta=frob_delta,
        frobenius_norm_base=frob_base,
        relative_frobenius=frob_delta / (frob_base + 1e-12),
        spectral_norm_before=spectral_norm(base_f),
        spectral_norm_after=spectral_norm(base_f + delta_f),
    )


def summarize_patch(
    base_params: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
    total_model_params: int,
    structural_deltas: Mapping[str, torch.Tensor] | None = None,
) -> tuple[list[TensorDeltaStats], int, float]:
    """Per-tensor stats, total modified parameters, and the modified fraction of the whole model."""
    stats = [
        tensor_delta_stats(
            name,
            base_params[name],
            d,
            structural_delta=structural_deltas.get(name) if structural_deltas is not None else None,
        )
        for name, d in deltas.items()
    ]
    modified = sum(s.nnz_count for s in stats)
    return stats, modified, modified / total_model_params if total_model_params else 0.0


def state_fingerprint(model: nn.Module) -> str:
    """SHA-256 over sorted parameter names, shapes, dtypes and raw bytes."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items(), key=lambda kv: kv[0]):
        t = p.detach().cpu().contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}\n".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
