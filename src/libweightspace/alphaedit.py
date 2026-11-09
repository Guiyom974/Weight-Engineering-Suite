"""AlphaEdit empirical covariance nullspace and closed-form multi-fact MEMIT (NF-BL-02).

Implements the AlphaEdit (ICLR 2025 Outstanding Paper) formulation for lifelong weight editing:
    1. ``compute_covariance_nullspace`` captures pre-linear input activations ``K_0`` on reference texts,
       computes the uncentered second-moment matrix ``C = K_0 @ K_0^T``, extracts eigenvectors with
       eigenvalue ``<= eig_threshold * lambda_max``, and returns the symmetric projection matrix
       ``P_null = U_null @ U_null^T`` of shape ``[d_in, d_in]``.
    2. ``alphaedit_batch_update`` solves for the closed-form multi-fact update projected onto ``P_null``:
       ``delta_W = R @ K^T @ P_null @ (K @ K^T @ P_null + ridge * I)^{-1}``
       where ``R = V - W @ K`` is the residual matrix for target key/value pairs ``(K, V)``.
       Because ``delta_W`` lies in the column space of ``P_null`` on the right, ``delta_W @ k_0 approx 0``
       for every vector ``k_0`` in the empirical covariance range.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .prober import capture_linear_inputs


def project_matrix_from_activations(
    activations: torch.Tensor,
    eig_threshold: float = 1e-2,
) -> torch.Tensor:
    """Computes ``P_null = U_null @ U_null^T`` from sample rows ``activations`` of shape ``[n_samples, d_in]``.

    Args:
        activations: 2-D tensor ``[n_samples, d_in]`` of input vectors ``k_0^T``.
        eig_threshold: Relative threshold against the largest eigenvalue; eigenvectors with
            ``lambda_i <= eig_threshold * max(lambda_max, 1e-12)`` form ``U_null``.

    Returns:
        Symmetric nullspace projection matrix ``P_null`` of shape ``[d_in, d_in]`` in float32.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2-D [n_samples, d_in], got {tuple(activations.shape)}")
    x = activations.detach().double()
    d_in = x.shape[1]
    if x.shape[0] == 0:
        return torch.eye(d_in, dtype=torch.float32, device=activations.device)

    cov = x.T @ x  # [d_in, d_in]
    evals, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues in float64
    max_eval = float(evals[-1].item()) if evals.numel() > 0 else 0.0
    if max_eval <= 1e-12:
        return torch.eye(d_in, dtype=torch.float32, device=activations.device)

    cutoff = eig_threshold * max_eval
    mask = evals <= cutoff
    if not bool(mask.any().item()):
        # Fall back to the smallest eigenvector so the nullspace is never empty.
        u_null = evecs[:, :1]
    else:
        u_null = evecs[:, mask]
    return (u_null @ u_null.T).float()


@torch.no_grad()
def compute_covariance_nullspace(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    param_names: Sequence[str],
    eig_threshold: float = 1e-2,
) -> dict[str, torch.Tensor]:
    """Captures input activations for ``param_names`` on ``texts`` and returns per-tensor ``P_null`` matrices.

    Args:
        model: Causal LM module.
        tokenizer: Tokenizer callable returning ``input_ids``.
        texts: Calibration corpus texts.
        param_names: Weight parameter names (e.g. ``layers.2.mlp.down_proj.weight``).
        eig_threshold: Relative eigenvalue threshold for the nullspace cutoff.

    Returns:
        Mapping ``param_name -> P_null`` where each ``P_null`` has shape ``[d_in, d_in]``.
    """
    captured = capture_linear_inputs(model, tokenizer, list(texts), list(param_names))
    result: dict[str, torch.Tensor] = {}
    params = dict(model.named_parameters())
    for name in param_names:
        if name in captured and captured[name].numel() > 0:
            result[name] = project_matrix_from_activations(captured[name], eig_threshold=eig_threshold)
        elif name in params and params[name].ndim == 2:
            d_in = int(params[name].shape[1])
            result[name] = torch.eye(d_in, dtype=torch.float32)
    return result


def alphaedit_batch_update(
    W: torch.Tensor,
    keys: torch.Tensor,
    target_values: torch.Tensor,
    P_null: torch.Tensor | None = None,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """Computes the closed-form AlphaEdit / MEMIT update ``delta_W`` for a batch of associative facts.

    Accepts ``keys`` as either ``[d_in, m]`` (columns are key vectors) or ``[m, d_in]`` (rows are key vectors),
    and ``target_values`` as ``[d_out, m]`` or ``[m, d_out]``.

    Args:
        W: Base weight matrix ``[d_out, d_in]``.
        keys: Fact key activations ``K``.
        target_values: Desired output vectors ``V`` such that ``(W + delta_W) @ K approx V``.
        P_null: Optional nullspace projector ``[d_in, d_in]`` (defaults to identity).
        ridge: Tikhonov regularization scalar ``lambda > 0``.

    Returns:
        ``delta_W`` of shape ``[d_out, d_in]`` matching ``W``'s dtype and device.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D [d_out, d_in], got {tuple(W.shape)}")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    d_out, d_in = W.shape
    w_f = W.detach().float()

    k_f = keys.detach().float().to(w_f.device)
    v_f = target_values.detach().float().to(w_f.device)
    if k_f.ndim == 1:
        k_f = k_f.unsqueeze(1)
    if v_f.ndim == 1:
        v_f = v_f.unsqueeze(1)
    if k_f.shape[0] != d_in and k_f.shape[1] == d_in:
        k_f = k_f.T
    if v_f.shape[0] != d_out and v_f.shape[1] == d_out:
        v_f = v_f.T
    if k_f.shape[0] != d_in or v_f.shape[0] != d_out or k_f.shape[1] != v_f.shape[1]:
        raise ValueError(
            f"Incompatible shapes: W={tuple(W.shape)}, keys={tuple(keys.shape)}, "
            f"target_values={tuple(target_values.shape)}"
        )

    p = (
        torch.eye(d_in, dtype=torch.float32, device=w_f.device)
        if P_null is None
        else P_null.detach().float().to(w_f.device)
    )
    if p.shape != (d_in, d_in):
        raise ValueError(f"P_null must have shape ({d_in}, {d_in}), got {tuple(p.shape)}")

    # Project keys into the nullspace so delta_W = R @ (P_null K)^T @ ((P_null K)(P_null K)^T + ridge * I)^{-1}
    # which guarantees delta_W = (...) @ P_null^T = (...) @ P_null, so delta_W @ k_0 = 0 for all k_0 in col(K_0).
    k_proj = p @ k_f  # [d_in, m]
    residual = v_f - (w_f @ k_f)  # [d_out, m]
    gram = (k_proj @ k_proj.T) + float(ridge) * torch.eye(d_in, dtype=torch.float32, device=w_f.device)
    rhs = (residual @ k_proj.T).T  # [d_in, d_out]
    solved = torch.linalg.solve(gram, rhs).T  # [d_out, d_in]
    # Enforce exact right-projection onto P_null to eliminate floating-point drift outside the nullspace.
    delta = solved @ p
    return delta.to(dtype=W.dtype, device=W.device)
