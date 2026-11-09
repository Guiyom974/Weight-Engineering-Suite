"""Inner-loop hyperparameter autotuner for weight surgery scripts (NF-BL-06).

Decouples structural operator synthesis (which the LLM Reasoner proposes) from continuous hyperparameter search
over projection strength ``alpha`` and column sparsity ``k`` inside ``CoWWeightOverlay`` forks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.sandbox import CloningWeightView, SandboxError, execute_weight_surgery_script

from .convergence_engine import check_constraints
from .schemas import EditConstraints


@dataclass
class AutotuneTrial:
    """Single ``(alpha, k)`` trial evaluated in a CoW overlay fork."""

    alpha: float
    k: int
    target_score: float
    invariant_degradation: float
    feasible: bool
    modified_params: int
    error_or_violation: str | None = None


@dataclass
class AutotuneResult:
    """Best hyperparameter configuration discovered by ``autotune_candidate``."""

    best_alpha: float
    best_k: int
    best_deltas: dict[str, torch.Tensor]
    best_target_score: float
    best_invariant_degradation: float
    trials: list[AutotuneTrial] = field(default_factory=list)


def _slice_diagnostics_top_k(diagnostics: Mapping[str, Any], k: int) -> dict[str, Any]:
    """Produces a copy of ``diagnostics`` with per-layer ``top_culpable_mlp_cols`` truncated to ``k``."""
    out = dict(diagnostics)
    out["autotune_k"] = k
    if "layers" in out and isinstance(out["layers"], list):
        sliced_layers = []
        for layer_entry in out["layers"]:
            if isinstance(layer_entry, Mapping):
                le = dict(layer_entry)
                if "top_culpable_mlp_cols" in le and isinstance(le["top_culpable_mlp_cols"], list):
                    le["top_culpable_mlp_cols"] = list(le["top_culpable_mlp_cols"][:k])
                if "top_mlp_col_alignments" in le and isinstance(le["top_mlp_col_alignments"], list):
                    le["top_mlp_col_alignments"] = list(le["top_mlp_col_alignments"][:k])
                sliced_layers.append(le)
            else:
                sliced_layers.append(layer_entry)
        out["layers"] = sliced_layers
    if "top_culpable_mlp_cols" in out and isinstance(out["top_culpable_mlp_cols"], list):
        out["top_culpable_mlp_cols"] = list(out["top_culpable_mlp_cols"][:k])
    return out


def autotune_candidate(
    model: nn.Module,
    overlay: CoWWeightOverlay,
    params: Mapping[str, torch.Tensor],
    total_params: int,
    constraints: EditConstraints,
    script_code: str,
    diagnostics: Mapping[str, Any],
    evaluate_fn: Callable[[nn.Module], tuple[float, float, bool]],
    alphas: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    k_values: Sequence[int] = (2, 3, 5, 8),
) -> AutotuneResult:
    """Grid/coordinate search over ``(alpha, k)`` for a synthesized ``script_code``.

    For each ``k`` in ``k_values``, ``diagnostics`` is truncated to the top-``k`` culpable columns and
    ``script_code`` is executed once; its output delta is then scaled by each ``alpha`` in ``alphas``,
    checked against ``constraints``, and evaluated inside ``overlay``.

    Args:
        model: Target causal LM.
        overlay: Active ``CoWWeightOverlay`` bound to ``model``.
        params: Mapping of parameter names to base tensors.
        total_params: Total scalar count in ``model``.
        constraints: Mathematical edit bounds.
        script_code: Synthesized ``compute_delta(weights, diagnostics)`` source.
        diagnostics: Contrastive prober diagnostics.
        evaluate_fn: Callback returning ``(target_score, invariant_degradation, invariants_ok)`` on a CoW fork.
        alphas: Candidate scalar multipliers on the base delta.
        k_values: Candidate column-count caps on ``top_culpable_mlp_cols``.

    Returns:
        ``AutotuneResult`` containing the Pareto-best feasible ``(alpha, k, deltas)`` and the trial log.
    """
    trials: list[AutotuneTrial] = []
    best_alpha = float(alphas[0]) if alphas else 1.0
    best_k = int(k_values[0]) if k_values else 3
    best_deltas: dict[str, torch.Tensor] = {}
    best_key: tuple[bool, float, float, int] | None = None
    best_target = float("-inf")
    best_deg = float("inf")

    for k in k_values:
        diag_k = _slice_diagnostics_top_k(diagnostics, int(k))
        try:
            base_deltas = execute_weight_surgery_script(script_code, CloningWeightView(params), diag_k)
        except SandboxError as exc:
            for alpha in alphas:
                trials.append(
                    AutotuneTrial(
                        alpha=float(alpha),
                        k=int(k),
                        target_score=float("-inf"),
                        invariant_degradation=float("inf"),
                        feasible=False,
                        modified_params=0,
                        error_or_violation=str(exc),
                    )
                )
            continue

        for alpha in alphas:
            scaled = {name: d * float(alpha) for name, d in base_deltas.items()}
            nnz = sum(int((d != 0).sum().item()) for d in scaled.values())
            violations = check_constraints(scaled, params, total_params, constraints)
            if violations:
                trials.append(
                    AutotuneTrial(
                        alpha=float(alpha),
                        k=int(k),
                        target_score=float("-inf"),
                        invariant_degradation=float("inf"),
                        feasible=False,
                        modified_params=nnz,
                        error_or_violation="; ".join(violations),
                    )
                )
                continue

            with torch.no_grad(), overlay.apply_delta(scaled) as forked:
                t_score, inv_deg, inv_ok = evaluate_fn(forked)

            feasible = bool(inv_ok)
            trials.append(
                AutotuneTrial(
                    alpha=float(alpha),
                    k=int(k),
                    target_score=float(t_score),
                    invariant_degradation=float(inv_deg),
                    feasible=feasible,
                    modified_params=nnz,
                )
            )
            cand_key = (feasible, float(t_score), -float(inv_deg), -nnz)
            if best_key is None or cand_key > best_key:
                best_key = cand_key
                best_alpha = float(alpha)
                best_k = int(k)
                best_deltas = scaled
                best_target = float(t_score)
                best_deg = float(inv_deg)

    return AutotuneResult(
        best_alpha=best_alpha,
        best_k=best_k,
        best_deltas=best_deltas,
        best_target_score=best_target,
        best_invariant_degradation=best_deg,
        trials=trials,
    )
