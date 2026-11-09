"""Multi-island speciation and nullspace-protected patch crossover for W-RSI (WRSI-BL-01).

Maintains ``num_islands`` domain-specialized weight-delta overlays over a shared base model ``M_0``. Each island
evolves on its assigned ``TaskDomain`` inside a ``CoWWeightOverlay``, and every ``crossover_interval`` steps a
crossover tournament merges non-empty island deltas using ``NullspaceLocker.project_into_nullspace`` +
``merge_deltas(..., strategy="nullspace-chain")``. If the crossed patch improves total cross-domain fitness while
satisfying the Anchor Vault KL tether, it is promoted into the shared model and synchronized across all islands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from libweightspace.adapters.base import ArchitectureAdapter
from libweightspace.candidates import CandidateSource, ProposalContext
from libweightspace.cow_overlay import SliceBackup, apply_delta_with_backup
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from libweightspace.sandbox import CloningWeightView, SandboxError, execute_weight_surgery_script
from neuroforge.merger import merge_deltas

from .scientist_agent import AnchorKLFn, PromotedGeneration, TaskDomain, WRSIScientistLoop


@dataclass
class IslandState:
    """State of one evolutionary island."""

    island_id: int
    domain: TaskDomain
    cumulative_delta: dict[str, torch.Tensor] = field(default_factory=dict)
    best_score: float = float("-inf")
    promotions: int = 0


@dataclass
class CrossoverReport:
    """Outcome of a multi-island crossover tournament."""

    step: int
    participating_islands: list[int]
    score_before: float
    score_after: float
    kl_divergence: float
    promoted: bool
    note: str = ""


class IslandPopulation:
    """Manages ``num_islands`` specialized weight-space lineages with periodic nullspace-protected crossover."""

    def __init__(
        self,
        loop: WRSIScientistLoop,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        domains: Sequence[TaskDomain],
        proposer: CandidateSource,
        anchor_kl_fn: AnchorKLFn,
        num_islands: int = 2,
        crossover_interval: int = 2,
        candidates_per_island: int = 2,
    ):
        if num_islands < 1:
            raise ValueError("num_islands must be >= 1")
        if not domains:
            raise ValueError("IslandPopulation requires at least one TaskDomain")
        self.loop = loop
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.domains = list(domains)
        self.proposer = proposer
        self.anchor_kl_fn = anchor_kl_fn
        self.num_islands = num_islands
        self.crossover_interval = max(1, crossover_interval)
        self.candidates_per_island = max(1, candidates_per_island)
        self.islands: list[IslandState] = [
            IslandState(island_id=i, domain=self.domains[i % len(self.domains)])
            for i in range(num_islands)
        ]

    def _total_domain_score(self, model: nn.Module) -> float:
        return sum(float(d.evaluate(model)) for d in self.domains)

    def step_island(self, island: IslandState, step: int) -> bool:
        """Runs one evolutionary mutation step on ``island`` inside a CoW overlay and updates ``island.cumulative_delta``."""
        model = self.loop.model
        overlay = self.loop.overlay
        dom = island.domain

        with overlay.apply_delta(island.cumulative_delta) as isl_model:
            base_score = float(dom.evaluate(isl_model))
            if island.best_score == float("-inf"):
                island.best_score = base_score
            prober = MechanisticProber(isl_model, self.adapter, self.tokenizer)
            diagnostics = build_diagnostics(prober.trace_contrastive(dom.fail_texts, dom.pass_texts, top_k_cols=32))
            params = dict(isl_model.named_parameters())

        ctx = ProposalContext(
            iteration=step,
            diagnostics=diagnostics,
            topology={"num_layers": self.adapter.num_layers(), "hidden_size": self.adapter.hidden_size()},
            history=[],
            intent_description=f"Island {island.island_id} specialization on domain {dom.name} ({dom.failure_taxonomy})",
            num_candidates=self.candidates_per_island,
        )
        candidates = self.proposer.propose(ctx)[: self.candidates_per_island]
        total_params = sum(p.numel() for p in model.parameters())

        best_new_cum: dict[str, torch.Tensor] | None = None
        best_cand_score = base_score
        for cand in candidates:
            try:
                raw = execute_weight_surgery_script(cand.python_script, CloningWeightView(params), diagnostics)
            except SandboxError:
                continue
            if not raw:
                continue
            projected = {k: self.loop.nullspace.project_into_nullspace(k, v) for k, v in raw.items()}
            nnz = sum(int((v != 0).sum().item()) for v in raw.values())
            if total_params > 0 and (nnz / total_params) > self.loop.max_sparsity:
                continue

            combined = {k: v.clone() for k, v in island.cumulative_delta.items()}
            for k, v in projected.items():
                combined[k] = combined[k] + v.float() if k in combined else v.float().clone()

            self.loop.candidate_tensors = sorted(combined)
            try:
                with overlay.apply_delta(combined) as forked:
                    kl_ok, _kl_val = self.anchor_kl_fn(forked)
                    if not kl_ok:
                        continue
                    score = float(dom.evaluate(forked))
            finally:
                self.loop.candidate_tensors = []

            if score > best_cand_score + 1e-9:
                best_cand_score = score
                best_new_cum = combined

        if best_new_cum is not None:
            island.cumulative_delta = best_new_cum
            island.best_score = best_cand_score
            island.promotions += 1
            return True
        return False

    def crossover_tournament(self, step: int) -> CrossoverReport:
        """Combines active island deltas via ``nullspace-chain`` fusion and promotes if cross-domain fitness improves."""
        active = [isl for isl in self.islands if isl.cumulative_delta]
        model = self.loop.model
        overlay = self.loop.overlay
        score_before = self._total_domain_score(model)

        if not active:
            return CrossoverReport(
                step=step,
                participating_islands=[],
                score_before=score_before,
                score_after=score_before,
                kl_divergence=0.0,
                promoted=False,
                note="no active island deltas",
            )

        if len(active) == 1:
            merged = {
                k: self.loop.nullspace.project_into_nullspace(k, v)
                for k, v in active[0].cumulative_delta.items()
            }
        else:
            raw_merged = merge_deltas([isl.cumulative_delta for isl in active], strategy="nullspace-chain")
            merged = {
                k: self.loop.nullspace.project_into_nullspace(k, v)
                for k, v in raw_merged.items()
            }

        self.loop.candidate_tensors = sorted(merged)
        try:
            with overlay.apply_delta(merged) as forked:
                kl_ok, kl_val = self.anchor_kl_fn(forked)
                score_after = self._total_domain_score(forked) if kl_ok else float("-inf")
        finally:
            self.loop.candidate_tensors = []

        if not kl_ok or score_after <= score_before + 1e-9:
            return CrossoverReport(
                step=step,
                participating_islands=[isl.island_id for isl in active],
                score_before=score_before,
                score_after=score_after if score_after != float("-inf") else score_before,
                kl_divergence=float(kl_val),
                promoted=False,
                note="kl veto" if not kl_ok else "no cross-domain improvement",
            )

        # Promote merged delta into the shared base model and record rollback point.
        params = dict(model.named_parameters())
        backups: list[SliceBackup] = []
        for t_name, d_tensor in merged.items():
            backup = apply_delta_with_backup(params[t_name], t_name, d_tensor)
            if backup is not None:
                backups.append(backup)
        null_before = {k: self.loop.nullspace.basis(k) for k in merged}
        pass_texts = [t for d in self.domains for t in d.pass_texts[:2]]
        if pass_texts:
            for tname, acts in capture_linear_inputs(model, self.tokenizer, pass_texts, list(merged)).items():
                self.loop.nullspace.register_protected_directions(tname, acts, top_k=self.loop.nullspace_top_k)

        self.loop.rollback_stack.append(
            PromotedGeneration(
                generation=self.loop.generation,
                trial_id=f"crossover_step_{step}",
                tensors=sorted(merged),
                backups=backups,
                nullspace_before=null_before,
            )
        )
        self.loop.record_trial(
            trial_id=f"crossover_step_{step}",
            domain="island_crossover",
            taxonomy="speciation_crossover",
            hypothesis=f"Nullspace-chained crossover across islands {[isl.island_id for isl in active]}",
            code="# nullspace-chain crossover",
            target_tensors=sorted(merged),
            sparsity_ratio=0.0,
            frobenius_rel_norm=0.0,
            target_score_delta=score_after - score_before,
            holdout_kl_divergence=float(kl_val),
            promoted=True,
        )
        self.loop.generation += 1
        # Reset island deltas relative to the newly promoted trunk state.
        for isl in self.islands:
            isl.cumulative_delta.clear()
            isl.best_score = float("-inf")

        return CrossoverReport(
            step=step,
            participating_islands=[isl.island_id for isl in active],
            score_before=score_before,
            score_after=score_after,
            kl_divergence=float(kl_val),
            promoted=True,
            note=f"promoted generation {self.loop.generation}",
        )
