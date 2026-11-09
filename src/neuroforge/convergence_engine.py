"""Mode 3: iterative test-driven convergence with a multi-candidate Pareto tournament.

Each iteration:
    1. Evaluate target goals and invariant guards on the current model; stop if every goal is met.
    2. Probe (contrastive tracing on the intent's fail/pass texts) -> diagnostics.
    3. Ask the candidate source (LLM reasoner or template proposer) for K scripts, given the history.
    4. Per candidate: sandboxed ``compute_delta`` -> ``EditConstraints`` check (allowed modules, modified-param
       ratio, relative Frobenius, spectral shift) -> nullspace projection against invariant-prompt directions
       -> CoW evaluation of target + invariant suites.
    5. Tournament: feasible candidates (no error, no violation, every invariant within ``max_allowed_drop`` of the
       loop's starting baseline, target score strictly improved) form a Pareto front over (target score up,
       invariant degradation down, modified params down); the winner is the front member with the best target
       score (ties: less degradation, fewer params).
    6. Apply the winner in place and commit it to ``.wgit`` with a full ``PatchManifest``.

Invariants are always measured against the baseline captured when ``run`` starts, so small per-iteration drifts
cannot accumulate past the guard.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from libweightspace.adapters.base import ArchitectureAdapter
from libweightspace.candidates import CandidateParseError, CandidatePatch, CandidateSource, ProposalContext
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.llm_backends import LLMBackendError
from libweightspace.nullspace import NullspaceLocker
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from libweightspace.sandbox import CloningWeightView, SandboxError, execute_weight_surgery_script
from libweightspace.spectral import profile_matrix, spectral_norm_shift

from .intent_compiler import contrastive_texts, invariant_prompts
from .metrics import EvaluationSuite, metric_drop
from .patch_stats import state_fingerprint, summarize_patch
from .reasoner_bridge import topology_summary
from .schemas import EditConstraints, IntentSpec, PatchManifest, TensorDeltaStats
from .weight_git import WeightGitRepository, apply_deltas_inplace

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    """Outcome of one candidate in one iteration."""

    index: int
    candidate: CandidatePatch
    error: str | None = None
    violations: list[str] = field(default_factory=list)
    target_metrics: dict[str, float] = field(default_factory=dict)
    invariant_metrics: dict[str, float] = field(default_factory=dict)
    target_score: float = float("-inf")
    invariant_degradation: float = float("inf")
    invariants_ok: bool = False
    goals_met: bool = False
    modified_params: int = 0
    sparsity_ratio: float = 0.0
    tensor_stats: list[TensorDeltaStats] = field(default_factory=list)
    deltas: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def edit_size(self) -> float:
        """Sum of per-tensor relative Frobenius norms (surgical-minimality tie-breaker)."""
        return sum(s.relative_frobenius for s in self.tensor_stats)

    @property
    def feasible(self) -> bool:
        """No error, no constraint violation, invariants within bounds."""
        return self.error is None and not self.violations and self.invariants_ok


@dataclass
class IterationRecord:
    """One iteration of the loop."""

    iteration: int
    target_before: dict[str, float]
    invariant_before: dict[str, float]
    evaluations: list[CandidateEvaluation]
    winner: int | None = None
    committed_patch_id: str | None = None
    note: str = ""


@dataclass
class ConvergenceResult:
    """Final loop outcome."""

    converged: bool
    iterations: list[IterationRecord]
    target_metrics: dict[str, float]
    invariant_metrics: dict[str, float]
    head: str | None
    cancelled: bool = False


def check_constraints(
    deltas: Mapping[str, torch.Tensor],
    params: Mapping[str, torch.Tensor],
    total_params: int,
    constraints: EditConstraints,
) -> list[str]:
    """Returns human-readable violations of ``constraints`` (empty when compliant)."""
    violations: list[str] = []
    nnz = 0
    for name, d in deltas.items():
        if not any(fnmatch.fnmatch(name, pat) for pat in constraints.allowed_module_patterns):
            violations.append(f"{name}: not in allowed_module_patterns")
            continue
        base = params[name].detach().float()
        dd = d.detach().float().to(base.device)
        nnz += int((dd != 0).sum())
        rel = float(torch.linalg.vector_norm(dd) / (torch.linalg.vector_norm(base) + 1e-12))
        if rel > constraints.max_frobenius_rel_norm:
            violations.append(f"{name}: relative Frobenius {rel:.4f} > {constraints.max_frobenius_rel_norm}")
        if base.ndim == 2:
            shift = spectral_norm_shift(base, dd)
            if shift > constraints.max_spectral_norm_shift:
                violations.append(
                    f"{name}: spectral shift {shift:.4f} > {constraints.max_spectral_norm_shift}"
                )
            if constraints.min_post_edit_alpha is not None:
                post_alpha = profile_matrix(base + dd, tensor_name=name).power_law_alpha
                if post_alpha < constraints.min_post_edit_alpha:
                    violations.append(
                        f"{name}: post-edit HTSR alpha {post_alpha:.3f} < {constraints.min_post_edit_alpha}"
                    )
    ratio = nnz / total_params if total_params else 0.0
    if ratio > constraints.max_modified_param_ratio:
        violations.append(f"modified ratio {ratio:.6f} > {constraints.max_modified_param_ratio}")
    return violations


class ConvergenceEngine:
    """Runs the Mode 3 loop for one intent on one model."""

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        intent: IntentSpec,
        target_suites: Mapping[str, EvaluationSuite],
        invariant_suites: Mapping[str, EvaluationSuite],
        candidate_source: CandidateSource,
        repo: WeightGitRepository | None = None,
        nullspace: NullspaceLocker | None = None,
        base_model_id: str = "unknown",
        max_iterations: int = 3,
        candidates_per_iteration: int = 3,
        top_k_cols: int = 64,
        nullspace_reference: nn.Module | None = None,
        baseline_model: nn.Module | None = None,
        progress: Callable[[str, float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        protected_tensors: Sequence[str] = (),
        approve: Callable[[CandidateEvaluation], bool] | None = None,
        commit: bool = True,
    ):
        # Optional hooks (CR5, CR7): progress reporting, cooperative cancel, protected tensors,
        # approval before commit, and dry runs that evaluate without changing the model or the history.
        self.progress = progress
        self.should_cancel = should_cancel
        self.protected_tensors = tuple(protected_tensors)
        self.approve = approve
        self.commit_enabled = commit
        if max_iterations < 1 or candidates_per_iteration < 1:
            raise ValueError("max_iterations and candidates_per_iteration must be >= 1")
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.intent = intent
        self.target_suites = dict(target_suites)
        self.invariant_suites = dict(invariant_suites)
        self.source = candidate_source
        self.repo = repo
        self.nullspace = nullspace if nullspace is not None else NullspaceLocker()
        self.base_model_id = base_model_id
        self.max_iterations = max_iterations
        self.candidates_per_iteration = candidates_per_iteration
        self.top_k_cols = top_k_cols
        self.nullspace_reference = nullspace_reference
        # Invariants are measured against this model, so separate runs cannot each spend the full drift budget.
        self.baseline_model = baseline_model
        self.overlay = CoWWeightOverlay(model)
        self.history: list[dict[str, Any]] = []
        self._baseline_invariants: dict[str, float] = {}
        missing = {g.suite_name for g in intent.target_goals} - set(self.target_suites)
        missing |= {g.suite_name for g in intent.invariants} - set(self.invariant_suites)
        if missing:
            raise ValueError(f"no evaluation suite for: {sorted(missing)}")

    # --- metrics --------------------------------------------------------------------------------------

    def _evaluate(self, model: nn.Module) -> tuple[dict[str, float], dict[str, float]]:
        target: dict[str, float] = {}
        for name, suite in self.target_suites.items():
            target.update({f"{name}.{k}": v for k, v in suite.evaluate(model).items()})
        return target, self._evaluate_invariants(model)

    def _evaluate_invariants(self, model: nn.Module) -> dict[str, float]:
        invariant: dict[str, float] = {}
        for name, suite in self.invariant_suites.items():
            invariant.update({f"{name}.{k}": v for k, v in suite.evaluate(model).items()})
        return invariant

    def target_score(self, target: Mapping[str, float]) -> float:
        """Weighted shortfall to the goals, saturating at 0 once every goal is met.

        Overshooting a satisfied goal earns nothing: the loop is goal-directed, so among candidates that meet
        the contract the tournament falls through to invariant degradation, parameter count and edit size
        (surgical minimality) instead of rewarding the most aggressive edit.
        """
        score = 0.0
        for goal in self.intent.target_goals:
            value = target[f"{goal.suite_name}.{goal.metric_name}"]
            if goal.operator == ">=":
                score += goal.weight * min(value - goal.threshold, 0.0)
            elif goal.operator == "<=":
                score += goal.weight * min(goal.threshold - value, 0.0)
            else:
                score -= goal.weight * abs(value - goal.threshold)
        return score

    def goals_met(self, target: Mapping[str, float]) -> bool:
        """True when every target goal is satisfied."""
        return all(
            g.is_satisfied(target[f"{g.suite_name}.{g.metric_name}"]) for g in self.intent.target_goals
        )

    def invariant_status(self, invariant: Mapping[str, float]) -> tuple[bool, float]:
        """``(all guards ok, worst drop / max_allowed_drop)`` relative to the loop's baseline.

        The ratio is negative when every guarded metric improved, so the tournament can prefer the candidate
        that restores baseline behaviour best. With no guards it is 0.
        """
        worst = float("-inf") if self.intent.invariants else 0.0
        ok = True
        for guard in self.intent.invariants:
            key = f"{guard.suite_name}.{guard.metric_name}"
            drop = metric_drop(guard.metric_name, self._baseline_invariants[key], invariant[key])
            worst = max(worst, drop / guard.max_allowed_drop)
            ok &= drop <= guard.max_allowed_drop
        return ok, worst

    # --- nullspace ------------------------------------------------------------------------------------

    def _lock_invariant_directions(self, tensor_names: list[str]) -> None:
        """Locks principal input directions of invariant prompts for tensors not yet locked.

        Directions come from ``nullspace_reference`` (a known-good model) when given. Capturing them on the
        defective model instead would lock the defect's own leakage onto invariant prompts and block the repair.
        """
        prompts = invariant_prompts(self.intent)
        todo = [n for n in tensor_names if self.nullspace.locked_rank(n) == 0]
        if not prompts or not todo:
            return
        source = self.nullspace_reference if self.nullspace_reference is not None else self.model
        for name, acts in capture_linear_inputs(source, self.tokenizer, prompts, todo).items():
            self.nullspace.register_protected_directions(name, acts, top_k=2)

    # --- candidates -----------------------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        index: int,
        candidate: CandidatePatch,
        diagnostics: dict[str, Any],
        params: Mapping[str, torch.Tensor],
    ) -> CandidateEvaluation:
        ev = CandidateEvaluation(index=index, candidate=candidate)
        try:
            raw = execute_weight_surgery_script(
                candidate.python_script, CloningWeightView(params), diagnostics
            )
        except SandboxError as exc:
            ev.error = str(exc)
            return ev
        if not raw:
            ev.error = "script returned no deltas"
            return ev
        constraints = self.intent.constraints
        total = sum(p.numel() for p in self.model.parameters())
        ev.violations = check_constraints(raw, params, total, constraints)
        if ev.violations:
            return ev
        deltas = raw
        if constraints.enforce_nullspace_projection:
            self._lock_invariant_directions(list(raw))
            deltas = {k: self.nullspace.project_into_nullspace(k, v) for k, v in raw.items()}
            for name, delta in deltas.items():  # projection can change the spectrum that was just approved
                if params[name].ndim == 2:
                    shift = spectral_norm_shift(params[name], delta)
                    if shift > constraints.max_spectral_norm_shift:
                        ev.violations.append(f"{name}: spectral shift {shift:.4f} after projection")
            if ev.violations:
                return ev
        ev.deltas = deltas
        ev.tensor_stats, ev.modified_params, ev.sparsity_ratio = summarize_patch(
            params, deltas, total, structural_deltas=raw
        )
        try:
            with self.overlay.apply_delta(deltas) as forked:
                ev.target_metrics, ev.invariant_metrics = self._evaluate(forked)
        except Exception as exc:  # one failing candidate must not abort the tournament
            ev.error = f"Evaluation error: {type(exc).__name__}: {exc}"
            return ev
        ev.target_score = self.target_score(ev.target_metrics)
        ev.invariants_ok, ev.invariant_degradation = self.invariant_status(ev.invariant_metrics)
        ev.goals_met = self.goals_met(ev.target_metrics)
        return ev

    @staticmethod
    def select_winner(evaluations: list[CandidateEvaluation], baseline_score: float) -> int | None:
        """Pareto tournament over feasible, improving candidates; returns the winner's index or None."""
        pool = [e for e in evaluations if e.feasible and e.target_score > baseline_score + 1e-9]

        def dominated(a: CandidateEvaluation) -> bool:
            for b in pool:
                if b is a:
                    continue
                no_worse = (
                    b.target_score >= a.target_score
                    and b.invariant_degradation <= a.invariant_degradation
                    and b.modified_params <= a.modified_params
                )
                better = (
                    b.target_score > a.target_score
                    or b.invariant_degradation < a.invariant_degradation
                    or b.modified_params < a.modified_params
                )
                if no_worse and better:
                    return True
            return False

        front = [e for e in pool if not dominated(e)]
        if not front:
            return None
        best = max(
            front, key=lambda e: (e.target_score, -e.invariant_degradation, -e.modified_params, -e.edit_size)
        )
        return best.index

    # --- main loop ------------------------------------------------------------------------------------

    def run(self) -> ConvergenceResult:
        """Executes up to ``max_iterations`` iterations; stops early once every goal is met."""
        fail_texts, pass_texts = contrastive_texts(self.intent)
        target, invariant = self._evaluate(self.model)
        baseline = self.baseline_model if self.baseline_model is not None else self.model
        self._baseline_invariants = (
            dict(invariant) if baseline is self.model else self._evaluate_invariants(baseline)
        )
        base_sha = state_fingerprint(baseline)
        topology = topology_summary(self.adapter, self.model)
        records: list[IterationRecord] = []
        prober = MechanisticProber(self.model, self.adapter, self.tokenizer)
        cancelled = False
        total_steps = max(1, self.max_iterations * self.candidates_per_iteration)

        for iteration in range(self.max_iterations):
            if self.goals_met(target):
                break
            if self.should_cancel is not None and self.should_cancel():
                cancelled = True
                break
            record = IterationRecord(iteration, dict(target), dict(invariant), [])
            records.append(record)
            diagnostics = build_diagnostics(prober.trace_contrastive(fail_texts, pass_texts, self.top_k_cols))
            context = ProposalContext(
                iteration=iteration,
                diagnostics=diagnostics,
                topology=topology,
                history=list(self.history),
                intent_description=f"{self.intent.description}\nTarget: {self.intent.target_behavior}\n"
                f"Failure: {self.intent.failure_behavior}",
                num_candidates=self.candidates_per_iteration,
            )
            try:
                candidates = self.source.propose(context)[: self.candidates_per_iteration]
            except (LLMBackendError, CandidateParseError) as exc:
                record.note = f"candidate source failed: {exc}"
                logger.warning(record.note)
                continue
            if not candidates:
                record.note = "candidate source returned no candidates"
                continue

            params = dict(self.model.named_parameters())
            baseline_score = self.target_score(target)
            record.evaluations = []
            for i, c in enumerate(candidates):
                if self.should_cancel is not None and self.should_cancel():
                    cancelled = True
                    break
                if self.progress is not None:
                    done = iteration * self.candidates_per_iteration + i
                    self.progress(
                        f"iteration {iteration + 1} of {self.max_iterations}: evaluating candidate {i + 1} of {len(candidates)}",
                        done / total_steps,
                    )
                ev = self._evaluate_candidate(i, c, diagnostics, params)
                touched = [n for n in ev.deltas if any(n == p or n.startswith(p + ".") or p in n for p in self.protected_tensors)]
                if touched:
                    ev.violations.append(f"touches protected tensor {touched[0]}")
                record.evaluations.append(ev)
            if cancelled:
                record.note = "cancelled by user"
                break
            for ev in record.evaluations:
                self.history.append(
                    {
                        "iteration": iteration,
                        "hypothesis": ev.candidate.hypothesis,
                        "python_script": ev.candidate.python_script,
                        "target_score": None
                        if ev.target_score == float("-inf")
                        else round(ev.target_score, 4),
                        "invariant_degradation": None
                        if ev.invariant_degradation == float("inf")
                        else round(ev.invariant_degradation, 4),
                        "outcome": ev.error
                        or (
                            "; ".join(ev.violations)
                            if ev.violations
                            else ("feasible" if ev.feasible else "invariant violation")
                        ),
                    }
                )
            record.winner = self.select_winner(record.evaluations, baseline_score)
            if record.winner is None:
                record.note = "no feasible improving candidate"
                continue

            win = record.evaluations[record.winner]
            if not self.commit_enabled:
                record.note = "dry run: best candidate shown, nothing applied or committed"
                break
            if self.approve is not None and not self.approve(win):
                record.note = "best candidate rejected by the user"
                record.winner = None
                continue
            if self.should_cancel is not None and self.should_cancel():
                cancelled = True
                record.note = "cancelled by user"
                record.winner = None
                break
            apply_deltas_inplace(self.model, win.deltas)
            if self.repo is not None:
                manifest = PatchManifest(
                    patch_id="pending",
                    parent_patch_id=None,
                    base_model_id=self.base_model_id,
                    base_model_sha256=base_sha,
                    intent_id=self.intent.intent_id,
                    iteration=iteration,
                    synthesized_code=win.candidate.python_script,
                    reasoner_rationale=win.candidate.hypothesis,
                    tensor_stats=win.tensor_stats,
                    total_modified_params=win.modified_params,
                    overall_sparsity_ratio=win.sparsity_ratio,
                    target_metrics_before=record.target_before,
                    target_metrics_after=win.target_metrics,
                    invariant_metrics_before=record.invariant_before,
                    invariant_metrics_after=win.invariant_metrics,
                    passed_gate=win.goals_met and win.invariants_ok,
                )
                record.committed_patch_id = self.repo.commit(win.deltas, manifest).patch_id
            target, invariant = win.target_metrics, win.invariant_metrics

        return ConvergenceResult(
            converged=self.goals_met(target),
            iterations=records,
            target_metrics=target,
            invariant_metrics=invariant,
            head=self.repo.head() if self.repo is not None else None,
            cancelled=cancelled,
        )
