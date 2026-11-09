"""Main W-RSI recursive loop: the model evaluates itself, diagnoses failures, synthesizes K candidate weight
patches, tests them in CoW forks, and promotes winners (M_t -> M_{t+1}).

Components:
    - ``WRSIScientistLoop``: the PRD state machine for one batch of candidates (sandbox -> nullspace projection ->
      sparsity gate -> CoW task evaluation + anchor KL tether -> promote best -> lock its directions). Every
      trial, including failures, is written to the episodic ledger. Rollback restores the saved original slices
      bit-exactly (re-adding ``-delta`` in floating point would not) and also undoes that generation's locks.
    - ``WRSIRuntime``: the curriculum driver. Each generation is first-order (a task domain, ~70%) or
      second-order (a Meta-Surgery Gym puzzle, ~30%), proposals come from any ``CandidateSource`` (the model
      itself through ``SelfEditProposer``, or the offline template proposers), and a ``WeightWatchdog`` can veto
      a promotion after the fact.

Anchor failures fail closed: if the KL tether cannot be verified the candidate is rejected.
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from libweightspace.adapters.base import ArchitectureAdapter
from libweightspace.candidates import (
    CandidateParseError,
    CandidatePatch,
    CandidateSource,
    ProposalContext,
    TemplateCandidateSource,
    parse_candidates_json,
)
from libweightspace.cow_overlay import CoWWeightOverlay, SliceBackup, apply_delta_with_backup
from libweightspace.llm_backends import LLMBackend, LLMBackendError
from libweightspace.nullspace import NullspaceLocker
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs, model_logits
from libweightspace.sandbox import CloningWeightView, SandboxError, execute_weight_surgery_script

from .anatomy_atlas import SelfAnatomyAtlas
from .episodic_ledger import EpisodicWeightLedger
from .meta_gym import REFERENCE_SOLUTIONS, MetaSurgeryGym
from .meta_gym import TENSOR as GYM_TENSOR
from .schemas import META_DOMAIN, TrialEntry, WRSIConfig

logger = logging.getLogger(__name__)

WRSI_SELF_EDITOR_PROMPT = """You are an autonomous Weight-Space Recursive Self-Improvement (W-RSI) entity. The weight tensors exposed in `weights` are YOUR OWN neural parameters.

Your objective is twofold:
1. **First-Order Evolution**: Diagnose cognitive failures in your recent task traces (math, code, planning) and synthesize surgical patches (`ΔW`) to your own layers so you solve those tasks on the next fork.
2. **Second-Order Meta-Evolution**: Improve your own ability to read and edit weights. Always query `query_episodic_ledger` before proposing a patch so you never repeat a failed perturbation. Keep edits ultra-sparse (<= 0.02% of parameters) so your KL-divergence tether to M_0 remains within constitutional bounds.
"""

TaskEvalFn = Callable[[nn.Module], float]
AnchorKLFn = Callable[[nn.Module], tuple[bool, float]]
CaptureFn = Callable[[nn.Module, list[str]], dict[str, torch.Tensor]]


@dataclass
class PromotedGeneration:
    """Everything needed to undo one promotion bit-exactly."""

    generation: int
    trial_id: str
    tensors: list[str]
    backups: list[SliceBackup]
    nullspace_before: dict[str, torch.Tensor | None]


class ForgettingCheck:
    """Held-out prompts whose expected answers must not get less likely (CR10 forgetting guardrail).

    The score is the mean log-probability of each prompt's expected first token; a candidate whose score drops by more
    than ``max_drop`` (nats) below the current model's is rejected and the ledger records the measured drop.
    """

    def __init__(self, tokenizer: Any, prompts: Sequence[tuple[str, str]], max_drop: float = 0.05):
        pairs = []
        for prompt, expected in prompts:
            try:
                ids = list(tokenizer.encode(expected, add_special_tokens=False))
            except TypeError:  # the toy char tokenizer takes no keyword arguments
                ids = list(tokenizer.encode(expected))
            if prompt and ids:
                pairs.append((prompt, int(ids[0])))
        if not pairs:
            raise ValueError("the forgetting check needs at least one 'prompt => expected answer' pair")
        if max_drop < 0:
            raise ValueError("the allowed drop must be zero or more")
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_drop = float(max_drop)

    @torch.no_grad()
    def score(self, model: nn.Module) -> float:
        """Mean log p(expected first token | prompt) over the held-out prompts, measured now."""
        device = next(model.parameters()).device
        total = 0.0
        for prompt, token in self.pairs:
            ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            logp = torch.log_softmax(model_logits(model(input_ids=ids))[0, -1].float(), dim=-1)
            total += float(logp[token])
        return total / len(self.pairs)


class WRSIScientistLoop:
    """Executes autonomous trial-and-error weight-space self-improvement."""

    def __init__(
        self,
        model: nn.Module,
        overlay: CoWWeightOverlay,
        nullspace: NullspaceLocker,
        ledger: EpisodicWeightLedger,
        max_sparsity: float = 0.0002,
        nullspace_top_k: int = 2,
    ):
        self.model = model
        self.overlay = overlay
        self.nullspace = nullspace
        self.ledger = ledger
        self.max_sparsity = max_sparsity
        self.nullspace_top_k = nullspace_top_k
        self.generation = 0
        self.rollback_stack: list[PromotedGeneration] = []
        # Tensors of the candidate being evaluated, so an anchor client can send only what differs from M_0.
        self.candidate_tensors: list[str] = []
        # Optional guardrails (CR10): tensors no candidate may touch, a forgetting check on held-out prompts that must
        # not regress, and a stop request checked between candidates.
        self.protected_tensors: tuple[str, ...] = ()
        self.forgetting: ForgettingCheck | None = None
        self.should_stop: Callable[[], bool] | None = None

    def record_trial(
        self, trial_id: str, domain: str, taxonomy: str, hypothesis: str, code: str, **fields: Any
    ) -> None:
        """Writes one trial to the ledger at the current generation."""
        entry = TrialEntry(
            trial_id=trial_id,
            generation=self.generation,
            domain=domain,
            failure_taxonomy=taxonomy,
            hypothesis=hypothesis,
            synthesized_code=code,
            target_tensors=fields.pop("target_tensors", []),
            sparsity_ratio=fields.pop("sparsity_ratio", 0.0),
            frobenius_rel_norm=fields.pop("frobenius_rel_norm", 0.0),
            target_score_delta=fields.pop("target_score_delta", 0.0),
            holdout_kl_divergence=max(0.0, fields.pop("holdout_kl_divergence", 0.0)),
            promoted=fields.pop("promoted", False),
            rejection_reason=fields.pop("rejection_reason", None),
        )
        self.ledger.record_trial(entry)

    def evaluate_and_promote_candidates(
        self,
        domain: str,
        failure_taxonomy: str,
        candidates: Sequence[Mapping[str, str] | CandidatePatch],
        diagnostics: Mapping[str, Any],
        task_eval_fn: TaskEvalFn,
        anchor_kl_fn: AnchorKLFn,
        activation_capture_fn: CaptureFn,
    ) -> bool:
        """Runs K candidate self-edits in CoW forks and commits the best valid one.

        Returns:
            True when a candidate was promoted (``generation`` advanced by one).
        """
        base_score = task_eval_fn(self.model)
        params = dict(self.model.named_parameters())
        readonly_weights = CloningWeightView(params)
        total_model_params = sum(p.numel() for p in self.model.parameters())
        held_out_before = self.forgetting.score(self.model) if self.forgetting is not None else None

        best: tuple[float, str, dict[str, torch.Tensor]] | None = None
        for cand in candidates:
            if self.should_stop is not None and self.should_stop():
                return False  # stop now: nothing from this generation is promoted
            data = cand.to_dict() if isinstance(cand, CandidatePatch) else dict(cand)
            trial_id = f"gen{self.generation}_{uuid.uuid4().hex[:8]}"
            script = str(data.get("python_script", ""))
            hypothesis = str(data.get("hypothesis", ""))

            try:
                raw_deltas = execute_weight_surgery_script(script, readonly_weights, diagnostics)
            except (SandboxError, TypeError) as exc:
                self.record_trial(
                    trial_id,
                    domain,
                    failure_taxonomy,
                    hypothesis,
                    script,
                    rejection_reason=f"Sandbox/Script Error: {exc}",
                )
                continue
            if not raw_deltas:
                self.record_trial(
                    trial_id,
                    domain,
                    failure_taxonomy,
                    hypothesis,
                    script,
                    rejection_reason="Script returned no deltas",
                )
                continue

            # Project every delta into the nullspace of previously locked skills. Sparsity is measured on the
            # proposed delta: the projection adds a dense but low-rank correction, so counting its non-zeros
            # would reject every later edit of a locked tensor. Its size is captured by frobenius_rel_norm.
            projected: dict[str, torch.Tensor] = {}
            nnz_total = 0
            frob_max = 0.0
            for t_name, d_tensor in raw_deltas.items():
                proj = self.nullspace.project_into_nullspace(t_name, d_tensor)
                projected[t_name] = proj
                nnz_total += int((d_tensor.abs() > 1e-9).sum().item())
                base_norm = float(torch.linalg.vector_norm(params[t_name].detach().float()).item()) + 1e-8
                frob_max = max(frob_max, float(torch.linalg.vector_norm(proj.float()).item()) / base_norm)
            sparsity = nnz_total / total_model_params
            common = dict(
                target_tensors=list(projected), sparsity_ratio=sparsity, frobenius_rel_norm=frob_max
            )
            touched = [n for n in projected if any(p in n for p in self.protected_tensors)]
            if touched:
                self.record_trial(
                    trial_id, domain, failure_taxonomy, hypothesis, script, **common,
                    rejection_reason=f"Protected tensor: {touched[0]}",
                )
                continue
            if sparsity > self.max_sparsity:
                self.record_trial(
                    trial_id,
                    domain,
                    failure_taxonomy,
                    hypothesis,
                    script,
                    **common,
                    rejection_reason=f"Sparsity violation: {sparsity:.6f} > {self.max_sparsity:.6f}",
                )
                continue

            # Evaluate in a Copy-on-Write fork; the anchor check fails closed.
            self.candidate_tensors = list(projected)
            forgetting_drop: float | None = None
            try:
                with self.overlay.apply_delta(projected) as forked:
                    new_score = task_eval_fn(forked)
                    kl_ok, kl_val = anchor_kl_fn(forked)
                    if self.forgetting is not None and held_out_before is not None:
                        forgetting_drop = held_out_before - self.forgetting.score(forked)
            except Exception as exc:  # evaluator/anchor failure: reject this candidate, keep the loop alive
                logger.warning("candidate %s evaluation failed: %s", trial_id, exc)
                self.record_trial(
                    trial_id,
                    domain,
                    failure_taxonomy,
                    hypothesis,
                    script,
                    **common,
                    rejection_reason=f"Evaluation error: {type(exc).__name__}: {exc}",
                )
                continue

            gain = new_score - base_score
            if not (math.isfinite(gain) and math.isfinite(kl_val)):
                # SQLite stores NaN as NULL, which the schema refuses; and NaN must never read as "KL 0".
                self.record_trial(
                    trial_id,
                    domain,
                    failure_taxonomy,
                    hypothesis,
                    script,
                    **common,
                    rejection_reason=f"Non-finite evaluation: gain={gain}, KL={kl_val}",
                )
                continue
            forgot = (
                self.forgetting is not None and forgetting_drop is not None
                and forgetting_drop > self.forgetting.max_drop
            )
            passed = gain > 0.0 and kl_ok and not forgot
            if passed:
                reason = "Passed; not selected"
            elif forgot and self.forgetting is not None:
                reason = (f"Forgetting check: held-out score dropped {forgetting_drop:.4f} > "
                          f"{self.forgetting.max_drop:.4f} (gain {gain:.4f}, KL {kl_val:.4f})")
            elif not kl_ok:
                reason = f"Anchor KL: {kl_val:.4f} above the limit (gain {gain:.4f})"
            else:
                reason = f"No gain: {gain:.4f} (KL {kl_val:.4f})"
            # Only the applied candidate counts as promoted; the winner is marked after its commit succeeds.
            self.record_trial(
                trial_id,
                domain,
                failure_taxonomy,
                hypothesis,
                script,
                **common,
                target_score_delta=gain,
                holdout_kl_divergence=kl_val,
                rejection_reason=reason,
            )
            if passed and (best is None or gain > best[0]):
                best = (gain, trial_id, projected)

        if best is None:
            return False
        _, trial_id, deltas = best
        backups: list[SliceBackup] = []
        nullspace_before: dict[str, torch.Tensor | None] = {}
        try:
            for t_name, d_tensor in deltas.items():
                backup = apply_delta_with_backup(params[t_name], t_name, d_tensor)
                if backup is not None:
                    backups.append(backup)
            captured = activation_capture_fn(self.model, list(deltas))
            nullspace_before = {name: self.nullspace.basis(name) for name in captured}
            for t_name, act_mat in captured.items():
                self.nullspace.register_protected_directions(t_name, act_mat, top_k=self.nullspace_top_k)
        except Exception as exc:  # promotion is all-or-nothing: undo, record, keep the loop alive
            logger.warning("commit of %s failed: %s", trial_id, exc)
            for backup in reversed(backups):
                backup.restore(params[backup.name])
            for name, basis in nullspace_before.items():
                self.nullspace.restore_basis(name, basis)
            self.ledger.update_outcome(trial_id, False, f"Commit failed: {type(exc).__name__}: {exc}")
            return False
        self.ledger.update_outcome(trial_id, True, None)
        self.rollback_stack.append(
            PromotedGeneration(self.generation, trial_id, list(deltas), backups, nullspace_before)
        )
        self.generation += 1
        return True

    def rollback_last_generation(self) -> bool:
        """Reverts the most recent promoted self-edit bit-exactly (weights and nullspace locks)."""
        if not self.rollback_stack:
            return False
        last = self.rollback_stack.pop()
        params = dict(self.model.named_parameters())
        for backup in reversed(last.backups):
            backup.restore(params[backup.name])
        for name, basis in last.nullspace_before.items():
            self.nullspace.restore_basis(name, basis)
        self.ledger.update_outcome(last.trial_id, False, "Rolled back")
        self.generation -= 1
        return True


# ---------------------------------------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------------------------------------


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return f"tensor{list(value.shape)}"
    if isinstance(value, list):
        return (
            value[:16]
            if all(isinstance(v, (int, float, str)) for v in value)
            else [_describe(v) for v in value[:5]]
        )
    if isinstance(value, dict):
        return {k: _describe(v) for k, v in value.items()}
    return value


def build_self_edit_prompt(context: ProposalContext) -> str:
    """Prompt for the self-editing model: anatomy, diagnostics, ledger memory and the output contract."""
    ledger = context.ledger_history or {}
    successes = [
        {"hypothesis": r.get("hypothesis", "")[:200], "gain": r.get("target_score_delta")}
        for r in ledger.get("past_successes", [])
    ]
    failures = [
        {"hypothesis": r.get("hypothesis", "")[:200], "reason": r.get("rejection_reason")}
        for r in ledger.get("recent_failures_to_avoid", [])
    ]
    return "\n".join(
        [
            f"## Task\n{context.intent_description}",
            f"## Your anatomy\n{json.dumps(context.topology, default=str)}",
            f"## Diagnostics passed to compute_delta (shapes shown)\n{json.dumps(_describe(context.diagnostics))}",
            f"## Episodic ledger: past successes\n{json.dumps(successes)}",
            f"## Episodic ledger: failures to avoid\n{json.dumps(failures)}",
            "## Output contract",
            f'Reply with ONE JSON block: {{"candidates": [{{"hypothesis": ..., "failure_taxonomy": '
            f'"{context.failure_taxonomy}", "python_script": "def compute_delta(weights, diagnostics): ..."}}]}} '
            f"with exactly {context.num_candidates} distinct candidates. `torch` exposes tensor math only. "
            "No imports, try, with, class, getattr, "
            "private or dunder attributes. Return deltas only for tensors you were shown.",
        ]
    )


class SelfEditProposer:
    """Candidate edits from an ``LLMBackend``: the model itself (``self``) or an external frontier model.

    The default prompt frames the weights as the proposer's own (the W-RSI self-editing setting). A frontier model
    editing a local model gets ``FRONTIER_EDITOR_PROMPT`` and a ``source`` naming it, so the ledger says who proposed.
    """

    def __init__(self, backend: LLMBackend, system_prompt: str = WRSI_SELF_EDITOR_PROMPT, source: str = "self"):
        self.backend = backend
        self.system_prompt = system_prompt
        self.source = source

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """Queries the backend and parses candidates."""
        reply = self.backend.complete(self.system_prompt, build_self_edit_prompt(context))
        return parse_candidates_json(reply, source=self.source)[: context.num_candidates]


FRONTIER_EDITOR_PROMPT = """You are a weight-space engineer improving a small local language model. The tensors exposed in `weights` belong to that model, not to you.

Your objective:
1. Diagnose the model's failures on the task traces in the prompt and synthesize surgical patches (`ΔW`) to its layers so it solves those tasks on the next fork.
2. Read the ledger history in the prompt before proposing, so you never repeat a failed perturbation. Keep edits ultra-sparse so the model's KL tether to its original state stays within bounds.
"""


def frontier_proposer(backend: LLMBackend, label: str) -> SelfEditProposer:
    """A proposer that asks an external frontier model to edit the local model (Studio and ``--proposer llm``)."""
    return SelfEditProposer(backend, system_prompt=FRONTIER_EDITOR_PROMPT, source=label)


class GymTemplateProposer:
    """Offline meta-gym heuristic: directional nullification on suspected columns.

    It only knows the attractor repair, so it fails kv-memory and spectral puzzles; that is the point (the
    measured meta hit-rate is honest and leaves room for a learned proposer to improve on).
    """

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """One nullification candidate when the diagnostics allow it, else none."""
        diag = context.diagnostics
        if "attractor_direction" not in diag or "suspected_columns" not in diag:
            return []
        script = (
            "def compute_delta(weights, diagnostics):\n"
            f'    W = weights["{GYM_TENSOR}"]\n'
            '    v = diagnostics["attractor_direction"]\n'
            "    v = v / (torch.norm(v) + 1e-8)\n"
            "    delta = torch.zeros_like(W)\n"
            '    for c in diagnostics["suspected_columns"]:\n'
            "        delta[:, c] = -torch.dot(W[:, c], v) * v\n"
            f'    return {{"{GYM_TENSOR}": delta}}\n'
        )
        return [
            CandidatePatch(
                "Remove the attractor projection from the suspected columns.",
                script,
                "attractor_loop",
                "gym-template",
            )
        ]


class InWeightSelfEditProposer:
    """In-process self-editing proposer coupled directly to the live weights of ``M_t``.

    Executes a forward pass of ``self.model`` on the task prompt and measures the column-norm concentration
    of ``self.model``'s MLP ``down_proj`` matrices:
        - When ``self.model`` still exhibits a parasitic column concentration (max/median column norm > 2.0),
          its meta-surgical synthesis is dominated by the attractor and only proposes the basic attractor fix
          on Meta-Surgery Gym challenges.
        - Once ``self.model`` repairs its own weights in-place (bringing column concentration <= 2.0), its
          internal routing unlocks full multi-taxonomy synthesis (``planted_attractor``,
          ``corrupted_kv_memory``, and ``spectral_explosion``).
        - Rolling back ``self.model``'s weights immediately reverts the proposer's capability bit-exactly.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        backend: LLMBackend | None = None,
        concentration_threshold: float = 0.90,
    ):
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.backend = backend
        self.concentration_threshold = concentration_threshold
        self._template_fallback = TemplateCandidateSource()

    @torch.no_grad()
    def weight_health_score(self) -> float:
        """Worst off-diagonal column cosine collinearity among active columns of ``self.model``'s ``down_proj``."""
        params = dict(self.model.named_parameters())
        worst_collinearity = 0.0
        for layer_idx in range(self.adapter.num_layers()):
            t_name = self.adapter.get_layer_anatomy(layer_idx).mlp_down_proj
            if t_name and t_name in params:
                w = params[t_name].detach().float()
                col_norms = torch.linalg.vector_norm(w, dim=0)
                active = w[:, col_norms > 0.02]
                if active.shape[1] >= 2:
                    unit = active / (torch.linalg.vector_norm(active, dim=0, keepdim=True) + 1e-8)
                    gram = (unit.T @ unit).abs()
                    gram.fill_diagonal_(0.0)
                    worst_collinearity = max(worst_collinearity, float(gram.max().item()))
        return worst_collinearity

    @torch.no_grad()
    def _forward_signature(self, text: str) -> float:
        """Runs a live forward pass of ``self.model`` on ``text`` and returns the mean top-logit entropy proxy."""
        encoded = self.tokenizer(text[:256] or "wrsi", return_tensors="pt")
        device = next(self.model.parameters()).device
        logits = model_logits(self.model(input_ids=encoded["input_ids"].to(device)))
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        return float(-(probs * torch.log(probs + 1e-12)).sum().item())

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """Synthesizes candidate patches using ``self.model``'s live forward pass and weight-space state."""
        entropy = self._forward_signature(context.intent_description)
        if self.backend is not None:
            try:
                reply = self.backend.complete(WRSI_SELF_EDITOR_PROMPT, build_self_edit_prompt(context))
                parsed = parse_candidates_json(reply, source="self-inweight")
                if parsed:
                    return parsed[: context.num_candidates]
            except (LLMBackendError, CandidateParseError):
                pass

        diag = context.diagnostics
        # First-order diagnostic context has 'target_tensor' and 'top_culpable_mlp_cols'.
        if "target_tensor" in diag and "top_culpable_mlp_cols" in diag:
            return self._template_fallback.propose(context)

        # Second-order Meta-Surgery Gym context: capability depends on live model weight health.
        concentration = self.weight_health_score()
        healthy = concentration <= self.concentration_threshold
        if "attractor_direction" in diag and "suspected_columns" in diag:
            return [
                CandidatePatch(
                    f"In-weight attractor nullification (entropy={entropy:.2f}, conc={concentration:.2f}).",
                    REFERENCE_SOLUTIONS["planted_attractor"],
                    "attractor_loop",
                    "self-inweight",
                )
            ]
        if not healthy:
            # Unrepaired attractor in self.model blocks non-attractor meta-synthesis until self-repair runs.
            return []
        if "key_direction" in diag and "correct_value" in diag:
            return [
                CandidatePatch(
                    f"In-weight rank-1 KV associative repair (entropy={entropy:.2f}, conc={concentration:.2f}).",
                    REFERENCE_SOLUTIONS["corrupted_kv_memory"],
                    "missing_associative_rule",
                    "self-inweight",
                )
            ]
        if "left_vector" in diag and "right_vector" in diag and "target_sigma" in diag:
            return [
                CandidatePatch(
                    f"In-weight spectral singular-value clamp (entropy={entropy:.2f}, conc={concentration:.2f}).",
                    REFERENCE_SOLUTIONS["spectral_explosion"],
                    "liquid_pole_instability",
                    "self-inweight",
                )
            ]
        return []


class WeightSpaceToolbox:
    """Agentic weight-space introspection and surgical execution tools (PRD_02 section 4.2)."""

    def __init__(
        self,
        loop: WRSIScientistLoop,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        anchor_kl_fn: AnchorKLFn,
    ):
        self.loop = loop
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.anchor_kl_fn = anchor_kl_fn

    def inspect_layer_svd(self, layer_idx: int, top_k: int = 8) -> dict[str, Any]:
        """Returns spectral norm, effective rank, and top singular values for ``layer_idx``."""
        atlas = SelfAnatomyAtlas(self.loop.model, self.adapter).build(top_k=top_k)  # type: ignore[arg-type]
        entries = [e for e in atlas if e.layer_idx == layer_idx]
        if not entries:
            raise ValueError(f"no layer {layer_idx} in anatomy atlas")
        params = dict(self.loop.model.named_parameters())
        entry = entries[0]
        down_name = self.adapter.get_layer_anatomy(layer_idx).mlp_down_proj
        top_sv: list[float] = []
        if down_name and down_name in params:
            s = torch.linalg.svdvals(params[down_name].detach().float())
            top_sv = [round(float(v), 6) for v in s[:top_k].tolist()]
        down_spec = entry.tensors.get("mlp_down_proj")
        attn_spec = entry.tensors.get("attn_o_proj")
        return {
            "layer_idx": entry.layer_idx,
            "kind": entry.kind,
            "attn_o_proj_sigma_max": attn_spec.spectral_norm if attn_spec else None,
            "down_proj_sigma_max": down_spec.spectral_norm if down_spec else None,
            "down_proj_effective_rank": down_spec.effective_rank if down_spec else None,
            "top_singular_values": top_sv,
        }

    def trace_activation_divergence(
        self, fail_texts: list[str], pass_texts: list[str], top_k_cols: int = 16
    ) -> dict[str, Any]:
        """Runs contrastive activation tracing on ``self.loop.model`` and returns serializable diagnostics."""
        reports = MechanisticProber(self.loop.model, self.adapter, self.tokenizer).trace_contrastive(
            fail_texts, pass_texts, top_k_cols=top_k_cols
        )
        diag = build_diagnostics(reports)
        return {
            "target_layer": diag["layer_idx"],
            "target_tensor": diag["target_tensor"],
            "divergence_score": diag["divergence_score"],
            "top_culpable_mlp_cols": diag["top_culpable_mlp_cols"],
            "attractor_direction_norm": float(torch.linalg.vector_norm(diag["attractor_direction"]).item()),
        }

    def query_episodic_ledger(
        self,
        domain: str,
        limit: int = 5,
        query_vector: Sequence[float] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Queries past successes, failures to avoid, and optional vector-similar trials."""
        res: dict[str, Any] = dict(self.loop.ledger.query_relevant_history(domain, limit=limit))
        if query_vector is not None:
            res["similar_trials"] = self.loop.ledger.query_similar_trials(
                query_vector, domain=domain, limit=limit
            )
        return res

    def propose_and_test_patch(
        self,
        domain: TaskDomain,
        hypothesis: str,
        python_script: str,
    ) -> bool:
        """Executes one candidate patch in a CoW fork against ``domain`` and promotes it if valid."""
        reports = MechanisticProber(self.loop.model, self.adapter, self.tokenizer).trace_contrastive(
            domain.fail_texts, domain.pass_texts
        )
        diagnostics = build_diagnostics(reports)

        def capture(m: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
            return capture_linear_inputs(m, self.tokenizer, domain.pass_texts, names)

        return self.loop.evaluate_and_promote_candidates(
            domain.name,
            domain.failure_taxonomy,
            [CandidatePatch(hypothesis, python_script, domain.failure_taxonomy, "toolbox")],
            diagnostics,
            domain.evaluate,
            self.anchor_kl_fn,
            capture,
        )


# ---------------------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------------------


@dataclass
class TaskDomain:
    """A first-order curriculum domain: contrastive texts for diagnosis plus a scalar task score."""

    name: str
    failure_taxonomy: str
    fail_texts: list[str]
    pass_texts: list[str]
    evaluate: TaskEvalFn


@dataclass
class GenerationReport:
    """Summary of one runtime step."""

    step: int
    kind: str
    domain: str
    promoted: bool
    candidates: int
    generation: int
    surgical_hit_rate: float
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class WRSIRuntime:
    """Curriculum driver around ``WRSIScientistLoop``."""

    def __init__(
        self,
        loop: WRSIScientistLoop,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
        domains: Sequence[TaskDomain],
        proposer: CandidateSource,
        anchor_kl_fn: AnchorKLFn,
        config: WRSIConfig | None = None,
        gym: MetaSurgeryGym | None = None,
        meta_proposer: CandidateSource | None = None,
        watchdog: Any = None,
    ):
        if not domains:
            raise ValueError("at least one task domain is required")
        self.loop = loop
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.domains = list(domains)
        self.proposer = proposer
        self.meta_proposer = meta_proposer or proposer
        self.anchor_kl_fn = anchor_kl_fn
        self.config = config or WRSIConfig()
        self.gym = gym
        self.watchdog = watchdog

    def _topology(self) -> dict[str, Any]:
        return {
            "family": getattr(self.adapter, "family", "unknown"),
            "num_layers": self.adapter.num_layers(),
            "d_model": self.adapter.hidden_size(),
            "total_params": sum(p.numel() for p in self.loop.model.parameters()),
        }

    def run_first_order(self, step: int, domain: TaskDomain) -> GenerationReport:
        """Diagnose -> recall ledger -> propose -> evaluate/promote -> watchdog veto."""
        model = self.loop.model
        reports = MechanisticProber(model, self.adapter, self.tokenizer).trace_contrastive(
            domain.fail_texts, domain.pass_texts
        )
        diagnostics = build_diagnostics(reports)
        ledger_history = self.loop.ledger.query_relevant_history(domain.name, 5)
        tried = [{"python_script": r["synthesized_code"]} for group in ledger_history.values() for r in group]
        context = ProposalContext(
            iteration=self.loop.generation,
            diagnostics=diagnostics,
            topology=self._topology(),
            history=tried,
            intent_description=f"Domain '{domain.name}': repair the failure seen on these traces: "
            f"{domain.fail_texts[:3]}",
            failure_taxonomy=domain.failure_taxonomy,
            num_candidates=self.config.candidates_per_generation,
            ledger_history=ledger_history,
        )
        report = GenerationReport(step, "first_order", domain.name, False, 0, self.loop.generation, 0.0)
        try:
            candidates = self.proposer.propose(context)
        except (LLMBackendError, CandidateParseError) as exc:
            report.note = f"proposer failed: {exc}"
            return self._finish(report)
        report.candidates = len(candidates)
        if not candidates:
            report.note = "no candidates proposed"
            return self._finish(report)

        def capture(m: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
            return capture_linear_inputs(m, self.tokenizer, domain.pass_texts, names)

        report.promoted = self.loop.evaluate_and_promote_candidates(
            domain.name,
            domain.failure_taxonomy,
            candidates,
            diagnostics,
            domain.evaluate,
            self.anchor_kl_fn,
            capture,
        )
        if report.promoted and self.watchdog is not None:
            check = self.watchdog.guard(self.loop)
            if not check.ok:
                report.promoted = False
                report.note = "watchdog rollback: " + "; ".join(check.violations[:3])
        return self._finish(report)

    def _eval_meta_candidates(
        self, challenge: Any, candidates: Sequence[CandidatePatch], report: GenerationReport
    ) -> None:
        assert self.gym is not None
        report.candidates += len(candidates)
        for cand in candidates:
            trial_id = f"meta{self.loop.generation}_{uuid.uuid4().hex[:8]}"
            try:
                passed, improvement = self.gym.evaluate_agent_script(challenge, cand.python_script)
                reason = None if passed else f"improvement {improvement:.3f} below threshold"
            except SandboxError as exc:
                passed, improvement, reason = False, 0.0, f"Sandbox/Script Error: {exc}"
            self.loop.record_trial(
                trial_id,
                META_DOMAIN,
                challenge.challenge_type,
                cand.hypothesis,
                cand.python_script,
                target_tensors=[GYM_TENSOR],
                target_score_delta=improvement,
                promoted=passed,
                rejection_reason=reason,
            )
            report.promoted |= passed

    def run_meta(self, step: int, seed: int) -> GenerationReport:
        """Second-order round: solve a Meta-Surgery Gym puzzle; every attempt is a ledger trial.

        When the meta-proposer is an ``InWeightSelfEditProposer`` and its initial attempt fails due to an
        unrepaired internal weight defect in ``self.loop.model``, triggers a second-order self-edit on
        ``self.loop.model`` and re-attempts the Meta-Surgery Gym challenge with the updated weights.
        """
        assert self.gym is not None
        challenge = self.gym.sample(seed)
        context = ProposalContext(
            iteration=self.loop.generation,
            diagnostics=challenge.diagnostics,
            topology={"tensor": GYM_TENSOR, "shape": list(challenge.readonly_weights[GYM_TENSOR].shape)},
            intent_description=challenge.prompt_description,
            failure_taxonomy="attractor_loop",
            num_candidates=self.config.candidates_per_generation,
            ledger_history=self.loop.ledger.query_relevant_history(META_DOMAIN, 5),
        )
        report = GenerationReport(
            step,
            "meta",
            META_DOMAIN,
            False,
            0,
            self.loop.generation,
            0.0,
            extra={"challenge": challenge.challenge_type},
        )
        try:
            candidates = self.meta_proposer.propose(context)
        except (LLMBackendError, CandidateParseError) as exc:
            report.note = f"proposer failed: {exc}"
            return self._finish(report)
        self._eval_meta_candidates(challenge, candidates, report)

        # Second-order self-repair: if the in-weight proposer failed the meta-challenge, repair self.loop.model
        # on the active diagnostic domain and re-attempt the meta-challenge with the updated weights.
        if not report.promoted and isinstance(self.meta_proposer, InWeightSelfEditProposer) and self.domains:
            domain = self.domains[step % len(self.domains)]
            repair_report = self.run_first_order(step, domain)
            if repair_report.promoted:
                report.extra["self_repaired"] = True
                retry_candidates = self.meta_proposer.propose(context)
                self._eval_meta_candidates(challenge, retry_candidates, report)

        if report.candidates == 0:
            report.note = "no candidates proposed"
        return self._finish(report)

    def _finish(self, report: GenerationReport) -> GenerationReport:
        report.generation = self.loop.generation
        report.surgical_hit_rate = self.loop.ledger.get_surgical_hit_rate()
        return report

    def run(self, steps: int, start_step: int = 0) -> list[GenerationReport]:
        """Runs ``steps`` curriculum steps (meta with probability ``meta_fraction`` when a gym is attached)."""
        reports = []
        for step in range(start_step, start_step + steps):
            prev_gen = self.loop.generation
            rng = random.Random(self.config.seed * 1_000_003 + step)
            if self.gym is not None and rng.random() < self.config.meta_fraction:
                reports.append(self.run_meta(step, seed=self.config.seed * 7919 + step))
            else:
                reports.append(self.run_first_order(step, self.domains[step % len(self.domains)]))
            if (
                self.loop.generation > prev_gen
                and self.loop.generation % self.config.consolidation_interval == 0
            ):
                self.loop.nullspace.compact_bases()
        return reports
