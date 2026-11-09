"""One W-RSI session, shared by ``wrsi start`` and the Studio (CR10): setup, guardrails, hooks, timeline, rollback.

``run_session`` opens (or resumes) the run in ``<root>/.wrsi``, builds the scientist loop with the anchor vault as a
real child process, applies the guardrails, and runs curriculum steps until the generation limit, a pause request
(stops after the current step) or a stop request (stops at the next candidate, promoting nothing from that step).
State is saved after every step, undo data first, and each step is appended to ``.wrsi/timeline.jsonl`` with the
outcome and measured values of every candidate it tried.

Guardrails: the anchor KL limit (the tether to M_0), the sparsity budget, protected tensors, a forgetting check on
held-out prompts that blocks promotion when their expected answers become less likely, and the number of generations.

Main entry points: ``Guardrails``, ``SessionInputs``, ``run_session``, ``timeline``, ``rollback_preview``,
``rollback``, ``status``, ``DEMO_DOMAIN``, ``ANCHOR_TEXTS``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from libweightspace.candidates import CandidateSource
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.loading import LoadedModel, load_model
from neuroforge.metrics import ProbeSuite
from neuroforge.schemas import ProbePair

from .anchor_client import AnchorVaultClient
from .islands import IslandPopulation
from .meta_gym import MetaSurgeryGym
from .schemas import META_DOMAIN, GenerationState, WRSIConfig
from .scientist_agent import ForgettingCheck, GenerationReport, TaskDomain, WRSIRuntime, WRSIScientistLoop
from .watchdog import WeightWatchdog
from .workspace import WRSIWorkspace

ANCHOR_TEXTS = ["hello world", "abc def ghi", "the quick brown", "lorem ipsum do", "1 + 1 = 2", "red green blue"]
DEMO_DOMAIN: dict[str, Any] = {
    "name": "logical_deduction",
    "failure_taxonomy": "attractor_loop",
    "probes": [
        {"probe_id": f"loop{i}", "prompt": p, "expected_completion_prefix": "yzyz", "undesired_completion_prefix": "xxxx"}
        for i, p in enumerate(["ab x", "cd x", "the x", "q: x"])
    ],
}
WATCHDOG_PROBE = "the quick brown fox"
TOY_MAX_SPARSITY = 0.005  # a 27k-parameter model cannot be repaired within the 0.02% budget meant for LLMs
REAL_MAX_SPARSITY = 0.0002
ProposerFactory = Callable[[LoadedModel], tuple[CandidateSource, CandidateSource]]
Progress = Callable[[str, float], None]


@dataclass
class Guardrails:
    """Limits every promotion must respect (shown and configurable in the Studio)."""

    max_kl: float = 0.08
    max_sparsity: float | None = None  # None: 0.5% for toy models, 0.02% otherwise
    protected_tensors: tuple[str, ...] = ()
    forgetting: list[tuple[str, str]] = field(default_factory=list)  # (held-out prompt, expected answer)
    max_forgetting_drop: float = 0.05
    candidates_per_generation: int = 3
    meta_fraction: float = 0.30
    islands: int = 1
    seed: int = 0


@dataclass
class SessionInputs:
    """What a first start needs; a resumed run keeps its saved model and vault."""

    model: str = "toy:qwen3_8"
    demo_defect: bool = True
    domains: list[dict[str, Any]] | None = None
    anchors: list[str] | None = None


def build_domain(spec: dict[str, Any], tokenizer: Any) -> TaskDomain:
    """A task domain from ``{name, failure_taxonomy, probes: [ProbePair...]}``.

    Raises:
        ValueError: no probe with an undesired completion.
    """
    probes = [ProbePair.model_validate(p) for p in spec["probes"]]
    contrastive = [p for p in probes if p.undesired_completion_prefix]
    if not contrastive:
        raise ValueError(f"domain '{spec['name']}' needs probes with undesired_completion_prefix")
    suite = ProbeSuite(spec["name"], contrastive, tokenizer)
    return TaskDomain(
        name=spec["name"],
        failure_taxonomy=spec.get("failure_taxonomy", "attractor_loop"),
        fail_texts=[p.prompt + str(p.undesired_completion_prefix) for p in contrastive],
        pass_texts=[p.prompt + p.expected_completion_prefix for p in contrastive],
        evaluate=lambda m: suite.evaluate(m)["logit_margin"],
    )


def _trials_since(ws: WRSIWorkspace, started: float) -> list[dict[str, Any]]:
    rows = []
    for t in ws.ledger().list_trials(limit=200):
        if t.timestamp + 1e-6 < started:
            continue
        rows.append({
            "trial_id": t.trial_id, "promoted": t.promoted, "reason": t.rejection_reason, "gain": round(t.target_score_delta, 6),
            "kl": round(t.holdout_kl_divergence, 6), "sparsity": round(t.sparsity_ratio, 8),
            "tensors": t.target_tensors, "hypothesis": t.hypothesis[:200], "domain": t.domain,
        })
    return list(reversed(rows))


def _append_timeline(ws: WRSIWorkspace, entry: dict[str, Any]) -> None:
    ws.dir.mkdir(parents=True, exist_ok=True)
    with (ws.dir / "timeline.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def run_session(
    root: Path,
    inputs: SessionInputs,
    guardrails: Guardrails,
    proposer_factory: ProposerFactory,
    generations: int,
    progress: Progress | None = None,
    should_pause: Callable[[], bool] | None = None,
    should_stop: Callable[[], bool] | None = None,
    proposer_label: str = "",
) -> dict[str, Any]:
    """Run up to ``generations`` steps (see the module docstring); returns reports and why the run ended.

    Raises:
        ValueError / RuntimeError / AnchorVaultError: invalid inputs or a vault integrity failure (nothing promoted).
    """
    if not 1 <= generations <= 100_000:
        raise ValueError("generations must be between 1 and 100000")
    root.mkdir(parents=True, exist_ok=True)
    ws = WRSIWorkspace(root)
    is_new = not ws.exists()
    state = GenerationState(model=inputs.model, demo_defect=inputs.demo_defect and inputs.model.startswith("toy:")) \
        if is_new else ws.load_state()
    is_toy = state.model.startswith("toy:")
    budget = guardrails.max_sparsity if guardrails.max_sparsity is not None else (TOY_MAX_SPARSITY if is_toy else REAL_MAX_SPARSITY)
    config = WRSIConfig(max_sparsity=budget, max_kl=guardrails.max_kl, meta_fraction=guardrails.meta_fraction,
                        seed=guardrails.seed, candidates_per_generation=guardrails.candidates_per_generation)
    anchors = inputs.anchors or (ANCHOR_TEXTS if is_toy else None)
    if not anchors:
        raise ValueError("non-toy models need anchor prompts (held-out texts for the KL tether)")
    domain_specs = inputs.domains or ([DEMO_DOMAIN] if is_toy else None)
    if not domain_specs:
        raise ValueError("non-toy models need task domains (prompts with an undesired answer to improve)")

    loaded = load_model(state.model, demo_defect=state.demo_defect)
    domain_list = [build_domain(s, loaded.tokenizer) for s in domain_specs]
    vault, base = ws.ensure_vault(loaded.model, loaded.tokenizer, anchors, state=state)  # before the overlay: M_0
    ws.apply_overlay(loaded.model)
    ledger = ws.ledger()
    nullspace = ws.load_nullspace()
    loop = WRSIScientistLoop(loaded.model, CoWWeightOverlay(loaded.model), nullspace, ledger, config.max_sparsity,
                             config.nullspace_top_k)
    loop.generation = state.generation
    loop.protected_tensors = tuple(guardrails.protected_tensors)
    loop.should_stop = should_stop
    if guardrails.forgetting:
        loop.forgetting = ForgettingCheck(loaded.tokenizer, guardrails.forgetting, guardrails.max_forgetting_drop)
    source, meta_source = proposer_factory(loaded)

    def changed_tensors() -> list[str]:
        promoted = {name for g in loop.rollback_stack for name in g.tensors}
        return sorted(set(state.modified_tensors) | promoted | set(loop.candidate_tensors))

    saved_generations: set[int] = set()
    earlier_promotions = list(state.promoted_generations)

    def persist() -> None:
        """Saves after every step, undo data first, so a hard kill loses at most the running step."""
        for promoted in loop.rollback_stack:
            if promoted.generation not in saved_generations:
                ws.save_rollback(promoted)
                saved_generations.add(promoted.generation)
        state.modified_tensors = sorted(set(state.modified_tensors) | {n for g in loop.rollback_stack for n in g.tensors})
        state.promoted_generations = earlier_promotions + [g.generation for g in loop.rollback_stack]
        state.generation = loop.generation
        state.surgical_hit_rate = ledger.get_surgical_hit_rate()
        state.last_kl = next(
            (t.holdout_kl_divergence for t in ledger.list_trials(limit=100) if t.promoted and t.domain != META_DOMAIN),
            state.last_kl,
        )
        ws.save_overlay(loaded.model, state.modified_tensors)
        ws.save_nullspace(nullspace)
        ws.save_state(state)

    reports: list[GenerationReport] = []
    ended = "limit"
    with AnchorVaultClient(
        vault, base, "libweightspace.loading:model_from_spec", {"spec": state.model}, max_kl=config.max_kl,
        vault_sha256=state.vault_sha256, base_weights_sha256=state.base_weights_sha256,
    ) as anchor:
        probe = loaded.tokenizer(WATCHDOG_PROBE, return_tensors="pt")["input_ids"]
        runtime = WRSIRuntime(
            loop, loaded.adapter, loaded.tokenizer, domain_list, source, anchor.make_kl_fn(changed_tensors), config,
            gym=MetaSurgeryGym(), meta_proposer=meta_source,
            watchdog=WeightWatchdog(loaded.adapter, probe_input_ids=probe),
        )
        population = (
            IslandPopulation(loop, loaded.adapter, loaded.tokenizer, domain_list, source,
                             anchor.make_kl_fn(changed_tensors), num_islands=guardrails.islands)
            if guardrails.islands > 1 else None
        )
        first = state.steps_run
        for step in range(first, first + generations):
            if progress is not None:
                progress(f"step {step - first + 1} of {generations} (generation {loop.generation})",
                         (step - first) / generations)
            started = time.time()
            gen_before = loop.generation
            if population is not None:
                isl = population.islands[step % len(population.islands)]
                isl_promoted = population.step_island(isl, step)
                xover = population.crossover_tournament(step)
                report = GenerationReport(step=step, kind="island", domain=f"island-{isl.island_id}:{isl.domain.name}",
                                          candidates=population.candidates_per_island,
                                          promoted=xover.promoted or isl_promoted, generation=loop.generation,
                                          surgical_hit_rate=ledger.get_surgical_hit_rate(), note=xover.note)
            else:
                report = runtime.run(1, start_step=step)[0]
            reports.append(report)
            state.steps_run = step + 1
            persist()
            stopped = should_stop is not None and should_stop()
            trials = _trials_since(ws, started)
            promoted_kl = next((t["kl"] for t in trials if t["promoted"]), None)
            _append_timeline(ws, {
                "step": step, "kind": report.kind, "domain": report.domain, "candidates": report.candidates,
                "promoted": loop.generation > gen_before, "generation": loop.generation,
                "hit_rate": round(report.surgical_hit_rate, 4), "note": report.note or ("stopped" if stopped else ""),
                "anchor_kl": promoted_kl, "trials": trials, "proposer": proposer_label, "at": time.time(),
            })
            if stopped:
                ended = "stopped"
                break
            if should_pause is not None and should_pause():
                ended = "paused"
                break
    return {
        "ended": ended, "generation": state.generation, "steps_run": state.steps_run,
        "hit_rate": state.surgical_hit_rate, "reports": [r.__dict__ for r in reports],
    }


def timeline(root: Path, limit: int = 200) -> list[dict[str, Any]]:
    """The newest ``limit`` timeline entries, oldest first."""
    path = WRSIWorkspace(root).dir / "timeline.jsonl"
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]


def status(root: Path) -> dict[str, Any]:
    """Saved state, ledger totals and rollback points (``FileNotFoundError`` when no run exists)."""
    ws = WRSIWorkspace(root)
    if not ws.exists():
        raise FileNotFoundError(f"no W-RSI run in {root} (run `wrsi start`)")
    return {**ws.load_state().model_dump(), "ledger": ws.ledger().summary(), "rollback_points": len(ws.rollback_files())}


def rollback_preview(root: Path) -> dict[str, Any] | None:
    """What one rollback would change: the newest promoted generation, its trial, tensors and hypothesis."""
    ws = WRSIWorkspace(root)
    files = ws.rollback_files() if ws.exists() else []
    if not files:
        return None
    from safetensors import safe_open

    with safe_open(str(files[-1]), framework="pt") as f:
        meta = f.metadata()
    trial_id = meta.get("trial_id", "")
    tensors = json.loads(meta.get("wrsi", "{}")).get("backups", [])
    trial = next((t for t in ws.ledger().list_trials(limit=500) if t.trial_id == trial_id), None)
    return {
        "generation": int(files[-1].stem.split("_")[1]), "trial_id": trial_id, "tensors": tensors,
        "hypothesis": trial.hypothesis[:400] if trial else None,
        "gain": round(trial.target_score_delta, 6) if trial else None,
        "returns_to_generation": int(files[-1].stem.split("_")[1]),
    }


def rollback(root: Path, steps: int = 1) -> dict[str, Any]:
    """Undo ``steps`` promoted generations bit-exactly (weights and nullspace locks).

    Raises:
        FileNotFoundError: no run; ValueError: nothing to roll back.
    """
    ws = WRSIWorkspace(root)
    if not ws.exists():
        raise FileNotFoundError(f"no W-RSI run in {root} (run `wrsi start`)")
    state = ws.load_state()
    if not ws.rollback_files():
        raise ValueError("nothing to roll back")
    loaded = load_model(state.model, demo_defect=state.demo_defect)
    ws.apply_overlay(loaded.model)
    nullspace = ws.load_nullspace()
    undone: list[int] = []
    for _ in range(steps):
        generation = ws.pop_rollback(loaded.model, nullspace)
        if generation is None:
            break
        undone.append(generation)
    state.generation = undone[-1]
    state.promoted_generations = [g for g in state.promoted_generations if g not in undone]
    if not ws.rollback_files():
        state.modified_tensors = []  # back at M_0: no overlay needed
    ws.save_overlay(loaded.model, state.modified_tensors)
    ws.save_nullspace(nullspace)
    ws.save_state(state)
    ws.commit_rollback()
    # A rollback is not a curriculum step: it takes no step number, so the next generation's number stays unique.
    _append_timeline(ws, {"step": None, "kind": "rollback", "domain": "-", "candidates": 0, "promoted": False,
                          "generation": state.generation, "hit_rate": round(state.surgical_hit_rate, 4),
                          "note": f"rolled back generations {undone}", "anchor_kl": None, "trials": [], "at": time.time()})
    return {"undone": undone, "generation": state.generation}


def forgetting_pairs(lines: Sequence[str]) -> list[tuple[str, str]]:
    """``prompt => expected answer`` lines to pairs (the Studio and CLI format); blank lines are skipped.

    Raises:
        ValueError: a line without ``=>`` or with an empty side.
    """
    pairs = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if "=>" not in line:
            raise ValueError(f"forgetting check line {i}: write 'prompt => expected answer'")
        prompt, expected = line.split("=>", 1)
        if not prompt.strip() or not expected.strip():
            raise ValueError(f"forgetting check line {i}: both the prompt and the answer are needed")
        pairs.append((prompt.rstrip(), " " + expected.strip() if expected.startswith(" ") else expected.strip()))
    return pairs
