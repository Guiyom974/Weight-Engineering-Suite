"""Mode 3 convergence loop tests (PRD_01 section 5.6, blueprint step 10).

Scenario: a micro-model with a planted 'x' repetition attractor and a known-good (pre-defect) reference for the
KL invariant. A scripted reasoner proposes an over-budget rewrite (A), an over-aggressive fix that maximises the
target but breaks the KL invariant (B), and the surgical three-column directional nullification (C). The
tournament must commit C, never A or B, and finish with invariant KL < 0.01.
"""

import json
from typing import Any

import pytest
import torch

from libweightspace.adapters import Qwen3Adapter
from libweightspace.candidates import CandidatePatch, TemplateCandidateSource, nullification_script
from libweightspace.llm_backends import CallableBackend
from libweightspace.toy_models import build_tiny_qwen3
from neuroforge.convergence_engine import CandidateEvaluation, ConvergenceEngine, check_constraints
from neuroforge.intent_compiler import compile_suites
from neuroforge.metrics import ProbeSuite
from neuroforge.reasoner_bridge import ReasonerBridge
from neuroforge.schemas import EditConstraints, IntentSpec
from neuroforge.weight_git import WeightGitRepository

from .conftest import NEUTRAL_PROMPTS

PROBE_PROMPTS = ["ab x", "cd x", "the x", "q: x"]

OVER_BUDGET = (
    "def compute_delta(weights, diagnostics):\n"
    "    name = diagnostics['target_tensor']\n"
    "    return {name: -weights[name]}\n"
)
OVER_AGGRESSIVE = """def compute_delta(weights, diagnostics):
    name = diagnostics["target_tensor"]
    W = weights[name]
    U = weights["lm_head.weight"]
    u = U[121] - U[120]
    u = u / (torch.norm(u) + 1e-8)
    delta = torch.zeros_like(W)
    for c in diagnostics["top_culpable_mlp_cols"][:3]:
        delta[:, c] = -W[:, c] + 0.1 * u
    return {name: delta}
"""


def _intent(threshold: float) -> IntentSpec:
    probes: list[dict[str, Any]] = [
        {
            "probe_id": f"loop{i}",
            "prompt": p,
            "expected_completion_prefix": "yzyz",
            "undesired_completion_prefix": "xxxx",
        }
        for i, p in enumerate(PROBE_PROMPTS)
    ]
    probes += [
        {
            "probe_id": f"inv{i}",
            "prompt": p,
            "expected_completion_prefix": " ",
            "metadata": {"role": "invariant"},
        }
        for i, p in enumerate(NEUTRAL_PROMPTS)
    ]
    return IntentSpec.model_validate(
        {
            "intent_id": "break-x-loop",
            "description": "Stop the model from looping on 'x'",
            "target_behavior": "after 'x' the model no longer prefers another 'x' over 'y'",
            "failure_behavior": "x x x x ...",
            "contrastive_probes": probes,
            "target_goals": [
                {
                    "suite_name": "loop",
                    "metric_name": "logit_margin",
                    "operator": ">=",
                    "threshold": threshold,
                }
            ],
            "invariants": [
                {"suite_name": "baseline", "metric_name": "kl_divergence", "max_allowed_drop": 0.01}
            ],
            "constraints": {
                "max_modified_param_ratio": 0.01,
                "max_frobenius_rel_norm": 0.15,
                "max_spectral_norm_shift": 0.05,
            },
        }
    )


CANDIDATES = {
    "A": ("over-budget rewrite", OVER_BUDGET),
    "B": ("overwrite culpable columns toward y", OVER_AGGRESSIVE),
    "C": ("nullify top three columns", nullification_script(3)),
}


class ScriptedReasoner:
    """Returns pre-written candidate JSON per call (one round per iteration), like a reasoner that iterates."""

    def __init__(self, rounds: list[str]):
        self.rounds = rounds
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        batch = [CANDIDATES[k] for k in self.rounds[min(self.calls, len(self.rounds) - 1)]]
        self.calls += 1
        body = {
            "candidates": [
                {"rationale": r, "failure_taxonomy": "attractor_loop", "python_script": s} for r, s in batch
            ]
        }
        return f"Here you go:\n```json\n{json.dumps(body)}\n```"


@pytest.fixture
def setup(planted_qwen, tokenizer, tmp_path):
    model, defect = planted_qwen
    clean = build_tiny_qwen3(seed=0)  # known-good reference (pre-defect checkpoint)
    loop_probes = _intent(0.0).contrastive_probes[:4]
    clean_margin = ProbeSuite("loop", loop_probes, tokenizer).evaluate(clean)["logit_margin"]
    intent = _intent(clean_margin - 0.5)  # goal: restore (approximately) the known-good preference
    targets, invariants = compile_suites(intent, tokenizer, reference_model=clean)
    repo = WeightGitRepository.init(tmp_path)
    return model, defect, clean, intent, targets, invariants, repo


def _engine(setup, tokenizer, reasoner, max_iterations=3):
    model, _defect, clean, intent, targets, invariants, repo = setup
    return ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        ReasonerBridge(CallableBackend(reasoner)),
        repo=repo,
        base_model_id="toy:qwen3_8+defect",
        max_iterations=max_iterations,
        candidates_per_iteration=3,
        nullspace_reference=clean,
    )


def test_tournament_selects_surgical_fix_and_keeps_kl(setup, tokenizer):
    model, defect, _clean, intent, _targets, _invariants, repo = setup
    reasoner = ScriptedReasoner(["ABC"])
    result = _engine(setup, tokenizer, reasoner).run()

    assert result.converged and len(result.iterations) == 1 and reasoner.calls == 1
    record = result.iterations[0]
    a, b, c = record.evaluations
    assert any("modified ratio" in v for v in a.violations) and any("Frobenius" in v for v in a.violations)
    assert not b.violations and b.error is None and b.invariants_ok is False  # passes budget, breaks KL
    assert b.target_metrics["loop.logit_margin"] > c.target_metrics["loop.logit_margin"]  # greedier on target
    assert record.winner == 2 and c.feasible and c.goals_met

    assert result.invariant_metrics["baseline.kl_divergence"] < 0.01
    assert result.target_metrics["loop.logit_margin"] >= intent.target_goals[0].threshold

    # Planted columns no longer write the attractor direction (nullspace projection leaves a small residue).
    w_after = dict(model.named_parameters())[defect.tensor_name].detach()
    for col in defect.columns:
        assert abs(float(torch.dot(w_after[:, col], defect.direction))) < 0.02

    (manifest,) = repo.log()
    assert manifest.patch_id == record.committed_patch_id and manifest.passed_gate
    assert manifest.reasoner_rationale == "nullify top three columns"
    assert manifest.total_modified_params == c.modified_params > 0
    assert (
        manifest.target_metrics_before["loop.logit_margin"]
        < manifest.target_metrics_after["loop.logit_margin"]
    )
    assert repo.verify(manifest.patch_id)


def test_loop_iterates_until_a_feasible_fix_appears(setup, tokenizer):
    reasoner = ScriptedReasoner(["AB", "BC"])
    result = _engine(setup, tokenizer, reasoner).run()
    assert result.converged and len(result.iterations) == 2
    assert (
        result.iterations[0].winner is None and result.iterations[0].note == "no feasible improving candidate"
    )
    assert result.iterations[1].winner == 1
    # Prompt protocol: topology, attribution table, history with outcomes, output contract.
    first, second = reasoner.prompts
    assert "### 1. Model Topology Summary" in first and "### 2. Mechanistic Attribution Table" in first
    assert "(first iteration)" in first
    assert "invariant violation" in second and "modified ratio" in second
    assert "### 4. Output Contract" in second


def test_template_reasoner_converges_offline(setup, tokenizer):
    model, _defect, clean, intent, targets, invariants, repo = setup
    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        TemplateCandidateSource(),
        repo=repo,
        max_iterations=3,
        nullspace_reference=clean,
    )
    result = engine.run()
    assert result.converged and result.invariant_metrics["baseline.kl_divergence"] < 0.01


def test_no_feasible_candidate_commits_nothing(setup, tokenizer):
    model, _defect, clean, intent, targets, invariants, repo = setup
    before = {k: v.clone() for k, v in model.state_dict().items()}
    backend = CallableBackend(lambda s, p: json.dumps([{"python_script": OVER_BUDGET}]))
    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        ReasonerBridge(backend),
        repo=repo,
        max_iterations=2,
    )
    result = engine.run()
    assert not result.converged and all(r.winner is None for r in result.iterations)
    assert repo.list_commits() == []
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_reasoner_failure_is_recorded_not_raised(setup, tokenizer):
    model, _defect, _clean, intent, targets, invariants, repo = setup
    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        ReasonerBridge(CallableBackend(lambda s, p: "I cannot help with that.")),
        max_iterations=1,
    )
    result = engine.run()
    assert "candidate source failed" in result.iterations[0].note


def test_missing_suite_rejected(setup, tokenizer):
    model, _defect, _clean, intent, targets, _invariants, _repo = setup
    with pytest.raises(ValueError, match="no evaluation suite"):
        ConvergenceEngine(
            model, Qwen3Adapter(model), tokenizer, intent, targets, {}, TemplateCandidateSource()
        )


def test_check_constraints_patterns(tiny_qwen):
    params = dict(tiny_qwen.named_parameters())
    total = sum(p.numel() for p in tiny_qwen.parameters())
    up = "model.layers.0.mlp.up_proj.weight"
    assert check_constraints({up: torch.zeros_like(params[up])}, params, total, EditConstraints()) == [
        f"{up}: not in allowed_module_patterns"
    ]


def test_pareto_selection_rules():
    def ev(i, score, deg, params, ok=True):
        return CandidateEvaluation(
            index=i,
            candidate=CandidatePatch("h", "s"),
            target_score=score,
            invariant_degradation=deg,
            modified_params=params,
            invariants_ok=ok,
        )

    evaluations = [
        ev(0, 1.0, 0.5, 10),
        ev(1, 2.0, 0.2, 10),
        ev(2, 5.0, 0.0, 10, ok=False),
        ev(3, 0.5, 0.0, 1),
    ]
    assert ConvergenceEngine.select_winner(evaluations, baseline_score=0.0) == 1
    assert ConvergenceEngine.select_winner(evaluations, baseline_score=3.0) is None
