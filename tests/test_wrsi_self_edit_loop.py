"""W-RSI scientist loop tests (PRD_02 section 4.5, blueprint step 11)."""

import pytest
import torch

from libweightspace.adapters import Qwen3Adapter
from libweightspace.candidates import CandidatePatch, TemplateCandidateSource, nullification_script
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.nullspace import NullspaceLocker
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from neuroforge.metrics import KLDivergenceSuite, ProbeSuite
from neuroforge.schemas import ProbePair
from wrsi.episodic_ledger import EpisodicWeightLedger
from wrsi.meta_gym import MetaSurgeryGym
from wrsi.schemas import META_DOMAIN, WRSIConfig
from wrsi.scientist_agent import (
    WRSI_SELF_EDITOR_PROMPT,
    GymTemplateProposer,
    InWeightSelfEditProposer,
    TaskDomain,
    WeightSpaceToolbox,
    WRSIRuntime,
    WRSIScientistLoop,
    build_self_edit_prompt,
)
from wrsi.watchdog import WeightWatchdog

from .conftest import FAIL_PROMPTS, NEUTRAL_PROMPTS, PASS_PROMPTS

SYNTAX_ERROR = "def compute_delta(weights, diagnostics)\n    return {}"
DENSE = (
    "def compute_delta(weights, diagnostics):\n"
    "    name = diagnostics['target_tensor']\n"
    "    return {name: torch.ones_like(weights[name]) * 0.001}\n"
)
PROBES = [
    ProbePair(
        probe_id=f"p{i}", prompt=p, expected_completion_prefix="yzyz", undesired_completion_prefix="xxxx"
    )
    for i, p in enumerate(["ab x", "cd x", "the x", "q: x"])
]


@pytest.fixture
def harness(planted_qwen, tokenizer, tmp_path):
    model, defect = planted_qwen
    adapter = Qwen3Adapter(model)
    ledger = EpisodicWeightLedger(tmp_path / "ledger.db")
    loop = WRSIScientistLoop(model, CoWWeightOverlay(model), NullspaceLocker(), ledger, max_sparsity=0.005)
    suite = ProbeSuite("task", PROBES, tokenizer)
    kl = KLDivergenceSuite("anchor", NEUTRAL_PROMPTS, tokenizer, model)  # tether to M_0 (snapshot now)

    def task_eval(m):
        return suite.evaluate(m)["logit_margin"]

    def anchor_kl(m):
        value = kl.evaluate(m)["kl_divergence"]
        return value <= 0.08, value

    def capture(m, names):
        return capture_linear_inputs(m, tokenizer, PASS_PROMPTS, names)

    diagnostics = build_diagnostics(
        MechanisticProber(model, adapter, tokenizer).trace_contrastive(FAIL_PROMPTS, PASS_PROMPTS)
    )
    return model, defect, adapter, ledger, loop, task_eval, anchor_kl, capture, diagnostics


def test_only_valid_fix_promoted_and_rollback_is_bit_exact(harness):
    model, defect, _adapter, ledger, loop, task_eval, anchor_kl, capture, diagnostics = harness
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    candidates = [
        {"hypothesis": "typo", "python_script": SYNTAX_ERROR},
        {"hypothesis": "dense nudge", "python_script": DENSE},
        {"hypothesis": "nullify planted attractor columns", "python_script": nullification_script(3)},
    ]
    promoted = loop.evaluate_and_promote_candidates(
        "logical_deduction", "attractor_loop", candidates, diagnostics, task_eval, anchor_kl, capture
    )
    assert promoted is True
    assert loop.generation == 1
    assert loop.nullspace.locked_rank(defect.tensor_name) == 2
    assert task_eval(model) > -3.0  # loop broken in the live model

    trials = {t.hypothesis: t for t in ledger.list_trials()}
    assert len(trials) == 3
    assert "Sandbox/Script Error" in trials["typo"].rejection_reason and not trials["typo"].promoted
    assert (
        "Sparsity violation" in trials["dense nudge"].rejection_reason and not trials["dense nudge"].promoted
    )
    fix = trials["nullify planted attractor columns"]
    assert fix.promoted and fix.target_score_delta > 0 and fix.target_tensors == [defect.tensor_name]
    assert ledger.get_surgical_hit_rate() == pytest.approx(1 / 3)

    assert loop.rollback_last_generation() is True
    assert loop.generation == 0 and loop.nullspace.locked_rank(defect.tensor_name) == 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name]), name  # bit-for-bit
    assert loop.rollback_last_generation() is False


def test_anchor_failure_fails_closed(harness):
    model, _defect, _a, ledger, loop, task_eval, _kl, capture, diagnostics = harness
    before = {k: v.clone() for k, v in model.state_dict().items()}

    def broken_anchor(_m):
        raise RuntimeError("anchor vault unreachable")

    promoted = loop.evaluate_and_promote_candidates(
        "logical_deduction",
        "attractor_loop",
        [CandidatePatch("fix", nullification_script(3))],
        diagnostics,
        task_eval,
        broken_anchor,
        capture,
    )
    assert promoted is False and loop.generation == 0
    assert "Evaluation error" in ledger.list_trials()[0].rejection_reason
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_kl_violation_blocks_promotion(harness):
    _model, _defect, _a, ledger, loop, task_eval, _kl, capture, diagnostics = harness
    promoted = loop.evaluate_and_promote_candidates(
        "logical_deduction",
        "attractor_loop",
        [{"hypothesis": "h", "python_script": nullification_script(3)}],
        diagnostics,
        task_eval,
        lambda m: (False, 0.5),
        capture,
    )
    assert promoted is False and "Anchor KL: 0.5000 above the limit" in str(ledger.list_trials()[0].rejection_reason)


def test_later_edits_are_projected_away_from_locked_skill(harness):
    model, defect, _a, _ledger, loop, task_eval, anchor_kl, capture, diagnostics = harness
    loop.evaluate_and_promote_candidates(
        "d",
        "attractor_loop",
        [{"hypothesis": "fix", "python_script": nullification_script(3)}],
        diagnostics,
        task_eval,
        anchor_kl,
        capture,
    )
    basis = loop.nullspace.basis(defect.tensor_name)
    w = dict(model.named_parameters())[defect.tensor_name]
    raw = torch.randn_like(w) * 0.01
    safe = loop.nullspace.project_into_nullspace(defect.tensor_name, raw)
    assert torch.linalg.matrix_norm(safe @ basis) < 1e-5


def test_runtime_curriculum_with_meta_rounds(harness, tokenizer):
    model, _defect, adapter, ledger, loop, task_eval, anchor_kl, _capture, _diag = harness
    domain = TaskDomain("logical_deduction", "attractor_loop", FAIL_PROMPTS, PASS_PROMPTS, task_eval)
    runtime = WRSIRuntime(
        loop,
        adapter,
        tokenizer,
        [domain],
        TemplateCandidateSource(),
        anchor_kl,
        # seed 1 schedules meta rounds with both an attractor and a spectral puzzle
        WRSIConfig(max_sparsity=0.005, meta_fraction=0.5, seed=1),
        gym=MetaSurgeryGym(),
        meta_proposer=GymTemplateProposer(),
        watchdog=WeightWatchdog(adapter),
    )
    reports = runtime.run(6)
    kinds = {r.kind for r in reports}
    assert kinds == {"first_order", "meta"}
    assert any(r.promoted for r in reports if r.kind == "first_order")
    meta_trials = [t for t in ledger.list_trials(limit=500) if t.domain == META_DOMAIN]
    meta_reports = [r for r in reports if r.kind == "meta"]
    assert len(meta_trials) == sum(r.candidates for r in meta_reports)  # every attempt is remembered
    for r in meta_reports:  # the offline proposer only knows the attractor repair
        assert (r.candidates > 0) == (r.extra["challenge"] == "planted_attractor")
        assert r.promoted == (r.extra["challenge"] == "planted_attractor")
    assert {r.extra["challenge"] for r in meta_reports} == {"planted_attractor", "spectral_explosion"}
    assert reports[-1].surgical_hit_rate == ledger.get_surgical_hit_rate()
    # Already-tried scripts are not proposed again.
    codes = [t.synthesized_code for t in ledger.list_trials(limit=200) if t.domain == "logical_deduction"]
    assert len(codes) == len(set(codes))


def test_watchdog_vetoes_destabilising_promotion(harness, tokenizer):
    model, defect, adapter, ledger, loop, _task_eval, anchor_kl, _capture, _diag = harness
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    huge = (
        "def compute_delta(weights, diagnostics):\n"
        "    name = diagnostics['target_tensor']\n"
        "    delta = torch.zeros_like(weights[name])\n"
        "    delta[0, 0] = 50000.0\n"
        "    return {name: delta}\n"
    )

    class OneShot:
        def propose(self, context):
            return [CandidatePatch("explode", huge)]

    def rewards_magnitude(m):  # a mis-specified task metric that a bad edit can game
        return float(dict(m.named_parameters())[defect.tensor_name].detach().abs().max())

    domain = TaskDomain("tool_execution", "attractor_loop", FAIL_PROMPTS, PASS_PROMPTS, rewards_magnitude)
    runtime = WRSIRuntime(
        loop,
        adapter,
        tokenizer,
        [domain],
        OneShot(),
        lambda m: (True, 0.0),
        WRSIConfig(max_sparsity=0.005, meta_fraction=0.0),
        watchdog=WeightWatchdog(adapter),
    )
    (report,) = runtime.run(1)
    assert report.promoted is False and "watchdog rollback" in report.note and loop.generation == 0
    assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())


def test_self_edit_prompt_contains_memory(harness):
    *_, diagnostics = harness
    from libweightspace.candidates import ProposalContext

    ctx = ProposalContext(
        iteration=0,
        diagnostics=diagnostics,
        topology={"family": "qwen3_8"},
        num_candidates=2,
        ledger_history={
            "past_successes": [{"hypothesis": "worked", "target_score_delta": 1.0}],
            "recent_failures_to_avoid": [{"hypothesis": "broke", "rejection_reason": "KL"}],
        },
    )
    prompt = build_self_edit_prompt(ctx)
    assert "worked" in prompt and "broke" in prompt and "tensor[32]" in prompt and "exactly 2" in prompt
    assert "YOUR OWN neural parameters" in WRSI_SELF_EDITOR_PROMPT


def test_inweight_self_edit_proposer_and_second_order_meta_repair(harness, tokenizer):
    """When M_t has a planted attractor defect, InWeightSelfEditProposer fails non-attractor meta-challenges
    until run_meta triggers a second-order self-repair on M_t, after which M_t solves all 3 challenge types.
    """
    from libweightspace.candidates import ProposalContext
    from libweightspace.llm_backends import InProcessModelBackend

    model, _defect, adapter, ledger, loop, task_eval, anchor_kl, _capture, _diag = harness
    backend = InProcessModelBackend(model, tokenizer, max_new_tokens=8)
    proposer = InWeightSelfEditProposer(model, adapter, tokenizer, backend=backend)
    gym = MetaSurgeryGym()

    # Before self-repair: model has high column collinearity (> 0.90) from the planted rank-1 attractor.
    assert proposer.weight_health_score() > 0.90
    kv_challenge = gym.generate("corrupted_kv_memory", seed=5)
    kv_ctx = ProposalContext(
        iteration=0,
        diagnostics=kv_challenge.diagnostics,
        topology={"tensor": "mlp.down_proj.weight"},
        intent_description=kv_challenge.prompt_description,
    )
    assert proposer.propose(kv_ctx) == []

    # Running a meta round on a non-attractor challenge triggers second-order self-repair on loop.model
    # and then succeeds on the retry!
    domain = TaskDomain("logical_deduction", "attractor_loop", FAIL_PROMPTS, PASS_PROMPTS, task_eval)
    runtime = WRSIRuntime(
        loop,
        adapter,
        tokenizer,
        [domain],
        proposer,
        anchor_kl,
        WRSIConfig(max_sparsity=0.005, meta_fraction=1.0, consolidation_interval=1, seed=1),
        gym=gym,
        meta_proposer=proposer,
        watchdog=WeightWatchdog(adapter),
    )
    reports = runtime.run(4)
    assert loop.generation >= 1
    assert proposer.weight_health_score() <= 0.90
    assert any(r.extra.get("self_repaired") for r in reports)
    assert all(r.promoted for r in reports)

    # Rolling back the self-edit restores the defective weights and reverts the proposer's capability.
    while loop.rollback_last_generation():
        pass
    assert proposer.weight_health_score() > 0.90
    assert proposer.propose(kv_ctx) == []


def test_weight_space_toolbox_introspection_and_patch(harness, tokenizer):
    model, _defect, adapter, _ledger, loop, task_eval, anchor_kl, _capture, _diag = harness
    toolbox = WeightSpaceToolbox(loop, adapter, tokenizer, anchor_kl)
    svd_info = toolbox.inspect_layer_svd(1, top_k=4)
    assert svd_info["layer_idx"] == 1 and len(svd_info["top_singular_values"]) == 4

    trace = toolbox.trace_activation_divergence(FAIL_PROMPTS, PASS_PROMPTS, top_k_cols=4)
    assert trace["target_layer"] == 1 and len(trace["top_culpable_mlp_cols"]) == 4

    domain = TaskDomain("logical_deduction", "attractor_loop", FAIL_PROMPTS, PASS_PROMPTS, task_eval)
    assert toolbox.propose_and_test_patch(domain, "toolbox fix", nullification_script(3)) is True
    mem = toolbox.query_episodic_ledger("logical_deduction", limit=3, query_vector=[0.002, 0.05, 1.0, 0.0])
    assert len(mem["past_successes"]) == 1 and len(mem["similar_trials"]) == 1
