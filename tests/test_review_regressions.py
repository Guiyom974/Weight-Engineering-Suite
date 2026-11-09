"""Regressions for findings of the independent review (numbers refer to .bic/work/001-*/review.md)."""

import io
import json
import queue
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from libweightspace.adapters import Qwen3Adapter
from libweightspace.candidates import (
    CandidateParseError,
    TemplateCandidateSource,
    nullification_script,
    parse_candidates_json,
)
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.llm_backends import (
    CallableBackend,
    make_backend,
)
from libweightspace.nullspace import NullspaceLocker
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from libweightspace.toy_models import build_tiny_qwen3, plant_attractor_defect
from neuroforge.cli import app as nf
from neuroforge.convergence_engine import ConvergenceEngine
from neuroforge.intent_compiler import compile_suites
from neuroforge.metrics import CallableSuite, ProbeSuite
from neuroforge.reasoner_bridge import ReasonerBridge
from neuroforge.schemas import ProbePair
from neuroforge.weight_git import WeightGitError, WeightGitRepository, apply_deltas_inplace
from wrsi.anchor_client import AnchorVaultClient
from wrsi.anchor_vault_server import serve
from wrsi.cli import app as wrsi
from wrsi.episodic_ledger import EpisodicWeightLedger
from wrsi.scientist_agent import WRSIScientistLoop
from wrsi.workspace import WRSIWorkspace

from .conftest import FAIL_PROMPTS, PASS_PROMPTS
from .test_convergence_loop import _intent
from .test_weight_git import _delta, _manifest

W = "model.layers.1.mlp.down_proj.weight"
runner = CliRunner()
PROBES = [
    ProbePair(
        probe_id=f"p{i}", prompt=p, expected_completion_prefix="yzyz", undesired_completion_prefix="xxxx"
    )
    for i, p in enumerate(["ab x", "cd x", "the x", "q: x"])
]
OVERFLOW = (
    "def compute_delta(weights, diagnostics):\n"
    "    return {'model.norm.weight': torch.ones_like(weights['model.norm.weight']) * 3e38}\n"
)


def _cli(app, *args, code=0):
    result = runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})
    assert result.exit_code == code, f"{args}\n{result.output}\n{result.exception!r}"
    return result.output


@pytest.fixture
def loop_setup(planted_qwen, tokenizer, tmp_path):
    model, defect = planted_qwen
    loop = WRSIScientistLoop(
        model,
        CoWWeightOverlay(model),
        NullspaceLocker(),
        EpisodicWeightLedger(tmp_path / "l.db"),
        max_sparsity=0.01,
    )
    suite = ProbeSuite("task", PROBES, tokenizer)
    reports = MechanisticProber(model, Qwen3Adapter(model), tokenizer).trace_contrastive(
        FAIL_PROMPTS, PASS_PROMPTS
    )

    def task(m):
        return suite.evaluate(m)["logit_margin"]

    def capture(m, names):
        return capture_linear_inputs(m, tokenizer, PASS_PROMPTS, names)

    return model, defect, loop, task, capture, build_diagnostics(reports)


def _fix(name="fix"):
    return {"hypothesis": name, "python_script": nullification_script(3)}


def test_01_failed_commit_leaves_no_half_promotion(loop_setup):
    model, _defect, loop, task, _capture, diag = loop_setup
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def broken_capture(m, names):
        raise RuntimeError("capture failed")

    promoted = loop.evaluate_and_promote_candidates(
        "d", "attractor_loop", [_fix()], diag, task, lambda m: (True, 0.0), broken_capture
    )
    assert promoted is False and loop.generation == 0 and loop.rollback_stack == []
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    (trial,) = loop.ledger.list_trials()
    assert trial.promoted is False and "Commit failed" in trial.rejection_reason


def test_02_non_finite_score_rejects_only_that_candidate(loop_setup):
    _model, _defect, loop, task, capture, diag = loop_setup
    candidates = [{"hypothesis": "overflow", "python_script": OVERFLOW}, _fix()]
    assert loop.evaluate_and_promote_candidates(
        "d", "attractor_loop", candidates, diag, task, lambda m: (True, 0.0), capture
    )
    trials = {t.hypothesis: t for t in loop.ledger.list_trials()}
    assert "Non-finite" in trials["overflow"].rejection_reason and trials["fix"].promoted


def test_02_nan_kl_is_not_stored_as_zero(loop_setup):
    _model, _defect, loop, task, capture, diag = loop_setup
    assert not loop.evaluate_and_promote_candidates(
        "d", "attractor_loop", [_fix()], diag, task, lambda m: (True, float("nan")), capture
    )
    assert "Non-finite" in loop.ledger.list_trials()[0].rejection_reason


def test_03_invariant_baseline_comes_from_base_model(planted_qwen, tokenizer):
    model, _defect = planted_qwen
    clean = build_tiny_qwen3(seed=0)
    intent = _intent(-2.85)
    targets, invariants = compile_suites(intent, tokenizer, clean)
    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        TemplateCandidateSource(),
        nullspace_reference=clean,
        baseline_model=clean,
    )
    result = engine.run()
    assert engine._baseline_invariants["baseline.kl_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert result.converged


def test_05_main_only_moves_for_patches_that_pass_the_gate(planted_qwen, tokenizer, tmp_path):
    model, _defect = planted_qwen
    clean = build_tiny_qwen3(seed=0)
    intent = _intent(-2.85)
    targets, invariants = compile_suites(intent, tokenizer, clean)
    repo = WeightGitRepository.init(tmp_path)
    weak = CallableBackend(lambda s, p: json.dumps([{"python_script": nullification_script(1)}]))
    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        targets,
        invariants,
        ReasonerBridge(weak),
        repo=repo,
        max_iterations=1,
        nullspace_reference=clean,
    )
    result = engine.run()
    assert not result.converged and repo.head() == result.iterations[0].committed_patch_id
    assert repo.main() == "ROOT"


def test_06_only_the_applied_candidate_counts_as_promoted(loop_setup):
    _model, _defect, loop, task, capture, diag = loop_setup
    candidates = [
        {"hypothesis": "narrow", "python_script": nullification_script(2)},
        {"hypothesis": "full", "python_script": nullification_script(3)},
    ]
    assert loop.evaluate_and_promote_candidates(
        "d", "attractor_loop", candidates, diag, task, lambda m: (True, 0.0), capture
    )
    trials = {t.hypothesis: t for t in loop.ledger.list_trials()}
    assert sum(t.promoted for t in trials.values()) == 1 and trials["full"].promoted
    assert "not selected" in trials["narrow"].rejection_reason
    assert loop.ledger.get_surgical_hit_rate() == pytest.approx(0.5)


def test_07_rollback_updates_the_ledger(loop_setup):
    _model, _defect, loop, task, capture, diag = loop_setup
    loop.evaluate_and_promote_candidates(
        "d", "attractor_loop", [_fix()], diag, task, lambda m: (True, 0.0), capture
    )
    assert loop.rollback_last_generation()
    (trial,) = loop.ledger.list_trials()
    assert trial.promoted is False and "Rolled back" in trial.rejection_reason
    assert loop.ledger.get_surgical_hit_rate() == 0.0


def test_07_cli_rollback_updates_the_ledger(tmp_path):
    _cli(wrsi, "start", "-w", tmp_path, "--generations", 1, "--meta-fraction", 0)
    _cli(wrsi, "rollback", "-w", tmp_path)
    assert json.loads(_cli(wrsi, "status", "-w", tmp_path, "--json"))["ledger"]["promoted_trials"] == 0


def test_08_stale_reply_after_timeout_is_skipped():
    client = AnchorVaultClient.__new__(AnchorVaultClient)
    client.timeout = 1.0
    client._next_id = 1
    client._proc = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
    client._lines = queue.Queue()
    client._lines.put(json.dumps({"jsonrpc": "2.0", "id": 1, "result": "stale"}))
    client._lines.put(json.dumps({"jsonrpc": "2.0", "id": 2, "result": "fresh"}))
    assert client.request("ping") == "fresh"


def test_09_rollback_file_survives_until_commit(tmp_path):
    _cli(wrsi, "start", "-w", tmp_path, "--generations", 1, "--meta-fraction", 0)
    workspace = WRSIWorkspace(tmp_path)
    model = build_tiny_qwen3(seed=0)
    plant_attractor_defect(model)
    workspace.apply_overlay(model)
    (path,) = workspace.rollback_files()
    assert workspace.pop_rollback(model, workspace.load_nullspace()) == 0
    assert path.exists() and workspace.rollback_files() == []
    workspace.commit_rollback()
    assert not path.exists()


def test_10_revert_requires_an_unreverted_ancestor(tmp_path, tiny_qwen):
    repo = WeightGitRepository.init(tmp_path)
    first = repo.commit({W: _delta(tiny_qwen, W, [1], 0.5)}, _manifest("a"))
    repo.revert(first.patch_id)
    with pytest.raises(WeightGitError, match="already reverted"):
        repo.revert(first.patch_id)
    repo.checkout("ROOT")
    other = repo.commit({W: _delta(tiny_qwen, W, [2], 0.5)}, _manifest("b"))
    with pytest.raises(WeightGitError, match="not an ancestor"):
        repo.revert(first.patch_id)
    assert repo.head() == other.patch_id


def test_11_verify_detects_swapped_object_and_hash(tmp_path, tiny_qwen):
    repo = WeightGitRepository.init(tmp_path)
    good = repo.commit({W: _delta(tiny_qwen, W, [1], 0.5)}, _manifest("a"))
    forged = repo.commit({W: _delta(tiny_qwen, W, [2], 9.0)}, _manifest("b"))
    objects, commits = tmp_path / ".wgit" / "objects", tmp_path / ".wgit" / "commits"
    (objects / f"{good.patch_id}.COO.safetensors").write_bytes(
        (objects / f"{forged.patch_id}.COO.safetensors").read_bytes()
    )
    path = commits / f"{good.patch_id}.json"
    path.write_text(path.read_text().replace(good.delta_sha256, forged.delta_sha256), encoding="utf-8")
    assert repo.verify(good.patch_id) is False


def test_11_history_is_bound_to_its_base_model(tmp_path):
    ws = tmp_path / "w"
    _cli(nf, "init", "--demo", "-w", ws)
    _cli(nf, "loop", "-i", ws / "intent.yaml", "-w", ws)
    config = ws / "neuroforge.yaml"
    config.write_text(
        config.read_text().replace("model: toy:qwen3_8\n", "model: toy:qwen3_8:7\n", 1), encoding="utf-8"
    )
    assert "different base model" in _cli(nf, "log", "-w", ws) + _cli(
        nf, "probe", "-i", ws / "intent.yaml", "-w", ws, code=1
    )


def test_13_unsupported_metric_fails_at_compile_time(tokenizer, tiny_qwen):
    intent = _intent(0.0)
    intent.target_goals[0].metric_name = "pass_at_1"
    with pytest.raises(ValueError, match="custom suite"):
        compile_suites(intent, tokenizer, tiny_qwen)


def test_16_api_key_is_never_sent_implicitly(monkeypatch):
    monkeypatch.delenv("WEIGHTLAB_LLM_KEY_ENV", raising=False)
    monkeypatch.delenv("WEIGHTLAB_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("POE_API_KEY", "poe-secret")
    backend = make_backend("llm", model="m", base_url="https://example.invalid/v1")
    assert backend.build_request("s", "p").get_header("Authorization") is None  # Poe key stays with Poe
    monkeypatch.setenv("MY_ENDPOINT_KEY", "sk-opt-in")
    remote = make_backend("llm", model="m", base_url="https://example.invalid/v1", api_key_env="MY_ENDPOINT_KEY")
    assert remote.build_request("s", "p").get_header("Authorization") == "Bearer sk-opt-in"
    with pytest.raises(ValueError, match="plain HTTP"):
        make_backend("llm", model="m", base_url="http://example.invalid/v1", api_key_env="MY_ENDPOINT_KEY")
    poe = make_backend("llm", model="m")  # default endpoint is Poe
    assert poe.build_request("s", "p").get_header("Authorization") == "Bearer poe-secret"


def test_17_hostile_llm_reply_is_a_parse_error():
    with pytest.raises(CandidateParseError):
        parse_candidates_json("[" * 200_000)
    with pytest.raises(CandidateParseError, match="too large"):
        parse_candidates_json(" " * 10_000_000 + "{}")


def test_18_deeply_nested_request_does_not_kill_the_vault():
    server = SimpleNamespace(handle=lambda req: ({"jsonrpc": "2.0", "id": 1, "result": "pong"}, True))
    out = io.StringIO()
    serve(server, io.StringIO("[" * 60_000 + "\n" + '{"id": 1}\n'), out)
    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["error"]["code"] == -32700 and second["result"] == "pong"


def test_19_capture_only_covers_linear_layers(tiny_qwen, tokenizer):
    captured = capture_linear_inputs(tiny_qwen, tokenizer, ["abc", "de"], ["model.embed_tokens.weight", W])
    assert list(captured) == [W]


def test_20_suite_failure_rejects_the_candidate_only(planted_qwen, tokenizer):
    model, _defect = planted_qwen
    clean = build_tiny_qwen3(seed=0)
    intent = _intent(-2.85)
    targets, invariants = compile_suites(intent, tokenizer, clean)
    real = targets["loop"]
    calls = {"n": 0}

    def flaky(m):
        calls["n"] += 1
        if calls["n"] == 2:  # first call is the baseline, second is the candidate fork
            raise RuntimeError("suite exploded")
        return real.evaluate(m)

    engine = ConvergenceEngine(
        model,
        Qwen3Adapter(model),
        tokenizer,
        intent,
        {"loop": CallableSuite("loop", flaky)},
        invariants,
        TemplateCandidateSource(),
        max_iterations=1,
        nullspace_reference=clean,
    )
    result = engine.run()
    assert "suite exploded" in result.iterations[0].evaluations[0].error


def test_21_apply_deltas_validates_before_writing(tiny_qwen):
    before = dict(tiny_qwen.named_parameters())[W].detach().clone()
    with pytest.raises(KeyError):
        apply_deltas_inplace(tiny_qwen, {W: torch.ones_like(before), "nope": torch.zeros(1)})
    assert torch.equal(dict(tiny_qwen.named_parameters())[W], before)


def test_22_patch_ids_reject_trailing_newline(tmp_path):
    repo = WeightGitRepository.init(tmp_path)
    with pytest.raises(WeightGitError, match="invalid patch id"):
        repo.load_manifest("0123456789abcdef\n")


def test_23_read_only_commands_create_nothing(tmp_path):
    typo = tmp_path / "typo"
    assert "no W-RSI run" in _cli(wrsi, "status", "-w", typo, code=1)
    assert not typo.exists()


def test_24_os_errors_are_one_line_messages(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert "error:" in _cli(wrsi, "start", "-w", blocker, code=1)


def test_14_non_toy_models_need_anchor_file(tmp_path):
    domains = tmp_path / "d.yaml"
    domains.write_text("- name: d\n  probes: []\n", encoding="utf-8")
    missing = tmp_path / "anchors.txt"
    assert "anchors file not found" in _cli(wrsi, "start", "-w", tmp_path / "w", "--anchors", missing, code=1)


