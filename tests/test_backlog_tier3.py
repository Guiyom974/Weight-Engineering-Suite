"""Unit, integration, and security tests for Tier 3 backlog items (WRSI-BL-01..04) and new CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from libweightspace.candidates import TemplateCandidateSource
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.isolated_sandbox import (
    execute_code_pass_at_1_trial,
    execute_weight_surgery_subprocess,
)
from libweightspace.loading import load_model
from libweightspace.nullspace import NullspaceLocker
from libweightspace.prober import MechanisticProber
from libweightspace.sandbox import SandboxSecurityViolation
from neuroforge.cli import app as nf_app
from neuroforge.intent_compiler import compile_suites
from neuroforge.metrics import CodePassAt1Suite
from neuroforge.schemas import IntentSpec, MetricGoal, ProbePair
from wrsi.cli import app as wrsi_app
from wrsi.episodic_ledger import EpisodicWeightLedger
from wrsi.islands import IslandPopulation
from wrsi.scientist_agent import TaskDomain, WRSIScientistLoop

runner = CliRunner()


def test_isolated_subprocess_sandbox_and_negative_controls() -> None:
    """WRSI-BL-02: execute_weight_surgery_subprocess runs valid scripts and rejects escapes/infinite loops."""
    w = {"layer.weight": torch.ones(8, 8)}
    valid_script = """
def compute_delta(weights, diagnostics):
    W = weights["layer.weight"]
    return {"layer.weight": -0.1 * W}
"""
    deltas = execute_weight_surgery_subprocess(
        valid_script, w, {"gain": 0.1}, timeout=10.0, max_memory_mb=2048  # torch needs about 1-2 GB of commit
    )
    assert "layer.weight" in deltas
    assert torch.allclose(deltas["layer.weight"], torch.full((8, 8), -0.1))

    # Negative security test 1: static AST violation rejected before subprocess launch.
    with pytest.raises(SandboxSecurityViolation):
        execute_weight_surgery_subprocess("import os\ndef compute_delta(w, d): return {}", w, {})

    # Negative security test 2: line-budget / infinite loop inside child process raises SandboxSecurityViolation.
    infinite_script = """
def compute_delta(weights, diagnostics):
    x = 0
    while True:
        x = x + 1
    return {}
"""
    with pytest.raises(SandboxSecurityViolation, match="Execution budget exceeded"):
        execute_weight_surgery_subprocess(
            infinite_script, w, {}, timeout=10.0, max_line_events=5000
        )


def test_code_pass_at_1_suite_and_intent_compiler() -> None:
    """WRSI-BL-02: CodePassAt1Suite executes unit tests in subprocess and blocks forbidden imports."""
    assert execute_code_pass_at_1_trial("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert not execute_code_pass_at_1_trial("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")
    # Negative security control: dangerous import is blocked.
    assert not execute_code_pass_at_1_trial("import os\ndef f(): return 1", "assert f() == 1")

    bundle = load_model("toy:qwen3_8")
    probes = [
        ProbePair(
            probe_id="p1",
            prompt="def sq(x): ",
            expected_completion_prefix="return x * x",
            undesired_completion_prefix="return 0",
            metadata={"candidate_code": "def sq(x):\n    return x * x", "test_code": "assert sq(4) == 16"},
        )
    ]
    suite = CodePassAt1Suite("code_suite", probes, bundle.tokenizer)
    metrics = suite.evaluate(bundle.model)
    assert metrics["pass_at_1"] == 1.0

    intent = IntentSpec(
        intent_id="code-test",
        description="test pass_at_1 compilation",
        target_behavior="pass unit test",
        failure_behavior="fail unit test",
        contrastive_probes=probes,
        target_goals=[
            MetricGoal(suite_name="code_suite", metric_name="pass_at_1", operator=">=", threshold=1.0)
        ],
        invariants=[],
    )
    target_suites, _ = compile_suites(intent, bundle.tokenizer, bundle.model)
    assert isinstance(target_suites["code_suite"], CodePassAt1Suite)


def test_subcomponent_circuit_tracing() -> None:
    """WRSI-BL-03: MechanisticProber.trace_circuit isolates subcomponent edges and culpable columns."""
    bundle = load_model("toy:qwen3_8", demo_defect=True)
    prober = MechanisticProber(bundle.model, bundle.adapter, bundle.tokenizer)
    edges = prober.trace_circuit(["ab xxxxx", "cd xxxxx"], ["ab xyzyz", "cd xyzyz"], top_k_edges=6)
    assert len(edges) == 6
    assert abs(edges[0].attribution_score) >= abs(edges[-1].attribution_score)
    assert any(e.source_node == "L1.mlp" for e in edges)


def test_multi_island_speciation_and_crossover(tmp_path: Path) -> None:
    """WRSI-BL-01: IslandPopulation evolves specialized islands and crosses winning deltas."""
    bundle = load_model("toy:qwen3_8", demo_defect=True)
    ledger = EpisodicWeightLedger(tmp_path / "ledger.sqlite")
    nullspace = NullspaceLocker()
    loop = WRSIScientistLoop(
        bundle.model,
        CoWWeightOverlay(bundle.model),
        nullspace,
        ledger,
        max_sparsity=0.01,
    )
    dom_a = TaskDomain(
        name="dom_a",
        failure_taxonomy="attractor_loop",
        fail_texts=["ab xxxxx"],
        pass_texts=["ab xyzyz"],
        evaluate=lambda m: float(
            -MechanisticProber(m, bundle.adapter, bundle.tokenizer)
            .trace_contrastive(["ab xxxxx"], ["ab xyzyz"])[0]
            .divergence_score
        ),
    )
    dom_b = TaskDomain(
        name="dom_b",
        failure_taxonomy="attractor_loop",
        fail_texts=["cd xxxxx"],
        pass_texts=["cd xyzyz"],
        evaluate=lambda m: float(
            -MechanisticProber(m, bundle.adapter, bundle.tokenizer)
            .trace_contrastive(["cd xxxxx"], ["cd xyzyz"])[0]
            .divergence_score
        ),
    )
    pop = IslandPopulation(
        loop,
        bundle.adapter,
        bundle.tokenizer,
        [dom_a, dom_b],
        TemplateCandidateSource(),
        anchor_kl_fn=lambda _m: (True, 0.001),
        num_islands=2,
    )
    for i, isl in enumerate(pop.islands):
        assert pop.step_island(isl, step=i)
    report = pop.crossover_tournament(step=2)
    assert report.promoted
    assert report.score_after > report.score_before
    assert loop.generation == 1


def test_new_cli_commands_end_to_end(tmp_path: Path) -> None:
    """Smoke tests for all new NeuroForge and W-RSI CLI subcommands."""
    ws = tmp_path / "nf_ws"
    res = runner.invoke(nf_app, ["init", "--demo", "-w", str(ws)])
    assert res.exit_code == 0

    # spectrum
    res = runner.invoke(nf_app, ["spectrum", "-w", str(ws), "--layer", "1", "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout)[0]["layer"] == 1

    # trace-circuit
    res = runner.invoke(
        nf_app, ["trace-circuit", "-w", str(ws), "-i", str(ws / "intent.yaml"), "--top-edges", "4", "--json"]
    )
    assert res.exit_code == 0
    assert len(json.loads(res.stdout)) == 4

    # diff-models
    res = runner.invoke(
        nf_app,
        ["diff-models", "--model-a", "toy:qwen3_8", "--model-b", "toy:qwen3_8", "--demo-defect-b", "--json"],
    )
    assert res.exit_code == 0
    assert len(json.loads(res.stdout)) >= 1

    # calibrate-nullspace & inject-facts
    ns_path = tmp_path / "nullspace.safetensors"
    res = runner.invoke(nf_app, ["calibrate-nullspace", "-w", str(ws), "--layer", "1", "-o", str(ns_path)])
    assert res.exit_code == 0
    assert ns_path.is_file()

    facts_file = tmp_path / "facts.json"
    facts_file.write_text(json.dumps([{"prompt": "Paris is in", "target": "France"}]), encoding="utf-8")
    res = runner.invoke(
        nf_app,
        ["inject-facts", "-w", str(ws), "--layer", "1", "-f", str(facts_file), "--nullspace", str(ns_path)],
    )
    assert res.exit_code == 0

    # branch & merge & export --format lora
    assert runner.invoke(nf_app, ["branch", "b1", "-w", str(ws)]).exit_code == 0
    assert runner.invoke(nf_app, ["loop", "-w", str(ws), "-i", str(ws / "intent.yaml")]).exit_code == 0
    assert runner.invoke(nf_app, ["branch", "b2", "-w", str(ws)]).exit_code == 0
    assert (
        runner.invoke(nf_app, ["merge", "b1", "b2", "--strategy", "orthomerge", "-w", str(ws)]).exit_code == 0
    )
    lora_out = tmp_path / "lora_adapter"
    assert (
        runner.invoke(nf_app, ["export", "-w", str(ws), "--format", "lora", "-o", str(lora_out)]).exit_code
        == 0
    )
    assert (lora_out / "adapter_config.json").is_file()

    # studio --check-only
    assert (
        runner.invoke(nf_app, ["studio", "-w", str(ws), "--port", "0", "--check-only"]).exit_code == 0
    )

    # wrsi start --islands 2
    wrsi_ws = tmp_path / "wrsi_ws"
    res_wrsi = runner.invoke(
        wrsi_app, ["start", "-w", str(wrsi_ws), "--generations", "2", "--islands", "2"]
    )
    assert res_wrsi.exit_code == 0
