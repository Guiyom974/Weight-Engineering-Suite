"""End-to-end CLI tests on toy models (blueprint step 12): real commands, real files, real anchor subprocess."""

import json

import pytest
import torch
from safetensors.torch import load_file, load_model
from typer.testing import CliRunner

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.prober import capture_linear_inputs
from libweightspace.toy_models import build_tiny_qwen3, plant_attractor_defect
from neuroforge.cli import app as nf
from neuroforge.weight_git import WeightGitRepository
from wrsi.cli import app as wrsi
from wrsi.scientist_agent import WRSIScientistLoop
from wrsi.workspace import WRSIWorkspace

runner = CliRunner()
TENSOR = "model.layers.1.mlp.down_proj.weight"


def run(app, *args, code=0):
    result = runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})
    assert result.exit_code == code, f"{args}\n{result.output}\n{result.exception!r}"
    return result


@pytest.fixture
def demo(tmp_path):
    ws = tmp_path / "demo"
    run(nf, "init", "--demo", "-w", ws)
    return ws


def test_init_creates_workspace(demo):
    assert (demo / "neuroforge.yaml").is_file() and (demo / "intent.yaml").is_file()
    assert WeightGitRepository.exists(demo)


def test_probe_loop_log_export_flow(demo, tmp_path):
    intent = demo / "intent.yaml"
    out = run(nf, "probe", "-i", intent, "-w", demo).output
    assert "Mechanistic attribution" in out and "down_proj" in out

    out = run(nf, "loop", "-i", intent, "-w", demo, "--reasoner", "template").output
    assert "converged=True" in out and "COMMITTED" in out

    rows = json.loads(run(nf, "log", "-w", demo, "--json").output)
    assert len(rows) == 1 and rows[0]["passed_gate"] is True

    # A second loop sees goals already met: nothing new is committed (state replayed from .wgit).
    assert "converged=True" in run(nf, "loop", "-i", intent, "-w", demo).output
    assert len(json.loads(run(nf, "log", "-w", demo, "--json").output)) == 1

    merged = tmp_path / "merged.safetensors"
    run(nf, "export", "-w", demo, "--out", merged)
    patched = build_tiny_qwen3(seed=0)
    load_model(patched, str(merged))
    defective = build_tiny_qwen3(seed=0)
    defect = plant_attractor_defect(defective)
    p, d = dict(patched.named_parameters()), dict(defective.named_parameters())
    assert not torch.equal(p[TENSOR], d[TENSOR])
    assert abs(float(p[TENSOR].detach()[:, 5] @ defect.direction)) < 0.02

    sparse = tmp_path / "delta.safetensors"
    run(nf, "export", "-w", demo, "--out", sparse, "--format", "sparse")
    assert f"{TENSOR}__indices" in load_file(str(sparse))
    assert (
        "save_pretrained"
        in run(nf, "export", "-w", demo, "--out", tmp_path / "hf", "--format", "hf", code=1).output
    )


def test_edit_gate(demo, tmp_path):
    intent = demo / "intent.yaml"
    good = tmp_path / "fix.py"
    good.write_text(
        "def compute_delta(weights, diagnostics):\n"
        "    name = diagnostics['target_tensor']\n"
        "    W = weights[name]\n"
        "    v = diagnostics['attractor_direction']\n"
        "    delta = torch.zeros_like(W)\n"
        "    for c in diagnostics['top_culpable_mlp_cols'][:3]:\n"
        "        delta[:, c] = -torch.dot(W[:, c], v) * v\n"
        "    return {name: delta}\n",
        encoding="utf-8",
    )
    run(nf, "edit", "-i", intent, "-s", good, "-w", demo)  # dry run
    assert WeightGitRepository(demo).list_commits() == []
    run(nf, "edit", "-i", intent, "-s", good, "-w", demo, "--commit")
    assert len(WeightGitRepository(demo).list_commits()) == 1

    bad = tmp_path / "bad.py"
    bad.write_text("import os\ndef compute_delta(weights, diagnostics):\n    return {}\n", encoding="utf-8")
    fresh = tmp_path / "fresh"
    run(nf, "init", "--demo", "-w", fresh)
    rejected = run(nf, "edit", "-i", fresh / "intent.yaml", "-s", bad, "-w", fresh, code=1).output
    assert "Import" in rejected and "error: Exit" not in rejected


def test_compile_blueprint(demo, tmp_path):
    out = tmp_path / "bp" / "model.safetensors"
    assert "compiled denovo-demo" in run(nf, "compile", demo / "blueprint.yaml", "--out", out).output
    assert out.is_file() and json.loads(out.with_suffix(".config.json").read_text())["hidden_size"] == 32


@pytest.mark.parametrize(
    "args, message",
    [
        (("probe", "-i", "missing.yaml", "-w", "{ws}"), "not found"),
        (("log", "-w", "{tmp}"), "no .wgit"),
        (("compile", "{ws}/blueprint.yaml", "--out", "x.bin"), ".safetensors"),
        (("export", "-w", "{ws}", "--out", "o", "--format", "gguf"), "--format"),
        (("export", "-w", "{ws}", "--out", "o", "--ref", "../../x"), "invalid ref"),
        (("export", "-w", "{ws}", "--out", "o", "--format", "sparse"), "no patches"),
        (("loop", "-i", "{ws}/intent.yaml", "-w", "{ws}", "--reasoner", "telepathy"), "unknown backend"),
    ],
)
def test_invalid_input_is_one_line_error(demo, tmp_path, args, message):
    formatted = [a.format(ws=demo, tmp=tmp_path) for a in args]
    result = run(nf, *formatted, code=1)
    assert message in result.output and "Traceback" not in result.output


def test_wrsi_start_status_rollback_resume(tmp_path):
    ws = tmp_path / "run"
    out = run(wrsi, "start", "-w", ws, "--generations", 2, "--meta-fraction", 0).output
    assert "generation=1" in out  # the first step repairs the attractor; the second has nothing better

    status = json.loads(run(wrsi, "status", "-w", ws, "--json").output)
    assert status["generation"] == 1 and status["steps_run"] == 2 and status["rollback_points"] == 1
    assert status["last_kl"] is not None and status["last_kl"] < 0.08
    assert status["ledger"]["promoted_trials"] >= 1 and TENSOR in status["modified_tensors"]

    trials = json.loads(run(wrsi, "telemetry", "-w", ws, "--json").output)
    assert any(t["promoted"] for t in trials) and any("Sparsity" in t["reason"] for t in trials)
    atlas = json.loads(run(wrsi, "telemetry", "-w", ws, "--atlas", "--json").output)
    assert {r["kind"] for r in atlas} == {"attention"}

    # Resume in a new process-like invocation: step counter and generation continue.
    run(wrsi, "start", "-w", ws, "--generations", 1, "--meta-fraction", 0)
    assert json.loads(run(wrsi, "status", "-w", ws, "--json").output)["steps_run"] == 3

    # Rollback across invocations restores M_0 bit-exactly and clears the overlay.
    state = WRSIWorkspace(ws).load_state()
    out = run(wrsi, "rollback", "-w", ws, "--steps", state.generation).output
    assert "now at generation 0" in out
    m0 = load_file(str(ws / ".wrsi" / "vault" / "m0.safetensors"))
    reloaded = build_tiny_qwen3(seed=0)
    plant_attractor_defect(reloaded)
    WRSIWorkspace(ws).apply_overlay(reloaded)
    assert not (ws / ".wrsi" / "overlay.safetensors").exists()
    assert torch.equal(dict(reloaded.named_parameters())[TENSOR], m0[TENSOR])
    assert "nothing to roll back" in run(wrsi, "rollback", "-w", ws, code=1).output


def _nullify_columns(cols):
    return (
        "def compute_delta(weights, diagnostics):\n"
        "    name = diagnostics['target_tensor']\n"
        "    W = weights[name]\n"
        "    v = diagnostics['attractor_direction']\n"
        "    delta = torch.zeros_like(W)\n"
        f"    for c in {cols}:\n"
        "        delta[:, c] = -torch.dot(W[:, c], v) * v\n"
        "    return {name: delta}\n"
    )


def test_persisted_rollback_is_bit_exact_across_two_generations(tmp_path, tokenizer):
    """Undoing only the newest of two saved generations lands exactly on M_1, then on M_0."""
    model = build_tiny_qwen3(seed=0)
    defect = plant_attractor_defect(model)
    workspace = WRSIWorkspace(tmp_path)
    loop = WRSIScientistLoop(
        model, CoWWeightOverlay(model), workspace.load_nullspace(), workspace.ledger(), max_sparsity=0.01
    )
    param = dict(model.named_parameters())[TENSOR]
    diagnostics = {"target_tensor": TENSOR, "attractor_direction": defect.direction}

    def attractor_energy_removed(m):  # strictly improves as planted columns are repaired
        w = dict(m.named_parameters())[TENSOR].detach()
        return -float((defect.direction @ w[:, defect.columns]).abs().sum())

    def capture(m, names):
        return capture_linear_inputs(m, tokenizer, ["ab yzyz", "cd zyzy"], names)

    def promote(cols):
        assert loop.evaluate_and_promote_candidates(
            "d",
            "attractor_loop",
            [{"hypothesis": "h", "python_script": _nullify_columns(cols)}],
            diagnostics,
            attractor_energy_removed,
            lambda m: (True, 0.0),
            capture,
        )

    m0 = param.detach().clone()
    promote([5])
    m1 = param.detach().clone()
    rank_after_gen0 = loop.nullspace.locked_rank(TENSOR)
    promote([12, 19])
    assert loop.generation == 2 and not torch.equal(param, m1)
    for promoted in loop.rollback_stack:
        workspace.save_rollback(promoted)
    workspace.save_overlay(model, [TENSOR])
    workspace.save_nullspace(loop.nullspace)

    restored = build_tiny_qwen3(seed=0)
    plant_attractor_defect(restored)
    workspace.apply_overlay(restored)
    locker = workspace.load_nullspace()
    restored_param = dict(restored.named_parameters())[TENSOR]
    assert torch.equal(restored_param, param)

    assert workspace.pop_rollback(restored, locker) == 1
    assert torch.equal(restored_param, m1) and locker.locked_rank(TENSOR) == rank_after_gen0
    assert workspace.pop_rollback(restored, locker) == 0
    assert torch.equal(restored_param, m0) and locker.locked_rank(TENSOR) == 0
    assert workspace.pop_rollback(restored, locker) is None


def test_wrsi_errors(tmp_path):
    assert "no W-RSI run" in run(wrsi, "status", "-w", tmp_path, code=1).output
    assert "no W-RSI run" in run(wrsi, "rollback", "-w", tmp_path, code=1).output
    missing = tmp_path / "nope.yaml"
    assert (
        "domains file not found"
        in run(wrsi, "start", "-w", tmp_path / "x", "--domains", missing, code=1).output
    )
    assert "invalid toy spec" in run(wrsi, "start", "-w", tmp_path / "z", "--model", "toy:", code=1).output
    bad = tmp_path / "domains.yaml"
    bad.write_text("not-a-list: 1\n", encoding="utf-8")
    assert (
        " non-empty YAML list" in run(wrsi, "start", "-w", tmp_path / "y", "--domains", bad, code=1).output
    )


def test_neuroforge_draft_intent_cli(tmp_path):
    out_file = tmp_path / "drafted.yaml"
    res = run(nf, "draft-intent", "Prevent x-token repetition loops", "--out", out_file)
    assert "offline template" in res.output and out_file.is_file()  # no LLM: the output says so
    assert "Prevent x-token repetition loops" in out_file.read_text(encoding="utf-8")


def test_wrsi_start_with_self_proposer_and_vault_pinning(tmp_path):
    ws = tmp_path / "self_run"
    out = run(wrsi, "start", "-w", ws, "--generations", 2, "--proposer", "self", "--meta-fraction", 0.5).output
    assert "generation=" in out
    state = WRSIWorkspace(ws).load_state()
    assert state.vault_sha256 is not None and len(state.vault_sha256) == 64
    assert state.base_weights_sha256 is not None and len(state.base_weights_sha256) == 64

    # Tampering with anchors.safetensors on disk is caught by the SHA-256 pin before the vault runs.
    vault_file = ws / ".wrsi" / "vault" / "anchors.safetensors"
    vault_file.write_bytes(vault_file.read_bytes() + b"\x00")
    err = run(wrsi, "start", "-w", ws, "--generations", 1, code=1).output
    assert "SHA-256 mismatch" in err


def test_real_huggingface_qwen2_checkpoint_end_to_end(tmp_path):
    """Creates a tiny randomly initialised Qwen2 checkpoint in Hugging Face format, loads, probes and exports it."""
    pytest.importorskip("transformers", reason="needs the hf extra: pip install -e '.[hf]'")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from libweightspace.loading import load_model as ws_load_model

    ckpt = tmp_path / "hf_qwen2"
    vocab = {"<unk>": 0, "hello": 1, "world": 2, "x": 3, "y": 4, "ab": 5, "cd": 6}
    tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_backend.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok_backend, unk_token="<unk>")
    fast_tok.save_pretrained(str(ckpt))

    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    hf_model = Qwen2ForCausalLM(cfg)
    hf_model.save_pretrained(str(ckpt))

    loaded = ws_load_model(str(ckpt))
    assert loaded.adapter.num_layers() == 2 and loaded.adapter.hidden_size() == 16

    ws = tmp_path / "nf_hf"
    run(nf, "init", "--model", str(ckpt), "-w", ws)
    export_dir = tmp_path / "exported_hf"
    run(nf, "export", "-w", ws, "--out", export_dir, "--format", "hf")
    assert (export_dir / "config.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_demo_loop_on_cuda(tmp_path):
    ws = tmp_path / "gpu"
    run(nf, "init", "--demo", "-w", ws)
    config = ws / "neuroforge.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("device: cpu", "device: cuda"), encoding="utf-8"
    )
    assert "converged=True" in run(nf, "loop", "-i", ws / "intent.yaml", "-w", ws).output
    assert len(WeightGitRepository(ws).list_commits()) == 1
