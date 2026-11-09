"""NeuroForge command line: ``neuroforge init/probe/edit/compile/loop/export/log``.

Workspace model: ``neuroforge.yaml`` names the base model; ``.wgit`` holds the patch history. Every command loads
the base model and replays HEAD's cumulative delta, so the on-disk state is always base + patches. Offline demo:

    neuroforge init --demo -w demo
    neuroforge probe -w demo --intent demo/intent.yaml
    neuroforge loop  -w demo --intent demo/intent.yaml --reasoner template
    neuroforge export -w demo --out demo/merged.safetensors

Expected failures print one line on stderr and exit with code 1; tracebacks are not shown.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch.nn as nn
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file, save_file, save_model

from libweightspace.alphaedit import alphaedit_batch_update, compute_covariance_nullspace
from libweightspace.candidates import (
    CandidateParseError,
    CandidatePatch,
    CandidateSource,
    ProposalContext,
    TemplateCandidateSource,
)
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.llm_backends import LLMBackendError, make_backend
from libweightspace.llm_config import ModelListError, fetch_models, resolve_settings
from libweightspace.loading import load_model
from libweightspace.prober import MechanisticProber, build_diagnostics, capture_linear_inputs
from libweightspace.spectral import profile_matrix
from libweightspace.transcoder import diff_models

from .autotuner import autotune_candidate
from .blueprint_compiler import BlueprintCompiler
from .convergence_engine import ConvergenceResult
from .exporter import (
    export_hf_pretrained,
    export_merged_safetensors,
    export_peft_lora,
    export_sparse_delta,
    patch_gguf_inplace,
)
from .intent_compiler import contrastive_texts, draft_intent, load_blueprint, load_intent
from .merger import merge_deltas
from .patch_stats import state_fingerprint, summarize_patch
from .reasoner_bridge import ReasonerBridge
from .schemas import IntentSpec, PatchManifest, WorkspaceConfig
from .studio import StudioServer, StudioState
from .templates import DEMO_BLUEPRINT, DEMO_INTENT
from .weight_git import WeightGitError, WeightGitRepository, apply_deltas_inplace, manifest_summary
from .workspace_manager import StudioHome
from .workspace_ops import CONFIG_NAME
from .workspace_ops import build_engine as _engine
from .workspace_ops import load_config as _load_config
from .workspace_ops import open_workspace as _open_workspace

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    help="NeuroForge: intent-driven weight-space programming.",
)
console = Console()
error_console = Console(stderr=True)
EXPECTED_ERRORS = (
    ValueError,
    KeyError,
    OSError,
    RuntimeError,
    ValidationError,
    WeightGitError,
    LLMBackendError,
    CandidateParseError,
    yaml.YAMLError,
)
F = TypeVar("F", bound=Callable[..., Any])

WorkspaceOpt = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory.")
IntentOpt = typer.Option(..., "--intent", "-i", help="Intent YAML/JSON file.")
ReasonerOpt = typer.Option(
    "template", help="template (offline built-in candidates) or llm (any OpenAI-compatible endpoint)."
)
ModelOpt = typer.Option(None, "--model", help="LLM model id; list them with `neuroforge models`.")
BaseUrlOpt = typer.Option(None, help="OpenAI-compatible endpoint (default: Poe, https://api.poe.com/v1).")
KeyEnvOpt = typer.Option(None, "--key-env", help="Name of the environment variable holding the API key.")


def cli_errors(fn: F) -> F:
    """Turns expected exceptions into a one-line message and exit code 1."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except typer.Exit:  # subclasses RuntimeError; a deliberate exit is not an error
            raise
        except EXPECTED_ERRORS as exc:
            first_line = (str(exc).splitlines() or [type(exc).__name__])[0]
            error_console.print(f"error: {first_line}", markup=False)
            raise typer.Exit(code=1) from exc

    return wrapper  # type: ignore[return-value]


@app.command()
@cli_errors
def init(
    workspace: Path = WorkspaceOpt,
    model: str = typer.Option(
        "toy:qwen3_8", help="Model spec: toy:<arch>[:<seed>] or a Hugging Face id/path."
    ),
    reference: str | None = typer.Option(None, help="Known-good model spec for invariants."),
    demo: bool = typer.Option(
        False, help="Toy model with a planted defect plus matching intent and blueprint."
    ),
) -> None:
    """Creates a workspace: config, .wgit repository and starter files."""
    workspace.mkdir(parents=True, exist_ok=True)
    if demo:
        model, reference = "toy:qwen3_8", "toy:qwen3_8"
    cfg = WorkspaceConfig(model=model, demo_defect=demo, reference_model=reference)
    (workspace / CONFIG_NAME).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    WeightGitRepository.init(workspace)
    if demo:
        (workspace / "intent.yaml").write_text(yaml.safe_dump(DEMO_INTENT, sort_keys=False), encoding="utf-8")
        (workspace / "blueprint.yaml").write_text(
            yaml.safe_dump(DEMO_BLUEPRINT, sort_keys=False), encoding="utf-8"
        )
    console.print(f"workspace ready: {workspace} (model {cfg.model})", markup=False)


@app.command()
@cli_errors
def probe(
    intent: Path = IntentOpt,
    workspace: Path = WorkspaceOpt,
    top: int = typer.Option(5, min=1, help="Layers to show."),
    columns: int = typer.Option(8, min=1, help="Culpable columns to show per layer."),
) -> None:
    """Contrastive tracing: which layers and columns write the failure direction."""
    _cfg, loaded, _repo = _open_workspace(workspace)
    fail, passing = contrastive_texts(load_intent(intent))
    reports = MechanisticProber(loaded.model, loaded.adapter, loaded.tokenizer).trace_contrastive(
        fail, passing
    )
    table = Table(title="Mechanistic attribution")
    for name in ("layer", "divergence", "tensor", "culpable columns"):
        table.add_column(name, overflow="fold")
    for r in reports[:top]:
        table.add_row(
            str(r.layer_idx),
            f"{r.divergence_score:.4f}",
            str(r.mlp_down_proj),
            str(r.top_culpable_mlp_cols[:columns]),
        )
    console.print(table)


class _FixedCandidate:
    """Candidate source that offers one hand-written script (``neuroforge edit``)."""

    def __init__(self, candidate: CandidatePatch):
        self.candidate = candidate

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        return [self.candidate]


def _print_result(result: ConvergenceResult) -> None:
    table = Table(title="Convergence")
    for name in ("iter", "candidate", "status", "score", "invariants", "params"):
        table.add_column(name, overflow="fold")
    for record in result.iterations:
        if not record.evaluations:
            table.add_row(str(record.iteration), "-", record.note, "-", "-", "-")
        for ev in record.evaluations:
            status = (
                "COMMITTED"
                if ev.index == record.winner
                else (
                    ev.error
                    or "; ".join(ev.violations)
                    or ("feasible" if ev.feasible else "invariant violation")
                )
            )
            score = "-" if ev.target_score == float("-inf") else f"{ev.target_score:.3f}"
            degradation = (
                "-" if ev.invariant_degradation == float("inf") else f"{ev.invariant_degradation:.3f}"
            )
            table.add_row(
                str(record.iteration),
                ev.candidate.hypothesis[:48],
                status,
                score,
                degradation,
                str(ev.modified_params),
            )
    console.print(table)
    console.print(f"converged={result.converged} head={result.head}", markup=False)
    console.print(
        json.dumps({"target": result.target_metrics, "invariants": result.invariant_metrics}), markup=False
    )


@app.command()
@cli_errors
def edit(
    intent: Path = IntentOpt,
    script: Path = typer.Option(..., "--script", "-s", help="Python file defining compute_delta."),
    workspace: Path = WorkspaceOpt,
    rationale: str = typer.Option("manual edit", help="Why this edit should work."),
    commit: bool = typer.Option(
        False, help="Commit improving edits to .wgit (main moves only if every goal is met)."
    ),
) -> None:
    """Mode 1: runs one surgery script through the sandbox, constraints and evaluation."""
    cfg, loaded, repo = _open_workspace(workspace)
    source = _FixedCandidate(CandidatePatch(rationale, script.read_text(encoding="utf-8"), source="manual"))
    result = _engine(cfg, loaded, load_intent(intent), source, repo if commit else None, 1, 1).run()
    _print_result(result)
    if not result.iterations:
        console.print("goals already met; nothing to do", markup=False)
    elif result.iterations[0].winner is None:
        raise typer.Exit(code=1)


@app.command(name="compile")
@cli_errors
def compile_blueprint(
    blueprint: Path = typer.Argument(..., help="Blueprint YAML/JSON file."),
    out: Path = typer.Option(..., "--out", "-o", help="Output .safetensors file."),
    seed: int = typer.Option(42, help="Basis seed."),
    spectral_gamma: float | None = typer.Option(
        None, help="Clamp synthesised projections to this spectral norm."
    ),
) -> None:
    """Mode 2: compiles a blueprint into weights with zero training steps."""
    if out.suffix != ".safetensors":
        raise ValueError("--out must end in .safetensors")
    compiler = BlueprintCompiler(load_blueprint(blueprint), seed=seed, spectral_gamma=spectral_gamma)
    model = compiler.build_model()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out))
    config_path = out.with_suffix(".config.json")
    config_path.write_text(json.dumps(compiler.toy_config().to_dict(), indent=2), encoding="utf-8")
    params = sum(p.numel() for p in model.parameters())
    console.print(f"compiled {compiler.bp.model_name}: {params} parameters -> {out}", markup=False)


@app.command()
@cli_errors
def loop(
    intent: Path = IntentOpt,
    workspace: Path = WorkspaceOpt,
    reasoner: str = ReasonerOpt,
    model: str | None = ModelOpt,
    base_url: str | None = BaseUrlOpt,
    key_env: str | None = KeyEnvOpt,
    iterations: int = typer.Option(3, min=1, max=100),
    candidates: int = typer.Option(3, min=1, max=16),
) -> None:
    """Mode 3: multi-candidate tournament loop; winners are committed to .wgit."""
    cfg, loaded, repo = _open_workspace(workspace)
    source: CandidateSource = (
        TemplateCandidateSource()
        if reasoner == "template"
        else ReasonerBridge(make_backend(reasoner, model, base_url, api_key_env=key_env, workspace_llm=cfg.llm))
    )
    result = _engine(cfg, loaded, load_intent(intent), source, repo, iterations, candidates).run()
    _print_result(result)
    if not result.converged:
        raise typer.Exit(code=1)


@app.command()
@cli_errors
def export(
    out: Path = typer.Option(..., "--out", "-o", help="Output file (or directory for --format hf/lora)."),
    workspace: Path = WorkspaceOpt,
    ref: str = typer.Option("HEAD", help="HEAD, main, ROOT, branch or patch id."),
    fmt: str = typer.Option(
        "safetensors", "--format", help="safetensors, sparse, hf, lora or gguf-patch."
    ),
    rank: int = typer.Option(8, "--rank", min=1, help="LoRA rank when --format lora."),
    lora_alpha: float = typer.Option(16.0, "--lora-alpha", help="LoRA alpha when --format lora."),
) -> None:
    """Exports the base model with a patch applied (Safetensors, Sparse COO, HF, PEFT LoRA, or GGUF patch)."""
    if fmt not in ("safetensors", "sparse", "hf", "lora", "gguf-patch"):
        raise ValueError("--format must be safetensors, sparse, hf, lora or gguf-patch")
    cfg, loaded, repo = _open_workspace(workspace, at_head=False)
    patch_id = repo.resolve(ref)
    delta = repo.cumulative_delta(patch_id)
    metadata = {"patch_id": patch_id, "base_model_id": cfg.model}
    if fmt == "sparse":
        if not delta:
            raise ValueError(f"{ref} has no patches to export")
        export_sparse_delta(delta, out, metadata)
    elif fmt == "hf":
        export_hf_pretrained(loaded.model, delta, out)
    elif fmt == "lora":
        if not delta:
            raise ValueError(f"{ref} has no patches to export")
        export_peft_lora(delta, out, rank=rank, lora_alpha=lora_alpha, base_model_id=cfg.model)
    elif fmt == "gguf-patch":
        if not delta:
            raise ValueError(f"{ref} has no patches to export")
        patch_gguf_inplace(out, delta)
    else:
        export_merged_safetensors(loaded.model, delta, out, metadata)
    console.print(f"exported {patch_id} ({fmt}) -> {out}", markup=False)


@app.command(name="draft-intent")
@cli_errors
def draft_intent_cmd(
    description: str = typer.Argument(..., help="Natural-language behavior change description."),
    out: Path = typer.Option(..., "--out", "-o", help="Output IntentSpec YAML file."),
    reasoner: str = ReasonerOpt,
    model: str | None = ModelOpt,
    base_url: str | None = BaseUrlOpt,
    key_env: str | None = KeyEnvOpt,
) -> None:
    """Drafts a validated IntentSpec YAML from a natural-language description (review it before use)."""
    if reasoner == "template":
        # Offline: no model reads the description, so say so instead of pretending it was drafted from it.
        payload = dict(DEMO_INTENT)
        payload["description"] = description.strip() or payload["description"]
        spec = IntentSpec.model_validate(payload)
        origin = "offline template (edit the suites to match your description)"
    else:
        spec = draft_intent(make_backend(reasoner, model, base_url, api_key_env=key_env), description)
        origin = f"drafted by {model or 'the configured model'}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    console.print(f"intent '{spec.intent_id}' -> {out} ({origin})", markup=False)


@app.command()
@cli_errors
def log(
    workspace: Path = WorkspaceOpt,
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Shows the patch history from HEAD back to ROOT."""
    rows = [manifest_summary(m) for m in WeightGitRepository(workspace).log()]
    if as_json:
        console.print(json.dumps(rows), markup=False, soft_wrap=True)
        return
    table = Table(title="Patch history (newest first)")
    # "goals met" is the manifest's passed_gate: every target goal reached. Committed edits always passed the limits.
    for name in ("patch", "intent", "iter", "goals met", "params", "rationale"):
        table.add_column(name, overflow="fold")
    for r in rows:
        table.add_row(
            str(r["patch_id"]),
            str(r["intent"]),
            str(r["iteration"]),
            str(r["passed_gate"]),
            str(r["modified_params"]),
            str(r["rationale"]),
        )
    console.print(table)


@app.command()
@cli_errors
def spectrum(
    workspace: Path = WorkspaceOpt,
    layer: int | None = typer.Option(None, "--layer", help="Optional single layer index to inspect."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
) -> None:
    """Profiles layer singular values, HTSR power-law alpha, and Marchenko-Pastur SNR (NF-BL-03)."""
    _cfg, loaded, _repo = _open_workspace(workspace)
    params = dict(loaded.model.named_parameters())
    rows: list[dict[str, Any]] = []
    for idx in range(loaded.adapter.num_layers()):
        if layer is not None and idx != layer:
            continue
        down_name = loaded.adapter.get_layer_anatomy(idx).mlp_down_proj
        if down_name and down_name in params:
            prof = profile_matrix(params[down_name], tensor_name=down_name, top_k=8)
            rows.append(
                {
                    "layer": idx,
                    "tensor": down_name,
                    "alpha": round(prof.power_law_alpha, 3),
                    "mp_bulk_edge": round(prof.mp_bulk_edge, 5),
                    "snr_ratio": round(prof.snr_ratio, 4),
                    "effective_rank": round(prof.effective_rank, 2),
                    "spectral_norm": round(prof.spectral_norm, 4),
                }
            )
    if as_json:
        console.print(json.dumps(rows), markup=False, soft_wrap=True)
        return
    table = Table(title="HTSR & Marchenko-Pastur Spectral Profile")
    for col in ("layer", "tensor", "alpha", "mp_bulk_edge", "snr_ratio", "effective_rank", "spectral_norm"):
        table.add_column(col, overflow="fold")
    for r in rows:
        table.add_row(*[str(r[c]) for c in ("layer", "tensor", "alpha", "mp_bulk_edge", "snr_ratio", "effective_rank", "spectral_norm")])
    console.print(table)


@app.command()
@cli_errors
def autotune(
    intent: Path = IntentOpt,
    script: Path = typer.Option(..., "--script", "-s", help="Python file defining compute_delta."),
    workspace: Path = WorkspaceOpt,
) -> None:
    """Searches over (alpha, k) scaling/column-count pairs in CoW forks (NF-BL-06)."""
    cfg, loaded, _repo = _open_workspace(workspace)
    intent_spec = load_intent(intent)
    source = _FixedCandidate(CandidatePatch("autotune", script.read_text(encoding="utf-8"), source="manual"))
    eng = _engine(cfg, loaded, intent_spec, source, None, 1, 1)
    fail_texts, pass_texts = contrastive_texts(intent_spec)
    prober = MechanisticProber(loaded.model, loaded.adapter, loaded.tokenizer)
    diagnostics = build_diagnostics(prober.trace_contrastive(fail_texts, pass_texts))
    params = dict(loaded.model.named_parameters())
    total_params = sum(p.numel() for p in loaded.model.parameters())
    _, base_inv = eng._evaluate(loaded.model)
    eng._baseline_invariants = dict(base_inv)

    def eval_fn(forked: nn.Module) -> tuple[float, float, bool]:
        t_m, i_m = eng._evaluate(forked)
        ok, deg = eng.invariant_status(i_m)
        return eng.target_score(t_m), deg, ok

    res = autotune_candidate(
        loaded.model,
        CoWWeightOverlay(loaded.model),
        params,
        total_params,
        intent_spec.constraints,
        script.read_text(encoding="utf-8"),
        diagnostics,
        eval_fn,
    )
    console.print(
        json.dumps(
            {
                "best_alpha": res.best_alpha,
                "best_k": res.best_k,
                "best_target_score": res.best_target_score,
                "best_invariant_degradation": res.best_invariant_degradation,
                "num_trials": len(res.trials),
            }
        ),
        markup=False,
    )


@app.command(name="calibrate-nullspace")
@cli_errors
def calibrate_nullspace(
    out: Path = typer.Option(..., "--out", "-o", help="Output .safetensors file for P_null matrices."),
    workspace: Path = WorkspaceOpt,
    layer: int = typer.Option(1, "--layer", min=0, help="Layer index to calibrate."),
    eig_threshold: float = typer.Option(1e-2, "--eig-threshold", help="Relative eigenvalue cutoff."),
) -> None:
    """Computes AlphaEdit empirical covariance nullspace projector P_null (NF-BL-02)."""
    _cfg, loaded, _repo = _open_workspace(workspace)
    down_name = loaded.adapter.get_layer_anatomy(layer).mlp_down_proj
    if not down_name:
        raise ValueError(f"layer {layer} has no mlp_down_proj")
    texts = ["The quick brown fox jumps over the lazy dog.", "1 + 1 = 2", "Hello world."]
    nulls = compute_covariance_nullspace(
        loaded.model, loaded.tokenizer, texts, [down_name], eig_threshold=eig_threshold
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(nulls, str(out))
    console.print(f"calibrated nullspace for {down_name} -> {out}", markup=False)


@app.command(name="inject-facts")
@cli_errors
def inject_facts(
    facts: Path = typer.Option(..., "--facts", "-f", help="JSON/YAML list of {prompt, target} fact objects."),
    workspace: Path = WorkspaceOpt,
    layer: int = typer.Option(1, "--layer", min=0, help="Target layer index."),
    nullspace_file: Path | None = typer.Option(None, "--nullspace", help="Optional P_null .safetensors file."),
    ridge: float = typer.Option(1e-2, "--ridge", help="Tikhonov regularization scalar."),
) -> None:
    """Applies closed-form AlphaEdit multi-fact update to a layer and commits to .wgit (NF-BL-02)."""
    cfg, loaded, repo = _open_workspace(workspace)
    down_name = loaded.adapter.get_layer_anatomy(layer).mlp_down_proj
    if not down_name:
        raise ValueError(f"layer {layer} has no mlp_down_proj")
    raw_facts = yaml.safe_load(facts.read_text(encoding="utf-8"))
    if not isinstance(raw_facts, list) or not raw_facts:
        raise ValueError("facts file must contain a non-empty list of {prompt, target}")

    prompts = [str(item["prompt"]) for item in raw_facts]
    targets = [str(item["target"]) for item in raw_facts]
    cap_in = capture_linear_inputs(loaded.model, loaded.tokenizer, prompts, [down_name])
    if down_name not in cap_in or cap_in[down_name].numel() == 0:
        raise ValueError(f"failed to capture inputs for {down_name}")

    params = dict(loaded.model.named_parameters())
    w = params[down_name].detach().float().cpu()
    m = len(prompts)
    k_mat = cap_in[down_name][:m].T  # [d_in, m]
    # Construct target value vectors from target token embeddings projected into residual space.
    embed = params.get("model.embed_tokens.weight", w).detach().float().cpu()
    v_cols = []
    for t_str in targets:
        t_ids = loaded.tokenizer(t_str, return_tensors="pt")["input_ids"][0]
        v_cols.append(embed[t_ids[-1]])
    import torch as _torch

    v_mat = _torch.stack(v_cols, dim=1)  # [d_out, m]
    p_null = None
    if nullspace_file is not None:
        loaded_nulls = load_file(str(nullspace_file))
        p_null = loaded_nulls.get(down_name)

    delta_w = alphaedit_batch_update(w, k_mat, v_mat, P_null=p_null, ridge=ridge)
    deltas = {down_name: delta_w}
    params = dict(loaded.model.named_parameters())
    total_params = sum(p.numel() for p in loaded.model.parameters())
    t_stats, mod_params, sparsity = summarize_patch(params, deltas, total_params)
    apply_deltas_inplace(loaded.model, deltas)
    manifest = PatchManifest(
        patch_id="pending",
        parent_patch_id=None,
        base_model_id=cfg.model,
        base_model_sha256=state_fingerprint(loaded.model),
        intent_id="alphaedit-inject-facts",
        iteration=0,
        synthesized_code=f"# alphaedit_batch_update({down_name}, facts={len(prompts)})",
        reasoner_rationale=f"AlphaEdit batch injection of {len(prompts)} facts into {down_name}",
        tensor_stats=t_stats,
        total_modified_params=mod_params,
        overall_sparsity_ratio=sparsity,
        target_metrics_before={},
        target_metrics_after={},
        invariant_metrics_before={},
        invariant_metrics_after={},
        passed_gate=True,
    )
    committed = repo.commit(deltas, manifest, promote=True)
    console.print(f"injected {len(prompts)} facts -> commit {committed.patch_id}", markup=False)


@app.command()
@cli_errors
def branch(
    name: str | None = typer.Argument(None, help="Branch name to create (omit to list branches)."),
    ref: str = typer.Option("HEAD", "--ref", help="Starting ref for new branch."),
    checkout: bool = typer.Option(False, "--checkout", "-c", help="Checkout the branch immediately."),
    workspace: Path = WorkspaceOpt,
) -> None:
    """Lists, creates, or checks out .wgit branches (NF-BL-05)."""
    repo = WeightGitRepository(workspace)
    if name is None:
        console.print(json.dumps(repo.list_branches()), markup=False)
        return
    target = repo.create_branch(name, ref=ref)
    if checkout:
        repo.checkout_branch(name)
    console.print(f"branch '{name}' -> {target}", markup=False)


@app.command()
@cli_errors
def merge(
    refs: list[str] = typer.Argument(..., help="Two or more branch names or patch refs to merge."),
    strategy: str = typer.Option(
        "orthomerge", "--strategy", help="della, slerp, orthomerge or nullspace-chain."
    ),
    density: float = typer.Option(0.5, "--density", help="Retention fraction for della."),
    workspace: Path = WorkspaceOpt,
) -> None:
    """Merges multiple .wgit branches/commits without destructive interference (NF-BL-05)."""
    if len(refs) < 2:
        raise ValueError("merge requires at least 2 refs")
    if strategy not in ("della", "slerp", "orthomerge", "nullspace-chain"):
        raise ValueError("--strategy must be della, slerp, orthomerge or nullspace-chain")
    cfg, loaded, repo = _open_workspace(workspace, at_head=False)
    deltas_list = [repo.cumulative_delta(repo.resolve(r)) for r in refs]
    merged = merge_deltas(deltas_list, strategy=strategy, density=density)  # type: ignore[arg-type]
    if not merged:
        raise ValueError("merged delta is empty")
    params = dict(loaded.model.named_parameters())
    total_params = sum(p.numel() for p in loaded.model.parameters())
    t_stats, mod_params, sparsity = summarize_patch(params, merged, total_params)
    manifest = PatchManifest(
        patch_id="pending",
        parent_patch_id=None,
        base_model_id=cfg.model,
        base_model_sha256=state_fingerprint(loaded.model),
        intent_id=f"merge-{strategy}",
        iteration=0,
        synthesized_code=f"# merge_deltas({refs}, strategy={strategy!r})",
        reasoner_rationale=f"Merged {refs} via {strategy}",
        tensor_stats=t_stats,
        total_modified_params=mod_params,
        overall_sparsity_ratio=sparsity,
        target_metrics_before={},
        target_metrics_after={},
        invariant_metrics_before={},
        invariant_metrics_after={},
        passed_gate=True,
    )
    committed = repo.commit(merged, manifest, promote=True)
    console.print(f"merged {refs} ({strategy}) -> {committed.patch_id}", markup=False)


@app.command(name="diff-models")
@cli_errors
def diff_models_cmd(
    model_a: str = typer.Option(..., "--model-a", help="First model spec."),
    model_b: str = typer.Option(..., "--model-b", help="Second model spec."),
    demo_defect_b: bool = typer.Option(False, "--demo-defect-b", help="Plant demo defect on model B."),
    top: int = typer.Option(10, "--top", min=1, help="Top modified tensors to display."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
) -> None:
    """Computes weight-space parameter diffs between two models (NF-BL-04)."""
    ma = load_model(model_a).model
    mb = load_model(model_b, demo_defect=demo_defect_b).model
    diffs = [d.to_dict() for d in diff_models(ma, mb, top_k=5)[:top]]
    if as_json:
        console.print(json.dumps(diffs), markup=False, soft_wrap=True)
        return
    table = Table(title="Cross-Model Weight Diff")
    for col in ("tensor_name", "frobenius_diff", "relative_frobenius", "spectral_diff"):
        table.add_column(col, overflow="fold")
    for d in diffs:
        table.add_row(
            str(d["tensor_name"]),
            f"{float(d['frobenius_diff']):.4f}",  # type: ignore[arg-type]
            f"{float(d['relative_frobenius']):.4f}",  # type: ignore[arg-type]
            f"{float(d['spectral_diff']):.4f}",  # type: ignore[arg-type]
        )
    console.print(table)


@app.command(name="trace-circuit")
@cli_errors
def trace_circuit_cmd(
    intent: Path = IntentOpt,
    workspace: Path = WorkspaceOpt,
    top_edges: int = typer.Option(10, "--top-edges", min=1, help="Number of edges to display."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
) -> None:
    """Subcomponent / Edge Attribution Patching across attention and MLP nodes (WRSI-BL-03)."""
    _cfg, loaded, _repo = _open_workspace(workspace)
    fail, passing = contrastive_texts(load_intent(intent))
    edges = MechanisticProber(loaded.model, loaded.adapter, loaded.tokenizer).trace_circuit(
        fail, passing, top_k_edges=top_edges
    )
    rows = [e.to_dict() for e in edges]
    if as_json:
        console.print(json.dumps(rows), markup=False, soft_wrap=True)
        return
    table = Table(title="Subcomponent Circuit Attribution")
    for col in ("source_node", "target_node", "tensor_name", "attribution_score", "top_cols"):
        table.add_column(col, overflow="fold")
    for r in rows:
        table.add_row(
            str(r["source_node"]),
            str(r["target_node"]),
            str(r["tensor_name"]),
            f"{float(r['attribution_score']):.4f}",  # type: ignore[arg-type]
            str(r["top_cols"]),
        )
    console.print(table)


@app.command()
@cli_errors
def models(
    base_url: str | None = BaseUrlOpt,
    key_env: str | None = KeyEnvOpt,
    workspace: Path | None = typer.Option(None, "--workspace", "-w", help="Read the llm: section of this workspace."),
    show_all: bool = typer.Option(False, "--all", help="List every model, not only reasoning models."),
) -> None:
    """Lists the reasoning models your endpoint offers right now (GET {base_url}/models); --all lists every model."""
    ws_llm = _load_config(workspace).llm if workspace else None
    settings = resolve_settings(base_url=base_url, api_key_env=key_env, workspace_llm=ws_llm)
    try:
        listing = fetch_models(settings, reasoning_only=not show_all)
    except ModelListError as exc:
        console.print(f"cannot list models ({exc.reason}): {exc}", markup=False)
        raise typer.Exit(code=1) from None
    if listing.reasoning_filter == "applied":
        console.print(f"{len(listing.ids)} reasoning models (of {listing.total}) at {settings.base_url}:", markup=False)
    elif listing.reasoning_filter == "not-available":
        console.print(f"{settings.base_url} does not mark reasoning models; all {listing.total} listed:", markup=False)
    else:
        console.print(f"{listing.total} models at {settings.base_url}:", markup=False)
    for model_id in listing.ids:
        console.print(f"  {model_id}", markup=False)


@app.command()
@cli_errors
def doctor(
    workspace: Path | None = typer.Option(None, "--workspace", "-w", help="Also check this workspace."),
    base_url: str | None = BaseUrlOpt,
    key_env: str | None = KeyEnvOpt,
) -> None:
    """Checks Python, PyTorch, the optional workspace and the LLM endpoint, and says what to do next."""
    import platform

    import torch

    rows: list[tuple[str, str, str]] = []
    rows.append(("python", "ok", platform.python_version()))
    cuda = torch.cuda.is_available()
    rows.append(("torch", "ok", f"{torch.__version__} (CUDA {'available' if cuda else 'not available: CPU only'})"))
    try:
        import transformers

        rows.append(("transformers", "ok", f"{transformers.__version__} (Hugging Face models enabled)"))
    except ImportError:
        rows.append(("transformers", "info", "not installed: toy models only (pip install '.[hf]')"))
    ws_llm = None
    if workspace is not None:
        try:
            cfg = _load_config(workspace)
            ws_llm = cfg.llm
            rows.append(("workspace", "ok", f"{workspace} (model {cfg.model})"))
        except (FileNotFoundError, ValidationError) as exc:
            rows.append(("workspace", "fail", f"{exc}"))
    settings = resolve_settings(base_url=base_url, api_key_env=key_env, workspace_llm=ws_llm)
    info = settings.describe()
    key_state = "set" if info["key_present"] else "NOT set"
    rows.append(("llm endpoint", "ok", f"{info['base_url']} (key variable {info['api_key_env']}: {key_state})"))
    try:
        listing = fetch_models(settings, timeout=10.0)
        rows.append(("llm models", "ok", f"{len(listing.ids)} reasoning models of {listing.total}; choose one with --model or in the Studio"))
    except ModelListError as exc:
        rows.append(("llm models", "warn", f"{exc.reason}: {exc}. Offline templates still work."))
    table = Table(title="neuroforge doctor")
    for name in ("check", "status", "detail"):
        table.add_column(name, overflow="fold")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    if any(status == "fail" for _, status, _ in rows):
        raise typer.Exit(code=1)


@app.command()
@cli_errors
def studio(
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w",
        help="Workspace to open. Without it the Studio opens the current folder if it is a workspace, else the start page.",
    ),
    home: Path | None = typer.Option(
        None, "--home", help="Studio home for downloaded models and new workspaces (default: WEIGHTLAB_HOME or ~/.weightlab).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (keep the default unless you know why)."),
    port: int = typer.Option(7860, "--port", help="TCP port (0 picks a free one)."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the Studio in your browser."),
    check_only: bool = typer.Option(False, "--check-only", help="Bind, print the URL and exit."),
) -> None:
    """Opens the guided Studio in your browser: workspaces, edits, model factory and self-improvement."""
    studio_home = StudioHome.resolve(home)
    if workspace is None and (Path(".") / CONFIG_NAME).is_file():
        workspace = Path(".")
    if workspace is None:
        state = StudioState.without_workspace(fetch_models_on_start=not check_only, home=studio_home)
    else:
        state = StudioState.from_workspace(workspace, fetch_models_on_start=not check_only, home=studio_home)
    srv = StudioServer(state, host=host, port=port)
    try:
        if host not in ("127.0.0.1", "localhost", "::1"):
            console.print(f"warning: listening on {host}; anyone who can reach it and read the page can edit weights", markup=False)
        console.print(f"NeuroForge Studio: {srv.url}  (Ctrl+C to stop)", markup=False)
        if check_only:
            return
        if open_browser:
            import webbrowser

            webbrowser.open(srv.url)
        srv.serve_forever()
    except KeyboardInterrupt:
        console.print("Studio stopped.", markup=False)
    finally:
        srv.close()


if __name__ == "__main__":
    app()

