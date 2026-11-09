"""Workspace operations shared by the CLI and the Studio: open, probe, repair loop, history, checkout, export.

Every function works on a workspace directory created by ``neuroforge init`` and returns plain data (dicts and
lists of JSON-safe values) so the Studio can serve results without re-implementing engine logic. Results contain
only measured values; nothing here invents or rounds-up metrics.

Main entry points: ``load_config``, ``load_workspace_model``, ``open_workspace``, ``build_engine``, ``probe_report``, ``convergence_summary``,
``history``, ``checkout``, ``export_patch``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from libweightspace.candidates import CandidateSource
from libweightspace.loading import LoadedModel, load_model
from libweightspace.prober import MechanisticProber, model_logits

from .convergence_engine import ConvergenceEngine, ConvergenceResult
from .exporter import (
    export_hf_pretrained,
    export_merged_safetensors,
    export_peft_lora,
    export_sparse_delta,
)
from .intent_compiler import compile_suites, contrastive_texts
from .patch_stats import state_fingerprint
from .schemas import IntentSpec, WorkspaceConfig
from .weight_git import ROOT, WeightGitRepository, apply_deltas_inplace, manifest_summary

CONFIG_NAME = "neuroforge.yaml"
EXPORT_FORMATS = ("safetensors", "sparse", "hf", "lora")


def load_config(workspace: Path) -> WorkspaceConfig:
    """Read and validate ``neuroforge.yaml``.

    Raises:
        FileNotFoundError: the workspace has no config (run ``neuroforge init``).
    """
    path = workspace / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {CONFIG_NAME} in {workspace} (run `neuroforge init`)")
    return WorkspaceConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_workspace_model(cfg: WorkspaceConfig) -> LoadedModel:
    """The workspace's base model, loaded from its local copy when it has one and labelled with its public id."""
    loaded = load_model(cfg.load_spec, cfg.device, cfg.demo_defect)
    loaded.model_id = cfg.model  # manifests and bundles name the public id, never a local folder
    return loaded


def open_workspace(workspace: Path, at_head: bool = True) -> tuple[WorkspaceConfig, LoadedModel, WeightGitRepository]:
    """Load config, repository and the model (with HEAD's cumulative delta unless ``at_head`` is False)."""
    cfg = load_config(workspace)
    repo = WeightGitRepository(workspace)
    loaded = load_workspace_model(cfg)
    repo.check_base(state_fingerprint(loaded.model))
    if at_head and repo.head() != ROOT:
        apply_deltas_inplace(loaded.model, repo.cumulative_delta(repo.head()))
    return cfg, loaded, repo


def reference_models(cfg: WorkspaceConfig, base: nn.Module) -> tuple[nn.Module, nn.Module | None]:
    """``(KL reference, nullspace reference)``: the configured known-good model, else the unedited base."""
    if cfg.reference_model:
        ref = load_model(cfg.reference_model, cfg.device).model
        return ref, ref
    return base, None


def build_engine(
    cfg: WorkspaceConfig,
    loaded: LoadedModel,
    intent_spec: IntentSpec,
    source: CandidateSource,
    repo: WeightGitRepository | None,
    iterations: int,
    candidates: int,
    **hooks: Any,
) -> ConvergenceEngine:
    """Convergence engine whose invariants are measured against the unedited base model.

    ``hooks`` are passed to ``ConvergenceEngine`` (progress, should_cancel, protected_tensors, approve, commit).
    """
    base = load_workspace_model(cfg).model  # invariants span all runs, not just this one
    kl_reference, nullspace_reference = reference_models(cfg, base)
    targets, invariants = compile_suites(intent_spec, loaded.tokenizer, kl_reference)
    return ConvergenceEngine(
        loaded.model,
        loaded.adapter,
        loaded.tokenizer,
        intent_spec,
        targets,
        invariants,
        source,
        repo=repo,
        base_model_id=loaded.model_id,
        max_iterations=iterations,
        candidates_per_iteration=candidates,
        nullspace_reference=nullspace_reference,
        baseline_model=base,
        **hooks,
    )


def probe_report(loaded: LoadedModel, intent_spec: IntentSpec, top: int = 5, columns: int = 8) -> list[dict[str, Any]]:
    """Contrastive tracing: layers ranked by how strongly they write the failure direction."""
    fail, passing = contrastive_texts(intent_spec)
    reports = MechanisticProber(loaded.model, loaded.adapter, loaded.tokenizer).trace_contrastive(fail, passing)
    return [
        {
            "layer": r.layer_idx,
            "divergence": round(float(r.divergence_score), 6),
            "tensor": r.mlp_down_proj,
            "culprit_columns": [int(c) for c in r.top_culpable_mlp_cols[:columns]],
            "column_alignment": [round(float(a), 4) for a in r.top_mlp_col_alignments[:columns]],
        }
        for r in reports[:top]
    ]


def _finite(value: float) -> float | None:
    """JSON-safe number: None for inf/nan (e.g. an unevaluated candidate)."""
    return None if not math.isfinite(value) else round(float(value), 6)


def convergence_summary(result: ConvergenceResult) -> dict[str, Any]:
    """Every candidate of every iteration with its measured score, degradation, size and gate outcome."""
    rows: list[dict[str, Any]] = []
    for record in result.iterations:
        if not record.evaluations:
            rows.append({"iteration": record.iteration, "candidate": None, "status": record.note})
        for ev in record.evaluations:
            if ev.index == record.winner:
                status = "committed"
            elif ev.error:
                status = f"error: {ev.error}"
            elif ev.violations:
                status = "rejected: " + "; ".join(ev.violations)
            else:
                status = "feasible (not best)" if ev.feasible else "rejected: invariant degradation"
            rows.append(
                {
                    "iteration": record.iteration,
                    "candidate": ev.candidate.hypothesis[:160],
                    "source": ev.candidate.source,
                    "status": status,
                    "target_score": _finite(ev.target_score),
                    "invariant_degradation": _finite(ev.invariant_degradation),
                    "modified_params": int(ev.modified_params),
                    "patch_id": record.committed_patch_id if ev.index == record.winner else None,
                }
            )
    return {
        "converged": bool(result.converged),
        "head": result.head,
        "target_metrics": {k: _finite(v) for k, v in result.target_metrics.items()},
        "invariant_metrics": {k: _finite(v) for k, v in result.invariant_metrics.items()},
        "rows": rows,
    }


def history(repo: WeightGitRepository) -> list[dict[str, Any]]:
    """Patch history from HEAD back to ROOT (newest first), JSON-safe."""
    return [{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
             for k, v in manifest_summary(m).items()} for m in repo.log()]


def checkout(workspace: Path, patch_id: str) -> tuple[str, LoadedModel]:
    """Move HEAD to ``patch_id`` (or ``ROOT``) and return a freshly loaded model at that state (bit-exact)."""
    repo = WeightGitRepository(workspace)
    resolved = repo.resolve(patch_id)
    repo.checkout(resolved)
    _cfg, loaded, _repo = open_workspace(workspace)
    return resolved, loaded


def export_patch(
    workspace: Path,
    fmt: str,
    out: Path,
    ref: str = "HEAD",
    rank: int = 8,
    lora_alpha: float = 16.0,
) -> dict[str, Any]:
    """Export the base model with ``ref`` applied (or just the delta for ``sparse``/``lora``).

    Raises:
        ValueError: unknown format, or a delta-only format requested for a state without patches.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}")
    cfg, loaded, repo = open_workspace(workspace, at_head=False)
    patch_id = repo.resolve(ref)
    delta = repo.cumulative_delta(patch_id)
    metadata = {"patch_id": patch_id, "base_model_id": cfg.model}
    if fmt in ("sparse", "lora") and not delta:
        raise ValueError(f"{ref} has no patches to export")
    if fmt == "sparse":
        export_sparse_delta(delta, out, metadata)
    elif fmt == "hf":
        export_hf_pretrained(loaded.model, delta, out)
    elif fmt == "lora":
        export_peft_lora(delta, out, rank=rank, lora_alpha=lora_alpha, base_model_id=cfg.model)
    else:
        export_merged_safetensors(loaded.model, delta, out, metadata)
    return {"patch_id": patch_id, "format": fmt, "path": str(out)}


# --- CR3 / CR4: answers before and after ---------------------------------------------------------------------


def intent_prompts(intent_spec: IntentSpec) -> list[dict[str, Any]]:
    """The intent's prompts with their role: ``failing`` (should change) or ``protected`` (must not change)."""
    rows: list[dict[str, Any]] = []
    for probe in intent_spec.contrastive_probes:
        role = "protected" if probe.metadata.get("role") == "invariant" else "failing"
        rows.append({"probe_id": probe.probe_id, "prompt": probe.prompt, "role": role,
                     "expected": probe.expected_completion_prefix, "undesired": probe.undesired_completion_prefix or ""})
    return rows


@torch.no_grad()
def generate_answers(model: nn.Module, tokenizer: Any, prompts: list[str], max_new_tokens: int = 24) -> list[str]:
    """Greedy completions (deterministic: no sampling), so before/after answers are directly comparable."""
    device = next(model.parameters()).device
    answers: list[str] = []
    for prompt in prompts:
        cur = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = model_logits(model(input_ids=cur))
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=cur.dtype, device=device)], dim=1)
        answers.append(str(tokenizer.decode(generated)))
    return answers


def compare_answers(cfg: WorkspaceConfig, loaded: LoadedModel, intent_spec: IntentSpec, max_new_tokens: int = 24) -> dict[str, Any]:
    """Answers of the original model and of the current state for every intent prompt, generated now."""
    rows = intent_prompts(intent_spec)
    prompts = [r["prompt"] for r in rows]
    base = load_workspace_model(cfg).model
    before = generate_answers(base, loaded.tokenizer, prompts, max_new_tokens)
    after = generate_answers(loaded.model, loaded.tokenizer, prompts, max_new_tokens)
    for row, b, a in zip(rows, before, after, strict=True):
        row.update({"original": b, "current": a, "changed": b != a})
    return {"settings": {"decoding": "greedy", "max_new_tokens": max_new_tokens}, "rows": rows}


# --- CR8: edit bundle ------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(workspace: Path, out_dir: Path) -> dict[str, Any]:
    """Write an edit bundle: sparse patch, manifest history, intent, model card and checksums.

    Raises:
        ValueError: HEAD is the original model (nothing to bundle).
    """
    cfg, loaded, repo = open_workspace(workspace, at_head=False)
    head = repo.head()
    if head == ROOT:
        raise ValueError("the model is at its original state: there is no edit to bundle")
    out_dir.mkdir(parents=True, exist_ok=False)
    delta = repo.cumulative_delta(head)
    export_sparse_delta(delta, out_dir / "patch.sparse.safetensors", {"patch_id": head, "base_model_id": cfg.model})
    commits = history(repo)
    (out_dir / "manifests.json").write_text(json.dumps(commits, indent=2), encoding="utf-8")
    intent_file = workspace / "intent.yaml"
    if intent_file.is_file():
        (out_dir / "intent.yaml").write_text(intent_file.read_text(encoding="utf-8"), encoding="utf-8")
    fingerprint = state_fingerprint(loaded.model)
    changed = sum(int(c.get("modified_params") or 0) for c in commits)
    card = [
        f"# Edit bundle for {cfg.model}",
        "",
        f"- Base model: `{cfg.model}` (weights fingerprint `{fingerprint[:16]}...`)",
        f"- Head patch: `{head}`; {len(commits)} committed edit(s); {changed:,} weights targeted in total "
        "(nullspace protection may add a small correction to other entries of the same tensors)",
        "- Every metric in manifests.json was measured when the edit was evaluated.",
        "",
        "## Edits (newest first)",
        "",
    ] + [f"- `{c['patch_id'][:12]}` intent `{c['intent']}`: {c['rationale']}" for c in commits] + [
        "",
        "## Limits",
        "",
        "Edits were evaluated on the prompts in intent.yaml. Check behaviour you care about before relying on it.",
        "Apply this bundle only to the same base model; `checksums.txt` lists SHA-256 hashes of every file.",
    ]
    (out_dir / "MODEL_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    files = sorted(p for p in out_dir.iterdir() if p.is_file())
    (out_dir / "checksums.txt").write_text(
        "".join(f"{_sha256_file(p)}  {p.name}\n" for p in files) + f"base-fingerprint  {fingerprint}\n", encoding="utf-8"
    )
    return {"patch_id": head, "format": "bundle", "files": [p.name for p in files] + ["checksums.txt"]}
