"""Edit modes beyond repair (CR6): fact batches and behaviour directions, each previewed with measured results.

Fact mode (closed-form, AlphaEdit-style, ``plan_fact_edit``):
    1. For each fact, the key ``k`` is the input of one layer's down projection at the last prompt token.
    2. The value ``v = W k + delta`` is found by optimising only ``delta`` (an activation, never a weight) so that the
       model, with ``delta`` added to that output, predicts the new object (teacher forcing, norm-clamped).
    3. The update is restricted to the budgeted columns with the highest key energy, so the structural edit stays
       within the weight budget, and it is projected into the nullspace of protected inputs (neighbour and general
       prompts) inside those columns: ``Delta[:, S] = alphaedit(W[:, S], K[S], V, P_S)``.
    4. Success, paraphrase and neighbourhood results, KL on general prompts and the cumulative norm drift of the
       tensor are measured on a copy-on-write fork before anything is applied.

Direction mode (``analyse_direction`` and ``plan_direction_edit``):
    1. For two prompt sets that differ only in a behaviour, the mean difference of what each layer's down projection
       writes at the last token gives a direction per layer; its strength is the separation in standard deviations.
    2. The edit removes (strength > 0) or amplifies (strength < 0) that direction in the top-k columns most aligned
       with it, then projects into the nullspace of general-prompt inputs. KL to the original model on general
       prompts and the measured shift along the direction are reported before commit.

Both modes produce ordinary ``.wgit`` patches (one patch per batch), so restoring the parent is bit-exact.
Main entry points: ``FactBatch``, ``parse_facts_csv``, ``plan_fact_edit``, ``DirectionSpec``, ``analyse_direction``,
``plan_direction_edit``, ``EditPlan``, ``commit_plan``, ``draft_fact_probes``, ``draft_direction_sets``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator, model_validator

from libweightspace.alphaedit import alphaedit_batch_update
from libweightspace.candidates import extract_json
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.llm_backends import LLMBackend
from libweightspace.loading import LoadedModel
from libweightspace.prober import capture_linear_inputs, model_device, model_logits

from .cancellation import OperationCancelled
from .convergence_engine import check_constraints
from .metrics import encode_continuation, greedy_continuation
from .patch_stats import state_fingerprint, summarize_patch
from .schemas import EditConstraints, PatchManifest
from .weight_git import WeightGitRepository, apply_deltas_inplace

MAX_FACTS = 32
MAX_PROBES_PER_FACT = 8
MAX_DIRECTION_PROMPTS = 64
MAX_TEXT = 400
DEFAULT_GENERAL_PROMPTS = (
    "The weather today is",
    "In mathematics, a prime number is",
    "def add(a, b):",
    "Once upon a time there was",
    "The quick brown fox",
    "Water boils at",
)
_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
Progress = Callable[[str, float], None]


# --- schemas -------------------------------------------------------------------------------------------------


def _clean_list(values: list[str], limit: int, what: str) -> list[str]:
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if len(cleaned) > limit or any(len(v) > MAX_TEXT for v in cleaned):
        raise ValueError(f"at most {limit} {what} of at most {MAX_TEXT} characters each")
    return cleaned


class Fact(BaseModel):
    """One fact to write: a prompt that should now be completed by ``new_object``."""

    prompt: str = Field(min_length=1, max_length=MAX_TEXT)
    new_object: str = Field(min_length=1, max_length=200)
    old_object: str | None = Field(default=None, max_length=200)
    # The subject the fact is about (for example "France" in "The capital of France is"). When given, the edit is keyed
    # on its last token, as in ROME, so other subjects are left alone and rewordings that mention the subject can change
    # too. That last part is not guaranteed: on LFM 3 1.7B the rewordings tested did not change; the preview measures it.
    subject: str | None = Field(default=None, max_length=200)
    paraphrases: list[str] = Field(default_factory=list)
    neighbours: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _subject_in_prompt(self) -> Fact:
        if self.subject is not None and self.subject.strip() and self.subject not in self.prompt:
            raise ValueError(f"the subject {self.subject!r} must appear in the prompt")
        if self.subject is not None and not self.subject.strip():
            self.subject = None
        return self

    @field_validator("paraphrases")
    @classmethod
    def _paraphrases(cls, v: list[str]) -> list[str]:
        return _clean_list(v, MAX_PROBES_PER_FACT, "paraphrases")

    @field_validator("neighbours")
    @classmethod
    def _neighbours(cls, v: list[str]) -> list[str]:
        return _clean_list(v, MAX_PROBES_PER_FACT, "neighbour prompts")


class FactBatch(BaseModel):
    """A batch of facts applied as one patch."""

    batch_id: str = Field(pattern=_ID_RE)
    facts: list[Fact] = Field(min_length=1, max_length=MAX_FACTS)
    layers: list[int] | None = Field(default=None, max_length=8)  # None: four layers ending at the middle
    general_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_GENERAL_PROMPTS))
    ridge: float = Field(default=1e-2, gt=0.0, le=10.0)
    steps: int = Field(default=30, ge=1, le=200)
    margin: float = Field(default=1.0, ge=0.05, le=5.0)  # nats the new object must lead by at each target position
    # Real-model defaults: the repair loop's weight budget (0.05% of the weights) with norm limits that allow a fact.
    constraints: EditConstraints = Field(default_factory=lambda: EditConstraints(
        max_modified_param_ratio=0.0005, max_frobenius_rel_norm=0.10, max_spectral_norm_shift=0.10))

    @field_validator("layers")
    @classmethod
    def _layers(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and (not v or any(i < 0 for i in v)):
            raise ValueError("layers must be a non-empty list of layer numbers")
        return v

    @field_validator("general_prompts")
    @classmethod
    def _general(cls, v: list[str]) -> list[str]:
        return _clean_list(v, 32, "general prompts")


class DirectionSpec(BaseModel):
    """Two prompt sets that differ only in one behaviour (``positives`` show it, ``negatives`` do not)."""

    name: str = Field(pattern=_ID_RE)
    positives: list[str] = Field(min_length=2)
    negatives: list[str] = Field(min_length=2)
    general_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_GENERAL_PROMPTS))
    layer: int | None = Field(default=None, ge=0)
    strength: float = Field(default=1.0, ge=-1.0, le=1.0)
    constraints: EditConstraints = Field(default_factory=lambda: EditConstraints(
        max_modified_param_ratio=0.0005, max_frobenius_rel_norm=0.10, max_spectral_norm_shift=0.10))

    @field_validator("positives", "negatives")
    @classmethod
    def _sets(cls, v: list[str]) -> list[str]:
        return _clean_list(v, MAX_DIRECTION_PROMPTS, "prompts per set")

    @field_validator("general_prompts")
    @classmethod
    def _general(cls, v: list[str]) -> list[str]:
        return _clean_list(v, 32, "general prompts")


def parse_facts_csv(text: str) -> list[Fact]:
    """Facts from CSV text with a header ``prompt,new_object[,old_object,paraphrases,neighbours]``.

    Paraphrases and neighbours hold several prompts separated by ``|``.

    Raises:
        ValueError: missing columns, too many rows, or an invalid row (the message names the row).
    """
    reader = csv.DictReader(io.StringIO(text.strip()))
    header = {h.strip() for h in (reader.fieldnames or [])}
    if not {"prompt", "new_object"} <= header:
        raise ValueError("CSV needs a header with at least: prompt,new_object")
    facts: list[Fact] = []
    for i, row in enumerate(reader, start=2):
        if len(facts) >= MAX_FACTS:
            raise ValueError(f"at most {MAX_FACTS} facts per batch")
        clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        # Answers keep their leading space: real tokenizers start the answer word with it (" Abuja", not "Abuja").
        answers = {k.strip(): (v or "").rstrip() for k, v in row.items() if k and k.strip() in ("new_object", "old_object")}
        try:
            facts.append(Fact(
                prompt=clean["prompt"], new_object=answers.get("new_object", ""),
                old_object=answers.get("old_object") or None,
                subject=clean.get("subject") or None,
                paraphrases=[p for p in clean.get("paraphrases", "").split("|") if p.strip()],
                neighbours=[p for p in clean.get("neighbours", "").split("|") if p.strip()],
            ))
        except ValueError as exc:
            raise ValueError(f"row {i}: {str(exc).splitlines()[0]}") from None
    if not facts:
        raise ValueError("the CSV has no fact rows")
    return facts


# --- shared helpers ------------------------------------------------------------------------------------------


@dataclass
class EditPlan:
    """A previewed edit: the deltas and everything measured about them, ready to commit unchanged."""

    plan_id: str
    mode: str
    head: str
    deltas: dict[str, torch.Tensor] = field(repr=False)
    structural: dict[str, torch.Tensor] = field(repr=False)
    report: dict[str, Any]
    manifest_fields: dict[str, Any]
    # The edit history the preview was measured against: two fresh workspaces share the head "ROOT", so the head
    # alone cannot stop a preview from one workspace being committed to another.
    workspace: str


def _workspace_id(repo: WeightGitRepository) -> str:
    """Identity of the workspace an edit history belongs to (its resolved history folder)."""
    return str(repo.root.resolve())


def _down_proj(loaded: LoadedModel, layer: int) -> tuple[str, nn.Linear]:
    """Parameter name and module of ``layer``'s down projection."""
    if not 0 <= layer < loaded.adapter.num_layers():
        raise ValueError(f"layer must be between 0 and {loaded.adapter.num_layers() - 1}")
    name = loaded.adapter.get_layer_anatomy(layer).mlp_down_proj
    if not name:
        raise ValueError(f"layer {layer} has no down projection to edit (for example a mixture-of-experts layer)")
    module = loaded.model.get_submodule(name[: -len(".weight")])
    if not isinstance(module, nn.Linear):
        raise ValueError(f"{name} is not a linear layer")
    return name, module


def editable_layers(loaded: LoadedModel) -> list[int]:
    """Layers with a dense down projection."""
    out = []
    for idx in range(loaded.adapter.num_layers()):
        try:
            _down_proj(loaded, idx)
            out.append(idx)
        except ValueError:
            continue
    return out


def _budget_columns(constraints: EditConstraints, total_params: int, d_out: int) -> int:
    weights = int(constraints.max_modified_param_ratio * total_params)
    return weights // d_out


def _ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def _answer(model: nn.Module, tokenizer: Any, prompt: str, n: int) -> list[int]:
    return greedy_continuation(model, tokenizer, prompt, n)


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(ids)))


@torch.no_grad()
def _log_probs(model: nn.Module, tokenizer: Any, prompts: Sequence[str]) -> list[torch.Tensor]:
    """Next-token log-probabilities at every position of every prompt (captured before an edit is applied).

    The copy-on-write overlay edits the model in place, so the reference must be captured first.
    """
    device = model_device(model)
    return [F.log_softmax(model_logits(model(input_ids=_ids(tokenizer, p, device)))[0].float(), dim=-1).cpu()
            for p in prompts]


@torch.no_grad()
def _mean_kl(reference: Sequence[torch.Tensor], model: nn.Module, tokenizer: Any, prompts: Sequence[str]) -> float:
    """Mean over prompts of the per-position KL(P_reference || P_model)."""
    if not prompts:
        return 0.0
    current = _log_probs(model, tokenizer, prompts)
    total = sum(float(F.kl_div(c, r, reduction="batchmean", log_target=True)) for c, r in zip(current, reference, strict=True))
    return total / len(prompts)


def _drift(repo: WeightGitRepository, params: Mapping[str, torch.Tensor], name: str, delta: torch.Tensor) -> dict[str, Any]:
    """Relative Frobenius distance of ``name`` from the original model, before and after ``delta``."""
    current = params[name].detach().float()
    cumulative = repo.cumulative_delta(repo.head()).get(name)
    original = current - (cumulative.float().to(current.device) if cumulative is not None else 0.0)
    base_norm = float(torch.linalg.vector_norm(original)) + 1e-12
    before = float(torch.linalg.vector_norm(current - original)) / base_norm
    after = float(torch.linalg.vector_norm(current + delta.float().to(current.device) - original)) / base_norm
    return {"tensor": name, "before": round(before, 6), "after": round(after, 6)}


def commit_plan(plan: EditPlan, loaded: LoadedModel, repo: WeightGitRepository, base_model_id: str) -> str:
    """Apply a previewed plan to the model and commit it as one patch; returns the patch id.

    Raises:
        ValueError: the preview belongs to another workspace, the model moved since the preview, or the plan broke a
            constraint.
    """
    if plan.workspace != _workspace_id(repo):
        raise ValueError("this preview was measured on another workspace; preview again in this one")
    if repo.head() != plan.head:
        raise ValueError("the model changed since this preview; preview again before applying")
    if plan.report.get("violations"):
        raise ValueError("this edit breaks the limits shown in the preview; adjust the budget or the inputs")
    params = dict(loaded.model.named_parameters())
    total = sum(p.numel() for p in loaded.model.parameters())
    stats, modified, sparsity = summarize_patch(params, plan.deltas, total, structural_deltas=plan.structural)
    base_sha = state_fingerprint(loaded.model)
    apply_deltas_inplace(loaded.model, plan.deltas)
    fields = plan.manifest_fields
    manifest = PatchManifest(
        patch_id="pending", parent_patch_id=None, base_model_id=base_model_id, base_model_sha256=base_sha,
        intent_id=fields["intent_id"], iteration=0, synthesized_code=fields["description"],
        reasoner_rationale=fields["rationale"], tensor_stats=stats, total_modified_params=modified,
        overall_sparsity_ratio=sparsity, target_metrics_before=fields["target_before"],
        target_metrics_after=fields["target_after"], invariant_metrics_before=fields["invariant_before"],
        invariant_metrics_after=fields["invariant_after"], passed_gate=True,
    )
    return repo.commit(plan.deltas, manifest).patch_id


# --- fact mode -----------------------------------------------------------------------------------------------


def default_fact_layers(loaded: LoadedModel) -> list[int]:
    """Edit window: four editable layers starting at about one sixth of the depth (one middle layer for small models).

    Locate-then-edit methods store facts in early-middle MLP layers (MEMIT edits layers 3-8 of GPT-J's 28). On the
    local LFM 3 1.7B check (24 layers), layers 4-7 wrote two facts within the default limits while layers 9-12 needed
    edits above them. Spreading the change over several layers divides each layer's share of it.
    """
    layers = editable_layers(loaded)
    if not layers:
        raise ValueError("this model has no layer with a dense down projection to edit")
    if len(layers) < 8:
        return [layers[len(layers) // 2]]
    start = max(0, round(len(layers) / 6))
    return layers[start: start + 4]


def _keys(loaded: LoadedModel, name: str, prompt: str) -> torch.Tensor:
    """Inputs ``[n_tokens, d_in]`` of the module owning ``name`` at every position of ``prompt``."""
    return capture_linear_inputs(loaded.model, loaded.tokenizer, [prompt], [name], max_rows=10_000)[name]


CLAMP_FACTORS = (4.0, 16.0, 64.0)
# Protected prompts are a handful of explicit texts, not a corpus: their whole span is protected (only numerically
# zero directions count as free), unlike AlphaEdit's 1% cut-off over a large covariance.
EXACT_PROTECTION = 1e-6
# By default the search stops once the new object wins greedy decoding at every target position by 1 nat (about 2.7x
# more likely than the runner-up). Pushing further only makes the weight change larger.
DEFAULT_MARGIN = 1.0


def key_position(tokenizer: Any, prompt: str, subject: str | None) -> int:
    """Token index the edit is keyed on: the subject's last token when a subject is given, else the prompt's last."""
    n_prompt = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    if not subject:
        return n_prompt - 1
    end = prompt.index(subject) + len(subject)
    return min(n_prompt, len(tokenizer(prompt[:end], return_tensors="pt")["input_ids"][0])) - 1


def _optimise_value(
    loaded: LoadedModel, module: nn.Linear, prompt: str, target_ids: list[int], steps: int,
    target_margin: float = DEFAULT_MARGIN, key_pos: int | None = None,
) -> tuple[torch.Tensor, float, float, float]:
    """``delta`` for the down projection's output at ``key_pos`` (default: the last prompt token).

    The target is predicted from the last prompt token; when ``key_pos`` is the subject's token, the change reaches
    the prediction through the later layers' attention, as in ROME.

    Only the activation offset is optimised; every weight keeps ``requires_grad=False`` during the search. The norm of
    ``delta`` is clamped to a multiple of the original output's norm, starting at 4x (as in ROME) and raised to 16x
    and 64x only while the target is not reached; the resulting weight change is then judged by the edit limits.

    Returns:
        ``(delta, loss before, loss after, clamp factor used)``.
    """
    model = loaded.model
    device = model_device(model)
    prompt_ids = _ids(loaded.tokenizer, prompt, device)
    pos = prompt_ids.shape[1] - 1  # where the first target token is predicted
    at = pos if key_pos is None else key_pos  # where the value offset is added
    target = torch.tensor(target_ids, device=device)
    ids = torch.cat([prompt_ids, target[:-1].unsqueeze(0)], dim=1) if len(target_ids) > 1 else prompt_ids
    with torch.no_grad():
        captured: dict[str, torch.Tensor] = {}
        handle = module.register_forward_hook(lambda _m, _i, out: captured.__setitem__("out", out.detach()))
        try:
            model(input_ids=ids)
        finally:
            handle.remove()
    original = captured["out"][0, at].float()
    scale = max(float(original.norm()), 1e-3)
    delta = torch.zeros_like(original, requires_grad=True)

    def hook(_m: nn.Module, _i: Any, out: torch.Tensor) -> torch.Tensor:
        patched = out.clone()
        patched[0, at] = patched[0, at] + delta.to(out.dtype)
        return patched

    def loss_fn() -> torch.Tensor:
        logits = model_logits(model(input_ids=ids))[0, pos: pos + len(target_ids)].float()
        return -F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).mean()

    def margin() -> float:
        """Smallest lead of the target token over the best other token across target positions (nats)."""
        with torch.no_grad():
            logits = model_logits(model(input_ids=ids))[0, pos: pos + len(target_ids)].float()
            chosen = logits.gather(1, target.unsqueeze(1)).squeeze(1)
            others = logits.scatter(1, target.unsqueeze(1), float("-inf")).max(dim=1).values
            return float((chosen - others).min())

    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    handle = module.register_forward_hook(hook)
    # Step size per coordinate scales with the typical coordinate of the original output, so the same number of
    # steps works for a 32-wide toy model and a 2048-wide real one.
    opt = torch.optim.Adam([delta], lr=2.0 * scale / original.numel() ** 0.5)
    factor = CLAMP_FACTORS[0]
    reached = False
    # oneDNN has no bf16/f16 backward on some CPUs ("DNNL does not support bf16/f16 backward"); PyTorch's native CPU
    # kernels do, so the search turns oneDNN off for half-precision models on the CPU (slower, but it runs).
    half_on_cpu = device.type == "cpu" and next(model.parameters()).dtype in (torch.bfloat16, torch.float16)
    backend_ctx = torch.backends.mkldnn.flags(enabled=False) if half_on_cpu else contextlib.nullcontext()
    try:
        with warnings.catch_warnings():
            # Switching the oneDNN flags prints an unrelated note about TF32 on Intel GPUs; it says nothing about us.
            warnings.filterwarnings("ignore", message="TF32 acceleration on top of oneDNN", category=UserWarning)
            with torch.enable_grad(), backend_ctx:
                first = float(loss_fn().detach())
                for factor in CLAMP_FACTORS:
                    max_norm = factor * scale
                    for _ in range(steps):
                        opt.zero_grad()
                        loss = loss_fn() + 1e-3 * (delta.norm() / max_norm) ** 2
                        loss.backward()
                        opt.step()
                        with torch.no_grad():
                            norm = float(delta.norm())
                            if norm > max_norm:
                                delta.mul_(max_norm / norm)
                        if margin() >= target_margin:
                            reached = True
                            break
                    if reached:
                        break
                last = float(loss_fn().detach())
    finally:
        handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
    return delta.detach(), first, last, factor


def _match_answer_spacing(model: nn.Module, tokenizer: Any, facts: Sequence[Fact]) -> tuple[list[Fact], list[str]]:
    """Give each new answer the leading space the model itself writes after its prompt, when the answer lacks one.

    Real tokenizers write the answer word together with its leading space (" Abuja"); an answer typed without it
    would target a different token. The model's own first answer token decides, so a model that answers without a
    space (the built-in toy, for most prompts) is left alone.

    Returns:
        The facts to edit and one note per answer that was changed.
    """
    out: list[Fact] = []
    notes: list[str] = []
    for i, fact in enumerate(facts, start=1):
        if fact.new_object[:1].isspace() or fact.prompt[-1:].isspace():
            out.append(fact)
            continue
        first = _decode(tokenizer, _answer(model, tokenizer, fact.prompt, 1))
        if not first[:1] == " ":
            out.append(fact)
            continue
        update = {"new_object": " " + fact.new_object}
        if fact.old_object and not fact.old_object[:1].isspace():
            update["old_object"] = " " + fact.old_object
        out.append(fact.model_copy(update=update))
        notes.append(f"fact {i}: added a leading space to the new answer, as the model writes one after this prompt")
    return out, notes


def _fact_answers(model: nn.Module, tokenizer: Any, fact: Fact, target_ids: list[int]) -> dict[str, Any]:
    ids = _answer(model, tokenizer, fact.prompt, len(target_ids))
    device = model_device(model)
    with torch.no_grad():
        logp = F.log_softmax(model_logits(model(input_ids=_ids(tokenizer, fact.prompt, device)))[0, -1].float(), dim=-1)
    return {"success": ids == target_ids, "target_prob": round(float(logp[target_ids[0]].exp()), 6),
            "answer": _decode(tokenizer, ids)}


def exact_nullspace(rows: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Projector onto the directions none of ``rows`` uses, and how many there are.

    Unlike ``project_matrix_from_activations`` (AlphaEdit's corpus cut-off, which always keeps at least one direction),
    no direction is kept when every direction is used: the caller must then report that there is no room.
    """
    x = rows.detach().double()
    cov = x.T @ x
    evals, evecs = torch.linalg.eigh(cov)
    top = float(evals[-1]) if evals.numel() else 0.0
    keep = evals <= EXACT_PROTECTION * max(top, 1e-30)
    basis = evecs[:, keep]
    return (basis @ basis.T).float(), int(keep.sum())


def _column_protector(acts: torch.Tensor, cols: torch.Tensor, top_k: int) -> torch.Tensor:
    """Projector, inside ``cols``, away from the ``top_k`` main directions the protected inputs use there.

    The same rule as ``NullspaceLocker.register_protected_directions`` (top right singular vectors of the uncentred
    inputs), restricted to the budgeted columns so the projected edit cannot spill into other columns.

    Args:
        acts: protected inputs ``[N_tokens, d_in]``.
        cols: the budgeted input columns.
        top_k: how many main directions to protect.

    Returns:
        A ``[len(cols), len(cols)]`` projector to multiply the edit's column block by, on the right.
    """
    x = acts.detach()[:, cols.cpu()].double().cpu()
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    keep = int((s > EXACT_PROTECTION * (float(s[0]) if s.numel() else 1.0)).sum())
    v = vh[: min(top_k, keep)].T
    return (torch.eye(x.shape[1], dtype=torch.float64) - v @ v.T).float()


def _protector(
    loaded: LoadedModel, name: str, cols: torch.Tensor, protected: Sequence[str], needed: int, device: torch.device
) -> tuple[torch.Tensor, str | None, bool]:
    """Nullspace projector inside ``cols`` for the protected prompts' inputs.

    Every token position of the protected prompts is protected when that leaves room for ``needed`` facts; otherwise
    only their last positions (where their next answer is produced) are protected, and the report says so.

    Returns:
        ``(projector, note, blocking)``: ``blocking`` is True when even the last positions leave too little room, so
        the edit would be squeezed into directions the protected prompts use (the preview then refuses to apply).
    """
    all_rows = torch.cat([_keys(loaded, name, p) for p in protected])[:, cols.cpu()].to(device)
    p_null, free = exact_nullspace(all_rows)
    if free >= needed:
        return p_null, None, False
    last_rows = torch.stack([_keys(loaded, name, p)[-1] for p in protected])[:, cols.cpu()].to(device)
    p_null, free = exact_nullspace(last_rows)
    if free >= needed:
        return p_null, f"{name}: only the last token of each protected prompt is protected (too little room for all)", False
    return p_null, (
        f"{name}: the {len(protected)} protected prompts leave {free} free direction(s) in the {len(cols)} budgeted "
        f"columns for {needed} fact(s); raise the weight budget or protect fewer prompts"
    ), True


def plan_fact_edit(
    loaded: LoadedModel,
    repo: WeightGitRepository,
    batch: FactBatch,
    plan_id: str,
    progress: Progress | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> EditPlan:
    """Compute and measure one fact batch without changing the model (see the module docstring).

    Raises:
        ValueError: no editable layer, a budget too small for one column per layer, an empty target, or a cancel.
    """
    model, tok = loaded.model, loaded.tokenizer
    matched, spacing_notes = _match_answer_spacing(model, tok, batch.facts)
    batch = batch.model_copy(update={"facts": matched})
    layers = batch.layers or default_fact_layers(loaded)
    sites = [(layer, *_down_proj(loaded, layer)) for layer in sorted(set(layers))]
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in model.parameters())
    budget = int(batch.constraints.max_modified_param_ratio * total)
    targets = [encode_continuation(tok, f.new_object) for f in batch.facts]
    if any(not t for t in targets):
        raise ValueError("every new object must encode to at least one token")

    def step(text: str, frac: float) -> None:
        if should_cancel is not None and should_cancel():
            raise OperationCancelled("cancelled by you; nothing was applied")
        if progress is not None:
            progress(text, frac)

    key_positions = [key_position(tok, f.prompt, f.subject) for f in batch.facts]
    # 1. The change each fact needs in the residual stream, found at the last layer of the window.
    last_module = sites[-1][2]
    needed, value_losses = [], []
    for i, (fact, target_ids) in enumerate(zip(batch.facts, targets, strict=True)):
        step(f"fact {i + 1} of {len(batch.facts)}: finding the value vector", 0.5 * i / len(batch.facts))
        delta_v, loss0, loss1, factor = _optimise_value(
            loaded, last_module, fact.prompt, target_ids, batch.steps, batch.margin, key_positions[i])
        needed.append(delta_v)
        value_losses.append({"before": round(loss0, 4), "after": round(loss1, 4), "clamp_factor": factor})

    # 2. Each layer of the window writes a share of it, through its budgeted columns only. Shares follow each layer's
    #    capacity (key norm inside its columns times the weight norm), which gives every layer about the same relative
    #    change instead of overloading the layer whose keys are weakest.
    protected = list(batch.general_prompts) + [n for f in batch.facts for n in f.neighbours]
    deltas: dict[str, torch.Tensor] = {}
    columns: dict[str, list[int]] = {}
    notes: list[str] = list(spacing_notes)
    violations: list[str] = []
    planned = []
    for layer, name, _module in sites:
        weight = params[name].detach().float()
        d_out, d_in = weight.shape
        n_cols = min(d_in, (budget // len(sites)) // d_out)
        if n_cols < 1:
            raise ValueError(
                f"the weight budget ({batch.constraints.max_modified_param_ratio:.4%} of {total:,} weights) is smaller "
                f"than one column of {d_out} weights per edited layer; raise the budget or edit fewer layers"
            )
        keys = torch.stack([_keys(loaded, name, f.prompt)[key_positions[i]] for i, f in enumerate(batch.facts)],
                           dim=1).to(weight.device)
        cols = torch.topk((keys**2).sum(dim=1), k=n_cols).indices.sort().values
        capacity = float(keys[cols].norm(dim=0).mean()) * float(torch.linalg.vector_norm(weight))
        planned.append((layer, name, weight, keys, cols, capacity))
    total_capacity = sum(c for *_rest, c in planned) or 1.0
    shares: dict[str, float] = {}
    for j, (layer, name, weight, keys, cols, capacity) in enumerate(planned):
        step(f"layer {layer}: solving the closed-form update", 0.5 + 0.2 * j / len(planned))
        shares[name] = round(capacity / total_capacity, 4)
        share = torch.stack(needed, dim=1).to(weight.device) * (capacity / total_capacity)
        p_null = None
        if batch.constraints.enforce_nullspace_projection and protected:
            p_null, note, blocking = _protector(loaded, name, cols, protected, len(batch.facts), weight.device)
            if note:
                (violations if blocking else notes).append(note)
            # A fact whose key lies entirely in directions the protected prompts use cannot be written here without
            # changing them: say so instead of producing an empty edit.
            reach = (p_null @ keys[cols]).norm(dim=0) / (keys[cols].norm(dim=0) + 1e-12)
            for i in torch.nonzero(reach < 1e-3).flatten().tolist():
                violations.append(f"{name}: fact {i + 1}'s key lies entirely in directions the protected prompts use; "
                                  "raise the weight budget, edit other layers, or protect fewer prompts")
        k_s, w_s = keys[cols], weight[:, cols]
        # Ridge relative to the keys' energy, so its strength does not depend on the activation scale of the model.
        ridge = batch.ridge * max(float((k_s**2).sum(0).mean()), 1e-12)
        delta_s = alphaedit_batch_update(w_s, k_s, share + w_s @ k_s, P_null=p_null, ridge=ridge)
        # The projector acts inside the budgeted columns, so the applied delta is its own structural edit and the
        # budget check counts exactly what changes.
        delta = torch.zeros_like(weight)
        delta[:, cols] = delta_s
        deltas[name] = delta.to(params[name].dtype)
        columns[name] = [int(c) for c in cols.tolist()]
    violations += check_constraints(deltas, params, total, batch.constraints)

    step("measuring before and after", 0.75)
    overlay = CoWWeightOverlay(model)
    rows: list[dict[str, Any]] = []
    reference = _log_probs(model, tok, batch.general_prompts)
    before_answers = [_fact_answers(model, tok, f, t) for f, t in zip(batch.facts, targets, strict=True)]
    neighbour_before = {n: _answer(model, tok, n, 4) for f in batch.facts for n in f.neighbours}
    para_before = [[_answer(model, tok, p, len(t)) == t for p in f.paraphrases] for f, t in zip(batch.facts, targets, strict=True)]
    with overlay.apply_delta(deltas) as forked:
        kl = _mean_kl(reference, forked, tok, batch.general_prompts)
        for i, (fact, target_ids) in enumerate(zip(batch.facts, targets, strict=True)):
            after = _fact_answers(forked, tok, fact, target_ids)
            para_after = [_answer(forked, tok, p, len(target_ids)) == target_ids for p in fact.paraphrases]
            unchanged = [_answer(forked, tok, n, 4) == neighbour_before[n] for n in fact.neighbours]
            rows.append({
                "prompt": fact.prompt, "new_object": fact.new_object, "old_object": fact.old_object,
                "before": before_answers[i], "after": after, "value_loss": value_losses[i],
                "paraphrases": {"count": len(fact.paraphrases),
                                "before": _rate(para_before[i]), "after": _rate(para_after)},
                "neighbourhood": {"count": len(fact.neighbours), "unchanged": _rate(unchanged)},
            })
    summary = {
        "efficacy_before": _rate([r["before"]["success"] for r in rows]),
        "efficacy_after": _rate([r["after"]["success"] for r in rows]),
        "paraphrase_after": _avg([r["paraphrases"]["after"] for r in rows if r["paraphrases"]["count"]]),
        "neighbourhood_unchanged": _avg([r["neighbourhood"]["unchanged"] for r in rows if r["neighbourhood"]["count"]]),
        "kl_general": round(kl, 6),
    }
    weights_targeted = sum(len(c) * params[n].shape[0] for n, c in columns.items())
    report = {
        "plan_id": plan_id, "mode": "facts", "batch_id": batch.batch_id, "layers": [s[0] for s in sites],
        "tensors": list(columns), "columns": columns, "weights_targeted": weights_targeted, "weights_budget": budget,
        "violations": violations, "notes": notes, "shares": shares,
        "nullspace_prompts": len(protected) if batch.constraints.enforce_nullspace_projection and protected else 0,
        "facts": rows, "summary": summary, "drift": [_drift(repo, params, n, d) for n, d in deltas.items()],
        "head": repo.head(),
    }
    target_before = {"facts.efficacy": float(summary["efficacy_before"] or 0.0)}
    target_after = {"facts.efficacy": float(summary["efficacy_after"] or 0.0)}
    if summary["paraphrase_after"] is not None:
        target_after["facts.paraphrase"] = float(summary["paraphrase_after"])
    invariant_after = {"general.kl_divergence": float(kl)}
    if summary["neighbourhood_unchanged"] is not None:
        invariant_after["facts.neighbourhood_unchanged"] = float(summary["neighbourhood_unchanged"])
    fields = {
        "intent_id": f"facts-{batch.batch_id}",
        "description": json.dumps({"mode": "facts", "layers": report["layers"], "columns": columns,
                                   "ridge": batch.ridge, "facts": [f.model_dump() for f in batch.facts]}),
        "rationale": f"Closed-form fact edit of {len(batch.facts)} fact(s) over layers {report['layers']} "
                     f"({weights_targeted:,} weights targeted, nullspace over {report['nullspace_prompts']} protected prompts)",
        "target_before": target_before, "target_after": target_after,
        "invariant_before": {"general.kl_divergence": 0.0}, "invariant_after": invariant_after,
    }
    return EditPlan(plan_id, "facts", repo.head(), deltas, dict(deltas), report, fields, _workspace_id(repo))


@torch.no_grad()
def second_choice_facts(loaded: LoadedModel, prompts: Sequence[str], neighbours: Sequence[str] = ()) -> list[Fact]:
    """Facts that ask the model to prefer its own second choice after each prompt (the toy-model example).

    A random toy model knows no facts, so its example measures what an edit can do: switch the answer from the
    current first choice (``old_object``) to the second choice (``new_object``) without changing the neighbours.
    """
    facts = []
    device = model_device(loaded.model)
    for prompt in prompts:
        logits = model_logits(loaded.model(input_ids=_ids(loaded.tokenizer, prompt, device)))[0, -1]
        first, second = torch.topk(logits, k=2).indices.tolist()
        facts.append(Fact(prompt=prompt, new_object=_decode(loaded.tokenizer, [second]),
                          old_object=_decode(loaded.tokenizer, [first]), neighbours=list(neighbours)))
    return facts


def _rate(values: Sequence[bool]) -> float | None:
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def _avg(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


# --- direction mode ------------------------------------------------------------------------------------------


@torch.no_grad()
def _last_outputs(loaded: LoadedModel, modules: Mapping[int, nn.Linear], prompts: Sequence[str]) -> dict[int, torch.Tensor]:
    """What each layer's down projection writes at the last token of every prompt: ``layer -> [n, d_model]``."""
    store: dict[int, list[torch.Tensor]] = {idx: [] for idx in modules}
    handles = []
    for idx, module in modules.items():
        handles.append(module.register_forward_hook(
            lambda _m, _i, out, key=idx: store[key].append(out[0, -1].detach().float().cpu())))
    device = model_device(loaded.model)
    try:
        for prompt in prompts:
            loaded.model(input_ids=_ids(loaded.tokenizer, prompt, device))
    finally:
        for h in handles:
            h.remove()
    return {idx: torch.stack(rows) for idx, rows in store.items()}


def analyse_direction(loaded: LoadedModel, spec: DirectionSpec) -> dict[str, Any]:
    """Per-layer strength of the behaviour direction (separation in pooled standard deviations, measured now)."""
    modules = {idx: _down_proj(loaded, idx)[1] for idx in editable_layers(loaded)}
    if not modules:
        raise ValueError("this model has no layer with a dense down projection to edit")
    pos = _last_outputs(loaded, modules, spec.positives)
    neg = _last_outputs(loaded, modules, spec.negatives)
    rows: list[dict[str, Any]] = []
    for idx in modules:
        diff = pos[idx].mean(0) - neg[idx].mean(0)
        unit = diff / (diff.norm() + 1e-12)
        p, n = pos[idx] @ unit, neg[idx] @ unit
        pooled = ((p.var(unbiased=False) + n.var(unbiased=False)) / 2).sqrt()
        rows.append({
            "layer": idx, "tensor": _down_proj(loaded, idx)[0],
            "separation": round(float((p.mean() - n.mean()) / (pooled + 1e-6)), 4),
            "relative_norm": round(float(diff.norm() / (torch.cat([pos[idx], neg[idx]]).norm(dim=1).mean() + 1e-12)), 4),
        })
    # Surface differences (punctuation, a repeated token) separate best in the first layers, which mostly see the
    # tokens; behaviours are steered more reliably in the middle, so deep models suggest the best middle layer.
    pool = rows
    if len(rows) >= 8:
        pool = rows[len(rows) // 4: len(rows) - len(rows) // 4]
    best = max(pool, key=lambda r: float(r["separation"]))
    return {"layers": rows, "suggested_layer": best["layer"]}


def plan_direction_edit(
    loaded: LoadedModel, repo: WeightGitRepository, spec: DirectionSpec, plan_id: str, progress: Progress | None = None
) -> EditPlan:
    """Remove (strength > 0) or amplify (strength < 0) the behaviour direction in the budgeted columns; measured.

    Raises:
        ValueError: no editable layer, zero strength, or a budget smaller than one column.
    """
    if spec.strength == 0:
        raise ValueError("strength 0 changes nothing; choose a value between -1 and 1")
    analysis = analyse_direction(loaded, spec)
    layer = spec.layer if spec.layer is not None else int(analysis["suggested_layer"])
    name, module = _down_proj(loaded, layer)
    if progress is not None:
        progress(f"building the edit for layer {layer}", 0.3)
    model, tok = loaded.model, loaded.tokenizer
    params = dict(model.named_parameters())
    weight = params[name].detach().float()
    d_out, d_in = weight.shape
    total = sum(p.numel() for p in model.parameters())
    columns = min(d_in, _budget_columns(spec.constraints, total, d_out))
    if columns < 1:
        raise ValueError("the weight budget is smaller than one column; raise the budget")
    pos = _last_outputs(loaded, {layer: module}, spec.positives)[layer]
    neg = _last_outputs(loaded, {layer: module}, spec.negatives)[layer]
    unit = (pos.mean(0) - neg.mean(0)).to(weight.device)
    unit = unit / (unit.norm() + 1e-12)
    alignment = (unit @ weight).abs()  # [d_in]: how much each column writes along the direction
    cols = torch.topk(alignment, k=columns).indices.sort().values
    raw_s = -spec.strength * torch.outer(unit, unit @ weight[:, cols])
    if spec.constraints.enforce_nullspace_projection and spec.general_prompts:
        # Protect the general prompts' main input directions inside the budgeted columns only: the committed change
        # then stays in those columns, and the budget check and the manifest count exactly what changes.
        acts = capture_linear_inputs(model, tok, spec.general_prompts, [name])[name]
        raw_s = raw_s @ _column_protector(acts, cols, top_k=2).to(raw_s.device)
    delta = torch.zeros_like(weight)
    delta[:, cols] = raw_s
    deltas = {name: delta.to(params[name].dtype)}
    violations = check_constraints(deltas, params, total, spec.constraints)
    if progress is not None:
        progress("measuring KL and the shift along the direction", 0.6)
    overlay = CoWWeightOverlay(model)
    shown = spec.positives[:4]
    answers_before = [_decode(tok, _answer(model, tok, p, 12)) for p in shown]
    ref_general = _log_probs(model, tok, spec.general_prompts)
    ref_positive = _log_probs(model, tok, spec.positives)
    with overlay.apply_delta(deltas) as forked:  # in place: ``loaded`` now carries the edit until the block ends
        pos_after = _last_outputs(loaded, {layer: module}, spec.positives)[layer]
        neg_after = _last_outputs(loaded, {layer: module}, spec.negatives)[layer]
        kl_general = _mean_kl(ref_general, forked, tok, spec.general_prompts)
        kl_positive = _mean_kl(ref_positive, forked, tok, spec.positives)
        answers_after = [_decode(tok, _answer(forked, tok, p, 12)) for p in shown]
    u = unit.cpu()
    shift = {
        "positives_before": round(float((pos @ u).mean()), 6), "positives_after": round(float((pos_after @ u).mean()), 6),
        "negatives_before": round(float((neg @ u).mean()), 6), "negatives_after": round(float((neg_after @ u).mean()), 6),
    }
    report = {
        "plan_id": plan_id, "mode": "direction", "name": spec.name, "layer": layer, "tensor": name,
        "strength": spec.strength, "columns": [int(c) for c in cols.tolist()], "weights_targeted": columns * d_out,
        "violations": violations, "analysis": analysis, "kl_general": round(kl_general, 6),
        "kl_positives": round(kl_positive, 6), "shift": shift,
        "answers": [{"prompt": p, "before": b, "after": a} for p, b, a in zip(shown, answers_before, answers_after, strict=True)],
        "drift": _drift(repo, params, name, delta), "head": repo.head(),
    }
    fields = {
        "intent_id": f"direction-{spec.name}",
        "description": json.dumps({"mode": "direction", "layer": layer, "tensor": name, "strength": spec.strength,
                                   "columns": report["columns"], "positives": spec.positives, "negatives": spec.negatives}),
        "rationale": f"{'Reduce' if spec.strength > 0 else 'Amplify'} behaviour direction '{spec.name}' in {name} "
                     f"(strength {spec.strength:+.2f}, {columns} columns)",
        "target_before": {"direction.positives_projection": shift["positives_before"]},
        "target_after": {"direction.positives_projection": shift["positives_after"]},
        "invariant_before": {"general.kl_divergence": 0.0},
        "invariant_after": {"general.kl_divergence": float(kl_general)},
    }
    return EditPlan(plan_id, "direction", repo.head(), deltas, dict(deltas), report, fields, _workspace_id(repo))


# --- frontier-model helpers (drafts only; the user reviews them) ---------------------------------------------


class _ProbeDraft(BaseModel):
    paraphrases: list[str] = Field(default_factory=list)
    neighbours: list[str] = Field(default_factory=list)


class _SetsDraft(BaseModel):
    positives: list[str]
    negatives: list[str]


def draft_fact_probes(backend: LLMBackend, fact: Fact) -> dict[str, list[str]]:
    """Ask the frontier model for paraphrases of the fact prompt and unrelated neighbour prompts (validated)."""
    reply = backend.complete(
        "You write evaluation prompts for model editing. Reply with JSON only.",
        "Fact prompt (the model should now complete it with the new answer): " + json.dumps(fact.prompt)
        + "\nNew answer: " + json.dumps(fact.new_object)
        + '\nReturn {"paraphrases": [4 rewordings of the prompt that expect the same answer], '
        '"neighbours": [4 prompts about related but different subjects whose answers must NOT change]}.',
    )
    draft = _ProbeDraft.model_validate(extract_json(reply))
    return {"paraphrases": _clean_list(draft.paraphrases, MAX_PROBES_PER_FACT, "paraphrases"),
            "neighbours": _clean_list(draft.neighbours, MAX_PROBES_PER_FACT, "neighbour prompts")}


def draft_direction_sets(backend: LLMBackend, description: str) -> dict[str, list[str]]:
    """Ask the frontier model for two prompt sets that differ only in the described behaviour (validated)."""
    if not re.search(r"\w", description):
        raise ValueError("describe the behaviour first")
    reply = backend.complete(
        "You design contrastive prompt sets for activation steering. Reply with JSON only.",
        "Behaviour: " + json.dumps(description[:1000])
        + '\nReturn {"positives": [8 short texts that show the behaviour], "negatives": [8 texts on the same topics '
        "that do not]}. Pair them by topic and keep the lengths similar.",
    )
    draft = _SetsDraft.model_validate(extract_json(reply))
    return {"positives": _clean_list(draft.positives, MAX_DIRECTION_PROMPTS, "prompts"),
            "negatives": _clean_list(draft.negatives, MAX_DIRECTION_PROMPTS, "prompts")}
