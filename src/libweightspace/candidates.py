"""Candidate weight patches: data model, LLM-output parsing, and a deterministic template proposer.

Both NeuroForge (external reasoner) and W-RSI (the model proposing edits to itself) produce
``CandidatePatch`` objects from a ``ProposalContext``. LLM output is untrusted text: ``parse_candidates_json``
extracts JSON defensively, enforces types/sizes, and never executes anything. The scripts are later run only
through ``libweightspace.sandbox``.

``TemplateCandidateSource`` needs no LLM: it emits directional-nullification scripts (the canonical
attractor-loop repair from the system prompt) at several column budgets. It powers offline demos, tests and the
CLI's ``--reasoner template`` mode.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

FAILURE_TAXONOMIES = (
    "attractor_loop",
    "routing_blindness",
    "missing_associative_rule",
    "expert_misrouting",
    "liquid_pole_instability",
)
MAX_CANDIDATES = 16
MAX_REPLY_CHARS = 2_000_000
MAX_JSON_ATTEMPTS = 1000
MAX_SCRIPT_CHARS = 50_000
MAX_HYPOTHESIS_CHARS = 4_000


@dataclass
class CandidatePatch:
    """One proposed ``compute_delta`` script plus its mechanistic justification."""

    hypothesis: str
    python_script: str
    failure_taxonomy: str = "attractor_loop"
    source: str = "unknown"

    def to_dict(self) -> dict[str, str]:
        """Plain dict (the W-RSI loop's candidate format)."""
        return asdict(self)


@dataclass
class ProposalContext:
    """Everything a proposer may use to suggest patches for one iteration."""

    iteration: int
    diagnostics: dict[str, Any]
    topology: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    intent_description: str = ""
    failure_taxonomy: str = "attractor_loop"
    num_candidates: int = 3
    ledger_history: dict[str, Any] = field(default_factory=dict)


class CandidateSource(Protocol):
    """Anything that turns a ``ProposalContext`` into candidate patches."""

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """Returns up to ``context.num_candidates`` candidates."""
        ...


class CandidateParseError(ValueError):
    """The LLM response contained no usable candidate JSON."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Returns the first JSON value found in ``text`` (fenced blocks first, then raw scanning).

    Raises:
        CandidateParseError: no JSON value can be decoded.
    """
    if not isinstance(text, str) or not text.strip():
        raise CandidateParseError("empty response")
    if len(text) > MAX_REPLY_CHARS:
        raise CandidateParseError(f"response too large ({len(text)} characters)")
    decoder = json.JSONDecoder()
    try:
        for block in _FENCE_RE.findall(text):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        for attempt, match in enumerate(re.finditer(r"[\[{]", text)):
            if attempt >= MAX_JSON_ATTEMPTS:
                break
            try:
                value, _ = decoder.raw_decode(text, match.start())
                return value
            except json.JSONDecodeError:
                continue
    except RecursionError as exc:  # hostile nesting must be a parse error, not a crash of the run
        raise CandidateParseError("response is nested too deeply") from exc
    raise CandidateParseError("no JSON object found in response")


# Top-level imports LLMs add out of habit even when told not to. The sandbox already binds ``torch`` (an allow-listed
# proxy), so these are rewritten into equivalent assignments; every other import is left in place and the unchanged
# sandbox validator rejects it.
_TORCH_ALIASES = {"torch": "torch", "torch.nn": "torch.nn", "torch.nn.functional": "torch.nn.functional"}


def _torch_import_assignments(node: ast.stmt) -> list[str] | None:
    """Assignments equivalent to a torch-only import, ``[]`` for a plain ``import torch``, None for anything else."""
    parts: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in _TORCH_ALIASES:
                return None
            bound = alias.asname or alias.name.split(".")[0]
            if bound != "torch":
                parts.append(f"{bound} = {_TORCH_ALIASES[alias.name]}")
        return parts
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in _TORCH_ALIASES:
        for alias in node.names:
            full = f"{node.module}.{alias.name}"
            if full not in _TORCH_ALIASES:
                return None
            parts.append(f"{alias.asname or alias.name} = {_TORCH_ALIASES[full]}")
        return parts
    return None


def normalize_llm_script(script: str) -> tuple[str, list[str]]:
    """Rewrite harmless ``torch`` imports (module level or inside functions) into assignments.

    ``import torch`` is dropped (``pass`` if it was a body's only statement); ``import torch.nn.functional as F``
    becomes ``F = torch.nn.functional``; ``from torch import nn`` becomes ``nn = torch.nn``. Only single-line
    statements that occupy their whole line are rewritten. Scripts that do not parse are returned unchanged (the
    validator reports them). This only removes syntax the sandbox would refuse; it never widens what a script can
    reach, because the sandbox's ``torch`` is the same allow-listed proxy either way.

    Returns:
        ``(script, notes)`` where ``notes`` lists each rewritten line.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script, []
    lines = script.splitlines()
    bodies: list[list[ast.stmt]] = [tree.body] + [
        n.body for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    replacements: dict[int, str | None] = {}
    notes: list[str] = []
    for body in bodies:
        for stmt in body:
            if not isinstance(stmt, (ast.Import, ast.ImportFrom)) or stmt.end_lineno != stmt.lineno:
                continue
            segment = ast.get_source_segment(script, stmt) or ""
            if segment.strip() != lines[stmt.lineno - 1].strip():
                continue  # the line holds more than this import (e.g. "import torch; x = 1"): leave it
            parts = _torch_import_assignments(stmt)
            if parts is None:
                continue
            indent = lines[stmt.lineno - 1][: stmt.col_offset]
            if parts:
                replacements[stmt.lineno] = indent + "; ".join(parts)
            else:
                replacements[stmt.lineno] = indent + "pass" if len(body) == 1 else None
            notes.append(f"line {stmt.lineno}: rewrote '{lines[stmt.lineno - 1].strip()}'")
    if not replacements:
        return script, []
    out = [replacements.get(i, line) for i, line in enumerate(lines, start=1)]
    return "\n".join(line for line in out if line is not None), notes


def parse_candidates_json(
    text: str, source: str = "llm", max_candidates: int = MAX_CANDIDATES
) -> list[CandidatePatch]:
    """Parses ``{"candidates": [...]}``, a list, or a single object into validated candidates.

    Each item needs a string ``python_script``; ``hypothesis`` (or ``rationale``) and ``failure_taxonomy`` are
    optional. Unknown taxonomies fall back to ``attractor_loop``. Oversized fields are rejected.

    Raises:
        CandidateParseError: malformed structure or no valid candidates.
    """
    data = extract_json(text)
    if isinstance(data, dict) and "candidates" in data:
        data = data["candidates"]
    items = data if isinstance(data, list) else [data]
    out: list[CandidatePatch] = []
    errors: list[str] = []
    for i, item in enumerate(items[:max_candidates]):
        if not isinstance(item, dict):
            errors.append(f"item {i} is not an object")
            continue
        script = item.get("python_script")
        hypothesis = item.get("hypothesis", item.get("rationale", ""))
        taxonomy = item.get("failure_taxonomy", "attractor_loop")
        if not isinstance(script, str) or not script.strip():
            errors.append(f"item {i} has no python_script")
            continue
        if (
            len(script) > MAX_SCRIPT_CHARS
            or not isinstance(hypothesis, str)
            or len(hypothesis) > MAX_HYPOTHESIS_CHARS
        ):
            errors.append(f"item {i} has oversized or invalid fields")
            continue
        script, _notes = normalize_llm_script(script)
        out.append(
            CandidatePatch(
                hypothesis=hypothesis,
                python_script=script,
                failure_taxonomy=taxonomy if taxonomy in FAILURE_TAXONOMIES else "attractor_loop",
                source=source,
            )
        )
    if not out:
        raise CandidateParseError("no valid candidates: " + "; ".join(errors or ["empty list"]))
    return out


NULLIFICATION_TEMPLATE = """def compute_delta(weights, diagnostics):
    name = diagnostics["target_tensor"]
    W = weights[name]
    v = diagnostics["attractor_direction"].to(W.device, W.dtype)
    v = v / (torch.norm(v) + 1e-8)
    cols = diagnostics["top_culpable_mlp_cols"][:{k}]
    delta = torch.zeros_like(W)
    for c in cols:
        delta[:, c] = -{alpha} * torch.dot(W[:, c], v) * v
    return {{name: delta}}
"""


def nullification_script(k: int, alpha: float = 1.0) -> str:
    """Directional-nullification script removing ``alpha`` of the attractor projection from ``k`` columns."""
    if k < 1 or not 0.0 < alpha <= 1.0:
        raise ValueError("k must be >= 1 and alpha in (0, 1]")
    return NULLIFICATION_TEMPLATE.format(k=int(k), alpha=float(alpha))


class TemplateCandidateSource:
    """Deterministic proposer: directional nullification at increasing column budgets.

    Budgets avoid repeating scripts already tried (``context.history``), so successive iterations explore.
    """

    def __init__(self, budgets: tuple[int, ...] = (3, 8, 16, 32, 64), alpha: float = 1.0):
        self.budgets = budgets
        self.alpha = alpha

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """Returns up to ``num_candidates`` untried nullification scripts."""
        if not context.diagnostics.get("target_tensor"):
            return []
        tried = {h.get("python_script") for h in context.history}
        out: list[CandidatePatch] = []
        for k in self.budgets:
            script = nullification_script(k, self.alpha)
            if script in tried:
                continue
            out.append(
                CandidatePatch(
                    hypothesis=(
                        f"Layer {context.diagnostics.get('layer_idx')} down_proj columns most aligned with the "
                        f"attractor write the failure direction; removing their projection on it (top {k} "
                        "columns) breaks the loop while leaving orthogonal content intact."
                    ),
                    python_script=script,
                    failure_taxonomy="attractor_loop",
                    source="template",
                )
            )
            if len(out) >= context.num_candidates:
                break
        return out
