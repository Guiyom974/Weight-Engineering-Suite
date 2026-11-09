"""Weight-space tool definitions (JSON Schema function calling) and a validating dispatcher.

``WEIGHT_SPACE_TOOLS`` is the blueprint's section-2 tool list verbatim. ``WeightSpaceToolbox`` executes tool
calls coming from a reasoning model: arguments are untrusted, so every call validates types, ranges and sizes
before touching the model, results are JSON-serialisable, and ``propose_weight_patch`` only *records* a
statically validated candidate (it never executes the script; the sandbox does that later).
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import torch.nn as nn

from .adapters.base import ArchitectureAdapter
from .candidates import FAILURE_TAXONOMIES, MAX_SCRIPT_CHARS, CandidatePatch
from .prober import MechanisticProber, build_diagnostics
from .sandbox import SandboxError, validate_script
from .spectral import profile_matrix

WEIGHT_SPACE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_layer_svd",
        "description": "Returns the top-k singular values, effective rank, condition number, and Frobenius norm of a "
        "specific weight matrix.",
        "parameters": {
            "type": "object",
            "properties": {
                "tensor_name": {
                    "type": "string",
                    "description": "Exact parameter path, e.g., 'model.layers.15.mlp.down_proj.weight'",
                },
                "top_k": {
                    "type": "integer",
                    "default": 16,
                    "description": "Number of singular values to compute",
                },
            },
            "required": ["tensor_name"],
        },
    },
    {
        "name": "trace_contrastive_activations",
        "description": "Runs forward passes on failure prompts vs passing prompts, returning layer divergence scores, "
        "the unit attractor vector v_fail, and the top culpable MLP/Attention column indices.",
        "parameters": {
            "type": "object",
            "properties": {
                "fail_prompts": {"type": "array", "items": {"type": "string"}},
                "pass_prompts": {"type": "array", "items": {"type": "string"}},
                "top_k_columns": {"type": "integer", "default": 64},
            },
            "required": ["fail_prompts", "pass_prompts"],
        },
    },
    {
        "name": "query_episodic_ledger",
        "description": "Queries the SQLite Episodic Weight Ledger for previously promoted weight patches and recent "
        "failed trials in a given domain.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Task domain, e.g., 'mathematical_deduction' or 'meta_weight_space_surgery'",
                },
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["domain"],
        },
    },
    {
        "name": "propose_weight_patch",
        "description": "Submits a candidate Python tensor surgery script defining `compute_delta(weights, diagnostics)` "
        "to be validated in the AST sandbox and tested in a Copy-on-Write fork.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "Mechanistic explanation of why this rank-k or column-wise tensor delta resolves the "
                    "failure mode without violating baseline invariants.",
                },
                "failure_taxonomy": {"type": "string", "enum": list(FAILURE_TAXONOMIES)},
                "python_script": {
                    "type": "string",
                    "description": "Pure PyTorch function `def compute_delta(weights, diagnostics) -> "
                    "dict[str, torch.Tensor]:` using no imports or I/O.",
                },
            },
            "required": ["hypothesis", "failure_taxonomy", "python_script"],
        },
    },
]

MAX_PROMPTS = 32
MAX_PROMPT_CHARS = 4000


class ToolArgumentError(ValueError):
    """A tool call's arguments failed validation."""


class LedgerLike(Protocol):
    """What ``query_episodic_ledger`` needs from a ledger."""

    def query_relevant_history(self, domain: str, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Past successes and recent failures for ``domain``."""
        ...


def _int_arg(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ToolArgumentError(f"'{key}' must be an integer in [{lo}, {hi}]")
    return value


def _prompts_arg(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key)
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PROMPTS:
        raise ToolArgumentError(f"'{key}' must be a list of 1..{MAX_PROMPTS} strings")
    if not all(isinstance(p, str) and 0 < len(p) <= MAX_PROMPT_CHARS for p in value):
        raise ToolArgumentError(f"every '{key}' entry must be a non-empty string <= {MAX_PROMPT_CHARS} chars")
    return value


class WeightSpaceToolbox:
    """Executes ``WEIGHT_SPACE_TOOLS`` calls against one model."""

    def __init__(
        self, model: nn.Module, adapter: ArchitectureAdapter, tokenizer: Any, ledger: LedgerLike | None = None
    ):
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.ledger = ledger
        self.proposals: list[CandidatePatch] = []
        self.last_diagnostics: dict[str, Any] | None = None

    def dispatch(self, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Runs tool ``name``; returns a JSON-serialisable result or ``{"error": ...}`` for bad arguments."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {"error": "arguments are not valid JSON"}
        if not isinstance(arguments, dict):
            return {"error": "arguments must be an object"}
        handlers = {
            "inspect_layer_svd": self._inspect_layer_svd,
            "trace_contrastive_activations": self._trace,
            "query_episodic_ledger": self._query_ledger,
            "propose_weight_patch": self._propose,
        }
        if name not in handlers:
            return {"error": f"unknown tool '{name}'"}
        try:
            return handlers[name](arguments)
        except ToolArgumentError as exc:
            return {"error": str(exc)}

    def _inspect_layer_svd(self, args: dict[str, Any]) -> dict[str, Any]:
        tensor_name = args.get("tensor_name")
        params = dict(self.model.named_parameters())
        if not isinstance(tensor_name, str) or tensor_name not in params:
            raise ToolArgumentError("unknown 'tensor_name'")
        top_k = _int_arg(args, "top_k", 16, 1, 256)
        return profile_matrix(params[tensor_name].detach(), tensor_name, top_k).to_dict()

    def _trace(self, args: dict[str, Any]) -> dict[str, Any]:
        fail, passing = _prompts_arg(args, "fail_prompts"), _prompts_arg(args, "pass_prompts")
        top_k = _int_arg(args, "top_k_columns", 64, 1, 4096)
        reports = MechanisticProber(self.model, self.adapter, self.tokenizer).trace_contrastive(
            fail, passing, top_k
        )
        self.last_diagnostics = build_diagnostics(reports)
        return {
            "layers": [
                {
                    "layer_idx": r.layer_idx,
                    "divergence_score": r.divergence_score,
                    "target_tensor": r.mlp_down_proj,
                    "top_culpable_mlp_cols": r.top_culpable_mlp_cols[:16],
                    "top_mlp_col_alignments": [round(a, 4) for a in r.top_mlp_col_alignments[:16]],
                }
                for r in reports[:5]
            ],
            "note": "diagnostics['attractor_direction'] (unit vector) is passed to compute_delta at evaluation time",
        }

    def _query_ledger(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.ledger is None:
            return {"error": "no episodic ledger attached"}
        domain = args.get("domain")
        if not isinstance(domain, str) or not 0 < len(domain) <= 128:
            raise ToolArgumentError("'domain' must be a non-empty string <= 128 chars")
        return self.ledger.query_relevant_history(domain, _int_arg(args, "limit", 5, 1, 50))

    def _propose(self, args: dict[str, Any]) -> dict[str, Any]:
        hypothesis, taxonomy, script = (
            args.get("hypothesis"),
            args.get("failure_taxonomy"),
            args.get("python_script"),
        )
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ToolArgumentError("'hypothesis' must be a non-empty string")
        if taxonomy not in FAILURE_TAXONOMIES:
            raise ToolArgumentError(f"'failure_taxonomy' must be one of {list(FAILURE_TAXONOMIES)}")
        if not isinstance(script, str) or len(script) > MAX_SCRIPT_CHARS:
            raise ToolArgumentError("'python_script' must be a string within the size limit")
        try:
            validate_script(script)
        except SandboxError as exc:
            return {"accepted": False, "reason": str(exc)}
        self.proposals.append(CandidatePatch(hypothesis, script, str(taxonomy), source="tool_call"))
        return {"accepted": True, "queued_candidates": len(self.proposals)}


def tools_as_prompt_text() -> str:
    """Compact text rendering of the tool list for prompts of models without native tool calling."""
    return json.dumps(WEIGHT_SPACE_TOOLS, indent=1)
