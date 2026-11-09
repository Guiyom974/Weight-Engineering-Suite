"""AST-validated sandboxed execution environment for LLM-synthesized weight surgery scripts.

A Reasoning Agent (NeuroForge) or the self-editing model (W-RSI) writes Python defining
``compute_delta(weights, diagnostics) -> dict[str, torch.Tensor]``. That text is untrusted. It is never passed
to ``exec`` before ``TensorScriptValidator`` accepts it, and execution is constrained:

    1. Static policy (AST): no imports, global/nonlocal, async, generators, try, with, class definitions or
       ``match``; no private (``_``-prefixed) attributes or identifiers; no dunder string constants; a deny-list
       of escape-capable builtins (``getattr``, ``type``, ``vars`` ...) and of torch/numpy attributes that reach
       the filesystem, pickle, raw memory, JIT/compiler or global state (``load``, ``save``, ``hub``, ``tofile``,
       ``as_strided``, ``set_*`` ...); only function definitions and constant assignments at module level.
    2. Runtime: restricted builtins; ``torch`` is ``SAFE_TORCH``, an immutable allow-list proxy of pure tensor
       math, never the real module (which exposes importlib, ctypes and pickle through ordinary attribute
       chains); a line-execution budget (aborts infinite loops); inputs served as clones so in-place mutation
       cannot corrupt the caller's model.
    3. Output: must be a dict mapping exposed tensor names to finite floating tensors of the exact shape.

Limitation (by design, documented): this is an in-process, defense-in-depth sandbox as specified by the PRD,
not an OS isolation boundary. It cannot cap C-level memory or CPU time inside a single torch call. Run untrusted
models' scripts in a separate low-privilege process/container when the threat model requires it.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Mapping
from types import FrameType
from typing import Any

import torch

from .surgery_ops import biproject_direction, norm_preserving_column_edit

SANDBOX_FILENAME = "<weight_surgery_sandbox>"
MAX_SCRIPT_CHARS = 50_000
DEFAULT_MAX_LINE_EVENTS = 2_000_000

FORBIDDEN_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.ClassDef,
    ast.Match,  # class patterns perform attribute lookups by string name
)

# PRD prefix list (identifiers and attributes).
FORBIDDEN_ATTR_PREFIXES = (
    "__",
    "func_",
    "f_",
    "gi_",
    "cr_",
    "sys",
    "os",
    "subprocess",
    "open",
    "eval",
    "exec",
)

# Attribute names that reach I/O, pickle, raw memory, code generation, hooks or process-global state.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "load",
        "save",
        "hub",
        "from_file",
        "jit",
        "compile",
        "compiler",
        "ops",
        "classes",
        "utils",
        "distributed",
        "multiprocessing",
        "package",
        "serialization",
        "export",
        "onnx",
        "library",
        "backends",
        "fx",
        "overrides",
        "testing",
        "profiler",
        "autograd",
        "func",
        "storage",
        "untyped_storage",
        "typed_storage",
        "UntypedStorage",
        "TypedStorage",
        "share_memory_",
        "tofile",
        "dump",
        "dumps",
        "numpy",
        "as_strided",
        "as_strided_",
        "frombuffer",
        "from_dlpack",
        "to_dlpack",
        "data_ptr",
        "pin_memory",
        "register_hook",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "format",
        "format_map",
        "manual_seed",
        "seed",
        "initial_seed",
        "mro",
    }
)
FORBIDDEN_ATTRIBUTE_PREFIXES = ("_", "set_", "use_")

# Builtins that enable reflection/escape; they are not provided at runtime either, but rejecting them
# statically gives the author a clear policy error instead of a NameError.
FORBIDDEN_NAMES = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "compile",
        "type",
        "object",
        "breakpoint",
        "input",
        "help",
        "memoryview",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "print",
        "exit",
        "quit",
        "id",
    }
)

SAFE_BUILTINS: dict[str, Any] = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "enumerate": enumerate,
    "zip": zip,
    "int": int,
    "float": float,
    "bool": bool,
    "round": round,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "ValueError": ValueError,
}

_ALLOWED_TOP_LEVEL = (ast.FunctionDef, ast.Assign, ast.AnnAssign)


def _torch_internal_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """``__import__`` for C-level lookups only: ``torch.tensor`` & co. import torch via the caller's builtins.

    Scripts cannot reach this (``import`` statements and dunder names are rejected statically).
    """
    if name.split(".")[0] != "torch":
        raise ImportError(f"import of '{name}' is not allowed in the sandbox")
    return __import__(name, *args, **kwargs)


class SandboxError(Exception):
    """Base class for every sandbox failure."""


class SandboxSecurityViolation(SandboxError):
    """Raised when synthesized weight surgery code violates AST security or resource policies."""


class SandboxExecutionError(SandboxError):
    """Raised when a policy-compliant script fails to parse or raises while running."""


# Allow-lists for the ``torch`` object scripts receive. Deny-listing cannot work here: the real module exposes
# importlib, ctypes, builtins and pickle through dozens of attribute chains (``torch.importlib``,
# ``torch.cuda.tunable.shutil`` ...), so scripts only ever see these pure tensor-math members.
TORCH_ALLOWED = frozenset(
    """
    abs add addmm all allclose amax amin any arange argmax argmin argsort atan bmm cat ceil chunk clamp clip clone
    concat cos cosine_similarity count_nonzero cumsum diag diagonal div dot einsum eq exp eye flatten flip floor
    full full_like gather ge gt hstack index_select inner isfinite isinf isnan kron le lerp linspace log log1p
    logical_and logical_not logical_or logsumexp lt masked_fill matmul max maximum mean median min minimum mm mul
    mv ne neg nonzero norm numel ones ones_like outer permute pow prod reciprocal relu repeat_interleave reshape
    roll round rsqrt scatter sigmoid sign sin softmax sort split sqrt square squeeze stack std sub sum svd_lowrank
    tan tanh tensor tensordot tile topk trace transpose tril triu unbind unique unsqueeze var vstack where zeros
    zeros_like
    bfloat16 bool double float float16 float32 float64 half int int32 int64 long uint8
    e inf nan pi
    """.split()
)
FUNCTIONAL_ALLOWED = frozenset(
    "cosine_similarity gelu log_softmax normalize one_hot pad relu sigmoid silu softmax tanh".split()
)
LINALG_ALLOWED = frozenset(
    "cholesky det eigh eigvalsh inv lstsq matrix_norm matrix_rank norm pinv qr solve svd svdvals vector_norm".split()
)


class _SafeNamespace:
    """Immutable namespace exposing only allow-listed members; anything else is a policy violation."""

    def __init__(self, label: str, members: dict[str, Any]):
        self.__dict__.update(members)
        self.__dict__["_label"] = label

    def __getattr__(self, name: str) -> Any:
        raise SandboxSecurityViolation(f"{self.__dict__['_label']}.{name} is not available in the sandbox")

    def __setattr__(self, name: str, value: Any) -> None:
        raise SandboxSecurityViolation(f"{self.__dict__['_label']} is read-only in the sandbox")

    def __delattr__(self, name: str) -> None:
        raise SandboxSecurityViolation(f"{self.__dict__['_label']} is read-only in the sandbox")


def _pick(module: Any, names: frozenset[str]) -> dict[str, Any]:
    # hasattr: torch versions differ; a missing name just stays unavailable.
    return {n: getattr(module, n) for n in names if hasattr(module, n)}


SAFE_TORCH = _SafeNamespace(
    "torch",
    {
        **_pick(torch, TORCH_ALLOWED),
        "linalg": _SafeNamespace("torch.linalg", _pick(torch.linalg, LINALG_ALLOWED)),
        "surgery": _SafeNamespace(
            "torch.surgery",
            {
                "biproject_direction": biproject_direction,
                "norm_preserving_column_edit": norm_preserving_column_edit,
            },
        ),
        "nn": _SafeNamespace(
            "torch.nn",
            {
                "functional": _SafeNamespace(
                    "torch.nn.functional", _pick(torch.nn.functional, FUNCTIONAL_ALLOWED)
                )
            },
        ),
    },
)


class TensorScriptValidator(ast.NodeVisitor):
    """Verifies that a Python snippet contains only pure mathematical/tensor manipulations."""

    def generic_visit(self, node: ast.AST) -> None:
        """Rejects forbidden node types, then visits children."""
        if isinstance(node, FORBIDDEN_AST_NODES):
            raise SandboxSecurityViolation(f"Forbidden AST node type: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Rejects private, dunder and deny-listed attribute names."""
        attr = node.attr
        if (
            any(attr.startswith(p) for p in FORBIDDEN_ATTR_PREFIXES)
            or any(attr.startswith(p) for p in FORBIDDEN_ATTRIBUTE_PREFIXES)
            or attr in FORBIDDEN_ATTRIBUTES
        ):
            raise SandboxSecurityViolation(f"Forbidden attribute access: '{attr}'")
        # Early, clear rejection; the runtime proxy enforces the same rule for aliases and nested namespaces.
        if isinstance(node.value, ast.Name) and node.value.id == "torch" and attr not in SAFE_TORCH.__dict__:
            raise SandboxSecurityViolation(f"torch.{attr} is not available in the sandbox")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Rejects dunder/prefixed identifiers and reflective builtins."""
        if any(node.id.startswith(p) for p in FORBIDDEN_ATTR_PREFIXES) or node.id.startswith("_"):
            raise SandboxSecurityViolation(f"Forbidden identifier: '{node.id}'")
        if node.id in FORBIDDEN_NAMES:
            raise SandboxSecurityViolation(f"Forbidden builtin: '{node.id}'")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Rejects string constants that name dunder attributes (reflection payloads)."""
        if isinstance(node.value, str) and "__" in node.value:
            raise SandboxSecurityViolation("Forbidden dunder string constant")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        """Applies identifier rules to function parameter names."""
        if node.arg.startswith("_") or any(node.arg.startswith(p) for p in FORBIDDEN_ATTR_PREFIXES):
            raise SandboxSecurityViolation(f"Forbidden parameter name: '{node.arg}'")
        self.generic_visit(node)


def validate_script(script_code: str) -> ast.Module:
    """Parses and statically validates a surgery script.

    Returns:
        The parsed module.

    Raises:
        SandboxSecurityViolation: policy violation (including oversize scripts and missing ``compute_delta``).
        SandboxExecutionError: the text is not valid Python.
    """
    if not isinstance(script_code, str):
        raise SandboxSecurityViolation("Script must be a string.")
    if len(script_code) > MAX_SCRIPT_CHARS:
        raise SandboxSecurityViolation(f"Script exceeds {MAX_SCRIPT_CHARS} characters.")
    try:
        tree = ast.parse(script_code, filename=SANDBOX_FILENAME, mode="exec")
    except SyntaxError as exc:
        raise SandboxExecutionError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    for stmt in tree.body:
        is_docstring = isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
        if not (isinstance(stmt, _ALLOWED_TOP_LEVEL) or is_docstring):
            raise SandboxSecurityViolation(
                f"Only function definitions and assignments are allowed at module level, got {type(stmt).__name__}"
            )
    TensorScriptValidator().visit(tree)
    if not any(isinstance(s, ast.FunctionDef) and s.name == "compute_delta" for s in tree.body):
        raise SandboxSecurityViolation(
            "Synthesized script must define a callable `compute_delta(weights, diagnostics)`."
        )
    return tree


class CloningWeightView(Mapping[str, torch.Tensor]):
    """Read-only mapping that hands out detached clones of the underlying tensors on access.

    Only tensors the script actually reads are copied, so exposing every parameter of a large model costs
    nothing up front, and in-place edits by the script never reach the caller's tensors.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        self._tensors = tensors

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._tensors[key].detach().clone()

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, key: object) -> bool:
        return key in self._tensors

    def shape_of(self, key: str) -> torch.Size:
        """Shape of a tensor without cloning it (used for output validation)."""
        return self._tensors[key].shape


def _clone_diagnostics(value: Any, depth: int = 0) -> Any:
    """Deep-copies diagnostics, allowing only tensors, scalars, strings and containers of them."""
    if depth > 8:
        raise SandboxSecurityViolation("diagnostics nesting too deep")
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_clone_diagnostics(v, depth + 1) for v in value]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, Mapping):
        return {str(k): _clone_diagnostics(v, depth + 1) for k, v in value.items()}
    raise TypeError(f"diagnostics value of type {type(value).__name__} cannot be exposed to the sandbox")


class _LineBudget:
    """``sys.settrace`` hook that counts executed lines of sandbox code and aborts past a budget."""

    def __init__(self, max_events: int):
        self.max_events = max_events
        self.events = 0

    def global_trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename == SANDBOX_FILENAME:
            return self.local_trace
        return None

    def local_trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event == "line":
            self.events += 1
            if self.events > self.max_events:
                raise SandboxSecurityViolation(f"Execution budget exceeded ({self.max_events} line events)")
        return self.local_trace


def execute_weight_surgery_script(
    script_code: str,
    readonly_weights: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Any],
    max_line_events: int = DEFAULT_MAX_LINE_EVENTS,
) -> dict[str, torch.Tensor]:
    """Validates and executes a synthesized ``compute_delta(weights, diagnostics)`` function.

    Args:
        script_code: untrusted Python source.
        readonly_weights: tensors the script may read (served as clones).
        diagnostics: prober output etc. (deep-copied; tensors, scalars, strings and containers only).
        max_line_events: line-execution budget for the script's own code.

    Returns:
        Dictionary mapping parameter_name -> delta_tensor.

    Raises:
        SandboxSecurityViolation: static or runtime policy violation, malformed or non-finite output.
        SandboxExecutionError: syntax error or exception raised by the script.
    """
    tree = validate_script(script_code)
    weights = (
        readonly_weights
        if isinstance(readonly_weights, CloningWeightView)
        else CloningWeightView(readonly_weights)
    )
    safe_diagnostics = _clone_diagnostics(
        dict(diagnostics)
    )  # caller errors surface as TypeError, not script errors
    builtins_map = {**SAFE_BUILTINS, "__import__": _torch_internal_import}
    namespace: dict[str, Any] = {"__builtins__": builtins_map, "torch": SAFE_TORCH}
    budget = _LineBudget(max_line_events)
    previous_trace = sys.gettrace()
    sys.settrace(budget.global_trace)
    try:
        exec(compile(tree, filename=SANDBOX_FILENAME, mode="exec"), namespace)  # nosec B102 - AST-validated above; the sandbox's purpose
        compute_delta = namespace.get("compute_delta")
        if not callable(compute_delta):
            raise SandboxSecurityViolation("`compute_delta` is not callable.")
        with torch.no_grad():
            deltas = compute_delta(weights, safe_diagnostics)
    except SandboxError:
        raise
    except Exception as exc:  # script-raised errors are reported, never swallowed
        raise SandboxExecutionError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        sys.settrace(previous_trace)

    return _validate_output(deltas, weights)


def _validate_output(deltas: Any, weights: CloningWeightView) -> dict[str, torch.Tensor]:
    if not isinstance(deltas, dict):
        raise SandboxSecurityViolation("`compute_delta` must return a Dict[str, torch.Tensor].")
    validated: dict[str, torch.Tensor] = {}
    for k, v in deltas.items():
        if not isinstance(k, str) or k not in weights:
            raise SandboxSecurityViolation(f"Delta targets unknown or unexposed tensor '{k}'.")
        if not isinstance(v, torch.Tensor):
            raise SandboxSecurityViolation(f"Delta for '{k}' is not a torch.Tensor.")
        if not v.is_floating_point():
            raise SandboxSecurityViolation(f"Delta for '{k}' must be a floating-point tensor.")
        if v.shape != weights.shape_of(k):
            raise SandboxSecurityViolation(
                f"Delta for '{k}' has shape {tuple(v.shape)}, expected {tuple(weights.shape_of(k))}."
            )
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise SandboxSecurityViolation(f"Delta for '{k}' contains NaN or Inf values.")
        validated[k] = v.detach()
    return validated
