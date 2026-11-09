"""Red-team regressions for the sandbox.

A deny-list validator once let ``torch.importlib.import_module('subprocess')`` run. Scripts now receive an
allow-list proxy instead of the real ``torch`` module; these tests pin that property for the whole reachable
attribute graph, not just for known names.
"""

import types

import pytest
import torch

from libweightspace import sandbox
from libweightspace.sandbox import (
    SAFE_BUILTINS,
    SAFE_TORCH,
    CloningWeightView,
    SandboxError,
    SandboxSecurityViolation,
    execute_weight_surgery_script,
    validate_script,
)

WEIGHTS = {"w": torch.zeros(2, 2)}


def _script(body: str) -> str:
    return f"def compute_delta(weights, diagnostics):\n    {body}\n    return {{}}\n"


@pytest.mark.parametrize(
    "chain",
    [
        "torch.importlib.import_module('subprocess')",
        "torch.ctypes.CDLL('msvcrt')",
        "torch.builtins.open('x')",
        "torch.cuda.importlib.import_module('subprocess')",
        "torch.xpu.memory.pickle.loads(b'')",
        "torch.cuda.tunable.shutil.rmtree('x')",
        "torch.nn.modules.module.warnings",
        "torch.nn.functional.importlib",
        "torch.linalg.importlib",
        "torch.random.warnings",
        "torch.functional.importlib",
    ],
)
def test_module_escape_chains_are_blocked(chain):
    with pytest.raises(SandboxSecurityViolation):
        execute_weight_surgery_script(_script(f"x = {chain}"), WEIGHTS, {})


def test_subprocess_exploit_does_not_execute(tmp_path):
    marker = tmp_path / "pwned.txt"
    script = (
        "def compute_delta(weights, diagnostics):\n"
        "    m = torch.importlib.import_module('pathlib')\n"
        f"    m.Path({str(marker)!r}).write_text('x')\n"
        "    return {}\n"
    )
    with pytest.raises(SandboxError):
        execute_weight_surgery_script(script, WEIGHTS, {})
    assert not marker.exists()


def _allowed(attr: str) -> bool:
    return not (
        any(attr.startswith(p) for p in sandbox.FORBIDDEN_ATTR_PREFIXES)
        or any(attr.startswith(p) for p in sandbox.FORBIDDEN_ATTRIBUTE_PREFIXES)
        or attr in sandbox.FORBIDDEN_ATTRIBUTES
    )


def test_no_module_is_reachable_through_allowed_attributes():
    """Walks every attribute chain a script can write from the objects it can touch."""
    roots = [
        ("torch", SAFE_TORCH),
        ("tensor", torch.zeros(2, 2)),
        ("weights", CloningWeightView(WEIGHTS)),
        ("str", "text"),
        ("lambda", lambda: 0),
        *SAFE_BUILTINS.items(),
    ]
    seen: set[int] = set()
    stack = [(name, obj, 0) for name, obj in roots]
    visited = 0
    while stack:
        path, obj, depth = stack.pop()
        if id(obj) in seen or depth > 3 or isinstance(obj, (int, float, str, bytes, bool, type(None))):
            continue
        seen.add(id(obj))
        visited += 1
        for attr in dir(obj):
            if not _allowed(attr):
                continue
            try:
                value = getattr(obj, attr)
            except Exception:  # noqa: BLE001 - properties may raise; unreachable values are not an escape
                continue
            assert not isinstance(value, types.ModuleType), f"{path}.{attr} is a module"
            stack.append((f"{path}.{attr}", value, depth + 1))
    assert visited > 500  # the walk really covered the surface


def test_proxy_surface_covers_surgery_math():
    script = """
def compute_delta(weights, diagnostics):
    W = weights["w"]
    x = torch.nn.functional.silu(W) + torch.linalg.vector_norm(W)
    y = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    z = torch.tensor([1.0, 2.0], dtype=torch.float32) * torch.pi
    return {"w": y + z.to(W.dtype) * 0.0}
"""
    assert execute_weight_surgery_script(script, {"w": torch.ones(2)}, {})["w"].shape == (2,)


def test_unknown_torch_name_rejected_statically():
    with pytest.raises(SandboxSecurityViolation, match="not available"):
        validate_script(_script("x = torch.vmap"))


def test_proxy_is_immutable():
    with pytest.raises(SandboxSecurityViolation):
        execute_weight_surgery_script(_script("torch.zeros = 1"), WEIGHTS, {})
    assert callable(SAFE_TORCH.zeros)


def test_class_reflection_blocked():
    with pytest.raises(SandboxSecurityViolation, match="mro"):
        execute_weight_surgery_script(_script("x = int.mro()"), WEIGHTS, {})
