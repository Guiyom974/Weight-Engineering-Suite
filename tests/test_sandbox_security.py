"""AST sandbox tests (PRD_01 section 5.5, blueprint step 5): valid surgery succeeds, escapes are rejected."""

import textwrap

import pytest
import torch

from libweightspace.sandbox import (
    CloningWeightView,
    SandboxExecutionError,
    SandboxSecurityViolation,
    execute_weight_surgery_script,
    validate_script,
)

NAME = "model.layers.1.mlp.down_proj.weight"

RANK1_SUBTRACTIVE = textwrap.dedent(
    """
    def compute_delta(weights, diagnostics):
        W = weights["model.layers.1.mlp.down_proj.weight"]
        v = diagnostics["attractor_direction"].to(W.device, W.dtype)
        v = v / (torch.norm(v) + 1e-8)
        cols = diagnostics["top_culpable_mlp_cols"][:64]
        delta = torch.zeros_like(W)
        for c in cols:
            proj = torch.dot(W[:, c], v) * v
            delta[:, c] = -1.0 * proj
        return {"model.layers.1.mlp.down_proj.weight": delta}
    """
)


@pytest.fixture
def weights():
    g = torch.Generator().manual_seed(0)
    return {NAME: torch.randn(8, 16, generator=g)}


@pytest.fixture
def diagnostics():
    v = torch.zeros(8)
    v[0] = 1.0
    return {"attractor_direction": v, "top_culpable_mlp_cols": [1, 4, 9]}


def test_valid_rank1_surgery(weights, diagnostics):
    deltas = execute_weight_surgery_script(RANK1_SUBTRACTIVE, weights, diagnostics)
    assert set(deltas) == {NAME}
    d = deltas[NAME]
    assert d.shape == weights[NAME].shape
    patched = weights[NAME] + d
    assert torch.allclose(patched[0, [1, 4, 9]], torch.zeros(3), atol=1e-6)  # projection on v removed
    assert torch.equal(patched[:, 0], weights[NAME][:, 0])  # untouched column


def test_helper_functions_can_call_each_other(weights, diagnostics):
    script = textwrap.dedent(
        """
        SCALE = 0.5

        def helper(W):
            return torch.zeros_like(W) + SCALE

        def compute_delta(weights, diagnostics):
            return {"model.layers.1.mlp.down_proj.weight": helper(weights["model.layers.1.mlp.down_proj.weight"])}
        """
    )
    assert torch.allclose(
        execute_weight_surgery_script(script, weights, diagnostics)[NAME], torch.full((8, 16), 0.5)
    )


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("import os\ndef compute_delta(weights, diagnostics):\n    return {}", "Import"),
        (
            "def compute_delta(weights, diagnostics):\n    e = __builtins__['eval']\n    return {}",
            "__builtins__",
        ),
        (
            "def compute_delta(weights, diagnostics):\n    d = getattr(torch, '__dict__')\n    return {}",
            "getattr",
        ),
        ("def compute_delta(weights, diagnostics):\n    torch.save(weights, 'x.pt')\n    return {}", "save"),
        ("def compute_delta(weights, diagnostics):\n    return torch.load('evil.pt')", "load"),
        ("def compute_delta(weights, diagnostics):\n    torch.hub.load('a', 'b')\n    return {}", "load|hub"),
        ("def compute_delta(weights, diagnostics):\n    w = weights['a'].numpy()\n    return {}", "numpy"),
        (
            "def compute_delta(weights, diagnostics):\n    w = weights['a'].tofile('f')\n    return {}",
            "tofile",
        ),
        (
            "def compute_delta(weights, diagnostics):\n    w = torch.as_strided(weights['a'], (9,), (99,))\n"
            "    return {}",
            "as_strided",
        ),
        ("def compute_delta(weights, diagnostics):\n    c = weights._tensors\n    return {}", "_tensors"),
        (
            "def compute_delta(weights, diagnostics):\n    s = '{0.x}'.format(weights)\n    return {}",
            "format",
        ),
        (
            "def compute_delta(weights, diagnostics):\n    torch.set_default_dtype(torch.float64)\n    return {}",
            "set_default_dtype",
        ),
        (
            "def compute_delta(weights, diagnostics):\n    try:\n        pass\n    except Exception:\n        pass\n"
            "    return {}",
            "Try",
        ),
        ("class X:\n    pass\ndef compute_delta(weights, diagnostics):\n    return {}", "ClassDef"),
        (
            "def compute_delta(weights, diagnostics):\n    match weights:\n        case _:\n            pass\n"
            "    return {}",
            "Match",
        ),
        ("def compute_delta(weights, diagnostics):\n    t = type(weights)\n    return {}", "type"),
        (
            "for i in range(3):\n    pass\ndef compute_delta(weights, diagnostics):\n    return {}",
            "module level",
        ),
        ("def other(weights, diagnostics):\n    return {}", "must define a callable"),
        ("def compute_delta(weights, diagnostics):\n    yield 1", "Yield"),
    ],
)
def test_policy_violations_rejected(body, fragment, weights, diagnostics):
    with pytest.raises(SandboxSecurityViolation, match=fragment):
        execute_weight_surgery_script(body, weights, diagnostics)


def test_nan_output_rejected(weights, diagnostics):
    script = (
        "def compute_delta(weights, diagnostics):\n"
        "    W = weights['model.layers.1.mlp.down_proj.weight']\n"
        "    return {'model.layers.1.mlp.down_proj.weight': torch.zeros_like(W) * float('nan')}"
    )
    with pytest.raises(SandboxSecurityViolation, match="NaN or Inf"):
        execute_weight_surgery_script(script, weights, diagnostics)


@pytest.mark.parametrize(
    "ret, fragment",
    [
        ("[1, 2]", "must return a Dict"),
        ("{'unknown.weight': torch.zeros(2)}", "unknown or unexposed"),
        ("{'model.layers.1.mlp.down_proj.weight': 3.0}", "not a torch.Tensor"),
        ("{'model.layers.1.mlp.down_proj.weight': torch.zeros(2, 2)}", "has shape"),
        ("{'model.layers.1.mlp.down_proj.weight': torch.zeros(8, 16, dtype=torch.long)}", "floating-point"),
    ],
)
def test_malformed_outputs_rejected(ret, fragment, weights, diagnostics):
    script = f"def compute_delta(weights, diagnostics):\n    return {ret}"
    with pytest.raises(SandboxSecurityViolation, match=fragment):
        execute_weight_surgery_script(script, weights, diagnostics)


def test_infinite_loop_hits_budget(weights, diagnostics):
    script = "def compute_delta(weights, diagnostics):\n    while True:\n        x = 1\n    return {}"
    with pytest.raises(SandboxSecurityViolation, match="Execution budget exceeded"):
        execute_weight_surgery_script(script, weights, diagnostics, max_line_events=10_000)


def test_in_place_mutation_does_not_reach_caller(weights, diagnostics):
    before = weights[NAME].clone()
    v_before = diagnostics["attractor_direction"].clone()
    script = (
        "def compute_delta(weights, diagnostics):\n"
        "    W = weights['model.layers.1.mlp.down_proj.weight']\n"
        "    W.zero_()\n"
        "    W[:, 0] = 5.0\n"
        "    diagnostics['attractor_direction'].fill_(9.0)\n"
        "    return {'model.layers.1.mlp.down_proj.weight': torch.zeros_like(W)}"
    )
    execute_weight_surgery_script(script, weights, diagnostics)
    assert torch.equal(weights[NAME], before)
    assert torch.equal(diagnostics["attractor_direction"], v_before)


def test_runtime_and_syntax_errors_are_execution_errors(weights, diagnostics):
    with pytest.raises(SandboxExecutionError, match="SyntaxError"):
        execute_weight_surgery_script(
            "def compute_delta(weights, diagnostics)\n    return {}", weights, diagnostics
        )
    with pytest.raises(SandboxExecutionError, match="KeyError"):
        execute_weight_surgery_script(
            "def compute_delta(weights, diagnostics):\n    return {'x': weights['missing']}",
            weights,
            diagnostics,
        )
    with pytest.raises(SandboxExecutionError, match="NameError"):
        execute_weight_surgery_script(
            "def compute_delta(weights, diagnostics):\n    return math.pi", weights, diagnostics
        )


def test_oversized_script_rejected():
    with pytest.raises(SandboxSecurityViolation, match="exceeds"):
        validate_script("x = 1\n" * 20_000)


def test_unsupported_diagnostics_type_rejected(weights):
    with pytest.raises(TypeError, match="cannot be exposed"):
        execute_weight_surgery_script(RANK1_SUBTRACTIVE, weights, {"obj": object()})


def test_previous_tracer_restored(weights, diagnostics):
    import sys

    marker = sys.gettrace()
    execute_weight_surgery_script(RANK1_SUBTRACTIVE, weights, diagnostics)
    assert sys.gettrace() is marker


def test_cloning_view_is_lazy_and_isolated(weights):
    view = CloningWeightView(weights)
    assert NAME in view and len(view) == 1 and list(view) == [NAME]
    clone = view[NAME]
    clone.add_(1.0)
    assert not torch.equal(clone, weights[NAME])
