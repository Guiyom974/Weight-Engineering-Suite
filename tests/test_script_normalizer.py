"""Tests for normalize_llm_script: harmless torch imports become assignments, everything else is left to the sandbox.

Found in a live run on 2026-11-08: a Poe-hosted model diagnosed the planted defect correctly but both of its scripts
were rejected because they started with `import torch`.
"""

from __future__ import annotations

import pytest

from libweightspace.candidates import normalize_llm_script, parse_candidates_json
from libweightspace.sandbox import SandboxSecurityViolation, validate_script

BODY = "def compute_delta(weights, diagnostics):\n    return {}\n"


def test_import_torch_is_dropped_and_script_validates() -> None:
    script, notes = normalize_llm_script("import torch\n" + BODY)
    assert script.splitlines()[0].startswith("def compute_delta") and notes
    validate_script(script)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("import torch.nn.functional as F", "F = torch.nn.functional"),
        ("from torch import nn", "nn = torch.nn"),
        ("from torch.nn import functional as F", "F = torch.nn.functional"),
        ("import torch.nn as nn", "nn = torch.nn"),
    ],
)
def test_torch_aliases_become_assignments(line: str, expected: str) -> None:
    script, notes = normalize_llm_script(line + "\n" + BODY)
    assert script.splitlines()[0] == expected and notes
    validate_script(script)


@pytest.mark.parametrize("line", ["import os", "import numpy as np", "from subprocess import run", "import torch, os"])
def test_other_imports_are_left_for_the_validator_to_reject(line: str) -> None:
    script, notes = normalize_llm_script(line + "\n" + BODY)
    assert script.startswith(line) and not notes
    with pytest.raises(SandboxSecurityViolation):
        validate_script(script)


def test_imports_inside_the_function_body_are_rewritten() -> None:
    # Exactly what a Poe-hosted model produced in the live run: the import sits inside compute_delta.
    script = (
        "def compute_delta(weights, diagnostics):\n"
        "    import torch\n"
        "    import torch.nn.functional as F\n"
        "    return {}\n"
    )
    out, notes = normalize_llm_script(script)
    assert "import" not in out and "    F = torch.nn.functional" in out and len(notes) == 2
    validate_script(out)


def test_body_that_only_imports_keeps_valid_syntax() -> None:
    out, _ = normalize_llm_script("def helper():\n    import torch\n" + BODY)
    assert "    pass" in out
    validate_script(out)


def test_import_sharing_a_line_with_other_code_is_left_alone() -> None:
    script = "import torch; X = 1\n" + BODY
    out, notes = normalize_llm_script(script)
    assert out == script and not notes


def test_unparseable_scripts_are_returned_unchanged() -> None:
    assert normalize_llm_script("def broken(:\n") == ("def broken(:\n", [])


def test_parse_candidates_applies_the_normalizer() -> None:
    reply = '{"candidates": [{"hypothesis": "h", "python_script": "import torch\\n' + BODY.replace("\n", "\\n") + '"}]}'
    (candidate,) = parse_candidates_json(reply)
    assert "import" not in candidate.python_script
    validate_script(candidate.python_script)
