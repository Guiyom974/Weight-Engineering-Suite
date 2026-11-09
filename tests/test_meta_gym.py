"""Meta-Surgery Gym tests (PRD_02 section 4.4, blueprint step 11)."""

import pytest

from libweightspace.sandbox import SandboxSecurityViolation
from wrsi.meta_gym import MetaSurgeryGym, reference_solution

PRD_SCRIPT = """
def compute_delta(weights, diagnostics):
    W = weights["mlp.down_proj.weight"]
    v = diagnostics["attractor_direction"]
    v = v / (torch.norm(v) + 1e-8)
    delta = torch.zeros_like(W)
    for c in diagnostics["suspected_columns"]:
        delta[:, c] = -torch.dot(W[:, c], v) * v
    return {"mlp.down_proj.weight": delta}
"""

NOOP = "def compute_delta(weights, diagnostics):\n    return {}\n"


@pytest.fixture
def gym():
    return MetaSurgeryGym()


def test_planted_attractor_solved_by_subtractive_surgery(gym):
    challenge = gym.generate_planted_attractor_challenge(seed=7)
    assert challenge.challenge_type == "planted_attractor"
    assert challenge.readonly_weights["mlp.down_proj.weight"].shape == (128, 256)
    passed, improvement = gym.evaluate_agent_script(challenge, PRD_SCRIPT)
    assert passed is True and improvement >= 0.95


@pytest.mark.parametrize("kind", MetaSurgeryGym.CHALLENGE_TYPES)
def test_reference_solutions_pass_and_noop_fails(gym, kind):
    challenge = gym.generate(kind, seed=3)
    passed, improvement = gym.evaluate_agent_script(challenge, reference_solution(kind))
    assert passed and improvement >= 0.95, (kind, improvement)
    passed, improvement = gym.evaluate_agent_script(challenge, NOOP)
    assert passed is False and improvement == pytest.approx(0.0, abs=1e-6)


def test_wrong_repair_does_not_pass(gym):
    """The attractor fix applied to a spectral-explosion puzzle must not score as a pass (no free credit)."""
    challenge = gym.generate("spectral_explosion", seed=1)
    challenge.diagnostics.update(
        {"attractor_direction": challenge.diagnostics["left_vector"], "suspected_columns": [0, 1, 2]}
    )
    passed, _ = gym.evaluate_agent_script(challenge, PRD_SCRIPT)
    assert passed is False


def test_script_cannot_mutate_challenge(gym):
    challenge = gym.generate_planted_attractor_challenge(seed=2)
    cheat = (
        "def compute_delta(weights, diagnostics):\n"
        "    W = weights['mlp.down_proj.weight']\n"
        "    W.zero_()\n"
        "    return {}\n"
    )
    before = challenge.readonly_weights["mlp.down_proj.weight"].clone()
    passed, _ = gym.evaluate_agent_script(challenge, cheat)
    assert passed is False and challenge.readonly_weights["mlp.down_proj.weight"].equal(before)


def test_sandbox_violations_propagate(gym):
    challenge = gym.generate_planted_attractor_challenge(seed=0)
    with pytest.raises(SandboxSecurityViolation):
        gym.evaluate_agent_script(
            challenge, "import os\ndef compute_delta(weights, diagnostics):\n    return {}"
        )


def test_sampling_is_deterministic_and_serializable(gym):
    a, b = gym.sample(11), gym.sample(11)
    assert a.challenge_id == b.challenge_id and a.readonly_weights["mlp.down_proj.weight"].equal(
        b.readonly_weights["mlp.down_proj.weight"]
    )
    task = a.task()
    assert task.challenge_type == a.challenge_type and task.seed == 11
    assert {gym.sample(s).challenge_type for s in range(30)} == set(MetaSurgeryGym.CHALLENGE_TYPES)
    with pytest.raises(ValueError):
        gym.generate("unknown", 0)
    with pytest.raises(ValueError):
        MetaSurgeryGym(d_inter=64)
