"""Procedural generator for Second-Order Meta-Surgery tasks.

These puzzles train and benchmark the agent's ability to read, diagnose, and surgically edit neural weights:
each is a small MLP ``down_proj`` with a known corruption and an exact ground truth, so a proposed
``compute_delta`` script can be scored objectively (relative recovery of the clean outputs). The surgical
hit-rate on these puzzles is the second-order fitness W-RSI tries to raise across generations.

Challenge types:
    - ``planted_attractor``: a rank-1 parasitic direction added to 8 known columns (PRD reference task).
    - ``corrupted_kv_memory``: one associative memory (key -> value) overwritten with a wrong value.
    - ``spectral_explosion``: one singular direction inflated by a large factor.

``reference_solution(challenge_type)`` returns a correct script for each type (used by tests and as a
demonstration for template proposers). Scripts run only through the sandbox.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from libweightspace.sandbox import execute_weight_surgery_script

from .schemas import MetaGymTask

TENSOR = "mlp.down_proj.weight"
PASS_THRESHOLD = 0.85


@dataclass
class MetaSurgeryChallenge:
    """One puzzle: corrupted weights, diagnostics, and the ground truth used for scoring."""

    challenge_id: str
    challenge_type: str  # "planted_attractor" | "corrupted_kv_memory" | "spectral_explosion"
    readonly_weights: dict[str, torch.Tensor]
    diagnostics: dict[str, Any]
    prompt_description: str
    ground_truth_clean_W: torch.Tensor
    test_inputs: torch.Tensor
    expected_outputs: torch.Tensor
    seed: int = 0

    def task(self) -> MetaGymTask:
        """Serializable description (no tensors)."""
        return MetaGymTask(
            challenge_id=self.challenge_id,
            challenge_type=self.challenge_type,  # type: ignore[arg-type]
            seed=self.seed,
            prompt_description=self.prompt_description,
        )


class MetaSurgeryGym:
    """Procedurally generates synthetic weight-editing challenges to measure and sharpen meta-surgical skill."""

    CHALLENGE_TYPES = ("planted_attractor", "corrupted_kv_memory", "spectral_explosion")

    def __init__(self, d_model: int = 128, d_inter: int = 256):
        if d_inter < 231:
            raise ValueError("d_inter must be >= 231 (planted columns go up to index 230)")
        self.d_model = d_model
        self.d_inter = d_inter

    def _clean_mlp(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_gate = torch.randn(self.d_inter, self.d_model, generator=g) / (self.d_model**0.5)
        w_up = torch.randn(self.d_inter, self.d_model, generator=g) / (self.d_model**0.5)
        w_down = torch.randn(self.d_model, self.d_inter, generator=g) / (self.d_inter**0.5)
        return w_gate, w_up, w_down

    def _hidden(
        self, g: torch.Generator, w_gate: torch.Tensor, w_up: torch.Tensor, n: int = 16
    ) -> torch.Tensor:
        x = torch.randn(n, self.d_model, generator=g)
        return torch.nn.functional.silu(x @ w_gate.T) * (x @ w_up.T)

    def generate_planted_attractor_challenge(self, seed: int) -> MetaSurgeryChallenge:
        """Creates a synthetic MLP layer corrupted by a known rank-1 parasitic loop attractor."""
        g = torch.Generator().manual_seed(seed)
        w_gate, w_up, w_down_clean = self._clean_mlp(g)
        v_loop = torch.randn(self.d_model, generator=g)
        v_loop = v_loop / torch.norm(v_loop)
        corrupted_cols = [3, 17, 42, 88, 105, 144, 199, 230]
        w_down_corrupted = w_down_clean.clone()
        for c in corrupted_cols:
            w_down_corrupted[:, c] += 5.0 * v_loop
        hidden = self._hidden(g, w_gate, w_up)
        return MetaSurgeryChallenge(
            challenge_id=f"meta_attractor_{seed}",
            challenge_type="planted_attractor",
            readonly_weights={TENSOR: w_down_corrupted.clone()},
            diagnostics={"attractor_direction": v_loop, "suspected_columns": corrupted_cols},
            prompt_description=(
                "Given `weights['mlp.down_proj.weight']` of shape [d_model, d_inter] and "
                "`diagnostics['attractor_direction']` (unit vector v_loop), write `compute_delta(weights, diagnostics)` "
                "that removes the projection onto `v_loop` strictly from `diagnostics['suspected_columns']`."
            ),
            ground_truth_clean_W=w_down_clean,
            test_inputs=hidden,
            expected_outputs=hidden @ w_down_clean.T,
            seed=seed,
        )

    def generate_corrupted_kv_memory_challenge(self, seed: int) -> MetaSurgeryChallenge:
        """One stored association ``W k ~ u`` has been overwritten so that ``W k`` now returns ``u_bad``."""
        g = torch.Generator().manual_seed(seed)
        w_gate, w_up, w_down_clean = self._clean_mlp(g)
        key = torch.randn(self.d_inter, generator=g)
        key_unit = key / key.norm()
        u_bad = torch.randn(self.d_model, generator=g) * 4.0
        # Rank-1 corruption along the key: W_c k_hat = W k_hat + (u_bad - W k_hat).
        w_down_corrupted = w_down_clean + torch.outer(u_bad - w_down_clean @ key_unit, key_unit)
        hidden = self._hidden(g, w_gate, w_up)
        hidden = torch.cat([hidden, (key * 3.0)[None, :]], dim=0)  # the corrupted memory must be exercised
        return MetaSurgeryChallenge(
            challenge_id=f"meta_kv_{seed}",
            challenge_type="corrupted_kv_memory",
            readonly_weights={TENSOR: w_down_corrupted.clone()},
            diagnostics={"key_direction": key_unit, "correct_value": w_down_clean @ key_unit},
            prompt_description=(
                "The association for hidden key `diagnostics['key_direction']` (unit, [d_inter]) is corrupted. Write "
                "`compute_delta` returning a rank-1 update so that `W @ key_direction == diagnostics['correct_value']`."
            ),
            ground_truth_clean_W=w_down_clean,
            test_inputs=hidden,
            expected_outputs=hidden @ w_down_clean.T,
            seed=seed,
        )

    def generate_spectral_explosion_challenge(self, seed: int) -> MetaSurgeryChallenge:
        """The top singular direction of ``W`` was inflated by ``factor``; diagnostics expose that direction."""
        g = torch.Generator().manual_seed(seed)
        w_gate, w_up, w_down_clean = self._clean_mlp(g)
        u, s, vh = torch.linalg.svd(w_down_clean, full_matrices=False)
        factor = 20.0
        w_down_corrupted = w_down_clean + (factor - 1.0) * s[0] * torch.outer(u[:, 0], vh[0])
        hidden = self._hidden(g, w_gate, w_up)
        return MetaSurgeryChallenge(
            challenge_id=f"meta_spectral_{seed}",
            challenge_type="spectral_explosion",
            readonly_weights={TENSOR: w_down_corrupted.clone()},
            diagnostics={
                "left_vector": u[:, 0].clone(),
                "right_vector": vh[0].clone(),
                "target_sigma": float(s[0]),
            },
            prompt_description=(
                "One singular direction (left `diagnostics['left_vector']`, right `diagnostics['right_vector']`) "
                "has exploded. Write `compute_delta` that restores its singular value to `diagnostics['target_sigma']`."
            ),
            ground_truth_clean_W=w_down_clean,
            test_inputs=hidden,
            expected_outputs=hidden @ w_down_clean.T,
            seed=seed,
        )

    def generate(self, challenge_type: str, seed: int) -> MetaSurgeryChallenge:
        """Generates a challenge of ``challenge_type``.

        Raises:
            ValueError: unknown type.
        """
        generators = {
            "planted_attractor": self.generate_planted_attractor_challenge,
            "corrupted_kv_memory": self.generate_corrupted_kv_memory_challenge,
            "spectral_explosion": self.generate_spectral_explosion_challenge,
        }
        if challenge_type not in generators:
            raise ValueError(f"unknown challenge type '{challenge_type}'")
        return generators[challenge_type](seed)

    def sample(self, seed: int) -> MetaSurgeryChallenge:
        """Generates a challenge of a type chosen deterministically from ``seed``."""
        return self.generate(random.Random(seed).choice(self.CHALLENGE_TYPES), seed)

    def evaluate_agent_script(self, challenge: MetaSurgeryChallenge, script_code: str) -> tuple[bool, float]:
        """Executes the agent's proposed surgery script and computes relative recovery accuracy.

        Returns:
            ``(passed, improvement_ratio)`` with ``improvement = 1 - ||err_after|| / ||err_before||``.

        Raises:
            SandboxError: the script is rejected or fails (callers record it as a failed trial).
        """
        deltas = execute_weight_surgery_script(
            script_code=script_code,
            readonly_weights=challenge.readonly_weights,
            diagnostics=challenge.diagnostics,
        )
        w_corrupt = challenge.readonly_weights[TENSOR]
        w_patched = w_corrupt + deltas.get(TENSOR, torch.zeros_like(w_corrupt))
        err_before = torch.norm(challenge.test_inputs @ w_corrupt.T - challenge.expected_outputs)
        err_after = torch.norm(challenge.test_inputs @ w_patched.T - challenge.expected_outputs)
        improvement_ratio = float(1.0 - (err_after / (err_before + 1e-8)).item())
        return improvement_ratio >= PASS_THRESHOLD, improvement_ratio


REFERENCE_SOLUTIONS = {
    "planted_attractor": """def compute_delta(weights, diagnostics):
    W = weights["mlp.down_proj.weight"]
    v = diagnostics["attractor_direction"]
    v = v / (torch.norm(v) + 1e-8)
    delta = torch.zeros_like(W)
    for c in diagnostics["suspected_columns"]:
        delta[:, c] = -torch.dot(W[:, c], v) * v
    return {"mlp.down_proj.weight": delta}
""",
    "corrupted_kv_memory": """def compute_delta(weights, diagnostics):
    W = weights["mlp.down_proj.weight"]
    k = diagnostics["key_direction"]
    k = k / (torch.norm(k) + 1e-8)
    residual = diagnostics["correct_value"] - W @ k
    return {"mlp.down_proj.weight": torch.outer(residual, k)}
""",
    "spectral_explosion": """def compute_delta(weights, diagnostics):
    W = weights["mlp.down_proj.weight"]
    u = diagnostics["left_vector"]
    v = diagnostics["right_vector"]
    current = torch.dot(u, W @ v)
    return {"mlp.down_proj.weight": (diagnostics["target_sigma"] - current) * torch.outer(u, v)}
""",
}


def reference_solution(challenge_type: str) -> str:
    """A correct surgery script for ``challenge_type``.

    Raises:
        KeyError: unknown type.
    """
    return REFERENCE_SOLUTIONS[challenge_type]
