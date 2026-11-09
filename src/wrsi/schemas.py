"""Pydantic data contracts for W-RSI: trials, generation state, meta-gym tasks and runtime config.

``TrialEntry`` is what the scientist loop writes to the episodic ledger; ``TrialRecord`` is what the ledger
returns (adds the server-side timestamp). ``GenerationState`` is the persisted runtime state used by the CLI.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FailureTaxonomy = Literal[
    "attractor_loop",
    "routing_blindness",
    "missing_associative_rule",
    "expert_misrouting",
    "liquid_pole_instability",
]

FIRST_ORDER_DOMAINS = ("mathematical_deduction", "code_generation", "logical_deduction", "tool_execution")
META_DOMAIN = "meta_weight_space_surgery"


class TrialEntry(BaseModel):
    """One attempted self-edit, as recorded by the scientist loop."""

    trial_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=0)
    domain: str = Field(min_length=1, max_length=128)
    failure_taxonomy: str = Field(min_length=1, max_length=64)
    target_tensors: list[str]
    hypothesis: str
    synthesized_code: str
    sparsity_ratio: float = Field(ge=0.0)
    frobenius_rel_norm: float = Field(ge=0.0)
    target_score_delta: float
    holdout_kl_divergence: float = Field(ge=0.0)
    promoted: bool
    rejection_reason: str | None = None
    embedding: list[float] | None = None


class TrialRecord(TrialEntry):
    """A trial as stored in the ledger (adds the insertion timestamp)."""

    timestamp: float


class GenerationState(BaseModel):
    """Persisted W-RSI runtime state (``.wrsi/state.json``)."""

    model: str = "toy:qwen3_8"
    demo_defect: bool = True
    generation: int = Field(default=0, ge=0)
    steps_run: int = Field(default=0, ge=0)
    promoted_generations: list[int] = Field(default_factory=list)
    modified_tensors: list[str] = Field(default_factory=list)
    surgical_hit_rate: float = 0.0
    last_kl: float | None = None
    vault_sha256: str | None = None
    base_weights_sha256: str | None = None


class MetaGymTask(BaseModel):
    """Serializable description of a meta-surgery challenge (tensors stay in memory)."""

    challenge_id: str
    challenge_type: Literal["planted_attractor", "corrupted_kv_memory", "spectral_explosion"]
    seed: int
    prompt_description: str


class WRSIConfig(BaseModel):
    """Runtime knobs for the scientist loop and curriculum."""

    max_sparsity: float = Field(default=0.0002, gt=0.0, le=0.05)
    max_kl: float = Field(default=0.08, gt=0.0)
    candidates_per_generation: int = Field(default=3, ge=1, le=16)
    meta_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    nullspace_top_k: int = Field(default=2, ge=1, le=64)
    consolidation_interval: int = Field(default=10, ge=1)
    seed: int = 0
