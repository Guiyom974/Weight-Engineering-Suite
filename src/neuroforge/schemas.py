"""Core data contracts for NeuroForge intent specifications, blueprints, diagnostics, and patches.

These Pydantic v2 models are the single source of truth across NeuroForge modules (PRD_01 section 3).
Additions beyond the PRD are backward compatible and marked as such:
    - ``PatchManifest.delta_sha256`` / ``created_at``: integrity hash of the stored sparse delta object.
    - ``NeuralBlueprint`` cross-field validation (subspace bounds, overlaps, references).
    - ``WorkspaceConfig``: persisted per-workspace model selection used by the CLI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class OperationMode(str, Enum):
    """NeuroForge operating mode."""

    BASE_EDIT = "base_edit"
    FROM_SCRATCH = "from_scratch"
    HYBRID = "hybrid"


class ArchitectureType(str, Enum):
    """Supported architecture families."""

    QWEN_3_8 = "qwen3_8"
    GEMMA_4 = "gemma4"
    DEEPSEEK_MOE = "deepseek_moe"
    LFM_3 = "lfm3"
    CUSTOM_BLUEPRINT = "custom_blueprint"


class ProbePair(BaseModel):
    """Contrastive prompt pair used to isolate behavioral circuits."""

    probe_id: str
    prompt: str
    expected_completion_prefix: str
    undesired_completion_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricGoal(BaseModel):
    """Target metric evaluation criterion."""

    suite_name: str
    metric_name: Literal["exact_match", "pass_at_1", "repetition_rate", "kl_divergence", "logit_margin"]
    operator: Literal[">=", "<=", "=="]
    threshold: float
    weight: float = 1.0

    def is_satisfied(self, value: float, tol: float = 1e-9) -> bool:
        """Returns True when ``value`` meets this goal's operator/threshold."""
        if self.operator == ">=":
            return value >= self.threshold - tol
        if self.operator == "<=":
            return value <= self.threshold + tol
        return abs(value - self.threshold) <= max(tol, 1e-6)


class InvariantGuard(BaseModel):
    """Non-regression constraint on baseline model capabilities."""

    suite_name: str
    metric_name: str
    max_allowed_drop: float = Field(gt=0.0, le=0.10, default=0.01)


class EditConstraints(BaseModel):
    """Mathematical bounds enforced on every weight-space intervention."""

    max_modified_param_ratio: float = Field(default=0.0005, gt=0.0, le=0.05)
    max_frobenius_rel_norm: float = Field(default=0.02, gt=0.0, le=0.20)
    max_spectral_norm_shift: float = Field(default=0.015, gt=0.0, le=0.10)
    min_post_edit_alpha: float | None = Field(default=None, ge=1.0, le=20.0)
    allowed_module_patterns: list[str] = Field(
        default_factory=lambda: [
            "*mlp.down_proj*",
            "*self_attn.o_proj*",
            "*mlp.gate_proj*",
            "*feed_forward.w2*",
            "*self_attn.out_proj*",
            "*conv.out_proj*",
        ]
    )
    enforce_nullspace_projection: bool = True


class IntentSpec(BaseModel):
    """Complete Intent Specification for Mode 1 and Mode 3."""

    intent_id: str
    description: str
    target_behavior: str
    failure_behavior: str
    contrastive_probes: list[ProbePair]
    target_goals: list[MetricGoal]
    invariants: list[InvariantGuard]
    constraints: EditConstraints = Field(default_factory=EditConstraints)


# --- Mode 2: De Novo Blueprint Schemas ---


class SubspaceSlice(BaseModel):
    """Named orthogonal subspace slice in the residual stream."""

    name: str
    start_dim: int = Field(ge=0)
    end_dim: int = Field(gt=0)
    description: str

    @field_validator("end_dim")
    @classmethod
    def validate_range(cls, v: int, info: ValidationInfo) -> int:
        """Rejects empty or inverted slices."""
        if "start_dim" in info.data and v <= info.data["start_dim"]:
            raise ValueError("end_dim must be strictly greater than start_dim")
        return v

    @property
    def dim(self) -> int:
        """Number of residual dimensions in this slice."""
        return self.end_dim - self.start_dim


class AttentionHeadBlueprint(BaseModel):
    """Declarative specification of a single attention head's behavior."""

    head_idx: int
    role: Literal["previous_token_copy", "induction_head", "bracket_matcher", "subspace_router", "sink_head"]
    query_subspace: str
    key_subspace: str
    value_subspace: str
    output_subspace: str
    positional_offset: int | None = None
    routing_strength: float = 10.0


class MLPMemoryRule(BaseModel):
    """Declarative key-value associative rule compiled into MLP weights."""

    rule_id: str
    input_subspace: str
    output_subspace: str
    key_vectors: list[list[float]]  # Shape: [N_rules, dim_in_slice]
    value_vectors: list[list[float]]  # Shape: [N_rules, dim_out_slice]
    gating_threshold: float = 0.5

    @model_validator(mode="after")
    def _check_rule_counts(self) -> MLPMemoryRule:
        if len(self.key_vectors) != len(self.value_vectors):
            raise ValueError(f"rule '{self.rule_id}': key_vectors and value_vectors must have equal length")
        if not self.key_vectors:
            raise ValueError(f"rule '{self.rule_id}': at least one key/value pair is required")
        return self


class LayerBlueprint(BaseModel):
    """Specification for a single de novo layer."""

    layer_idx: int
    attention_heads: list[AttentionHeadBlueprint] = Field(default_factory=list)
    mlp_rules: list[MLPMemoryRule] = Field(default_factory=list)
    graft_source_checkpoint: str | None = None
    graft_source_layer: int | None = None


class DonorGraftSpec(BaseModel):
    """Specification for transplanting donor layers or embeddings with Orthogonal Procrustes alignment (NF-BL-08)."""

    checkpoint_path: str
    donor_layers: list[int] = Field(default_factory=list)
    target_layers: list[int] = Field(default_factory=list)
    transplant_embeddings: bool = True
    procrustes_align: bool = True

    @model_validator(mode="after")
    def _validate_layer_pairs(self) -> DonorGraftSpec:
        if len(self.donor_layers) != len(self.target_layers):
            raise ValueError("donor_layers and target_layers must have equal length")
        return self


class NeuralBlueprint(BaseModel):
    """Top-level schema for Mode 2: Starting from Scratch from a Blueprint."""

    model_name: str
    architecture_type: ArchitectureType = ArchitectureType.QWEN_3_8
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    subspaces: list[SubspaceSlice]
    layers: list[LayerBlueprint]
    donor_graft: DonorGraftSpec | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> NeuralBlueprint:
        # Cross-field checks catch blueprint typos before any tensor is allocated.
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        names = [s.name for s in self.subspaces]
        if len(set(names)) != len(names):
            raise ValueError("subspace names must be unique")
        ordered = sorted(self.subspaces, key=lambda s: s.start_dim)
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            if cur.start_dim < prev.end_dim:
                raise ValueError(f"subspaces '{prev.name}' and '{cur.name}' overlap")
        for s in self.subspaces:
            if s.end_dim > self.hidden_size:
                raise ValueError(f"subspace '{s.name}' exceeds hidden_size={self.hidden_size}")
        known = set(names)
        for layer in self.layers:
            if not 0 <= layer.layer_idx < self.num_hidden_layers:
                raise ValueError(f"layer_idx {layer.layer_idx} out of range")
            for head in layer.attention_heads:
                if not 0 <= head.head_idx < self.num_attention_heads:
                    raise ValueError(f"head_idx {head.head_idx} out of range in layer {layer.layer_idx}")
                for ref in (
                    head.query_subspace,
                    head.key_subspace,
                    head.value_subspace,
                    head.output_subspace,
                ):
                    if ref not in known:
                        raise ValueError(f"unknown subspace '{ref}' in layer {layer.layer_idx}")
            for rule in layer.mlp_rules:
                for ref in (rule.input_subspace, rule.output_subspace):
                    if ref not in known:
                        raise ValueError(f"unknown subspace '{ref}' in rule '{rule.rule_id}'")
        return self


class TensorDeltaStats(BaseModel):
    """Audit metrics for a single modified parameter tensor."""

    tensor_name: str
    shape: list[int]
    nnz_count: int
    total_elements: int
    sparsity_ratio: float
    frobenius_norm_delta: float
    frobenius_norm_base: float
    relative_frobenius: float
    spectral_norm_before: float
    spectral_norm_after: float


class PatchManifest(BaseModel):
    """Cryptographically verifiable manifest emitted with every weight patch."""

    patch_id: str
    parent_patch_id: str | None
    base_model_id: str
    base_model_sha256: str
    intent_id: str
    iteration: int
    synthesized_code: str
    reasoner_rationale: str
    tensor_stats: list[TensorDeltaStats]
    total_modified_params: int
    overall_sparsity_ratio: float
    target_metrics_before: dict[str, float]
    target_metrics_after: dict[str, float]
    invariant_metrics_before: dict[str, float]
    invariant_metrics_after: dict[str, float]
    passed_gate: bool
    # Extension (backward compatible): SHA-256 of the stored COO object and commit time.
    delta_sha256: str | None = None
    created_at: float | None = None


class WorkspaceConfig(BaseModel):
    """Per-workspace settings persisted by ``neuroforge init`` (``neuroforge.yaml``)."""

    model: str = "toy:qwen3_8"
    device: str = "cpu"
    demo_defect: bool = False
    # Known-good checkpoint for KL invariants and nullspace directions (None: the model at loop start).
    reference_model: str | None = None
    # Optional LLM endpoint: base_url, api_key_env (the variable NAME) and model. Keys never live here;
    # libweightspace.llm_config rejects an ``api_key`` entry.
    llm: dict[str, str] | None = None
    # Extension (backward compatible, CR2): where the weights are loaded from and where they came from. ``model`` stays
    # the public id written into manifests and bundles; ``model_path`` is the local folder actually loaded (inside the
    # Studio's models folder), so no local path ends up in exported files.
    model_path: str | None = None
    source: str | None = None
    revision: str | None = None
    license: str | None = None

    @property
    def load_spec(self) -> str:
        """The spec passed to ``load_model``: the local copy when there is one, else ``model``."""
        return self.model_path or self.model
