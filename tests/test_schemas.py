"""Tests for NeuroForge and W-RSI Pydantic schemas (blueprint step 2)."""

import pytest
from pydantic import ValidationError

from neuroforge.schemas import (
    EditConstraints,
    IntentSpec,
    InvariantGuard,
    MetricGoal,
    NeuralBlueprint,
    PatchManifest,
    ProbePair,
    SubspaceSlice,
)
from wrsi.schemas import TrialEntry, WRSIConfig


def _intent() -> IntentSpec:
    return IntentSpec(
        intent_id="fix-loop",
        description="Stop repetition loops",
        target_behavior="varied continuation",
        failure_behavior="repeats x",
        contrastive_probes=[
            ProbePair(
                probe_id="p1", prompt="ab x", expected_completion_prefix="y", undesired_completion_prefix="x"
            )
        ],
        target_goals=[
            MetricGoal(suite_name="target", metric_name="logit_margin", operator=">=", threshold=0.5)
        ],
        invariants=[InvariantGuard(suite_name="inv", metric_name="kl_divergence")],
    )


def test_intent_round_trip_and_defaults():
    intent = _intent()
    restored = IntentSpec.model_validate_json(intent.model_dump_json())
    assert restored == intent
    assert restored.constraints.max_modified_param_ratio == pytest.approx(0.0005)
    assert restored.constraints.enforce_nullspace_projection is True


def test_metric_goal_operators():
    assert MetricGoal(suite_name="s", metric_name="exact_match", operator=">=", threshold=0.5).is_satisfied(
        0.5
    )
    assert not MetricGoal(
        suite_name="s", metric_name="kl_divergence", operator="<=", threshold=0.1
    ).is_satisfied(0.2)
    assert MetricGoal(suite_name="s", metric_name="exact_match", operator="==", threshold=1.0).is_satisfied(
        1.0
    )


@pytest.mark.parametrize("drop", [0.0, -0.1, 0.11])
def test_invariant_guard_bounds(drop):
    with pytest.raises(ValidationError):
        InvariantGuard(suite_name="s", metric_name="m", max_allowed_drop=drop)


def test_edit_constraints_bounds():
    with pytest.raises(ValidationError):
        EditConstraints(max_modified_param_ratio=0.5)


def test_subspace_slice_rejects_inverted_range():
    with pytest.raises(ValidationError, match="end_dim must be strictly greater"):
        SubspaceSlice(name="a", start_dim=8, end_dim=8, description="")


def _blueprint(**overrides):
    data = dict(
        model_name="bp",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        subspaces=[
            {"name": "in", "start_dim": 0, "end_dim": 8, "description": "keys"},
            {"name": "out", "start_dim": 8, "end_dim": 16, "description": "values"},
        ],
        layers=[
            {
                "layer_idx": 0,
                "mlp_rules": [
                    {
                        "rule_id": "r",
                        "input_subspace": "in",
                        "output_subspace": "out",
                        "key_vectors": [[1.0] * 8],
                        "value_vectors": [[1.0] * 8],
                    }
                ],
            }
        ],
    )
    data.update(overrides)
    return NeuralBlueprint.model_validate(data)


def test_blueprint_valid():
    bp = _blueprint()
    assert bp.subspaces[1].dim == 8


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "subspaces": [
                    {"name": "in", "start_dim": 0, "end_dim": 10, "description": ""},
                    {"name": "out", "start_dim": 8, "end_dim": 16, "description": ""},
                ]
            },
            "overlap",
        ),
        (
            {"subspaces": [{"name": "in", "start_dim": 0, "end_dim": 40, "description": ""}]},
            "exceeds hidden_size",
        ),
        ({"layers": [{"layer_idx": 5}]}, "layer_idx 5 out of range"),
        ({"num_key_value_heads": 3}, "multiple of num_key_value_heads"),
        (
            {
                "layers": [
                    {
                        "layer_idx": 0,
                        "mlp_rules": [
                            {
                                "rule_id": "r",
                                "input_subspace": "nope",
                                "output_subspace": "out",
                                "key_vectors": [[1.0]],
                                "value_vectors": [[1.0]],
                            }
                        ],
                    }
                ]
            },
            "unknown subspace",
        ),
    ],
)
def test_blueprint_rejects_inconsistent(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _blueprint(**overrides)


def test_mlp_rule_requires_matching_counts():
    with pytest.raises(ValidationError, match="equal length"):
        _blueprint(
            layers=[
                {
                    "layer_idx": 0,
                    "mlp_rules": [
                        {
                            "rule_id": "r",
                            "input_subspace": "in",
                            "output_subspace": "out",
                            "key_vectors": [[1.0]],
                            "value_vectors": [],
                        }
                    ],
                }
            ]
        )


def test_patch_manifest_extension_fields_optional():
    manifest = PatchManifest(
        patch_id="abc",
        parent_patch_id=None,
        base_model_id="toy",
        base_model_sha256="0" * 64,
        intent_id="i",
        iteration=0,
        synthesized_code="",
        reasoner_rationale="",
        tensor_stats=[],
        total_modified_params=0,
        overall_sparsity_ratio=0.0,
        target_metrics_before={},
        target_metrics_after={},
        invariant_metrics_before={},
        invariant_metrics_after={},
        passed_gate=False,
    )
    assert manifest.delta_sha256 is None and manifest.created_at is None


def test_trial_entry_validation():
    entry = TrialEntry(
        trial_id="t",
        generation=0,
        domain="d",
        failure_taxonomy="attractor_loop",
        target_tensors=[],
        hypothesis="h",
        synthesized_code="",
        sparsity_ratio=0.0,
        frobenius_rel_norm=0.0,
        target_score_delta=0.0,
        holdout_kl_divergence=0.0,
        promoted=False,
    )
    assert entry.rejection_reason is None
    with pytest.raises(ValidationError):
        TrialEntry(**{**entry.model_dump(), "generation": -1})


def test_wrsi_config_bounds():
    assert WRSIConfig().meta_fraction == pytest.approx(0.3)
    with pytest.raises(ValidationError):
        WRSIConfig(candidates_per_generation=0)
