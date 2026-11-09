"""Mode 2 de novo synthesis tests (PRD_01 section 5.4, blueprint step 8)."""

import pytest
import torch
from safetensors.torch import save_file

from libweightspace.spectral import spectral_norm
from neuroforge.blueprint_compiler import BlueprintCompiler
from neuroforge.schemas import NeuralBlueprint

D = 32


def _rules(seed=0, n=4, dim=8):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, dim, generator=g)
    vals = torch.randn(n, dim, generator=g)
    return keys, vals


def _blueprint(layers, **overrides):
    data = dict(
        model_name="denovo",
        vocab_size=128,
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        subspaces=[
            {"name": "in", "start_dim": 0, "end_dim": 8, "description": "keys"},
            {"name": "out", "start_dim": 8, "end_dim": 16, "description": "values"},
        ],
        layers=layers,
    )
    data.update(overrides)
    return NeuralBlueprint.model_validate(data)


def _memory_blueprint(keys, vals):
    rules = [
        {
            "rule_id": f"r{i}",
            "input_subspace": "in",
            "output_subspace": "out",
            "key_vectors": [keys[i].tolist()],
            "value_vectors": [vals[i].tolist()],
        }
        for i in range(len(keys))
    ]
    return _blueprint([{"layer_idx": 0, "mlp_rules": rules}])


def test_compiled_memories_recall_values_with_zero_training():
    keys, vals = _rules()
    compiler = BlueprintCompiler(_memory_blueprint(keys, vals))
    model = compiler.build_model()
    b_out = compiler.subspace_basis("out")
    for i in range(4):
        x = compiler.embed_in_subspace("in", keys[i])[None, None, :]
        with torch.no_grad():
            h = model(inputs_embeds=x, output_hidden_states=True).hidden_states[-1][0, 0]
        recalled = b_out.T @ h
        cos = torch.nn.functional.cosine_similarity(recalled, vals[i], dim=0)
        assert float(cos) >= 0.98, f"rule {i}: cos={float(cos):.4f}"
        assert torch.linalg.vector_norm(recalled - vals[i]) / torch.linalg.vector_norm(vals[i]) < 0.05


def test_negative_control_without_rules_recalls_nothing():
    keys, vals = _rules()
    compiler = BlueprintCompiler(_blueprint([]))
    model = compiler.build_model()
    x = compiler.embed_in_subspace("in", keys[0])[None, None, :]
    with torch.no_grad():
        h = model(inputs_embeds=x, output_hidden_states=True).hidden_states[-1][0, 0]
    assert torch.linalg.vector_norm(compiler.subspace_basis("out").T @ h) < 1e-5


def test_subspaces_are_orthogonal_and_deterministic():
    keys, vals = _rules()
    c1 = BlueprintCompiler(_memory_blueprint(keys, vals), seed=7)
    c2 = BlueprintCompiler(_memory_blueprint(keys, vals), seed=7)
    assert torch.allclose(c1.basis.T @ c1.basis, torch.eye(D), atol=1e-5)
    assert torch.linalg.matrix_norm(c1.subspace_basis("in").T @ c1.subspace_basis("out")) < 1e-5
    s1, s2 = c1.compile(), c2.compile()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_previous_token_copy_head_routes_previous_value():
    bp = _blueprint(
        [
            {
                "layer_idx": 0,
                "attention_heads": [
                    {
                        "head_idx": 0,
                        "role": "previous_token_copy",
                        "query_subspace": "const",
                        "key_subspace": "const",
                        "value_subspace": "val",
                        "output_subspace": "out",
                        "routing_strength": 20.0,
                    }
                ],
            }
        ],
        subspaces=[
            {"name": "const", "start_dim": 0, "end_dim": 2, "description": "constant position anchor"},
            {"name": "val", "start_dim": 2, "end_dim": 10, "description": "token content"},
            {"name": "out", "start_dim": 10, "end_dim": 18, "description": "copied previous content"},
        ],
    )
    compiler = BlueprintCompiler(bp)
    model = compiler.build_model()
    g = torch.Generator().manual_seed(4)
    t = 6
    values = torch.randn(t, 8, generator=g)
    anchor = compiler.embed_in_subspace("const", torch.tensor([3.0, 0.0]))
    x = (anchor + compiler.embed_in_subspace("val", values))[None]
    with torch.no_grad():
        h = model(inputs_embeds=x, output_hidden_states=True).hidden_states[1][0]
    copied = h @ compiler.subspace_basis("out")  # [T, 8]
    for pos in range(1, t):
        cos_prev = torch.nn.functional.cosine_similarity(copied[pos], values[pos - 1], dim=0)
        cos_self = torch.nn.functional.cosine_similarity(copied[pos], values[pos], dim=0)
        assert float(cos_prev) > 0.9 and float(cos_prev) > float(cos_self)


def test_sink_head_is_silent_and_gqa_conflict_detected():
    sink = {
        "head_idx": 1,
        "role": "sink_head",
        "query_subspace": "in",
        "key_subspace": "in",
        "value_subspace": "in",
        "output_subspace": "out",
    }
    attn = BlueprintCompiler(_blueprint([{"layer_idx": 0, "attention_heads": [sink]}])).compile()
    assert not attn["model.layers.0.self_attn.o_proj.weight"].any()
    a = {**sink, "head_idx": 0, "role": "subspace_router"}
    b = {
        **sink,
        "head_idx": 1,
        "role": "subspace_router",
        "value_subspace": "out",
    }  # same kv group, other value
    with pytest.raises(ValueError, match="GQA group"):
        BlueprintCompiler(_blueprint([{"layer_idx": 0, "attention_heads": [a, b]}])).compile()


def test_rule_errors():
    keys, vals = _rules(n=4)
    too_many = _blueprint(
        [
            {
                "layer_idx": 0,
                "mlp_rules": [
                    {
                        "rule_id": "r",
                        "input_subspace": "in",
                        "output_subspace": "out",
                        "key_vectors": [[1.0] * 8] * 65,
                        "value_vectors": [[1.0] * 8] * 65,
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="exceed intermediate_size"):
        BlueprintCompiler(too_many).compile()
    wrong_dim = _blueprint(
        [
            {
                "layer_idx": 0,
                "mlp_rules": [
                    {
                        "rule_id": "r",
                        "input_subspace": "in",
                        "output_subspace": "out",
                        "key_vectors": [[1.0] * 3],
                        "value_vectors": [[1.0] * 8],
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="must equal subspace dims"):
        BlueprintCompiler(wrong_dim).compile()
    with pytest.raises(ValueError, match="ridge_lambda"):
        BlueprintCompiler(_memory_blueprint(keys, vals), ridge_lambda=0.0)
    with pytest.raises(KeyError):
        BlueprintCompiler(_memory_blueprint(keys, vals)).subspace_basis("missing")


def test_spectral_calibration_clamps():
    keys, vals = _rules()
    state = BlueprintCompiler(_memory_blueprint(keys * 10, vals * 10), spectral_gamma=0.5).compile()
    assert spectral_norm(state["model.layers.0.mlp.down_proj.weight"]) <= 0.5 + 1e-4


def test_hybrid_graft_copies_layer(tmp_path):
    keys, vals = _rules()
    donor = BlueprintCompiler(_memory_blueprint(keys, vals)).compile()
    ckpt = tmp_path / "donor.safetensors"
    save_file({k: v.contiguous() for k, v in donor.items() if k.startswith("model.layers.0.")}, str(ckpt))
    bp = _blueprint([{"layer_idx": 1, "graft_source_checkpoint": str(ckpt), "graft_source_layer": 0}])
    state = BlueprintCompiler(bp).compile()
    assert torch.equal(
        state["model.layers.1.mlp.down_proj.weight"], donor["model.layers.0.mlp.down_proj.weight"]
    )
    missing = _blueprint([{"layer_idx": 1, "graft_source_checkpoint": str(tmp_path / "nope.safetensors")}])
    with pytest.raises(FileNotFoundError):
        BlueprintCompiler(missing).compile()
    wrong_layer = _blueprint(
        [{"layer_idx": 1, "graft_source_checkpoint": str(ckpt), "graft_source_layer": 1}]
    )
    with pytest.raises(ValueError, match="found no tensors"):
        BlueprintCompiler(wrong_layer).compile()
