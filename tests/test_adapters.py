"""Adapter tests on tiny synthetic Qwen-3.8, Gemma 4 and DeepSeek MoE models (blueprint step 3)."""

import pytest
import torch

from libweightspace.adapters import (
    DeepSeekAdapter,
    Gemma4Adapter,
    LFM3Adapter,
    Qwen3Adapter,
    resolve_adapter,
)
from libweightspace.toy_models import (
    build_tiny_deepseek,
    build_tiny_gemma4,
    build_tiny_lfm3,
    build_tiny_qwen3,
)

FAMILIES = [
    (build_tiny_qwen3, Qwen3Adapter),
    (build_tiny_gemma4, Gemma4Adapter),
    (build_tiny_deepseek, DeepSeekAdapter),
    (build_tiny_lfm3, LFM3Adapter),
]


@pytest.mark.parametrize("factory, adapter_cls", FAMILIES)
def test_resolve_and_anatomy_paths_exist(factory, adapter_cls, tokenizer):
    model = factory(seed=0)
    adapter = resolve_adapter(model)
    assert isinstance(adapter, adapter_cls)
    params = dict(model.named_parameters())
    d = model.config.hidden_size
    for i in range(adapter.num_layers()):
        anatomy = adapter.get_layer_anatomy(i)
        present = anatomy.present()
        assert present, f"layer {i} resolved nothing"
        for value in present.values():
            for name in value if isinstance(value, list) else [value]:
                assert name in params, name
        # Every family has a residual-writing MLP/channel mixer whose output dim is d_model.
        assert params[anatomy.mlp_down_proj].shape[0] == d


@pytest.mark.parametrize("factory, _cls", FAMILIES)
def test_layer_module_hooks_fire_with_residual_shape(factory, _cls, tokenizer):
    model = factory(seed=0)
    adapter = resolve_adapter(model)
    seen = {}
    handles = [
        adapter.get_layer_module(i).register_forward_hook(
            lambda _m, _inp, out, i=i: seen.__setitem__(i, (out[0] if isinstance(out, tuple) else out).shape)
        )
        for i in range(adapter.num_layers())
    ]
    try:
        model(**tokenizer("hello there"))
    finally:
        for h in handles:
            h.remove()
    assert sorted(seen) == list(range(adapter.num_layers()))
    assert all(shape == (1, 11, model.config.hidden_size) for shape in seen.values())
    assert adapter.get_lm_head_weight().shape == (model.config.vocab_size, model.config.hidden_size)


def test_qwen_projection_shapes_and_qk_norm(tiny_qwen):
    adapter = Qwen3Adapter(tiny_qwen)
    cfg = tiny_qwen.config
    params = dict(tiny_qwen.named_parameters())
    a = adapter.get_layer_anatomy(0)
    assert params[a.attn_q_proj].shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    assert params[a.attn_k_proj].shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    assert params[a.mlp_down_proj].shape == (cfg.hidden_size, cfg.intermediate_size)
    q_inv, k_inv = adapter.qk_norm_inverse_scales(0)
    assert q_inv.shape == (cfg.head_dim,) and torch.allclose(q_inv, torch.ones(cfg.head_dim), atol=1e-5)


def test_qwen_logit_lens_matches_forward(tiny_qwen, tokenizer):
    adapter = Qwen3Adapter(tiny_qwen)
    out = tiny_qwen(**tokenizer("abc"), output_hidden_states=True)
    assert torch.allclose(adapter.logit_lens(out.hidden_states[-1]), out.logits, atol=1e-5)


def test_gemma_offset_softcap_and_sliding(tiny_gemma, tokenizer):
    adapter = Gemma4Adapter(tiny_gemma)
    out = tiny_gemma(**tokenizer("abcdef"), output_hidden_states=True)
    # logit_lens must reproduce the model's own (1 + w) norm and soft-capping exactly.
    assert torch.allclose(adapter.logit_lens(out.hidden_states[-1]), out.logits, atol=1e-5)
    raw = torch.linspace(-20, 20, 11)
    capped = 30.0 * torch.tanh(raw / 30.0)
    assert torch.allclose(adapter.invert_softcap(capped), raw, atol=1e-3)
    assert torch.allclose(adapter.effective_norm_weight(torch.zeros(3)), torch.ones(3))
    assert adapter.is_sliding_layer(0) is True and adapter.is_sliding_layer(1) is False


def test_deepseek_mla_and_moe_anatomy(tiny_deepseek):
    adapter = DeepSeekAdapter(tiny_deepseek)
    cfg = tiny_deepseek.config
    params = dict(tiny_deepseek.named_parameters())
    dense, moe = adapter.get_layer_anatomy(0), adapter.get_layer_anatomy(1)
    for a in (dense, moe):
        assert a.attn_k_proj is None and a.attn_v_proj is None
        assert params[a.mla_kv_a_proj].shape == (cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size)
        assert params[a.mla_kv_b_proj].shape[1] == cfg.kv_lora_rank
    assert dense.moe_router_gate is None and dense.mlp_down_proj.endswith("layers.0.mlp.down_proj.weight")
    assert params[moe.moe_router_gate].shape == (cfg.n_routed_experts, cfg.hidden_size)
    assert moe.moe_router_bias.endswith("mlp.gate.e_score_correction_bias")
    assert len(moe.moe_expert_down_projs) == cfg.n_routed_experts
    assert moe.moe_expert_down_projs[2].endswith("experts.2.down_proj.weight")
    assert "shared_experts" in moe.mlp_down_proj


def test_deepseek_expert_rerouting_changes_output(tiny_deepseek, tokenizer):
    adapter = DeepSeekAdapter(tiny_deepseek)
    inputs = tokenizer("route me please")
    before = tiny_deepseek(**inputs).logits.clone()
    delta = adapter.expert_bias_delta(1, expert_idx=0, shift=-100.0)
    ((name, d),) = delta.items()
    assert d.numel() == tiny_deepseek.config.n_routed_experts and int((d != 0).sum()) == 1
    with torch.no_grad():
        dict(tiny_deepseek.named_parameters())[name].add_(d)
    assert not torch.allclose(before, tiny_deepseek(**inputs).logits)
    with pytest.raises(ValueError, match="not an MoE layer"):
        adapter.expert_bias_delta(0, 0, 1.0)
    with pytest.raises(ValueError, match="out of range"):
        adapter.expert_bias_delta(1, 99, 1.0)


def test_resolve_adapter_errors(tiny_qwen):
    with pytest.raises(ValueError, match="unknown architecture"):
        resolve_adapter(tiny_qwen, arch="gpt2")
    tiny_qwen.config.model_type = "mamba"
    with pytest.raises(ValueError, match="cannot infer adapter"):
        resolve_adapter(tiny_qwen)
    assert isinstance(resolve_adapter(tiny_qwen, arch="gemma4"), Gemma4Adapter)


def test_adapter_rejects_model_without_layers():
    with pytest.raises(ValueError, match="could not locate the layer stack"):
        Qwen3Adapter(torch.nn.Linear(2, 2))


def test_lfm_factory_runs(tokenizer):
    model = build_tiny_lfm3(seed=1)
    assert model(**tokenizer("liquid")).logits.shape == (1, 6, model.config.vocab_size)
