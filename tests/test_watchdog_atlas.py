"""Watchdog and Self-Anatomy Atlas tests."""

import torch

from libweightspace.adapters import DeepSeekAdapter, LFM3Adapter, Qwen3Adapter
from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.nullspace import NullspaceLocker
from wrsi.anatomy_atlas import SelfAnatomyAtlas
from wrsi.episodic_ledger import EpisodicWeightLedger
from wrsi.scientist_agent import WRSIScientistLoop
from wrsi.watchdog import WeightWatchdog


def test_healthy_models_pass(tiny_qwen, tiny_lfm3, tokenizer):
    probe = tokenizer("hello")["input_ids"]
    assert WeightWatchdog(Qwen3Adapter(tiny_qwen), probe_input_ids=probe).inspect(tiny_qwen).ok
    assert WeightWatchdog(LFM3Adapter(tiny_lfm3), probe_input_ids=probe).inspect(tiny_lfm3).ok


def test_detects_non_finite_and_exploding_weights(tiny_qwen):
    dog = WeightWatchdog(Qwen3Adapter(tiny_qwen), max_abs_weight=100.0)
    with torch.no_grad():
        tiny_qwen.model.layers[0].mlp.down_proj.weight[0, 0] = float("inf")
        tiny_qwen.model.layers[1].mlp.up_proj.weight[0, 0] = 500.0
    report = dog.inspect(tiny_qwen)
    assert not report.ok and any("non-finite" in v for v in report.violations)
    assert any("exceeds 100.0" in v for v in report.violations)


def test_detects_liquid_pole_instability(tiny_lfm3):
    with torch.no_grad():
        tiny_lfm3.model.layers[0].operator.A_log[2] = -50.0
    report = WeightWatchdog(LFM3Adapter(tiny_lfm3)).inspect(tiny_lfm3)
    assert report.violations == ["layer 0: liquid pole instability"]


def test_detects_residual_explosion(tiny_qwen, tokenizer):
    with torch.no_grad():
        tiny_qwen.model.layers[0].mlp.down_proj.weight.mul_(5000.0)
    dog = WeightWatchdog(
        Qwen3Adapter(tiny_qwen),
        max_abs_weight=1e9,
        probe_input_ids=tokenizer("hi there")["input_ids"],
        max_residual_norm=1e3,
    )
    assert any("residual norm" in v for v in dog.inspect(tiny_qwen).violations)


def test_guard_rolls_back(tiny_qwen, tmp_path):
    loop = WRSIScientistLoop(
        tiny_qwen, CoWWeightOverlay(tiny_qwen), NullspaceLocker(), EpisodicWeightLedger(tmp_path / "l.db")
    )
    before = {k: v.clone() for k, v in tiny_qwen.state_dict().items()}
    name = "model.layers.1.mlp.down_proj.weight"
    from libweightspace.cow_overlay import apply_delta_with_backup
    from wrsi.scientist_agent import PromotedGeneration

    delta = torch.zeros_like(dict(tiny_qwen.named_parameters())[name])
    delta[0, 0] = 1e6
    backup = apply_delta_with_backup(dict(tiny_qwen.named_parameters())[name], name, delta)
    loop.rollback_stack.append(PromotedGeneration(0, "t", [name], [backup], {}))
    loop.generation = 1
    report = WeightWatchdog(Qwen3Adapter(tiny_qwen)).guard(loop)
    assert not report.ok and loop.generation == 0
    assert all(torch.equal(before[k], v) for k, v in tiny_qwen.state_dict().items())


def test_atlas_kinds_and_rows(tiny_lfm3, tiny_deepseek):
    lfm = SelfAnatomyAtlas(tiny_lfm3, LFM3Adapter(tiny_lfm3))
    profiles = lfm.build(top_k=4)
    assert [p.kind for p in profiles] == ["liquid", "attention", "liquid"]
    assert "ssm_out_proj" in profiles[0].tensors and "attn_o_proj" in profiles[1].tensors
    assert all(p.param_count > 0 for p in profiles)
    rows = lfm.to_rows(profiles)
    assert {r["field"] for r in rows} >= {"mlp_down_proj", "ssm_out_proj", "attn_o_proj"}
    assert "L0 [liquid" in lfm.summarize(profiles)

    ds = SelfAnatomyAtlas(tiny_deepseek, DeepSeekAdapter(tiny_deepseek)).build()
    assert [p.kind for p in ds] == ["attention", "moe"]
    assert "moe_router_gate" in ds[1].tensors
