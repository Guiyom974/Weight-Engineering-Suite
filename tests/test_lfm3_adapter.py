"""LFM 3 hybrid adapter tests (PRD_02 section 4.1): liquid + attention blocks resolve accurately."""

import torch
import torch.nn as nn

from libweightspace.adapters import LFM3Adapter
from libweightspace.toy_models import RMSNorm, ToyConfig, ToyLFMBlock


def test_liquid_and_attention_blocks_resolve(tiny_lfm3):
    adapter = LFM3Adapter(tiny_lfm3)
    params = dict(tiny_lfm3.named_parameters())
    d = tiny_lfm3.config.hidden_size
    assert adapter.num_layers() == 3

    liquid = adapter.get_layer_anatomy(0)
    assert liquid.liquid_tau_proj == "model.layers.0.operator.tau_proj.weight"
    assert liquid.ssm_in_proj == "model.layers.0.operator.in_proj.weight"
    assert liquid.ssm_out_proj == "model.layers.0.operator.out_proj.weight"
    assert liquid.ssm_A_log == "model.layers.0.operator.A_log"
    assert liquid.attn_q_proj is None and liquid.attn_o_proj is None  # no false attention attribution
    assert liquid.mlp_down_proj == "model.layers.0.channel_mixer.down_proj.weight"
    assert params[liquid.ssm_in_proj].shape == (2 * d, d)
    assert params[liquid.ssm_A_log].shape == (d,)

    attn = adapter.get_layer_anatomy(1)
    assert attn.attn_q_proj == "model.layers.1.operator.q_proj.weight"
    assert attn.attn_k_proj == "model.layers.1.operator.k_proj.weight"
    assert attn.attn_v_proj == "model.layers.1.operator.v_proj.weight"
    assert attn.attn_o_proj == "model.layers.1.operator.out_proj.weight"
    assert attn.ssm_out_proj is None and attn.ssm_A_log is None and attn.liquid_tau_proj is None
    assert attn.mlp_gate_proj == "model.layers.1.channel_mixer.gate_proj.weight"

    assert adapter.get_layer_anatomy(2).ssm_A_log == "model.layers.2.operator.A_log"


def test_pole_stability(tiny_lfm3):
    adapter = LFM3Adapter(tiny_lfm3)
    poles = adapter.liquid_poles(0)
    assert poles is not None and bool((poles < 0).all())
    assert adapter.liquid_poles(1) is None
    assert adapter.pole_stability_ok(0) and adapter.pole_stability_ok(1)
    with torch.no_grad():
        tiny_lfm3.model.layers[0].operator.A_log[3] = float("nan")
    assert not adapter.pole_stability_ok(0)
    with torch.no_grad():
        tiny_lfm3.model.layers[2].operator.A_log[0] = -40.0  # decay rate exp(-40) ~ marginal stability
    assert not adapter.pole_stability_ok(2)


def test_stable_poles_keep_liquid_state_bounded(tiny_lfm3):
    """With poles in the left half-plane the per-step decay is in (0, 1): state is a convex mix of inputs."""
    op = tiny_lfm3.model.layers[0].operator
    x = torch.randn(1, 128, tiny_lfm3.config.hidden_size) * 3.0
    u, gate = op.in_proj(x).chunk(2, dim=-1)
    out = op(x, None, None)
    assert torch.isfinite(out).all()
    # |state| <= max|u| for a convex recurrence, so the pre-gate output is bounded by the input range.
    bound = u.abs().max() * torch.nn.functional.silu(gate).abs().max()
    hidden_norm_bound = op.out_proj.weight.abs().sum(dim=1).max() * bound
    assert out.abs().max() <= hidden_norm_bound + 1e-4


class _BlocksModel(nn.Module):
    """Alternate hierarchy from the PRD (``.blocks`` instead of ``.model.layers``)."""

    def __init__(self):
        super().__init__()
        cfg = ToyConfig(model_type="lfm3", use_qk_norm=False)
        self.blocks = nn.ModuleList([ToyLFMBlock(cfg, "liquid"), ToyLFMBlock(cfg, "attention")])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)


def test_blocks_hierarchy_supported():
    adapter = LFM3Adapter(_BlocksModel())
    assert adapter.num_layers() == 2
    assert adapter.get_layer_anatomy(0).ssm_A_log == "blocks.0.operator.A_log"
    assert adapter.get_layer_anatomy(1).attn_o_proj == "blocks.1.operator.out_proj.weight"
    assert isinstance(adapter.get_final_norm(), RMSNorm)
