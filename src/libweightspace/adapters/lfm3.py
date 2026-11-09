"""Architecture adapter for Liquid Foundation Model 3 (LFM 3) and hybrid state-space backbones.

LFM 3 interleaves liquid/state-space operator blocks (``operator.tau_proj``, ``operator.A_log``,
``operator.in_proj``, ``operator.out_proj``) with attention blocks (``operator.q_proj`` ... ``operator.out_proj``)
and gated channel mixers. Because both block kinds name their output projection ``operator.out_proj``, the
anatomy is disambiguated by block kind (PRD code reported it as both ``attn_o_proj`` and ``ssm_out_proj``).

Pole stability invariant: continuous-time eigenvalues ``lambda_i = -exp(A_log_i)`` must stay strictly negative;
``liquid_poles`` / ``pole_stability_ok`` expose this for the watchdog.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import DecoderAdapter, LayerAnatomy


class LFM3Adapter(DecoderAdapter):
    """Provides uniform parameter and activation hook access for LFM 3 hybrid blocks."""

    family = "lfm3"

    def is_attention_block(self, layer_idx: int) -> bool:
        """True when the block's operator is attention (has a query projection)."""
        return self.find_param(layer_idx, "self_attn.q_proj.weight", "operator.q_proj.weight") is not None

    def get_layer_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """Resolves attention or liquid operator projections plus channel-mixer projections."""
        f = self.find_param
        attention = self.is_attention_block(layer_idx)
        mixer: dict[str, Any] = dict(
            mlp_gate_proj=f(
                layer_idx,
                "mlp.gate_proj.weight",
                "channel_mixer.gate_proj.weight",
                "ffn.w1.weight",
                "feed_forward.w1.weight",
            ),
            mlp_up_proj=f(
                layer_idx,
                "mlp.up_proj.weight",
                "channel_mixer.up_proj.weight",
                "ffn.w3.weight",
                "feed_forward.w3.weight",
            ),
            mlp_down_proj=f(
                layer_idx,
                "mlp.down_proj.weight",
                "channel_mixer.down_proj.weight",
                "ffn.w2.weight",
                "feed_forward.w2.weight",
            ),
        )
        if attention:
            return LayerAnatomy(
                layer_idx=layer_idx,
                attn_q_proj=f(layer_idx, "self_attn.q_proj.weight", "operator.q_proj.weight"),
                attn_k_proj=f(layer_idx, "self_attn.k_proj.weight", "operator.k_proj.weight"),
                attn_v_proj=f(layer_idx, "self_attn.v_proj.weight", "operator.v_proj.weight"),
                attn_o_proj=f(
                    layer_idx,
                    "self_attn.o_proj.weight",
                    "self_attn.out_proj.weight",
                    "operator.out_proj.weight",
                ),
                **mixer,
            )
        return LayerAnatomy(
            layer_idx=layer_idx,
            attn_q_proj=None,
            attn_k_proj=None,
            attn_v_proj=None,
            attn_o_proj=None,
            liquid_tau_proj=f(layer_idx, "operator.tau_proj.weight", "liquid.dt_proj.weight"),
            ssm_in_proj=f(
                layer_idx,
                "conv.in_proj.weight",
                "operator.in_proj.weight",
                "liquid.in_proj.weight",
            ),
            ssm_out_proj=f(
                layer_idx,
                "conv.out_proj.weight",
                "operator.out_proj.weight",
                "liquid.out_proj.weight",
            ),
            ssm_A_log=f(layer_idx, "operator.A_log", "liquid.A_log"),
            **mixer,
        )

    def liquid_poles(self, layer_idx: int) -> torch.Tensor | None:
        """Continuous-time poles ``-exp(A_log)`` of a liquid block (None for attention blocks)."""
        name = self.get_layer_anatomy(layer_idx).ssm_A_log
        if name is None:
            return None
        a_log = dict(self.model.named_parameters())[name].detach().float()
        return -torch.exp(a_log)

    def pole_stability_ok(self, layer_idx: int, min_decay_rate: float = 1e-6) -> bool:
        """True when every pole is finite and strictly left of ``-min_decay_rate``."""
        poles = self.liquid_poles(layer_idx)
        if poles is None:
            return True
        return bool(torch.isfinite(poles).all() and (poles < -min_decay_rate).all())
