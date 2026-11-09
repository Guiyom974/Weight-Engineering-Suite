"""Architecture adapter for DeepSeek (Multi-head Latent Attention + routed/shared MoE).

MLA has no separate k/v projections: keys and values come from the latent ``kv_a_proj_with_mqa`` ->
``kv_a_layernorm`` -> ``kv_b_proj`` path, so ``attn_k_proj``/``attn_v_proj`` are None and the MLA fields are set.
Layers ``i >= first_k_dense_replace`` are MoE: ``mlp.gate.weight`` routes to ``mlp.experts.{j}`` with
``mlp.gate.e_score_correction_bias``; ``mlp.shared_experts`` is always on and is reported as the layer's
``mlp_*`` projections. ``expert_bias_delta`` builds the PRD's zero-gradient expert re-routing edit.
"""

from __future__ import annotations

import torch

from .base import DecoderAdapter, LayerAnatomy


class DeepSeekAdapter(DecoderAdapter):
    """Adapter for DeepSeek-V2/V3/R1 style decoders."""

    family = "deepseek_moe"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """True when the layer has a router gate."""
        return self.find_param(layer_idx, "mlp.gate.weight") is not None

    def expert_down_projs(self, layer_idx: int) -> list[str]:
        """``down_proj`` names of every routed expert in index order."""
        prefix = f"{self.layer_prefix(layer_idx)}.mlp.experts."
        names = [n for n in self._param_names if n.startswith(prefix) and n.endswith(".down_proj.weight")]
        return sorted(names, key=lambda n: int(n[len(prefix) :].split(".")[0]))

    def get_layer_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """MLA attention paths plus dense-MLP or MoE paths."""
        f = self.find_param
        moe = self.is_moe_layer(layer_idx)
        mlp_root = "mlp.shared_experts" if moe else "mlp"
        return LayerAnatomy(
            layer_idx=layer_idx,
            attn_q_proj=f(layer_idx, "self_attn.q_proj.weight", "self_attn.q_b_proj.weight"),
            attn_k_proj=None,
            attn_v_proj=None,
            attn_o_proj=f(layer_idx, "self_attn.o_proj.weight"),
            mlp_gate_proj=f(layer_idx, f"{mlp_root}.gate_proj.weight"),
            mlp_up_proj=f(layer_idx, f"{mlp_root}.up_proj.weight"),
            mlp_down_proj=f(layer_idx, f"{mlp_root}.down_proj.weight"),
            mla_kv_a_proj=f(layer_idx, "self_attn.kv_a_proj_with_mqa.weight"),
            mla_kv_b_proj=f(layer_idx, "self_attn.kv_b_proj.weight"),
            moe_router_gate=f(layer_idx, "mlp.gate.weight") if moe else None,
            moe_expert_down_projs=self.expert_down_projs(layer_idx) if moe else None,
            moe_router_bias=f(layer_idx, "mlp.gate.e_score_correction_bias") if moe else None,
        )

    def expert_bias_delta(self, layer_idx: int, expert_idx: int, shift: float) -> dict[str, torch.Tensor]:
        """Delta that shifts one expert's routing score-correction bias (silence: shift < 0; boost: > 0).

        Raises:
            ValueError: the layer is not MoE, has no bias parameter, or ``expert_idx`` is out of range.
        """
        name = self.find_param(layer_idx, "mlp.gate.e_score_correction_bias")
        if name is None:
            raise ValueError(f"layer {layer_idx} has no e_score_correction_bias (not an MoE layer?)")
        bias = dict(self.model.named_parameters())[name]
        if not 0 <= expert_idx < bias.numel():
            raise ValueError(f"expert_idx {expert_idx} out of range for {bias.numel()} experts")
        delta = torch.zeros_like(bias)
        delta[expert_idx] = shift
        return {name: delta}
