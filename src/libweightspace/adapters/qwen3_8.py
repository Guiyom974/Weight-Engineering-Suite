"""Architecture adapter for Qwen-3.8 (GQA, SwiGLU, per-head QK-Norm, RoPE).

Qwen3-family attention applies RMSNorm ``q_norm``/``k_norm`` to each head before RoPE. Vectors synthesised for
query/key space must be divided by those norm weights (``qk_norm_inverse_scales``) so that, after the model
re-applies the norm scale, the intended direction is recovered.
"""

from __future__ import annotations

import torch

from .base import DecoderAdapter, LayerAnatomy


class Qwen3Adapter(DecoderAdapter):
    """Adapter for Qwen3/Qwen-3.8 style decoders (``model.layers.{i}.self_attn|mlp``)."""

    family = "qwen3_8"

    def get_layer_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """Standard Qwen projection paths."""
        return self.standard_anatomy(layer_idx)

    def qk_norm_names(self, layer_idx: int) -> tuple[str | None, str | None]:
        """Parameter names of the per-head ``q_norm`` / ``k_norm`` weights (None when absent)."""
        return (
            self.find_param(layer_idx, "self_attn.q_norm.weight"),
            self.find_param(layer_idx, "self_attn.k_norm.weight"),
        )

    def qk_norm_inverse_scales(self, layer_idx: int, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``1 / q_norm.weight`` and ``1 / k_norm.weight`` (ones when the layer has no QK-norm)."""
        params = dict(self.model.named_parameters())
        q_name, k_name = self.qk_norm_names(layer_idx)
        head_dim = int(self.config_value("head_dim", 0) or 0)
        ones = torch.ones(head_dim) if head_dim else torch.ones(1)
        q = 1.0 / (params[q_name].detach().float() + eps) if q_name else ones
        k = 1.0 / (params[k_name].detach().float() + eps) if k_name else ones
        return q, k
