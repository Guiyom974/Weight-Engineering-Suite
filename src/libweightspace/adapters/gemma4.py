"""Architecture adapter for Gemma 4 (interleaved sliding/global attention, GeGLU, soft-capping).

Gemma nuances handled here:
    - RMSNorm uses a +1 unit offset: ``x / rms(x) * (1 + w)``; ``effective_norm_weight`` returns ``1 + w``.
    - ``final_logit_softcapping``: ``logits = cap * tanh(raw / cap)``; ``logit_lens`` applies it and
      ``invert_softcap`` recovers raw logits with ``arctanh`` clamped to its valid domain.
    - Sliding-window layers alternate with global layers (``is_sliding_layer``).
"""

from __future__ import annotations

import torch

from .base import DecoderAdapter, LayerAnatomy


class Gemma4Adapter(DecoderAdapter):
    """Adapter for Gemma 3/4 style decoders."""

    family = "gemma4"
    rmsnorm_unit_offset = True

    def get_layer_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """Standard projection paths (Gemma keeps the Llama naming)."""
        return self.standard_anatomy(layer_idx)

    def effective_norm_weight(self, norm_weight: torch.Tensor) -> torch.Tensor:
        """Multiplicative scale actually applied by a Gemma RMSNorm with raw weight ``w``."""
        return 1.0 + norm_weight.float()

    def final_softcap(self) -> float | None:
        """Final logit soft-cap value, if configured."""
        cap = self.config_value("final_logit_softcapping")
        return float(cap) if cap else None

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm (with +1 offset, applied by the module), unembedding, then soft-cap."""
        logits = super().logit_lens(hidden)
        cap = self.final_softcap()
        return cap * torch.tanh(logits / cap) if cap else logits

    def invert_softcap(self, capped_logits: torch.Tensor, margin: float = 1e-6) -> torch.Tensor:
        """Inverse of the soft-cap: ``cap * arctanh(logits / cap)`` within ``(-1, 1)``."""
        cap = self.final_softcap()
        if not cap:
            return capped_logits
        ratio = (capped_logits.float() / cap).clamp(-1.0 + margin, 1.0 - margin)
        return cap * torch.atanh(ratio)

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """True when layer ``layer_idx`` uses local sliding-window attention."""
        layer_types = self.config_value("layer_types")
        if isinstance(layer_types, (list, tuple)) and layer_types and isinstance(layer_types[0], str):
            if layer_types[layer_idx] in ("sliding_attention", "full_attention"):
                return layer_types[layer_idx] == "sliding_attention"
        window = self.config_value("sliding_window")
        if not window:
            return False
        pattern = int(self.config_value("sliding_window_pattern", 2) or 2)
        return (layer_idx + 1) % pattern != 0
