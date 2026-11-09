"""Self-Anatomy Atlas: per-layer specialization and spectral map of the running model.

The atlas tells the self-editing model what it is made of: which layers are attention, liquid (state-space),
MoE or dense, how many parameters each holds, and the spectrum (top singular values, effective rank, norms) of
the residual-writing projections it is most likely to edit. ``summarize`` renders it compactly for prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from libweightspace.adapters.base import DecoderAdapter
from libweightspace.spectral import SpectralProfile, profile_matrix

DEFAULT_FIELDS = ("attn_o_proj", "mlp_down_proj", "ssm_out_proj", "moe_router_gate")


@dataclass
class LayerProfile:
    """Anatomy and spectra of one layer."""

    layer_idx: int
    kind: str
    param_count: int
    tensors: dict[str, SpectralProfile] = field(default_factory=dict)


class SelfAnatomyAtlas:
    """Builds ``LayerProfile``s for every layer of a model."""

    def __init__(self, model: nn.Module, adapter: DecoderAdapter):
        self.model = model
        self.adapter = adapter

    def _kind(self, anatomy: Any) -> str:
        if anatomy.ssm_A_log or anatomy.liquid_tau_proj:
            return "liquid"
        if anatomy.moe_router_gate:
            return "moe"
        if anatomy.attn_q_proj:
            return "attention"
        return "dense"

    def build(self, top_k: int = 8, fields: tuple[str, ...] = DEFAULT_FIELDS) -> list[LayerProfile]:
        """Profiles the selected projection fields of every layer."""
        params = dict(self.model.named_parameters())
        profiles = []
        for i in range(self.adapter.num_layers()):
            anatomy = self.adapter.get_layer_anatomy(i)
            names = self.adapter.layer_param_names(i)
            profile = LayerProfile(i, self._kind(anatomy), sum(params[n].numel() for n in names))
            for f in fields:
                name = getattr(anatomy, f)
                if isinstance(name, str) and params[name].ndim == 2:
                    profile.tensors[f] = profile_matrix(params[name].detach(), name, top_k)
            profiles.append(profile)
        return profiles

    def to_rows(self, profiles: list[LayerProfile]) -> list[dict[str, Any]]:
        """Flat rows (one per profiled tensor) for tables and JSON telemetry."""
        return [
            {
                "layer": p.layer_idx,
                "kind": p.kind,
                "params": p.param_count,
                "field": f,
                "tensor": s.tensor_name,
                "sigma_max": round(s.spectral_norm, 4),
                "effective_rank": round(s.effective_rank, 2),
                "frobenius": round(s.frobenius_norm, 4),
            }
            for p in profiles
            for f, s in p.tensors.items()
        ]

    def summarize(self, profiles: list[LayerProfile], max_layers: int = 8) -> str:
        """Compact multi-line description for prompts."""
        lines = []
        for p in profiles[:max_layers]:
            spectra = ", ".join(
                f"{f}: s1={s.spectral_norm:.3g} erank={s.effective_rank:.1f}" for f, s in p.tensors.items()
            )
            lines.append(f"L{p.layer_idx} [{p.kind}, {p.param_count} params] {spectra}")
        return "\n".join(lines)
