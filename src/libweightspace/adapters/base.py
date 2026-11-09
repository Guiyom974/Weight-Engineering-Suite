"""Canonical interface for inspecting and addressing model layers across architectures.

``ArchitectureAdapter`` is the PRD abstract interface. ``DecoderAdapter`` implements the parts every Hugging
Face style decoder shares (layer discovery through several known module hierarchies, parameter lookup by
candidate suffixes, unembedding and final-norm access, logit lens). Family adapters subclass it and only
describe their anatomy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LayerAnatomy:
    """Exact parameter paths of one layer; ``None`` means the layer has no such projection."""

    layer_idx: int
    attn_q_proj: str | None
    attn_k_proj: str | None
    attn_v_proj: str | None
    attn_o_proj: str | None
    mlp_gate_proj: str | None
    mlp_up_proj: str | None
    mlp_down_proj: str | None
    # DeepSeek MLA & MoE specific
    mla_kv_a_proj: str | None = None
    mla_kv_b_proj: str | None = None
    moe_router_gate: str | None = None
    moe_expert_down_projs: list[str] | None = None
    # LFM 3 State-Space / Liquid specific
    liquid_tau_proj: str | None = None
    ssm_in_proj: str | None = None
    ssm_out_proj: str | None = None
    ssm_A_log: str | None = None
    # Extension (backward compatible): DeepSeek router score-correction bias.
    moe_router_bias: str | None = None

    def present(self) -> dict[str, str | list[str]]:
        """Returns only the fields that resolved to parameters (excluding ``layer_idx``)."""
        return {k: v for k, v in asdict(self).items() if k != "layer_idx" and v}


class ArchitectureAdapter(ABC):
    """Abstract adapter providing uniform mechanistic access to diverse LLM topologies."""

    family: str = "abstract"

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def num_layers(self) -> int:
        """Returns the total number of repeating blocks/layers."""

    @abstractmethod
    def get_layer_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """Returns exact parameter paths for a specific layer index."""

    @abstractmethod
    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Returns the PyTorch nn.Module for hooking residual outputs."""

    @abstractmethod
    def get_lm_head_weight(self) -> torch.Tensor:
        """Returns the unembedding projection matrix [vocab_size, d_model]."""

    @abstractmethod
    def get_final_norm(self) -> nn.Module:
        """Returns the final RMSNorm/LayerNorm prior to unembedding."""

    def hidden_size(self) -> int:
        """Residual stream width ``d_model``."""
        return int(self.get_lm_head_weight().shape[1])

    def logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Projects residual states ``[..., d_model]`` to vocabulary logits through the final norm."""
        norm = self.get_final_norm()
        w = self.get_lm_head_weight()
        return norm(hidden.to(w.dtype)) @ w.T


_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.model.layers",
    "transformer.h",
    "blocks",
)
_NORM_PATHS = ("model.norm", "model.language_model.norm", "language_model.model.norm", "norm")


def _get_path(root: nn.Module, dotted: str) -> nn.Module | None:
    obj: object = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, nn.Module) else None


class DecoderAdapter(ArchitectureAdapter):
    """Shared implementation for HF-style decoder stacks."""

    family = "decoder"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._layers = self._resolve_layers()
        module_names = {id(m): n for n, m in model.named_modules()}
        self._prefixes = [module_names[id(layer)] for layer in self._layers]
        self._param_names = set(dict(model.named_parameters()).keys())

    def _resolve_layers(self) -> nn.ModuleList:
        for path in _LAYER_PATHS:
            found = _get_path(self.model, path)
            if isinstance(found, nn.ModuleList) and len(found) > 0:
                return found
        raise ValueError(
            f"{type(self).__name__}: could not locate the layer stack (tried {', '.join(_LAYER_PATHS)})"
        )

    def num_layers(self) -> int:
        """Number of decoder blocks."""
        return len(self._layers)

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Decoder block ``layer_idx`` (its forward output is the residual stream after the block)."""
        return self._layers[layer_idx]

    def layer_prefix(self, layer_idx: int) -> str:
        """Qualified module name of block ``layer_idx`` (e.g. ``model.layers.3``)."""
        return self._prefixes[layer_idx]

    def find_param(self, layer_idx: int, *suffixes: str) -> str | None:
        """Returns ``{prefix}.{suffix}`` for the first suffix that names an existing parameter."""
        prefix = self.layer_prefix(layer_idx)
        for suffix in suffixes:
            full = f"{prefix}.{suffix}"
            if full in self._param_names:
                return full
        return None

    def layer_param_names(self, layer_idx: int) -> list[str]:
        """All parameter names inside block ``layer_idx``."""
        prefix = self.layer_prefix(layer_idx) + "."
        return sorted(n for n in self._param_names if n.startswith(prefix))

    def get_lm_head_weight(self) -> torch.Tensor:
        """Unembedding matrix ``[vocab, d_model]``."""
        head = getattr(self.model, "lm_head", None)
        weight = getattr(head, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
        get_out = getattr(self.model, "get_output_embeddings", None)
        if callable(get_out):
            out = get_out()
            if out is not None:
                return out.weight
        raise ValueError(f"{type(self).__name__}: model has no lm_head / output embeddings")

    def get_final_norm(self) -> nn.Module:
        """Final norm applied before the unembedding."""
        for path in _NORM_PATHS:
            found = _get_path(self.model, path)
            if found is not None:
                return found
        raise ValueError(f"{type(self).__name__}: could not locate the final norm")

    def config_value(self, name: str, default: Any = None) -> Any:
        """Reads ``model.config.<name>`` (works for HF configs and toy dataclass configs)."""
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return default
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, name):
            return getattr(text_cfg, name)
        return getattr(cfg, name, default)

    def standard_anatomy(self, layer_idx: int) -> LayerAnatomy:
        """Anatomy for the common ``self_attn.{q,k,v,o}_proj`` + ``mlp.{gate,up,down}_proj`` layout."""
        f = self.find_param
        return LayerAnatomy(
            layer_idx=layer_idx,
            attn_q_proj=f(layer_idx, "self_attn.q_proj.weight"),
            attn_k_proj=f(layer_idx, "self_attn.k_proj.weight"),
            attn_v_proj=f(layer_idx, "self_attn.v_proj.weight"),
            attn_o_proj=f(layer_idx, "self_attn.o_proj.weight"),
            mlp_gate_proj=f(layer_idx, "mlp.gate_proj.weight"),
            mlp_up_proj=f(layer_idx, "mlp.up_proj.weight"),
            mlp_down_proj=f(layer_idx, "mlp.down_proj.weight"),
        )
