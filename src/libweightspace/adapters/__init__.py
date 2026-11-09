"""Architecture-specific parameter and hook adapters.

``resolve_adapter(model, arch=None)`` picks the adapter from an explicit architecture key or from
``model.config.model_type`` (Hugging Face and toy models both expose it).
"""

from __future__ import annotations

import torch.nn as nn

from .base import ArchitectureAdapter, DecoderAdapter, LayerAnatomy
from .deepseek import DeepSeekAdapter
from .gemma4 import Gemma4Adapter
from .lfm3 import LFM3Adapter
from .qwen3_8 import Qwen3Adapter

ADAPTERS: dict[str, type[DecoderAdapter]] = {
    "qwen3_8": Qwen3Adapter,
    "gemma4": Gemma4Adapter,
    "deepseek_moe": DeepSeekAdapter,
    "lfm3": LFM3Adapter,
}

# model_type prefixes (HF config) -> adapter key. Llama/Mistral share the Qwen layout.
_MODEL_TYPE_PREFIXES = (
    ("qwen", "qwen3_8"),
    ("llama", "qwen3_8"),
    ("mistral", "qwen3_8"),
    ("gemma", "gemma4"),
    ("deepseek", "deepseek_moe"),
    ("lfm", "lfm3"),
)


def adapter_key_for_model_type(model_type: str | None) -> str | None:
    """Adapter key for a Hugging Face ``model_type`` (e.g. ``llama`` -> ``qwen3_8``), or None when unsupported.

    Used before a download to tell the user whether the architecture can be edited at all.
    """
    lowered = str(model_type or "").lower()
    for prefix, key in _MODEL_TYPE_PREFIXES:
        if lowered.startswith(prefix):
            return key
    return None


def resolve_adapter(model: nn.Module, arch: str | None = None) -> DecoderAdapter:
    """Returns the adapter for ``model``.

    Args:
        model: the model to adapt.
        arch: optional explicit key (``qwen3_8``, ``gemma4``, ``deepseek_moe``, ``lfm3``).

    Raises:
        ValueError: unknown ``arch`` or a ``model_type`` with no matching adapter.
    """
    if arch is not None:
        if arch not in ADAPTERS:
            raise ValueError(f"unknown architecture '{arch}'; choose from {sorted(ADAPTERS)}")
        return ADAPTERS[arch](model)
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    key = adapter_key_for_model_type(model_type)
    if key is None:
        raise ValueError(f"cannot infer adapter for model_type '{model_type}'; pass arch explicitly")
    return ADAPTERS[key](model)


__all__ = [
    "ADAPTERS",
    "ArchitectureAdapter",
    "DecoderAdapter",
    "DeepSeekAdapter",
    "Gemma4Adapter",
    "LFM3Adapter",
    "LayerAnatomy",
    "Qwen3Adapter",
    "adapter_key_for_model_type",
    "resolve_adapter",
]
