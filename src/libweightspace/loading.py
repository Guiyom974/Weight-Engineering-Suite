"""Model loading from a spec string, shared by both CLIs and the anchor vault.

Specs:
    - ``toy:<arch>`` or ``toy:<arch>:<seed>``: bundled tiny models (``qwen3_8``, ``gemma4``, ``deepseek_moe``,
      ``lfm3``) with the char tokenizer; fully offline. ``demo_defect=True`` plants the documented 'x'
      repetition attractor so the demos have something real to repair.
    - anything else: a Hugging Face model id or local directory, loaded with ``transformers`` (``hf`` extra)
      and ``trust_remote_code=False`` (remote code is never executed). Append ``@<revision>`` to pin a commit
      (``org/model@abc123``); without a pin the repository's default branch can change under you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from .adapters import resolve_adapter
from .adapters.base import DecoderAdapter
from .toy_models import CharTokenizer, build_toy_model, plant_attractor_defect

_TOY_RE = re.compile(r"^toy:([a-z0-9_]+)(?::(\d{1,9}))?$")


@dataclass
class LoadedModel:
    """A model with its tokenizer and adapter."""

    model: nn.Module
    tokenizer: Any
    adapter: DecoderAdapter
    model_id: str


def model_from_spec(spec: str, device: str = "cpu", demo_defect: bool = False) -> nn.Module:
    """Builds only the model (used as the anchor-vault factory, where weights are loaded afterwards)."""
    return load_model(spec, device, demo_defect).model


def load_model(spec: str, device: str = "cpu", demo_defect: bool = False) -> LoadedModel:
    """Loads ``spec`` onto ``device``.

    Raises:
        ValueError: malformed toy spec or demo defect requested for a non-toy model.
        RuntimeError: ``transformers`` missing for a Hugging Face spec.
    """
    toy = _TOY_RE.match(spec)
    if spec.startswith("toy:") and not toy:
        raise ValueError(f"invalid toy spec '{spec}' (expected toy:<arch>[:<seed>])")
    if toy:
        arch, seed = toy.group(1), int(toy.group(2) or 0)
        toy_model = build_toy_model(arch, seed=seed)
        if demo_defect:
            # DeepSeek layer 1 is MoE; its dense MLP lives in layer 0.
            plant_attractor_defect(toy_model, layer_idx=0 if arch == "deepseek_moe" else 1)
        toy_model = toy_model.to(device).eval()
        return LoadedModel(
            toy_model,
            CharTokenizer(toy_model.config.vocab_size),
            resolve_adapter(toy_model, arch),
            spec,
        )
    if demo_defect:
        raise ValueError("demo_defect is only available for toy models")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("loading Hugging Face models needs the 'hf' extra: pip install '.[hf]'") from exc

    # Register the vendored LFM 3 architecture so trust_remote_code=False loads it natively.
    from .vendor_lfm3 import Lfm3Config, Lfm3ForCausalLM

    try:
        AutoConfig.register("lfm3", Lfm3Config)
        AutoModelForCausalLM.register(Lfm3Config, Lfm3ForCausalLM)
    except ValueError:
        pass  # already registered by an earlier load in this process (or by a transformers release that ships lfm3)

    # A local folder is loaded as it is; only Hub ids carry an ``@<revision>`` pin.
    name, _, pinned = ("", "", "") if Path(spec).is_dir() else spec.partition("@")
    name = name or spec
    revision = pinned or None
    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype="auto", trust_remote_code=False
    )
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    return LoadedModel(model, tokenizer, resolve_adapter(model), spec)
