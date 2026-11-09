"""Tiny deterministic synthetic models that mirror real architecture families.

Purpose: fast CPU fixtures for tests and offline CLI demos, and the runtime model for compiled blueprints.
Each family reproduces the mechanistically relevant structure and the Hugging Face parameter naming of its
real counterpart, so adapters, the prober and edit scripts address them exactly as they would a real model:

    - ``qwen3``: GQA attention with per-head q_norm/k_norm, RoPE, SwiGLU MLP.
    - ``gemma4``: RMSNorm with +1 offset, GeGLU, alternating sliding/global attention, logit soft-capping.
    - ``deepseek_v3``: Multi-head Latent Attention (kv_a/kv_b), routed + shared MoE with score-correction bias.
    - ``lfm3``: hybrid of liquid time-constant state-space blocks and attention blocks with channel mixers.

Main entry points: ``build_toy_model(arch)``, the ``build_tiny_*`` factories, ``CharTokenizer``,
``plant_attractor_defect`` (plants a known, causal repetition attractor for probing/repair demos) and the RoPE
helpers ``apply_rope`` / ``rope_rotation_matrix`` reused by the blueprint compiler.

Assumptions: single unpadded sequences (``attention_mask`` is accepted for HF call compatibility and ignored);
weights are initialised from a seeded generator, so a factory call with the same seed is bit-reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------
# Configuration and outputs
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ToyConfig:
    """Hyper-parameters for every toy family (unused fields are ignored by other families)."""

    model_type: str = "qwen3"
    vocab_size: int = 128
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    # Gemma family
    sliding_window: int | None = None
    sliding_window_pattern: int = 2
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    # DeepSeek family
    n_routed_experts: int = 4
    num_experts_per_tok: int = 2
    n_shared_experts: int = 1
    moe_intermediate_size: int = 16
    first_k_dense_replace: int = 1
    kv_lora_rank: int = 16
    qk_nope_head_dim: int = 4
    qk_rope_head_dim: int = 4
    v_head_dim: int = 8
    # LFM family: per-layer block kinds ("liquid" | "attention")
    layer_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the config."""
        return asdict(self)


@dataclass
class ToyOutput:
    """Minimal stand-in for a Hugging Face ``CausalLMOutput``."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


# ---------------------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm; ``unit_offset=True`` reproduces Gemma's ``x / rms(x) * (1 + w)`` convention."""

    def __init__(self, dim: int, eps: float = 1e-6, unit_offset: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim) if unit_offset else torch.ones(dim))
        self.eps = eps
        self.unit_offset = unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalises over the last dimension in float32 and casts back."""
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        scale = (1.0 + self.weight.float()) if self.unit_offset else self.weight.float()
        return (normed * scale).to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns RoPE ``cos``/``sin`` tables of shape ``[T, dim]`` (HF ``rotate_half`` convention)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Maps ``[x1, x2]`` to ``[-x2, x1]`` over the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embedding; ``cos``/``sin`` broadcast against ``x``'s trailing ``[T, dim]``."""
    return x * cos + rotate_half(x) * sin


def rope_rotation_matrix(dim: int, position: float, theta: float) -> torch.Tensor:
    """Returns ``R`` (``[dim, dim]``) with ``apply_rope(x)`` at ``position`` equal to ``R @ x``.

    ``R(a)^T R(b) = R(b - a)``, which is what lets the blueprint compiler synthesise offset-selective heads.
    """
    cos, sin = rope_cos_sin(torch.tensor([position]), dim, theta)
    return apply_rope(torch.eye(dim), cos[0], sin[0]).T.contiguous()


def _causal_mask(t: int, window: int | None, device: torch.device) -> torch.Tensor:
    """Boolean ``[T, T]`` mask, True where attention is allowed (causal, optionally windowed)."""
    i = torch.arange(t, device=device)[:, None]
    j = torch.arange(t, device=device)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= j > i - window
    return allowed


class GatedMLP(nn.Module):
    """SwiGLU (``silu``) or GeGLU (``gelu_tanh``) feed-forward block with HF projection names."""

    def __init__(self, hidden: int, inter: int, act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``down(act(gate(x)) * up(x))``."""
        gate = self.gate_proj(x)
        gate = F.silu(gate) if self.act == "silu" else F.gelu(gate, approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class ToyAttention(nn.Module):
    """GQA attention with optional QK-norm, sliding window and logit soft-capping.

    ``o_name`` sets the output projection attribute (``o_proj`` for Qwen/Gemma, ``out_proj`` for LFM).
    """

    def __init__(
        self,
        cfg: ToyConfig,
        sliding_window: int | None = None,
        softcap: float | None = None,
        o_name: str = "o_proj",
        unit_offset_norm: bool = False,
    ):
        super().__init__()
        h, kv, hd, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
        self.n_heads, self.n_kv, self.head_dim = h, kv, hd
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, kv * hd, bias=False)
        self.v_proj = nn.Linear(d, kv * hd, bias=False)
        self.o_name = o_name
        setattr(self, o_name, nn.Linear(h * hd, d, bias=False))
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(hd, cfg.rms_norm_eps, unit_offset_norm)
            self.k_norm = RMSNorm(hd, cfg.rms_norm_eps, unit_offset_norm)
        self.sliding_window = sliding_window
        self.softcap = softcap

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Causal self-attention over ``x`` of shape ``[B, T, d]``."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        groups = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.softcap is not None:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        scores = scores.masked_fill(~_causal_mask(t, self.sliding_window, x.device), float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return getattr(self, self.o_name)(out)


class ToyMLAttention(nn.Module):
    """DeepSeek Multi-head Latent Attention: KV compressed through ``kv_a_proj_with_mqa`` / ``kv_b_proj``."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        d, h = cfg.hidden_size, cfg.num_attention_heads
        self.n_heads, self.nope, self.rope, self.v_dim = (
            h,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.v_head_dim,
        )
        self.kv_lora_rank = cfg.kv_lora_rank
        self.q_proj = nn.Linear(d, h * (self.nope + self.rope), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, cfg.kv_lora_rank + self.rope, bias=False)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, h * (self.nope + self.v_dim), bias=False)
        self.o_proj = nn.Linear(h * self.v_dim, d, bias=False)
        self.rope_theta = cfg.rope_theta

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Causal MLA over ``x`` of shape ``[B, T, d]`` (``cos``/``sin`` unused: rope dim differs)."""
        b, t, _ = x.shape
        pos_cos, pos_sin = rope_cos_sin(torch.arange(t, device=x.device), self.rope, self.rope_theta)
        q = self.q_proj(x).view(b, t, self.n_heads, self.nope + self.rope).transpose(1, 2)
        q_nope, q_pe = q[..., : self.nope], q[..., self.nope :]
        compressed = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe = compressed[..., : self.kv_lora_rank], compressed[..., self.kv_lora_rank :]
        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv)).view(b, t, self.n_heads, self.nope + self.v_dim)
        kv = kv.transpose(1, 2)
        k_nope, v = kv[..., : self.nope], kv[..., self.nope :]
        q_pe = apply_rope(q_pe, pos_cos, pos_sin)
        k_pe = apply_rope(k_pe[:, None, :, :], pos_cos, pos_sin).expand(b, self.n_heads, t, self.rope)
        q_full, k_full = torch.cat([q_nope, q_pe], dim=-1), torch.cat([k_nope, k_pe], dim=-1)
        scores = (q_full @ k_full.transpose(-1, -2)) / math.sqrt(self.nope + self.rope)
        scores = scores.masked_fill(~_causal_mask(t, None, x.device), float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.n_heads * self.v_dim)
        return self.o_proj(out)


class ToyMoEGate(nn.Module):
    """DeepSeek-V3 style router: sigmoid affinities, bias-corrected top-k selection."""

    def __init__(self, hidden: int, n_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_experts, hidden))
        # A Parameter (not a buffer) so zero-gradient re-routing edits are addressable like any weight.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))


class ToyMoE(nn.Module):
    """Routed experts plus always-on shared experts."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.gate = ToyMoEGate(cfg.hidden_size, cfg.n_routed_experts)
        self.experts = nn.ModuleList(
            [GatedMLP(cfg.hidden_size, cfg.moe_intermediate_size) for _ in range(cfg.n_routed_experts)]
        )
        self.shared_experts = GatedMLP(cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts)
        self.top_k = cfg.num_experts_per_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense evaluation of all experts weighted by the routing mask (tiny models only)."""
        scores = torch.sigmoid(x @ self.gate.weight.T)
        _, idx = torch.topk(scores + self.gate.e_score_correction_bias, k=self.top_k, dim=-1)
        weights = scores.gather(-1, idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        out = self.shared_experts(x)
        for e, expert in enumerate(self.experts):
            w = (weights * (idx == e)).sum(dim=-1, keepdim=True)
            out = out + w * expert(x)
        return out


class ToyLiquidOperator(nn.Module):
    """Liquid time-constant diagonal state-space operator.

    ``dt = softplus(tau_proj(x))`` modulates decay ``exp(-dt * exp(A_log))``; poles ``-exp(A_log)`` are in the
    left half-plane for any finite ``A_log``.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.tau_proj = nn.Linear(hidden, hidden, bias=False)
        self.A_log = nn.Parameter(torch.zeros(hidden))
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Sequential scan over ``x`` of shape ``[B, T, d]``."""
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        dt = F.softplus(self.tau_proj(x))
        decay = torch.exp(-dt * torch.exp(self.A_log))
        state = torch.zeros_like(u[:, 0])
        outs = []
        for step in range(x.shape[1]):
            state = decay[:, step] * state + (1.0 - decay[:, step]) * u[:, step]
            outs.append(state)
        return self.out_proj(torch.stack(outs, dim=1) * F.silu(gate))


# ---------------------------------------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------------------------------------


class ToyDecoderLayer(nn.Module):
    """Qwen3 / Gemma / DeepSeek decoder block (pre-norm residual)."""

    def __init__(self, cfg: ToyConfig, layer_idx: int):
        super().__init__()
        family = cfg.model_type
        self.is_gemma = family.startswith("gemma")
        eps = cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps, self.is_gemma)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps, self.is_gemma)
        self.self_attn: nn.Module
        self.mlp: nn.Module
        if family.startswith("deepseek"):
            self.self_attn = ToyMLAttention(cfg)
            if layer_idx < cfg.first_k_dense_replace:
                self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size)
            else:
                self.mlp = ToyMoE(cfg)
        else:
            window = None
            if self.is_gemma and cfg.sliding_window and (layer_idx + 1) % cfg.sliding_window_pattern != 0:
                window = cfg.sliding_window
            self.sliding_window = window
            self.self_attn = ToyAttention(cfg, window, cfg.attn_logit_softcapping, "o_proj", self.is_gemma)
            self.mlp = GatedMLP(
                cfg.hidden_size, cfg.intermediate_size, "gelu_tanh" if self.is_gemma else "silu"
            )
        if self.is_gemma:
            self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps, True)
            self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps, True)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """One residual block."""
        if self.is_gemma:
            h = h + self.post_attention_layernorm(self.self_attn(self.input_layernorm(h), cos, sin))
            return h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
        h = h + self.self_attn(self.input_layernorm(h), cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


class ToyLFMBlock(nn.Module):
    """LFM 3 hybrid block: ``operator`` (liquid or attention) followed by a gated ``channel_mixer``."""

    def __init__(self, cfg: ToyConfig, kind: str):
        super().__init__()
        if kind not in ("liquid", "attention"):
            raise ValueError(f"unknown LFM block kind '{kind}'")
        self.kind = kind
        self.operator_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.operator: nn.Module
        if kind == "liquid":
            self.operator = ToyLiquidOperator(cfg.hidden_size)
        else:
            self.operator = ToyAttention(cfg, o_name="out_proj")
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.channel_mixer = GatedMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """One residual hybrid block."""
        h = h + self.operator(self.operator_norm(h), cos, sin)
        return h + self.channel_mixer(self.ffn_norm(h))


class ToyBackbone(nn.Module):
    """Embedding, decoder stack and final norm (the HF ``model.model`` submodule)."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        layers: list[nn.Module]
        if cfg.model_type.startswith("lfm"):
            kinds = cfg.layer_types or ["liquid", "attention"] * cfg.num_hidden_layers
            layers = [ToyLFMBlock(cfg, kinds[i]) for i in range(cfg.num_hidden_layers)]
        else:
            layers = [ToyDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.model_type.startswith("gemma"))


class ToyCausalLM(nn.Module):
    """Causal LM wrapper exposing ``.model``, ``.lm_head`` and a HF-compatible ``forward``."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.config = cfg
        self.model = ToyBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Token embedding table."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Unembedding projection."""
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> ToyOutput:
        """Runs the model on ``input_ids`` ``[B, T]`` or ``inputs_embeds`` ``[B, T, d]``."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("either input_ids or inputs_embeds is required")
            inputs_embeds = self.model.embed_tokens(input_ids)
            if self.config.model_type.startswith("gemma"):
                inputs_embeds = inputs_embeds * math.sqrt(self.config.hidden_size)
        h = inputs_embeds
        t = h.shape[1]
        cos, sin = rope_cos_sin(
            torch.arange(t, device=h.device), self.config.head_dim, self.config.rope_theta
        )
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        hidden: list[torch.Tensor] = [h]
        for layer in self.model.layers:
            h = layer(h, cos, sin)
            hidden.append(h)
        logits = self.lm_head(self.model.norm(h))
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        return ToyOutput(logits=logits, hidden_states=tuple(hidden) if output_hidden_states else None)


# ---------------------------------------------------------------------------------------------------------
# Initialisation and factories
# ---------------------------------------------------------------------------------------------------------


def init_toy_weights(model: ToyCausalLM, seed: int = 0) -> ToyCausalLM:
    """Deterministically initialises every parameter from a seeded generator.

    Scales are chosen so token embeddings dominate the residual stream (norm ~ sqrt(d)) while each block adds a
    small contribution; planted defects are therefore clearly causal and attributable in tests.
    """
    g = torch.Generator().manual_seed(seed)
    unit_offset = model.config.model_type.startswith("gemma")
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.fill_(0.0 if unit_offset and not name.endswith("kv_a_layernorm.weight") else 1.0)
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.linspace(0.5, 4.0, p.numel())))
            elif name.endswith("e_score_correction_bias"):
                p.zero_()
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name == "lm_head.weight":
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
            else:
                fan_in = p.shape[-1] if p.ndim > 1 else p.numel()
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 / math.sqrt(fan_in)))
    return model


def build_tiny_qwen3(seed: int = 0, **overrides: Any) -> ToyCausalLM:
    """Tiny Qwen-3.8-style model (QK-norm, GQA, SwiGLU)."""
    cfg = ToyConfig(**{"model_type": "qwen3", "use_qk_norm": True, **overrides})
    return init_toy_weights(ToyCausalLM(cfg), seed)


def build_tiny_gemma4(seed: int = 0, **overrides: Any) -> ToyCausalLM:
    """Tiny Gemma-4-style model (+1 RMSNorm offset, GeGLU, sliding/global attention, soft-capping)."""
    defaults: dict[str, Any] = {
        "model_type": "gemma4",
        "use_qk_norm": True,
        "sliding_window": 4,
        "sliding_window_pattern": 2,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
    }
    return init_toy_weights(ToyCausalLM(ToyConfig(**{**defaults, **overrides})), seed)


def build_tiny_deepseek(seed: int = 0, **overrides: Any) -> ToyCausalLM:
    """Tiny DeepSeek-V3-style model (MLA attention; layer 0 dense, later layers MoE)."""
    cfg = ToyConfig(**{"model_type": "deepseek_v3", "use_qk_norm": False, **overrides})
    return init_toy_weights(ToyCausalLM(cfg), seed)


def build_tiny_lfm3(seed: int = 0, **overrides: Any) -> ToyCausalLM:
    """Tiny LFM-3-style hybrid: liquid, attention, liquid blocks with channel mixers."""
    defaults: dict[str, Any] = {
        "model_type": "lfm3",
        "use_qk_norm": False,
        "num_hidden_layers": 3,
        "layer_types": ["liquid", "attention", "liquid"],
    }
    return init_toy_weights(ToyCausalLM(ToyConfig(**{**defaults, **overrides})), seed)


TOY_FACTORIES: dict[str, Callable[..., ToyCausalLM]] = {
    "qwen3_8": build_tiny_qwen3,
    "gemma4": build_tiny_gemma4,
    "deepseek_moe": build_tiny_deepseek,
    "lfm3": build_tiny_lfm3,
}


def build_toy_model(arch: str, seed: int = 0, **overrides: Any) -> ToyCausalLM:
    """Builds a toy model by architecture key (``qwen3_8``, ``gemma4``, ``deepseek_moe``, ``lfm3``).

    Raises:
        ValueError: unknown architecture key.
    """
    if arch not in TOY_FACTORIES:
        raise ValueError(f"unknown toy architecture '{arch}'; choose from {sorted(TOY_FACTORIES)}")
    return TOY_FACTORIES[arch](seed=seed, **overrides)


# ---------------------------------------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------------------------------------


class TokenBatch(dict):
    """Dict of tensors with a HF ``BatchEncoding``-like ``.to(device)``."""

    def to(self, device: torch.device | str) -> TokenBatch:
        """Moves every tensor to ``device``."""
        return TokenBatch({k: v.to(device) for k, v in self.items()})


class CharTokenizer:
    """Byte-level character tokenizer: token id = ``ord(char)`` for ASCII, ``?`` otherwise."""

    def __init__(self, vocab_size: int = 128):
        if vocab_size < 128:
            raise ValueError("CharTokenizer needs vocab_size >= 128")
        self.vocab_size = vocab_size
        self.unk_token_id = ord("?")

    def encode(self, text: str) -> list[int]:
        """Encodes ``text`` to token ids."""
        return [ord(c) if ord(c) < 128 else self.unk_token_id for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decodes ids back to text."""
        values = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return "".join(chr(int(i)) if 0 <= int(i) < 128 else "?" for i in values)

    def __call__(self, text: str, return_tensors: str = "pt") -> TokenBatch:
        """Tokenises one string into ``input_ids``/``attention_mask`` of shape ``[1, T]``.

        Raises:
            ValueError: empty text or unsupported ``return_tensors``.
        """
        if return_tensors != "pt":
            raise ValueError("CharTokenizer only supports return_tensors='pt'")
        ids = self.encode(text)
        if not ids:
            raise ValueError("cannot tokenize empty text")
        input_ids = torch.tensor([ids], dtype=torch.long)
        return TokenBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


# ---------------------------------------------------------------------------------------------------------
# Planted defects
# ---------------------------------------------------------------------------------------------------------


@dataclass
class PlantedDefect:
    """Ground truth of a planted attractor (for tests and demos)."""

    layer_idx: int
    tensor_name: str
    columns: list[int]
    direction: torch.Tensor
    trigger: str
    boosted: str
    suppressed: str


def plant_attractor_defect(
    model: ToyCausalLM,
    layer_idx: int = 1,
    columns: tuple[int, ...] = (5, 12, 19),
    trigger: str = "x",
    boosted: str = "x",
    suppressed: str = "y",
    gate_scale: float = 1.5,
    strength: float = 0.08,
) -> PlantedDefect:
    """Plants a causal rank-1 repetition attractor into an MLP's ``down_proj`` columns.

    Neurons ``columns`` are rewired to fire on the ``trigger`` token's embedding direction and their down
    projection columns are set to ``strength * v`` where ``v`` = unit(unembed[boosted] - unembed[suppressed]); after
    seeing ``trigger`` the model therefore strongly predicts ``boosted`` (a self-reinforcing loop when equal).
    The gain sits mostly in the gate (large ``gate_scale``) so the repairing ``down_proj`` edit stays small
    relative to the tensor, within ``EditConstraints`` bounds, as it would in a real model.

    Works on families whose layer exposes ``mlp.gate_proj/up_proj/down_proj`` (Qwen3, Gemma, DeepSeek dense
    layers) or ``channel_mixer.*`` (LFM).
    """
    layer = model.model.layers[layer_idx]
    mlp = getattr(layer, "mlp", None) or getattr(layer, "channel_mixer", None)
    if mlp is None or not isinstance(mlp, GatedMLP):
        raise ValueError(f"layer {layer_idx} has no dense gated MLP to plant into")
    prefix = "mlp" if hasattr(layer, "mlp") else "channel_mixer"
    with torch.no_grad():
        emb = model.model.embed_tokens.weight[ord(trigger)].float()
        emb_unit = emb / (emb.norm() + 1e-8)
        unembed = model.lm_head.weight.float()
        v = unembed[ord(boosted)] - unembed[ord(suppressed)]
        v = v / (v.norm() + 1e-8)
        for c in columns:
            mlp.gate_proj.weight[c] = (gate_scale * emb_unit).to(mlp.gate_proj.weight.dtype)
            mlp.up_proj.weight[c] = (gate_scale * emb_unit).to(mlp.up_proj.weight.dtype)
            # Overwrite (not add): the rewired neuron fires strongly, so any leftover random column content
            # would dominate the planted direction and the defect would no longer be rank-1.
            mlp.down_proj.weight[:, c] = (strength * v).to(mlp.down_proj.weight.dtype)
    return PlantedDefect(
        layer_idx=layer_idx,
        tensor_name=f"model.layers.{layer_idx}.{prefix}.down_proj.weight",
        columns=list(columns),
        direction=v,
        trigger=trigger,
        boosted=boosted,
        suppressed=suppressed,
    )
