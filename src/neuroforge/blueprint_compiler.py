"""Mode 2: compiles a declarative ``NeuralBlueprint`` directly into working Transformer weight tensors.

No gradient steps are taken. Construction (PRD_01 section 4.4):

1. Orthogonal residual partitioning: a seeded Gaussian matrix is QR-decomposed into an orthonormal basis ``Q``;
   subspace ``S_a = [s_a, e_a)`` owns columns ``Q[:, s_a:e_a]``. Writes into one subspace never leak into another.
2. Attention heads: ``W_Q^T R(-dp) W_K = c * Q_SQ Q_SK^T`` and ``W_O W_V = Q_SO Q_SV^T`` where ``R`` is the RoPE
   rotation, so a head fires when query-subspace content matches key-subspace content at relative offset ``dp``.
   Offset-selective roles use the highest-frequency rotary pairs (sharp positional selectivity); content roles
   use the lowest-frequency pairs (least positional modulation). ``sink_head`` compiles to a silent head.
3. SwiGLU memories: rule ``i`` gets neuron ``i`` with gate/up rows on the normalised key; ``W_down`` solves the
   ridge system ``W_down H = V`` in push-through form ``V (H^T H + lambda I)^-1 H^T`` (an M x M solve).
4. Optional spectral calibration clamps each synthesised projection to ``sigma_max <= gamma``.

RMSNorm awareness: the toy Qwen-style runtime applies RMSNorm (weight 1) before attention and MLP, which rescales
an input ``x`` to norm ``sqrt(d)``. Keys are pre-scaled the same way and attention gains divided by ``d`` so
compiled behaviour matches the forward pass exactly for inputs living in the declared subspaces.

Token embeddings are not specified by blueprints; they are seeded Gaussian rows (tied with the unembedding).
Hybrid grafting copies ``model.layers.{src}.*`` tensors from a safetensors checkpoint after synthesis.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from libweightspace.spectral import clamp_spectral_norm
from libweightspace.toy_models import ToyCausalLM, ToyConfig, rope_rotation_matrix

from .schemas import AttentionHeadBlueprint, DonorGraftSpec, MLPMemoryRule, NeuralBlueprint

DEFAULT_OFFSETS = {
    "previous_token_copy": 1,
    "induction_head": 0,
    "bracket_matcher": 0,
    "subspace_router": 0,
    "sink_head": 0,
}
POSITIONAL_ROLES = frozenset({"previous_token_copy"})


def procrustes_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the orthogonal Procrustes rotation ``R`` minimizing ``||source @ R - target||_F`` (NF-BL-08).

    Args:
        source: 2-D matrix ``[n, d]`` in donor coordinate space.
        target: 2-D matrix ``[n, d]`` in target blueprint coordinate space.

    Returns:
        Orthogonal matrix ``R`` of shape ``[d, d]`` satisfying ``R.T @ R = I``.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError(
            f"source and target must be 2-D tensors of identical shape, got {tuple(source.shape)} vs {tuple(target.shape)}"
        )
    cross = source.detach().double().T @ target.detach().double()  # [d, d]
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    r = u @ vh
    return r.to(dtype=source.dtype, device=source.device)


class BlueprintCompiler:
    """Synthesizes neural network weights from scratch without gradient descent."""

    def __init__(
        self,
        blueprint: NeuralBlueprint,
        seed: int = 42,
        ridge_lambda: float = 1e-4,
        spectral_gamma: float | None = None,
        rmsnorm_aware: bool = True,
    ):
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.bp = blueprint
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self.ridge_lambda = ridge_lambda
        self.spectral_gamma = spectral_gamma
        self.rmsnorm_aware = rmsnorm_aware
        self.basis = self._build_orthonormal_basis()
        self.subspace_map = {s.name: (s.start_dim, s.end_dim) for s in blueprint.subspaces}

    def _build_orthonormal_basis(self) -> torch.Tensor:
        """Generates a deterministic orthonormal basis [d_model, d_model] (sign-fixed QR)."""
        raw = torch.randn(self.bp.hidden_size, self.bp.hidden_size, generator=self.rng)
        q, r = torch.linalg.qr(raw)
        d = torch.diag(r)
        return q * (d / (d.abs() + 1e-12))

    def subspace_basis(self, name: str) -> torch.Tensor:
        """Orthonormal columns ``[d_model, dim]`` assigned to subspace ``name``.

        Raises:
            KeyError: unknown subspace.
        """
        if name not in self.subspace_map:
            raise KeyError(f"unknown subspace '{name}'")
        start, end = self.subspace_map[name]
        return self.basis[:, start:end]

    def embed_in_subspace(self, name: str, coords: torch.Tensor) -> torch.Tensor:
        """Maps local coordinates ``[..., dim]`` of subspace ``name`` into the residual stream ``[..., d]``."""
        return coords @ self.subspace_basis(name).T

    # --- MLP ------------------------------------------------------------------------------------------

    def compile_mlp_rules(
        self, rules: list[MLPMemoryRule]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compiles symbolic key-value rules into SwiGLU ``(W_gate, W_up, W_down)`` matrices.

        Raises:
            ValueError: vector dimensions disagree with their subspaces, or rules exceed ``intermediate_size``.
        """
        d_model, d_inter = self.bp.hidden_size, self.bp.intermediate_size
        w_gate = torch.zeros(d_inter, d_model)
        w_up = torch.zeros(d_inter, d_model)
        if not rules:
            return w_gate, w_up, torch.zeros(d_model, d_inter)
        total = sum(len(r.key_vectors) for r in rules)
        if total > d_inter:
            raise ValueError(f"{total} memory rules exceed intermediate_size={d_inter}")

        keys, vals = [], []
        cursor = 0
        for rule in rules:
            b_in, b_out = self.subspace_basis(rule.input_subspace), self.subspace_basis(rule.output_subspace)
            k_local = torch.tensor(rule.key_vectors, dtype=torch.float32)
            v_local = torch.tensor(rule.value_vectors, dtype=torch.float32)
            if k_local.shape[1] != b_in.shape[1] or v_local.shape[1] != b_out.shape[1]:
                raise ValueError(
                    f"rule '{rule.rule_id}': key dim {k_local.shape[1]} / value dim {v_local.shape[1]} must equal "
                    f"subspace dims {b_in.shape[1]} / {b_out.shape[1]}"
                )
            k_full = b_in @ k_local.T  # [d_model, N]
            v_full = b_out @ v_local.T  # [d_model, N]
            if self.rmsnorm_aware:
                # RMSNorm(weight=1) maps x to x * sqrt(d) / ||x||; solve for what the MLP will actually see.
                k_full = k_full * (math.sqrt(d_model) / (k_full.norm(dim=0, keepdim=True) + 1e-8))
            gain = 1.0 / max(rule.gating_threshold, 0.1)
            for idx in range(k_full.shape[1]):
                k_unit = k_full[:, idx] / (torch.norm(k_full[:, idx]) + 1e-8)
                w_gate[cursor] = k_unit * gain
                w_up[cursor] = k_unit
                cursor += 1
            keys.append(k_full)
            vals.append(v_full)

        k_mat, v_mat = torch.cat(keys, dim=1), torch.cat(vals, dim=1)  # [d_model, M]
        h = torch.nn.functional.silu(w_gate @ k_mat) * (w_up @ k_mat)  # [d_inter, M]
        gram = h.T @ h + self.ridge_lambda * torch.eye(h.shape[1])  # symmetric positive definite [M, M]
        coeffs = torch.linalg.solve(gram, v_mat.T).T  # V (H^T H + lambda I)^-1
        return w_gate, w_up, coeffs @ h.T

    # --- Attention ------------------------------------------------------------------------------------

    def _rotary_dims(self, positional: bool) -> list[int]:
        """Head dims ordered by rotary pair frequency (highest first if ``positional``, else lowest first)."""
        half = self.bp.head_dim // 2
        pairs = list(range(half)) if positional else list(range(half - 1, -1, -1))
        return [d for i in pairs for d in (i, i + half)]

    def compile_attention_heads(self, heads: list[AttentionHeadBlueprint]) -> dict[str, torch.Tensor]:
        """Synthesises ``q/k/v/o`` projection matrices for one layer.

        Raises:
            ValueError: two heads sharing a GQA key/value head require different key/value subspaces.
        """
        bp = self.bp
        d, hd, n_h, n_kv = bp.hidden_size, bp.head_dim, bp.num_attention_heads, bp.num_key_value_heads
        q = torch.zeros(n_h * hd, d)
        k = torch.zeros(n_kv * hd, d)
        v = torch.zeros(n_kv * hd, d)
        o = torch.zeros(d, n_h * hd)
        group = n_h // n_kv
        kv_owner: dict[int, tuple[str, str]] = {}
        norm_gain = 1.0 / d if self.rmsnorm_aware else 1.0  # undo RMSNorm's sqrt(d) on both q and k
        for head in heads:
            if head.role == "sink_head":
                continue  # silent head: zero projections
            kv_idx = head.head_idx // group
            signature = (head.key_subspace, head.value_subspace)
            if kv_owner.setdefault(kv_idx, signature) != signature:
                raise ValueError(
                    f"heads in GQA group {kv_idx} need different key/value subspaces {kv_owner[kv_idx]} vs {signature}"
                )
            b_q, b_k = self.subspace_basis(head.query_subspace), self.subspace_basis(head.key_subspace)
            b_v, b_o = self.subspace_basis(head.value_subspace), self.subspace_basis(head.output_subspace)
            positional = head.role in POSITIONAL_ROLES or head.positional_offset is not None
            r_qk = min(b_q.shape[1], b_k.shape[1], hd)
            dims = self._rotary_dims(positional)[:r_qk]
            p_qk = torch.eye(hd)[:, dims]  # [hd, r_qk], orthonormal columns
            offset = (
                head.positional_offset if head.positional_offset is not None else DEFAULT_OFFSETS[head.role]
            )
            rot = rope_rotation_matrix(hd, -float(offset), bp.rope_theta)
            scale = head.routing_strength * math.sqrt(hd) * norm_gain
            rows_q = slice(head.head_idx * hd, (head.head_idx + 1) * hd)
            rows_kv = slice(kv_idx * hd, (kv_idx + 1) * hd)
            q[rows_q] = scale * rot @ p_qk @ b_q[:, :r_qk].T
            k[rows_kv] = p_qk @ b_k[:, :r_qk].T
            r_vo = min(b_v.shape[1], b_o.shape[1], hd)
            p_vo = torch.eye(hd)[:, :r_vo]
            v[rows_kv] = p_vo @ b_v[:, :r_vo].T
            o[:, rows_q] = b_o[:, :r_vo] @ p_vo.T
        return {"q_proj": q, "k_proj": k, "v_proj": v, "o_proj": o}

    # --- Whole model ----------------------------------------------------------------------------------

    def toy_config(self) -> ToyConfig:
        """Runtime config: Qwen-style decoder without QK-norm (keeps the bilinear construction exact)."""
        bp = self.bp
        return ToyConfig(
            model_type="qwen3",
            vocab_size=bp.vocab_size,
            hidden_size=bp.hidden_size,
            intermediate_size=bp.intermediate_size,
            num_hidden_layers=bp.num_hidden_layers,
            num_attention_heads=bp.num_attention_heads,
            num_key_value_heads=bp.num_key_value_heads,
            head_dim=bp.head_dim,
            rms_norm_eps=bp.rms_norm_eps,
            rope_theta=bp.rope_theta,
            use_qk_norm=False,
            tie_word_embeddings=True,
        )

    def _calibrate(self, w: torch.Tensor) -> torch.Tensor:
        if self.spectral_gamma is None or not bool(w.any()):
            return w
        return clamp_spectral_norm(w, self.spectral_gamma)

    def compile(self) -> dict[str, torch.Tensor]:
        """Builds the complete state dict for ``ToyCausalLM(self.toy_config())``."""
        bp = self.bp
        d = bp.hidden_size
        embed_rng = torch.Generator().manual_seed(self.seed + 1)
        embed = torch.randn(bp.vocab_size, d, generator=embed_rng)
        state: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": embed,
            "lm_head.weight": embed,
            "model.norm.weight": torch.ones(d),
        }
        by_idx = {layer.layer_idx: layer for layer in bp.layers}
        for i in range(bp.num_hidden_layers):
            spec = by_idx.get(i)
            heads = spec.attention_heads if spec else []
            rules = spec.mlp_rules if spec else []
            attn = self.compile_attention_heads(heads)
            gate, up, down = self.compile_mlp_rules(rules)
            prefix = f"model.layers.{i}"
            state[f"{prefix}.input_layernorm.weight"] = torch.ones(d)
            state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(d)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                state[f"{prefix}.self_attn.{name}.weight"] = self._calibrate(attn[name])
            state[f"{prefix}.mlp.gate_proj.weight"] = gate
            state[f"{prefix}.mlp.up_proj.weight"] = up
            state[f"{prefix}.mlp.down_proj.weight"] = self._calibrate(down)
            if spec and spec.graft_source_checkpoint is not None:
                self._graft(state, i, spec.graft_source_checkpoint, spec.graft_source_layer)
        if bp.donor_graft is not None:
            self._apply_donor_graft(state, bp.donor_graft)
        return state

    def _apply_donor_graft(self, state: dict[str, torch.Tensor], graft: DonorGraftSpec) -> None:
        """Transplants donor embeddings and layers into ``state`` using Orthogonal Procrustes alignment."""
        path = Path(graft.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"donor graft checkpoint not found: {graft.checkpoint_path}")
        tensors = load_file(str(path))
        d = self.bp.hidden_size
        rot = torch.eye(d, dtype=torch.float32)

        donor_embed = tensors.get("model.embed_tokens.weight")
        if donor_embed is not None and tuple(donor_embed.shape) == tuple(state["model.embed_tokens.weight"].shape):
            if graft.procrustes_align:
                rot = procrustes_align(donor_embed.float(), state["model.embed_tokens.weight"].float())
            if graft.transplant_embeddings:
                aligned_embed = donor_embed.float() @ rot
                state["model.embed_tokens.weight"] = aligned_embed.clone()
                state["lm_head.weight"] = aligned_embed.clone()

        for src_idx, dst_idx in zip(graft.donor_layers, graft.target_layers, strict=True):
            self._graft(state, dst_idx, graft.checkpoint_path, src_idx)
            if graft.procrustes_align:
                prefix = f"model.layers.{dst_idx}"
                # Residual vector x_target = x_donor @ R => W_in_target = W_in_donor @ R, W_out_target = R^T @ W_out_donor
                for in_proj in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                ):
                    key = f"{prefix}.{in_proj}"
                    if key in state and state[key].shape[1] == d:
                        state[key] = state[key] @ rot
                for out_proj in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
                    key = f"{prefix}.{out_proj}"
                    if key in state and state[key].shape[0] == d:
                        state[key] = rot.T @ state[key]

    def _graft(self, state: dict[str, torch.Tensor], dst: int, checkpoint: str, src: int | None) -> None:
        """Copies ``model.layers.{src}.*`` from a safetensors checkpoint into layer ``dst``.

        Raises:
            FileNotFoundError / ValueError: missing file, no matching tensors, or shape mismatch.
        """
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"graft checkpoint not found: {checkpoint}")
        src_idx = dst if src is None else src
        src_prefix, dst_prefix = f"model.layers.{src_idx}.", f"model.layers.{dst}."
        tensors = load_file(str(path))
        copied = 0
        for name, tensor in tensors.items():
            if not name.startswith(src_prefix):
                continue
            target = dst_prefix + name[len(src_prefix) :]
            if target not in state:
                continue
            if tuple(state[target].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"graft shape mismatch for {target}: {tuple(tensor.shape)} vs {tuple(state[target].shape)}"
                )
            state[target] = tensor.float().clone()
            copied += 1
        if copied == 0:
            raise ValueError(f"graft found no tensors under '{src_prefix}' in {checkpoint}")

    def build_model(self) -> ToyCausalLM:
        """Instantiates the runtime model and loads the compiled weights (strict)."""
        model = ToyCausalLM(self.toy_config())
        model.load_state_dict(self.compile(), strict=True)
        return model.eval()
