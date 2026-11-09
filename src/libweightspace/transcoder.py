"""Sparse transcoder monosemantic feature surgery and cross-model weight diffing (NF-BL-04).

Provides:
    - ``MLPTranscoder``: sparse ReLU transcoder ``f(x) = ReLU(W_enc x + b_enc)`` with decoder ``W_dec f(x) + b_dec``
      that attributes contrastive activation differences to sparse monosemantic features and folds a feature
      intervention ``scale * w_dec[:, f]`` directly into a layer's ``down_proj`` weight matrix ``W_down`` as a
      zero-runtime-overhead rank-1 weight update.
    - ``diff_models``: crosscoder-style parameter diffing between two checkpoints/models, reporting per-tensor
      Frobenius norms, spectral norms, and top singular directions of ``W_B - W_A``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class FeatureAttribution:
    """Contrastive attribution for a single sparse transcoder feature."""

    feature_idx: int
    differential_activation: float
    mean_fail_activation: float
    mean_pass_activation: float
    decoder_norm: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return asdict(self)


@dataclass
class TensorModelDiff:
    """Per-tensor diff summary between two models."""

    tensor_name: str
    shape: list[int]
    frobenius_diff: float
    relative_frobenius: float
    spectral_diff: float
    top_singular_values: list[float]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return asdict(self)


class MLPTranscoder(nn.Module):
    """Sparse ReLU transcoder mapping MLP inputs ``[*, d_model]`` to sparse features ``[*, d_features]``."""

    def __init__(self, d_model: int, d_features: int, seed: int = 0):
        super().__init__()
        if d_model < 1 or d_features < 1:
            raise ValueError("d_model and d_features must be >= 1")
        self.d_model = d_model
        self.d_features = d_features
        rng = torch.Generator().manual_seed(seed)
        w_enc = torch.randn(d_features, d_model, generator=rng) / (d_model**0.5)
        w_dec = torch.randn(d_model, d_features, generator=rng)
        w_dec = w_dec / (torch.linalg.vector_norm(w_dec, dim=0, keepdim=True) + 1e-8)
        self.w_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(d_features))
        self.w_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sparse feature activations ``ReLU(x @ W_enc^T + b_enc)``."""
        return torch.relu(x.float() @ self.w_enc.T + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Maps sparse feature activations back to ``d_model`` output space."""
        return features.float() @ self.w_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full transcoder reconstruction ``decode(encode(x))``."""
        return self.decode(self.encode(x))

    @torch.no_grad()
    def attribute_contrastive(
        self,
        fail_inputs: torch.Tensor,
        pass_inputs: torch.Tensor,
        top_k: int = 8,
    ) -> list[FeatureAttribution]:
        """Ranks sparse features by ``mean(encode(fail_inputs)) - mean(encode(pass_inputs))``.

        Args:
            fail_inputs: Tensor ``[N_fail, d_model]`` of MLP inputs on failing prompts.
            pass_inputs: Tensor ``[N_pass, d_model]`` of MLP inputs on passing prompts.
            top_k: Number of highest-magnitude differential features to return.

        Returns:
            List of ``FeatureAttribution`` sorted by ``|differential_activation|`` descending.
        """
        f_fail = self.encode(fail_inputs.reshape(-1, self.d_model)).mean(dim=0)
        f_pass = self.encode(pass_inputs.reshape(-1, self.d_model)).mean(dim=0)
        diff = f_fail - f_pass
        k = max(1, min(top_k, self.d_features))
        _, indices = torch.topk(diff.abs(), k=k)
        dec_norms = torch.linalg.vector_norm(self.w_dec.detach().float(), dim=0)
        reports: list[FeatureAttribution] = []
        for idx_t in indices:
            idx = int(idx_t.item())
            reports.append(
                FeatureAttribution(
                    feature_idx=idx,
                    differential_activation=float(diff[idx].item()),
                    mean_fail_activation=float(f_fail[idx].item()),
                    mean_pass_activation=float(f_pass[idx].item()),
                    decoder_norm=float(dec_norms[idx].item()),
                )
            )
        return reports

    @torch.no_grad()
    def fold_feature_edit_into_down_proj(
        self,
        w_down: torch.Tensor,
        feature_idx: int,
        scale: float = -1.0,
        top_cols: int | None = None,
    ) -> torch.Tensor:
        """Folds a sparse feature steering/ablation edit directly into ``W_down`` with zero runtime overhead.

        Computes the rank-1 update ``delta_W_down = scale * (d_f @ a_f^T)`` where ``d_f = w_dec[:, feature_idx]``
        is the feature's output direction in ``R^{d_model}`` and ``a_f`` is the normalized alignment of
        ``W_down``'s columns with ``d_f`` (optionally restricted to ``top_cols`` columns).

        Args:
            w_down: MLP ``down_proj`` weight tensor ``[d_model, d_inter]``.
            feature_idx: Feature index in ``[0, d_features)``.
            scale: Steering or suppression scalar (negative suppresses the feature's output direction).
            top_cols: Optional column sparsity cap; when set, only the ``top_cols`` most aligned columns of
                ``w_down`` are modified.

        Returns:
            ``delta_W_down`` of the same shape, dtype, and device as ``w_down``.
        """
        if not 0 <= feature_idx < self.d_features:
            raise ValueError(f"feature_idx {feature_idx} out of range [0, {self.d_features})")
        if w_down.ndim != 2 or w_down.shape[0] != self.d_model:
            raise ValueError(
                f"w_down must have shape ({self.d_model}, d_inter), got {tuple(w_down.shape)}"
            )
        w_f = w_down.detach().float()
        d_f = self.w_dec.detach().float()[:, feature_idx].to(w_f.device)
        d_norm = torch.linalg.vector_norm(d_f)
        if float(d_norm.item()) <= 1e-12:
            return torch.zeros_like(w_down)
        d_unit = d_f / d_norm

        # Alignment of each column of W_down with d_unit: coeff_c = d_unit^T W_down[:, c]
        coeffs = d_unit @ w_f  # [d_inter]
        if top_cols is not None and 0 < top_cols < w_f.shape[1]:
            _, keep = torch.topk(coeffs.abs(), k=top_cols)
            mask = torch.zeros_like(coeffs)
            mask[keep] = 1.0
            coeffs = coeffs * mask

        delta = float(scale) * torch.outer(d_unit, coeffs)
        return delta.to(dtype=w_down.dtype, device=w_down.device)


@torch.no_grad()
def diff_models(
    model_a: nn.Module,
    model_b: nn.Module,
    top_k: int = 5,
) -> list[TensorModelDiff]:
    """Computes per-parameter weight-space diffs between ``model_a`` and ``model_b``, sorted by Frobenius diff."""
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    diffs: list[TensorModelDiff] = []
    for name, pa in params_a.items():
        if name not in params_b:
            continue
        pb = params_b[name]
        if pa.shape != pb.shape:
            continue
        da = pa.detach().float().cpu()
        db = pb.detach().float().cpu()
        delta = db - da
        frob_diff = float(torch.linalg.vector_norm(delta).item())
        if frob_diff <= 1e-12:
            continue
        frob_base = float(torch.linalg.vector_norm(da).item())
        mat = delta.reshape(delta.shape[0], -1) if delta.ndim >= 2 else delta.unsqueeze(0)
        s = torch.linalg.svdvals(mat)
        k = min(top_k, int(s.numel()))
        diffs.append(
            TensorModelDiff(
                tensor_name=name,
                shape=list(pa.shape),
                frobenius_diff=frob_diff,
                relative_frobenius=frob_diff / (frob_base + 1e-12),
                spectral_diff=float(s[0].item()) if s.numel() > 0 else 0.0,
                top_singular_values=[float(x) for x in s[:k].tolist()],
            )
        )
    diffs.sort(key=lambda d: d.frobenius_diff, reverse=True)
    return diffs
