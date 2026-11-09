"""Singular-value profiling, effective rank and spectral-norm utilities.

Exact SVD is used up to ``EXACT_SVD_MAX_ELEMENTS``; larger matrices use randomized low-rank SVD (seeded, so
results are reproducible) and deterministic power iteration for the spectral norm. All computations run in
float32 on the tensor's device and never modify the input.

Main entry points: ``profile_matrix`` (backs the ``inspect_layer_svd`` agent tool), ``spectral_norm``,
``spectral_norm_shift`` (edit-constraint check), ``clamp_spectral_norm`` (blueprint calibration).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

EXACT_SVD_MAX_ELEMENTS = 16_000_000
_LOWRANK_SEED = 1234


@dataclass
class SpectralProfile:
    """Summary of a weight matrix's spectrum (condition number is over the computed spectrum)."""

    tensor_name: str
    shape: list[int]
    top_singular_values: list[float]
    effective_rank: float
    condition_number: float
    frobenius_norm: float
    spectral_norm: float
    power_law_alpha: float = 0.0
    mp_bulk_edge: float = 0.0
    snr_ratio: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view."""
        return asdict(self)


def _as_matrix(w: torch.Tensor) -> torch.Tensor:
    w = w.detach().float()
    if w.ndim == 1:
        return w[None, :]
    if w.ndim > 2:
        return w.reshape(w.shape[0], -1)
    return w


def singular_values(w: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Descending singular values (all of them, or the top ``k``)."""
    m = _as_matrix(w)
    full_k = min(m.shape)
    k = full_k if k is None else max(1, min(k, full_k))
    if m.numel() <= EXACT_SVD_MAX_ELEMENTS:
        return torch.linalg.svdvals(m)[:k]
    with torch.random.fork_rng(devices=[m.device] if m.is_cuda else []):
        torch.manual_seed(_LOWRANK_SEED)
        _, s, _ = torch.svd_lowrank(m, q=min(k + 8, full_k), niter=4)
    return s[:k]


def spectral_norm(w: torch.Tensor, iters: int = 64) -> float:
    """Largest singular value (exact for small matrices, deterministic power iteration otherwise)."""
    m = _as_matrix(w)
    if m.numel() <= EXACT_SVD_MAX_ELEMENTS:
        return float(torch.linalg.matrix_norm(m, ord=2).item())
    v = torch.ones(m.shape[1], device=m.device) / m.shape[1] ** 0.5
    sigma = 0.0
    for _ in range(iters):
        u = m @ v
        u = u / (u.norm() + 1e-12)
        v = m.T @ u
        sigma = float(v.norm().item())
        v = v / (sigma + 1e-12)
    return sigma


def effective_rank(w: torch.Tensor) -> float:
    """Entropy effective rank ``exp(H(p))`` with ``p = s / sum(s)`` (Roy & Vetterli, 2007)."""
    s = singular_values(w)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def estimate_power_law_alpha(s: torch.Tensor) -> float:
    """Estimates the HTSR Heavy-Tailed Self-Regularization power-law exponent ``alpha`` (Martin & Mahoney).

    Uses the Clauset-Shalizi-Newman Hill maximum-likelihood estimator over the upper tail of empirical
    eigenvalues ``lambda_i = sigma_i^2``:
        ``alpha = 1 + k / sum_{i=1..k} ln(lambda_i / lambda_min)``
    """
    pos = s.detach().float()
    pos = pos[pos > 1e-12]
    if pos.numel() < 4:
        return 0.0
    evals = pos.square()
    k = max(2, evals.numel() // 2)
    tail = evals[:k]
    xmin = tail[-1].clamp_min(1e-12)
    logs = torch.log((tail / xmin).clamp_min(1.0))
    denom = float(logs.sum().item())
    if denom <= 1e-12:
        return float("inf")
    return float(1.0 + k / denom)


def estimate_marchenko_pastur(s: torch.Tensor, shape: tuple[int, int]) -> tuple[float, float]:
    """Estimates the Marchenko-Pastur bulk noise edge ``lambda_+`` and signal-to-noise ratio (SNR).

    Args:
        s: Descending singular values of a 2-D matrix.
        shape: Matrix dimensions ``(d_out, d_in)``.

    Returns:
        Tuple ``(mp_bulk_edge, snr_ratio)`` where ``mp_bulk_edge`` is the estimated upper edge of the random
        bulk in eigenvalue space (``sigma^2``) and ``snr_ratio`` is the fraction of total spectral energy
        residing above ``mp_bulk_edge``.
    """
    pos = s.detach().float()
    pos = pos[pos > 1e-12]
    if pos.numel() == 0:
        return 0.0, 0.0
    evals = pos.square()
    d_min = max(1, min(shape[0], shape[1]))
    d_max = max(1, max(shape[0], shape[1]))
    q = d_min / d_max
    # Robust bulk variance estimate from the lower half of the spectrum.
    lower_half = evals[evals.numel() // 2 :]
    sigma_sq = float(lower_half.median().item()) if lower_half.numel() > 0 else float(evals.median().item())
    mp_edge = sigma_sq * (1.0 + (q**0.5)) ** 2
    total_energy = float(evals.sum().item())
    if total_energy <= 1e-12:
        return mp_edge, 0.0
    signal_energy = float(evals[evals > mp_edge].sum().item())
    return float(mp_edge), float(signal_energy / total_energy)


def profile_matrix(w: torch.Tensor, tensor_name: str = "", top_k: int = 16) -> SpectralProfile:
    """Top-k singular values, effective rank, condition number, Frobenius/spectral norms, HTSR alpha and MP SNR."""
    m = _as_matrix(w)
    s_top = singular_values(w, top_k)
    s_all = singular_values(w) if m.numel() <= EXACT_SVD_MAX_ELEMENTS else s_top
    smallest = float(s_all[-1].item()) if s_all.numel() else 0.0
    largest = float(s_all[0].item()) if s_all.numel() else 0.0
    alpha = estimate_power_law_alpha(s_all)
    mp_edge, snr = estimate_marchenko_pastur(s_all, (int(m.shape[0]), int(m.shape[1])))
    return SpectralProfile(
        tensor_name=tensor_name,
        shape=list(w.shape),
        top_singular_values=[float(x) for x in s_top.tolist()],
        effective_rank=effective_rank(w),
        condition_number=largest / smallest if smallest > 0 else float("inf"),
        frobenius_norm=float(torch.linalg.vector_norm(w.detach().float()).item()),
        spectral_norm=largest,
        power_law_alpha=alpha,
        mp_bulk_edge=mp_edge,
        snr_ratio=snr,
    )


def spectral_norm_shift(w: torch.Tensor, delta: torch.Tensor) -> float:
    """Relative change of the spectral norm caused by adding ``delta``: ``|s(W+D) - s(W)| / s(W)``."""
    base = spectral_norm(w)
    after = spectral_norm(w.detach().float() + delta.detach().float().to(w.device))
    return abs(after - base) / (base + 1e-12)


def clamp_spectral_norm(w: torch.Tensor, max_sigma: float) -> torch.Tensor:
    """Returns ``w`` rescaled so its spectral norm is at most ``max_sigma`` (unchanged when already below)."""
    if max_sigma <= 0:
        raise ValueError("max_sigma must be positive")
    sigma = spectral_norm(w)
    if sigma <= max_sigma or sigma == 0.0:
        return w.clone()
    return w * (max_sigma / sigma)
