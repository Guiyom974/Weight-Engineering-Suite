"""Monotonic nullspace locking of protected capability directions.

Maintains, per weight tensor ``W`` of shape ``[d_out, d_in]``, an orthonormal basis ``V_locked`` (``[d_in, r]``)
of input directions that earlier promoted edits rely on, and projects every new delta into the orthogonal
complement: ``dW_safe = dW - (dW @ V) @ V^T``. Then ``dW_safe @ v = 0`` for every locked ``v``, so later edits
cannot disturb how ``W`` responds to protected inputs.

New directions are added by Gram-Schmidt against the existing basis: directions already (numerically) inside
the locked span are dropped instead of being turned into spurious basis vectors, so ``locked_rank`` grows only
by genuinely new directions. Once ``max_rank_per_tensor`` is reached, the oldest directions are kept
(monotonic: earlier skills stay protected) and the overflow is reported via the return value.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors.torch import save as st_save

from .io_utils import write_atomic


class NullspaceLocker:
    """Guarantees monotonic skill accumulation by projecting dW into the nullspace of locked subspaces."""

    def __init__(self, max_rank_per_tensor: int = 128, tolerance: float = 1e-6):
        if max_rank_per_tensor < 1:
            raise ValueError("max_rank_per_tensor must be >= 1")
        self.max_rank = max_rank_per_tensor
        self.tolerance = tolerance
        # Maps tensor_name -> Orthonormal basis matrix V_locked of shape [d_in, current_rank]
        self._locked_bases: dict[str, torch.Tensor] = {}

    def project_into_nullspace(
        self,
        tensor_name: str,
        delta_W: torch.Tensor,
        preserve_sparsity: bool = False,
    ) -> torch.Tensor:
        """Projects a candidate 2D weight delta [d_out, d_in] into the right nullspace of V_locked.

        By default (``preserve_sparsity=False``), applies the minimum-Frobenius-norm projection
        ``dW - (dW @ V_locked) @ V_locked^T``. When ``preserve_sparsity=True`` and ``delta_W`` is
        column-sparse (touching ``k < d_in`` columns), keeps ``delta_W`` intact on its active support ``S``
        and cancels ``(delta_W[:, S] @ V[S, :])`` using the minimum-norm pseudoinverse over at most ``2 * r``
        high-leverage companion columns ``P`` outside ``S``, guaranteeing ``dW_safe @ V_locked == 0`` while
        bounding column fill-in to ``|S| + |P|``.

        Raises:
            ValueError: the delta's input dimension does not match the locked basis.
        """
        if tensor_name not in self._locked_bases or delta_W.ndim != 2:
            return delta_W
        basis = self._locked_bases[tensor_name]
        if delta_W.shape[1] != basis.shape[0]:
            raise ValueError(
                f"delta for '{tensor_name}' has d_in={delta_W.shape[1]}, locked basis has d_in={basis.shape[0]}"
            )
        v = basis.to(device=delta_W.device, dtype=torch.float32)
        dw = delta_W.float()
        d_in, rank = v.shape

        if preserve_sparsity:
            col_norms = dw.abs().sum(dim=0)
            active_cols = torch.nonzero(col_norms > 0, as_tuple=False).reshape(-1)
            k = int(active_cols.numel())
            companion_count = min(max(2 * rank, rank + 1), d_in - k)
            if 0 < k < d_in and companion_count >= rank and (k + companion_count) < d_in:
                leverage = v.norm(dim=1)
                leverage[active_cols] = -1.0
                pivots = torch.topk(leverage, k=companion_count).indices
                v_piv = v[pivots, :]  # [m, r]
                # Check that V[pivots, :] has full column rank r so compensation is exact.
                s_vals = torch.linalg.svdvals(v_piv)
                if int((s_vals > self.tolerance).sum().item()) == rank:
                    residual = dw[:, active_cols] @ v[active_cols, :]  # [d_out, r]
                    # Solve X @ v_piv = -residual via minimum-norm pseudoinverse: X = -residual @ pinv(v_piv)
                    comp = -residual @ torch.linalg.pinv(v_piv)
                    out = dw.clone()
                    out[:, pivots] = comp
                    return out.to(dtype=delta_W.dtype)

        return (dw - (dw @ v) @ v.T).to(dtype=delta_W.dtype)

    def compact_bases(self, max_rank: int | None = None) -> dict[str, int]:
        """Compacts locked bases via SVD to at most ``max_rank`` (defaults to ``self.max_rank // 2``).

        Returns:
            Mapping of tensor_name -> new rank after compaction.
        """
        target_rank = max_rank if max_rank is not None else max(1, self.max_rank // 2)
        if target_rank < 1:
            raise ValueError("max_rank must be >= 1")
        new_ranks: dict[str, int] = {}
        for name, basis in list(self._locked_bases.items()):
            if basis.shape[1] > target_rank:
                u, _, _ = torch.linalg.svd(basis, full_matrices=False)
                self._locked_bases[name] = u[:, :target_rank].contiguous()
            new_ranks[name] = int(self._locked_bases[name].shape[1])
        return new_ranks

    def register_protected_directions(
        self, tensor_name: str, input_activations: torch.Tensor, top_k: int = 2
    ) -> int:
        """Locks the top principal input directions of a newly promoted skill.

        Args:
            tensor_name: Name of the weight matrix [d_out, d_in].
            input_activations: Tensor of shape [N_tokens, d_in] recorded on passing target probes.
            top_k: Number of principal singular vectors to lock.

        Returns:
            Number of directions actually added (0..top_k).

        Raises:
            ValueError: activations are not 2-D, ``top_k`` < 1, or ``d_in`` mismatches the existing basis.
        """
        if input_activations.ndim != 2:
            raise ValueError("input_activations must have shape [N_tokens, d_in]")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        x = input_activations.detach().float().cpu()
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        keep = s > self.tolerance * (s[0] if s.numel() else 1.0)
        candidates = vh[: min(top_k, int(keep.sum().item()))].T  # [d_in, k]

        existing = self._locked_bases.get(tensor_name)
        if existing is not None and existing.shape[0] != x.shape[1]:
            raise ValueError(
                f"'{tensor_name}': d_in {x.shape[1]} does not match locked basis {existing.shape[0]}"
            )
        basis = existing if existing is not None else torch.zeros(x.shape[1], 0)
        added = 0
        for j in range(candidates.shape[1]):
            if basis.shape[1] >= self.max_rank:
                break
            v = candidates[:, j]
            # Two Gram-Schmidt passes for numerical orthogonality.
            for _ in range(2):
                v = v - basis @ (basis.T @ v)
            norm = float(v.norm())
            if norm <= 1e-4:
                continue
            basis = torch.cat([basis, (v / norm)[:, None]], dim=1)
            added += 1
        if basis.shape[1] > 0:
            self._locked_bases[tensor_name] = basis
        return added

    def locked_rank(self, tensor_name: str) -> int:
        """Current number of locked directions for ``tensor_name``."""
        if tensor_name not in self._locked_bases:
            return 0
        return int(self._locked_bases[tensor_name].shape[1])

    def locked_tensors(self) -> list[str]:
        """Names of tensors with at least one locked direction."""
        return sorted(self._locked_bases)

    def basis(self, tensor_name: str) -> torch.Tensor | None:
        """Copy of the locked basis ``[d_in, r]`` (None when nothing is locked)."""
        b = self._locked_bases.get(tensor_name)
        return None if b is None else b.clone()

    def restore_basis(self, tensor_name: str, basis: torch.Tensor | None) -> None:
        """Sets (or with None, removes) the locked basis of ``tensor_name``; used to undo locks on rollback."""
        if basis is None:
            self._locked_bases.pop(tensor_name, None)
        else:
            self._locked_bases[tensor_name] = basis.detach().float().clone()

    def save(self, path: Path) -> None:
        """Persists all bases to a safetensors file."""
        write_atomic(path, st_save({k: v.contiguous() for k, v in self._locked_bases.items()}))

    @classmethod
    def load(cls, path: Path, max_rank_per_tensor: int = 128) -> NullspaceLocker:
        """Restores a locker saved with ``save`` (empty locker when the file does not exist)."""
        locker = cls(max_rank_per_tensor)
        if path.exists():
            locker._locked_bases = {k: v.float() for k, v in load_file(str(path)).items()}
        return locker
