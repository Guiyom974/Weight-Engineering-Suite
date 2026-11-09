"""Zero-duplication Copy-on-Write (CoW) weight overlay for fast sandboxed evaluation.

``CoWWeightOverlay.apply_delta`` adds deltas to a model's parameters in place and restores the original values
on exit, bitwise, even when the body raises. For 2-D parameters only the columns the delta touches are backed
up (a 64-column edit of a 16 GB model backs up 64 columns, not the model), which is what lets K candidates be
evaluated without K model copies.

Also provides ``SliceBackup`` (the same column-level snapshot) for permanent commits that must stay bit-exactly
revertible, e.g. the W-RSI rollback stack.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SliceBackup:
    """Original values of the slice of one parameter that a delta modifies.

    ``mask`` selects columns for 2-D parameters; ``None`` means the whole tensor was saved.
    """

    name: str
    mask: torch.Tensor | None
    values: torch.Tensor

    def restore(self, param: torch.Tensor) -> None:
        """Writes the saved values back into ``param`` (bit-exact)."""
        with torch.no_grad():
            if self.mask is None:
                param.data.copy_(self.values)
            else:
                param.data[:, self.mask] = self.values


def _validate_delta(name: str, param: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if not isinstance(delta, torch.Tensor):
        raise TypeError(f"delta for '{name}' must be a torch.Tensor, got {type(delta).__name__}")
    if tuple(delta.shape) != tuple(param.shape):
        raise ValueError(
            f"delta for '{name}' has shape {tuple(delta.shape)}, parameter has {tuple(param.shape)}"
        )
    if not torch.isfinite(delta).all():
        raise ValueError(f"delta for '{name}' contains NaN or Inf")
    return delta.to(device=param.device, dtype=param.dtype)


def apply_delta_with_backup(param: torch.Tensor, name: str, delta: torch.Tensor) -> SliceBackup | None:
    """Adds ``delta`` to ``param`` in place and returns a backup of the touched slice.

    Returns None when the delta is all zeros (nothing modified).
    """
    delta_dev = _validate_delta(name, param, delta)
    with torch.no_grad():
        if delta_dev.ndim == 2:
            col_mask = delta_dev.abs().sum(dim=0) > 0
            if not bool(col_mask.any()):
                return None
            backup = SliceBackup(name, col_mask, param.data[:, col_mask].clone())
            param.data[:, col_mask] += delta_dev[:, col_mask]
            return backup
        if not bool((delta_dev != 0).any()):
            return None
        backup = SliceBackup(name, None, param.data.clone())
        param.data.add_(delta_dev)
        return backup


class CoWWeightOverlay:
    """Applies temporary, reversible in-place weight deltas to a frozen base model.

    Guarantees bitwise restoration of base weights upon context exit, even if an exception occurs.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._param_dict: dict[str, nn.Parameter] = dict(model.named_parameters())

    def parameter(self, name: str) -> nn.Parameter:
        """Looks up a parameter by name.

        Raises:
            KeyError: the model has no such parameter.
        """
        if name not in self._param_dict:
            raise KeyError(f"Parameter '{name}' not found in target model.")
        return self._param_dict[name]

    @contextmanager
    def apply_delta(self, delta_dict: Mapping[str, torch.Tensor]) -> Iterator[nn.Module]:
        """Context manager that applies sparse/low-rank deltas and restores originals on exit.

        Args:
            delta_dict: Mapping of parameter_name -> delta tensor (same shape as the parameter).

        Raises:
            KeyError / ValueError / TypeError: unknown parameter, shape mismatch, non-finite or non-tensor delta.
            Any deltas applied before the failing one are restored before the error propagates.
        """
        backups: list[SliceBackup] = []
        try:
            for name, delta in delta_dict.items():
                backup = apply_delta_with_backup(self.parameter(name), name, delta)
                if backup is not None:
                    backups.append(backup)
            yield self.model
        finally:
            # Restore in reverse order so a parameter listed twice ends at its true original value.
            for backup in reversed(backups):
                backup.restore(self._param_dict[backup.name])
