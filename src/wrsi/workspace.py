"""On-disk W-RSI workspace so self-improvement survives across CLI invocations.

Layout (``<root>/.wrsi/``)::

    state.json                  GenerationState
    ledger.db                   EpisodicWeightLedger
    overlay.safetensors         current values of every tensor ever modified (base + overlay = M_t)
    nullspace.safetensors       locked bases
    rollback/gen_00000.safetensors   pre-edit slices + pre-promotion nullspace bases of each generation
    vault/anchors.safetensors   anchor sequences + M_0 reference log-probs (read only by the vault process)
    vault/m0.safetensors        M_0 weights for the vault process

Rollback files store the exact original slices (column mask + values), so ``pop_rollback`` restores M_{t-1}
bit-exactly, like the in-memory rollback stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from libweightspace.cow_overlay import SliceBackup
from libweightspace.io_utils import write_atomic
from libweightspace.nullspace import NullspaceLocker

from .anchor_client import anchor_ids_from_texts, save_base_weights
from .anchor_vault_server import build_anchor_vault, file_sha256
from .episodic_ledger import EpisodicWeightLedger
from .schemas import GenerationState
from .scientist_agent import PromotedGeneration


class WRSIWorkspace:
    """Paths and persistence helpers for one W-RSI run."""

    def __init__(self, root: Path):
        # No directories here: read-only commands must not create anything (writers make their own parents).
        self.dir = Path(root) / ".wrsi"
        self._popped: list[tuple[Path, str]] = []

    @property
    def state_path(self) -> Path:
        """``state.json``."""
        return self.dir / "state.json"

    def exists(self) -> bool:
        """True once a run has saved state."""
        return self.state_path.is_file()

    def load_state(self) -> GenerationState:
        """Saved state (defaults when none)."""
        if not self.exists():
            return GenerationState()
        return GenerationState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: GenerationState) -> None:
        """Writes ``state.json``."""
        write_atomic(self.state_path, state.model_dump_json(indent=2).encode())

    def ledger(self) -> EpisodicWeightLedger:
        """The run's episodic ledger."""
        return EpisodicWeightLedger(self.dir / "ledger.db")

    # --- weights ----------------------------------------------------------------------------------------

    def apply_overlay(self, model: nn.Module) -> list[str]:
        """Overwrites modified tensors with their saved current values; returns their names."""
        path = self.dir / "overlay.safetensors"
        if not path.is_file():
            return []
        params = dict(model.named_parameters())
        tensors = st_load(path.read_bytes())
        with torch.no_grad():
            for name, value in tensors.items():
                params[name].copy_(value.to(params[name].dtype))
        return sorted(tensors)

    def save_overlay(self, model: nn.Module, names: list[str]) -> None:
        """Persists current values of ``names`` (removes the overlay when empty)."""
        path = self.dir / "overlay.safetensors"
        if not names:
            path.unlink(missing_ok=True)
            return
        params = dict(model.named_parameters())
        write_atomic(path, st_save({n: params[n].detach().cpu().contiguous().clone() for n in names}))

    def load_nullspace(self) -> NullspaceLocker:
        """Saved nullspace bases (empty locker when none)."""
        return NullspaceLocker.load(self.dir / "nullspace.safetensors")

    def save_nullspace(self, locker: NullspaceLocker) -> None:
        """Persists nullspace bases."""
        locker.save(self.dir / "nullspace.safetensors")

    # --- rollback ---------------------------------------------------------------------------------------

    def save_rollback(self, promoted: PromotedGeneration) -> Path:
        """Writes one generation's undo information."""
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, Any] = {"backups": [], "nullspace_none": []}
        for i, b in enumerate(promoted.backups):
            tensors[f"backup.{i}.values"] = b.values.detach().cpu().contiguous()
            if b.mask is not None:
                tensors[f"backup.{i}.mask"] = b.mask.to(torch.uint8).cpu().contiguous()
            meta["backups"].append(b.name)
        for name, basis in promoted.nullspace_before.items():
            if basis is None:
                meta["nullspace_none"].append(name)
            else:
                tensors[f"nullspace.{len(meta.setdefault('nullspace', []))}"] = basis.contiguous()
                meta["nullspace"].append(name)
        path = self.dir / "rollback" / f"gen_{promoted.generation:05d}.safetensors"
        write_atomic(
            path, st_save(tensors, metadata={"wrsi": json.dumps(meta), "trial_id": promoted.trial_id})
        )
        return path

    def rollback_files(self) -> list[Path]:
        """Rollback files not yet popped, oldest first."""
        popped = {path for path, _ in self._popped}
        return sorted(p for p in (self.dir / "rollback").glob("gen_*.safetensors") if p not in popped)

    def pop_rollback(self, model: nn.Module, locker: NullspaceLocker) -> int | None:
        """Undoes the newest saved generation in ``model`` and ``locker``; returns its generation number."""
        files = self.rollback_files()
        if not files:
            return None
        path = files[-1]
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
            meta = json.loads(metadata["wrsi"])
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        params = dict(model.named_parameters())
        for i in reversed(range(len(meta["backups"]))):
            name = meta["backups"][i]
            mask = tensors.get(f"backup.{i}.mask")
            SliceBackup(name, None if mask is None else mask.bool(), tensors[f"backup.{i}.values"]).restore(
                params[name]
            )
        for j, name in enumerate(meta.get("nullspace", [])):
            locker.restore_basis(name, tensors[f"nullspace.{j}"])
        for name in meta["nullspace_none"]:
            locker.restore_basis(name, None)
        self._popped.append((path, metadata["trial_id"]))
        return int(path.stem.split("_")[1])

    def commit_rollback(self) -> None:
        """Finalises popped generations; call only after the overlay and state are saved.

        Undo files are deleted last, so a crash during rollback can never leave M_t on disk with no undo.
        """
        ledger = self.ledger()
        for path, trial_id in self._popped:
            ledger.update_outcome(trial_id, False, "Rolled back")
            path.unlink()
        self._popped.clear()

    # --- anchor vault -----------------------------------------------------------------------------------

    def ensure_vault(
        self,
        model: nn.Module,
        tokenizer: Any,
        anchor_texts: list[str],
        state: GenerationState | None = None,
    ) -> tuple[Path, Path]:
        """Creates the vault from the current model (M_0) on first use and verifies SHA-256 pins."""
        vault = self.dir / "vault" / "anchors.safetensors"
        base = self.dir / "vault" / "m0.safetensors"
        if not vault.is_file() or not base.is_file():
            if state is not None and (state.vault_sha256 is not None or state.base_weights_sha256 is not None):
                raise ValueError("anchor vault files missing despite existing state SHA-256 pin")
            build_anchor_vault(model, anchor_ids_from_texts(tokenizer, anchor_texts), vault)
            save_base_weights(model, base)
        v_sha = file_sha256(vault)
        b_sha = file_sha256(base)
        if state is not None:
            if state.vault_sha256 is None:
                state.vault_sha256 = v_sha
            elif state.vault_sha256.lower() != v_sha.lower():
                raise ValueError("anchor vault integrity check failed: anchors.safetensors SHA-256 mismatch")
            if state.base_weights_sha256 is None:
                state.base_weights_sha256 = b_sha
            elif state.base_weights_sha256.lower() != b_sha.lower():
                raise ValueError("anchor vault integrity check failed: m0.safetensors SHA-256 mismatch")
        return vault, base
