"""Git-for-Weights: content-addressed DAG versioning for sparse weight deltas.

Storage layout (blueprint section 4)::

    .wgit/
    |-- HEAD                              current patch_id (or "ROOT")
    |-- refs/heads/main                   latest promoted patch_id
    |-- commits/<patch_id>.json           PatchManifest (+ delta_sha256, created_at)
    `-- objects/<patch_id>.COO.safetensors

Each modified tensor ``name`` is stored as ``{name}__indices`` (int32 ``[ndim, nnz]``), ``{name}__values``
(float32/bfloat16/float16 ``[nnz]``) and ``{name}__shape`` (int64 ``[ndim]``). Weight-space patches touch
< 0.05% of parameters, so this costs a tiny fraction of a dense checkpoint.

Patch ids are the first 16 hex chars of ``SHA-256(parent | object_sha256 | synthesized_code)``; ids are
validated against ``^[0-9a-f]{16}$`` before they are ever used in a path. ``verify`` re-hashes an object against
its manifest. Writes go through a temp file + ``os.replace`` so a crash never leaves a half-written ref.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from libweightspace.io_utils import write_atomic as _write_atomic

from .schemas import PatchManifest

ROOT = "ROOT"
PATCH_ID_RE = re.compile(r"^[0-9a-f]{16}$")
BRANCH_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")
_VALUE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _derive_patch_id(parent: str, object_sha: str, code: str) -> str:
    return hashlib.sha256(f"{parent}|{object_sha}|{code}".encode()).hexdigest()[:16]


class WeightGitError(Exception):
    """Repository misuse or corruption (unknown id, failed integrity check, missing repo)."""


def to_coo(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse COO triple ``(indices int32 [ndim, nnz], values [nnz], shape int64 [ndim])`` of a dense delta."""
    d = delta.detach().cpu()
    if d.ndim == 0:
        raise ValueError("0-dim deltas are not supported; reshape scalars to [1]")
    if max(d.shape, default=0) >= 2**31:
        raise ValueError("tensor dimension too large for int32 COO indices")
    idx = d.nonzero(as_tuple=False).T.contiguous()
    values = d[tuple(idx)] if idx.numel() else d.reshape(-1)[:0]
    if values.dtype not in _VALUE_DTYPES:
        values = values.float()
    return idx.to(torch.int32), values.contiguous(), torch.tensor(list(d.shape), dtype=torch.int64)


def from_coo(indices: torch.Tensor, values: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Dense tensor from a COO triple."""
    dense = torch.zeros(tuple(int(s) for s in shape.tolist()), dtype=values.dtype)
    if indices.numel():
        dense[tuple(indices.long())] = values
    return dense


def serialize_deltas(deltas: Mapping[str, torch.Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """Encodes a delta dict as COO safetensors bytes."""
    tensors: dict[str, torch.Tensor] = {}
    for name, delta in sorted(deltas.items()):
        idx, vals, shape = to_coo(delta)
        tensors[f"{name}__indices"] = idx
        tensors[f"{name}__values"] = vals
        tensors[f"{name}__shape"] = shape
    return st_save(tensors, metadata=metadata)


def deserialize_deltas(blob: bytes) -> dict[str, torch.Tensor]:
    """Decodes COO safetensors bytes into dense deltas.

    Raises:
        WeightGitError: incomplete triples.
    """
    raw = st_load(blob)
    names = {k.rsplit("__", 1)[0] for k in raw}
    out: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        try:
            out[name] = from_coo(raw[f"{name}__indices"], raw[f"{name}__values"], raw[f"{name}__shape"])
        except KeyError as exc:
            raise WeightGitError(f"incomplete COO triple for '{name}'") from exc
    return out


class WeightGitRepository:
    """A ``.wgit`` repository inside a workspace directory."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.root = self.workspace / ".wgit"
        if not (self.root / "HEAD").is_file():
            raise WeightGitError(f"no .wgit repository in {self.workspace} (run init first)")

    @classmethod
    def init(cls, workspace: Path) -> WeightGitRepository:
        """Creates (or opens, if present) a repository in ``workspace``."""
        root = Path(workspace) / ".wgit"
        if not (root / "HEAD").is_file():
            for sub in ("refs/heads", "commits", "objects"):
                (root / sub).mkdir(parents=True, exist_ok=True)
            _write_atomic(root / "refs" / "heads" / "main", ROOT.encode())
            _write_atomic(root / "HEAD", ROOT.encode())
        return cls(Path(workspace))

    @staticmethod
    def exists(workspace: Path) -> bool:
        """True when ``workspace`` contains a repository."""
        return (Path(workspace) / ".wgit" / "HEAD").is_file()

    # --- refs -----------------------------------------------------------------------------------------

    @staticmethod
    def _check_id(patch_id: str, allow_root: bool = True) -> str:
        if allow_root and patch_id == ROOT:
            return patch_id
        if not isinstance(patch_id, str) or not PATCH_ID_RE.fullmatch(patch_id):
            raise WeightGitError(f"invalid patch id '{patch_id}'")
        return patch_id

    def _read_ref(self, path: Path) -> str:
        return self._check_id(path.read_text(encoding="utf-8").strip())

    def check_base(self, fingerprint: str) -> None:
        """Binds the history to one base model on first use; later mismatches are refused.

        Raises:
            WeightGitError: the workspace history belongs to a different base model.
        """
        path = self.root / "base_model.sha256"
        if not path.is_file():
            _write_atomic(path, fingerprint.encode())
        elif path.read_text(encoding="utf-8").strip() != fingerprint:
            raise WeightGitError("this history was built on a different base model; refusing to replay it")

    def head(self) -> str:
        """Current HEAD patch id (``ROOT`` before the first commit)."""
        return self._read_ref(self.root / "HEAD")

    def main(self) -> str:
        """Latest promoted patch id."""
        return self._read_ref(self.root / "refs" / "heads" / "main")

    @staticmethod
    def _check_branch_name(name: str) -> str:
        if (
            not isinstance(name, str)
            or not BRANCH_NAME_RE.fullmatch(name)
            or name in {"HEAD", ROOT}
            or ".." in name
        ):
            raise WeightGitError(f"invalid branch name '{name}'")
        return name

    def create_branch(self, name: str, ref: str = "HEAD") -> str:
        """Creates or updates branch ``refs/heads/<name>`` pointing to ``resolve(ref)``."""
        branch = self._check_branch_name(name)
        target = self.resolve(ref)
        _write_atomic(self.root / "refs" / "heads" / branch, target.encode())
        return target

    def list_branches(self) -> dict[str, str]:
        """Returns mapping ``branch_name -> patch_id`` for all branches under ``refs/heads/``."""
        heads_dir = self.root / "refs" / "heads"
        out: dict[str, str] = {}
        if not heads_dir.is_dir():
            return out
        for p in sorted(heads_dir.iterdir()):
            if p.is_file() and BRANCH_NAME_RE.fullmatch(p.name):
                out[p.name] = self._read_ref(p)
        return out

    def checkout_branch(self, name: str, model: nn.Module | None = None) -> str:
        """Checks out branch ``name`` and moves HEAD to its commit."""
        branch = self._check_branch_name(name)
        path = self.root / "refs" / "heads" / branch
        if not path.is_file():
            raise WeightGitError(f"unknown branch '{branch}'")
        return self.checkout(self._read_ref(path), model=model)

    def resolve(self, ref: str) -> str:
        """Resolves ``HEAD``, ``main``, branch names, ``ROOT`` or a patch id (unique prefixes >= 4 chars)."""
        if ref == "HEAD":
            return self.head()
        if ref == "main":
            return self.main()
        if ref == ROOT or PATCH_ID_RE.fullmatch(ref):
            return self._check_id(ref)
        if BRANCH_NAME_RE.fullmatch(ref) and ".." not in ref:
            branch_path = self.root / "refs" / "heads" / ref
            if branch_path.is_file():
                return self._read_ref(branch_path)
        if re.fullmatch(r"[0-9a-f]{4,15}", ref):
            matches = [p for p in self.list_commits() if p.startswith(ref)]
            if len(matches) == 1:
                return matches[0]
            raise WeightGitError(f"ref '{ref}' is {'ambiguous' if matches else 'unknown'}")
        raise WeightGitError(f"invalid ref '{ref}'")

    def _commit_path(self, patch_id: str) -> Path:
        return self.root / "commits" / f"{self._check_id(patch_id, allow_root=False)}.json"

    def _object_path(self, patch_id: str) -> Path:
        return self.root / "objects" / f"{self._check_id(patch_id, allow_root=False)}.COO.safetensors"

    # --- commits --------------------------------------------------------------------------------------

    def commit(
        self, deltas: Mapping[str, torch.Tensor], manifest: PatchManifest, promote: bool | None = None
    ) -> PatchManifest:
        """Stores ``deltas`` as a child of HEAD and moves HEAD (and ``main`` when promoted).

        ``manifest.patch_id`` / ``parent_patch_id`` / ``delta_sha256`` / ``created_at`` are assigned here.
        ``promote`` defaults to ``manifest.passed_gate``.
        """
        if not deltas:
            raise WeightGitError("refusing to commit an empty delta")
        parent = self.head()
        blob = serialize_deltas(deltas)
        object_sha = hashlib.sha256(blob).hexdigest()
        patch_id = _derive_patch_id(parent, object_sha, manifest.synthesized_code)
        committed = manifest.model_copy(
            update={
                "patch_id": patch_id,
                "parent_patch_id": None if parent == ROOT else parent,
                "delta_sha256": object_sha,
                "created_at": time.time(),
            }
        )
        _write_atomic(self._object_path(patch_id), blob)
        _write_atomic(self._commit_path(patch_id), committed.model_dump_json(indent=2).encode())
        _write_atomic(self.root / "HEAD", patch_id.encode())
        if committed.passed_gate if promote is None else promote:
            _write_atomic(self.root / "refs" / "heads" / "main", patch_id.encode())
        return committed

    def list_commits(self) -> list[str]:
        """All stored patch ids."""
        ids = [p.name[: -len(".json")] for p in (self.root / "commits").glob("*.json")]
        return sorted(i for i in ids if PATCH_ID_RE.fullmatch(i))

    def load_manifest(self, patch_id: str) -> PatchManifest:
        """Reads a commit's manifest.

        Raises:
            WeightGitError: unknown id.
        """
        path = self._commit_path(patch_id)
        if not path.is_file():
            raise WeightGitError(f"unknown patch '{patch_id}'")
        return PatchManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def verify(self, patch_id: str) -> bool:
        """True when the object matches the manifest and the id still derives from their contents."""
        manifest = self.load_manifest(patch_id)
        path = self._object_path(patch_id)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.delta_sha256:
            return False
        parent = manifest.parent_patch_id or ROOT
        return patch_id == _derive_patch_id(parent, manifest.delta_sha256 or "", manifest.synthesized_code)

    def load_delta(self, patch_id: str) -> dict[str, torch.Tensor]:
        """Dense deltas of one commit (integrity-checked).

        Raises:
            WeightGitError: unknown id or hash mismatch.
        """
        if not self.verify(patch_id):
            raise WeightGitError(f"integrity check failed for patch '{patch_id}'")
        return deserialize_deltas(self._object_path(patch_id).read_bytes())

    def ancestry(self, patch_id: str) -> list[str]:
        """Patch ids from ``patch_id`` back to (excluding) ROOT, newest first."""
        chain: list[str] = []
        current: str | None = self._check_id(patch_id)
        while current and current != ROOT:
            if current in chain:
                raise WeightGitError("cycle detected in commit graph")
            chain.append(current)
            current = self.load_manifest(current).parent_patch_id
        return chain

    def log(self, start: str = "HEAD") -> list[PatchManifest]:
        """Manifests along the first-parent chain from ``start``, newest first."""
        return [self.load_manifest(p) for p in self.ancestry(self.resolve(start))]

    def cumulative_delta(self, patch_id: str) -> dict[str, torch.Tensor]:
        """Sum of every delta from ROOT to ``patch_id`` (float32)."""
        total: dict[str, torch.Tensor] = {}
        for pid in reversed(self.ancestry(self._check_id(patch_id))):
            for name, d in self.load_delta(pid).items():
                total[name] = total[name] + d.float() if name in total else d.float().clone()
        return total

    def diff(self, from_id: str, to_id: str) -> dict[str, torch.Tensor]:
        """Delta that moves weights from state ``from_id`` to state ``to_id``."""
        a, b = self.cumulative_delta(from_id), self.cumulative_delta(to_id)
        out = {k: b.get(k, torch.zeros_like(v)) - v for k, v in a.items()}
        out.update({k: v.clone() for k, v in b.items() if k not in a})
        return out

    def checkout(self, patch_id: str, model: nn.Module | None = None) -> str:
        """Moves HEAD to ``patch_id``; when ``model`` (currently at HEAD) is given, updates it in place.

        In-place updates add a float diff; reload base weights + ``cumulative_delta`` for bit-exact states.
        """
        target = self.resolve(patch_id)
        if target != ROOT:
            self.load_manifest(target)
        if model is not None:
            apply_deltas_inplace(model, self.diff(self.head(), target))
        _write_atomic(self.root / "HEAD", target.encode())
        return target

    def revert(self, patch_id: str, model: nn.Module | None = None) -> PatchManifest:
        """Commits the inverse of ``patch_id`` on top of HEAD (git-revert semantics)."""
        target = self.resolve(patch_id)
        if target == ROOT:
            raise WeightGitError("cannot revert ROOT")
        history = self.ancestry(self.head())
        if target not in history:
            raise WeightGitError(f"{target} is not an ancestor of HEAD")
        if any(self.load_manifest(p).reasoner_rationale == f"Revert {target}" for p in history):
            raise WeightGitError(f"{target} is already reverted")
        original = self.load_manifest(target)
        inverse = {k: -v for k, v in self.load_delta(target).items()}
        manifest = original.model_copy(
            update={
                "reasoner_rationale": f"Revert {target}",
                "synthesized_code": f"# revert of {target}\n{original.synthesized_code}",
                "target_metrics_before": original.target_metrics_after,
                "target_metrics_after": original.target_metrics_before,
                "invariant_metrics_before": original.invariant_metrics_after,
                "invariant_metrics_after": original.invariant_metrics_before,
            }
        )
        committed = self.commit(inverse, manifest, promote=True)
        if model is not None:
            apply_deltas_inplace(model, inverse)
        return committed


def apply_deltas_inplace(model: nn.Module, deltas: Mapping[str, torch.Tensor]) -> None:
    """Permanently adds ``deltas`` to ``model`` parameters.

    Raises:
        KeyError / ValueError: unknown parameter or shape mismatch.
    """
    params = dict(model.named_parameters())
    for name, d in deltas.items():  # validate everything first: a failure must not leave a half-patched model
        if name not in params:
            raise KeyError(f"Parameter '{name}' not found in target model.")
        if tuple(d.shape) != tuple(params[name].shape):
            raise ValueError(
                f"delta for '{name}' has shape {tuple(d.shape)}, parameter has {tuple(params[name].shape)}"
            )
    with torch.no_grad():
        for name, d in deltas.items():
            p = params[name]
            p.data.add_(d.to(device=p.device, dtype=p.dtype))


def manifest_summary(manifest: PatchManifest) -> dict[str, object]:
    """Compact JSON-friendly view used by the CLI ``log`` command."""
    return {
        "patch_id": manifest.patch_id,
        "parent": manifest.parent_patch_id,
        "intent": manifest.intent_id,
        "iteration": manifest.iteration,
        "passed_gate": manifest.passed_gate,
        "modified_params": manifest.total_modified_params,
        "rationale": manifest.reasoner_rationale[:80],
    }
