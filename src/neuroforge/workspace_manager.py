"""Workspace and model manager for the Studio (CR2): Studio home, recent workspaces, Hub checks, verified downloads.

The Studio home (``WEIGHTLAB_HOME``, default ``~/.weightlab``) holds three things::

    models/        downloaded models, one folder per repository and revision (``org--name--<sha12>``)
    workspaces/    workspaces created from the Studio
    recent.json    workspaces opened recently (absolute paths, kept on this machine only)

Nothing here accepts a filesystem path from the browser: workspaces are addressed by an opaque id derived from the
server's own list, model folders by a name that must appear in the server's own listing, and Hub file names are
validated before they are joined to a folder.

Downloads use the Hub's HTTP API directly (``/api/models/<repo>/revision/<rev>?blobs=true`` for metadata and
``/<repo>/resolve/<sha>/<file>`` for files), so progress, cancel, size caps and hash checks are under our control:

    - only top-level ``*.safetensors`` weights and tokenizer/config files are fetched (pickle-based ``.bin`` weights
      can run code when loaded, so repositories without safetensors are refused);
    - every file is checked against its declared size and hash (SHA-256 for LFS files, the git blob SHA-1 otherwise);
    - files go to a ``.partial-*`` folder that is removed on cancel or failure, then renamed into place;
    - ``HF_TOKEN`` (optional, for gated models) is only sent to the Hub host over https and dropped on redirects to
      another host.

Main entry points: ``StudioHome``, ``HubClient``, ``check_model``, ``download_model``, ``create_workspace``,
``model_facts``, ``memory_probe``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import yaml

from libweightspace.adapters import adapter_key_for_model_type
from libweightspace.loading import LoadedModel

from .cancellation import OperationCancelled
from .schemas import WorkspaceConfig
from .templates import DEMO_BLUEPRINT, DEMO_INTENT
from .weight_git import WeightGitRepository

HOME_ENV = "WEIGHTLAB_HOME"
HUB_ENV = "WEIGHTLAB_HF_ENDPOINT"
HUB_DEFAULT = "https://huggingface.co"
TOKEN_ENV = "HF_TOKEN"
MAX_DOWNLOAD_ENV = "WEIGHTLAB_MAX_DOWNLOAD_GB"
CONFIG_NAME = "neuroforge.yaml"
SOURCE_FILE = ".weightlab-source.json"
MAX_RECENT = 12
MAX_METADATA_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 1 << 20

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MODEL_FOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
_HUB_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Weights must be safetensors; the rest are configs and tokenizers. Anything else (pickle weights, ONNX, GGUF,
# subfolders) is not downloaded.
_WANTED_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".jinja", ".tiktoken")
_SKIPPED_NAMES = frozenset({"README.md", ".gitattributes"})
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class WorkspaceManagerError(ValueError):
    """A refused or failed workspace/model operation; the message is safe to show to the user."""


class DownloadCancelled(WorkspaceManagerError, OperationCancelled):
    """The user cancelled a download; partial files were removed."""


# --- Studio home ---------------------------------------------------------------------------------------------


def _workspace_id(path: Path) -> str:
    """Opaque, stable id for a workspace path (the browser only ever sees this)."""
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StudioHome:
    """The per-user folder that holds downloaded models, Studio-created workspaces and the recent list."""

    root: Path

    @classmethod
    def resolve(cls, explicit: Path | None = None) -> StudioHome:
        """``explicit`` > ``WEIGHTLAB_HOME`` > ``~/.weightlab``."""
        raw = explicit or (Path(os.environ[HOME_ENV]) if os.environ.get(HOME_ENV) else Path.home() / ".weightlab")
        return cls(Path(raw).resolve())

    @property
    def models_dir(self) -> Path:
        """Downloaded models."""
        return self.root / "models"

    @property
    def workspaces_dir(self) -> Path:
        """Workspaces created from the Studio."""
        return self.root / "workspaces"

    @property
    def recent_path(self) -> Path:
        """Recent-workspaces list."""
        return self.root / "recent.json"

    def ensure(self) -> None:
        """Create the home folders (idempotent)."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    # --- recent and known workspaces --------------------------------------------------------------------------
    def _read_recent(self) -> list[dict[str, Any]]:
        if not self.recent_path.is_file():
            return []
        try:
            data = json.loads(self.recent_path.read_text(encoding="utf-8"))
        except ValueError:
            return []  # a damaged list is rebuilt on the next remember(); it only holds convenience entries
        return [e for e in data if isinstance(e, dict) and isinstance(e.get("path"), str)] if isinstance(data, list) else []

    def remember(self, workspace: Path) -> None:
        """Put ``workspace`` at the top of the recent list (at most ``MAX_RECENT`` entries)."""
        path = str(workspace.resolve())
        entries = [e for e in self._read_recent() if e["path"] != path]
        entries.insert(0, {"path": path, "opened": time.time()})
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.recent_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries[:MAX_RECENT], indent=1), encoding="utf-8")
        tmp.replace(self.recent_path)

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Recent workspaces first, then other workspaces in the home folder; only folders that still hold a config."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        candidates: list[tuple[Path, float | None, bool]] = [
            (Path(e["path"]), float(e.get("opened") or 0.0), True) for e in self._read_recent()
        ]
        if self.workspaces_dir.is_dir():
            candidates += [(p, None, False) for p in sorted(self.workspaces_dir.iterdir()) if p.is_dir()]
        for path, opened, recent in candidates:
            key = str(path.resolve())
            if key in seen or not (path / CONFIG_NAME).is_file():
                continue
            seen.add(key)
            try:
                cfg = WorkspaceConfig.model_validate(yaml.safe_load((path / CONFIG_NAME).read_text(encoding="utf-8")) or {})
                model, revision = cfg.model, cfg.revision
            except (ValueError, yaml.YAMLError):
                model, revision = "(unreadable neuroforge.yaml)", None
            out.append({
                "id": _workspace_id(path), "name": path.name, "model": model, "revision": revision,
                "opened": opened, "recent": recent, "in_home": path.resolve().parent == self.workspaces_dir.resolve(),
            })
        return out

    def workspace_by_id(self, ws_id: str) -> Path:
        """Resolve an id from ``list_workspaces`` back to its folder.

        Raises:
            WorkspaceManagerError: the id is malformed or not in the server's list.
        """
        if not re.fullmatch(r"[0-9a-f]{16}", ws_id or ""):
            raise WorkspaceManagerError("invalid workspace id")
        for entry in self._read_recent():
            if _workspace_id(Path(entry["path"])) == ws_id and (Path(entry["path"]) / CONFIG_NAME).is_file():
                return Path(entry["path"]).resolve()
        if self.workspaces_dir.is_dir():
            for path in self.workspaces_dir.iterdir():
                if path.is_dir() and _workspace_id(path) == ws_id and (path / CONFIG_NAME).is_file():
                    return path.resolve()
        raise WorkspaceManagerError("unknown workspace; refresh the list")

    # --- local model folders ----------------------------------------------------------------------------------
    def list_model_folders(self) -> list[dict[str, Any]]:
        """Model folders in ``models/`` that hold a ``config.json`` (downloaded or copied there by the user)."""
        if not self.models_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.models_dir.iterdir()):
            if not path.is_dir() or not MODEL_FOLDER_RE.fullmatch(path.name):
                continue
            if not (path / "config.json").is_file():
                continue
            model_type = _read_model_type(path / "config.json")
            source: dict[str, Any] = {}
            if (path / SOURCE_FILE).is_file():
                try:
                    source = json.loads((path / SOURCE_FILE).read_text(encoding="utf-8"))
                except ValueError:
                    source = {}
            rows.append({
                "name": path.name,
                "model_type": model_type,
                "adapter": adapter_key_for_model_type(model_type),
                "size_bytes": sum(f.stat().st_size for f in path.iterdir() if f.is_file()),
                "repo": source.get("repo"),
                "revision": source.get("revision"),
                "license": source.get("license"),
            })
        return rows

    def model_folder(self, name: str) -> Path:
        """A folder from ``list_model_folders`` by name.

        Raises:
            WorkspaceManagerError: the name is malformed or not listed (paths and ``..`` are never accepted).
        """
        if not isinstance(name, str) or not MODEL_FOLDER_RE.fullmatch(name):
            raise WorkspaceManagerError("invalid model folder name")
        if name not in {row["name"] for row in self.list_model_folders()}:
            raise WorkspaceManagerError("that model folder is not in the Studio's models folder")
        path = (self.models_dir / name).resolve()
        if not path.is_relative_to(self.models_dir.resolve()):
            raise WorkspaceManagerError("that model folder is outside the Studio's models folder")
        return path


def _read_model_type(config_path: Path) -> str | None:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("model_type") if isinstance(data, dict) else None
    return str(value) if isinstance(value, str) else None


# --- memory --------------------------------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _cpu_free_bytes() -> int | None:
    """Available physical memory, or None when the platform does not tell us."""
    if sys.platform == "win32":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def memory_probe() -> dict[str, Any]:
    """Devices this Studio can use with their free memory in bytes (None when unknown)."""
    import torch

    devices: list[dict[str, Any]] = [{"device": "cpu", "free_bytes": _cpu_free_bytes(), "name": "CPU (system memory)"}]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        devices.append({"device": "cuda", "free_bytes": int(free), "total_bytes": int(total),
                        "name": torch.cuda.get_device_name(0)})
    return {"devices": devices}


# --- Hub client ----------------------------------------------------------------------------------------------


def _check_endpoint(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" or (parsed.scheme == "http" and host in _LOOPBACK):
        return url.rstrip("/")
    raise WorkspaceManagerError("the Hub endpoint must be https (plain http is only allowed on this machine)")


class _ScopedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects over https only, and send the token only to the Hub's own scheme and host.

    A CDN never sees the token, and a redirect that would downgrade to plain http (except on this machine, for tests)
    is refused, so neither the token nor a file travels in clear text.
    """

    def __init__(self, hub_netloc: str, token: str | None, hub_scheme: str = "https"):
        super().__init__()
        self.hub_netloc = hub_netloc
        self.hub_scheme = hub_scheme
        self.token = token

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        target = urlparse(newurl)
        if target.scheme != "https" and (target.hostname or "").lower() not in _LOOPBACK:
            raise urllib.error.HTTPError(newurl, code, "refusing to follow a redirect to plain http", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and self.token and target.netloc == self.hub_netloc and target.scheme == self.hub_scheme:
            new.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return new


class HubClient:
    """Minimal Hugging Face Hub client: model metadata and file streams (no third-party dependency)."""

    def __init__(self, endpoint: str | None = None, timeout: float = 30.0, token_env: str = TOKEN_ENV):
        self.endpoint = _check_endpoint(endpoint or os.environ.get(HUB_ENV) or HUB_DEFAULT)
        self.timeout = timeout
        self._netloc = urlparse(self.endpoint).netloc
        self._token = os.environ.get(token_env) or None
        self._opener = urllib.request.build_opener(
            _ScopedRedirect(self._netloc, self._token, urlparse(self.endpoint).scheme))

    @property
    def has_token(self) -> bool:
        """True when ``HF_TOKEN`` is set (its value is never returned)."""
        return self._token is not None

    def _request(self, url: str) -> urllib.request.Request:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "weight-engineering-studio"})
        if self._token:
            req.add_unredirected_header("Authorization", f"Bearer {self._token}")
        return req

    def model_info(self, repo: str, revision: str = "main") -> dict[str, Any]:
        """Metadata of ``repo`` at ``revision`` including per-file sizes and hashes.

        Raises:
            WorkspaceManagerError: invalid id, unknown repository, gated without access, or an unreachable Hub.
        """
        if not REPO_RE.fullmatch(repo or ""):
            raise WorkspaceManagerError("model id must look like organisation/model-name")
        if not REVISION_RE.fullmatch(revision or ""):
            raise WorkspaceManagerError("revision must be a branch, tag or commit SHA")
        url = f"{self.endpoint}/api/models/{quote(repo, safe='/')}/revision/{quote(revision, safe='')}?blobs=true"
        try:
            with self._opener.open(self._request(url), timeout=self.timeout) as resp:  # nosec B310 - scheme checked
                raw = resp.read(MAX_METADATA_BYTES + 1)
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code in (401, 403):
                raise WorkspaceManagerError(f"{repo} needs access: accept its licence on the Hub and set {TOKEN_ENV}") from None
            if code == 404:
                raise WorkspaceManagerError(f"{repo} at revision {revision} was not found on the Hub") from None
            raise WorkspaceManagerError(f"the Hub answered HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WorkspaceManagerError(f"cannot reach {self.endpoint}: {type(exc).__name__}") from None
        if len(raw) > MAX_METADATA_BYTES:
            raise WorkspaceManagerError("the Hub metadata exceeds the size limit")
        try:
            info = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise WorkspaceManagerError("the Hub returned unreadable metadata") from None
        if not isinstance(info, dict) or not isinstance(info.get("siblings"), list):
            raise WorkspaceManagerError("the Hub metadata has no file list")
        return info

    def open_file(self, repo: str, sha: str, filename: str) -> Any:
        """Stream one file at a pinned commit (the caller reads and closes it)."""
        url = f"{self.endpoint}/{quote(repo, safe='/')}/resolve/{sha}/{quote(filename, safe='')}"
        try:
            return self._opener.open(self._request(url), timeout=self.timeout)  # nosec B310 - scheme checked
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise WorkspaceManagerError(f"download of {filename} failed: HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WorkspaceManagerError(f"download of {filename} failed: {type(exc).__name__}") from None


# --- check ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HubFile:
    """One file to download with its declared size and hash."""

    name: str
    size: int
    sha256: str | None = None
    git_sha1: str | None = None


@dataclass
class ModelCheck:
    """Everything the wizard shows before a download starts (all values come from the Hub or this machine)."""

    repo: str
    requested_revision: str
    revision: str
    files: list[HubFile]
    download_bytes: int
    parameters: int | None
    weights_bytes_bf16: int
    approximate: bool
    license: str | None
    gated: bool
    model_type: str | None
    adapter: str | None
    device: str
    free_bytes: int | None
    needed_bytes: int
    warnings: list[str] = field(default_factory=list)
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the Studio."""
        data = asdict(self)
        data["files"] = [{"name": f.name, "size": f.size} for f in self.files]
        return data


def _hub_files(info: dict[str, Any]) -> list[HubFile]:
    files: list[HubFile] = []
    for sib in info.get("siblings", []):
        if not isinstance(sib, dict):
            continue
        name = sib.get("rfilename")
        if not isinstance(name, str) or name in _SKIPPED_NAMES or not _HUB_FILE_RE.fullmatch(name):
            continue  # subfolders, hidden files and anything with unusual characters are not downloaded
        if not name.endswith(_WANTED_SUFFIXES):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else None
        size = int((lfs or {}).get("size") or sib.get("size") or -1)
        sha256 = str(lfs["sha256"]) if lfs and isinstance(lfs.get("sha256"), str) else None
        blob = sib.get("blobId") if isinstance(sib.get("blobId"), str) and not lfs else None
        if size < 0 or (sha256 is None and blob is None):
            raise WorkspaceManagerError(f"the Hub did not give a size and hash for {name}; refusing an unverified download")
        files.append(HubFile(name, size, sha256, blob))
    return files


def _license_of(info: dict[str, Any]) -> str | None:
    raw_card = info.get("cardData")
    card: dict[str, Any] = raw_card if isinstance(raw_card, dict) else {}
    value = card.get("license")
    if isinstance(value, str) and value:
        name = card.get("license_name")
        return f"other ({name})" if value == "other" and isinstance(name, str) else value
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def max_download_bytes() -> int:
    """Upper bound for one download (``WEIGHTLAB_MAX_DOWNLOAD_GB``, default 64 GB)."""
    try:
        gb = float(os.environ.get(MAX_DOWNLOAD_ENV, "64"))
    except ValueError:
        gb = 64.0
    return int(gb * 1024**3)


def _gb(n: int | None) -> str:
    return "unknown" if n is None else f"{n / 1024**3:.2f} GB"


def check_model(
    client: HubClient,
    repo: str,
    revision: str = "main",
    device: str = "cpu",
    memory: dict[str, Any] | None = None,
) -> ModelCheck:
    """Size, memory estimate, licence, revision and architecture support of ``repo`` before any download.

    The memory estimate assumes bfloat16 weights (2 bytes per parameter). Editing keeps a second copy of the base
    weights for the before/after checks, so the need compared with free memory is twice the weights.

    Raises:
        WorkspaceManagerError: see ``HubClient.model_info``; no safetensors weights; no config.json.
    """
    if device not in ("cpu", "cuda"):
        raise WorkspaceManagerError("device must be cpu or cuda")
    info = client.model_info(repo, revision)
    sha = str(info.get("sha") or "")
    if not SHA_RE.fullmatch(sha):
        raise WorkspaceManagerError("the Hub did not return a commit SHA to pin")
    files = _hub_files(info)
    names = {f.name for f in files}
    if not any(n.endswith(".safetensors") for n in names):
        raise WorkspaceManagerError(
            f"{repo} has no safetensors weights; only safetensors are downloaded (pickle-based .bin files can run code when loaded)"
        )
    if "config.json" not in names:
        raise WorkspaceManagerError(f"{repo} has no config.json at the top level")
    weights_file_bytes = sum(f.size for f in files if f.name.endswith(".safetensors"))
    st_info = info.get("safetensors")
    total = st_info.get("total") if isinstance(st_info, dict) else None
    params = int(total) if isinstance(total, int) and total > 0 else None
    weights_bf16 = params * 2 if params else weights_file_bytes
    config = info.get("config")
    raw_type = config.get("model_type") if isinstance(config, dict) else None
    model_type = str(raw_type) if isinstance(raw_type, str) and raw_type else None
    adapter = adapter_key_for_model_type(model_type)
    probe = memory if memory is not None else memory_probe()
    dev = next((d for d in probe.get("devices", []) if d.get("device") == device), None)
    if dev is None:
        raise WorkspaceManagerError(f"device {device} is not available on this machine")
    free = dev.get("free_bytes")
    needed = 2 * weights_bf16
    check = ModelCheck(
        repo=repo, requested_revision=revision, revision=sha, files=files,
        download_bytes=sum(f.size for f in files), parameters=params, weights_bytes_bf16=weights_bf16,
        approximate=params is None, license=_license_of(info), gated=bool(info.get("gated")),
        model_type=model_type, adapter=adapter, device=device, free_bytes=free, needed_bytes=needed,
    )
    if adapter is None:
        check.warnings.append(
            f"Unsupported architecture: model_type '{model_type}' has no adapter (supported families: Qwen, Llama, "
            "Mistral, Gemma, DeepSeek, LFM). The model would download but could not be edited."
        )
    if check.approximate:
        check.warnings.append("The Hub gave no parameter count; the memory estimate uses the weight file sizes and is approximate.")
    if check.gated and not client.has_token:
        check.warnings.append(f"This model is gated: accept its licence on the Hub and set {TOKEN_ENV} before downloading.")
    if check.download_bytes > max_download_bytes():
        check.warnings.append(f"The download ({_gb(check.download_bytes)}) exceeds the limit set by {MAX_DOWNLOAD_ENV}.")
    if free is not None and needed > free:
        cpu = next((d for d in probe.get("devices", []) if d.get("device") == "cpu"), None)
        cpu_free = cpu.get("free_bytes") if cpu else None
        check.warnings.append(
            f"Editing needs about {_gb(needed)} (two copies of the weights in bfloat16); {device} has {_gb(free)} free."
        )
        if device != "cpu" and cpu_free is not None and needed <= cpu_free:
            check.suggestion = "Choose CPU: it has enough free memory (slower, but it fits)."
        else:
            check.suggestion = "Choose a smaller model (fewer parameters) or free some memory first."
    return check


# --- download ------------------------------------------------------------------------------------------------


def _safe_folder_name(repo: str, sha: str) -> str:
    return f"{repo.replace('/', '--')}--{sha[:12]}"


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    client: HubClient,
    check: ModelCheck,
    models_dir: Path,
    progress: Callable[[str, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Download the files of ``check`` into ``models_dir`` and verify every size and hash.

    Returns:
        The final model folder (``models_dir/<org>--<name>--<sha12>``); reused when that revision is already there.

    Raises:
        DownloadCancelled: cancel requested (partial files removed).
        WorkspaceManagerError: size limit, HTTP failure, size or hash mismatch (partial files removed).
    """
    if check.download_bytes > max_download_bytes():
        raise WorkspaceManagerError(f"the download exceeds the limit set by {MAX_DOWNLOAD_ENV}")
    models_dir.mkdir(parents=True, exist_ok=True)
    final = models_dir / _safe_folder_name(check.repo, check.revision)
    if (final / "config.json").is_file() and (final / SOURCE_FILE).is_file():
        return final
    partial = models_dir / f".partial-{uuid.uuid4().hex[:12]}"
    partial.mkdir()
    total = max(1, check.download_bytes)
    done = 0
    try:
        for f in check.files:
            target = (partial / f.name).resolve()
            if not target.is_relative_to(partial.resolve()):
                raise WorkspaceManagerError(f"refusing file name {f.name!r}")
            sha256 = hashlib.sha256()
            written = 0
            with client.open_file(check.repo, check.revision, f.name) as resp, target.open("wb") as out:
                while True:
                    if should_cancel is not None and should_cancel():
                        raise DownloadCancelled("download cancelled; partial files were removed")
                    chunk = resp.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > f.size:
                        raise WorkspaceManagerError(f"{f.name} is larger than the Hub declared ({f.size} bytes)")
                    sha256.update(chunk)
                    out.write(chunk)
                    if progress is not None:
                        progress(f"downloading {f.name} ({_gb(done + written)} of {_gb(total)})", (done + written) / total)
            if written != f.size:
                raise WorkspaceManagerError(f"{f.name} is incomplete ({written} of {f.size} bytes)")
            if f.sha256 is not None and sha256.hexdigest() != f.sha256.lower():
                raise WorkspaceManagerError(f"{f.name} failed its SHA-256 check")
            if f.sha256 is None and f.git_sha1 is not None and _git_blob_sha1(target, written) != f.git_sha1.lower():
                raise WorkspaceManagerError(f"{f.name} failed its git SHA-1 check")
            done += written
        (partial / SOURCE_FILE).write_text(json.dumps({
            "repo": check.repo, "revision": check.revision, "license": check.license,
            "files": {f.name: f.sha256 or f.git_sha1 for f in check.files}, "downloaded_at": time.time(),
        }, indent=1), encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)  # an earlier, unverified copy (no source file) is replaced
        partial.rename(final)
        return final
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


# --- workspaces ----------------------------------------------------------------------------------------------


def create_workspace(
    home: StudioHome,
    name: str,
    model: str,
    *,
    model_path: Path | None = None,
    device: str = "cpu",
    demo: bool = False,
    source: str | None = None,
    revision: str | None = None,
    license_name: str | None = None,
) -> Path:
    """Create ``home/workspaces/<name>`` with its config, an empty edit history and (for the demo) starter files.

    Raises:
        WorkspaceManagerError: invalid or taken name, or a model folder outside the models folder.
    """
    if not NAME_RE.fullmatch(name or ""):
        raise WorkspaceManagerError("workspace name: letters, digits, '.', '_' or '-' (at most 64)")
    if device not in ("cpu", "cuda"):
        raise WorkspaceManagerError("device must be cpu or cuda")
    if model_path is not None and not model_path.resolve().is_relative_to(home.models_dir.resolve()):
        raise WorkspaceManagerError("model folders must be inside the Studio's models folder")
    home.ensure()
    path = home.workspaces_dir / name
    if path.exists():
        raise WorkspaceManagerError(f"a workspace named {name} already exists")
    path.mkdir()
    try:
        cfg = WorkspaceConfig(
            model=model, device=device, demo_defect=demo, reference_model=model if demo else None,
            model_path=str(model_path.resolve()) if model_path is not None else None,
            source=source, revision=revision, license=license_name,
        )
        (path / CONFIG_NAME).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
        WeightGitRepository.init(path)
        if demo:
            (path / "intent.yaml").write_text(yaml.safe_dump(DEMO_INTENT, sort_keys=False), encoding="utf-8")
            (path / "blueprint.yaml").write_text(yaml.safe_dump(DEMO_BLUEPRINT, sort_keys=False), encoding="utf-8")
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    home.remember(path)
    return path


def model_facts(cfg: WorkspaceConfig, loaded: LoadedModel) -> dict[str, Any]:
    """Facts panel: what is loaded, from where, at which revision, under which licence."""
    first = next(loaded.model.parameters())
    return {
        "model": cfg.model,
        "source": cfg.source or ("built-in toy model" if cfg.model.startswith("toy:") else "local or Hub id"),
        "revision": cfg.revision,
        "license": cfg.license,
        "family": loaded.adapter.family,
        "adapter": type(loaded.adapter).__name__,
        "layers": loaded.adapter.num_layers(),
        "parameters": int(sum(p.numel() for p in loaded.model.parameters())),
        "dtype": str(first.dtype).replace("torch.", ""),
        "device": str(first.device),
        "toy": cfg.model.startswith("toy:"),
    }
