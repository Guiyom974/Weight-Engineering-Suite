"""JSON-RPC client that launches and talks to the air-gapped anchor vault subprocess.

``AnchorVaultClient`` starts ``python -m wrsi.anchor_vault_server`` with stdin/stdout pipes (stderr goes to a log
file so a chatty server can never dead-lock the pipe), writes candidate weights into a private exchange
directory, and returns ``(passed, mean_kl)``. Every request has a timeout; a dead or silent server raises
``AnchorVaultError`` and callers must treat the candidate as rejected (fail closed).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .anchor_vault_server import file_sha256

_SRC_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


class AnchorVaultError(RuntimeError):
    """The vault process failed, timed out, or returned an error."""


class AnchorVaultClient:
    """Owns one anchor-vault subprocess for the lifetime of the client."""

    def __init__(
        self,
        vault_path: Path,
        base_weights_path: Path,
        model_factory: str = "libweightspace.toy_models:build_tiny_qwen3",
        factory_kwargs: dict[str, Any] | None = None,
        max_kl: float = 0.08,
        exchange_dir: Path | None = None,
        timeout: float = 120.0,
        startup_timeout: float = 600.0,
        vault_sha256: str | None = None,
        base_weights_sha256: str | None = None,
    ):
        self.timeout = startup_timeout  # loading a large checkpoint can take far longer than one request
        self._owns_exchange = exchange_dir is None
        self.exchange_dir = Path(exchange_dir or tempfile.mkdtemp(prefix="wrsi-anchor-"))
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.exchange_dir / "anchor_vault.log", "ab")  # closed in close()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC_DIR), env.get("PYTHONPATH")]))
        v_hash = vault_sha256 or (file_sha256(vault_path) if Path(vault_path).is_file() else None)
        b_hash = base_weights_sha256 or (
            file_sha256(base_weights_path) if Path(base_weights_path).is_file() else None
        )
        argv = [
            sys.executable,
            "-m",
            "wrsi.anchor_vault_server",
            "--vault",
            str(vault_path),
            "--base-weights",
            str(base_weights_path),
            "--exchange-dir",
            str(self.exchange_dir),
            "--model-factory",
            model_factory,
            "--factory-kwargs",
            json.dumps(factory_kwargs or {}),
            "--max-kl",
            str(max_kl),
        ]
        if v_hash is not None:
            argv += ["--vault-sha256", v_hash]
        if b_hash is not None:
            argv += ["--base-weights-sha256", b_hash]
        self._proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._next_id = 0
        try:
            if self.request("ping") != "pong":
                raise AnchorVaultError("anchor vault did not answer ping")
            self.timeout = timeout
        except BaseException:
            self.close()
            raise

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Sends one JSON-RPC request and waits for its response.

        Raises:
            AnchorVaultError: process exited, timed out, mismatched id, or JSON-RPC error.
        """
        if self._proc.poll() is not None:
            raise AnchorVaultError(f"anchor vault process exited with code {self._proc.returncode}")
        self._next_id += 1
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            raise AnchorVaultError("anchor vault pipe closed") from exc
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0.001))
            except queue.Empty as exc:
                raise AnchorVaultError(f"anchor vault timed out after {self.timeout}s") from exc
            if line is None:
                raise AnchorVaultError("anchor vault closed its output (see anchor_vault.log)")
            response = json.loads(line)
            reply_id = response.get("id")
            if reply_id is None or reply_id == self._next_id:
                break
            # A reply that arrives after its request timed out must not poison every later request.
            logger.warning("discarding stale anchor vault reply %s", reply_id)
        if "error" in response:
            raise AnchorVaultError(f"anchor vault error: {response['error'].get('message')}")
        return response.get("result")

    def info(self) -> dict[str, Any]:
        """Anchor count and threshold (never the anchors themselves)."""
        return self.request("info")

    def verify(self, model: nn.Module, tensor_names: Sequence[str] | None = None) -> tuple[bool, float]:
        """Sends the candidate's weights (all parameters, or only ``tensor_names``) and returns ``(passed, kl)``."""
        params = dict(model.named_parameters())
        names = list(params) if tensor_names is None else list(tensor_names)
        unknown = [n for n in names if n not in params]
        if unknown:
            raise KeyError(f"unknown tensors: {unknown[:3]}")
        file_name = f"cand_{uuid.uuid4().hex}.safetensors"
        path = self.exchange_dir / file_name
        save_file({n: params[n].detach().cpu().contiguous().clone() for n in names}, str(path))
        try:
            result = self.request("verify_kl_tether", {"weights_file": file_name})
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # server may still hold it on Windows; must not mask the real error
                logger.warning("could not remove %s: %s", path.name, exc)
        return bool(result["passed"]), float(result["mean_kl"])

    def make_kl_fn(
        self, tensor_names_fn: Callable[[], Sequence[str]] | None = None
    ) -> Callable[[nn.Module], tuple[bool, float]]:
        """Adapter for ``WRSIScientistLoop``: ``fn(model) -> (passed, kl)``.

        ``tensor_names_fn`` returns which tensors to send (e.g. every tensor ever modified); default all.
        """

        def kl_fn(model: nn.Module) -> tuple[bool, float]:
            return self.verify(model, None if tensor_names_fn is None else tensor_names_fn())

        return kl_fn

    def close(self) -> None:
        """Asks the server to exit, kills it if needed, and removes a self-created exchange directory."""
        if self._proc.poll() is None:
            try:
                self.request("shutdown")
            except AnchorVaultError as exc:
                logger.warning("anchor vault did not shut down cleanly (%s); killing it", exc)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError as exc:  # flushing into a pipe whose reader died (EINVAL/EPIPE on Windows)
                logger.debug("ignoring error while closing anchor vault pipe: %s", exc)
        self._log.close()
        if self._owns_exchange:
            shutil.rmtree(self.exchange_dir, ignore_errors=True)

    def __enter__(self) -> AnchorVaultClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_base_weights(model: nn.Module, path: Path) -> Path:
    """Operator-side: writes M_0's weights for the vault (tied tensors handled by ``save_model``)."""
    from safetensors.torch import save_model

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))
    return path


def anchor_ids_from_texts(tokenizer: Any, texts: Sequence[str]) -> list[torch.Tensor]:
    """Tokenises anchor texts for ``build_anchor_vault``."""
    return [tokenizer(t, return_tensors="pt")["input_ids"][0] for t in texts]
