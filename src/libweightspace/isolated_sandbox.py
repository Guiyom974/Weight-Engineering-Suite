"""OS-isolated subprocess sandbox for weight surgery scripts and code-execution pass@1 (WRSI-BL-02).

Executes untrusted ``compute_delta(weights, diagnostics)`` scripts inside a child Python interpreter with:
    1. Pre-flight static AST validation via ``validate_script`` in the parent process.
    2. Zero-pickle IPC: referenced tensors and diagnostics are serialized via ``safetensors`` + JSON to a
       temporary directory; only tensors whose name appears as a string literal in the script (or all tensors
       when under ``LAZY_EXPORT_MAX_TENSORS``) are exported.
    3. OS resource caps: on Windows, attaches a Job Object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY`` and
       ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` when available; on POSIX, applies ``resource.RLIMIT_AS`` in the
       child process; plus hard wall-clock ``subprocess.run(..., timeout=timeout)`` termination.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .check_runner import close_job, run_python_check, windows_job_limit
from .sandbox import (
    DEFAULT_MAX_LINE_EVENTS,
    CloningWeightView,
    SandboxExecutionError,
    SandboxSecurityViolation,
    _validate_output,
    validate_script,
)

LAZY_EXPORT_MAX_TENSORS = 64

_CHILD_RUNNER = """
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from libweightspace.sandbox import execute_weight_surgery_script, SandboxSecurityViolation, SandboxExecutionError

def _apply_posix_rlimit(max_mb: int) -> None:
    # Fail closed: if the limit cannot be set, the child stops here instead of running unlimited.
    import resource
    limit_bytes = int(max_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def main() -> int:
    work_dir = Path(sys.argv[1])
    meta = json.loads((work_dir / "meta.json").read_text(encoding="utf-8"))
    if sys.platform != "win32" and meta.get("max_memory_mb"):
        _apply_posix_rlimit(int(meta["max_memory_mb"]))
    script_code = (work_dir / "script.py").read_text(encoding="utf-8")
    weights = load_file(str(work_dir / "weights.safetensors")) if (work_dir / "weights.safetensors").is_file() else {}
    diag_tensors = load_file(str(work_dir / "diag.safetensors")) if (work_dir / "diag.safetensors").is_file() else {}
    diagnostics = meta["diagnostics"]

    def _hydrate(val):
        if isinstance(val, dict) and "__tensor_key__" in val:
            return diag_tensors[val["__tensor_key__"]]
        if isinstance(val, dict):
            return {k: _hydrate(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_hydrate(v) for v in val]
        return val

    hydrated_diag = _hydrate(diagnostics)
    try:
        deltas = execute_weight_surgery_script(
            script_code,
            weights,
            hydrated_diag,
            max_line_events=int(meta["max_line_events"]),
        )
    except SandboxSecurityViolation as exc:
        (work_dir / "error.json").write_text(
            json.dumps({"kind": "security", "message": str(exc)}), encoding="utf-8"
        )
        return 2
    except SandboxExecutionError as exc:
        (work_dir / "error.json").write_text(
            json.dumps({"kind": "execution", "message": str(exc)}), encoding="utf-8"
        )
        return 3
    if deltas:
        save_file({k: v.contiguous().cpu() for k, v in deltas.items()}, str(work_dir / "deltas.safetensors"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def _referenced_string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _dehydrate_diagnostics(
    value: Any, tensor_bag: dict[str, torch.Tensor], prefix: str = "d"
) -> Any:
    if isinstance(value, torch.Tensor):
        key = f"{prefix}_{len(tensor_bag)}"
        tensor_bag[key] = value.detach().cpu().contiguous()
        return {"__tensor_key__": key}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_dehydrate_diagnostics(v, tensor_bag, prefix) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _dehydrate_diagnostics(v, tensor_bag, prefix) for k, v in value.items()}
    raise TypeError(f"diagnostics value of type {type(value).__name__} cannot be serialized to subprocess")


def _try_attach_windows_job_limit(proc_handle: int, max_memory_mb: int) -> int | None:
    """Windows job object with a real per-process memory limit and kill-on-close (returns the job handle or None).

    Earlier versions created the job without setting any limit; the limit is now applied by
    ``check_runner.windows_job_limit``. Child processes stay allowed here because the surgery runner imports torch.

    Raises:
        OSError: Windows refused to apply the limits (the caller kills the child instead of running unlimited).
    """
    if sys.platform != "win32" or max_memory_mb <= 0:
        return None
    return windows_job_limit(proc_handle, max_memory_mb, max_processes=None)


def execute_weight_surgery_subprocess(
    script_code: str,
    readonly_weights: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Any],
    timeout: float = 30.0,
    max_memory_mb: int = 2048,
    max_line_events: int = DEFAULT_MAX_LINE_EVENTS,
) -> dict[str, torch.Tensor]:
    """Executes ``script_code`` in an OS-isolated child Python process with safetensors IPC and wall-clock timeout.

    Args:
        script_code: Untrusted Python source defining ``compute_delta(weights, diagnostics)``.
        readonly_weights: Mapping of parameter names to base tensors.
        diagnostics: Contrastive prober diagnostics.
        timeout: Wall-clock timeout in seconds before ``SIGKILL`` / ``TerminateProcess``.
        max_memory_mb: Address-space limit in megabytes.
        max_line_events: Line execution budget enforced inside the child sandbox.

    Returns:
        Validated dictionary ``param_name -> delta_tensor``.

    Raises:
        SandboxSecurityViolation: Static policy violation, runtime policy violation, or wall-clock timeout.
        SandboxExecutionError: Syntax error or runtime error raised by the script.
    """
    tree = validate_script(script_code)
    weights_view = (
        readonly_weights
        if isinstance(readonly_weights, CloningWeightView)
        else CloningWeightView(readonly_weights)
    )

    literals = _referenced_string_constants(tree)
    if len(readonly_weights) <= LAZY_EXPORT_MAX_TENSORS:
        exported_weights = {k: v.detach().cpu().contiguous() for k, v in readonly_weights.items()}
    else:
        exported_weights = {
            k: v.detach().cpu().contiguous() for k, v in readonly_weights.items() if k in literals
        }

    diag_tensors: dict[str, torch.Tensor] = {}
    serializable_diag = _dehydrate_diagnostics(dict(diagnostics), diag_tensors)

    src_root = str(Path(__file__).resolve().parents[1])
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONPATH": src_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }

    with tempfile.TemporaryDirectory(prefix="nf_subsandbox_") as tmp_str:
        tmp_dir = Path(tmp_str)
        (tmp_dir / "runner.py").write_text(_CHILD_RUNNER, encoding="utf-8")
        (tmp_dir / "script.py").write_text(script_code, encoding="utf-8")
        (tmp_dir / "meta.json").write_text(
            json.dumps(
                {
                    "diagnostics": serializable_diag,
                    "max_memory_mb": max_memory_mb,
                    "max_line_events": max_line_events,
                }
            ),
            encoding="utf-8",
        )
        if exported_weights:
            save_file(exported_weights, str(tmp_dir / "weights.safetensors"))
        if diag_tensors:
            save_file(diag_tensors, str(tmp_dir / "diag.safetensors"))

        proc = subprocess.Popen(  # nosec B603 - fixed interpreter path and temp runner file
            [sys.executable, str(tmp_dir / "runner.py"), str(tmp_dir)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        try:
            job_handle = _try_attach_windows_job_limit(getattr(proc, "_handle", 0), max_memory_mb)
        except OSError as exc:
            proc.kill()
            proc.communicate()
            raise SandboxSecurityViolation(f"could not apply the memory limit: {exc}") from exc
        try:
            try:
                _stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.communicate()
                raise SandboxSecurityViolation(
                    f"Subprocess surgery execution timed out after {timeout}s"
                ) from exc
        finally:
            close_job(job_handle)

        err_file = tmp_dir / "error.json"
        if err_file.is_file():
            err_info = json.loads(err_file.read_text(encoding="utf-8"))
            if err_info.get("kind") == "security":
                raise SandboxSecurityViolation(str(err_info.get("message", "Security violation")))
            raise SandboxExecutionError(str(err_info.get("message", "Execution error")))

        if proc.returncode != 0:
            raise SandboxExecutionError(
                f"Subprocess exited with code {proc.returncode}: {(stderr or '').strip()[:300]}"
            )

        deltas_path = tmp_dir / "deltas.safetensors"
        raw_deltas = load_file(str(deltas_path)) if deltas_path.is_file() else {}
        return _validate_output(raw_deltas, weights_view)


def execute_code_pass_at_1_trial(
    candidate_code: str,
    test_code: str,
    timeout: float = 5.0,
) -> bool:
    """True when ``candidate_code`` followed by ``test_code`` passes in the hardened check runner.

    Used by ``CodePassAt1Suite``. A static pre-check still rejects imports of ``os``, ``sys``, ``subprocess``,
    ``socket``, ``shutil``, ``ctypes``, ``pathlib``, ``urllib`` and ``http``; the child then runs with the audit-hook
    guard, memory limit and completion marker of ``check_runner.run_python_check``.
    """
    combined = f"{candidate_code}\n\n{test_code}\n"
    try:
        tree = ast.parse(combined)
    except SyntaxError:
        return False
    blocked_modules = {"os", "sys", "subprocess", "socket", "shutil", "ctypes", "pathlib", "urllib", "http"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in blocked_modules for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in blocked_modules:
                return False
    return run_python_check(candidate_code, test_code, timeout=timeout).passed
