"""Hardened child-process runner for model-written code: Python unit-test checks and SQL result checks (CR9).

Each check runs in a fresh ``python -I -S`` child (isolated mode, no site-packages, so no third-party code is
importable) in an empty temporary folder with a minimal environment (no API keys or tokens). Before any untrusted
code runs, the child installs an audit hook that blocks:

    - network and process creation (``socket.*``, ``subprocess.*``, ``os.system``, ``os.exec*``/``spawn*``/``fork``),
    - native code and the registry (``ctypes.*``, ``winreg.*``, ``_winapi.*``) and their imports,
    - reading outside the folder and the Python standard library, and any write, delete or rename outside the folder,
    - frame and object introspection used to forge results (``sys._getframe``, ``gc.get_objects``, ``sys.settrace``).

On Windows the child is placed in a job object with a per-process memory limit, a one-process limit (no children)
and kill-on-close; on POSIX the child sets ``RLIMIT_AS``. The child waits for a start line on stdin, so the limits are
in place before untrusted code runs. A wall-clock timeout kills it.

A Python check passes only when the child prints a random per-run marker after the test code finished, so code that
exits early (``os._exit``) or raises ``SystemExit`` fails. SQL results are compared by the parent.

This is a strong filter, not an operating-system security boundary: native extension code can bypass Python audit
hooks. Run suites from untrusted sources in a container.

Main entry points: ``CheckResult``, ``run_python_check``, ``run_sql_check``, ``windows_job_limit``.
"""

from __future__ import annotations

import ctypes
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 4000
MAX_CODE_CHARS = 100_000

_BOOTSTRAP = r'''
import json
import os
import sys


def _install_guard(root, readable):
    blocked_prefixes = (
        "socket.", "subprocess.", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty", "os.kill",
        "os.killpg", "os.system", "os.startfile", "os.putenv", "os.unsetenv", "os.chdir", "ctypes.", "winreg.",
        "_winapi.", "msvcrt.", "webbrowser.", "urllib.", "http.", "ftplib.", "smtplib.", "poplib.", "imaplib.",
        "sqlite3.load_extension", "sqlite3.enable_load_extension", "sys.settrace", "sys.setprofile",
        "gc.get_objects", "gc.get_referrers", "gc.get_referents", "shutil.", "os.remove", "os.rename", "os.rmdir",
        "os.mkdir", "os.symlink", "os.link", "os.truncate", "os.chmod", "os.chown", "os.utime",
    )
    # Exact events: "sys._getframe" must not catch "sys._getframemodulename", which ordinary stdlib imports raise.
    blocked_exact = {"sys._getframe", "sys._current_frames"}
    # Frame objects reached through tracebacks, generators or coroutines would let the checked code read the
    # harness's locals (the completion marker) and forge a pass; CPython audits these attribute reads.
    frame_attributes = {"tb_frame", "gi_frame", "cr_frame", "ag_frame", "f_back", "f_locals", "f_globals", "f_builtins"}
    blocked_imports = {
        "socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "multiprocessing", "_multiprocessing",
        "ctypes", "_ctypes", "winreg", "_winapi", "msvcrt", "select", "selectors", "asyncio", "mmap",
    }
    blocked_submodules = ("urllib.request", "http.client", "http.server", "xmlrpc.client", "xmlrpc.server")
    for name in list(sys.modules):  # modules loaded at start-up raise no import event: make a new import load them
        if name.split(".")[0] in blocked_imports:
            del sys.modules[name]
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def inside(path, roots):
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return any(real == r or real.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event == "import":
            module = str(args[0])
            if module.split(".")[0] in blocked_imports or any(
                    module == m or module.startswith(m + ".") for m in blocked_submodules):
                raise RuntimeError("blocked by the check sandbox: import " + module)
            return
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if isinstance(path, int):
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool((flags or 0) & write_flags)
            if writing and not inside(path, [root]):
                raise RuntimeError("blocked by the check sandbox: writing outside the check folder")
            if not writing and not inside(path, readable):
                raise RuntimeError("blocked by the check sandbox: reading outside the check folder")
            return
        if event == "object.__getattr__":
            if len(args) > 1 and str(args[1]) in frame_attributes:
                raise RuntimeError("blocked by the check sandbox: frame access (" + str(args[1]) + ")")
            return
        if event in ("os.listdir", "os.scandir"):
            if args and args[0] is not None and not isinstance(args[0], int) and not inside(args[0], readable):
                raise RuntimeError("blocked by the check sandbox: listing outside the check folder")
            return
        if event in blocked_exact or event.startswith(blocked_prefixes):
            raise RuntimeError("blocked by the check sandbox: " + event)

    sys.addaudithook(hook)


def _limit_memory(max_mb):
    if sys.platform != "win32":
        import resource
        limit = int(max_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def main():
    header = sys.stdin.readline().split()
    if len(header) != 4 or header[0] != "go":
        return 3
    nonce, mode, max_mb = header[1], header[2], header[3]
    payload = json.loads(sys.stdin.read())
    _limit_memory(max_mb)
    root = os.path.realpath(os.getcwd())
    readable = [root] + sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                                 os.path.dirname(os.__file__))})
    if mode == "sql":
        import sqlite3

        conn = sqlite3.connect(":memory:")
        denied = {19, 24, 25}  # PRAGMA, ATTACH, DETACH

        def authorizer(action, arg1, arg2, dbname, source):
            if action in denied or (action == 31 and str(arg2).lower() == "load_extension"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        _install_guard(root, readable)
        # The authorizer guards the suite's setup script as well as the query: both are untrusted text.
        conn.set_authorizer(authorizer)
        try:
            if payload.get("setup"):
                conn.executescript(payload["setup"])
            rows = conn.execute(payload["query"]).fetchmany(1001)
        except BaseException as exc:
            print(nonce + " FAIL " + type(exc).__name__ + ": " + str(exc)[:300])
            return 1
        print(nonce + " ROWS " + json.dumps([list(r) for r in rows], default=str))
        return 0
    _install_guard(root, readable)
    source = payload["code"] + "\n\n" + payload["test"] + "\n"
    scope = {"__name__": "__candidate__"}
    try:
        exec(compile(source, "candidate.py", "exec"), scope)
    except AssertionError as exc:
        print(nonce + " FAIL assertion failed: " + (str(exc) or "(no message)")[:300])
        return 1
    except BaseException as exc:
        print(nonce + " FAIL " + type(exc).__name__ + ": " + str(exc)[:300])
        return 1
    print(nonce + " PASS")
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    code = main()
    sys.stdout.flush()
    os._exit(code)
'''


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check, with what the child printed (truncated) and how long it took."""

    passed: bool
    reason: str
    stdout: str
    stderr: str
    seconds: float
    rows: list[list[Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for run logs."""
        return asdict(self)


# --- Windows job object -----------------------------------------------------------------------------------------


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def windows_job_limit(proc_handle: int, max_memory_mb: int, max_processes: int | None = 1) -> int | None:
    """Put a process in a new job object with a per-process memory limit, kill-on-close and a process count limit.

    Args:
        proc_handle: handle of the started (and still waiting) child process.
        max_memory_mb: committed-memory limit for each process in the job.
        max_processes: active processes allowed in the job (1: the child cannot start others); None for no limit.

    Returns:
        The job handle (close it with ``close_job`` when the process ended), or None off Windows.

    Raises:
        OSError: Windows refused to create, configure or assign the job (the caller must not run untrusted code).
    """
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")
    info = _ExtendedLimits()
    flags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if max_processes is not None:
        flags |= _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        info.BasicLimitInformation.ActiveProcessLimit = int(max_processes)
    info.BasicLimitInformation.LimitFlags = flags
    info.ProcessMemoryLimit = int(max_memory_mb) * 1024 * 1024
    ok = kernel32.SetInformationJobObject(ctypes.c_void_p(job), _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                          ctypes.byref(info), ctypes.sizeof(info))
    if not ok or not kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(proc_handle))):
        kernel32.CloseHandle(ctypes.c_void_p(job))
        raise OSError("could not apply the job object limits")
    return int(job)


def close_job(job: int | None) -> None:
    """Close a job handle from ``windows_job_limit`` (kills anything still running in it)."""
    if job is not None and sys.platform == "win32":
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job))


# --- running ---------------------------------------------------------------------------------------------------


def _minimal_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    if sys.platform == "win32":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def _interpreter() -> tuple[str, int]:
    """The interpreter to start and how many processes its job may hold.

    Virtual environments created by some tools (uv, the Windows ``venv`` launcher) give a ``python.exe`` that starts
    the real interpreter as a child. The base interpreter is started directly when it is known (``-S`` means the
    venv's packages are not wanted anyway); otherwise the job allows the launcher plus one interpreter.
    """
    base = getattr(sys, "_base_executable", None)
    if isinstance(base, str) and base and Path(base).is_file():
        return base, 1
    return sys.executable, 2


def _run(mode: str, payload: dict[str, Any], timeout: float, max_memory_mb: int) -> tuple[str | None, str, str, float, bool]:
    """Run the bootstrap; returns (marker line or None, stdout, stderr, seconds, timed out)."""
    nonce = secrets.token_hex(16)
    started = time.perf_counter()
    interpreter, processes = _interpreter()
    with tempfile.TemporaryDirectory(prefix="wl_check_") as tmp:
        boot = Path(tmp) / "bootstrap.py"
        boot.write_text(_BOOTSTRAP, encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(  # nosec B603 - fixed interpreter, fixed bootstrap file, no shell
            [interpreter, "-I", "-S", str(boot)], cwd=tmp, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=_minimal_env(), creationflags=flags,
        )
        job: int | None = None
        try:
            job = windows_job_limit(getattr(proc, "_handle", 0), max_memory_mb, max_processes=processes)
        except OSError:
            proc.kill()
            proc.communicate()
            raise
        timed_out = False
        start_line = f"go {nonce} {mode} {int(max_memory_mb)}\n".encode()
        try:
            out, err = proc.communicate(input=start_line + json.dumps(payload).encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
        finally:
            close_job(job)
    text = out.decode("utf-8", "replace")
    marker = next((line[len(nonce) + 1:] for line in reversed(text.splitlines()) if line.startswith(nonce + " ")), None)
    visible = "\n".join(line for line in text.splitlines() if not line.startswith(nonce + " "))
    return marker, visible[-MAX_OUTPUT_CHARS:], err.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:], \
        time.perf_counter() - started, timed_out


def _no_marker_reason(timed_out: bool, timeout: float, stderr: str) -> str:
    if timed_out:
        return f"timed out after {timeout:g} s"
    if "MemoryError" in stderr:
        return "ran out of its memory limit"
    return "the code exited before the check finished (os._exit, a crash or the memory limit)"


def run_python_check(code: str, test: str, timeout: float = 5.0, max_memory_mb: int = 512) -> CheckResult:
    """Run ``code`` followed by ``test`` (assert statements) in the hardened child; passes when the test completes."""
    if len(code) > MAX_CODE_CHARS or len(test) > MAX_CODE_CHARS:
        return CheckResult(False, "code or test is too large", "", "", 0.0)
    marker, out, err, secs, timed_out = _run("python", {"code": code, "test": test}, timeout, max_memory_mb)
    if marker is None:
        return CheckResult(False, _no_marker_reason(timed_out, timeout, err), out, err, secs)
    if marker == "PASS":
        return CheckResult(True, "all assertions passed", out, err, secs)
    return CheckResult(False, marker.removeprefix("FAIL ").strip(), out, err, secs)


def _normalise_rows(rows: list[list[Any]], ordered: bool) -> list[list[str]]:
    as_text = [[json.dumps(v, default=str) for v in row] for row in rows]
    return as_text if ordered else sorted(as_text)


def run_sql_check(
    setup: str,
    query: str,
    expected_rows: list[list[Any]],
    ordered: bool = False,
    timeout: float = 5.0,
    max_memory_mb: int = 512,
) -> CheckResult:
    """Run ``setup`` then ``query`` on an in-memory SQLite database in the hardened child and compare result rows.

    ``ATTACH``, ``DETACH``, ``PRAGMA`` and ``load_extension`` are denied by an authorizer; row order matters only when
    ``ordered`` is True.
    """
    if len(setup) > MAX_CODE_CHARS or len(query) > MAX_CODE_CHARS:
        return CheckResult(False, "setup or query is too large", "", "", 0.0)
    marker, out, err, secs, timed_out = _run("sql", {"setup": setup, "query": query}, timeout, max_memory_mb)
    if marker is None:
        return CheckResult(False, _no_marker_reason(timed_out, timeout, err), out, err, secs)
    if not marker.startswith("ROWS "):
        return CheckResult(False, marker.removeprefix("FAIL ").strip(), out, err, secs)
    rows = json.loads(marker[5:])
    if _normalise_rows(rows, ordered) == _normalise_rows(expected_rows, ordered):
        return CheckResult(True, "result rows match", out, err, secs, rows=rows[:50])
    return CheckResult(False, f"result rows differ ({len(rows)} returned, {len(expected_rows)} expected)", out, err, secs,
                       rows=rows[:50])
