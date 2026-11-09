"""Tests for the hardened check runner (CR9, threat T15): passes, failures, and every blocked escape route.

Each test spawns a real ``python -I -S`` child; nothing here reaches the network or writes outside a temp folder.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from libweightspace.check_runner import close_job, run_python_check, run_sql_check, windows_job_limit
from libweightspace.isolated_sandbox import execute_code_pass_at_1_trial


def test_passing_and_failing_python_checks() -> None:
    ok = run_python_check("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert ok.passed and ok.reason == "all assertions passed"
    bad = run_python_check("def add(a, b):\n    return a - b", "assert add(2, 3) == 5, 'wrong sum'")
    assert not bad.passed and "assertion failed: wrong sum" in bad.reason
    broken = run_python_check("def add(a, b) return", "assert True")
    assert not broken.passed and "SyntaxError" in broken.reason


def test_output_is_captured_and_marker_hidden() -> None:
    res = run_python_check("print('hello from the model')", "assert True")
    assert res.passed and "hello from the model" in res.stdout and "PASS" not in res.stdout


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("import socket", "import socket"),
        ("import subprocess", "import subprocess"),
        ("import ctypes", "import ctypes"),
        ("import os\nos.system('echo hi')", "os.system"),
        ("open('../escape.txt', 'w').write('x')", "writing outside"),
        ("import os\nos.remove('bootstrap.py')", "os.remove"),
        ("import sys\nsys._getframe(0)", "sys._getframe"),
        ("import gc\ngc.get_objects()", "gc.get_objects"),
        ("import urllib.request", "urllib"),
    ],
)
def test_escape_routes_are_blocked(code: str, fragment: str) -> None:
    res = run_python_check(code, "assert True")
    assert not res.passed and "blocked by the check sandbox" in res.reason and fragment in res.reason


def test_reading_outside_the_folder_is_blocked(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("api-key-123", encoding="utf-8")
    res = run_python_check(f"print(open({str(secret)!r}).read())", "assert True")
    assert not res.passed and "reading outside" in res.reason and "api-key-123" not in res.stdout


def test_environment_holds_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POE_API_KEY", "should-not-leak")
    res = run_python_check("import os\nleak = os.environ.get('POE_API_KEY')", "assert leak is None")
    assert res.passed


def test_early_exit_cannot_fake_a_pass() -> None:
    res = run_python_check("import os\nos._exit(0)", "assert False")
    assert not res.passed and "exited before the check finished" in res.reason
    res = run_python_check("import sys\nsys.exit(0)", "assert False")
    assert not res.passed and "SystemExit" in res.reason


def test_timeout_and_memory_limit() -> None:
    res = run_python_check("while True:\n    pass", "assert True", timeout=1.5)
    assert not res.passed and "timed out" in res.reason
    hog = run_python_check("blob = bytearray(1500 * 1024 * 1024)", "assert True", timeout=20, max_memory_mb=256)
    assert not hog.passed


@pytest.mark.skipif(sys.platform != "win32", reason="the one-process job limit is Windows-specific")
def test_windows_job_blocks_child_processes_even_without_python_hooks() -> None:
    res = run_python_check("import _winapi", "assert True")
    assert not res.passed and "_winapi" in res.reason
    # Second line of defence, tested without any audit hook: a child in a one-process job cannot start another.
    code = "import subprocess, sys\ntry:\n    subprocess.run([sys.executable, '-c', 'pass'])\n    print('spawned')\nexcept OSError:\n    print('refused')"
    base = getattr(sys, "_base_executable", sys.executable)
    proc = subprocess.Popen([base, "-I", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    job = windows_job_limit(proc._handle, 256, max_processes=1)  # type: ignore[attr-defined]
    try:
        out, _ = proc.communicate(timeout=30)
    finally:
        close_job(job)
    assert b"refused" in out and b"spawned" not in out


def test_sql_checks_compare_rows_and_deny_attach() -> None:
    setup = "create table t(name text, n int); insert into t values ('a', 1), ('b', 2);"
    ok = run_sql_check(setup, "select name, n from t where n > 1", [["b", 2]])
    assert ok.passed and ok.rows == [["b", 2]]
    unordered = run_sql_check(setup, "select name from t order by name desc", [["a"], ["b"]])
    assert unordered.passed
    ordered = run_sql_check(setup, "select name from t order by name desc", [["a"], ["b"]], ordered=True)
    assert not ordered.passed and "differ" in ordered.reason
    attach = run_sql_check(setup, "attach database 'x.db' as x", [])
    assert not attach.passed and "not authorized" in attach.reason.lower()
    pragma = run_sql_check(setup, "pragma table_info(t)", [])
    assert not pragma.passed


def test_legacy_pass_at_1_uses_the_hardened_runner() -> None:
    assert execute_code_pass_at_1_trial("def f():\n    return 1", "assert f() == 1")
    assert not execute_code_pass_at_1_trial("import os\nos._exit(0)", "assert False")
    assert not execute_code_pass_at_1_trial("import os\ndef f(): return 1", "assert f() == 1")
