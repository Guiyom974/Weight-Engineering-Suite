"""Model Factory (CR9): specialise a model for one job, measured on a task suite with automatic checks.

A factory run never reports a number it did not measure:

    1. Baseline: the model answers every task (greedy); each answer is checked (Python unit test or SQL result set in
       the hardened check runner, or a text check) and the check result is written to the run log.
    2. Prompt-only baseline: the same model with the best of the suite's system prompts (chosen on training tasks
       only), so the report can say when weight editing is not worth it.
    3. Repair: failing training tasks that have a reference answer become the intent's failing prompts, passing
       training tasks and general prompts become invariants, and the normal guarded loop runs (approval, budgets,
       nullspace, frontier proposer). Held-out tasks are never shown to the proposer.
    4. After: the edited model answers every task again; pass@1 is reported per split.
    5. Verdict: packaging is offered only when the edited model beats both the baseline and the prompt-only baseline on
       the held-out tasks (or the training tasks when there are none).

Run folder: ``<workspace>/.factory/<run_id>/`` with ``suite.yaml``, ``log.jsonl`` (one line per check), ``intent.yaml``
and ``report.json``.

Main entry points: ``TaskSuite``, ``Task``, ``evaluate_suite``, ``prompt_only_baseline``, ``build_repair_intent``,
``verdict``, ``FactoryRun``, ``suite_examples``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from libweightspace.check_runner import CheckResult, run_python_check, run_sql_check
from libweightspace.prober import model_device, model_logits

from .cancellation import OperationCancelled
from .metrics import encode_continuation
from .schemas import EditConstraints, IntentSpec, InvariantGuard, MetricGoal, ProbePair

_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
MAX_TASKS = 64
Progress = Callable[[str, float], None]
CheckKind = Literal["python_test", "sql_rows", "starts_with", "not_starts_with"]
DEFAULT_GENERAL = ("The weather today is", "Once upon a time there was", "The quick brown fox", "hello world")


class Task(BaseModel):
    """One task: a prompt, how its answer is checked, and (for training tasks) an optional reference answer."""

    id: str = Field(pattern=_ID_RE)
    prompt: str = Field(min_length=1, max_length=4000)
    check: CheckKind
    test: str | None = Field(default=None, max_length=20_000)  # python_test: assert statements
    setup: str | None = Field(default=None, max_length=20_000)  # sql_rows: SQL that creates the data
    expected: str | list[list[Any]] | None = None  # text for text checks, rows for sql_rows
    ordered: bool = False  # sql_rows: does row order matter
    answer_starts_with_prompt: bool = False  # python_test on code-completion prompts: check prompt + completion
    reference: str | None = Field(default=None, max_length=4000)  # a correct answer (training tasks, for repair)
    split: Literal["train", "heldout"] = "train"

    @model_validator(mode="after")
    def _fields_for_check(self) -> Task:
        if self.check == "python_test" and not (self.test or "").strip():
            raise ValueError(f"task {self.id}: python_test needs test (assert statements)")
        if self.check == "sql_rows" and not isinstance(self.expected, list):
            raise ValueError(f"task {self.id}: sql_rows needs expected rows (a list of lists)")
        if self.check in ("starts_with", "not_starts_with") and not (isinstance(self.expected, str) and self.expected.strip()):
            raise ValueError(f"task {self.id}: {self.check} needs expected text")
        return self


class TaskSuite(BaseModel):
    """A domain defined by tasks with automatic checks (no task suite, no factory run)."""

    suite_id: str = Field(pattern=_ID_RE)
    description: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=48, ge=1, le=512)
    stop: list[str] = Field(default_factory=lambda: ["\n\n"], max_length=8)
    prompt_baselines: list[str] = Field(default_factory=list, max_length=6)
    general_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_GENERAL), max_length=16)
    tasks: list[Task] = Field(min_length=1, max_length=MAX_TASKS)

    @field_validator("prompt_baselines", "general_prompts", "stop")
    @classmethod
    def _short_texts(cls, v: list[str]) -> list[str]:
        if any(not isinstance(s, str) or len(s) > 2000 for s in v):
            raise ValueError("entries must be texts of at most 2000 characters")
        return v

    @model_validator(mode="after")
    def _unique_and_trainable(self) -> TaskSuite:
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        if not any(t.split == "train" for t in self.tasks):
            raise ValueError("the suite needs at least one training task")
        return self


# --- generation and checks -------------------------------------------------------------------------------------


@torch.no_grad()
def generate_completion(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int, stop: Sequence[str]) -> str:
    """Greedy completion, cut at the first stop string or the end-of-sequence token."""
    device = model_device(model)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    eos = getattr(tokenizer, "eos_token_id", None)
    out: list[int] = []
    text = ""
    for _ in range(max_new_tokens):
        nxt = int(model_logits(model(input_ids=ids))[0, -1].argmax())
        if eos is not None and nxt == eos:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        text = str(tokenizer.decode(out))
        if any(s and s in text for s in stop):
            break
    for s in stop:
        if s and s in text:
            text = text[: text.index(s)]
    return text


def check_task(task: Task, completion: str) -> CheckResult:
    """Run the task's check on ``completion`` (code and SQL in the hardened child process)."""
    if task.check == "python_test":
        code = (task.prompt + completion) if task.answer_starts_with_prompt else completion
        return run_python_check(code, task.test or "")
    if task.check == "sql_rows":
        rows = task.expected if isinstance(task.expected, list) else []
        return run_sql_check(task.setup or "", completion.strip().rstrip(";"), rows, ordered=task.ordered)
    expected = str(task.expected).strip()
    starts = completion.strip().startswith(expected)
    if task.check == "starts_with":
        return CheckResult(starts, "starts with the expected text" if starts else f"does not start with {expected!r}",
                           "", "", 0.0)
    return CheckResult(not starts, f"avoids {expected!r}" if not starts else f"starts with the forbidden {expected!r}",
                       "", "", 0.0)


def evaluate_suite(
    model: nn.Module,
    tokenizer: Any,
    suite: TaskSuite,
    phase: str,
    prefix: str = "",
    splits: Sequence[str] = ("train", "heldout"),
    log: Callable[[dict[str, Any]], None] | None = None,
    progress: Progress | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Answer and check every task in ``splits``; every result is also passed to ``log``.

    Raises:
        OperationCancelled: cancel requested (nothing after the current task is measured).
    """
    tasks = [t for t in suite.tasks if t.split in splits]
    results = []
    for i, task in enumerate(tasks):
        if should_cancel is not None and should_cancel():
            raise OperationCancelled("cancelled by you; nothing was applied")
        if progress is not None:
            progress(f"{phase}: task {i + 1} of {len(tasks)} ({task.id})", i / max(1, len(tasks)))
        completion = generate_completion(model, tokenizer, prefix + task.prompt, suite.max_new_tokens, suite.stop)
        res = check_task(task, completion)
        row = {"phase": phase, "task": task.id, "split": task.split, "check": task.check, "prefix": prefix,
               "completion": completion[:2000], **res.to_dict(), "at": time.time()}
        results.append(row)
        if log is not None:
            log(row)
    return results


def pass_at_1(results: Sequence[dict[str, Any]], split: str) -> float | None:
    """Share of passed tasks in ``split`` (None when the split has no tasks)."""
    rows = [r for r in results if r["split"] == split]
    return round(sum(1 for r in rows if r["passed"]) / len(rows), 4) if rows else None


def prompt_only_baseline(
    model: nn.Module, tokenizer: Any, suite: TaskSuite, log: Callable[[dict[str, Any]], None] | None = None,
    progress: Progress | None = None, should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Best system prompt chosen on training tasks only, then measured on every split."""
    if not suite.prompt_baselines:
        return {"configured": False}
    scored = []
    for i, prefix in enumerate(suite.prompt_baselines):
        rows = evaluate_suite(model, tokenizer, suite, f"prompt-only {i + 1}", prefix=prefix, splits=("train",),
                              log=log, progress=progress, should_cancel=should_cancel)
        scored.append((pass_at_1(rows, "train") or 0.0, -i, prefix))
    best_score, neg_index, best = max(scored)
    held = evaluate_suite(model, tokenizer, suite, "prompt-only best (held-out)", prefix=best, splits=("heldout",),
                          log=log, progress=progress, should_cancel=should_cancel)
    return {"configured": True, "prefix": best, "index": -neg_index, "train": best_score,
            "heldout": pass_at_1(held, "heldout"), "candidates": [{"prefix": p, "train": s} for s, _, p in scored]}


# --- repair intent from failures -------------------------------------------------------------------------------


def build_repair_intent(
    suite: TaskSuite, baseline: Sequence[dict[str, Any]], tokenizer: Any, constraints: EditConstraints
) -> tuple[IntentSpec | None, list[str]]:
    """Intent from failing training tasks with a reference answer; held-out tasks are never used.

    A failing task becomes a contrastive probe only when its reference and its actual answer start with different
    tokens (the loop's ``logit_margin`` compares first tokens). Returns ``(intent or None, notes)``.
    """
    by_id = {t.id: t for t in suite.tasks}
    notes: list[str] = []
    probes: list[ProbePair] = []
    for row in baseline:
        task = by_id[row["task"]]
        if task.split != "train" or row["passed"]:
            continue
        if not task.reference:
            notes.append(f"{task.id}: failing but has no reference answer, so it cannot guide a repair")
            continue
        ref_ids = encode_continuation(tokenizer, task.reference)
        got_ids = encode_continuation(tokenizer, row["completion"])
        if not ref_ids or (got_ids and got_ids[0] == ref_ids[0]):
            notes.append(f"{task.id}: the answer already starts like the reference; a first-token repair cannot target it")
            continue
        probes.append(ProbePair(
            probe_id=f"fail-{task.id}", prompt=task.prompt, expected_completion_prefix=task.reference[:200],
            undesired_completion_prefix=(row["completion"] or " ")[:200], metadata={"task": task.id},
        ))
    if not probes:
        return None, notes
    keep = [t for t in suite.tasks if t.split == "train" and any(r["task"] == t.id and r["passed"] for r in baseline)]
    invariants = [ProbePair(probe_id=f"keep-{t.id}", prompt=t.prompt, expected_completion_prefix=" ",
                            metadata={"role": "invariant", "task": t.id}) for t in keep]
    invariants += [ProbePair(probe_id=f"general-{i}", prompt=p, expected_completion_prefix=" ",
                             metadata={"role": "invariant"}) for i, p in enumerate(suite.general_prompts)]
    if not invariants:
        notes.append("no passing task or general prompt to protect; add general prompts to the suite")
        return None, notes
    spec = IntentSpec(
        intent_id=f"factory-{suite.suite_id}", description=f"Factory repair for suite {suite.suite_id}: {suite.description}",
        target_behavior="On failing training tasks the model prefers the first token of the reference answer.",
        failure_behavior="On failing training tasks the model starts its answer differently from the reference.",
        contrastive_probes=probes + invariants,
        target_goals=[MetricGoal(suite_name="factory", metric_name="logit_margin", operator=">=", threshold=0.0)],
        invariants=[InvariantGuard(suite_name="general", metric_name="kl_divergence", max_allowed_drop=0.01)],
        constraints=constraints,
    )
    return spec, notes


def verdict(baseline: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]] | None, prompt_only: dict[str, Any],
            committed: bool) -> dict[str, Any]:
    """Compare after vs baseline vs prompt-only on held-out tasks (or training tasks when there are none)."""
    split = "heldout" if any(r["split"] == "heldout" for r in baseline) else "train"
    label = "held-out" if split == "heldout" else "training"
    base = pass_at_1(baseline, split)
    aft = pass_at_1(after, split) if after is not None else None
    prompt = prompt_only.get(split) if prompt_only.get("configured") else None
    messages: list[str] = []
    beats_base = aft is not None and base is not None and aft > base
    beats_prompt: bool | None = None if prompt is None else (aft is not None and aft > prompt)
    if not committed:
        messages.append("No edit was committed, so the model is unchanged; nothing to package.")
    elif not beats_base:
        messages.append(f"The edited model does not beat the baseline on {label} tasks ({aft} vs {base}).")
    if beats_prompt is False:
        messages.append(
            f"The edited model does not beat the prompt-only baseline on {label} tasks ({aft} vs {prompt}): a system "
            "prompt is enough here, so packaging is not offered."
        )
    if prompt is None:
        messages.append("No prompt-only baseline was configured; add prompt_baselines to compare against a system prompt.")
    allowed = committed and beats_base and beats_prompt is not False
    if allowed:
        messages.insert(0, f"The edited model beats the baseline ({aft} vs {base}) on {label} tasks"
                           + (f" and the prompt-only baseline ({prompt})." if prompt is not None else "."))
    return {"split": split, "baseline": base, "after": aft, "prompt_only": prompt, "beats_baseline": beats_base,
            "beats_prompt_only": beats_prompt, "packaging_allowed": allowed, "messages": messages}


# --- run folder ------------------------------------------------------------------------------------------------


class FactoryRun:
    """One run folder: suite, append-only check log, intent and report."""

    def __init__(self, workspace: Path, run_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", run_id):
            raise ValueError("invalid run id")
        self.dir = workspace / ".factory" / run_id
        self.run_id = run_id

    def create(self, suite: TaskSuite) -> FactoryRun:
        """Make the folder and store the suite."""
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "suite.yaml").write_text(yaml.safe_dump(suite.model_dump(), sort_keys=False), encoding="utf-8")
        return self

    def log(self, row: dict[str, Any]) -> None:
        """Append one check result to ``log.jsonl``."""
        with (self.dir / "log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def save_intent(self, spec: IntentSpec) -> None:
        """Store the intent built from failures."""
        (self.dir / "intent.yaml").write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")

    def save_report(self, report: dict[str, Any]) -> None:
        """Store the final report."""
        (self.dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    def report(self) -> dict[str, Any]:
        """The stored report (``ValueError`` when the run has none yet)."""
        path = self.dir / "report.json"
        if not path.is_file():
            raise ValueError("this factory run has no report yet")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("the report is damaged")
        return data

    def log_rows(self) -> list[dict[str, Any]]:
        """Every logged check result."""
        path = self.dir / "log.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_runs(workspace: Path) -> list[dict[str, Any]]:
    """Factory runs of a workspace, newest first, with their verdict when finished."""
    root = workspace / ".factory"
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir() or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", path.name):
            continue
        entry: dict[str, Any] = {"run_id": path.name}
        report = path / "report.json"
        if report.is_file():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
                entry.update(suite_id=data.get("suite_id"), verdict=data.get("verdict"))
            except ValueError:
                entry["error"] = "damaged report"
        out.append(entry)
    return out


# --- examples --------------------------------------------------------------------------------------------------


def suite_examples() -> list[dict[str, Any]]:
    """Example suites for the Studio. The demo suite matches the toy demo model; the others are templates."""
    demo = {
        "suite_id": "demo-no-x-loop",
        "description": "Toy demo: after 'x' the model must not continue with another 'x'.",
        "max_new_tokens": 4,
        "stop": [],
        "prompt_baselines": ["Never repeat a letter. ", "Answer with y after x. "],
        "general_prompts": ["hello world", "abc def ghi", "the quick brown", "lorem ipsum do"],
        "tasks": [
            {"id": f"t{i}", "prompt": p, "check": "not_starts_with", "expected": "x", "reference": "y", "split": "train"}
            for i, p in enumerate(["ab x", "cd x", "the x", "q: x"])
        ] + [
            {"id": f"h{i}", "prompt": p, "check": "not_starts_with", "expected": "x", "split": "heldout"}
            for i, p in enumerate(["mn x", "go x"])
        ],
    }
    python = {
        "suite_id": "python-functions",
        "description": "Template: small Python functions checked by unit tests. Adapt the tasks to your domain.",
        "max_new_tokens": 96,
        "stop": ["\ndef ", "\nclass ", "\n\n\n"],
        "prompt_baselines": ["# Write correct, simple Python.\n"],
        "tasks": [
            {"id": "add", "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "check": "python_test",
             "answer_starts_with_prompt": True, "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
             "reference": "    return a + b\n"},
            {"id": "is_even", "prompt": "def is_even(n):\n    \"\"\"True when n is even.\"\"\"\n", "check": "python_test",
             "answer_starts_with_prompt": True, "test": "assert is_even(4)\nassert not is_even(7)",
             "reference": "    return n % 2 == 0\n"},
            {"id": "reverse", "prompt": "def reverse(s):\n    \"\"\"Return s reversed.\"\"\"\n", "check": "python_test",
             "answer_starts_with_prompt": True, "test": "assert reverse('abc') == 'cba'", "split": "heldout"},
        ],
    }
    sql = {
        "suite_id": "sql-queries",
        "description": "Template: SQL queries checked against expected result rows on an in-memory database.",
        "max_new_tokens": 64,
        "stop": [";", "\n\n"],
        "tasks": [
            {"id": "adults", "prompt": "Table people(name TEXT, age INT). SQL query for the names of people aged 18 or more:\n",
             "check": "sql_rows", "setup": "create table people(name text, age int); insert into people values ('Ann', 30), ('Bo', 12);",
             "expected": [["Ann"]], "reference": "SELECT name FROM people WHERE age >= 18"},
            {"id": "count", "prompt": "Table people(name TEXT, age INT). SQL query that counts the people:\n",
             "check": "sql_rows", "setup": "create table people(name text, age int); insert into people values ('Ann', 30), ('Bo', 12);",
             "expected": [[2]], "split": "heldout"},
        ],
    }
    return [
        {"id": "demo", "title": "Stop the 'x' loop (matches the toy demo model)", "suite": demo},
        {"id": "python", "title": "Python functions with unit tests (template)", "suite": python},
        {"id": "sql", "title": "SQL queries with expected rows (template)", "suite": sql},
    ]


# --- frontier helper -------------------------------------------------------------------------------------------


def draft_prompt_baselines(backend: Any, suite: TaskSuite) -> list[str]:
    """Ask the frontier model for three system prompts, showing it training tasks only (never held-out)."""
    from libweightspace.candidates import extract_json

    shown = [{"prompt": t.prompt[:500], "check": t.check} for t in suite.tasks if t.split == "train"][:8]
    reply = backend.complete(
        "You write short system prompts that make a small model solve a task family. Reply with JSON only.",
        "Task family: " + json.dumps(suite.description) + "\nExample tasks: " + json.dumps(shown)
        + '\nReturn {"prompts": [three different system prompts, each under 300 characters]}.',
    )
    data = extract_json(reply)
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("the model did not return a list of prompts")
    return [p.strip()[:300] + ("\n" if not p.endswith("\n") else "") for p in prompts if p.strip()][:3]
