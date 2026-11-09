"""Studio routes for the Model Factory (CR9): run a task suite end to end, read reports, package results.

One factory run is one job: baseline -> prompt-only baseline -> guarded repair on failing training tasks (with the
same approval, budgets and frontier proposer as the Edit view) -> after -> verdict. Every check result is appended to
the run's ``log.jsonl``; the report is saved as ``report.json``. Packaging writes inside the workspace and is refused
unless the verdict allows it.

Main entry points: ``FactoryMixin`` and ``FACTORY_GET_ROUTES`` / ``FACTORY_POST_ROUTES``.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from datetime import datetime
from http import HTTPStatus
from typing import Any

import yaml
from pydantic import ValidationError

from .factory import (
    FactoryRun,
    TaskSuite,
    build_repair_intent,
    draft_prompt_baselines,
    evaluate_suite,
    list_runs,
    pass_at_1,
    prompt_only_baseline,
    suite_examples,
    verdict,
)
from .packaging import PACKAGE_KINDS, build_package, package_destination
from .schemas import EditConstraints
from .studio_common import StudioCore, StudioError, bool_field, int_field, str_field

_RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
TOY_REPAIR_LIMITS = EditConstraints(max_modified_param_ratio=0.01, max_frobenius_rel_norm=0.15, max_spectral_norm_shift=0.05)


def parse_suite(text: str) -> TaskSuite:
    """A suite from YAML or JSON text (400 with the first problem when it is invalid)."""
    try:
        return TaskSuite.model_validate(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise StudioError(HTTPStatus.BAD_REQUEST, f"suite is not valid YAML: {str(exc).splitlines()[0]}") from None
    except ValidationError as exc:
        err = exc.errors()[0]
        where = ".".join(str(p) for p in err.get("loc", ()))
        raise StudioError(HTTPStatus.BAD_REQUEST, f"suite is invalid at {where or 'top level'}: {err['msg']}") from None


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"train": pass_at_1(results, "train"), "heldout": pass_at_1(results, "heldout"),
            "tasks": [{k: r[k] for k in ("task", "split", "passed", "reason", "completion")} for r in results]}


class FactoryMixin(StudioCore):
    """Model Factory (CR9)."""

    def factory_examples(self) -> dict[str, Any]:
        """Example suites as YAML for the editor."""
        return {"examples": [{**ex, "yaml": yaml.safe_dump(ex["suite"], sort_keys=False)} for ex in suite_examples()]}

    def factory_runs(self) -> dict[str, Any]:
        """Runs of this workspace, newest first."""
        return {"runs": list_runs(self.workspace)}

    def factory_report(self, run_id: str) -> dict[str, Any]:
        """A finished run's report plus every logged check."""
        if not _RUN_RE.fullmatch(run_id or ""):
            raise StudioError(HTTPStatus.BAD_REQUEST, "invalid run id")
        run = FactoryRun(self.workspace, run_id)
        try:
            return {"report": run.report(), "log": run.log_rows()}
        except ValueError as exc:
            raise StudioError(HTTPStatus.NOT_FOUND, str(exc)) from None

    def start_factory(self, body: Mapping[str, Any]) -> str:
        """Start a factory run as a job (see the module docstring)."""
        self._require_idle()
        suite = parse_suite(str_field(body, "suite_yaml", 60_000))
        iterations = int_field(body, "iterations", 2, 1, 10)
        candidates = int_field(body, "candidates", 3, 1, 8)
        approve_each = bool_field(body, "approve_each", True)
        source, proposer = self.repair_proposer(bool_field(body, "use_llm"))
        constraints = TOY_REPAIR_LIMITS if self.is_toy else EditConstraints()
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

        def job(job_id: str) -> dict[str, Any]:
            run = FactoryRun(self.workspace, run_id).create(suite)
            head_before = self.repo.head()

            def progress(phase_share: tuple[float, float]) -> Any:
                low, high = phase_share
                return lambda text, frac: self.jobs.update_progress(job_id, text, low + (high - low) * frac)

            cancel = lambda: self.jobs.cancel_requested(job_id)  # noqa: E731 - passed to several helpers
            with self.model_lock:
                model, tok = self.loaded.model, self.loaded.tokenizer
                baseline = evaluate_suite(model, tok, suite, "baseline", log=run.log, progress=progress((0.0, 0.2)),
                                          should_cancel=cancel)
                prompt_only = prompt_only_baseline(model, tok, suite, log=run.log, progress=progress((0.2, 0.35)),
                                                   should_cancel=cancel)
            intent, notes = build_repair_intent(suite, baseline, tok, constraints)
            repair: dict[str, Any] | None = None
            if intent is not None:
                run.save_intent(intent)
                self._check_cancel(job_id)
                repair = self.run_repair(job_id, intent, source, iterations, candidates, approve_each=approve_each)
            committed = self.repo.head() != head_before
            after = None
            if committed:
                with self.model_lock:
                    after = evaluate_suite(self.loaded.model, tok, suite, "after", log=run.log,
                                           progress=progress((0.85, 1.0)), should_cancel=cancel)
            report = {
                "run_id": run_id, "suite_id": suite.suite_id, "suite": suite.model_dump(), "proposer": proposer,
                "baseline": _summary(baseline), "prompt_only": prompt_only, "after": _summary(after) if after else None,
                "repair": repair, "notes": notes, "verdict": verdict(baseline, after, prompt_only, committed),
                "head_before": head_before, "head_after": self.repo.head(),
                "heldout_shown_to_proposer": False,
            }
            run.save_report(report)
            return report

        return self.jobs.start("factory", job)

    def factory_destination(self, kind: str) -> dict[str, Any]:
        """Where a package would be written (chosen by the server, inside the workspace)."""
        try:
            path = package_destination(self.workspace, kind)
        except ValueError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        return {"kind": kind, "path": str(path.relative_to(self.workspace)), "workspace": self.workspace.name}

    def factory_package(self, run_id: str, kind: str) -> dict[str, Any]:
        """Build the package for a run whose verdict allows it."""
        self._require_idle()
        if kind not in PACKAGE_KINDS:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"kind must be one of {', '.join(PACKAGE_KINDS)}")
        report = self.factory_report(run_id)["report"]
        with self.model_lock:
            try:
                return build_package(self.workspace, self.cfg, self.repo, report, kind)
            except ValueError as exc:
                raise StudioError(HTTPStatus.CONFLICT, str(exc)) from None

    def factory_draft_prompts(self, suite_yaml: str) -> dict[str, Any]:
        """System prompts for the prompt-only baseline drafted by the frontier model from training tasks only."""
        suite = parse_suite(suite_yaml)
        return {"prompts": draft_prompt_baselines(self._backend(), suite), "drafted_by": self.selected_model}


FACTORY_GET_ROUTES: dict[str, Any] = {
    "/api/factory/examples": lambda s, q: (HTTPStatus.OK, s.factory_examples()),
    "/api/factory/runs": lambda s, q: (HTTPStatus.OK, s.factory_runs()),
    "/api/factory/report": lambda s, q: (HTTPStatus.OK, s.factory_report((q.get("run") or [""])[0])),
    "/api/factory/destination": lambda s, q: (HTTPStatus.OK, s.factory_destination((q.get("kind") or [""])[0])),
}
FACTORY_POST_ROUTES: dict[str, Any] = {
    "/api/factory/run": lambda s, b: (HTTPStatus.ACCEPTED, {"job_id": s.start_factory(b)}),
    "/api/factory/package": lambda s, b: (HTTPStatus.OK, s.factory_package(str_field(b, "run_id", 32), str_field(b, "kind", 8))),
    "/api/factory/draft_prompts": lambda s, b: (HTTPStatus.OK, s.factory_draft_prompts(str_field(b, "suite_yaml", 60_000))),
}
