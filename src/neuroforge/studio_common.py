"""Shared core of the Studio server: errors, the job manager, the workspace session and field validation.

``StudioCore`` holds what every feature needs: the optional open workspace (``WorkspaceSession``), LLM settings and
the selected frontier model, the job manager and the model lock. Feature mixins (``studio_modes``, ``studio_factory``,
``studio_selfimprove``) inherit from it, and ``studio.StudioState`` combines them, so each feature lives in its own
module while sharing one session and one job queue.

Proposer policy (owner decision 2026-11-09, REQ-241): edits on real models are proposed only by the frontier model
chosen in step 1. Offline templates are accepted only for the bundled toy models (demo and tests).

Main entry points: ``StudioError``, ``JobManager``, ``WorkspaceSession``, ``StudioCore``, ``bool_field``,
``int_field``, ``float_field``, ``str_field``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from libweightspace.candidates import CandidateSource, TemplateCandidateSource
from libweightspace.llm_config import LLMSettings, ModelListError
from libweightspace.loading import LoadedModel

from .cancellation import OperationCancelled
from .reasoner_bridge import ReasonerBridge
from .schemas import IntentSpec, WorkspaceConfig
from .weight_git import WeightGitError, WeightGitRepository
from .workspace_manager import StudioHome
from .workspace_ops import build_engine, convergence_summary

logger = logging.getLogger(__name__)

MAX_KEPT_JOBS = 50
TOY_ONLY_MESSAGE = (
    "Edits on real models are proposed by a frontier model: choose one in step 1. "
    "Offline templates only work with the built-in toy models."
)


class StudioError(Exception):
    """A request the Studio refuses with a specific status and a one-line, user-facing message."""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


class JobCancelled(OperationCancelled):
    """Raised inside a job when the user cancelled it; the job ends in state ``cancelled``."""


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` next to ``path`` and rename it into place."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def finite_or_none(value: float) -> float | None:
    """JSON-safe number for scores that may be infinite before evaluation."""
    return None if not math.isfinite(value) else round(float(value), 6)


class JobManager:
    """Runs at most one background job; results are kept in memory for polling (the newest ``MAX_KEPT_JOBS``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: str | None = None
        self._decisions: dict[str, list[Any]] = {}

    @property
    def busy(self) -> bool:
        """True while a job is running."""
        with self._lock:
            return self._active is not None

    def start(self, kind: str, fn: Callable[[str], dict[str, Any]]) -> str:
        """Start ``fn(job_id)`` in a daemon thread; raises ``StudioError`` (409) when another job is running."""
        with self._lock:
            if self._active is not None:
                raise StudioError(HTTPStatus.CONFLICT, "another job is still running; wait for it to finish or cancel it")
            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {"id": job_id, "kind": kind, "state": "running", "started": time.time()}
            self._active = job_id
            for old in list(self._jobs)[:-MAX_KEPT_JOBS]:
                self._jobs.pop(old, None)

        def runner() -> None:
            try:
                result = fn(job_id)
                update: dict[str, Any] = {"state": "done", "result": result}
            except OperationCancelled as exc:
                update = {"state": "cancelled", "error": str(exc) or "cancelled"}
            except (ValueError, WeightGitError, ValidationError, ModelListError, StudioError) as exc:
                update = {"state": "failed", "error": str(exc)}
            except Exception:  # surfaced generically; details go to the server log only
                logger.exception("Studio job %s failed", job_id)
                update = {"state": "failed", "error": "internal error (see the terminal running the Studio)"}
            with self._lock:
                self._jobs[job_id].update(update, finished=time.time())
                self._active = None

        threading.Thread(target=runner, name=f"studio-job-{job_id}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        """Snapshot of one job; raises ``StudioError`` (404) for unknown ids."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise StudioError(HTTPStatus.NOT_FOUND, "unknown job id")
            return dict(job)

    # --- progress, cancel and approval (CR5, CR7) -------------------------------------------------------------
    def update_progress(self, job_id: str, text: str, fraction: float) -> None:
        """Record the job's current step for polling."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = {"text": text, "fraction": round(max(0.0, min(1.0, fraction)), 3)}

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to stop at the next safe point."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise StudioError(HTTPStatus.NOT_FOUND, "unknown job id")
            if job["state"] not in ("running", "awaiting_approval"):
                raise StudioError(HTTPStatus.CONFLICT, "the job is not running")
            job["cancel"] = True
            event = self._decisions.get(job_id)
        if event is not None:
            event[0].set()
        return {"cancel_requested": True}

    def cancel_requested(self, job_id: str) -> bool:
        """True once cancel was requested for this job."""
        with self._lock:
            return bool(self._jobs.get(job_id, {}).get("cancel"))

    def set_flag(self, job_id: str, name: str) -> None:
        """Set a named flag on a running job (for example ``pause``); raises 409 when the job is not running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise StudioError(HTTPStatus.NOT_FOUND, "unknown job id")
            if job["state"] not in ("running", "awaiting_approval"):
                raise StudioError(HTTPStatus.CONFLICT, "the job is not running")
            job[name] = True

    def flag(self, job_id: str, name: str) -> bool:
        """True once ``name`` was set on this job."""
        with self._lock:
            return bool(self._jobs.get(job_id, {}).get(name))

    def await_decision(self, job_id: str, summary: dict[str, Any], timeout: float = 1800.0) -> bool:
        """Block the job until the user approves or rejects ``summary`` (rejects on cancel or timeout)."""
        event = threading.Event()
        holder: list[Any] = [event, False]
        with self._lock:
            self._decisions[job_id] = holder
            self._jobs[job_id].update(state="awaiting_approval", awaiting=summary)
        event.wait(timeout)
        with self._lock:
            self._decisions.pop(job_id, None)
            job = self._jobs[job_id]
            job.pop("awaiting", None)
            job["state"] = "running"
            return bool(holder[1]) and not job.get("cancel")

    def decide(self, job_id: str, approve: bool) -> dict[str, Any]:
        """Deliver the user's decision to a job waiting for approval."""
        with self._lock:
            holder = self._decisions.get(job_id)
            if holder is None:
                raise StudioError(HTTPStatus.CONFLICT, "this job is not waiting for a decision")
            holder[1] = approve
        holder[0].set()
        return {"approved": approve}

    def active(self) -> str | None:
        """Id of the running job, if any."""
        with self._lock:
            return self._active

    def active_kind(self) -> str | None:
        """Kind of the running job, if any."""
        with self._lock:
            return self._jobs[self._active]["kind"] if self._active else None


@dataclass
class WorkspaceSession:
    """One open workspace: folder, config, the model at HEAD and the edit history."""

    workspace: Path
    cfg: WorkspaceConfig
    loaded: LoadedModel
    repo: WeightGitRepository


NO_WORKSPACE = "open or create a workspace first (Workspaces view)"


class StudioCore:
    """Session, LLM settings, jobs and helpers shared by every Studio feature."""

    def __init__(self, settings: LLMSettings, home: StudioHome, session: WorkspaceSession | None, max_body_bytes: int):
        self.settings = settings
        self.selected_model: str | None = settings.model
        self.home = home
        self._session = session
        self.max_body_bytes = max_body_bytes
        self.jobs = JobManager()
        self.model_lock = threading.Lock()

    # --- workspace session -------------------------------------------------------------------------------------
    @property
    def has_workspace(self) -> bool:
        """True when a workspace is open."""
        return self._session is not None

    def _require_session(self) -> WorkspaceSession:
        if self._session is None:
            raise StudioError(HTTPStatus.CONFLICT, NO_WORKSPACE)
        return self._session

    @property
    def workspace(self) -> Path:
        """Folder of the open workspace (409 when none is open)."""
        return self._require_session().workspace

    @property
    def cfg(self) -> WorkspaceConfig:
        """Config of the open workspace."""
        return self._require_session().cfg

    @cfg.setter
    def cfg(self, value: WorkspaceConfig) -> None:
        self._require_session().cfg = value

    @property
    def loaded(self) -> LoadedModel:
        """The model of the open workspace, at HEAD."""
        return self._require_session().loaded

    @loaded.setter
    def loaded(self, value: LoadedModel) -> None:
        self._require_session().loaded = value

    @property
    def repo(self) -> WeightGitRepository:
        """Edit history of the open workspace."""
        return self._require_session().repo

    @repo.setter
    def repo(self, value: WeightGitRepository) -> None:
        self._require_session().repo = value

    @property
    def is_toy(self) -> bool:
        """True for the bundled toy models (the only ones offline templates may edit)."""
        return self.cfg.model.startswith("toy:")

    # --- LLM ---------------------------------------------------------------------------------------------------
    def _backend(self) -> Any:
        if not self.selected_model:
            raise StudioError(HTTPStatus.CONFLICT, "choose a model in step 1 first (a frontier model from your endpoint)")
        return self.settings.backend(self.selected_model)

    def proposer_label(self) -> str:
        """How the selected frontier model is named in results."""
        return f"frontier model: {self.selected_model} via {self.settings.describe()['provider']}"

    def repair_proposer(self, use_llm: bool) -> tuple[CandidateSource, str]:
        """The candidate source for repair-style loops under the frontier-only policy.

        Raises:
            StudioError: 409 without a selected model; 400 when templates are asked for a non-toy model.
        """
        if use_llm:
            return ReasonerBridge(self._backend()), self.proposer_label()
        if not self.is_toy:
            raise StudioError(HTTPStatus.BAD_REQUEST, TOY_ONLY_MESSAGE)
        return TemplateCandidateSource(), "offline demo templates (toy model, no LLM)"

    # --- jobs and results --------------------------------------------------------------------------------------
    def _require_idle(self) -> None:
        if self.jobs.busy:
            raise StudioError(HTTPStatus.CONFLICT, "a job is running; wait for it to finish or cancel it")

    def _save_result(self, name: str, payload: dict[str, Any]) -> None:
        folder = self.workspace / ".studio"
        folder.mkdir(exist_ok=True)
        record = {**payload, "saved_at": time.time(), "measured_at_head": self.repo.head()}
        write_text_atomic(folder / name, json.dumps(record))

    def _check_cancel(self, job_id: str) -> None:
        if self.jobs.cancel_requested(job_id):
            raise JobCancelled("cancelled by you; nothing was applied")

    def run_repair(
        self,
        job_id: str,
        intent: IntentSpec,
        source: CandidateSource,
        iterations: int,
        candidates: int,
        dry_run: bool = False,
        approve_each: bool = False,
        protected: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run the guarded repair loop inside job ``job_id`` and return its measured summary.

        ``approve_each`` pauses before every commit until the user decides (the approval shows the script, its size and
        its measured effect); ``dry_run`` evaluates without changing the model or the history.
        """

        def approve(ev: Any) -> bool:
            stats = [{"tensor": s.tensor_name, "modified": int(s.nnz_count)} for s in ev.tensor_stats]
            summary = {
                "hypothesis": ev.candidate.hypothesis[:400],
                "script": ev.candidate.python_script[:6000],
                "source": ev.candidate.source,
                "target_score": finite_or_none(ev.target_score),
                "invariant_degradation": finite_or_none(ev.invariant_degradation),
                "modified_params": int(ev.modified_params),
                "tensors": stats,
            }
            return self.jobs.await_decision(job_id, summary)

        with self.model_lock:
            engine = build_engine(
                self.cfg, self.loaded, intent, source, None if dry_run else self.repo, iterations, candidates,
                progress=lambda text, frac: self.jobs.update_progress(job_id, text, frac),
                should_cancel=lambda: self.jobs.cancel_requested(job_id),
                protected_tensors=protected,
                approve=approve if approve_each and not dry_run else None,
                commit=not dry_run,
            )
            result = engine.run()
            summary = convergence_summary(result)
        summary.update(dry_run=dry_run, cancelled=result.cancelled, head=self.repo.head())
        return summary


# --- request field validation ----------------------------------------------------------------------------------


def bool_field(body: Mapping[str, Any], key: str, default: bool = False) -> bool:
    """A JSON boolean or 400."""
    value = body.get(key, default)
    if not isinstance(value, bool):
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be true or false")
    return value


def int_field(body: Mapping[str, Any], key: str, default: int, low: int, high: int) -> int:
    """A JSON integer within ``[low, high]`` or 400."""
    value = body.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be an integer between {low} and {high}")
    return value


def float_field(body: Mapping[str, Any], key: str, default: float, low: float, high: float) -> float:
    """A JSON number within ``[low, high]`` or 400."""
    value = body.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be a number between {low} and {high}")
    return float(value)


def str_field(body: Mapping[str, Any], key: str, max_len: int, default: str = "") -> str:
    """JSON text of at most ``max_len`` characters or 400."""
    value = body.get(key, default)
    if not isinstance(value, str) or len(value) > max_len:
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be text of at most {max_len} characters")
    return value


def lines_field(body: Mapping[str, Any], key: str, max_lines: int, max_line_len: int = 1000, min_lines: int = 0) -> list[str]:
    """A list of non-empty strings (or newline-separated text) or 400."""
    value = body.get(key, [])
    items = value.splitlines() if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be a list of texts")
    cleaned = [v.strip() for v in items if v.strip()]
    if len(cleaned) > max_lines or any(len(v) > max_line_len for v in cleaned) or len(cleaned) < min_lines:
        raise StudioError(
            HTTPStatus.BAD_REQUEST,
            f"{key}: {min_lines} to {max_lines} entries of at most {max_line_len} characters each",
        )
    return cleaned
