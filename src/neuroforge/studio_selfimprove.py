"""Studio routes for the self-improvement runtime (CR10): start, pause, stop, roll back, timeline.

The run lives in ``<workspace>/.wrsi`` and starts from the workspace's original model; it is separate from the edit
history of the Edit view (the page says so). Its task is the intent's failing prompts (step 2), its KL anchors are the
intent's protected prompts, and the forgetting check uses held-out ``prompt => answer`` lines given here. Proposals
come from the frontier model chosen in step 1; offline templates are allowed only for the toy models.

Main entry points: ``SelfImproveMixin`` and ``SELFIMPROVE_GET_ROUTES`` / ``SELFIMPROVE_POST_ROUTES``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import yaml

from libweightspace.candidates import CandidateSource, TemplateCandidateSource
from libweightspace.loading import LoadedModel
from wrsi.runner import Guardrails, SessionInputs, forgetting_pairs, rollback, rollback_preview, run_session, timeline
from wrsi.runner import status as wrsi_status
from wrsi.scientist_agent import GymTemplateProposer, frontier_proposer
from wrsi.workspace import WRSIWorkspace

from .intent_compiler import is_invariant
from .schemas import IntentSpec
from .studio_common import (
    TOY_ONLY_MESSAGE,
    StudioCore,
    StudioError,
    bool_field,
    float_field,
    int_field,
    lines_field,
    str_field,
)
from .studio_common import write_text_atomic as _write


class SelfImproveMixin(StudioCore):
    """Self-improvement runtime (CR10)."""

    def _wrsi_inputs(self) -> SessionInputs:
        intent_file = self.workspace / "intent.yaml"
        domains = anchors = None
        if intent_file.is_file():
            spec = IntentSpec.model_validate(yaml.safe_load(intent_file.read_text(encoding="utf-8")))
            failing = [p.model_dump() for p in spec.contrastive_probes if not is_invariant(p) and p.undesired_completion_prefix]
            if failing:
                domains = [{"name": spec.intent_id, "failure_taxonomy": "attractor_loop", "probes": failing}]
            anchors = [p.prompt for p in spec.contrastive_probes if is_invariant(p)] or None
        if domains is None and not self.is_toy:
            raise StudioError(HTTPStatus.CONFLICT,
                              "describe a failure in step 2 first: self-improvement works on the intent's failing prompts")
        if anchors is None and not self.is_toy:
            raise StudioError(HTTPStatus.CONFLICT,
                              "the intent needs prompts that must not change (role: invariant) to anchor the KL limit")
        return SessionInputs(model=self.cfg.load_spec, demo_defect=self.cfg.demo_defect, domains=domains, anchors=anchors)

    def _wrsi_proposers(self, use_llm: bool) -> tuple[Any, str]:
        if use_llm:
            backend = self._backend()

            def frontier(_loaded: LoadedModel) -> tuple[CandidateSource, CandidateSource]:
                llm = frontier_proposer(backend, f"frontier:{self.selected_model}")
                return llm, llm

            return frontier, self.proposer_label()
        if not self.is_toy:
            raise StudioError(HTTPStatus.BAD_REQUEST, TOY_ONLY_MESSAGE)

        def templates(_loaded: LoadedModel) -> tuple[CandidateSource, CandidateSource]:
            return TemplateCandidateSource(), GymTemplateProposer()

        return templates, "offline demo templates (toy model, no LLM)"

    def selfimprove_status(self) -> dict[str, Any]:
        """Saved state, guardrails, timeline, the rollback point and whether a run is active."""
        ws = WRSIWorkspace(self.workspace)
        out: dict[str, Any] = {"exists": ws.exists(), "toy": self.is_toy, "timeline": timeline(self.workspace, 100),
                               "rollback": rollback_preview(self.workspace), "running": None}
        if self.jobs.active_kind() == "selfimprove":
            out["running"] = self.jobs.active()
        if ws.exists():
            out["state"] = wrsi_status(self.workspace)
        guard = ws.dir / "studio.json"
        if guard.is_file():
            try:
                out["guardrails"] = json.loads(guard.read_text(encoding="utf-8"))
            except ValueError:
                out["guardrails"] = None
        return out

    def start_selfimprove(self, body: Mapping[str, Any]) -> str:
        """Start (or resume) a run as a job; pause and stop are job flags checked between steps and candidates."""
        self._require_idle()
        protected = tuple(p.strip() for p in str_field(body, "protected", 400).split(",") if p.strip())
        try:
            pairs = forgetting_pairs(lines_field(body, "forgetting", 32, 600))
        except ValueError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        guard = Guardrails(
            max_kl=float_field(body, "max_kl", 0.08, 0.001, 1.0),
            protected_tensors=protected,
            forgetting=pairs,
            max_forgetting_drop=float_field(body, "max_forgetting_drop", 0.05, 0.0, 5.0),
            candidates_per_generation=int_field(body, "candidates", 3, 1, 8),
            meta_fraction=float_field(body, "meta_fraction", 0.0, 0.0, 1.0),
        )
        generations = int_field(body, "generations", 3, 1, 50)
        factory, label = self._wrsi_proposers(bool_field(body, "use_llm"))
        inputs = self._wrsi_inputs()
        record = {"max_kl": guard.max_kl, "protected": list(protected), "forgetting": [f"{p} => {a}" for p, a in pairs],
                  "max_forgetting_drop": guard.max_forgetting_drop, "candidates": guard.candidates_per_generation,
                  "generations": generations, "proposer": label, "meta_fraction": guard.meta_fraction}

        def job(job_id: str) -> dict[str, Any]:
            ws = WRSIWorkspace(self.workspace)
            ws.dir.mkdir(parents=True, exist_ok=True)
            _write(ws.dir / "studio.json", json.dumps(record))
            result = run_session(
                self.workspace, inputs, guard, factory, generations,
                progress=lambda t, f: self.jobs.update_progress(job_id, t, f),
                should_pause=lambda: self.jobs.flag(job_id, "pause"),
                should_stop=lambda: self.jobs.cancel_requested(job_id),
                proposer_label=label,
            )
            return {**result, "proposer": label}

        return self.jobs.start("selfimprove", job)

    def selfimprove_rollback(self, confirm: bool) -> dict[str, Any]:
        """Undo the newest promoted generation bit-exactly (needs ``confirm``; the preview says what changes)."""
        self._require_idle()
        if not confirm:
            raise StudioError(HTTPStatus.BAD_REQUEST, "confirm the rollback (the preview lists what will change)")
        try:
            return rollback(self.workspace, 1)
        except (FileNotFoundError, ValueError) as exc:
            raise StudioError(HTTPStatus.CONFLICT, str(exc)) from None


SELFIMPROVE_GET_ROUTES: dict[str, Any] = {
    "/api/selfimprove/status": lambda s, q: (HTTPStatus.OK, s.selfimprove_status()),
}
SELFIMPROVE_POST_ROUTES: dict[str, Any] = {
    "/api/selfimprove/start": lambda s, b: (HTTPStatus.ACCEPTED, {"job_id": s.start_selfimprove(b)}),
    "/api/selfimprove/rollback": lambda s, b: (HTTPStatus.OK, s.selfimprove_rollback(bool_field(b, "confirm"))),
}
