"""Studio routes for the edit modes of CR6: update facts and adjust a behaviour direction.

Previews run as jobs (they generate and measure); the server keeps each previewed plan under an id, and applying
commits exactly that plan as one patch, provided the model has not moved since the preview. Defaults differ for the
bundled toy models (whose blocks write very little into the residual stream) and real checkpoints.

Main entry points: ``ModesMixin`` and ``MODES_GET_ROUTES`` / ``MODES_POST_ROUTES``.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from pydantic import ValidationError

from .edit_modes import (
    DirectionSpec,
    EditPlan,
    Fact,
    FactBatch,
    commit_plan,
    default_fact_layers,
    draft_direction_sets,
    draft_fact_probes,
    editable_layers,
    parse_facts_csv,
    plan_direction_edit,
    plan_fact_edit,
)
from .schemas import EditConstraints
from .studio_common import StudioCore, StudioError, float_field, int_field, lines_field, str_field

MAX_PLANS = 10
TOY_LIMITS = {"budget_percent": 1.0, "max_frobenius": 0.15, "max_spectral": 0.10}
REAL_LIMITS = {"budget_percent": 0.05, "max_frobenius": 0.10, "max_spectral": 0.10}
TOY_NOTE = (
    "The toy model's blocks write very little into its residual stream, so a fact edit large enough to change an "
    "answer exceeds the safety limits: the preview shows the measured effect and refuses to apply it. Direction edits "
    "fit. On real checkpoints both fit (validated on LFM 3 1.7B)."
)


class ModesMixin(StudioCore):
    """Fact and direction modes (CR6)."""

    _plans: dict[str, EditPlan]

    def _remember_plan(self, plan: EditPlan) -> None:
        if not hasattr(self, "_plans"):
            self._plans = {}
        self._plans[plan.plan_id] = plan
        for old in list(self._plans)[:-MAX_PLANS]:
            self._plans.pop(old, None)

    def _limits(self, body: Mapping[str, Any]) -> EditConstraints:
        defaults = TOY_LIMITS if self.is_toy else REAL_LIMITS
        budget = float_field(body, "budget_percent", defaults["budget_percent"], 0.001, 5.0)
        return EditConstraints(
            max_modified_param_ratio=budget / 100.0,
            max_frobenius_rel_norm=float_field(body, "max_frobenius", defaults["max_frobenius"], 0.001, 0.20),
            max_spectral_norm_shift=float_field(body, "max_spectral", defaults["max_spectral"], 0.001, 0.10),
        )

    def _layers_field(self, body: Mapping[str, Any]) -> list[int] | None:
        raw = str_field(body, "layers", 80).strip()
        if not raw:
            return None
        try:
            layers = sorted({int(p) for p in raw.replace(" ", "").split(",") if p})
        except ValueError:
            raise StudioError(HTTPStatus.BAD_REQUEST, "layers: comma-separated layer numbers, or empty for automatic") from None
        allowed = set(editable_layers(self.loaded))
        bad = [i for i in layers if i not in allowed]
        if bad or not layers or len(layers) > 8:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"layers must be up to 8 of: {sorted(allowed)}")
        return layers

    # --- info and examples -------------------------------------------------------------------------------------
    def modes_info(self) -> dict[str, Any]:
        """Editable layers, the default fact window, default limits and the toy-model note."""
        layers = editable_layers(self.loaded)
        return {
            "editable_layers": layers,
            "default_fact_layers": default_fact_layers(self.loaded) if layers else [],
            "limits": TOY_LIMITS if self.is_toy else REAL_LIMITS,
            "toy": self.is_toy,
            "note": TOY_NOTE if self.is_toy else None,
            "examples": self._examples(),
        }

    def _examples(self) -> dict[str, Any]:
        if self.is_toy:
            facts = [
                {"prompt": "ab x", "new_object": "y", "old_object": "x", "paraphrases": ["cd x"],
                 "neighbours": ["hello world", "abc def ghi"]},
            ]
            direction = {"name": "exclaim", "positives": ["wow!!!", "yes!!!", "go!!!", "hey!!!"],
                         "negatives": ["wow...", "yes...", "go...", "hey..."]}
        else:
            facts = [
                {"prompt": "The capital of Exampleland is", "subject": "Exampleland", "new_object": " Newtown",
                 "old_object": " Oldtown", "paraphrases": ["Exampleland's capital city is"],
                 "neighbours": ["The capital of France is", "The capital of Spain is"]},
            ]
            direction = {
                "name": "exclaim",
                "positives": ["What a wonderful day!!!", "I love this so much!!!", "This is amazing!!!", "We won!!!"],
                "negatives": ["What a wonderful day.", "I love this so much.", "This is amazing.", "We won."],
            }
        return {"facts": facts, "direction": direction}

    # --- facts -------------------------------------------------------------------------------------------------
    def _fact_batch(self, body: Mapping[str, Any]) -> FactBatch:
        raw_csv = str_field(body, "csv", 60_000)
        try:
            facts = parse_facts_csv(raw_csv) if raw_csv.strip() else [Fact.model_validate(f) for f in _list(body, "facts", 32)]
            return FactBatch(
                batch_id=str_field(body, "batch_id", 64, "facts") or "facts", facts=facts,
                layers=self._layers_field(body), constraints=self._limits(body),
                margin=float_field(body, "margin", 1.0, 0.05, 5.0), steps=int_field(body, "steps", 30, 1, 200),
            )
        except ValidationError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"facts: {exc.errors()[0]['msg']}") from None
        except ValueError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None

    def start_fact_preview(self, body: Mapping[str, Any]) -> str:
        """Compute and measure a fact batch as a job; the plan is kept for ``apply_plan``."""
        self._require_idle()
        batch = self._fact_batch(body)

        def job(job_id: str) -> dict[str, Any]:
            plan_id = secrets.token_hex(8)
            with self.model_lock:
                plan = plan_fact_edit(
                    self.loaded, self.repo, batch, plan_id,
                    progress=lambda t, f: self.jobs.update_progress(job_id, t, f),
                    should_cancel=lambda: self.jobs.cancel_requested(job_id),
                )
            self._remember_plan(plan)
            return plan.report

        return self.jobs.start("facts-preview", job)

    def parse_csv(self, text: str) -> dict[str, Any]:
        """Validate CSV text and return the facts it holds (nothing is computed)."""
        try:
            return {"facts": [f.model_dump() for f in parse_facts_csv(text)]}
        except ValueError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None

    def draft_fact(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Paraphrases and neighbour prompts for one fact, drafted by the frontier model (for review)."""
        try:
            fact = Fact(prompt=str_field(body, "prompt", 400), new_object=str_field(body, "new_object", 200))
        except ValidationError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"fact: {exc.errors()[0]['msg']}") from None
        return {**draft_fact_probes(self._backend(), fact), "drafted_by": self.selected_model}

    # --- direction ---------------------------------------------------------------------------------------------
    def _direction_spec(self, body: Mapping[str, Any]) -> DirectionSpec:
        try:
            return DirectionSpec(
                name=str_field(body, "name", 64, "behaviour") or "behaviour",
                positives=lines_field(body, "positives", 64, 400, min_lines=2),
                negatives=lines_field(body, "negatives", 64, 400, min_lines=2),
                layer=_optional_layer(body), strength=float_field(body, "strength", 1.0, -1.0, 1.0),
                constraints=self._limits(body),
            )
        except ValidationError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"direction: {exc.errors()[0]['msg']}") from None

    def start_direction_preview(self, body: Mapping[str, Any]) -> str:
        """Per-layer strength, the edit and its measured effect (KL, shift, answers), as a job."""
        self._require_idle()
        spec = self._direction_spec(body)
        if spec.layer is not None and spec.layer not in editable_layers(self.loaded):
            raise StudioError(HTTPStatus.BAD_REQUEST, f"layer must be one of {editable_layers(self.loaded)}")

        def job(job_id: str) -> dict[str, Any]:
            plan_id = secrets.token_hex(8)
            with self.model_lock:
                plan = plan_direction_edit(self.loaded, self.repo, spec, plan_id,
                                           progress=lambda t, f: self.jobs.update_progress(job_id, t, f))
            self._remember_plan(plan)
            return plan.report

        return self.jobs.start("direction-preview", job)

    def draft_direction(self, description: str) -> dict[str, Any]:
        """Two contrasting prompt sets drafted by the frontier model (for review)."""
        if not description.strip():
            raise StudioError(HTTPStatus.BAD_REQUEST, "describe the behaviour first")
        return {**draft_direction_sets(self._backend(), description), "drafted_by": self.selected_model}

    # --- apply -------------------------------------------------------------------------------------------------
    def apply_plan(self, plan_id: str) -> dict[str, Any]:
        """Commit a previewed plan as one patch (409 when the model moved since the preview)."""
        self._require_idle()
        plan = getattr(self, "_plans", {}).get(plan_id)
        if plan is None:
            raise StudioError(HTTPStatus.BAD_REQUEST, "unknown or expired preview; preview again")
        if plan.head != self.repo.head():
            raise StudioError(HTTPStatus.CONFLICT, "the model changed since this preview; preview again before applying")
        if plan.report.get("violations"):
            raise StudioError(HTTPStatus.CONFLICT, "this edit breaks the limits shown in the preview; it cannot be applied")
        with self.model_lock:
            patch_id = commit_plan(plan, self.loaded, self.repo, self.cfg.model)
        self._plans.pop(plan_id, None)
        return {"patch_id": patch_id, "head": self.repo.head(), "mode": plan.mode}


def _list(body: Mapping[str, Any], key: str, limit: int) -> list[Any]:
    value = body.get(key, [])
    if not isinstance(value, list) or len(value) > limit:
        raise StudioError(HTTPStatus.BAD_REQUEST, f"{key} must be a list of at most {limit} entries")
    return value


def _optional_layer(body: Mapping[str, Any]) -> int | None:
    value = body.get("layer")
    if value is None or value == "":
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StudioError(HTTPStatus.BAD_REQUEST, "layer must be a layer number or empty for the suggested one")
    return value


MODES_GET_ROUTES: dict[str, Any] = {
    "/api/modes/info": lambda s, q: (HTTPStatus.OK, s.modes_info()),
}
MODES_POST_ROUTES: dict[str, Any] = {
    "/api/facts/preview": lambda s, b: (HTTPStatus.ACCEPTED, {"job_id": s.start_fact_preview(b)}),
    "/api/facts/parse_csv": lambda s, b: (HTTPStatus.OK, s.parse_csv(str_field(b, "csv", 60_000))),
    "/api/facts/draft": lambda s, b: (HTTPStatus.OK, s.draft_fact(b)),
    "/api/direction/preview": lambda s, b: (HTTPStatus.ACCEPTED, {"job_id": s.start_direction_preview(b)}),
    "/api/direction/draft": lambda s, b: (HTTPStatus.OK, s.draft_direction(str_field(b, "description", 1000))),
    "/api/plans/apply": lambda s, b: (HTTPStatus.OK, s.apply_plan(str_field(b, "plan_id", 32))),
}
