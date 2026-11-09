"""Weight-Space Studio: a hardened, same-origin local HTTP server for the guided web UI.

The Studio drives the same engine as the CLI (``neuroforge.workspace_ops``). Views:

    - Workspaces (CR2): start page with recent workspaces, a new-workspace wizard (built-in demo, Hugging Face model
      checked before download, or a model folder), the facts panel, and switching without a restart;
    - Edit: connect a frontier LLM -> describe the change (fix a failure, update facts, adjust a behaviour; CR6) ->
      diagnose -> repair -> keep, roll back or export;
    - Model factory (CR9) and Self-improvement (CR10), in their own modules.

Every number it returns is measured. Edits on real models are proposed by the frontier model chosen in step 1;
offline templates are labelled and limited to the toy models.

Security model (see the threat models of work items 004 and 005):
    - binds to 127.0.0.1 by default; ``Host`` must be the server's own host:port (blocks DNS rebinding);
    - no CORS headers at all; ``Origin``, when sent, must equal the Studio origin;
    - every POST needs ``Content-Type: application/json`` and the per-run token in ``X-Studio-Token`` (the token is
      embedded in the served page, which other origins cannot read);
    - request bodies are capped before parsing (413); one background job at a time (409);
    - responses carry a nonce-based CSP, ``nosniff``, ``frame-ancestors 'none'`` and ``no-referrer``;
    - no route accepts a filesystem path, URL or code: workspaces and model folders are chosen by id from the server's
      own lists; exports go to ``<workspace>/exports/<timestamp>-<format>``.

Main entry points: ``StudioState.from_workspace``, ``StudioState.without_workspace``, ``StudioServer``.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from pydantic import ValidationError

from libweightspace.llm_config import LLMSettings, ModelListError, fetch_models, resolve_settings
from libweightspace.spectral import profile_matrix

from .intent_compiler import draft_intent
from .schemas import IntentSpec
from .studio_common import (
    JobManager,
    StudioCore,
    StudioError,
    WorkspaceSession,
    bool_field,
    int_field,
    str_field,
    write_text_atomic,
)
from .studio_factory import FACTORY_GET_ROUTES, FACTORY_POST_ROUTES, FactoryMixin
from .studio_modes import MODES_GET_ROUTES, MODES_POST_ROUTES, ModesMixin
from .studio_selfimprove import SELFIMPROVE_GET_ROUTES, SELFIMPROVE_POST_ROUTES, SelfImproveMixin
from .templates import intent_examples
from .weight_git import WeightGitError, WeightGitRepository
from .workspace_manager import (
    HubClient,
    ModelCheck,
    StudioHome,
    WorkspaceManagerError,
    check_model,
    create_workspace,
    download_model,
    memory_probe,
    model_facts,
)
from .workspace_ops import (
    CONFIG_NAME,
    EXPORT_FORMATS,
    checkout,
    compare_answers,
    export_bundle,
    export_patch,
    history,
    open_workspace,
    probe_report,
)

__all__ = ["JobManager", "StudioError", "StudioServer", "StudioState", "MAX_BODY_BYTES", "TOKEN_HEADER"]

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024
_REF_RE = re.compile(r"^(ROOT|HEAD|main|[A-Za-z0-9_-]{4,80})$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/@-]{1,128}$")
_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
TOKEN_HEADER = "X-Studio-Token"
MAX_CHECKS = 20


def _load_ui_html() -> str:
    """The packaged single-file UI (works from a source checkout and from an installed wheel)."""
    return resources.files("neuroforge").joinpath("ui/index.html").read_text(encoding="utf-8")


class StudioState(ModesMixin, FactoryMixin, SelfImproveMixin, StudioCore):
    """Workspace session, LLM settings, model list and jobs shared by request handlers, plus every feature view."""

    def __init__(
        self,
        settings: LLMSettings,
        home: StudioHome,
        session: WorkspaceSession | None = None,
        fetch_models_on_start: bool = True,
        max_body_bytes: int = MAX_BODY_BYTES,
        hub: HubClient | None = None,
    ):
        super().__init__(settings, home, session, max_body_bytes)
        self.token = secrets.token_urlsafe(32)
        self._hub = hub
        self._checks: dict[str, ModelCheck] = {}
        self._models: dict[str, Any] = {"state": "idle", "ids": [], "error": None, "reason": None}
        if fetch_models_on_start:
            self.refresh_models(background=True)

    @classmethod
    def from_workspace(cls, workspace: Path, fetch_models_on_start: bool = True, home: StudioHome | None = None) -> StudioState:
        """Open ``workspace`` (model at HEAD) and resolve LLM settings from env and the workspace config."""
        cfg, loaded, repo = open_workspace(workspace)
        settings = resolve_settings(workspace_llm=cfg.llm)
        studio_home = home or StudioHome.resolve()
        studio_home.remember(workspace)
        session = WorkspaceSession(workspace, cfg, loaded, repo)
        return cls(settings, studio_home, session, fetch_models_on_start=fetch_models_on_start)

    @classmethod
    def without_workspace(cls, fetch_models_on_start: bool = True, home: StudioHome | None = None) -> StudioState:
        """Start on the Workspaces view (CR2): no model is loaded until the user opens or creates a workspace."""
        studio_home = home or StudioHome.resolve()
        studio_home.ensure()
        return cls(resolve_settings(), studio_home, None, fetch_models_on_start=fetch_models_on_start)

    @property
    def hub(self) -> HubClient:
        """Hub client (created on first use so a broken ``WEIGHTLAB_HF_ENDPOINT`` only affects downloads)."""
        if self._hub is None:
            self._hub = HubClient()
        return self._hub

    # --- LLM ------------------------------------------------------------------------------------------------
    def refresh_models(self, background: bool = False) -> None:
        """Fetch ``GET {base_url}/models`` now (or in a background thread at launch)."""

        def fetch() -> None:
            self._models = {"state": "loading", "ids": [], "error": None, "reason": None}
            try:
                listing = fetch_models(self.settings, reasoning_only=True)
                self._models = {
                    "state": "ready",
                    "ids": listing.ids,
                    "total": listing.total,
                    "reasoning_filter": listing.reasoning_filter,
                    "error": None,
                    "reason": None,
                    "fetched_at": time.time(),
                }
            except ModelListError as exc:
                self._models = {"state": "failed", "ids": [], "error": str(exc), "reason": exc.reason}

        if background:
            threading.Thread(target=fetch, name="studio-models", daemon=True).start()
        else:
            fetch()

    def models_payload(self) -> dict[str, Any]:
        """Current model list state plus endpoint description (never the key)."""
        return {**self._models, "endpoint": self.settings.describe(), "selected": self.selected_model}

    def select_model(self, model: str) -> dict[str, Any]:
        """Remember the chosen model (written to the open workspace's ``neuroforge.yaml``, never a key)."""
        if not _MODEL_ID_RE.fullmatch(model):
            raise StudioError(HTTPStatus.BAD_REQUEST, "invalid model id")
        ids = self._models.get("ids") or []
        if ids and model not in ids:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"{model} is not offered by {self.settings.base_url}")
        self.selected_model = model
        if self.has_workspace:
            llm = dict(self.cfg.llm or {})
            llm["model"] = model
            self.cfg = self.cfg.model_copy(update={"llm": llm})
            write_text_atomic(self.workspace / CONFIG_NAME, yaml.safe_dump(self.cfg.model_dump(), sort_keys=False))
        return {"selected": model}

    # --- workspaces (CR2) ------------------------------------------------------------------------------------
    def workspaces_payload(self) -> dict[str, Any]:
        """Known workspaces (by id), model folders, devices and the current workspace id."""
        rows = self.home.list_workspaces()
        current = None
        if self.has_workspace:
            here = str(self.workspace.resolve())
            current = next((r["id"] for r in rows if str(self.home.workspace_by_id(r["id"])) == here), None)
        return {
            "home": str(self.home.root),
            "current": current,
            "workspaces": rows,
            "model_folders": self.home.list_model_folders(),
            **memory_probe(),
        }

    def facts_payload(self) -> dict[str, Any]:
        """Facts panel for the open workspace."""
        return model_facts(self.cfg, self.loaded)

    def check_hub_model(self, repo: str, revision: str, device: str) -> dict[str, Any]:
        """Hub check before any download; the server keeps the result and returns an id to confirm it with."""
        try:
            check = check_model(self.hub, repo.strip(), revision.strip() or "main", device)
        except WorkspaceManagerError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        check_id = secrets.token_hex(8)
        self._checks[check_id] = check
        for old in list(self._checks)[:-MAX_CHECKS]:
            self._checks.pop(old, None)
        return {"check_id": check_id, **check.to_dict()}

    def _open_session(self, workspace: Path, job_id: str | None = None) -> dict[str, Any]:
        if job_id:
            self.jobs.update_progress(job_id, "loading the model", 0.9)
        cfg, loaded, repo = open_workspace(workspace)
        with self.model_lock:
            self._session = WorkspaceSession(workspace, cfg, loaded, repo)
            # Previews were measured on the previous workspace's weights; they must not survive the switch.
            if hasattr(self, "_plans"):
                self._plans.clear()
            self.settings = resolve_settings(workspace_llm=cfg.llm)
            self.selected_model = self.settings.model or self.selected_model
        self.home.remember(workspace)
        return {"workspace": workspace.name, "facts": self.facts_payload()}

    def start_open_workspace(self, ws_id: str) -> str:
        """Switch to another known workspace as a job (the session token is unchanged)."""
        self._require_idle()
        try:
            path = self.home.workspace_by_id(ws_id)
        except WorkspaceManagerError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        return self.jobs.start("open-workspace", lambda job_id: self._open_session(path, job_id))

    def start_create_workspace(self, kind: str, name: str, device: str, check_id: str = "", folder: str = "") -> str:
        """Create a workspace from the demo, a checked Hub model (download job) or a model folder, then open it."""
        self._require_idle()
        if (self.home.workspaces_dir / name).exists():
            raise StudioError(HTTPStatus.CONFLICT, f"a workspace named {name} already exists")
        hub_check: ModelCheck | None = None
        model_dir: Path | None = None
        if kind == "hub":
            hub_check = self._checks.get(check_id)
            if hub_check is None:
                raise StudioError(HTTPStatus.BAD_REQUEST, "check the model first (the check expired or is unknown)")
            if hub_check.device != device:
                raise StudioError(HTTPStatus.BAD_REQUEST, "the device differs from the checked one; check again")
        elif kind == "folder":
            try:
                model_dir = self.home.model_folder(folder)
            except WorkspaceManagerError as exc:
                raise StudioError(HTTPStatus.BAD_REQUEST, str(exc)) from None
        elif kind != "demo":
            raise StudioError(HTTPStatus.BAD_REQUEST, "kind must be demo, hub or folder")

        def job(job_id: str) -> dict[str, Any]:
            if hub_check is not None:
                downloaded = download_model(
                    self.hub, hub_check, self.home.models_dir,
                    progress=lambda text, frac: self.jobs.update_progress(job_id, text, 0.85 * frac),
                    should_cancel=lambda: self.jobs.cancel_requested(job_id),
                )
                path = create_workspace(
                    self.home, name, f"{hub_check.repo}@{hub_check.revision}", model_path=downloaded, device=device,
                    source=f"Hugging Face: {hub_check.repo}", revision=hub_check.revision,
                    license_name=hub_check.license,
                )
            elif model_dir is None:
                path = create_workspace(self.home, name, "toy:qwen3_8", device=device, demo=True)
            else:
                source = _folder_source(model_dir)
                path = create_workspace(
                    self.home, name, source.get("repo_spec") or model_dir.name, model_path=model_dir, device=device,
                    source=source.get("source") or "local model folder", revision=source.get("revision"),
                    license_name=source.get("license"),
                )
            return self._open_session(path, job_id)

        return self.jobs.start("create-workspace", job)

    # --- intent ----------------------------------------------------------------------------------------------
    @property
    def intent_path(self) -> Path:
        """Fixed location of the workspace intent."""
        return self.workspace / "intent.yaml"

    def load_intent(self) -> IntentSpec:
        """Validated intent, or ``StudioError`` (409) when none is saved yet."""
        if not self.intent_path.is_file():
            raise StudioError(HTTPStatus.CONFLICT, "no intent.yaml yet: describe the behaviour first (step 2)")
        return IntentSpec.model_validate(yaml.safe_load(self.intent_path.read_text(encoding="utf-8")))

    def intent_payload(self) -> dict[str, Any]:
        """Intent text and summary for the editor (empty when none exists)."""
        if not self.intent_path.is_file():
            return {"exists": False, "yaml": "", "summary": None}
        text = self.intent_path.read_text(encoding="utf-8")
        try:
            spec = IntentSpec.model_validate(yaml.safe_load(text))
            summary: dict[str, Any] | None = {
                "intent_id": spec.intent_id,
                "description": spec.description,
                "targets": len(spec.target_goals),
                "invariants": len(spec.invariants),
            }
            error = None
        except (ValidationError, yaml.YAMLError) as exc:
            summary, error = None, f"intent.yaml is invalid: {exc}"
        return {"exists": True, "yaml": text, "summary": summary, "error": error}

    def save_intent(self, text: str) -> dict[str, Any]:
        """Validate YAML text as an ``IntentSpec`` and store it as ``intent.yaml``."""
        try:
            spec = IntentSpec.model_validate(yaml.safe_load(text))
        except yaml.YAMLError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"not valid YAML: {exc}") from None
        except ValidationError as exc:
            raise StudioError(HTTPStatus.BAD_REQUEST, f"intent is invalid: {exc.errors()[0]['msg']}") from None
        write_text_atomic(self.intent_path, yaml.safe_dump(spec.model_dump(), sort_keys=False))
        return self.intent_payload()

    def draft_intent(self, description: str) -> dict[str, Any]:
        """Ask the selected model to draft an intent; the draft is returned for review, not saved."""
        if not description.strip():
            raise StudioError(HTTPStatus.BAD_REQUEST, "describe the behaviour you want first")
        spec = draft_intent(self._backend(), description)
        return {"yaml": yaml.safe_dump(spec.model_dump(), sort_keys=False), "drafted_by": self.selected_model}

    # --- engine ----------------------------------------------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        """Workspace, model, history head, intent and LLM summary for the header and step checks."""
        llm = {**self.settings.describe(), "selected": self.selected_model, "models_state": self._models["state"]}
        base: dict[str, Any] = {"llm": llm, "job": self.jobs.active(), "job_kind": self.jobs.active_kind()}
        if not self.has_workspace:
            return {**base, "workspace": None}
        model = self.loaded.model
        return {
            **base,
            "workspace": self.workspace.name,
            "model": self.cfg.model,
            "toy": self.is_toy,
            "family": self.loaded.adapter.family,
            "layers": self.loaded.adapter.num_layers(),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "device": self.cfg.device,
            "head": self.repo.head(),
            "commits": len(self.repo.list_commits()),
            "intent": self.intent_payload().get("summary"),
        }

    def probe(self, top: int, columns: int) -> dict[str, Any]:
        """Contrastive diagnosis on the intent's failing vs passing prompts."""
        self._require_idle()
        spec = self.load_intent()
        with self.model_lock:
            result = {"layers": probe_report(self.loaded, spec, top=top, columns=columns)}
        self._save_result("last_probe.json", result)
        return result

    def start_loop(
        self,
        iterations: int,
        candidates: int,
        use_llm: bool,
        dry_run: bool = False,
        approve_each: bool = False,
        protected: tuple[str, ...] = (),
    ) -> str:
        """Start the repair tournament as a background job and return its id.

        ``dry_run`` evaluates candidates without changing the model or the history; ``approve_each`` pauses before
        every commit until the user approves or rejects the best candidate; ``protected`` tensors (name substrings)
        reject any candidate that touches them.
        """
        source, proposer = self.repair_proposer(use_llm)  # policy first: it does not depend on the intent
        intent = self.load_intent()

        def job(job_id: str) -> dict[str, Any]:
            summary = self.run_repair(job_id, intent, source, iterations, candidates, dry_run, approve_each, protected)
            summary["proposer"] = proposer
            self._save_result("last_run.json", summary)
            return summary

        return self.jobs.start("loop", job)

    # --- persisted results (CR1) -----------------------------------------------------------------------------
    def last_results(self) -> dict[str, Any]:
        """Last saved probe and repair results, each with the HEAD it was measured at."""
        out: dict[str, Any] = {"head": self.repo.head()}
        for key, name in (("probe", "last_probe.json"), ("run", "last_run.json"), ("compare", "last_compare.json")):
            path = self.workspace / ".studio" / name
            if path.is_file():
                try:
                    out[key] = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    out[key] = None
        return out

    # --- answers before/after (CR3, CR4) ---------------------------------------------------------------------
    def compare(self) -> dict[str, Any]:
        """Original vs current greedy answers for every intent prompt, generated now."""
        self._require_idle()
        spec = self.load_intent()
        with self.model_lock:
            result = compare_answers(self.cfg, self.loaded, spec)
        self._save_result("last_compare.json", result)
        return result

    def history_payload(self) -> dict[str, Any]:
        """Commit history, newest first."""
        return {"head": self.repo.head(), "commits": history(self.repo)}

    def checkout(self, ref: str) -> dict[str, Any]:
        """Restore the model to a commit (or ROOT) bit-exactly."""
        self._require_idle()
        if not _REF_RE.fullmatch(ref):
            raise StudioError(HTTPStatus.BAD_REQUEST, "invalid commit reference")
        with self.model_lock:
            resolved, loaded = checkout(self.workspace, ref)
            self.loaded = loaded
            self.repo = WeightGitRepository(self.workspace)
        return {"head": resolved}

    def export(self, fmt: str) -> dict[str, Any]:
        """Export HEAD to ``<workspace>/exports/<timestamp>-<format>`` (the only place the Studio writes)."""
        self._require_idle()
        if fmt not in EXPORT_FORMATS and fmt != "bundle":
            raise StudioError(HTTPStatus.BAD_REQUEST, f"format must be one of {', '.join(EXPORT_FORMATS)}, bundle")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if fmt == "bundle":
            out_dir = self.workspace / "exports" / f"{stamp}-bundle"
            with self.model_lock:
                result = export_bundle(self.workspace, out_dir)
            result["path"] = str(Path("exports") / out_dir.name)
            return result
        out = self.workspace / "exports" / f"{stamp}-{fmt}"
        if fmt in ("safetensors", "sparse"):
            out = out.with_suffix(".safetensors")
        out.parent.mkdir(parents=True, exist_ok=True)
        with self.model_lock:
            result = export_patch(self.workspace, fmt, out)
        result["path"] = str(Path("exports") / out.name)
        return result

    def spectrum_payload(self, layer: int | None) -> dict[str, Any]:
        """Spectral profile (incl. HTSR alpha) of each layer's down projection."""
        params = dict(self.loaded.model.named_parameters())
        layers: list[dict[str, Any]] = []
        with self.model_lock:
            for idx in range(self.loaded.adapter.num_layers()):
                if layer is not None and idx != layer:
                    continue
                name = self.loaded.adapter.get_layer_anatomy(idx).mlp_down_proj
                if name and name in params:
                    entry = profile_matrix(params[name], tensor_name=name, top_k=16).to_dict()
                    entry["layer_idx"] = idx
                    layers.append(entry)
        return {"layers": layers}

    # --- route handlers (POST bodies are validated here) ----------------------------------------------------
    def route_models(self, query: Mapping[str, list[str]]) -> tuple[int, Mapping[str, Any]]:
        """GET /api/llm/models (``?refresh=1`` fetches the list again first)."""
        if query.get("refresh", ["0"])[0] == "1":
            self.refresh_models()
        return HTTPStatus.OK, self.models_payload()

    def route_loop(self, body: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """POST /api/loop."""
        iterations = int_field(body, "iterations", 3, 1, 10)
        candidates = int_field(body, "candidates", 3, 1, 8)
        protected = protected_field(body)
        job_id = self.start_loop(
            iterations, candidates, bool_field(body, "use_llm"),
            dry_run=bool_field(body, "dry_run"), approve_each=bool_field(body, "approve_each"), protected=protected,
        )
        return HTTPStatus.ACCEPTED, {"job_id": job_id}

    def route_create_workspace(self, body: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """POST /api/workspaces/create."""
        job_id = self.start_create_workspace(
            str_field(body, "kind", 16), str_field(body, "name", 64), str_field(body, "device", 8, "cpu"),
            check_id=str_field(body, "check_id", 32), folder=str_field(body, "folder", 220),
        )
        return HTTPStatus.ACCEPTED, {"job_id": job_id}


def _folder_source(model_dir: Path) -> dict[str, Any]:
    """Provenance recorded next to a downloaded model, if any (repo, revision, licence)."""
    path = model_dir / ".weightlab-source.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    repo, rev = data.get("repo"), data.get("revision")
    out: dict[str, Any] = {"revision": rev, "license": data.get("license")}
    if isinstance(repo, str) and isinstance(rev, str):
        out.update(repo_spec=f"{repo}@{rev}", source=f"Hugging Face: {repo}")
    return out


def protected_field(body: Mapping[str, Any]) -> tuple[str, ...]:
    """Comma-separated protected tensor names (or name parts) from ``body["protected"]``, validated."""
    raw = str_field(body, "protected", 400)
    protected = tuple(p.strip() for p in raw.split(",") if p.strip())
    if any(not re.fullmatch(r"[A-Za-z0-9_.]{1,120}", p) for p in protected):
        raise StudioError(HTTPStatus.BAD_REQUEST, "protected tensors: comma-separated names or name parts")
    return protected


# Route tables: path -> handler(state, body or query). Feature modules add their own entries in ``_routes``.
PostRoute = Callable[[StudioState, Mapping[str, Any]], tuple[int, Mapping[str, Any]]]
GetRoute = Callable[[StudioState, Mapping[str, list[str]]], tuple[int, Mapping[str, Any]]]


def _routes() -> tuple[dict[str, GetRoute], dict[str, PostRoute]]:
    get: dict[str, GetRoute] = {
        "/api/status": lambda s, q: (HTTPStatus.OK, s.status_payload()),
        "/api/llm/models": StudioState.route_models,
        "/api/intent": lambda s, q: (HTTPStatus.OK, s.intent_payload()),
        "/api/history": lambda s, q: (HTTPStatus.OK, s.history_payload()),
        "/api/last": lambda s, q: (HTTPStatus.OK, s.last_results()),
        "/api/examples": lambda s, q: (HTTPStatus.OK, {
            "examples": [{**ex, "yaml": yaml.safe_dump(ex["intent"], sort_keys=False)} for ex in intent_examples()]
        }),
        "/api/spectrum": lambda s, q: (HTTPStatus.OK, s.spectrum_payload(
            int(q["layer"][0]) if q.get("layer") else None)),
        "/api/workspaces": lambda s, q: (HTTPStatus.OK, s.workspaces_payload()),
        "/api/workspace/facts": lambda s, q: (HTTPStatus.OK, s.facts_payload()),
    }
    post: dict[str, PostRoute] = {
        "/api/llm/select": lambda s, b: (HTTPStatus.OK, s.select_model(str_field(b, "model", 128))),
        "/api/intent/draft": lambda s, b: (HTTPStatus.OK, s.draft_intent(str_field(b, "description", 4000))),
        "/api/intent/save": lambda s, b: (HTTPStatus.OK, s.save_intent(str_field(b, "yaml", s.max_body_bytes))),
        "/api/probe": lambda s, b: (HTTPStatus.OK, s.probe(int_field(b, "top", 5, 1, 64), int_field(b, "columns", 8, 1, 64))),
        "/api/loop": StudioState.route_loop,
        "/api/compare": lambda s, b: (HTTPStatus.OK, s.compare()),
        "/api/checkout": lambda s, b: (HTTPStatus.OK, s.checkout(str_field(b, "ref", 80))),
        "/api/export": lambda s, b: (HTTPStatus.OK, s.export(str_field(b, "format", 16))),
        "/api/workspaces/open": lambda s, b: (HTTPStatus.ACCEPTED, {"job_id": s.start_open_workspace(str_field(b, "id", 32))}),
        "/api/workspaces/create": StudioState.route_create_workspace,
        "/api/models/check": lambda s, b: (HTTPStatus.OK, s.check_hub_model(
            str_field(b, "repo", 200), str_field(b, "revision", 64), str_field(b, "device", 8, "cpu"))),
    }
    for extra_get, extra_post in (
        (MODES_GET_ROUTES, MODES_POST_ROUTES),
        (FACTORY_GET_ROUTES, FACTORY_POST_ROUTES),
        (SELFIMPROVE_GET_ROUTES, SELFIMPROVE_POST_ROUTES),
    ):
        overlap = (set(extra_get) & set(get)) | (set(extra_post) & set(post))
        if overlap:  # a feature module must never shadow a core route
            raise RuntimeError(f"duplicate Studio routes: {sorted(overlap)}")
        get.update(extra_get)
        post.update(extra_post)
    return get, post


GET_ROUTES, POST_ROUTES = _routes()


def _make_handler(state: StudioState, allowed_hosts: frozenset[str]) -> type[BaseHTTPRequestHandler]:
    allowed_origins = frozenset(f"http://{h}" for h in allowed_hosts)

    class StudioRequestHandler(BaseHTTPRequestHandler):
        server_version = "NeuroForgeStudio/2.0"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
            logger.debug(format, *args)

        # --- responses ---------------------------------------------------------------------------------------
        def _send(self, status: int, content_type: str, body: bytes, nonce: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            script_src = f"'nonce-{nonce}'" if nonce else "'none'"
            self.send_header(
                "Content-Security-Policy",
                f"default-src 'self'; script-src {script_src}; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

        # --- request checks ----------------------------------------------------------------------------------
        def _check_host_and_origin(self) -> None:
            if self.headers.get("Host", "") not in allowed_hosts:
                raise StudioError(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                raise StudioError(HTTPStatus.FORBIDDEN, "cross-origin requests are not allowed")

        def _read_body_bytes(self) -> bytes:
            """Read the request body within the cap. Reading first means every refusal reaches the client
            (closing a socket with unread data makes Windows reset the connection before the reply arrives)."""
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                self.close_connection = True
                raise StudioError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            raw_len = self.headers.get("Content-Length")
            try:
                length = int(raw_len) if raw_len is not None else -1
            except ValueError:
                length = -1
            if length < 0:
                self.close_connection = True
                raise StudioError(HTTPStatus.LENGTH_REQUIRED, "a valid Content-Length is required")
            if length > state.max_body_bytes:
                # Read (never parse) at most 1 MiB so the 413 reaches the client; the rest is left unread and the
                # connection is closed.
                remaining = min(length, 1024 * 1024)
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self.close_connection = True
                raise StudioError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body exceeds {state.max_body_bytes} bytes")
            return self.rfile.read(length) if length else b""

        def _require_token(self) -> None:
            """403 unless the request carries this session's token."""
            # Constant-time comparison on bytes: compare_digest rejects non-ASCII str with TypeError, and header
            # values arrive decoded as latin-1, so encoding both sides keeps a bad token a 403 instead of a 500.
            sent = self.headers.get(TOKEN_HEADER, "").encode("utf-8")
            if not hmac.compare_digest(sent, state.token.encode("ascii")):
                raise StudioError(HTTPStatus.FORBIDDEN, "missing or invalid Studio token; reload the page")

        def _parse_json(self, raw: bytes) -> dict[str, Any]:
            """Token, content type and JSON object checks, in that order."""
            self._require_token()
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                raise StudioError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
            try:
                body = json.loads((raw or b"{}").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise StudioError(HTTPStatus.BAD_REQUEST, "malformed JSON") from None
            if not isinstance(body, dict):
                raise StudioError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return body

        def _dispatch(self, fn: Callable[[], tuple[int, Mapping[str, Any]] | None]) -> None:
            try:
                self._check_host_and_origin()
                out = fn()
                if out is not None:
                    self._json(*out)
            except StudioError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (ValueError, WeightGitError, ModelListError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except ValidationError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid input: {exc.errors()[0]['msg']}"})
            except Exception:  # never leak tracebacks to the browser
                logger.exception("Unhandled Studio error on %s %s", self.command, self.path)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error (see the terminal running the Studio)"})

        # --- routes ------------------------------------------------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802 - no CORS: preflights are refused
            self._dispatch(lambda: (HTTPStatus.FORBIDDEN, {"error": "cross-origin requests are not allowed"}))

        def do_GET(self) -> None:  # noqa: N802
            route, query = urlparse_path(self.path)

            def handle() -> tuple[int, Mapping[str, Any]] | None:
                if route in ("/", "/index.html"):
                    nonce = secrets.token_urlsafe(16)
                    html = _load_ui_html().replace("__STUDIO_TOKEN__", state.token).replace("__CSP_NONCE__", nonce)
                    self._send(HTTPStatus.OK, "text/html; charset=utf-8", html.encode("utf-8"), nonce=nonce)
                    return None
                if route.startswith("/api/jobs/"):
                    return HTTPStatus.OK, state.jobs.get(route.rsplit("/", 1)[-1])
                if query.get("refresh", ["0"])[0] == "1":
                    # A refresh makes an outbound call to the LLM endpoint: another site must not trigger it.
                    self._require_token()
                handler = GET_ROUTES.get(route)
                if handler is None:
                    raise StudioError(HTTPStatus.NOT_FOUND, "route not found")
                return handler(state, query)

            self._dispatch(handle)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse_path(self.path)[0]
            try:
                raw = self._read_body_bytes()
            except StudioError as exc:
                self._json(exc.status, {"error": str(exc)})
                return

            def handle() -> tuple[int, Mapping[str, Any]]:
                body = self._parse_json(raw)
                if route.startswith("/api/jobs/"):
                    parts = route.split("/")
                    if len(parts) == 5 and parts[4] == "cancel":
                        return HTTPStatus.OK, state.jobs.request_cancel(parts[3])
                    if len(parts) == 5 and parts[4] == "decision":
                        return HTTPStatus.OK, state.jobs.decide(parts[3], bool_field(body, "approve"))
                    if len(parts) == 5 and parts[4] == "pause":
                        state.jobs.set_flag(parts[3], "pause")
                        return HTTPStatus.OK, {"pause_requested": True}
                handler = POST_ROUTES.get(route)
                if handler is None:
                    raise StudioError(HTTPStatus.NOT_FOUND, "route not found")
                return handler(state, body)

            self._dispatch(handle)

    return StudioRequestHandler


def urlparse_path(path: str) -> tuple[str, dict[str, list[str]]]:
    """``(route, query)`` of a request target."""
    parsed = urlparse(path)
    return parsed.path, parse_qs(parsed.query)


class StudioServer:
    """Context-managed local HTTP server for the Studio (loopback by default)."""

    def __init__(self, state: StudioState, host: str = "127.0.0.1", port: int = 7860):
        self.state = state
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(state, frozenset()))
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        hosts = {f"{h}:{self.port}" for h in _LOOPBACK}
        if host not in ("127.0.0.1", "localhost", "::1"):
            hosts.add(f"{host}:{self.port}")
        # Rebuild the handler now that the real (possibly ephemeral) port is known.
        self.httpd.RequestHandlerClass = _make_handler(state, frozenset(hosts))
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Address to open in the browser."""
        return f"http://127.0.0.1:{self.port}/"

    def serve_forever(self) -> None:
        """Serve until interrupted."""
        self.httpd.serve_forever()

    def start_background(self) -> None:
        """Serve from a daemon thread (tests and embedding)."""
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="studio-http", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop serving and release the socket."""
        if self._thread is not None:
            self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> StudioServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

