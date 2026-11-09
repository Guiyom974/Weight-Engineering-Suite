"""W-RSI command line: ``wrsi start/status/rollback/telemetry``.

``wrsi start`` runs curriculum steps against the model, with the anchor vault as a real child process. All
state lives in ``<workspace>/.wrsi`` (see ``wrsi.workspace``), so runs resume where they stopped and
``wrsi rollback`` undoes generations bit-exactly in a later process. Offline demo:

    wrsi start -w run --generations 6
    wrsi status -w run
    wrsi rollback -w run

Non-toy models need ``--domains FILE`` (YAML list of ``{name, failure_taxonomy, probes: [ProbePair...]}``).
Expected failures print one line on stderr and exit with code 1.
"""

from __future__ import annotations

import functools
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from libweightspace.candidates import CandidateSource, TemplateCandidateSource
from libweightspace.llm_backends import InProcessModelBackend, LLMBackendError, make_backend
from libweightspace.loading import LoadedModel, load_model

from .anatomy_atlas import SelfAnatomyAtlas
from .anchor_client import AnchorVaultError
from .runner import Guardrails, SessionInputs, forgetting_pairs, run_session
from .runner import rollback as run_rollback
from .runner import status as run_status
from .schemas import GenerationState
from .scientist_agent import GymTemplateProposer, InWeightSelfEditProposer, frontier_proposer
from .workspace import WRSIWorkspace

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    help="W-RSI: autonomous weight-space recursive self-improvement.",
)
console = Console()
error_console = Console(stderr=True)
EXPECTED_ERRORS = (
    ValueError,
    KeyError,
    OSError,
    sqlite3.Error,
    RuntimeError,
    ValidationError,
    LLMBackendError,
    AnchorVaultError,
    yaml.YAMLError,
)
F = TypeVar("F", bound=Callable[..., Any])
WorkspaceOpt = typer.Option(Path("."), "--workspace", "-w", help="Workspace directory.")

def cli_errors(fn: F) -> F:
    """Turns expected exceptions into a one-line message and exit code 1."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except typer.Exit:  # subclasses RuntimeError; a deliberate exit is not an error
            raise
        except EXPECTED_ERRORS as exc:
            first_line = (str(exc).splitlines() or [type(exc).__name__])[0]
            error_console.print(f"error: {first_line}", markup=False)
            raise typer.Exit(code=1) from exc

    return wrapper  # type: ignore[return-value]


def _load_domain_specs(path: Path | None, is_toy: bool) -> list[dict[str, Any]] | None:
    if path is None:
        if not is_toy:
            raise ValueError("non-toy models need --domains FILE")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"domains file not found: {path}")
    specs = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("domains file must be a non-empty YAML list")
    return specs


def _load_anchors(path: Path | None, is_toy: bool) -> list[str] | None:
    if path is None:
        if not is_toy:
            raise ValueError("non-toy models need --anchors FILE (one holdout prompt per line)")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"anchors file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("anchors file is empty")
    return lines


def _open_model(ws: WRSIWorkspace, state: GenerationState) -> LoadedModel:
    return load_model(state.model, demo_defect=state.demo_defect)


def _proposer_factory(
    proposer: str, proposer_model: str | None, base_url: str | None, key_env: str | None
) -> Callable[[LoadedModel], tuple[CandidateSource, CandidateSource]]:
    def build(loaded: LoadedModel) -> tuple[CandidateSource, CandidateSource]:
        if proposer == "template":
            return TemplateCandidateSource(), GymTemplateProposer()
        if proposer == "self":
            inweight = InWeightSelfEditProposer(
                loaded.model, loaded.adapter, loaded.tokenizer,
                backend=InProcessModelBackend(loaded.model, loaded.tokenizer, max_new_tokens=32),
            )
            return inweight, inweight
        llm = frontier_proposer(make_backend(proposer, proposer_model, base_url, api_key_env=key_env), f"llm:{proposer_model}")
        return llm, llm

    return build


@app.command()
@cli_errors
def start(
    workspace: Path = WorkspaceOpt,
    model: str = typer.Option("toy:qwen3_8", help="Model spec (first start only)."),
    demo_defect: bool = typer.Option(
        True, help="Plant the demo attractor in a toy model (first start only)."
    ),
    generations: int = typer.Option(5, min=1, max=100_000, help="Curriculum steps to run."),
    proposer: str = typer.Option(
        "template", help="template (offline), self (the model edits itself) or llm (any OpenAI-compatible endpoint)."
    ),
    proposer_model: str | None = typer.Option(
        None, "--proposer-model", help="LLM model id for --proposer llm (see `neuroforge models`)."
    ),
    base_url: str | None = typer.Option(None, help="OpenAI-compatible endpoint (default: Poe, https://api.poe.com/v1)."),
    key_env: str | None = typer.Option(None, "--key-env", help="Name of the environment variable holding the API key."),
    domains: Path | None = typer.Option(None, help="YAML task domains (required for non-toy models)."),
    anchors: Path | None = typer.Option(
        None, help="Holdout prompts, one per line (required for non-toy models; first start only)."
    ),
    meta_fraction: float = typer.Option(0.30, min=0.0, max=1.0, help="Share of second-order steps."),
    max_kl: float = typer.Option(0.08, help="KL tether to M_0."),
    max_sparsity: float | None = typer.Option(None, help="Modified-parameter budget per edit."),
    islands: int = typer.Option(1, "--islands", min=1, max=16, help="Number of speciation islands."),
    protect: str = typer.Option("", "--protect", help="Comma-separated tensor names (or parts) no edit may touch."),
    forgetting: Path | None = typer.Option(
        None, "--forgetting", help="Held-out 'prompt => expected answer' lines; a candidate that makes them less likely is not promoted."
    ),
    max_forgetting_drop: float = typer.Option(0.05, min=0.0, help="Allowed drop of the held-out score (nats)."),
    seed: int = typer.Option(0),
) -> None:
    """Runs self-improvement steps and persists the result."""
    ws = WRSIWorkspace(workspace)
    is_toy = (ws.load_state().model if ws.exists() else model).startswith("toy:")
    if forgetting is not None and not forgetting.is_file():
        raise FileNotFoundError(f"forgetting file not found: {forgetting}")
    pairs = forgetting_pairs(forgetting.read_text(encoding="utf-8").splitlines()) if forgetting else []
    guard = Guardrails(
        max_kl=max_kl, max_sparsity=max_sparsity, protected_tensors=tuple(p.strip() for p in protect.split(",") if p.strip()),
        forgetting=pairs, max_forgetting_drop=max_forgetting_drop, meta_fraction=meta_fraction, islands=islands, seed=seed,
    )
    inputs = SessionInputs(model=model, demo_defect=demo_defect, domains=_load_domain_specs(domains, is_toy),
                           anchors=_load_anchors(anchors, is_toy))
    result = run_session(workspace, inputs, guard, _proposer_factory(proposer, proposer_model, base_url, key_env),
                         generations, proposer_label=proposer)
    table = Table(title="W-RSI run")
    for name in ("step", "kind", "domain", "candidates", "promoted", "generation", "hit rate", "note"):
        table.add_column(name, overflow="fold")
    for r in result["reports"]:
        table.add_row(str(r["step"]), r["kind"], r["domain"], str(r["candidates"]), str(r["promoted"]),
                      str(r["generation"]), f"{r['surgical_hit_rate']:.3f}", r["note"])
    console.print(table)
    console.print(f"generation={result['generation']} hit_rate={result['hit_rate']:.3f}", markup=False)


@app.command()
@cli_errors
def status(
    workspace: Path = WorkspaceOpt,
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Shows generation, hit rate and ledger totals."""
    console.print(json.dumps(run_status(workspace), indent=None if as_json else 2), markup=False, soft_wrap=True)


@app.command()
@cli_errors
def rollback(
    workspace: Path = WorkspaceOpt,
    steps: int = typer.Option(1, min=1, help="Generations to undo."),
) -> None:
    """Undoes promoted generations bit-exactly (weights and nullspace locks)."""
    result = run_rollback(workspace, steps)
    console.print(f"rolled back generations {result['undone']}; now at generation {result['generation']}", markup=False)


@app.command()
@cli_errors
def telemetry(
    workspace: Path = WorkspaceOpt,
    limit: int = typer.Option(20, min=1, max=1000),
    atlas: bool = typer.Option(False, help="Show the self-anatomy atlas instead of trials."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Shows recent trials, or the spectral anatomy of the current model."""
    ws = WRSIWorkspace(workspace)
    if not ws.exists():
        raise FileNotFoundError(f"no W-RSI run in {workspace} (run `wrsi start`)")
    if atlas:
        loaded = _open_model(ws, ws.load_state())
        ws.apply_overlay(loaded.model)
        builder = SelfAnatomyAtlas(loaded.model, loaded.adapter)
        rows: list[dict[str, Any]] = builder.to_rows(builder.build())
    else:
        rows = [
            {
                "trial": t.trial_id,
                "gen": t.generation,
                "domain": t.domain,
                "promoted": t.promoted,
                "gain": round(t.target_score_delta, 4),
                "kl": round(t.holdout_kl_divergence, 5),
                "reason": (t.rejection_reason or "")[:60],
            }
            for t in ws.ledger().list_trials(limit=limit)
        ]
    if as_json:
        console.print(json.dumps(rows), markup=False, soft_wrap=True)
        return
    table = Table(title="Self-anatomy atlas" if atlas else "Recent trials")
    for name in rows[0] if rows else ["(empty)"]:
        table.add_column(str(name), overflow="fold")
    for row in rows:
        table.add_row(*[str(v) for v in row.values()])
    console.print(table)


if __name__ == "__main__":
    app()

