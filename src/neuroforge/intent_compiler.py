"""Intent compiler: turns an ``IntentSpec`` into contrastive probe texts and evaluation-suite contracts.

Conventions (within the PRD schema):
    - A ``ProbePair`` with ``metadata["role"] == "invariant"`` is an invariant prompt (baseline behaviour to
      preserve), not a contrastive probe. Its ``prompt`` feeds KL invariants and nullspace locking.
    - Contrastive texts: ``fail = prompt + undesired_completion_prefix`` and ``pass = prompt + expected...``.
      Probes without an undesired completion contribute to target suites but not to contrastive tracing.
    - Target goals map to ``ProbeSuite``s over the contrastive probes, named by ``goal.suite_name``. Invariant
      guards on ``kl_divergence`` map to ``KLDivergenceSuite``s over the invariant prompts; other invariant
      metrics map to ``ProbeSuite``s over the invariant probes.

``draft_intent`` asks a reasoner to turn a natural-language description into an ``IntentSpec`` (validated by
Pydantic; the LLM output is never trusted unvalidated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from libweightspace.candidates import extract_json
from libweightspace.llm_backends import LLMBackend

from .metrics import CodePassAt1Suite, EvaluationSuite, KLDivergenceSuite, ProbeSuite
from .schemas import IntentSpec, NeuralBlueprint, ProbePair

MAX_SPEC_BYTES = 1_000_000
PROBE_METRICS = frozenset({"logit_margin", "exact_match", "repetition_rate", "pass_at_1"})


def _load_yaml(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise ValueError(f"{path} exceeds {MAX_SPEC_BYTES} bytes")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_intent(path: Path) -> IntentSpec:
    """Loads and validates an intent YAML/JSON file."""
    return IntentSpec.model_validate(_load_yaml(path))


def load_blueprint(path: Path) -> NeuralBlueprint:
    """Loads and validates a blueprint YAML/JSON file."""
    return NeuralBlueprint.model_validate(_load_yaml(path))


def is_invariant(probe: ProbePair) -> bool:
    """True for probes tagged ``metadata.role = invariant``."""
    return probe.metadata.get("role") == "invariant"


def contrastive_probes(intent: IntentSpec) -> list[ProbePair]:
    """Probes that express the target behaviour (everything not tagged invariant)."""
    return [p for p in intent.contrastive_probes if not is_invariant(p)]


def invariant_prompts(intent: IntentSpec) -> list[str]:
    """Prompts whose baseline behaviour must be preserved."""
    return [p.prompt for p in intent.contrastive_probes if is_invariant(p)]


def contrastive_texts(intent: IntentSpec) -> tuple[list[str], list[str]]:
    """``(fail_texts, pass_texts)`` for the prober.

    Raises:
        ValueError: no probe defines an undesired completion.
    """
    fail, passing = [], []
    for p in contrastive_probes(intent):
        if p.undesired_completion_prefix:
            fail.append(p.prompt + p.undesired_completion_prefix)
            passing.append(p.prompt + p.expected_completion_prefix)
    if not fail:
        raise ValueError(f"intent '{intent.intent_id}' has no probe with an undesired_completion_prefix")
    return fail, passing


def compile_suites(
    intent: IntentSpec, tokenizer: Any, reference_model: nn.Module
) -> tuple[dict[str, EvaluationSuite], dict[str, EvaluationSuite]]:
    """Builds ``(target_suites, invariant_suites)`` keyed by suite name.

    ``reference_model`` is snapshotted immediately (KL reference log-probs), so pass the unedited model.

    Raises:
        ValueError: a guard needs invariant prompts but the intent has none.
    """
    targets: dict[str, EvaluationSuite] = {}
    probes = contrastive_probes(intent)
    for goal in intent.target_goals:
        if goal.metric_name not in PROBE_METRICS:
            raise ValueError(
                f"target metric '{goal.metric_name}' needs a custom suite (see neuroforge.metrics.CallableSuite)"
            )
        if goal.metric_name == "pass_at_1":
            if not any(isinstance(p.metadata.get("test_code"), str) for p in probes):
                raise ValueError(
                    "target metric 'pass_at_1' needs probes with metadata.test_code or a custom suite "
                    "(see neuroforge.metrics.CallableSuite)"
                )
            targets[goal.suite_name] = CodePassAt1Suite(goal.suite_name, probes, tokenizer)
        elif goal.suite_name not in targets:
            targets[goal.suite_name] = ProbeSuite(goal.suite_name, probes, tokenizer)
    invariants: dict[str, EvaluationSuite] = {}
    inv_probes = [p for p in intent.contrastive_probes if is_invariant(p)]
    for guard in intent.invariants:
        if guard.metric_name != "kl_divergence" and guard.metric_name not in PROBE_METRICS:
            raise ValueError(
                f"invariant metric '{guard.metric_name}' needs a custom suite (see neuroforge.metrics.CallableSuite)"
            )
        if guard.suite_name in invariants:
            if isinstance(invariants[guard.suite_name], KLDivergenceSuite) != (
                guard.metric_name == "kl_divergence"
            ):
                raise ValueError(f"suite '{guard.suite_name}' cannot mix KL and probe metrics")
            continue
        if not inv_probes:
            raise ValueError(f"invariant '{guard.suite_name}' needs probes tagged metadata.role=invariant")
        if guard.metric_name == "kl_divergence":
            invariants[guard.suite_name] = KLDivergenceSuite(
                guard.suite_name, [p.prompt for p in inv_probes], tokenizer, reference_model
            )
        else:
            invariants[guard.suite_name] = ProbeSuite(guard.suite_name, inv_probes, tokenizer)
    return targets, invariants


DRAFT_PROMPT = """Convert the behaviour change below into a NeuroForge IntentSpec JSON object with keys:
intent_id, description, target_behavior, failure_behavior,
contrastive_probes: [{{probe_id, prompt, expected_completion_prefix, undesired_completion_prefix, metadata}}],
target_goals: [{{suite_name, metric_name (exact_match|pass_at_1|repetition_rate|kl_divergence|logit_margin),
operator (>=|<=|==), threshold}}], invariants: [{{suite_name, metric_name, max_allowed_drop (0 < x <= 0.1)}}].
Tag 3-8 probes that capture behaviour to preserve with metadata {{"role": "invariant"}}.
Reply with the JSON object only.

Behaviour change: {description}
"""


def draft_intent(backend: LLMBackend, description: str) -> IntentSpec:
    """Asks a reasoner to draft an intent; the reply is parsed and validated.

    Raises:
        CandidateParseError / pydantic.ValidationError: unusable reply.
    """
    reply = backend.complete(
        "You write precise, testable model-editing specifications.",
        DRAFT_PROMPT.format(description=description.strip()[:4000]),
    )
    return IntentSpec.model_validate(extract_json(reply))
