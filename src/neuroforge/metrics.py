"""Evaluation suites behind ``MetricGoal`` / ``InvariantGuard`` contracts.

A suite is any object with ``name`` and ``evaluate(model) -> dict[metric_name, float]``:
    - ``ProbeSuite``: ``logit_margin`` (log p(expected first token) - log p(undesired first token) at the end of
      the prompt), ``exact_match`` (greedy continuation equals the expected prefix tokens), ``repetition_rate``
      (1 - unique/total over a greedy continuation).
    - ``KLDivergenceSuite``: mean KL(P_ref || P_model) over every position of fixed prompts, with the reference
      log-probs captured once at construction (typically from the unedited model).
    - ``CallableSuite``: wraps a user function, e.g. a code-execution ``pass_at_1`` harness.

``metric_drop`` defines degradation direction: lower-is-better metrics (``kl_divergence``, ``repetition_rate``)
degrade when they rise; all others degrade when they fall.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from libweightspace.isolated_sandbox import execute_code_pass_at_1_trial
from libweightspace.prober import model_device, model_logits

from .schemas import ProbePair

LOWER_IS_BETTER = frozenset({"kl_divergence", "repetition_rate"})


class EvaluationSuite(Protocol):
    """Named metric producer."""

    name: str

    def evaluate(self, model: nn.Module) -> dict[str, float]:
        """Returns metric_name -> value."""
        ...


def metric_drop(metric_name: str, before: float, after: float) -> float:
    """Positive when ``after`` is worse than ``before``."""
    return after - before if metric_name in LOWER_IS_BETTER else before - after


def encode_continuation(tokenizer: Any, text: str) -> list[int]:
    """Token ids of ``text`` without special tokens (HF tokenizers) or plain ``encode`` (char tokenizer)."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


@torch.no_grad()
def greedy_continuation(model: nn.Module, tokenizer: Any, prompt: str, n_tokens: int) -> list[int]:
    """Greedy-decodes ``n_tokens`` ids after ``prompt`` (full re-forward per step; fine for probes)."""
    device = model_device(model)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    out: list[int] = []
    for _ in range(n_tokens):
        nxt = int(model_logits(model(input_ids=ids))[0, -1].argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return out


class ProbeSuite:
    """Metrics computed on ``ProbePair`` prompts."""

    def __init__(self, name: str, probes: Sequence[ProbePair], tokenizer: Any, repetition_tokens: int = 8):
        if not probes:
            raise ValueError(f"suite '{name}' needs at least one probe")
        self.name = name
        self.probes = list(probes)
        self.tokenizer = tokenizer
        self.repetition_tokens = repetition_tokens

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> dict[str, float]:
        """Computes ``logit_margin``, ``exact_match`` and ``repetition_rate`` averaged over probes."""
        device = model_device(model)
        margins: list[float] = []
        matches: list[float] = []
        repetition: list[float] = []
        for probe in self.probes:
            ids = self.tokenizer(probe.prompt, return_tensors="pt")["input_ids"].to(device)
            logp = F.log_softmax(model_logits(model(input_ids=ids))[0, -1].float(), dim=-1)
            expected = encode_continuation(self.tokenizer, probe.expected_completion_prefix)
            if probe.undesired_completion_prefix:
                undesired = encode_continuation(self.tokenizer, probe.undesired_completion_prefix)
                if expected and undesired:
                    margins.append(float(logp[expected[0]] - logp[undesired[0]]))
            if expected:
                matches.append(
                    float(greedy_continuation(model, self.tokenizer, probe.prompt, len(expected)) == expected)
                )
            cont = greedy_continuation(model, self.tokenizer, probe.prompt, self.repetition_tokens)
            repetition.append(1.0 - len(set(cont)) / len(cont) if cont else 0.0)
        result = {
            "exact_match": sum(matches) / len(matches) if matches else 0.0,
            "repetition_rate": sum(repetition) / len(repetition),
        }
        if margins:
            result["logit_margin"] = sum(margins) / len(margins)
        return result


class KLDivergenceSuite:
    """Mean forward KL divergence to reference next-token distributions on fixed prompts."""

    def __init__(self, name: str, prompts: Sequence[str], tokenizer: Any, reference_model: nn.Module):
        if not prompts:
            raise ValueError(f"suite '{name}' needs at least one prompt")
        self.name = name
        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.reference = [self._log_probs(reference_model, p) for p in self.prompts]

    @torch.no_grad()
    def _log_probs(self, model: nn.Module, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(model_device(model))
        return F.log_softmax(model_logits(model(input_ids=ids))[0].float(), dim=-1).cpu()

    def evaluate(self, model: nn.Module) -> dict[str, float]:
        """``kl_divergence`` = mean over prompts of the per-position mean KL(P_ref || P_model)."""
        total = 0.0
        for prompt, ref in zip(self.prompts, self.reference, strict=True):
            cand = self._log_probs(model, prompt)
            total += float(F.kl_div(cand, ref, reduction="batchmean", log_target=True))
        return {"kl_divergence": total / len(self.prompts)}


class CallableSuite:
    """Suite backed by ``fn(model) -> dict[str, float]``."""

    def __init__(self, name: str, fn: Callable[[nn.Module], dict[str, float]]):
        self.name = name
        self.fn = fn

    def evaluate(self, model: nn.Module) -> dict[str, float]:
        """Delegates to the wrapped function."""
        return dict(self.fn(model))


class CodePassAt1Suite:
    """Sandboxed code-execution ``pass_at_1`` evaluation suite (WRSI-BL-02).

    Evaluates each ``ProbePair`` by generating a greedy continuation (or using ``probe.metadata["candidate_code"]``
    when provided) and running ``probe.metadata.get("test_code", "assert True")`` inside an isolated child
    Python process via ``execute_code_pass_at_1_trial``. Also computes ``ProbeSuite`` metrics so mixed goals on
    the same suite name are supported.
    """

    def __init__(
        self,
        name: str,
        probes: Sequence[ProbePair],
        tokenizer: Any,
        max_new_tokens: int = 32,
        timeout: float = 5.0,
    ):
        if not probes:
            raise ValueError(f"suite '{name}' needs at least one probe")
        self.name = name
        self.probes = list(probes)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self._probe_suite = ProbeSuite(name, probes, tokenizer)

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> dict[str, float]:
        """Returns ``pass_at_1`` plus standard probe metrics."""
        base_metrics = self._probe_suite.evaluate(model)
        passed: list[float] = []
        for probe in self.probes:
            test_code = probe.metadata.get("test_code")
            if isinstance(test_code, str) and test_code.strip():
                override_code = probe.metadata.get("candidate_code")
                if isinstance(override_code, str):
                    candidate_src = override_code
                else:
                    expected_ids = encode_continuation(self.tokenizer, probe.expected_completion_prefix)
                    n_toks = len(expected_ids) if expected_ids else self.max_new_tokens
                    cont_ids = greedy_continuation(model, self.tokenizer, probe.prompt, n_toks)
                    decode_fn = getattr(self.tokenizer, "decode", None)
                    completion = (
                        decode_fn(cont_ids)
                        if callable(decode_fn)
                        else "".join(chr(c % 128) for c in cont_ids)
                    )
                    candidate_src = probe.prompt + completion
                ok = execute_code_pass_at_1_trial(candidate_src, test_code, timeout=self.timeout)
                passed.append(1.0 if ok else 0.0)
            else:
                expected = encode_continuation(self.tokenizer, probe.expected_completion_prefix)
                if expected:
                    cont = greedy_continuation(model, self.tokenizer, probe.prompt, len(expected))
                    passed.append(1.0 if cont == expected else 0.0)
                else:
                    passed.append(0.0)
        base_metrics["pass_at_1"] = sum(passed) / len(passed) if passed else 0.0
        return base_metrics
