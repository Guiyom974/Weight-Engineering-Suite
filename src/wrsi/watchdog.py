"""Hardware/process watchdog with checkpoint rollback for self-edited models.

``WeightWatchdog.inspect`` checks a model for states a promoted self-edit must never produce: non-finite or
exploding parameters, unstable liquid poles (LFM: ``-exp(A_log)`` must stay strictly negative and finite),
non-finite logits or exploding residual norms on a probe input, and (optionally) CUDA memory over budget.
``guard(loop)`` rolls the scientist loop back one generation when a violation is found; restoration uses the
loop's saved slices, so it is bit-exact and takes milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from libweightspace.adapters.base import ArchitectureAdapter
from libweightspace.adapters.lfm3 import LFM3Adapter
from libweightspace.prober import model_logits


@dataclass
class WatchdogReport:
    """Violations found by one inspection (empty means healthy)."""

    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no violation was found."""
        return not self.violations


class WeightWatchdog:
    """Health checks run after every promotion."""

    def __init__(
        self,
        adapter: ArchitectureAdapter | None = None,
        max_abs_weight: float = 1e4,
        min_decay_rate: float = 1e-6,
        probe_input_ids: torch.Tensor | None = None,
        max_residual_norm: float = 1e4,
        max_cuda_memory_fraction: float | None = None,
    ):
        self.adapter = adapter
        self.max_abs_weight = max_abs_weight
        self.min_decay_rate = min_decay_rate
        self.probe_input_ids = probe_input_ids
        self.max_residual_norm = max_residual_norm
        self.max_cuda_memory_fraction = max_cuda_memory_fraction

    @torch.no_grad()
    def inspect(self, model: nn.Module) -> WatchdogReport:
        """Runs every configured check."""
        report = WatchdogReport()
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                report.violations.append(f"{name}: non-finite values")
            elif float(p.detach().abs().max()) > self.max_abs_weight:
                report.violations.append(f"{name}: |w| exceeds {self.max_abs_weight}")
        if isinstance(self.adapter, LFM3Adapter):
            for i in range(self.adapter.num_layers()):
                if not self.adapter.pole_stability_ok(i, self.min_decay_rate):
                    report.violations.append(f"layer {i}: liquid pole instability")
        if self.probe_input_ids is not None:
            self._check_forward(model, self.probe_input_ids, report)
        if self.max_cuda_memory_fraction is not None and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            used = torch.cuda.memory_allocated(0) / total
            if used > self.max_cuda_memory_fraction:
                report.violations.append(
                    f"CUDA memory {used:.2%} exceeds {self.max_cuda_memory_fraction:.2%}"
                )
        return report

    def _check_forward(self, model: nn.Module, probe: torch.Tensor, report: WatchdogReport) -> None:
        norms: list[float] = []
        handles = []
        if self.adapter is not None:
            for i in range(self.adapter.num_layers()):

                def hook(_m: nn.Module, _i: Any, out: Any) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    norms.append(float(h.float().norm(dim=-1).max()))

                handles.append(self.adapter.get_layer_module(i).register_forward_hook(hook))
        try:
            device = next(model.parameters()).device
            logits = model_logits(model(input_ids=probe.to(device)))
        finally:
            for h in handles:
                h.remove()
        if not torch.isfinite(logits).all():
            report.violations.append("probe forward: non-finite logits")
        if norms and max(norms) > self.max_residual_norm:
            report.violations.append(
                f"probe forward: residual norm {max(norms):.3g} > {self.max_residual_norm}"
            )

    def guard(self, loop: Any) -> WatchdogReport:
        """Inspects ``loop.model`` and rolls back the last generation on any violation."""
        report = self.inspect(loop.model)
        if not report.ok and loop.rollback_stack:
            loop.rollback_last_generation()
        return report
