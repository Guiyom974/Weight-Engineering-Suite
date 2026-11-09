"""Contrastive activation tracing, causal attribution, and attractor extraction.

``MechanisticProber.trace_contrastive`` compares mean residual-stream states (last ``window`` positions) of
failure texts against passing texts at every layer. Per layer it reports the divergence norm of the residual
difference (PRD definition), the unit attractor direction and the ``down_proj`` columns most aligned with it
(the neurons that write the failure direction). Reports are sorted by divergence, most divergent first.

Deviation from the PRD reference: the attractor direction is taken from the difference of the layer's
``down_proj`` *output* (what its MLP writes), not of the whole residual stream. The residual difference also
contains token-embedding and earlier-layer differences, which bias the direction (cosine ~0.9 to a planted
attractor instead of ~1.0) and leave a large residue when the direction is nullified.

Also provided:
    - ``layer_indirect_effects``: activation patching (clean -> corrupt, last position) per layer.
    - ``capture_linear_inputs``: inputs seen by given weight matrices on a set of texts (nullspace locking).
    - ``build_diagnostics``: converts reports into the ``diagnostics`` dict handed to surgery scripts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .adapters.base import ArchitectureAdapter
from .spectral import singular_values


@dataclass
class LayerAttributionReport:
    """Per-layer contrastive attribution result."""

    layer_idx: int
    divergence_score: float
    attractor_direction: torch.Tensor  # Unit vector [d_model] in residual stream
    top_culpable_mlp_cols: list[int]  # Column indices in down_proj most aligned with attractor
    top_mlp_col_alignments: list[float]  # Cosine similarities of those columns
    singular_values_top5: list[float]
    mlp_down_proj: str | None = field(default=None)  # Extension: the attributed tensor's name


@dataclass
class CircuitEdge:
    """Subcomponent-to-subcomponent edge attribution result (WRSI-BL-03)."""

    source_node: str
    target_node: str
    source_layer: int
    target_layer: int
    tensor_name: str | None
    attribution_score: float
    top_cols: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "tensor_name": self.tensor_name,
            "attribution_score": self.attribution_score,
            "top_cols": list(self.top_cols),
        }


def _hidden(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


def model_logits(output: Any) -> torch.Tensor:
    """Extracts logits from HF outputs, toy outputs, tuples or raw tensors."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def model_device(model: nn.Module) -> torch.device:
    """Device of the model's first parameter."""
    return next(model.parameters()).device


class MechanisticProber:
    """Traces residual stream trajectories and isolates culpable weight columns."""

    def __init__(self, model: nn.Module, adapter: ArchitectureAdapter, tokenizer: Any, window: int = 8):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.model = model
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.window = window

    def _layer_means(self, texts: Sequence[str]) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
        """Mean (over texts and the last ``window`` positions) residual output of every layer, and of every
        layer's ``down_proj`` output (what the MLP writes into the residual stream)."""
        num_layers = self.adapter.num_layers()
        d_model = self.adapter.hidden_size()
        accum = [torch.zeros(d_model) for _ in range(num_layers)]
        mlp_accum: dict[int, torch.Tensor] = {}
        modules = dict(self.model.named_modules())
        down_modules: dict[int, nn.Module] = {}
        for l_idx in range(num_layers):
            name = self.adapter.get_layer_anatomy(l_idx).mlp_down_proj
            module = modules.get(name[: -len(".weight")]) if name and name.endswith(".weight") else None
            if module is not None:
                down_modules[l_idx] = module
                mlp_accum[l_idx] = torch.zeros(d_model)
        device = model_device(self.model)
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            cache: dict[int, torch.Tensor] = {}
            mlp_cache: dict[int, torch.Tensor] = {}
            handles = []
            for l_idx in range(num_layers):

                def hook(_m: nn.Module, _inp: Any, out: Any, idx: int = l_idx, store: dict = cache) -> None:
                    store[idx] = _hidden(out)[0, -self.window :, :].float().mean(dim=0).detach().cpu()

                handles.append(self.adapter.get_layer_module(l_idx).register_forward_hook(hook))
            for l_idx, module in down_modules.items():

                def mlp_hook(
                    _m: nn.Module, _inp: Any, out: Any, idx: int = l_idx, store: dict = mlp_cache
                ) -> None:
                    store[idx] = _hidden(out)[0, -self.window :, :].float().mean(dim=0).detach().cpu()

                handles.append(module.register_forward_hook(mlp_hook))
            try:
                self.model(**inputs)
            finally:
                for h in handles:
                    h.remove()
            for l_idx in range(num_layers):
                accum[l_idx] += cache[l_idx] / len(texts)
            for l_idx, value in mlp_cache.items():
                mlp_accum[l_idx] += value / len(texts)
        return accum, mlp_accum

    @torch.no_grad()
    def trace_contrastive(
        self,
        fail_texts: Sequence[str],
        pass_texts: Sequence[str],
        top_k_cols: int = 64,
    ) -> list[LayerAttributionReport]:
        """Computes contrastive residual directions (v_fail - v_pass) per layer and attributes MLP columns.

        Raises:
            ValueError: either text list is empty or ``top_k_cols`` < 1.
        """
        if not fail_texts or not pass_texts:
            raise ValueError("trace_contrastive needs at least one fail text and one pass text")
        if top_k_cols < 1:
            raise ValueError("top_k_cols must be >= 1")
        fail_means, fail_mlp = self._layer_means(fail_texts)
        pass_means, pass_mlp = self._layer_means(pass_texts)
        params = dict(self.model.named_parameters())

        reports: list[LayerAttributionReport] = []
        for l_idx in range(self.adapter.num_layers()):
            diff = fail_means[l_idx] - pass_means[l_idx]
            div_score = float(torch.norm(diff).item())
            # The attractor is what this layer's MLP writes differently on fail vs pass inputs. The residual
            # difference also carries token-embedding and earlier-layer differences, which would bias both the
            # direction and the column attribution; fall back to it only when down_proj cannot be hooked.
            write = fail_mlp[l_idx] - pass_mlp[l_idx] if l_idx in fail_mlp else diff
            v_attractor = write / (torch.norm(write) + 1e-8)

            down_name = self.adapter.get_layer_anatomy(l_idx).mlp_down_proj
            top_cols: list[int] = []
            top_aligns: list[float] = []
            sv_top5: list[float] = []
            if down_name and down_name in params:
                w_down = params[down_name].detach().float().cpu()  # [d_model, d_intermediate]
                col_norms = torch.norm(w_down, dim=0) + 1e-8
                cos_sims = (v_attractor @ w_down) / col_norms
                _, idxs = torch.topk(cos_sims.abs(), k=min(top_k_cols, w_down.shape[1]))
                top_cols = [int(i) for i in idxs.tolist()]
                top_aligns = [float(cos_sims[i].item()) for i in top_cols]
                sv_top5 = [float(s) for s in singular_values(w_down, 5).tolist()]
            reports.append(
                LayerAttributionReport(
                    layer_idx=l_idx,
                    divergence_score=div_score,
                    attractor_direction=v_attractor,
                    top_culpable_mlp_cols=top_cols,
                    top_mlp_col_alignments=top_aligns,
                    singular_values_top5=sv_top5,
                    mlp_down_proj=down_name,
                )
            )
        return sorted(reports, key=lambda r: r.divergence_score, reverse=True)

    @torch.no_grad()
    def layer_indirect_effects(self, clean_text: str, corrupt_text: str, target_token_id: int) -> list[float]:
        """Activation-patching indirect effect of each layer on ``target_token_id`` at the last position.

        For layer ``l`` the corrupt run's last-position residual after ``l`` is replaced by the clean run's; the
        effect is ``(patched - corrupt) / (clean - corrupt)`` of the target logit (1.0 = fully restores clean).
        """
        device = model_device(self.model)
        clean_in = self.tokenizer(clean_text, return_tensors="pt").to(device)
        corrupt_in = self.tokenizer(corrupt_text, return_tensors="pt").to(device)
        num_layers = self.adapter.num_layers()
        clean_cache: dict[int, torch.Tensor] = {}
        handles = []
        for l_idx in range(num_layers):

            def save(_m: nn.Module, _i: Any, out: Any, idx: int = l_idx) -> None:
                clean_cache[idx] = _hidden(out)[0, -1].detach().clone()

            handles.append(self.adapter.get_layer_module(l_idx).register_forward_hook(save))
        try:
            clean_logit = float(model_logits(self.model(**clean_in))[0, -1, target_token_id])
        finally:
            for h in handles:
                h.remove()
        corrupt_logit = float(model_logits(self.model(**corrupt_in))[0, -1, target_token_id])
        denom = clean_logit - corrupt_logit
        effects: list[float] = []
        for l_idx in range(num_layers):

            def patch(_m: nn.Module, _i: Any, out: Any, idx: int = l_idx) -> Any:
                h = _hidden(out).clone()
                h[0, -1] = clean_cache[idx].to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            handle = self.adapter.get_layer_module(l_idx).register_forward_hook(patch)
            try:
                patched = float(model_logits(self.model(**corrupt_in))[0, -1, target_token_id])
            finally:
                handle.remove()
            effects.append((patched - corrupt_logit) / denom if abs(denom) > 1e-8 else 0.0)
        return effects

    def _subcomponent_means(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Captures mean output vectors ``[d_model]`` of every layer's ``attn`` (``o_proj``) and ``mlp`` (``down_proj``)."""
        num_layers = self.adapter.num_layers()
        d_model = self.adapter.hidden_size()
        modules = dict(self.model.named_modules())
        targets: dict[str, nn.Module] = {}
        accum: dict[str, torch.Tensor] = {}
        for l_idx in range(num_layers):
            anat = self.adapter.get_layer_anatomy(l_idx)
            for label, param_name in (("attn", anat.attn_o_proj), ("mlp", anat.mlp_down_proj)):
                if param_name and param_name.endswith(".weight"):
                    mod = modules.get(param_name[: -len(".weight")])
                    if mod is not None:
                        node_key = f"L{l_idx}.{label}"
                        targets[node_key] = mod
                        accum[node_key] = torch.zeros(d_model)
        device = model_device(self.model)
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            step_cache: dict[str, torch.Tensor] = {}
            handles = []
            for node_key, mod in targets.items():

                def hook(_m: nn.Module, _inp: Any, out: Any, nk: str = node_key, store: dict = step_cache) -> None:
                    store[nk] = _hidden(out)[0, -self.window :, :].float().mean(dim=0).detach().cpu()

                handles.append(mod.register_forward_hook(hook))
            try:
                self.model(**inputs)
            finally:
                for h in handles:
                    h.remove()
            for nk, val in step_cache.items():
                accum[nk] += val / len(texts)
        return accum

    @torch.no_grad()
    def trace_circuit(
        self,
        fail_texts: Sequence[str],
        pass_texts: Sequence[str],
        top_k_edges: int = 10,
        top_k_cols: int = 8,
    ) -> list[CircuitEdge]:
        """Subcomponent / Edge Attribution Patching across ``L{i}.attn`` and ``L{i}.mlp`` writers (WRSI-BL-03).

        Computes contrastive subcomponent writes ``delta_h_u = h_u(fail) - h_u(pass)`` and scores each
        directed edge ``u -> v`` (including ``u -> output``) by the inner-product attribution of ``delta_h_u``
        along ``v``'s contrastive direction, plus the top culpable columns in ``u``'s projection tensor.
        """
        if not fail_texts or not pass_texts:
            raise ValueError("trace_circuit needs at least one fail text and one pass text")
        fail_sub = self._subcomponent_means(fail_texts)
        pass_sub = self._subcomponent_means(pass_texts)
        fail_res, _ = self._layer_means(fail_texts)
        pass_res, _ = self._layer_means(pass_texts)
        params = dict(self.model.named_parameters())
        num_layers = self.adapter.num_layers()

        final_diff = fail_res[-1] - pass_res[-1] if fail_res else torch.zeros(self.adapter.hidden_size())
        final_dir = final_diff / (torch.norm(final_diff) + 1e-8)

        node_diffs: dict[str, tuple[int, str | None, torch.Tensor]] = {}
        for l_idx in range(num_layers):
            anat = self.adapter.get_layer_anatomy(l_idx)
            for label, pname in (("attn", anat.attn_o_proj), ("mlp", anat.mlp_down_proj)):
                nk = f"L{l_idx}.{label}"
                if nk in fail_sub:
                    node_diffs[nk] = (l_idx, pname, fail_sub[nk] - pass_sub[nk])

        edges: list[CircuitEdge] = []
        for src_node, (src_layer, pname, src_vec) in node_diffs.items():
            src_norm = float(torch.norm(src_vec).item())
            if src_norm <= 1e-12:
                continue
            src_unit = src_vec / src_norm
            cols: list[int] = []
            if pname and pname in params and params[pname].ndim == 2:
                w = params[pname].detach().float().cpu()
                c_norms = torch.norm(w, dim=0) + 1e-8
                sims = (src_unit @ w) / c_norms
                _, idxs = torch.topk(sims.abs(), k=min(top_k_cols, w.shape[1]))
                cols = [int(i) for i in idxs.tolist()]

            # Direct contribution edge to final residual output.
            out_score = float(torch.dot(src_vec, final_dir).item())
            edges.append(
                CircuitEdge(
                    source_node=src_node,
                    target_node="output",
                    source_layer=src_layer,
                    target_layer=num_layers,
                    tensor_name=pname,
                    attribution_score=out_score,
                    top_cols=cols,
                )
            )
            # Edges to downstream subcomponents in later layers.
            for dst_node, (dst_layer, _, dst_vec) in node_diffs.items():
                if dst_layer <= src_layer:
                    continue
                dst_dir = dst_vec / (torch.norm(dst_vec) + 1e-8)
                score = float(torch.dot(src_vec, dst_dir).item())
                edges.append(
                    CircuitEdge(
                        source_node=src_node,
                        target_node=dst_node,
                        source_layer=src_layer,
                        target_layer=dst_layer,
                        tensor_name=pname,
                        attribution_score=score,
                        top_cols=cols,
                    )
                )

        edges.sort(key=lambda e: abs(e.attribution_score), reverse=True)
        return edges[: max(1, top_k_edges)]


@torch.no_grad()
def capture_linear_inputs(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    param_names: Sequence[str],
    max_rows: int = 4096,
) -> dict[str, torch.Tensor]:
    """Records the inputs ``[N_tokens, d_in]`` that the modules owning ``param_names`` receive on ``texts``.

    Only 2-D ``*.weight`` parameters of modules with a forward pass are captured; others are skipped.
    """
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []
    for name in param_names:
        if not name.endswith(".weight") or name not in params or params[name].ndim != 2:
            continue
        module = modules.get(name[: -len(".weight")])
        if not isinstance(module, nn.Linear):  # e.g. an embedding's input is token ids, not activations
            continue
        captured[name] = []

        def pre_hook(_m: nn.Module, inputs: Any, key: str = name) -> None:
            x = inputs[0]
            captured[key].append(x.reshape(-1, x.shape[-1]).detach().float().cpu())

        handles.append(module.register_forward_pre_hook(pre_hook))
    device = model_device(model)
    try:
        for text in texts:
            model(**tokenizer(text, return_tensors="pt").to(device))
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, dim=0)[-max_rows:] for k, v in captured.items() if v}


def build_diagnostics(reports: Sequence[LayerAttributionReport], top_n: int = 5) -> dict[str, Any]:
    """Diagnostics dict for surgery scripts: the most divergent layer's fields plus a ``layers`` summary.

    Raises:
        ValueError: ``reports`` is empty.
    """
    if not reports:
        raise ValueError("no attribution reports")
    top = reports[0]
    return {
        "layer_idx": top.layer_idx,
        "target_tensor": top.mlp_down_proj,
        "divergence_score": top.divergence_score,
        "attractor_direction": top.attractor_direction.clone(),
        "top_culpable_mlp_cols": list(top.top_culpable_mlp_cols),
        "top_mlp_col_alignments": list(top.top_mlp_col_alignments),
        "singular_values_top5": list(top.singular_values_top5),
        "layers": [
            {
                "layer_idx": r.layer_idx,
                "target_tensor": r.mlp_down_proj,
                "divergence_score": r.divergence_score,
                "attractor_direction": r.attractor_direction.clone(),
                "top_culpable_mlp_cols": list(r.top_culpable_mlp_cols),
                "top_mlp_col_alignments": list(r.top_mlp_col_alignments),
                "singular_values_top5": list(r.singular_values_top5),
            }
            for r in reports[:top_n]
        ],
    }
