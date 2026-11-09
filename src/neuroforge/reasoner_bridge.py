"""Reasoner bridge: prompt protocol between NeuroForge and an external frontier reasoning model.

``build_reasoner_prompt`` assembles the four PRD sections (PRD_01 section 4.6): model topology, mechanistic
attribution table (top-5 layers by divergence with singular values and culpable columns), previous iteration
history, and the output contract. ``ReasonerBridge`` sends it with ``NEUROFORGE_SYSTEM_PROMPT`` through any
``LLMBackend`` (Gemini CLI, OpenAI-compatible, Anthropic, callable) and parses candidates. Replies are untrusted;
scripts are only ever executed by the sandbox.
"""

from __future__ import annotations

import json
from typing import Any

from libweightspace.candidates import CandidatePatch, ProposalContext, parse_candidates_json
from libweightspace.llm_backends import LLMBackend

NEUROFORGE_SYSTEM_PROMPT = """You are the NeuroForge Weight-Space Programmer. You directly read, diagnose, and surgically edit the numerical weight tensors of neural networks without gradient descent or fine-tuning.

### Mathematical Principles You Must Follow:
1. **Surgical Minimality**: Never modify more than 0.05% of total model parameters. Prefer modifying specific columns of `mlp.down_proj.weight` or specific rows of MoE router gates.
2. **Directional Nullification (Breaking Attractor Loops)**:
   Given a unit attractor vector `v = diagnostics["attractor_direction"]` of shape `[d_model]` and culpable column indices `cols = diagnostics["top_culpable_mlp_cols"]` in `W = weights[target_name]` of shape `[d_model, d_inter]`:
   Subtract the projection onto `v` from only those columns:
   ```python
   def compute_delta(weights, diagnostics):
       W = weights["model.layers.15.mlp.down_proj.weight"]
       v = diagnostics["attractor_direction"].to(W.device, W.dtype)
       v = v / (torch.norm(v) + 1e-8)
       cols = diagnostics["top_culpable_mlp_cols"][:64]
       delta = torch.zeros_like(W)
       for c in cols:
           proj = torch.dot(W[:, c], v) * v
           delta[:, c] = -1.0 * proj
       return {"model.layers.15.mlp.down_proj.weight": delta}
   ```
3. **Associative Memory Insertion (Adding Rules)**:
   To inject a mapping from hidden key direction `k` (`[d_inter]`) to residual value vector `u` (`[d_model]`):
   $$\\Delta W_{\\text{down}} = \\alpha \\frac{u k^T}{\\|k\\|_2^2}$$
4. **Strict Sandbox Compliance**:
   - Do NOT use `import`, `open`, `exec`, `eval`, or dunder attributes (`__`).
   - Always define `def compute_delta(weights, diagnostics):` returning a `dict[str, torch.Tensor]`.
   - Ensure zero `NaN` or `Inf` values by adding `1e-8` to all denominators.
"""

OUTPUT_CONTRACT = """### 4. Output Contract
Reply with ONE JSON block and nothing else:
```json
{{"candidates": [{{"rationale": "<mechanistic hypothesis>", "failure_taxonomy": "attractor_loop", "python_script": "def compute_delta(weights, diagnostics):\\n    ..."}}]}}
```
Propose exactly {n} distinct candidates. Scripts may read `diagnostics` keys: layer_idx, target_tensor,
divergence_score, attractor_direction (tensor [d_model]), top_culpable_mlp_cols, top_mlp_col_alignments,
singular_values_top5, layers (list of per-layer dicts with the same keys). Prefer `diagnostics["target_tensor"]`
over hard-coded names. Allowed builtins: range len min max abs sum enumerate zip int float bool round list dict
tuple sorted reversed any all. No try/with/class/import/getattr; no private or dunder attributes. `torch` exposes tensor math only
(creation, arithmetic, reductions, torch.linalg, torch.nn.functional); tensor methods work as usual.
"""


def _fmt_floats(values: list[float], n: int) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in values[:n]) + "]"


def build_reasoner_prompt(context: ProposalContext) -> str:
    """Renders the four-section reasoner prompt for one iteration."""
    topo = context.topology
    lines = [
        f"## Intent\n{context.intent_description or '(unspecified)'}",
        "",
        "### 1. Model Topology Summary",
        f"architecture={topo.get('architecture')} layers={topo.get('num_layers')} d_model={topo.get('d_model')} "
        f"d_intermediate={topo.get('d_intermediate')} total_params={topo.get('total_params')}",
        "",
        "### 2. Mechanistic Attribution Table (top layers by divergence)",
        "| layer | divergence | target tensor | singular_values_top5 | top culpable cols | cosine alignments |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in context.diagnostics.get("layers", [])[:5]:
        lines.append(
            f"| {row.get('layer_idx')} | {row.get('divergence_score', 0.0):.4f} | {row.get('target_tensor')} | "
            f"{_fmt_floats(row.get('singular_values_top5', []), 5)} | {row.get('top_culpable_mlp_cols', [])[:16]} | "
            f"{_fmt_floats(row.get('top_mlp_col_alignments', []), 16)} |"
        )
    lines += ["", "### 3. Previous Iteration History"]
    if not context.history:
        lines.append("(first iteration)")
    for h in context.history[-10:]:
        lines.append(
            f"- iter {h.get('iteration')}: {h.get('outcome')} | target_score={h.get('target_score')} "
            f"| invariant_degradation={h.get('invariant_degradation')} | {str(h.get('hypothesis', ''))[:160]}"
        )
    if context.ledger_history:
        lines += ["", "### Episodic ledger", json.dumps(context.ledger_history, default=str)[:4000]]
    lines += ["", OUTPUT_CONTRACT.format(n=context.num_candidates)]
    return "\n".join(lines)


class ReasonerBridge:
    """``CandidateSource`` backed by an LLM reasoner."""

    def __init__(self, backend: LLMBackend, system_prompt: str = NEUROFORGE_SYSTEM_PROMPT):
        self.backend = backend
        self.system_prompt = system_prompt
        self.last_prompt: str | None = None
        self.last_reply: str | None = None

    def propose(self, context: ProposalContext) -> list[CandidatePatch]:
        """Queries the reasoner and parses up to ``num_candidates`` candidates.

        Raises:
            LLMBackendError / CandidateParseError: backend failure or unusable reply.
        """
        self.last_prompt = build_reasoner_prompt(context)
        self.last_reply = self.backend.complete(self.system_prompt, self.last_prompt)
        return parse_candidates_json(self.last_reply, source=getattr(self.backend, "name", "llm"))[
            : context.num_candidates
        ]


def topology_summary(adapter: Any, model: Any) -> dict[str, Any]:
    """Topology dict for prompts: architecture, layer count, widths, parameter count."""
    params = dict(model.named_parameters())
    down = adapter.get_layer_anatomy(0).mlp_down_proj
    return {
        "architecture": getattr(adapter, "family", type(adapter).__name__),
        "num_layers": adapter.num_layers(),
        "d_model": adapter.hidden_size(),
        "d_intermediate": int(params[down].shape[1]) if down else None,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
