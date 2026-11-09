"""Case study: correct three outdated capitals in LiquidAI LFM 3 1.7B with one fact edit, then undo it exactly.

LFM 3 1.7B names the former capitals of three countries: Lagos for Nigeria (capital until 1991), Karachi for
Pakistan (until 1959) and Almaty for Kazakhstan (until 1997). This script runs the Studio's "Update facts" mode on
the real checkpoint with its default settings, the same code the Studio uses:

1. records the model's greedy answers to the fact prompts, to rewordings it currently gets wrong and to neighbour
   prompts whose answers must not change (including each country's largest city, the old capital itself);
2. previews the edit (nothing changes), then applies it as one patch in a real workspace;
3. records the answers again on the edited model;
4. restores the original model through the workspace history and checks, tensor by tensor, that it is bit-exact.

Usage (needs the ``hf`` extra and the checkpoint on disk; a GPU makes it take about a minute, a CPU about 10):
    python examples/lfm3_capitals.py --model /path/to/LFM3-1.7B --device cuda --out lfm3_capitals_out

Writes ``results.json`` (every measured value) and ``results.md`` (the tables) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from neuroforge.edit_modes import Fact, FactBatch, commit_plan, plan_fact_edit
from neuroforge.workspace_ops import checkout, open_workspace

FACTS = [
    Fact(prompt="The capital of Nigeria is", subject="Nigeria", new_object=" Abuja", old_object=" Lagos",
         paraphrases=["Nigeria's capital city is", "The capital city of Nigeria is"],
         neighbours=["The largest city in Nigeria is", "The official language of Nigeria is", "The capital of Ghana is"]),
    Fact(prompt="The capital of Pakistan is", subject="Pakistan", new_object=" Islamabad", old_object=" Karachi",
         paraphrases=["The capital city of Pakistan is", "The capital of Pakistan is the city of"],
         neighbours=["The largest city in Pakistan is", "The official language of Pakistan is",
                     "The capital of Bangladesh is"]),
    Fact(prompt="The capital of Kazakhstan is", subject="Kazakhstan", new_object=" Astana", old_object=" Almaty",
         paraphrases=["Kazakhstan's capital city is", "The capital city of Kazakhstan is"],
         neighbours=["The largest city in Kazakhstan is", "The official language of Kazakhstan is",
                     "The capital of Uzbekistan is"]),
]
# Prompts that are not part of the edit at all, to show what else the edited model still says.
UNRELATED = ["The capital of France is", "The capital of Japan is", "The chemical symbol for gold is",
             "The novel Pride and Prejudice was written by"]


def greedy(loaded: Any, prompt: str, tokens: int = 12) -> str:
    """Greedy continuation of ``prompt`` (the same decoding the Studio uses for answers)."""
    tok, model = loaded.tokenizer, loaded.model
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False)
    return str(tok.decode(out[0, ids.shape[1]:]))


def all_prompts() -> list[tuple[str, str]]:
    """Every prompt with its role: fact, rewording, neighbour or unrelated."""
    rows: list[tuple[str, str]] = []
    for f in FACTS:
        rows.append(("fact", f.prompt))
        rows += [("rewording", p) for p in f.paraphrases]
        rows += [("neighbour", n) for n in f.neighbours]
    rows += [("unrelated", u) for u in UNRELATED]
    return rows


def write_markdown(out: Path, result: dict[str, Any]) -> None:
    """Human-readable tables of the measured results."""
    s = result["preview"]["summary"]
    lines = [
        "# LFM 3 1.7B: three outdated capitals, one fact edit",
        "",
        f"Model: {result['model']} on {result['device']}; measured {result['measured_at']}.",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Facts written (preview) | {s['efficacy_after']:.0%} (before: {s['efficacy_before']:.0%}) |",
        f"| Rewordings with the new answer | {s['paraphrase_after']:.0%} |",
        f"| Neighbour prompts unchanged | {s['neighbourhood_unchanged']:.0%} |",
        f"| KL on general prompts | {s['kl_general']} |",
        f"| Weights changed | {result['preview']['weights_targeted']:,} of {result['parameters']:,} "
        f"({result['preview']['weights_targeted'] / result['parameters']:.3%}); budget {result['preview']['weights_budget']:,} |",
        f"| Layers | {result['preview']['layers']} |",
        f"| Limits broken | {len(result['preview']['violations'])} |",
        f"| Restore of the original | {'bit-exact' if result['restore_bit_exact'] else 'NOT bit-exact'} |",
        "",
        "| Role | Prompt | Before | After the edit | After undo |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["answers"]:
        cells = [row["role"], row["prompt"], row["before"], row["after"], row["restored"]]
        lines.append("| " + " | ".join(c.replace("|", "/").replace("\n", " ").strip() for c in cells) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the case study and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="folder holding the LFM 3 1.7B checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="lfm3_capitals_out", help="folder for results.json and results.md")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lfm3_case_") as tmp:
        ws = Path(tmp) / "workspace"
        ws.mkdir()
        cfg = {"model": "LiquidAI/LFM3-1.7B", "model_path": str(Path(args.model).resolve()), "device": args.device}
        (ws / "neuroforge.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        from neuroforge.weight_git import WeightGitRepository

        WeightGitRepository.init(ws)
        _cfg, loaded, repo = open_workspace(ws)
        parameters = sum(p.numel() for p in loaded.model.parameters())
        prompts = all_prompts()
        before = {p: greedy(loaded, p) for _, p in prompts}

        started = time.perf_counter()
        plan = plan_fact_edit(loaded, repo, FactBatch(batch_id="capitals", facts=FACTS), "capitals")
        preview_seconds = time.perf_counter() - started
        report = plan.report
        params = dict(loaded.model.named_parameters())
        originals = {name: params[name].detach().cpu().clone() for name in report["tensors"]}
        patch_id = commit_plan(plan, loaded, repo, "LiquidAI/LFM3-1.7B")
        after = {p: greedy(loaded, p) for _, p in prompts}

        del loaded, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _root, restored = checkout(ws, "ROOT")
        restored_params = dict(restored.model.named_parameters())
        bit_exact = all(torch.equal(restored_params[n].detach().cpu(), t) for n, t in originals.items())
        after_undo = {p: greedy(restored, p) for _, p in prompts}

    result = {
        "model": "LiquidAI/LFM3-1.7B", "device": args.device, "parameters": parameters,
        "measured_at": time.strftime("%Y-%m-%d %H:%M"), "patch_id": patch_id,
        "preview_seconds": round(preview_seconds, 1),
        "preview": {k: report[k] for k in ("layers", "tensors", "weights_targeted", "weights_budget", "violations",
                                           "notes", "summary", "facts", "drift")},
        "restore_bit_exact": bit_exact,
        "answers": [{"role": r, "prompt": p, "before": before[p], "after": after[p], "restored": after_undo[p]}
                    for r, p in prompts],
    }
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown(out, result)
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
