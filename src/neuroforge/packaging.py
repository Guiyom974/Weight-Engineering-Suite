"""Packaging for Model Factory results (CR9): an MCP server or an agent skill that calls the edited local model.

Packages are built from fixed templates (``neuroforge/package_templates``). Nothing from a request or a suite is
placed into generated code: the tool name, description, model and settings go into ``tool_config.json``, which the
template reads at run time; the edit goes into ``patch.sparse.safetensors``. A skill's ``SKILL.md`` carries the name and
description in YAML front matter written with ``yaml.safe_dump``.

Packaging is refused unless the factory verdict allows it and the model is still at the state the factory measured.
Packages are written to ``<workspace>/packages/<timestamp>-<kind>/`` (shown in the Studio before confirmation).

Main entry points: ``PACKAGE_KINDS``, ``build_package``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .schemas import WorkspaceConfig
from .weight_git import WeightGitRepository
from .workspace_ops import export_patch

PACKAGE_KINDS = ("mcp", "skill")


def _template(name: str) -> str:
    return resources.files("neuroforge").joinpath(f"package_templates/{name}").read_text(encoding="utf-8")


def tool_name_for(suite_id: str) -> str:
    """A tool name safe for MCP clients: lower-case letters, digits and underscores (at most 64)."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", suite_id.lower()).strip("_")
    return (cleaned or "edited_model")[:64]


def package_destination(workspace: Path, kind: str, now: datetime | None = None) -> Path:
    """Where ``build_package`` writes (inside the workspace; the browser never chooses this path)."""
    if kind not in PACKAGE_KINDS:
        raise ValueError(f"package kind must be one of {', '.join(PACKAGE_KINDS)}")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return workspace / "packages" / f"{stamp}-{kind}"


def build_package(
    workspace: Path, cfg: WorkspaceConfig, repo: WeightGitRepository, report: dict[str, Any], kind: str,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a package for a finished factory run.

    Raises:
        ValueError: unknown kind, a verdict that does not allow packaging, or a model that moved since the run.
    """
    verdict = report.get("verdict") or {}
    if not verdict.get("packaging_allowed"):
        raise ValueError("packaging is only offered when the edited model beats the baseline and the prompt-only baseline")
    if report.get("head_after") != repo.head():
        raise ValueError("the model changed since this factory run; package from the state the run measured")
    out = out_dir or package_destination(workspace, kind)
    out.mkdir(parents=True, exist_ok=False)
    suite = report["suite"]
    name = tool_name_for(str(suite["suite_id"]))
    description = " ".join(str(suite["description"]).split())[:300]
    export_patch(workspace, "sparse", out / "patch.sparse.safetensors")
    measured: dict[str, Any] = {k: verdict.get(k) for k in ("split", "baseline", "prompt_only", "after")}
    config: dict[str, Any] = {
        "tool_name": name,
        "description": description,
        "model": cfg.model,
        "load_spec": cfg.load_spec,
        "device": cfg.device,
        "demo_defect": cfg.demo_defect,
        "prefix": "",
        "max_new_tokens": int(suite.get("max_new_tokens", 64)),
        "stop": list(suite.get("stop", [])),
        "patch_id": repo.head(),
        "factory_run": report.get("run_id"),
        "measured": measured,
    }
    (out / "tool_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    files = ["tool_config.json", "patch.sparse.safetensors"]
    if kind == "mcp":
        (out / "server.py").write_text(_template("mcp_server.py.txt"), encoding="utf-8")
        files.append("server.py")
        how = "python server.py   (register this command as a stdio MCP server in your agent)"
    else:
        (out / "run_tool.py").write_text(_template("run_tool.py.txt"), encoding="utf-8")
        front = yaml.safe_dump({"name": name.replace("_", "-"), "description": description}, sort_keys=False)
        (out / "SKILL.md").write_text(
            "---\n" + front + "---\n\n# Edited local model\n\nRun `python run_tool.py \"<prompt>\"` in this folder and use "
            "its output. The tool's purpose and settings are in tool_config.json.\n", encoding="utf-8")
        files += ["run_tool.py", "SKILL.md"]
        how = 'python run_tool.py "<prompt>"'
    m = measured
    (out / "README.md").write_text(
        "# Packaged edited model\n\n"
        f"Built by the Model Factory from run `{report.get('run_id')}` at patch `{repo.head()}`.\n\n"
        f"Measured pass@1 on {m['split']} tasks: baseline {m['baseline']}, prompt-only {m['prompt_only']}, "
        f"edited {m['after']}.\n\n"
        f"Run: `{how}`. Needs the weight-engineering-suite package (and its `hf` extra for Hugging Face models). "
        "tool_config.json names the local model folder on this machine; edit `load_spec` if you move the model.\n",
        encoding="utf-8")
    files.append("README.md")
    return {"kind": kind, "path": str(Path("packages") / out.name), "files": files, "tool_name": name}
