# Changelog

## 0.2.0

- Workspaces view (CR2): the Studio starts without `-w` on a start page with recent workspaces and those in the Studio
  home (`WEIGHTLAB_HOME`, `--home`). New-workspace wizard: built-in demo, a Hugging Face model checked before download
  (size, bfloat16 memory against free CPU/GPU memory, licence, pinned revision, adapter support), or a model folder.
  Downloads are https-only, safetensors-only, hash-verified, cancellable (partial files removed) and never send
  `HF_TOKEN` to another host. Facts panel; switching workspaces runs as a job and keeps the session token.
- Edit modes (CR6): "Update facts" (closed-form edits keyed on the subject token over a window of early-middle layers,
  capacity-weighted shares, inside the weight budget, exact nullspace protection of neighbour and general prompts,
  per-fact success, paraphrase and neighbourhood results, KL and cumulative drift before applying) and "Adjust a
  behaviour" (per-layer strength, signed strength, KL and measured shift before applying). Each batch is one patch.
- Model factory (CR9): task suites with Python unit-test, SQL result and text checks, baseline, prompt-only baseline,
  guarded repair on failing training tasks (held-out tasks never reach the proposer), after-measurement, verdict, and
  packaging as an MCP server or agent skill from fixed templates (suite text only in `tool_config.json`).
- Self-improvement view (CR10): guardrails (anchor KL limit, generations, protected tensors, forgetting check on
  held-out prompts), pause after the current generation, stop, rollback with a preview, and a timeline of every
  candidate with the gate and measured value that decided it. `wrsi start` and the Studio share `wrsi.runner`;
  `wrsi start` gains `--protect`, `--forgetting` and `--max-forgetting-drop`.
- Frontier-only proposals: in the Studio, edits on real models are proposed by the frontier model chosen in step 1;
  offline templates are limited to the toy models and labelled "offline demo templates".
- New hardened check runner (`libweightspace.check_runner`): `python -I -S` child, audit hook, Windows job object
  (memory limit, one process, kill on close) or POSIX `RLIMIT_AS`, completion marker. `CodePassAt1Suite` uses it.
- Fixed: the Windows memory limit of the surgery subprocess was created but never applied; it is now enforced (the
  default 2048 MB is what a CUDA build of torch needs to import).
- Fixed: a local model folder whose name contains `@` was read as a Hub revision pin.
- Fact previews run on half-precision models on the CPU (oneDNN has no bf16 backward on some CPUs; it is switched
  off for that search).
- W-RSI rejection reasons name the gate and its measured value ("Anchor KL: ...", "Forgetting check: ...").
- Fixed: an external LLM proposing W-RSI edits (`wrsi start --proposer llm`, the Studio's frontier option) was given
  the self-editing prompt ("the weights are your own") and its candidates were labelled "self"; it now gets a prompt
  for editing a local model and a label naming it.
- Fixed: a self-improvement rollback took the next generation's step number, so two timeline rows showed the same
  step; rollback rows now have no step number.
- Fixed: an LLM run with its key variable unset sent an unauthenticated request and reported a bare "HTTP 400"; it
  now stops before any request with a message such as `POE_API_KEY is not set` (local servers on 127.0.0.1 still
  work without a key).
- `neuroforge log` labels its column "goals met" (it was "gate"): every committed edit passed the limits, and the
  column says whether all target goals were also reached.

## 0.1.0 (first public release)

- Guided local Studio (five steps: connect, describe, diagnose, repair, keep or undo) served by a hardened,
  same-origin server with a per-session token.
- One configuration path for every OpenAI-compatible endpoint; Poe preconfigured; model list fetched live from
  `GET /models`; `neuroforge models` and `neuroforge doctor`.
- Engine: contrastive probing, sandboxed candidate edits, copy-on-write evaluation, nullspace protection,
  sparsity budgets, spectral checks, content-addressed `.wgit` history with exact restore, exports
  (safetensors, sparse delta, PEFT LoRA, Hugging Face folder; experimental GGUF patch in the CLI).
- Architectures: Qwen-style, Gemma-style, DeepSeek-style (MLA/MoE) and LiquidAI LFM 3 hybrids; toy models of each
  for offline tests.
- Real-checkpoint tests are opt-in (`WEIGHTLAB_REAL_MODEL_DIR`).
- Studio controls (CR1, CR3, CR4, CR5, CR7, CR8 first slice): next-step bar, persisted results, intent examples,
  original-vs-current answers, approve-before-commit, dry run, protected tensors, progress and cancel, edit bundles.
- Model list filtered to reasoning models (Poe `reasoning` metadata, OpenRouter `supported_parameters`); `--all` lists
  everything.
- LLM edit scripts: harmless `import torch` lines are rewritten into assignments before the sandbox validator runs.
- The Studio, the edit history and bundle model cards report an edit's size as "weights targeted" (the weights the
  script changes, which the size budget limits) and explain that nullspace protection can add a small correction to
  other entries of the same tensors.
- The Studio compares the session token in constant time; a malformed or non-ASCII token is a 403, never a 500.
