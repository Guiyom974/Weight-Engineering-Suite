# Guide: changing a model step by step

This guide follows the Studio: Workspaces, the five Edit steps (with the three kinds of change of step 2), the Model
factory and Self-improvement. Most steps have a CLI equivalent, shown at the end of each section.

## Concepts in one minute

- **Workspace**: a folder holding the model choice (`neuroforge.yaml`), the intent (`intent.yaml`), the edit history
  (`.wgit/`), exports (`exports/`), factory runs (`.factory/`), packages (`packages/`) and a self-improvement run
  (`.wrsi/`). Create one in the Studio's Workspaces view or with `neuroforge init`.
- **Studio home**: where the Studio keeps downloaded models (`models/`), the workspaces it creates (`workspaces/`) and
  the recent list. `WEIGHTLAB_HOME`, default `~/.weightlab`; `neuroforge studio --home DIR` overrides it.
- **Intent**: what is wrong and what must stay the same, as prompts plus measurable goals.
- **Probe**: compares the model's internal activity on failing and passing prompts to find which layers and weight
  columns push it toward the failure.
- **Candidate edit**: a small script that computes a weight change. It comes from an LLM or a built-in template and
  runs in a sandbox that can only do tensor math.
- **Gates**: every candidate is measured on a copy of the touched weights. It is rejected if it breaks an invariant
  (KL drift on prompts that must not change), exceeds the size budget, or disturbs protected directions.
- **Patch**: the winning edit, stored as a sparse, content-addressed file with a manifest of the measured metrics.
  Any patch, and the original model, can be restored exactly.

## Workspaces: open, create, check before download

`neuroforge studio` without `-w` opens the Workspaces view (with `-w`, or inside a workspace folder, it opens that
workspace). Recent workspaces and those in the Studio home are listed by name; press **Open** to switch (the model
loads as a job; the page keeps working).

**New workspace** offers three sources:

1. **Built-in demo**: a toy model with a planted repetition loop and a matching intent; works offline.
2. **A Hugging Face model**: type `organisation/model-name` (and a branch, tag or commit), choose the device, then
   **Check the model**. Nothing downloads yet. The check shows the download size (safetensors weights and
   config/tokenizer files only), the parameter count, the memory needed to edit (two bfloat16 copies of the weights:
   the model and the reference used for before/after checks) against the free memory of that device, the licence from
   the model card, the commit it pins, and whether an adapter supports the architecture. Warnings say when memory is
   short (with the suggestion to use the CPU or a smaller model), when the architecture cannot be edited, or when a
   gated model needs `HF_TOKEN`. **Download and create** fetches every file, checks its size and hash, and creates the
   workspace with the revision in `neuroforge.yaml`; **Cancel** removes partial files and creates nothing.
3. **A model folder**: any folder with a `config.json` that you copied into the Studio home's `models/` folder.

The **Current workspace** panel shows model, source, revision, licence, family, adapter, layers, parameters, data
type and device.

## Step 1: Connect a frontier model

The Studio fetches the live list of reasoning models from your endpoint when it starts. Choose a model and press
**Use this model**; the choice is saved in the workspace (never the key). Edits on real models are proposed only by
this model. On the built-in toy models the repair, factory and self-improvement steps can also use offline demo
templates, and results say so.

CLI: `neuroforge models`, then pass `--reasoner llm --model <id>` to `neuroforge loop`.

## Step 2: Describe the behaviour

An intent has three parts:

1. **Contrastive probes**: prompts where the model fails, each with the completion you want and the one you get;
   plus invariant prompts (`metadata: {role: invariant}`) whose behaviour must not change.
2. **Target goals**: metrics computed on the probes, for example the logit margin between the wanted and the
   unwanted completion, with a threshold.
3. **Invariants and constraints**: how much KL drift is allowed on invariant prompts, and size limits for an edit
   (fraction of parameters touched, relative Frobenius norm, spectral norm shift).

The demo intent (`neuroforge init --demo`) shows the format:

```yaml
intent_id: break-x-loop
description: After the token 'x' the model falls into a repetition loop (x x x ...). Remove the loop.
target_behavior: After 'x' the model prefers 'y' over another 'x' at least as much as the known-good model.
failure_behavior: After 'x' the model keeps predicting 'x'.
contrastive_probes:
- {probe_id: loop0, prompt: "ab x", expected_completion_prefix: "yzyz", undesired_completion_prefix: "xxxx"}
- {probe_id: inv0, prompt: "hello world", expected_completion_prefix: " ", metadata: {role: invariant}}
target_goals:
- {suite_name: loop, metric_name: logit_margin, operator: ">=", threshold: -2.85}
invariants:
- {suite_name: baseline, metric_name: kl_divergence, max_allowed_drop: 0.01}
constraints: {max_modified_param_ratio: 0.01, max_frobenius_rel_norm: 0.15, max_spectral_norm_shift: 0.05}
```

With a model selected, type a plain-language description and press **Draft intent with the LLM**; review the
draft in the editor, then **Validate and save**. Write at least four failing probes and four invariant prompts:
the gates are only as good as the prompts behind them.

CLI: `neuroforge draft-intent "description" -o intent.yaml --reasoner llm --model <id>`.

## Step 3: Diagnose

**Run diagnosis** ranks layers by divergence between failing and passing prompts and lists the culprit columns of
each layer's down projection. On the demo it finds layer 1, columns 5, 12 and 19, where the defect was planted.
If no layer stands out, your probes may not isolate the behaviour well: refine them.

CLI: `neuroforge probe -i intent.yaml -w <workspace>`.

## Step 4: Repair

Choose who proposes edits, the number of iterations and candidates per iteration, and press **Start repair**.
The table shows every candidate with its measured target score, invariant drift and number of weights targeted,
and whether it was committed, rejected (and why) or feasible but not the best. Each iteration commits the best
candidate that improves the target score and breaks no limit, even when the goals are not fully met yet; the history
records whether they were (`neuroforge log`, column "goals met"). A run stops early when every goal is met.

**Weights targeted** counts the weights the edit script changes, and the size budget applies to that count. When
nullspace protection is active it projects the edit so that protected directions do not move, which can add a small
correction to other entries of the same tensors. That correction is stored in the patch, included in the Frobenius
and spectral checks, and never touches other tensors. On the demo, the edit targets 3 columns (96 weights) of one
matrix; the stored patch changes that matrix only.

What the numbers mean: **goal shortfall** is the weighted distance still missing to your goals, 0 (shown as
"met") once every goal is satisfied, negative otherwise; overshooting a met goal earns nothing, so among candidates
that meet the goals the smallest, safest edit wins. **Invariant budget used** is the worst change on invariant
prompts divided by the change your intent allows: above 100% the candidate is rejected, "improved" means the
protected behaviour moved closer to the reference model.

Control options: **Ask me before each edit is applied** (on by default) pauses at the best feasible candidate and
shows its hypothesis, script, touched tensors, size and measured scores; you apply or reject it. **Dry run** evaluates
candidates without changing anything. **Protected tensors** (comma-separated name parts, for example
`layers.0.self_attn`) reject any candidate that touches them. A progress bar shows the current step and **Cancel**
stops at the next safe point; only edits you already approved are kept.

CLI: `neuroforge loop -i intent.yaml -w <workspace> [--reasoner llm --model <id>] --iterations 3 --candidates 3`.

## Step 5: Keep, undo or export

The history lists committed patches, newest first. **Restore** moves the model to that point exactly;
**Restore the original model** goes back to the unedited weights. Export writes to `<workspace>/exports/`:

| Format | Content | Use |
| --- | --- | --- |
| safetensors | full model with the edits merged | load directly |
| sparse | only the changed weights | share or audit an edit |
| lora | PEFT LoRA adapter approximating the edit | apply with PEFT-compatible tools |
| hf (Hugging Face models only) | Hugging Face folder | convert to GGUF with llama.cpp, load with transformers |
| bundle (Studio only) | patch, edit history, intent, model card, SHA-256 checksums and base fingerprint | share or archive an edit with its evidence |

**Compare answers** generates, for every intent prompt, the original model's answer and the current one (greedy
decoding), so you can read what changed, including on prompts that must not change.

CLI: `neuroforge log -w <workspace>`, `neuroforge export -w <workspace> --format hf -o out/`.

## Step 2, other modes: update facts

Choose **Update facts** in step 2. Each fact has a prompt, the subject it is about (it must appear in the prompt), the
new answer, optionally the old answer, paraphrases (rewordings that should get the new answer too) and neighbours
(prompts about other subjects whose answers must not change). Type facts in the form or paste CSV with the header
`prompt,new_object,old_object,subject,paraphrases,neighbours` (several paraphrases or neighbours separated by `|`).

**Preview the edit** computes the change without applying it:

1. For each fact, the value the edited layers must write at the subject's last token is found by optimising an
   activation offset (never a weight) until the new answer leads the runner-up by the chosen margin (1 nat by default).
2. The change is shared between the layers of a window (four early-middle layers by default; layers 4 to 7 on a
   24-layer model) in proportion to each layer's capacity, and solved in closed form inside each layer's share of the
   weight budget (0.05% of the weights by default on real models).
3. It is projected so that the inputs of the neighbour and general prompts at every token stay untouched (only their
   last token when there is not enough room, which the preview notes).

The preview shows, per fact, the answer before and after, whether the fact is written, the share of paraphrases that
changed and the share of neighbours that did not; plus KL on general prompts, the weights targeted against the
budget, and each tensor's distance from the original model before and after (cumulative over earlier batches).
**Apply this edit** commits exactly the previewed batch as one patch; it is refused when a limit is broken or the
model changed since the preview. Undo it in step 5 like any patch.

On the built-in toy models, fact edits large enough to change an answer exceed the limits: the preview says why and
refuses to apply. Fact mode was validated on LiquidAI LFM 3 1.7B (README, "A real case"): three outdated capitals were
corrected and every neighbour kept its answer, but only 4 of 6 rewordings followed (the possessive "Nigeria's capital
city is" did not). Read the paraphrase column of the preview, and validate other architectures before relying on fact
mode.

Answers start with a space on real models (" Abuja", not "Abuja"): keep it in the CSV (`prompt, Abuja`). If you leave
it out and the model writes one after the prompt, the preview adds it and says so in its notes.

## Step 2, other modes: adjust a behaviour

Choose **Adjust a behaviour**. Give prompts that show the behaviour and the same prompts without it, one per line
(or draft both sets with your frontier model from a description). **Preview the edit** measures, for every layer, how
strongly what its down projection writes separates the two sets (in standard deviations), picks the strongest middle
layer unless you choose one, and removes (positive strength) or amplifies (negative strength) that direction in the
budgeted columns most aligned with it. The preview shows the per-layer strengths, KL to the original model on general
and behaviour prompts, the gap between the two sets along the direction before and after, and sample answers.
**Apply this edit** commits it as one patch.

## Model factory

The factory specialises the model for one job and measures whether that worked. A **task suite** (YAML) lists tasks:

```yaml
suite_id: python-functions
description: Small Python functions checked by unit tests.
max_new_tokens: 96
stop: ["\ndef "]
prompt_baselines: ["# Write correct, simple Python.\n"]   # system prompts for the prompt-only baseline
tasks:
  - id: add
    prompt: "def add(a, b):\n"
    check: python_test              # python_test | sql_rows | starts_with | not_starts_with
    answer_starts_with_prompt: true # check prompt + answer (code completion)
    test: "assert add(2, 3) == 5"
    reference: "    return a + b\n" # a correct answer: lets a failing training task guide the repair
    split: train                    # train | heldout
```

`sql_rows` tasks give `setup` SQL and `expected` rows; text checks give `expected` text. **Run the factory** then:

1. answers every task with the current model and runs its check (code and SQL in an isolated child process with no
   network, a memory limit and a timeout); every result goes to `.factory/<run>/log.jsonl`;
2. measures the prompt-only baseline: each system prompt on the training tasks, the best one also on held-out tasks;
3. turns failing training tasks that have a reference into an intent (passing tasks and general prompts become
   invariants) and runs the guarded repair loop with your approval; held-out tasks are never shown to the proposer;
4. answers every task again with the edited model.

The report gives pass@1 per split for the baseline, the prompt-only baseline and the edited model, every task's check
result before and after, and a verdict on held-out tasks (training tasks when there are none). It says plainly when a
system prompt does as well as the edit. Only then is **Write the package** enabled: an MCP server (stdio) or an agent
skill, written to `<workspace>/packages/`, built from fixed templates with your suite's text only in
`tool_config.json`.

## Self-improvement

Self-improvement runs repeated generations on the workspace's original model, separately from the Edit history. Each
generation diagnoses the intent's failing prompts, asks the proposer for candidates and promotes the best one only if
it improves the task and passes every guardrail:

- **Anchor KL limit**: KL to the original model on the intent's protected prompts, checked by a separate process;
- **protected tensors**: candidates touching them are rejected;
- **forgetting check**: held-out `prompt => expected answer` lines; a candidate that makes those answers less likely
  by more than the allowed drop is rejected;
- **generations this run** and **candidates per generation**.

**Pause after this generation** finishes the running generation and saves everything; **Stop now** stops at the next
candidate and promotes nothing from that generation; **Roll back one generation** first shows what it will restore,
then returns the weights and nullspace locks bit-exactly. The timeline lists every step with each candidate's outcome:
the gate and measured value that rejected it, or "promoted".

CLI: `wrsi start -w <workspace> --generations 3 --protect layers.0 --forgetting heldout.txt`, `wrsi status`,
`wrsi rollback`.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Step 1 says `no-key` | the key variable is not set in the shell that started the Studio | set it, restart the Studio |
| Step 1 says `unauthorized` | wrong or expired key | create a new key with your provider |
| "Missing or invalid Studio token" | the page was loaded from another Studio run | reload the page |
| Every candidate is rejected for invariant drift | edits too large for your invariants | add more targeted probes, lower the constraints only if justified |
| Repair converges on the probes but behaviour elsewhere changed | invariant prompts do not cover it | add invariant prompts for that behaviour and repeat |
| A fact preview says a key "lies entirely in directions the protected prompts use" | the protected prompts use every free direction in the budgeted columns | raise the weight budget, edit other layers, or protect fewer prompts |
| A fact preview breaks the norm limits on a real model | the facts need a larger change than the limits allow | add layers to the window, raise the budget, or lower the margin |
| The factory says a system prompt is enough | the prompt-only baseline matches the edited model | use the system prompt; weight editing is not worth it for this job |
| "the model changed since this preview" | another edit or a restore happened after the preview | preview again |
| The page loads incompletely, or a request fails with "connection reset" (Windows) | security software that inspects local network traffic can interrupt connections to 127.0.0.1 | reload the page; if it keeps happening, allow Python's local connections in that software |
| Compare answers shows a prompt that must not change with a different long answer | the invariant check measures the next-token distribution, not whole generations | read the compare table, add the prompt as an invariant, or tighten the drift limit |
