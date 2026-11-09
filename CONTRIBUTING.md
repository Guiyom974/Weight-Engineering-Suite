# Contributing

Thanks for helping. A few rules keep the tool trustworthy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest && ruff check src tests && mypy src
```

The default test suite is offline and uses toy models only. Real-checkpoint tests are opt-in:
`WEIGHTLAB_REAL_MODEL_DIR=/path/to/model pytest -m real_model`.

## Rules

1. **Measured, never invented.** Any metric shown in the UI, printed by the CLI or written to a manifest must come
   from an actual measurement in this run. Placeholder or demo numbers are not accepted. When a result comes from
   an offline template instead of an LLM, say so in the output.
2. **Tests with every change.** Add a test for new behaviour and a failing test first for every bug fix.
3. **No secrets, no personal paths.** Keys come from environment variables only. Never commit keys, `.env` files,
   machine-specific paths or model weights.
4. **Security-relevant paths need negative tests**: bad input, foreign origin, oversized body, path traversal.
5. **Document public code.** Every module starts with a docstring; public functions and classes have docstrings.
6. **Keep the Studio honest and simple.** Every screen states what it does and what comes next.

## Pull requests

Describe the change, how you tested it (commands and results), and anything you could not verify.
