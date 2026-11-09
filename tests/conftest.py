"""Shared pytest fixtures: tiny synthetic Qwen/Gemma/DeepSeek/LFM3 models and a planted-attractor model.

All models are built on CPU from seeded generators, so every test is deterministic and runs in milliseconds.
Real-checkpoint tests are opt-in: ``require_real_model_dir`` skips them unless WEIGHTLAB_REAL_MODEL_DIR is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from libweightspace.toy_models import (
    CharTokenizer,
    PlantedDefect,
    ToyCausalLM,
    build_tiny_deepseek,
    build_tiny_gemma4,
    build_tiny_lfm3,
    build_tiny_qwen3,
    plant_attractor_defect,
)

# Fail texts drive the planted 'x' attractor; pass texts are matched in length but never contain 'x'.
FAIL_PROMPTS = ["ab xxxxxxxxxx", "cd xxxxxxxxxx", "the xxxxxxxxx", "q: xxxxxxxxxx"]
PASS_PROMPTS = ["ab yzyzyzyzyz", "cd zyzyzyzyzy", "the yzyzyzyzy", "q: zyzyzyzyzy"]
NEUTRAL_PROMPTS = ["hello world", "abc def ghi", "the quick brown", "lorem ipsum do"]


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _isolated_studio_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the Studio home at a fresh temporary folder so no test reads or writes the real ``~/.weightlab``."""
    home = tmp_path_factory.mktemp("weightlab-home")
    monkeypatch.setenv("WEIGHTLAB_HOME", str(home))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WEIGHTLAB_HF_ENDPOINT", raising=False)
    return home


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def tiny_qwen() -> ToyCausalLM:
    return build_tiny_qwen3(seed=0)


@pytest.fixture
def tiny_gemma() -> ToyCausalLM:
    return build_tiny_gemma4(seed=0)


@pytest.fixture
def tiny_deepseek() -> ToyCausalLM:
    return build_tiny_deepseek(seed=0)


@pytest.fixture
def tiny_lfm3() -> ToyCausalLM:
    return build_tiny_lfm3(seed=0)


@pytest.fixture
def planted_qwen() -> tuple[ToyCausalLM, PlantedDefect]:
    model = build_tiny_qwen3(seed=0)
    defect = plant_attractor_defect(model, layer_idx=1, columns=(5, 12, 19))
    return model, defect


REAL_MODEL_ENV = "WEIGHTLAB_REAL_MODEL_DIR"


def require_real_model_dir() -> Path:
    """Local real checkpoint folder for opt-in tests; skips with the reason when it is not configured.

    The default suite never downloads or loads multi-gigabyte models. To run the real-model tests, point
    WEIGHTLAB_REAL_MODEL_DIR at a local copy of LiquidAI/LFM3-1.7B (a folder holding config.json and
    model.safetensors).
    """
    raw = os.environ.get(REAL_MODEL_ENV)
    if not raw:
        pytest.skip(f"opt-in real-model test: set {REAL_MODEL_ENV} to a local LiquidAI/LFM3-1.7B folder")
    path = Path(raw)
    if not (path / "config.json").is_file():
        pytest.skip(f"{REAL_MODEL_ENV}={raw} does not contain config.json")
    return path


_REAL_MODEL_CACHE: dict[str, object] = {}


def load_real_lfm3() -> object:
    """Load the opt-in real checkpoint once per test session, synchronously.

    Loading it twice with transformers' threaded loader crashed the interpreter on Windows (access violation in
    core_model_loading), so the copy is shared and HF_DEACTIVATE_ASYNC_LOAD is set before loading.
    """
    path = require_real_model_dir()
    if "bundle" not in _REAL_MODEL_CACHE:
        from libweightspace.loading import load_model

        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        _REAL_MODEL_CACHE["bundle"] = load_model(str(path), device="cpu")
    return _REAL_MODEL_CACHE["bundle"]
