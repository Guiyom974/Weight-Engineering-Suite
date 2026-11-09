"""Real experiments and validation tests executed on the local copy of LiquidAI/LFM3-1.7B.

Validates:
1. Loading from the local folder named by WEIGHTLAB_REAL_MODEL_DIR (opt-in).
2. Forward generation on diverse creative use cases (3D realistic forest scene, rock song lyrics).
3. Empirical Spectral Density (HTSR power-law alpha and Marchenko-Pastur bulk noise) across hybrid conv/transformer layers.
4. Mechanistic circuit tracing and contrastive divergence attribution.
5. CoW surgical intervention and measurement of behavioral steering.
"""

from __future__ import annotations

import pytest
import torch

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.loading import LoadedModel
from libweightspace.prober import MechanisticProber
from libweightspace.spectral import profile_matrix
from libweightspace.surgery_ops import norm_preserving_column_edit

from .conftest import load_real_lfm3

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def lfm3_local_bundle() -> LoadedModel:
    """Loads the local LiquidAI/LFM3-1.7B copy named by WEIGHTLAB_REAL_MODEL_DIR (opt-in)."""
    return load_real_lfm3()  # type: ignore[return-value]


def test_real_lfm3_local_copy_load(lfm3_local_bundle: LoadedModel) -> None:
    """Experiment 1: Verifies that the local LFM3 copy loads cleanly with complete hybrid anatomy."""
    model = lfm3_local_bundle.model
    adapter = lfm3_local_bundle.adapter

    assert type(model).__name__ == "Lfm3ForCausalLM"
    assert adapter.family == "lfm3"
    assert adapter.num_layers() == 24
    assert adapter.hidden_size() == 2048

    # Confirm both Conv and Attention hybrid layer anatomies are resolved
    anat0 = adapter.get_layer_anatomy(0)
    assert anat0.ssm_in_proj == "model.layers.0.conv.in_proj.weight"
    assert anat0.mlp_down_proj == "model.layers.0.feed_forward.w2.weight"

    anat2 = adapter.get_layer_anatomy(2)
    assert anat2.attn_q_proj == "model.layers.2.self_attn.q_proj.weight"


def test_real_lfm3_creative_generations(lfm3_local_bundle: LoadedModel) -> None:
    """Experiment 2: Evaluates forward generation on creative prompt use cases."""
    model = lfm3_local_bundle.model
    tokenizer = lfm3_local_bundle.tokenizer

    prompts = [
        # Use Case A: 3D realistic forest HTML
        "<!-- 3D Forest Scene in Three.js -->\n<canvas id='forest'></canvas>\n<script>\nconst trees = ",
        # Use Case B: 70s rock song lyrics
        "[Verse 1]\nMidnight highway and the thunder rolling loud\nElectric guitars roaring ",
    ]

    for p in prompts:
        with torch.inference_mode():
            toks = tokenizer(p, return_tensors="pt")
            curr_ids = toks["input_ids"].clone()
            for _ in range(8):
                logits = model(curr_ids).logits
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_tok], dim=-1)

            decoded = tokenizer.decode(curr_ids[0].tolist())
            assert len(decoded) > len(p)
            assert not torch.isnan(logits).any()


def test_real_lfm3_spectral_density_experiments(lfm3_local_bundle: LoadedModel) -> None:
    """Experiment 3: Analyzes empirical spectral density on real conv and MLP layers."""
    params = dict(lfm3_local_bundle.model.named_parameters())

    # Layer 0 Convolution projection weight [2048, 2048]
    conv_w = params["model.layers.0.conv.in_proj.weight"]
    prof_conv = profile_matrix(conv_w, tensor_name="conv0_in", top_k=16)
    assert prof_conv.power_law_alpha > 0.0
    assert prof_conv.mp_bulk_edge > 0.0
    assert prof_conv.snr_ratio > 0.0
    assert prof_conv.effective_rank > 1.0

    # Layer 0 MLP feed-forward down-projection weight [2048, 6144]
    mlp_w = params["model.layers.0.feed_forward.w2.weight"]
    prof_mlp = profile_matrix(mlp_w, tensor_name="mlp0_down", top_k=16)
    assert prof_mlp.power_law_alpha > 0.0
    assert prof_mlp.mp_bulk_edge > 0.0
    assert prof_mlp.snr_ratio > 0.0


def test_real_lfm3_circuit_tracing_attribution(lfm3_local_bundle: LoadedModel) -> None:
    """Experiment 4: Mechanistic contrastive circuit tracing on real LFM3 activations."""
    model = lfm3_local_bundle.model
    adapter = lfm3_local_bundle.adapter
    tokenizer = lfm3_local_bundle.tokenizer

    prober = MechanisticProber(model, adapter, tokenizer)
    fail_prompts = ["The forest is empty with zero objects."]
    pass_prompts = ["The forest is dense with realistic tall pines."]

    reports = prober.trace_contrastive(fail_prompts, pass_prompts)
    assert len(reports) == adapter.num_layers()
    # Reports should identify divergent layers and culpable columns
    assert any(r.divergence_score > 0.0 for r in reports)
    peak = max(reports, key=lambda r: r.divergence_score)
    assert len(peak.top_culpable_mlp_cols) > 0


def test_real_lfm3_surgical_steering_intervention(lfm3_local_bundle: LoadedModel) -> None:
    """Experiment 5: Executes surgical norm-preserving intervention and measures logit steering."""
    model = lfm3_local_bundle.model
    adapter = lfm3_local_bundle.adapter
    tokenizer = lfm3_local_bundle.tokenizer

    target_name = adapter.get_layer_anatomy(0).mlp_down_proj
    assert target_name is not None
    params = dict(model.named_parameters())
    orig_weight = params[target_name]

    # Target specific columns for norm-preserving edit
    cols = [10, 25, 42]
    delta = norm_preserving_column_edit(orig_weight, cols=cols, scale=0.8)
    delta_dict = {target_name: delta}

    cow = CoWWeightOverlay(model)
    prompt = "Create a 3D forest scene in HTML:"
    toks = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.inference_mode():
        logits_orig = model(toks).logits.clone()
        with cow.apply_delta(delta_dict) as forked_model:
            logits_fork = forked_model(toks).logits.clone()

    divergence = float(torch.norm(logits_fork - logits_orig, p=2).item())
    assert divergence > 0.0
    assert not torch.isnan(logits_fork).any()
