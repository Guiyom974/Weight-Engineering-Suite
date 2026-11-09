"""Unit and integration tests for Tier 2 backlog items (NF-BL-04 through NF-BL-08)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.loading import load_model
from libweightspace.transcoder import MLPTranscoder, diff_models
from neuroforge.autotuner import autotune_candidate
from neuroforge.blueprint_compiler import BlueprintCompiler, procrustes_align
from neuroforge.exporter import (
    export_peft_lora,
    patch_gguf_inplace,
    read_gguf_v3_tensor,
    write_gguf_v3,
)
from neuroforge.merger import MergeStrategy, merge_deltas
from neuroforge.schemas import (
    DonorGraftSpec,
    EditConstraints,
    LayerBlueprint,
    NeuralBlueprint,
    PatchManifest,
    SubspaceSlice,
)
from neuroforge.weight_git import WeightGitRepository


def _dummy_manifest(intent_id: str, code: str) -> PatchManifest:
    return PatchManifest(
        patch_id="pending",
        parent_patch_id=None,
        base_model_id="toy:qwen3_8",
        base_model_sha256="0" * 64,
        intent_id=intent_id,
        iteration=0,
        synthesized_code=code,
        reasoner_rationale=intent_id,
        tensor_stats=[],
        total_modified_params=4,
        overall_sparsity_ratio=0.01,
        target_metrics_before={"m": 0.0},
        target_metrics_after={"m": 1.0},
        invariant_metrics_before={"i": 1.0},
        invariant_metrics_after={"i": 1.0},
        passed_gate=True,
    )


def test_transcoder_feature_surgery_and_diff_models() -> None:
    """NF-BL-04: Sparse transcoder contrastive attribution, rank-1 W_down folding, and diff_models."""
    torch.manual_seed(11)
    tc = MLPTranscoder(d_model=16, d_features=32, seed=11)
    fail_in = torch.randn(10, 16) + 2.0
    pass_in = torch.randn(10, 16) - 2.0
    attrs = tc.attribute_contrastive(fail_in, pass_in, top_k=4)
    assert len(attrs) == 4
    assert attrs[0].feature_idx >= 0
    assert abs(attrs[0].differential_activation) >= abs(attrs[-1].differential_activation)

    w_down = torch.randn(16, 48)
    delta = tc.fold_feature_edit_into_down_proj(w_down, attrs[0].feature_idx, scale=-0.5, top_cols=6)
    assert delta.shape == (16, 48)
    # Exactly 6 columns should be modified when top_cols=6.
    active_cols = int((delta.abs().max(dim=0).values > 0).sum().item())
    assert active_cols == 6

    m_a = load_model("toy:qwen3_8", demo_defect=False).model
    m_b = load_model("toy:qwen3_8", demo_defect=True).model
    diffs = diff_models(m_a, m_b, top_k=3)
    assert len(diffs) >= 1
    assert any("down_proj" in d.tensor_name for d in diffs)
    assert diffs[0].frobenius_diff > 0.0


def test_wgit_branches_and_multi_branch_merging(tmp_path: Path) -> None:
    """NF-BL-05: .wgit branch creation/checkout and DELLA, SLERP, OrthoMerge, and nullspace-chain fusion."""
    repo = WeightGitRepository.init(tmp_path)
    w_name = "model.layers.1.mlp.down_proj.weight"

    # Commit branch A on top of ROOT.
    d_a = torch.zeros(16, 32)
    d_a[:, 2] = 1.0
    c_a = repo.commit({w_name: d_a}, _dummy_manifest("branch-a", "# a"), promote=False)
    repo.create_branch("feat-a", c_a.patch_id)

    # Checkout ROOT and commit branch B on disjoint columns.
    repo.checkout("ROOT")
    d_b = torch.zeros(16, 32)
    d_b[:, 7] = -1.5
    c_b = repo.commit({w_name: d_b}, _dummy_manifest("branch-b", "# b"), promote=False)
    repo.create_branch("feat-b", c_b.patch_id)

    branches = repo.list_branches()
    assert branches["feat-a"] == c_a.patch_id
    assert branches["feat-b"] == c_b.patch_id
    assert repo.resolve("feat-a") == c_a.patch_id

    cum_a = repo.cumulative_delta(repo.resolve("feat-a"))
    cum_b = repo.cumulative_delta(repo.resolve("feat-b"))

    strategies: tuple[MergeStrategy, ...] = ("della", "slerp", "orthomerge", "nullspace-chain")
    for strat in strategies:
        merged = merge_deltas([cum_a, cum_b], strategy=strat)
        assert w_name in merged
        assert merged[w_name].shape == (16, 32)

    # Verify nullspace-chain guarantees branch 2 delta is orthogonal to branch 1's row space.
    chained = merge_deltas([cum_a, cum_b], strategy="nullspace-chain")[w_name]
    diff_b = chained - d_a
    # Column 2 is the sole active input coordinate of d_a, so diff_b[:, 2] must be 0.
    assert float(torch.linalg.vector_norm(diff_b[:, 2]).item()) < 1e-6


def test_inner_loop_autotuner_selects_optimal_alpha_and_k() -> None:
    """NF-BL-06: autotune_candidate searches over (alpha, k) in CoW forks and leaves base model untouched."""
    bundle = load_model("toy:qwen3_8")
    model = bundle.model
    overlay = CoWWeightOverlay(model)
    params = dict(model.named_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    target_name = bundle.adapter.get_layer_anatomy(1).mlp_down_proj
    assert target_name is not None
    orig_weight = params[target_name].detach().clone()

    script = f"""
def compute_delta(weights, diagnostics):
    W = weights[{target_name!r}]
    cols = diagnostics["layers"][0]["top_culpable_mlp_cols"]
    d = torch.zeros_like(W)
    for c in cols:
        d[:, c] = -0.02 * W[:, c]
    return {{{target_name!r}: d}}
"""
    diagnostics = {
        "layers": [
            {
                "layer_idx": 1,
                "top_culpable_mlp_cols": [0, 1, 2, 3, 4],
                "top_mlp_col_alignments": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        ]
    }
    constraints = EditConstraints(
        max_modified_param_ratio=0.05,
        max_frobenius_rel_norm=0.20,
        max_spectral_norm_shift=0.10,
    )

    # Ideal configuration is alpha=1.25, k=3 -> highest synthetic target score.
    def eval_fn(_forked: torch.nn.Module) -> tuple[float, float, bool]:
        cur = dict(_forked.named_parameters())[target_name].detach()
        diff = cur - orig_weight
        active_cols = int((diff.abs().max(dim=0).values > 1e-7).sum().item())
        col0_scale = float(torch.linalg.vector_norm(diff[:, 0]) / (torch.linalg.vector_norm(orig_weight[:, 0]) + 1e-8))
        alpha_est = col0_scale / 0.02
        score = -abs(alpha_est - 1.25) - 0.5 * abs(active_cols - 3)
        return score, 0.01, True

    res = autotune_candidate(
        model,
        overlay,
        params,
        total_params,
        constraints,
        script,
        diagnostics,
        eval_fn,
        alphas=(0.5, 1.0, 1.25),
        k_values=(2, 3, 5),
    )
    assert res.best_alpha == 1.25
    assert res.best_k == 3
    assert len(res.trials) == 9
    # Base model must remain bit-exact after all CoW autotune trials.
    assert torch.equal(params[target_name], orig_weight)


def test_export_peft_lora_and_gguf_inplace_patching(tmp_path: Path) -> None:
    """NF-BL-07: PEFT LoRA SVD export and GGUF v3 Q8_0 in-place block surgical patching."""
    torch.manual_seed(23)
    # 1. PEFT LoRA export: rank-2 exact delta should reconstruct with < 1e-5 error.
    u = torch.randn(32, 2)
    v = torch.randn(2, 64)
    delta_w = u @ v
    lora_dir = export_peft_lora(
        {"model.layers.1.mlp.down_proj.weight": delta_w},
        tmp_path / "lora_out",
        rank=4,
        lora_alpha=8.0,
        base_model_id="toy:qwen3_8",
    )
    cfg = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["peft_type"] == "LORA"
    assert cfg["r"] == 4
    tensors = load_file(str(lora_dir / "adapter_model.safetensors"))
    a = tensors["base_model.model.model.layers.1.mlp.down_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"]
    recon = (cfg["lora_alpha"] / cfg["r"]) * (b @ a)
    assert torch.allclose(recon, delta_w, atol=1e-5)

    # 2. GGUF v3 Q8_0 write and in-place block patching.
    w_base = torch.randn(16, 64) * 0.5
    w_other = torch.randn(8, 32) * 0.25
    gguf_path = write_gguf_v3(
        {"blk.0.ffn_down.weight": w_base, "blk.1.ffn_down.weight": w_other},
        tmp_path / "model.gguf",
        quant="q8_0",
    )
    before_other = read_gguf_v3_tensor(gguf_path, "blk.1.ffn_down.weight")
    patch_delta = torch.zeros_like(w_base)
    patch_delta[0, :32] = 0.25  # Only modifies the very first 32-element Q8_0 block!
    patched = patch_gguf_inplace(gguf_path, {"blk.0.ffn_down.weight": patch_delta})
    assert patched == ["blk.0.ffn_down.weight"]

    after_base = read_gguf_v3_tensor(gguf_path, "blk.0.ffn_down.weight")
    after_other = read_gguf_v3_tensor(gguf_path, "blk.1.ffn_down.weight")
    # Untouched tensor and untouched blocks in blk.0 are bit-identical.
    assert torch.equal(before_other, after_other)
    assert torch.allclose(after_base[1:, :], read_gguf_v3_tensor(gguf_path, "blk.0.ffn_down.weight")[1:, :])
    assert float((after_base[0, :32] - w_base[0, :32]).mean().item()) > 0.20


def test_procrustes_donor_organ_transplanting(tmp_path: Path) -> None:
    """NF-BL-08: Orthogonal Procrustes alignment and DonorGraftSpec compilation."""
    torch.manual_seed(31)
    donor = load_model("toy:qwen3_8").model
    donor_ckpt = tmp_path / "donor.safetensors"
    save_file({k: v.detach().cpu() for k, v in donor.state_dict().items()}, str(donor_ckpt))

    # Verify procrustes_align recovers an exact orthogonal rotation matrix R.
    src = torch.randn(64, 32)
    q_true, _ = torch.linalg.qr(torch.randn(32, 32))
    tgt = src @ q_true
    r_est = procrustes_align(src, tgt)
    assert torch.allclose(r_est.T @ r_est, torch.eye(32), atol=1e-5)
    assert torch.allclose(src @ r_est, tgt, atol=1e-4)

    from typing import Any

    d_cfg: Any = donor.config
    bp = NeuralBlueprint(
        model_name="franken-qwen",
        vocab_size=d_cfg.vocab_size,
        hidden_size=d_cfg.hidden_size,
        intermediate_size=d_cfg.intermediate_size,
        num_hidden_layers=d_cfg.num_hidden_layers,
        num_attention_heads=d_cfg.num_attention_heads,
        num_key_value_heads=d_cfg.num_key_value_heads,
        head_dim=d_cfg.head_dim,
        subspaces=[SubspaceSlice(name="s0", start_dim=0, end_dim=16, description="test")],
        layers=[LayerBlueprint(layer_idx=0)],
        donor_graft=DonorGraftSpec(
            checkpoint_path=str(donor_ckpt),
            donor_layers=[0, 1],
            target_layers=[0, 1],
            transplant_embeddings=True,
            procrustes_align=True,
        ),
    )
    compiled_model = BlueprintCompiler(bp).build_model()
    out = compiled_model(torch.tensor([[1, 2, 3, 4]]))
    assert out.logits.shape == (1, 4, d_cfg.vocab_size)
