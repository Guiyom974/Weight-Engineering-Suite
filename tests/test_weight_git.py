"""Git-for-Weights, exporter and patch-stats tests (blueprint step 9)."""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_model

from libweightspace.toy_models import build_tiny_qwen3
from neuroforge.exporter import (
    export_hf_pretrained,
    export_merged_safetensors,
    export_sparse_delta,
    load_sparse_delta,
)
from neuroforge.patch_stats import state_fingerprint, summarize_patch, tensor_delta_stats
from neuroforge.schemas import PatchManifest
from neuroforge.weight_git import ROOT, WeightGitError, WeightGitRepository, from_coo, to_coo

W = "model.layers.1.mlp.down_proj.weight"
O_PROJ = "model.layers.0.self_attn.o_proj.weight"


def _manifest(code="x", passed=True) -> PatchManifest:
    return PatchManifest(
        patch_id="pending",
        parent_patch_id=None,
        base_model_id="toy:qwen3_8",
        base_model_sha256="0" * 64,
        intent_id="intent",
        iteration=0,
        synthesized_code=code,
        reasoner_rationale="because",
        tensor_stats=[],
        total_modified_params=0,
        overall_sparsity_ratio=0.0,
        target_metrics_before={},
        target_metrics_after={},
        invariant_metrics_before={},
        invariant_metrics_after={},
        passed_gate=passed,
    )


def _delta(model, name, cols, value):
    d = torch.zeros_like(dict(model.named_parameters())[name])
    d[:, cols] = value
    return d


@pytest.fixture
def repo(tmp_path):
    return WeightGitRepository.init(tmp_path)


def test_init_layout(repo, tmp_path):
    root = tmp_path / ".wgit"
    assert (root / "HEAD").read_text() == ROOT and (root / "refs/heads/main").read_text() == ROOT
    assert (root / "commits").is_dir() and (root / "objects").is_dir()
    assert WeightGitRepository.exists(tmp_path) and WeightGitRepository.init(tmp_path).head() == ROOT
    with pytest.raises(WeightGitError, match="run init"):
        WeightGitRepository(tmp_path / "elsewhere")


def test_commit_writes_coo_object_and_moves_refs(repo, tmp_path, tiny_qwen):
    d = _delta(tiny_qwen, W, [2, 7], 0.25)
    m = repo.commit({W: d}, _manifest())
    assert repo.head() == m.patch_id == repo.main()
    assert m.parent_patch_id is None and len(m.delta_sha256) == 64 and m.created_at
    obj = tmp_path / ".wgit" / "objects" / f"{m.patch_id}.COO.safetensors"
    with safe_open(str(obj), framework="pt") as f:
        assert set(f.keys()) == {f"{W}__indices", f"{W}__values", f"{W}__shape"}
        idx, vals, shape = (
            f.get_tensor(f"{W}__indices"),
            f.get_tensor(f"{W}__values"),
            f.get_tensor(f"{W}__shape"),
        )
    assert idx.dtype == torch.int32 and idx.shape == (2, 2 * d.shape[0])
    assert vals.shape == (2 * d.shape[0],) and shape.dtype == torch.int64 and shape.tolist() == list(d.shape)
    assert torch.equal(repo.load_delta(m.patch_id)[W], d)
    # Unpromoted commits move HEAD but not main.
    m2 = repo.commit({W: d * 2}, _manifest("y", passed=False))
    assert repo.head() == m2.patch_id and repo.main() == m.patch_id and m2.parent_patch_id == m.patch_id


def test_dag_traversal_checkout_and_revert(repo):
    base = build_tiny_qwen3(seed=0)
    model = build_tiny_qwen3(seed=0)
    d1, d2 = _delta(model, W, [1], 0.5), _delta(model, O_PROJ, [3], -0.2)
    params = dict(model.named_parameters())
    c1 = repo.commit({W: d1}, _manifest("one"))
    with torch.no_grad():
        params[W].add_(d1)
    c2 = repo.commit({O_PROJ: d2}, _manifest("two"))
    with torch.no_grad():
        params[O_PROJ].add_(d2)
    assert [m.patch_id for m in repo.log()] == [c2.patch_id, c1.patch_id]
    cum = repo.cumulative_delta(c2.patch_id)
    assert torch.equal(cum[W], d1) and torch.equal(cum[O_PROJ], d2)

    base_params = dict(base.named_parameters())
    repo.checkout(c1.patch_id, model)
    assert repo.head() == c1.patch_id
    assert torch.allclose(params[O_PROJ], base_params[O_PROJ]) and torch.allclose(
        params[W], base_params[W] + d1
    )
    repo.checkout(ROOT, model)
    assert torch.allclose(params[W], base_params[W], atol=1e-7)

    repo.checkout(c2.patch_id, model)
    reverted = repo.revert(c2.patch_id, model)
    assert reverted.parent_patch_id == c2.patch_id and "Revert" in reverted.reasoner_rationale
    after = repo.cumulative_delta(repo.head())
    assert torch.allclose(after[O_PROJ], torch.zeros_like(d2)) and torch.equal(after[W], d1)
    assert torch.allclose(params[O_PROJ], base_params[O_PROJ])
    with pytest.raises(WeightGitError):
        repo.revert(ROOT)


def test_tampering_detected(repo, tmp_path, tiny_qwen):
    m = repo.commit({W: _delta(tiny_qwen, W, [0], 1.0)}, _manifest())
    obj = tmp_path / ".wgit" / "objects" / f"{m.patch_id}.COO.safetensors"
    raw = bytearray(obj.read_bytes())
    raw[-1] ^= 0xFF
    obj.write_bytes(bytes(raw))
    assert repo.verify(m.patch_id) is False
    with pytest.raises(WeightGitError, match="integrity"):
        repo.load_delta(m.patch_id)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ZZZZZZZZZZZZZZZZ", "abc", "0123456789abcdef0"])
def test_invalid_ids_rejected(repo, bad):
    with pytest.raises(WeightGitError):
        repo.load_manifest(bad)


def test_resolve_refs(repo, tiny_qwen):
    m = repo.commit({W: _delta(tiny_qwen, W, [0], 1.0)}, _manifest())
    assert repo.resolve("HEAD") == repo.resolve("main") == repo.resolve(m.patch_id[:6]) == m.patch_id
    with pytest.raises(WeightGitError, match="unknown"):
        repo.resolve("ffff" if not m.patch_id.startswith("ffff") else "eeee")
    with pytest.raises(WeightGitError, match="empty delta"):
        repo.commit({}, _manifest())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_coo_round_trip(dtype):
    d = torch.zeros(4, 6, dtype=dtype)
    d[1, 2], d[3, 5] = 1.5, -2.0
    idx, vals, shape = to_coo(d)
    assert vals.dtype == dtype and idx.shape == (2, 2)
    assert torch.equal(from_coo(idx, vals, shape), d)
    zero = torch.zeros(3)
    assert torch.equal(from_coo(*to_coo(zero)), zero)
    with pytest.raises(ValueError, match="0-dim"):
        to_coo(torch.tensor(1.0))


def test_exports_leave_model_unchanged(tmp_path, tiny_qwen):
    d = {W: _delta(tiny_qwen, W, [4], 0.3)}
    before = {k: v.clone() for k, v in tiny_qwen.state_dict().items()}
    out = export_merged_safetensors(tiny_qwen, d, tmp_path / "merged.safetensors", {"patch": "p"})
    assert all(torch.equal(before[k], v) for k, v in tiny_qwen.state_dict().items())
    fresh = build_tiny_qwen3(seed=0)
    load_model(fresh, str(out))
    assert torch.allclose(dict(fresh.named_parameters())[W], before[W] + d[W])

    sparse = export_sparse_delta(d, tmp_path / "delta.safetensors", {"base": "toy"})
    assert torch.equal(load_sparse_delta(sparse)[W], d[W])
    with safe_open(str(sparse), framework="pt") as f:
        assert f.metadata() == {"base": "toy"}
    with pytest.raises(ValueError, match="save_pretrained"):
        export_hf_pretrained(tiny_qwen, d, tmp_path / "hf")


def test_patch_stats_and_fingerprint(tiny_qwen):
    params = dict(tiny_qwen.named_parameters())
    d = _delta(tiny_qwen, W, [0, 1], 0.1)
    s = tensor_delta_stats(W, params[W], d)
    assert s.nnz_count == 2 * d.shape[0] and s.total_elements == d.numel()
    assert s.sparsity_ratio == pytest.approx(2 / d.shape[1])
    assert s.relative_frobenius == pytest.approx(s.frobenius_norm_delta / s.frobenius_norm_base)
    total = sum(p.numel() for p in tiny_qwen.parameters())
    stats, modified, ratio = summarize_patch(params, {W: d}, total)
    assert modified == s.nnz_count and ratio == pytest.approx(modified / total) and len(stats) == 1
    fp = state_fingerprint(tiny_qwen)
    assert fp == state_fingerprint(build_tiny_qwen3(seed=0)) and len(fp) == 64
    with torch.no_grad():
        params[W][0, 0] += 1e-3
    assert state_fingerprint(tiny_qwen) != fp
