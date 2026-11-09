"""Copy-on-Write overlay tests (PRD_01 section 5.2, blueprint step 4)."""

import pytest
import torch

from libweightspace.cow_overlay import CoWWeightOverlay, apply_delta_with_backup

TARGET = "model.layers.1.mlp.down_proj.weight"


def _sparse_delta(model, cols=(3, 9)):
    w = dict(model.named_parameters())[TARGET]
    delta = torch.zeros_like(w)
    for c in cols:
        delta[:, c] = 0.5
    return delta


def test_delta_changes_output_then_restores_bitwise(tiny_qwen, tokenizer):
    overlay = CoWWeightOverlay(tiny_qwen)
    inputs = tokenizer("copy on write")
    w_before = overlay.parameter(TARGET).detach().clone()
    out_before = tiny_qwen(**inputs).logits.clone()
    with overlay.apply_delta({TARGET: _sparse_delta(tiny_qwen)}) as forked:
        out_inside = forked(**inputs).logits
        assert not torch.allclose(out_inside, out_before)
        assert not torch.equal(overlay.parameter(TARGET), w_before)
    assert torch.equal(overlay.parameter(TARGET), w_before)
    assert torch.equal(tiny_qwen(**inputs).logits, out_before)


def test_restores_bitwise_when_body_raises(tiny_qwen):
    overlay = CoWWeightOverlay(tiny_qwen)
    w_before = overlay.parameter(TARGET).detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with overlay.apply_delta({TARGET: _sparse_delta(tiny_qwen)}):
            raise RuntimeError("boom")
    assert torch.equal(overlay.parameter(TARGET), w_before)


def test_partial_application_rolled_back_on_bad_name(tiny_qwen):
    overlay = CoWWeightOverlay(tiny_qwen)
    w_before = overlay.parameter(TARGET).detach().clone()
    deltas = {TARGET: _sparse_delta(tiny_qwen), "model.layers.9.nope.weight": torch.zeros(1)}
    with pytest.raises(KeyError, match="not found"):
        with overlay.apply_delta(deltas):
            pass
    assert torch.equal(overlay.parameter(TARGET), w_before)


def test_only_touched_columns_are_backed_up(tiny_qwen):
    param = dict(tiny_qwen.named_parameters())[TARGET]
    backup = apply_delta_with_backup(param, TARGET, _sparse_delta(tiny_qwen, cols=(1, 2, 7)))
    assert backup is not None and backup.values.shape == (param.shape[0], 3)
    backup.restore(param)
    assert apply_delta_with_backup(param, TARGET, torch.zeros_like(param)) is None


def test_one_dimensional_parameter(tiny_qwen):
    overlay = CoWWeightOverlay(tiny_qwen)
    name = "model.norm.weight"
    before = overlay.parameter(name).detach().clone()
    with overlay.apply_delta({name: torch.full_like(before, 0.25)}):
        assert torch.allclose(overlay.parameter(name), before + 0.25)
    assert torch.equal(overlay.parameter(name), before)


@pytest.mark.parametrize(
    "bad, exc, msg",
    [
        (torch.zeros(3, 3), ValueError, "has shape"),
        ("not a tensor", TypeError, "must be a torch.Tensor"),
    ],
)
def test_rejects_malformed_deltas(tiny_qwen, bad, exc, msg):
    overlay = CoWWeightOverlay(tiny_qwen)
    with pytest.raises(exc, match=msg):
        with overlay.apply_delta({TARGET: bad}):
            pass


def test_rejects_non_finite_delta(tiny_qwen):
    overlay = CoWWeightOverlay(tiny_qwen)
    bad = _sparse_delta(tiny_qwen)
    bad[0, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        with overlay.apply_delta({TARGET: bad}):
            pass
