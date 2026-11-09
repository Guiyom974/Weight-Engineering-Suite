"""Air-gapped anchor vault tests: real subprocess over JSON-RPC, plus in-process protocol/validation tests."""

import io
import json

import pytest
import torch
from safetensors.torch import save_file

from libweightspace.cow_overlay import CoWWeightOverlay
from libweightspace.toy_models import build_tiny_qwen3
from wrsi.anchor_client import AnchorVaultClient, AnchorVaultError, anchor_ids_from_texts, save_base_weights
from wrsi.anchor_vault_server import (
    MAX_LINE_BYTES,
    AnchorEvaluator,
    AnchorVaultServer,
    build_anchor_vault,
    resolve_factory,
    serve,
)

from .conftest import NEUTRAL_PROMPTS

TARGET = "model.layers.1.mlp.down_proj.weight"


@pytest.fixture
def vault_files(tmp_path, tokenizer):
    m0 = build_tiny_qwen3(seed=0)
    vault = build_anchor_vault(
        m0, anchor_ids_from_texts(tokenizer, NEUTRAL_PROMPTS), tmp_path / "vault" / "anchors.safetensors"
    )
    base = save_base_weights(m0, tmp_path / "vault" / "m0.safetensors")
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    return vault, base, exchange


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """One real subprocess shared by the IPC tests (startup imports torch, which takes a few seconds)."""
    from libweightspace.toy_models import CharTokenizer

    root = tmp_path_factory.mktemp("anchor")
    m0 = build_tiny_qwen3(seed=0)
    vault = build_anchor_vault(
        m0, anchor_ids_from_texts(CharTokenizer(), NEUTRAL_PROMPTS), root / "anchors.safetensors"
    )
    base = save_base_weights(m0, root / "m0.safetensors")
    with AnchorVaultClient(vault, base, max_kl=0.05, timeout=120) as c:
        yield c


def test_unmodified_model_passes_with_zero_kl(client):
    passed, kl = client.verify(build_tiny_qwen3(seed=0))
    assert passed is True and kl == pytest.approx(0.0, abs=1e-6)


def test_perturbed_model_fails_tether(client):
    model = build_tiny_qwen3(seed=0)
    w = dict(model.named_parameters())[TARGET]
    with CoWWeightOverlay(model).apply_delta({TARGET: torch.randn_like(w) * 2.0}) as forked:
        passed, kl = client.verify(forked, tensor_names=[TARGET])
        passed_full, kl_full = client.verify(forked)
    assert passed is False and kl > 0.05 and kl == pytest.approx(kl_full, rel=1e-5) and not passed_full


def test_info_never_reveals_anchors(client):
    info = client.info()
    assert info == {"num_anchors": len(NEUTRAL_PROMPTS), "max_kl": 0.05}
    assert "input_ids" not in json.dumps(info)


def test_malicious_requests_rejected_and_server_survives(client):
    for bad in [
        "../anchors.safetensors",
        "..\\m0.safetensors",
        "C:/Windows/win.ini",
        "x.txt",
        "",
        "missing.safetensors",
    ]:
        with pytest.raises(AnchorVaultError):
            client.request("verify_kl_tether", {"weights_file": bad})
    evil = client.exchange_dir / "evil.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(3, 3)}, str(evil))
    with pytest.raises(AnchorVaultError, match="wrong shape"):
        client.request("verify_kl_tether", {"weights_file": "evil.safetensors"})
    save_file({"not.a.param": torch.zeros(2)}, str(evil))
    with pytest.raises(AnchorVaultError, match="unknown tensor"):
        client.request("verify_kl_tether", {"weights_file": "evil.safetensors"})
    evil.write_bytes(b"\x00garbage")
    with pytest.raises(AnchorVaultError, match="not a valid safetensors"):
        client.request("verify_kl_tether", {"weights_file": "evil.safetensors"})
    with pytest.raises(AnchorVaultError, match="method not found"):
        client.request("read_anchors")
    assert client.request("ping") == "pong"  # still alive


def test_make_kl_fn_and_unknown_tensor(client):
    fn = client.make_kl_fn(lambda: [TARGET])
    assert fn(build_tiny_qwen3(seed=0))[0] is True
    with pytest.raises(KeyError):
        client.verify(build_tiny_qwen3(seed=0), tensor_names=["nope"])


def test_client_close_stops_process(vault_files):
    vault, base, _ = vault_files
    c = AnchorVaultClient(vault, base, timeout=120)
    exchange = c.exchange_dir
    c.close()
    assert c._proc.poll() is not None and not exchange.exists()
    with pytest.raises(AnchorVaultError, match="exited"):
        c.request("ping")


def test_bad_startup_raises(tmp_path, vault_files):
    vault, _, _ = vault_files
    with pytest.raises(AnchorVaultError):
        AnchorVaultClient(vault, tmp_path / "missing.safetensors", timeout=60)


# --- in-process protocol tests ------------------------------------------------------------------------


def test_serve_protocol_errors(vault_files):
    vault, base, exchange = vault_files
    server = AnchorVaultServer(vault, base, exchange)
    big = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping", "pad": "x" * MAX_LINE_BYTES})
    lines = (
        "\n".join(
            [
                "not json",
                big,
                json.dumps({"jsonrpc": "1.0", "id": 1, "method": "ping"}),
                json.dumps([1, 2]),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "verify_kl_tether", "params": [1]}),
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
                json.dumps({"jsonrpc": "2.0", "id": 4, "method": "shutdown"}),
                json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping"}),
            ]
        )
        + "\n"
    )
    out = io.StringIO()
    serve(server, io.StringIO(lines), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    codes = [r.get("error", {}).get("code") for r in responses]
    assert codes == [-32700, -32600, -32600, -32600, -32602, None, None]
    assert responses[5]["result"] == "pong" and responses[6]["result"] == "bye"  # nothing after shutdown
    assert all("Traceback" not in json.dumps(r) for r in responses)


def test_evaluator_and_factory_validation():
    ref = {"a": torch.log_softmax(torch.randn(3, 5), dim=-1)}
    ok, kl = AnchorEvaluator(ref, 0.01).verify_kl_tether({"a": ref["a"].clone()})
    assert ok and kl == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        resolve_factory("os.system")
    with pytest.raises(ValueError):
        resolve_factory("os:system; rm -rf /")
    assert callable(resolve_factory("libweightspace.toy_models:build_tiny_qwen3"))


def test_vault_sha256_pinning_detects_tampering(vault_files):
    from wrsi.anchor_vault_server import file_sha256

    vault, base, exchange = vault_files
    good_vault_sha = file_sha256(vault)
    good_base_sha = file_sha256(base)
    server = AnchorVaultServer(
        vault, base, exchange, vault_sha256=good_vault_sha, base_weights_sha256=good_base_sha
    )
    assert len(server.inputs) == len(NEUTRAL_PROMPTS)

    bad_sha = "0" * 64
    with pytest.raises(ValueError, match="vault SHA-256 mismatch"):
        AnchorVaultServer(vault, base, exchange, vault_sha256=bad_sha)
    with pytest.raises(ValueError, match="base weights SHA-256 mismatch"):
        AnchorVaultServer(vault, base, exchange, base_weights_sha256=bad_sha)
    with pytest.raises(ValueError, match="64-character"):
        AnchorVaultServer(vault, base, exchange, vault_sha256="not-hex")
