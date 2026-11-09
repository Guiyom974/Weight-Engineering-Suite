"""Tests for the workspace and model manager (CR2): home listing, Hub checks, verified downloads, path refusal.

All Hub traffic goes to ``tests.fake_hub.FakeHub`` on loopback; nothing here touches the network or the real
``~/.weightlab`` (conftest points ``WEIGHTLAB_HOME`` at a temporary folder).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from neuroforge.workspace_manager import (
    DownloadCancelled,
    HubClient,
    StudioHome,
    WorkspaceManagerError,
    _ScopedRedirect,
    check_model,
    create_workspace,
    download_model,
)

from .fake_hub import SHA, FakeHub, FakeRepo


def test_redirects_keep_the_token_on_the_hub_over_https_only() -> None:
    handler = _ScopedRedirect("huggingface.co", "hf_test_secret", "https")
    req = urllib.request.Request("https://huggingface.co/api/models/org/name")
    same = handler.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/org/name/resolve/x")
    assert same is not None and same.unredirected_hdrs.get("Authorization") == "Bearer hf_test_secret"
    cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.huggingface.co/x")
    assert cdn is not None and "Authorization" not in cdn.unredirected_hdrs
    # A downgrade to plain http (same host or not) is refused: the token and the file would travel in clear text.
    with pytest.raises(urllib.error.HTTPError, match="plain http"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://huggingface.co/org/name/resolve/x")

GB = 1024**3
PLENTY = {"devices": [{"device": "cpu", "free_bytes": 64 * GB}, {"device": "cuda", "free_bytes": 64 * GB}]}


def _repo(**kw: object) -> FakeRepo:
    files = {
        "config.json": json.dumps({"model_type": "llama"}).encode(),
        "tokenizer.json": b'{"version": "1.0"}',
        "model.safetensors": b"\x00" * 4096,
        "README.md": b"# card",
    }
    return FakeRepo(files=files, lfs={"model.safetensors"}, **kw)  # type: ignore[arg-type]


@pytest.fixture()
def hub() -> Iterator[FakeHub]:
    fake = FakeHub({"org/tiny": _repo(), "org/pickle": FakeRepo(files={"config.json": b"{}", "pytorch_model.bin": b"x"})})
    try:
        yield fake
    finally:
        fake.close()


def test_check_reports_size_memory_licence_and_revision_before_download(hub: FakeHub, tmp_path: Path) -> None:
    check = check_model(HubClient(hub.url), "org/tiny", "main", "cpu", memory=PLENTY)
    assert check.revision == SHA and check.license == "apache-2.0" and check.adapter == "qwen3_8"
    assert check.parameters == 1000 and check.weights_bytes_bf16 == 2000 and check.needed_bytes == 4000
    assert {f.name for f in check.files} == {"config.json", "tokenizer.json", "model.safetensors"}  # README skipped
    assert check.download_bytes == sum(f.size for f in check.files)
    assert check.warnings == []
    assert not any("/resolve/" in h.get("Host", "") for h in hub.hub_seen.headers)
    assert not list((tmp_path).glob("**/*.safetensors"))  # nothing downloaded by a check


def test_memory_warning_suggests_cpu_or_smaller_model(hub: FakeHub) -> None:
    tight = {"devices": [{"device": "cpu", "free_bytes": 10 * GB}, {"device": "cuda", "free_bytes": 1000}]}
    check = check_model(HubClient(hub.url), "org/tiny", "main", "cuda", memory=tight)
    assert any("Editing needs about" in w for w in check.warnings)
    assert check.suggestion and "CPU" in check.suggestion
    both_tight = {"devices": [{"device": "cpu", "free_bytes": 100}]}
    check = check_model(HubClient(hub.url), "org/tiny", "main", "cpu", memory=both_tight)
    assert check.suggestion and "smaller model" in check.suggestion


def test_unsupported_architecture_is_explained(tmp_path: Path) -> None:
    fake = FakeHub({"org/odd": _repo(model_type="t5")})
    try:
        check = check_model(HubClient(fake.url), "org/odd", "main", "cpu", memory=PLENTY)
    finally:
        fake.close()
    assert check.adapter is None and any("Unsupported architecture" in w for w in check.warnings)


def test_pickle_only_repo_is_refused(hub: FakeHub) -> None:
    with pytest.raises(WorkspaceManagerError, match="no safetensors"):
        check_model(HubClient(hub.url), "org/pickle", "main", "cpu", memory=PLENTY)


def test_bad_ids_and_endpoints_are_refused(hub: FakeHub) -> None:
    for repo in ("../etc", "org/../x", "/abs/path", "org", "org/name with space"):
        with pytest.raises(WorkspaceManagerError):
            check_model(HubClient(hub.url), repo, "main", "cpu", memory=PLENTY)
    with pytest.raises(WorkspaceManagerError, match="https"):
        HubClient("http://example.com")


def test_download_verifies_and_pins_revision(hub: FakeHub, tmp_path: Path) -> None:
    home = StudioHome(tmp_path / "home")
    client = HubClient(hub.url)
    check = check_model(client, "org/tiny", "main", "cpu", memory=PLENTY)
    seen: list[float] = []
    folder = download_model(client, check, home.models_dir, progress=lambda _t, f: seen.append(f))
    assert folder.name == f"org--tiny--{SHA[:12]}" and (folder / "model.safetensors").stat().st_size == 4096
    assert not (folder / "README.md").exists()
    assert seen and seen[-1] == pytest.approx(1.0)
    source = json.loads((folder / ".weightlab-source.json").read_text(encoding="utf-8"))
    assert source["revision"] == SHA and source["license"] == "apache-2.0"
    assert [r["name"] for r in home.list_model_folders()] == [folder.name]
    assert not list(home.models_dir.glob(".partial-*"))


def test_download_rejects_hash_mismatch_and_cleans_up(tmp_path: Path) -> None:
    fake = FakeHub({"org/bad": _repo(bad_hash={"model.safetensors"})})
    try:
        client = HubClient(fake.url)
        check = check_model(client, "org/bad", "main", "cpu", memory=PLENTY)
        with pytest.raises(WorkspaceManagerError, match="SHA-256"):
            download_model(client, check, tmp_path / "models")
    finally:
        fake.close()
    assert list((tmp_path / "models").iterdir()) == []


def test_cancelled_download_removes_partial_files(hub: FakeHub, tmp_path: Path) -> None:
    client = HubClient(hub.url)
    check = check_model(client, "org/tiny", "main", "cpu", memory=PLENTY)
    calls = {"n": 0}

    def cancel_after_first_chunk() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(DownloadCancelled):
        download_model(client, check, tmp_path / "models", should_cancel=cancel_after_first_chunk)
    assert list((tmp_path / "models").iterdir()) == []


def test_token_is_sent_to_the_hub_but_never_to_the_cdn(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
    client = HubClient(hub.url)
    check = check_model(client, "org/tiny", "main", "cpu", memory=PLENTY)
    download_model(client, check, tmp_path / "models")
    assert all(h.get("Authorization") == "Bearer hf_test_secret_token" for h in hub.hub_seen.headers)
    assert hub.cdn_seen.headers and all("Authorization" not in h for h in hub.cdn_seen.headers)


def test_hostile_hub_file_names_are_never_written(tmp_path: Path) -> None:
    repo = _repo(extra_siblings=[
        {"rfilename": "../evil.safetensors", "size": 1, "blobId": "a" * 40},
        {"rfilename": "sub/inner.json", "size": 1, "blobId": "b" * 40},
        {"rfilename": "C:\\x.json", "size": 1, "blobId": "c" * 40},
    ])
    fake = FakeHub({"org/hostile": repo})
    try:
        client = HubClient(fake.url)
        check = check_model(client, "org/hostile", "main", "cpu", memory=PLENTY)
        assert {f.name for f in check.files} == {"config.json", "tokenizer.json", "model.safetensors"}
        download_model(client, check, tmp_path / "models")
    finally:
        fake.close()
    assert not (tmp_path / "evil.safetensors").exists() and not (tmp_path / "models" / "evil.safetensors").exists()


def test_home_lists_recent_and_home_workspaces_by_opaque_id(tmp_path: Path) -> None:
    home = StudioHome(tmp_path / "home")
    demo = create_workspace(home, "demo1", "toy:qwen3_8", demo=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "neuroforge.yaml").write_text(yaml.safe_dump({"model": "toy:gemma4"}), encoding="utf-8")
    home.remember(outside)
    rows = home.list_workspaces()
    assert [r["name"] for r in rows] == ["elsewhere", "demo1"]
    assert all(len(r["id"]) == 16 and "path" not in r for r in rows)
    assert home.workspace_by_id(rows[1]["id"]) == demo.resolve()
    cfg = yaml.safe_load((demo / "neuroforge.yaml").read_text(encoding="utf-8"))
    assert cfg["demo_defect"] is True and (demo / "intent.yaml").is_file()


def test_paths_outside_home_are_refused(tmp_path: Path) -> None:
    home = StudioHome(tmp_path / "home")
    home.ensure()
    for bad in ("../secrets", "..", "/etc", "C:\\Windows", "a/b", "", "x" * 300):
        with pytest.raises(WorkspaceManagerError):
            home.model_folder(bad)
    with pytest.raises(WorkspaceManagerError):
        home.workspace_by_id("../../etc/passwd")
    with pytest.raises(WorkspaceManagerError):
        home.workspace_by_id("0" * 16)
    for bad_name in ("../x", "a/b", "", "con fig"):
        with pytest.raises(WorkspaceManagerError):
            create_workspace(home, bad_name, "toy:qwen3_8")
    with pytest.raises(WorkspaceManagerError, match="inside"):
        create_workspace(home, "ok", "org/x", model_path=tmp_path)
    assert not (home.workspaces_dir / "ok").exists()
