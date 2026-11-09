"""A local fake of the Hugging Face Hub HTTP API for offline tests (metadata, file resolve, CDN redirects).

``FakeHub`` serves ``/api/models/<repo>/revision/<rev>?blobs=true`` and ``/<repo>/resolve/<sha>/<file>`` on a loopback
port. Files can be redirected to a second server (``FakeHub.cdn``) that records the headers it received, so tests can
check that the token never leaves the Hub host. Everything binds to 127.0.0.1 and is torn down by ``close``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

SHA = "0123456789abcdef0123456789abcdef01234567"


def git_sha1(data: bytes) -> str:
    """Git blob id of ``data``."""
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


@dataclass
class FakeRepo:
    """A repository: files with content, optional overrides for what the metadata declares."""

    files: dict[str, bytes]
    lfs: set[str] = field(default_factory=set)
    params: int | None = 1000
    model_type: str = "llama"
    license: str = "apache-2.0"
    gated: bool = False
    bad_hash: set[str] = field(default_factory=set)
    extra_siblings: list[dict[str, Any]] = field(default_factory=list)


class _Recorder:
    def __init__(self) -> None:
        self.headers: list[dict[str, str]] = []


def _server(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class FakeHub:
    """Fake Hub plus a fake CDN; ``url`` is the endpoint to pass to ``HubClient``."""

    def __init__(
        self,
        repos: dict[str, FakeRepo],
        redirect_lfs: bool = True,
        slow_bytes_per_read: int | None = None,
        delay_per_chunk: float = 0.0,
    ):
        self.repos = repos
        self.redirect_lfs = redirect_lfs
        self.slow = slow_bytes_per_read
        self.delay = delay_per_chunk
        self.hub_seen = _Recorder()
        self.cdn_seen = _Recorder()
        hub = self

        class CDN(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                hub.cdn_seen.headers.append(dict(self.headers.items()))
                repo, _, name = unquote(urlparse(self.path).path).lstrip("/").rpartition("/")
                hub._send_file(self, repo, name)

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                hub.hub_seen.headers.append(dict(self.headers.items()))
                path = unquote(urlparse(self.path).path)
                if path.startswith("/api/models/"):
                    repo = path[len("/api/models/"):].split("/revision/")[0]
                    hub._send_metadata(self, repo)
                    return
                if "/resolve/" in path:
                    repo, rest = path.lstrip("/").split("/resolve/", 1)
                    name = rest.split("/", 1)[1]
                    if name in hub.repos.get(repo, FakeRepo({})).lfs and hub.redirect_lfs:
                        self.send_response(302)
                        self.send_header("Location", f"{hub.cdn_url}/{repo}/{name}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    hub._send_file(self, repo, name)
                    return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._cdn = _server(CDN)
        self._hub = _server(Hub)
        self.cdn_url = f"http://127.0.0.1:{self._cdn.server_address[1]}"
        self.url = f"http://127.0.0.1:{self._hub.server_address[1]}"

    def _send_metadata(self, handler: BaseHTTPRequestHandler, repo: str) -> None:
        spec = self.repos.get(repo)
        if spec is None:
            handler.send_response(404)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        siblings: list[dict[str, Any]] = []
        for name, data in spec.files.items():
            declared = data if name not in spec.bad_hash else data + b"tampered"
            entry: dict[str, Any] = {"rfilename": name, "size": len(data)}
            if name in spec.lfs:
                entry["lfs"] = {"sha256": hashlib.sha256(declared).hexdigest(), "size": len(data), "pointerSize": 134}
                entry["blobId"] = "f" * 40
            else:
                entry["blobId"] = git_sha1(declared)
            siblings.append(entry)
        siblings += spec.extra_siblings
        body = json.dumps({
            "id": repo, "sha": SHA, "siblings": siblings, "gated": spec.gated,
            "cardData": {"license": spec.license},
            "config": {"architectures": ["X"], "model_type": spec.model_type},
            "safetensors": {"total": spec.params} if spec.params else None,
        }).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _send_file(self, handler: BaseHTTPRequestHandler, repo: str, name: str) -> None:
        data = self.repos.get(repo, FakeRepo({})).files.get(name)
        if data is None:
            handler.send_response(404)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        step = self.slow or len(data) or 1
        try:
            for i in range(0, len(data), step):
                handler.wfile.write(data[i:i + step])
                handler.wfile.flush()
                if self.delay:
                    time.sleep(self.delay)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # the client cancelled mid-file

    def close(self) -> None:
        """Stop both servers."""
        for srv in (self._hub, self._cdn):
            srv.shutdown()
            srv.server_close()
