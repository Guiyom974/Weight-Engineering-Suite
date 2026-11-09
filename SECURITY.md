# Security policy

## Reporting a vulnerability

Please open a private security advisory on this repository (GitHub: Security, then "Report a vulnerability") with
steps to reproduce. Do not open a public issue for vulnerabilities.

## Threat model in short

| Boundary | Risk | Protection |
| --- | --- | --- |
| Browser to the local Studio | Another web page tries to drive the loopback server (drive-by requests, DNS rebinding) | Loopback bind by default; `Host` must be the Studio's own address; `Origin` must match when present; no CORS headers; per-session token in `X-Studio-Token` on every POST; JSON only; body size cap; nonce-based CSP |
| Studio routes | A request chooses where files are written or what code runs | No route accepts a path, URL or code; workspaces and model folders are chosen by id from the server's own lists; exports go to `<workspace>/exports/`, packages to `<workspace>/packages/` |
| Hugging Face downloads | Pickle weights, tampered or oversized files, hostile file names, token leaks | Only top-level safetensors weights and config/tokenizer files; declared size cap and SHA-256 (LFS) or git SHA-1 check per file; partial folder removed on cancel or failure; https only (plain http only on loopback, for tests); `HF_TOKEN` sent only to the Hub host and dropped on redirects to another host |
| Factory checks (model-written code and SQL) | Code escapes to the network, other processes, files or secrets; a fake pass | `python -I -S` child in an empty folder with a minimal environment (no keys); audit hook blocks sockets, subprocesses, `ctypes`, writes and reads outside its folder and frame introspection; Windows job object (memory limit, one process, kill on close) or POSIX `RLIMIT_AS`; timeout; a random completion marker so early exits fail; SQL runs in memory with `ATTACH`, `PRAGMA` and extension loading denied |
| Packaged tools | Request text injected into generated code | Fixed templates; suite text goes only into `tool_config.json` (tested) |
| LLM endpoint | Key leaks to the wrong host; hostile replies | Key read from an environment variable only; the Poe key is only sent to `api.poe.com`; other hosts need an explicitly named key variable; never over plain HTTP to a non-local host; timeouts and size limits on replies |
| LLM-generated edit scripts | Prompt-injected or hostile code | Allow-list sandbox: only tensor math is reachable (no modules, files, network or imports), inputs are cloned, a line budget stops loops; edits are committed only after measured gates pass |
| Model downloads | Remote code in model repositories | `trust_remote_code=False`; LFM 3 runs through vendored, reviewed modeling code; pin revisions with `org/model@<sha>` |

## Known limits

- The sandbox runs in the same process. It cannot limit memory or CPU used inside native torch kernels, and a
  memory-safety bug in a kernel would bypass it. If you point the tool at an endpoint you do not trust, run it in a
  container or a virtual machine.
- The factory's check runner relies on Python audit hooks, which native extension code can bypass; the job object or
  rlimit and the timeout still apply. Run task suites from sources you do not trust in a container.
- Licences shown by the workspace wizard come from the model card metadata; checking them remains your responsibility.
- `--host 0.0.0.0` (or any non-loopback address) exposes the Studio to your network. Anyone who can load the page
  can edit weights in that workspace. The CLI prints a warning when you do this.
- Integrity hashes (SHA-256) protect patches against accidental corruption; manifests are not signed.
