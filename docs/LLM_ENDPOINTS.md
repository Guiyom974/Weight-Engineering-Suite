# Connecting an LLM endpoint

The suite talks to one kind of API: OpenAI-compatible chat completions (`POST {base_url}/chat/completions`) and
model listing (`GET {base_url}/models`). That covers hosted aggregators, providers and local servers.

## Settings and where they come from

| Setting | Flag | Environment variable | Workspace (`neuroforge.yaml`, `llm:` section) | Default |
| --- | --- | --- | --- | --- |
| Base URL | `--base-url` | `WEIGHTLAB_LLM_BASE_URL` | `base_url` | `https://api.poe.com/v1` |
| Key variable name | `--key-env` | `WEIGHTLAB_LLM_KEY_ENV` | `api_key_env` | `POE_API_KEY`, for Poe only |
| Model id | `--model` (`--proposer-model` for `wrsi`) | `WEIGHTLAB_LLM_MODEL` | `model` | none: pick one |

Flags win over environment variables, which win over the workspace file. The Studio's model dropdown writes
`llm.model` into the workspace file. The workspace file never holds a key: an `api_key` entry is rejected.

## Example environment

Put the lines you need in your shell profile (do not commit them):

```bash
# Poe (recommended): one key, hundreds of models
export POE_API_KEY="your-poe-api-key"

# Another OpenAI-compatible endpoint: base URL plus the NAME of the variable holding its key
export WEIGHTLAB_LLM_BASE_URL="https://openrouter.ai/api/v1"
export WEIGHTLAB_LLM_KEY_ENV="OPENROUTER_API_KEY"
export OPENROUTER_API_KEY="your-openrouter-key"

# Optional default model
export WEIGHTLAB_LLM_MODEL="some-model-id"
```

Windows PowerShell uses `$env:POE_API_KEY = "your-poe-api-key"` (or set it in System Properties for persistence).

## Why Poe is the default

Poe exposes many frontier and open models behind one OpenAI-compatible endpoint and one key, so you can compare
proposers (for example a large reasoning model against a small fast one) without new accounts. The model list in
the Studio comes live from your key, so you always see what is actually available to you.

## Safety rules

- The default key variable `POE_API_KEY` is only sent to `api.poe.com`. If you change the base URL, name the key
  variable for that host explicitly, otherwise requests go without a key. This stops a Poe key from leaking to an
  arbitrary server.
- Keys are never sent over plain `http://` to a host other than `127.0.0.1`, `localhost` or `::1`.
- Keys are never logged, printed, returned by the Studio API or written to disk.
- Every request has a timeout and a size limit on the reply. Replies are parsed as data; edit scripts they contain
  run only in the sandbox and are committed only after measured gates pass.

## Checking a setup

```bash
neuroforge doctor            # endpoint, key presence (not the key), model count
neuroforge models            # the live list
```

Common results: `no-key` (the key variable is not set), `unauthorized` (the server rejected the key),
`unreachable` (network, proxy or wrong URL), `bad-response` (the URL is not an OpenAI-compatible `/v1` base).
