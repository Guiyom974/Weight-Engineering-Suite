"""Tests for candidate parsing, LLM backends, agent tools, reasoner prompts and the intent compiler."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from libweightspace.adapters import Qwen3Adapter
from libweightspace.agent_tools import WEIGHT_SPACE_TOOLS, WeightSpaceToolbox, tools_as_prompt_text
from libweightspace.candidates import (
    CandidateParseError,
    ProposalContext,
    TemplateCandidateSource,
    extract_json,
    nullification_script,
    parse_candidates_json,
)
from libweightspace.llm_backends import (
    CallableBackend,
    LLMBackendError,
    OpenAICompatibleBackend,
    make_backend,
)
from libweightspace.sandbox import validate_script
from neuroforge.intent_compiler import contrastive_texts, draft_intent, load_blueprint, load_intent
from neuroforge.reasoner_bridge import NEUROFORGE_SYSTEM_PROMPT, ReasonerBridge, build_reasoner_prompt

from .conftest import FAIL_PROMPTS, PASS_PROMPTS

# --- candidates ------------------------------------------------------------------------------------------


def test_parse_fenced_list_and_single_object():
    text = 'blah\n```json\n{"candidates": [{"rationale": "r", "python_script": "def compute_delta(w, d):\\n  return {}"}]}\n```'
    (c,) = parse_candidates_json(text, source="t")
    assert c.hypothesis == "r" and c.source == "t" and c.failure_taxonomy == "attractor_loop"
    raw = 'Sure! [{"hypothesis": "h", "failure_taxonomy": "expert_misrouting", "python_script": "x=1"}] done'
    (c2,) = parse_candidates_json(raw)
    assert c2.failure_taxonomy == "expert_misrouting"
    (c3,) = parse_candidates_json('{"python_script": "y=2", "failure_taxonomy": "bogus"}')
    assert c3.failure_taxonomy == "attractor_loop"


@pytest.mark.parametrize(
    "text, msg",
    [
        ("", "empty"),
        ("no json here", "no JSON"),
        ('[{"hypothesis": "h"}]', "no python_script"),
        ('[{"python_script": "' + "a" * 60_000 + '"}]', "oversized"),
        ("[1, 2]", "not an object"),
    ],
)
def test_parse_rejects_bad_payloads(text, msg):
    with pytest.raises(CandidateParseError, match=msg):
        parse_candidates_json(text)


def test_extract_json_skips_invalid_fence():
    assert extract_json('```json\n{bad}\n```\n then {"ok": 1}') == {"ok": 1}


def test_template_source_scripts_are_sandbox_valid_and_explore():
    src = TemplateCandidateSource()
    ctx = ProposalContext(
        iteration=0, diagnostics={"target_tensor": "t", "layer_idx": 1}, topology={}, num_candidates=2
    )
    first = src.propose(ctx)
    assert len(first) == 2 and all(validate_script(c.python_script) for c in first)
    ctx.history = [{"python_script": c.python_script} for c in first]
    second = src.propose(ctx)
    assert {c.python_script for c in second}.isdisjoint({c.python_script for c in first})
    assert src.propose(ProposalContext(iteration=0, diagnostics={}, topology={})) == []
    with pytest.raises(ValueError):
        nullification_script(0)


# --- backends --------------------------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if body["model"] == "fail":
            self.send_response(500)
            self.end_headers()
            return
        payload = {"choices": [{"message": {"content": "reply:" + body["messages"][1]["content"]}}]}
        if body["model"] == "malformed":
            payload = {"nope": True}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def chat_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_openai_compatible_backend(chat_server, monkeypatch):
    base = f"http://127.0.0.1:{chat_server.server_address[1]}/v1"
    monkeypatch.setenv("NF_TEST_KEY", "sk-secret")
    backend = OpenAICompatibleBackend(base, "local-model", api_key_env="NF_TEST_KEY", timeout=5)
    assert backend.complete("sys", "hello") == "reply:hello"
    seen = chat_server.seen[-1]
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-secret"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    with pytest.raises(LLMBackendError, match="HTTP 500") as exc:
        OpenAICompatibleBackend(base, "fail", api_key_env="NF_TEST_KEY", timeout=5).complete("s", "p")
    assert "sk-secret" not in str(exc.value)
    with pytest.raises(LLMBackendError, match="malformed"):
        OpenAICompatibleBackend(base, "malformed", timeout=5).complete("s", "p")


def test_a_missing_key_for_a_remote_host_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("NF_TEST_KEY", raising=False)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise OSError("no network in this test")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(LLMBackendError, match="NF_TEST_KEY is not set"):
        OpenAICompatibleBackend("https://api.example.invalid/v1", "m", api_key_env="NF_TEST_KEY").complete("s", "p")
    assert calls == []  # nothing was sent
    # A local server may run without a key: the request goes out.
    with pytest.raises(LLMBackendError, match="failed"):
        OpenAICompatibleBackend("http://127.0.0.1:9/v1", "m", api_key_env="NF_TEST_KEY").complete("s", "p")
    assert calls == ["http://127.0.0.1:9/v1/chat/completions"]


def test_openai_backend_validation_and_unreachable():
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleBackend("file:///etc/passwd", "m")
    with pytest.raises(ValueError, match="invalid model"):
        OpenAICompatibleBackend("http://127.0.0.1:1", "bad model; rm -rf")
    with pytest.raises(LLMBackendError, match="failed"):
        OpenAICompatibleBackend("http://127.0.0.1:9/v1", "m", timeout=2).complete("s", "p")


def test_make_backend(monkeypatch):
    monkeypatch.delenv("WEIGHTLAB_LLM_MODEL", raising=False)
    monkeypatch.delenv("WEIGHTLAB_LLM_BASE_URL", raising=False)
    backend = make_backend("llm", model="m", base_url="http://127.0.0.1:8080/v1")
    assert isinstance(backend, OpenAICompatibleBackend) and backend.base_url == "http://127.0.0.1:8080/v1"
    assert isinstance(make_backend("openai", model="m"), OpenAICompatibleBackend)  # alias kept
    with pytest.raises(ValueError, match="no LLM model selected"):
        make_backend("llm")
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("telepathy")
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("gemini")


# --- agent tools -----------------------------------------------------------------------------------------


class _Ledger:
    def query_relevant_history(self, domain, limit=5):
        return {"past_successes": [{"domain": domain}], "recent_failures_to_avoid": [], "limit": limit}


def test_tool_schemas_match_blueprint():
    names = [t["name"] for t in WEIGHT_SPACE_TOOLS]
    assert names == [
        "inspect_layer_svd",
        "trace_contrastive_activations",
        "query_episodic_ledger",
        "propose_weight_patch",
    ]
    assert (
        WEIGHT_SPACE_TOOLS[3]["parameters"]["properties"]["failure_taxonomy"]["enum"][-1]
        == "liquid_pole_instability"
    )
    assert json.loads(tools_as_prompt_text()) == WEIGHT_SPACE_TOOLS


def test_toolbox_dispatch(planted_qwen, tokenizer):
    model, defect = planted_qwen
    box = WeightSpaceToolbox(model, Qwen3Adapter(model), tokenizer, ledger=_Ledger())
    svd = box.dispatch("inspect_layer_svd", {"tensor_name": defect.tensor_name, "top_k": 4})
    assert len(svd["top_singular_values"]) == 4 and svd["frobenius_norm"] > 0
    trace = box.dispatch(
        "trace_contrastive_activations",
        json.dumps({"fail_prompts": FAIL_PROMPTS, "pass_prompts": PASS_PROMPTS}),
    )
    assert (
        trace["layers"][0]["layer_idx"] == 1 and box.last_diagnostics["target_tensor"] == defect.tensor_name
    )
    assert box.dispatch("query_episodic_ledger", {"domain": "math", "limit": 2})["limit"] == 2
    ok = box.dispatch(
        "propose_weight_patch",
        {"hypothesis": "h", "failure_taxonomy": "attractor_loop", "python_script": nullification_script(3)},
    )
    assert ok == {"accepted": True, "queued_candidates": 1}
    bad = box.dispatch(
        "propose_weight_patch",
        {
            "hypothesis": "h",
            "failure_taxonomy": "attractor_loop",
            "python_script": "import os\ndef compute_delta(w, d):\n  return {}",
        },
    )
    assert bad["accepted"] is False and len(box.proposals) == 1


@pytest.mark.parametrize(
    "name, args, msg",
    [
        ("inspect_layer_svd", {"tensor_name": "nope"}, "unknown 'tensor_name'"),
        ("inspect_layer_svd", {"tensor_name": "lm_head.weight", "top_k": 10_000}, "top_k"),
        ("trace_contrastive_activations", {"fail_prompts": [], "pass_prompts": ["a"]}, "fail_prompts"),
        (
            "trace_contrastive_activations",
            {"fail_prompts": ["a" * 5000], "pass_prompts": ["a"]},
            "fail_prompts",
        ),
        ("query_episodic_ledger", {"domain": ""}, "domain"),
        (
            "propose_weight_patch",
            {"hypothesis": "h", "failure_taxonomy": "magic", "python_script": "x"},
            "taxonomy",
        ),
        ("teleport", {}, "unknown tool"),
    ],
)
def test_toolbox_rejects_bad_arguments(tiny_qwen, tokenizer, name, args, msg):
    box = WeightSpaceToolbox(tiny_qwen, Qwen3Adapter(tiny_qwen), tokenizer, ledger=_Ledger())
    assert msg in box.dispatch(name, args)["error"]


def test_toolbox_without_ledger_and_bad_json(tiny_qwen, tokenizer):
    box = WeightSpaceToolbox(tiny_qwen, Qwen3Adapter(tiny_qwen), tokenizer)
    assert "no episodic ledger" in box.dispatch("query_episodic_ledger", {"domain": "d"})["error"]
    assert "not valid JSON" in box.dispatch("inspect_layer_svd", "{oops")["error"]


# --- reasoner prompt & intent compiler -------------------------------------------------------------------


def test_reasoner_prompt_sections_and_bridge():
    ctx = ProposalContext(
        iteration=1,
        diagnostics={
            "layers": [
                {
                    "layer_idx": 3,
                    "divergence_score": 2.5,
                    "target_tensor": "t",
                    "singular_values_top5": [3.0, 2.0],
                    "top_culpable_mlp_cols": [1, 2],
                    "top_mlp_col_alignments": [0.9, -0.8],
                }
            ]
        },
        topology={"architecture": "qwen3_8", "num_layers": 4, "d_model": 32, "d_intermediate": 64},
        history=[
            {"iteration": 0, "outcome": "invariant violation", "target_score": -1.0, "hypothesis": "h0"}
        ],
        num_candidates=2,
    )
    prompt = build_reasoner_prompt(ctx)
    for section in (
        "### 1. Model Topology Summary",
        "### 2. Mechanistic Attribution Table",
        "### 3. Previous Iteration History",
        "### 4. Output Contract",
    ):
        assert section in prompt
    assert (
        "| 3 | 2.5000 | t |" in prompt and "invariant violation" in prompt and "exactly 2 distinct" in prompt
    )
    seen = {}

    def fn(system, p):
        seen["system"] = system
        return json.dumps([{"python_script": "a"}, {"python_script": "b"}, {"python_script": "c"}])

    bridge = ReasonerBridge(CallableBackend(fn))
    assert len(bridge.propose(ctx)) == 2 and seen["system"] == NEUROFORGE_SYSTEM_PROMPT
    assert "Surgical Minimality" in NEUROFORGE_SYSTEM_PROMPT


def test_load_intent_and_contrastive_texts(tmp_path):
    spec = {
        "intent_id": "i",
        "description": "d",
        "target_behavior": "t",
        "failure_behavior": "f",
        "contrastive_probes": [
            {
                "probe_id": "a",
                "prompt": "ab x",
                "expected_completion_prefix": "y",
                "undesired_completion_prefix": "x",
            },
            {
                "probe_id": "b",
                "prompt": "hello",
                "expected_completion_prefix": " ",
                "metadata": {"role": "invariant"},
            },
        ],
        "target_goals": [
            {"suite_name": "s", "metric_name": "logit_margin", "operator": ">=", "threshold": 0}
        ],
        "invariants": [{"suite_name": "inv", "metric_name": "kl_divergence"}],
    }
    path = tmp_path / "intent.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    intent = load_intent(path)
    assert contrastive_texts(intent) == (["ab xx"], ["ab xy"])
    with pytest.raises(FileNotFoundError):
        load_intent(tmp_path / "missing.yaml")
    spec["contrastive_probes"] = spec["contrastive_probes"][1:]
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="undesired_completion_prefix"):
        contrastive_texts(load_intent(path))
    bp = tmp_path / "bp.yaml"
    bp.write_text(
        yaml.safe_dump(
            {
                "model_name": "m",
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "subspaces": [],
                "layers": [],
            }
        ),
        encoding="utf-8",
    )
    assert load_blueprint(bp).hidden_size == 16


def test_draft_intent_validates_llm_output():
    good = {
        "intent_id": "x",
        "description": "d",
        "target_behavior": "t",
        "failure_behavior": "f",
        "contrastive_probes": [],
        "target_goals": [],
        "invariants": [],
    }
    assert (
        draft_intent(CallableBackend(lambda s, p: "```json\n" + json.dumps(good) + "\n```"), "desc").intent_id
        == "x"
    )
    with pytest.raises(Exception, match="validation error|intent_id"):
        draft_intent(CallableBackend(lambda s, p: '{"intent_id": 5}'), "desc")
