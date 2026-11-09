"""Starter files written by ``neuroforge init``: a demo intent (repairs the toy 'x' attractor) and a blueprint."""

from __future__ import annotations

DEMO_INTENT = {
    "intent_id": "break-x-loop",
    "description": "After the token 'x' the model falls into a repetition loop (x x x ...). Remove the loop.",
    "target_behavior": "After 'x' the model prefers 'y' over another 'x' at least as much as the known-good model.",
    "failure_behavior": "After 'x' the model keeps predicting 'x'.",
    "contrastive_probes": [
        {
            "probe_id": f"loop{i}",
            "prompt": p,
            "expected_completion_prefix": "yzyz",
            "undesired_completion_prefix": "xxxx",
        }
        for i, p in enumerate(["ab x", "cd x", "the x", "q: x"])
    ]
    + [
        {
            "probe_id": f"inv{i}",
            "prompt": p,
            "expected_completion_prefix": " ",
            "metadata": {"role": "invariant"},
        }
        for i, p in enumerate(["hello world", "abc def ghi", "the quick brown", "lorem ipsum do"])
    ],
    # Threshold: the known-good toy model's margin (-2.34) minus a 0.5 tolerance.
    "target_goals": [
        {"suite_name": "loop", "metric_name": "logit_margin", "operator": ">=", "threshold": -2.85}
    ],
    "invariants": [{"suite_name": "baseline", "metric_name": "kl_divergence", "max_allowed_drop": 0.01}],
    # Toy models are tiny, so a real repair touches a larger fraction of parameters than in a 7B model.
    "constraints": {
        "max_modified_param_ratio": 0.01,
        "max_frobenius_rel_norm": 0.15,
        "max_spectral_norm_shift": 0.05,
    },
}

DEMO_BLUEPRINT = {
    "model_name": "denovo-demo",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "subspaces": [
        {"name": "anchor", "start_dim": 0, "end_dim": 2, "description": "constant positional anchor"},
        {"name": "content", "start_dim": 2, "end_dim": 10, "description": "token content"},
        {"name": "previous", "start_dim": 10, "end_dim": 18, "description": "copied previous-token content"},
        {"name": "facts", "start_dim": 18, "end_dim": 26, "description": "associative memory outputs"},
    ],
    "layers": [
        {
            "layer_idx": 0,
            "attention_heads": [
                {
                    "head_idx": 0,
                    "role": "previous_token_copy",
                    "query_subspace": "anchor",
                    "key_subspace": "anchor",
                    "value_subspace": "content",
                    "output_subspace": "previous",
                    "routing_strength": 20.0,
                }
            ],
        },
        {
            "layer_idx": 1,
            "mlp_rules": [
                {
                    "rule_id": "fact-a",
                    "input_subspace": "previous",
                    "output_subspace": "facts",
                    "key_vectors": [[1, 0, 0, 0, 0, 0, 0, 0]],
                    "value_vectors": [[0, 0, 0, 0, 0, 0, 0, 1]],
                },
                {
                    "rule_id": "fact-b",
                    "input_subspace": "previous",
                    "output_subspace": "facts",
                    "key_vectors": [[0, 1, 0, 0, 0, 0, 0, 0]],
                    "value_vectors": [[0, 0, 0, 0, 0, 0, 1, 0]],
                },
            ],
        },
    ],
}


def intent_examples() -> list[dict[str, object]]:
    """Offline example intents for the Studio's intent editor (CR3). Each validates against ``IntentSpec``.

    The demo example matches the toy demo model. The fact template must be edited (prompts and threshold) for a real
    model; its thresholds are placeholders and the description says so.
    """
    fact = {
        "intent_id": "update-one-fact",
        "description": "Template: the model gives an outdated answer to one factual question. Edit the prompts, the "
        "expected answers and the threshold for your model before running.",
        "target_behavior": "The model completes the question with the new answer.",
        "failure_behavior": "The model completes the question with the old answer.",
        "contrastive_probes": [
            {"probe_id": "fact0", "prompt": "The capital of Exampleland is", "expected_completion_prefix": " Newtown",
             "undesired_completion_prefix": " Oldtown"},
            {"probe_id": "fact1", "prompt": "Exampleland's capital city is called", "expected_completion_prefix": " Newtown",
             "undesired_completion_prefix": " Oldtown"},
            {"probe_id": "inv0", "prompt": "The capital of France is", "expected_completion_prefix": " Paris",
             "metadata": {"role": "invariant"}},
            {"probe_id": "inv1", "prompt": "Water boils at", "expected_completion_prefix": " 100",
             "metadata": {"role": "invariant"}},
        ],
        "target_goals": [{"suite_name": "fact", "metric_name": "logit_margin", "operator": ">=", "threshold": 0.0}],
        "invariants": [{"suite_name": "baseline", "metric_name": "kl_divergence", "max_allowed_drop": 0.01}],
    }
    return [
        {"id": "demo-loop", "title": "Stop a repetition loop (matches the demo model)", "intent": DEMO_INTENT},
        {"id": "fact-template", "title": "Update one fact (template to adapt)", "intent": fact},
    ]
