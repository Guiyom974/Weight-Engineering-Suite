"""Episodic ledger tests (PRD_02 section 4.3, blueprint step 11)."""

import pytest
from pydantic import ValidationError

from wrsi.episodic_ledger import EpisodicWeightLedger
from wrsi.schemas import TrialEntry


def _entry(i, generation, promoted, domain="mathematical_deduction", gain=0.0):
    return TrialEntry(
        trial_id=f"t{i}",
        generation=generation,
        domain=domain,
        failure_taxonomy="attractor_loop",
        target_tensors=[f"model.layers.{i}.mlp.down_proj.weight"],
        hypothesis=f"hypothesis {i}",
        synthesized_code=f"# code {i}",
        sparsity_ratio=0.0001 * i,
        frobenius_rel_norm=0.01,
        target_score_delta=gain,
        holdout_kl_divergence=0.002,
        promoted=promoted,
        rejection_reason=None if promoted else f"reason {i}",
    )


@pytest.fixture
def ledger(tmp_path):
    return EpisodicWeightLedger(tmp_path / "nested" / "ledger.db")


def _populate(ledger):
    # 5 failed + 3 promoted trials across 2 generations.
    for i, (gen, promoted, gain) in enumerate(
        [
            (0, False, -0.1),
            (0, False, 0.0),
            (0, True, 0.2),
            (0, False, -0.3),
            (1, True, 0.5),
            (1, False, 0.0),
            (1, True, 0.1),
            (1, False, -0.2),
        ]
    ):
        ledger.record_trial(_entry(i, gen, promoted, gain=gain))


def test_hit_rate_and_history(ledger):
    _populate(ledger)
    assert ledger.get_surgical_hit_rate() == pytest.approx(3 / 8)
    assert ledger.get_surgical_hit_rate(window_generations=1) == pytest.approx(2 / 4)

    history = ledger.query_relevant_history("mathematical_deduction", limit=5)
    successes, failures = history["past_successes"], history["recent_failures_to_avoid"]
    assert [r["trial_id"] for r in successes] == ["t4", "t2", "t6"]  # best gain first
    assert [r["trial_id"] for r in failures] == ["t7", "t5", "t3", "t1", "t0"]  # most recent first
    top = successes[0]
    assert top == {
        "trial_id": "t4",
        "generation": 1,
        "timestamp": top["timestamp"],
        "domain": "mathematical_deduction",
        "failure_taxonomy": "attractor_loop",
        "target_tensors": ["model.layers.4.mlp.down_proj.weight"],
        "hypothesis": "hypothesis 4",
        "synthesized_code": "# code 4",
        "sparsity_ratio": pytest.approx(0.0004),
        "frobenius_rel_norm": 0.01,
        "target_score_delta": 0.5,
        "holdout_kl_divergence": 0.002,
        "promoted": True,
        "rejection_reason": None,
    }
    assert failures[0]["rejection_reason"] == "reason 7" and failures[0]["promoted"] is False
    assert (
        ledger.query_relevant_history("mathematical_deduction", limit=2)["recent_failures_to_avoid"][1][
            "trial_id"
        ]
        == "t5"
    )


def test_empty_ledger_and_domain_isolation(ledger):
    assert ledger.get_surgical_hit_rate() == 0.0
    ledger.record_trial(_entry(1, 0, True, domain="code_generation"))
    assert ledger.query_relevant_history("mathematical_deduction") == {
        "past_successes": [],
        "recent_failures_to_avoid": [],
    }


def test_list_summary_and_persistence(ledger, tmp_path):
    _populate(ledger)
    recent = ledger.list_trials(limit=3)
    assert [r.trial_id for r in recent] == ["t7", "t6", "t5"] and recent[0].timestamp > 0
    assert {r.trial_id for r in ledger.list_trials(generation=0)} == {"t0", "t1", "t2", "t3"}
    summary = ledger.summary()
    assert (
        summary["total_trials"] == 8 and summary["promoted_trials"] == 3 and summary["latest_generation"] == 1
    )
    assert summary["domains"]["mathematical_deduction"] == {"trials": 8, "promoted": 3}
    reopened = EpisodicWeightLedger(tmp_path / "nested" / "ledger.db")
    assert reopened.get_surgical_hit_rate() == pytest.approx(0.375)


def test_connections_are_closed(tmp_path):
    """On Windows an open sqlite handle locks the file; deleting it proves every call closed its connection."""
    path = tmp_path / "l.db"
    ledger = EpisodicWeightLedger(path)
    ledger.record_trial(_entry(1, 0, True))
    ledger.query_relevant_history("mathematical_deduction")
    ledger.summary()
    path.unlink()
    assert not path.exists()


def test_injection_strings_are_data(ledger):
    evil = "x'; DROP TABLE weight_trials; --"
    ledger.record_trial(_entry(1, 0, False, domain=evil))
    assert ledger.query_relevant_history(evil)["recent_failures_to_avoid"][0]["domain"] == evil
    assert ledger.summary()["total_trials"] == 1


def test_duplicate_ids_and_bad_arguments(ledger):
    ledger.record_trial(_entry(1, 0, True))
    with pytest.raises(Exception, match="UNIQUE"):
        ledger.record_trial(_entry(1, 0, True))
    with pytest.raises(ValueError):
        ledger.get_surgical_hit_rate(window_generations=0)
    with pytest.raises(ValueError):
        ledger.query_relevant_history("d", limit=0)
    with pytest.raises(ValidationError):
        _entry(1, -1, True)


def test_query_similar_trials_cosine_ranking(ledger):
    e1 = _entry(1, 0, True, gain=0.8)
    e1.embedding = [1.0, 0.0, 0.0]
    e2 = _entry(2, 0, False, gain=-0.2)
    e2.embedding = [0.0, 1.0, 0.0]
    e3 = _entry(3, 1, True, gain=0.5)
    e3.embedding = [0.9, 0.1, 0.0]
    ledger.record_trial(e1)
    ledger.record_trial(e2)
    ledger.record_trial(e3)

    matches = ledger.query_similar_trials([1.0, 0.0, 0.0], domain="mathematical_deduction", limit=2)
    assert [m["trial_id"] for m in matches] == ["t1", "t3"]
    assert matches[0]["similarity"] > matches[1]["similarity"] > 0.9
    with pytest.raises(ValueError):
        ledger.query_similar_trials([], limit=1)
