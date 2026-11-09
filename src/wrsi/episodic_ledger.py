"""Persistent SQLite ledger recording every self-edit trial, hypothesis, affected subspace, and fitness outcome.

W-RSI learns by trial and error, so failed trials matter as much as promoted ones: the scientist queries this
ledger before proposing a patch so it never repeats a failed perturbation. The schema is the PRD DDL.

All statements are parameterized. Each call opens and *closes* its connection (``sqlite3``'s context manager
only commits; an unclosed handle keeps the file locked on Windows).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

import torch

from .schemas import TrialEntry, TrialRecord

LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS weight_trials (
    trial_id TEXT PRIMARY KEY,
    generation INT NOT NULL,
    timestamp REAL NOT NULL,
    domain TEXT NOT NULL,
    failure_taxonomy TEXT NOT NULL,
    target_tensors TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    synthesized_code TEXT NOT NULL,
    sparsity_ratio REAL NOT NULL,
    frobenius_rel_norm REAL NOT NULL,
    target_score_delta REAL NOT NULL,
    holdout_kl_divergence REAL NOT NULL,
    promoted INTEGER NOT NULL,
    rejection_reason TEXT,
    embedding_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_generation ON weight_trials(generation);
CREATE INDEX IF NOT EXISTS idx_domain_promoted ON weight_trials(domain, promoted);
"""


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["target_tensors"] = json.loads(data["target_tensors"])
    data["promoted"] = bool(data["promoted"])
    raw_emb = data.pop("embedding_json", None)
    if raw_emb is not None:
        data["embedding"] = json.loads(raw_emb)
    return data


class EpisodicWeightLedger:
    """Stores and queries the history of weight-space self-modification trials."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(LEDGER_DDL)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(weight_trials)").fetchall()}
            if "embedding_json" not in cols:
                conn.execute("ALTER TABLE weight_trials ADD COLUMN embedding_json TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:  # commit on success, rollback on error
                yield conn

    def record_trial(self, entry: TrialEntry) -> None:
        """Inserts one trial (``trial_id`` must be unique)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO weight_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.trial_id,
                    entry.generation,
                    time.time(),
                    entry.domain,
                    entry.failure_taxonomy,
                    json.dumps(entry.target_tensors),
                    entry.hypothesis,
                    entry.synthesized_code,
                    entry.sparsity_ratio,
                    entry.frobenius_rel_norm,
                    entry.target_score_delta,
                    entry.holdout_kl_divergence,
                    1 if entry.promoted else 0,
                    entry.rejection_reason,
                    json.dumps(entry.embedding) if entry.embedding is not None else None,
                ),
            )

    def update_outcome(self, trial_id: str, promoted: bool, reason: str | None) -> None:
        """Rewrites a trial's outcome once it is final (selected, failed to commit, or rolled back)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE weight_trials SET promoted = ?, rejection_reason = ? WHERE trial_id = ?",
                (1 if promoted else 0, reason, trial_id),
            )

    def get_surgical_hit_rate(self, window_generations: int = 10) -> float:
        """Computes eta_t = (Promoted Trials) / (Total Trials) over the recent generation window."""
        if window_generations < 1:
            raise ValueError("window_generations must be >= 1")
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(generation) FROM weight_trials").fetchone()
            max_gen = row[0] if row and row[0] is not None else 0
            min_gen = max(0, max_gen - window_generations + 1)
            total, promoted = conn.execute(
                "SELECT COUNT(*), SUM(promoted) FROM weight_trials WHERE generation >= ?", (min_gen,)
            ).fetchone()
        total, promoted = total or 0, promoted or 0
        return float(promoted) / float(total) if total > 0 else 0.0

    def query_relevant_history(self, domain: str, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Returns the top promoted and most recent failed trials for a domain to inform the Scientist Agent."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1, 1000]")
        with self._connect() as conn:
            successes = conn.execute(
                "SELECT * FROM weight_trials WHERE domain = ? AND promoted = 1 "
                "ORDER BY target_score_delta DESC, rowid DESC LIMIT ?",
                (domain, limit),
            ).fetchall()
            failures = conn.execute(
                "SELECT * FROM weight_trials WHERE domain = ? AND promoted = 0 "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (domain, limit),
            ).fetchall()
        return {
            "past_successes": [_row_to_dict(r) for r in successes],
            "recent_failures_to_avoid": [_row_to_dict(r) for r in failures],
        }

    def query_similar_trials(
        self,
        query_vector: Sequence[float] | torch.Tensor,
        domain: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Ranks stored trials by cosine similarity against ``query_vector``.

        Uses each trial's stored ``embedding`` (or falls back to its 4-D metric signature
        ``[sparsity_ratio, frobenius_rel_norm, target_score_delta, holdout_kl_divergence]``).
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1, 1000]")
        q = (
            query_vector.detach().float().reshape(-1).tolist()
            if isinstance(query_vector, torch.Tensor)
            else [float(x) for x in query_vector]
        )
        if not q:
            raise ValueError("query_vector must not be empty")
        q_norm = math.sqrt(sum(x * x for x in q)) + 1e-12
        with self._connect() as conn:
            if domain is None:
                rows = conn.execute("SELECT * FROM weight_trials ORDER BY rowid DESC LIMIT 1000").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM weight_trials WHERE domain = ? ORDER BY rowid DESC LIMIT 1000", (domain,)
                ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            item = _row_to_dict(r)
            vec = item.get("embedding") or [
                float(item["sparsity_ratio"]),
                float(item["frobenius_rel_norm"]),
                float(item["target_score_delta"]),
                float(item["holdout_kl_divergence"]),
            ]
            dim = min(len(q), len(vec))
            v_norm = math.sqrt(sum(float(vec[i]) ** 2 for i in range(dim))) + 1e-12
            q_sub_norm = math.sqrt(sum(q[i] ** 2 for i in range(dim))) if dim < len(q) else q_norm
            sim = sum(q[i] * float(vec[i]) for i in range(dim)) / ((q_sub_norm + 1e-12) * v_norm)
            item["similarity"] = float(sim)
            scored.append((sim, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def list_trials(self, generation: int | None = None, limit: int = 100) -> list[TrialRecord]:
        """Most recent trials first, optionally filtered by generation."""
        with self._connect() as conn:
            if generation is None:
                rows = conn.execute(
                    "SELECT * FROM weight_trials ORDER BY rowid DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM weight_trials WHERE generation = ? ORDER BY rowid DESC LIMIT ?",
                    (generation, limit),
                ).fetchall()
        return [TrialRecord.model_validate(_row_to_dict(r)) for r in rows]

    def summary(self) -> dict[str, Any]:
        """Totals for status displays: trials, promoted, per-domain counts, latest generation."""
        with self._connect() as conn:
            total, promoted, max_gen = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(promoted), 0), MAX(generation) FROM weight_trials"
            ).fetchone()
            per_domain = conn.execute(
                "SELECT domain, COUNT(*), COALESCE(SUM(promoted), 0) FROM weight_trials GROUP BY domain ORDER BY domain"
            ).fetchall()
        return {
            "total_trials": total,
            "promoted_trials": promoted,
            "latest_generation": max_gen,
            "domains": {d: {"trials": n, "promoted": p} for d, n, p in per_domain},
        }
