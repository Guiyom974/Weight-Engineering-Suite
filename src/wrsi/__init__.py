"""W-RSI: autonomous weight-space recursive self-improvement.

An open-weights model diagnoses its own failures, proposes sandboxed patches to its own parameters, tests them
in Copy-on-Write forks, and promotes winners under hard rails: nullspace locking (earlier skills are protected),
an air-gapped KL tether to M_0 (``anchor_vault_server``), a sparsity budget, and a bit-exact rollback stack.

The CLI lives in ``wrsi.cli`` (``wrsi start/status/rollback/telemetry``).
"""

__version__ = "0.2.0"
