"""NeuroForge: intent-driven weight-space programming and neural synthesis.

Modes:
    1. ``base_edit``: localise a behaviour with the prober and apply a sparse, sandboxed tensor delta.
    2. ``from_scratch``: compile a declarative ``NeuralBlueprint`` into working weights (``blueprint_compiler``).
    3. ``convergence_loop``: a reasoner proposes candidate patches; a Pareto tournament commits winners
       into a Git-for-Weights DAG (``convergence_engine``, ``weight_git``).

The CLI lives in ``neuroforge.cli`` (``neuroforge init/probe/edit/compile/loop/export/log``).
"""

__version__ = "0.2.0"
