"""libweightspace: shared low-level kernel for weight-space engineering.

Used by both NeuroForge (intent-driven editing of a target model) and W-RSI (a model editing itself).

Main entry points:
    - ``adapters``: uniform ``LayerAnatomy`` access across Qwen-3.8, Gemma 4, DeepSeek and LFM 3 families.
    - ``cow_overlay.CoWWeightOverlay``: reversible, slice-level in-place weight deltas.
    - ``prober.MechanisticProber``: contrastive activation tracing and culpable-column attribution.
    - ``spectral``: singular-value profiling and spectral-norm utilities.
    - ``nullspace.NullspaceLocker``: protects previously locked input directions from later edits.
    - ``sandbox.execute_weight_surgery_script``: validated execution of LLM-written ``compute_delta`` scripts.

Key assumption: models are ``torch.nn.Module`` objects whose parameter names follow Hugging Face conventions
(``model.layers.{i}...``); the bundled ``toy_models`` follow the same naming.
"""

__version__ = "0.2.0"
