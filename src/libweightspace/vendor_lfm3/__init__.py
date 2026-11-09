"""Vendored modeling and configuration files for LiquidAI LFM3 models.

Vendored under Apache 2.0 license from LiquidAI/LFM3-1.7B checkpoint repository
to enable secure local execution without trust_remote_code=True.
"""

from libweightspace.vendor_lfm3.configuration_lfm3 import Lfm3Config
from libweightspace.vendor_lfm3.modeling_lfm3 import (
    Lfm3DecoderLayer,
    Lfm3ForCausalLM,
    Lfm3Model,
    Lfm3PreTrainedModel,
)

__all__ = [
    "Lfm3Config",
    "Lfm3DecoderLayer",
    "Lfm3ForCausalLM",
    "Lfm3Model",
    "Lfm3PreTrainedModel",
]
