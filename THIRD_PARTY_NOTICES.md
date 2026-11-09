# Third-party notices

## Vendored code

`src/libweightspace/vendor_lfm3/` contains the configuration and modeling code for LiquidAI LFM 3 models
(`configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py`), copied from the model repository's remote-code files
so that the model can run without `trust_remote_code=True`. The files carry their original header:
"Copyright 2026 The HuggingFace Team. All rights reserved. Licensed under the Apache License, Version 2.0".
They are kept verbatim (linting and type checking are disabled for that folder). A copy of the licence is in
`src/libweightspace/vendor_lfm3/LICENSE-Apache-2.0.txt`.

## Model weights

No model weights are included in this repository. Checkpoints you load (for example from Hugging Face) are
governed by their own licences, which you accept when you download them. LiquidAI LFM models are distributed under
LiquidAI's own licence; check the model card before use.

## Runtime dependencies

Installed from PyPI under their own licences: PyTorch (BSD-3-Clause), safetensors (Apache-2.0), pydantic (MIT),
PyYAML (MIT), Typer (MIT), Rich (MIT), NumPy (BSD-3-Clause); optional: transformers and accelerate (Apache-2.0).
