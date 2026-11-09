"""Merge and export weight patches (Safetensors, Sparse COO, HF, PEFT LoRA, and GGUF v3 in-place patching).

Formats:
    - merged SafeTensors (``export_merged_safetensors``): full weights with the patch applied; tied tensors are
      de-duplicated by ``safetensors.torch.save_model``.
    - sparse delta (``export_sparse_delta``): the same COO layout as ``.wgit`` objects, with metadata; load with
      ``load_sparse_delta``.
    - Hugging Face directory (``export_hf_pretrained``): ``save_pretrained`` output (config + weights) for models
      that support it.
    - PEFT LoRA adapter (``export_peft_lora``): decomposes 2-D weight deltas via truncated SVD into
      ``adapter_model.safetensors`` + ``adapter_config.json`` so ``(lora_alpha / r) * (B @ A)`` reproduces the
      low-rank delta on vLLM, SGLang, or Hugging Face PEFT.
    - Direct in-place GGUF v3 patching (``patch_gguf_inplace``, ``write_gguf_v3``, ``read_gguf_v3_tensor``):
      dequantizes only the modified rows/blocks of ``F32``, ``F16``, or ``Q8_0`` (32-element blocks) tensors,
      applies ``delta_W``, and re-quantizes in place without re-writing the untouched gigabytes of the file.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, save_model

from libweightspace.cow_overlay import CoWWeightOverlay

from .weight_git import deserialize_deltas, serialize_deltas

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34  # 2-byte fp16 scale + 32 int8 quants


@dataclass
class GGUFTensorInfo:
    """Parsed GGUF v3 tensor directory entry."""

    name: str
    dims: tuple[int, ...]  # PyTorch row-major shape (reversed from GGUF ne[0..n-1])
    ggml_type: int
    offset: int  # relative to data_offset
    abs_offset: int  # absolute byte offset in the file
    byte_size: int


def export_merged_safetensors(
    model: nn.Module,
    deltas: Mapping[str, torch.Tensor],
    out_path: Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Writes ``model`` weights with ``deltas`` applied to ``out_path`` (model left unchanged)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with CoWWeightOverlay(model).apply_delta(dict(deltas)) as patched:
        save_model(patched, str(out_path), metadata=metadata)
    return out_path


def export_sparse_delta(
    deltas: Mapping[str, torch.Tensor], out_path: Path, metadata: dict[str, str] | None = None
) -> Path:
    """Writes ``deltas`` in COO safetensors form."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialize_deltas(deltas, metadata=metadata))
    return out_path


def load_sparse_delta(path: Path) -> dict[str, torch.Tensor]:
    """Reads a file written by ``export_sparse_delta`` (no pickle involved)."""
    return deserialize_deltas(Path(path).read_bytes())


def export_hf_pretrained(model: nn.Module, deltas: Mapping[str, torch.Tensor], out_dir: Path) -> Path:
    """``save_pretrained`` with ``deltas`` applied.

    Raises:
        ValueError: the model has no ``save_pretrained`` (e.g. toy models; use safetensors export instead).
    """
    save = getattr(model, "save_pretrained", None)
    if not callable(save):
        raise ValueError("model has no save_pretrained(); export to safetensors instead")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with CoWWeightOverlay(model).apply_delta(dict(deltas)):
        save(str(out_dir), safe_serialization=True)
    return out_dir


def export_peft_lora(
    deltas: Mapping[str, torch.Tensor],
    out_dir: Path,
    rank: int = 8,
    lora_alpha: float = 16.0,
    base_model_id: str = "unknown",
) -> Path:
    """Exports 2-D weight deltas as a Hugging Face PEFT LoRA adapter directory (NF-BL-07).

    Decomposes each 2-D ``delta_W`` via SVD into ``lora_B`` ``[d_out, r]`` and ``lora_A`` ``[r, d_in]`` scaled
    so that ``(lora_alpha / r) * (lora_B @ lora_A)`` equals the top-``r`` SVD approximation of ``delta_W``.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_tensors: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()

    for name, delta in sorted(deltas.items()):
        if delta.ndim != 2 or not bool((delta != 0).any().item()):
            continue
        mod_name = name[: -len(".weight")] if name.endswith(".weight") else name
        short_mod = mod_name.rsplit(".", 1)[-1]
        target_modules.add(short_mod)

        mat = delta.detach().double().cpu()
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r_eff = min(rank, int(s.numel()))
        # Pad to `rank` if min(d_out, d_in) < rank so all adapter matrices have uniform rank r.
        d_out, d_in = mat.shape
        b_mat = torch.zeros((d_out, rank), dtype=torch.float32)
        a_mat = torch.zeros((rank, d_in), dtype=torch.float32)

        # (lora_alpha / rank) * (B @ A) = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]
        scale = float(rank) / float(lora_alpha)
        sqrt_s = torch.sqrt(s[:r_eff] * scale)
        b_mat[:, :r_eff] = (u[:, :r_eff] * sqrt_s.unsqueeze(0)).float()
        a_mat[:r_eff, :] = (sqrt_s.unsqueeze(1) * vh[:r_eff, :]).float()

        prefix = f"base_model.model.{mod_name}"
        lora_tensors[f"{prefix}.lora_A.weight"] = a_mat.contiguous()
        lora_tensors[f"{prefix}.lora_B.weight"] = b_mat.contiguous()

    if not lora_tensors:
        raise ValueError("no non-zero 2-D weight deltas to export as PEFT LoRA")

    save_file(lora_tensors, str(out_dir / "adapter_model.safetensors"))
    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "fan_in_fan_out": False,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")
    return out_dir


# --- GGUF v3 binary helpers & in-place block surgery (NF-BL-07) ----------------------------------------


def _quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """Encodes a float tensor (numel divisible by 32) into GGML_TYPE_Q8_0 34-byte blocks."""
    arr = tensor.detach().float().cpu().numpy().reshape(-1)
    if arr.size % Q8_0_BLOCK_ELEMS != 0:
        raise ValueError(f"Q8_0 requires numel divisible by 32, got {arr.size}")
    blocks = arr.reshape(-1, Q8_0_BLOCK_ELEMS)
    amax = np.max(np.abs(blocks), axis=1)
    d = (amax / 127.0).astype(np.float16)
    d_f32 = d.astype(np.float32)
    inv_d = np.where(d_f32 > 0, 1.0 / d_f32, 0.0)[:, None]
    quants = np.clip(np.round(blocks * inv_d), -128, 127).astype(np.int8)

    out = bytearray(blocks.shape[0] * Q8_0_BLOCK_BYTES)
    d_bytes = d.view(np.uint8).reshape(-1, 2)
    q_bytes = quants.view(np.uint8)
    out_view = np.frombuffer(out, dtype=np.uint8).reshape(-1, Q8_0_BLOCK_BYTES)
    out_view[:, :2] = d_bytes
    out_view[:, 2:] = q_bytes
    return bytes(out)


def _dequantize_q8_0(raw: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """Decodes GGML_TYPE_Q8_0 bytes into a float32 tensor of ``shape``."""
    n_blocks = len(raw) // Q8_0_BLOCK_BYTES
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    d = buf[:, :2].copy().view(np.float16).astype(np.float32)  # [n_blocks, 1]
    quants = buf[:, 2:].view(np.int8).astype(np.float32)  # [n_blocks, 32]
    vals = (d * quants).reshape(shape)
    return torch.from_numpy(vals.copy())


def _ggml_byte_size(numel: int, ggml_type: int) -> int:
    if ggml_type == GGML_TYPE_F32:
        return numel * 4
    if ggml_type == GGML_TYPE_F16:
        return numel * 2
    if ggml_type == GGML_TYPE_Q8_0:
        if numel % Q8_0_BLOCK_ELEMS != 0:
            raise ValueError(f"Q8_0 tensor numel {numel} is not a multiple of 32")
        return (numel // Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
    raise ValueError(f"Unsupported ggml_type {ggml_type}")


def _skip_gguf_value(fh: BinaryIO, vtype: int) -> None:
    """Advances ``fh`` past a GGUF v3 metadata value of type ``vtype``."""
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in scalar_sizes:
        fh.read(scalar_sizes[vtype])
    elif vtype == 8:  # string
        (slen,) = struct.unpack("<Q", fh.read(8))
        fh.read(slen)
    elif vtype == 9:  # array
        elem_type, count = struct.unpack("<IQ", fh.read(12))
        for _ in range(count):
            _skip_gguf_value(fh, elem_type)
    else:
        raise ValueError(f"Unknown GGUF metadata value type {vtype}")


def parse_gguf_v3_directory(gguf_path: Path) -> dict[str, GGUFTensorInfo]:
    """Parses a GGUF v3 header and returns a mapping ``tensor_name -> GGUFTensorInfo``."""
    with Path(gguf_path).open("rb") as fh:
        magic = fh.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version {version} (expected 3)")
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", fh.read(8))
            fh.read(klen)
            (vtype,) = struct.unpack("<I", fh.read(4))
            _skip_gguf_value(fh, vtype)

        entries: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            (nlen,) = struct.unpack("<Q", fh.read(8))
            name = fh.read(nlen).decode("utf-8")
            (n_dims,) = struct.unpack("<I", fh.read(4))
            ne = struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims))
            # GGUF stores inner dimension first (ne[0] = cols, ne[1] = rows); reverse for PyTorch shape.
            dims = tuple(int(x) for x in reversed(ne))
            ggml_type, rel_offset = struct.unpack("<IQ", fh.read(12))
            entries.append((name, dims, ggml_type, rel_offset))

        pos = fh.tell()
        pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        data_offset = pos + pad

    out: dict[str, GGUFTensorInfo] = {}
    for name, dims, ggml_type, rel_offset in entries:
        numel = 1
        for d in dims:
            numel *= d
        bsize = _ggml_byte_size(numel, ggml_type)
        out[name] = GGUFTensorInfo(
            name=name,
            dims=dims,
            ggml_type=ggml_type,
            offset=rel_offset,
            abs_offset=data_offset + rel_offset,
            byte_size=bsize,
        )
    return out


def write_gguf_v3(
    tensors: Mapping[str, torch.Tensor],
    out_path: Path,
    quant: Literal["f32", "f16", "q8_0"] = "q8_0",
) -> Path:
    """Writes ``tensors`` into a valid GGUF v3 file with ``F32``, ``F16``, or ``Q8_0`` quantization."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ggml_type = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0}[quant]

    encoded_blobs: list[bytes] = []
    header_entries: list[bytes] = []
    rel_offset = 0

    for name, t in sorted(tensors.items()):
        t_cpu = t.detach().float().cpu().contiguous()
        dims = tuple(int(s) for s in t_cpu.shape)
        t_type = ggml_type
        if t_type == GGML_TYPE_Q8_0 and t_cpu.numel() % Q8_0_BLOCK_ELEMS != 0:
            t_type = GGML_TYPE_F32

        if t_type == GGML_TYPE_F32:
            blob = t_cpu.numpy().astype(np.float32).tobytes()
        elif t_type == GGML_TYPE_F16:
            blob = t_cpu.numpy().astype(np.float16).tobytes()
        else:
            blob = _quantize_q8_0(t_cpu)

        name_bytes = name.encode("utf-8")
        ne = list(reversed(dims))
        entry = (
            struct.pack("<Q", len(name_bytes))
            + name_bytes
            + struct.pack("<I", len(ne))
            + struct.pack(f"<{len(ne)}Q", *ne)
            + struct.pack("<IQ", t_type, rel_offset)
        )
        header_entries.append(entry)
        pad = (GGUF_DEFAULT_ALIGNMENT - (len(blob) % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        padded_blob = blob + (b"\x00" * pad)
        encoded_blobs.append(padded_blob)
        rel_offset += len(padded_blob)

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(header_entries), 0)
    for entry in header_entries:
        header += entry
    h_pad = (GGUF_DEFAULT_ALIGNMENT - (len(header) % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
    header += b"\x00" * h_pad

    with out_path.open("wb") as fh:
        fh.write(header)
        for blob in encoded_blobs:
            fh.write(blob)
    return out_path


def read_gguf_v3_tensor(gguf_path: Path, tensor_name: str) -> torch.Tensor:
    """Reads and dequantizes a single tensor from a GGUF v3 file into float32."""
    directory = parse_gguf_v3_directory(gguf_path)
    if tensor_name not in directory:
        raise KeyError(f"Tensor '{tensor_name}' not found in {gguf_path}")
    info = directory[tensor_name]
    with Path(gguf_path).open("rb") as fh:
        fh.seek(info.abs_offset)
        raw = fh.read(info.byte_size)
    if info.ggml_type == GGML_TYPE_F32:
        arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(info.dims)
        return torch.from_numpy(arr)
    if info.ggml_type == GGML_TYPE_F16:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32). reshape(info.dims)
        return torch.from_numpy(arr)
    if info.ggml_type == GGML_TYPE_Q8_0:
        return _dequantize_q8_0(raw, info.dims)
    raise ValueError(f"Unsupported ggml_type {info.ggml_type}")


def patch_gguf_inplace(gguf_path: Path, deltas: Mapping[str, torch.Tensor]) -> list[str]:
    """Patches modified tensors or Q8_0 blocks directly in place inside ``gguf_path`` (NF-BL-07).

    Returns:
        List of tensor names patched in place.
    """
    gguf_path = Path(gguf_path)
    directory = parse_gguf_v3_directory(gguf_path)
    patched_names: list[str] = []

    with gguf_path.open("r+b") as fh:
        for name, delta in sorted(deltas.items()):
            if name not in directory or not bool((delta != 0).any().item()):
                continue
            info = directory[name]
            d_cpu = delta.detach().float().cpu()
            if tuple(d_cpu.shape) != info.dims:
                raise ValueError(
                    f"GGUF shape mismatch for '{name}': delta {tuple(d_cpu.shape)} vs GGUF {info.dims}"
                )

            if info.ggml_type == GGML_TYPE_Q8_0:
                # Surgical block-level patching: only touch 32-element blocks where delta != 0.
                d_blocks = d_cpu.reshape(-1, Q8_0_BLOCK_ELEMS)
                active_blocks = torch.nonzero(d_blocks.abs().max(dim=1).values > 0, as_tuple=False).reshape(-1)
                for b_idx_t in active_blocks:
                    b_idx = int(b_idx_t.item())
                    block_offset = info.abs_offset + b_idx * Q8_0_BLOCK_BYTES
                    fh.seek(block_offset)
                    raw_block = fh.read(Q8_0_BLOCK_BYTES)
                    cur_vals = _dequantize_q8_0(raw_block, (Q8_0_BLOCK_ELEMS,))
                    updated = cur_vals + d_blocks[b_idx]
                    fh.seek(block_offset)
                    fh.write(_quantize_q8_0(updated))
            elif info.ggml_type == GGML_TYPE_F32:
                fh.seek(info.abs_offset)
                raw = fh.read(info.byte_size)
                cur = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy().reshape(info.dims))
                updated_bytes = (cur + d_cpu).numpy().astype(np.float32).tobytes()
                fh.seek(info.abs_offset)
                fh.write(updated_bytes)
            elif info.ggml_type == GGML_TYPE_F16:
                fh.seek(info.abs_offset)
                raw = fh.read(info.byte_size)
                cur = torch.from_numpy(
                    np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(info.dims)
                )
                updated_bytes = (cur + d_cpu).numpy().astype(np.float16).tobytes()
                fh.seek(info.abs_offset)
                fh.write(updated_bytes)
            else:
                raise ValueError(f"Unsupported ggml_type {info.ggml_type} for '{name}'")
            patched_names.append(name)
    return patched_names
