"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import tvm_ffi

from ...utils import get_compute_capability, get_device_sm_count
from .cake_jit import ARCHES, PROGRAM, load_cake_nvfp4_attention_module

HEAD_DIM = 128
SF_VEC = 16
# Work-tile rows of the persistent kernel (four 128-row Q blocks per CTA pair).
TILE_ROWS = 512
# E2M1 magnitude thresholds (midpoints between representable values); the
# code of ``|x|`` is the number of thresholds strictly below it.
E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
SCALE_MIN = 2.0**-9
SCALE_MAX = torch.finfo(torch.float8_e4m3fn).max


def _device_key(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


def _device_facts(device):
    """``(compute_capability, multiprocessor_count)`` through FlashInfer's cached helpers."""
    device = torch.device("cuda", _device_key(device)[1])
    return get_compute_capability(device), int(get_device_sm_count(device))


@functools.cache
def _boundaries(device_key):
    # One threshold table per device for the process lifetime (a plan-time
    # constant, not a per-call construction).
    kind, index = device_key
    device = torch.device(kind) if index is None else torch.device(kind, index)
    return torch.tensor(E2M1_BOUNDARIES, dtype=torch.float32, device=device)


def _block_scales(blocks, dim):
    """Positive E4M3 scale per group of ``SF_VEC`` values along ``dim``.

    The scale is ``max|x| / 6`` clamped to ``[2^-9, 448]`` and rounded to E4M3;
    the maximum is exact for any input layout, so strided inputs quantize
    exactly like contiguous copies.
    """
    amax = torch.linalg.vector_norm(blocks, float("inf"), dim=dim, dtype=torch.float32)
    amax.div_(6.0).clamp_(min=SCALE_MIN, max=SCALE_MAX)
    return amax.to(torch.float8_e4m3fn)


def _e2m1_codes(blocks, scale_fp8, dim):
    """Signed 4-bit E2M1 codes (one ``uint8`` per value) of ``blocks / scale``."""
    normalized = blocks / scale_fp8.float().unsqueeze(dim)
    # A negative value keeps its sign bit only when its magnitude code is
    # nonzero, i.e. when it lies strictly below the first threshold.
    negative = normalized < -E2M1_BOUNDARIES[0]
    magnitude = normalized.abs_()
    code = torch.bucketize(
        magnitude, _boundaries(_device_key(blocks.device)), out_int32=True
    ).to(torch.uint8)
    code.add_(negative.to(torch.uint8), alpha=8)
    return code


def quantize_nvfp4_rows(x):
    """Quantize the last dimension of ``x`` (any strides) to NVFP4.

    Returns the packed E2M1 pairs ``[..., D/2]`` (low nibble first) and the
    E4M3 block scales ``[..., D/16]`` as ``uint8`` bytes.
    """
    blocks = x.unflatten(-1, (x.shape[-1] // SF_VEC, SF_VEC))
    scale = _block_scales(blocks, -1)
    code = _e2m1_codes(blocks, scale, -1).flatten(-2)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, scale.view(torch.uint8)


def quantize_nvfp4_columns(v):
    """Quantize ``v`` ``[B,H,S,128]`` (any strides) along ``S`` into the transposed
    MMA operand.

    Returns the packed pairs along ``S`` as ``[B,H,128,S/2]`` (the kernel's
    ``Vt`` layout, written in place without a BF16 transpose copy) and the E4M3
    block scales ``[B,H,S/16,128]`` as ``uint8`` bytes.
    """
    batch, heads, seqlen, dim = v.shape
    blocks = v.unflatten(2, (seqlen // SF_VEC, SF_VEC))
    scale = _block_scales(blocks, 3)
    code = _e2m1_codes(blocks, scale, 3).flatten(2, 3)
    vt = torch.empty(
        (batch, heads, dim, seqlen // 2), dtype=torch.uint8, device=v.device
    )
    torch.bitwise_or(
        code[:, :, 0::2, :], code[:, :, 1::2, :] << 4, out=vt.transpose(-1, -2)
    )
    return vt, scale.view(torch.uint8)


def _prepack_sf_tiles(scale_u8):
    """Pack ``[tiles,128,8]`` natural UE4M3 bytes into UTCCP source tiles."""
    tiles = scale_u8.shape[0]
    sf = scale_u8.reshape(tiles, 4, 4, 8, 1, 2, 4)
    return sf.permute(0, 4, 2, 5, 3, 1, 6).reshape(tiles * 32, 32)


def _prepack_sf_tiles_split_ksets(scale_u8):
    """Pack two 4-scale K-sets of ``[tiles,128,8]`` as separate 512 B UTCCP tiles."""
    tiles = scale_u8.shape[0]

    def pack_half(values):
        sf = values.reshape(tiles, 4, 4, 8, 1, 1, 4)
        return sf.permute(0, 4, 2, 5, 3, 1, 6).reshape(tiles * 16, 32)

    return pack_half(scale_u8[..., :4]), pack_half(scale_u8[..., 4:])


def pack_nvfp4_attention_inputs(q, k, v):
    """Quantize and pack ``[B,H,S,128]`` BF16 ``q``/``k``/``v`` (any strides) into
    the kernel's operand layouts.

    Returns ``Q``/``K`` ``[B*H*S, 64]``, ``Vt`` ``[B*H*128, S/2]`` and the
    scale tiles ``SFQ``/``SFK`` ``[B*H*S/4, 32]``, ``SFVtLo``/``SFVtHi``
    ``[B*H*S/8, 32]``; every tensor is dense.
    """
    batch, heads, seqlen, dim = q.shape
    bh = batch * heads
    blocks = seqlen // 128
    packed_q, sq = quantize_nvfp4_rows(q)
    packed_k, sk = quantize_nvfp4_rows(k)
    vt, sv = quantize_nvfp4_columns(v)
    # V scales: natural ``[tile, head-dim row, 16-row group within the tile]``.
    sv = sv.reshape(bh, blocks, 8, dim).transpose(-1, -2).reshape(-1, dim, 8)
    lo, hi = _prepack_sf_tiles_split_ksets(sv)
    return dict(
        Q=packed_q.reshape(bh * seqlen, dim // 2),
        K=packed_k.reshape(bh * seqlen, dim // 2),
        Vt=vt.reshape(bh * dim, seqlen // 2),
        SFQ=_prepack_sf_tiles(sq.reshape(-1, 128, 8)),
        SFK=_prepack_sf_tiles(sk.reshape(-1, 128, 8)),
        SFVtLo=lo,
        SFVtHi=hi,
    )


@dataclass(frozen=True)
class NVFP4AttentionRunner:
    """Run QK, softmax and PV attention on quantized, bound Q/K/V tensors.

    Calling the runner or ``launch()`` writes and returns the caller-owned
    output. Prepare a new runner when input values or bindings change.
    """

    main_kwargs: dict
    out: torch.Tensor
    entry: object
    arguments: tuple

    def launch(self):
        # The binding encodes host-known TMA descriptors by value. No device
        # descriptor update, scratch allocation or graph capture occurs here.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch


def _arch_of(capability):
    arch = f"sm_{capability[0]}{capability[1]}a"
    if arch not in ARCHES:
        raise ValueError(
            f"NVFP4 attention requires compute capability {[a[3:-1] for a in ARCHES]}, "
            f"got {capability[0]}.{capability[1]}"
        )
    return arch


def bind_attention_payload(main_kwargs, out):
    """Bind prepacked buffers to the generated physical argument order."""
    arch = _arch_of(_device_facts(out.device)[0])
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), main_kwargs["grid"], strict=True))
    arguments = tuple(
        grid[key] if kind == "grid" else main_kwargs[key]
        for kind, key in PROGRAM["arg_plan"]
    )
    module = load_cake_nvfp4_attention_module(arch)
    entry = getattr(module, PROGRAM["ffi_entry"])
    return NVFP4AttentionRunner(main_kwargs, out, entry, arguments)


def prepare_nvfp4_attention(q, k, v, out, *, backend="cake"):
    if backend != "cake":
        raise ValueError("NVFP4 attention supports backend='cake'")
    if (
        q.ndim != 4
        or q.shape[-1] != HEAD_DIM
        or k.shape != q.shape
        or v.shape != q.shape
        or out.shape != q.shape
    ):
        raise ValueError("Expected matching [B,H,S,128] inputs and output")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, out)):
        raise TypeError("NVFP4 attention inputs and output must be BF16")
    if not all(x.is_cuda and x.device == q.device for x in (q, k, v, out)):
        raise ValueError("Expected CUDA tensors on one device")
    if not out.is_contiguous():
        raise ValueError("The output must be a contiguous [B,H,S,128] tensor")
    batch, heads, seqlen, dim = q.shape
    if min(batch, heads, seqlen) <= 0 or seqlen % TILE_ROWS:
        raise ValueError(
            f"This route requires positive extents and S divisible by {TILE_ROWS}"
        )
    capability, sm_count = _device_facts(q.device)
    _arch_of(capability)
    bh = batch * heads
    tiles = bh * (seqlen // TILE_ROWS)
    grid_x = 2 * min(sm_count // 2, tiles)
    kwargs = dict(
        grid=(grid_x, 1, 1),
        **pack_nvfp4_attention_inputs(q, k, v),
        O=out.reshape(bh, seqlen, dim),
        seqlen_q=seqlen,
        seqlen_kv=seqlen,
        q_stride=seqlen,
        kv_stride=seqlen,
        total_bh=bh,
        softmax_scale_log2=1.0 / math.sqrt(dim) / math.log(2.0),
    )
    return bind_attention_payload(kwargs, out)
