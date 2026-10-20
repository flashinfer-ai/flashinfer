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

Backend of the Cake Sage block-sparse attention kernels as one self-contained
package: the host API bodies, the ``cake_jit`` host plan and the generated CUDA
sources under ``csrc/``.

SM100 / SM103 (B200, B300 / GB300): :func:`bsa_attn_sm100_sage_fwd_cake`, the
Sage-FP8 block-sparse attention forward on ``tcgen05``, and
:func:`sage_fp8_quantize_sm100`, its fused quantizer.  Compared with the
``sm100_blk64`` CuTe-DSL Sage-FP8 path the attention accepts any batch size and
head count, GQA (``num_heads`` a multiple of ``num_kv_heads``), ragged ``seqlen_q``
/ ``seqlen_k`` (no 64-token alignment), per-row KV counts (``q2k_block_nums``,
empty rows produce exact zeros), rank-1/2/3 ``block_sizes`` token masks and an
optional LSE.  Inputs and outputs use the ``bsa_attn_sm100_blk64_fwd`` contract:
BSHD ``float8_e4m3fn`` Q/K/V, ``q_scale[B,H,Sq]``, ``k_scale[B,Hkv,ceil(Sk/16)]``,
``v_scale[Hkv,D]`` or ``v_scale[B,Hkv,D]`` (float32), BF16 BSHD output.

SM120 (RTX PRO 6000 / RTX 5090): :func:`bsa_attn_sm120_blk64_sage_fwd_cake`, the
prequantized INT8-QK / FP8-PV ``mma.sync`` forward of the
``bsa_attn_sm120_blk64_sage_fwd`` contract (INT8 BHSD Q/K, FP8 HDS V, BF16 BHSD
output), one exported specialization per combination of its metadata flags.

Every launch passes its TMA descriptors by value as kernel parameters: a call
writes nothing on the device besides its outputs, owns no workspace, and
captures into a CUDA Graph as one kernel node that replays the descriptor bytes
it captured.  The stable public entry points live in
:mod:`flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake` (SM100 / SM103) and
:mod:`flashinfer.cute_dsl.sparse.bsa_attn_sm120` (SM120); import those modules.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import tvm_ffi

from . import cake_jit as _loader
from .cake_jit import (
    SM120_FLAGS,
    ARCH_BY_CAPABILITY,
    ARCHES,
    BLOCK_M,
    HEAD_DIM,
    K_SCALE_GROUP,
    KV_BLOCK,
    SM100_ARCHES,
    SM120_ARCH,
    TOKENS_PER_CTA,
    arch_for_capability,
    select_sm120,
    sm120_module_key,
)

_LOG2E = 1.4426950408889634
SM100_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    capability
    for capability, arch in ARCH_BY_CAPABILITY.items()
    if arch in SM100_ARCHES
)
SM120_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    capability for capability, arch in ARCH_BY_CAPABILITY.items() if arch == SM120_ARCH
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _device_index(device: torch.device) -> int:
    return int(
        device.index if device.index is not None else torch.cuda.current_device()
    )


def _capability(device: Optional[torch.device]) -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return (int(major), int(minor))


def is_cake_sage_sm100_supported(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) runs the SM100-family programs."""

    return _capability(device) in SM100_CAPABILITIES


def is_cake_sage_sm120_supported(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) runs the SM120 programs."""

    return _capability(device) in SM120_CAPABILITIES


def _sm100_arch(device_index: int) -> str:
    arch = _loader.device_arch(device_index)
    if arch not in SM100_ARCHES:
        raise RuntimeError(
            "Cake Sage-FP8 block-sparse attention requires SM100 (B200/GB200) or SM103 "
            f"(B300/GB300); device {device_index} is {arch}"
        )
    return arch


def _check_tensor(
    name: str, tensor: torch.Tensor, device: torch.device, dtype, shape=None
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def _softmax_scale(softmax_scale: Optional[float], dim: int) -> float:
    if softmax_scale is None:
        softmax_scale = dim**-0.5
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a number")
    value = float(softmax_scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")
    return value


# --------------------------------------------------------------------------- SM100 / SM103


def bsa_attn_sm100_sage_fwd_cake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    *,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sage-FP8 block-sparse attention forward on SM100/SM103 with the Cake kernel.

    Args:
        q: ``float8_e4m3fn`` ``(B, Sq, H, 128)`` contiguous.
        k, v: ``float8_e4m3fn`` ``(B, Sk, Hkv, 128)`` contiguous; ``H % Hkv == 0``.
        q2k_block_index: int32 ``(B, H, ceil(Sq/64), capacity)`` selected 64-token KV blocks.
        block_sparse_num: uniform number of selected blocks per row (``<= capacity``);
            ignored when ``q2k_block_nums`` is given.
        q_scale: float32 ``(B, H, Sq)`` per-token Q scales.
        k_scale: float32 ``(B, Hkv, ceil(Sk/16))`` per-16-token K scales.
        v_scale: float32 ``(Hkv, 128)`` or ``(B, Hkv, 128)`` per-channel V scales.
        block_sizes: optional int32 valid-token counts per KV block, shaped ``(Kb,)``,
            ``(B, Kb)`` or ``(B, Hkv, Kb)`` with ``Kb = ceil(Sk/64)``.
        q2k_block_nums: optional int32 ``(B, H, ceil(Sq/64))`` per-row selected counts
            (rows with 0 blocks produce zeros and ``-inf`` LSE).
        softmax_scale: defaults to ``1/sqrt(128)``.
        return_lse: also return float32 ``(B, H, Sq)`` log-sum-exp (natural log).
        out, lse: optional preallocated outputs (BF16 BSHD / float32 BHS).  With both
            given the call allocates nothing.

    Returns:
        ``(out, lse)``; ``lse`` is ``None`` unless ``return_lse``.

    One kernel launch per call; the three TMA descriptors travel by value in the
    kernel parameters, so the call is CUDA-Graph capturable and shares no device
    state between streams.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor) or t.ndim != 4:
            raise ValueError(f"{name} must be a rank-4 BSHD tensor")
    device = q.device
    if device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    device_index = _device_index(device)
    arch = _sm100_arch(device_index)
    batch, seqlen_q, heads, dim = (int(x) for x in q.shape)
    seqlen_k, kv_heads = int(k.shape[1]), int(k.shape[2])
    if dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {dim}")
    if kv_heads < 1 or heads % kv_heads != 0:
        raise ValueError("num_heads must be a positive multiple of num_kv_heads")
    _check_tensor("q", q, device, torch.float8_e4m3fn)
    _check_tensor("k", k, device, torch.float8_e4m3fn, (batch, seqlen_k, kv_heads, dim))
    _check_tensor("v", v, device, torch.float8_e4m3fn, (batch, seqlen_k, kv_heads, dim))
    q_blocks = _ceil_div(seqlen_q, BLOCK_M)
    k_blocks = _ceil_div(seqlen_k, KV_BLOCK)
    k_scale_groups = _ceil_div(seqlen_k, K_SCALE_GROUP)
    _check_tensor("q_scale", q_scale, device, torch.float32, (batch, heads, seqlen_q))
    _check_tensor(
        "k_scale", k_scale, device, torch.float32, (batch, kv_heads, k_scale_groups)
    )
    if tuple(v_scale.shape) == (kv_heads, dim):
        v_scale_batch_stride = 0
    elif tuple(v_scale.shape) == (batch, kv_heads, dim):
        v_scale_batch_stride = kv_heads * dim
    else:
        raise ValueError(
            f"v_scale must be [Hkv,128]={(kv_heads, dim)} or [B,Hkv,128]={(batch, kv_heads, dim)}, "
            f"got {tuple(v_scale.shape)}"
        )
    _check_tensor("v_scale", v_scale, device, torch.float32)
    if q2k_block_index.ndim != 4 or tuple(q2k_block_index.shape[:3]) != (
        batch,
        heads,
        q_blocks,
    ):
        raise ValueError(f"q2k_block_index must be int32 [B,H,{q_blocks},capacity]")
    _check_tensor("q2k_block_index", q2k_block_index, device, torch.int32)
    capacity = int(q2k_block_index.shape[-1])
    if capacity < 1:
        raise ValueError("q2k_block_index must have positive capacity")

    has_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if has_block_nums:
        _check_tensor(
            "q2k_block_nums",
            q2k_block_nums,
            device,
            torch.int32,
            (batch, heads, q_blocks),
        )
        block_nums_arg = q2k_block_nums
        block_sparse_num = capacity
    else:
        if isinstance(block_sparse_num, bool) or not isinstance(block_sparse_num, int):
            raise TypeError("block_sparse_num must be an integer")
        if block_sparse_num <= 0:
            block_sparse_num = capacity
        if block_sparse_num > capacity:
            raise ValueError(
                f"block_sparse_num ({block_sparse_num}) exceeds capacity ({capacity})"
            )
        block_nums_arg = q2k_block_index

    has_block_sizes = block_sizes is not None and block_sizes.numel() > 0
    if has_block_sizes:
        shape = tuple(block_sizes.shape)
        mode = {
            (k_blocks,): 1,
            (batch, k_blocks): 2,
            (batch, kv_heads, k_blocks): 3,
        }.get(shape)
        if mode is None:
            raise ValueError(
                f"block_sizes shape {shape} must be [Kb], [B,Kb] or [B,Hkv,Kb] with Kb={k_blocks}"
            )
        _check_tensor("block_sizes", block_sizes, device, torch.int32)
        block_sizes_mode, block_sizes_arg = mode, block_sizes
    else:
        block_sizes_mode, block_sizes_arg = 0, q2k_block_index

    scale = _softmax_scale(softmax_scale, dim)

    if out is None:
        out = torch.empty(
            (batch, seqlen_q, heads, dim), dtype=torch.bfloat16, device=device
        )
    else:
        _check_tensor("out", out, device, torch.bfloat16, (batch, seqlen_q, heads, dim))
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, heads, seqlen_q), dtype=torch.float32, device=device
            )
        else:
            _check_tensor("lse", lse, device, torch.float32, (batch, heads, seqlen_q))
        lse_arg = lse
    else:
        lse_arg = q_scale

    total_tiles = batch * heads * q_blocks
    run = _loader.entry("attention", arch)
    # Positional binding in the record's ``arg_plan`` order (buffers, parameters, grid).
    with tvm_ffi.use_torch_stream():
        run(
            q.view(torch.uint8),
            k.view(torch.uint8),
            v.view(torch.uint8),
            out,
            lse_arg,
            q_scale,
            k_scale,
            v_scale,
            q2k_block_index,
            block_nums_arg,
            block_sizes_arg,
            seqlen_q,
            seqlen_k,
            heads,
            kv_heads,
            q_blocks,
            k_blocks,
            k_scale_groups,
            capacity,
            int(block_sparse_num),
            int(has_block_nums),
            block_sizes_mode,
            v_scale_batch_stride,
            total_tiles,
            scale * _LOG2E,
            int(return_lse),
            max(1, min(total_tiles, _loader.sm_count(device_index))),
            1,
            1,
        )
    return out, (lse if return_lse else None)


def sage_fp8_quantize_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fused Sage-FP8 quantization of BF16 BSHD Q/K/V for :func:`bsa_attn_sm100_sage_fwd_cake`.

    Returns ``(q8, k8, v8, q_scale[B,H,Sq], k_scale[B,Hkv,ceil(Sk/16)], v_scale[B,Hkv,128])``
    with symmetric E4M3 quantization: per-token Q scales (``amax/448``), per-16-token
    K scales (shared by the 16 tokens and all 128 channels of a KV head) and per-channel V
    scales over the whole sequence.  Scales are bit-exact with the torch recipe
    ``amax.clamp_min(1e-12) / 448``; two kernel launches plus one ``zero_`` on the V amax scratch.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor) or t.ndim != 4:
            raise ValueError(f"{name} must be a rank-4 BSHD tensor")
    device = q.device
    if device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    arch = _sm100_arch(_device_index(device))
    batch, seqlen_q, heads, dim = (int(x) for x in q.shape)
    seqlen_k, kv_heads = int(k.shape[1]), int(k.shape[2])
    if dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}, got {dim}")
    if kv_heads < 1 or heads % kv_heads != 0:
        raise ValueError("num_heads must be a positive multiple of num_kv_heads")
    _check_tensor("q", q, device, torch.bfloat16)
    _check_tensor("k", k, device, torch.bfloat16, (batch, seqlen_k, kv_heads, dim))
    _check_tensor("v", v, device, torch.bfloat16, (batch, seqlen_k, kv_heads, dim))
    groups = _ceil_div(seqlen_k, K_SCALE_GROUP)
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=device)
    k_scale = torch.empty((batch, kv_heads, groups), dtype=torch.float32, device=device)
    v_scale = torch.empty((batch, kv_heads, dim), dtype=torch.float32, device=device)
    v_amax = torch.zeros((batch, kv_heads, dim), dtype=torch.float32, device=device)
    q_chunks = _ceil_div(seqlen_q, TOKENS_PER_CTA)
    k_chunks = _ceil_div(seqlen_k, TOKENS_PER_CTA)
    chunks = max(q_chunks, k_chunks)
    grid_heads = max(heads, kv_heads)
    run_qk = _loader.entry("quantize_qk_vamax", arch)
    run_v = _loader.entry("quantize_v", arch)
    with tvm_ffi.use_torch_stream():
        run_qk(
            q,
            k,
            v,
            q8,
            k8,
            q_scale,
            k_scale,
            v_amax,
            seqlen_q,
            seqlen_k,
            heads,
            kv_heads,
            grid_heads,
            groups,
            chunks,
            q_chunks,
            k_chunks,
            batch * grid_heads * chunks,
            1,
            1,
        )
        run_v(
            v,
            v8,
            v_amax,
            v_scale,
            seqlen_k,
            kv_heads,
            k_chunks,
            batch * kv_heads * k_chunks,
            1,
            1,
        )
    return q8, k8, v8, q_scale, k_scale, v_scale


# --------------------------------------------------------------------------- SM120


def bsa_attn_sm120_blk64_sage_fwd_cake(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    *,
    out: torch.Tensor,
    tma_descriptor_workspace: Optional[torch.Tensor] = None,
    uniform_block_count: bool = False,
    contiguous_block_indices: bool = False,
) -> torch.Tensor:
    """Run prequantized SM120 Sage block-sparse attention.

    All CUDA storage is caller-owned. ``out`` is contiguous BF16 BHSD.
    Tensor-map descriptors are passed by value as grid-constant kernel
    parameters; ``tma_descriptor_workspace`` is accepted and ignored.
    Q/K use contiguous INT8 BHSD, V uses contiguous FP8 E4M3 HDS, and the
    operation supports MHA, head dimension 128, non-causal forward without LSE.
    The specialization is selected from the five host-known metadata flags
    (:func:`cake_jit.select_sm120`); one kernel launch per call.
    """
    if (
        type(uniform_block_count) is not bool
        or type(contiguous_block_indices) is not bool
    ):
        raise TypeError("uniform_block_count and contiguous_block_indices must be bool")
    if contiguous_block_indices and not uniform_block_count:
        raise ValueError("contiguous_block_indices requires uniform_block_count")
    if not isinstance(block_sparse_num, int) or isinstance(block_sparse_num, bool):
        raise TypeError("block_sparse_num must be an integer")

    tensors = {
        "q_int8": q_int8,
        "k_int8": k_int8,
        "v_fp8": v_fp8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q2k_block_index": q2k_block_index,
        "out": out,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all input and output values must be torch.Tensor")
    device = q_int8.device
    if device.type != "cuda":
        raise ValueError("q_int8 must be a CUDA tensor")
    for name, tensor in tensors.items():
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {device}")

    device_index = _device_index(device)
    arch = _loader.device_arch(device_index)
    if arch != SM120_ARCH:
        raise RuntimeError(
            f"SM120 Sage attention requires compute capability 12.0, got device {device_index} ({arch})"
        )
    if q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8:
        raise TypeError("q_int8 and k_int8 must use torch.int8")
    if v_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("v_fp8 must use torch.float8_e4m3fn")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must use torch.bfloat16")
    for name, tensor in (
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32")
    if q_int8.ndim != 4:
        raise ValueError("q_int8 must be rank-4 BHSD")
    batch, heads, seqlen_q, head_dim = (int(value) for value in q_int8.shape)
    if batch < 1 or heads < 1 or seqlen_q < 1 or head_dim != HEAD_DIM:
        raise ValueError("q_int8 must have positive B/H/Sq and head dimension 128")
    if (
        k_int8.ndim != 4
        or tuple(k_int8.shape[:2]) != (batch, heads)
        or int(k_int8.shape[3]) != HEAD_DIM
    ):
        raise ValueError("k_int8 must be BHSD MHA with head dimension 128")
    seqlen_k = int(k_int8.shape[2])
    if seqlen_k < 1:
        raise ValueError("seqlen_k must be positive")
    q_blocks = _ceil_div(seqlen_q, BLOCK_M)
    k_blocks = _ceil_div(seqlen_k, KV_BLOCK)
    padded_k = k_blocks * KV_BLOCK
    if tuple(v_fp8.shape) != (batch, heads, HEAD_DIM, padded_k):
        raise ValueError("v_fp8 must use contiguous HDS [B,H,128,padded_Sk]")
    if tuple(out.shape) != (batch, heads, seqlen_q, HEAD_DIM):
        raise ValueError("out must be contiguous BF16 BHSD [B,H,Sq,128]")
    if tuple(q_scale.shape) != (batch, heads, _ceil_div(seqlen_q, 128) * 4):
        raise ValueError("q_scale has an invalid shape")
    if tuple(k_scale.shape) != (batch, heads, k_blocks):
        raise ValueError("k_scale has an invalid shape")
    if tuple(v_scale.shape) != (batch, heads, HEAD_DIM):
        raise ValueError("v_scale has an invalid shape")
    if q2k_block_index.ndim != 4 or tuple(q2k_block_index.shape[:3]) != (
        batch,
        heads,
        q_blocks,
    ):
        raise ValueError("q2k_block_index must be [B,H,ceil(Sq/64),capacity]")
    if q2k_block_index.dtype != torch.int32 or int(q2k_block_index.shape[-1]) < 1:
        raise ValueError("q2k_block_index must be int32 with positive capacity")
    q2k_capacity = int(q2k_block_index.shape[-1])
    if not 0 <= block_sparse_num <= q2k_capacity:
        raise ValueError("block_sparse_num must be in [0, q2k capacity]")

    if q2k_block_nums is not None:
        if (
            not isinstance(q2k_block_nums, torch.Tensor)
            or q2k_block_nums.device != device
            or not q2k_block_nums.is_contiguous()
            or q2k_block_nums.dtype != torch.int32
            or tuple(q2k_block_nums.shape) != (batch, heads, q_blocks)
        ):
            raise ValueError("q2k_block_nums must be contiguous int32 [B,H,Q64]")
    q2k_nums_arg = q2k_block_index if q2k_block_nums is None else q2k_block_nums

    block_sizes_mode = 0
    if block_sizes is not None:
        if not isinstance(block_sizes, torch.Tensor) or block_sizes.device != device:
            raise ValueError("block_sizes must be a CUDA tensor on the input device")
        if block_sizes.dtype != torch.int32 or not block_sizes.is_contiguous():
            raise ValueError("block_sizes must be contiguous int32")
        expected = {
            1: (k_blocks,),
            2: (batch, k_blocks),
            3: (batch, heads, k_blocks),
        }.get(block_sizes.ndim)
        if expected is None or tuple(block_sizes.shape) != expected:
            raise ValueError("block_sizes must have the canonical 1D, 2D, or 3D shape")
        block_sizes_mode = int(block_sizes.ndim)
    block_sizes_arg = q2k_block_index if block_sizes is None else block_sizes

    scale = _softmax_scale(softmax_scale, head_dim)

    key = sm120_module_key(
        has_block_nums=q2k_block_nums is not None,
        block_sizes_mode=block_sizes_mode,
        seqlen_k=seqlen_k,
        uniform_block_count=uniform_block_count,
        contiguous_block_indices=contiguous_block_indices,
        block_sparse_num=block_sparse_num,
    )
    record = select_sm120(**dict(zip(SM120_FLAGS, key, strict=True)))
    run = _loader.entry(record, arch)
    with tvm_ffi.use_torch_stream():
        run(
            q_int8,
            k_int8,
            v_fp8.view(torch.uint8),
            out,
            q_scale,
            k_scale,
            v_scale,
            q2k_block_index,
            q2k_nums_arg,
            block_sizes_arg,
            seqlen_q,
            seqlen_k,
            heads,
            q2k_capacity,
            k_blocks,
            block_sparse_num,
            scale,
            q_blocks,
            heads,
            batch,
        )
    return out


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "SM100_CAPABILITIES",
    "SM120_CAPABILITIES",
    "arch_for_capability",
    "bsa_attn_sm100_sage_fwd_cake",
    "bsa_attn_sm120_blk64_sage_fwd_cake",
    "is_cake_sage_sm100_supported",
    "is_cake_sage_sm120_supported",
    "sage_fp8_quantize_sm100",
]
