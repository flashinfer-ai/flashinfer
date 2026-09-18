# SPDX-License-Identifier: Apache-2.0
"""Per-head quantization fused with destination-major QKV packing."""

import torch
import triton
import triton.language as tl
from ...comm.ulysses_head_chunk import _positive_int, _storage_ranges_overlap


@triton.jit
def _quant_pack_qkv_kernel(
    output,
    q,
    k,
    v,
    q_scale,
    k_scale,
    v_scale,
    total_elements,
    rows,
    local_heads,
    head_dim,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    dim = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * local_heads + local_head

    q_value = tl.load(
        q + row * stride_q_row + global_head * stride_q_head + dim, mask=mask
    )
    k_value = tl.load(
        k + row * stride_k_row + global_head * stride_k_head + dim, mask=mask
    )
    v_value = tl.load(
        v + row * stride_v_row + global_head * stride_v_head + dim, mask=mask
    )
    qs = tl.load(q_scale + global_head, mask=mask)
    ks = tl.load(k_scale + global_head, mask=mask)
    vs = tl.load(v_scale + global_head, mask=mask)

    # Approximate division can move FP32 values across an E4M3 midpoint.
    # Match the reference's rounded division before the FP8 conversion.
    q_value = tl.maximum(
        -fp8_max, tl.minimum(fp8_max, tl.div_rn(q_value.to(tl.float32), qs))
    )
    k_value = tl.maximum(
        -fp8_max, tl.minimum(fp8_max, tl.div_rn(k_value.to(tl.float32), ks))
    )
    v_value = tl.maximum(
        -fp8_max, tl.minimum(fp8_max, tl.div_rn(v_value.to(tl.float32), vs))
    )
    output_base = head_slot * (3 * head_dim) + dim
    tl.store(output + output_base, q_value, mask=mask)
    tl.store(output + output_base + head_dim, k_value, mask=mask)
    tl.store(output + output_base + 2 * head_dim, v_value, mask=mask)


def quant_pack(q, k, v, scales, *, world_size, out):
    world_size = _positive_int(world_size, "world_size")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or not q.is_cuda:
        raise ValueError("query must be CUDA [S_local,H,D]")
    if any(size <= 0 for size in q.shape) or q.dtype != torch.bfloat16:
        raise ValueError("QKV must have positive dimensions and BF16 dtype")
    for tensor in (q, k, v):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != q.shape:
            raise ValueError("QKV shapes must match")
        if tensor.device != q.device or tensor.dtype != q.dtype:
            raise ValueError("QKV device/dtype must match")
        if any(stride <= 0 for stride in tensor.stride()) or tensor.stride(-1) != 1:
            raise ValueError("QKV must have positive strides and contiguous D")
        if tensor.requires_grad:
            raise ValueError("quantization is inference-only")
    rows, heads, dim = q.shape
    if heads % world_size:
        raise ValueError("heads must be divisible by world_size")
    if not isinstance(scales, (tuple, list)) or len(scales) != 3:
        raise ValueError("three globally agreed per-head scales are required")
    for scale in scales:
        if not isinstance(scale, torch.Tensor) or scale.shape != (heads,):
            raise ValueError("each scale must be [H]")
        if (
            scale.device != q.device
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            raise ValueError("scales must be contiguous FP32 on the QKV device")
        if scale.requires_grad:
            raise ValueError("scales must not require gradients")
    expected = (world_size, rows, heads // world_size, 3 * dim)
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected:
        raise ValueError(f"out must have shape {expected}")
    if (
        out.device != q.device
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous E4M3 on the QKV device")
    if any(_storage_ranges_overlap(out, t) for t in (q, k, v, *scales)):
        raise ValueError("output must not overlap input/scales")
    if out.numel() > 2**31 - 1:
        raise ValueError("payload exceeds the transport int32 element limit")
    # Scale positivity/finiteness is a caller contract: synchronizing a scalar
    # check here would break graph capture and overlap. Clamp amax to >=1e-12
    # before division by 448, and agree scales across ranks before calling.
    with torch.cuda.device(q.device):
        _quant_pack_qkv_kernel[(triton.cdiv(q.numel(), 1024),)](
            out,
            q,
            k,
            v,
            *scales,
            q.numel(),
            rows,
            heads // world_size,
            dim,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            fp8_max=448.0,
            BLOCK=1024,
            num_warps=8,
        )
    return out
