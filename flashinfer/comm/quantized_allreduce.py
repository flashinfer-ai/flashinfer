# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 two-shot AllReduce that halves NVLink bandwidth via per-group quantization.

Transfers FP8 (1 byte/element + one FP32 scale per group) instead of BF16
(2 bytes/element) across symmetric memory, then dequants back to BF16.
"""

from collections import namedtuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from ..api_logging import flashinfer_api
from ..utils import register_custom_op
from .torch_symmetric_memory import _enable_symm_mem_for_group

FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
# Number of contiguous elements sharing one FP32 scale factor.
# Smaller = better accuracy, larger = less scale overhead.
SCALE_GROUP_DEFAULT = 256


# Barrier helpers derived from:
# https://github.com/meta-pytorch/kraken/blob/main/kraken/_ptx_utils/symm_mem_barrier.py


@triton.jit
def _get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _get_flat_bid():
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def _send_signal(addrs, sem: tl.constexpr):
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.{sem}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs, sem: tl.constexpr):
    tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.{sem}.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def symm_mem_sync(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,
    hasSubsequentMemAccess: tl.constexpr = False,
):
    """Synchronizes blocks across devices via symmetric memory signal pads.

    CUDA graph compatible: resets to zero-filled state after each sync.
    """
    if block_id is None:
        block_id = _get_flat_bid()
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if hasPreviousMemAccess:
        tl.debug_barrier()

    if flat_tid < world_size:
        _send_signal(send_addrs, "release" if hasPreviousMemAccess else "relaxed")
        _wait_signal(wait_addrs, "acquire" if hasSubsequentMemAccess else "relaxed")

    if hasSubsequentMemAccess:
        tl.debug_barrier()


@triton.jit
def _fp8_blockscale_2shot_kernel(
    buf_ptrs_dev,
    signal_pad_ptrs,
    input_ptr,
    output_ptr,
    numel,
    num_groups_total: tl.constexpr,
    data_offset: tl.constexpr,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    P2P_PHASE3: tl.constexpr = False,
):
    pid = tl.program_id(0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    ptrs = buf_ptrs_dev.to(tl.pointer_type(tl.uint64))
    my_buf = tl.load(ptrs + rank).to(tl.pointer_type(tl.uint8))

    my_scales = my_buf.to(tl.pointer_type(tl.float32))
    my_data = (my_buf + data_offset).to(tl.pointer_type(tl.float8e4nv))
    my_data = tl.multiple_of(my_data, 16)

    group_ids = tl.arange(0, GROUPS_PER_BLOCK).to(tl.int64)
    elem_ids = tl.arange(0, SCALE_GROUP).to(tl.int64)

    # Phase 1: quantize BF16 → FP8, write to symmetric memory buffer
    # Uses same stride pattern as Phase 2 so per-block barrier is sufficient.
    block_start = pid * stride_per_program
    while block_start < numel:
        for local_blk in tl.static_range(world_size):
            blk_off = (block_start + local_blk * BLOCK_SIZE).to(tl.int64)
            first_group = blk_off // SCALE_GROUP

            offsets_2d = blk_off + group_ids[:, None] * SCALE_GROUP + elem_ids[None, :]
            mask_2d = offsets_2d < numel

            x = tl.load(input_ptr + offsets_2d, mask=mask_2d, other=0.0).to(tl.float32)
            grp_amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)
            grp_inv_scales = (
                grp_amax / FP8_MAX
            )  # stored for Phase 2/3 dequant (multiply)

            scale_mask = (first_group + group_ids) < num_groups_total
            tl.store(
                my_scales + first_group + group_ids, grp_inv_scales, mask=scale_mask
            )

            grp_quant_scales = FP8_MAX / grp_amax
            x_scaled = tl.clamp(x * grp_quant_scales[:, None], -FP8_MAX, FP8_MAX)

            offsets_1d = blk_off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            mask_1d = offsets_1d < numel
            tl.store(
                my_data + offsets_1d,
                tl.reshape(x_scaled, [BLOCK_SIZE]).to(tl.float8e4nv),
                mask=mask_1d,
            )

        block_start += tl.num_programs(0) * stride_per_program

    # Barrier 1
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Phase 2: each rank reduces its stripe by reading FP8 from all peers
    block_start = pid * stride_per_program
    while block_start < numel:
        stripe_off = (block_start + rank * BLOCK_SIZE).to(tl.int64)
        first_group = stripe_off // SCALE_GROUP

        offsets = stripe_off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < numel

        acc_2d = tl.zeros([GROUPS_PER_BLOCK, SCALE_GROUP], dtype=tl.float32)

        for i in tl.static_range(world_size):
            peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
            peer_scales_ptr = peer.to(tl.pointer_type(tl.float32))
            peer_scales_ptr = tl.multiple_of(peer_scales_ptr, 16)
            peer_inv_scales = tl.load(
                peer_scales_ptr + first_group + group_ids,
                mask=(first_group + group_ids) < num_groups_total,
                other=1.0,
            )
            peer_data = (peer + data_offset).to(tl.pointer_type(tl.float8e4nv))
            peer_data = tl.multiple_of(peer_data, 16)
            fp8_vals = tl.load(peer_data + offsets, mask=mask, other=0.0)
            vals_2d = tl.reshape(
                fp8_vals.to(tl.float32), [GROUPS_PER_BLOCK, SCALE_GROUP]
            )
            acc_2d += vals_2d * peer_inv_scales[:, None]

        out_amax = tl.maximum(tl.max(tl.abs(acc_2d), axis=1), 1e-12)
        out_inv_scales = out_amax / FP8_MAX
        out_quant_scales = FP8_MAX / out_amax

        result_2d = tl.clamp(acc_2d * out_quant_scales[:, None], -FP8_MAX, FP8_MAX)
        result_fp8 = tl.reshape(result_2d, [BLOCK_SIZE]).to(tl.float8e4nv)

        if P2P_PHASE3:
            tl.store(
                my_scales + num_groups_total + first_group + group_ids,
                out_inv_scales,
                mask=(first_group + group_ids) < num_groups_total,
            )
            tl.store(my_data + offsets, result_fp8, mask=mask)
        else:
            for i in tl.static_range(world_size):
                peer = tl.load(ptrs + i).to(tl.pointer_type(tl.uint8))
                tl.store(
                    peer.to(tl.pointer_type(tl.float32))
                    + num_groups_total
                    + first_group
                    + group_ids,
                    out_inv_scales,
                    mask=(first_group + group_ids) < num_groups_total,
                )
                peer_data = (peer + data_offset).to(tl.pointer_type(tl.float8e4nv))
                peer_data = tl.multiple_of(peer_data, 16)
                tl.store(peer_data + offsets, result_fp8, mask=mask)

        # Own stripe: write BF16 directly (no Phase 3 needed for this stripe)
        result_bf16 = tl.reshape(acc_2d, [BLOCK_SIZE]).to(tl.bfloat16)
        tl.store(output_ptr + offsets, result_bf16, mask=mask)

        block_start += tl.num_programs(0) * stride_per_program

    # Barrier 2
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Phase 3: dequant other ranks' reduced FP8 stripes → BF16 output
    if P2P_PHASE3:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = (block_start + r * BLOCK_SIZE).to(tl.int64)
                    first_group = stripe_off // SCALE_GROUP

                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                    mask = offsets < numel

                    reducer = tl.load(ptrs + r).to(tl.pointer_type(tl.uint8))
                    reducer_data = (reducer + data_offset).to(
                        tl.pointer_type(tl.float8e4nv)
                    )
                    reducer_data = tl.multiple_of(reducer_data, 16)
                    fp8_vals = tl.load(reducer_data + offsets, mask=mask, other=0.0)
                    vals_2d = tl.reshape(
                        fp8_vals.to(tl.float32), [GROUPS_PER_BLOCK, SCALE_GROUP]
                    )

                    r_out_inv_scales = tl.load(
                        reducer.to(tl.pointer_type(tl.float32))
                        + num_groups_total
                        + first_group
                        + group_ids,
                        mask=(first_group + group_ids) < num_groups_total,
                        other=1.0,
                    )

                    result_2d = vals_2d * r_out_inv_scales[:, None]
                    result_bf16 = tl.reshape(result_2d, [BLOCK_SIZE]).to(tl.bfloat16)
                    tl.store(output_ptr + offsets, result_bf16, mask=mask)

            block_start += tl.num_programs(0) * stride_per_program
    else:
        block_start = pid * stride_per_program
        while block_start < numel:
            for r in tl.static_range(world_size):
                if r != rank:
                    stripe_off = (block_start + r * BLOCK_SIZE).to(tl.int64)
                    first_group = stripe_off // SCALE_GROUP

                    offsets = stripe_off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                    mask = offsets < numel

                    fp8_vals = tl.load(my_data + offsets, mask=mask, other=0.0)
                    vals_2d = tl.reshape(
                        fp8_vals.to(tl.float32), [GROUPS_PER_BLOCK, SCALE_GROUP]
                    )

                    r_out_inv_scales = tl.load(
                        my_scales + num_groups_total + first_group + group_ids,
                        mask=(first_group + group_ids) < num_groups_total,
                        other=1.0,
                    )

                    result_2d = vals_2d * r_out_inv_scales[:, None]
                    result_bf16 = tl.reshape(result_2d, [BLOCK_SIZE]).to(tl.bfloat16)
                    tl.store(output_ptr + offsets, result_bf16, mask=mask)

            block_start += tl.num_programs(0) * stride_per_program


_cache: dict = {}

_CachedBuf = namedtuple(
    "_CachedBuf", ["buf", "hdl", "max_numel", "scale_group", "world_size"]
)


def _compute_layout(numel: int, scale_group: int, world_size: int):
    num_groups_total = triton.cdiv(numel, scale_group)
    num_out_groups = num_groups_total + world_size - 1
    num_scale_slots = num_groups_total + num_out_groups
    scale_header = num_scale_slots * 4
    data_offset = ((scale_header + 15) // 16) * 16
    packed_size = data_offset + numel
    return num_groups_total, data_offset, packed_size


def _get_buf(numel: int, scale_group: int, device: torch.device, group) -> _CachedBuf:
    """Get or allocate a symmetric memory buffer that fits numel elements.

    Allocates once at the first numel seen per (scale_group, device, group).
    If a later call needs more space, re-allocates at the larger size.
    """
    group_name = group.group_name if hasattr(group, "group_name") else "default"
    key = (scale_group, device, group_name)

    cached = _cache.get(key)
    if cached is not None and cached.max_numel >= numel:
        return cached

    # Allocate (or re-allocate at larger size)
    ws = dist.get_world_size(group)
    alloc_numel = numel
    _enable_symm_mem_for_group(group_name)
    _, _, packed_size = _compute_layout(alloc_numel, scale_group, ws)
    buf = symm_mem.empty((packed_size,), dtype=torch.uint8, device=device)
    hdl = symm_mem.rendezvous(buf, group=group)
    _cache[key] = _CachedBuf(buf, hdl, alloc_numel, scale_group, ws)
    return _cache[key]


def reset():
    """Free all cached symmetric memory buffers."""
    _cache.clear()


def _select_params(numel: int, world_size: int) -> dict:
    """Select kernel parameters based on problem size.

    No config files needed — AllReduce is bandwidth-bound with a simple
    performance model based on NVLink packet efficiency and barrier overhead.

    block_size: elements processed per thread block per iteration. Larger values
        amortize barrier overhead but increase register pressure.
    num_warps: warps per thread block (32 threads each). 16 warps = 512 threads.
    max_num_blocks: grid size cap (H200 has 132 SMs).
    """
    if numel <= 32768:  # <= 64KB
        block_size = 4096
        num_warps = 16
        max_num_blocks = 24
    elif numel <= 524288:  # <= 1MB
        block_size = 4096
        num_warps = 16
        max_num_blocks = 48
    elif numel <= 33554432:  # <= 64MB
        block_size = 4096
        num_warps = 16
        max_num_blocks = 132
    else:  # > 64MB: larger blocks to amortize barrier overhead
        block_size = 16384
        num_warps = 16
        max_num_blocks = 132

    # P2P Phase 3 saves write bandwidth at the cost of cross-GPU reads.
    # Wins when write savings outweigh Phase 3 read latency (large tensors).
    p2p_phase3 = numel >= 16 * 1024 * 1024  # >= 32MB in BF16

    return dict(
        block_size=block_size,
        num_warps=num_warps,
        max_num_blocks=max_num_blocks,
        p2p_phase3=p2p_phase3,
    )


@flashinfer_api
@register_custom_op("flashinfer::quantized_all_reduce", mutates_args=["output"])
def quantized_all_reduce(
    inp: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    scale_group: int = SCALE_GROUP_DEFAULT,
    block_size: int | None = None,
    num_warps: int | None = None,
    max_num_blocks: int | None = None,
    p2p_phase3: bool | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 quantized two-shot AllReduce with per-block scaling.

    Quantizes activations to FP8 with per-block scales before P2P transfer
    via symmetric memory, halving inter-GPU bandwidth during the collective.

    Requires SM90+ (Hopper or later) for FP8 E4M3 support and NVSwitch for
    symmetric memory. World size must divide block_size (default 4096),
    so world_size must be a power of 2 up to 16.

    Best suited for tensors >= 2MB where NVLink bandwidth dominates. For
    smaller tensors, torch.ops.symm_mem.multimem_all_reduce_ or
    flashinfer.comm.allreduce_fusion may be faster due to lower fixed overhead.

    Note:
        The first call allocates a symmetric memory buffer (collective across
        all ranks). Call once with the largest expected tensor during warmup
        to avoid re-allocation during inference.

    Args:
        inp: Input tensor (BF16, contiguous, numel divisible by 8).
        group: Process group for the collective.
        scale_group: Elements per scale group (default 256). Smaller values
            give better accuracy at marginal bandwidth cost.
        block_size: Compute tile size. None = auto (4096 for <= 64MB,
            16384 for larger).
        num_warps: Warps per thread block. None = auto (16).
        max_num_blocks: Max SMs to use. None = auto-tuned by tensor size.
        p2p_phase3: Phase 3 read strategy. None = auto (True for >= 32MB).
        output: Optional pre-allocated output tensor. If None, allocates
            a new tensor.

    Returns:
        AllReduced output tensor (same shape/dtype as input).
    """
    if inp.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {inp.dtype}")
    if not inp.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    numel = inp.numel()
    if numel == 0:
        if output is None:
            return torch.empty_like(inp)
        return output
    ws = dist.get_world_size(group)
    if numel % 8 != 0:
        raise ValueError(f"numel must be divisible by 8, got {numel}")

    major, _ = torch.cuda.get_device_capability(inp.device)
    if major < 9:
        raise RuntimeError(
            f"quantized_all_reduce requires SM90+ (Hopper or later), got SM{major}0"
        )

    # Apply analytical heuristics for unspecified params
    defaults = _select_params(numel, ws)
    if block_size is None:
        block_size = defaults["block_size"]
    if max_num_blocks is None:
        max_num_blocks = defaults["max_num_blocks"]
    if p2p_phase3 is None:
        p2p_phase3 = defaults["p2p_phase3"]
    if num_warps is None:
        num_warps = defaults["num_warps"]

    # Clamp for small tensors
    scale_group = min(scale_group, numel)
    block_size = min(block_size, numel)
    block_size = max(block_size, scale_group)

    if block_size % scale_group != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by"
            f" scale_group ({scale_group})"
        )
    if block_size % ws != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by world_size ({ws})"
        )

    GROUPS_PER_BLOCK = block_size // scale_group
    cached = _get_buf(numel, scale_group, inp.device, group)
    num_groups_total, data_offset, _ = _compute_layout(numel, scale_group, ws)
    stride_per_program = block_size * ws
    num_blocks = min(triton.cdiv(numel, stride_per_program), max_num_blocks)

    if output is None:
        output = torch.empty_like(inp)
    else:
        if output.shape != inp.shape:
            raise ValueError(
                f"output shape {output.shape} must match input shape {inp.shape}"
            )
        if output.dtype != inp.dtype:
            raise ValueError(
                f"output dtype {output.dtype} must match input dtype {inp.dtype}"
            )
        if not output.is_contiguous():
            raise ValueError("Output tensor must be contiguous")

    _fp8_blockscale_2shot_kernel[(num_blocks,)](
        cached.hdl.buffer_ptrs_dev,
        cached.hdl.signal_pad_ptrs_dev,
        inp,
        output,
        numel=numel,
        num_groups_total=num_groups_total,
        data_offset=data_offset,
        stride_per_program=stride_per_program,
        rank=cached.hdl.rank,
        world_size=ws,
        BLOCK_SIZE=block_size,
        SCALE_GROUP=scale_group,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        FP8_MAX=FP8_E4M3_MAX,
        P2P_PHASE3=p2p_phase3,
        num_warps=num_warps,
    )

    return output
