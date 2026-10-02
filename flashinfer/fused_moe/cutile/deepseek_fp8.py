# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE with E4M3 weights and FP32 scales per 128x128 block."""

from dataclasses import dataclass
from typing import TypeAlias

import cuda.tile as ct
import torch

from ...cutile.cutile_common import cached_replace_hints
from ...gemm.kernels.cutile.ragged_block_scaled_bmm_cutile import (
    _block_scaled_matmul_acc,
    _block_scaled_matmul_tile,
)
from ...tllm_enums import ActivationType
from .activation import launch_activation, _row_tile_is_live, _apply_ungated_activation
from .indexing import INT32_INDEX_LIMIT, needs_int64_indexing
from .fp8 import _pack_sorted_input
from .moe import (
    GemmConfig as GemmConfig,
    Workspace as RoutingWorkspace,
    _combine,
    _combine_i64,
    _combine_impl,
    _combine_tile_h,
    _permute,
    allocate_workspace as allocate_routing_workspace,
)

ConstInt: TypeAlias = ct.Constant[int]
ConstBool: TypeAlias = ct.Constant[bool]


def _allocate_scale(rows, groups, device, transposed):
    if transposed:
        return torch.empty(
            (groups, (rows + 3) // 4 * 4), device=device, dtype=torch.float32
        ).T[:rows]
    return torch.empty((rows, groups), device=device, dtype=torch.float32)


@dataclass
class Workspace(RoutingWorkspace):
    sorted_input: torch.Tensor
    sorted_input_scale: torch.Tensor
    input_q: torch.Tensor
    input_scale: torch.Tensor
    activation_q: torch.Tensor
    activation_scale: torch.Tensor


def allocate_workspace(
    *, activation_fp8: bool, scale_transposed=False, **kwargs
) -> Workspace:
    tokens, hidden = kwargs["num_tokens"], kwargs["hidden_size"]
    assignments = tokens * kwargs["top_k"]
    max_block = max(kwargs["block_sizes"])
    sorted_io = assignments >= 128
    kwargs["allocate_gemm1_output"] = True
    kwargs["sorted_io"] = sorted_io
    reuse_fp8_storage = sorted_io and activation_fp8 and max_block >= 64
    if sorted_io and (not activation_fp8 or reuse_fp8_storage):
        kwargs["gemm2_output_rows"] = 0
    base = allocate_routing_workspace(**kwargs)
    rows, inter = base.activation_out.shape
    device = kwargs["device"]
    packed_rows = base.sorted_slots.numel()
    packed_storage = torch.empty(
        (packed_rows * (2 if reuse_fp8_storage else 1), hidden) if sorted_io else (0,),
        device=device,
        dtype=torch.float8_e4m3fn if activation_fp8 else torch.bfloat16,
    )
    packed = packed_storage[:packed_rows] if reuse_fp8_storage else packed_storage
    # GEMM2 overwrites packed GEMM1 inputs after their last read.
    if reuse_fp8_storage:
        base.gemm2_out = packed_storage.view(torch.bfloat16).reshape(
            packed_rows, hidden
        )
    elif sorted_io and not activation_fp8:
        base.gemm2_out = packed
    if not activation_fp8:
        tokens = rows = 0
    return Workspace(
        **vars(base),
        sorted_input=packed,
        sorted_input_scale=(
            _allocate_scale(
                base.sorted_slots.numel(), hidden // 128, device, scale_transposed
            )
            if sorted_io and activation_fp8
            else torch.empty(0, device=device, dtype=torch.float32)
        ),
        input_q=torch.empty((tokens, hidden), device=device, dtype=torch.float8_e4m3fn),
        input_scale=_allocate_scale(tokens, hidden // 128, device, scale_transposed),
        activation_q=torch.empty(
            (rows, inter), device=device, dtype=torch.float8_e4m3fn
        ),
        activation_scale=_allocate_scale(rows, inter // 128, device, scale_transposed),
    )


@ct.function
def _quantize_impl(X, Q, S, TILE_ROWS: ConstInt, ACT: ConstInt):
    values = ct.astype(
        ct.load(
            X,
            (ct.bid(0), ct.bid(1)),
            (TILE_ROWS, 128),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    if ACT >= 0:
        values = ct.astype(
            ct.astype(
                _apply_ungated_activation(values, ACT, 0.0, 0.0, 0.0), ct.bfloat16
            ),
            ct.float32,
        )
    scale = ct.maximum(ct.max(ct.abs(values), 1) * (1.0 / 448.0), 1e-12)
    divisor = ct.expand_dims(scale, 1)
    zero = (values == 0.0) & (divisor > 0.0)
    # Zero lanes divide scale by itself and retain their original sign.
    normalized = ct.truediv(
        ct.where(zero, divisor, values), divisor, rounding_mode=ct.RoundingMode.RN
    )
    normalized = ct.where(zero, values, normalized)
    q = ct.astype(ct.minimum(ct.maximum(normalized, -448.0), 448.0), Q.dtype)
    ct.store(Q, (ct.bid(0), ct.bid(1)), q)
    ct.store(S, (ct.bid(0), ct.bid(1)), ct.reshape(scale, (TILE_ROWS, 1)))


@ct.kernel
def _quantize(
    X,
    Q,
    S,
    TILE_ROWS: ConstInt,
    VALID_ROWS,
    HAS_LIMIT: ConstBool,
    ROW_STEP: ConstInt,
    ACT: ConstInt,
):
    if _row_tile_is_live(VALID_ROWS, HAS_LIMIT, ROW_STEP):
        _quantize_impl(X, Q, S, TILE_ROWS, ACT)


@ct.kernel
def _quantize_i64(
    X: ct.IndexedWithInt64,
    Q: ct.IndexedWithInt64,
    S: ct.IndexedWithInt64,
    TILE_ROWS: ConstInt,
    VALID_ROWS,
    HAS_LIMIT: ConstBool,
    ROW_STEP: ConstInt,
    ACT: ConstInt,
):
    if _row_tile_is_live(VALID_ROWS, HAS_LIMIT, ROW_STEP):
        _quantize_impl(X, Q, S, TILE_ROWS, ACT)


def quantize(
    x,
    q,
    scale,
    valid_rows=None,
    *,
    sorted_rows=False,
    activation_type=-1,
):
    kernel = _quantize_i64 if needs_int64_indexing(x, q, scale) else _quantize
    tile_rows = 1 if x.numel() < 1 << 15 else (8 if x.numel() < 1 << 18 else 16)
    if tile_rows == 16:
        kernel = cached_replace_hints(kernel, occupancy=2)
    ct.launch(
        torch.cuda.current_stream(x.device),
        ((x.shape[0] + tile_rows - 1) // tile_rows, x.shape[1] // 128),
        kernel,
        (
            x,
            q,
            scale,
            tile_rows,
            q if valid_rows is None else valid_rows,
            valid_rows is not None,
            tile_rows if sorted_rows else 0,
            activation_type,
        ),
    )


@ct.function
def _sorted_grouped_impl(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    PAD_OFF,
    OUT,
    grid,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    EXPERT_SCHEDULE: ConstBool,
    LOAD_LATENCY: ConstInt,
    SCALE_LATENCY: ConstInt,
    N_TILES: ConstInt,
    EXPERT_OFFSET: ConstInt,
):
    blocks_n = N_TILES
    grouped_tiles = 8 * blocks_n
    if EXPERT_SCHEDULE:
        job = ct.bid(0)
        end = 0
        for expert in range(ct.astype(W.shape[0], ct.int32)):
            begin = ct.load(PAD_OFF, (expert + EXPERT_OFFSET,), (1,)).item() // TM
            stop = ct.load(PAD_OFF, (expert + EXPERT_OFFSET + 1,), (1,)).item() // TM
            expert_blocks = stop - begin
            count = expert_blocks * blocks_n
            while job >= end and job < end + count:
                local_job = job - end
                first_m = local_job // grouped_tiles * 8
                group_m = ct.minimum(expert_blocks - first_m, 8)
                block = begin + first_m + local_job % group_m
                column = local_job % grouped_tiles // group_m
                acc = _block_scaled_matmul_acc(
                    X,
                    W,
                    XS,
                    WS,
                    expert,
                    block,
                    column,
                    ct.astype(X.shape[1], ct.int32) // 128,
                    TM,
                    TN,
                    128,
                    A8,
                    True,
                    not A8,
                    False,
                    LOAD_LATENCY,
                    SCALE_LATENCY,
                )
                ct.store(OUT, (block, column), ct.astype(acc, OUT.dtype))
                job += grid
            end += count
        return
    live = ct.load(POST_PAD, (0,), (1,)).item()
    blocks_m = ct.cdiv(live, TM)
    for job in range(ct.bid(0), blocks_m * blocks_n, grid):
        first_m = job // grouped_tiles * 8
        group_m = ct.minimum(blocks_m - first_m, 8)
        block = first_m + job % group_m
        column = job % grouped_tiles // group_m
        expert = ct.load(EXPERTS, (block,), (1,)).item()
        _block_scaled_matmul_tile(
            X,
            W,
            XS,
            WS,
            OUT,
            expert,
            block,
            column,
            ct.astype(X.shape[1], ct.int32) // 128,
            TM,
            TN,
            128,
            A8,
            True,
            not A8,
            False,
        )


@ct.kernel
def _sorted_grouped(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    PAD_OFF,
    OUT,
    grid,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    EXPERT_SCHEDULE: ConstBool,
    LOAD_LATENCY: ConstInt,
    SCALE_LATENCY: ConstInt,
    N_TILES: ConstInt,
    EXPERT_OFFSET: ConstInt,
):
    _sorted_grouped_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        PAD_OFF,
        OUT,
        grid,
        TM,
        TN,
        A8,
        EXPERT_SCHEDULE,
        LOAD_LATENCY,
        SCALE_LATENCY,
        N_TILES,
        EXPERT_OFFSET,
    )


@ct.kernel
def _sorted_grouped_w64(
    X,
    XS,
    W: ct.IndexedWithInt64,
    WS,
    EXPERTS,
    POST_PAD,
    PAD_OFF,
    OUT,
    grid,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    EXPERT_SCHEDULE: ConstBool,
    LOAD_LATENCY: ConstInt,
    SCALE_LATENCY: ConstInt,
    N_TILES: ConstInt,
    EXPERT_OFFSET: ConstInt,
):
    _sorted_grouped_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        PAD_OFF,
        OUT,
        grid,
        TM,
        TN,
        A8,
        EXPERT_SCHEDULE,
        LOAD_LATENCY,
        SCALE_LATENCY,
        N_TILES,
        EXPERT_OFFSET,
    )


@ct.kernel
def _sorted_grouped_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    WS: ct.IndexedWithInt64,
    EXPERTS,
    POST_PAD,
    PAD_OFF,
    OUT: ct.IndexedWithInt64,
    grid,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    EXPERT_SCHEDULE: ConstBool,
    LOAD_LATENCY: ConstInt,
    SCALE_LATENCY: ConstInt,
    N_TILES: ConstInt,
    EXPERT_OFFSET: ConstInt,
):
    _sorted_grouped_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        PAD_OFF,
        OUT,
        grid,
        TM,
        TN,
        A8,
        EXPERT_SCHEDULE,
        LOAD_LATENCY,
        SCALE_LATENCY,
        N_TILES,
        EXPERT_OFFSET,
    )


@ct.kernel
def _combine_sorted(Y, WEIGHTS, OUT, ROWS, top_k, H: ConstInt, TILE_H: ConstInt):
    _combine_impl(Y, WEIGHTS, OUT, ROWS, top_k, H, TILE_H, False, True)


@ct.function
def _sorted_tile_impl(
    X, XS, W, WS, EXPERTS, POST_PAD, OUT, TM: ConstInt, TN: ConstInt, N_TILES: ConstInt
):
    job = ct.bid(0)
    block = job // (4 * N_TILES) * 4 + job % 4
    column = job % (4 * N_TILES) // 4
    live = ct.load(POST_PAD, (0,), (1,)).item()
    if block * TM < live:
        expert = ct.load(EXPERTS, (block,), (1,)).item()
        acc = _block_scaled_matmul_acc(
            X,
            W,
            XS,
            WS,
            expert,
            block,
            column,
            ct.astype(X.shape[1], ct.int32) // 128,
            TM,
            TN,
            128,
            False,
            True,
            True,
            False,
        )
        ct.store(OUT, (block, column), ct.astype(acc, OUT.dtype))


@ct.kernel
def _sorted_tile_w64(
    X,
    XS,
    W: ct.IndexedWithInt64,
    WS,
    EXPERTS,
    POST_PAD,
    OUT,
    TM: ConstInt,
    TN: ConstInt,
    N_TILES: ConstInt,
):
    _sorted_tile_impl(X, XS, W, WS, EXPERTS, POST_PAD, OUT, TM, TN, N_TILES)


@ct.kernel
def _sorted_tile_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    WS: ct.IndexedWithInt64,
    EXPERTS,
    POST_PAD,
    OUT: ct.IndexedWithInt64,
    TM: ConstInt,
    TN: ConstInt,
    N_TILES: ConstInt,
):
    _sorted_tile_impl(X, XS, W, WS, EXPERTS, POST_PAD, OUT, TM, TN, N_TILES)


@ct.kernel
def _combine_sorted_i64(
    Y: ct.IndexedWithInt64,
    WEIGHTS,
    OUT: ct.IndexedWithInt64,
    ROWS,
    top_k,
    H: ConstInt,
    TILE_H: ConstInt,
):
    _combine_impl(Y, WEIGHTS, OUT, ROWS, top_k, H, TILE_H, True, True)


@ct.function
def _grouped_impl(
    X,
    XS,
    W,
    WS,
    SLOTS,
    EXPERTS,
    POST_PAD,
    OUT,
    top_k,
    grid_m,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    I64: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    if INPUT_SORTED:
        ct.static_assert(TN <= 128 and 128 % TN == 0)
    initial = ct.bid(0)
    nblock = ct.bid(1)
    live = ct.load(POST_PAD, (0,), (1,)).item()
    blocks = ct.cdiv(live, TM)
    cols = nblock * TN + ct.arange(TN, dtype=ct.int32)
    for block in range(initial, blocks, grid_m):
        slots = ct.load(SLOTS, (block,), (TM,))
        expert = ct.load(EXPERTS, (block,), (1,)).item()
        rows = slots // top_k
        if I64:
            rows = ct.astype(rows, ct.int64)
        acc = ct.zeros((TM, TN), dtype=ct.float32)
        # Sorted loads and 64-bit gathers use a dynamic int32 K loop.
        k_tiles = (
            ct.astype(X.shape[1] // 128, ct.int32)
            if A8 or INPUT_SORTED or I64
            else K // 128
        )
        for kb in range(k_tiles):
            ks = kb * 128 + ct.arange(128, dtype=ct.int32)
            if INPUT_SORTED:
                a = ct.load(
                    X,
                    (block, kb),
                    (TM, 128),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=3,
                )
                w = ct.load(
                    W,
                    (expert * ct.astype(OUT.shape[1] // TN, ct.int32) + nblock, kb),
                    (TN, 128),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=3,
                )
                scale = ct.gather(
                    WS,
                    (expert, ct.full((1,), nblock * TN // 128, dtype=ct.int32), kb),
                    padding_value=0,
                    latency=4,
                ).item()
            else:
                a = ct.gather(
                    X,
                    (ct.reshape(rows, (TM, 1)), ct.reshape(ks, (1, 128))),
                    check_bounds=True,
                    padding_value=0,
                )
                w = ct.reshape(
                    ct.load(
                        W,
                        (expert, nblock, kb),
                        (1, TN, 128),
                        padding_mode=ct.PaddingMode.ZERO,
                        latency=3,
                    ),
                    (TN, 128),
                )
                scale = ct.reshape(
                    ct.gather(
                        WS,
                        (expert, cols // 128, kb),
                        check_bounds=True,
                        padding_value=0,
                    ),
                    (1, TN),
                )
            if A8:
                partial = ct.transpose(
                    ct.mma(w, ct.transpose(a), ct.zeros((TN, TM), dtype=ct.float32))
                )
                scale_rows = (
                    block * TM + ct.arange(TM, dtype=ct.int32) if INPUT_SORTED else rows
                )
                if INPUT_SORTED:
                    xs = ct.gather(
                        XS,
                        (scale_rows, kb),
                        check_bounds=True,
                        padding_value=0,
                        latency=4,
                    )
                else:
                    xs = ct.gather(
                        XS, (scale_rows, kb), check_bounds=True, padding_value=0
                    )
                partial = partial * ct.reshape(xs, (TM, 1))
            else:
                w = ct.astype(w, ct.bfloat16)
                if INPUT_SORTED or I64:
                    partial = ct.transpose(
                        ct.mma(w, ct.transpose(a), ct.zeros((TN, TM), dtype=ct.float32))
                    )
                else:
                    partial = ct.mma(
                        a, ct.transpose(w), ct.zeros((TM, TN), dtype=ct.float32)
                    )
            acc = acc + partial * scale
        out_rows = (
            block * TM + ct.arange(TM, dtype=ct.int32) if OUTPUT_SORTED else slots
        )
        out_rows = ct.astype(out_rows, ct.int64) if I64 else out_rows
        ct.scatter(
            OUT,
            (ct.reshape(out_rows, (TM, 1)), ct.reshape(cols, (1, TN))),
            ct.astype(acc, OUT.dtype),
            check_bounds=True,
        )


@ct.kernel
def _grouped(
    X,
    XS,
    W,
    WS,
    SLOTS,
    EXPERTS,
    POST_PAD,
    OUT,
    top_k,
    grid_m,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_impl(
        X,
        XS,
        W,
        WS,
        SLOTS,
        EXPERTS,
        POST_PAD,
        OUT,
        top_k,
        grid_m,
        K,
        TM,
        TN,
        A8,
        False,
        INPUT_SORTED,
        OUTPUT_SORTED,
    )


@ct.kernel
def _grouped_w64(
    X,
    XS,
    W: ct.IndexedWithInt64,
    WS,
    SLOTS,
    EXPERTS,
    POST_PAD,
    OUT,
    top_k,
    grid_m,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_impl(
        X,
        XS,
        W,
        WS,
        SLOTS,
        EXPERTS,
        POST_PAD,
        OUT,
        top_k,
        grid_m,
        K,
        TM,
        TN,
        A8,
        False,
        INPUT_SORTED,
        OUTPUT_SORTED,
    )


@ct.kernel
def _grouped_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    WS: ct.IndexedWithInt64,
    SLOTS,
    EXPERTS,
    POST_PAD,
    OUT: ct.IndexedWithInt64,
    top_k,
    grid_m,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    A8: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_impl(
        X,
        XS,
        W,
        WS,
        SLOTS,
        EXPERTS,
        POST_PAD,
        OUT,
        top_k,
        grid_m,
        K,
        TM,
        TN,
        A8,
        True,
        INPUT_SORTED,
        OUTPUT_SORTED,
    )


def grouped_gemm(
    x,
    xs,
    w,
    ws,
    slots,
    experts,
    post_pad,
    out,
    *,
    top_k,
    block_size,
    config,
    activation_fp8,
    input_sorted=False,
    output_sorted=False,
    persistent=False,
    expert_offsets=None,
):
    blocks = slots.numel() // block_size
    grid_n = (out.shape[1] + config.tile_n - 1) // config.tile_n
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    if persistent:
        elements_per_expert = w.shape[1] * w.shape[2]
        if (
            expert_offsets is not None
            and needs_int64_indexing(w)
            and elements_per_expert < INT32_INDEX_LIMIT
        ):
            shard_size = min(64, (INT32_INDEX_LIMIT - 1) // elements_per_expert)
            grid = max(1, min(blocks * grid_n, sms * config.occupancy))
            kernel = (
                _sorted_grouped_i64
                if needs_int64_indexing(x, xs, ws, out)
                else _sorted_grouped
            )
            kernel = cached_replace_hints(
                kernel, num_ctas=1, occupancy=config.occupancy
            )
            for offset in range(0, w.shape[0], shard_size):
                # Each view rebases the weight pointer to int32-sized expert ranges.
                ct.launch(
                    torch.cuda.current_stream(x.device),
                    (grid,),
                    kernel,
                    (
                        x,
                        xs,
                        w[offset : offset + shard_size],
                        ws[offset : offset + shard_size],
                        experts,
                        post_pad,
                        expert_offsets,
                        out,
                        grid,
                        block_size,
                        config.tile_n,
                        activation_fp8,
                        True,
                        3,
                        4,
                        grid_n,
                        offset,
                    ),
                )
            return
        if not activation_fp8 and needs_int64_indexing(w):
            kernel = (
                _sorted_tile_i64
                if needs_int64_indexing(x, xs, ws, out)
                else _sorted_tile_w64
            )
            kernel = cached_replace_hints(
                kernel, num_ctas=1, occupancy=config.occupancy
            )
            ct.launch(
                torch.cuda.current_stream(x.device),
                (((blocks + 3) // 4) * 4 * grid_n,),
                kernel,
                (
                    x,
                    xs,
                    w,
                    ws,
                    experts,
                    post_pad,
                    out,
                    block_size,
                    config.tile_n,
                    grid_n,
                ),
            )
            return
        grid = max(1, min(blocks * grid_n, sms * config.occupancy))
        kernel = _sorted_grouped
        if needs_int64_indexing(w):
            kernel = _sorted_grouped_w64
        if needs_int64_indexing(x, xs, ws, out):
            kernel = _sorted_grouped_i64
        kernel = cached_replace_hints(
            kernel, num_ctas=1 if activation_fp8 else None, occupancy=config.occupancy
        )
        ct.launch(
            torch.cuda.current_stream(x.device),
            (grid,),
            kernel,
            (
                x,
                xs,
                w,
                ws,
                experts,
                post_pad,
                out if expert_offsets is None else expert_offsets,
                out,
                grid,
                block_size,
                config.tile_n,
                activation_fp8,
                activation_fp8 and expert_offsets is not None,
                3,
                4,
                grid_n,
                0,
            ),
        )
        return
    waves = 4 if input_sorted else 1
    grid_m = max(1, min(blocks, (sms + grid_n - 1) // grid_n * waves))
    kernel = _grouped
    if needs_int64_indexing(w):
        kernel = _grouped_w64 if activation_fp8 or input_sorted else _grouped_i64
    if needs_int64_indexing(x, xs, ws, out):
        kernel = _grouped_i64
    kernel = cached_replace_hints(kernel, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, grid_n),
        kernel,
        (
            x,
            xs,
            w.reshape(-1, w.shape[-1]) if input_sorted else w,
            ws,
            slots,
            experts,
            post_pad,
            out,
            top_k,
            grid_m,
            x.shape[1],
            block_size,
            config.tile_n,
            activation_fp8,
            input_sorted,
            output_sorted,
        ),
    )


def run_moe(
    hidden_states,
    topk_ids,
    topk_weights,
    w1,
    s1,
    w2,
    s2,
    output,
    workspace,
    *,
    activation,
    activation_fp8,
    block_size,
    gemm1_config,
    gemm2_config,
):
    tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    rows = tokens * top_k
    sorted_io = rows >= 128
    sorted_output = sorted_io and block_size >= 64
    slots, experts, post_pad = _permute(
        topk_ids,
        w1.shape[0],
        block_size,
        workspace,
        inverse_ranks=sorted_output,
    )
    x = hidden_states
    xs = workspace.input_scale[:tokens]
    if activation_fp8:
        x = workspace.input_q[:tokens]
        quantize(
            hidden_states,
            x,
            xs,
            post_pad,
        )
    if sorted_io:
        packed = workspace.sorted_input[: slots.numel()]
        scales = workspace.sorted_input_scale[: slots.numel()]
        _pack_sorted_input(
            x,
            xs,
            slots,
            packed,
            scales,
            top_k=top_k,
            block_scaled=activation_fp8,
            quantize_input=False,
            scale_group_size=128,
            tile_rows=block_size,
            valid_rows=post_pad,
        )
        x = packed
        xs = scales
    stage_rows = slots.numel() if sorted_io else rows
    g1 = workspace.gemm1_out[:stage_rows]
    act = workspace.activation_out[:stage_rows]
    grouped_gemm(
        x,
        xs,
        w1,
        s1,
        slots,
        experts,
        post_pad,
        g1,
        top_k=top_k,
        block_size=block_size,
        config=gemm1_config,
        activation_fp8=activation_fp8,
        input_sorted=sorted_io,
        output_sorted=sorted_io,
        persistent=sorted_output,
        expert_offsets=workspace.pad_off,
    )
    fused = activation_fp8 and activation.type in (
        ActivationType.Relu,
        ActivationType.Relu2,
    )
    if activation.type == ActivationType.Identity or fused:
        act = g1
    else:
        launch_activation(
            g1, act, activation, valid_rows=post_pad if sorted_io else None
        )
    xs = workspace.activation_scale[:stage_rows]
    if activation_fp8:
        x = workspace.activation_q[:stage_rows]
        quantize(
            act,
            x,
            xs,
            post_pad,
            sorted_rows=sorted_io,
            activation_type=int(activation.type) if fused else -1,
        )
    else:
        x = act
    g2 = (
        workspace.gemm2_out[:stage_rows]
        if sorted_output
        else workspace.gemm2_out[:rows]
    )
    grouped_gemm(
        x,
        xs,
        w2,
        s2,
        slots,
        experts,
        post_pad,
        g2,
        top_k=1,
        block_size=block_size,
        config=gemm2_config,
        activation_fp8=activation_fp8,
        input_sorted=sorted_io,
        output_sorted=sorted_output,
        persistent=sorted_output,
        expert_offsets=workspace.pad_off,
    )
    if sorted_output:
        tile_h = _combine_tile_h(tokens, hidden)
        ct.launch(
            torch.cuda.current_stream(output.device),
            (tokens, (hidden + tile_h - 1) // tile_h),
            _combine_sorted_i64
            if needs_int64_indexing(g2, output)
            else _combine_sorted,
            (
                g2.reshape(-1),
                topk_weights.reshape(-1),
                output.reshape(-1),
                workspace.ranks,
                top_k,
                hidden,
                tile_h,
            ),
        )
        return output
    tile_h = _combine_tile_h(tokens, hidden)
    ct.launch(
        torch.cuda.current_stream(output.device),
        (tokens, (hidden + tile_h - 1) // tile_h),
        _combine_i64 if needs_int64_indexing(g2, output) else _combine,
        (
            g2.reshape(-1),
            topk_weights.reshape(-1),
            output.reshape(-1),
            top_k,
            hidden,
            tile_h,
        ),
    )
    return output
