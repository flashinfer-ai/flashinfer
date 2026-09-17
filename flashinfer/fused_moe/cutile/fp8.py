# Copyright (c) 2026 by FlashInfer team.
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

"""FP8-weight and MXFP4 x MXFP8 cuTile MoE kernels.

Activation precision is identical for both GEMMs. A8 accepts BF16 at the API
boundary and quantizes each GEMM input dynamically: one E4M3 scale per tensor,
or one E8M0 scale per 32 values for MXFP8. A16 never quantizes activations.
"""

from dataclasses import dataclass
from typing import TypeAlias

import cuda.tile as ct
import torch

from ...cutile.cutile_common import cached_replace_hints
from ..api import ActivationConfig
from .activation import (
    _activation_kernel_args,
    _apply_gated_activation,
    _apply_ungated_activation,
    launch_activation,
)
from .fp4 import _decode_e4m3_bytes, _load_w4a4_weight_tile
from .indexing import needs_int64_indexing
from .moe import (
    GemmConfig,
    Workspace as Bf16Workspace,
    _combine,
    _combine_i64,
    _combine_tile_h,
    _permute,
    allocate_workspace as allocate_bf16_workspace,
)

ConstInt: TypeAlias = ct.Constant[int]
ConstBool: TypeAlias = ct.Constant[bool]
ConstFloat: TypeAlias = ct.Constant[float]


@dataclass
class Workspace(Bf16Workspace):
    input_q: torch.Tensor
    input_scale: torch.Tensor
    input_q_unsorted: torch.Tensor
    input_scale_unsorted: torch.Tensor
    activation_q: torch.Tensor
    activation_scale: torch.Tensor
    amax_partials: torch.Tensor


def allocate_workspace(*, activation_fp8: bool, block_scaled: bool, **kwargs):
    base = allocate_bf16_workspace(**kwargs)
    tokens, hidden = kwargs["num_tokens"], kwargs["hidden_size"]
    inter = kwargs["intermediate_size"]
    device = kwargs["device"]
    max_rows = base.sorted_slots.numel()

    def buffers(rows, columns):
        dtype = torch.float8_e4m3fn if activation_fp8 else torch.bfloat16
        q = torch.empty((rows, columns), dtype=dtype, device=device)
        scale = (
            torch.empty(
                (rows, columns // 32) if block_scaled else (1,),
                dtype=torch.float8_e8m0fnu if block_scaled else torch.float32,
                device=device,
            )
            if activation_fp8
            else torch.empty(0, device=device)
        )
        return q, scale

    # The leading token/assignment slices serve the gather path. The full
    # capacity stores expert-padded rows for the independently tuned sorted path.
    xq, xs = buffers(max_rows, hidden)
    xq_unsorted, xs_unsorted = buffers(tokens, hidden)
    if activation_fp8 and not block_scaled:
        xs_unsorted = xs
    aq, ass = buffers(max_rows, inter)
    if kwargs["is_gated"]:
        base.gemm1_out = torch.empty(
            max_rows, 2 * inter, dtype=torch.bfloat16, device=device
        )
    base.activation_out = torch.empty(
        max_rows, inter, dtype=torch.bfloat16, device=device
    )
    partials = (
        max(
            (tokens * hidden + 4095) // 4096,
            (max_rows * inter + 4095) // 4096,
            ((max_rows + 15) // 16) * ((inter + 127) // 128),
        )
        if activation_fp8 and not block_scaled
        else 0
    )
    return Workspace(
        **vars(base),
        input_q=xq,
        input_scale=xs,
        input_q_unsorted=xq_unsorted,
        input_scale_unsorted=xs_unsorted,
        activation_q=aq,
        activation_scale=ass,
        amax_partials=torch.empty(partials, device=device),
    )


@ct.function
def _amax_partials_impl(X, PARTIAL):
    x = ct.astype(
        ct.load(X, (ct.bid(0),), (4096,), padding_mode=ct.PaddingMode.ZERO), ct.float32
    )
    ct.store(PARTIAL, (ct.bid(0),), ct.reshape(ct.max(ct.abs(x), 0), (1,)))


@ct.kernel
def _amax_partials(X, PARTIAL):
    _amax_partials_impl(X, PARTIAL)


@ct.kernel
def _amax_partials_i64(X: ct.IndexedWithInt64, PARTIAL: ct.IndexedWithInt64):
    _amax_partials_impl(X, PARTIAL)


@ct.function
def _amax_partials_rows_impl(
    X,
    PARTIAL,
    POST_PAD,
    ROW_SIZE: ConstInt,
    I64: ConstBool,
):
    x = ct.astype(
        ct.load(X, (ct.bid(0),), (4096,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    offsets = ct.bid(0) * 4096 + ct.arange(4096, dtype=ct.int32)
    if I64:
        offsets = ct.astype(offsets, ct.int64)
    valid_elements = (
        ct.astype(ct.load(POST_PAD, (0,), (1,)).item(), offsets.dtype) * ROW_SIZE
    )
    x = ct.where(offsets < valid_elements, x, 0.0)
    ct.store(PARTIAL, (ct.bid(0),), ct.reshape(ct.max(ct.abs(x), 0), (1,)))


@ct.kernel
def _amax_partials_rows(X, PARTIAL, POST_PAD, ROW_SIZE: ConstInt):
    _amax_partials_rows_impl(X, PARTIAL, POST_PAD, ROW_SIZE, False)


@ct.kernel
def _amax_partials_rows_i64(
    X: ct.IndexedWithInt64,
    PARTIAL: ct.IndexedWithInt64,
    POST_PAD,
    ROW_SIZE: ConstInt,
):
    _amax_partials_rows_impl(X, PARTIAL, POST_PAD, ROW_SIZE, True)


@ct.kernel
def _amax_scale(PARTIAL, SCALE):
    maximum = ct.full((), 0.0, ct.float32)
    for i in range(ct.cdiv(PARTIAL.shape[0], 1024)):
        values = ct.load(PARTIAL, (i,), (1024,), padding_mode=ct.PaddingMode.ZERO)
        maximum = ct.maximum(maximum, ct.max(values, 0))
    ct.store(SCALE, (0,), ct.reshape(ct.maximum(maximum / 448.0, 2.0**-126), (1,)))


@ct.kernel
def _amax_scale_tiles(
    PARTIAL,
    SCALE,
    POST_PAD,
    TILE_M: ConstInt,
    GRID_N: ConstInt,
):
    valid = ct.cdiv(ct.load(POST_PAD, (0,), (1,)).item(), TILE_M) * GRID_N
    maximum = ct.full((), 0.0, ct.float32)
    for i in range(ct.cdiv(PARTIAL.shape[0], 1024)):
        offsets = i * 1024 + ct.arange(1024, dtype=ct.int32)
        values = ct.load(PARTIAL, (i,), (1024,), padding_mode=ct.PaddingMode.ZERO)
        maximum = ct.maximum(maximum, ct.max(ct.where(offsets < valid, values, 0), 0))
    ct.store(SCALE, (0,), ct.reshape(ct.maximum(maximum / 448.0, 2.0**-126), (1,)))


def _quantize_scale(x, scale, partials, valid_rows=None):
    count = (x.numel() + 4095) // 4096
    partials = partials[:count]
    use_i64 = needs_int64_indexing(x, partials)
    if valid_rows is None:
        kernel = _amax_partials_i64 if use_i64 else _amax_partials
        args = (x.reshape(-1), partials)
    else:
        kernel = _amax_partials_rows_i64 if use_i64 else _amax_partials_rows
        args = (x.reshape(-1), partials, valid_rows, x.shape[1])
    stream = torch.cuda.current_stream(x.device)
    ct.launch(stream, (count,), kernel, args)
    ct.launch(stream, (1,), _amax_scale, (partials, scale))


@ct.function
def _quantize_impl(
    X,
    Q,
    S,
    MX: ConstBool,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
):
    # Whole 32-value scale groups, with bounds-checked tails in both dimensions.
    x = ct.astype(
        ct.load(
            X,
            (ct.bid(0), ct.bid(1)),
            (TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    if MX:
        groups = ct.reshape(x, (TILE_M, TILE_K // 32, 32))
        amax = ct.max(ct.abs(groups), 2)
        exponent = ct.ceil(ct.log2(ct.maximum(amax / 448.0, 2.0**-127)))
        exponent = ct.minimum(exponent, 127.0)
        scale = ct.bitcast(ct.astype(exponent + 127, ct.uint8), ct.float8_e8m0fnu)
        normalized = ct.reshape(
            groups / ct.expand_dims(ct.exp2(exponent), 2), (TILE_M, TILE_K)
        )
        ct.store(S, (ct.bid(0), ct.bid(1)), scale)
    else:
        scale = ct.load(S, (0,), (1,)).item()
        normalized = x / scale
    q = ct.astype(ct.minimum(ct.maximum(normalized, -448.0), 448.0), Q.dtype)
    ct.store(Q, (ct.bid(0), ct.bid(1)), q)


@ct.kernel
def _quantize(X, Q, S, MX: ConstBool, TILE_M: ConstInt, TILE_K: ConstInt):
    _quantize_impl(X, Q, S, MX, TILE_M, TILE_K)


@ct.kernel
def _quantize_i64(
    X: ct.IndexedWithInt64,
    Q: ct.IndexedWithInt64,
    S: ct.IndexedWithInt64,
    MX: ConstBool,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
):
    _quantize_impl(X, Q, S, MX, TILE_M, TILE_K)


def quantize(
    x, q, scale, partials, *, block_scaled, valid_rows=None, scale_ready=False
):
    stream = torch.cuda.current_stream(x.device)
    if not block_scaled and not scale_ready:
        _quantize_scale(x, scale, partials, valid_rows)
    tile_m, tile_k, occupancy = (
        (32, 128, 4) if block_scaled and x.numel() >= 2**20 else (16, 128, 0)
    )
    kernel = _quantize_i64 if needs_int64_indexing(x, q, scale) else _quantize
    if occupancy:
        kernel = cached_replace_hints(kernel, occupancy=occupancy)
    ct.launch(
        stream,
        (
            (x.shape[0] + tile_m - 1) // tile_m,
            (x.shape[1] + tile_k - 1) // tile_k,
        ),
        kernel,
        (x, q, scale, block_scaled, tile_m, tile_k),
    )


@ct.function
def _pack_sorted_impl(X, XS, SLOTS, OUT, OS, TOP_K: ConstInt, MX: ConstBool):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    slots = ct.gather(SLOTS, (rows,), check_bounds=True, padding_value=2147483647)
    source_rows = ct.reshape(slots // TOP_K, (16, 1))
    values = ct.gather(
        X,
        (source_rows, ct.reshape(columns, (1, 128))),
        check_bounds=True,
        padding_value=0,
    )
    ct.store(OUT, (ct.bid(0), ct.bid(1)), values)
    if MX:
        groups = ct.bid(1) * 4 + ct.arange(4, dtype=ct.int32)
        scales = ct.gather(
            XS,
            (source_rows, ct.reshape(groups, (1, 4))),
            check_bounds=True,
            padding_value=0,
        )
        ct.store(OS, (ct.bid(0), ct.bid(1)), scales)


@ct.kernel
def _pack_sorted(X, XS, SLOTS, OUT, OS, TOP_K: ConstInt, MX: ConstBool):
    _pack_sorted_impl(X, XS, SLOTS, OUT, OS, TOP_K, MX)


@ct.kernel
def _pack_sorted_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    SLOTS,
    OUT: ct.IndexedWithInt64,
    OS: ct.IndexedWithInt64,
    TOP_K: ConstInt,
    MX: ConstBool,
):
    _pack_sorted_impl(X, XS, SLOTS, OUT, OS, TOP_K, MX)


@ct.function
def _pack_quantize_mx_impl(X, SLOTS, OUT, OS, TOP_K: ConstInt):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    slots = ct.gather(SLOTS, (rows,), check_bounds=True, padding_value=2147483647)
    source_rows = ct.reshape(slots // TOP_K, (16, 1))
    values = ct.astype(
        ct.gather(
            X,
            (source_rows, ct.reshape(columns, (1, 128))),
            check_bounds=True,
            padding_value=0,
        ),
        ct.float32,
    )
    groups = ct.reshape(values, (16, 4, 32))
    amax = ct.max(ct.abs(groups), 2)
    exponent = ct.ceil(ct.log2(ct.maximum(amax / 448.0, 2.0**-127)))
    exponent = ct.minimum(exponent, 127.0)
    scale = ct.bitcast(ct.astype(exponent + 127, ct.uint8), ct.float8_e8m0fnu)
    normalized = ct.reshape(groups / ct.expand_dims(ct.exp2(exponent), 2), (16, 128))
    quantized = ct.astype(ct.minimum(ct.maximum(normalized, -448.0), 448.0), OUT.dtype)
    ct.store(OUT, (ct.bid(0), ct.bid(1)), quantized)
    ct.store(OS, (ct.bid(0), ct.bid(1)), scale)


@ct.kernel
def _pack_quantize_mx(X, SLOTS, OUT, OS, TOP_K: ConstInt):
    _pack_quantize_mx_impl(X, SLOTS, OUT, OS, TOP_K)


@ct.kernel
def _pack_quantize_mx_i64(
    X: ct.IndexedWithInt64,
    SLOTS,
    OUT: ct.IndexedWithInt64,
    OS: ct.IndexedWithInt64,
    TOP_K: ConstInt,
):
    _pack_quantize_mx_impl(X, SLOTS, OUT, OS, TOP_K)


@ct.function
def _pack_quantize_tensor_impl(X, SCALE, SLOTS, OUT, TOP_K: ConstInt):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    slots = ct.gather(SLOTS, (rows,), check_bounds=True, padding_value=2147483647)
    source_rows = ct.reshape(slots // TOP_K, (16, 1))
    values = ct.astype(
        ct.gather(
            X,
            (source_rows, ct.reshape(columns, (1, 128))),
            check_bounds=True,
            padding_value=0,
        ),
        ct.float32,
    )
    scale = ct.load(SCALE, (0,), (1,)).item()
    quantized = ct.astype(
        ct.minimum(ct.maximum(values / scale, -448.0), 448.0), OUT.dtype
    )
    ct.store(OUT, (ct.bid(0), ct.bid(1)), quantized)


@ct.kernel
def _pack_quantize_tensor(X, SCALE, SLOTS, OUT, TOP_K: ConstInt):
    _pack_quantize_tensor_impl(X, SCALE, SLOTS, OUT, TOP_K)


@ct.kernel
def _pack_quantize_tensor_i64(
    X: ct.IndexedWithInt64,
    SCALE,
    SLOTS,
    OUT: ct.IndexedWithInt64,
    TOP_K: ConstInt,
):
    _pack_quantize_tensor_impl(X, SCALE, SLOTS, OUT, TOP_K)


@ct.function
def _gated_activation_quantize_mx_impl(
    X,
    OUT,
    OS,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    base = ct.reshape(rows, (16, 1)) * (2 * INTER) + ct.reshape(columns, (1, 128))
    gate = ct.astype(
        ct.gather(X, (base,), check_bounds=True, padding_value=0), ct.float32
    )
    up = ct.astype(
        ct.gather(X, (base + INTER,), check_bounds=True, padding_value=0),
        ct.float32,
    )
    values = ct.where(
        ct.reshape(columns < INTER, (1, 128)),
        _apply_gated_activation(gate, up, ACT, P1, P2, P3),
        0.0,
    )
    # Match the explicit BF16 activation boundary of the unfused path before
    # deriving MXFP8 block scales.
    values = ct.astype(ct.astype(values, ct.bfloat16), ct.float32)
    groups = ct.reshape(values, (16, 4, 32))
    amax = ct.max(ct.abs(groups), 2)
    exponent = ct.ceil(ct.log2(ct.maximum(amax / 448.0, 2.0**-127)))
    exponent = ct.minimum(exponent, 127.0)
    scale = ct.bitcast(ct.astype(exponent + 127, ct.uint8), ct.float8_e8m0fnu)
    normalized = ct.reshape(groups / ct.expand_dims(ct.exp2(exponent), 2), (16, 128))
    quantized = ct.astype(ct.minimum(ct.maximum(normalized, -448.0), 448.0), OUT.dtype)
    ct.store(OUT, (ct.bid(0), ct.bid(1)), quantized)
    ct.store(OS, (ct.bid(0), ct.bid(1)), scale)


@ct.kernel
def _gated_activation_quantize_mx(
    X,
    OUT,
    OS,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    _gated_activation_quantize_mx_impl(X, OUT, OS, ACT, P1, P2, P3, INTER)


@ct.kernel
def _gated_activation_quantize_mx_i64(
    X: ct.IndexedWithInt64,
    OUT: ct.IndexedWithInt64,
    OS: ct.IndexedWithInt64,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    _gated_activation_quantize_mx_impl(X, OUT, OS, ACT, P1, P2, P3, INTER)


@ct.function
def _gated_activation_amax_impl(
    X,
    PARTIAL,
    POST_PAD,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
    NUM_COLUMN_TILES: ConstInt,
):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    base = ct.reshape(rows, (16, 1)) * (2 * INTER) + ct.reshape(columns, (1, 128))
    gate = ct.astype(
        ct.gather(X, (base,), check_bounds=True, padding_value=0), ct.float32
    )
    up = ct.astype(
        ct.gather(X, (base + INTER,), check_bounds=True, padding_value=0),
        ct.float32,
    )
    values = ct.where(
        ct.reshape(columns < INTER, (1, 128)),
        _apply_gated_activation(gate, up, ACT, P1, P2, P3),
        0.0,
    )
    valid_rows = ct.load(POST_PAD, (0,), (1,)).item()
    values = ct.where(ct.reshape(rows < valid_rows, (16, 1)), values, 0.0)
    # Match the explicit BF16 activation boundary of the unfused path.
    values = ct.astype(ct.astype(values, ct.bfloat16), ct.float32)
    maximum = ct.reshape(ct.max(ct.abs(ct.reshape(values, (-1,))), 0), (1,))
    ct.store(PARTIAL, (ct.bid(0) * NUM_COLUMN_TILES + ct.bid(1),), maximum)


@ct.kernel
def _gated_activation_amax(
    X,
    PARTIAL,
    POST_PAD,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
    NUM_COLUMN_TILES: ConstInt,
):
    _gated_activation_amax_impl(
        X, PARTIAL, POST_PAD, ACT, P1, P2, P3, INTER, NUM_COLUMN_TILES
    )


@ct.kernel
def _gated_activation_amax_i64(
    X: ct.IndexedWithInt64,
    PARTIAL: ct.IndexedWithInt64,
    POST_PAD,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
    NUM_COLUMN_TILES: ConstInt,
):
    _gated_activation_amax_impl(
        X, PARTIAL, POST_PAD, ACT, P1, P2, P3, INTER, NUM_COLUMN_TILES
    )


@ct.function
def _gated_activation_quantize_tensor_impl(
    X,
    SCALE,
    OUT,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    rows = ct.bid(0) * 16 + ct.arange(16, dtype=ct.int32)
    columns = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int32)
    base = ct.reshape(rows, (16, 1)) * (2 * INTER) + ct.reshape(columns, (1, 128))
    gate = ct.astype(
        ct.gather(X, (base,), check_bounds=True, padding_value=0), ct.float32
    )
    up = ct.astype(
        ct.gather(X, (base + INTER,), check_bounds=True, padding_value=0),
        ct.float32,
    )
    values = ct.where(
        ct.reshape(columns < INTER, (1, 128)),
        _apply_gated_activation(gate, up, ACT, P1, P2, P3),
        0.0,
    )
    values = ct.astype(ct.astype(values, ct.bfloat16), ct.float32)
    scale = ct.load(SCALE, (0,), (1,)).item()
    quantized = ct.astype(
        ct.minimum(ct.maximum(values / scale, -448.0), 448.0), OUT.dtype
    )
    ct.store(OUT, (ct.bid(0), ct.bid(1)), quantized)


@ct.kernel
def _gated_activation_quantize_tensor(
    X,
    SCALE,
    OUT,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    _gated_activation_quantize_tensor_impl(X, SCALE, OUT, ACT, P1, P2, P3, INTER)


@ct.kernel
def _gated_activation_quantize_tensor_i64(
    X: ct.IndexedWithInt64,
    SCALE,
    OUT: ct.IndexedWithInt64,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    INTER: ConstInt,
):
    _gated_activation_quantize_tensor_impl(X, SCALE, OUT, ACT, P1, P2, P3, INTER)


@ct.function
def _e8m0_to_bfloat16(scale):
    exponent = ct.astype(ct.bitcast(scale, ct.uint8), ct.uint16)
    bits = ct.where(exponent == 0, 1 << 6, exponent << 7)
    return ct.bitcast(bits, ct.bfloat16)


@ct.function
def _e8m0_to_float32(scale):
    exponent = ct.astype(ct.bitcast(scale, ct.uint8), ct.uint32)
    bits = ct.where(exponent == 0, 1 << 22, exponent << 23)
    return ct.bitcast(bits, ct.float32)


@ct.function
def _sorted_gemm_tile(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    SLOTS,
    OUT,
    AMAX_PARTIAL,
    mblock,
    nblock,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    BF16_DEQUANT: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    SCATTER: ConstBool,
    WRITE_AMAX: ConstBool,
    GRID_N: ConstInt,
    ASSIGNMENTS: ConstInt,
):
    padded = ct.load(POST_PAD, (0,), (1,)).item()
    if mblock * TM < padded:
        expert = ct.load(EXPERTS, (mblock,), (1,)).item()
        accumulator = ct.full((TM, TN), 0, ct.float32)
        up_accumulator = ct.full((TM, TN), 0, ct.float32)
        num_output_blocks = ct.cdiv(OUT.shape[1], TN)
        for ktile in range(ct.cdiv(X.shape[1], TK)):
            a = ct.load(
                X,
                (mblock, ktile),
                (TM, TK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            )
            if W4:
                w, ws = _load_w4a4_weight_tile(W, WS, expert, nblock, ktile, TN, TK, 32)
                w = ct.astype(w, ct.float8_e4m3fn)
                if GATED:
                    up_w, up_ws = _load_w4a4_weight_tile(
                        W,
                        WS,
                        expert,
                        nblock + num_output_blocks,
                        ktile,
                        TN,
                        TK,
                        32,
                    )
                    up_w = ct.astype(up_w, ct.float8_e4m3fn)
            else:
                w = ct.reshape(
                    ct.load(
                        W,
                        (expert, nblock, ktile),
                        (1, TN, TK),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=True,
                        latency=3,
                    ),
                    (TN, TK),
                )
                if W.dtype == ct.uint8:
                    w = _decode_e4m3_bytes(w)
                if GATED:
                    up_w = ct.reshape(
                        ct.load(
                            W,
                            (expert, nblock + num_output_blocks, ktile),
                            (1, TN, TK),
                            padding_mode=ct.PaddingMode.ZERO,
                            allow_tma=True,
                            latency=3,
                        ),
                        (TN, TK),
                    )
                    if W.dtype == ct.uint8:
                        up_w = _decode_e4m3_bytes(up_w)
                if MX:
                    ws = ct.transpose(
                        ct.reshape(
                            ct.load(
                                WS,
                                (expert, nblock, ktile),
                                (1, TN, TK // 32),
                                padding_mode=ct.PaddingMode.ZERO,
                                latency=3,
                            ),
                            (TN, TK // 32),
                        )
                    )
                    if not A8 and not BF16_DEQUANT:
                        ws = _e8m0_to_float32(ws)
                    if GATED:
                        up_ws = ct.transpose(
                            ct.reshape(
                                ct.load(
                                    WS,
                                    (
                                        expert,
                                        nblock + num_output_blocks,
                                        ktile,
                                    ),
                                    (1, TN, TK // 32),
                                    padding_mode=ct.PaddingMode.ZERO,
                                    latency=3,
                                ),
                                (TN, TK // 32),
                            )
                        )
                        if not A8 and not BF16_DEQUANT:
                            up_ws = _e8m0_to_float32(up_ws)
            if A8:
                if MX:
                    activation_scale = ct.load(
                        XS,
                        (mblock, ktile),
                        (TM, TK // 32),
                        padding_mode=ct.PaddingMode.ZERO,
                        latency=3,
                    )
                    accumulator = ct.mma_scaled(
                        a,
                        activation_scale,
                        ct.transpose(w),
                        ws,
                        accumulator,
                    )
                    if GATED:
                        up_accumulator = ct.mma_scaled(
                            a,
                            activation_scale,
                            ct.transpose(up_w),
                            up_ws,
                            up_accumulator,
                        )
                else:
                    accumulator = ct.mma(a, ct.transpose(w), accumulator)
                    if GATED:
                        up_accumulator = ct.mma(a, ct.transpose(up_w), up_accumulator)
            else:
                if MX:
                    if BF16_DEQUANT:
                        groups = ct.reshape(
                            ct.astype(w, ct.bfloat16), (TN, TK // 32, 32)
                        )
                        scales = _e8m0_to_bfloat16(ws)
                    else:
                        groups = ct.reshape(
                            ct.astype(w, ct.float32), (TN, TK // 32, 32)
                        )
                        scales = ct.astype(ws, ct.float32)
                    w = ct.reshape(
                        groups * ct.expand_dims(ct.transpose(scales), 2), (TN, TK)
                    )
                    if GATED:
                        if BF16_DEQUANT:
                            up_groups = ct.reshape(
                                ct.astype(up_w, ct.bfloat16), (TN, TK // 32, 32)
                            )
                            up_scales = _e8m0_to_bfloat16(up_ws)
                        else:
                            up_groups = ct.reshape(
                                ct.astype(up_w, ct.float32), (TN, TK // 32, 32)
                            )
                            up_scales = ct.astype(up_ws, ct.float32)
                        up_w = ct.reshape(
                            up_groups * ct.expand_dims(ct.transpose(up_scales), 2),
                            (TN, TK),
                        )
                accumulator = ct.mma(
                    a, ct.transpose(ct.astype(w, ct.bfloat16)), accumulator
                )
                if GATED:
                    up_accumulator = ct.mma(
                        a,
                        ct.transpose(ct.astype(up_w, ct.bfloat16)),
                        up_accumulator,
                    )
        if not MX:
            scale = ct.load(WS, (expert,), (1,)).item()
            if A8:
                scale = scale * ct.load(XS, (0,), (1,)).item()
            accumulator = accumulator * scale
            if GATED:
                up_accumulator = up_accumulator * scale
        values = ct.astype(accumulator, OUT.dtype)
        if GATED:
            up = ct.astype(up_accumulator, OUT.dtype)
            values = ct.astype(
                _apply_gated_activation(
                    ct.astype(values, ct.float32),
                    ct.astype(up, ct.float32),
                    ACT,
                    P1,
                    P2,
                    P3,
                ),
                OUT.dtype,
            )
        elif ACT >= 0:
            values = ct.astype(
                _apply_ungated_activation(
                    ct.astype(values, ct.float32), ACT, P1, P2, P3
                ),
                OUT.dtype,
            )
        if WRITE_AMAX:
            tile_amax = ct.max(ct.max(ct.abs(ct.astype(values, ct.float32)), 1), 0)
            ct.store(
                AMAX_PARTIAL,
                (mblock * GRID_N + nblock,),
                ct.reshape(tile_amax, (1,)),
            )
        slots = ct.load(SLOTS, (mblock,), (TM,), padding_mode=ct.PaddingMode.ZERO)
        if SCATTER:
            columns = nblock * TN + ct.arange(TN, dtype=ct.int32)
            ct.scatter(
                OUT,
                (ct.reshape(slots, (TM, 1)), ct.reshape(columns, (1, TN))),
                values,
                check_bounds=True,
            )
        else:
            values = ct.where(ct.reshape(slots < ASSIGNMENTS, (TM, 1)), values, 0)
            ct.store(OUT, (mblock, nblock), values)


@ct.function
def _sorted_grouped_impl(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    SLOTS,
    OUT,
    AMAX_PARTIAL,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    BF16_DEQUANT: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    SCATTER: ConstBool,
    WRITE_AMAX: ConstBool,
    GRID_N: ConstInt,
    ASSIGNMENTS: ConstInt,
):
    _sorted_gemm_tile(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        SLOTS,
        OUT,
        AMAX_PARTIAL,
        ct.bid(0),
        ct.bid(1),
        TM,
        TN,
        TK,
        A8,
        MX,
        W4,
        BF16_DEQUANT,
        GATED,
        ACT,
        P1,
        P2,
        P3,
        SCATTER,
        WRITE_AMAX,
        GRID_N,
        ASSIGNMENTS,
    )


@ct.kernel
def _sorted_grouped(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    SLOTS,
    OUT,
    AMAX_PARTIAL,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    BF16_DEQUANT: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    SCATTER: ConstBool,
    WRITE_AMAX: ConstBool,
    GRID_N: ConstInt,
    ASSIGNMENTS: ConstInt,
):
    _sorted_grouped_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        SLOTS,
        OUT,
        AMAX_PARTIAL,
        TM,
        TN,
        TK,
        A8,
        MX,
        W4,
        BF16_DEQUANT,
        GATED,
        ACT,
        P1,
        P2,
        P3,
        SCATTER,
        WRITE_AMAX,
        GRID_N,
        ASSIGNMENTS,
    )


@ct.kernel
def _sorted_grouped_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    WS: ct.IndexedWithInt64,
    EXPERTS,
    POST_PAD,
    SLOTS,
    OUT: ct.IndexedWithInt64,
    AMAX_PARTIAL,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    BF16_DEQUANT: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    SCATTER: ConstBool,
    WRITE_AMAX: ConstBool,
    GRID_N: ConstInt,
    ASSIGNMENTS: ConstInt,
):
    _sorted_grouped_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        SLOTS,
        OUT,
        AMAX_PARTIAL,
        TM,
        TN,
        TK,
        A8,
        MX,
        W4,
        BF16_DEQUANT,
        GATED,
        ACT,
        P1,
        P2,
        P3,
        SCATTER,
        WRITE_AMAX,
        GRID_N,
        ASSIGNMENTS,
    )


@ct.kernel
def _sorted_grouped_persistent(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    SLOTS,
    OUT,
    AMAX_PARTIAL,
    GRID_M: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    BF16_DEQUANT: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    SCATTER: ConstBool,
    WRITE_AMAX: ConstBool,
    GRID_N: ConstInt,
    ASSIGNMENTS: ConstInt,
):
    padded = ct.load(POST_PAD, (0,), (1,)).item()
    iterations = (ct.cdiv(padded, TM) - ct.bid(0) + GRID_M - 1) // GRID_M
    for iteration in range(iterations):
        _sorted_gemm_tile(
            X,
            XS,
            W,
            WS,
            EXPERTS,
            POST_PAD,
            SLOTS,
            OUT,
            AMAX_PARTIAL,
            ct.bid(0) + iteration * GRID_M,
            ct.bid(1),
            TM,
            TN,
            TK,
            A8,
            MX,
            W4,
            BF16_DEQUANT,
            GATED,
            ACT,
            P1,
            P2,
            P3,
            SCATTER,
            WRITE_AMAX,
            GRID_N,
            ASSIGNMENTS,
        )


@ct.function
def _sorted_grouped_gemm1_fused_mx_impl(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    OUT,
    OUT_SCALE,
    TM: ConstInt,
    TI: ConstInt,
    TK: ConstInt,
    W4: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
):
    mblock = ct.bid(0)
    iblock = ct.bid(1)
    padded = ct.load(POST_PAD, (0,), (1,)).item()
    if mblock * TM < padded:
        expert = ct.load(EXPERTS, (mblock,), (1,)).item()
        num_intermediate_blocks = ct.cdiv(OUT.shape[1], TI)
        gate_accumulator = ct.zeros((TM, TI), dtype=ct.float32)
        up_accumulator = ct.zeros((TM, TI), dtype=ct.float32)
        for ktile in range(ct.cdiv(X.shape[1], TK)):
            activation = ct.load(
                X,
                (mblock, ktile),
                (TM, TK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            )
            activation_scale = ct.load(
                XS,
                (mblock, ktile),
                (TM, TK // 32),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            )
            if W4:
                gate_weight, gate_scale = _load_w4a4_weight_tile(
                    W, WS, expert, iblock, ktile, TI, TK, 32
                )
                gate_weight = ct.astype(gate_weight, ct.float8_e4m3fn)
                if GATED:
                    up_weight, up_scale = _load_w4a4_weight_tile(
                        W,
                        WS,
                        expert,
                        iblock + num_intermediate_blocks,
                        ktile,
                        TI,
                        TK,
                        32,
                    )
                    up_weight = ct.astype(up_weight, ct.float8_e4m3fn)
            else:
                gate_weight = ct.reshape(
                    ct.load(
                        W,
                        (expert, iblock, ktile),
                        (1, TI, TK),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=True,
                        latency=3,
                    ),
                    (TI, TK),
                )
                gate_scale = ct.transpose(
                    ct.reshape(
                        ct.load(
                            WS,
                            (expert, iblock, ktile),
                            (1, TI, TK // 32),
                            padding_mode=ct.PaddingMode.ZERO,
                            latency=3,
                        ),
                        (TI, TK // 32),
                    )
                )
                if GATED:
                    up_weight = ct.reshape(
                        ct.load(
                            W,
                            (expert, iblock + num_intermediate_blocks, ktile),
                            (1, TI, TK),
                            padding_mode=ct.PaddingMode.ZERO,
                            allow_tma=True,
                            latency=3,
                        ),
                        (TI, TK),
                    )
                    up_scale = ct.transpose(
                        ct.reshape(
                            ct.load(
                                WS,
                                (
                                    expert,
                                    iblock + num_intermediate_blocks,
                                    ktile,
                                ),
                                (1, TI, TK // 32),
                                padding_mode=ct.PaddingMode.ZERO,
                                latency=3,
                            ),
                            (TI, TK // 32),
                        )
                    )
            gate_accumulator = ct.mma_scaled(
                activation,
                activation_scale,
                ct.transpose(gate_weight),
                gate_scale,
                gate_accumulator,
            )
            if GATED:
                up_accumulator = ct.mma_scaled(
                    activation,
                    activation_scale,
                    ct.transpose(up_weight),
                    up_scale,
                    up_accumulator,
                )
        gate = ct.astype(ct.astype(gate_accumulator, ct.bfloat16), ct.float32)
        if GATED:
            up = ct.astype(ct.astype(up_accumulator, ct.bfloat16), ct.float32)
            values = _apply_gated_activation(gate, up, ACT, P1, P2, P3)
        else:
            values = _apply_ungated_activation(gate, ACT, P1, P2, P3)
        values = ct.astype(ct.astype(values, ct.bfloat16), ct.float32)
        groups = ct.reshape(values, (TM, TI // 32, 32))
        amax = ct.max(ct.abs(groups), 2)
        exponent = ct.ceil(ct.log2(ct.maximum(amax / 448.0, 2.0**-127)))
        exponent = ct.minimum(exponent, 127.0)
        scale = ct.bitcast(ct.astype(exponent + 127, ct.uint8), ct.float8_e8m0fnu)
        normalized = ct.reshape(groups / ct.expand_dims(ct.exp2(exponent), 2), (TM, TI))
        quantized = ct.astype(
            ct.minimum(ct.maximum(normalized, -448.0), 448.0), OUT.dtype
        )
        ct.store(OUT, (mblock, iblock), quantized)
        ct.store(OUT_SCALE, (mblock, iblock), scale)


@ct.kernel
def _sorted_grouped_gemm1_fused_mx(
    X,
    XS,
    W,
    WS,
    EXPERTS,
    POST_PAD,
    OUT,
    OUT_SCALE,
    TM: ConstInt,
    TI: ConstInt,
    TK: ConstInt,
    W4: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
):
    _sorted_grouped_gemm1_fused_mx_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        OUT,
        OUT_SCALE,
        TM,
        TI,
        TK,
        W4,
        GATED,
        ACT,
        P1,
        P2,
        P3,
    )


@ct.kernel
def _sorted_grouped_gemm1_fused_mx_i64(
    X: ct.IndexedWithInt64,
    XS: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    WS: ct.IndexedWithInt64,
    EXPERTS,
    POST_PAD,
    OUT: ct.IndexedWithInt64,
    OUT_SCALE: ct.IndexedWithInt64,
    TM: ConstInt,
    TI: ConstInt,
    TK: ConstInt,
    W4: ConstBool,
    GATED: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
):
    _sorted_grouped_gemm1_fused_mx_impl(
        X,
        XS,
        W,
        WS,
        EXPERTS,
        POST_PAD,
        OUT,
        OUT_SCALE,
        TM,
        TI,
        TK,
        W4,
        GATED,
        ACT,
        P1,
        P2,
        P3,
    )


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
    TOP_K: ConstInt,
    GRID_M: ConstInt,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
    I64: ConstBool,
):
    padded = ct.load(POST_PAD, (0,), (1,)).item()
    nblock = ct.bid(1)
    ns = ct.reshape(nblock * TN + ct.arange(TN, dtype=ct.int32), (1, TN))
    iterations = (ct.cdiv(padded, TM) - ct.bid(0) + GRID_M - 1) // GRID_M
    for iteration in range(iterations):
        mblock = ct.bid(0) + iteration * GRID_M
        if mblock * TM < padded:
            expert = ct.load(EXPERTS, (mblock,), (1,)).item()
            slots = ct.load(SLOTS, (mblock,), (TM,), padding_mode=ct.PaddingMode.ZERO)
            rows = slots // TOP_K
            if I64:
                rows = ct.astype(rows, ct.int64)
            acc = ct.full((TM, TN), 0, ct.float32)
            for ktile in range(ct.cdiv(K, TK)):
                ks = ct.reshape(ktile * TK + ct.arange(TK, dtype=ct.int32), (1, TK))
                # Padded routing slots are sentinels, so these loads must be checked.
                a = ct.gather(
                    X,
                    (ct.reshape(rows, (TM, 1)), ks),
                    check_bounds=True,
                    padding_value=0,
                    latency=3,
                )
                if W4:
                    w, ws = _load_w4a4_weight_tile(
                        W, WS, expert, nblock, ktile, TN, TK, 32
                    )
                    # cuTile requires equal MMA operand dtypes. E2M1 -> E4M3 is
                    # exact; do NOT apply the E8M0 scale before this conversion.
                    w = ct.astype(w, ct.float8_e4m3fn)
                else:
                    w = ct.reshape(
                        ct.load(
                            W,
                            (expert, nblock, ktile),
                            (1, TN, TK),
                            padding_mode=ct.PaddingMode.ZERO,
                            latency=3,
                        ),
                        (TN, TK),
                    )
                    if W.dtype == ct.uint8:
                        # cuTile does not expose E4M3 on SM89. Decode its bits
                        # exactly for BF16 MMA without expanding weight storage.
                        w = _decode_e4m3_bytes(w)
                    if MX:
                        ws = ct.transpose(
                            ct.reshape(
                                ct.load(
                                    WS,
                                    (expert, nblock, ktile),
                                    (1, TN, TK // 32),
                                    padding_mode=ct.PaddingMode.ZERO,
                                    latency=3,
                                ),
                                (TN, TK // 32),
                            )
                        )
                        if not A8:
                            ws = ct.exp2(ct.astype(ws, ct.int32) - 127)
                if A8:
                    if MX:
                        groups = ct.reshape(
                            ktile * (TK // 32) + ct.arange(TK // 32, dtype=ct.int32),
                            (1, TK // 32),
                        )
                        ass = ct.gather(
                            XS,
                            (ct.reshape(rows, (TM, 1)), groups),
                            check_bounds=True,
                            padding_value=0,
                            latency=3,
                        )
                        acc = ct.mma_scaled(a, ass, ct.transpose(w), ws, acc)
                    else:
                        acc = ct.mma(a, ct.transpose(w), acc)
                else:
                    if MX:
                        groups = ct.reshape(
                            ct.astype(w, ct.float32), (TN, TK // 32, 32)
                        )
                        w = ct.reshape(
                            groups
                            * ct.expand_dims(
                                ct.transpose(ct.astype(ws, ct.float32)), 2
                            ),
                            (TN, TK),
                        )
                    acc = ct.mma(a, ct.transpose(ct.astype(w, ct.bfloat16)), acc)
            if not MX:
                scale = ct.load(WS, (expert,), (1,)).item()
                if A8:
                    scale = scale * ct.load(XS, (0,), (1,)).item()
                acc = acc * scale
            # Match the explicit BF16 GEMM-output boundary used by gated paths.
            values = ct.astype(acc, OUT.dtype)
            if ACT >= 0:
                activated = _apply_ungated_activation(
                    ct.astype(values, ct.float32), ACT, P1, P2, P3
                )
                values = ct.astype(activated, OUT.dtype)
            ct.scatter(
                OUT,
                (ct.reshape(slots, (TM, 1)), ns),
                values,
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
    TOP_K: ConstInt,
    GRID_M: ConstInt,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
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
        TOP_K,
        GRID_M,
        K,
        TM,
        TN,
        TK,
        A8,
        MX,
        W4,
        ACT,
        P1,
        P2,
        P3,
        False,
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
    TOP_K: ConstInt,
    GRID_M: ConstInt,
    K: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    A8: ConstBool,
    MX: ConstBool,
    W4: ConstBool,
    ACT: ConstInt,
    P1: ConstFloat,
    P2: ConstFloat,
    P3: ConstFloat,
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
        TOP_K,
        GRID_M,
        K,
        TM,
        TN,
        TK,
        A8,
        MX,
        W4,
        ACT,
        P1,
        P2,
        P3,
        True,
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
    block_scaled,
    weight_fp4,
    activation=None,
):
    if not activation_fp8:
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            w = w.view(torch.uint8)
        if block_scaled:
            # E8M0 is not a cuTile dtype on SM89/90; bytes retain exact scales.
            ws = ws.view(torch.uint8)
    grid_m = max(1, min((slots.numel() + block_size - 1) // block_size, out.shape[0]))
    kernel = _grouped_i64 if needs_int64_indexing(x, xs, w, ws, out) else _grouped
    kernel = cached_replace_hints(kernel, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, (out.shape[1] + config.tile_n - 1) // config.tile_n),
        kernel,
        (
            x,
            xs,
            w,
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
            config.tile_k,
            activation_fp8,
            block_scaled,
            weight_fp4,
            *(
                (-1, 0.0, 0.0, 0.0)
                if activation is None
                else _activation_kernel_args(activation)
            ),
        ),
    )


def _pack_sorted_input(
    x,
    xs,
    slots,
    out,
    out_scale,
    *,
    top_k,
    block_scaled,
    quantize_input,
):
    stream = torch.cuda.current_stream(x.device)
    grid = ((out.shape[0] + 15) // 16, (out.shape[1] + 127) // 128)
    if quantize_input:
        if block_scaled:
            kernel = (
                _pack_quantize_mx_i64
                if needs_int64_indexing(x, out, out_scale)
                else _pack_quantize_mx
            )
            ct.launch(stream, grid, kernel, (x, slots, out, out_scale, top_k))
        else:
            kernel = (
                _pack_quantize_tensor_i64
                if needs_int64_indexing(x, out)
                else _pack_quantize_tensor
            )
            ct.launch(stream, grid, kernel, (x, xs, slots, out, top_k))
        return
    kernel = (
        _pack_sorted_i64
        if needs_int64_indexing(x, xs, out, out_scale)
        else _pack_sorted
    )
    ct.launch(
        stream,
        grid,
        kernel,
        (x, xs, slots, out, out_scale, top_k, block_scaled),
    )


def _gated_activation_quantize(
    x,
    out,
    scale,
    partials,
    post_pad,
    activation,
    *,
    block_scaled,
):
    rows, inter = out.shape
    stream = torch.cuda.current_stream(x.device)
    row_tiles = (rows + 15) // 16
    column_tiles = (inter + 127) // 128
    activation_args = (*_activation_kernel_args(activation), inter)
    if block_scaled:
        kernel = (
            _gated_activation_quantize_mx_i64
            if needs_int64_indexing(x, out, scale)
            else _gated_activation_quantize_mx
        )
        ct.launch(
            stream,
            (row_tiles, column_tiles),
            kernel,
            (x.reshape(-1), out, scale, *activation_args),
        )
        return
    count = row_tiles * column_tiles
    partials = partials[:count]
    amax_kernel = (
        _gated_activation_amax_i64
        if needs_int64_indexing(x, partials)
        else _gated_activation_amax
    )
    ct.launch(
        stream,
        (row_tiles, column_tiles),
        amax_kernel,
        (x.reshape(-1), partials, post_pad, *activation_args, column_tiles),
    )
    ct.launch(stream, (1,), _amax_scale, (partials, scale))
    quantize_kernel = (
        _gated_activation_quantize_tensor_i64
        if needs_int64_indexing(x, out)
        else _gated_activation_quantize_tensor
    )
    ct.launch(
        stream,
        (row_tiles, column_tiles),
        quantize_kernel,
        (x.reshape(-1), scale, out, *activation_args),
    )


def sorted_grouped_gemm(
    x,
    xs,
    w,
    ws,
    slots,
    experts,
    post_pad,
    out,
    *,
    config,
    activation_fp8,
    block_scaled,
    weight_fp4,
    activation=None,
    gated_activation=False,
    scatter=False,
    assignments,
    block_size,
    num_sms,
    persistent_factor=0,
    amax_partials=None,
    amax_scale=None,
):
    if not activation_fp8:
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            w = w.view(torch.uint8)
        if block_scaled:
            ws = ws.view(torch.uint8)
    full_grid_m = (slots.numel() + block_size - 1) // block_size
    grid_n = (out.shape[1] + config.tile_n - 1) // config.tile_n
    write_amax = amax_partials is not None
    if write_amax != (amax_scale is not None):
        raise ValueError("amax_partials and amax_scale must be provided together")
    partial_count = full_grid_m * grid_n
    if write_amax:
        if partial_count > amax_partials.numel():
            raise ValueError(
                f"amax partial workspace needs {partial_count} elements, "
                f"but only {amax_partials.numel()} are available"
            )
        partials = amax_partials[:partial_count]
    else:
        partials = out
    use_i64 = needs_int64_indexing(x, xs, w, ws, out)
    grid_m = full_grid_m
    kernel = _sorted_grouped_i64 if use_i64 else _sorted_grouped
    leading_args = ()
    if persistent_factor and not use_i64:
        grid_m = max(1, min(full_grid_m, persistent_factor * num_sms // grid_n))
        kernel = _sorted_grouped_persistent
        leading_args = (grid_m,)
    kernel = cached_replace_hints(kernel, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, grid_n),
        kernel,
        (
            x,
            xs,
            w,
            ws,
            experts,
            post_pad,
            slots,
            out,
            partials,
            *leading_args,
            block_size,
            config.tile_n,
            config.tile_k,
            activation_fp8,
            block_scaled,
            weight_fp4,
            torch.cuda.get_device_capability(x.device)[0] < 10,
            gated_activation,
            *(
                (-1, 0.0, 0.0, 0.0)
                if activation is None
                else _activation_kernel_args(activation)
            ),
            scatter,
            write_amax,
            grid_n,
            assignments,
        ),
    )
    if write_amax:
        ct.launch(
            torch.cuda.current_stream(x.device),
            (1,),
            _amax_scale_tiles,
            (partials, amax_scale, post_pad, block_size, grid_n),
        )


def sorted_grouped_gemm1_fused_mx(
    x,
    xs,
    w,
    ws,
    experts,
    post_pad,
    out,
    out_scale,
    *,
    activation,
    block_size,
    config,
    weight_fp4,
):
    kernel = (
        _sorted_grouped_gemm1_fused_mx_i64
        if needs_int64_indexing(x, xs, w, ws, out, out_scale)
        else _sorted_grouped_gemm1_fused_mx
    )
    kernel = cached_replace_hints(kernel, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (
            (x.shape[0] + block_size - 1) // block_size,
            (out.shape[1] + config.tile_n - 1) // config.tile_n,
        ),
        kernel,
        (
            x,
            xs,
            w,
            ws,
            experts,
            post_pad,
            out,
            out_scale,
            block_size,
            config.tile_n,
            config.tile_k,
            weight_fp4,
            activation.is_gated,
            *_activation_kernel_args(activation),
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
    activation: ActivationConfig,
    activation_fp8: bool,
    block_scaled: bool,
    weight_fp4: bool,
    block_size: int,
    gemm1_config: GemmConfig,
    gemm2_config: GemmConfig,
    sorted_input: bool = False,
    fuse_quantization: bool = False,
    num_sms: int = 0,
    persistent_factors: tuple[int, int] = (0, 0),
    fuse_gemm1_quant: bool = False,
):
    tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    assignments = tokens * top_k
    slots, experts, post_pad = _permute(topk_ids, w1.shape[0], block_size, workspace)
    if sorted_input:
        rows = slots.numel()
        x = workspace.input_q[:rows]
        xs = workspace.input_scale[:rows] if block_scaled else workspace.input_scale
        if activation_fp8:
            if fuse_quantization:
                if not block_scaled:
                    _quantize_scale(hidden_states, xs, workspace.amax_partials)
                _pack_sorted_input(
                    hidden_states,
                    xs,
                    slots,
                    x,
                    xs,
                    top_k=top_k,
                    block_scaled=block_scaled,
                    quantize_input=True,
                )
            else:
                unsorted_x = workspace.input_q_unsorted[:tokens]
                unsorted_xs = (
                    workspace.input_scale_unsorted[:tokens]
                    if block_scaled
                    else workspace.input_scale_unsorted
                )
                quantize(
                    hidden_states,
                    unsorted_x,
                    unsorted_xs,
                    workspace.amax_partials,
                    block_scaled=block_scaled,
                )
                _pack_sorted_input(
                    unsorted_x,
                    unsorted_xs,
                    slots,
                    x,
                    xs,
                    top_k=top_k,
                    block_scaled=block_scaled,
                    quantize_input=False,
                )
        else:
            _pack_sorted_input(
                hidden_states,
                xs,
                slots,
                x,
                xs,
                top_k=top_k,
                block_scaled=False,
                quantize_input=False,
            )
        input_scale = xs
        activation_out = workspace.activation_out[:rows]
        xs = (
            workspace.activation_scale[:rows]
            if block_scaled
            else workspace.activation_scale
        )
        if fuse_gemm1_quant:
            x = workspace.activation_q[:rows]
            sorted_grouped_gemm1_fused_mx(
                workspace.input_q[:rows],
                workspace.input_scale[:rows],
                w1,
                s1,
                experts,
                post_pad,
                x,
                xs,
                activation=activation,
                block_size=block_size,
                config=gemm1_config,
                weight_fp4=weight_fp4,
            )
        else:
            fuse_gated_gemm1 = (
                activation.is_gated
                and not weight_fp4
                # Sparse Hopper SwiGLU reuses the packed input more effectively
                # when gate and up projections share one GEMM1 program.
                and tokens <= 64
                and torch.cuda.get_device_capability(hidden_states.device)[0] == 9
                and activation_out.shape[1] % gemm1_config.tile_n == 0
            )
            g1 = (
                workspace.gemm1_out[:rows]
                if activation.is_gated and not fuse_gated_gemm1
                else activation_out
            )
            gemm1_partial_count = (
                (slots.numel() + block_size - 1)
                // block_size
                * (g1.shape[1] + gemm1_config.tile_n - 1)
                // gemm1_config.tile_n
            )
            fuse_gemm1_amax = (
                activation_fp8
                and not block_scaled
                and (not activation.is_gated or fuse_gated_gemm1)
                and gemm1_partial_count <= workspace.amax_partials.numel()
            )
            sorted_grouped_gemm(
                x,
                input_scale,
                w1,
                s1,
                slots,
                experts,
                post_pad,
                g1,
                config=gemm1_config,
                activation_fp8=activation_fp8,
                block_scaled=block_scaled,
                weight_fp4=weight_fp4,
                activation=(
                    activation if not activation.is_gated or fuse_gated_gemm1 else None
                ),
                gated_activation=fuse_gated_gemm1,
                assignments=assignments,
                block_size=block_size,
                num_sms=num_sms,
                persistent_factor=persistent_factors[0],
                amax_partials=(workspace.amax_partials if fuse_gemm1_amax else None),
                amax_scale=(xs if fuse_gemm1_amax else None),
            )
            x = activation_out
            fused_activation_quantize = (
                activation.is_gated
                and activation_fp8
                and not fuse_gated_gemm1
                and (
                    fuse_quantization
                    or torch.cuda.get_device_capability(hidden_states.device)[0] == 9
                )
            )
            if fused_activation_quantize:
                x = workspace.activation_q[:rows]
                _gated_activation_quantize(
                    g1,
                    x,
                    xs,
                    workspace.amax_partials,
                    post_pad,
                    activation,
                    block_scaled=block_scaled,
                )
            else:
                if activation.is_gated and not fuse_gated_gemm1:
                    launch_activation(g1, activation_out, activation)
                if activation_fp8:
                    x = workspace.activation_q[:rows]
                    quantize(
                        activation_out,
                        x,
                        xs,
                        workspace.amax_partials,
                        block_scaled=block_scaled,
                        valid_rows=post_pad,
                        scale_ready=fuse_gemm1_amax,
                    )
        g2 = workspace.gemm2_out[:assignments]
        sorted_grouped_gemm(
            x,
            xs,
            w2,
            s2,
            slots,
            experts,
            post_pad,
            g2,
            config=gemm2_config,
            activation_fp8=activation_fp8,
            block_scaled=block_scaled,
            weight_fp4=weight_fp4,
            scatter=True,
            assignments=assignments,
            block_size=block_size,
            num_sms=num_sms,
            persistent_factor=persistent_factors[1],
        )
        tile_h = _combine_tile_h(tokens, hidden)
        ct.launch(
            torch.cuda.current_stream(hidden_states.device),
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

    x = hidden_states
    xs = workspace.input_scale_unsorted
    if activation_fp8:
        x = workspace.input_q_unsorted[:tokens]
        xs = xs[:tokens] if block_scaled else xs
        quantize(
            hidden_states, x, xs, workspace.amax_partials, block_scaled=block_scaled
        )
    activation_out = workspace.activation_out[:assignments]
    g1 = workspace.gemm1_out[:assignments] if activation.is_gated else activation_out
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
        block_scaled=block_scaled,
        weight_fp4=weight_fp4,
        activation=None if activation.is_gated else activation,
    )
    if activation.is_gated:
        launch_activation(g1, activation_out, activation)
    x = activation_out
    xs = workspace.activation_scale
    if activation_fp8:
        x = workspace.activation_q[:assignments]
        xs = xs[:assignments] if block_scaled else xs
        quantize(
            activation_out, x, xs, workspace.amax_partials, block_scaled=block_scaled
        )
    g2 = workspace.gemm2_out[:assignments]
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
        block_scaled=block_scaled,
        weight_fp4=weight_fp4,
    )
    tile_h = _combine_tile_h(tokens, hidden)
    ct.launch(
        torch.cuda.current_stream(hidden_states.device),
        (tokens, (hidden + tile_h - 1) // tile_h),
        _combine_i64 if needs_int64_indexing(g2, output) else _combine,
        (g2.view(-1), topk_weights.view(-1), output.view(-1), top_k, hidden, tile_h),
    )
    return output
