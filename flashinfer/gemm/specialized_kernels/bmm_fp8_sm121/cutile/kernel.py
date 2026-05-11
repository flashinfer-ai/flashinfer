"""cuTile FP8 BMM backend for SM121 specialized routing."""

from typing import Tuple, TypeAlias

import cuda.tile as ct
import torch

ConstInt: TypeAlias = ct.Constant[int]
NUM_SMS = 48


def _swizzle_2d_from_bid(M, N, tm, tn, group_size_m, bid):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = group_size_m * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size_m
    current_group_size_m = min(num_bid_m - first_bid_m, group_size_m)
    bid_m = first_bid_m + (bid % current_group_size_m)
    bid_n = (bid % num_bid_in_group) // current_group_size_m
    return bid_m, bid_n


@ct.kernel
def bmm_fp8_kernel(
    A,
    B,
    A_scale,
    B_scale,
    out,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
    num_tiles_k: ConstInt,
    group_size_m: ConstInt,
):
    bid = ct.bid(0)
    M = A.shape[1]
    N = B.shape[2]
    sa = ct.load(A_scale, (0,), shape=(1,)).astype(ct.float32)
    sb = ct.load(B_scale, (0,), shape=(1,)).astype(ct.float32)
    scale = sa.item() * sb.item()
    bid_m, bid_n = _swizzle_2d_from_bid(M, N, tm, tn, group_size_m, bid)
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(0, bid_m, k), shape=(1, tm, tk), padding_mode=zero_pad, latency=10
        )
        a = ct.reshape(a, (tm, tk))
        b = ct.load(
            B, index=(0, k, bid_n), shape=(1, tk, tn), padding_mode=zero_pad, latency=10
        )
        b = ct.reshape(b, (tk, tn))
        accumulator = ct.mma(a, b, acc=accumulator)
    accumulator = accumulator * scale
    result = ct.astype(accumulator, ct.bfloat16)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(out, index=(0, bid_m, bid_n), tile=result_3d)


@ct.kernel(occupancy=ct.ByTarget(sm_121=3, default=2))
def bmm_fp8_single_row_kernel(
    A,
    B,
    A_scale,
    B_scale,
    out,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
    num_tiles_k: ConstInt,
):
    bid_n = ct.bid(0)
    sa = ct.load(A_scale, (0,), shape=(1,)).astype(ct.float32)
    sb = ct.load(B_scale, (0,), shape=(1,)).astype(ct.float32)
    scale = sa.item() * sb.item()
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(0, 0, k), shape=(1, tm, tk), padding_mode=zero_pad, latency=10
        )
        a = ct.reshape(a, (tm, tk))
        b = ct.load(
            B, index=(0, k, bid_n), shape=(1, tk, tn), padding_mode=zero_pad, latency=10
        )
        b = ct.reshape(b, (tk, tn))
        accumulator = ct.mma(a, b, acc=accumulator)
    accumulator = accumulator * scale
    result = ct.astype(accumulator, ct.bfloat16)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(out, index=(0, 0, bid_n), tile=result_3d)


@ct.kernel(occupancy=ct.ByTarget(sm_121=3, default=2))
def bmm_fp8_kernel_large_m(
    A,
    B,
    A_scale,
    B_scale,
    out,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
    num_tiles_k: ConstInt,
    group_size_m: ConstInt,
):
    bid = ct.bid(0)
    M = A.shape[1]
    N = B.shape[2]
    sa = ct.load(A_scale, (0,), shape=(1,)).astype(ct.float32)
    sb = ct.load(B_scale, (0,), shape=(1,)).astype(ct.float32)
    scale = sa.item() * sb.item()
    bid_m, bid_n = _swizzle_2d_from_bid(M, N, tm, tn, group_size_m, bid)
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(0, bid_m, k), shape=(1, tm, tk), padding_mode=zero_pad, latency=10
        )
        a = ct.reshape(a, (tm, tk))
        b = ct.load(
            B, index=(0, k, bid_n), shape=(1, tk, tn), padding_mode=zero_pad, latency=10
        )
        b = ct.reshape(b, (tk, tn))
        accumulator = ct.mma(a, b, acc=accumulator)
    accumulator = accumulator * scale
    result = ct.astype(accumulator, ct.bfloat16)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(out, index=(0, bid_m, bid_n), tile=result_3d)


@ct.kernel
def bmm_fp8_persistent_kernel(
    A,
    B,
    A_scale,
    B_scale,
    out,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
    num_tiles_k: ConstInt,
    group_size_m: ConstInt,
):
    bid = ct.bid(0)
    M = A.shape[1]
    N = B.shape[2]
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    upper_bound = num_bid_m * num_bid_n
    num_blocks = ct.num_blocks(0)
    zero_pad = ct.PaddingMode.ZERO
    sa = ct.load(A_scale, (0,), shape=(1,)).astype(ct.float32)
    sb = ct.load(B_scale, (0,), shape=(1,)).astype(ct.float32)
    scale = sa.item() * sb.item()
    for current_bid in range(bid, upper_bound, num_blocks):
        bid_m, bid_n = _swizzle_2d_from_bid(M, N, tm, tn, group_size_m, current_bid)
        accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
        for k in range(num_tiles_k):
            a = ct.load(
                A,
                index=(0, bid_m, k),
                shape=(1, tm, tk),
                padding_mode=zero_pad,
                latency=10,
            )
            a = ct.reshape(a, (tm, tk))
            b = ct.load(
                B,
                index=(0, k, bid_n),
                shape=(1, tk, tn),
                padding_mode=zero_pad,
                latency=10,
            )
            b = ct.reshape(b, (tk, tn))
            accumulator = ct.mma(a, b, acc=accumulator)
        accumulator = accumulator * scale
        result = ct.astype(accumulator, ct.bfloat16)
        result_3d = ct.reshape(result, (1, tm, tn))
        ct.store(out, index=(0, bid_m, bid_n), tile=result_3d)


def _pick_tiles(M: int, N: int, K: int) -> Tuple[int, int, int]:
    if M <= 8:
        tm = 16
        if N >= 8192:
            tn = 128
        elif N >= 4096 and K >= 5376:
            tn = 32
        else:
            tn = 64
        if K >= 8192:
            tk = 512
        elif K >= 4096:
            tk = 256
        elif K >= 2048:
            tk = 256 if N > 128 else 512
        else:
            tk = 128
    elif M <= 16:
        tm = 16
        if N <= 128:
            tn = 64
            tk = 512 if K >= 2048 else 128
        elif N >= 4096 and K >= 5376:
            tn = 32
            tk = 512 if K >= 8192 else 256
        else:
            tn = 128
            if K >= 8192:
                tk = 512
            elif K >= 2048:
                tk = 256
            else:
                tk = 128
    elif M <= 128:
        if N <= 1024 and M >= 72:
            tm = 32 if K >= 4096 else 16
            tn = 64
            tk = 256 if K >= 4096 else 128
        elif (N <= 1024 and 33 <= M < 72) or (M <= 32 and N <= 4096):
            tm = 32
            tn = 64
            tk = 256 if K >= 4096 else 128
        elif M >= 72 and N >= 12288:
            tm = 64
            tn = 128
            tk = 128
        else:
            tm = 32
            tn = 128
            tk = 256 if K >= 4096 else 128
    elif M < 4096:
        if N <= 1024 or N == 5376:
            tm = 64
        else:
            tm = 128
        tn = 128
        tk = 128
    else:
        tm = 128
        tn = 128
        tk = 128
    return tm, tn, tk


_DISPATCH_CACHE: dict[
    tuple[int, int, int], tuple[int, tuple[int, int, int], int, int, int, int]
] = {}

_PATH_SINGLE_ROW = 0
_PATH_PERSISTENT = 1
_PATH_LARGE_M = 2
_PATH_STANDARD = 3


def _build_dispatch(M, N, K):
    tm, tn, tk = _pick_tiles(M, N, K)
    num_blocks_m = (M + tm - 1) // tm
    num_blocks_n = (N + tn - 1) // tn
    total_tiles = num_blocks_m * num_blocks_n
    num_tiles_k = (K + tk - 1) // tk
    if num_blocks_m == 1:
        path = _PATH_SINGLE_ROW
        grid_size = num_blocks_n
    elif 256 <= M < 4096:
        if total_tiles <= NUM_SMS * 4:
            path = _PATH_STANDARD
            grid_size = total_tiles
        else:
            path = _PATH_PERSISTENT
            grid_size = NUM_SMS * 4
    elif M >= 4096:
        path = _PATH_LARGE_M
        grid_size = total_tiles
    else:
        path = _PATH_STANDARD
        grid_size = total_tiles
    grid = (grid_size, 1, 1)
    return path, grid, tm, tn, tk, num_tiles_k


def run(A, B, A_scale, B_scale, out):
    M = A.shape[1]
    K = A.shape[2]
    N = B.shape[2]
    key = (M, N, K)
    cached = _DISPATCH_CACHE.get(key)
    if cached is None:
        cached = _build_dispatch(M, N, K)
        _DISPATCH_CACHE[key] = cached
    path, grid, tm, tn, tk, num_tiles_k = cached

    A_scale_1d = A_scale[None]
    B_scale_1d = B_scale[None]

    stream = torch.cuda.current_stream(device=A.device)

    if path == _PATH_SINGLE_ROW:
        ct.launch(
            stream,
            grid,
            bmm_fp8_single_row_kernel,
            (A, B, A_scale_1d, B_scale_1d, out, tm, tn, tk, num_tiles_k),
        )
    elif path == _PATH_PERSISTENT:
        ct.launch(
            stream,
            grid,
            bmm_fp8_persistent_kernel,
            (A, B, A_scale_1d, B_scale_1d, out, tm, tn, tk, num_tiles_k, 8),
        )
    elif path == _PATH_LARGE_M:
        ct.launch(
            stream,
            grid,
            bmm_fp8_kernel_large_m,
            (A, B, A_scale_1d, B_scale_1d, out, tm, tn, tk, num_tiles_k, 8),
        )
    else:
        ct.launch(
            stream,
            grid,
            bmm_fp8_kernel,
            (A, B, A_scale_1d, B_scale_1d, out, tm, tn, tk, num_tiles_k, 8),
        )
