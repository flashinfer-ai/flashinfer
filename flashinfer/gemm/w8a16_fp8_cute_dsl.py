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

"""W8A16 FP8 GEMM, CuTe-DSL kernels (Blackwell and Hopper).

Same contract as the C++ op in ``w8a16_fp8.py``:
``out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])``.  While the
two implementations coexist this module is reached through
``w8a16_fp8_gemm_cute_dsl``; the variant selection (``algo`` codes, tile walk) is
the C++ dispatchers' (``include/flashinfer/gemm/w8a16_sm100/dispatch.hpp`` and
``tile_order.hpp``, ``w8a16_sm90/dispatch.hpp`` and ``tile_order.hpp``) ported
line by line so the two run the same kernel for a problem.
"""

import math
import os
from typing import Dict, Optional, Tuple

import torch

from ..utils import get_compute_capability, get_device_sm_count

# Blackwell tile configurations by algo code (W8A16Sm100Algo): (weight rows, activation
# rows, k) per MMA tile, 2SM MMA, cluster, transformed A kept in smem.
_SM100_TILES: Dict[int, Tuple[Tuple[int, int, int], bool, Tuple[int, int], bool]] = {
    1: ((256, 128, 128), True, (2, 1), False),  # 2sm256x128
    2: ((128, 128, 128), False, (1, 1), False),  # 1sm128x128
    3: ((256, 256, 64), True, (2, 1), True),  # 2sm256x256, A in smem
    4: ((256, 192, 128), True, (2, 1), False),  # 2sm256x192
    7: ((256, 160, 128), True, (2, 1), False),  # 2sm256x160
}
_SM100_SPLITK = 5
_SM100_STREAMK = 6


def _tiles(x: int, t: int) -> int:
    return -(-x // t)


def _sm100_split_k(m: int, n: int, k: int, num_sms: int) -> int:
    """W8A16Sm100SplitK: slices for the 256x128 tile (1 = do not split)."""
    if k % 128 != 0:
        return 1
    clusters = num_sms // 2 if num_sms > 1 else 1
    units = _tiles(n, 256) * _tiles(m, 128)
    ktiles = k // 128
    if units * 2 <= clusters:
        best = 1
        for s in range(2, 9):
            if ktiles % s != 0 or ktiles // s < 2 or units * s > clusters:
                continue
            best = s
        return best
    if ktiles >= 100:
        waves1 = _tiles(units, clusters)
        for s in range(2, 9):
            if ktiles % s != 0 or ktiles // s < 50:
                continue
            waves = _tiles(units * s, clusters)
            if waves * 100 <= waves1 * s * 85:
                return s
    return 1


def _sm100_use_stream_k(m: int, n: int, k: int, num_sms: int) -> bool:
    """W8A16Sm100UseStreamK."""
    if k % 128 != 0 or k // 128 < 64:
        return False
    clusters = num_sms // 2 if num_sms > 1 else 1
    units = _tiles(n, 256) * _tiles(m, 128)
    if units * 2 <= clusters:
        return False
    waves = _tiles(units, clusters)
    tail = units - (waves - 1) * clusters
    return 2 <= waves <= 3 and tail * 3 <= clusters


def sm100_splits_for(m: int, n: int, k: int, num_sms: int) -> int:
    """W8A16Sm100SplitsFor: the split-K slice count actually used: the model's
    choice, or, when the variant is forced on a grid the model would not split,
    the smallest s >= 2 dividing the k-tile count.  0 = K cannot be split."""
    s = _sm100_split_k(m, n, k, num_sms)
    if s >= 2:
        return s
    ktiles = k // 128
    if k % 128 != 0 or ktiles < 2:
        return 0
    for c in range(2, 9):
        if ktiles % c == 0:
            return c
    return 0


def _sm100_streamk_substitute(m: int, n: int, k: int, num_sms: int) -> Tuple[int, int]:
    """(algo, splits) for a problem the C++ dispatcher runs as stream-K: the DSL has
    no stream-K, so the plain tiles and the split-K slice counts are scored by wave
    quantization (whole waves of k-tiles, the last wave counted whole) plus the
    split-K reduce (its fp32 partial traffic in k-tile-waves), and the cheapest wins.
    Falls back to the plain 256x128 tile when K cannot be sliced."""
    clusters = num_sms // 2 if num_sms > 1 else 1
    ktiles = k // 128
    w_tiles = _tiles(n, 256)

    def plain(tile_a: int, eff: float) -> float:
        return (
            _tiles(w_tiles * _tiles(m, tile_a), clusters)
            * ktiles
            * (tile_a / 128)
            / eff
        )

    best_algo, best_cost = 1, plain(128, 1.00)
    for algo, tile_a, eff in ((7, 160, 0.96), (4, 192, 1.02)):
        c = plain(tile_a, eff)
        if c < best_cost:
            best_algo, best_cost = algo, c
    best_splits = 1
    units = w_tiles * _tiles(m, 128)
    # One k-tile-wave: the whole grid advancing one k-tile.  The reduce streams
    # (splits + 0.5) * m * n fp32 words; the ratio to one k-tile-wave's MMA work is
    # a constant of the machine (about 1/400 per byte-per-FLOP here).
    reduce_per_split = 4.0 * m * n / (clusters * 2.0 * 128 * 128 * 128 * 2.0) * 400.0
    for s in range(2, 9):
        if ktiles % s != 0 or ktiles // s < 8:
            continue
        c = _tiles(units * s, clusters) * (ktiles // s) + reduce_per_split * (s + 0.5)
        if c < best_cost:
            best_algo, best_cost, best_splits = _SM100_SPLITK, c, s
    if best_algo == _SM100_SPLITK:
        return 1, best_splits
    return best_algo, 1


def sm100_pick_algo(m: int, n: int, k: int, num_sms: int) -> int:
    """W8A16Sm100PickAlgo: the launcher's variant for a problem."""
    if n <= 128 and m >= 16384:
        return 2
    if _sm100_use_stream_k(m, n, k, num_sms):
        return _SM100_STREAMK
    if _sm100_split_k(m, n, k, num_sms) > 1:
        return _SM100_SPLITK
    clusters = num_sms // 2 if num_sms > 1 else 1
    w_tiles = _tiles(n, 256)

    def cost(tile_a: int, eff: float) -> float:
        units = w_tiles * _tiles(m, tile_a)
        return _tiles(units, clusters) * tile_a / eff

    c128 = cost(128, 1.00)
    c160 = cost(160, 0.96)
    c192 = cost(192, 1.02)
    if c192 <= c128 and c192 <= c160:
        return 4
    if c160 < c128:
        return 7
    return 1


def sm100_pick_swizzle(m: int, n: int, k: int) -> int:
    """W8A16Sm100PickSwizzle: L2 tile group in clusters (1 = plain raster)."""
    weight_bytes = n * k
    if weight_bytes < (64 << 20) or m < 4096:
        return 1
    s = ((24 << 20) + 128 * k) // (256 * k)
    return max(2, min(16, s))


def sm100_raster_along_m(
    m: int, n: int, tile: Tuple[int, int, int], two_sm: bool
) -> bool:
    """CUTLASS's heuristic raster: along the kernel's M (weight rows) when there
    are more activation tiles than weight tiles, else along N."""
    cta_tile_m = tile[0] // (2 if two_sm else 1)
    tiles_m = _tiles(n, cta_tile_m)
    tiles_n = _tiles(m, tile[1])
    return tiles_n > tiles_m


def _compile_pointers(specs: Tuple):
    """CuTe pointer objects for ``cute.compile`` from (dtype, address, alignment)
    triples.  Built only when a kernel is compiled: the compiled function takes the
    raw addresses."""
    from ..cute_dsl.utils import make_ptr

    return tuple(
        make_ptr(dtype, addr, cute_gmem(), assumed_align=align)
        for dtype, addr, align in specs
    )


def _raw_stream(device: torch.device):
    """The current Torch stream of ``device`` as a driver stream handle (the public
    torch.cuda.current_stream() path costs ~20 us per call, most of a small GEMM)."""
    import cuda.bindings.driver as cuda

    index = device.index if device.index is not None else torch.cuda.current_device()
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(index))


_KERNEL_CACHE: Dict[Tuple, object] = {}


def _get_sm100_kernel(
    algo: int,
    act_dtype: torch.dtype,
    has_bias: bool,
    swizzle_size: int,
    raster_along_m: bool,
    max_active_clusters: int,
    pointer_specs: Tuple,
    n: int,
    m: int,
    k: int,
    stream,
    enable_pdl: bool = False,
    debug: bool = False,
    partial_output: bool = False,
    splits: int = 1,
):
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_gemm_sm100 import Sm100W8A16Fp8GemmKernel

    mma_tiler, two_sm, cluster, a_in_smem = _SM100_TILES[algo]
    # Bytes of weight each transform thread converts per step: a whole stage when K
    # is one k-tile (no pipelining to overlap), 32 otherwise (as the W4A16 kernel).
    transform_fragment_size = 128 if k == mma_tiler[2] else 32
    key = (
        algo,
        act_dtype,
        has_bias,
        swizzle_size,
        raster_along_m,
        transform_fragment_size,
        max_active_clusters,
        enable_pdl,
        debug,
        partial_output,
        splits,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    kernel = Sm100W8A16Fp8GemmKernel(
        mma_tiler_mnk=mma_tiler,
        use_2cta_instrs=two_sm,
        cluster_shape_mn=cluster,
        a_in_smem=a_in_smem,
        has_bias=has_bias,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        transform_fragment_size=transform_fragment_size,
        enable_pdl=enable_pdl,
        debug=debug,
        partial_output=partial_output,
    )
    compiled = cute.compile(
        kernel.wrapper,
        *_compile_pointers(pointer_specs),
        n,
        m,
        k,
        splits=splits,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    if debug:
        print(
            f"w8a16 dsl kernel algo {algo}: stages load2trans "
            f"{kernel.num_load2trans_stage} trans2mma {kernel.num_trans2mma_stage} acc "
            f"{kernel.num_acc_stage} c {kernel.num_c_stage} b {kernel.num_b_stage}, tmem cols "
            f"{kernel.num_tmem_alloc_cols}",
            flush=True,
        )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _get_reduce_kernel(
    act_dtype: torch.dtype, has_bias: bool, splits: int, pointer_specs: Tuple, stream
):
    """The split-K reduce (sum of slices, scale, bias, cast), compiled once per
    activation dtype, bias presence and slice count."""
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_splitk_reduce import W8A16SplitKReduce

    key = ("reduce", act_dtype, has_bias, splits)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    compiled = cute.compile(
        W8A16SplitKReduce(has_bias, splits).wrapper,
        *_compile_pointers(pointer_specs),
        1,
        8,
        1,
        stream=stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


_PLAN_CACHE: Dict[Tuple, Tuple] = {}


def _sm100_plan(algo: int, m: int, n: int, k: int, num_sms: int) -> Tuple:
    """(algo, splits, swizzle, raster_along_m, max active clusters) of a Blackwell
    call.  A pure function of the shape, cached: the cost model is most of the host
    cost of a call otherwise."""
    key = ("sm100", algo, m, n, k, num_sms)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    from ..cute_dsl.utils import get_max_active_clusters

    if algo == 0:
        algo = sm100_pick_algo(m, n, k, num_sms)
    splits = 1
    if algo == _SM100_STREAMK:
        # No stream-K here: the best of what this kernel has for a grid of two or
        # three partly filled waves (a plain tile or split-K).
        algo, splits = _sm100_streamk_substitute(m, n, k, num_sms)
    elif algo == _SM100_SPLITK:
        splits = sm100_splits_for(m, n, k, num_sms)
        if splits < 2:
            raise ValueError(
                f"w8a16_fp8_gemm split-K cannot slice K={k}: K must be a multiple of "
                "128 whose k-tile count has a divisor in 2..8"
            )
        algo = 1
    mma_tiler, two_sm, cluster, _ = _SM100_TILES[algo]
    plan = (
        algo,
        splits,
        sm100_pick_swizzle(m, n, k),
        sm100_raster_along_m(m, n, mma_tiler, two_sm),
        get_max_active_clusters(cluster[0] * cluster[1]),
    )
    _PLAN_CACHE[key] = plan
    return plan


def w8a16_fp8_gemm_cute_dsl(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    algo: int = 0,
    debug_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blackwell W8A16 GEMM in CuTe-DSL.  ``algo`` as the C++ op's codes (0 = the
    dispatcher's choice); split-K and stream-K codes are not implemented yet and
    fall back to the plain 256x128 tile.  ``debug_buffer``: an Int32 tensor of at
    least 64 * 4096 entries (pinned host memory to watch a hang) that the kernel's
    debug build writes its per-CTA progress into; None = the production build."""
    import cutlass

    if act.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"act must be bf16 or fp16, got {act.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    major = get_compute_capability(act.device)[0]
    if major not in (9, 10):
        raise RuntimeError(
            "the CuTe-DSL W8A16 kernel runs on Hopper and Blackwell only"
        )
    if act.dim() != 2 or weight.dim() != 2 or weight.shape[1] != act.shape[1]:
        raise ValueError(
            f"act [M, K] and weight [N, K] must share K, got {tuple(act.shape)} and "
            f"{tuple(weight.shape)}"
        )
    m, k = act.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=act.dtype, device=act.device)
    if out.shape != (m, n) or out.dtype != act.dtype:
        raise ValueError(f"out must be [{m}, {n}] {act.dtype}")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() != 1:
        raise ValueError("weight_scale must be one fp32 value")
    if bias is not None and (bias.dtype != act.dtype or bias.numel() != n):
        raise ValueError(f"bias must be [{n}] {act.dtype}")
    for name, t in (("act", act), ("weight", weight), ("out", out), ("bias", bias)):
        if t is not None and not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    # TMA needs 16-byte rows: 16 fp8 weights, 8 16-bit activations / outputs.
    if k % 16 != 0 or n % 8 != 0:
        raise ValueError(f"K must be a multiple of 16 and N of 8, got K={k}, N={n}")
    for name, t in (("act", act), ("weight", weight), ("out", out)):
        if t.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    if major == 9:
        if debug_buffer is not None:
            raise ValueError("debug_buffer is a Blackwell-kernel feature")
        return _w8a16_fp8_gemm_sm90(act, weight, weight_scale, bias, out, algo)
    plan = _sm100_plan(algo, m, n, k, get_device_sm_count(act.device))
    algo, splits, swizzle_size, raster_along_m, max_active_clusters = plan
    # Split-K: the 256x128 tile over `splits` K-slices writing fp32 partials, then
    # the reduce.  The GEMM's `out` is then the partials buffer and the kernel does
    # no scale / bias.
    partials = None
    if splits > 1:
        partials = torch.empty((splits, m, n), dtype=torch.float32, device=act.device)
    k_slice = k // splits
    has_bias = bias is not None
    act_ctype = cutlass.BFloat16 if act.dtype == torch.bfloat16 else cutlass.Float16
    bias_tensor = bias if has_bias else weight_scale  # any valid pointer when unused
    bias_ctype = act_ctype if has_bias else cutlass.Float32
    gemm_out = partials if partials is not None else out
    gemm_out_ctype = cutlass.Float32 if partials is not None else act_ctype
    debug_tensor = debug_buffer if debug_buffer is not None else weight_scale
    pointer_specs = (
        (cutlass.Float8E4M3FN, weight.data_ptr(), 16),
        (act_ctype, act.data_ptr(), 16),
        (cutlass.Float32, weight_scale.data_ptr(), 4),
        (bias_ctype, bias_tensor.data_ptr(), 4),
        (gemm_out_ctype, gemm_out.data_ptr(), 16),
        (cutlass.Int32, debug_tensor.data_ptr(), 4),
    )
    stream = _raw_stream(act.device)
    compiled = _get_sm100_kernel(
        algo,
        act.dtype,
        has_bias,
        swizzle_size,
        raster_along_m,
        max_active_clusters,
        pointer_specs,
        n,
        m,
        k_slice,
        stream,
        debug=debug_buffer is not None,
        partial_output=partials is not None,
        splits=splits,
    )
    compiled(
        weight.data_ptr(),
        act.data_ptr(),
        weight_scale.data_ptr(),
        bias_tensor.data_ptr(),
        gemm_out.data_ptr(),
        debug_tensor.data_ptr(),
        n,
        m,
        k_slice,
        stream,
    )
    if partials is not None:
        reduce_specs = (
            (cutlass.Float32, partials.data_ptr(), 16),
            (cutlass.Float32, weight_scale.data_ptr(), 4),
            (bias_ctype, bias_tensor.data_ptr(), 16),
            (act_ctype, out.data_ptr(), 16),
        )
        reduce = _get_reduce_kernel(act.dtype, has_bias, splits, reduce_specs, stream)
        reduce(
            partials.data_ptr(),
            weight_scale.data_ptr(),
            bias_tensor.data_ptr(),
            out.data_ptr(),
            m,
            n,
            m * n // 8,
            stream,
        )
    return out


# ======================================================================= Hopper
# The C++ Hopper dispatcher (``w8a16_sm90/dispatch.hpp``, ``tile_order.hpp``) ported.
# Tile configurations by algo code (W8A16Sm90Algo): (weight rows, activation rows).
_SM90_TILES: Dict[int, Tuple[int, int]] = {
    2: (128, 256),
    3: (128, 192),
    4: (128, 160),
    5: (128, 128),
    6: (64, 256),
    9: (64, 128),
}
# Stream-K codes run the plain tile (CUTLASS's stream-K scheduler is not ported);
# split-K codes run the tile over K-slices.
_SM90_STREAMK_TILE: Dict[int, int] = {7: 2, 8: 3}
_SM90_SPLITK_TILE: Dict[int, int] = {10: 5, 11: 2, 12: 9}
_SM90_TILE_K = 64
_SM90_MAX_SPLITS = 8
_SM90_MIN_KTILES_PER_SPLIT = 8
_SM90_MAX_PARTIALS_BYTES = 64 << 20

# The cost model's tile menu: (algo, weight rows, activation rows, ns per unit of K of one
# tile alone on an SM, ping-pong).  Candidate order breaks ties.
_SM90_TILE_COSTS: Tuple[Tuple[int, int, int, float, bool], ...] = (
    (2, 128, 256, 11.37, False),
    (3, 128, 192, 9.00, False),
    (5, 128, 128, 7.45, False),
    (6, 64, 256, 6.15, True),
    (9, 64, 128, 4.50, True),
)


def _sm90_tile_of(algo: int) -> Tuple[int, int, int, float, bool]:
    base = _SM90_STREAMK_TILE.get(algo, _SM90_SPLITK_TILE.get(algo, algo))
    for t in _SM90_TILE_COSTS:
        if t[0] == base:
            return t
    return _SM90_TILE_COSTS[0]


def _sm90_is_legal_split(k: int, s: int) -> bool:
    if s < 2 or s > _SM90_MAX_SPLITS or k % _SM90_TILE_K != 0:
        return False
    ktiles = k // _SM90_TILE_K
    return ktiles % s == 0 and ktiles // s >= _SM90_MIN_KTILES_PER_SPLIT


def _sm90_tile_us(t, k_len: float) -> float:
    _, _, tile_a, ns_per_k, pingpong = t
    epi = 0.0 if pingpong else 2.0 * tile_a / 256
    return ns_per_k * 1e-3 * k_len + epi


def _sm90_tiles_of(t, m: int, n: int) -> float:
    return float(_tiles(n, t[1])) * float(_tiles(m, t[2]))


def _sm90_split_us(t, m: int, n: int, k: int, num_sms: int, s: int) -> float:
    if 4.0 * m * n * s > _SM90_MAX_PARTIALS_BYTES:
        return 1e30
    return (
        math.ceil(_sm90_tiles_of(t, m, n) * s / max(num_sms, 1))
        * _sm90_tile_us(t, k / s)
        + 2.0
        + 2e-6 * m * n * s
    )


def _sm90_pick_splits(t, m: int, n: int, k: int, num_sms: int) -> int:
    best, best_us = 1, 1e30
    for s in range(2, _SM90_MAX_SPLITS + 1):
        if not _sm90_is_legal_split(k, s):
            continue
        us = _sm90_split_us(t, m, n, k, num_sms, s)
        if us < best_us:
            best_us, best = us, s
    return best


def sm90_splits_for(algo: int, m: int, n: int, k: int, num_sms: int) -> int:
    """W8A16Sm90SplitsFor: the model's split, or the smallest legal one when the
    variant is forced; 1 = K cannot be sliced."""
    t = _sm90_tile_of(algo)
    s = _sm90_pick_splits(t, m, n, k, num_sms)
    if s >= 2:
        return s
    for c in range(2, _SM90_MAX_SPLITS + 1):
        if _sm90_is_legal_split(k, c):
            return c
    return 1


def _sm90_plain_us(t, m: int, n: int, k: int, num_sms: int) -> float:
    return math.ceil(_sm90_tiles_of(t, m, n) / max(num_sms, 1)) * _sm90_tile_us(t, k)


def _sm90_streamk_us(t, m: int, n: int, k: int, num_sms: int) -> float:
    sms = float(max(num_sms, 1))
    ktiles = k // _SM90_TILE_K
    tiles = _sm90_tiles_of(t, m, n)
    tile_us = _sm90_tile_us(t, k)
    full_waves, total_waves = math.floor(tiles / sms), math.ceil(tiles / sms)
    tail_tiles = tiles - full_waves * sms
    if (
        full_waves == total_waves
        or ktiles <= 8
        or (full_waves >= 1 and 2 * tail_tiles >= sms)
    ):
        return total_waves * tile_us
    dp_waves = full_waves - 1 if full_waves > 1 else 0
    sk_tiles = tiles - dp_waves * sms
    units = max(1.0, min(sms, math.floor(sk_tiles * ktiles / 8.0)))
    return (
        dp_waves * tile_us
        + sk_tiles * ktiles / units * (tile_us / ktiles) * 1.1
        + 8.0
        + 2.0 * units / sk_tiles
    )


def _sm90_pick_by_cost(m: int, n: int, k: int, num_sms: int) -> int:
    best, best_us = 2, 1e30

    def consider(a: int, us: float) -> None:
        nonlocal best, best_us
        if us < best_us:
            best_us, best = us, a

    for t in _SM90_TILE_COSTS:
        consider(t[0], _sm90_plain_us(t, m, n, k, num_sms))
    for a in (7, 8):
        consider(a, _sm90_streamk_us(_sm90_tile_of(a), m, n, k, num_sms))
    for a in (10, 11, 12):
        t = _sm90_tile_of(a)
        for s in range(2, _SM90_MAX_SPLITS + 1):
            if _sm90_is_legal_split(k, s):
                consider(a, _sm90_split_us(t, m, n, k, num_sms, s))
    return best


def sm90_pick_algo(m: int, n: int, k: int, num_sms: int) -> int:
    """W8A16Sm90PickAlgo: the cost model up to M = 8192, wave-count rules above."""
    units = _tiles(n, 128) * _tiles(m, 256)
    waves = units / max(num_sms, 1)
    frac = waves - math.floor(waves)
    if m <= 8192:
        return _sm90_pick_by_cost(m, n, k, num_sms)
    if n <= 512:
        if waves < 0.75:
            return 5
        if m >= 16384:
            return 3
        return 2
    if k <= 1024 and m >= 16384:
        return 6
    if waves <= 1.5:
        if k >= 4096 and waves >= 0.7:
            return 7
        if n <= 2048 or k <= 3200:
            return 6
        return 2
    if frac >= 0.8:
        return 2
    if frac < 0.15 and k >= 2048 and waves <= 16.0:
        return 7
    if waves <= 3.0:
        return 6
    return 2


def sm90_pick_group(m: int, n: int, k: int, tile_w: int) -> int:
    """W8A16Sm90PickGroup: activation tiles per L2 group (1 = ungrouped)."""
    if n * k < 40 * 1000 * 1000 or m < 16384:
        return 1
    return 4 if _tiles(n, tile_w) >= 64 else 2


def _get_sm90_kernel(
    tile_algo: int,
    act_dtype: torch.dtype,
    has_bias: bool,
    group: int,
    num_persistent: int,
    pointer_specs: Tuple,
    n: int,
    m: int,
    k: int,
    tail_blocks: int,
    stream,
    has_tail: bool = False,
):
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_gemm_sm90 import Sm90W8A16Fp8GemmKernel

    key = ("sm90", tile_algo, act_dtype, has_bias, group, num_persistent, has_tail)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    kernel = Sm90W8A16Fp8GemmKernel(
        _SM90_TILES[tile_algo],
        has_bias=has_bias,
        group_size=group,
        has_tail=has_tail,
    )
    compiled = cute.compile(
        kernel.wrapper,
        *_compile_pointers(pointer_specs),
        n,
        m,
        k,
        tail_blocks,
        1,  # body_items: dynamic
        1,  # tail_splits: dynamic
        num_persistent=num_persistent,
        stream=stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    if os.environ.get("W8A16_DSL_VERBOSE") == "1":
        print(
            f"w8a16 dsl sm90 tile {_SM90_TILES[tile_algo]}: ab stages {kernel.ab_stage}, epi "
            f"stages {kernel.epi_stage}, group {group}, tail {has_tail}",
            flush=True,
        )
    _KERNEL_CACHE[key] = compiled
    return compiled


_SM90_TAIL_WORKSPACE_CAP = 64 << 20
_SM90_TAIL_MAX_SPLITS = 16
_TAIL_COUNTERS: Dict[torch.device, torch.Tensor] = {}


def _sm90_tail_counters(device: torch.device, count: int) -> torch.Tensor:
    """Per-device 64-bit arrival counters for the tail tiles.  They only grow (every
    finished tile advances its counter by one ``TAIL_COUNTER_STRIDE``, so a counter is
    always at a stride boundary between calls); grown when a call needs more."""
    buf = _TAIL_COUNTERS.get(device)
    if buf is None or buf.numel() < count:
        buf = torch.zeros(max(count, 1024), dtype=torch.int64, device=device)
        _TAIL_COUNTERS[device] = buf
    return buf


def sm90_tail_plan(
    algo: int, m: int, n: int, k: int, num_sms: int
) -> Tuple[int, int, int]:
    """The work decomposition of a Hopper variant: (tile algo, body tiles, tail
    splits).  Plain tiles: every tile whole.  Split-K codes: no body, every tile cut
    into W8A16Sm90SplitsFor slices.  Stream-K codes: the whole waves plain and the
    last partial wave cut into the slice count that best fills the SMs (fewest rounds
    of the shortest slices under the workspace cap), or plain when CUTLASS's own
    stream-K would run data-parallel (no tail, or a tail of at least half a wave)."""
    tile_algo = _SM90_STREAMK_TILE.get(algo, _SM90_SPLITK_TILE.get(algo, algo))
    tile_w, tile_a = _SM90_TILES[tile_algo]
    tiles = _tiles(n, tile_w) * _tiles(m, tile_a)
    k_tiles = k // _SM90_TILE_K
    if algo in _SM90_SPLITK_TILE:
        return tile_algo, 0, sm90_splits_for(algo, m, n, k, num_sms)
    if algo not in _SM90_STREAMK_TILE:
        return tile_algo, tiles, 1
    full = tiles // num_sms
    tail = tiles - full * num_sms
    if tail == 0 or 2 * tail >= num_sms or k_tiles < 4:
        return tile_algo, tiles, 1
    # The most slices that still run in one round (the slices of a tile reduce it
    # among themselves and must be resident together), at least two k-tiles each,
    # under the workspace cap.
    block_bytes = tile_w * tile_a * 4
    best_s = 1
    for s in range(2, _SM90_TAIL_MAX_SPLITS + 1):
        if tail * s > num_sms or k_tiles // s < 2:
            break
        if tail * s * block_bytes > _SM90_TAIL_WORKSPACE_CAP:
            break
        best_s = s
    if best_s == 1:
        return tile_algo, tiles, 1
    return tile_algo, full * num_sms, best_s


def _sm90_plan(algo: int, m: int, n: int, k: int, num_sms: int) -> Tuple:
    """(tile algo, body items, tail splits, tail tiles, L2 group) of a Hopper call:
    the variant choice and the tail plan, pure functions of the shape, cached."""
    key = ("sm90", algo, m, n, k, num_sms)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if algo == 0:
        algo = sm90_pick_algo(m, n, k, num_sms)
    if algo == 1 or algo > 12:
        raise ValueError(f"w8a16_fp8_gemm: unknown Hopper algo code {algo}")
    if algo in _SM90_SPLITK_TILE and k % _SM90_TILE_K != 0:
        raise ValueError("w8a16_fp8_gemm split-K: K must be a multiple of 64")
    tile_algo, body_items, tail_splits = sm90_tail_plan(algo, m, n, k, num_sms)
    if algo in _SM90_SPLITK_TILE and tail_splits < 2:
        raise ValueError(
            f"w8a16_fp8_gemm split-K cannot slice K={k}: K must be a multiple of 64 "
            "with at least 8 k-tiles per slice"
        )
    tile_w, tile_a = _SM90_TILES[tile_algo]
    tiles = _tiles(n, tile_w) * _tiles(m, tile_a)
    tail_tiles = tiles - body_items
    if tail_splits < 2 or tail_tiles * tail_splits > num_sms:
        body_items, tail_splits, tail_tiles = tiles, 1, 0
    plan = (
        tile_algo,
        body_items,
        tail_splits,
        tail_tiles,
        sm90_pick_group(m, n, k, tile_w),
    )
    _PLAN_CACHE[key] = plan
    return plan


def _w8a16_fp8_gemm_sm90(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    algo: int,
) -> torch.Tensor:
    """The Hopper path of ``w8a16_fp8_gemm_cute_dsl`` (inputs already validated)."""
    import cutlass

    m, k = act.shape
    n = weight.shape[0]
    num_sms = get_device_sm_count(act.device)
    tile_algo, body_items, tail_splits, tail_tiles, group = _sm90_plan(
        algo, m, n, k, num_sms
    )
    tile_w, tile_a = _SM90_TILES[tile_algo]
    has_tail = tail_tiles > 0
    tail_blocks = tail_tiles * tail_splits
    workspace = (
        torch.empty(
            (max(tail_blocks, 1), tile_a, tile_w),
            dtype=torch.float32,
            device=act.device,
        )
        if has_tail
        else weight_scale
    )
    counters = _sm90_tail_counters(act.device, tail_tiles) if has_tail else weight_scale
    has_bias = bias is not None
    act_ctype = cutlass.BFloat16 if act.dtype == torch.bfloat16 else cutlass.Float16
    bias_tensor = bias if has_bias else weight_scale  # any valid pointer when unused
    bias_ctype = act_ctype if has_bias else cutlass.Float32
    pointer_specs = (
        (cutlass.Float8E4M3FN, weight.data_ptr(), 16),
        (act_ctype, act.data_ptr(), 16),
        (cutlass.Float32, weight_scale.data_ptr(), 4),
        (bias_ctype, bias_tensor.data_ptr(), 4),
        (act_ctype, out.data_ptr(), 16),
        (cutlass.Float32, workspace.data_ptr(), 16),
        (cutlass.Int64, counters.data_ptr(), 8),
    )
    stream = _raw_stream(act.device)
    compiled = _get_sm90_kernel(
        tile_algo,
        act.dtype,
        has_bias,
        group,
        num_sms,
        pointer_specs,
        n,
        m,
        k,
        max(tail_blocks, 1),
        stream,
        has_tail=has_tail,
    )
    compiled(
        weight.data_ptr(),
        act.data_ptr(),
        weight_scale.data_ptr(),
        bias_tensor.data_ptr(),
        out.data_ptr(),
        workspace.data_ptr(),
        counters.data_ptr(),
        n,
        m,
        k,
        max(tail_blocks, 1),
        body_items,
        tail_splits,
        stream,
    )
    return out


def cute_gmem():
    import cutlass.cute as cute

    return cute.AddressSpace.gmem
