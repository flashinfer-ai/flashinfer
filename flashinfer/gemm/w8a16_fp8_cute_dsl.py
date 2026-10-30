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

"""W8A16 FP8 GEMM, CuTe-DSL kernels (Blackwell first).

Same contract as the C++ op in ``w8a16_fp8.py``:
``out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])``.  While the
two implementations coexist this module is reached through
``w8a16_fp8_gemm_cute_dsl``; the variant selection (``algo`` codes, tile walk) is
the C++ dispatcher's (``include/flashinfer/gemm/w8a16_sm100/dispatch.hpp``,
``tile_order.hpp``) ported line by line so the two run the same kernel for a
problem.
"""

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


_KERNEL_CACHE: Dict[Tuple, object] = {}


def _get_sm100_kernel(
    algo: int,
    act_dtype: torch.dtype,
    has_bias: bool,
    swizzle_size: int,
    raster_along_m: bool,
    max_active_clusters: int,
    pointers: Tuple,
    n: int,
    m: int,
    k: int,
    stream,
    enable_pdl: bool = False,
    debug: bool = False,
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
    )
    compiled = cute.compile(
        kernel.wrapper,
        *pointers,
        n,
        m,
        k,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


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
    least 16 * 4096 entries (pinned host memory to watch a hang) that the kernel's
    debug build writes its per-CTA progress into; None = the production build."""
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl.utils import current_cuda_stream, get_max_active_clusters, make_ptr

    if act.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"act must be bf16 or fp16, got {act.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if get_compute_capability(act.device)[0] != 10:
        raise RuntimeError("the CuTe-DSL W8A16 kernel is Blackwell-only for now")
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
    num_sms = get_device_sm_count(act.device)
    if algo == 0:
        algo = sm100_pick_algo(m, n, k, num_sms)
    if algo in (_SM100_SPLITK, _SM100_STREAMK):
        algo = 1
    mma_tiler, two_sm, cluster, _ = _SM100_TILES[algo]
    swizzle_size = sm100_pick_swizzle(m, n, k)
    raster_along_m = sm100_raster_along_m(m, n, mma_tiler, two_sm)
    has_bias = bias is not None
    act_ctype = cutlass.BFloat16 if act.dtype == torch.bfloat16 else cutlass.Float16
    bias_tensor = bias if has_bias else weight_scale  # any valid pointer when unused
    bias_ctype = act_ctype if has_bias else cutlass.Float32
    pointers = (
        make_ptr(
            cutlass.Float8E4M3FN,
            weight.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(act_ctype, act.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.Float32,
            weight_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            bias_ctype, bias_tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        make_ptr(act_ctype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.Int32,
            (debug_buffer if debug_buffer is not None else weight_scale).data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
    )
    stream = current_cuda_stream()
    max_active_clusters = get_max_active_clusters(cluster[0] * cluster[1])
    compiled = _get_sm100_kernel(
        algo,
        act.dtype,
        has_bias,
        swizzle_size,
        raster_along_m,
        max_active_clusters,
        pointers,
        n,
        m,
        k,
        stream,
        debug=debug_buffer is not None,
    )
    compiled(
        weight.data_ptr(),
        act.data_ptr(),
        weight_scale.data_ptr(),
        bias_tensor.data_ptr(),
        out.data_ptr(),
        (debug_buffer if debug_buffer is not None else weight_scale).data_ptr(),
        n,
        m,
        k,
        stream,
    )
    return out
