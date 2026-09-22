# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Architecture-specific CuTe-DSL backend for the BF16 x NVFP4 GEMM."""

import functools
from typing import Dict, List, Optional, Tuple, cast

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..utils import _get_cache_buf, get_compute_capability, get_device_sm_count
from .gemm_base import _check_cute_dsl_availability
from .gemm_bf16_fp4 import _unswizzle_sf_128x4

_BF16_FP4_ALPHA_ONE_CACHE: dict = {}


def _prepare_bf16_fp4_alpha(
    alpha: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Normalize ``alpha`` to a ``(1,) float32`` tensor for the kernel."""
    if alpha is None:
        cached = _BF16_FP4_ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _BF16_FP4_ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha.dim() == 0:
        return alpha.to(device=device, dtype=torch.float32).unsqueeze(0)
    return alpha.to(device=device, dtype=torch.float32).reshape(1)


def _select_bf16_fp4_tile_shape(
    m: int, n: int, k: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pick a CTA tile shape AND MMA atom_layout for the cute-DSL bf16 x fp4 kernel.

    Returns ``(tile_shape_mnk, atom_layout)``.

    Tile shape selection:
      tile_M choice
        * M <= 16 (and tile_K=128 path): use tile_M=16 with atom_layout
          (1,2,1).  Halves wasted M-rows vs tile_M=32, and a 1-M-warp
          layout removes the duplicate dequant that (2,2,1) suffers from.
        * 16 < M <= 32: use tile_M=32 with atom_layout (2,2,1).  Smaller
          MMA + epilogue waste than tile_M=64.
        * M > 32: use tile_M=64 with atom_layout (2,2,1) -- standard tile,
          more rows to amortize across.

      tile_K choice
        * K % 128 == 0: tile_K=128 (halves K-tile count and barrier
          overhead).
        * Otherwise: tile_K=64.

    Why atom_layout differs:
      * (2,2,1) (default for tile_M >= 32): 4 MMA warps as 2 M x 2 N --
        well-tested cute layout, but the 2 M-warps redundantly dequant
        the same B values into their own register files (~50% waste in
        dequant compute).
      * (1,2,1) (used for tile_M=16): 2 MMA warps as 1 M x 2 N -- no
        M-warp duplication.  Permutation_m = 16, so tile_M must be 16.
    """
    tile_k = 128 if k % 128 == 0 else 64
    if m <= 16 and tile_k == 128:
        return ((16, 64, 128), (1, 2, 1))
    if m <= 32:
        return ((32, 64, tile_k), (2, 2, 1))
    return ((64, 64, tile_k), (2, 2, 1))


def _select_bf16_fp4_k_splits(
    m: int, n: int, k: int, tile_shape_mnk: Tuple[int, int, int], sm_count: int
) -> int:
    """Static split-K pick for the no-autotune fallback: only clear-win
    underfill grids get a split, closer calls are left to the autotuner."""
    tile_m, tile_n, tile_k = tile_shape_mnk
    if tile_m != (16 if tile_k == 128 else 32):
        # Split only the tile shapes that the tactic space pairs with splits.
        return 1
    tiles = -(-m // tile_m) * -(-n // tile_n)
    if tiles * 3 > sm_count:
        return 1
    k_tiles = k // tile_k
    for splits in (8, 4, 2):
        if splits <= k_tiles and tiles * splits <= sm_count:
            return splits
    return 1


# Stream-GEMV residency window (warps/SM): below the floor DRAM latency is
# uncovered, far above the ceiling splits just multiply partial traffic.
_GEMV_MIN_WARPS_PER_SM = 12
_GEMV_MAX_WARPS_PER_SM = 96


def _bf16_fp4_gemv_knee_split(n: int, k: int, sm_count: int) -> int:
    """Smallest K-split reaching ~20 warps/SM, where throughput saturates
    and deeper splits only add partial traffic; capped under one full wave
    (6 CTAs/SM = 1536-thread limit / 256) and at >= 4 K-tiles per split."""
    ctas = -(-(n // 64) // 8)
    cap = max(1, min(int(0.95 * sm_count * 6 / ctas), (k // 16) // 4))
    return min(-(-20 * sm_count // (n // 64)), cap)


def _select_bf16_fp4_gemv_split(
    n: int, k: int, cc_major: int, sm_count: int
) -> Optional[int]:
    """m=1 no-autotune fallback pick: the stream GEMV where it applies,
    ``None`` for the MMA heuristic.  Serving stacks do not tune every
    shape (vLLM never tunes the logits GEMM), so untuned m=1 matters."""
    if cc_major != 12 or n % 64 != 0 or k // 16 < 4:
        return None
    split = _bf16_fp4_gemv_knee_split(n, k, sm_count)
    warps_per_sm = (n // 64) * split / sm_count
    if not _GEMV_MIN_WARPS_PER_SM <= warps_per_sm <= _GEMV_MAX_WARPS_PER_SM:
        return None
    return split


_CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE: dict = {}


def _get_cute_dsl_bf16_fp4_gemm(
    tile_shape_mnk: Tuple[int, int, int],
    a_dtype: torch.dtype,
    c_dtype: torch.dtype,
    atom_layout: Tuple[int, int, int] = (2, 2, 1),
    pipeline_depth: int = 1,
    use_fp16_mma: int = 1,
    enable_pdl: bool = True,
    tile_swizzle: int = 1,
    k_splits: int = 1,
    occupancy: int = 1,
):
    # Normalize to a tuple (callers may pass a list) so the cache key is hashable.
    atom_layout = cast(Tuple[int, int, int], tuple(atom_layout))
    pipeline_depth = int(pipeline_depth)
    use_fp16_mma = int(use_fp16_mma)
    enable_pdl = bool(enable_pdl)
    tile_swizzle = int(tile_swizzle)
    k_splits = int(k_splits)
    occupancy = int(occupancy)
    cache_key = (
        tile_shape_mnk,
        a_dtype,
        c_dtype,
        atom_layout,
        pipeline_depth,
        use_fp16_mma,
        enable_pdl,
        tile_swizzle,
        k_splits,
        occupancy,
    )
    cached = _CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _check_cute_dsl_availability()

    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernels.cute_dsl.dense_gemm_bf16_fp4_sm12x import (
        Sm12xDenseGemmBf16Fp4Kernel,
    )

    from ..cute_dsl.utils import torch_to_cutlass_dtype

    a_cutlass_dtype = torch_to_cutlass_dtype(a_dtype)
    c_cutlass_dtype = torch_to_cutlass_dtype(c_dtype)

    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()
    sym_k_tiles = cute.sym_int()
    sym_n_packed = cute.sym_int()

    a_fake = cute.runtime.make_fake_compact_tensor(
        a_cutlass_dtype, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
    )
    b_packed_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_k_tiles, sym_n_packed),
        stride_order=(1, 0),
        assumed_align=16,
    )
    b_sf_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_k_tiles, sym_n), stride_order=(1, 0), assumed_align=16
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_cutlass_dtype, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16
    )
    sym_partial = cute.sym_int()
    partial_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (sym_partial,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    gemm = Sm12xDenseGemmBf16Fp4Kernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mnk=tile_shape_mnk,
        atom_layout=atom_layout,
        # At occupancy 3 the SMEM budget per CTA only fits 2 ab stages next
        # to the default 4-stage epilogue; a 2-stage epilogue keeps 3.
        epi_stage=2 if occupancy >= 3 else 4,
        pipeline_depth=pipeline_depth,
        use_fp16_mma=use_fp16_mma,
        enable_pdl=enable_pdl,
        tile_swizzle=tile_swizzle,
        k_splits=k_splits,
        occupancy=occupancy,
    )
    # The persistent grid launches occupancy CTAs per SM; the kernel's stage
    # budget (_compute_stages) shrinks SMEM per CTA so they co-reside.
    max_active_clusters = get_max_active_clusters(1) * occupancy

    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_packed_fake,
        b_sf_fake,
        c_fake,
        partial_fake,
        alpha_fake,
        1,  # l (batch)
        max_active_clusters,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE[cache_key] = compiled
    return compiled


def _get_cute_dsl_bf16_fp4_gemv(
    splits: int,
    a_dtype: torch.dtype,
    c_dtype: torch.dtype,
    enable_pdl: bool = True,
):
    """Compile-cache the m=1 streaming GEMV (same call signature as the
    compiled MMA kernel, so the runner dispatches either transparently)."""
    splits = int(splits)
    enable_pdl = bool(enable_pdl)
    cache_key = ("gemv", splits, a_dtype, c_dtype, enable_pdl)
    cached = _CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _check_cute_dsl_availability()

    import cutlass
    import cutlass.cute as cute

    from .kernels.cute_dsl.gemv_bf16_fp4_sm12x import GemvBf16Fp4Sm12x

    from ..cute_dsl.utils import torch_to_cutlass_dtype

    sym = [cute.sym_int() for _ in range(6)]
    a_fake = cute.runtime.make_fake_compact_tensor(
        torch_to_cutlass_dtype(a_dtype),
        (sym[0], sym[1]),
        stride_order=(1, 0),
        assumed_align=16,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym[2], sym[3]), stride_order=(1, 0), assumed_align=16
    )
    sf_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym[2], sym[4]), stride_order=(1, 0), assumed_align=16
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        torch_to_cutlass_dtype(c_dtype),
        (sym[0], sym[4]),
        stride_order=(1, 0),
        assumed_align=16,
    )
    partial_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (sym[5],), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile(
        GemvBf16Fp4Sm12x(splits=splits, enable_pdl=enable_pdl),
        a_fake,
        b_fake,
        sf_fake,
        c_fake,
        partial_fake,
        alpha_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE[cache_key] = compiled
    return compiled


def _e4m3_to_s0e5m3(sf_u8: torch.Tensor) -> torch.Tensor:
    """Reformat a uint8 tensor of E4M3 scale bytes to S0E5M3 bytes.
    Used in cute-dsl backend for faster in-kernel scale decode.
    """
    f16 = sf_u8.contiguous().view(torch.float8_e4m3fn).to(torch.float16)
    bits = f16.view(torch.int16).to(torch.int32) & 0xFFFF
    return ((bits >> 7) & 0xFF).to(torch.uint8)


_CUTE_DSL_PACK_TILE_K: int = 16  # K-tile size = MMA K-block size
_CUTE_DSL_PACK_TILE_N: int = 64  # N-tile size = kernel tile_N
_CUTE_DSL_PACK_INTS_PER_TILE: int = 128  # int32s per (16K x 64N) repack block


def _cute_dsl_pack_fp4_weight(b: torch.Tensor) -> torch.Tensor:
    """Repack a packed FP4 weight for the bf16 x fp4 cute-DSL kernel."""
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)

    k_half, n = b.shape
    k = k_half * 2
    if k % _CUTE_DSL_PACK_TILE_K != 0:
        raise ValueError(f"K must be a multiple of {_CUTE_DSL_PACK_TILE_K} (got K={k})")
    if n % _CUTE_DSL_PACK_TILE_N != 0:
        raise ValueError(f"N must be a multiple of {_CUTE_DSL_PACK_TILE_N} (got N={n})")

    device = b.device
    k_tiles = k // _CUTE_DSL_PACK_TILE_K
    n_tiles = n // _CUTE_DSL_PACK_TILE_N
    k_half_per_tile = _CUTE_DSL_PACK_TILE_K // 2  # 8 packed K-rows per tile

    u32_pos = torch.arange(
        _CUTE_DSL_PACK_INTS_PER_TILE, device=device, dtype=torch.long
    )
    u32_idx_local = u32_pos % 2
    lane = (u32_pos // 2) % 32
    n_warp_idx = u32_pos // 64

    tc_col = lane // 4  # in [0, 8)
    tc_row_half = lane % 4  # tc_row = tc_row_half * 2 in {0, 2, 4, 6}
    base_n = n_warp_idx * 8 + tc_col  # in [0, 16)

    byte_k_half_offset = torch.tensor([0, 4, 0, 4], device=device, dtype=torch.long)
    n_offset_stack = torch.tensor(
        [[0, 0, 16, 16], [32, 32, 48, 48]], device=device, dtype=torch.long
    )
    byte_n_offset = n_offset_stack[u32_idx_local]  # (128, 4)

    # Source byte within the (8, 64) tile for each (u32_pos, byte_idx).
    k_half_in_tile = tc_row_half[:, None] + byte_k_half_offset[None, :]  # (128, 4)
    n_in_tile = base_n[:, None] + byte_n_offset  # (128, 4)
    within_idx = (k_half_in_tile * _CUTE_DSL_PACK_TILE_N + n_in_tile).reshape(
        -1
    )  # (512,) flat index into a row-major (8, 64) tile

    # (K/2, N) -> (K_tiles, 8, N_tiles, 64) -> (K_tiles, N_tiles, 8*64) so the
    # 512 source bytes of each tile are contiguous, then gather them in
    # (u32_pos, byte_idx) order.
    tile_bytes = (
        b.reshape(k_tiles, k_half_per_tile, n_tiles, _CUTE_DSL_PACK_TILE_N)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles, n_tiles, k_half_per_tile * _CUTE_DSL_PACK_TILE_N)
    )
    gathered = tile_bytes[:, :, within_idx].reshape(
        k_tiles, n_tiles, _CUTE_DSL_PACK_INTS_PER_TILE, 4
    )

    # Each 4 consecutive bytes are one little-endian int32 (byte 0 = bits 0-7),
    # exactly what the kernel's 32-bit loads read -- reinterpret in place.
    return gathered.view(torch.int32).reshape(
        k_tiles, n_tiles * _CUTE_DSL_PACK_INTS_PER_TILE
    )


def _prepare_cute_dsl_sm12x(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cute-DSL-backend prep: repack the weight + unswizzle the SF.

    Produces the bespoke layout the cute-DSL kernel consumes:
      * weight: ``(K // 16, N * 2)`` int32 (see :func:`_cute_dsl_pack_fp4_weight`).
      * SF:     ``(K // block_size, N)`` uint8 -- per-block scales reformatted to
        S0E5M3, the format the cute-DSL kernel decodes.
    ``alpha`` is passed through unchanged (the compute step normalizes it
    to a ``(1,) float32`` scalar).  Pair the returned tensors with
    ``mm_bf16_fp4(a, b, b_descale, alpha, backend='cute-dsl')``.
    """
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size

    b_kn = b.t().contiguous()
    b_packed = _cute_dsl_pack_fp4_weight(b_kn)  # (K//16, N*2) int32

    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf)  # (N, K_sf) uint8
    sf_ksf_n = linear_sf.t().contiguous()  # (K_sf, N) uint8 (E4M3)
    sf_ksf_n = _e4m3_to_s0e5m3(sf_ksf_n)  # -> S0E5M3
    return b_packed, sf_ksf_n, alpha


def _prepare_cute_dsl_sm100(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cute-DSL-backend prep: keep SF in swizzled layout.

    ``convert_sf_to_mma_layout`` only creates the six-dimensional strided view
    consumed by the SM100 TMA descriptor; it does not copy or linearize the
    canonical 128x4 scale-factor buffer.
    """
    from ..cute_dsl.utils import convert_sf_to_mma_layout

    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    b_descale = b_descale.contiguous()
    weight_sf = convert_sf_to_mma_layout(
        b_descale,
        m=n,
        k=k,
        num_groups=1,
        sf_vec_size=block_size,
    )
    return b.contiguous(), weight_sf, alpha


def _prepare_cute_dsl(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch weight preparation to the architecture-specific DSL kernel."""
    major, minor = get_compute_capability(b.device)
    if (major, minor) in ((10, 0), (10, 3)):
        return _prepare_cute_dsl_sm100(b, b_descale, alpha, block_size)
    elif major == 12:
        return _prepare_cute_dsl_sm12x(b, b_descale, alpha, block_size)
    else:
        raise NotImplementedError(
            f"cute-dsl w4a16 GEMM only supports SM100/103 and SM12x; got {major}.{minor}"
        )


@functools.cache
def _bf16_fp4_cute_dsl_tactic_configs(
    n: int, k: int, sm_count: int
) -> Tuple[
    Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int, int, int, str],
    ...,
]:
    """Enumerate cute-DSL tactic configs for a given ``(N, K)``.

    Returns a tuple of ``(tile_shape_mnk, atom_layout, pipeline_depth,
    use_fp16_mma, tile_swizzle, k_splits, occupancy, kind)`` entries,
    where ``kind`` selects the kernel: ``"mma"`` (the dense GEMM, all
    fields live) or ``"gemv"`` (the m=1 streaming kernel, which only
    reads ``k_splits``).  The gemv entries include device-derived splits,
    so a tactic index is only meaningful for the ``(n, k, sm_count)`` it
    was enumerated with.
    """
    tile_k = 128 if k % 128 == 0 else 64

    # (tile_M, atom_layout) shapes the kernel is designed/validated for, at the
    # default tile_N=64; a tile_N=128 variant is added below for very large N.
    tile_m_atoms: List[Tuple[int, Tuple[int, int, int]]] = []
    if tile_k == 128:
        tile_m_atoms.append((16, (1, 2, 1)))
    tile_m_atoms.append((32, (2, 2, 1)))
    tile_m_atoms.append((64, (2, 2, 1)))

    configs: List[
        Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int, int, int, str]
    ] = []
    seen = set()

    def add(tile_m, atom, pdepth, fp16, tile_n=64, tk=None, swz=1, splits=1, occ=1):
        cfg = (
            (tile_m, tile_n, tile_k if tk is None else tk),
            atom,
            pdepth,
            fp16,
            swz,
            splits,
            occ,
            "mma",
        )
        key = (cfg[0], cfg[1], pdepth, fp16, swz, splits, occ)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    base_tile_m, base_atom = tile_m_atoms[0]
    add(base_tile_m, base_atom, 1, 1)  # 0: baseline
    add(base_tile_m, base_atom, 0, 1)  # no dequant prefetch (helps short-K)
    for tile_m, atom in tile_m_atoms[1:]:
        add(tile_m, atom, 1, 1)

    # Small-N grids underfill the GPU; offer them split-K variants.
    if n // 64 <= 256:
        k_tiles = k // tile_k
        for splits in (2, 4, 8):
            if splits <= k_tiles:
                add(base_tile_m, base_atom, 1, 1, splits=splits)

    # tile_N=128 halves the (m,n)-tile count but needs large wave count.
    if tile_k == 128 and n >= 12288 and n % 128 == 0:
        add(base_tile_m, base_atom, 1, 1, tile_n=128)

    # tile_K=64 has more ab stages, but requires larger problem size.
    if tile_k == 128 and n >= 8192:
        add(base_tile_m, base_atom, 1, 1, tile_n=64, tk=64)

    # tile_M=128 (taller M tile, atom (2,2,1)) -- the large-M *prefill* lever.
    if tile_k == 128:
        add(128, (2, 2, 1), 1, 1)

    # Threadblock swizzle (tile_swizzle=8) -- for large-M prefill.
    if tile_k == 128 and n * k >= 16 * 1024 * 1024:
        add(64, (2, 2, 1), 1, 1, swz=8)
    if tile_k == 128:
        add(128, (2, 2, 1), 1, 1, swz=8)

    # tile_N=128 (with tile_M=64, atom (2,2,1)) -- large shapes.
    if tile_k == 128 and n % 128 == 0 and n >= 4096:
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=8)
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=1)

    # occupancy=2: two co-resident CTAs per SM at half the ab_stage depth
    # (the kernel's occupancy note explains the trade).  The 128-tile gate
    # fills two CTAs per SM on ~188-SM parts; 5080-class still qualifies.
    if tile_k == 128 and n // 64 >= 128:
        add(base_tile_m, base_atom, 1, 1, occ=2)
        add(base_tile_m, base_atom, 0, 1, occ=2)

    # occupancy=2 x split-K: grids too small to fill two CTAs per SM (the
    # down-proj class) get there by splitting K; the makespan gate decides
    # per device whether the pieces fit.
    if tile_k == 128 and n // 64 <= 256:
        k_tiles = k // tile_k
        for splits in (2, 4, 8):
            if splits <= k_tiles:
                add(base_tile_m, base_atom, 1, 1, splits=splits, occ=2)

    # occupancy=3: 9 resident warps per SM; needs the 2-stage epilogue to
    # keep 3 ab stages in SMEM (paired in _get_cute_dsl_bf16_fp4_gemm).
    if tile_k == 128 and n // 64 >= 128:
        add(base_tile_m, base_atom, 1, 1, occ=3)

    # Streaming GEMV for m=1 decode (SMEM-free, no tensor cores; see
    # GemvBf16Fp4Sm12x).  splits shards K across grid.y for grid fill.
    if n % 64 == 0:
        k_tiles16 = k // 16
        gemv_splits = [s for s in (1, 2, 4, 8, 16, 32) if s * 4 <= k_tiles16]
        if gemv_splits:
            knee = _bf16_fp4_gemv_knee_split(n, k, sm_count)
            if knee not in gemv_splits:
                gemv_splits.append(knee)
        for splits in gemv_splits:
            configs.append(((1, 64, 16), (0, 0, 0), 0, 0, 1, splits, 1, "gemv"))

    return tuple(configs)


_BF16_FP4_CUTE_DSL_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),  # M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            5,  # out_tensor_index follows M
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


_SM100_BF16_FP4_KERNEL_CACHE: Dict[Tuple, object] = {}

_SM100_BF16_FP4_N_TILES = (8, 16, 32, 64, 128, 192)
_SM100_BF16_FP4_K_TILE = 256


def _sm100_bf16_fp4_tactic_configs() -> List[Tuple]:
    """Return the SM100 W4A16 tactics shared with the MoE GEMMs.

    Dense GEMM maps public output channels to the kernel M mode and public
    rows to its N mode.  The 128x192x256 one-CTA tile is excluded because it
    cannot retain the two load and transform stages required by the kernel.
    """
    tactics: List[Tuple] = []
    for route_tile in _SM100_BF16_FP4_N_TILES:
        for gemm_m, cluster_shape_mn in (
            (128, (1, 1)),
            (128, (2, 1)),
            (256, (2, 1)),
        ):
            if route_tile < 16 and gemm_m == 256:
                continue
            if route_tile == 192 and gemm_m == 128:
                continue
            for raster_along_m in (True, False):
                tactics.append(
                    (
                        (gemm_m, route_tile, _SM100_BF16_FP4_K_TILE),
                        cluster_shape_mn,
                        raster_along_m,
                    )
                )
    return tactics


_SM100_BF16_FP4_TACTICS = tuple(_sm100_bf16_fp4_tactic_configs())

_SM100_BF16_FP4_CUTE_DSL_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),  # public M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            5,  # out_tensor_index follows public M
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


def _get_sm100_bf16_fp4_kernel(
    weight_ptr,
    weight_sf_ptr,
    activation_ptr,
    alpha_ptr,
    output_ptr,
    n: int,
    m: int,
    k: int,
    max_active_clusters: int,
    stream,
    enable_pdl: bool,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    raster_along_m: bool,
):
    """Compile the SM100/103 dense W4A16 kernel for one tactic."""
    import cutlass
    import cutlass.cute as cute

    from .kernels.cute_dsl.dense_gemm_bf16_fp4_sm100 import (
        Sm100DenseGemmBf16Fp4Kernel,
    )

    use_2cta_instrs = mma_tiler_mnk[0] == 256
    transform_fragment_size = 128 if k == mma_tiler_mnk[2] else 32
    cache_key = (
        mma_tiler_mnk,
        cluster_shape_mn,
        bool(raster_along_m),
        transform_fragment_size,
        int(max_active_clusters),
        bool(enable_pdl),
    )
    compiled = _SM100_BF16_FP4_KERNEL_CACHE.get(cache_key)
    if compiled is not None:
        return compiled

    kernel = Sm100DenseGemmBf16Fp4Kernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mn=cluster_shape_mn,
        enable_pdl=enable_pdl,
        raster_along_m=raster_along_m,
        transform_fragment_size=transform_fragment_size,
    )
    compiled = cute.compile(
        kernel.wrapper,
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        alpha_ptr,
        output_ptr,
        n,
        m,
        k,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options="--opt-level 3 --enable-tvm-ffi",
    )
    _SM100_BF16_FP4_KERNEL_CACHE[cache_key] = compiled
    return compiled


def _launch_cute_dsl_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha_for_launch: torch.Tensor,
    out: torch.Tensor,
    tactic: Tuple,
    enable_pdl: bool,
) -> torch.Tensor:
    """Compile, cache, and launch one SM100 W4A16 tactic."""
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl.utils import current_cuda_stream, get_max_active_clusters, make_ptr

    m, k = map(int, a.shape)
    n = int(b.shape[0])
    mma_tiler_mnk, cluster_shape_mn, raster_along_m = tactic
    cluster_size = int(cluster_shape_mn[0]) * int(cluster_shape_mn[1])
    for name, tensor, align in (
        ("b", b, 32),
        ("b_descale", b_descale, 16),
        ("a", a, 32),
        ("alpha", alpha_for_launch, 16),
        ("out", out, 32),
    ):
        if tensor.data_ptr() % align:
            raise ValueError(
                f"SM100 cute-dsl requires a {align}-byte aligned {name}; "
                f"got address {tensor.data_ptr():`#x`}. Pass a freshly "
                "allocated contiguous tensor."
            )
    stream = current_cuda_stream()
    weight_ptr = make_ptr(
        cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    weight_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        b_descale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    activation_ptr = make_ptr(
        cutlass.BFloat16,
        a.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    alpha_ptr = make_ptr(
        cutlass.Float32,
        alpha_for_launch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = make_ptr(
        cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    max_active_clusters = get_max_active_clusters(cluster_size)
    compiled = _get_sm100_bf16_fp4_kernel(
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        alpha_ptr,
        output_ptr,
        n,
        m,
        k,
        max_active_clusters,
        stream,
        enable_pdl,
        mma_tiler_mnk,
        cluster_shape_mn,
        raster_along_m,
    )
    compiled(
        b.data_ptr(),
        b_descale.data_ptr(),
        a.data_ptr(),
        alpha_for_launch.data_ptr(),
        out.data_ptr(),
        n,
        m,
        k,
        stream,
    )
    return out


def _cute_dsl_sm100_bf16_fp4_runner(enable_pdl: bool = True) -> TunableRunner:
    """Build the tunable runner for the native-layout SM100 W4A16 GEMM."""

    class CuteDslSm100Bf16Fp4Runner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            _, b, _, _, out_dtype, _, block_size = inputs
            n = int(b.shape[0])
            k = int(b.shape[1]) * 2
            return (out_dtype, n, k, int(block_size), bool(enable_pdl))

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[Tuple]:
            import cutlass

            from .kernels.cute_dsl.dense_gemm_bf16_fp4_sm100 import (
                Sm100DenseGemmBf16Fp4Kernel,
            )

            a, b, _, _, _, _, _ = inputs
            m, k = map(int, a.shape)
            n = int(b.shape[0])
            valid_tactics: List[Tuple] = []
            for tactic in _SM100_BF16_FP4_TACTICS:
                mma_tiler_mnk, cluster_shape_mn, _ = tactic
                if Sm100DenseGemmBf16Fp4Kernel.can_implement(
                    mnkl=(n, m, k, 1),
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.BFloat16,
                    c_dtype=cutlass.BFloat16,
                    a_major="k",
                    b_major="k",
                    c_major="m",
                    mma_tiler=mma_tiler_mnk,
                    cluster_shape_mn=cluster_shape_mn,
                    use_2cta_instrs=mma_tiler_mnk[0] == 256,
                ):
                    valid_tactics.append(tactic)
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=-1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, b_descale, alpha_for_launch, _, out, _ = inputs
            if tactic == -1:
                valid = self.get_valid_tactics(inputs, None)
                if valid:
                    tactic = valid[0]
                else:
                    m, k = map(int, a.shape)
                    raise ValueError(
                        "no SM100 cute-dsl w4a16 tactic supports "
                        f"m={m}, n={int(b.shape[0])}, k={k}"
                    )
            return _launch_cute_dsl_sm100(
                a,
                b,
                b_descale,
                alpha_for_launch,
                out,
                tactic,
                enable_pdl,
            )

    return CuteDslSm100Bf16Fp4Runner()


def _compute_cute_dsl_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Launch the SM100 native-layout NVFP4 x BF16 GEMM."""
    if a.device != b.device or a.device != b_descale.device:
        raise ValueError(
            "a, b, and b_descale must be on the same CUDA device; got "
            f"{a.device}, {b.device}, and {b_descale.device}."
        )
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("SM100 cute-dsl requires contiguous a and b tensors.")
    if a.dtype != torch.bfloat16:
        raise TypeError(
            f"SM100 cute-dsl currently requires a bfloat16 activation; got {a.dtype}."
        )
    if b.dtype != torch.uint8:
        raise TypeError(
            "SM100 cute-dsl expects the uint8 NVFP4 weight returned by "
            "prepare_bf16_fp4_weights(..., backend='cute-dsl'); "
            f"got {b.dtype}."
        )
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"SM100 cute-dsl currently requires a bfloat16 output; got {out_dtype}."
        )

    m, k = map(int, a.shape)
    n = int(b.shape[0])
    if int(b.shape[1]) * 2 != k:
        raise ValueError(
            f"a.shape[1]={k} but b.shape={tuple(b.shape)} encodes K="
            f"{int(b.shape[1]) * 2}"
        )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (m, n):
            raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
        if out.device != a.device or not out.is_contiguous():
            raise ValueError(
                "out must be contiguous and on the same device as a; got "
                f"device={out.device}, contiguous={out.is_contiguous()}."
            )

    alpha_for_launch = _prepare_bf16_fp4_alpha(alpha, a.device)
    tuner = AutoTuner.get()
    runner = _cute_dsl_sm100_bf16_fp4_runner(enable_pdl=enable_pdl)
    inputs = [a, b, b_descale, alpha_for_launch, out_dtype, out, block_size]
    chosen_runner, tactic = tuner.choose_one(
        "bf16_fp4_cute_dsl_sm100_gemm",
        [runner],
        _SM100_BF16_FP4_CUTE_DSL_TUNING_CONFIG,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out


def _cute_dsl_bf16_fp4_runner(enable_pdl: bool = True) -> TunableRunner:
    """Build a ``CuteDslBf16Fp4Runner`` for the cute-DSL bf16 x fp4 GEMM."""

    class CuteDslBf16Fp4Runner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out_dtype, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            # The config list holds device-derived gemv splits, so a tuned
            # index must not replay on a different SM count.
            return (out_dtype, n, k, get_device_sm_count(a.device))

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a, b, _, _, _, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            m_opt = int(profile.get_opt_shapes()[0][0])
            sm_count = get_device_sm_count(a.device)
            # GEMV is validated only on SM12x; the autotuner ranks by time
            # and would not catch a wrong-but-fast kernel elsewhere.
            gemv_ok = get_compute_capability(a.device)[0] == 12
            configs = _bf16_fp4_cute_dsl_tactic_configs(n, k, sm_count)

            def split_reduces_makespan(cfg) -> bool:
                # Offer a split only if it shrinks the last wave >= 25%: the
                # autotuner's warm-cache timing hides the partials cost, so it
                # cannot reject bad splits itself.
                splits = cfg[5]
                if splits == 1:
                    return True
                tile_m, tile_n, _ = cfg[0]
                tiles = (-(-m_opt // tile_m)) * (-(-n // tile_n))
                # occupancy multiplies the co-resident CTA slots per SM, so
                # occ2 x split-K fits 2x the pieces in one wave.
                slots = sm_count * cfg[6]
                base_waves = -(-tiles // sm_count)
                split_waves = (-(-tiles * splits // slots)) / splits
                return split_waves <= 0.75 * base_waves

            def gemv_valid(cfg) -> bool:
                # The m buckets round up, so only runtime m == 1 lands in
                # the m_opt == 1 bucket.
                if m_opt != 1 or not gemv_ok:
                    return False
                warps_per_sm = (n // 64) * cfg[5] / sm_count
                return _GEMV_MIN_WARPS_PER_SM <= warps_per_sm <= _GEMV_MAX_WARPS_PER_SM

            return [
                i
                for i, cfg in enumerate(configs)
                if (
                    gemv_valid(cfg) if cfg[7] == "gemv" else split_reduces_makespan(cfg)
                )
            ]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            m = int(a.shape[0])
            cfg = (
                _bf16_fp4_cute_dsl_tactic_configs(n, k, get_device_sm_count(a.device))[
                    tactic
                ]
                if tactic >= 0
                else None
            )
            splits: Optional[int] = None
            if cfg is not None and cfg[7] == "gemv":
                if m != 1:
                    raise ValueError(f"the gemv tactic requires m == 1, got m={m}")
                splits = cfg[5]
            elif cfg is None and m == 1 and not do_preparation:
                # Prep calls run outside the autotuner's per-tactic exception
                # guard; a gemv failure there would abort the whole op.
                splits = _select_bf16_fp4_gemv_split(
                    n,
                    k,
                    get_compute_capability(a.device)[0],
                    get_device_sm_count(a.device),
                )
            if splits is not None:
                compiled = _get_cute_dsl_bf16_fp4_gemv(
                    splits, a.dtype, out_dtype, enable_pdl=enable_pdl
                )
                if splits > 1:
                    partial = _get_cache_buf(
                        "mm_bf16_fp4_split_k_partial",
                        splits * m * n * 4,
                        a.device,
                    ).view(torch.float32)
                else:
                    partial = _get_cache_buf(
                        "mm_bf16_fp4_split_k_partial_dummy", 16, a.device
                    ).view(torch.float32)
                compiled(a, b, b_sf_u8, out, partial, alpha_for_launch)
                return out
            if tactic < 0:
                # Fallback == pre-autotuner heuristic (M-aware), default knobs.
                tile_shape_mnk, atom_layout = _select_bf16_fp4_tile_shape(m, n, k)
                pipeline_depth, use_fp16_mma, tile_swizzle, occupancy = 1, 1, 1, 1
                k_splits = _select_bf16_fp4_k_splits(
                    m, n, k, tile_shape_mnk, get_device_sm_count(a.device)
                )
            else:
                (
                    tile_shape_mnk,
                    atom_layout,
                    pipeline_depth,
                    use_fp16_mma,
                    tile_swizzle,
                    k_splits,
                    occupancy,
                ) = cfg[:7]
            compiled = _get_cute_dsl_bf16_fp4_gemm(
                tile_shape_mnk,
                a.dtype,
                out_dtype,
                atom_layout,
                pipeline_depth,
                use_fp16_mma,
                enable_pdl=enable_pdl,
                tile_swizzle=tile_swizzle,
                k_splits=k_splits,
                occupancy=occupancy,
            )
            if k_splits > 1:
                if k_splits > k // tile_shape_mnk[2]:
                    raise ValueError(
                        f"k_splits={k_splits} exceeds the K-tile count for "
                        f"k={k}, tile_k={tile_shape_mnk[2]}"
                    )
                # The compiled call launches the reduce kernel as well, so
                # ``out`` is final on return.
                partial = _get_cache_buf(
                    "mm_bf16_fp4_split_k_partial",
                    k_splits * m * n * 4,
                    a.device,
                ).view(torch.float32)
            else:
                partial = _get_cache_buf(
                    "mm_bf16_fp4_split_k_partial_dummy", 16, a.device
                ).view(torch.float32)
            compiled(a, b, b_sf_u8, out, partial, alpha_for_launch)
            return out

    return CuteDslBf16Fp4Runner()


def _compute_cute_dsl(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Dispatch to the SM100/103 or SM12x compiled Blackwell kernel.

    SM100 consumes the native ``(N, K // 2)`` uint8 NVFP4 weight and 128x4 SF.

    SM12x consumes the packed ``(K // 16, N * 2)`` int32 weight and
    ``(K // block_size, N)`` uint8 SF in S0E5M3 format (reformatted from FP8-E4M3
    by :func:`_e4m3_to_s0e5m3`) returned by :func:`_prepare_cute_dsl`.
    """
    if get_compute_capability(a.device) in ((10, 0), (10, 3)):
        return _compute_cute_dsl_sm100(
            a,
            b,
            b_descale,
            alpha,
            out_dtype,
            out,
            block_size,
            enable_pdl,
        )

    if b.dtype != torch.int32:
        raise TypeError(
            f"cute-dsl backend expects the packed int32 weight from "
            f"prepare_bf16_fp4_weights(..., backend='cute-dsl'); got {b.dtype}."
        )
    if out_dtype != a.dtype:
        raise NotImplementedError(
            f"cute-dsl backend requires out_dtype == a.dtype (got "
            f"out_dtype={out_dtype}, a.dtype={a.dtype}).  Use the cudnn "
            f"backend for a mismatched output dtype."
        )
    k_tiles = int(b.shape[0])
    n = int(b.shape[1]) // 2
    k = k_tiles * block_size
    m = int(a.shape[0])
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from prepared b.shape="
            f"{tuple(b.shape)} is {k}"
        )

    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (m, n):
            raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")

    b_sf_u8 = b_descale.view(torch.uint8).contiguous()
    alpha_for_launch = _prepare_bf16_fp4_alpha(alpha, a.device)

    tuner = AutoTuner.get()
    runner = _cute_dsl_bf16_fp4_runner(enable_pdl=enable_pdl)
    inputs = [a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size]
    chosen_runner, tactic = tuner.choose_one(
        "bf16_fp4_cute_dsl_gemm",
        [runner],
        _BF16_FP4_CUTE_DSL_TUNING_CONFIG,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out
