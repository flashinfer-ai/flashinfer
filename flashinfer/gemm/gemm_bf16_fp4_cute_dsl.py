# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CuTe-DSL backend for the bf16 x fp4 GEMM (weight repack / kernel launch)."""

from typing import List, Optional, Tuple, cast

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
from .gemm_base import _TORCH_TO_CUTLASS_DTYPE_ATTR, _check_cute_dsl_availability
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
):
    # Normalize to a tuple (callers may pass a list) so the cache key is hashable.
    atom_layout = cast(Tuple[int, int, int], tuple(atom_layout))
    pipeline_depth = int(pipeline_depth)
    use_fp16_mma = int(use_fp16_mma)
    enable_pdl = bool(enable_pdl)
    tile_swizzle = int(tile_swizzle)
    cache_key = (
        tile_shape_mnk,
        a_dtype,
        c_dtype,
        atom_layout,
        pipeline_depth,
        use_fp16_mma,
        enable_pdl,
        tile_swizzle,
    )
    cached = _CUTE_DSL_MM_BF16_FP4_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _check_cute_dsl_availability()

    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernels.cute_dsl.dense_gemm_bf16_fp4_blackwell import (
        BlackwellDenseGemmBf16Fp4Kernel,
    )

    a_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[a_dtype])
    c_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[c_dtype])

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
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    gemm = BlackwellDenseGemmBf16Fp4Kernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mnk=tile_shape_mnk,
        atom_layout=atom_layout,
        pipeline_depth=pipeline_depth,
        use_fp16_mma=use_fp16_mma,
        enable_pdl=enable_pdl,
        tile_swizzle=tile_swizzle,
    )
    max_active_clusters = get_max_active_clusters(1)

    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_packed_fake,
        b_sf_fake,
        c_fake,
        alpha_fake,
        1,  # l (batch)
        max_active_clusters,
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


def _prepare_cute_dsl(
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


def _bf16_fp4_cute_dsl_tactic_configs(
    n: int, k: int
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]]:
    """Enumerate cute-DSL tactic configs for a given ``(N, K)``.

    Returns a list of ``(tile_shape_mnk, atom_layout, pipeline_depth,
    use_fp16_mma)`` tuples.
    """
    tile_k = 128 if k % 128 == 0 else 64

    # (tile_M, atom_layout) shapes the kernel is designed/validated for, at the
    # default tile_N=64; a tile_N=128 variant is added below for very large N.
    tile_m_atoms: List[Tuple[int, Tuple[int, int, int]]] = []
    if tile_k == 128:
        tile_m_atoms.append((16, (1, 2, 1)))
    tile_m_atoms.append((32, (2, 2, 1)))
    tile_m_atoms.append((64, (2, 2, 1)))

    configs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]] = []
    seen = set()

    def add(tile_m, atom, pdepth, fp16, tile_n=64, tk=None, swz=1):
        cfg = (
            (tile_m, tile_n, tile_k if tk is None else tk),
            atom,
            pdepth,
            fp16,
            swz,
        )
        key = (cfg[0], cfg[1], pdepth, fp16, swz)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    base_tile_m, base_atom = tile_m_atoms[0]
    add(base_tile_m, base_atom, 1, 1)  # 0: baseline
    add(base_tile_m, base_atom, 0, 1)  # no dequant prefetch (helps short-K)
    for tile_m, atom in tile_m_atoms[1:]:
        add(tile_m, atom, 1, 1)

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

    return configs


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


def _cute_dsl_bf16_fp4_runner(enable_pdl: bool = True) -> TunableRunner:
    """Build a ``CuteDslBf16Fp4Runner`` for the cute-DSL bf16 x fp4 GEMM."""

    class CuteDslBf16Fp4Runner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out_dtype, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return (out_dtype, n, k)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            _, b, _, _, _, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return list(range(len(_bf16_fp4_cute_dsl_tactic_configs(n, k))))

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
            if tactic < 0:
                # Fallback == pre-autotuner heuristic (M-aware), default knobs.
                tile_shape_mnk, atom_layout = _select_bf16_fp4_tile_shape(m, n, k)
                pipeline_depth, use_fp16_mma, tile_swizzle = 1, 1, 1
            else:
                (
                    tile_shape_mnk,
                    atom_layout,
                    pipeline_depth,
                    use_fp16_mma,
                    tile_swizzle,
                ) = _bf16_fp4_cute_dsl_tactic_configs(n, k)[tactic]
            compiled = _get_cute_dsl_bf16_fp4_gemm(
                tile_shape_mnk,
                a.dtype,
                out_dtype,
                atom_layout,
                pipeline_depth,
                use_fp16_mma,
                enable_pdl=enable_pdl,
                tile_swizzle=tile_swizzle,
            )
            compiled(a, b, b_sf_u8, out, alpha_for_launch)
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
    """cute-DSL-backend compute: dispatch to the compiled Blackwell kernel.

    ``b`` is the packed ``(K // 16, N * 2)`` int32 weight and
    ``b_descale`` the ``(K // block_size, N)`` uint8 SF in S0E5M3 format
    (reformatted from FP8-E4M3 by :func:`_e4m3_to_s0e5m3`) returned by
    :func:`_prepare_cute_dsl`.
    """
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
