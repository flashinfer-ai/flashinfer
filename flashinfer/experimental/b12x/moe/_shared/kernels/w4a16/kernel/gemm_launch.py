"""W4A16 GEMM compilation and launch bindings."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x._lib.compiler import KernelCompileSpec
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.utils.pointers import make_ptr
from b12x.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.program_cache import register_program_cache
from dataclasses import replace
import torch
from b12x.moe._shared.kernels.w4a16.kernel.buffers import _get_c_tmp
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _LUT_E4M3_VALUE_TABLE_ENTRIES,
    _candidate_tile_fits,
    _covering_count,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _normalize_scale_format,
    _scale_fake_int32_elements,
    _select_tile_config,
)
from b12x.moe._shared.kernels.w4a16.kernel.gemm import W4A16GemmKernel
from b12x.moe._shared.kernels.w4a16.kernel.types import (
    W4A16GemmCompileResult,
    _W4A16GemmLaunch,
)


_CACHE: dict[tuple, W4A16GemmCompileResult] = {}

register_program_cache(_CACHE)


def compile_w4a16_gemm(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    tile_n: int,
    tile_k: int,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "packed",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    trellis_pair_kind: str | None = None,
    trellis_rate_axis: str | None = None,
    dense_route_fast_path: bool = False,
) -> W4A16GemmCompileResult:
    scale_format = _normalize_scale_format(scale_format)
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if torch.cuda.is_available():
        device = int(torch.cuda.current_device())
    else:
        device = None
    kernel = W4A16GemmKernel(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
        dense_route_fast_path=bool(dense_route_fast_path),
        schedule_whole_tiles=weight_layout == "trellis_t256",
    )
    cache_key = (
        "w4a16_gemm",
        device,
        kernel.__cache_key__,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return attach_programs(
            replace(
                cached, max_m_blocks=max_m_blocks, blocks_per_sm=kernel.blocks_per_sm
            ),
            cached.compiled,
        )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_route_blocks = 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    if weight_layout == "trellis_t256":
        b_fake_elements = (
            num_experts * (size_k // 16) * (size_n // 16) * (8 * int(trellis_bits))
        )
    else:
        b_fake_elements = num_experts * (size_k // 16) * (size_n // 16 * 32)
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (b_fake_elements,),
        assumed_align=16,
    )
    c_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (
            _scale_fake_int32_elements(
                num_experts=num_experts,
                size_k=size_k,
                size_n=size_n,
                scale_format=scale_format,
            ),
        ),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_slots,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    topk_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (compile_size_m * top_k,),
        assumed_align=4,
    )
    c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                size_n * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256,),
        assumed_align=16,
    )
    trellis_lut_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (_LUT_E4M3_VALUE_TABLE_ENTRIES,),
        assumed_align=16,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        a_fake,
        a_fake,
        b_fake,
        c_fake,
        scales_fake,
        global_scale_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        topk_fake,
        c_tmp_fake,
        locks_fake,
        trellis_lut_fake,
        Int32(compile_size_m),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.gemm",
            4,
            cache_key,
        ),
    )
    result = W4A16GemmCompileResult(
        compiled=compiled,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        dense_route_fast_path=bool(dense_route_fast_path),
        trellis_bits=int(trellis_bits),
        trellis_codebook=str(trellis_codebook).lower(),
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
    )
    attach_programs(result, compiled)
    _CACHE[cache_key] = result
    return result


def _compile_w4a16_gemm_launch(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    packed_route_indices: torch.Tensor | None,
    sms: int,
    max_shared_mem: int,
    device: torch.device,
    c_tmp: torch.Tensor | None = None,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "packed",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    trellis_pair_kind: str | None = None,
    trellis_rate_axis: str | None = None,
    dense_route_fast_path: bool = False,
    route_slots: int | None = None,
    force_tile_config: tuple[int, int] | None = None,
) -> _W4A16GemmLaunch:
    planner_weight_bits = (
        max(4, int(trellis_bits)) if weight_layout == "trellis_t256" else 4
    )
    if force_tile_config is None:
        tile_k, tile_n, _, _ = _select_tile_config(
            problem_m=size_m,
            problem_n=size_n,
            problem_k=size_k,
            top_k=top_k,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=planner_weight_bits,
        )
    else:
        tile_k, tile_n = (int(v) for v in force_tile_config)
        cta_threads = tile_n * tile_k // 64
        if not _candidate_tile_fits(
            problem_n=size_n,
            problem_k=size_k,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=tile_n,
            tile_k=tile_k,
            cta_threads=cta_threads,
            max_shared_mem=max_shared_mem - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=planner_weight_bits,
        ):
            raise ValueError(
                "forced standalone W4A16 tile does not fit: "
                f"tile_k/tile_n={tile_k}/{tile_n}, N/K={size_n}/{size_k}"
            )
    kernel = compile_w4a16_gemm(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
        dense_route_fast_path=bool(dense_route_fast_path),
    )
    if route_slots is None:
        if packed_route_indices is None:
            raise ValueError(
                "standalone W4A16 launch requires route_slots or packed_route_indices"
            )
        route_slots = int(packed_route_indices.numel())
    c_tmp = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=int(route_slots),
            moe_block_size=moe_block_size,
            sms=sms,
            weight_layout=weight_layout,
        ),
        device=device,
        scratch=c_tmp,
    )
    return _W4A16GemmLaunch(kernel=kernel, c_tmp=c_tmp)
