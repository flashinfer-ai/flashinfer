"""Public isolated API for the primary paged-attention backend."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    compile as b12x_compile,
    launch as b12x_launch,
    run_compiled,
)
from b12x._lib.utils import current_cuda_stream
from b12x._lib.compile_plan import attach_programs, compile_only_launches_enabled
from ._controls import paged_control, paged_controls, snapshot_paged_controls

from .forward_paged import (
    PagedFp8ExtendRawForwardKernel,
    PagedForwardKernel,
)
from .forward_extend_generic import build_extend_forward_kernel
from .graph_replay import build_msa_prefill_union_metadata
from .merge import (
    LagunaVerifierMergeKernel,
    PagedPersistentMergeKernel,
    default_paged_persistent_ctas,
)
from .planner import use_paged_extend_fp8_pv_repack
from .traits import (
    PagedForwardTraits,
    select_paged_forward_traits_from_plan,
)
import os


@dataclass
class _PagedLaunchers:
    """Resident native programs and the static route chosen during preparation."""

    controls: Mapping[str, str | None]
    traits: PagedForwardTraits | None = None
    route: tuple[bool | int, ...] | None = None
    forward_kernel: Any | None = None
    metadata: Mapping[str, Any] | None = None
    forward: Any | None = None
    merge_kernel: Any | None = None
    merge: Any | None = None
    merge_regular_decode_graph: bool = False


def _attach_paged_programs(launchers: _PagedLaunchers) -> _PagedLaunchers:
    dependencies = tuple(
        program
        for program in (
            launchers.forward,
            launchers.merge,
            *(launchers.metadata or {}).values(),
        )
        if program is not None
    )
    return attach_programs(launchers, *dependencies)


_DECODE_NATIVE_FP8_QKV_MAX_SMALL_BATCH = 2
_DECODE_NATIVE_FP8_QKV_MIN_LONG_CHUNK_PAGES = 11


def _turbo_attention_enabled() -> bool:
    return paged_control("B12X_TURBO_ATTN") == "1"


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported dtype {dtype}")


def _torch_to_cutlass_storage_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float8_e4m3fn:
        return cutlass.Uint8
    return _torch_to_cutlass_dtype(dtype)


def _to_kernel_tensor(
    tensor: torch.Tensor | None,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
    dynamic_rank1: bool = False,
) -> torch.Tensor | cutlass.cute.Tensor | None:
    if compile_only_launches_enabled() and hasattr(tensor, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor

        leading_dim = next(
            (idx for idx, stride in enumerate(tensor.stride()) if stride == 1),
            None,
        )
        if tensor.numel() == 0:
            # DLPack preserves the native empty optional-tensor ABI, which the
            # CuTe fake-layout constructor cannot represent (positive extents only).
            from torch._subclasses.fake_tensor import unset_fake_temporarily

            with unset_fake_temporarily():
                empty = torch.empty_strided(
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    device="cpu",
                )
            converted = from_dlpack(empty, assumed_align=assumed_align)
            converted.element_type = dtype
            return converted
        if tensor.ndim >= 2 and leading_dim is not None:
            shape = tuple(cutlass.cute.sym_int(32) for _ in tensor.shape)
            strides = tuple(
                1 if idx == leading_dim else cutlass.cute.sym_int(64)
                for idx in range(tensor.ndim)
            )
        elif dynamic_rank1 and tensor.ndim == 1:
            shape, strides = (cutlass.cute.sym_int(32),), (1,)
        else:
            shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        return make_fake_tensor(dtype, shape, strides, assumed_align=assumed_align)
    if tensor is None:
        return None
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next(
        (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    elif dynamic_rank1 and tensor.ndim == 1:
        # Rank-1 tensors are normally exact because most metadata vectors
        # define a compile-time ABI. Extend worklists are different: their
        # selected capacity is a graph-static launch extent, not kernel policy.
        # Pass that extent through the CuTe JIT wrapper at launch so fixed
        # Q-capacity buckets can share one compiled entry without inheriting
        # the first bucket's CTA count.
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=0)
    return cute_tensor


def _as_int32_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.int32 else tensor.to(torch.int32)


def _uses_native_fp8_attention_mma(
    *,
    plan,
) -> bool:
    return _turbo_attention_enabled() and plan.kv_dtype == torch.float8_e4m3fn


def _resolve_native_fp8_attention_mma_flags(
    *,
    plan,
) -> tuple[bool, bool, bool]:
    if getattr(plan, "msa_block_sparse", False):
        return False, False, False
    use_native_fp8_qk = _uses_native_fp8_attention_mma(plan=plan)
    decode_runtime_chunk_guard = False
    if use_native_fp8_qk and plan.mode in ("decode", "verify"):
        if (
            plan.total_q > _DECODE_NATIVE_FP8_QKV_MAX_SMALL_BATCH
            and plan.kv_chunk_size
            < _DECODE_NATIVE_FP8_QKV_MIN_LONG_CHUNK_PAGES * plan.page_size
        ):
            # Mid-batch short-chunk decode sees the native FP8 QK loss without enough replay gain.
            use_native_fp8_qk = False
    use_native_fp8_pv = (
        use_native_fp8_qk
        and plan.mode in ("decode", "verify")
        and plan.kv_chunk_size <= 384
    )
    return use_native_fp8_qk, use_native_fp8_pv, decode_runtime_chunk_guard


@lru_cache(maxsize=16)
def _dummy_plane_tma_desc_ptrs(device_index: int, num_heads: int) -> torch.Tensor:
    return torch.zeros(
        (num_heads,), dtype=torch.int64, device=torch.device("cuda", device_index)
    )


def _encode_plane_tma_descriptors(
    cache: torch.Tensor,
    *,
    plane_cols: int,
    tile_rows: int | None = None,
) -> torch.Tensor:
    if cache.ndim != 4:
        raise ValueError(
            "cache must have shape [num_pages, page_size, kv_heads, head_dim]"
        )
    num_pages, page_size, kv_heads, head_dim = [int(dim) for dim in cache.shape]
    if plane_cols <= 0 or head_dim % plane_cols != 0:
        raise ValueError(f"plane_cols={plane_cols} must divide head_dim={head_dim}")
    if tile_rows is None:
        tile_rows = page_size
    if tile_rows <= 0 or page_size % tile_rows != 0:
        raise ValueError(
            f"tile_rows={tile_rows} must be positive and divide page_size={page_size}"
        )

    swizzle_name = os.environ.get("B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "")
    swizzle = (
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        if swizzle_name == "none"
        else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    )
    if cache.dtype == torch.float8_e4m3fn:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        elem_bytes = 1
    elif cache.dtype == torch.bfloat16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        elem_bytes = 2
    elif cache.dtype == torch.float16:
        data_type = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
        elem_bytes = 2
    else:
        raise TypeError(f"unsupported plane TMA cache dtype {cache.dtype}")
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = kv_heads * head_dim * elem_bytes
    # The plane descriptors flatten the cache to rank-2 [head_dim, total_rows]
    # with row pitch row_bytes; rows within a page must therefore be contiguous
    # at exactly kv_heads * head_dim elements, and total_rows must be derived
    # from the PAGE STRIDE (not the shape) so strided caches (e.g. vLLM's
    # combined [N, 2, page, H, D] K/V slices) keep flat row coordinates
    # `tile_idx * tile_rows` in-bounds.
    if int(cache.stride(3)) != 1:
        raise ValueError(
            "plane TMA cache requires contiguous head_dim (stride(3) == 1)"
        )
    if int(cache.stride(2)) != head_dim:
        raise ValueError(
            "plane TMA cache requires contiguous kv-head rows (stride(2) == head_dim)"
        )
    row_stride_elems = int(cache.stride(1))
    if row_stride_elems != kv_heads * head_dim:
        raise ValueError(
            "plane TMA cache requires rows-within-page contiguous: "
            f"stride(1)={row_stride_elems} != kv_heads*head_dim={kv_heads * head_dim}"
        )
    page_stride_elems = int(cache.stride(0))
    if page_stride_elems % row_stride_elems != 0:
        raise ValueError(
            "plane TMA cache page stride must be a whole number of rows: "
            f"stride(0)={page_stride_elems}, stride(1)={row_stride_elems}"
        )
    page_stride_rows = page_stride_elems // row_stride_elems
    if page_stride_rows % tile_rows != 0:
        raise ValueError(
            f"plane TMA cache page stride rows ({page_stride_rows}) must be a multiple of tile_rows={tile_rows}"
        )
    total_rows = num_pages * page_stride_rows
    base_ptr = int(cache.view(torch.uint8).data_ptr())
    head_stride_bytes = head_dim * elem_bytes

    host_desc = torch.empty((kv_heads, 16), dtype=torch.uint64)
    for kv_head_idx in range(kv_heads):
        result, tensor_map = cuda.cuTensorMapEncodeTiled(
            data_type,
            2,
            base_ptr + kv_head_idx * head_stride_bytes,
            [U64(head_dim), U64(total_rows)],
            [U64(row_bytes)],
            [U32(plane_cols), U32(tile_rows)],
            [U32(1), U32(1)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
        host_desc[kv_head_idx] = torch.tensor(
            [int(word) for word in tensor_map.opaque],
            dtype=torch.uint64,
        )
    return host_desc.to(device=cache.device, non_blocking=False)


def _paged_kv_tiles_per_table_entry(
    cache: torch.Tensor,
    *,
    page_size: int,
    tile_rows: int,
    name: str,
) -> int:
    """Physical `tile_rows`-row tiles spanned by one page-table entry stride.

    Derived from the CACHE STRIDE, not assumed contiguous: vLLM's combined
    [num_blocks, 2, 128, H, D] cache exposes K/V as strided slices whose page
    stride covers both planes; the off-plane rows appear as phantom tiles that
    are never addressed.
    """
    if cache.ndim != 4:
        raise ValueError(
            f"{name} must be rank-4 [num_pages, page_size, kv_heads, head_dim]"
        )
    if int(cache.stride(3)) != 1:
        raise ValueError(f"{name} requires contiguous head_dim (stride(3) == 1)")
    row_stride_elems = int(cache.stride(1))
    if row_stride_elems <= 0:
        raise ValueError(f"{name} requires a positive token stride")
    page_stride_elems = int(cache.stride(0))
    if page_stride_elems % row_stride_elems != 0:
        raise ValueError(
            f"{name} page stride must be a whole number of token rows: "
            f"stride(0)={page_stride_elems}, stride(1)={row_stride_elems}"
        )
    page_stride_rows = page_stride_elems // row_stride_elems
    if page_stride_rows % tile_rows != 0:
        raise ValueError(
            f"{name} page stride rows ({page_stride_rows}) must be a multiple of {tile_rows}"
        )
    if page_stride_rows < int(page_size):
        raise ValueError(
            f"{name} page stride rows ({page_stride_rows}) are smaller than page_size={page_size}"
        )
    return page_stride_rows // tile_rows


def _descriptor_row_ptrs(desc: torch.Tensor) -> torch.Tensor:
    row_bytes = int(desc.stride(0)) * desc.element_size()
    base_ptr = int(desc.data_ptr())
    ptrs = [base_ptr + idx * row_bytes for idx in range(int(desc.shape[0]))]
    return torch.tensor(ptrs, dtype=torch.int64, device=desc.device)


def _get_cached_plane_tma_descs(
    scratch: object,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plane_cols: int,
    tile_rows: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
    cached = getattr(scratch, "_live_plane_tma_desc_cache", None)
    if cached is None:
        return None
    key = (
        int(k_cache.data_ptr()),
        int(v_cache.data_ptr()),
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        plane_cols,
        tile_rows,
    )
    return cached.get(key)


def _tensor_meta_key(
    tensor: torch.Tensor | None,
    *,
    dynamic_dims: tuple[int, ...] = (),
    dynamic_strides: tuple[int, ...] = (),
) -> tuple[tuple[object, ...], tuple[object, ...], str, tuple[str, int | None]] | None:
    if tensor is None:
        return None
    dynamic_dim_set = set(dynamic_dims)
    dynamic_stride_set = set(dynamic_strides)
    return (
        tuple(
            DimKey.dynamic() if idx in dynamic_dim_set else int(dim)
            for idx, dim in enumerate(tensor.shape)
        ),
        tuple(
            DimKey.dynamic() if idx in dynamic_stride_set else int(stride)
            for idx, stride in enumerate(tensor.stride())
        ),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _paged_kv_cache_meta_key(
    tensor: torch.Tensor | None,
) -> tuple[tuple[object, ...], tuple[object, ...], str, tuple[str, int | None]] | None:
    # K/V cache page capacity is runtime storage capacity. The selected kernel
    # policy depends on page layout, dtype, heads, and strides, not total pages.
    return _tensor_meta_key(tensor, dynamic_dims=(0,))


def _traits_compile_key(traits: PagedForwardTraits) -> tuple[object, ...]:
    return (
        int(traits.cta_tile_q),
        int(traits.cta_tile_kv),
        int(traits.num_mma_q),
        int(traits.num_mma_kv),
        int(traits.num_mma_d_qk),
        int(traits.num_mma_d_vo),
        int(traits.num_warps_q),
        int(traits.num_warps_kv),
        int(traits.num_threads),
        int(traits.head_dim_qk),
        int(traits.head_dim_vo),
        int(traits.upcast_stride_q),
        int(traits.upcast_stride_k),
        int(traits.upcast_stride_v),
        int(traits.upcast_stride_o),
        str(traits.q_dtype),
        str(traits.kv_dtype),
        str(traits.o_dtype),
        int(traits.q_smem_bytes),
        int(traits.shared_storage_bytes),
        int(traits.num_ctas_per_sm),
        int(traits.max_smem_per_threadblock),
    )


@lru_cache(maxsize=64)
def _build_forward_kernel(
    traits: PagedForwardTraits,
    gqa_group_size: int,
    split_kv: bool,
    single_request_decode_graph: bool,
    single_qtile_decode_graph: bool,
    regularized_decode_graph: bool,
    analytic_laguna_decode_graph: bool,
    analytic_laguna_decode_batch: int,
    analytic_laguna_decode_total_chunks: int,
    use_native_fp8_qk: bool,
    use_native_fp8_pv: bool,
    decode_only: bool,
    decode_native_fp8_runtime_chunk_guard: bool,
    window_left: int,
    has_attention_sink_bias: bool,
    has_relative_attention_bias: bool,
    msa_block_sparse: bool,
    page_size: int,
    page_tiles_per_entry: int,
    controls_key: tuple = (),
) -> PagedForwardKernel:
    return PagedForwardKernel(
        _torch_to_cutlass_dtype(traits.q_dtype),
        _torch_to_cutlass_dtype(traits.kv_dtype),
        _torch_to_cutlass_storage_dtype(traits.kv_dtype),
        _torch_to_cutlass_dtype(traits.o_dtype),
        traits=traits,
        gqa_group_size=gqa_group_size,
        split_kv=split_kv,
        single_request_decode_graph=single_request_decode_graph,
        single_qtile_decode_graph=single_qtile_decode_graph,
        regularized_decode_graph=regularized_decode_graph,
        analytic_laguna_decode_graph=analytic_laguna_decode_graph,
        analytic_laguna_decode_batch=analytic_laguna_decode_batch,
        analytic_laguna_decode_total_chunks=analytic_laguna_decode_total_chunks,
        use_native_fp8_qk=use_native_fp8_qk,
        use_native_fp8_pv=use_native_fp8_pv,
        decode_only=decode_only,
        decode_native_fp8_runtime_chunk_guard=decode_native_fp8_runtime_chunk_guard,
        window_left=window_left,
        has_attention_sink_bias=has_attention_sink_bias,
        has_relative_attention_bias=has_relative_attention_bias,
        msa_block_sparse=msa_block_sparse,
        page_size=page_size,
        page_tiles_per_entry=page_tiles_per_entry,
    )


@lru_cache(maxsize=32)
def _build_laguna_verify_forward_kernel(
    page_tiles_per_entry: int,
    batch: int,
    max_chunks_per_request: int,
    two_wave_b1: bool,
    use_tma_kv_load: bool,
    controls_key: tuple = (),
) -> PagedFp8ExtendRawForwardKernel:
    return PagedFp8ExtendRawForwardKernel(
        split_kv=True,
        cta_tile_q=64,
        page_size=128,
        head_dim=128,
        page_tiles_per_entry=page_tiles_per_entry,
        analytic_verify_batch=batch,
        analytic_verify_max_chunks=max_chunks_per_request,
        analytic_verify_two_wave_b1=two_wave_b1,
        use_tma_kv_load=use_tma_kv_load,
    )


def _use_laguna_verify_forward_kernel(
    *,
    plan,
    traits: PagedForwardTraits,
    use_native_fp8_qk: bool,
    has_attention_sink_bias: bool,
    has_relative_attention_bias: bool,
) -> bool:
    batch_capacity = int(plan.page_table_shape[0])
    return (
        plan.mode == "verify"
        and plan.enable_cuda_graph
        and plan.split_kv
        and not plan.msa_block_sparse
        and plan.window_left < 0
        and plan.page_size == 128
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == torch.float8_e4m3fn
        and plan.num_q_heads == 24
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 6
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and 1 <= batch_capacity <= 8
        and plan.cta_tile_q == 64
        and traits.cta_tile_kv == 64
        and traits.num_warps_q == 4
        and traits.num_warps_kv == 1
        and int(plan.total_q) == batch_capacity * 8
        and int(plan.num_qo_tiles) == batch_capacity
        and all(int(qo_tile) == 0 for qo_tile in plan.qo_tile_indices)
        and not use_native_fp8_qk
        and not has_attention_sink_bias
        and not has_relative_attention_bias
    )


def _use_laguna_decode_analytic_kernel(
    *,
    plan,
    traits: PagedForwardTraits,
    use_native_fp8_qk: bool,
    has_attention_sink_bias: bool,
    has_relative_attention_bias: bool,
) -> bool:
    batch_capacity = int(plan.page_table_shape[0])
    return (
        plan.mode == "decode"
        and plan.enable_cuda_graph
        and plan.graph_chunk_policy
        and plan.fixed_split_size < 0
        and plan.split_kv
        and not plan.msa_block_sparse
        and plan.window_left < 0
        and plan.page_size == 128
        and plan.dtype == torch.bfloat16
        and plan.kv_dtype == torch.float8_e4m3fn
        and plan.num_q_heads == 24
        and plan.num_kv_heads == 4
        and plan.gqa_group_size == 6
        and plan.head_dim_qk == 128
        and plan.head_dim_vo == 128
        and 1 <= batch_capacity <= 8
        and int(plan.total_q) == batch_capacity
        and int(plan.num_qo_tiles) == batch_capacity
        and all(int(qo_tile) == 0 for qo_tile in plan.qo_tile_indices)
        and plan.cta_tile_q == 16
        and traits.cta_tile_kv in (64, 128)
        and traits.num_warps_q == 1
        and traits.num_warps_kv == 4
        and traits.num_mma_kv == (2 if traits.cta_tile_kv == 128 else 1)
        and not use_native_fp8_qk
        and not has_attention_sink_bias
        and not has_relative_attention_bias
    )


@lru_cache(maxsize=32)
def _build_extend_forward_kernel(
    traits: PagedForwardTraits,
    use_native_fp8_qk: bool,
    use_native_fp8_pv: bool,
    window_left: int,
    has_attention_sink_bias: bool,
    has_relative_attention_bias: bool,
    msa_block_sparse: bool,
    msa_union_tile: bool,
    page_size: int,
    use_fp8_pv_repack: bool,
    controls_key: tuple = (),
) -> object:
    return build_extend_forward_kernel(
        traits,
        use_native_fp8_qk,
        use_native_fp8_pv,
        window_left=window_left,
        has_attention_sink_bias=has_attention_sink_bias,
        has_relative_attention_bias=has_relative_attention_bias,
        msa_block_sparse=msa_block_sparse,
        msa_union_tile=msa_union_tile,
        page_size=page_size,
        use_fp8_pv_repack=use_fp8_pv_repack,
    )


@lru_cache(maxsize=16)
def _build_merge_kernel(
    dtype: torch.dtype,
    head_dim: int,
    total_q: int,
    persistent_ctas: int,
    direct_grid: bool,
    regular_decode_graph: bool,
    analytic_laguna_verify_graph: bool,
    analytic_laguna_decode_graph: bool,
    pair_bf16_partial_loads: bool,
) -> PagedPersistentMergeKernel:
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    merge_vec_size = head_dim // 32
    merge_bdx = 32
    merge_bdy = (
        3 if dtype == torch.bfloat16 and head_dim == 128 and regular_decode_graph else 4
    )
    laguna_b1_decode_merge = (
        dtype == torch.bfloat16
        and head_dim == 128
        and regular_decode_graph
        and int(total_q) == 1
    )
    if analytic_laguna_verify_graph or laguna_b1_decode_merge:
        # Match the proven FlashInfer merge geometry for BF16 D128:
        # 16 lanes each move a native 16-byte vector while eight y-lanes
        # partition the split states.  The previous 32x4 layout needed
        # paired half-warp loads for the same bytes.
        merge_vec_size = 8
        merge_bdx = 16
        merge_bdy = 8
    if (
        dtype == torch.bfloat16
        and head_dim == 128
        and regular_decode_graph
        and int(total_q) == 4
    ):
        merge_bdy = 4
    return PagedPersistentMergeKernel(
        cutlass_dtype,
        cutlass_dtype,
        head_dim=head_dim,
        vec_size=merge_vec_size,
        bdx=merge_bdx,
        bdy=merge_bdy,
        persistent_ctas=persistent_ctas,
        direct_grid=direct_grid,
        regular_decode_graph=regular_decode_graph,
        analytic_laguna_verify_graph=analytic_laguna_verify_graph,
        analytic_laguna_decode_graph=analytic_laguna_decode_graph,
        pair_bf16_partial_loads=pair_bf16_partial_loads,
    )


@lru_cache(maxsize=8)
def _build_laguna_verify_merge_kernel(
    persistent_ctas: int,
    two_wave_b1: bool,
) -> LagunaVerifierMergeKernel:
    return LagunaVerifierMergeKernel(
        persistent_ctas,
        two_wave_b1=two_wave_b1,
    )


def _resolve_paged_attention_binding(
    *,
    binding,
    q: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    output: torch.Tensor | None,
    q2k_indices: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    attention_sink_bias: torch.Tensor | None,
    relative_attention_bias: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if binding is None:
        raise TypeError("paged_attention_forward requires binding")

    extras = [
        name
        for name, value in (
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("output", output),
            ("q2k_indices", q2k_indices),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
            ("attention_sink_bias", attention_sink_bias),
            ("relative_attention_bias", relative_attention_bias),
        )
        if value is not None
    ]
    if extras:
        raise ValueError(
            "paged attention binding owns runtime tensors and scratch; "
            f"do not also pass {', '.join(extras)}"
        )
    binding_scratch = getattr(binding, "scratch", None)
    if binding_scratch is None:
        raise TypeError("paged attention binding must expose scratch")
    return (
        binding.q,
        binding.k_cache,
        binding.v_cache,
        binding_scratch,
        binding.output,
        getattr(binding, "q2k_indices", None),
        binding.k_descale,
        binding.v_descale,
        binding.attention_sink_bias,
        binding.relative_attention_bias,
    )


def _capture_decode_graph_replay_metadata_if_needed(
    scratch: object,
    *,
    analytic_device_schedule: bool = False,
) -> None:
    scratch_device = getattr(scratch, "device", None)
    if scratch_device is not None and torch.device(scratch_device).type != "cuda":
        return
    if not torch.cuda.is_current_stream_capturing():
        return
    if not (
        getattr(scratch, "use_cuda_graph", False)
        and getattr(scratch, "mode", None) == "decode"
    ):
        return
    if getattr(scratch, "_plan", None) is None:
        raise RuntimeError(
            "decode CUDA graph capture requires "
            "prepare_decode_graph_replay_state before capture"
        )
    owner_scratch_plan = getattr(scratch, "_owner_scratch_plan", None)
    if owner_scratch_plan is not None:
        # ``Plan.bind()`` may have run before capture.  In that case the graph
        # first sees the Plan-owned schedule addresses here, in ``run()``, and
        # the owner must be pinned before a later prepare can replace them.
        owner_scratch_plan._mark_decode_graph_replay_state_captured(
            getattr(scratch, "_plan_metadata_cache", None)
        )
    if analytic_device_schedule:
        # This specialization derives its active split count and balanced
        # chunk ranges directly from the persistent device cache-length
        # tensor.  Capturing the generic metadata updater would add a launch
        # without contributing any state consumed by the forward or merge.
        return
    if getattr(scratch, "_decode_graph_chunk_pages_lut", None) is None:
        raise RuntimeError(
            "decode CUDA graph capture requires "
            "prepare_decode_graph_replay_state before capture"
        )
    update_metadata = getattr(
        scratch,
        "update_decode_graph_replay_metadata_from_runtime_cache_seqlens",
        None,
    )
    if update_metadata is None:
        raise RuntimeError(
            "decode CUDA graph scratch does not expose a replay metadata updater"
        )
    # Capture one device updater for every forward invocation.  The lifetime
    # flag below records that fixed addresses escaped into at least one graph;
    # it must not suppress the updater when the same prepared state is captured
    # into a second graph.
    update_metadata()
    scratch._decode_graph_metadata_captured_in_graph = True


def paged_attention_forward(
    q: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    *,
    output: torch.Tensor | None = None,
    q2k_indices: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    relative_attention_bias: torch.Tensor | None = None,
    binding=None,
    _compile_only: bool = False,
    _launchers: _PagedLaunchers | None = None,
    _collect_launchers: _PagedLaunchers | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        q,
        k_cache,
        v_cache,
        workspace,
        output,
        q2k_indices,
        k_descale,
        v_descale,
        attention_sink_bias,
        relative_attention_bias,
    ) = _resolve_paged_attention_binding(
        binding=binding,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
        relative_attention_bias=relative_attention_bias,
    )
    plan = workspace.plan
    page_table = workspace.page_table
    cache_seqlens = workspace.cache_seqlens
    cu_seqlens_q = workspace.cu_seqlens_q
    if page_table is None or cache_seqlens is None or cu_seqlens_q is None:
        raise RuntimeError("paged attention scratch metadata has not been prepared")
    if plan.split_kv and (workspace.tmp_output is None or workspace.tmp_lse is None):
        raise ValueError(
            "split-kv plan requires tmp_output and tmp_lse in the scratch binding"
        )
    exact_decode_graph_bucket = (
        getattr(workspace, "use_cuda_graph", False)
        and getattr(workspace, "mode", None) == "decode"
        and getattr(workspace, "_decode_graph_chunk_pages_lut", None) is not None
    )
    capacity_extend_graph = (
        getattr(workspace, "fixed_capacity", False)
        and getattr(workspace, "use_cuda_graph", False)
        and getattr(workspace, "mode", None) == "extend"
    )
    if exact_decode_graph_bucket and int(q.shape[0]) != int(plan.total_q):
        raise ValueError(
            "decode graph q total_q must exactly match the prepared bucket: "
            f"got {int(q.shape[0])}, expected {int(plan.total_q)}"
        )
    if capacity_extend_graph and not (0 < int(q.shape[0]) <= int(plan.total_q)):
        raise ValueError(
            "extend graph q total_q must be within the prepared capacity: "
            f"got {int(q.shape[0])}, capacity {int(plan.total_q)}"
        )
    if k_descale is not None and k_descale.ndim == 2 and int(k_descale.shape[1]) == 1:
        source = k_descale[:, 0]
        if _launchers is not None and not source.is_contiguous():
            k_descale = workspace._prepared_optional_buffers["k_descale"]
            k_descale.copy_(source)
        else:
            k_descale = source.contiguous()
    if v_descale is not None and v_descale.ndim == 2 and int(v_descale.shape[1]) == 1:
        source = v_descale[:, 0]
        if _launchers is not None and not source.is_contiguous():
            v_descale = workspace._prepared_optional_buffers["v_descale"]
            v_descale.copy_(source)
        else:
            v_descale = source.contiguous()
    # A one-request descale is semantically scalar.  Normalize its otherwise
    # incidental singleton stride so engine warmup and the first live request
    # share one compiled ABI.  ``as_strided`` is an allocation-free view and
    # batches larger than one retain their per-request stride.
    if k_descale is not None and k_descale.ndim == 1 and int(k_descale.numel()) == 1:
        k_descale = k_descale.as_strided((1,), (0,))
    if v_descale is not None and v_descale.ndim == 1 and int(v_descale.numel()) == 1:
        v_descale = v_descale.as_strided((1,), (0,))
    if output.ndim != 3:
        raise ValueError(
            f"output must be rank-3 [total_q, heads, head_dim], got {tuple(output.shape)}"
        )
    if exact_decode_graph_bucket and int(output.shape[0]) != int(plan.total_q):
        raise ValueError(
            "decode graph output total_q must exactly match the prepared bucket: "
            f"got {int(output.shape[0])}, expected {int(plan.total_q)}"
        )
    required_output_rows = (
        int(q.shape[0]) if capacity_extend_graph else int(plan.total_q)
    )
    if int(output.shape[0]) < required_output_rows:
        raise ValueError(
            "output first dimension must be at least "
            f"total_q={required_output_rows}, got {int(output.shape[0])}"
        )
    if tuple(output.shape[1:]) != (plan.num_q_heads, plan.head_dim_vo):
        raise ValueError(
            "output shape must match the prepared scratch contract: "
            f"expected (*, {plan.num_q_heads}, {plan.head_dim_vo}), got {tuple(output.shape)}"
        )
    if (
        k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn
    ) and (k_descale is None or v_descale is None):
        raise ValueError("fp8 paged caches require k_descale and v_descale")
    if workspace.kv_window_start_tokens is None:
        raise RuntimeError("paged attention scratch is missing kv_window_start_tokens")
    if getattr(plan, "msa_block_sparse", False):
        if attention_sink_bias is not None:
            raise ValueError(
                "MSA block-sparse paged attention does not support attention_sink_bias"
            )
        if relative_attention_bias is not None:
            raise ValueError(
                "MSA block-sparse paged attention does not support "
                "relative_attention_bias"
            )
        if q2k_indices is None:
            raise ValueError("MSA block-sparse paged attention requires q2k_indices")
        if q2k_indices.ndim != 3:
            raise ValueError(
                "q2k_indices must have shape [kv_heads, total_q_capacity, 16]"
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError("q2k_indices must be torch.int32")
        if q2k_indices.device != q.device:
            raise ValueError("q2k_indices must be on the same CUDA device as q")
        if not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous")
        expected_prefix = (plan.num_kv_heads, 16)
        if (
            int(q2k_indices.shape[0]) != expected_prefix[0]
            or int(q2k_indices.shape[2]) != expected_prefix[1]
        ):
            raise ValueError(
                "q2k_indices must have shape "
                f"({plan.num_kv_heads}, >=total_q, 16), got {tuple(q2k_indices.shape)}"
            )
        if int(q2k_indices.shape[1]) < int(plan.total_q):
            raise ValueError(
                "q2k_indices total_q_capacity is smaller than active total_q"
            )
    elif q2k_indices is not None:
        raise ValueError("q2k_indices can only be supplied for MSA block-sparse plans")
    has_attention_sink_bias = attention_sink_bias is not None
    if attention_sink_bias is None:
        attention_sink_bias = (
            workspace._prepared_optional_buffers["attention_sink_bias"]
            if _launchers is not None
            else torch.empty(0, dtype=torch.float32, device=q.device)
        )
    else:
        if attention_sink_bias.ndim != 1:
            raise ValueError(
                f"attention_sink_bias must be rank-1 [num_q_heads], got {tuple(attention_sink_bias.shape)}"
            )
        if int(attention_sink_bias.shape[0]) != plan.num_q_heads:
            raise ValueError(
                f"attention_sink_bias must have {plan.num_q_heads} elements, got {int(attention_sink_bias.shape[0])}"
            )
        if attention_sink_bias.device != q.device:
            raise ValueError("attention_sink_bias must be on the same CUDA device as q")
        if _launchers is not None and (
            attention_sink_bias.dtype != torch.float32
            or not attention_sink_bias.is_contiguous()
        ):
            source = attention_sink_bias
            attention_sink_bias = workspace._prepared_optional_buffers[
                "attention_sink_bias"
            ]
            attention_sink_bias.copy_(source)
        else:
            if attention_sink_bias.dtype != torch.float32:
                attention_sink_bias = attention_sink_bias.to(torch.float32)
            if not attention_sink_bias.is_contiguous():
                attention_sink_bias = attention_sink_bias.contiguous()

    has_relative_attention_bias = relative_attention_bias is not None
    if relative_attention_bias is not None:
        if relative_attention_bias.ndim != 3:
            raise ValueError(
                "relative_attention_bias must be rank-3 "
                "[total_q_capacity, num_q_heads, relative_extent], got "
                f"{tuple(relative_attention_bias.shape)}"
            )
        if int(relative_attention_bias.shape[0]) < int(plan.total_q):
            raise ValueError(
                "relative_attention_bias first dimension must be at least "
                f"total_q={plan.total_q}, got "
                f"{int(relative_attention_bias.shape[0])}"
            )
        if int(relative_attention_bias.shape[1]) != plan.num_q_heads:
            raise ValueError(
                "relative_attention_bias must have "
                f"{plan.num_q_heads} query heads, got "
                f"{int(relative_attention_bias.shape[1])}"
            )
        if int(relative_attention_bias.shape[2]) <= 0:
            raise ValueError("relative_attention_bias extent must be positive")
        if relative_attention_bias.device != q.device:
            raise ValueError(
                "relative_attention_bias must be on the same CUDA device as q"
            )
        if relative_attention_bias.dtype != q.dtype:
            raise TypeError(
                "relative_attention_bias must have the same dtype as q; "
                f"got {relative_attention_bias.dtype} and {q.dtype}"
            )
        if not relative_attention_bias.is_contiguous():
            raise ValueError("relative_attention_bias must be contiguous")

    page_size = int(plan.page_size)
    if _launchers is not None:
        if _launchers.traits is None or _launchers.route is None:
            raise RuntimeError("unprepared paged launchers are missing static routing")
        traits = _launchers.traits
        (
            use_native_fp8_qk,
            use_native_fp8_pv,
            decode_native_fp8_runtime_chunk_guard,
            page_tiles_per_entry,
            single_request_decode_graph,
            single_qtile_decode_graph,
            regularized_decode_graph,
            use_laguna_verify_kernel,
            use_laguna_decode_analytic_kernel,
            laguna_verify_use_tma,
            laguna_verify_two_wave_b1,
            laguna_verify_max_chunks,
        ) = _launchers.route
        forward_kernel = _launchers.forward_kernel
    else:
        controls_key = tuple(
            (
                _collect_launchers.controls
                if _collect_launchers is not None
                else snapshot_paged_controls()
            ).items()
        )
        traits = select_paged_forward_traits_from_plan(plan)
        use_native_fp8_qk, use_native_fp8_pv, decode_native_fp8_runtime_chunk_guard = (
            _resolve_native_fp8_attention_mma_flags(plan=plan)
        )
        page_tiles_per_entry = 1
        if int(plan.page_size) == 128:
            if (
                k_cache.dtype
                not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
                or v_cache.dtype != k_cache.dtype
            ):
                raise ValueError(
                    "page_size=128 paged attention requires matching fp16, bf16, or fp8 e4m3 K/V caches"
                )
            k_tiles_per_entry = _paged_kv_tiles_per_table_entry(
                k_cache, page_size=128, tile_rows=traits.cta_tile_kv, name="k_cache"
            )
            v_tiles_per_entry = _paged_kv_tiles_per_table_entry(
                v_cache, page_size=128, tile_rows=traits.cta_tile_kv, name="v_cache"
            )
            if k_tiles_per_entry != v_tiles_per_entry:
                raise ValueError(
                    "page_size=128 paged attention requires equal K/V page-stride tile counts"
                )
            page_tiles_per_entry = k_tiles_per_entry
        single_request_decode_graph = False
        single_qtile_decode_graph = False
        regularized_decode_graph = False
        use_laguna_verify_kernel = _use_laguna_verify_forward_kernel(
            plan=plan,
            traits=traits,
            use_native_fp8_qk=use_native_fp8_qk,
            has_attention_sink_bias=has_attention_sink_bias,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        use_laguna_decode_analytic_kernel = _use_laguna_decode_analytic_kernel(
            plan=plan,
            traits=traits,
            use_native_fp8_qk=use_native_fp8_qk,
            has_attention_sink_bias=has_attention_sink_bias,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        laguna_verify_use_tma = bool(use_laguna_verify_kernel)
        laguna_verify_two_wave_b1 = bool(
            use_laguna_verify_kernel
            and int(plan.page_table_shape[0]) == 1
            and int(plan.graph_ctas_per_sm) >= 2
        )
        laguna_verify_max_chunks = (
            int(plan.total_num_partial_rows) // int(plan.total_q)
            if use_laguna_verify_kernel
            else 0
        )
        if plan.mode == "extend":
            if plan.split_kv:
                raise ValueError("extend plans no longer support split-kv")
            forward_kernel = _build_extend_forward_kernel(
                traits,
                use_native_fp8_qk,
                use_native_fp8_pv,
                plan.window_left,
                has_attention_sink_bias,
                has_relative_attention_bias,
                bool(plan.msa_block_sparse),
                bool(getattr(plan, "msa_union_tile", False)),
                int(plan.page_size),
                use_paged_extend_fp8_pv_repack(
                    plan, resident_ctas_per_sm=int(traits.num_ctas_per_sm)
                ),
                controls_key=controls_key,
            )
        elif use_laguna_verify_kernel:
            forward_kernel = _build_laguna_verify_forward_kernel(
                page_tiles_per_entry,
                int(plan.page_table_shape[0]),
                laguna_verify_max_chunks,
                laguna_verify_two_wave_b1,
                laguna_verify_use_tma,
                controls_key=controls_key,
            )
        else:
            single_request_decode_graph = (
                plan.mode == "decode"
                and plan.enable_cuda_graph
                and plan.split_kv
                and not bool(plan.msa_block_sparse)
                and plan.gqa_group_size <= plan.cta_tile_q
                and workspace._decode_graph_chunk_pages_lut is not None
                and plan.num_qo_tiles == 1
                and plan.page_table_shape[0] == 1
            )
            single_qtile_decode_graph = (
                plan.mode == "decode"
                and plan.enable_cuda_graph
                and plan.split_kv
                and not bool(plan.msa_block_sparse)
                and plan.gqa_group_size <= plan.cta_tile_q
                and workspace._decode_graph_chunk_pages_lut is not None
                and plan.page_table_shape[0] > 1
                and max(plan.qo_tile_indices, default=0) == 0
            )
            regularized_decode_graph = (
                single_qtile_decode_graph and workspace._use_regular_decode_graph_replay
            )
            forward_kernel = _build_forward_kernel(
                traits,
                plan.gqa_group_size,
                plan.split_kv,
                single_request_decode_graph,
                single_qtile_decode_graph,
                regularized_decode_graph,
                use_laguna_decode_analytic_kernel,
                int(plan.page_table_shape[0])
                if use_laguna_decode_analytic_kernel
                else 1,
                int(plan.total_num_partial_rows)
                if use_laguna_decode_analytic_kernel
                else 1,
                use_native_fp8_qk,
                use_native_fp8_pv,
                plan.mode == "decode",
                decode_native_fp8_runtime_chunk_guard,
                plan.window_left,
                has_attention_sink_bias,
                has_relative_attention_bias,
                bool(plan.msa_block_sparse),
                int(plan.page_size),
                page_tiles_per_entry,
                controls_key=controls_key,
            )
        _launchers = None
    _capture_decode_graph_replay_metadata_if_needed(
        workspace, analytic_device_schedule=use_laguna_decode_analytic_kernel
    )
    if _collect_launchers is not None:
        _collect_launchers.traits = traits
        _collect_launchers.route = (
            use_native_fp8_qk,
            use_native_fp8_pv,
            decode_native_fp8_runtime_chunk_guard,
            page_tiles_per_entry,
            single_request_decode_graph,
            single_qtile_decode_graph,
            regularized_decode_graph,
            use_laguna_verify_kernel,
            use_laguna_decode_analytic_kernel,
            laguna_verify_use_tma,
            laguna_verify_two_wave_b1,
            laguna_verify_max_chunks,
        )
        _collect_launchers.forward_kernel = forward_kernel
    forward_output = workspace.tmp_output if plan.split_kv else output
    forward_lse = workspace.tmp_lse if plan.split_kv else workspace.lse
    assert forward_output is not None
    assert forward_lse is not None
    if use_laguna_verify_kernel or use_laguna_decode_analytic_kernel:
        partial_rows = int(plan.total_num_partial_rows)
        forward_output = forward_output.narrow(0, 0, partial_rows)
        forward_lse = forward_lse.narrow(0, 0, partial_rows)
    msa_union_blocks = None
    msa_union_masks = None
    msa_union_counts = None
    if bool(getattr(plan, "msa_union_tile", False)):
        msa_union_blocks = getattr(workspace, "msa_union_blocks", None)
        msa_union_masks = getattr(workspace, "msa_union_masks", None)
        msa_union_counts = getattr(workspace, "msa_union_counts", None)
        if (
            msa_union_blocks is None
            or msa_union_masks is None
            or msa_union_counts is None
        ):
            raise RuntimeError(
                "MSA union-tile prefill requires union metadata scratch buffers"
            )
        assert q2k_indices is not None
        if not _compile_only:
            build_msa_prefill_union_metadata(
                q2k_indices=q2k_indices,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                request_indices=workspace.request_indices,
                qo_tile_indices=workspace.qo_tile_indices,
                block_valid_mask=workspace.block_valid_mask,
                union_blocks=msa_union_blocks,
                union_masks=msa_union_masks,
                union_counts=msa_union_counts,
                _prepared=None if _launchers is None else _launchers.metadata,
            )
    q_arg = _to_kernel_tensor(q, _torch_to_cutlass_dtype(q.dtype))
    k_cache_arg = (
        _to_kernel_tensor(k_cache.view(torch.uint8), cutlass.Uint8)
        if k_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(k_cache, _torch_to_cutlass_dtype(k_cache.dtype))
    )
    v_cache_arg = (
        _to_kernel_tensor(v_cache.view(torch.uint8), cutlass.Uint8)
        if v_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(v_cache, _torch_to_cutlass_dtype(v_cache.dtype))
    )
    forward_output_arg = _to_kernel_tensor(
        forward_output, _torch_to_cutlass_dtype(forward_output.dtype)
    )
    forward_lse_arg = _to_kernel_tensor(forward_lse, cutlass.Float32)
    page_table_arg = _to_kernel_tensor(
        _as_int32_tensor(page_table), cutlass.Int32, assumed_align=4
    )
    cache_seqlens_arg = _to_kernel_tensor(
        _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
    )
    cu_seqlens_q_arg = _to_kernel_tensor(
        _as_int32_tensor(cu_seqlens_q), cutlass.Int32, assumed_align=4
    )
    request_indices_arg = _to_kernel_tensor(
        workspace.request_indices,
        cutlass.Int32,
        assumed_align=4,
        dynamic_rank1=True,
    )
    qo_tile_indices_arg = _to_kernel_tensor(
        workspace.qo_tile_indices,
        cutlass.Int32,
        assumed_align=4,
        dynamic_rank1=True,
    )
    kv_tile_indices_arg = _to_kernel_tensor(
        workspace.kv_tile_indices,
        cutlass.Int32,
        assumed_align=4,
        dynamic_rank1=True,
    )
    o_indptr_arg = _to_kernel_tensor(workspace.o_indptr, cutlass.Int32, assumed_align=4)
    kv_chunk_size_arg = _to_kernel_tensor(
        workspace.kv_chunk_size_ptr, cutlass.Int32, assumed_align=4
    )
    kv_window_start_arg = _to_kernel_tensor(
        workspace.kv_window_start_tokens, cutlass.Int32, assumed_align=4
    )
    block_valid_mask_arg = _to_kernel_tensor(
        workspace.block_valid_mask,
        cutlass.Int32,
        assumed_align=4,
        dynamic_rank1=True,
    )
    q2k_indices_arg = _to_kernel_tensor(q2k_indices, cutlass.Int32, assumed_align=4)
    msa_union_blocks_arg = _to_kernel_tensor(
        msa_union_blocks, cutlass.Int32, assumed_align=4
    )
    msa_union_masks_arg = _to_kernel_tensor(
        msa_union_masks, cutlass.Int32, assumed_align=4
    )
    msa_union_counts_arg = _to_kernel_tensor(
        msa_union_counts, cutlass.Int32, assumed_align=4
    )
    attention_sink_bias_arg = _to_kernel_tensor(
        attention_sink_bias, cutlass.Float32, assumed_align=4
    )
    relative_attention_bias_arg = _to_kernel_tensor(
        relative_attention_bias,
        _torch_to_cutlass_dtype(q.dtype),
    )
    k_descale_arg = _to_kernel_tensor(k_descale, cutlass.Float32)
    v_descale_arg = _to_kernel_tensor(v_descale, cutlass.Float32)
    k_tma_desc_ptrs: torch.Tensor | None = None
    v_tma_desc_ptrs: torch.Tensor | None = None
    k_tma_desc: torch.Tensor | None = None
    v_tma_desc: torch.Tensor | None = None
    if (
        plan.mode == "extend"
        and (
            getattr(forward_kernel, "use_paged_kv_tma_raw_desc_issue", False)
            or getattr(forward_kernel, "use_paged_kv_tma_fp8_raw_issue", False)
        )
        and not _compile_only
        and _launchers is None
    ):
        cached_descs = _get_cached_plane_tma_descs(
            workspace,
            k_cache=k_cache,
            v_cache=v_cache,
            plane_cols=forward_kernel.kv_tma_plane_head_dim,
            tile_rows=forward_kernel.stage_tile_rows,
        )
        if cached_descs is not None:
            k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs = cached_descs
        else:
            k_tma_desc = _encode_plane_tma_descriptors(
                k_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            v_tma_desc = _encode_plane_tma_descriptors(
                v_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            k_tma_desc_ptrs = _descriptor_row_ptrs(k_tma_desc)
            v_tma_desc_ptrs = _descriptor_row_ptrs(v_tma_desc)
            workspace._live_plane_tma_desc_cache[
                (
                    int(k_cache.data_ptr()),
                    int(v_cache.data_ptr()),
                    tuple(k_cache.shape),
                    tuple(v_cache.shape),
                    forward_kernel.kv_tma_plane_head_dim,
                    forward_kernel.stage_tile_rows,
                )
            ] = (k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs)
    if k_tma_desc_ptrs is None or v_tma_desc_ptrs is None:
        if _launchers is not None:
            prepared_descs = getattr(workspace, "_prepared_plane_tma_descs", None)
            if prepared_descs is None:
                raise RuntimeError(
                    "prepared paged TMA launch requires persistent descriptor pointers"
                )
            k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs = prepared_descs
        elif _compile_only:
            dummy_desc_ptrs = torch.empty(
                (plan.num_kv_heads,), dtype=torch.int64, device=k_cache.device
            )
            k_tma_desc_ptrs = dummy_desc_ptrs
            v_tma_desc_ptrs = dummy_desc_ptrs
        else:
            dummy_desc_ptrs = _dummy_plane_tma_desc_ptrs(
                torch.cuda.current_device(), plan.num_kv_heads
            )
            k_tma_desc_ptrs = dummy_desc_ptrs
            v_tma_desc_ptrs = dummy_desc_ptrs
    workspace._live_plane_tma_descs = (
        k_tma_desc,
        v_tma_desc,
        k_tma_desc_ptrs,
        v_tma_desc_ptrs,
    )

    stream = current_cuda_stream()
    use_capacity_contract = (
        plan.mode in ("extend", "verify") and workspace.fixed_capacity
    )
    q_cache_tensor = (
        workspace._plan_q
        if use_capacity_contract and workspace._plan_q is not None
        else q
    )
    output_cache_tensor = (
        workspace._plan_output
        if use_capacity_contract and workspace._plan_output is not None
        else forward_output
    )
    cuda_graph_decode = plan.mode == "decode" and plan.enable_cuda_graph
    dynamic_first_dim = () if cuda_graph_decode else (0,)
    # The worklist length is a graph-static launch extent, not kernel policy.
    # Its rank-1 CuTe tensors are marked layout-dynamic below, so the launch
    # wrapper receives the selected bucket's real extent on every invocation.
    grid_worklist_dynamic_first_dim = (0,)
    forward_lse_dynamic_dims = (
        dynamic_first_dim if plan.split_kv else (() if cuda_graph_decode else (1,))
    )
    forward_lse_dynamic_strides = () if plan.split_kv or cuda_graph_decode else (0,)
    forward_cache_key = [
        _traits_compile_key(traits),
        (
            plan.mode,
            bool(plan.split_kv),
            bool(plan.enable_cuda_graph),
            int(plan.gqa_group_size),
            bool(single_request_decode_graph),
            bool(single_qtile_decode_graph),
            bool(regularized_decode_graph),
            bool(use_laguna_decode_analytic_kernel),
            bool(plan.mode == "decode"),
            bool(use_native_fp8_qk),
            bool(use_native_fp8_pv),
            bool(decode_native_fp8_runtime_chunk_guard),
            int(plan.window_left),
            bool(has_attention_sink_bias),
            bool(has_relative_attention_bias),
            bool(plan.msa_block_sparse),
            bool(getattr(plan, "msa_union_tile", False)),
            bool(use_laguna_verify_kernel),
            bool(laguna_verify_two_wave_b1),
            bool(laguna_verify_use_tma),
            int(page_size),
            int(page_tiles_per_entry),
        ),
        _tensor_meta_key(q_cache_tensor, dynamic_dims=dynamic_first_dim),
        _paged_kv_cache_meta_key(k_cache),
        _paged_kv_cache_meta_key(v_cache),
        _tensor_meta_key(page_table, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(cache_seqlens, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(cu_seqlens_q, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(
            workspace.request_indices,
            dynamic_dims=grid_worklist_dynamic_first_dim,
        ),
        _tensor_meta_key(
            workspace.qo_tile_indices,
            dynamic_dims=grid_worklist_dynamic_first_dim,
        ),
        _tensor_meta_key(
            workspace.kv_tile_indices,
            dynamic_dims=grid_worklist_dynamic_first_dim,
        ),
        _tensor_meta_key(workspace.o_indptr, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(workspace.kv_chunk_size_ptr),
        _tensor_meta_key(
            workspace.kv_window_start_tokens, dynamic_dims=dynamic_first_dim
        ),
        _tensor_meta_key(
            workspace.block_valid_mask,
            dynamic_dims=grid_worklist_dynamic_first_dim,
        ),
        _tensor_meta_key(attention_sink_bias),
        _tensor_meta_key(
            relative_attention_bias,
            dynamic_dims=dynamic_first_dim,
        ),
        _tensor_meta_key(output_cache_tensor, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(
            forward_lse,
            dynamic_dims=forward_lse_dynamic_dims,
            dynamic_strides=forward_lse_dynamic_strides,
        ),
        _tensor_meta_key(k_descale, dynamic_dims=dynamic_first_dim),
        _tensor_meta_key(v_descale, dynamic_dims=dynamic_first_dim),
    ]
    cache_key_labels = [
        "traits",
        "kernel_policy",
        "q_contract" if use_capacity_contract else "q",
        "k_cache",
        "v_cache",
        "page_table",
        "cache_seqlens",
        "cu_seqlens_q",
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "o_indptr",
        "kv_chunk_size_ptr",
        "kv_window_start_tokens",
        "block_valid_mask",
        "attention_sink_bias",
        "relative_attention_bias",
        "output_contract" if use_capacity_contract else "forward_output",
        "forward_lse",
        "k_descale",
        "v_descale",
    ]
    forward_args = [
        q_arg,
        k_cache_arg,
        v_cache_arg,
        page_table_arg,
        cache_seqlens_arg,
        cu_seqlens_q_arg,
        request_indices_arg,
        qo_tile_indices_arg,
        kv_tile_indices_arg,
        o_indptr_arg,
        kv_chunk_size_arg,
        kv_window_start_arg,
        block_valid_mask_arg,
    ]
    forward_cache_key.append(
        _tensor_meta_key(
            q2k_indices,
            dynamic_dims=(() if cuda_graph_decode else (1,)),
            dynamic_strides=(
                (0,)
                if (
                    not cuda_graph_decode
                    and q2k_indices is not None
                    and getattr(q2k_indices, "ndim", 0) >= 1
                    and int(q2k_indices.shape[0]) == 1
                )
                else ()
            ),
        )
    )
    cache_key_labels.append("q2k_indices")
    forward_args.append(q2k_indices_arg)
    if plan.mode == "extend":
        forward_cache_key.extend(
            (
                _tensor_meta_key(
                    msa_union_blocks,
                    dynamic_dims=grid_worklist_dynamic_first_dim,
                ),
                _tensor_meta_key(
                    msa_union_masks,
                    dynamic_dims=grid_worklist_dynamic_first_dim,
                ),
                _tensor_meta_key(
                    msa_union_counts,
                    dynamic_dims=grid_worklist_dynamic_first_dim,
                ),
            )
        )
        cache_key_labels.extend(
            ("msa_union_blocks", "msa_union_masks", "msa_union_counts")
        )
        forward_args.extend(
            (msa_union_blocks_arg, msa_union_masks_arg, msa_union_counts_arg)
        )
    forward_args.extend(
        [
            attention_sink_bias_arg,
            relative_attention_bias_arg,
            forward_output_arg,
            forward_lse_arg,
        ]
    )
    forward_args.extend((k_descale_arg, v_descale_arg))
    if plan.mode == "extend":
        k_tma_desc_arg = _to_kernel_tensor(
            k_tma_desc_ptrs, cutlass.Int64, assumed_align=8
        )
        v_tma_desc_arg = _to_kernel_tensor(
            v_tma_desc_ptrs, cutlass.Int64, assumed_align=8
        )
        forward_args.extend((k_tma_desc_arg, v_tma_desc_arg))
        forward_cache_key.extend(
            (
                _tensor_meta_key(k_tma_desc_ptrs),
                _tensor_meta_key(v_tma_desc_ptrs),
            )
        )
        cache_key_labels.extend(("k_tma_desc_ptrs", "v_tma_desc_ptrs"))
    if use_laguna_verify_kernel:
        # Keep the verifier entry's ABI as narrow as its device contract:
        # no window, sparse-index, bias, or extend-only metadata reaches the
        # specialization, so those paths cannot inflate its register footprint.
        forward_args = [
            q_arg,
            k_cache_arg,
            v_cache_arg,
            page_table_arg,
            cache_seqlens_arg,
            cu_seqlens_q_arg,
            request_indices_arg,
            qo_tile_indices_arg,
            kv_tile_indices_arg,
            o_indptr_arg,
            kv_chunk_size_arg,
            block_valid_mask_arg,
            forward_output_arg,
            forward_lse_arg,
            k_descale_arg,
            v_descale_arg,
        ]
    forward_args.append(stream)
    forward_spec = KernelCompileSpec.from_key(
        "attention.paged.forward",
        26,
        tuple(forward_cache_key),
        labels=tuple(cache_key_labels),
    )
    if _compile_only:
        compiled_forward = b12x_compile(
            forward_kernel, *forward_args, compile_spec=forward_spec
        )
        if _collect_launchers is not None:
            _collect_launchers.forward = compiled_forward
    elif _launchers is not None:
        if _launchers.forward is None:
            raise RuntimeError("unprepared paged forward launcher")
        run_compiled(_launchers.forward, tuple(forward_args))
    else:
        b12x_launch(
            forward_kernel,
            compile_spec=forward_spec,
            compile_args=tuple(forward_args),
            runtime_args=tuple(forward_args),
        )

    if plan.split_kv:
        if _launchers is not None:
            if _launchers.merge is None:
                raise RuntimeError("prepared split attention has no merge program")
            tmp_output_arg = _to_kernel_tensor(
                forward_output, _torch_to_cutlass_dtype(forward_output.dtype)
            )
            tmp_lse_arg = _to_kernel_tensor(forward_lse, cutlass.Float32)
            merge_cache_seqlens_arg = _to_kernel_tensor(
                _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
            )
            output_arg = _to_kernel_tensor(
                output, _torch_to_cutlass_dtype(output.dtype)
            )
            lse_arg = _to_kernel_tensor(workspace.lse, cutlass.Float32)
            if use_laguna_verify_kernel:
                merge_args = (
                    tmp_output_arg,
                    tmp_lse_arg,
                    merge_cache_seqlens_arg,
                    output_arg,
                    lse_arg,
                )
            else:
                merge_args = (
                    tmp_output_arg,
                    tmp_lse_arg,
                    _to_kernel_tensor(
                        workspace.merge_indptr, cutlass.Int32, assumed_align=4
                    ),
                    merge_cache_seqlens_arg,
                    kv_chunk_size_arg,
                    output_arg,
                    lse_arg,
                    None
                    if _launchers.merge_regular_decode_graph
                    else _to_kernel_tensor(
                        workspace.total_num_rows_ptr, cutlass.Int32, assumed_align=4
                    ),
                )
            run_compiled(_launchers.merge, (*merge_args, stream))
            rows = int(q.shape[0]) if capacity_extend_graph else plan.total_q
            return output.narrow(0, 0, rows), workspace.lse.narrow(
                1, 0, rows
            ).transpose(0, 1)
        persistent_ctas = default_paged_persistent_ctas(
            total_rows=plan.total_q,
            num_heads=plan.num_q_heads,
            device=output.device,
        )
        merge_regular_decode_graph = (
            plan.mode == "decode"
            and plan.enable_cuda_graph
            and plan.gqa_group_size <= plan.cta_tile_q
            and workspace._decode_graph_chunk_pages_lut is not None
            and workspace._use_regular_decode_graph_replay
            and max(plan.qo_tile_indices, default=0) == 0
        )
        if _collect_launchers is not None:
            _collect_launchers.merge_regular_decode_graph = merge_regular_decode_graph
        merge_direct_grid = merge_regular_decode_graph
        merge_analytic_laguna_verify_graph = bool(use_laguna_verify_kernel)
        merge_analytic_laguna_decode_graph = bool(use_laguna_decode_analytic_kernel)
        pair_bf16_merge_partial_loads = (
            (
                (
                    plan.mode == "decode"
                    and 2 <= int(plan.total_q) <= 4
                    and plan.gqa_group_size == 6
                )
                or (
                    plan.mode == "verify"
                    and plan.enable_cuda_graph
                    and plan.cta_tile_q == 64
                    and plan.page_size == 128
                    and plan.num_q_heads == 24
                    and plan.num_kv_heads == 4
                    and plan.gqa_group_size == 6
                    and plan.window_left < 0
                )
            )
            and output.dtype == torch.bfloat16
            and workspace.tmp_output is not None
            and workspace.tmp_output.dtype == torch.bfloat16
            and plan.head_dim_vo == 128
        )
        tmp_output_arg = _to_kernel_tensor(
            forward_output, _torch_to_cutlass_dtype(forward_output.dtype)
        )
        tmp_lse_arg = _to_kernel_tensor(forward_lse, cutlass.Float32)
        merge_indptr_arg = _to_kernel_tensor(
            workspace.merge_indptr, cutlass.Int32, assumed_align=4
        )
        merge_cache_seqlens_arg = _to_kernel_tensor(
            _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
        )
        output_arg = _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype))
        lse_arg = _to_kernel_tensor(workspace.lse, cutlass.Float32)
        if merge_analytic_laguna_verify_graph:
            exact_merge_work = int(plan.total_q * plan.num_q_heads)
            laguna_merge_ctas = exact_merge_work
            merge_kernel = _build_laguna_verify_merge_kernel(
                laguna_merge_ctas,
                laguna_verify_two_wave_b1,
            )
            merge_args = (
                tmp_output_arg,
                tmp_lse_arg,
                merge_cache_seqlens_arg,
                output_arg,
                lse_arg,
            )
            merge_spec = KernelCompileSpec.from_key(
                "attention.paged.laguna_verify_merge",
                3,
                (
                    _tensor_meta_key(forward_output),
                    _tensor_meta_key(forward_lse),
                    _tensor_meta_key(cache_seqlens),
                    _tensor_meta_key(output),
                    _tensor_meta_key(workspace.lse),
                    laguna_merge_ctas,
                    laguna_verify_two_wave_b1,
                ),
                labels=(
                    "tmp_output",
                    "tmp_lse",
                    "cache_seqlens",
                    "output",
                    "lse",
                    "persistent_ctas",
                    "two_wave_b1",
                ),
            )
        else:
            merge_kernel = _build_merge_kernel(
                output.dtype,
                plan.head_dim_vo,
                plan.total_q,
                persistent_ctas,
                merge_direct_grid,
                merge_regular_decode_graph,
                False,
                merge_analytic_laguna_decode_graph,
                pair_bf16_merge_partial_loads,
            )
            total_num_rows_arg = (
                None
                if merge_regular_decode_graph
                else _to_kernel_tensor(
                    workspace.total_num_rows_ptr, cutlass.Int32, assumed_align=4
                )
            )
            merge_args = (
                tmp_output_arg,
                tmp_lse_arg,
                merge_indptr_arg,
                merge_cache_seqlens_arg,
                kv_chunk_size_arg,
                output_arg,
                lse_arg,
                total_num_rows_arg,
            )
            merge_dynamic_first_dim = () if cuda_graph_decode else (0,)
            merge_cache_key = (
                _tensor_meta_key(forward_output, dynamic_dims=merge_dynamic_first_dim),
                _tensor_meta_key(forward_lse, dynamic_dims=merge_dynamic_first_dim),
                _tensor_meta_key(
                    workspace.merge_indptr, dynamic_dims=merge_dynamic_first_dim
                ),
                _tensor_meta_key(cache_seqlens, dynamic_dims=merge_dynamic_first_dim),
                _tensor_meta_key(workspace.kv_chunk_size_ptr),
                _tensor_meta_key(output, dynamic_dims=merge_dynamic_first_dim),
                _tensor_meta_key(
                    workspace.lse,
                    dynamic_dims=(() if cuda_graph_decode else (1,)),
                    dynamic_strides=(() if cuda_graph_decode else (0,)),
                ),
                None
                if merge_regular_decode_graph
                else _tensor_meta_key(workspace.total_num_rows_ptr),
                persistent_ctas,
                merge_direct_grid,
                merge_regular_decode_graph,
                False,
                merge_analytic_laguna_decode_graph,
                pair_bf16_merge_partial_loads,
            )
            merge_spec = KernelCompileSpec.from_key(
                "attention.paged.merge",
                9,
                merge_cache_key,
                labels=(
                    "tmp_output",
                    "tmp_lse",
                    "merge_indptr",
                    "cache_seqlens",
                    "kv_chunk_size_ptr",
                    "output",
                    "lse",
                    "total_num_rows_ptr",
                    "persistent_ctas",
                    "direct_grid",
                    "regular_decode_graph",
                    "analytic_laguna_verify_graph",
                    "analytic_laguna_decode_graph",
                    "pair_bf16_partial_loads",
                ),
            )
        if _compile_only:
            compiled_merge = b12x_compile(
                merge_kernel, *merge_args, stream, compile_spec=merge_spec
            )
            if _collect_launchers is not None:
                _collect_launchers.merge_kernel = merge_kernel
                _collect_launchers.merge = compiled_merge
        elif _launchers is not None:
            if _launchers.merge is None:
                raise RuntimeError("unprepared paged merge launcher")
            run_compiled(_launchers.merge, (*merge_args, stream))
        else:
            b12x_launch(
                merge_kernel,
                compile_spec=merge_spec,
                compile_args=(*merge_args, stream),
                runtime_args=(*merge_args, stream),
            )

    if capacity_extend_graph:
        active_rows = int(q.shape[0])
        return (
            output.narrow(0, 0, active_rows),
            workspace.lse.narrow(1, 0, active_rows).transpose(0, 1),
        )
    return output.narrow(0, 0, plan.total_q), workspace.current_lse_view()


def materialize_paged_resources(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plan: object,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    launchers: _PagedLaunchers,
) -> Mapping[str, object]:
    """Materialize durable resources from raw K/V storage before scratch binds."""
    kernel = launchers.forward_kernel
    plane_tma_descs: tuple[torch.Tensor | None, ...]
    if kernel is not None and (
        getattr(kernel, "use_paged_kv_tma_raw_desc_issue", False)
        or getattr(kernel, "use_paged_kv_tma_fp8_raw_issue", False)
    ):
        k_desc = _encode_plane_tma_descriptors(
            k_cache,
            plane_cols=kernel.kv_tma_plane_head_dim,
            tile_rows=kernel.stage_tile_rows,
        )
        v_desc = _encode_plane_tma_descriptors(
            v_cache,
            plane_cols=kernel.kv_tma_plane_head_dim,
            tile_rows=kernel.stage_tile_rows,
        )
        plane_tma_descs = (
            k_desc,
            v_desc,
            _descriptor_row_ptrs(k_desc),
            _descriptor_row_ptrs(v_desc),
        )
    else:
        ptrs = torch.zeros(
            int(plan.num_kv_heads), dtype=torch.int64, device=k_cache.device
        )
        plane_tma_descs = (None, None, ptrs, ptrs)
    optional = {}
    for name, source in (("k_descale", k_descale), ("v_descale", v_descale)):
        if (
            source is not None
            and source.ndim == 2
            and source.shape[1] == 1
            and not source[:, 0].is_contiguous()
        ):
            optional[name] = torch.empty(
                source.shape[0], dtype=source.dtype, device=source.device
            )
    if attention_sink_bias is None:
        optional["attention_sink_bias"] = torch.empty(
            0, dtype=torch.float32, device=k_cache.device
        )
    elif (
        attention_sink_bias.dtype != torch.float32
        or not attention_sink_bias.is_contiguous()
    ):
        optional["attention_sink_bias"] = torch.empty(
            attention_sink_bias.shape, dtype=torch.float32, device=k_cache.device
        )
    return MappingProxyType(
        {"plane_tma_descs": plane_tma_descs, "optional_buffers": optional}
    )


def install_paged_resources(binding: object, resources: Mapping[str, object]) -> None:
    """Install durable resources before scratch binding/scheduler preparation."""
    workspace = getattr(binding, "scratch", None)
    if workspace is None:
        raise TypeError("paged attention binding must expose scratch")
    plane_tma_descs = resources.get("plane_tma_descs")
    if plane_tma_descs is None:
        raise ValueError("paged resources are missing plane_tma_descs")
    install = getattr(workspace, "install_prepared_resources", None)
    if install is None:
        workspace._prepared_plane_tma_descs = plane_tma_descs
    else:
        install(plane_tma_descs=plane_tma_descs)
    workspace._prepared_optional_buffers = resources["optional_buffers"]


def compile_paged_launchers(
    binding: object, *, controls: Mapping[str, str]
) -> _PagedLaunchers:
    """Compile and retain the exact native paged programs for ``binding``."""
    from b12x._lib.compile_plan import compile_only_launches
    from .graph_replay import compile_graph_replay

    launchers = _PagedLaunchers(controls=MappingProxyType(dict(controls)))
    tensor_mode = (
        binding.q.fake_mode if hasattr(binding.q, "fake_mode") else nullcontext()
    )
    with paged_controls(launchers.controls), tensor_mode, compile_only_launches():
        paged_attention_forward(
            binding=binding,
            _compile_only=True,
            _collect_launchers=launchers,
        )
        launchers.metadata = compile_graph_replay(binding)
    if launchers.forward is None:
        raise RuntimeError("paged launch compilation produced no forward program")
    return _attach_paged_programs(launchers)


def run_paged_prepared(
    binding: object, launchers: _PagedLaunchers
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only resident programs retained by ``compile_paged_launchers``."""
    if not isinstance(launchers, _PagedLaunchers):
        raise TypeError("run_paged_prepared requires compiled paged launchers")
    return paged_attention_forward(binding=binding, _launchers=launchers)


def _compile_paged_attention(*, binding) -> None:
    """Prime the programs selected by a heuristic binding."""
    compile_paged_launchers(binding, controls=snapshot_paged_controls())


def clear_paged_caches() -> None:
    """Clear compiled-kernel caches for the primary paged backend."""
    _build_forward_kernel.cache_clear()
    _build_laguna_verify_forward_kernel.cache_clear()
    _build_extend_forward_kernel.cache_clear()
    _build_merge_kernel.cache_clear()
    _build_laguna_verify_merge_kernel.cache_clear()
