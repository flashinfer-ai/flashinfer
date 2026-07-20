# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/merge.py @ 17428af5 (2026-07-01) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Split-axis merge for the active sparse MLA SM120 kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from flashinfer.experimental.sm12x.attention._shared.cute import ops as attention_ops
from flashinfer.experimental.sm12x.attention._shared.workspace import _SPLIT_MAX_CHUNKS
from flashinfer.experimental.sm12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    launch as sm12x_launch,
    tensor_compile_fact,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream

from .decode_math import _exp2_approx_ftz_f32
from .reference import _MLA_GROUP_SIZE, _MLA_NOPE_DIM


_MLA_SCALE_GROUPS = _MLA_NOPE_DIM // _MLA_GROUP_SIZE
_MLA_WARP_THREADS = 32


def _raise_binding_extras(api_name: str, extras: list[str]) -> None:
    raise ValueError(
        f"{api_name} binding owns runtime tensors, scratch, and kernel options; "
        f"do not also pass {', '.join(extras)}"
    )


def _require_bound_arg(value, *, api_name: str, name: str):
    if value is None:
        raise TypeError(f"{api_name} requires {name} or binding")
    return value


@dataclass(frozen=True, kw_only=True)
class SparseMLASplitDecodeMergeBinding:
    tmp_output: torch.Tensor
    tmp_lse: torch.Tensor
    num_chunks_ptr: torch.Tensor
    output: torch.Tensor
    num_chunks: int | None = None
    attn_sink: torch.Tensor | None = None
    scratch: object | None = None

    def run(self) -> None:
        run_sparse_mla_split_decode_merge(binding=self)


def build_sparse_mla_split_decode_merge_binding(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    num_chunks: int | None = None,
    attn_sink: torch.Tensor | None = None,
    scratch: object | None = None,
) -> SparseMLASplitDecodeMergeBinding:
    return SparseMLASplitDecodeMergeBinding(
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        num_chunks_ptr=num_chunks_ptr,
        output=output,
        num_chunks=num_chunks,
        attn_sink=attn_sink,
        scratch=scratch,
    )


def _validate_tensor_storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    min_offset = int(tensor.storage_offset())
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        if extent >= 0:
            max_offset += extent
        else:
            min_offset += extent
    storage_elems = tensor.untyped_storage().nbytes() // tensor.element_size()
    if min_offset < 0 or max_offset >= storage_elems:
        raise ValueError(
            f"{name} view is out of storage bounds: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} storage_offset={int(tensor.storage_offset())} "
            f"storage_elems={storage_elems}"
        )


def _validate_split_control_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.shape != (1,):
        raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.uint32:
        return cutlass.Uint32
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next(
        (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
) -> tuple[tuple[object, ...], tuple[int, ...], str, tuple[str, int | None]]:
    dynamic_dim_set = set(dynamic_dims)
    return (
        tuple(
            DimKey.dynamic() if idx in dynamic_dim_set else int(dim)
            for idx, dim in enumerate(tensor.shape)
        ),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _tensor_compile_key(
    name: str,
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
    dynamic_strides: tuple[int, ...] = (),
) -> tuple[object, ...]:
    return tensor_compile_fact(
        name,
        tensor,
        dynamic_dims=dynamic_dims,
        dynamic_strides=dynamic_strides,
    )


@cute.jit
def _split_output_lane_view(
    tmp_output: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
    out_base: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_output, (q_idx, head_idx, Int32(0), out_base)),
        cute.make_layout(
            (tmp_output.shape[2], 4),
            stride=(tmp_output.stride[2], 1),
        ),
    )


@cute.jit
def _split_lse_head_view(
    tmp_lse: cute.Tensor,
    q_idx: Int32,
    head_idx: Int32,
) -> cute.Tensor:
    return cute.make_tensor(
        attention_ops.elem_pointer(tmp_lse, (q_idx, head_idx, Int32(0))),
        cute.make_layout(
            (tmp_lse.shape[2],),
            stride=(tmp_lse.stride[2],),
        ),
    )


class SparseMLASplitDecodeMergeKernel:
    """Reduce normalized chunk partials into the final decode output."""

    def __init__(self, static_num_chunks: int | None = None):
        self.static_num_chunks = static_num_chunks

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_idx, group_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_idx = Int32(head_idx)
        group_idx = Int32(group_idx)

        acc = cute.make_rmem_tensor((4,), Float32)
        for frag_idx in cutlass.range_constexpr(4):
            acc[frag_idx] = Float32(0.0)

        out_base = group_idx * Int32(_MLA_GROUP_SIZE) + lane * Int32(4)
        tmp_output_lane = _split_output_lane_view(tmp_output, q_idx, head_idx, out_base)
        tmp_lse_head = _split_lse_head_view(tmp_lse, q_idx, head_idx)
        merged_m = Float32(-Float32.inf)
        merged_d = Float32(1.0)
        chunk_idx = Int32(0)
        if cutlass.const_expr(self.static_num_chunks is None):
            num_chunks = Int32(num_chunks_ptr[Int32(0)])
        else:
            num_chunks = Int32(self.static_num_chunks)
        if num_chunks > Int32(_SPLIT_MAX_CHUNKS):
            num_chunks = Int32(_SPLIT_MAX_CHUNKS)

        while chunk_idx < num_chunks and merged_m == Float32(-Float32.inf):
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                acc[0] = Float32(tmp_output_lane[chunk_idx, Int32(0)])
                acc[1] = Float32(tmp_output_lane[chunk_idx, Int32(1)])
                acc[2] = Float32(tmp_output_lane[chunk_idx, Int32(2)])
                acc[3] = Float32(tmp_output_lane[chunk_idx, Int32(3)])
                merged_m = Float32(part_lse)
                merged_d = Float32(1.0)
            chunk_idx += Int32(1)

        while chunk_idx < num_chunks:
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                new_m = attention_ops.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(
                output.element_type
            )
        else:
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(acc[0] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(acc[1] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(acc[2] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(acc[3] * inv_d).to(
                output.element_type
            )


class SparseMLASplitDecodeSinkMergeKernel:
    """Reduce chunk partials and fold a zero-value attention sink into softmax."""

    def __init__(self, static_num_chunks: int | None = None):
        self.static_num_chunks = static_num_chunks

    @cute.jit
    def __call__(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            tmp_output,
            tmp_lse,
            num_chunks_ptr,
            attn_sink,
            output,
        ).launch(
            grid=(output.shape[0], output.shape[1], _MLA_SCALE_GROUPS),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tmp_output: cute.Tensor,
        tmp_lse: cute.Tensor,
        num_chunks_ptr: cute.Tensor,
        attn_sink: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_idx, group_idx = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        head_idx = Int32(head_idx)
        group_idx = Int32(group_idx)

        acc = cute.make_rmem_tensor((4,), Float32)
        for frag_idx in cutlass.range_constexpr(4):
            acc[frag_idx] = Float32(0.0)

        out_base = group_idx * Int32(_MLA_GROUP_SIZE) + lane * Int32(4)
        tmp_output_lane = _split_output_lane_view(tmp_output, q_idx, head_idx, out_base)
        tmp_lse_head = _split_lse_head_view(tmp_lse, q_idx, head_idx)
        merged_m = Float32(-Float32.inf)
        merged_d = Float32(1.0)
        chunk_idx = Int32(0)
        if cutlass.const_expr(self.static_num_chunks is None):
            num_chunks = Int32(num_chunks_ptr[Int32(0)])
        else:
            num_chunks = Int32(self.static_num_chunks)
        if num_chunks > Int32(_SPLIT_MAX_CHUNKS):
            num_chunks = Int32(_SPLIT_MAX_CHUNKS)

        while chunk_idx < num_chunks and merged_m == Float32(-Float32.inf):
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                acc[0] = Float32(tmp_output_lane[chunk_idx, Int32(0)])
                acc[1] = Float32(tmp_output_lane[chunk_idx, Int32(1)])
                acc[2] = Float32(tmp_output_lane[chunk_idx, Int32(2)])
                acc[3] = Float32(tmp_output_lane[chunk_idx, Int32(3)])
                merged_m = Float32(part_lse)
                merged_d = Float32(1.0)
            chunk_idx += Int32(1)

        while chunk_idx < num_chunks:
            part_lse = Float32(tmp_lse_head[chunk_idx])
            if part_lse != Float32(-Float32.inf):
                new_m = attention_ops.fmax(merged_m, part_lse)
                prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
                part_scale = _exp2_approx_ftz_f32(part_lse - new_m)
                merged_d = Float32(merged_d * prev_scale + part_scale)
                acc[0] = Float32(
                    acc[0] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(0)]) * part_scale
                )
                acc[1] = Float32(
                    acc[1] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(1)]) * part_scale
                )
                acc[2] = Float32(
                    acc[2] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(2)]) * part_scale
                )
                acc[3] = Float32(
                    acc[3] * prev_scale
                    + Float32(tmp_output_lane[chunk_idx, Int32(3)]) * part_scale
                )
                merged_m = Float32(new_m)
            chunk_idx += Int32(1)

        if merged_m == Float32(-Float32.inf):
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(0.0).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(0.0).to(
                output.element_type
            )
        else:
            sink_m = Float32(attn_sink[head_idx] * attention_ops.LOG2_E)
            new_m = attention_ops.fmax(merged_m, sink_m)
            prev_scale = _exp2_approx_ftz_f32(merged_m - new_m)
            sink_scale = _exp2_approx_ftz_f32(sink_m - new_m)
            merged_d = Float32(merged_d * prev_scale + sink_scale)
            inv_d = cute.arch.rcp_approx(merged_d)
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(
                acc[0] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(
                acc[1] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(
                acc[2] * prev_scale * inv_d
            ).to(output.element_type)
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(
                acc[3] * prev_scale * inv_d
            ).to(output.element_type)


@lru_cache(maxsize=None)
def _build_sparse_mla_split_merge_kernel(
    static_num_chunks: int | None = None,
) -> SparseMLASplitDecodeMergeKernel:
    return SparseMLASplitDecodeMergeKernel(static_num_chunks)


@lru_cache(maxsize=None)
def _build_sparse_mla_split_sink_merge_kernel(
    static_num_chunks: int | None = None,
) -> SparseMLASplitDecodeSinkMergeKernel:
    return SparseMLASplitDecodeSinkMergeKernel(static_num_chunks)


def clear_sparse_mla_merge_kernel_cache() -> None:
    _build_sparse_mla_split_merge_kernel.cache_clear()
    _build_sparse_mla_split_sink_merge_kernel.cache_clear()


def _sparse_mla_split_decode_merge_flat_launch(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
) -> None:
    static_num_chunks_or_none = (
        int(static_num_chunks) if int(static_num_chunks) > 0 else None
    )
    if not has_attn_sink:
        merge_kernel = _build_sparse_mla_split_merge_kernel(static_num_chunks_or_none)
        merge_args = (
            _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
            _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
            current_cuda_stream(),
        )
        merge_cache_key = (
            _tensor_compile_key(
                "tmp_output",
                contract_tmp_output,
                dynamic_dims=(0, 2),
                dynamic_strides=(2,),
            ),
            _tensor_compile_key(
                "tmp_lse",
                contract_tmp_lse,
                dynamic_dims=(0, 2),
                dynamic_strides=(0, 1),
            ),
            _tensor_meta_key(num_chunks_ptr),
            _tensor_compile_key(
                "output",
                contract_output,
                dynamic_dims=(0,),
            ),
            str(tmp_output.dtype),
            str(output.dtype),
            static_num_chunks_or_none,
        )
        merge_spec = KernelCompileSpec.from_key(
            "attention.mla.merge",
            4,
            merge_cache_key,
            labels=(
                "tmp_output",
                "tmp_lse",
                "num_chunks_ptr",
                "output",
                "tmp_output_dtype",
                "output_dtype",
                "static_num_chunks",
            ),
        )
        sm12x_launch(
            merge_kernel,
            compile_spec=merge_spec,
            compile_args=merge_args,
            runtime_args=merge_args,
        )
        return

    merge_kernel = _build_sparse_mla_split_sink_merge_kernel(static_num_chunks_or_none)
    merge_args = (
        _to_kernel_tensor(tmp_output, _torch_to_cutlass_dtype(tmp_output.dtype)),
        _to_kernel_tensor(tmp_lse, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(num_chunks_ptr, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(attn_sink, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    merge_cache_key = (
        _tensor_compile_key(
            "tmp_output",
            contract_tmp_output,
            dynamic_dims=(0, 2),
            dynamic_strides=(2,),
        ),
        _tensor_compile_key(
            "tmp_lse",
            contract_tmp_lse,
            dynamic_dims=(0, 2),
            dynamic_strides=(0, 1),
        ),
        _tensor_meta_key(num_chunks_ptr),
        _tensor_meta_key(attn_sink),
        _tensor_compile_key(
            "output",
            contract_output,
            dynamic_dims=(0,),
        ),
        str(tmp_output.dtype),
        str(output.dtype),
        "attn_sink",
        static_num_chunks_or_none,
    )
    merge_spec = KernelCompileSpec.from_key(
        "attention.mla.sink_merge",
        4,
        merge_cache_key,
        labels=(
            "tmp_output",
            "tmp_lse",
            "num_chunks_ptr",
            "attn_sink",
            "output",
            "tmp_output_dtype",
            "output_dtype",
            "kind",
            "static_num_chunks",
        ),
    )
    sm12x_launch(
        merge_kernel,
        compile_spec=merge_spec,
        compile_args=merge_args,
        runtime_args=merge_args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::sparse_mla_sm120_split_decode_merge",
    mutates_args=("output",),
)
def _sparse_mla_split_decode_merge_op(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
) -> None:
    _sparse_mla_split_decode_merge_flat_launch(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink,
        contract_tmp_output,
        contract_tmp_lse,
        contract_output,
        static_num_chunks,
        has_attn_sink,
    )


@_sparse_mla_split_decode_merge_op.register_fake
def _sparse_mla_split_decode_merge_fake(
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    attn_sink: torch.Tensor,
    contract_tmp_output: torch.Tensor,
    contract_tmp_lse: torch.Tensor,
    contract_output: torch.Tensor,
    static_num_chunks: int,
    has_attn_sink: bool,
) -> None:
    return None


def run_sparse_mla_split_decode_merge(
    *,
    tmp_output: torch.Tensor | None = None,
    tmp_lse: torch.Tensor | None = None,
    num_chunks_ptr: torch.Tensor | None = None,
    num_chunks: int | None = None,
    output: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    workspace: object | None = None,
    binding: SparseMLASplitDecodeMergeBinding | None = None,
) -> None:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("tmp_output", tmp_output),
                ("tmp_lse", tmp_lse),
                ("num_chunks_ptr", num_chunks_ptr),
                ("num_chunks", num_chunks),
                ("output", output),
                ("attn_sink", attn_sink),
                ("workspace", workspace),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_sparse_mla_split_decode_merge", extras)
        tmp_output = binding.tmp_output
        tmp_lse = binding.tmp_lse
        num_chunks_ptr = binding.num_chunks_ptr
        num_chunks = binding.num_chunks
        output = binding.output
        attn_sink = binding.attn_sink
        workspace = binding.scratch

    tmp_output = _require_bound_arg(
        tmp_output,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_output",
    )
    tmp_lse = _require_bound_arg(
        tmp_lse,
        api_name="run_sparse_mla_split_decode_merge",
        name="tmp_lse",
    )
    num_chunks_ptr = _require_bound_arg(
        num_chunks_ptr,
        api_name="run_sparse_mla_split_decode_merge",
        name="num_chunks_ptr",
    )
    output = _require_bound_arg(
        output,
        api_name="run_sparse_mla_split_decode_merge",
        name="output",
    )

    if tmp_output.device != output.device or tmp_lse.device != output.device:
        raise ValueError("sparse MLA merge tensors must be on the same device")
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}")
    if tmp_output.dtype != output.dtype:
        raise TypeError(
            f"tmp_output dtype {tmp_output.dtype} must match output dtype {output.dtype}"
        )
    if tmp_output.ndim != 4:
        raise ValueError(
            f"tmp_output must have shape [rows, heads, chunks, dim], got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"tmp_lse must have shape [rows, heads, chunks], got {tuple(tmp_lse.shape)}"
        )
    if output.ndim != 3:
        raise ValueError(
            f"output must have shape [rows, heads, dim], got {tuple(output.shape)}"
        )
    if (
        int(tmp_output.shape[0]) < int(output.shape[0])
        or int(tmp_output.shape[1]) < int(output.shape[1])
        or int(tmp_output.shape[3]) < int(output.shape[2])
        or int(tmp_lse.shape[0]) < int(output.shape[0])
        or int(tmp_lse.shape[1]) < int(output.shape[1])
        or int(tmp_lse.shape[2]) < int(tmp_output.shape[2])
    ):
        raise ValueError(
            "sparse MLA merge scratch/output shapes are inconsistent: "
            f"tmp_output={tuple(tmp_output.shape)} tmp_lse={tuple(tmp_lse.shape)} "
            f"output={tuple(output.shape)}"
        )
    _validate_tensor_storage_bounds(tmp_output, name="sparse MLA merge tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="sparse MLA merge tmp_lse")
    _validate_tensor_storage_bounds(output, name="sparse MLA merge output")
    _validate_split_control_tensor(
        num_chunks_ptr,
        name="num_chunks_ptr",
        device=output.device,
    )
    if num_chunks is not None:
        num_chunks = int(num_chunks)
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if num_chunks > min(int(tmp_output.shape[2]), _SPLIT_MAX_CHUNKS):
            raise ValueError(
                "num_chunks exceeds merge scratch capacity: "
                f"{num_chunks} > min({int(tmp_output.shape[2])}, {_SPLIT_MAX_CHUNKS})"
            )
    _cto = getattr(workspace, "_contract_tmp_output", None)
    _ctl = getattr(workspace, "_contract_tmp_lse", None)
    _co = getattr(workspace, "_contract_output", None)

    has_attn_sink = attn_sink is not None
    if has_attn_sink:
        attn_sink = attn_sink.detach()
        if attn_sink.dtype != torch.float32:
            raise ValueError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if attn_sink.device != output.device:
            raise ValueError("attn_sink must be on the same CUDA device as output")
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(output.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(output.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous for the fused merge path")

    torch.ops.flashinfer_sm12x.sparse_mla_sm120_split_decode_merge(
        tmp_output,
        tmp_lse,
        num_chunks_ptr,
        output,
        attn_sink if attn_sink is not None else tmp_lse,
        _cto if _cto is not None else tmp_output,
        _ctl if _ctl is not None else tmp_lse,
        _co if _co is not None else output,
        int(num_chunks or 0),
        bool(has_attn_sink),
    )


__all__ = [
    "SparseMLASplitDecodeMergeBinding",
    "build_sparse_mla_split_decode_merge_binding",
    "clear_sparse_mla_merge_kernel_cache",
    "run_sparse_mla_split_decode_merge",
]
