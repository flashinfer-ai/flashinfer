"""W4A16 top-k reduction compilation and launch bindings."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import compile as b12x_compile
import cuda.bindings.driver as cuda
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.utils.pointers import make_ptr
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.program_cache import register_program_cache
from dataclasses import replace
import torch
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _FP32_TOPK_WEIGHTS,
    _cutlass_element_dtype,
)
from b12x.moe._shared.kernels.w4a16.kernel.sum import W4A16TopKSumKernel
from b12x.moe._shared.kernels.w4a16.kernel.types import W4A16TopKSumCompileResult


_SUM_CACHE: dict[tuple, W4A16TopKSumCompileResult] = {}

register_program_cache(_SUM_CACHE)


def compile_w4a16_topk_sum(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str = "bf16",
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    num_experts: int = 0,
    route_num_experts: int = 0,
    route_ids_dtype: torch.dtype = torch.int32,
    use_expert_map: bool = False,
    broadcast_svh: bool = False,
    float32_output: bool = False,
    apply_topk_weights: bool | None = None,
) -> W4A16TopKSumCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("top-k route expert ids must be int32 or int64")
    if apply_topk_weights is None:
        apply_topk_weights = _FP32_TOPK_WEIGHTS and not full_rotation
    apply_topk_weights = bool(apply_topk_weights)
    route_cutlass_dtype = (
        cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    )
    cache_key = (
        "w4a16_topk_sum",
        element_dtype,
        topk,
        hidden_size,
        bool(full_rotation),
        bool(intermediate_hadamard),
        None if full_rotation else int(num_experts),
        None if full_rotation else int(route_num_experts),
        str(route_ids_dtype),
        bool(use_expert_map),
        bool(broadcast_svh),
        bool(float32_output),
        apply_topk_weights,
    )
    cached = _SUM_CACHE.get(cache_key)
    if cached is not None:
        return attach_programs(
            replace(
                cached,
                num_experts=int(num_experts),
                route_num_experts=int(route_num_experts),
            ),
            cached,
        )

    fc2_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    output_dtype = cutlass.Float32 if full_rotation or float32_output else cutlass_dtype
    output_fake = make_ptr(output_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    route_ids_align = 4 if route_ids_dtype == torch.int32 else 8
    route_ids_fake = make_ptr(
        route_cutlass_dtype,
        route_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=route_ids_align,
    )
    expert_map_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    svh_fake = make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16)
    kernel = W4A16TopKSumKernel(
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        num_experts=num_experts,
        route_num_experts=route_num_experts,
        use_expert_map=use_expert_map,
        broadcast_svh=broadcast_svh,
        float32_output=float32_output,
        apply_topk_weights=apply_topk_weights,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        fc2_fake,
        output_fake,
        topk_weights_fake,
        route_ids_fake,
        expert_map_fake,
        svh_fake,
        Int32(num_experts),
        Int32(route_num_experts),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.topk_sum",
            4,
            cache_key,
        ),
    )
    result = W4A16TopKSumCompileResult(
        compiled=compiled,
        m=0,
        topk=topk,
        hidden_size=hidden_size,
        full_rotation=bool(full_rotation),
        intermediate_hadamard=bool(intermediate_hadamard),
        num_experts=int(num_experts),
        route_num_experts=int(route_num_experts),
        route_ids_dtype=route_ids_dtype,
        use_expert_map=bool(use_expert_map),
        broadcast_svh=bool(broadcast_svh),
        apply_topk_weights=apply_topk_weights,
    )
    attach_programs(result, compiled)
    _SUM_CACHE[cache_key] = result
    return result


def _w4a16_topk_sum_launch_flat(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
    *,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    num_experts: int = 0,
    topk_weights: torch.Tensor | None = None,
    route_expert_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    svh_table: torch.Tensor | None = None,
    launcher: W4A16TopKSumCompileResult | None = None,
) -> None:
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    route_ids_dtype = (
        torch.int32 if route_expert_ids is None else route_expert_ids.dtype
    )
    if num_experts and route_expert_ids is None:
        raise ValueError("top-k sum with an expert count requires route expert IDs")
    if launcher is not None and launcher.route_ids_dtype != route_ids_dtype:
        raise ValueError("prepared top-k sum route-ID dtype differs from its binding")
    route_num_experts = 0 if expert_map is None else int(expert_map.numel())
    broadcast_svh = (
        full_rotation and svh_table is not None and svh_table.numel() == hidden_size
    )
    if launcher is not None and launcher.broadcast_svh != broadcast_svh:
        raise ValueError(
            "prepared W4A16 output rotation layout differs from the bound table"
        )
    applies_weights = (
        bool(getattr(launcher, "apply_topk_weights", False))
        if launcher is not None
        else (_FP32_TOPK_WEIGHTS and not full_rotation)
    )
    if applies_weights and topk_weights is None:
        raise RuntimeError(
            "B12X_W4A16_FP32_TOPK_WEIGHTS: this top-k sum applies router weights "
            "but none were passed"
        )
    sum_kernel = launcher or compile_w4a16_topk_sum(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        num_experts=num_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=expert_map is not None,
        broadcast_svh=broadcast_svh,
        float32_output=output.dtype == torch.float32,
    )
    dummy_addr = output.data_ptr()
    weights_addr = dummy_addr if topk_weights is None else topk_weights.data_ptr()
    route_ids_addr = (
        dummy_addr if route_expert_ids is None else route_expert_ids.data_ptr()
    )
    expert_map_addr = dummy_addr if expert_map is None else expert_map.data_ptr()
    svh_addr = dummy_addr if svh_table is None else svh_table.data_ptr()
    route_cutlass_dtype = (
        cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    )
    route_align = 4 if route_ids_dtype == torch.int32 else 8
    sum_kernel.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            fc2_out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32
            if full_rotation or output.dtype == torch.float32
            else _cutlass_element_dtype(element_dtype),
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            weights_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            route_cutlass_dtype,
            route_ids_addr,
            cute.AddressSpace.gmem,
            assumed_align=route_align,
        ),
        make_ptr(
            cutlass.Int32,
            expert_map_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float16,
            svh_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(num_experts),
        Int32(route_num_experts),
        m,
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_topk_sum_launch",
    mutates_args=("output",),
)
def _w4a16_topk_sum_launch_op(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    route_expert_ids: torch.Tensor,
    num_experts: int,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
) -> None:
    _w4a16_topk_sum_launch_flat(
        fc2_out=fc2_out,
        output=output,
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        stream_int=stream_int,
        route_expert_ids=route_expert_ids,
        num_experts=num_experts,
    )


@_w4a16_topk_sum_launch_op.register_fake
def _w4a16_topk_sum_launch_fake(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    route_expert_ids: torch.Tensor,
    num_experts: int,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
) -> None:
    return None
