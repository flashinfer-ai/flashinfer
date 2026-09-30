"""Capture-safe direct-route compact W4A8 materialized micro path."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.w4a8_compact_projection import (
    W4A8CompactMicroProjectionKernel,
)
from b12x.moe._shared.kernels.w4a8_compact_activation import (
    W4A8CompactMicroActivationKernel,
)
from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel


_ALIGN = 256


def _ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _align_up(x: int, alignment: int = _ALIGN) -> int:
    return _ceil_div(x, alignment) * alignment


@program_cache
def _layout(
    max_tokens: int, num_topk: int, k: int, n: int
) -> dict[str, tuple[int, int]]:
    """Return aligned byte ranges for the caller-owned micro workspace."""
    cap = int(max_tokens)
    topk = int(num_topk)
    if cap < 1 or topk < 1 or k < 1 or n < 1:
        raise ValueError("max_tokens, num_topk, k, and n must be positive")
    if k % 128 or n < 64 or n % 64:
        raise ValueError(
            "compact W4A8 micro requires K divisible by 128 and N divisible by 64"
        )
    rows = cap * topk
    n_tiles = _ceil_div(n, 128)
    n_padded = n_tiles * 128
    regions = (
        ("a_values", cap * k),
        ("a_scales", cap * (k // 32)),
        ("a_mma_scales", _ceil_div(cap, 128) * _ceil_div(k, 128) * 512),
        ("projections", cap * topk * 2 * n * 2),
        ("intermediate", rows * n_padded + rows * n_tiles * 4),
        ("route_output", cap * topk * k * 2),
    )
    offset = 0
    result: dict[str, tuple[int, int]] = {}
    for name, size in regions:
        offset = _align_up(offset)
        result[name] = (offset, int(size))
        offset += int(size)
    result["total"] = (0, _align_up(offset))
    return result


def micro_scratch_nbytes(max_tokens: int, k: int, n: int, num_topk: int) -> int:
    """Exact 256-byte-aligned workspace requirement for the compact micro path."""
    return _layout(max_tokens, num_topk, k, n)["total"][1]


def _ptr(dtype: object, tensor: torch.Tensor, *, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


class _DirectW4A8CompactLaunch:
    def __init__(
        self,
        *,
        max_tokens: int,
        num_topk: int,
        k: int,
        n: int,
        experts: int,
        input_scale_count: int,
        down_scale_count: int,
        swiglu_limit: float | None,
        fast_math: bool,
    ):
        self.capacity = max_tokens
        self.topk = num_topk
        self.k = k
        self.n = n
        self.experts = experts
        self.input_scale_count = input_scale_count
        self.down_scale_count = down_scale_count
        self.projection = W4A8CompactMicroProjectionKernel(k, n, num_topk)
        self.activation = W4A8CompactMicroActivationKernel(
            n,
            experts,
            swiglu_limit=swiglu_limit,
            fast_math=fast_math,
        )
        self.phase2 = W4A8MaterializedPhase2Kernel(
            source_tile_m=1,
            deterministic_output=True,
            direct_routes=True,
            n64_repacked=self.n % 128 == 64,
            n64_tail=self.n % 128 == 64,
        )

    @cute.jit
    def __call__(
        self,
        packed_a_ptr: cute.Pointer,
        a_scales_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        w13_scales_ptr: cute.Pointer,
        intermediate_ptr: cute.Pointer,
        projections_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        alpha1_ptr: cute.Pointer,
        input_scale_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_scales_ptr: cute.Pointer,
        route_output_ptr: cute.Pointer,
        alpha2_ptr: cute.Pointer,
        down_scale_ptr: cute.Pointer,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        pairs = self.capacity * self.topk
        rows = pairs
        packed_a = cute.make_tensor(
            packed_a_ptr, cute.make_layout((self.capacity * self.k,))
        )
        a_scales = cute.make_tensor(
            a_scales_ptr, cute.make_layout((self.capacity * self.k // 32,))
        )
        n_tiles = (self.n + 127) // 128
        n_padded = n_tiles * 128
        w13 = cute.make_tensor(
            w13_ptr,
            cute.make_layout((self.experts * self.n * self.k // 4,)),
        )
        w13_scales = cute.make_tensor(
            w13_scales_ptr,
            cute.make_layout((self.experts * self.n * self.k // 64,)),
        )
        w2 = cute.make_tensor(
            w2_ptr,
            cute.make_layout((self.experts * self.k * self.n // 8,)),
        )
        w2_scales = cute.make_tensor(
            w2_scales_ptr,
            cute.make_layout((self.experts * self.k * self.n // 128,)),
        )
        intermediate = cute.make_tensor(
            intermediate_ptr,
            cute.make_layout((rows * (n_padded + n_tiles * 4) // 4,)),
        )
        projections = cute.make_tensor(
            projections_ptr,
            cute.make_ordered_layout((pairs, 2 * self.n), order=(1, 0)),
        )
        # Direct kernels only use this tensor's bounded shape, never its data.
        token_map = cute.make_tensor(intermediate_ptr, cute.make_layout((rows,)))
        unused_metadata = cute.make_tensor(intermediate_ptr, cute.make_layout((1,)))
        token_weights = cute.make_tensor(token_weights_ptr, cute.make_layout((pairs,)))
        topk_ids = cute.make_tensor(topk_ids_ptr, cute.make_layout((pairs,)))
        alpha1 = cute.make_tensor(alpha1_ptr, cute.make_layout((self.experts,)))
        input_scale = cute.make_tensor(
            input_scale_ptr, cute.make_layout((self.input_scale_count,))
        )
        alpha2 = cute.make_tensor(alpha2_ptr, cute.make_layout((self.experts,)))
        down_scale = cute.make_tensor(
            down_scale_ptr, cute.make_layout((self.down_scale_count,))
        )
        route_output = cute.make_tensor(
            route_output_ptr,
            cute.make_ordered_layout((pairs, self.k), order=(1, 0)),
        )
        self.projection(
            packed_a,
            a_scales,
            w13,
            w13_scales,
            projections,
            topk_ids,
            alpha1,
            input_scale,
            num_pairs,
            max_active_clusters,
            stream,
        )
        self.activation(
            projections,
            intermediate,
            topk_ids,
            num_pairs,
            stream,
        )
        self.phase2(
            intermediate,
            w2,
            w2_scales,
            route_output,
            token_map,
            token_weights,
            topk_ids,
            unused_metadata,
            unused_metadata,
            alpha2,
            down_scale,
            packed_a,
            Int32(n_tiles),
            Int32(self.k // 256),
            max_active_clusters,
            num_pairs,
            stream,
        )


@program_cache
def _compiled_direct_w4a8_compact(
    device_index: int,
    max_tokens: int,
    num_topk: int,
    k: int,
    n: int,
    experts: int,
    ids_dtype: torch.dtype,
    input_scale_count: int,
    down_scale_count: int,
    swiglu_limit: float | None,
    fast_math: bool,
):
    launch = _DirectW4A8CompactLaunch(
        max_tokens=max_tokens,
        num_topk=num_topk,
        k=k,
        n=n,
        experts=experts,
        input_scale_count=input_scale_count,
        down_scale_count=down_scale_count,
        swiglu_limit=swiglu_limit,
        fast_math=fast_math,
    )
    ids_type = cutlass.Int32 if ids_dtype == torch.int32 else cutlass.Int64

    def dummy(dtype, align=16):
        return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)

    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=(
            device_index,
            max_tokens,
            num_topk,
            k,
            n,
            experts,
            ids_dtype,
            input_scale_count,
            down_scale_count,
            swiglu_limit,
            fast_math,
        ),
    )
    return b12x_compile(
        launch,
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint32),
        dummy(cutlass.BFloat16),
        dummy(cutlass.Float32, 4),
        dummy(ids_type, 4 if ids_dtype == torch.int32 else 8),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint32),
        dummy(cutlass.BFloat16),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        Int32(1),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a8.compact_micro",
            2,
            (
                device_index,
                max_tokens,
                num_topk,
                k,
                n,
                experts,
                str(ids_dtype),
                input_scale_count,
                down_scale_count,
                swiglu_limit,
                fast_math,
            ),
        ),
    )


def _as_bytes(scratch: torch.Tensor) -> torch.Tensor:
    if scratch.device.type != "cuda" or not scratch.is_contiguous():
        raise ValueError("scratch must be a contiguous CUDA tensor")
    if scratch.dtype not in (torch.uint8, torch.float32):
        raise TypeError("scratch must be torch.uint8 or torch.float32")
    return scratch.view(torch.uint8).reshape(-1)


def launch_w4a8_compact_micro(
    *,
    scratch: torch.Tensor,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    input_scale: torch.Tensor,
    down_scale: torch.Tensor,
    max_tokens: int,
    num_topk: int,
    swiglu_limit: float | None,
    fast_math: bool,
    _prepared_kernel=None,
    _prepared_quantize=None,
    max_active_clusters: int | None = None,
) -> torch.Tensor:
    """Run quantized projections, routed activation, and direct FC2 from fixed scratch."""
    if max_active_clusters is not None and max_active_clusters <= 0:
        raise ValueError("max_active_clusters must be positive when set")
    if a.dtype != torch.bfloat16 or a.ndim != 2 or not a.is_contiguous():
        raise ValueError("a must be a contiguous BF16 [tokens, K] CUDA tensor")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32 or int64")
    if topk_weights.dtype != torch.float32:
        raise TypeError("compact W4A8 micro requires FP32 topk_weights")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("top-k tensors must be contiguous")
    cap, k = int(max_tokens), int(a.shape[1])
    num_tokens = int(a.shape[0])
    if num_tokens > cap or topk_ids.numel() < num_tokens * num_topk:
        raise ValueError("runtime routes exceed the planned compact micro capacity")
    experts = int(alpha1.numel())
    if experts < 1 or alpha2.numel() != experts:
        raise ValueError("compact W4A8 alpha tensors must have one value per expert")
    if any(x.numel() not in (1, experts) for x in (input_scale, down_scale)):
        raise ValueError("compact W4A8 scales must be scalar or per-expert")
    w2_bytes = w2.numel() * w2.element_size()
    denom = experts * k
    if k % 256 or denom == 0 or (2 * w2_bytes) % denom:
        raise ValueError("w2 storage does not encode an exact prepared W4A8 layout")
    n = (2 * w2_bytes) // denom
    layout = _layout(cap, num_topk, k, n)
    storage = _as_bytes(scratch)
    if storage.numel() < layout["total"][1]:
        raise ValueError("scratch is smaller than micro_scratch_nbytes")

    def region(name: str) -> torch.Tensor:
        offset, size = layout[name]
        return storage.narrow(0, offset, size)

    values = region("a_values").view(cap, k)
    scale_rows = region("a_scales").view(cap, k // 32)
    scale_mma = region("a_mma_scales")
    intermediate = region("intermediate")
    projections = region("projections")
    route = region("route_output").view(torch.bfloat16).view(cap * num_topk, k)
    if _prepared_quantize is None:
        quantize_mxfp8_rows_cute(a, values, scale_rows, scale_mma, expected_m=cap)
    else:
        _prepared_quantize(a, values, scale_rows, scale_mma)

    compiled = _prepared_kernel or _compiled_direct_w4a8_compact(
        int(a.device.index or 0),
        cap,
        int(num_topk),
        k,
        n,
        experts,
        topk_ids.dtype,
        input_scale.numel(),
        down_scale.numel(),
        swiglu_limit,
        fast_math,
    )
    ids = topk_ids.reshape(-1)
    weights = topk_weights.reshape(-1)
    compiled(
        _ptr(cutlass.Uint8, values),
        _ptr(cutlass.Uint8, scale_rows),
        _ptr(cutlass.Uint32, w13),
        _ptr(cutlass.Uint32, w13_scales),
        _ptr(cutlass.Uint32, intermediate),
        _ptr(cutlass.BFloat16, projections),
        _ptr(cutlass.Float32, weights, align=4),
        _ptr(
            cutlass.Int32 if ids.dtype == torch.int32 else cutlass.Int64,
            ids,
            align=4 if ids.dtype == torch.int32 else 8,
        ),
        _ptr(cutlass.Float32, alpha1),
        _ptr(cutlass.Float32, input_scale),
        _ptr(cutlass.Uint32, w2),
        _ptr(cutlass.Uint32, w2_scales),
        _ptr(cutlass.BFloat16, route),
        _ptr(cutlass.Float32, alpha2),
        _ptr(cutlass.Float32, down_scale),
        num_tokens * int(num_topk),
        0 if max_active_clusters is None else max_active_clusters,
        current_cuda_stream(),
    )
    return route.narrow(0, 0, num_tokens * int(num_topk))


__all__ = ["micro_scratch_nbytes", "launch_w4a8_compact_micro"]
