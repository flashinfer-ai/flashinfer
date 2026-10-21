"""W4A16 direct-kernel compilation and launch bindings."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x._lib.compiler import KernelCompileSpec
from b12x.moe._shared.kernels.activations import SWIGLUOAI_UNINTERLEAVE
from b12x._lib.compiler import compile as b12x_compile
import cuda.bindings.driver as cuda
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.utils.pointers import make_ptr
from b12x.moe._shared.kernels.activations import normalize_moe_activation
import os
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.program_cache import register_program_cache
import torch
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _FC2_DIRECT_MIN_EXPERT_CAPACITY,
    _normalize_activation_swiglu_params,
)
from b12x.moe._shared.kernels.w4a16.kernel.direct import MoEMicroKernelW4A16SmallMDirect
from b12x.moe._shared.kernels.w4a16.kernel.types import (
    _W4A16FC2DirectLaunch,
    _W4A16SmallMDirectLaunch,
)


_SMALL_M_DIRECT_CACHE: dict[tuple, _W4A16SmallMDirectLaunch] = {}

_FC2_DIRECT_CACHE: dict[tuple, _W4A16FC2DirectLaunch] = {}

register_program_cache(_SMALL_M_DIRECT_CACHE)

register_program_cache(_FC2_DIRECT_CACHE)


def _small_m_direct_supported(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    swiglu_clamp_after_silu: bool = False,
    swiglu_alpha: float,
    swiglu_beta: float,
    element_dtype: str,
    weight_layout: str,
    w13_layout: str,
    scale_format: str = "e4m3_k16",
    expert_map: torch.Tensor | None = None,
) -> bool:
    if os.environ.get("B12X_W4A16_SMALL_M_DIRECT", "1") == "0":
        return False
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if activation == SWIGLUOAI_UNINTERLEAVE and swiglu_limit is None:
        return False
    return (
        element_dtype == "bf16"
        # Packed serving weights use the W4A16 TC-decode path for small M.
        # The direct micro kernel is kept limited to native ModelOpt weights.
        and weight_layout == "modelopt"
        and w13_layout in ("w13", "w31")
        and not bool(apply_router_weight_on_input)
        and (swiglu_limit is None or activation in ("silu", SWIGLUOAI_UNINTERLEAVE))
        and expert_map is None
        and MoEMicroKernelW4A16SmallMDirect.is_supported(
            m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topk=topk,
            num_experts=num_experts,
            scale_format=scale_format,
        )
    )


def _small_m_direct_host_barrier_reset_enabled() -> bool:
    """Whether to clear reusable small-M grid-barrier state before launch.

    The micro-kernel barrier resets its arrival counter and advances its epoch
    before releasing the grid, so completed launches can safely reuse both
    scalars.  Keep the historical host reset as the default while the
    persistent-epoch path is evaluated independently and end-to-end.
    """
    return os.environ.get("B12X_W4A16_SMALL_M_HOST_BARRIER_RESET", "1") != "0"


def _compile_w4a16_small_m_direct(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    topk_ids_dtype: torch.dtype,
    device: torch.device | None,
    scale_format: str = "e4m3_k16",
    swiglu_limit: float | None = None,
    swiglu_clamp_after_silu: bool = False,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    w13_layout: str = "w13",
) -> _W4A16SmallMDirectLaunch:
    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("small-M W4A16 direct path requires int32/int64 topk_ids")
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    cache_key = (
        "w4a16_small_m_direct",
        None if device is None else int(device.index or 0),
        int(m),
        int(hidden_size),
        int(intermediate_size),
        int(num_experts),
        int(topk),
        activation,
        bool(fast_math),
        topk_ids_dtype,
        scale_format,
        swiglu_limit,
        swiglu_clamp_after_silu,
        swiglu_alpha,
        swiglu_beta,
        w13_layout,
    )
    cached = _SMALL_M_DIRECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation=activation,
        fast_math=bool(fast_math),
        share_input_across_experts=(int(m) == 1),
        share_expert_scales=True,
        single_token=(int(m) == 1),
        scale_format=scale_format,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        w13_layout=w13_layout,
    )
    kernel.configure(
        int(m),
        int(hidden_size),
        int(intermediate_size),
        int(topk),
        int(num_experts),
        device=device,
    )

    def dummy(dt):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(ids_dtype),
        dummy(cutlass.Float32),
        dummy(cutlass.BFloat16),
        barrier_fake,
        barrier_fake,
        Int32(num_experts),
        Int32(m),
        Int32(kernel.grid_x),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "moe.w4a16.small_m_direct",
            3,
            ("device_index", None if device is None else int(device.index or 0)),
            ("m", int(m)),
            ("hidden_size", int(hidden_size)),
            ("intermediate_size", int(intermediate_size)),
            ("num_experts", int(num_experts)),
            ("topk", int(topk)),
            ("activation", activation),
            ("fast_math", bool(fast_math)),
            ("topk_ids_dtype", str(topk_ids_dtype)),
            ("scale_format", scale_format),
            ("swiglu_limit", swiglu_limit),
            ("swiglu_clamp_after_silu", swiglu_clamp_after_silu),
            ("swiglu_alpha", swiglu_alpha),
            ("swiglu_beta", swiglu_beta),
            ("w13_layout", w13_layout),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _W4A16SmallMDirectLaunch(
        compiled=compiled,
        grid_x=int(kernel.grid_x),
        m=int(m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(num_experts),
        topk=int(topk),
        activation=activation,
        fast_math=bool(fast_math),
        topk_ids_dtype=topk_ids_dtype,
    )
    _SMALL_M_DIRECT_CACHE[cache_key] = launch
    return launch


def _w4a16_fc2_direct_expert_capacity(num_experts: int) -> int:
    """Return the compile-time capacity bucket for FC2-only packed FP4 endpoints.

    The resident endpoint count is runtime artifact data. Keeping a bounded
    power-of-two capacity in the CuTe tensor type avoids one compilation per
    layer without changing route-ID based address arithmetic.
    """

    num_experts = int(num_experts)
    if num_experts < 1:
        raise ValueError("W4A16 FC2-only execution requires at least one expert")
    return max(
        _FC2_DIRECT_MIN_EXPERT_CAPACITY,
        1 << (num_experts - 1).bit_length(),
    )


def _compile_w4a16_fc2_direct(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk_ids_dtype: torch.dtype,
    device: torch.device,
) -> _W4A16FC2DirectLaunch:
    """Compile the runtime-M native-MXFP4 FC2-only microkernel."""

    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("W4A16 FC2-only route IDs must be int32 or int64")
    # Expert count changes the fake tensor extent used by CuTe, but it does
    # not change the FC2 address arithmetic: every access is based on the
    # runtime route ID and fixed per-expert strides.  Compiling the exact
    # resident count made a model with matrix-granular endpoints JIT one
    # otherwise-identical kernel per layer.  Use a capacity bucket instead;
    # route validation still guarantees IDs are within the actual tensors.
    expert_capacity = _w4a16_fc2_direct_expert_capacity(num_experts)
    cache_key = (
        "w4a16_fc2_direct",
        int(device.index or 0),
        int(hidden_size),
        int(intermediate_size),
        int(expert_capacity),
        topk_ids_dtype,
    )
    cached = _FC2_DIRECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation="silu",
        fast_math=False,
        share_input_across_experts=False,
        share_expert_scales=True,
        single_token=False,
        scale_format="e8m0_k32",
        compile_time_phase=2,
    )
    # m=2 selects the runtime-M body. FC2 does not retain token activations in
    # registers, so the compiled loop remains valid above the fused kernel's
    # ordinary eight-token ceiling.
    kernel.configure(
        2,
        int(hidden_size),
        int(intermediate_size),
        1,
        int(expert_capacity),
        device=device,
    )

    def dummy(dt):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(ids_dtype),
        dummy(cutlass.Float32),
        dummy(cutlass.BFloat16),
        barrier_fake,
        barrier_fake,
        Int32(expert_capacity),
        Int32(2),
        Int32(kernel.grid_x),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "moe.w4a16.fc2_direct",
            5,
            ("device_index", int(device.index or 0)),
            ("hidden_size", int(hidden_size)),
            ("intermediate_size", int(intermediate_size)),
            ("expert_capacity", int(expert_capacity)),
            ("topk_ids_dtype", str(topk_ids_dtype)),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _W4A16FC2DirectLaunch(
        compiled=compiled,
        grid_x=int(kernel.grid_x),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(expert_capacity),
        topk_ids_dtype=topk_ids_dtype,
    )
    _FC2_DIRECT_CACHE[cache_key] = launch
    return launch


def _w4a16_small_m_direct_launch_flat(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
    *,
    launcher: _W4A16SmallMDirectLaunch | None = None,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    direct_launch = (
        launcher
        if launcher is not None
        else _compile_w4a16_small_m_direct(
            m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
            activation=activation,
            fast_math=bool(fast_math),
            topk_ids_dtype=topk_ids.dtype,
            scale_format=scale_format,
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            w13_layout=w13_layout,
            device=a_input.device,
        )
    )

    def ptr(dt, tensor: torch.Tensor):
        return make_ptr(dt, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int64 if topk_ids.dtype == torch.int64 else cutlass.Int32
    direct_launch.compiled(
        ptr(cutlass.BFloat16, a_input),
        ptr(cutlass.Uint8, w13_u8),
        ptr(cutlass.Uint8, w13_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w13_global_scale),
        ptr(cutlass.Float32, w13_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Uint32, inter_u32.view(torch.uint32)),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(ids_dtype, topk_ids),
        ptr(cutlass.Float32, topk_weights),
        ptr(cutlass.BFloat16, output),
        barrier_count,
        barrier_epoch,
        Int32(num_experts),
        Int32(m),
        Int32(direct_launch.grid_x),
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_small_m_direct_launch",
    mutates_args="unknown",
)
def _w4a16_small_m_direct_launch_op(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
) -> None:
    _w4a16_small_m_direct_launch_flat(
        a_input=a_input,
        w13_u8=w13_u8,
        w13_scale_u8=w13_scale_u8,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        inter_u32=inter_u32,
        w2_u8=w2_u8,
        w2_scale_u8=w2_scale_u8,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        fast_math=fast_math,
        scale_format=scale_format,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        w13_layout=w13_layout,
        stream_int=stream_int,
    )


@_w4a16_small_m_direct_launch_op.register_fake
def _w4a16_small_m_direct_launch_fake(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
) -> None:
    return None


def _w4a16_fc2_direct_launch_flat(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
    *,
    launcher=None,
) -> None:
    launch = (
        launcher
        if launcher is not None
        else _compile_w4a16_fc2_direct(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk_ids_dtype=route_ids.dtype,
            device=intermediate.device,
        )
    )

    def ptr(dt, tensor: torch.Tensor):
        return make_ptr(dt, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int64 if route_ids.dtype == torch.int64 else cutlass.Int32
    launch.compiled(
        ptr(cutlass.BFloat16, intermediate),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Uint32, intermediate.view(torch.uint32)),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(ids_dtype, route_ids),
        ptr(cutlass.Float32, route_weights),
        ptr(cutlass.BFloat16, output),
        barrier_count,
        barrier_epoch,
        Int32(num_experts),
        Int32(m),
        Int32(launch.grid_x),
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_fc2_direct_launch",
    mutates_args="unknown",
)
def _w4a16_fc2_direct_launch_op(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
) -> None:
    _w4a16_fc2_direct_launch_flat(
        intermediate=intermediate,
        w2_u8=w2_u8,
        w2_scale_u8=w2_scale_u8,
        w2_global_scale=w2_global_scale,
        route_ids=route_ids,
        route_weights=route_weights,
        output=output,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        stream_int=stream_int,
    )


@_w4a16_fc2_direct_launch_op.register_fake
def _w4a16_fc2_direct_launch_fake(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
) -> None:
    return None
