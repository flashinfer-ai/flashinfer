# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host launchers for SM100 split MoE with activations on MMA operand B."""

from typing import Any, Dict, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import (
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    get_num_sm,
    make_ptr,
    torch_to_cutlass_dtype,
)
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from flashinfer.utils import get_compute_capability

from ..moe_utils import (
    normalize_cute_dsl_moe_activation_type,
    validate_cute_dsl_moe_situ_config,
)
from .moe_swap_ab_kernel import Sm100BlockScaledGroupedGemmKernel


_CUDA_GRID_Y_MAX = 65_535

_blockscaled_kernel_cache: Dict[Tuple, Any] = {}


def _blockscaled_ptr(tensor, dtype, alignment=16):
    if tensor is None:
        return None
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=alignment
    )


def _run_blockscaled_gemm(
    weight,
    weight_sf,
    activation,
    activation_sf,
    output,
    output_sf,
    alpha,
    norm_const,
    token_mapping,
    expert_idx,
    mn_limit,
    num_tiles,
    token_final_scales,
    per_token_scale,
    num_tokens,
    top_k,
    mma_tiler_mn,
    cluster_shape_mn,
    activation_type,
    swiglu_alpha,
    swiglu_beta,
    swiglu_limit,
    situ_beta,
    situ_linear_beta,
    use_fused_finalize,
    enable_pdl,
    activation_dtype,
    weight_dtype,
    sf_dtype,
    sf_vec_size,
    prepare_input,
    scatter_output,
    per_token_scale_is_routed,
    k_steps,
    overlap_activation_accumulators,
    pdl_signal_at,
    use_clc_scheduler,
    sm_count=None,
    publish_routing_before_signal=False,
    activation_cache_policy: str = "normal",
):
    cache_policies = {
        "normal": 0,
        "evict_first": 0x12F0000000000000,
        "evict_last": 0x14F0000000000000,
    }
    if activation_cache_policy not in cache_policies:
        raise ValueError(
            f"Unsupported block-scaled cache policy: {activation_cache_policy!r}"
        )
    activation_cache_hint = cache_policies[activation_cache_policy]
    if output.dtype == torch.uint8:
        output_dtype = cutlass.Float4E2M1FN
    else:
        output_dtype = torch_to_cutlass_dtype(output.dtype)
    if token_final_scales is None:
        scale_dtype = cutlass.Float32
    else:
        scale_dtype = torch_to_cutlass_dtype(token_final_scales.dtype)
    channels, reduction, experts = (
        weight.shape[1],
        weight.shape[2] * (2 if weight_dtype is cutlass.Float4E2M1FN else 1),
        weight.shape[0],
    )
    routes = token_mapping.numel()
    if not Sm100BlockScaledGroupedGemmKernel.can_implement(
        m=channels,
        n=routes,
        k=reduction,
        output_dtype=output_dtype,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        k_steps=k_steps,
    ):
        raise ValueError(
            "Unsupported split block-scaled GEMM shape, output dtype, or K steps"
        )
    mma_m, route_tile = mma_tiler_mn
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if prepare_input:
        from .moe_swap_ab_prepare import allocate, prepare

        routed, routed_sf = allocate(
            activation, activation_sf, token_mapping.numel(), sf_vec_size, route_tile
        )
        prepare(
            activation,
            activation_sf,
            routed,
            routed_sf,
            token_mapping,
            mn_limit,
            num_tiles,
            route_tile,
            top_k,
            sf_vec_size,
            enable_pdl,
        )
        activation, activation_sf = routed, routed_sf
    pointers = (
        _blockscaled_ptr(weight, weight_dtype, 32),
        _blockscaled_ptr(weight_sf, sf_dtype),
        _blockscaled_ptr(activation, activation_dtype, 32),
        _blockscaled_ptr(activation_sf, sf_dtype),
        _blockscaled_ptr(output, output_dtype, 32),
        _blockscaled_ptr(
            output_sf,
            cutlass.Uint8
            if output_dtype is cutlass.Float8E4M3FN
            and sf_dtype is cutlass.Float8E8M0FNU
            and sf_vec_size == 32
            else sf_dtype,
        ),
        _blockscaled_ptr(alpha, cutlass.Float32),
        _blockscaled_ptr(norm_const, cutlass.Float32),
        _blockscaled_ptr(token_mapping, cutlass.Int32),
        _blockscaled_ptr(expert_idx, cutlass.Int32),
        _blockscaled_ptr(mn_limit, cutlass.Int32),
        _blockscaled_ptr(num_tiles, cutlass.Int32),
        _blockscaled_ptr(token_final_scales, scale_dtype),
        _blockscaled_ptr(per_token_scale, cutlass.Float32),
    )
    device_sms = get_num_sm(activation.device)
    num_sms = device_sms if sm_count is None else sm_count
    # A CLC grid can occupy the whole device, so honor an explicit SM budget
    # with the bounded persistent launch. There is also no cancellation
    # opportunity when the entire allocated work grid fits concurrently.
    # The CLC grid's Y axis is the allocated route-tile capacity.
    work_clusters = ((channels + mma_m - 1) // mma_m) * (routes // route_tile)
    use_clc_scheduler = use_clc_scheduler and (
        num_sms >= device_sms
        and work_clusters > num_sms // cluster_shape_mn[0]
        and routes // route_tile <= _CUDA_GRID_Y_MAX
    )
    key = (
        activation.device.index,
        mma_tiler_mn,
        cluster_shape_mn,
        output_dtype,
        activation_dtype,
        weight_dtype,
        sf_dtype,
        sf_vec_size,
        output_sf is not None,
        norm_const is not None,
        scale_dtype,
        channels,
        reduction,
        experts,
        num_sms,
        top_k,
        scatter_output,
        per_token_scale_is_routed,
        k_steps,
        overlap_activation_accumulators,
        pdl_signal_at,
        use_clc_scheduler,
        publish_routing_before_signal,
        activation_cache_hint,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        situ_beta,
        situ_linear_beta,
        per_token_scale is not None,
        use_fused_finalize,
        enable_pdl,
    )
    tile_k = (
        256
        if activation_dtype is cutlass.Float4E2M1FN
        and weight_dtype is cutlass.Float4E2M1FN
        else 128
    )
    compiled = _blockscaled_kernel_cache.get(key)
    args = (*pointers, channels, routes, reduction, experts, num_tokens)
    if compiled is None:
        kernel = Sm100BlockScaledGroupedGemmKernel(
            mma_tiler_mnk=(mma_m, route_tile, tile_k),
            activation_type=activation_type,
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
            scatter_output=scatter_output,
            per_token_scale_is_routed=per_token_scale_is_routed,
            k_steps=k_steps,
            num_stages=None,
            overlap_activation_accumulators=overlap_activation_accumulators,
            pdl_signal_at=pdl_signal_at,
            use_clc_scheduler=use_clc_scheduler,
            num_tile_info_stages=1,
            prefetch_weights_before_wait=enable_pdl,
            publish_routing_before_signal=publish_routing_before_signal,
            activation_cache_policy=activation_cache_hint,
            generate_sfc=output_sf is not None,
            output_dtype=output_dtype,
            final_scale_dtype=scale_dtype,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_per_token_scale=per_token_scale is not None,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=enable_pdl,
            max_active_clusters=num_sms,
            use_k96=(
                get_compute_capability(activation.device) == (10, 3)
                and activation_dtype is weight_dtype is cutlass.Float4E2M1FN
                and sf_vec_size == 16
            ),
        )
        compiled = cute.compile(kernel.wrapper, *args, stream=stream)
        _blockscaled_kernel_cache[key] = compiled
    # Channels, reduction, and experts already specialize the cache entry.
    # Only routed capacity and token count remain runtime shape arguments.
    compiled(*pointers, routes, num_tokens, stream=stream)


def _check_gemm_inputs(a, b, a_dtype, b_dtype, a_per_token_scale, k_steps):
    assert a.device.type == "cuda", "Input tensors must be on CUDA device"
    assert b.device.type == "cuda", "Input tensors must be on CUDA device"
    arch = get_compute_capability(a.device)
    if arch not in ((10, 0), (10, 3)):
        raise ValueError(
            f"Swap-AB block-scaled MoE requires SM100 or SM103, got SM{arch[0]}{arch[1]}"
        )
    k = a.shape[1] * (2 if a_dtype == "float4_e2m1fn" else 1)
    b_k = b.shape[2] * (2 if b_dtype == "float4_e2m1fn" else 1)
    if b_k != k:
        raise ValueError(f"A and B logical K dimensions must match, got {k} and {b_k}")
    if k_steps is not None and (type(k_steps) is not int or k_steps not in (4, 8)):
        raise ValueError("Blackwell k_steps must be None, 4, or 8")
    if a_per_token_scale is not None:
        if a_per_token_scale.device.type != "cuda":
            raise ValueError("a_per_token_scale must be on CUDA device")
        if a_per_token_scale.dtype != torch.float32:
            raise ValueError("a_per_token_scale must have dtype torch.float32")
        if not a_per_token_scale.is_contiguous():
            raise ValueError("a_per_token_scale must be contiguous")
        if a_per_token_scale.shape != (a.shape[0],):
            raise ValueError(
                f"a_per_token_scale must have shape ({a.shape[0]},), "
                f"got {tuple(a_per_token_scale.shape)}"
            )


def launch_gather_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    token_id_mapping: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    a_per_token_scale: Optional[torch.Tensor] = None,
    topk: int = 8,
    a_dtype: str = "float4_e2m1fn",
    b_dtype: str = "float4_e2m1fn",
    sf_dtype: str = "float8_e4m3fn",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    quantize_output: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
    activation_type: Union[int, ActivationType] = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    gated: bool = True,
    k_steps: Optional[int] = None,
    use_clc_scheduler: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prepare routed input and launch FC1 with the swap-AB weight layout."""
    _check_gemm_inputs(a, b, a_dtype, b_dtype, a_per_token_scale, k_steps)
    normalized_activation_type, expected_gated = normalize_cute_dsl_moe_activation_type(
        activation_type
    )
    if gated != expected_gated:
        raise ValueError(
            f"gated={gated} is inconsistent with activation_type "
            f"{normalized_activation_type!r}"
        )
    validate_cute_dsl_moe_situ_config(
        normalized_activation_type, situ_beta, situ_linear_beta
    )
    if type(use_clc_scheduler) is not bool:
        raise ValueError("use_clc_scheduler must be a boolean")

    n = b.shape[1]
    intermediate_size = n // (2 if gated else 1)
    permuted_m = token_id_mapping.shape[0]
    if n % 128 != 0:
        raise ValueError(f"GEMM1 output dim n={n} must be a multiple of 128.")

    # Output quantization is explicit because Float8E4M3FN can also be used as
    # an ordinary (non-block-scaled) output by other internal configurations.
    generate_sfc = quantize_output
    if generate_sfc:
        if c_dtype not in {"float4_e2m1fn", "float8_e4m3fn"}:
            raise ValueError(
                f"Output scale generation is unsupported for c_dtype={c_dtype}"
            )
        is_mxfp8_output = (
            c_dtype == "float8_e4m3fn"
            and sf_dtype == "float8_e8m0fnu"
            and sf_vec_size == 32
        )
        if not is_mxfp8_output and global_scale is None:
            raise ValueError(
                "global_scale is required for normalized block-scaled output"
            )
    elif out_scale is not None or global_scale is not None:
        raise ValueError(
            "out_scale and global_scale are only supported when quantize_output is True"
        )

    a_dtype_cutlass = get_cutlass_dtype(a_dtype)
    b_dtype_cutlass = get_cutlass_dtype(b_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    c_dtype_cutlass = get_cutlass_dtype(c_dtype)
    if (
        a_dtype_cutlass is not b_dtype_cutlass
        and c_dtype_cutlass is not cutlass.Float8E4M3FN
    ):
        raise ValueError(
            "Mixed activation/weight formats require Float8E4M3FN FC1 output"
        )

    # Create output tensor if not provided
    if out is None:
        if generate_sfc and c_dtype == "float4_e2m1fn":
            # FP4 output: 2 values per byte
            out = torch.empty(
                (permuted_m, intermediate_size // 2),
                dtype=torch.uint8,
                device=a.device,
            )
        else:
            out = torch.empty(
                (permuted_m, intermediate_size),
                dtype=cutlass_to_torch_dtype(c_dtype_cutlass),
                device=a.device,
            )

    # Create output scale tensor if needed and not provided
    if generate_sfc:
        # Scale factor layout for output
        scale_atom_n = sf_vec_size * 4
        scale_rest_n = (intermediate_size + scale_atom_n - 1) // scale_atom_n
        route_tile = mma_tiler_mn[1]
        scale_rest_m = (
            (permuted_m + route_tile - 1) // route_tile * ((route_tile + 127) // 128)
        )
        # MMA-compatible scale factor shape, with complete atoms per route.
        if out_scale is None:
            out_scale = torch.empty(
                (32, 4, scale_rest_m, 4, scale_rest_n, 1),
                dtype=torch.uint8,
                device=a.device,
            )
        elif out_scale.numel() != 512 * scale_rest_m * scale_rest_n:
            raise ValueError("output SF storage must contain complete atoms per route")

    _run_blockscaled_gemm(
        weight=b,
        weight_sf=b_scale,
        activation=a,
        activation_sf=a_scale,
        output=out,
        output_sf=out_scale if generate_sfc else None,
        alpha=alpha,
        norm_const=global_scale,
        token_mapping=token_id_mapping,
        expert_idx=tile_idx_to_expert_idx,
        mn_limit=tile_idx_to_mn_limit,
        num_tiles=num_non_exiting_tiles,
        token_final_scales=None,
        per_token_scale=a_per_token_scale,
        num_tokens=a.shape[0],
        top_k=topk,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        activation_type=normalized_activation_type.value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        use_fused_finalize=False,
        enable_pdl=enable_pdl,
        activation_dtype=a_dtype_cutlass,
        weight_dtype=b_dtype_cutlass,
        sf_dtype=sf_dtype_cutlass,
        sf_vec_size=sf_vec_size,
        prepare_input=True,
        scatter_output=False,
        per_token_scale_is_routed=False,
        k_steps=k_steps,
        overlap_activation_accumulators=None,
        pdl_signal_at="startup",
        publish_routing_before_signal=enable_pdl,
        activation_cache_policy="evict_last",
        use_clc_scheduler=use_clc_scheduler,
        sm_count=sm_count,
    )
    return out, out_scale if generate_sfc else None


def launch_finalize_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales: torch.Tensor,
    a_dtype: str,
    b_dtype: str,
    out: Optional[torch.Tensor] = None,
    a_per_token_scale: Optional[torch.Tensor] = None,
    sf_dtype: str = "float8_e4m3fn",
    out_dtype: str = "bfloat16",
    sf_vec_size: int = 16,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sm_count: Optional[int] = None,
    enable_pdl: bool = True,
    use_fused_finalize: bool = True,
    k_steps: Optional[int] = None,
    use_clc_scheduler: bool = False,
) -> torch.Tensor:
    """Launch FC2 with either expanded-row output or fused atomic finalize."""
    _check_gemm_inputs(a, b, a_dtype, b_dtype, a_per_token_scale, k_steps)
    a_dtype_cutlass = get_cutlass_dtype(a_dtype)
    b_dtype_cutlass = get_cutlass_dtype(b_dtype)
    sf_dtype_cutlass = get_cutlass_dtype(sf_dtype)
    out_dtype_cutlass = get_cutlass_dtype(out_dtype)
    if token_final_scales.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"Unsupported routing weight dtype: {token_final_scales.dtype}")
    if out_dtype_cutlass not in (cutlass.Float32, cutlass.BFloat16, cutlass.Float16):
        raise ValueError(f"Unsupported FC2 output dtype: {out_dtype}")
    if (
        a_dtype_cutlass is not b_dtype_cutlass
        and out_dtype_cutlass is not cutlass.BFloat16
    ):
        raise ValueError("Mixed activation/weight formats require BF16 FC2 output")

    seq_len, topk = token_final_scales.shape
    n = b.shape[1]
    output_rows = seq_len if use_fused_finalize else seq_len * topk
    # Atomic fused finalize requires zero-initialized output.
    if out is None:
        allocator = torch.zeros if use_fused_finalize else torch.empty
        out = allocator(
            (output_rows, n),
            dtype=cutlass_to_torch_dtype(out_dtype_cutlass),
            device=a.device,
        )
    else:
        expected_out_dtype = cutlass_to_torch_dtype(out_dtype_cutlass)
        if out.shape != (output_rows, n):
            raise ValueError(
                f"out must have shape ({output_rows}, {n}), got {tuple(out.shape)}"
            )
        if out.dtype != expected_out_dtype:
            raise TypeError(
                f"out must have dtype {expected_out_dtype}, got {out.dtype}"
            )
        if out.device != a.device:
            raise ValueError(f"out must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    _run_blockscaled_gemm(
        weight=b,
        weight_sf=b_scale,
        activation=a,
        activation_sf=a_scale,
        output=out,
        output_sf=None,
        alpha=alpha,
        norm_const=None,
        token_mapping=permuted_idx_to_expanded_idx,
        expert_idx=tile_idx_to_expert_idx,
        mn_limit=tile_idx_to_mn_limit,
        num_tiles=num_non_exiting_tiles,
        token_final_scales=token_final_scales,
        per_token_scale=a_per_token_scale,
        num_tokens=seq_len,
        top_k=topk,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        activation_type=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=0.0,
        situ_beta=None,
        situ_linear_beta=None,
        use_fused_finalize=use_fused_finalize,
        enable_pdl=enable_pdl,
        activation_dtype=a_dtype_cutlass,
        weight_dtype=b_dtype_cutlass,
        sf_dtype=sf_dtype_cutlass,
        sf_vec_size=sf_vec_size,
        prepare_input=False,
        scatter_output=True,
        per_token_scale_is_routed=True,
        k_steps=k_steps,
        overlap_activation_accumulators=False,
        pdl_signal_at="producer_exit",
        activation_cache_policy="evict_last",
        use_clc_scheduler=use_clc_scheduler,
        sm_count=sm_count,
    )
    return out
