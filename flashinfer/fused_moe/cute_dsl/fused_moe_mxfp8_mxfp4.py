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
"""High-level MXFP8 activation x MXFP4 weight fused-MoE APIs for SM100.

The mixed path is deliberately separate from :mod:`.fused_moe`: it consumes
linear block-32 E8M0 activation scales, emits an MXFP8 FC1 intermediate, and
does not have NVFP4's ``fc2_input_scale`` argument.
"""

from __future__ import annotations

import functools
import math
import threading
from typing import Any, Dict, Optional, Tuple
import weakref

import torch

from ...api_logging import flashinfer_api
from ...autotuner import AutoTuner
from ...tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from ...trace.templates.moe import (
    cute_dsl_fused_moe_mxfp8_mxfp4_trace,
    cute_dsl_mxfp8_mxfp4_moe_wrapper_run_trace,
)
from ...utils import supported_compute_capability
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4,
)
from .mixed_tuner import (
    ALL_MXFP8_MXFP4_MOE_TACTICS,
    CuteDslFusedMoEMxfp8Mxfp4Runner,
    canonicalize_mxfp8_mxfp4_tactic,
)
from .moe_utils import (
    moe_output_memset_inplace,
    moe_sort,
    normalize_cute_dsl_moe_activation_type,
)


# Per-thread, per-device async-memset resources. These must not be shared
# across host threads: recording a CUDA event overwrites its previous capture,
# so two threads reusing one event can have one thread's `memset_event.wait()`
# satisfied by the other thread's record and start its GEMM2 before its own
# memset retired -- the fused finalize then atomically accumulates into a
# buffer that was never zeroed. The wrapper API serializes its own runs, but
# the functional API is free-threaded, so it gets one set per thread.
_mixed_functional_resources = threading.local()


def _get_mixed_functional_resources(device: torch.device) -> Dict[str, Any]:
    """Return per-thread async-memset resources owned by ``device``."""
    by_device: Optional[Dict[Tuple[str, Optional[int]], Dict[str, Any]]] = getattr(
        _mixed_functional_resources, "by_device", None
    )
    if by_device is None:
        by_device = {}
        _mixed_functional_resources.by_device = by_device
    key = (device.type, device.index)
    resources = by_device.get(key)
    if resources is None:
        with torch.cuda.device(device):
            resources = {
                "main_event": torch.cuda.Event(),
                "memset_event": torch.cuda.Event(),
                "aux_stream": torch.cuda.Stream(device=device),
            }
        by_device[key] = resources
    return resources


def _mxfp8_scale_storage_shape(
    num_rows: int, num_columns: int
) -> Tuple[int, int, int, int, int, int]:
    if num_rows <= 0 or num_columns <= 0:
        raise ValueError("MXFP8 scale-storage dimensions must be positive")
    return (
        32,
        4,
        math.ceil(num_rows / 128),
        4,
        math.ceil(num_columns / 128),
        1,
    )


def _mxfp8_weight_scale_shape_and_strides(
    num_rows: int, num_columns: int, num_groups: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the logical MMA view contract produced by scale conversion."""
    shape = _mxfp8_scale_storage_shape(num_rows, num_columns)[:-1] + (num_groups,)
    m_tiles = shape[2]
    k_tiles = shape[4]
    strides = (
        16,
        4,
        k_tiles * 512,
        1,
        512,
        m_tiles * k_tiles * 512,
    )
    return shape, strides


def _validate_mixed_configuration(
    *,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if num_experts <= 0 or num_local_experts <= 0:
        raise ValueError("num_experts and num_local_experts must be positive")
    if not 0 < top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if local_expert_offset < 0 or (
        local_expert_offset + num_local_experts > num_experts
    ):
        raise ValueError("local expert range must be contained in [0, num_experts)")
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("hidden_size must be a positive multiple of 128")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")


def _validate_mixed_inputs(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    moe_output: Optional[torch.Tensor],
    *,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    gated: bool,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
) -> None:
    if x.device.type != "cuda":
        raise ValueError("MXFP8 x MXFP4 fused MoE inputs must be CUDA tensors")
    device_tensors: Tuple[Tuple[str, torch.Tensor], ...] = (
        ("x_sf", x_sf),
        ("token_selected_experts", token_selected_experts),
        ("token_final_scales", token_final_scales),
        ("w1_weight", w1_weight),
        ("w1_weight_sf", w1_weight_sf),
        ("w1_alpha", w1_alpha),
        ("w2_weight", w2_weight),
        ("w2_weight_sf", w2_weight_sf),
        ("w2_alpha", w2_alpha),
    )
    if moe_output is not None:
        device_tensors += (("moe_output", moe_output),)
    for name, tensor in device_tensors:
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")

    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError("x must contain at least one token")
    inferred_hidden_size = x.shape[1]
    if hidden_size is None:
        hidden_size = inferred_hidden_size
    if inferred_hidden_size != hidden_size:
        raise ValueError(
            f"x must have hidden size {hidden_size}, got {inferred_hidden_size}"
        )
    if w2_weight.ndim != 3:
        raise ValueError("w2_weight must be a 3D grouped packed-weight tensor")
    if intermediate_size is None:
        intermediate_size = w2_weight.shape[2] * 2
    _validate_mixed_configuration(
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    num_tokens = x.shape[0]
    expected_w1_rows = intermediate_size * (2 if gated else 1)
    expected_shapes = {
        "x_sf": (num_tokens, hidden_size // 32),
        "token_selected_experts": (num_tokens, top_k),
        "token_final_scales": (num_tokens, top_k),
        "w1_weight": (
            num_local_experts,
            expected_w1_rows,
            hidden_size // 2,
        ),
        "w2_weight": (
            num_local_experts,
            hidden_size,
            intermediate_size // 2,
        ),
        "w1_alpha": (num_local_experts,),
        "w2_alpha": (num_local_experts,),
    }
    tensors_by_name = dict(device_tensors)
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors_by_name[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {actual_shape}"
            )

    for name in (
        "x",
        "x_sf",
        "token_selected_experts",
        "token_final_scales",
        "w1_weight",
        "w1_alpha",
        "w2_weight",
        "w2_alpha",
    ):
        tensor = x if name == "x" else tensors_by_name[name]
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    for name, rows, columns in (
        ("w1_weight_sf", expected_w1_rows, hidden_size),
        ("w2_weight_sf", hidden_size, intermediate_size),
    ):
        tensor = tensors_by_name[name]
        expected_shape, expected_strides = _mxfp8_weight_scale_shape_and_strides(
            rows, columns, num_local_experts
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have MMA scale shape {expected_shape}, got "
                f"{tuple(tensor.shape)}"
            )
        if tuple(tensor.stride()) != expected_strides:
            raise ValueError(
                f"{name} must use MMA scale strides {expected_strides}, got "
                f"{tuple(tensor.stride())}"
            )

    if x.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"x must have dtype torch.float8_e4m3fn, got {x.dtype}")
    if x_sf.dtype is not torch.uint8:
        raise TypeError(f"x_sf must have dtype torch.uint8, got {x_sf.dtype}")
    if token_selected_experts.dtype is not torch.int32:
        raise TypeError(
            "token_selected_experts must have dtype torch.int32, got "
            f"{token_selected_experts.dtype}"
        )
    if token_final_scales.dtype is not torch.float32:
        raise TypeError(
            f"token_final_scales must have dtype torch.float32, got "
            f"{token_final_scales.dtype}"
        )
    for name, tensor in (("w1_weight", w1_weight), ("w2_weight", w2_weight)):
        if tensor.dtype is not torch.uint8:
            raise TypeError(f"{name} must contain packed MXFP4 uint8 values")
    for name, tensor in (
        ("w1_weight_sf", w1_weight_sf),
        ("w2_weight_sf", w2_weight_sf),
    ):
        if tensor.dtype is not torch.uint8:
            raise TypeError(f"{name} must contain E8M0 uint8 scale codes")
    for name, tensor in (("w1_alpha", w1_alpha), ("w2_alpha", w2_alpha)):
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
    if moe_output is not None and moe_output.dtype is not torch.bfloat16:
        raise TypeError("moe_output must have dtype torch.bfloat16")
    if moe_output is not None:
        if tuple(moe_output.shape) != (num_tokens, hidden_size):
            raise ValueError(
                f"moe_output must have shape {(num_tokens, hidden_size)}, got "
                f"{tuple(moe_output.shape)}"
            )
        if not moe_output.is_contiguous():
            raise ValueError("moe_output must be contiguous")


def _single_flight_wrapper_run(method):
    """Reject host reentrancy for wrapper-owned reusable storage."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "CuteDslMxfp8Mxfp4MoEWrapper.run is not reentrant; use one "
                "wrapper instance per concurrent caller"
            )
        try:
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, torch.Tensor) and x.device != self.device:
                raise ValueError(
                    f"x must be on wrapper device {self.device}, got {x.device}"
                )
            current_stream = torch.cuda.current_stream(self.device)
            stream_handle = int(current_stream.cuda_stream)
            if self._bound_stream_handle is None:
                self._bound_stream_handle = stream_handle
            elif self._bound_stream_handle != stream_handle:
                raise RuntimeError(
                    "CuteDslMxfp8Mxfp4MoEWrapper is bound to the CUDA stream "
                    "used by its first run; create one wrapper per stream"
                )
            return method(self, *args, **kwargs)
        finally:
            self._run_lock.release()

    return wrapped


def _moe_core_impl_mxfp8_mxfp4(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    tile_size: int = 128,
    gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    output_dtype: torch.dtype = torch.bfloat16,
    moe_sort_buffers: Optional[Dict[str, torch.Tensor]] = None,
    gemm1_out: Optional[torch.Tensor] = None,
    gemm1_out_scale: Optional[torch.Tensor] = None,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    main_event: Optional[torch.cuda.Event] = None,
    memset_event: Optional[torch.cuda.Event] = None,
    use_async_memset: bool = True,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    if output_dtype is not torch.bfloat16:
        raise ValueError("MXFP8 x MXFP4 fused MoE only supports torch.bfloat16 output")
    activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    num_tokens = x.shape[0]
    hidden_size = w2_weight.shape[1]
    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=x.device
        )
    elif tuple(moe_output.shape) != (num_tokens, hidden_size):
        raise ValueError(
            f"moe_output must have shape {(num_tokens, hidden_size)}, got "
            f"{tuple(moe_output.shape)}"
        )

    if use_async_memset and (
        aux_stream is None or main_event is None or memset_event is None
    ):
        resources = _get_mixed_functional_resources(x.device)
        aux_stream = aux_stream or resources["aux_stream"]
        main_event = main_event or resources["main_event"]
        memset_event = memset_event or resources["memset_event"]

    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=tile_size,
        # The routing kernel has a separate PDL implementation contract.  The
        # current validated mixed pipeline enables PDL only for the GEMMs.
        enable_pdl=False,
        **(moe_sort_buffers or {}),
    )

    if use_async_memset:
        assert main_event is not None and aux_stream is not None
        main_event.record()

    intermediate, intermediate_sf = (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4(
            a=x,
            b=w1_weight,
            a_scale=x_sf,
            b_scale=w1_weight_sf,
            alpha=w1_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            token_id_mapping=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            out=gemm1_out,
            out_scale=gemm1_out_scale,
            topk=top_k,
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            gated=gated,
        )
    )

    if use_async_memset:
        assert aux_stream is not None and main_event is not None
        assert memset_event is not None
        moe_output.record_stream(aux_stream)
        with torch.cuda.stream(aux_stream):
            main_event.wait()
            moe_output_memset_inplace(moe_output)
            memset_event.record()
        memset_event.wait()
    else:
        moe_output_memset_inplace(moe_output)

    blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4(
        a=intermediate,
        b=w2_weight,
        a_scale=intermediate_sf,
        b_scale=w2_weight_sf,
        alpha=w2_alpha,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        token_final_scales=token_final_scales,
        out=moe_output,
        mma_tiler_mn=gemm2_mma_tiler_mn,
        cluster_shape_mn=gemm2_cluster_shape_mn,
        enable_pdl=enable_pdl,
    )
    return moe_output


def _cute_dsl_fused_moe_mxfp8_mxfp4_impl(**kwargs: Any) -> torch.Tensor:
    return _moe_core_impl_mxfp8_mxfp4(**kwargs)


class CuteDslMxfp8Mxfp4MoEWrapper:
    """Production wrapper for the MXFP8 x MXFP4 fused-MoE pipeline.

    With ``use_cuda_graph=True`` the wrapper holds persistent CUDA stream and
    event resources, created outside graph capture so they can be reused
    inside it. Workspace itself is not pre-allocated: graph capture records
    allocations made during capture in its private pool, so pre-sizing buffers
    for a maximum batch buys nothing but memory.

    Because the stream and event resources are reused, one wrapper instance is
    not reentrant or safe for concurrent calls. The first ``run`` binds the
    instance to that call's CUDA stream; create one wrapper per stream.
    """

    @supported_compute_capability([100, 103])
    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        # Deprecated; accepted for backwards compatibility but ignored. Graph
        # capture records allocations from its private pool, so there is no
        # workspace to pre-size.
        max_num_tokens: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        local_expert_offset: int = 0,
        use_cuda_graph: bool = False,
        device: str = "cuda",
        enable_pdl: bool = True,
        activation_type: int = ActivationType.Swiglu.value,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    ) -> None:
        """Initialize a reusable mixed-precision fused-MoE runner.

        Parameters
        ----------
        num_experts, top_k : int
            Global expert count and experts selected per token.
        hidden_size, intermediate_size : int
            Model hidden size and per-expert intermediate size.
        max_num_tokens : int, optional
            Deprecated compatibility argument; accepted but ignored.
        num_local_experts : int, optional
            Experts resident on this rank; defaults to ``num_experts``.
        local_expert_offset : int
            Global index of the first local expert.
        use_cuda_graph : bool
            Create persistent stream and event resources for graph capture.
        device : str
            CUDA device used for persistent resources.
        enable_pdl : bool
            Enable programmatic dependent launch in the generated kernels.
        activation_type : int
            Fused activation identifier.
        swiglu_alpha, swiglu_beta, swiglu_limit : float
            SwiGLU activation parameters.
        """
        activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = (
            num_experts if num_local_experts is None else num_local_experts
        )
        self.local_expert_offset = local_expert_offset
        self.use_cuda_graph = use_cuda_graph
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("CuteDslMxfp8Mxfp4MoEWrapper requires a CUDA device")
        device_index = (
            requested_device.index
            if requested_device.index is not None
            else torch.cuda.current_device()
        )
        self.device = torch.device("cuda", device_index)
        _validate_mixed_configuration(
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=local_expert_offset,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.enable_pdl = enable_pdl
        self.activation_type: ActivationType = activation
        self.gated = gated
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        # Persistent CUDA resources for async-memset / GEMM1 overlap. These are
        # created outside graph capture (so they can be reused inside it) when
        # ``use_cuda_graph=True``. When None, ``_moe_core_impl_mxfp8_mxfp4``
        # falls back to the per-thread resources for the current device.
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._main_event: Optional[torch.cuda.Event] = None
        self._memset_event: Optional[torch.cuda.Event] = None
        self._run_lock = threading.Lock()
        self._bound_stream_handle: Optional[int] = None

        wrapper_ref = weakref.ref(self)

        def _forward_with_tactic_weak(*args, **kwargs):
            wrapper = wrapper_ref()
            if wrapper is None:
                raise RuntimeError(
                    "CuteDslMxfp8Mxfp4MoEWrapper was destroyed before invocation"
                )
            return wrapper._forward_with_tactic(*args, **kwargs)

        self._runner = CuteDslFusedMoEMxfp8Mxfp4Runner(
            forward_impl=_forward_with_tactic_weak,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=local_expert_offset,
            output_dtype=torch.bfloat16,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )

        if use_cuda_graph:
            self._aux_stream = torch.cuda.Stream(device=self.device)
            self._main_event = torch.cuda.Event()
            self._memset_event = torch.cuda.Event()

    def _forward_with_tactic(self, **kwargs: Any) -> torch.Tensor:
        return _moe_core_impl_mxfp8_mxfp4(
            aux_stream=self._aux_stream,
            main_event=self._main_event,
            memset_event=self._memset_event,
            use_async_memset=True,
            **kwargs,
        )

    @flashinfer_api(trace=cute_dsl_mxfp8_mxfp4_moe_wrapper_run_trace)
    @_single_flight_wrapper_run
    def run(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        tactic: Optional[Tuple[Any, ...]] = None,
    ) -> torch.Tensor:
        """Execute one mixed-precision fused-MoE forward pass.

        ``x_sf`` is the linear ``[M, H / 32]`` E8M0 byte layout. Weight
        scales must already be in the MMA layout returned by
        ``convert_sf_to_mma_layout(..., sf_vec_size=32)``.

        The returned tensor is freshly allocated and owned by the caller.

        Parameters
        ----------
        x, x_sf : torch.Tensor
            MXFP8 activations and their linear block-32 E8M0 scales.
        token_selected_experts, token_final_scales : torch.Tensor
            Per-token expert indices and routing scales.
        w1_weight, w2_weight : torch.Tensor
            Packed MXFP4 expert weights.
        w1_weight_sf, w2_weight_sf : torch.Tensor
            Block-32 weight scales in MMA layout.
        w1_alpha, w2_alpha : torch.Tensor
            Per-expert dequantization multipliers.
        tactic : tuple, optional
            Explicit kernel tactic; the autotuner selects one when omitted.
        """

        _validate_mixed_inputs(
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            None,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            gated=self.gated,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )
        output = torch.empty(
            (x.shape[0], self.hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        inputs = [
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            output,
        ]
        if tactic is not None:
            return self._runner(inputs, tactic=tactic)

        tuner = AutoTuner.get()
        _, best_tactic = tuner.choose_one(
            f"CuteDslMxfp8Mxfp4MoEWrapper::run::{self.activation_type.name}",
            [self._runner],
            self._runner.tuning_config,
            inputs,
        )
        return self._runner(inputs, tactic=best_tactic)

    def get_valid_tactics(self) -> list:
        return list(ALL_MXFP8_MXFP4_MOE_TACTICS)


@supported_compute_capability([100, 103])
@flashinfer_api(trace=cute_dsl_fused_moe_mxfp8_mxfp4_trace)
def cute_dsl_fused_moe_mxfp8_mxfp4(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    tactic: Optional[Tuple[Any, ...]] = None,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    """Run fused MoE with MXFP8 activations and packed MXFP4 weights.

    No tensor conversion is performed. ``x`` must be E4M3, ``x_sf`` must be
    linear block-32 E8M0 bytes, packed weights must be uint8 E2M1 pairs, and
    weight scales must already use the block-32 MMA E8M0 layout. The output is
    BF16. Unlike the NVFP4 API, this interface has no ``fc2_input_scale``.

    Parameters
    ----------
    x, x_sf : torch.Tensor
        MXFP8 activations and their linear block-32 E8M0 scales.
    token_selected_experts, token_final_scales : torch.Tensor
        Per-token expert indices and routing scales.
    w1_weight, w2_weight : torch.Tensor
        Packed MXFP4 expert weights.
    w1_weight_sf, w2_weight_sf : torch.Tensor
        Block-32 weight scales in MMA layout.
    w1_alpha, w2_alpha : torch.Tensor
        Per-expert dequantization multipliers.
    num_experts, top_k : int
        Global expert count and experts selected per token.
    num_local_experts : int, optional
        Experts resident on this rank; defaults to ``num_experts``.
    local_expert_offset : int
        Global index of the first local expert.
    moe_output : torch.Tensor, optional
        Caller-owned BF16 output tensor.
    aux_stream : torch.cuda.Stream, optional
        Auxiliary CUDA stream used to overlap output clearing with GEMM1.
    tactic : tuple, optional
        Explicit kernel tactic; the autotuner selects one when omitted.
    enable_pdl : bool
        Enable programmatic dependent launch in the generated kernels.
    activation_type : int
        Fused activation identifier.
    swiglu_alpha, swiglu_beta, swiglu_limit : float
        SwiGLU activation parameters.
    """

    activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    num_local_experts = num_experts if num_local_experts is None else num_local_experts
    _validate_mixed_inputs(
        x,
        x_sf,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        moe_output,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        gated=gated,
    )
    if moe_output is None:
        moe_output = torch.empty(
            (x.shape[0], x.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
    runner = CuteDslFusedMoEMxfp8Mxfp4Runner(
        forward_impl=_cute_dsl_fused_moe_mxfp8_mxfp4_impl,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        output_dtype=torch.bfloat16,
        enable_pdl=enable_pdl,
        activation_type=activation.value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    inputs = [
        x,
        x_sf,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        moe_output,
    ]
    if tactic is not None:
        return runner(
            inputs,
            tactic=canonicalize_mxfp8_mxfp4_tactic(tactic),
            aux_stream=aux_stream,
        )
    tuner = AutoTuner.get()
    _, best_tactic = tuner.choose_one(
        f"CuteDslFusedMoE::run_mxfp8_mxfp4::{activation.name}",
        [runner],
        runner.tuning_config,
        inputs,
        aux_stream=aux_stream,
    )
    return runner(inputs, tactic=best_tactic, aux_stream=aux_stream)


__all__ = [
    "cute_dsl_fused_moe_mxfp8_mxfp4",
    "CuteDslMxfp8Mxfp4MoEWrapper",
]
