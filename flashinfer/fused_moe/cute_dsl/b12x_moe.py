"""
B12x fused MoE API for SM120/SM121.

Provides high-level APIs for running Mixture of Experts (MoE)
computations using b12x CuTe DSL kernels on Blackwell GeForce GPUs.

The b12x kernels take bf16 input and fuse quantization + routing +
FC1 + activation + FC2 + scatter in a single kernel launch.
Supports SiLU (gated, SwiGLU) and ReLU2 (non-gated, Nemotron-Super)
activations.

Two APIs are provided:

1. **Functional API** (`b12x_fused_moe`):
   Simple function call. Workspace is cached internally.

2. **Wrapper API** (`B12xMoEWrapper`):
   Class-based API with pre-allocated buffers for CUDA graph compatibility.

Example (Functional API):
    >>> from flashinfer import b12x_fused_moe
    >>> output = b12x_fused_moe(
    ...     x=hidden_states_bf16,
    ...     w1_weight=w1_fp4, w1_weight_sf=w1_sf, w1_alpha=w1_alpha,
    ...     fc2_input_scale=fc2_scale,
    ...     w2_weight=w2_fp4, w2_weight_sf=w2_sf, w2_alpha=w2_alpha,
    ...     token_selected_experts=topk_ids, token_final_scales=topk_weights,
    ...     num_experts=256, top_k=8,
    ... )

Example (Wrapper API with CUDA Graph):
    >>> from flashinfer import B12xMoEWrapper
    >>> moe = B12xMoEWrapper(
    ...     num_experts=256, top_k=8, hidden_size=4096,
    ...     intermediate_size=14336, use_cuda_graph=True,
    ... )
    >>> output = moe.run(x=hidden_states_bf16, ...)
"""

from typing import Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...utils import supported_compute_capability


@supported_compute_capability([120, 121])
@flashinfer_api
def b12x_fused_moe(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    num_local_experts: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
) -> torch.Tensor:
    """Run fused MoE on SM120/SM121 using b12x CuTe DSL kernels.

    The kernel takes bf16 input and fuses quantization + routing +
    FC1 + activation + FC2 + scatter in a single launch.
    Automatically selects micro (decode), static, or dynamic backend
    based on routed row count.

    Args:
        x: Input activations [num_tokens, hidden_size], bf16.
        w1_weight: FC1 weights, FP4 packed.
            Gated (SiLU): [E, 2*intermediate_size, hidden_size//2].
            Non-gated (ReLU2): [E, intermediate_size, hidden_size//2].
        w1_weight_sf: Scale factors for w1_weight.
        w2_weight: FC2 weights [E, hidden_size, intermediate_size//2], FP4.
        w2_weight_sf: Scale factors for w2_weight.
        token_selected_experts: Expert assignments [num_tokens, top_k].
        token_final_scales: Routing weights [num_tokens, top_k].
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        w1_alpha: Per-expert global scale for FC1.
        w2_alpha: Per-expert global scale for FC2.
        fc2_input_scale: Global scale for FC2 input quantization.
        num_local_experts: Local experts for EP. Default: num_experts.
        output: Pre-allocated output buffer [num_tokens, hidden_size], bf16.
        output_dtype: Output data type. Only torch.bfloat16 is currently
            supported. Default: torch.bfloat16.
        activation: Activation function — "silu" (gated/SwiGLU) or
            "relu2" (non-gated/Nemotron-Super). Default: "silu".

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    from ...jit.cpp_ext import get_cuda_version

    if get_cuda_version().major < 13:
        raise ValueError(
            "b12x fused MoE requires CUDA 13 or later. "
            f"Current CUDA version: {get_cuda_version()}."
        )

    # SM12x kernels hardcode BF16 for scatter_output (see Phase 0 zero-init
    # in moe_{static,micro,dynamic}_kernel.py). Other dtypes will fail the
    # kernel tensor binding.
    if output_dtype != torch.bfloat16:
        raise ValueError(
            f"b12x fused MoE only supports output_dtype=torch.bfloat16, "
            f"got {output_dtype}."
        )
    if output is not None and output.dtype != torch.bfloat16:
        raise ValueError(
            f"b12x fused MoE only supports bf16 output buffers, "
            f"got output.dtype={output.dtype}."
        )

    if num_local_experts is None:
        num_local_experts = num_experts

    num_tokens = token_selected_experts.size(0)
    hidden_size = x.size(1)

    if output is None:
        output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    from .blackwell_sm12x.moe_dispatch import launch_sm120_moe

    return launch_sm120_moe(
        a=x,
        topk_ids=token_selected_experts,
        topk_weights=token_final_scales,
        w1_weight=w1_weight,
        w1_weight_sf=w1_weight_sf,
        w1_alpha=w1_alpha,
        fc2_input_scale=fc2_input_scale,
        w2_weight=w2_weight,
        w2_weight_sf=w2_weight_sf,
        w2_alpha=w2_alpha,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        scatter_output=output,
        activation=activation,
    )


class B12xMoEWrapper:
    """B12x fused MoE wrapper for SM120/SM121 with CUDA graph support.

    Pre-allocates workspace buffers for CUDA graph compatibility.
    Automatically selects micro/static/dynamic backend per call.

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        hidden_size: Hidden dimension size.
        intermediate_size: Intermediate size.
        use_cuda_graph: Pre-allocate buffers for CUDA graph compatibility.
        max_num_tokens: Maximum tokens (only for use_cuda_graph=True).
        num_local_experts: Local experts for EP. Default: num_experts.
        output_dtype: Output data type. Only torch.bfloat16 is currently
            supported. Default: torch.bfloat16.
        device: Device for buffer allocation. Default: "cuda".
        activation: Activation function — "silu" or "relu2". Default: "silu".

    Example:
        >>> moe = B12xMoEWrapper(num_experts=256, top_k=8, ...)
        >>> output = moe.run(x=hidden_states_bf16, ...)
    """

    @supported_compute_capability([120, 121])
    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        use_cuda_graph: bool = False,
        max_num_tokens: int = 4096,
        num_local_experts: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        activation: str = "silu",
    ):
        from ...jit.cpp_ext import get_cuda_version

        if get_cuda_version().major < 13:
            raise ValueError(
                "b12x fused MoE requires CUDA 13 or later. "
                f"Current CUDA version: {get_cuda_version()}."
            )

        # SM12x kernels hardcode BF16 for scatter_output.
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"b12x fused MoE only supports output_dtype=torch.bfloat16, "
                f"got {output_dtype}."
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_cuda_graph = use_cuda_graph
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts or num_experts
        self.output_dtype = output_dtype
        self.device = device
        self.activation = activation

        # Pre-allocated objects. Both workspace slots may be populated so
        # run() can pick per-call; without this, the backend would be locked
        # to whichever workspace was allocated at init time.
        self._static_workspace: object = None
        self._dynamic_workspace: object = None
        self._weight_views: object = None
        self._weight_key: Optional[Tuple] = None
        self._moe_output: Optional[torch.Tensor] = None

        if use_cuda_graph:
            self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Pre-allocate buffers for CUDA graph compatibility."""
        from .blackwell_sm12x.moe_dispatch import (
            allocate_sm120_static_workspace,
            allocate_sm120_dynamic_workspace,
            select_sm120_moe_backend,
            _STATIC_COMPACT_CUTOVER_PAIRS,
        )

        max_routed_rows = self.max_num_tokens * self.top_k

        # Allocate a dynamic workspace alongside the static one when
        # max_num_tokens is large enough to cross the cutover. This lets
        # run() adapt backend per call rather than locking at init. The
        # dynamic kernel indexes row_counts/expert_write_rows with topk_ids
        # (sized by num_local_experts), so it requires num_local == num_experts.
        needs_dynamic = (
            select_sm120_moe_backend(
                num_tokens=self.max_num_tokens, num_topk=self.top_k
            )
            == "dynamic"
            and self.num_local_experts == self.num_experts
        )

        # When both workspaces exist, static only serves calls with
        # routed_rows <= cutover; size it accordingly to avoid paying for
        # the full capacity twice.
        static_max_rows = (
            min(max_routed_rows, _STATIC_COMPACT_CUTOVER_PAIRS)
            if needs_dynamic
            else max_routed_rows
        )
        self._static_workspace = allocate_sm120_static_workspace(
            state_E=self.num_local_experts,
            weight_E=self.num_experts,
            max_rows=max(1, static_max_rows),
            k=self.hidden_size,
            n=self.intermediate_size,
            num_topk=self.top_k,
            device=torch.device(self.device),
        )

        if needs_dynamic:
            self._dynamic_workspace = allocate_sm120_dynamic_workspace(
                state_E=self.num_local_experts,
                weight_E=self.num_experts,
                routed_rows=max_routed_rows,
                k=self.hidden_size,
                n=self.intermediate_size,
                num_topk=self.top_k,
                device=torch.device(self.device),
            )

        # Allocated after arch-specific buffers to preserve memory layout
        # that the autotuner's CUDA graph profiling is sensitive to.
        self._moe_output = torch.empty(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.output_dtype,
            device=self.device,
        )

    @flashinfer_api
    def run(
        self,
        x: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        *,
        w1_alpha: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run MoE computation.

        Args:
            x: Input activations [num_tokens, hidden_size], bf16.
            w1_weight: FC1 weights, FP4 packed.
            w1_weight_sf: Scale factors for w1_weight.
            w2_weight: FC2 weights, FP4 packed.
            w2_weight_sf: Scale factors for w2_weight.
            token_selected_experts: Expert assignments [num_tokens, top_k].
            token_final_scales: Routing weights [num_tokens, top_k].
            w1_alpha: Per-expert global scale for FC1.
            w2_alpha: Per-expert global scale for FC2.
            fc2_input_scale: Global scale for FC2 input quantization.

        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        num_tokens = token_selected_experts.size(0)

        if self.use_cuda_graph and num_tokens > self.max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens "
                f"({self.max_num_tokens})"
            )

        if self.use_cuda_graph:
            moe_output = self._moe_output[:num_tokens]
        else:
            moe_output = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.output_dtype,
                device=x.device,
            )

        from .blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
            select_sm120_moe_backend,
            _get_weight_views as _get_sm120_weight_views,
        )

        # Pick the right pre-allocated workspace for this call's token
        # count. launch_sm120_moe otherwise infers backend from workspace
        # type, which would lock us to whichever one was allocated at init.
        workspace = None
        if self.use_cuda_graph:
            if (
                self._dynamic_workspace is not None
                and select_sm120_moe_backend(num_tokens=num_tokens, num_topk=self.top_k)
                == "dynamic"
            ):
                workspace = self._dynamic_workspace
            else:
                workspace = self._static_workspace

        # Cache weight views; invalidate if weight pointers change.
        weight_key = (
            w1_weight.data_ptr(),
            w1_weight_sf.data_ptr(),
            w1_alpha.data_ptr(),
            w2_weight.data_ptr(),
            w2_weight_sf.data_ptr(),
            w2_alpha.data_ptr(),
        )
        if self._weight_views is None or self._weight_key != weight_key:
            self._weight_views = _get_sm120_weight_views(
                w1_fp4=w1_weight,
                w1_blockscale=w1_weight_sf,
                w2_fp4=w2_weight,
                w2_blockscale=w2_weight_sf,
                w1_alphas=w1_alpha,
                w2_alphas=w2_alpha,
                n=self.intermediate_size,
                k=self.hidden_size,
            )
            self._weight_key = weight_key

        return launch_sm120_moe(
            a=x,
            topk_ids=token_selected_experts,
            topk_weights=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            scatter_output=moe_output,
            activation=self.activation,
            _workspace=workspace,
            _weight_views=self._weight_views,
        )
