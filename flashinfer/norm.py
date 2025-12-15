"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from enum import Enum
from typing import List, Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from .jit.norm import gen_norm_module
from .utils import device_support_pdl, register_custom_op, register_fake_op




# cuDNN availability check
CUDNN_AVAILABLE = False
try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    pass
except OSError as e:
    error_msg = str(e).lower()
    is_lib_missing = any(ext in error_msg for ext in [".so", ".dll"])
    if not is_lib_missing:
        raise


@functools.cache
def get_norm_module():
    return gen_norm_module().build_and_load()


@flashinfer_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::layernorm", mutates_args=())
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Layer normalization.
    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Layer Normalized tensor, shape (batch_size, hidden_size). Same dtype as input.
    """
    out = torch.empty_like(input)
    get_norm_module().layernorm(out, input, gemma, beta, eps)
    return out


@register_fake_op("flashinfer::layernorm")
def _layernorm_fake(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, k = input.shape
    return input.new_empty([b, k])


# ============================================================================
# cuDNN Fused RMSNorm + FP4 Quantization
# ============================================================================


class _RMSNormFP4QuantUIDs(Enum):
    """UIDs for cuDNN RMSNorm + FP4 Quantization graph tensors."""

    X_UID = 0
    WEIGHT_UID = 1
    EPSILON_UID = 2
    Y_FP4_UID = 3
    BLOCK_SCALE_UID = 4


# Global cudnn handle for norm operations
_cudnn_norm_handle = None


def _get_cudnn_norm_handle(stream: torch.cuda.Stream):
    """Create and return a cached cuDNN handle for norm operations."""
    global _cudnn_norm_handle
    if _cudnn_norm_handle is None:
        if not CUDNN_AVAILABLE:
            raise RuntimeError(
                "cuDNN is not available. Please install cuDNN to use rmsnorm_fp4quant. "
                "You can install it with: pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend"
            )
        _cudnn_norm_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_norm_handle, stream.cuda_stream)
    return _cudnn_norm_handle


def _check_cudnn_rmsnorm_fp4quant_availability():
    """Check if cuDNN RMSNorm + FP4 quantization is available."""
    if not CUDNN_AVAILABLE:
        raise RuntimeError(
            "cuDNN is not available. Please install cuDNN to use rmsnorm_fp4quant. "
            "You can install it with: pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )

    # Check cuDNN backend version for FP4 block scale quantization support
    backend_version = cudnn.backend_version()
    if backend_version < 90700:
        raise RuntimeError(
            f"cuDNN RMSNorm + FP4 quantization requires backend version >= 90700 (9.7.0), "
            f"found {backend_version}. Please upgrade cuDNN."
        )


def _torch_dtype_to_cudnn_dtype(dtype: torch.dtype):
    """Convert PyTorch dtype to cuDNN data type."""
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    elif dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Only torch.float16 and torch.bfloat16 are supported."
        )


@functools.cache
def _create_rmsnorm_fp4quant_execution_plans(
    batch_size: int,
    hidden_size: int,
    block_size: int,
    input_dtype: torch.dtype,
    device,
):
    """
    Create cuDNN graph with execution plans for fused RMSNorm + FP4 quantization.

    This function creates the graph and execution plans but does NOT build them.
    The plans are built separately in _build_rmsnorm_fp4quant_graph to allow
    for tactic-specific caching.

    The graph performs:
    1. RMSNorm: y = (x / RMS(x)) * weight
    2. FP4 Block Scale Quantization: quantize y to FP4_E2M1 with FP8_E4M3 block scales

    Returns:
        Tuple of (graph, tensor_dict) where tensor_dict contains the tensor objects
        for building variant_pack.

    Note:
        Tensor layout is [batch_size, hidden_size, 1, 1] with row-major (NCHW) strides.
    """
    # Use global cached handle with stream set (matches gemm_base.py pattern)
    stream = torch.cuda.current_stream(device)
    handle = _get_cudnn_norm_handle(stream)

    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    # Convert PyTorch dtype to cuDNN dtype
    cudnn_input_dtype = _torch_dtype_to_cudnn_dtype(input_dtype)

    # Input tensor X: [batch_size, hidden_size, 1, 1] with NCHW strides
    x = graph.tensor(
        name="X",
        dim=[batch_size, hidden_size, 1, 1],
        stride=[hidden_size, 1, 1, 1],
        data_type=cudnn_input_dtype,
    )

    # Weight tensor: [1, hidden_size, 1, 1] (same dtype as input) with NCHW strides
    weight = graph.tensor(
        name="weight",
        dim=[1, hidden_size, 1, 1],
        stride=[hidden_size, 1, 1, 1],
        data_type=cudnn_input_dtype,
    )

    # Epsilon scalar (pass by value)
    epsilon = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    # RMSNorm operation (inference mode)
    y_rmsnorm, inv_variance = graph.rmsnorm(
        name="rmsnorm",
        input=x,
        scale=weight,
        bias=None,
        epsilon=epsilon,
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
    )

    # FP4 Block Scale Quantization
    # axis=1 means quantization along the hidden_size dimension
    y_fp4, block_scale_out = graph.block_scale_quantize(
        y_rmsnorm,
        block_size=block_size,
        axis=1,  # Quantize along hidden dimension
        transpose=False,
        name="fp4_quantize",
    )

    # Set output types
    y_fp4.set_output(True).set_data_type(cudnn.data_type.FP4_E2M1)
    block_scale_out.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)

    # Build the graph and create execution plans (but don't build plans yet)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    # Return graph and tensor objects for variant_pack
    tensor_dict = {
        "x": x,
        "weight": weight,
        "epsilon": epsilon,
        "y_fp4": y_fp4,
        "block_scale": block_scale_out,
    }

    return graph, tensor_dict


@functools.cache
def _build_rmsnorm_fp4quant_graph(
    batch_size: int,
    hidden_size: int,
    block_size: int,
    input_dtype: torch.dtype,
    device,
    tactic: int = -1,
):
    """
    Build execution plan at specific index for the RMSNorm + FP4 quantization graph.

    This function is cached with the tactic parameter to allow different tactics
    to be built and cached separately for autotuning.

    Args:
        batch_size: Batch size
        hidden_size: Hidden dimension size
        block_size: Block size for FP4 quantization
        input_dtype: Input tensor dtype (float16 or bfloat16)
        device: CUDA device
        tactic: Execution plan index to build. -1 means build all plans.

    Returns:
        Tuple of (graph, tensor_dict)
    """
    graph, tensor_dict = _create_rmsnorm_fp4quant_execution_plans(
        batch_size, hidden_size, block_size, input_dtype, device
    )

    graph.check_support()
    if tactic != -1:
        graph.build_plan_at_index(tactic)
    else:
        graph.build_plans()

    return graph, tensor_dict


def _execute_rmsnorm_fp4quant_graph(
    graph,
    tensor_dict,
    x_data: torch.Tensor,
    weight_data: torch.Tensor,
    eps: float,
    y_fp4_data: torch.Tensor,
    block_scale_data: torch.Tensor,
    tactic: int = -1,
):
    """Execute the cached cuDNN RMSNorm + FP4 quantization graph.

    Args:
        graph: The cuDNN graph object
        tensor_dict: Dictionary of tensor objects for building variant_pack
        x_data: Input tensor
        weight_data: Weight tensor
        eps: Epsilon value for RMSNorm
        y_fp4_data: Output tensor for FP4 quantized values
        block_scale_data: Output tensor for block scale factors
        tactic: Execution plan index to use. -1 means use default execution.
    """
    # Prepare epsilon as CPU tensor (pass by value)
    eps_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    # Build variant pack using tensor objects
    variant_pack = {
        tensor_dict["x"]: x_data,
        tensor_dict["weight"]: weight_data,
        tensor_dict["epsilon"]: eps_cpu,
        tensor_dict["y_fp4"]: y_fp4_data,
        tensor_dict["block_scale"]: block_scale_data,
    }

    # Allocate workspace
    workspace_size = graph.get_workspace_size()
    workspace = torch.empty(workspace_size, device=x_data.device, dtype=torch.uint8)

    # Get handle with current stream set (matches gemm_base.py pattern)
    # This ensures the operation runs on the correct CUDA stream
    stream = torch.cuda.current_stream(x_data.device)
    handle = _get_cudnn_norm_handle(stream)

    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=handle)
    else:
        graph.execute_plan_at_index(variant_pack, workspace, tactic, handle=handle)


@functools.cache
def _get_cudnn_rmsnorm_fp4quant_runner():
    """Get a cached TunableRunner for cuDNN RMSNorm + FP4 quantization."""

    class CudnnRMSNormFP4QuantRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            """Return list of valid execution plan indices for autotuning."""
            (
                input_tensor,
                weight,
                y_fp4,
                block_scale,
                eps,
                block_size,
            ) = inputs

            batch_size = input_tensor.shape[0]
            hidden_size = input_tensor.shape[1]

            # Get the graph with execution plans (not built yet)
            graph, _ = _create_rmsnorm_fp4quant_execution_plans(
                batch_size,
                hidden_size,
                block_size,
                input_tensor.dtype,
                input_tensor.device,
            )

            num_plans = graph.get_execution_plan_count()
            return list(range(num_plans))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            """Execute the RMSNorm + FP4 quantization with specified tactic."""
            (
                input_tensor,
                weight,
                y_fp4,
                block_scale,
                eps,
                block_size,
            ) = inputs

            batch_size = input_tensor.shape[0]
            hidden_size = input_tensor.shape[1]
            num_blocks = hidden_size // block_size

            # Reshape tensors for cuDNN 4D format
            x_4d = input_tensor.view(batch_size, hidden_size, 1, 1)
            weight_4d = weight.view(1, hidden_size, 1, 1)
            y_fp4_4d = y_fp4.view(batch_size, hidden_size // 2, 1, 1)
            block_scale_4d = block_scale.view(batch_size, num_blocks, 1, 1)

            # Build the graph with the specified tactic
            graph, tensor_dict = _build_rmsnorm_fp4quant_graph(
                batch_size,
                hidden_size,
                block_size,
                input_tensor.dtype,
                input_tensor.device,
                tactic=tactic,
            )

            # Execute the graph
            _execute_rmsnorm_fp4quant_graph(
                graph=graph,
                tensor_dict=tensor_dict,
                x_data=x_4d,
                weight_data=weight_4d,
                eps=eps,
                y_fp4_data=y_fp4_4d,
                block_scale_data=block_scale_4d,
                tactic=tactic,
            )

    return CudnnRMSNormFP4QuantRunner()


@flashinfer_api
def rmsnorm_fp4quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor,
    block_scale: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = 16,
) -> None:
    r"""Fused RMS normalization with FP4 quantization using cuDNN.

    This function performs RMSNorm followed by FP4 block scale quantization
    in a single fused kernel for optimal performance on Blackwell GPUs.

    The operation is:
        1. RMSNorm: ``y = (input / RMS(input)) * weight``
        2. FP4 Quantization: quantize ``y`` to FP4_E2M1 with FP8_E4M3 block scales

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
        Must be ``torch.float16`` or ``torch.bfloat16``.
    weight : torch.Tensor
        Weight tensor, shape ``(hidden_size,)``.
        Must have the same dtype as input.
    y_fp4 : torch.Tensor
        Output tensor for quantized values in FP4_E2M1 format, packed as uint8.
        Shape must be ``(batch_size, hidden_size // 2)`` for 2D input, or
        ``(batch_size, seq_len, hidden_size // 2)`` for 3D input.
        Two FP4 values are packed per byte. Must be dtype ``torch.uint8``.
    block_scale : torch.Tensor
        Output tensor for block scale factors in FP8_E4M3 format.
        Shape must be ``(batch_size, hidden_size // block_size)`` for 2D input, or
        ``(batch_size, seq_len, hidden_size // block_size)`` for 3D input.
        Must be dtype ``torch.float8_e4m3fn``.
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    block_size : int
        Block size for FP4 quantization. Default is ``16`` (standard for NVFP4).
        The hidden_size must be divisible by block_size.

    Raises
    ------
    RuntimeError
        If cuDNN is not available or the cuDNN version is too old.
    ValueError
        If the input dtype is not float16 or bfloat16, if input and weight
        dtypes don't match, or if hidden_size is not divisible by block_size.

    Notes
    -----
    - This function requires cuDNN >= 9.7.0 and a Blackwell GPU (compute capability >= 100).
    - The FP4_E2M1 format uses 4 bits per value with 2 exponent bits and 1 mantissa bit.
    - The block scale is computed as ``max_abs_in_block / 6.0`` (where 6.0 is FP4_E2M1_MAX),
      then rounded to FP8_E4M3 format.
    - Autotuning is supported: the function will automatically select the best cuDNN
      execution plan when autotuning is enabled.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.norm import rmsnorm_fp4quant
    >>> # Create input tensors
    >>> batch_size, hidden_size, block_size = 4, 128, 16
    >>> x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float16)
    >>> weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    >>> # Allocate output tensors
    >>> y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    >>> block_scale = torch.empty(batch_size, hidden_size // block_size, device="cuda", dtype=torch.float8_e4m3fn)
    >>> # Fused RMSNorm + FP4 quantization
    >>> rmsnorm_fp4quant(x, weight, y_fp4, block_scale)
    >>> y_fp4.shape
    torch.Size([4, 64])
    >>> block_scale.shape
    torch.Size([4, 8])
    """
    _check_cudnn_rmsnorm_fp4quant_availability()

    # Validate input dtype
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported input dtype: {input.dtype}. "
            f"Only torch.float16 and torch.bfloat16 are supported."
        )

    if input.dtype != weight.dtype:
        raise ValueError(
            f"Input and weight must have the same dtype. "
            f"Got input: {input.dtype}, weight: {weight.dtype}"
        )

    # Handle input shape
    if input.ndim == 2:
        batch_size, hidden_size = input.shape
    elif input.ndim == 3:
        batch_size, seq_len, hidden_size = input.shape
        # Flatten to 2D for processing
        input = input.view(batch_size * seq_len, hidden_size)
        batch_size = batch_size * seq_len
    else:
        raise ValueError(
            f"Input must be 2D or 3D tensor, got {input.ndim}D tensor with shape {input.shape}"
        )

    # Validate dimensions
    if hidden_size % block_size != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by block_size ({block_size})"
        )

    if hidden_size % 2 != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by 2 for FP4 packing"
        )

    # Flatten input to 2D for the runner
    input_2d = input.view(batch_size, hidden_size)

    # Use AutoTuner for execution plan selection
    runners = [_get_cudnn_rmsnorm_fp4quant_runner()]
    tuner = AutoTuner.get()

    # Package inputs for the runner
    # Note: eps and block_size are passed as Python values (not tensors)
    inputs = [
        input_2d,
        weight,
        y_fp4,
        block_scale,
        eps,
        block_size,
    ]

    runner, tactic = tuner.choose_one(
        "rmsnorm_fp4quant",
        runners,
        TuningConfig(),
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)
