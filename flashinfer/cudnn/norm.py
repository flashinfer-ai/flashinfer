"""
cuDNN graph-based fused RMSNorm + SiLU via the OSS engine.

Uses the cuDNN frontend graph API with heur_mode.OPENSOURCE to dispatch to
the Sm100RmsNormSiluEngine (Blackwell). The kernel is JIT-compiled via NVRTC
on first use for each unique (num_tokens, C, output_dtype) shape.

Supports bf16, FP8 E4M3, and NVFP4 E2M1 output types.

Requires: nvidia-cudnn-frontend with the knam/rmsnorm-silu-oss branch.
"""

import functools
from typing import Optional

import torch

try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    cudnn = None
    CUDNN_AVAILABLE = False


def _is_sm100_plus() -> bool:
    """Check if the current GPU is SM100+ (Blackwell)."""
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major >= 10


# Map torch dtypes to cuDNN data types for the output tensor
_TORCH_TO_CUDNN_OUTPUT_DTYPE = {}
if CUDNN_AVAILABLE:
    _TORCH_TO_CUDNN_OUTPUT_DTYPE = {
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    }


@functools.lru_cache(maxsize=128)
def _build_rmsnorm_silu_graph(
    num_tokens: int, C: int, output_cudnn_dtype: int, device_index: int
):
    """Build and cache a cuDNN graph for fused RMSNorm + SiLU.

    The graph is cached by (num_tokens, C, output_dtype, device_index) since
    the kernel variant and NVRTC compilation depend on the problem shape and
    output type.

    First call triggers NVRTC compilation (~2-5 seconds).
    Subsequent calls with the same key reuse the cached graph.
    """
    # Convert int back to cudnn enum (lru_cache needs hashable args)
    out_dtype = cudnn.data_type(output_cudnn_dtype)

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    X = graph.tensor(
        name="X",
        dim=[num_tokens, C, 1, 1],
        stride=[C, 1, 1, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )

    scale = graph.tensor(
        name="scale",
        dim=[1, C, 1, 1],
        stride=[C, 1, 1, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )

    epsilon = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    Y = graph.rmsnorm(
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=X,
        scale=scale,
        epsilon=epsilon,
    )[0]
    Y.set_dim([num_tokens, C, 1, 1])
    Y.set_stride([C, 1, 1, 1])
    Y.set_data_type(cudnn.data_type.BFLOAT16)

    Z = graph.swish(input=Y, swish_beta=1.0)
    Z.set_output(True).set_data_type(out_dtype)

    # Build with OPENSOURCE heuristic mode → dispatches to Sm100RmsNormSiluEngine
    graph.build([cudnn.heur_mode.OPENSOURCE])

    workspace_size = graph.get_workspace_size()

    return graph, X, scale, epsilon, Z, workspace_size


def cudnn_fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute fused RMSNorm + SiLU via the cuDNN OSS engine.

    The output dtype is determined by the `out` tensor dtype:
      - bf16: standard output
      - float8_e4m3fn: FP8 quantized output (scale=1.0)
      - For NVFP4: use cudnn_fused_rmsnorm_silu_nvfp4() instead

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (num_tokens, hidden_size). Must be bf16.
    weight : torch.Tensor
        Weight tensor, shape (hidden_size,). Must be bf16.
    eps : float
        RMSNorm epsilon.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor. Dtype determines output quantization.

    Returns
    -------
    torch.Tensor
        Output tensor, same shape as input.
    """
    num_tokens, C = input.shape
    device_index = input.device.index or 0

    # Determine output cuDNN dtype
    if out is not None:
        out_torch_dtype = out.dtype
    else:
        out_torch_dtype = input.dtype

    cudnn_out_dtype = _TORCH_TO_CUDNN_OUTPUT_DTYPE.get(out_torch_dtype)
    if cudnn_out_dtype is None:
        raise ValueError(f"Unsupported output dtype: {out_torch_dtype}")

    graph, X, scale, epsilon, Z, workspace_size = _build_rmsnorm_silu_graph(
        num_tokens, C, int(cudnn_out_dtype), device_index
    )

    if out is None:
        out = torch.empty_like(input)

    workspace = torch.empty(workspace_size, device=input.device, dtype=torch.uint8)

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    graph.execute(
        {
            X: input.view(num_tokens, C, 1, 1),
            scale: weight.view(1, C, 1, 1),
            epsilon: epsilon_cpu,
            Z: out.view(num_tokens, C, 1, 1),
        },
        workspace,
    )

    return out
