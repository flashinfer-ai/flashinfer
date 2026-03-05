"""
cuDNN graph-based fused RMSNorm + SiLU via the OSS engine.

Uses the cuDNN frontend graph API with heur_mode.OPENSOURCE to dispatch to
the Sm100RmsNormSiluEngine (Blackwell). The kernel is JIT-compiled via NVRTC
on first use for each unique (num_tokens, C) shape.

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


@functools.lru_cache(maxsize=64)
def _build_rmsnorm_silu_graph(num_tokens: int, C: int, device_index: int):
    """Build and cache a cuDNN graph for fused RMSNorm + SiLU.

    The graph is cached by (num_tokens, C, device_index) since the kernel
    variant and NVRTC compilation depend on the problem shape.

    First call triggers NVRTC compilation (~2-5 seconds).
    Subsequent calls with the same shape reuse the cached graph.
    """
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
    Z.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    # Build with OPENSOURCE heuristic mode → dispatches to Sm100RmsNormSiluEngine
    graph.build([cudnn.heur_mode.OPENSOURCE])

    # Pre-allocate workspace
    workspace_size = graph.get_workspace_size()

    return graph, X, scale, epsilon, Z, workspace_size


def cudnn_fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute fused RMSNorm + SiLU via the cuDNN OSS engine.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (num_tokens, hidden_size). Must be bf16.
    weight : torch.Tensor
        Weight tensor, shape (hidden_size,). Must be bf16.
    eps : float
        RMSNorm epsilon.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor.

    Returns
    -------
    torch.Tensor
        Output tensor, same shape and dtype as input.
    """
    num_tokens, C = input.shape
    device_index = input.device.index or 0

    graph, X, scale, epsilon, Z, workspace_size = _build_rmsnorm_silu_graph(
        num_tokens, C, device_index
    )

    if out is None:
        out = torch.empty_like(input)

    workspace = torch.empty(workspace_size, device=input.device, dtype=torch.uint8)

    epsilon_cpu = torch.full(
        (1, 1, 1, 1), eps, dtype=torch.float32, device="cpu"
    )

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
