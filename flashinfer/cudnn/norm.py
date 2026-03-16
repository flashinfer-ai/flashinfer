"""
cuDNN graph-based fused RMSNorm + SiLU via the OSS engine.

Uses the cuDNN frontend graph API with heur_mode.OPENSOURCE to dispatch to
the Sm100RmsNormSiluEngine. The kernel is JIT-compiled via NVRTC on first
use for each unique (num_tokens, C, output_dtype) shape.

Supports SM80+ GPUs (Ampere, Ada, Hopper, Blackwell). Performance is
optimized for WAN VAE problem sizes on B200 (SM100); other architectures
and problem sizes use a conservative fallback heuristic.

Output types: bf16, FP8 E4M3 (SM89+), NVFP4 E2M1 (SM100+).
"""

import functools
from typing import Optional
import torch
import cudnn

_TORCH_TO_CUDNN_OUTPUT_DTYPE = {
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    torch.uint8: cudnn.data_type.FP4_E2M1,  # packed NVFP4: 2 elements per byte
}

# torch.float4_e2m1fn_x2 is the native PyTorch packed FP4 dtype (same physical
# layout as uint8, 2 FP4 values per byte).  Accept it as an alias when available.
# See: https://github.com/pytorch/ao/blob/main/torchao/prototype/qat/nvfp4.py
if hasattr(torch, "float4_e2m1fn_x2"):
    _TORCH_TO_CUDNN_OUTPUT_DTYPE[torch.float4_e2m1fn_x2] = cudnn.data_type.FP4_E2M1


@functools.lru_cache(maxsize=128)
def _build_rmsnorm_silu_graph(
    num_tokens: int, C: int, output_cudnn_dtype: int, device_index: int
):
    """Build and cache a cuDNN graph for fused RMSNorm + SiLU.

    The graph is cached by (num_tokens, C, output_dtype, device_index) because
    cuDNN binds tensor dimensions at build time.  The compiled cubin (via
    NVRTC) depends only on C, output_dtype, and GPU arch — not num_tokens.

    First call for a new (C, output_dtype, device) triggers NVRTC compilation
    (~2-5 s).  Subsequent calls with a different num_tokens but the same C
    rebuild only the lightweight graph descriptor, not the cubin.
    """
    # Convert int back to cudnn enum (lru_cache needs hashable args)
    out_dtype = cudnn.data_type(output_cudnn_dtype)

    # Ensure cuDNN builds the graph on the correct device (matters for multi-GPU).
    with torch.cuda.device(device_index):
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

        # Build with OPENSOURCE heuristic mode → dispatches to Sm100RmsNormSiluEngine.
        # SM100 uses sweep-tuned LUT; other SM80+ archs use fallback heuristic.
        graph.build([cudnn.heur_mode.OPENSOURCE])

        workspace_size = graph.get_workspace_size()

    return graph, X, scale, epsilon, Z, workspace_size


def cudnn_fused_rmsnorm_silu(
    input: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Execute fused RMSNorm + SiLU via the cuDNN OSS engine.

    Computes:

    .. math::

        \text{out} = \text{SiLU}\!\Big(\frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}}
        \;\cdot\; \gamma\Big)

    where :math:`x` is the input, :math:`\gamma` is the per-channel scale,
    :math:`\varepsilon` is the epsilon, and
    :math:`\text{SiLU}(y) = y \cdot \sigma(y)`.

    The output dtype is determined by the ``out`` tensor dtype:
      - bf16: standard output
      - float8_e4m3fn: FP8 quantized output (scale=1.0)
      - uint8: NVFP4 E2M1 packed output (2 elements per byte)

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (num_tokens, hidden_size). Must be bf16.
    scale : torch.Tensor
        Per-channel scale (gamma), shape (hidden_size,). Must be bf16.
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

    is_nvfp4 = out_torch_dtype == torch.uint8

    graph, X_desc, scale_desc, eps_desc, Z_desc, workspace_size = (
        _build_rmsnorm_silu_graph(num_tokens, C, int(cudnn_out_dtype), device_index)
    )

    if out is None:
        if is_nvfp4:
            # NVFP4: 2 FP4 values per byte → half the elements
            out = torch.empty(
                num_tokens * C // 2, dtype=torch.uint8, device=input.device
            )
        else:
            out = torch.empty_like(input)

    workspace = torch.empty(workspace_size, device=input.device, dtype=torch.uint8)

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    # For NVFP4, the graph expects the logical FP4 shape but the memory is packed.
    # cuDNN handles the packing internally — we pass the raw buffer.
    out_for_graph = out if is_nvfp4 else out.view(num_tokens, C, 1, 1)

    graph.execute(
        {
            X_desc: input.view(num_tokens, C, 1, 1),
            scale_desc: scale.view(1, C, 1, 1),
            eps_desc: epsilon_cpu,
            Z_desc: out_for_graph,
        },
        workspace,
    )

    return out
