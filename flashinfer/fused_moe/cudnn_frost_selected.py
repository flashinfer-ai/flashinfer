"""Thin public entry points for offline-generated cuDNN Frost MoE kernels."""

from __future__ import annotations

from typing import Any

import torch

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def cudnn_frost_grouped_gemm1_swiglu_workspace_size(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
) -> int:
    """Return workspace bytes required by all matching generated tactics."""

    from ..experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
        workspace_size,
    )

    return workspace_size(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
    )


@flashinfer_experimental_api
def cudnn_frost_grouped_gemm1_swiglu(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor | None = None,
    tactic: Any = -1,
) -> torch.Tensor:
    """Run cuDNN Frost's fused dual grouped GEMM1 and SwiGLU epilogue.

    ``grouped_tokens`` is already sorted/materialized by group. Group ``g`` is
    the row range beginning at ``first_token_offset[g]`` (the last range ends
    at ``S``) and uses expert ``g % E``. Gate/up weights have shape ``[E,N,K]``.
    The operation is ``silu(tokens @ gate.T) * (tokens @ up.T) * scale``.
    Routing, token permutation, and GEMM2 are intentionally outside this API.
    """

    from ..experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
        CudnnFrostGroupedGemm1SwiGLURunner,
        workspace_size,
    )
    from ..autotuner import AutoTuner, TuningConfig

    if out is None:
        out = torch.empty(
            (grouped_tokens.shape[0], gate_weights.shape[1]),
            dtype=torch.bfloat16,
            device=grouped_tokens.device,
        )
    inputs = [
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        out,
        workspace,
    ]
    required = workspace_size(*inputs[:6])
    if workspace.dtype != torch.uint8 or workspace.numel() < required:
        raise ValueError(
            f"workspace must be uint8 with at least {required} elements; "
            f"got dtype={workspace.dtype}, numel={workspace.numel()}"
        )
    runner = CudnnFrostGroupedGemm1SwiGLURunner()
    # There are no dynamic profile dimensions, so the autotuner reuses the
    # caller's real tensors, including the content-bearing grouped offsets.
    # This avoids synthesizing invalid routing metadata while still allowing
    # every matching offline-generated cuDNN Frost artifact to race as a tactic.
    runner, tactic = AutoTuner.get().choose_one(
        "cudnn_frost_grouped_gemm1_swiglu",
        [runner],
        TuningConfig(profiling_repeat=30),
        inputs,
    )
    return runner(inputs=inputs, tactic=tactic)


__all__: list[str] = [
    "cudnn_frost_grouped_gemm1_swiglu",
    "cudnn_frost_grouped_gemm1_swiglu_workspace_size",
]
