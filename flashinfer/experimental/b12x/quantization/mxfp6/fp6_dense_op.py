"""Opaque ``torch.custom_op`` wrapper for the FP6 dense linear.

Registers ``b12x::fp6_dense_linear`` so vLLM's ``torch.compile`` path
(Dynamo fullgraph tracing + CUDA-graph capture) treats the already-prepared
native quantize/GEMM launch sequence as one opaque node.  The caller supplies
the session-owned prepared plan; this operation never selects,
compiles, allocates quantizer scratch, or resolves a launcher.

Importing this module performs the registration; the vLLM plugin imports it
while publishing loaded FP6 weights, before model compilation.
"""
from __future__ import annotations

import torch

from b12x.preparation.types import plan_from_handle, require_prepared



@torch.library.custom_op("b12x::fp6_dense_linear", mutates_args=())
def fp6_dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_storage: torch.Tensor,
    global_scale: torch.Tensor,
    fmt: str,
    out_features: int,
    in_features: int,
    plan_handle: int,
    act_fmt: str = "",
) -> torch.Tensor:
    """``y = x @ W.T`` in MX-FP6; ``x`` is ``(M, in_features)`` bf16.

    ``weight`` is :meth:`FP6DenseWeight.gemm_weight`: either the cached
    1-byte-per-code ``(N, K, 1)`` expansion or, for wide-N layers, the raw
    3:4-packed ``(N, 3K/4)`` codes streamed natively by the GEMM (the layout is
    detected from the K extent). ``scale_storage`` is the flat swizzled UE8M0
    scales. ``fmt`` is the weight sub-format; ``act_fmt`` is the runtime
    activation sub-format (empty string -> same as ``fmt``; ``"e4m3"`` for W6A8).
    Returns ``(M, out_features)`` bf16.
    """
    state = require_prepared(plan_from_handle(plan_handle), "quantization.mxfp6", x.device)
    if (state.query.weight_format != fmt
            or state.query.activation_format != (act_fmt or fmt)
            or state.query.out_features != out_features
            or state.query.in_features != in_features):
        raise ValueError("FP6 dense arguments differ from prepared plan")
    return state.run(x.to(torch.bfloat16), weight, scale_storage, global_scale)


@fp6_dense_linear.register_fake
def _fp6_dense_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_storage: torch.Tensor,
    global_scale: torch.Tensor,
    fmt: str,
    out_features: int,
    in_features: int,
    plan_handle: int,
    act_fmt: str = "",
) -> torch.Tensor:
    del weight, scale_storage, global_scale, fmt, in_features, act_fmt, plan_handle
    return x.new_empty((x.shape[0], out_features), dtype=torch.bfloat16)
