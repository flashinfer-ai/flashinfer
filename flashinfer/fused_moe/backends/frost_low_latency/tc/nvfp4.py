# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed NVFP4 weights with live block scales and grouped 16-bit MMA."""

import cutlass
from cutlass import cute

from .runtime import combine_kernel
from .warp import WarpGroupedProjection


@cute.jit
def launch(
    x,
    w1,
    s1,
    a1,
    w2,
    s2,
    a2,
    ids,
    scores,
    mid,
    routes,
    output,
    group: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    stream,
):
    top_k = cute.size(ids, mode=[1])
    first = WarpGroupedProjection(
        True, group, stages, tile_k, top_k, activation, gated, parameters
    )
    second = WarpGroupedProjection(False, group, stages, tile_k, top_k)
    first(w1, x, ids, mid, stream, a1, None, s1)
    second(w2, mid, ids, routes, stream, a2, None, s2)
    combine_kernel(routes, ids, scores, output, cute.size(w1, mode=[0]), narrow).launch(
        grid=(cute.ceil_div(cute.size(output), 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


def execute(inputs, config, tactic):
    """Bind current tensors; compiled entries retain no runtime pointers."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from ....api import QuantFormat, SwiGLU
    from .....utils import get_compute_capability
    from ..fma.activation_config import lower_activation
    from ..runtime import _COMPILED, _COMPILE_LOCK

    capability = get_compute_capability(inputs[0].device)
    choices = (
        (
            ("warp_mma", 4, 2, 128),
            ("warp_mma", 8, 4, 128),
            ("warp_mma", 4, 2, 256),
            ("warp_mma", 8, 2, 256),
        )
        if capability == (12, 0)
        and config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
        else (("warp_mma", 4, 2, 256), ("warp_mma", 8, 2, 256))
        if capability == (10, 3)
        and config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
        else ()
    )
    if (
        config.quant.weight is not QuantFormat.NVFP4
        or config.quant.output is not config.quant.activation
        or config.activation != SwiGLU()
        or not config.finalize.do_finalize
        or config.routing.top_k != 10
        or config.experts.intermediate_size != 512
        or len(inputs) != 13
        or inputs[0].shape != (8, 2048)
        or tactic not in choices
    ):
        raise ValueError(
            "Grouped NVFP4 requires SM103/SM120 BF16/FP16 T8 H2048 I512 top-k10 SwiGLU"
        )
    tensors, narrow = inputs[:-1], inputs[-1]
    key = (
        "grouped-nvfp4-fp16-pr-v5",
        tactic,
        config.activation,
        narrow,
        tuple((v.dtype, tuple(v.shape), tuple(v.stride()), v.device) for v in tensors),
    )
    with torch.cuda.device(tensors[0].device):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up frost_low_latency before CUDA Graph capture"
                )
            with _COMPILE_LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    name, parameters = lower_activation(config.activation)
                    compiled = cute.compile(
                        launch,
                        *(from_dlpack(v) for v in tensors),
                        *tactic[1:],
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch sm_{capability[0]}{capability[1]}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
    return tensors[-1]
