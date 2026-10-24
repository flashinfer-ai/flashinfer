# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared typed output and launch composition for grouped Tensor Core decode."""

import cutlass
from cutlass import cute

from ..quantization import _mul
from .hopper import HopperGroupedProjection
from .warp import WarpGroupedProjection


@cute.kernel
def combine_kernel(
    routes, ids, scores, output, experts: cutlass.Constexpr, narrow: cutlass.Constexpr
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + thread
    h = cute.size(output, mode=[1])
    k = cute.size(scores, mode=[1])
    if index < cute.size(output):
        token, column = index // h, index % h
        value = cutlass.Float32(0)
        # Match the existing separate FP32 multiply/add and ascending slot order.
        for slot in cutlass.range_constexpr(k):
            # Invalid slots contribute nothing, even with non-finite padding scores.
            expert = ids[token, slot]
            if expert >= 0 and expert < experts:
                score = scores[token, slot].to(cutlass.Float32)
                if cutlass.const_expr(narrow):
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                partial = routes[token * k + slot, column].to(cutlass.Float32)
                value = value + _mul(partial, score)
        output[token, column] = value.to(output.element_type)


@cute.jit
def launch(
    x,
    w1,
    ids,
    mid,
    w2,
    routes,
    scores,
    output,
    group_size: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    stream,
    architecture: cutlass.Constexpr = 90,
):
    top_k = cute.size(ids, mode=[1])
    if cutlass.const_expr(architecture == 90):
        first_hopper = HopperGroupedProjection(
            True, group_size, stages, tile_k, top_k, activation, gated, parameters
        )
        second_hopper = HopperGroupedProjection(
            False, group_size, stages, tile_k, top_k
        )
        first_hopper(w1, x, ids, mid, stream)
        second_hopper(w2, mid, ids, routes, stream)
    else:
        first_warp = WarpGroupedProjection(
            True, group_size, stages, tile_k, top_k, activation, gated, parameters
        )
        second_warp = WarpGroupedProjection(False, group_size, stages, tile_k, top_k)
        first_warp(w1, x, ids, mid, stream)
        second_warp(w2, mid, ids, routes, stream)
    if cutlass.const_expr(finalize):
        combine_kernel(
            routes, ids, scores, output, cute.size(w1, mode=[0]), narrow
        ).launch(
            grid=(cute.ceil_div(cute.size(output), 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


_TACTICS = tuple(
    ("wgmma", group, stages, tile_k)
    for group in (2, 4, 8)
    for stages in (2, 4, 8)
    for tile_k in (64, 128)
) + tuple(
    ("warp_mma", group, stages, tile_k)
    for group in (2, 4, 8)
    for tile_k in (64, 128)
    for stages in ((2, 4) if tile_k == 128 else (2, 4, 8))
)


def execute(inputs, config, tactic):
    """Bind live buffers to a compiled projection pair."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from ..fma.activation_config import lower_activation
    from ..runtime import _COMPILED, _COMPILE_LOCK
    from .....utils import get_compute_capability

    if tactic not in _TACTICS:
        raise ValueError(
            "Grouped decode requires a valid architecture/group/stage/K tactic"
        )
    x, ids, scores, w1, w2, mid, routes, output, narrow = inputs
    architecture = 90 if tactic[0] == "wgmma" else 120
    expected = (9, 0) if architecture == 90 else (12, 0)
    if get_compute_capability(x.device) != expected:
        raise ValueError(f"Grouped {tactic[0]} decode requires SM{architecture}")
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or w1.dtype != x.dtype
        or w2.dtype != x.dtype
    ):
        raise ValueError(
            "Grouped TC decode requires matching BF16/FP16 inputs and weights"
        )
    if x.shape[0] > 8 or ids.shape[1] > 32 or mid.shape[1] % 128 or x.shape[1] % 128:
        raise ValueError("Grouped TC decode requires T<=8, top-k<=32 and aligned H/I")
    group_size, stages, tile_k = tactic[1:]
    finalize = config.finalize.do_finalize
    tensors = (x, w1, ids, mid, w2, routes, scores, output)
    key = (
        "grouped-tc-ffi-v3",
        architecture,
        tuple((v.dtype, tuple(v.shape), tuple(v.stride()), v.device) for v in tensors),
        config.activation,
        tactic,
        narrow,
        finalize,
    )
    with torch.cuda.device(x.device):
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
                        group_size,
                        stages,
                        tile_k,
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        finalize,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        architecture,
                        options=f"--gpu-arch sm_{architecture}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
        if finalize:
            return output
        mapping = torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
        return [routes, scores, mapping]
