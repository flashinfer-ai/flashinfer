# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped DeepSeek FP8 with live block scales and shared dynamic quantization."""

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from .hopper import HopperGroupedProjection
from .warp import WarpGroupedProjection
from .runtime import combine_kernel, _TACTICS
from ..fma.deepseek_fp8_kernel import quantize
from ..fma.activation_config import lower_activation

_WEIGHT_FIRST_TACTICS = {("warp_mma", group, 2, 128, True) for group in (2, 4)}


@cute.jit
def launch(
    x,
    qx,
    xs,
    w1,
    s1,
    w2,
    s2,
    ids,
    scores,
    raw,
    mid,
    ms,
    routes,
    out,
    group: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    name: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    architecture: cutlass.Constexpr,
    stream,
    weight_first: cutlass.Constexpr = False,
):
    if cutlass.const_expr(x.element_type == cutlass.BFloat16):
        quantize(x, qx, xs).launch(
            grid=((cute.size(x) // 128 + 7) // 8, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    k = cute.size(ids, mode=[1])
    if cutlass.const_expr(architecture == 90):
        first_hopper = HopperGroupedProjection(
            True, group, stages, tile, k, name, gated, parameters, block_scaled=True
        )
        first_hopper(w1, qx, ids, raw, stream, s1, xs)
    else:
        first_warp = WarpGroupedProjection(
            True,
            group,
            stages,
            tile,
            k,
            name,
            gated,
            parameters,
            block_scaled=True,
            weight_first=weight_first,
        )
        first_warp(w1, qx, ids, raw, stream, s1, xs)
    quantize(raw, mid, ms).launch(
        grid=((cute.size(raw) // 128 + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )
    if cutlass.const_expr(architecture == 90):
        second_hopper = HopperGroupedProjection(
            False, group, stages, tile, k, block_scaled=True
        )
        second_hopper(w2, mid, ids, routes, stream, s2, ms)
    else:
        second_warp = WarpGroupedProjection(
            False, group, stages, tile, k, block_scaled=True, weight_first=weight_first
        )
        second_warp(w2, mid, ids, routes, stream, s2, ms)
    combine_kernel(routes, ids, scores, out, cute.size(w1, mode=[0]), narrow).launch(
        grid=(cute.ceil_div(cute.size(out), 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


def execute(inputs, config, tactic):
    from ....api import QuantFormat, ReLU2, SiLU, SwiGLU
    from ..runtime import _COMPILED, _COMPILE_LOCK
    from .....utils import get_compute_capability

    if (tactic not in _TACTICS and tactic not in _WEIGHT_FIRST_TACTICS) or tactic[
        0
    ] not in ("wgmma", "warp_mma"):
        raise ValueError("Grouped DeepSeek FP8 requires a valid SM90/SM120 tactic")
    if len(inputs) != 15:
        raise ValueError("Grouped DeepSeek FP8 requires canonical block-scale inputs")
    x, qx, xs, w1, s1, w2, s2, ids, scores, raw, mid, ms, routes, out, narrow = inputs
    architecture = 90 if tactic[0] == "wgmma" else 120
    if get_compute_capability(x.device) != (architecture // 10, architecture % 10):
        raise ValueError(f"Grouped {tactic[0]} decode requires SM{architecture}")
    weight_first = len(tactic) == 5
    if weight_first and not (
        architecture == 120
        and config.quant.weight is QuantFormat.DeepSeekFp8
        and config.quant.activation is QuantFormat.DeepSeekFp8
        and config.quant.output is QuantFormat.BF16
        and x.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and x.shape[1] == 2048
        and raw.shape[1] == config.experts.intermediate_size == 512
        and ids.shape[1] == config.routing.top_k == 10
        and config.finalize.do_finalize
        and (
            (x.shape[0] == 4 and config.activation == SwiGLU() and tactic[1] == 2)
            or (
                x.shape[0] == 8
                and config.activation in (ReLU2(), SiLU())
                and tactic[1] == 4
            )
        )
    ):
        raise ValueError(
            "Weight-first block-FP8 decode requires qualified SM120 T4 SwiGLU/group2 "
            "or T8 ReLU2/SiLU/group4, H2048/I512/top-k10 and finalized BF16 output"
        )
    if (
        not config.finalize.do_finalize
        or x.shape[0] > 8
        or ids.shape[1] > 32
        or x.shape[1] % 128
        or raw.shape[1] % 128
    ):
        raise ValueError(
            "Grouped DeepSeek FP8 requires finalized T<=8, top-k<=32 and aligned H/I"
        )
    tensors = tuple(inputs[:14])
    key = (
        "grouped-block-fp8-ffi-v2",
        architecture,
        tuple((v.dtype, tuple(v.shape), tuple(v.stride()), v.device) for v in tensors),
        config.activation,
        tactic,
        narrow,
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
                        *(from_dlpack(t) for t in tensors),
                        *tactic[1:4],
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        architecture,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        **({"weight_first": True} if weight_first else {}),
                        options=f"--gpu-arch sm_{architecture}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
    return out
