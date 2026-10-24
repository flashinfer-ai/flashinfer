# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed four-bit weight bindings to shared FC1/FC2 FMA scheduling."""

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from .activation_config import lower_activation
from .bf16.fma_fc1 import _kernel as fc1
from .bf16.fma_fc2 import _kernel as fc2
from .route_group import _kernel as grouped_fc1


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
    out,
    first: cutlass.Constexpr,
    second: cutlass.Constexpr,
    name: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    native_fp4: cutlass.Constexpr,
    scale_block_size: cutlass.Constexpr,
    integer_weights: cutlass.Constexpr,
    first_values: cutlass.Constexpr,
    second_values: cutlass.Constexpr,
    route_group: cutlass.Constexpr,
    stream,
):
    if cutlass.const_expr(route_group > 1):
        grouped_fc1(
            x,
            w1,
            s1,
            a1,
            ids,
            mid,
            first,
            route_group,
            native_fp4,
            scale_block_size,
            name,
            gated,
            parameters,
        ).launch(
            grid=(cute.size(mid) // (8 * first), 1, 1), block=(256, 1, 1), stream=stream
        )
    else:
        fc1(
            x,
            w1,
            ids,
            mid,
            first,
            name,
            gated,
            parameters,
            a1,
            None,
            s1,
            native_fp4,
            scale_block_size,
            integer_weights,
            first_values,
        ).launch(
            grid=(cute.size(mid) // (8 * first), 1, 1), block=(256, 1, 1), stream=stream
        )
    fc2(
        mid,
        w2,
        ids,
        scores,
        out,
        narrow,
        finalize,
        second,
        a2,
        s2,
        native_fp4,
        scale_block_size,
        integer_weights,
        second_values,
    ).launch(
        grid=(cute.size(ids, mode=[0]) * cute.size(out, mode=[1]) // second, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


def execute(inputs, config, tactic):
    from ..runtime import _COMPILED, _COMPILE_LOCK
    from ....api import QuantFormat

    integer_weights = config.quant.weight is QuantFormat.MXINT4
    if (
        len(tactic) not in (3, 4, 5)
        or tactic[0] != "ffma"
        or tactic[1] not in (1, 2, 4, 8)
        or tactic[2] not in (1, 2, 4, 8)
        or (
            len(tactic) == 4
            and (
                config.quant.weight not in (QuantFormat.NVFP4, QuantFormat.MXFP4)
                or tactic[1] not in (1, 2, 4)
                or tactic[3] not in (2, 4)
            )
        )
        or (len(tactic) == 5 and (tuple(tactic[3:]) not in ((4, 8), (8, 4), (8, 8))))
    ):
        raise ValueError("W4A16 Frost requires supported FMA column/load-width tactics")
    x, w1, s1, a1, w2, s2, a2, ids, scores, mid, routes, output, narrow = inputs
    first, second = tactic[1:3]
    first_values, second_values = tactic[3:] if len(tactic) == 5 else (4, 4)
    route_group = tactic[3] if len(tactic) == 4 else 1
    finalize = config.finalize.do_finalize
    scale_block_size = 16 if config.quant.weight is QuantFormat.NVFP4 else 32
    key = (
        "packed4-weight-only-ffi-v8",
        route_group,
        first_values,
        second_values,
        integer_weights,
        scale_block_size,
        x.dtype,
        output.dtype,
        tuple(x.shape),
        tuple(w1.shape),
        tuple(w2.shape),
        config.routing.top_k,
        config.activation,
        first,
        second,
        scores.dtype,
        narrow,
        finalize,
        x.device,
    )
    tensors = (
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
        output if finalize else routes,
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
                    major, minor = torch.cuda.get_device_capability(x.device)
                    name, parameters = lower_activation(config.activation)
                    compiled = cute.compile(
                        launch,
                        *(from_dlpack(v) for v in tensors),
                        first,
                        second,
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        finalize,
                        major >= 10,
                        scale_block_size,
                        integer_weights,
                        first_values,
                        second_values,
                        route_group,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
        if finalize:
            return output
        return [
            routes,
            scores,
            torch.arange(ids.numel(), device=x.device, dtype=torch.int32),
        ]
