# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic BF16 bindings to the shared legacy-compatible device bodies."""

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver

from ..activation_config import lower_activation
from .fma_fc1 import _kernel as fc1
from .fma_fc2 import _kernel as fc2


@cute.jit
def gate_up(
    x,
    w,
    ids,
    mid,
    outputs: cutlass.Constexpr,
    name: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    stream,
):
    fc1(x, w, ids, mid, outputs, name, gated, parameters).launch(
        grid=(cute.size(mid) // (8 * outputs), 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.jit
def down(
    mid,
    w,
    ids,
    scores,
    out,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    columns: cutlass.Constexpr,
    stream,
):
    fc2(mid, w, ids, scores, out, narrow, finalize, columns).launch(
        grid=(cute.size(ids, mode=[0]) * cute.size(out, mode=[1]) // columns, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


def execute(inputs, config, tactic):
    from ...runtime import _COMPILED, _COMPILE_LOCK

    if isinstance(tactic, tuple) and len(tactic) == 2:
        tactic = (*tactic, 4)  # Preserve the initial FC1-only tactic spelling.
    if tactic not in [
        ("ffma", first, second) for first in (1, 2, 4, 8) for second in (1, 2, 4, 8)
    ]:
        raise ValueError("BF16 Frost requires FMA FC1/FC2 output-column tactics")
    x, ids, scores, w1, w2, mid, routes, output, narrow = inputs
    name, parameters = lower_activation(config.activation)
    columns, down_columns = tactic[1:]
    finalize = config.finalize.do_finalize
    key = (
        "bf16-ffma-v2",
        tuple(x.shape),
        tuple(w1.shape),
        tuple(w2.shape),
        config.experts.intermediate_size,
        config.routing.top_k,
        config.activation,
        columns,
        down_columns,
        scores.dtype,
        narrow,
        finalize,
        x.device,
    )
    with torch.cuda.device(x.device):
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        first = tuple(from_dlpack(v) for v in (x, w1, ids, mid))
        second = tuple(
            from_dlpack(v)
            for v in (mid, w2, ids, scores, output if finalize else routes)
        )
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
                    options = f"--gpu-arch sm_{major}{minor}a"
                    compiled = (
                        cute.compile(
                            gate_up,
                            *first,
                            columns,
                            name,
                            config.activation.is_gated,
                            parameters,
                            stream,
                            options=options,
                        ),
                        cute.compile(
                            down,
                            *second,
                            narrow,
                            finalize,
                            down_columns,
                            stream,
                            options=options,
                        ),
                    )
                    _COMPILED[key] = compiled
        compiled[0](*first, stream)
        compiled[1](*second, stream)
        if finalize:
            return output
        mapping = torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
        return [routes, scores, mapping]
