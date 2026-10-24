# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 bindings to the shared FMA bodies."""

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
    ids,
    mid,
    w2,
    scores,
    out,
    dq1,
    aq2,
    dq2,
    columns: cutlass.Constexpr,
    down_columns: cutlass.Constexpr,
    name: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    route_group: cutlass.Constexpr,
    stream,
):
    if cutlass.const_expr(route_group > 1):
        grouped_fc1(
            x,
            w1,
            None,
            dq1,
            ids,
            mid,
            columns,
            route_group,
            False,
            16,
            name,
            gated,
            parameters,
        ).launch(
            grid=(cute.size(mid) // (8 * columns), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    else:
        fc1(x, w1, ids, mid, columns, name, gated, parameters, dq1, aq2).launch(
            grid=(cute.size(mid) // (8 * columns), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    fc2(mid, w2, ids, scores, out, narrow, finalize, down_columns, dq2).launch(
        grid=(
            cute.size(ids, mode=[0]) * cute.size(out, mode=[1]) // down_columns,
            1,
            1,
        ),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


def execute(inputs, config, tactic):
    from ..runtime import _COMPILED, _COMPILE_LOCK

    if isinstance(tactic, tuple) and len(tactic) == 2:
        tactic = (*tactic, 4)
    valid = [("ffma", a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)]
    if inputs[0].dtype in (torch.bfloat16, torch.float16):
        valid += [
            ("ffma", a, b, group)
            for a in (1, 2, 4)
            for b in (1, 2, 4, 8)
            for group in (2, 4)
        ]
    if tactic not in valid:
        raise ValueError("Per-tensor FP8 Frost requires FMA FC1/FC2 column tactics")
    x, ids, scores, w1, w2, mid, routes, output, narrow, dq1, aq2, dq2 = inputs
    columns, down_columns = tactic[1:3]
    route_group = tactic[3] if len(tactic) == 4 else 1
    finalize = config.finalize.do_finalize
    key = (
        "fp8-per-tensor-route-group-ffi-v4",
        route_group,
        x.dtype,
        mid.dtype,
        output.dtype,
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
        tensors = (
            x,
            w1,
            ids,
            mid,
            w2,
            scores,
            output if finalize else routes,
            dq1,
            aq2,
            dq2,
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
                    name, parameters = lower_activation(config.activation)
                    compiled = cute.compile(
                        launch,
                        *(from_dlpack(v) for v in tensors),
                        columns,
                        down_columns,
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        finalize,
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
