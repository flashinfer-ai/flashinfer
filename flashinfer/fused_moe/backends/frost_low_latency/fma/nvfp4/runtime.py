# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared NVFP4 FMA with per-call scratch and finalized or route-row output."""

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from .fma_fc1_o2 import _routed_fc1_kernel as first2
from .fma_fc1_o4 import _kernel as first4
from .fma_fc2 import _kernel as second
from ..activation_config import lower_activation


@cute.jit
def fc1(
    x,
    sf,
    w,
    ws,
    alpha,
    ids,
    mid,
    msf,
    gs,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    width: cutlass.Constexpr,
    stream,
):
    rows = cute.size(w, mode=[1]) // (2 if gated else 1)
    grid = (cute.size(ids) * rows // (8 * width), 1, 1)
    if cutlass.const_expr(width == 2):
        first2(
            x, sf, w, ws, alpha, ids, mid, msf, gs, activation, gated, True, parameters
        ).launch(grid=grid, block=(256, 1, 1), stream=stream)
    else:
        first4(
            x, sf, w, ws, alpha, ids, mid, msf, gs, activation, gated, True, parameters
        ).launch(grid=grid, block=(256, 1, 1), stream=stream)


@cute.jit
def fc2(
    mid,
    msf,
    w,
    ws,
    alpha,
    ids,
    scores,
    out,
    finalize: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    stream,
):
    second(mid, msf, w, ws, alpha, ids, scores, out, True, finalize, narrow).launch(
        grid=(cute.size(ids, mode=[0]) * cute.size(out, mode=[1]) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


@cute.jit
def launch(
    q,
    sf,
    w1,
    s1,
    alpha1,
    ids,
    mid,
    msf,
    gs,
    w2,
    s2,
    alpha2,
    scores,
    out,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    width: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    stream,
):
    fc1(
        q,
        sf,
        w1,
        s1,
        alpha1,
        ids,
        mid,
        msf,
        gs,
        activation,
        gated,
        parameters,
        width,
        stream,
    )
    fc2(mid, msf, w2, s2, alpha2, ids, scores, out, finalize, narrow, stream)


def execute(inputs, config, tactic):
    from ...runtime import _COMPILED, _COMPILE_LOCK

    if tactic not in (("nvfp4_ffma", 2), ("nvfp4_ffma", 4)):
        raise ValueError("NVFP4 FMA tactic must use 2 or 4 FC1 outputs per warp")
    q, sf, ids, scores, w1, s1, alpha1, w2, s2, alpha2, gs, mid, msf, out, narrow = (
        inputs
    )
    finalize = config.finalize.do_finalize
    key = (
        "nvfp4-ffi-v1",
        q.device,
        q.shape,
        w1.shape,
        w2.shape,
        tuple(s1.stride()),
        ids.shape,
        scores.dtype,
        config.activation,
        tactic,
        finalize,
        narrow,
    )
    with torch.cuda.device(q.device):
        tensors = (
            q,
            sf,
            w1,
            s1,
            alpha1,
            ids,
            mid,
            msf,
            gs,
            w2,
            s2,
            alpha2,
            scores,
            out,
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
                    major, minor = torch.cuda.get_device_capability(q.device)
                    name, parameters = lower_activation(config.activation)
                    compiled = cute.compile(
                        launch,
                        *(from_dlpack(v) for v in tensors),
                        name,
                        config.activation.is_gated,
                        parameters,
                        tactic[1],
                        finalize,
                        narrow,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
        if finalize:
            return out
        return [
            out,
            scores,
            torch.arange(ids.numel(), device=q.device, dtype=torch.int32),
        ]
