# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FMA bindings for the low-latency prepared weights and numerical contract.

The legacy adapters use the same device bodies with their original layout and
rounding. This adapter selects the low-latency contract at compile time; it is
not a user-visible numerics-changing tuning option.
"""

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver

from .mxfp8.fma_fc1 import _kernel as fc1
from .mxfp8.fma_fc2 import _kernel as fc2
from .activation_config import lower_activation


@cute.jit
def gate_up(
    x,
    xsf,
    w,
    wsf,
    ids,
    mid,
    msf,
    logical: cutlass.Constexpr,
    limit: cutlass.Constexpr,
    alpha: cutlass.Constexpr,
    beta: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    stream,
):
    # The alpha tensor argument belongs to the legacy contract and is pruned
    # from this specialization. All scales here are encoded in the MX bytes.
    fc1(
        x,
        xsf,
        w,
        wsf,
        xsf,
        ids,
        mid,
        msf,
        activation,
        gated,
        True,
        logical,
        limit,
        alpha,
        beta,
        parameters,
    ).launch(grid=(cute.size(mid) // 32, 1, 1), block=(256, 1, 1), stream=stream)


@cute.jit
def down(
    mid,
    msf,
    w,
    wsf,
    ids,
    scores,
    out,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    stream,
):
    fc2(mid, msf, w, wsf, msf, ids, scores, out, True, narrow, finalize).launch(
        grid=(cute.size(ids, mode=[0]) * cute.size(out, mode=[1]) // 4, 1, 1),
        block=(32 * cute.size(ids, mode=[1]), 1, 1),
        stream=stream,
    )


def execute(inputs, config):
    # Use the same metadata-only compile cache and lock as the TC tactic.
    from ..runtime import _COMPILED, _COMPILE_LOCK

    q, sf, ids, scores, w1, s1, w2, s2 = inputs[:8]
    mid, sf_storage, routes, output, narrow = inputs[10:]
    t, h = q.shape
    k = config.routing.top_k
    if not 1 <= k <= 32:
        raise ValueError("FMA requires top_k between 1 and 32")
    p = mid.shape[1]
    # FMA uses the linear prefix of the per-call TC scale workspace. No extra
    # allocation, weight packing, or GPU conversion is needed to switch tactic.
    msf = sf_storage[: t * k * (p // 32)].view(t * k, p // 32)
    a = config.activation
    name, parameters = lower_activation(a)
    finalize = config.finalize.do_finalize
    key = (
        "ffma-v1",
        t,
        h,
        p,
        config.routing.num_experts,
        k,
        config.experts.intermediate_size,
        a,
        w1.dtype,
        scores.dtype,
        narrow,
        finalize,
        q.device,
    )
    with torch.cuda.device(q.device):
        stream = driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        first = tuple(from_dlpack(v) for v in (q, sf, w1, s1, ids, mid, msf))
        second = tuple(
            from_dlpack(v)
            for v in (mid, msf, w2, s2, ids, scores, output if finalize else routes)
        )
        first += (
            config.experts.intermediate_size,
            getattr(a, "limit", None),
            getattr(a, "alpha", 1.0),
            getattr(a, "beta", 0.0),
            name,
            a.is_gated,
            parameters,
        )
        second += (narrow, finalize)
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
                    options = f"--gpu-arch sm_{major}{minor}a"
                    compiled = (
                        cute.compile(gate_up, *first, stream, options=options),
                        cute.compile(down, *second, stream, options=options),
                    )
                    _COMPILED[key] = compiled
        # Constexpr arguments specialize compilation and are erased from the
        # executable ABI. Only the seven tensor bindings and stream remain.
        compiled[0](*first[:7], stream)
        compiled[1](*second[:7], stream)
        if finalize:
            return output
        mapping = torch.arange(t * k, device=q.device, dtype=torch.int32)
        return [routes, scores, mapping]
