# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-call bindings; the compile cache contains executable metadata only."""

import threading
from typing import Any

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from cuda.bindings import driver

from ...api import QuantFormat
from .pipeline import Geometry, Pipeline
from .io import pack_scale, combine

_COMPILED: dict[tuple, Any] = {}
_COMPILE_LOCK = threading.Lock()


def execute(inputs, config, tile_k):
    if config.quant.weight is QuantFormat.BF16:
        from .fma.bf16.runtime import execute as execute_bf16

        return execute_bf16(inputs, config, tile_k)
    if tile_k == ("ffma", 1):
        from .fma.runtime import execute as execute_fma

        return execute_fma(inputs, config)
    with torch.cuda.device(inputs[0].device):
        return _execute(inputs, config, tile_k)


def _execute(inputs, config, tile_k):
    (
        q,
        sf,
        ids,
        weights,
        w1,
        s1,
        w2,
        s2,
        packed_sf,
        partials,
        midq,
        midsf,
        routes,
        output,
        narrow_weights,
    ) = inputs
    t, h = q.shape
    p = midq.shape[1]
    a = config.activation
    geometry = Geometry(
        t,
        h,
        p,
        config.experts.intermediate_size,
        config.routing.num_experts,
        config.routing.top_k,
        a.limit,
        a.alpha,
        a.beta,
    )
    pdl = config.execution.enable_pdl is not False
    stream = driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    wd = (
        cutlass.Float4E2M1FN
        if config.quant.weight is QuantFormat.MXFP4
        else cutlass.Float8E4M3FN
    )

    def ptr(tensor, dtype):
        return make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

    def args(w, ws, x, xs, out, m, nk, batches, outtype):
        return (
            ptr(w, wd),
            ptr(ws, cutlass.Float8E8M0FNU),
            ptr(x, cutlass.Float8E4M3FN),
            ptr(xs, cutlass.Float8E8M0FNU),
            ptr(out, outtype),
            None,
            None,
            tuple(
                cutlass.Int32(v)
                for v in (m, 1, nk, geometry.experts, batches, geometry.routes)
            ),
            from_dlpack(ids.view(-1)),
        )

    p1 = args(w1, s1, q, packed_sf, partials, 2 * p, h, t, cutlass.Float32)
    p2 = args(w2, s2, midq, midsf, routes, h, p, t * geometry.top_k, cutlass.BFloat16)
    activation = tuple(from_dlpack(v) for v in (partials, midq, midsf))
    sfargs = tuple(from_dlpack(v) for v in (sf, packed_sf, midsf))
    reduceargs = tuple(from_dlpack(v) for v in (routes, weights, output))
    key = (
        geometry,
        tile_k,
        pdl,
        wd,
        weights.dtype,
        narrow_weights,
        config.finalize.do_finalize,
        q.device,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up frost_low_latency before CUDA Graph capture")
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                pipeline = Pipeline(geometry, gu_k=tile_k, enable_pdl=pdl)
                compiled = (
                    cute.compile(pack_scale, *sfargs, stream),
                    cute.compile(pipeline.gate_up, p1, activation, stream),
                    cute.compile(pipeline.down_project, p2, stream),
                    (
                        cute.compile(combine, *reduceargs, narrow_weights, stream)
                        if config.finalize.do_finalize
                        else None
                    ),
                )
                _COMPILED[key] = compiled
    compiled[0](*sfargs, stream)
    compiled[1](p1, activation, stream)
    compiled[2](p2, stream)
    if config.finalize.do_finalize:
        compiled[3](*reduceargs, stream)
        return output
    # Rows already follow token-major slot order; no sorting permutation.
    mapping = torch.arange(t * geometry.top_k, device=q.device, dtype=torch.int32)
    return [routes, weights, mapping]
