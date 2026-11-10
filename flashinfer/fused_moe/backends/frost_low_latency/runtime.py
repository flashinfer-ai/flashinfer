# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-call bindings; the compile cache contains executable metadata only."""

import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from typing import Any

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from ...api import QuantFormat
from .pipeline import Geometry, Pipeline
from .fma.activation_config import lower_activation

_COMPILED: dict[tuple, Any] = {}
_COMPILE_LOCK = threading.Lock()


def _load_cross_compiled(entry):
    """Materialize a TVM-FFI entry when the DSL's default arch differs."""
    if getattr(entry, "engine", entry) is not None:
        return entry
    # A correct explicit GPU target can still produce no JIT engine when the
    # DSL singleton was initialized for another device. Export/load preserves
    # that target without changing process-global compiler state. This runs
    # only on a cold compile, before graph capture or insertion into the cache.
    with TemporaryDirectory(prefix="flashinfer-frost-mx-") as directory:
        name = "frost_mx_" + uuid4().hex
        path = str(Path(directory) / (name + ".o"))
        entry.export_to_c(path, function_name=name)
        module = cute.runtime.load_module(path, enable_tvm_ffi=True)
        # TVM-FFI retains the module's engine after the temporary object is gone.
        return module[name]


def execute(inputs, config, tile_k):
    if config.quant.activation in (
        QuantFormat.BF16,
        QuantFormat.FP16,
    ) and config.quant.weight in (
        QuantFormat.NVFP4,
        QuantFormat.MXFP4,
        QuantFormat.MXINT4,
    ):
        from .fp4_weight_only import execute as execute_w4a16

        return execute_w4a16(inputs, config, tile_k)
    if config.quant.weight is QuantFormat.DeepSeekFp8:
        if isinstance(tile_k, tuple) and tile_k and tile_k[0] in ("wgmma", "warp_mma"):
            from .tc.deepseek_fp8 import execute as execute_grouped_block

            return execute_grouped_block(inputs, config, tile_k)
        from .deepseek_fp8 import execute as execute_deepseek

        return execute_deepseek(inputs, config, tile_k)
    if config.quant.weight is QuantFormat.FP8PerTensor:
        if isinstance(tile_k, tuple) and tile_k and tile_k[0] in ("wgmma", "warp_mma"):
            from .tc.runtime import execute as execute_grouped

            return execute_grouped(inputs, config, tile_k)
        from .fp8_per_tensor import execute as execute_fp8

        return execute_fp8(inputs, config, tile_k)
    if config.quant.weight is QuantFormat.NVFP4:
        from .nvfp4 import execute as execute_nvfp4

        return execute_nvfp4(inputs, config, tile_k)
    if config.quant.weight in (QuantFormat.BF16, QuantFormat.FP16):
        if (
            isinstance(tile_k, tuple)
            and tile_k
            and tile_k[0] in ("wgmma", "warp_mma", "ampere_mma")
        ):
            from .tc.runtime import execute as execute_grouped

            return execute_grouped(inputs, config, tile_k)
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
    name, parameters = lower_activation(a)
    geometry = Geometry(
        t,
        h,
        p,
        config.experts.intermediate_size,
        config.routing.num_experts,
        config.routing.top_k,
        getattr(a, "limit", None),
        getattr(a, "alpha", 1.0),
        getattr(a, "beta", 0.0),
        name,
        parameters,
        a.is_gated,
    )
    pdl = config.execution.enable_pdl is not False
    wd = (
        cutlass.Float4E2M1FN
        if config.quant.weight is QuantFormat.MXFP4
        else cutlass.Float8E4M3FN
    )

    from ....utils import get_compute_capability

    capability = get_compute_capability(q.device)
    major, minor = capability
    key = (
        "mx-tc-ffi-v2",
        capability,
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
                first_shape = (
                    (2 if a.is_gated else 1) * p,
                    1,
                    h,
                    geometry.experts,
                    t,
                    geometry.routes,
                )
                second_shape = (
                    h,
                    1,
                    p,
                    geometry.experts,
                    geometry.routes,
                    geometry.routes,
                )
                entry = cute.compile(
                    pipeline.launch,
                    *(from_dlpack(v) for v in inputs[:14]),
                    tuple(cutlass.Int32(v) for v in first_shape),
                    tuple(cutlass.Int32(v) for v in second_shape),
                    wd,
                    narrow_weights,
                    config.finalize.do_finalize,
                    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                    options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
                )
                entry = _load_cross_compiled(entry)
                # The cache owns only executable and immutable shape metadata.
                compiled = (entry, first_shape, second_shape)
                _COMPILED[key] = compiled
    compiled[0](*inputs[:14], compiled[1], compiled[2])
    if config.finalize.do_finalize:
        return output
    # Rows already follow token-major slot order; no sorting permutation.
    mapping = torch.arange(t * geometry.top_k, device=q.device, dtype=torch.int32)
    return [routes, weights, mapping]
