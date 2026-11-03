# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile metadata only; block scales, buffers and stream stay per-call."""

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from .activation_config import lower_activation
from .deepseek_fp8_kernel import launch


def execute(inputs, config, tactic):
    from ..runtime import _COMPILED, _COMPILE_LOCK

    if isinstance(tactic, tuple) and len(tactic) == 2:
        tactic = (*tactic, 4)
    if tactic not in [("ffma", a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)]:
        raise ValueError("DeepSeek Frost requires FMA FC1/FC2 column tactics")
    x, qx, sf, w1, s1, w2, s2, ids, scores, raw, mid, ms, routes, out, narrow = inputs
    first, second = tactic[1:]
    finalize = config.finalize.do_finalize
    key = (
        "deepseek-fp8-fma-ffi-v1",
        x.dtype,
        tuple(x.shape),
        tuple(w1.shape),
        tuple(w2.shape),
        config.routing.top_k,
        config.activation,
        scores.dtype,
        narrow,
        finalize,
        first,
        second,
        x.device,
    )
    tensors = (
        x,
        qx,
        sf,
        w1,
        s1,
        w2,
        s2,
        ids,
        scores,
        raw,
        mid,
        ms,
        out if finalize else routes,
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
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
        if finalize:
            return out
        return [
            routes,
            scores,
            torch.arange(ids.numel(), device=x.device, dtype=torch.int32),
        ]
