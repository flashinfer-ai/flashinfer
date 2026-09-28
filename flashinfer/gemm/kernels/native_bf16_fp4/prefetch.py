# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decode configurations for native W4A16 register prefetch."""

from typing import Any

import torch

from ....utils import get_compute_capability, get_device_index

_COMPILED: dict[tuple, Any] = {}


def is_supported(inputs):
    a, b, sf, alpha, out, _ = inputs
    return (
        get_compute_capability(a.device) == (12, 1)
        and 1 <= a.shape[0] <= 16
        and (b.shape[0], a.shape[1]) in ((5120, 17408), (34816, 5120))
        and out.dtype == torch.bfloat16
        and alpha is not None
        and all(t.data_ptr() % 16 == 0 for t in (a, b, sf, out))
    )


def _config(m, n):
    # Compact B staging keeps the deeper K tile within the shared-memory limit.
    if n == 5120 and m in (14, 15):
        return (1024, 7, m, True, 112)
    if n == 5120 and m == 13:
        return (1024, 8, m, True, 128)
    bk = 1024 if (n == 5120 and m <= 12) or (n == 34816 and m <= 10) else 512
    return (bk, 8, m, False, 128)


def _compile(n, k, config):
    import cutlass
    import cutlass.cute as cute

    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import prefetch_kernel

    bk, warps, rows, compact_b, bn = config
    m = cute.sym_int()
    a = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n, k // 2), stride_order=(1, 0), assumed_align=16
    )
    sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n * k // 16,), assumed_align=16
    )
    alpha = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    out = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    op = prefetch_kernel.NativeBf16Fp4PrefetchKernel(
        16,
        bk,
        warps,
        1,
        n,
        k,
        rows,
        1,
        1,
        2,
        0 if compact_b else -1,
        False,
        compact_b,
        bn,
    )
    name = f"n{n}_k{k}_bk{bk}_w{warps}_r{rows}_cb{int(compact_b)}_bn{bn}"
    return build_and_load_cute_dsl_kernel(
        "native_bf16_fp4_prefetch_sm121",
        name,
        lambda: cute.compile(
            op, a, b, sf, alpha, out, None, stream, options="--enable-tvm-ffi"
        ),
        extra_key_files=(__file__, prefetch_kernel.__file__),
    )


def run(a, b, sf, alpha, out, *, do_preparation=False):
    m, k = a.shape
    n = b.shape[0]
    config = _config(m, n)
    key = (get_device_index(a.device), n, k, config)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the native W4A16 prefetch tactic before capture")
        with torch.cuda.device(a.device):
            compiled = _compile(n, k, config)
        _COMPILED[key] = compiled
    if not do_preparation:
        compiled(a, b, sf.view(torch.uint8).view(-1), alpha, out, None)
    return out
