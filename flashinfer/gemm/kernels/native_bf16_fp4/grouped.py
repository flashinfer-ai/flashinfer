# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Grouped M tiles and packed scale loads for canonical NVFP4 weights."""

from typing import Any

import torch

from ....utils import get_compute_capability, get_device_index

_COMPILED: dict[tuple, Any] = {}
_WORD_SCALES = {
    (5120, 128),
    (5120, 512),
    (5120, 1024),
    (5120, 3072),
    (34816, 128),
    (34816, 2048),
}
_TILE_CONFIGS = {
    (5120, 256): (8, 8, 128, 2),
    (5120, 512): (4, 8, 64, 2),
    (5120, 1024): (11, 8, 64, 2),
    (5120, 3072): (16, 8, 64, 2),
    (34816, 256): (8, 8, 128, 2),
    (34816, 512): (8, 8, 128, 2),
}
_UNROLL = {(5120, 128): 8, (5120, 512): 8, (5120, 3072): 16}
_GROUP_M = {
    (5120, 256): 2,
    (5120, 1568): 6,
    (5120, 2048): 6,
    (34816, 1568): 5,
    (5120, 1024): 6,
    (34816, 256): 2,
    (34816, 512): 4,
    (34816, 1024): 3,
    (5120, 512): 8,
    (5120, 3072): 2,
}
_COMPACT_A = {
    (5120, 128),
    (34816, 128),
    (5120, 256),
    (5120, 1024),
    (5120, 1568),
    (34816, 256),
    (34816, 512),
    (5120, 512),
    (5120, 3072),
}


def is_supported(inputs):
    a, b, sf, alpha, out, _ = inputs
    return (
        get_compute_capability(a.device) == (12, 1)
        and a.shape[0] >= 65
        and (b.shape[0], a.shape[1]) in ((5120, 17408), (34816, 5120))
        and out.dtype == torch.bfloat16
        and alpha is not None
        and all(t.data_ptr() % 16 == 0 for t in (a, b, sf, out))
    )


def _config(m, n):
    shape = (n, m)
    if shape in _TILE_CONFIGS:
        tiles, warps, tile_k, stages = _TILE_CONFIGS[shape]
    elif m == 352:
        tiles, warps, tile_k, stages = (22, 8, 64, 2)
    elif m < 512:
        tiles, warps, tile_k, stages = (8, 8, 128, 2)
    else:
        tiles = {
            1024: 22,
            1568: 20 if n == 34816 else 17,
            2000: 21,
            2048: 22,
            3072: 22 if n == 5120 else 20,
            4096: 20,
            8192: 21,
        }.get(m, 16)
        warps, tile_k, stages = (8, 64, 2)
    unroll = _UNROLL.get(shape, 2 if m in (352, 1024) or shape == (5120, 2048) else 4)
    return (
        tiles,
        warps,
        tile_k,
        stages,
        unroll,
        _GROUP_M.get(shape, 1),
        shape in _COMPACT_A,
        shape in _WORD_SCALES,
    )


def _compile(n, k, config):
    import cutlass
    import cutlass.cute as cute

    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import grouped_kernel

    tiles, warps, tile_k, stages, unroll, group_m, compact_a, word_scales = config
    m = cute.sym_int()
    a = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n, k // 2), stride_order=(1, 0), assumed_align=16
    )
    sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (n * k // 16,), assumed_align=16
    )
    alpha = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    out = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    op = grouped_kernel.NativeBf16Fp4GroupedKernel(
        m_tiles=tiles,
        warps=warps,
        tile_k=tile_k,
        stages=stages,
        alpha_one=False,
        loop_unroll=unroll,
        group_m=group_m,
        force_compact_a=compact_a,
        word_scales=word_scales,
    )
    name = f"n{n}_k{k}_tm{tiles}_w{warps}_tk{tile_k}_st{stages}_u{unroll}_gm{group_m}_ca{int(compact_a)}_ws{int(word_scales)}"
    return build_and_load_cute_dsl_kernel(
        "native_bf16_fp4_grouped_sm121",
        name,
        lambda: cute.compile(
            op, a, b, sf, alpha, out, stream, options="--enable-tvm-ffi"
        ),
        extra_key_files=(__file__, grouped_kernel.__file__),
    )


def run(a, b, sf, alpha, out, *, do_preparation=False):
    m, k = a.shape
    n = b.shape[0]
    config = _config(m, n)
    key = (get_device_index(a.device), n, k, config)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the native W4A16 grouped tactic before capture")
        with torch.cuda.device(a.device):
            compiled = _compile(n, k, config)
        _COMPILED[key] = compiled
    if not do_preparation:
        compiled(a, b, sf.view(torch.float8_e4m3fn).view(-1), alpha, out)
    return out
