# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-built GEMM: one TVM-FFI object per (knobs, B major, chunks, SM count) with dynamic M, N, K;
cache module ``batch_invariant_gemm_bf16`` in FlashInfer's CuTe-DSL disk cache. Operand conventions: A (M, K)
K-major, B (K, N) N-major (torch row-major) or K-major (a transposed weight), C (M, N) N-major, fp32 accumulate.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from .kernel import BlockGemm
from .operands import ALIGN, IN_DT, OUT_DT, can_implement  # noqa: F401  (re-exported)

MODULE = "batch_invariant_gemm_bf16"


def max_active_clusters(cfg: Knobs, sm_count: int) -> int:
    """Persistent grid in clusters: one CTA per SM of the launch device for this body (a launch parameter: the bits never
    depend on it)."""
    return sm_count // (cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1])


def kernel_name(cfg: Knobs, b_major: str, chunks: int, sm_count: int) -> str:
    return jit.sanitize_symbol_name(
        f"bf16_b{b_major}_C{chunks}_{cfg.describe()}_sm{sm_count}"
    )


@functools.cache
def get_compiled_gemm(cfg: Knobs, b_major: str, chunks: int, sm_count: int):
    """``kernel(a3, b3, c3)``: the unsplit class (``chunks == 1``, L = 1), the fixed-chunk split (batch index = K chunk,
    fp32 partials) or, with ``cfg.cta_chunks``, the in-CTA chunk fold (the same operand views, the output direct with
    L = 1). ``sm_count`` sizes the persistent grid (the launch device's) and is part of the cache key."""
    in_dt, al = IN_DT["bf16"], ALIGN["bf16"]
    in_cta = cfg.cta_chunks and chunks > 1
    direct = chunks == 1 or in_cta
    out_dt = OUT_DT["bf16"] if direct else cutlass.Float32
    oal = ALIGN["bf16"] if direct else ALIGN["f32"]
    name = kernel_name(cfg, b_major, chunks, sm_count)

    def compile_fn():
        sym_m, sym_n, sym_kc = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32, divisibility=al),
        )
        L = chunks
        Lc = 1 if in_cta else L
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_kc, L),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        if b_major == "n":  # (K, N) row-major viewed as (n, kc, l)
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (1, cute.sym_int64(divisibility=al), cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        else:  # (N, K) row-major, a transposed weight
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        c_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n, Lc),
            (cute.sym_int64(divisibility=oal), 1, cute.sym_int64(divisibility=oal)),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = cfg.grid_clusters or max_active_clusters(cfg, sm_count)
        return cute.compile(
            BlockGemm(cfg, "bf16" if direct else "f32", chunks),
            a_fake,
            b_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
