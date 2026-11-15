# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the Swaps-AB kernel: one TVM-FFI object per static tuple ``(D, g, page, n_pad)`` with dynamic
shapes, strides, page table, work units and scale; cache module ``batch_invariant_attention``. Operand views as the row body's (``engine.py``) minus the ``O_m`` scratch (the merged chunk state
is in TMEM at every D here).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..plan import GROUP_FIELDS
from .kernel import SwapsKernel

MODULE = "batch_invariant_attention"


def kernel_name(
    D: int, g: int, page: int, n_pad: int, ragged: bool = False, cap: int = 0
) -> str:
    return jit.sanitize_symbol_name(
        f"attn_swaps_d{D}_g{g}_p{page}_n{n_pad}"
        + ("_rg" if ragged else "")
        + (f"_c{cap >> 10}" if cap else "")
    )


@functools.cache
def get_compiled_swaps(
    D: int, g: int, page: int, n_pad: int, ragged: bool = False, cap: int = 0
):
    """``cap``: the shared-memory capacity the ring is sized for (the execution device's; the current device's when 0): part of
    the kernel's identity in this process's cache and on disk."""
    from .operands import arch_smem_cap

    cap = cap or arch_smem_cap()
    name = kernel_name(D, g, page, n_pad, ragged, cap)

    def compile_fn():
        bf16, f32, i32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32
        s_hq, s_tq, s_npages, s_nunits, s_hkv = (cute.sym_int(32) for _ in range(5))
        s_ws = (
            cute.sym_int64()
        )  # fp32 elements of the workspace: beyond 2^31 on large batches
        s_nb1, s_nidx = cute.sym_int(32), cute.sym_int(32)
        q_fake = cute.runtime.make_fake_tensor(
            bf16,
            (D, s_hq, s_tq),
            (1, cute.sym_int64(divisibility=8), cute.sym_int64(divisibility=8)),
            assumed_align=16,
        )
        kv_tok = cute.sym_int(32) if ragged else page
        k_fake = cute.runtime.make_fake_tensor(
            bf16,
            (D, kv_tok, s_hkv, s_npages),
            (
                1,
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
            ),
            assumed_align=16,
        )
        v_fake = cute.runtime.make_fake_tensor(
            bf16,
            (D, kv_tok, s_hkv, s_npages),
            (
                1,
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
                cute.sym_int64(divisibility=8),
            ),
            assumed_align=16,
        )
        o_fake = cute.runtime.make_fake_tensor(
            bf16,
            (cute.sym_int(32), D),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        ws_fake = cute.runtime.make_fake_tensor(f32, (s_ws,), (1,), assumed_align=16)
        groups_fake = cute.runtime.make_fake_tensor(
            i32, (cute.sym_int(32), GROUP_FIELDS), (GROUP_FIELDS, 1), assumed_align=4
        )
        counters_fake = cute.runtime.make_fake_tensor(
            i32, (cute.sym_int(32),), (1,), assumed_align=4
        )
        desc_fake = cute.runtime.make_fake_tensor(
            i32, (s_nunits, 16), (16, 1), assumed_align=16
        )
        indptr_fake = cute.runtime.make_fake_tensor(
            i32, (s_nb1,), (1,), assumed_align=4
        )
        indices_fake = cute.runtime.make_fake_tensor(
            i32, (s_nidx,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        k = SwapsKernel(D, g, page, n_pad, ragged, cap)
        return cute.compile(
            k,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            ws_fake,
            groups_fake,
            counters_fake,
            desc_fake,
            indptr_fake,
            indices_fake,
            cutlass.Float32(1.0),
            cutlass.Float32(-1.0),
            cutlass.Int32(-1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
