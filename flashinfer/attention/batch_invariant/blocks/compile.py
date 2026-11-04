# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the attention kernel: one TVM-FFI object per static tuple ``(D, g, page,
kv_slots, n_wg, q2)`` with dynamic shapes, strides, page table, work units and scale; cache module ``batch_invariant_attention``.

Operand views the kernel expects (all built by ``engine.py`` from the FlashInfer-shaped tensors):
Q as ``(D, Hq, total_q)`` (strides ``(1, D, Hq*D)`` for a packed ``(total_q, Hq, D)`` tensor), K and V as
``(D, page, Hkv, num_pages)`` (NHD or HND pools: runtime strides), O as ``(total_q * Hq, D)``, the fp32
workspace flat, the fp32 per-CTA ``O_m`` scratch flat (``grid x 128 x D`` at D = 256, one element otherwise), the int32 split-group table ``(num_groups, GROUP_FIELDS)`` and the per-group arrival
counters (the in-kernel chunk fold), the int32 descriptor table ``(num_units, 16)``, ``kv_indptr`` and ``kv_indices``.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..plan import GROUP_FIELDS
from .kernel import AttentionKernel

MODULE = "batch_invariant_attention"


def kernel_name(
    D: int,
    g: int,
    page: int,
    kv_slots: int,
    n_wg: int = 1,
    q2: bool = False,
    ragged: bool = False,
) -> str:
    return jit.sanitize_symbol_name(
        f"attn_d{D}_g{g}_p{page}_s{kv_slots}_w{n_wg}"
        + ("_q2" if q2 else "")
        + ("_rg" if ragged else "")
    )


@functools.cache
def get_compiled_attention(
    D: int,
    g: int,
    page: int,
    kv_slots: int,
    n_wg: int = 1,
    q2: bool = False,
    ragged: bool = False,
):
    name = kernel_name(D, g, page, kv_slots, n_wg, q2, ragged)

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
        kv_tok = (
            cute.sym_int(32) if ragged else page
        )  # the ragged pool: one page of total_kv (dynamic) tokens
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
        scratch_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(32),), (1,), assumed_align=16
        )
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
        k = AttentionKernel(D, g, page, kv_slots, n_wg, q2, ragged)
        return cute.compile(
            k,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            ws_fake,
            scratch_fake,
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
