# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the attention kernel: one TVM-FFI object per static tuple ``(D, g, page,
kv_slots, n_wg, q2, ragged, hw_rowmax, cap, causal)`` with dynamic shapes, strides, page table, work units and scale; cache module ``batch_invariant_attention``.
``cap`` is the shared-memory capacity the smem plan is sized for (the execution device's): part of the kernel's identity, in-process
and on disk, so a kernel compiled for one part is never loaded for another.

Operand views the kernel expects (all built by ``engine.py`` from the FlashInfer-shaped tensors):
Q as ``(D, Hq, total_q)`` (strides ``(1, D, Hq*D)`` for a packed ``(total_q, Hq, D)`` tensor), K and V as
``(D, page, Hkv, num_pages)`` (NHD or HND pools: runtime strides), O as ``(total_q * Hq, D)``, the fp32 LSE as
``(total_q * Hq,)`` (written under a nonzero ``lse_mode``: 1 base 2, 2 base e; any one-element tensor otherwise), the fp32
sinks as ``(Hq,)`` (read by a kernel compiled with ``sinks``; any one-element tensor otherwise), the fp32
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
    hw_rowmax: bool = False,
    cap: int = 0,
    *,
    causal: bool = True,
    sinks: bool = False,
) -> str:
    return jit.sanitize_symbol_name(
        f"attn_d{D}_g{g}_p{page}_s{kv_slots}_w{n_wg}"
        + ("_q2" if q2 else "")
        + ("_rg" if ragged else "")
        + ("_hwm" if hw_rowmax else "")
        + (f"_c{cap >> 10}" if cap else "")
        + ("" if causal else "_nc")
        + ("_sink" if sinks else "")
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
    hw_rowmax: bool = False,
    cap: int = 0,
    *,
    causal: bool = True,
    sinks: bool = False,
):
    """``hw_rowmax``: the softmax's row maximum from the TMEM load (``engine.hw_row_max_enabled``); part of the kernel's name.
    ``cap``: the shared-memory capacity the smem plan is checked against (the execution device's; the current device's when 0).
    ``causal``: the mask mode, a compile-time constant of the body (the non-causal kernel is a separate object, ``_nc``; the causal
    body's code is unchanged).
    ``sinks``: the kernel of the class's sink member (the sink step where a row is completed), in either mask mode; part of the
    kernel's name (``_sink``, after the mode)."""
    from .operands import arch_smem_cap

    cap = cap or arch_smem_cap()
    name = kernel_name(
        D,
        g,
        page,
        kv_slots,
        n_wg,
        q2,
        ragged,
        hw_rowmax,
        cap,
        causal=causal,
        sinks=sinks,
    )

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
        lse_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(32),), (1,), assumed_align=4
        )
        sink_fake = cute.runtime.make_fake_tensor(
            f32, (cute.sym_int(32),), (1,), assumed_align=4
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
        k = AttentionKernel(
            D,
            g,
            page,
            kv_slots,
            n_wg,
            q2,
            ragged,
            hw_rowmax,
            cap,
            causal=causal,
            sinks=sinks,
        )
        return cute.compile(
            k,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            lse_fake,
            sink_fake,
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
            cutlass.Int32(0),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
