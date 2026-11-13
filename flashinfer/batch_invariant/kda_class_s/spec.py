# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The KDA class ``kda.bf16.s16`` behind ``recurrent_kda(batch_invariant=True)``: its term, constants and fingerprint.

One term for prefill, chunked prefill and decode, per sequence and head, in 16-token chunks anchored at the sequence
start (``p`` = position in the chunk). The pool slot holds the boundary state ``S0`` (bf16 ``[V, K]``); the open chunk is
held next to it as rows ``ki_j`` (bf16), ``u_j`` (fp32) and the per-channel log2-decay prefix ``Lam_j`` (fp32), ``j < p``.
Every operation is binary32 round-to-nearest-even unless stated. ``MMA128(a, b)`` is the 128-term bf16 dot product
evaluated as eight ``mma.sync m16n8k16`` steps in ascending k from a zero fp32 accumulator. ``ex2``, ``tanh``, ``rcp``
and ``rsqrt`` are the hardware's ``ex2.approx.f32``, ``tanh.approx.f32``, ``rcp.approx.ftz.f32`` and
``rsqrt.approx.f32`` instructions, the ones the native KDA prefill kernel uses for its gate.

1. Normalisation: ``r_x = rsqrt(ss(x) + 1e-6)`` with ``ss`` = 32 partial sums of 4 squares each in index order, then a
   pairwise tree over the partials (xor butterfly 16, 8, 4, 2, 1); ``kn = k * r_k``, ``qn = q * (r_q * scale)``.
2. Gate, in the log2 domain: ``l_p[c] = fma(tanh(ea * (g_p[c] + dt_bias[c])), hs, hs)`` with
   ``ea = ex2(A_log * log2(e)) * 0.5`` and ``hs = float(lower_bound * log2(e) / 2)``, which is
   ``lower_bound * log2(e) * sigmoid(exp(A_log) * (g + dt_bias))``; ``Lam_p = Lam_{p-1} + l_p`` (``Lam_0 = l_0``);
   ``D_p = ex2(Lam_p)``; ``beta_p = fma(tanh(0.5 * b_p), 0.5, 0.5)``.
3. Operands: ``kt_p = bf16(kn * D_p)``, ``ki_p = bf16(kn * rcp(D_p))``, ``qt_p = bf16(qn * D_p)``.
4. Substitution, sequential in ``j``: ``pred = MMA128(S0[v], kt_p)``; for ``j < p``:
   ``pred = fma(MMA128(ki_j, kt_p), u_j[v], pred)``; ``u_p[v] = beta_p * (v_p[v] - pred)``.
5. Output: ``o = MMA128(S0[v], qt_p)``; for ``j <= p``: ``o = fma(MMA128(ki_j, qt_p), u_j[v], o)``; ``out = bf16(o)``.
6. At ``p = 15`` the chunk folds: ``S0[v, c] <- bf16(acc)``, ``acc`` starting at ``D_15[c] * S0[v, c]`` and receiving one
   ``mma.sync`` step over ``j`` with A = ``bf16(u_j[v])`` and B = ``bf16(float(ki_j[c]) * D_15[c])``.

The kernels (``include/flashinfer/kda/batch_invariant_class_s.cuh``) are two placements of this term: the prefill kernel
loops over a sequence's chunks with the state in registers, the decode kernel adds one row. ``reference.py`` evaluates
the term in torch with exact elementary functions and plain matmuls, so the kernels agree with it to a tolerance. The
operands ``kt`` and ``ki`` span ``exp(+-16 * |lower_bound|)``, which is why the class takes ``-5 <= lower_bound < 0``
(bf16's exponent range).
"""

from __future__ import annotations

import functools
import hashlib
import os

CLASS = "kda.bf16.s16"
#: Chunk boundaries are multiples of it from the sequence start.
CHUNK = 16
HEAD_DIM = 128
#: The compute capabilities the class is stated and tested for.
KDA_ARCHS = ((10, 0), (10, 3), (10, 7))
#: The smallest gate lower bound whose operands stay inside bf16's exponent range over a chunk.
MIN_LOWER_BOUND = -5.0
#: The sources that define the term: the kernels (C++ include directory), the binding that forms the gate scale
#: (csrc directory) and the JIT spec with the compile flags (the ``flashinfer`` package).
TERM_INCLUDES = ("flashinfer/kda/batch_invariant_class_s.cuh",)
TERM_CSRC = ("kda_batch_invariant_class_s.cu",)
TERM_PACKAGE = ("jit/kda_batch_invariant_class_s.py",)


def source_digest() -> str:
    from ...jit import env as jit_env

    package = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    h = hashlib.sha256()
    for base, rels in (
        (str(jit_env.FLASHINFER_INCLUDE_DIR), TERM_INCLUDES),
        (str(jit_env.FLASHINFER_CSRC_DIR), TERM_CSRC),
        (package, TERM_PACKAGE),
    ):
        for rel in rels:
            with open(os.path.join(base, rel), "rb") as f:
                h.update(rel.encode())
                h.update(f.read())
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def spec_fingerprint() -> str:
    """Identifier of the class: its constants and the digest of the sources that are the term (the kernels are
    hand-written, so their source is the dataflow; an edit to them renames the class)."""
    parts = [
        f"class={CLASS}",
        f"chunk={CHUNK}",
        f"head_dim={HEAD_DIM}",
        "io=bf16",
        "state=bf16",
        "open_chunk=ki:bf16,u:f32,prefix:f32",
        "dot=mma.sync.m16n8k16.bf16.f32",
        "gate=lower_bound*sigmoid(exp(A_log)*(g+dt_bias)),log2,tanh.approx,ex2.approx,rcp.approx.ftz",
        "norm=qk_l2,rsqrt.approx",
        "beta=sigmoid,tanh.approx",
        "flags=-fmad=false,no-fast-math",
        f"sources={source_digest()}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"kda-bf16-s16-{digest}"
