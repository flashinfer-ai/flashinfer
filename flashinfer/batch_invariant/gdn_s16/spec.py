# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The GDN class ``gdn.bf16.s16`` behind ``chunk_gated_delta_rule(batch_invariant=True)``: its constants, the chunk
arithmetic of a call and the fingerprint.

The term, per sequence and v-head, with 16-token chunks anchored at the sequence start (``p`` = position in the chunk,
every operation binary32 round-to-nearest unless stated, ``MMA128`` the 128-term bf16 dot product evaluated as eight
``mma.sync m16n8k16`` steps in ascending k from a zero accumulator):

1. ``kn = bf16(k * r_k)``, ``qn = bf16(q * (r_q * scale))``, ``r_x = 1 / sqrt(ss(x) + 1e-6)``, ``ss`` 32 partial sums
   of 4 squares in index order, then a pairwise tree over the partials (xor butterfly 16, 8, 4, 2, 1);
2. ``g = -exp(A_log) * softplus(a + dt_bias)`` (threshold 20), ``beta = 1 / (1 + exp(-b))``, ``L_p = L_{p-1} + g``,
   ``d_p = exp(L_p)``, ``w_jp = exp(L_p - L_j)``;
3. ``a_p = MMA128(S0, kn_p)``, ``b_p = MMA128(S0, qn_p)``, ``G_jp = MMA128(kn_j, kn_p)``, ``E_jp = MMA128(kn_j, qn_p)``;
4. ``pred = d_p * a_p``, then for ``j < p`` in order ``pred = fma(w_jp * G_jp, u_j, pred)``;
   ``u_p = beta * (v_p - pred)``;
5. ``o = d_p * b_p``, then for ``j < p`` in order ``o = fma(w_jp * E_jp, u_j, o)``, ``o = fma(E_pp, u_p, o)``; the
   output row is ``bf16(o)``;
6. at ``p = 15`` the state folds: ``S0 <- bf16(d_15 * S0 + sum_j bf16(w_{j,15} * u_j) kn_j^T)``, the sum one
   ``mma.sync`` step on the accumulator ``d_15 * S0``.

The state is rounded to bf16 once per 16 tokens; ``u`` stays fp32 inside the open chunk. The three kernels of
``include/flashinfer/gdn/batch_invariant_class_s.cuh`` (decode, short calls, long prefills) evaluate this one term, so
a row's bits depend on its sequence's own inputs and initial state alone.
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import Tuple

CLASS = "gdn.bf16.s16"
#: The chunk: the state is folded and rounded to bf16 at absolute multiples of it.
CHUNK = 16
HEAD_DIM = 128
#: The compute capabilities the class is stated and tested for.
GDN_ARCHS = ((10, 0), (10, 3), (10, 7))
#: The kernel source the fingerprint digests (relative to the C++ include directory).
TERM_INCLUDES = ("flashinfer/gdn/batch_invariant_class_s.cuh",)


def boundary(p: int) -> int:
    """The chunk boundary at or before position ``p``: the bf16 state row ``p`` is computed from."""
    return CHUNK * (p // CHUNK)


def phase(p: int) -> int:
    """The open rows row ``p`` depends on (the rows since the boundary)."""
    return p % CHUNK


def commits(count: int) -> bool:
    """Whether appending one row to a slot holding ``count`` open rows folds the chunk into the state."""
    return count + 1 == CHUNK


def extend_split(count: int, n: int) -> Tuple[int, int]:
    """A call appending ``n`` tokens to a slot holding ``count`` open rows folds ``whole`` rows of complete chunks into
    the state and leaves ``tail`` open rows."""
    total = count + n
    return boundary(total), phase(total)


def source_digest() -> str:
    from ...jit import env as jit_env

    h = hashlib.sha256()
    for rel in TERM_INCLUDES:
        with open(os.path.join(str(jit_env.FLASHINFER_INCLUDE_DIR), rel), "rb") as f:
            h.update(rel.encode())
            h.update(f.read())
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def spec_fingerprint() -> str:
    """Identifier of the class: its constants and the digest of the kernel source (the kernels are hand-written, so the
    source is the dataflow; an edit to it renames the class)."""
    parts = [
        f"class={CLASS}",
        f"chunk={CHUNK}",
        f"head_dim={HEAD_DIM}",
        "io=bf16",
        "state=bf16",
        "gate=fused",
        "norm=qk_l2",
        "fmad=false",
        f"sources={source_digest()}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"gdn-bf16-s16-{digest}"
