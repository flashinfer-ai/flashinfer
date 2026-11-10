# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The KDA class ``kda.bf16.c16`` behind ``recurrent_kda(batch_invariant=True)``: its constants, the chunk arithmetic
and the fingerprint.

The class's term is the native SM100 / SM103 chunked KDA prefill (the BT = 16 CuTe-DSL kernel of
``flashinfer/kda_kernels/kda_chunked_bt16.py`` on its persistent ``engine`` route, fused Q / K l2-norm, in-kernel gate
``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, in-kernel beta sigmoid, fp32 state): a sequence's rows and its
state at every multiple of 16 tokens are functions of the sequence's own inputs and its initial state. Decode computes
row ``p`` of that prefill from the boundary state and the rows since it: the term is the replay of the partial chunk
through the same kernel, and the step kernel of :mod:`.pool` (``include/flashinfer/kda/batch_invariant_decode.cuh``) is
a placement of it that keeps the earlier rows' intermediates instead of recomputing them, so decode row ``p`` is
prefill row ``p`` bit for bit. The arithmetic of that replay (which boundary, how many rows, when a chunk commits) is
here and stated in Lean (``lean/BIKernels/Kda.lean``, cases from :mod:`.lean_cases`).
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import List, Tuple

CLASS = "kda.bf16.c16"
#: The kernel's chunk: boundaries are absolute multiples of it.
CHUNK = 16
HEAD_DIM = 128
#: The two BT=16 routes ("decomp" and the persistent "engine") do not agree bitwise; the class is the engine's term.
ROUTE = "engine"
#: The compute capabilities the class is stated and tested for: SM100 (B200) and SM103 (B300).
KDA_ARCHS = ((10, 0), (10, 3))
#: The source files that define the term: relative to the ``flashinfer`` package, and to its C++ include directory.
TERM_SOURCES = ("kda_kernels/kda_chunked_bt16.py",)
TERM_INCLUDES = ("flashinfer/kda/batch_invariant_decode.cuh",)
#: The first fp32 values of a (slot, head) state compared by the decode step to notice a state the caller replaced.
STAMP = 8
#: ``log2(e)`` as the prefill kernel spells it (its gate scale is ``lower_bound * log2(e)``).
GATE_LOG2_E = 1.4426950408889634


def boundary(p: int) -> int:
    """The chunk boundary at or before position ``p``: the fp32 state the replay of row ``p`` starts from."""
    return CHUNK * (p // CHUNK)


def history_len(p: int) -> int:
    """The rows since the boundary that row ``p``'s replay re-runs before the new row."""
    return p % CHUNK


def commits(count: int) -> bool:
    """Whether appending one row to a slot holding ``count`` rows completes its chunk (the boundary state advances)."""
    return count + 1 == CHUNK


def next_count(count: int) -> int:
    return (count + 1) % CHUNK


def prefill_split(L: int) -> Tuple[int, int]:
    """A prefill of ``L`` tokens leaves ``(boundary, tail)``: the state at ``boundary`` is committed, ``tail`` rows
    (fewer than a chunk) become the slot's history."""
    return boundary(L), history_len(L)


def append_split(count: int, n: int) -> Tuple[int, int, int]:
    """``n`` tokens appended to a slot holding ``count`` open rows are ``(head, whole, tail)``: ``head`` rows up to the
    next boundary and ``tail`` rows after the last one run through the step kernel, the ``whole`` rows between (a
    multiple of 16, from a boundary) through the chunked prefill. A function of the count and the call's shape alone."""
    head = min(n, (CHUNK - count) % CHUNK)
    whole = boundary(n - head)
    return head, whole, n - head - whole


def _package_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def source_digest() -> str:
    from ...jit import env as jit_env

    h = hashlib.sha256()
    paths = [(rel, os.path.join(_package_dir(), rel)) for rel in TERM_SOURCES]
    paths += [
        (rel, os.path.join(str(jit_env.FLASHINFER_INCLUDE_DIR), rel))
        for rel in TERM_INCLUDES
    ]
    for rel, path in paths:
        with open(path, "rb") as f:
            h.update(rel.encode())
            h.update(f.read())
    return h.hexdigest()


def _dsl_version() -> str:
    """The CuTe-DSL build that compiles the prefill kernel (its contraction choices are part of the term)."""
    from importlib import metadata

    try:
        return metadata.version("nvidia-cutlass-dsl")
    except metadata.PackageNotFoundError:
        return "none"


@functools.lru_cache(maxsize=None)
def spec_fingerprint() -> str:
    """Identifier of the class: the class constants and the digest of the kernel sources that are the term (the kernels
    are hand-written, so their source is the dataflow; an edit to them renames the class)."""
    parts = [
        f"class={CLASS}",
        f"chunk={CHUNK}",
        f"head_dim={HEAD_DIM}",
        f"route={ROUTE}",
        "io=bf16",
        "state=f32",
        "gate=lower_bound*sigmoid(exp(A_log)*(g+dt_bias))",
        "norm=qk_l2",
        "beta=sigmoid",
        "step_flags=-fmad=false,no-fast-math",
        f"cute_dsl={_dsl_version()}",
        f"sources={source_digest()}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"kda-bf16-c16-engine-{digest}"


def replay_cases(positions: List[int]) -> List[Tuple[int, int, int, bool]]:
    """``(p, boundary, history_len, commits)`` per position: the rows the Lean cases pin."""
    return [
        (p, boundary(p), history_len(p), commits(history_len(p))) for p in positions
    ]
