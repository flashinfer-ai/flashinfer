# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The GDN class ``gdn.bf16.c64`` behind ``chunk_gated_delta_rule(batch_invariant=True)``: its constants, the chunk
arithmetic of a call and the fingerprint.

The class's term is the native SM100 / SM103 chunked GDN prefill (the chunk-64 CuTe-DSL kernel of
``flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`` without context parallelism, log-domain gates, the Q / K
l2-norm of ``gdn_kernels/qk_l2norm.py``, fp32 state): a sequence's rows and its state at every multiple of 64 tokens
are functions of the sequence's own inputs and its initial state. Decode computes row ``p`` of that term from rows
cached since the last boundary (the step kernel of ``include/flashinfer/gdn/batch_invariant_decode.cuh``), so decode
row ``p`` is prefill row ``p`` bit for bit; which boundary a row starts from, how many rows a call leaves open, when a
chunk commits and which placement a call takes are here and stated in Lean (``lean/BIKernels/Gdn.lean``, cases from
:mod:`.lean_cases`).
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import List, Tuple

CLASS = "gdn.bf16.c64"
#: The kernel's chunk: boundaries are absolute multiples of it.
CHUNK = 64
HEAD_DIM = 128
#: The compute capabilities the class is stated and tested for: SM100 (B200) and SM103 (B300).
GDN_ARCHS = ((10, 0), (10, 3))
#: Calls with fewer tokens per sequence than this run in the step kernel alone; the others take the prefill kernel for
#: their whole chunks and the step kernel for the rows of open chunks.
STEP_MAX_TOKENS = CHUNK
#: The kernel sources the fingerprint digests (package-relative, and relative to the C++ include directory).
TERM_SOURCES = (
    "gdn_kernels/blackwell/gated_delta_net_chunked.py",
    "gdn_kernels/qk_l2norm.py",
)
TERM_INCLUDES = ("flashinfer/gdn/batch_invariant_decode.cuh",)


def boundary(p: int) -> int:
    """The chunk boundary at or before position ``p``: the fp32 state row ``p`` is computed from."""
    return CHUNK * (p // CHUNK)


def phase(p: int) -> int:
    """The rows since the boundary that row ``p`` depends on (the open chunk's rows before it)."""
    return p % CHUNK


def commits(count: int) -> bool:
    """Whether appending one row to a slot holding ``count`` rows completes its chunk (the boundary state advances)."""
    return count + 1 == CHUNK


def next_count(count: int) -> int:
    return (count + 1) % CHUNK


def extend_split(count: int, n: int) -> Tuple[int, int]:
    """A call appending ``n`` tokens to a slot holding ``count`` open rows leaves ``(whole, tail)``: the open chunk plus
    the new tokens cover ``whole`` rows of complete chunks (their final state is committed) and ``tail`` open rows."""
    total = count + n
    return boundary(total), phase(total)


def uses_step_kernel(max_tokens: int) -> bool:
    """The placement of a call whose longest sequence brings ``max_tokens`` tokens."""
    return max_tokens < STEP_MAX_TOKENS


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


@functools.lru_cache(maxsize=None)
def spec_fingerprint() -> str:
    """Identifier of the class: the class constants and the digest of the kernel sources listed in ``TERM_SOURCES`` and
    ``TERM_INCLUDES`` (the kernels are hand-written, so their source is the dataflow; an edit to one of those files
    renames the class, an edit to a module they import does not)."""
    parts = [
        f"class={CLASS}",
        f"chunk={CHUNK}",
        f"head_dim={HEAD_DIM}",
        "route=uncut",
        "io=bf16",
        "state=f32",
        "gate=log",
        "norm=qk_l2",
        f"sources={source_digest()}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"gdn-bf16-c64-{digest}"


def extend_cases(cases: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int, bool]]:
    """``(count, n, whole, tail, step_kernel)`` per ``(count, n)``: the rows the Lean cases pin."""
    return [(c, n, *extend_split(c, n), uses_step_kernel(n)) for c, n in cases]
