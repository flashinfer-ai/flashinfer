"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Rows of record of the packed KDA context-parallel (KCP) tests: perf packs Q1..Q6, correctness rows C1..C8 (and C5b),
adversarial rows A1v2..A4v2, and the partition layouts C (contiguous), Z (whole-buffer zigzag), D (per-document
zigzag + ragged cut points) and R (explicit ragged tables, see ``RAGGED_TABLES``).

Pure Python (no torch) and deterministic (seed 20261021): the rows, cu_seqlens and fragment tables are the same in
every process.

Fragment table entry: (doc, offset_in_doc, length, rank, slot) with slot = position of the fragment in the rank's
local buffer order (by document, then by offset).  Predecessor / successor links are derived by `links()`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SEED = 20261021
D = 128
LAYOUTS = ("C", "Z", "D")
CP_SIZES = (2, 4, 8, 16)
PERF_H = 64


@dataclass(frozen=True)
class Row:
    name: str
    kind: str  # "perf" | "correct" | "adversarial"
    lengths: tuple[int, ...]  # document lengths (global pack order)
    h: int
    cp: tuple[int, ...]  # CP sizes the row is run at
    # layouts the row is run at ("R" = explicit ragged table, see RAGGED_TABLES)
    layouts: tuple[str, ...]
    note: str = ""
    # input generator regime (cake_kda_kcp_inputs): "default" or a regime string with parameters:
    # "iso_weak:b0=B:gmax=G:gzero=Z" | "aniso_weak:noise=E:gmax=G:gzero=Z" | "mixed_v2:b0=B:gmax=G"
    data: str = "default"

    @property
    def total(self) -> int:
        return sum(self.lengths)

    @property
    def cu_seqlens(self) -> list[int]:
        cu = [0]
        for n in self.lengths:
            cu.append(cu[-1] + n)
        return cu


# ----------------------------------------------------------------------------------------------------------------
# Perf packs: total tokens fixed, documents 2..6, lengths in [23_000, 131_071], none a multiple of 64, at least one
# with (len mod 64) < 32; Q4 carries a 131_07x document, Q5 a 23_0xx document.

# fmt: off
PACK_SPECS = {
    "Q1": dict(total=249_856, docs=4, note="issue #6919 pack"),
    "Q2": dict(total=226_336, docs=2, note="near the lower bound"),
    "Q3": dict(total=262_048, docs=6, note="near the upper bound, most documents"),
    "Q4": dict(total=240_160, docs=3, note="one 131,07x document (crosses >= 3 shards at CP16)", force=(131_070, 131_079)),
    "Q5": dict(total=255_008, docs=5, note="one 23,0xx document", force=(23_000, 23_099)),
    "Q6": dict(total=249_856, docs=2, note="same total as Q1, fewer documents"),
}
# fmt: on
LEN_MIN, LEN_MAX = 23_000, 131_071


def _ok_lengths(lengths: list[int]) -> bool:
    if any(n < LEN_MIN or n > LEN_MAX or n % 64 == 0 for n in lengths):
        return False
    return any(n % 64 < 32 for n in lengths)


def _gen_pack(name: str, spec: dict) -> tuple[int, ...]:
    rng = random.Random(f"{SEED}:{name}")
    total, docs = spec["total"], spec["docs"]
    for _attempt in range(100_000):
        lengths = []
        if "force" in spec:
            lo, hi = spec["force"]
            lengths.append(rng.randint(lo, hi))
        rest = docs - len(lengths)
        remaining = total - sum(lengths)
        # draw rest-1 lengths, the last one closes the total
        for i in range(rest - 1):
            # documents still to draw after this one (incl. the closing one)
            left_after = rest - 1 - i
            lo = max(LEN_MIN, remaining - left_after * LEN_MAX)
            hi = min(LEN_MAX, remaining - left_after * LEN_MIN)
            if lo > hi:
                break
            n = rng.randint(lo, hi)
            lengths.append(n)
            remaining -= n
        else:
            lengths.append(remaining)
            if len(lengths) == docs and _ok_lengths(lengths):
                rng.shuffle(lengths)
                return tuple(lengths)
    raise RuntimeError(f"pack {name}: no admissible lengths")


PACKS: dict[str, Row] = {
    name: Row(
        name=name,
        kind="perf",
        lengths=_gen_pack(name, spec),
        h=PERF_H,
        cp=CP_SIZES,
        layouts=LAYOUTS,
        note=spec["note"],
    )
    for name, spec in PACK_SPECS.items()
}

# ----------------------------------------------------------------------------------------------------------------
# Correctness rows.  "R" layouts take their explicit ragged table from RAGGED_TABLES[(row, cp)].

# fmt: off
CORRECT: dict[str, Row] = {
    "C1": Row("C1", "correct", (100, 640, 77, 1000, 129), 4, (2, 4), ("C",), "small documents, boundaries inside chunks"),
    "C2": Row("C2", "correct", (3000,), 4, (4,), ("C", "Z", "D"), "one document crossing >= 3 shards"),
    "C3": Row("C3", "correct", (2048, 5, 3000, 1), 4, (4,), ("D",), "empty and 1-token fragments"),
    "C4a": Row("C4a", "correct", (700 + 1 + 1300 + 999,), 4, (4,), ("R",), "explicit ragged T_r = [700, 1, 1300, 999], boundary inside a chunk"),
    "C4b": Row("C4b", "correct", (1200 + 900 + 900,), 4, (4,), ("R",), "explicit ragged T_r = [1200, 0, 900, 900], one rank with 0 tokens"),
    "C5": Row("C5", "correct", (63 + 64 + 65 + 1 + 63 + 500,), 4, (4,), ("R",), "fragments of 63 / 64 / 65 tokens at cut points == 1 and 63 (mod 64)"),
    "C6": Row("C6", "correct", (65_553,), 4, (8,), ("Z",), "one document, >= 64 chunks per fragment"),
    "C7": Row("C7", "correct", (5000, 1003), 1, (4,), ("R",), "conv halo: fragment of 1 token (rank 0) and 2 tokens (rank 1) precede the 4997-token fragment (rank 2): history rows from two ranks; document start on rank 3"),
    "C7b": Row("C7b", "correct", (5000, 1003), 1, (4,), ("R",), "conv halo: predecessor rank 2 holds 0 tokens; rank 3's fragment takes history from ranks 1 and 0"),
    "C8": Row("C8", "correct", (8_192 + 37, 6_144, 6_144), 64, (8,), ("C", "Z", "D"), "H = 64, CP8, 20,480 + 37 tokens in 3 documents"),
}

# Explicit ragged tables: list of per-rank token counts over the GLOBAL token order (contiguous ragged), or a full
# fragment table.  (row, cp) -> list of T_r  (contiguous ragged split of the global buffer).
RAGGED_TABLES: dict[tuple[str, int], list[int]] = {
    ("C4a", 4): [700, 1, 1300, 999],
    ("C4b", 4): [1200, 0, 900, 900],
    ("C5", 4): [63, 64 + 65, 1 + 63, 500],          # cut points at 63 (== 63 mod 64), 192 (== 0), 256 (== 0) -> see note below
    ("C7", 4): [1, 2, 4997, 1003],                  # doc0 = 1 | 2 | 4997 tokens on ranks 0..2, doc1 on rank 3
    ("C7b", 4): [1, 2, 0, 4997 + 1003],             # rank 2 empty; doc0's tail + doc1 on rank 3
}
# C5 note: cut points 63 | 192 | 256 give fragments of 63, 129, 64 and 500 tokens; the 65-token fragment and the
# "== 1 (mod 64)" cut come from the second C5 table below (both tables are run).
RAGGED_TABLES[("C5b", 4)] = [65, 64, 63 + 1 + 63, 500]  # cut points 65 (== 1 mod 64), 129 (== 1), 256 (== 0)
CORRECT["C5b"] = Row("C5b", "correct", (65 + 64 + 63 + 1 + 63 + 500,), 4, (4,), ("R",), "fragments of 65 / 64 / 127 tokens, cut points == 1 (mod 64)")
# fmt: on

# ----------------------------------------------------------------------------------------------------------------
# Adversarial precision rows: long-memory regimes (weak erasure, near-zero decay) in which the state carried across
# a fragment boundary stays visible in the outputs and gradients downstream.  The adversarial rows carry a ``v2``
# suffix: operands are seeded by row name, so the names are part of the inputs.

# fmt: off
ADVERSARIAL: dict[str, Row] = {
    "A1v2": Row("A1v2", "adversarial", (32_801,), 4, (8,), ("C",),
                "slow decay v2: isotropic keys, weak erasure beta ~ 0.02, g in (-1e-5, 0); the carried state survives a "
                "4,100-token fragment (carried fraction ~0.5)", data="iso_weak:b0=0.02:gmax=1e-5:gzero=0"),
    "A2v2": Row("A2v2", "adversarial", (40_000,), 4, (8,), ("C",),
                "anisotropic keys v2: 4 dims + 2 % noise, beta ~ 0.98, g in (-1e-5, 0), heads 0-1 with g == 0; the 124 "
                "never-erased directions carry the state", data="aniso_weak:noise=0.02:gmax=1e-5:gzero=2"),
    "A3v2": Row("A3v2", "adversarial", (32_801, 32_769), 4, (4,), ("D",),
                "mixed v2: document 0 long-memory (iso_weak b0=0.02, g in (-1e-5, 0)), document 1 fast decay g in (-5, -1); "
                "layout D CP4 -> 8 pieces of ~4,100 tokens (58-69 chunks with partition_D's jitter) per document",
                data="mixed_v2:b0=0.02:gmax=1e-5"),
    "A4v2": Row("A4v2", "adversarial", (131_071,), 4, (16, 8), ("C",),
                "long chain v2: isotropic keys, beta ~ 0.008 (memory ~16 K tokens), g in (-1e-5, 0); CP16 -> 8,192-token "
                "fragments (128 chunks); CP8 fallback -> 16,384 (256 chunks)", data="iso_weak:b0=0.008:gmax=1e-5:gzero=0"),
}
# fmt: on

ROWS: dict[str, Row] = {**PACKS, **CORRECT, **ADVERSARIAL}


# ----------------------------------------------------------------------------------------------------------------
# Partitions.  A fragment is (doc, off, length, rank, slot).  Layout C/Z/R split the GLOBAL token order into
# intervals and then cut each interval at document boundaries; layout D splits each document into 2W pieces with
# jittered cut points (ragged) and gives rank r the pieces r and 2W-1-r.


@dataclass(frozen=True)
class Fragment:
    doc: int
    off: int  # offset of the fragment inside its document
    length: int
    rank: int
    slot: int  # order in the rank's local buffer

    @property
    def end(self) -> int:
        return self.off + self.length


def _intervals_to_fragments(
    cu: list[int], bounds: list[tuple[int, int, int]]
) -> list[Fragment]:
    """bounds: (global_start, global_end, rank) in the rank's local order; cut at document boundaries.  A zero-token
    rank gets no fragment."""
    frags: list[Fragment] = []
    slots: dict[int, int] = {}
    for gs, ge, rank in bounds:
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= gs:
            d += 1
        pos = gs
        while pos < ge and d < len(cu) - 1:
            e = min(ge, cu[d + 1])
            if e > pos:
                frags.append(
                    Fragment(d, pos - cu[d], e - pos, rank, slots.get(rank, 0))
                )
                slots[rank] = slots.get(rank, 0) + 1
            pos = e
            d += 1
    return frags


def near_equal(T: int, n: int) -> list[tuple[int, int]]:
    """n near-equal intervals of [0, T): the first T % n intervals get one extra token (FLA's equal split when
    n divides T; Cake accepts the ragged remainder)."""
    base, extra = divmod(T, n)
    bounds, pos = [], 0
    for i in range(n):
        ln = base + (1 if i < extra else 0)
        bounds.append((pos, pos + ln))
        pos += ln
    return bounds


def partition_C(row: Row, W: int) -> list[Fragment]:
    return _intervals_to_fragments(
        row.cu_seqlens, [(s, e, r) for r, (s, e) in enumerate(near_equal(row.total, W))]
    )


def partition_Z(row: Row, W: int) -> list[Fragment]:
    pieces = near_equal(row.total, 2 * W)
    bounds = []
    for r in range(W):
        bounds.append((*pieces[r], r))
        bounds.append((*pieces[2 * W - 1 - r], r))
    return _intervals_to_fragments(row.cu_seqlens, bounds)


def partition_R(row: Row, W: int, t_r: list[int]) -> list[Fragment]:
    assert len(t_r) == W and sum(t_r) == row.total, (row.name, t_r, row.total)
    bounds, pos = [], 0
    for r, n in enumerate(t_r):
        bounds.append((pos, pos + n, r))
        pos += n
    return _intervals_to_fragments(row.cu_seqlens, bounds)


def doc_cuts_D(row: Row, W: int, doc: int) -> list[int]:
    """2W+1 cut offsets inside document `doc`: equal spacing with a deterministic jitter of up to +-6 % of the piece
    length, never landing on a multiple of 64 (unless the document is too short for jitter), monotone, first 0, last len."""
    n = row.lengths[doc]
    k = 2 * W
    rng = random.Random(f"{SEED}:{row.name}:D:{W}:{doc}")
    base = [round(i * n / k) for i in range(k + 1)]
    cuts = [0]
    for i in range(1, k):
        piece = n / k
        jit = int(rng.uniform(-0.06, 0.06) * piece)
        c = base[i] + jit
        c = max(cuts[-1], min(n, c))
        if n >= 64 * k and c % 64 == 0:
            c += 1
        c = max(cuts[-1], min(n, c))
        cuts.append(c)
    cuts.append(n)
    return cuts


def partition_D(row: Row, W: int) -> list[Fragment]:
    frags: list[Fragment] = []
    slots = [0] * W
    for d in range(len(row.lengths)):
        cuts = doc_cuts_D(row, W, d)
        pieces = [(cuts[i], cuts[i + 1] - cuts[i]) for i in range(2 * W)]
        for r in range(W):
            for i in (r, 2 * W - 1 - r):
                off, length = pieces[i]
                frags.append(Fragment(d, off, length, r, slots[r]))
                slots[r] += 1
    # local order on a rank: by document, then by offset (already the case: pieces r < 2W-1-r)
    return frags


def partition(
    row: Row, W: int, layout: str, ragged_key: str | None = None
) -> list[Fragment]:
    if layout == "C":
        return partition_C(row, W)
    if layout == "Z":
        return partition_Z(row, W)
    if layout == "D":
        return partition_D(row, W)
    if layout == "R":
        return partition_R(row, W, RAGGED_TABLES[(ragged_key or row.name, W)])
    raise ValueError(layout)


def links(frags: list[Fragment]) -> dict[tuple[int, int], dict]:
    """(rank, slot) -> {"prev": (rank, slot) | None, "next": (rank, slot) | None, "crossing_in": bool, "crossing_out": bool}.
    Predecessor = the fragment of the same document ending where this one starts (zero-length fragments are
    skipped in the chain)."""
    by_doc: dict[int, list[Fragment]] = {}
    for f in frags:
        by_doc.setdefault(f.doc, []).append(f)
    out: dict[tuple[int, int], dict] = {}
    for fs in by_doc.values():
        fs = sorted(fs, key=lambda f: (f.off, f.length == 0))
        chain = [f for f in fs if f.length > 0]
        for i, f in enumerate(chain):
            prev = chain[i - 1] if i > 0 else None
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            out[(f.rank, f.slot)] = {
                "prev": (prev.rank, prev.slot) if prev else None,
                "next": (nxt.rank, nxt.slot) if nxt else None,
                "crossing_in": prev is not None,
                "crossing_out": nxt is not None,
            }
        for f in fs:
            if f.length == 0:
                out[(f.rank, f.slot)] = {
                    "prev": None,
                    "next": None,
                    "crossing_in": False,
                    "crossing_out": False,
                }
    return out


def local_cu_seqlens(frags: list[Fragment], rank: int) -> list[int]:
    cu = [0]
    for f in sorted((f for f in frags if f.rank == rank), key=lambda f: f.slot):
        cu.append(cu[-1] + f.length)
    return cu


def comm_bytes(frags: list[Fragment], H: int) -> dict:
    """Crossing fragments and the minimal cross-rank bytes: one FP32 [H,128,128] state per crossing fragment and
    direction (forward S_in, backward dS_out); summary form (M, B) doubles the forward bytes."""
    lk = links(frags)
    crossing_in = sum(1 for v in lk.values() if v["crossing_in"])
    state = H * D * D * 4
    return {
        "crossing_fragments_in": crossing_in,
        "state_bytes": state,
        "fwd_state_bytes": crossing_in * state,
        "fwd_summary_bytes": crossing_in * 2 * state,
        "bwd_adjoint_bytes": crossing_in * state,
    }
