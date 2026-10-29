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

Metadata layer of the context-parallel (KCP) route of the Cake packed KDA
training backend: the partition of one packed batch of documents over the
ranks of a context-parallel group, the per-rank context (fragment table,
crossing lists, exchange splits, merge chains, the chunk / sequence / segment
tables of the existing stages, the causal-convolution halo plan) and the byte
queries.  Pure host metadata: nothing here reads the contents of a tensor or
launches a kernel; the module imports ``dataclasses``, ``os``, ``torch`` and
``torch.distributed`` only and is importable and testable on the CPU.

Partition.  ``kcp_partition(cu_seqlens_global_cpu, world_size, layout, fragments=None)``
cuts the global token order (``cu_seqlens`` of the packed documents, CPU
int64) into per-rank fragments ``(doc, offset in the document, length)``
stored on every rank in document order, then offset order (the rank's local
buffer order).  ``layout="contiguous"`` gives near-equal token intervals (no
``total % world_size`` requirement), ``"zigzag"`` ``2W`` near-equal parts with
rank ``r`` holding parts ``r`` and ``2W-1-r`` (``[front; back]``),
``"zigzag_per_document"`` cuts every document into ``2W`` pieces at the
caller's cut points (``cut_points``, default near-equal) and gives rank ``r``
the pieces ``r`` and ``2W-1-r`` of every document; ``fragments=`` takes an
explicit (ragged) table and validates it.  Document chains link the non-empty
fragments of a document in offset order (``prev`` / ``next`` as
``(rank, slot)``); a fragment with a predecessor is *crossing-in* (its
recurrence starts from a carried state), one with a successor is
*crossing-out* (it produces a summary).  Zero-length fragments are skipped by
the chains.  The partition is built once and reused across layers.

Context.  ``build_kcp_context(part, group, rank, device, conv_kernel_size=4)``
derives, for one rank, the local ``cu_seqlens`` (int32 device; one local
sequence per fragment), the device fragment table, the crossing slots, the
chunk / sequence tables of the existing chunk-parallel stages (bitwise the
output of the ``table`` program, built here on the CPU), the convolution halo
plan (which predecessor rows feed the first ``conv_kernel_size - 1`` outputs
of a continuing fragment, and the reverse routing of their ``dpre`` rows) and
the byte counters.  Everything that depends on the head count ``H`` (head
ownership ``Hg = ceil(H / W)``, ``owner(h) = h // Hg``, the split sizes of the
collectives, the merge chain tables, the segment plan ``P`` / ``NSEG`` and the
segment table) is derived on demand by ``ctx.heads(H)`` and cached.

Segment plan rule.  ``kcp_plan(n_frag, NC_max_frag, H, sm_count)`` picks the
segments per fragment ``P`` (``NSEG = n_frag * P``) with the shipped
``"wave_fill1_p1tie"`` rule (the one-wave fill rule plus the P = 1 tie line, on top of the
earlier ``"wave"`` rule): on a rank
with at most ``WAVE_N_FRAG_MAX = 3`` fragments (N_MAX) whose longest fragment
exceeds the chain cap (``NC_max_frag >= WAVE_MIN_NC_MAX_FRAG = 257`` chunks, i.e.
``P_chain >= 2``), the argmin over ``P in [P_chain, min(64, NC_max_frag)]`` of the
equal-segment wave cost
``ceil(n_frag P H / SM) * ceil(NC_max_frag / P) * t_walk + (P - 1) * t_compose``
(constants ``WAVE_RULE_10NS``, smallest ``P`` on ties; the earlier ``"wave"`` rule); on a
one-fragment rank below the
cap (``0 < NC_max_frag < 257``) the one-wave fill ``clamp(floor(SM / H), P_chain,
min(64, NC_max_frag))`` -- the largest ``P`` whose ``P H`` CTAs run one CTA per SM
instead of the occ ``ceil`` (the one-wave fill rule, :data:`WAVE_FILL_N_FRAG_MAX`; a rank with
``H > SM`` keeps the occ ``P``); every other rank (two or three fragments below
the cap, more than three fragments, or an empty longest fragment) keeps the
earlier ``"occ"`` rule ``clamp(max(P_occ, P_chain), 1, ...)`` (the occupancy rule); and -- the
P = 1 tie line -- on a two- or three-fragment rank below the cap whose ``P = 1``
walks run one wave (``n_frag H <= SM``) with an LPT makespan over the rank's
ACTUAL fragment chunk counts (:func:`p1_tie_steps`, the wave model's waves x steps
generalised to unequal fragments) not above the makespan at the base rule's ``P``,
``P = 1`` (one chain per fragment, no segment composition; :data:`P1_TIE_BASE`).
``"wave_fill1"`` (the previous rule), ``"wave"`` (the earlier rule), ``"occ"`` and
``"wave_fill"`` (the fill on every rank with at most three fragments; implemented,
not adopted) stay selectable through the private test knob
``_kcp_test_knobs={"p_rule": ...}`` (after :func:`enable_test_knobs`) for the A/B
record.  Every rule is a pure function of (fragment table, H, SM count).  A ``P``
change alters the segment recomposition (not bitwise in general);
for the tie line the bitwise proof set found every changed gate row
bitwise identical between ``P = 2`` and ``P = 1`` on their own operands (the
composition is exact there), re-proven on the merged tree.

Exchange layouts (one ``all_to_all_single`` per direction; ``s`` = one FP32
``[128, 128]`` state of one head = 16384 elements; owner ``j`` holds heads
``[j Hg, j Hg + Hg_j)`` with ``Hg_j = clamp(H - j Hg, 0, Hg)``):

* c1 (summaries to owners): sender ``i`` packs ``[dst j][out_slot][hg < Hg_j][2][s]``
  (half 0 = the offset ``B_f``, half 1 = the transposed map ``M_f^T``); owner
  ``j`` receives ``[src i][out_slot of i][hg][2][s]``.
* c2 (initial states home): owner ``j`` packs ``[dst i][in_slot of i][hg][s]``;
  the home rank receives ``S_in`` as ``[src j][in_slot][hg][s]`` (head-group major).
* b1 (adjoint offsets to owners): ``[dst j][in_slot][hg][s]`` followed, when the
  map is re-sent (saved mode without ``save_transfer``), by ``[out_slot][hg][s]``
  of ``M_f^T``; b2 (``dS_out`` home): the mirror of c2 with ``in_slot -> out_slot``.
* halo_fwd / halo_bwd: rows of ``C`` channels (BF16 forward, FP32 backward),
  one row per (consumer fragment, history / tail position), grouped by peer.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.distributed as dist

CHUNK = 64  # tokens per chunk of the existing stages
HEAD_DIM = 128  # head dimension of q, k, v, g
TILE = HEAD_DIM * HEAD_DIM  # elements of one FP32 [128, 128] state of one head
STATE_TILE_BYTES = TILE * 4  # 65536 B
MAX_FRAGS = 65536  # the layout table's 16-step document search (cake_backend.MAX_DOCS)
MAX_CHAIN_STEPS = 256  # kcp_plan: no walk chain longer than 256 chunk steps
MAX_SEGMENTS = 64  # kcp_plan: at most 64 segments per fragment
P_BAL_INACTIVE = 64  # kcp_plan: P_bal until a measurement fixes t_step / t_compose
P_RULES = (
    "occ",
    "wave",
    "wave_fill1",
    "wave_fill",
    "wave_fill1_p1tie",
)  # kcp_plan P rules: "wave_fill1_p1tie" = the shipped rule = "wave_fill1" + P = 1
# on two- / three-fragment ranks below the chain cap whose P = 1 LPT walk makespan over the actual fragment chunk counts is <= the
# base rule's (p1_tie_steps; one wave: n_frag H <= SM); "wave_fill1" = the previous shipped rule = "wave"
# + the one-wave fill on one-fragment ranks below the chain cap; "wave" = the earlier rule; "occ" = the design note §7.2
# rule; "wave_fill" = the fill on every rank with n_frag <= 3 (the second knob value; implemented, not adopted)
# Shipped P rule (the one-wave fill rule plus the P = 1 tie line, on top of the earlier
# wave rule): the public API plans with it; "wave_fill1", "wave", "occ" and
# "wave_fill" stay selectable through the private test knob ``_kcp_test_knobs={"p_rule": ...}`` for the A/B
# record.  Every rule is a pure function of (fragment table, H, SM count), so G3 holds under any of them.
P_RULE_DEFAULT = "wave_fill1_p1tie"  # shipped; "wave_fill1" = the previous default, via _kcp_test_knobs only
# The shipped default of ``save_transfer`` (the design note §6.2) -- the merge owner
# keeps its c1 receive tensor (and the home rank its M_seg at P > 1) for the backward instead of re-sending M_f^T in b1.
# Default of the byte queries below and of ``cake_kcp.KcpOptions``; ``False`` (the earlier recompute form) stays
# selectable through the ``save_transfer`` argument; ``kcp_saved_bytes`` reports the delta.
SAVE_TRANSFER_DEFAULT = True
# Wave-aware P rule: per-CTA-step constants at one CTA per SM on GB300, H = 64, in 10 ns units (exact integer arithmetic
# keeps the argmin and its ties identical on every rank), measured on the CP4 / CP8 profiles and the walk / merge
# micro-benchmarks: t_tape 2.85 us (the K1 tape step), t_xfer 3.5 us (the K3 transfer step), t_offset 1.2 us (the K1
# offset step), t_compose 20.4 us (the K4 map-mode step at 16 columns).
WAVE_RULE_10NS = {"tape": 285, "xfer": 350, "offset": 120, "compose": 2040}
# N_MAX (a plan constant, part of the pure function of the fragment table, H and SM count): the
# wave rule applies to a rank with at most this many fragments; ranks with more keep P_occ (the measured
# many-fragment D ranks lose 9-11 % at any P > 1: the compose / staging cost at P > 1 exceeds the model; D ranks
# of a multi-document pack have >= 4 fragments, two per document, so D keeps the bitwise "occ" program).
WAVE_N_FRAG_MAX = 3
# Candidate (iv): the wave rule applies only to ranks whose longest fragment exceeds the chain cap (P_chain >= 2): on
# fragments of <= 256 chunks (Q1 CP16: 244) the extra segments' compose / staging cost exceeds the shorter walks
# (measured CP16 C +5.0 % B1 / +6.4 % B3, CP16 Z +7.3 % B3 at the unconditioned wave P).
WAVE_MIN_NC_MAX_FRAG = MAX_CHAIN_STEPS + 1
# The one-wave fill rules: "wave_fill1_p1tie" is the shipped default ("wave_fill1" plus the P = 1 tie line; "wave_fill1"
# was the previous default), "wave_fill" stays a test knob value.  Below
# the chain cap (NC_max_frag < WAVE_MIN_NC_MAX_FRAG) the earlier "wave" rule keeps the occ P = ceil(SM / (n_frag H)), which on the one-fragment CP16 C ranks (NC 222-256, H = 64) launches 192 CTAs on
# 152 SMs (40 SMs host two CTAs: 148-172 step-times for 74-86-step chains); the fill rule takes the LARGEST P whose
# n_frag P H CTAs run one CTA per SM, floor(SM / (n_frag H)) (= 2 -> 128 CTAs, one wave of 111-128 steps, the floor's
# min_P), clamped to [P_chain, min(64, NC_max_frag)]; a rank without a one-wave P (n_frag H > SM) keeps the occ P.
# "wave_fill1" applies it to one-fragment ranks only (the shipped rule), "wave_fill" to every rank with
# n_frag <= WAVE_N_FRAG_MAX (two-fragment H = 64 ranks move P 2 -> 1).  Above the cap both are the shipped wave argmin.
# At H <= 4 on 152 SMs floor(SM / H) equals ceil(SM / H) (38; 64-cap at H = 1), so every H <= 4 row keeps its program.
WAVE_FILL_N_FRAG_MAX = {"wave_fill1": 1, "wave_fill": WAVE_N_FRAG_MAX}
# The P = 1 tie line of the shipped rule:
# below the chain cap a two-fragment H = 64 rank
# keeps the occ P = ceil(152 / 128) = 2 (256 CTAs: two waves of ceil(NC / 2) steps) although the wave model scores P = 1
# (128 CTAs, one wave of NC steps) at the same or fewer walk step-times (111 vs 2 x 56 = 112 on Q2 CP16 Z) and P = 2 adds the
# segment-composition K4 launches (map before c1, carry_in / carry_start after c2, tmap / dcarry in the backward) plus the
# backward's offset re-walk (replay mode; the saved-mode backward reads the kept carry plane).  "wave_fill1_p1tie" =
# "wave_fill1", then P := 1 on a rank with 2 <= n_frag <= WAVE_N_FRAG_MAX,
# 0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG, P > 1 and n_frag H <= SM (P = 1 runs one wave) when the LPT makespan of the walks
# at P = 1 over the rank's actual fragment chunk counts is <= the makespan at the base rule's P (p1_tie_steps: the wave
# model's waves x steps generalised to unequal fragments through lpt_makespan; equal fragments reproduce it exactly).  The
# compose term then favours P = 1 as well.  Unequal fragments (CP16 C two-fragment ranks, e.g. [170, 59] chunks) keep P = 2:
# their P = 1 makespan is the long fragment alone.  At H <= 4 the base P fills the SMs with short chains, so no H <= 4 rank
# changes.  A P change alters the segment recomposition and is not bitwise in general;
# for THIS rule the bitwise proof set found every changed gate row (the CP16 Z,
# CP8 Z and CP4 D / Z rows at H = 64) bitwise identical between P = 2 and P = 1 on their own operands -- the
# segment composition is exact on them -- so the adoption needed no FP64 re-judge; the proof was re-run on the merged tree.
P1_TIE_BASE = {
    "wave_fill1_p1tie": "wave_fill1"
}  # p1tie rule -> the rule whose P it starts from
ALIGN = 256  # every table starts on a 256 B boundary (cake_backend.ALIGN)
LAYOUTS = ("contiguous", "zigzag", "zigzag_per_document")
EXPLICIT_LAYOUT = "explicit"
# The int32 tables of the KCP saved set, in buffer order: the four chunk tables of the existing chunk-parallel stages and
# the boundary kernels' segment table (``[NSEG + 1]``, their ``seq_chunk_start`` argument).  The existing backend's own
# fifth table is the block table of its segmented recurrences (``cake_backend.TABLE_NAMES[4]``, ``seg_table``), which
# the KCP route does not launch.
TABLE_NAMES = (
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seq_order",
    "seg_chunk_start",
)
COLLECTIVES = ("c1", "c2", "b1", "b2", "halo_fwd", "halo_bwd")
TILE_B = 0  # half 0 of a c1 summary tile [2][128][128]: the offset B_f
TILE_M = 1  # half 1: the transposed map M_f^T
TILE_D = 0  # half 0 of a b1 pair tile [2][128][128]: the adjoint offset d_f (half 1 = the re-sent M_f^T, TILE_M)
TILE_STRIDE_PLANE = 1  # kernel argument tile_stride: dense plane (c2 S_in, b2 dS_out, b1 offsets-only, carry, the P > 1 staging planes)
TILE_STRIDE_PAIR = 2  # interleaved pair (c1 [B_f | M_f^T], b1 [d_f | M_f^T] with maps)
SUMMARY_TILE = 2 * TILE  # elements of one c1 summary tile of one head
# The process's opt-in to the private ``_kcp_test_knobs`` hook and the test-only record set (``build.test.json``):
# :func:`enable_test_knobs` sets it; the package reads no environment variable.
_TEST_KNOBS_ENABLED = False
PRECISION_MODES = (
    0,
    1,
    2,
)  # the TF32 hi/lo walk's static knob: 0 the hi/lo form the transfer_walk knob selects, 1 explicit three-MMA (R1 test), 2 plain TF32 (optional cross-check)
FLAG_CROSSING_IN = 1
FLAG_CROSSING_OUT = 2
FLAG_CONTINUING = 4  # the fragment's first outputs read convolution history rows (document offset > 0)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(n: int, m: int) -> int:
    return _ceil_div(n, m) * m


# ----------------------------------------------------------------------------------------------------------------
# Partition


@dataclasses.dataclass(frozen=True)
class KcpFragment:
    """One fragment of the partition: ``[off, off + length)`` of document ``doc``, held by ``rank`` at local
    position ``slot`` (the rank's buffer order); ``prev`` / ``next`` are the ``(rank, slot)`` of the neighbours in
    the document chain (``None`` at a document start / end and for zero-length fragments, which the chain skips)."""

    doc: int
    off: int
    length: int
    rank: int
    slot: int
    prev: tuple[int, int] | None = None
    next: tuple[int, int] | None = None

    @property
    def end(self) -> int:
        return self.off + self.length

    @property
    def crossing_in(self) -> bool:
        return self.prev is not None

    @property
    def crossing_out(self) -> bool:
        return self.next is not None

    def as_tuple(
        self,
    ) -> tuple[int, int, int, tuple[int, int] | None, tuple[int, int] | None]:
        """The KCP contract §3.1 record ``(doc, offset, length, prev, next)``."""
        return (self.doc, self.off, self.length, self.prev, self.next)


@dataclasses.dataclass(frozen=True)
class KcpPartition:
    """The fragment tables of every rank (``fragments[rank][slot]``), the global ``cu_seqlens`` they cut and the
    layout that produced them (``"explicit"`` for a caller-given table)."""

    world_size: int
    layout: str
    cu_seqlens: tuple[int, ...]
    fragments: tuple[tuple[KcpFragment, ...], ...]

    @property
    def num_docs(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def total(self) -> int:
        return self.cu_seqlens[-1]

    def doc_length(self, doc: int) -> int:
        return self.cu_seqlens[doc + 1] - self.cu_seqlens[doc]

    def fragment(self, rank: int, slot: int) -> KcpFragment:
        return self.fragments[rank][slot]

    def rank_tokens(self, rank: int) -> int:
        return sum(f.length for f in self.fragments[rank])

    def local_cu_seqlens(self, rank: int) -> tuple[int, ...]:
        """The rank's local ``cu_seqlens`` (one local sequence per fragment, zero-length ones included)."""
        cu = [0]
        for f in self.fragments[rank]:
            cu.append(cu[-1] + f.length)
        return tuple(cu)

    def records(
        self, rank: int
    ) -> tuple[
        tuple[int, int, int, tuple[int, int] | None, tuple[int, int] | None], ...
    ]:
        """The KCP contract §3.1 per-rank table ``(doc, offset, length, prev, next)`` in local order."""
        return tuple(f.as_tuple() for f in self.fragments[rank])

    def chains(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Per document the ``(rank, slot)`` of its non-empty fragments in offset order (the document chain)."""
        by_doc: dict[int, list[KcpFragment]] = {}
        for frs in self.fragments:
            for f in frs:
                if f.length > 0:
                    by_doc.setdefault(f.doc, []).append(f)
        out = []
        for d in range(self.num_docs):
            chain = sorted(by_doc.get(d, []), key=lambda f: f.off)
            out.append(tuple((f.rank, f.slot) for f in chain))
        return tuple(out)

    def crossing_counts(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """``(n_in per rank, n_out per rank)``: crossing-in / crossing-out fragments of every rank."""
        n_in = tuple(sum(1 for f in frs if f.crossing_in) for frs in self.fragments)
        n_out = tuple(sum(1 for f in frs if f.crossing_out) for frs in self.fragments)
        return n_in, n_out

    def max_chain_fragments(self) -> int:
        """The longest document chain in fragments (``n_d_max`` of the exchange rule)."""
        return max((len(c) for c in self.chains()), default=0)

    def to_dict(self) -> dict:
        return {
            "world_size": self.world_size,
            "layout": self.layout,
            "cu_seqlens": list(self.cu_seqlens),
            "fragments": [[list(f.as_tuple()) for f in frs] for frs in self.fragments],
        }


def _as_cu_seqlens(cu_seqlens) -> tuple[int, ...]:
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.device.type != "cpu":
            raise ValueError(
                "kcp_partition: cu_seqlens is CPU metadata (the partition reads no device tensor); "
                f"got a tensor on {cu_seqlens.device}"
            )
        if cu_seqlens.dim() != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"kcp_partition: cu_seqlens must be a 1-D int32 / int64 tensor, got {tuple(cu_seqlens.shape)} "
                f"{cu_seqlens.dtype}"
            )
        values = tuple(int(v) for v in cu_seqlens.tolist())
    else:
        values = tuple(int(v) for v in cu_seqlens)
    if len(values) < 1 or values[0] != 0:
        raise ValueError(
            f"kcp_partition: cu_seqlens must start with 0 and hold num_docs + 1 entries, got {values[:4]}..."
        )
    for a, b in zip(values, values[1:], strict=False):
        if b < a:
            raise ValueError("kcp_partition: cu_seqlens must be non-decreasing")
    return values


def near_equal(total: int, n: int) -> list[tuple[int, int]]:
    """``n`` near-equal intervals of ``[0, total)``: the first ``total % n`` intervals hold one token more."""
    base, extra = divmod(total, n)
    bounds, pos = [], 0
    for i in range(n):
        ln = base + (1 if i < extra else 0)
        bounds.append((pos, pos + ln))
        pos += ln
    return bounds


def _intervals_to_entries(
    cu: tuple[int, ...], bounds: list[tuple[int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """``(global_start, global_end, rank)`` intervals in each rank's local order -> ``(doc, off, len, rank)`` entries,
    cut at document boundaries; a zero-token interval yields no fragment and zero-length documents none."""
    entries: list[tuple[int, int, int, int]] = []
    num_docs = len(cu) - 1
    for gs, ge, rank in bounds:
        d = 0
        while d < num_docs and cu[d + 1] <= gs:
            d += 1
        pos = gs
        while pos < ge and d < num_docs:
            e = min(ge, cu[d + 1])
            if e > pos:
                entries.append((d, pos - cu[d], e - pos, rank))
            pos = e
            d += 1
    return entries


def _default_cut_points(length: int, pieces: int) -> list[int]:
    """``pieces + 1`` near-equal cut offsets of a document: the first ``length % pieces`` pieces one token longer."""
    cuts = [0]
    for _s, e in near_equal(length, pieces):
        cuts.append(e)
    return cuts


def _validate_cut_points(cuts, length: int, pieces: int, doc: int) -> list[int]:
    values = [int(c) for c in cuts]
    if len(values) != pieces + 1 or values[0] != 0 or values[-1] != length:
        raise ValueError(
            f"kcp_partition: cut_points[{doc}] must hold {pieces + 1} offsets from 0 to the document length {length}, "
            f"got {len(values)} entries {values[:3]}...{values[-2:]}"
        )
    for a, b in zip(values, values[1:], strict=False):
        if b < a:
            raise ValueError(f"kcp_partition: cut_points[{doc}] must be non-decreasing")
    return values


def _entries_zigzag_per_document(
    cu: tuple[int, ...], W: int, cut_points
) -> list[tuple[int, int, int, int]]:
    num_docs = len(cu) - 1
    pieces = 2 * W
    if cut_points is not None and len(cut_points) != num_docs:
        raise ValueError(
            f"kcp_partition: cut_points must give one list per document ({num_docs}), got {len(cut_points)}"
        )
    entries: list[tuple[int, int, int, int]] = []
    for d in range(num_docs):
        n = cu[d + 1] - cu[d]
        cuts = (
            _default_cut_points(n, pieces)
            if cut_points is None
            else _validate_cut_points(cut_points[d], n, pieces, d)
        )
        for r in range(W):
            for i in (r, pieces - 1 - r):
                entries.append((d, cuts[i], cuts[i + 1] - cuts[i], r))
    return entries


def _explicit_entries(
    cu: tuple[int, ...], W: int, fragments
) -> list[tuple[int, int, int, int]]:
    """Validate a caller-given table.  Accepted forms: per rank ``[[(doc, off, len), ...], ...]`` (``world_size``
    lists in local order) or flat ``[(doc, off, len, rank), ...]`` (local order per rank = the given order)."""
    num_docs = len(cu) - 1
    items = list(fragments)
    nested = len(items) == W and all(
        isinstance(x, (list, tuple)) and all(isinstance(y, (list, tuple)) for y in x)
        for x in items
    )
    entries: list[tuple[int, int, int, int]] = []
    if nested:
        for rank, frs in enumerate(items):
            for y in frs:
                if len(y) != 3:
                    raise ValueError(
                        f"kcp_partition: fragments[{rank}] entries must be (doc, off, len), got {y!r}"
                    )
                entries.append((int(y[0]), int(y[1]), int(y[2]), rank))
    else:
        for y in items:
            if not isinstance(y, (list, tuple)) or len(y) != 4:
                raise ValueError(
                    f"kcp_partition: flat fragments entries must be (doc, off, len, rank), got {y!r}"
                )
            entries.append((int(y[0]), int(y[1]), int(y[2]), int(y[3])))
    for doc, off, ln, rank in entries:
        if not 0 <= doc < num_docs:
            raise ValueError(
                f"kcp_partition: fragment document {doc} out of range [0, {num_docs})"
            )
        if not 0 <= rank < W:
            raise ValueError(
                f"kcp_partition: fragment rank {rank} out of range [0, {W})"
            )
        if ln < 0 or off < 0 or off + ln > cu[doc + 1] - cu[doc]:
            raise ValueError(
                f"kcp_partition: fragment (doc {doc}, off {off}, len {ln}) does not fit document {doc} of "
                f"{cu[doc + 1] - cu[doc]} tokens"
            )
    # every document covered exactly once by its non-empty fragments, offsets monotone
    for d in range(num_docs):
        n = cu[d + 1] - cu[d]
        parts = sorted((off, ln) for doc, off, ln, _r in entries if doc == d and ln > 0)
        pos = 0
        for off, ln in parts:
            if off != pos:
                raise ValueError(
                    f"kcp_partition: document {d} is not covered exactly once (gap or overlap at offset {pos}, "
                    f"fragment starts at {off})"
                )
            pos = off + ln
        if pos != n:
            raise ValueError(
                f"kcp_partition: document {d} is covered up to {pos} of {n} tokens"
            )
    # the given per-rank order is the local buffer order: document then offset (ties: empty pieces keep their place)
    for rank in range(W):
        keys = [(doc, off) for doc, off, _ln, r in entries if r == rank]
        for a, b in zip(keys, keys[1:], strict=False):
            if b < a:
                raise ValueError(
                    f"kcp_partition: rank {rank}'s fragments must be listed in document then offset order "
                    f"(the local buffer order), got {a} before {b}"
                )
    return entries


def _finish(
    cu: tuple[int, ...], W: int, layout: str, entries: list[tuple[int, int, int, int]]
) -> KcpPartition:
    per_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(W)]
    for doc, off, ln, rank in entries:
        per_rank[rank].append((doc, off, ln))
    by_doc: dict[int, list[tuple[int, int, int, int]]] = {}
    for r, frs in enumerate(per_rank):
        for s, (doc, off, ln) in enumerate(frs):
            by_doc.setdefault(doc, []).append((off, ln, r, s))
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    nxt: dict[tuple[int, int], tuple[int, int] | None] = {}
    for lst in by_doc.values():
        chain = sorted((x for x in lst if x[1] > 0), key=lambda x: x[0])
        for i, (_off, _ln, r, s) in enumerate(chain):
            prev[(r, s)] = (chain[i - 1][2], chain[i - 1][3]) if i > 0 else None
            nxt[(r, s)] = (
                (chain[i + 1][2], chain[i + 1][3]) if i + 1 < len(chain) else None
            )
    fragments = tuple(
        tuple(
            KcpFragment(doc, off, ln, r, s, prev.get((r, s)), nxt.get((r, s)))
            for s, (doc, off, ln) in enumerate(frs)
        )
        for r, frs in enumerate(per_rank)
    )
    return KcpPartition(world_size=W, layout=layout, cu_seqlens=cu, fragments=fragments)


def kcp_partition(
    cu_seqlens_global_cpu,
    world_size: int,
    layout: str = "contiguous",
    fragments=None,
    *,
    cut_points=None,
) -> KcpPartition:
    """The per-rank fragment tables of one packed batch (the KCP contract §3.1).

    ``cu_seqlens_global_cpu``: the packed documents' ``cu_seqlens`` (CPU int64 tensor or a sequence of ints);
    ``world_size``: ranks of the context-parallel group; ``layout``: ``"contiguous"``, ``"zigzag"`` or
    ``"zigzag_per_document"`` (``cut_points``: per document the ``2 * world_size + 1`` cut offsets, default
    near-equal); ``fragments``: an explicit table (see :func:`_explicit_entries`) that replaces the layout.
    Fragments may have any length (zero included) and cut points need not be multiples of 64.
    """
    cu = _as_cu_seqlens(cu_seqlens_global_cpu)
    W = int(world_size)
    if W < 1:
        raise ValueError(f"kcp_partition: world_size must be >= 1, got {world_size}")
    if fragments is not None:
        return _finish(cu, W, EXPLICIT_LAYOUT, _explicit_entries(cu, W, fragments))
    if layout not in LAYOUTS:
        raise ValueError(
            f"kcp_partition: layout must be one of {LAYOUTS}, got {layout!r}"
        )
    total = cu[-1]
    if layout == "contiguous":
        if cut_points is not None:
            raise ValueError(
                "kcp_partition: cut_points apply to layout 'zigzag_per_document' only"
            )
        entries = _intervals_to_entries(
            cu, [(s, e, r) for r, (s, e) in enumerate(near_equal(total, W))]
        )
    elif layout == "zigzag":
        if cut_points is not None:
            raise ValueError(
                "kcp_partition: cut_points apply to layout 'zigzag_per_document' only"
            )
        pieces = near_equal(total, 2 * W)
        bounds = []
        for r in range(W):
            bounds.append((*pieces[r], r))
            bounds.append((*pieces[2 * W - 1 - r], r))
        entries = _intervals_to_entries(cu, bounds)
    else:
        entries = _entries_zigzag_per_document(cu, W, cut_points)
    return _finish(cu, W, layout, entries)


# ----------------------------------------------------------------------------------------------------------------
# CPU-built tables: the four chunk tables of the existing chunk-parallel stages (the ``table`` program's contract) and the
# boundary kernels' segment table


def chunk_geometry(T: int, num_seqs: int) -> tuple[int, int, int]:
    """``(NC_max, NC_pad, R)`` of the existing plan: the chunk bound ``ceil(T / 64) + num_seqs``, its even round-up
    (the table length and grid bound of the chunk-shaped programs) and the internal row count ``64 * NC_pad``."""
    nc_max = _ceil_div(T, CHUNK) + num_seqs
    nc_pad = nc_max + nc_max % 2
    return nc_max, nc_pad, nc_pad * CHUNK


def chunk_tables(cu_seqlens_local: tuple[int, ...]) -> dict[str, list[int]]:
    """``chunk_bos``, ``chunk_len`` (``NC_pad`` entries, pad chunks zero), ``seq_chunk_start`` (``N + 1``) and
    ``seq_order`` (``N``: sequences by chunk count descending, index ascending) of the local sequences, bitwise the
    ``kda_chunk_train_layout_table_v1`` output for ``num_chunks_max = NC_pad``."""
    cu = tuple(int(x) for x in cu_seqlens_local)
    N = len(cu) - 1
    T = cu[-1]
    _nc_max, nc_pad, _rows = chunk_geometry(T, N)
    counts = [_ceil_div(cu[s + 1] - cu[s], CHUNK) for s in range(N)]
    seq_chunk_start = [0]
    for n in counts:
        seq_chunk_start.append(seq_chunk_start[-1] + n)
    nc = seq_chunk_start[-1]
    if nc > nc_pad:
        raise ValueError(
            f"chunk_tables: real chunk count {nc} exceeds the bound {nc_pad}"
        )
    chunk_bos = [0] * nc_pad
    chunk_len = [0] * nc_pad
    for s in range(N):
        length = cu[s + 1] - cu[s]
        for i in range(counts[s]):
            c = seq_chunk_start[s] + i
            chunk_bos[c] = cu[s] + i * CHUNK
            chunk_len[c] = min(CHUNK, length - i * CHUNK)
    seq_order = sorted(range(N), key=lambda s: (-counts[s], s))
    return {
        "chunk_bos": chunk_bos,
        "chunk_len": chunk_len,
        "seq_chunk_start": seq_chunk_start,
        "seq_order": seq_order,
    }


def seg_chunk_start(seq_chunk_start: list[int], P: int) -> list[int]:
    """``seg_chunk_start[d * P + p] = seq_chunk_start[d] + (p * n_d) // P``, ``[N * P] = NC``: segment ``p`` of local
    sequence ``d`` is pseudo-sequence ``d * P + p`` -- the ``[NSEG + 1]`` segment table the boundary kernels (K1-K4) take
    as their ``seq_chunk_start`` argument (the form of the earlier segment-table program of the chunk-parallel stages)."""
    N = len(seq_chunk_start) - 1
    P = int(P)
    if P < 1:
        raise ValueError(f"seg_chunk_start: P must be >= 1, got {P}")
    out = [0] * (N * P + 1)
    for d in range(N):
        c0 = seq_chunk_start[d]
        n_d = seq_chunk_start[d + 1] - c0
        for p in range(P):
            out[d * P + p] = c0 + (p * n_d) // P
    out[N * P] = seq_chunk_start[N]
    return out


def table_layout(NC_pad: int, N: int, NSEG: int) -> tuple[tuple[str, int, int], ...]:
    """``(name, word offset, words)`` of the five tables inside one int32 buffer, each 256 B aligned: the four chunk
    tables as the existing backend lays them out (``cake_backend._table_layout`` without a segment table), then the
    boundary kernels' ``[NSEG + 1]`` segment table in the same form, always present on the KCP route.  The existing
    backend's fifth table (``seg_table``, ``NSEG + 2 * N + 1`` words: the block table of its segmented recurrences) is
    another object: the KCP route never launches those recurrences and never hands this buffer to them."""
    words_per_align = ALIGN // 4
    tables = [
        ("chunk_bos", NC_pad),
        ("chunk_len", NC_pad),
        ("seq_chunk_start", N + 1),
        ("seq_order", N),
        ("seg_chunk_start", NSEG + 1),
    ]
    out: list[tuple[str, int, int]] = []
    off = 0
    for name, words in tables:
        out.append((name, off, words))
        off = _round_up(off + words, words_per_align)
    return tuple(out)


def table_words(NC_pad: int, N: int, NSEG: int) -> int:
    _name, off, words = table_layout(NC_pad, N, NSEG)[-1]
    return _round_up(off + words, ALIGN // 4)


def tables_bytes(NC_pad: int, N: int, NSEG: int) -> int:
    """Bytes of the int32 table buffer the saved set carries (``4 * (2 NC_pad + 2 N + NSEG + 2)`` plus alignment)."""
    return table_words(NC_pad, N, NSEG) * 4


# ----------------------------------------------------------------------------------------------------------------
# Segment plan (the design note §7.2) and head ownership (SPEC-5)


@dataclasses.dataclass(frozen=True)
class KcpSegmentPlan:
    """``P`` segments per fragment and ``NSEG = n_frag * P`` pseudo-sequences, with the two terms the rule took."""

    P: int
    NSEG: int
    P_occ: int
    P_chain: int
    P_bal: int
    n_frag: int
    NC_max_frag: int
    p_rule: str = P_RULE_DEFAULT


def lpt_makespan(chains: "list[tuple[int, int]]", machines: int) -> int:
    """Greedy LPT list-scheduling makespan (longest jobs first, each job to the least-loaded machine) of jobs given
    as ``(length, count)`` groups on ``machines`` identical machines (one CTA per SM).  Exact: a group of ``count``
    equal jobs is placed by water filling, which is the greedy's result for identical jobs (it raises the least-loaded
    machine each time); integer arithmetic throughout, order of equal-length groups irrelevant."""
    machines = int(machines)
    loads = [0] * machines
    for L, m in sorted(chains, reverse=True):
        L, m = int(L), int(m)
        if L <= 0 or m <= 0:
            continue

        def needed(
            level: int,
        ) -> int:  # jobs that raise every machine to at least `level`
            return sum(-(-(level - load) // L) for load in loads if load < level)

        lo = min(loads)
        hi = lo + (m + 1) * L  # needed(hi) >= m + 1 from the least-loaded machine alone
        while lo < hi:  # the largest level reachable with at most m jobs
            mid = (lo + hi + 1) // 2
            if needed(mid) <= m:
                lo = mid
            else:
                hi = mid - 1
        counts = [(-(-(lo - load) // L) if load < lo else 0) for load in loads]
        rest = m - sum(counts)
        loads = [load + c * L for load, c in zip(loads, counts, strict=True)]
        if rest:  # the leftover jobs (fewer than the machines at `lo`) go to the least-loaded machines, one each
            for i in sorted(range(machines), key=loads.__getitem__)[:rest]:
                loads[i] += L
    return max(loads)


def wave_rule_cost_10ns(
    n_frag: int, NC_max_frag: int, H: int, sm_count: int, P: int
) -> int:
    """Cost model of ``P`` segments per fragment (10 ns units, exact integers; the rule of
    record): the three walks (K1 offset, K3 transfer, K1 tape; one CTA per SM, equal segments) each take
    ``ceil(n_frag P H / SM)`` waves of ``ceil(NC_max_frag / P)`` CTA-steps, plus ``P - 1`` K4 compose steps."""
    waves = _ceil_div(n_frag * P * H, sm_count)
    steps = _ceil_div(NC_max_frag, P)
    walk = WAVE_RULE_10NS["tape"] + WAVE_RULE_10NS["xfer"] + WAVE_RULE_10NS["offset"]
    return waves * steps * walk + (P - 1) * WAVE_RULE_10NS["compose"]


def wave_rule_cost_us(
    n_frag: int, NC_max_frag: int, H: int, sm_count: int, P: int
) -> float:
    return wave_rule_cost_10ns(n_frag, NC_max_frag, H, sm_count, P) / 100.0


def _wave_rule_P(
    n_frag: int, NC_max_frag: int, H: int, sm_count: int, p_chain: int
) -> int:
    """``argmin_P wave_rule_cost`` over ``P in [P_chain, min(64, NC_max_frag)]`` (smallest ``P`` on ties)."""
    hi = max(1, min(MAX_SEGMENTS, NC_max_frag))
    lo = min(p_chain, hi)
    best_P, best_cost = lo, None
    for P in range(lo, hi + 1):
        cost = wave_rule_cost_10ns(n_frag, NC_max_frag, H, sm_count, P)
        if best_cost is None or cost < best_cost:
            best_P, best_cost = P, cost
    return best_P


def wave_rule_lpt_cost_10ns(
    nc_frags: "list[int]", H: int, sm_count: int, P: int
) -> int:
    """Information form (b) (NOT the rule of record): the walks take the LPT makespan of
    ``P * H`` chains of ``ceil(NC_f / P)`` steps per fragment ``f`` with ``NC_f`` chunks (empty fragments: none) on
    ``sm_count`` SMs, plus ``P - 1`` compose steps.  Equal fragments reproduce :func:`wave_rule_cost_10ns`."""
    P, H = int(P), int(H)
    chains = [(-(-int(nc) // P), P * H) for nc in nc_frags if int(nc) > 0]
    steps = lpt_makespan(chains, sm_count) if chains else 0
    walk = WAVE_RULE_10NS["tape"] + WAVE_RULE_10NS["xfer"] + WAVE_RULE_10NS["offset"]
    return steps * walk + (P - 1) * WAVE_RULE_10NS["compose"]


def wave_rule_lpt_P(nc_frags: "list[int]", H: int, sm_count: int) -> int:
    """``argmin_P`` of :func:`wave_rule_lpt_cost_10ns` over ``P in [P_chain, min(64, max(nc_frags))]`` (smallest ``P``
    on ties; ``1`` when every fragment is empty).  Information only."""
    nc_frags = [int(x) for x in nc_frags]
    nc_max = max(nc_frags, default=0)
    if nc_max == 0:
        return 1
    hi = max(1, min(MAX_SEGMENTS, nc_max))
    lo = min(max(1, _ceil_div(nc_max, MAX_CHAIN_STEPS)), hi)
    best_P, best_cost = lo, None
    for P in range(lo, hi + 1):
        cost = wave_rule_lpt_cost_10ns(nc_frags, H, sm_count, P)
        if best_cost is None or cost < best_cost:
            best_P, best_cost = P, cost
    return best_P


def p1_tie_steps(nc_frags: "list[int]", H: int, sm_count: int, P: int) -> int:
    """CTA step-times (LPT makespan at one CTA per SM) of one equal-segment walk of ``P`` segments per fragment over a
    rank's fragments of ``nc_frags`` chunks: ``P * H`` chains of ``ceil(NC_f / P)`` steps per non-empty fragment ``f`` on
    ``sm_count`` SMs (:func:`lpt_makespan`).  Equal fragments reproduce the wave model's ``ceil(n_frag P H / SM) x
    ceil(NC / P)`` exactly; unequal fragments are charged their actual makespan (the "wave_fill1_p1tie" tie test)."""
    P, H = int(P), int(H)
    chains = [(_ceil_div(int(nc), P), P * H) for nc in nc_frags if int(nc) > 0]
    return lpt_makespan(chains, int(sm_count)) if chains else 0


def _p1_tie(
    n_frag: int,
    NC_max_frag: int,
    H: int,
    sm_count: int,
    P: int,
    nc_frags: "list[int] | None",
) -> int:
    """The "wave_fill1_p1tie" line: ``1`` when the rank has 2..WAVE_N_FRAG_MAX fragments below the chain cap, the base
    rule took ``P > 1``, ``P = 1`` runs one wave (``n_frag H <= SM``) and the P = 1 walk makespan over the fragment chunk
    counts does not exceed the base ``P``'s; otherwise the base ``P``.  ``nc_frags`` = the rank's fragment chunk counts
    (``None``: ``n_frag`` equal fragments of ``NC_max_frag`` chunks, the pure (n_frag, NC_max) form of the examples)."""
    if nc_frags is None:
        frags = [NC_max_frag] * n_frag
    else:
        frags = [int(x) for x in nc_frags]
        if (
            len(frags) != n_frag
            or max(frags, default=0) != NC_max_frag
            or min(frags, default=0) < 0
        ):
            raise ValueError(
                f"kcp_plan: nc_frags={frags} must list n_frag={n_frag} fragment chunk counts >= 0 with max NC_max_frag={NC_max_frag}"
            )
    if not (
        2 <= n_frag <= WAVE_N_FRAG_MAX
        and 0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG
        and P > 1
        and n_frag * H <= sm_count
    ):
        return P
    if p1_tie_steps(frags, H, sm_count, 1) <= p1_tie_steps(frags, H, sm_count, P):
        return 1
    return P


def kcp_plan(
    n_frag: int,
    NC_max_frag: int,
    H: int,
    sm_count: int,
    *,
    P_bal: int = P_BAL_INACTIVE,
    p_rule: str = P_RULE_DEFAULT,
    nc_frags: "list[int] | None" = None,
) -> KcpSegmentPlan:
    """The KCP plan rule (``p_rule="wave_fill1_p1tie"``, the shipped default: the one-wave fill on the earlier
    ``"wave"`` rule plus the P = 1 tie line; ``"wave_fill1"`` = the previous default, the fill without the tie
    line).  The ``"wave"`` rule::

        P_occ   = ceil(SM / (n_frag * H))                 # one transfer-walk CTA per SM
        P_chain = ceil(NC_max_frag / 256)                 # no walk chain longer than 256 steps
        P_bal   = 64 (inactive until measured)
        P_wave  = argmin over P in [P_chain, min(64, NC_max_frag)] of wave_rule_cost(P)   # smallest P on ties
        P       = P_wave                                 if n_frag <= WAVE_N_FRAG_MAX (3) and NC_max_frag >= WAVE_MIN_NC_MAX_FRAG (257)
                = clamp(max(P_occ, P_chain), 1, min(64, NC_max_frag, max(P_chain, P_bal)))   otherwise (occ)
        NSEG    = n_frag * P

    with ``wave_rule_cost(P) = ceil(n_frag P H / SM) x ceil(NC_max_frag / P) x (t_tape + t_xfer + t_offset) +
    (P - 1) x t_compose`` (:func:`wave_rule_cost_10ns`, constants :data:`WAVE_RULE_10NS`): the three equal-segment
    walks at one CTA per SM plus the K4 compose chain.  The wave line applies only to ranks with
    ``n_frag <= WAVE_N_FRAG_MAX`` (3) whose longest fragment exceeds the chain cap (``NC_max_frag >=
    WAVE_MIN_NC_MAX_FRAG`` = 257 chunks, i.e. ``P_chain >= 2``); every other rank keeps the occ ``P``
    (many-fragment D ranks and short fragments measured slower at every ``P > 1``).  ``p_rule="occ"`` (the earlier
    §7.2 occupancy rule, selectable only through the ``p_rule`` test knob for the A/B record) uses the ``clamp`` line on every
    rank.  ``P_occ`` / ``P_chain`` / ``P_bal`` are reported under every rule.

    The one-wave fill rule (``"wave_fill1"`` the previous shipped rule, ``"wave_fill"`` test knob only; :data:`WAVE_FILL_N_FRAG_MAX`)::

        P_fill  = floor(SM / (n_frag * H))
        P       = clamp(P_fill, P_chain, min(64, NC_max_frag))   if p_rule in ("wave_fill1", "wave_fill") and
                                                                  n_frag <= WAVE_FILL_N_FRAG_MAX[p_rule] (1 | 3) and
                                                                  0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG and P_fill >= 1
                = the "wave" rule above                           otherwise

    i.e. below the chain cap the largest segment count that still runs one CTA per SM (``P_bal`` does not enter); ``p_rule="wave"`` (the earlier rule, selectable through the test knob) is the first
    block alone.

    The P = 1 tie rule ``"wave_fill1_p1tie"`` (the shipped rule)::

        P       = 1    if p_rule == "wave_fill1_p1tie" and 2 <= n_frag <= WAVE_N_FRAG_MAX and 0 < NC_max_frag < 257 and
                       P_wave_fill1 > 1 and n_frag H <= SM and
                       p1_tie_steps(nc_frags, H, SM, 1) <= p1_tie_steps(nc_frags, H, SM, P_wave_fill1)
                = the "wave_fill1" rule above                                                    otherwise

    with ``nc_frags`` the rank's fragment chunk counts (``None`` = ``n_frag`` equal fragments of ``NC_max_frag`` chunks;
    :func:`_build_head_plan` passes the actual counts) and :func:`p1_tie_steps` the LPT makespan of one equal-segment
    walk at one CTA per SM -- the wave model's ``waves x steps`` for equal fragments.  The segment-composition K4
    launches and the backward's offset re-walk (replay mode; the saved-mode backward reads the kept carry plane) exist
    only at ``P > 1``; the one-wave P = 1 walk of 128 CTAs runs at a
    higher per-step cost than the two-wave 256-CTA walk, which the GPU A/B measures.

    A pure function of ``(fragment table, H, SM count)``.  A rank without fragments has ``P = 1``, ``NSEG = 0``;
    when every fragment is empty (``NC_max_frag = 0``) the lower bound wins: ``P = 1``.
    """
    n_frag, NC_max_frag, H, sm_count = (
        int(n_frag),
        int(NC_max_frag),
        int(H),
        int(sm_count),
    )
    if n_frag < 0 or NC_max_frag < 0 or H < 1 or sm_count < 1:
        raise ValueError(
            f"kcp_plan: n_frag={n_frag}, NC_max_frag={NC_max_frag} must be >= 0 and H={H}, sm_count={sm_count} >= 1"
        )
    if p_rule not in P_RULES:
        raise ValueError(f"kcp_plan: p_rule must be one of {P_RULES}, got {p_rule!r}")
    if n_frag == 0:
        return KcpSegmentPlan(
            P=1,
            NSEG=0,
            P_occ=1,
            P_chain=1,
            P_bal=int(P_bal),
            n_frag=0,
            NC_max_frag=NC_max_frag,
            p_rule=p_rule,
        )
    p_occ = _ceil_div(sm_count, n_frag * H)
    p_chain = max(1, _ceil_div(NC_max_frag, MAX_CHAIN_STEPS))
    base_rule = P1_TIE_BASE.get(
        p_rule, p_rule
    )  # "wave_fill1_p1tie" starts from the "wave_fill1" P
    fill_n_frag_max = WAVE_FILL_N_FRAG_MAX.get(base_rule)
    p_fill = sm_count // (n_frag * H)
    if (
        fill_n_frag_max is not None
        and n_frag <= fill_n_frag_max
        and 0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG
        and p_fill >= 1
    ):
        P = max(p_chain, min(MAX_SEGMENTS, NC_max_frag, p_fill))
    elif (
        base_rule != "occ"
        and n_frag <= WAVE_N_FRAG_MAX
        and NC_max_frag >= WAVE_MIN_NC_MAX_FRAG
    ):
        P = (
            1
            if NC_max_frag == 0
            else _wave_rule_P(n_frag, NC_max_frag, H, sm_count, p_chain)
        )
    else:
        upper = min(MAX_SEGMENTS, NC_max_frag, max(p_chain, int(P_bal)))
        P = max(1, min(max(p_occ, p_chain), upper))
    if p_rule in P1_TIE_BASE:
        P = _p1_tie(n_frag, NC_max_frag, H, sm_count, P, nc_frags)
    elif nc_frags is not None:
        _p1_tie(
            n_frag, NC_max_frag, H, sm_count, 1, nc_frags
        )  # validation only: the other rules ignore the counts
    return KcpSegmentPlan(
        P=P,
        NSEG=n_frag * P,
        P_occ=p_occ,
        P_chain=p_chain,
        P_bal=int(P_bal),
        n_frag=n_frag,
        NC_max_frag=NC_max_frag,
        p_rule=p_rule,
    )


def head_groups(H: int, world_size: int) -> tuple[int, int, tuple[int, ...]]:
    """``(Hg, n_owners, Hg_r per rank)``: ``Hg = ceil(H / W)``, owner ``r`` holds heads ``[r Hg, r Hg + Hg_r)`` with
    ``Hg_r = clamp(H - r Hg, 0, Hg)``; ranks ``>= ceil(H / Hg)`` own no heads (``H < W``: owners ``0 .. H-1``)."""
    H, W = int(H), int(world_size)
    if H < 1 or W < 1:
        raise ValueError(f"head_groups: H={H} and world_size={W} must be >= 1")
    Hg = _ceil_div(H, W)
    n_owners = _ceil_div(H, Hg)
    hg_r = tuple(max(0, min(Hg, H - r * Hg)) for r in range(W))
    return Hg, n_owners, hg_r


def owner_of_head(h: int, H: int, world_size: int) -> int:
    Hg, _n, _hg = head_groups(H, world_size)
    if not 0 <= h < H:
        raise ValueError(f"owner_of_head: head {h} out of range [0, {H})")
    return h // Hg


# ----------------------------------------------------------------------------------------------------------------
# Convolution halo plan (the design note §8.2): history rows of continuing fragments and the reverse routing of dpre rows


@dataclasses.dataclass(frozen=True)
class KcpHaloPlan:
    """This rank's halo plan for the width-``halo + 1`` causal convolution.

    Forward: every continuing fragment ``f`` (document offset ``> 0``, non-empty) reads the document tokens
    ``off - halo .. off - 1`` as history; ``hist_row_off[slot][k]`` is the row of token ``off - halo + k`` in the
    ``hist`` pool (the ``halo_fwd`` ``all_to_all_single`` output, grouped by source rank), ``-1`` = zero history.
    ``hist_index`` gathers this rank's rows into the send buffer (destination-major).  Backward: every fragment
    ``g`` with a successor needs the ``dpre`` rows of the document tokens ``end + 0 .. end + halo - 1`` (held by
    successor fragments): ``tail_row_off[slot][m]`` is their row in the ``dpre_tail`` pool, ``-1`` = none;
    ``head_row_off[slot][k][c]`` are the send-layout slots of this rank's head row ``k`` of fragment ``slot``
    (``halo`` slots; ``-1`` = no consumer; a row with several consumers is stored into every consumer's slot).  Row
    counts per peer are the split
    sizes in rows (multiply by ``C`` for elements)."""

    halo: int
    n_cont: int
    hist_valid: tuple[int, ...]
    hist_row_off: tuple[tuple[int, ...], ...]
    tail_row_off: tuple[tuple[int, ...], ...]
    head_row_off: tuple[tuple[tuple[int, ...], ...], ...]
    hist_index: tuple[int, ...]
    fwd_send_rows: tuple[int, ...]
    fwd_recv_rows: tuple[int, ...]
    bwd_send_rows: tuple[int, ...]
    bwd_recv_rows: tuple[int, ...]

    @property
    def n_hist_rows(self) -> int:
        return sum(self.fwd_recv_rows)

    @property
    def n_head_rows(self) -> int:
        return sum(self.bwd_send_rows)

    @property
    def n_tail_rows(self) -> int:
        return sum(self.bwd_recv_rows)


MAX_HEAD_CONSUMERS = 3  # send-layout slots per head dpre row for conv_kernel_size = 4 (head_row_off [n_frag, 3, 3]);
# a head token p is consumed by the fragments ending at p, p-1, ..., p-halo+1: at most `halo` of them (three
# consecutive 1-token predecessors, e.g. the C3 row of record at layout D), so the table holds `halo` slots.


def _halo_sources(part: KcpPartition, halo: int):
    """Global halo sources.  ``hist[(rank, slot)][k]`` = ``(src_rank, src_slot, row_in_src)`` of history row ``k`` of a
    continuing fragment (``None`` before the document start); ``tail[(rank, slot)][m]`` = ``(succ_rank, succ_slot,
    head_row_k)`` of the successor token ``end + m`` of a fragment with a successor (``None`` past the document end)."""
    chains: dict[int, list[KcpFragment]] = {}
    for frs in part.fragments:
        for f in frs:
            if f.length > 0:
                chains.setdefault(f.doc, []).append(f)
    for lst in chains.values():
        lst.sort(key=lambda f: f.off)

    def owner(doc: int, t: int) -> KcpFragment:
        for g in chains[doc]:
            if g.off <= t < g.end:
                return g
        raise ValueError(f"halo plan: document {doc} token {t} has no owning fragment")

    hist: dict[tuple[int, int], tuple] = {}
    tail: dict[tuple[int, int], tuple] = {}
    for frs in part.fragments:
        for f in frs:
            if f.length > 0 and f.off > 0:
                rows: list[tuple[int, int, int] | None] = []
                for k in range(halo):
                    t = f.off - halo + k
                    if t < 0:
                        rows.append(None)
                    else:
                        g = owner(f.doc, t)
                        rows.append((g.rank, g.slot, t - g.off))
                hist[(f.rank, f.slot)] = tuple(rows)
            if f.crossing_out:
                length = part.doc_length(f.doc)
                rows = []
                for m in range(halo):
                    p = f.end + m
                    if p >= length:
                        rows.append(None)
                    else:
                        s = owner(f.doc, p)
                        rows.append((s.rank, s.slot, p - s.off))
                tail[(f.rank, f.slot)] = tuple(rows)
    return hist, tail


def _halo_plan(
    part: KcpPartition, rank: int, halo: int, local_cu: tuple[int, ...]
) -> KcpHaloPlan:
    W = part.world_size
    hist, tail = _halo_sources(part, halo)
    mine = part.fragments[rank]
    n_frag = len(mine)
    hist_row_off = [[-1] * halo for _ in range(n_frag)]
    tail_row_off = [[-1] * halo for _ in range(n_frag)]
    head_row_off = [[[-1] * halo for _ in range(halo)] for _ in range(n_frag)]
    fwd_recv = [0] * W
    fwd_send = [0] * W
    bwd_recv = [0] * W
    bwd_send = [0] * W
    hist_index: list[int] = []
    # forward receive pool (grouped by source rank; within a group: consumer slot, then k)
    n = 0
    for src in range(W):
        for f in mine:
            rows = hist.get((rank, f.slot))
            if rows is None:
                continue
            for k, e in enumerate(rows):
                if e is not None and e[0] == src:
                    hist_row_off[f.slot][k] = n
                    n += 1
                    fwd_recv[src] += 1
    # forward send (destination-major; the same inner order as the destination's receive group of this source)
    for dst in range(W):
        for f in part.fragments[dst]:
            rows = hist.get((dst, f.slot))
            if rows is None:
                continue
            for e in rows:
                if e is not None and e[0] == rank:
                    hist_index.append(local_cu[e[1]] + e[2])
                    fwd_send[dst] += 1
    # backward receive pool (dpre_tail; grouped by source rank; within a group: consumer slot, then m)
    n = 0
    for src in range(W):
        for g in mine:
            rows = tail.get((rank, g.slot))
            if rows is None:
                continue
            for m, e in enumerate(rows):
                if e is not None and e[0] == src:
                    tail_row_off[g.slot][m] = n
                    n += 1
                    bwd_recv[src] += 1
    # backward send (dpre_head rows written straight into the send layout; a head row may feed up to `halo` consumers)
    n = 0
    for dst in range(W):
        for g in part.fragments[dst]:
            rows = tail.get((dst, g.slot))
            if rows is None:
                continue
            for e in rows:
                if e is not None and e[0] == rank:
                    slots = head_row_off[e[1]][e[2]]
                    free = [c for c in range(halo) if slots[c] == -1]
                    if not free:
                        raise ValueError(
                            f"build_kcp_context: head row {e[2]} of fragment {e[1]} on rank {rank} has more than "
                            f"{halo} consumers; this cannot happen (a head token has at most {halo} consuming "
                            "predecessor fragments): the halo plan is inconsistent"
                        )
                    slots[free[0]] = n
                    n += 1
                    bwd_send[dst] += 1
    hist_valid = tuple(
        min(halo, f.off) if (f.length > 0 and f.off > 0) else 0 for f in mine
    )
    return KcpHaloPlan(
        halo=halo,
        n_cont=sum(1 for v in hist_valid if v > 0),
        hist_valid=hist_valid,
        hist_row_off=tuple(tuple(r) for r in hist_row_off),
        tail_row_off=tuple(tuple(r) for r in tail_row_off),
        head_row_off=tuple(tuple(tuple(c) for c in r) for r in head_row_off),
        hist_index=tuple(hist_index),
        fwd_send_rows=tuple(fwd_send),
        fwd_recv_rows=tuple(fwd_recv),
        bwd_send_rows=tuple(bwd_send),
        bwd_recv_rows=tuple(bwd_recv),
    )


# ----------------------------------------------------------------------------------------------------------------
# Per-head-count plan: ownership, exchange splits, compact-buffer addressing, merge chains, segment tables


@dataclasses.dataclass(frozen=True, eq=False)
class KcpHeadPlan:
    """Everything of one rank's context that depends on the head count ``H`` (``ctx.heads(H)``).

    Ownership (SPEC-5): ``Hg = ceil(H / W)``, owner ``j`` holds the compact head set ``[j Hg, j Hg + Hg_j)``,
    ``Hg_j = clamp(H - j Hg, 0, Hg)``; ``Hg_r`` / ``head0`` are this rank's.  Every owner with heads holds ``Hg`` tile
    slots per fragment slot (``Hg_pad_by_rank``): the single partial owner (``padded_owner``, ``0 < Hg_j < Hg``, only
    when ``H % Hg != 0``) is padded by ``pad_tiles = n_owners Hg - H`` tiles per slot so that the kernels' two-integer
    addressing ``((h // Hg) * n + slot) * Hg + h % Hg`` is exact for every ``H``; the padding is counted in that
    owner's splits, never read, and reported in ``pad_numel`` (elements per split; all zero for ``H = 64``).
    ``full_groups`` (= no padding anywhere) and the explicit per-head tile tables are validation aids, not kernel ABI.
    Buffer forms (the kernels' ``tile_stride`` argument per buffer is ``tile_stride``): c1 and the b1 buffer
    with maps are interleaved PAIRS per ``(slot, head)`` — ``[B_f | M_f^T]`` (``TILE_B`` / ``TILE_M``) and
    ``[d_f | M_f^T]`` (``TILE_D`` / ``TILE_M``; the ``M`` half of a document's last fragment is never written or
    read); c2 (``S_in``), b2 (``dS_out``) and the offsets-only b1 buffer are dense planes.  Sender-side layout
    ``[dst owner j][slot][hg < Hg]`` (pairs: ``[2][128][128]`` each), owner-side ``[src rank i][slot][hg]``.  Split
    sizes are per peer rank in elements (FP32) of the flattened buffers; the buffer sizes are their sums (``numel``).
    Tile tables (units of ``TILE`` unless noted): ``out_tile[out_slot][h]`` (c1 send, units of ``SUMMARY_TILE``),
    ``h0_tile[in_slot][h]`` (``S_in`` = c2 receive), ``dht_tile[out_slot][h]`` (``dS_out`` = b2 receive),
    ``b1_d_tile_d[in_slot][h]`` (offsets-only b1 send), ``b1_d_tile_dm[in_slot][h]`` / ``b1_m_tile_dm[in_slot][h]``
    (the ``d_f`` and ``M_f^T`` halves of the b1 pair of a crossing-in fragment: ``2 * pair + TILE_D`` / ``+ TILE_M``).
    Segment tables of K1 / K2 / K3
    (``NSEG`` entries, segment ``p`` of fragment ``f`` is ``f * P + p``): ``h0_index`` (slot in ``S_in`` of segment 0
    of a crossing-in fragment, else ``-1``), ``carry_index`` (slot in the forward carry buffer of segments ``>= 1``,
    ``f * (P - 1) + p - 1``, else ``-1``), ``dht_index`` / ``dcarry_index`` (the adjoint mirrors: last segment of a
    crossing-out fragment -> ``dS_out`` slot; segments ``< P - 1`` -> adjoint carry slot), ``out_index`` (``P = 1``:
    the c1 send element of a crossing-out fragment, else ``-1``; ``P > 1``: the staging slot ``= segment``),
    ``walk_seg_offset`` (segments the offset walks visit) and ``walk_seg_tape`` (all).  ``h0_index_fused`` /
    ``dht_index_fused`` are the one-pool form of the same tables for the fused walks (``fused_walks``; the shipped
    form since an earlier revision; ``fused_walks=0`` through the test knob): at ``P > 1`` every segment's initial state (final adjoint) sits in the carry plane -- segment 0 of a
    crossing-in fragment in the extra slot ``n_frag (P - 1) + in_slot`` that the ``carry_in`` merge fills with its
    ``S_in`` tiles (``write_states``), the last segment of a crossing-out fragment in ``n_frag (P - 1) + out_slot``
    (``dcarry_cross`` publishes ``dS_out`` there), ``-1`` = the zero state of a document start (end); ``numel["carry_fused"]``
    is that plane (``+ max(n_in, n_out)`` tiles per head: the forward and the adjoint never hold the extra slots at the
    same time); at ``P = 1`` they equal ``h0_index`` / ``dht_index`` (``S_in`` / ``dS_out`` are read directly).  Merge chains of this rank as
    an owner (``Hg_r > 0``; empty otherwise), CSR over elements: forward ``fwd_chain_start`` / ``fwd_chain_doc`` with
    per element the element offsets (hg 0) of ``M^T`` and ``B`` in the c1 receive buffer (``fwd_m_off`` /
    ``fwd_b_off``) and of the produced ``S_in`` in the c2 send buffer (``fwd_out_off``); adjoint chains (document end
    -> start) with ``adj_d_off_d`` / ``adj_d_off_dm`` (``d_f`` in the b1 receive buffer without / with the re-sent
    maps), ``adj_m_off_b1`` (``M^T`` in the b1 pair; ``-1`` for the document's last fragment, whose map is never
    sent), ``adj_m_off_c1`` (``M^T`` in the c1 receive buffer: replay mode or ``save_transfer``) and ``adj_out_off``
    (``dS_out`` of the predecessor in the b2 send buffer).  ``hg_stride[table]`` is the element stride between
    consecutive heads of each offset table (``SUMMARY_TILE`` in the pair buffers, ``TILE`` in the planes).  Every
    offset is ``-1`` or a non-negative multiple of 4 floats (checked here, before any launch).
    ``device_tables`` holds the int32 / int64 device copies of the tables the kernels read."""

    H: int
    Hg: int
    n_owners: int
    Hg_r: int
    head0: int
    Hg_by_rank: tuple[int, ...]
    full_groups: bool
    Hg_pad_by_rank: tuple[int, ...]
    padded_owner: int
    pad_tiles: int
    pad_numel: dict[str, int]
    tile_stride: dict[str, int]
    hg_stride: dict[str, int]
    segments: KcpSegmentPlan
    seg_chunk_start: tuple[int, ...]
    walk_seg_offset: tuple[int, ...]
    walk_seg_tape: tuple[int, ...]
    h0_index: tuple[int, ...]
    carry_index: tuple[int, ...]
    dht_index: tuple[int, ...]
    dcarry_index: tuple[int, ...]
    out_index: tuple[int, ...]
    h0_index_fused: tuple[int, ...]
    dht_index_fused: tuple[int, ...]
    out_tile: tuple[tuple[int, ...], ...]
    h0_tile: tuple[tuple[int, ...], ...]
    dht_tile: tuple[tuple[int, ...], ...]
    b1_d_tile_d: tuple[tuple[int, ...], ...]
    b1_d_tile_dm: tuple[tuple[int, ...], ...]
    b1_m_tile_dm: tuple[tuple[int, ...], ...]
    splits: dict[str, tuple[int, ...]]
    numel: dict[str, int]
    fwd_chain_doc: tuple[int, ...]
    fwd_chain_start: tuple[int, ...]
    fwd_m_off: tuple[int, ...]
    fwd_b_off: tuple[int, ...]
    fwd_out_off: tuple[int, ...]
    adj_chain_doc: tuple[int, ...]
    adj_chain_start: tuple[int, ...]
    adj_d_off_d: tuple[int, ...]
    adj_d_off_dm: tuple[int, ...]
    adj_m_off_b1: tuple[int, ...]
    adj_m_off_c1: tuple[int, ...]
    adj_out_off: tuple[int, ...]
    device_tables: dict[str, torch.Tensor]

    @property
    def P(self) -> int:
        return self.segments.P

    @property
    def NSEG(self) -> int:
        return self.segments.NSEG

    def owner(self, h: int) -> int:
        return int(h) // self.Hg

    def b1_splits(self, with_maps: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """``(send, recv)`` split sizes of b1: with the re-sent ``M^T`` (saved mode, ``save_transfer=False``) or
        the offsets ``d_f`` only (replay mode, or the maps kept on the owner)."""
        key = "dm" if with_maps else "d"
        return self.splits[f"b1_send_{key}"], self.splits[f"b1_recv_{key}"]


def _prefix(values) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + v)
    return out


def _check_tile_offsets(name: str, values) -> None:
    """Every chain-table tile offset is ``-1`` (unused) or a non-negative multiple of 4 floats (16 B);
    anything else is rejected here, before any launch."""
    for i, v in enumerate(values):
        if v == -1:
            continue
        if v < 0 or v % 4 != 0:
            raise ValueError(
                f"{name}[{i}] = {v}: chain-table tile offsets must be -1 or non-negative multiples of 4 floats"
            )


def _build_head_plan(ctx: KcpContext, H: int) -> KcpHeadPlan:
    part = ctx.partition
    W, r = part.world_size, ctx.rank
    H = int(H)
    if H < 1:
        raise ValueError(f"heads: H must be >= 1, got {H}")
    if H * HEAD_DIM * CHUNK >= 2**31:
        raise ValueError(f"heads: H={H} overflows the kernels' per-chunk int32 offsets")
    if ctx.R * H * CHUNK >= 2**31:
        raise ValueError(
            f"heads: R={ctx.R} padded rows (T_r={ctx.T_r}, n_frag={ctx.n_frag}) x H={H} overflow the kernels' int32 "
            "row-head offsets: R * H * 64 must stay below 2**31"
        )
    Hg, n_owners, hg_by_rank = head_groups(H, W)
    Hg_r = hg_by_rank[r]
    # every owner with heads holds Hg tile slots per fragment slot; the single partial owner (0 < Hg_j < Hg,
    # only when H % Hg != 0) is padded to Hg so that the kernels' two-integer addressing is exact for every H.  The
    # padding tiles are counted in that owner's splits, never read, and reported separately (pad_numel).
    hg_pad_by_rank = tuple(Hg if x > 0 else 0 for x in hg_by_rank)
    Hg_pad_r = hg_pad_by_rank[r]
    full_groups = n_owners * Hg == H  # no padding anywhere (validation aid)
    padded_owner = next((j for j, x in enumerate(hg_by_rank) if 0 < x < Hg), -1)
    pad_tiles = n_owners * Hg - H
    n_in_by, n_out_by = ctx.n_in_by_rank, ctx.n_out_by_rank
    n_in, n_out = n_in_by[r], n_out_by[r]

    # -- split sizes (elements) per peer: sender side [dst owner j][slot][hg < Hg_pad_j], owner side
    # [src rank i][slot][hg].  c1 and b1-with-maps are interleaved pairs (SUMMARY_TILE per (slot, head));
    # c2, b2 and the offsets-only b1 are dense planes (TILE).
    def _split_sizes(hg_owner, hg_self):
        return {
            "c1_send": tuple(n_out * hg_owner[j] * SUMMARY_TILE for j in range(W)),
            "c1_recv": tuple(n_out_by[i] * hg_self * SUMMARY_TILE for i in range(W)),
            "c2_send": tuple(n_in_by[i] * hg_self * TILE for i in range(W)),
            "c2_recv": tuple(n_in * hg_owner[j] * TILE for j in range(W)),
            "b1_send_d": tuple(n_in * hg_owner[j] * TILE for j in range(W)),
            "b1_recv_d": tuple(n_in_by[i] * hg_self * TILE for i in range(W)),
            "b1_send_dm": tuple(n_in * hg_owner[j] * SUMMARY_TILE for j in range(W)),
            "b1_recv_dm": tuple(n_in_by[i] * hg_self * SUMMARY_TILE for i in range(W)),
            "b2_send": tuple(n_out_by[i] * hg_self * TILE for i in range(W)),
            "b2_recv": tuple(n_out * hg_owner[j] * TILE for j in range(W)),
        }

    splits = _split_sizes(hg_pad_by_rank, Hg_pad_r)
    compact = _split_sizes(hg_by_rank, Hg_r)
    pad_numel = {k: sum(splits[k]) - sum(compact[k]) for k in splits}
    bases = {k: _prefix(v) for k, v in splits.items()}

    # sender side: the two-integer form ((h // Hg) * n + slot) * Hg + h % Hg (exact for every H with the padding)
    def c1_send_tile(out_slot: int, h: int) -> int:  # units of SUMMARY_TILE
        j, hg = h // Hg, h % Hg
        return bases["c1_send"][j] // SUMMARY_TILE + out_slot * Hg + hg

    def s_in_tile(
        in_slot: int, h: int
    ) -> int:  # units of TILE, S_in = the c2 receive buffer
        j, hg = h // Hg, h % Hg
        return bases["c2_recv"][j] // TILE + in_slot * Hg + hg

    def ds_out_tile(
        out_slot: int, h: int
    ) -> int:  # units of TILE, dS_out = the b2 receive buffer
        j, hg = h // Hg, h % Hg
        return bases["b2_recv"][j] // TILE + out_slot * Hg + hg

    def b1_send_d_tile(
        in_slot: int, h: int
    ) -> int:  # units of TILE, the offsets-only b1 send plane
        j, hg = h // Hg, h % Hg
        return bases["b1_send_d"][j] // TILE + in_slot * Hg + hg

    def b1_send_pair(
        in_slot: int, h: int
    ) -> int:  # units of SUMMARY_TILE, the b1 send pairs [d_f | M_f^T]
        j, hg = h // Hg, h % Hg
        return bases["b1_send_dm"][j] // SUMMARY_TILE + in_slot * Hg + hg

    # owner side: element offsets at hg 0 of the chain tables (used only when Hg_r > 0, where Hg_pad_r == Hg)
    def c1_recv_elem(src: int, out_slot: int) -> int:
        return bases["c1_recv"][src] + out_slot * Hg * SUMMARY_TILE

    def c2_send_elem(dst: int, in_slot: int) -> int:
        return bases["c2_send"][dst] + in_slot * Hg * TILE

    def b1_recv_d_elem(src: int, in_slot: int) -> int:
        return bases["b1_recv_d"][src] + in_slot * Hg * TILE

    def b1_recv_pair_elem(src: int, in_slot: int) -> int:
        return bases["b1_recv_dm"][src] + in_slot * Hg * SUMMARY_TILE

    def b2_send_elem(dst: int, out_slot: int) -> int:
        return bases["b2_send"][dst] + out_slot * Hg * TILE

    mine = part.fragments[r]
    out_tile = tuple(
        tuple(c1_send_tile(ctx.out_slot[f.slot], h) for h in range(H))
        for f in mine
        if f.crossing_out
    )
    h0_tile = tuple(
        tuple(s_in_tile(ctx.in_slot[f.slot], h) for h in range(H))
        for f in mine
        if f.crossing_in
    )
    dht_tile = tuple(
        tuple(ds_out_tile(ctx.out_slot[f.slot], h) for h in range(H))
        for f in mine
        if f.crossing_out
    )
    b1_d_tile_d = tuple(
        tuple(b1_send_d_tile(ctx.in_slot[f.slot], h) for h in range(H))
        for f in mine
        if f.crossing_in
    )
    b1_d_tile_dm = tuple(
        tuple(2 * b1_send_pair(ctx.in_slot[f.slot], h) + TILE_D for h in range(H))
        for f in mine
        if f.crossing_in
    )
    b1_m_tile_dm = tuple(
        tuple(2 * b1_send_pair(ctx.in_slot[f.slot], h) + TILE_M for h in range(H))
        for f in mine
        if f.crossing_in
    )

    # -- segment plan and the K1 / K2 / K3 segment index tables
    nc_frags = [_ceil_div(f.length, CHUNK) for f in mine]
    nc_max_frag = max(nc_frags, default=0)
    seg = kcp_plan(
        ctx.n_frag, nc_max_frag, H, ctx.sm_count, p_rule=ctx.p_rule, nc_frags=nc_frags
    )
    P, NSEG = seg.P, seg.NSEG
    seg_table = (
        tuple(seg_chunk_start(list(ctx.seq_chunk_start), P)) if ctx.n_frag else (0,)
    )
    h0_index, carry_index, dht_index, dcarry_index, out_index, walk_offset = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for f in mine:
        for p in range(P):
            s = f.slot * P + p
            h0_index.append(ctx.in_slot[f.slot] if (p == 0 and f.crossing_in) else -1)
            carry_index.append(f.slot * (P - 1) + p - 1 if p >= 1 else -1)
            dht_index.append(
                ctx.out_slot[f.slot] if (p == P - 1 and f.crossing_out) else -1
            )
            dcarry_index.append(f.slot * (P - 1) + p if p < P - 1 else -1)
            if P == 1:
                out_index.append(ctx.out_slot[f.slot] if f.crossing_out else -1)
                if f.crossing_out:
                    walk_offset.append(s)
            else:
                out_index.append(s)
                walk_offset.append(s)
    walk_tape = list(range(NSEG))
    # the one-pool tables of the fused walks (fused_walks; the shipped form since an earlier revision, fused_walks=0 via the knob)
    extra = ctx.n_frag * (P - 1)
    h0_fused, dht_fused = [], []
    for f in mine:
        for p in range(P):
            s = f.slot * P + p
            if P == 1:
                h0_fused.append(h0_index[s])
                dht_fused.append(dht_index[s])
            else:
                h0_fused.append(
                    extra + ctx.in_slot[f.slot]
                    if (p == 0 and f.crossing_in)
                    else carry_index[s]
                )
                dht_fused.append(
                    extra + ctx.out_slot[f.slot]
                    if (p == P - 1 and f.crossing_out)
                    else dcarry_index[s]
                )

    # -- merge chains of this rank as an owner
    fwd_doc, fwd_start, fwd_m, fwd_b, fwd_out = [], [0], [], [], []
    adj_doc, adj_start, adj_dd, adj_ddm, adj_mb1, adj_mc1, adj_out = (
        [],
        [0],
        [],
        [],
        [],
        [],
        [],
    )
    if Hg_r > 0:
        for doc, chain in enumerate(ctx.chains):
            if len(chain) < 2:
                continue
            frs = [part.fragment(*rs) for rs in chain]
            fwd_doc.append(doc)
            for e, succ in zip(frs[:-1], frs[1:], strict=True):
                pair_base = c1_recv_elem(e.rank, ctx.out_slot_all[e.rank][e.slot])
                fwd_m.append(pair_base + TILE_M * TILE)
                fwd_b.append(pair_base + TILE_B * TILE)
                fwd_out.append(
                    c2_send_elem(succ.rank, ctx.in_slot_all[succ.rank][succ.slot])
                )
            fwd_start.append(len(fwd_m))
            adj_doc.append(doc)
            for i in range(len(frs) - 1, 0, -1):
                e, pred = frs[i], frs[i - 1]
                in_s = ctx.in_slot_all[e.rank][e.slot]
                adj_dd.append(b1_recv_d_elem(e.rank, in_s))
                pair = b1_recv_pair_elem(e.rank, in_s)
                adj_ddm.append(pair + TILE_D * TILE)
                if e.crossing_out:
                    out_s = ctx.out_slot_all[e.rank][e.slot]
                    adj_mb1.append(pair + TILE_M * TILE)
                    adj_mc1.append(c1_recv_elem(e.rank, out_s) + TILE_M * TILE)
                else:
                    adj_mb1.append(-1)
                    adj_mc1.append(-1)
                adj_out.append(
                    b2_send_elem(pred.rank, ctx.out_slot_all[pred.rank][pred.slot])
                )
            adj_start.append(len(adj_dd))

    offsets = {
        "fwd_m_off": fwd_m,
        "fwd_b_off": fwd_b,
        "fwd_out_off": fwd_out,
        "adj_d_off_d": adj_dd,
        "adj_d_off_dm": adj_ddm,
        "adj_m_off_b1": adj_mb1,
        "adj_m_off_c1": adj_mc1,
        "adj_out_off": adj_out,
    }
    for name_, values in offsets.items():
        _check_tile_offsets(name_, values)
    hg_stride = {
        "fwd_m_off": SUMMARY_TILE,
        "fwd_b_off": SUMMARY_TILE,
        "fwd_out_off": TILE,
        "adj_d_off_d": TILE,
        "adj_d_off_dm": SUMMARY_TILE,
        "adj_m_off_b1": SUMMARY_TILE,
        "adj_m_off_c1": SUMMARY_TILE,
        "adj_out_off": TILE,
    }
    tile_stride = {
        "c1": TILE_STRIDE_PAIR,
        "c2": TILE_STRIDE_PLANE,
        "b1_d": TILE_STRIDE_PLANE,
        "b1_dm": TILE_STRIDE_PAIR,
        "b2": TILE_STRIDE_PLANE,
        "staging": TILE_STRIDE_PLANE,
        "carry": TILE_STRIDE_PLANE,
    }
    numel = {k: sum(v) for k, v in splits.items()}
    # the P > 1 staging is two dense planes stage_b / stage_m of [NSEG, H, 128, 128] each (K1 / K2 write stage_b, K3 writes
    # stage_m, both at tile_stride 1) -- counted together as one `staging` term of NSEG * H * SUMMARY_TILE elements;
    # the carry buffer holds the dense inter-segment states
    numel["staging"] = (NSEG * H * SUMMARY_TILE) if P > 1 else 0
    numel["carry"] = (ctx.n_frag * (P - 1) * H * TILE) if P > 1 else 0
    numel["carry_fused"] = (
        ((extra + max(n_in, n_out)) * H * TILE) if P > 1 else 0
    )  # the fused walks' carry plane
    dev = ctx.device
    i32, i64 = torch.int32, torch.int64

    def t(values, dtype=i32):
        return torch.tensor(list(values), dtype=dtype, device=dev)

    device_tables = {
        "seg_chunk_start": t(seg_table),
        "h0_index": t(h0_index),
        "carry_index": t(carry_index),
        "dht_index": t(dht_index),
        "dcarry_index": t(dcarry_index),
        "out_index": t(out_index),
        "walk_seg_offset": t(walk_offset),
        "walk_seg_tape": t(walk_tape),
        "out_tile": torch.tensor(
            [list(x) for x in out_tile], dtype=i32, device=dev
        ).reshape(len(out_tile), H),
        "h0_tile": torch.tensor(
            [list(x) for x in h0_tile], dtype=i32, device=dev
        ).reshape(len(h0_tile), H),
        "dht_tile": torch.tensor(
            [list(x) for x in dht_tile], dtype=i32, device=dev
        ).reshape(len(dht_tile), H),
        "b1_d_tile_d": torch.tensor(
            [list(x) for x in b1_d_tile_d], dtype=i32, device=dev
        ).reshape(len(b1_d_tile_d), H),
        "b1_d_tile_dm": torch.tensor(
            [list(x) for x in b1_d_tile_dm], dtype=i32, device=dev
        ).reshape(len(b1_d_tile_dm), H),
        "b1_m_tile_dm": torch.tensor(
            [list(x) for x in b1_m_tile_dm], dtype=i32, device=dev
        ).reshape(len(b1_m_tile_dm), H),
        "fwd_chain_start": t(fwd_start),
        "fwd_chain_doc": t(fwd_doc),
        "fwd_m_off": t(fwd_m, i64),
        "fwd_b_off": t(fwd_b, i64),
        "fwd_out_off": t(fwd_out, i64),
        "adj_chain_start": t(adj_start),
        "adj_chain_doc": t(adj_doc),
        "adj_d_off_d": t(adj_dd, i64),
        "adj_d_off_dm": t(adj_ddm, i64),
        "adj_m_off_b1": t(adj_mb1, i64),
        "adj_m_off_c1": t(adj_mc1, i64),
        "adj_out_off": t(adj_out, i64),
    }
    return KcpHeadPlan(
        H=H,
        Hg=Hg,
        n_owners=n_owners,
        Hg_r=Hg_r,
        head0=r * Hg,
        Hg_by_rank=hg_by_rank,
        full_groups=full_groups,
        Hg_pad_by_rank=hg_pad_by_rank,
        padded_owner=padded_owner,
        pad_tiles=pad_tiles,
        pad_numel=pad_numel,
        tile_stride=tile_stride,
        hg_stride=hg_stride,
        segments=seg,
        seg_chunk_start=seg_table,
        walk_seg_offset=tuple(walk_offset),
        walk_seg_tape=tuple(walk_tape),
        h0_index=tuple(h0_index),
        carry_index=tuple(carry_index),
        dht_index=tuple(dht_index),
        dcarry_index=tuple(dcarry_index),
        out_index=tuple(out_index),
        h0_index_fused=tuple(h0_fused),
        dht_index_fused=tuple(dht_fused),
        out_tile=out_tile,
        h0_tile=h0_tile,
        dht_tile=dht_tile,
        b1_d_tile_d=b1_d_tile_d,
        b1_d_tile_dm=b1_d_tile_dm,
        b1_m_tile_dm=b1_m_tile_dm,
        splits=splits,
        numel=numel,
        fwd_chain_doc=tuple(fwd_doc),
        fwd_chain_start=tuple(fwd_start),
        fwd_m_off=tuple(fwd_m),
        fwd_b_off=tuple(fwd_b),
        fwd_out_off=tuple(fwd_out),
        adj_chain_doc=tuple(adj_doc),
        adj_chain_start=tuple(adj_start),
        adj_d_off_d=tuple(adj_dd),
        adj_d_off_dm=tuple(adj_ddm),
        adj_m_off_b1=tuple(adj_mb1),
        adj_m_off_c1=tuple(adj_mc1),
        adj_out_off=tuple(adj_out),
        device_tables=device_tables,
    )


# ----------------------------------------------------------------------------------------------------------------
# Context


class KcpSequence:
    """Debug-only collective sequence counter (ruling R7): ``next(name)`` numbers the collectives a process issues,
    so a debug build can assert one issue order per call across ranks.  Metadata only; never read by a kernel."""

    def __init__(self) -> None:
        self.count = 0
        self.log: list[str] = []

    def next(self, name: str) -> int:
        self.count += 1
        self.log.append(str(name))
        return self.count

    def reset(self) -> None:
        self.count = 0
        self.log.clear()


@dataclasses.dataclass(frozen=True, eq=False)
class KcpContext:
    """One rank's view of the partition: CPU ints / tuples plus small int32 device tables; built once, reused across
    layers; never reads a tensor's contents.

    Local problem: ``T_r`` tokens in ``n_frag`` local sequences (``cu_seqlens_local`` int32 device ``[n_frag + 1]``,
    ``frag_table`` int32 device ``[n_frag, 4]`` = ``(doc, off, len, flags)`` with ``FLAG_CROSSING_IN`` /
    ``FLAG_CROSSING_OUT`` / ``FLAG_CONTINUING``).  Crossing lists: ``in_slot[slot]`` / ``out_slot[slot]`` (``-1``
    when not crossing) number the rank's crossing-in / crossing-out fragments in local order (``n_in`` / ``n_out``;
    ``N_in`` / ``N_out`` are the group totals; ``*_by_rank`` and ``*_all`` the per-rank counts / tables every owner
    needs).  ``chains[doc]`` are the ``(rank, slot)`` of each document's non-empty fragments in order.  Tables of the
    existing stages (CPU-built; bitwise the ``table`` program's output): ``chunk_bos`` / ``chunk_len`` (``NC_pad``),
    ``seq_chunk_start`` (``n_frag + 1``), ``seq_order`` (``n_frag``), as tuples and as int32 device tensors in
    ``tables``; ``NC`` real chunks, ``NC_max`` / ``NC_pad`` / ``R`` the existing plan's geometry.  ``halo`` is the
    convolution halo plan (its device tables in ``halo_tables``: ``hist_row_off``, ``tail_row_off``, ``head_row_off``
    int32, ``hist_index`` int64).  ``heads(H)`` derives and caches the head-dependent plan (:class:`KcpHeadPlan`).
    ``sequence`` is the debug collective counter."""

    partition: KcpPartition
    rank: int
    world_size: int
    group: object
    device: torch.device
    conv_kernel_size: int
    sm_count: int
    T_r: int
    n_frag: int
    cu_seqlens_local_cpu: tuple[int, ...]
    cu_seqlens_local: torch.Tensor
    frag_table: torch.Tensor
    in_slot: tuple[int, ...]
    out_slot: tuple[int, ...]
    n_in: int
    n_out: int
    N_in: int
    N_out: int
    n_in_by_rank: tuple[int, ...]
    n_out_by_rank: tuple[int, ...]
    in_slot_all: tuple[tuple[int, ...], ...]
    out_slot_all: tuple[tuple[int, ...], ...]
    chains: tuple[tuple[tuple[int, int], ...], ...]
    n_d_max: int
    exchange_scheme: str
    NC: int
    NC_max: int
    NC_pad: int
    R: int
    chunk_bos: tuple[int, ...]
    chunk_len: tuple[int, ...]
    seq_chunk_start: tuple[int, ...]
    seq_order: tuple[int, ...]
    tables: dict[str, torch.Tensor]
    halo: KcpHaloPlan
    halo_tables: dict[str, torch.Tensor]
    p_rule: str = P_RULE_DEFAULT  # plan rule: "wave_fill1_p1tie" shipped; "wave_fill1" (previous) / "wave" / "occ" / "wave_fill" via the test knob
    sequence: KcpSequence = dataclasses.field(default_factory=KcpSequence, repr=False)
    _head_plans: dict[int, KcpHeadPlan] = dataclasses.field(
        default_factory=dict, repr=False
    )

    @property
    def fragments(self) -> tuple[KcpFragment, ...]:
        return self.partition.fragments[self.rank]

    @property
    def ring_eligible(self) -> bool:
        """The design's ring alternative applies when every document chain has at most three fragments."""
        return self.n_d_max - 1 <= 2

    def heads(self, H: int) -> KcpHeadPlan:
        """The head-dependent plan for ``H`` heads (cached per ``H``)."""
        H = int(H)
        hp = self._head_plans.get(H)
        if hp is None:
            hp = _build_head_plan(self, H)
            self._head_plans[H] = hp
        return hp

    def table_layout(self, H: int) -> tuple[tuple[str, int, int], ...]:
        """``(name, word offset, words)`` of the five tables in the saved set's int32 buffer for ``H`` heads."""
        return table_layout(self.NC_pad, self.n_frag, self.heads(H).NSEG)

    def describe(self, H: int | None = None) -> dict:
        """Plain-int summary for reports (optionally with the head plan of ``H``)."""
        out = {
            "rank": self.rank,
            "world_size": self.world_size,
            "layout": self.partition.layout,
            "T_r": self.T_r,
            "n_frag": self.n_frag,
            "cu_seqlens_local": list(self.cu_seqlens_local_cpu),
            "n_in": self.n_in,
            "n_out": self.n_out,
            "N_in": self.N_in,
            "N_out": self.N_out,
            "in_slot": list(self.in_slot),
            "out_slot": list(self.out_slot),
            "n_d_max": self.n_d_max,
            "exchange_scheme": self.exchange_scheme,
            "NC": self.NC,
            "NC_max": self.NC_max,
            "NC_pad": self.NC_pad,
            "R": self.R,
            "sm_count": self.sm_count,
            "p_rule": self.p_rule,
            "conv_kernel_size": self.conv_kernel_size,
            "halo": {
                "n_cont": self.halo.n_cont,
                "hist_valid": list(self.halo.hist_valid),
                "n_hist_rows": self.halo.n_hist_rows,
                "n_head_rows": self.halo.n_head_rows,
                "n_tail_rows": self.halo.n_tail_rows,
                "fwd_send_rows": list(self.halo.fwd_send_rows),
                "fwd_recv_rows": list(self.halo.fwd_recv_rows),
                "bwd_send_rows": list(self.halo.bwd_send_rows),
                "bwd_recv_rows": list(self.halo.bwd_recv_rows),
            },
        }
        if H is not None:
            hp = self.heads(H)
            out["heads"] = {
                "H": hp.H,
                "Hg": hp.Hg,
                "n_owners": hp.n_owners,
                "Hg_r": hp.Hg_r,
                "full_groups": hp.full_groups,
                "Hg_pad_by_rank": list(hp.Hg_pad_by_rank),
                "padded_owner": hp.padded_owner,
                "pad_tiles": hp.pad_tiles,
                "pad_numel": dict(hp.pad_numel),
                "tile_stride": dict(hp.tile_stride),
                "P": hp.P,
                "NSEG": hp.NSEG,
                "P_occ": hp.segments.P_occ,
                "P_chain": hp.segments.P_chain,
                "NC_max_frag": hp.segments.NC_max_frag,
                "splits": {k: list(v) for k, v in hp.splits.items()},
                "numel": dict(hp.numel),
                "fwd_chains": len(hp.fwd_chain_doc),
                "fwd_elements": len(hp.fwd_m_off),
                "adj_chains": len(hp.adj_chain_doc),
                "adj_elements": len(hp.adj_d_off_d),
            }
        return out


def _device_of(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def build_kcp_context(
    part: KcpPartition,
    group,
    rank: int,
    device,
    conv_kernel_size: int = 4,
    *,
    sm_count: int | None = None,
    _kcp_test_knobs: dict | None = None,
) -> KcpContext:
    """The :class:`KcpContext` of ``rank`` (the KCP contract §3.1).

    ``group`` is the caller's ``torch.distributed`` process group (or ``None`` on the CPU / in a single process);
    when a group is given and the default process group is initialized, its size and this process's rank are
    checked against ``part.world_size`` / ``rank``.  ``device`` hosts the int32 tables.  ``conv_kernel_size - 1``
    history rows feed a continuing fragment's first outputs.  ``sm_count`` drives the segment plan (default: the
    CUDA device's multiprocessor count; ``1`` on a non-CUDA device = no occupancy pressure, ``P = P_chain``).
    Out-of-range problems are rejected here, before any launch: ``n_frag > 65536`` (the layout table's document
    search), int32 overflows of the kernels' offsets (``heads(H)``).  ``_kcp_test_knobs`` is the private test hook
    (SPEC-11; accepted only after :func:`enable_test_knobs`): its ``p_rule`` selects the plan's ``P`` rule for this
    context (:data:`P_RULE_DEFAULT` = ``"wave_fill1_p1tie"`` without the knob;
    ``"wave_fill1"`` = the previous rule (the fill without the tie line), ``"wave"`` = the equal-segment wave rule
    above the chain cap alone, ``"occ"`` = the occupancy rule, ``"wave_fill"`` = the unadopted fill on every rank with
    at most three fragments, kept for the A/B record); the
    public API never selects a rule other than the default.
    """
    if not isinstance(part, KcpPartition):
        raise TypeError(
            "build_kcp_context: part must be the KcpPartition of kcp_partition()"
        )
    W = part.world_size
    rank = int(rank)
    if not 0 <= rank < W:
        raise ValueError(f"build_kcp_context: rank {rank} out of range [0, {W})")
    if group is not None and dist.is_available() and dist.is_initialized():
        gw, gr = dist.get_world_size(group), dist.get_rank(group)
        if gw != W or gr != rank:
            raise ValueError(
                f"build_kcp_context: the process group has world_size {gw} / rank {gr}, the partition and the call "
                f"say {W} / {rank}"
            )
    halo = int(conv_kernel_size) - 1
    if halo < 0:
        raise ValueError(
            f"build_kcp_context: conv_kernel_size must be >= 1, got {conv_kernel_size}"
        )
    device = _device_of(device)
    if sm_count is None:
        sm_count = (
            torch.cuda.get_device_properties(device).multi_processor_count
            if device.type == "cuda"
            else 1
        )
    sm_count = int(sm_count)
    if sm_count < 1:
        raise ValueError(f"build_kcp_context: sm_count must be >= 1, got {sm_count}")
    p_rule = validate_test_knobs(_kcp_test_knobs)["p_rule"]
    mine = part.fragments[rank]
    n_frag = len(mine)
    if n_frag > MAX_FRAGS:
        raise ValueError(
            f"build_kcp_context: rank {rank} holds {n_frag} fragments; the kernels' tables admit at most {MAX_FRAGS}"
        )
    local_cu = part.local_cu_seqlens(rank)
    T_r = local_cu[-1]

    # crossing slots of every rank (the owners address every rank's send layouts)
    in_slot_all: list[tuple[int, ...]] = []
    out_slot_all: list[tuple[int, ...]] = []
    for frs in part.fragments:
        ins, outs, ni, no = [], [], 0, 0
        for f in frs:
            if f.crossing_in:
                ins.append(ni)
                ni += 1
            else:
                ins.append(-1)
            if f.crossing_out:
                outs.append(no)
                no += 1
            else:
                outs.append(-1)
        in_slot_all.append(tuple(ins))
        out_slot_all.append(tuple(outs))
    n_in_by, n_out_by = part.crossing_counts()
    chains = part.chains()
    n_d_max = max((len(c) for c in chains), default=0)

    tables_cpu = chunk_tables(local_cu)
    nc_max, nc_pad, rows = chunk_geometry(T_r, n_frag)
    i32 = torch.int32
    tables = {
        name: torch.tensor(list(values), dtype=i32, device=device)
        for name, values in tables_cpu.items()
    }
    flags = [
        (FLAG_CROSSING_IN if f.crossing_in else 0)
        | (FLAG_CROSSING_OUT if f.crossing_out else 0)
        | (FLAG_CONTINUING if (f.length > 0 and f.off > 0) else 0)
        for f in mine
    ]
    frag_rows = [
        [f.doc, f.off, f.length, fl] for f, fl in zip(mine, flags, strict=True)
    ]
    frag_table = torch.tensor(frag_rows, dtype=i32, device=device).reshape(n_frag, 4)
    halo_plan = _halo_plan(part, rank, halo, local_cu)
    halo_tables = {
        "hist_row_off": torch.tensor(
            [list(r) for r in halo_plan.hist_row_off], dtype=i32, device=device
        ).reshape(n_frag, halo),
        "tail_row_off": torch.tensor(
            [list(r) for r in halo_plan.tail_row_off], dtype=i32, device=device
        ).reshape(n_frag, halo),
        "head_row_off": torch.tensor(
            [[list(c) for c in r] for r in halo_plan.head_row_off],
            dtype=i32,
            device=device,
        ).reshape(n_frag, halo, halo),
        "hist_index": torch.tensor(
            list(halo_plan.hist_index), dtype=torch.int64, device=device
        ),
    }
    return KcpContext(
        partition=part,
        rank=rank,
        world_size=W,
        group=group,
        device=device,
        conv_kernel_size=int(conv_kernel_size),
        sm_count=sm_count,
        p_rule=p_rule,
        T_r=T_r,
        n_frag=n_frag,
        cu_seqlens_local_cpu=local_cu,
        cu_seqlens_local=torch.tensor(list(local_cu), dtype=i32, device=device),
        frag_table=frag_table,
        in_slot=in_slot_all[rank],
        out_slot=out_slot_all[rank],
        n_in=n_in_by[rank],
        n_out=n_out_by[rank],
        N_in=sum(n_in_by),
        N_out=sum(n_out_by),
        n_in_by_rank=tuple(n_in_by),
        n_out_by_rank=tuple(n_out_by),
        in_slot_all=tuple(in_slot_all),
        out_slot_all=tuple(out_slot_all),
        chains=chains,
        n_d_max=n_d_max,
        exchange_scheme="a2a",
        NC=tables_cpu["seq_chunk_start"][-1],
        NC_max=nc_max,
        NC_pad=nc_pad,
        R=rows,
        chunk_bos=tuple(tables_cpu["chunk_bos"]),
        chunk_len=tuple(tables_cpu["chunk_len"]),
        seq_chunk_start=tuple(tables_cpu["seq_chunk_start"]),
        seq_order=tuple(tables_cpu["seq_order"]),
        tables=tables,
        halo=halo_plan,
        halo_tables=halo_tables,
    )


# ----------------------------------------------------------------------------------------------------------------
# Byte queries (the design note §9; keyword-only beyond ``ctx``: the context carries no H)


def _check_save_mode(save_mode: str) -> None:
    if save_mode not in ("light", "tape"):
        raise ValueError(f"save_mode must be 'light' or 'tape', got {save_mode!r}")


def kcp_saved_bytes(
    ctx: KcpContext,
    *,
    H: int = 64,
    save_mode: str = "light",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    with_conv: bool = True,
    C: int = 8192,
    saved_carry: bool = True,
) -> int:
    """Saved-set bytes of one layer on this rank (the KCP contract §3.2 ``kcp_saved_bytes(ctx, ...)``)::

    tables_bytes(NC_pad, n_frag, NSEG)                         # the int32 tables
    + n_in * s                                                 # S_in, s = H * 128 * 128 * 4
    + [save_transfer] * N_out * 2 * s * Hg_r / H               # the owner's whole c1 receive tensor
    + [save_transfer] * [P > 1] * NSEG * s                     # M_seg of the segmented fragments (home rank)
    + [saved_carry] * [P > 1] * (n_frag * (P - 1) + max(n_in, n_out)) * s   # the forward's carry plane (shipped)
    + [with_conv] * sum_cont hist_valid * C * 2                # the hist row pool (BF16)
    save_mode="tape": + NC_pad * H * 32768 + R * H * 256       # the forward's h tape and v_new

    ``save_transfer`` defaults to :data:`SAVE_TRANSFER_DEFAULT` (``True`` since an earlier revision);
    the two ``[save_transfer]`` terms are the delta against the earlier recompute form (``save_transfer=False``).
    ``saved_carry`` (default ``True`` = the shipped default: the forward of a ``P > 1`` rank keeps its carry plane in
    the saved set) counts the whole carry plane of a ``P > 1`` rank under the shipped fused tape walks (one extra slot
    per crossing fragment); ``False`` = the re-walk form of the test knob ``saved_carry=0``.

    The set carries no token-row ``Aqk`` / ``Akk`` (the ``256 * T_r * H`` term of its earlier version): the backward
    recomputes the internal rows in both modes.

    The retained ``S_in`` and c1 tensors are the padded receive buffers (``s`` -> ``n_owners Hg`` tiles,
    ``Hg_r`` -> ``Hg_pad_r``); identical to the formula for ``H = 64``, and the padding is reported by
    ``kcp_comm_bytes(...)["padding"]``.  ``M_seg`` stays the dense copy of §9 (``NSEG s``).
    """
    _check_save_mode(save_mode)
    H, C = int(H), int(C)
    hp = ctx.heads(H)
    s = H * STATE_TILE_BYTES
    s_pad = hp.n_owners * hp.Hg * STATE_TILE_BYTES  # = s without padding
    total = tables_bytes(ctx.NC_pad, ctx.n_frag, hp.NSEG)
    total += ctx.n_in * s_pad
    if save_transfer:
        total += ctx.N_out * 2 * hp.Hg_pad_by_rank[ctx.rank] * STATE_TILE_BYTES
        if hp.P > 1:
            total += hp.NSEG * s
    if saved_carry and hp.P > 1:
        total += (ctx.n_frag * (hp.P - 1) + max(ctx.n_in, ctx.n_out)) * s
    if with_conv:
        total += ctx.halo.n_hist_rows * C * 2
    if save_mode == "tape":
        total += ctx.NC_pad * H * 32768 + ctx.R * H * 256
    return total


def kcp_comm_bytes(
    ctx: KcpContext,
    *,
    H: int = 64,
    C: int = 8192,
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    boundary_mode: str = "saved",
) -> dict:
    """Bytes this rank moves per collective and per boundary (the KCP contract §3.2 ``kcp_comm_bytes(ctx)``; ``send`` / ``recv``
    per collective, totals ``B1`` = forward (c1, c2, halo_fwd), ``B2`` = backward from the saved context (b1 with
    the re-sent maps unless ``save_transfer``, b2, halo_bwd), ``B3`` = the replayed forward + the backward with b1
    in its replay form (offsets only); plus the crossing-fragment counts and the per-fragment state bytes.
    Summed over the ranks, ``c2.recv`` = ``fwd_state_bytes``, ``c1.send`` = ``fwd_summary_bytes`` and ``b2.recv`` =
    ``bwd_adjoint_bytes`` of the harness ``kcp_shapes.comm_bytes`` (plus the padding when ``H % Hg != 0``).
    ``padding`` reports the padding bytes contained in c1 / c2 / b1 / b2 (``send`` / ``recv``), their ``B1`` /
    ``B2`` sums, the padding retained inside the saved ``S_in`` (``s_in_saved``), the ``padded_owner`` and the
    ``pad_tiles_per_slot``; all zero for the production ``H = 64``."""
    if boundary_mode not in ("saved", "replay"):
        raise ValueError(
            f"boundary_mode must be 'saved' or 'replay', got {boundary_mode!r}"
        )
    H, C = int(H), int(C)
    hp = ctx.heads(H)
    sp, halo = hp.splits, ctx.halo
    with_maps = boundary_mode == "saved" and not save_transfer
    b1_send, b1_recv = hp.b1_splits(with_maps)
    b1_send_replay, b1_recv_replay = hp.b1_splits(False)
    coll = {
        "c1": {"send": sum(sp["c1_send"]) * 4, "recv": sum(sp["c1_recv"]) * 4},
        "c2": {"send": sum(sp["c2_send"]) * 4, "recv": sum(sp["c2_recv"]) * 4},
        "b1": {"send": sum(b1_send) * 4, "recv": sum(b1_recv) * 4},
        "b2": {"send": sum(sp["b2_send"]) * 4, "recv": sum(sp["b2_recv"]) * 4},
        "halo_fwd": {
            "send": sum(halo.fwd_send_rows) * C * 2,
            "recv": sum(halo.fwd_recv_rows) * C * 2,
        },
        "halo_bwd": {
            "send": sum(halo.bwd_send_rows) * C * 4,
            "recv": sum(halo.bwd_recv_rows) * C * 4,
        },
    }
    b1_replay = {"send": sum(b1_send_replay) * 4, "recv": sum(b1_recv_replay) * 4}

    def total(*parts):
        return {
            "send": sum(p["send"] for p in parts),
            "recv": sum(p["recv"] for p in parts),
        }

    fwd = total(coll["c1"], coll["c2"], coll["halo_fwd"])
    bwd = total(coll["b1"], coll["b2"], coll["halo_bwd"])
    b3 = total(fwd, b1_replay, coll["b2"], coll["halo_bwd"])
    s = H * STATE_TILE_BYTES
    pn = hp.pad_numel
    b1_key = "dm" if with_maps else "d"
    padding = {
        "c1": {"send": pn["c1_send"] * 4, "recv": pn["c1_recv"] * 4},
        "c2": {"send": pn["c2_send"] * 4, "recv": pn["c2_recv"] * 4},
        "b1": {
            "send": pn[f"b1_send_{b1_key}"] * 4,
            "recv": pn[f"b1_recv_{b1_key}"] * 4,
        },
        "b2": {"send": pn["b2_send"] * 4, "recv": pn["b2_recv"] * 4},
        "s_in_saved": pn["c2_recv"] * 4,
        "padded_owner": hp.padded_owner,
        "pad_tiles_per_slot": hp.pad_tiles,
    }
    padding["B1"] = total(padding["c1"], padding["c2"])
    padding["B2"] = total(padding["b1"], padding["b2"])
    return {
        **coll,
        "b1_replay": b1_replay,
        "padding": padding,
        "B1": fwd,
        "B2": bwd,
        "B3": b3,
        "crossing_fragments_in": ctx.n_in,
        "crossing_fragments_out": ctx.n_out,
        "N_in": ctx.N_in,
        "N_out": ctx.N_out,
        "state_bytes": s,
        "fwd_state_bytes": ctx.n_in * s,
        "fwd_summary_bytes": ctx.n_out * 2 * s,
        "bwd_adjoint_bytes": ctx.n_out * s,
        "H": H,
        "C": C,
        "save_transfer": bool(save_transfer),
        "boundary_mode": boundary_mode,
    }


# The pools of a rank's KCP workspace (``cake_kcp.KcpWorkspace``) that share one storage: each backward send pool
# reuses the bytes of the forward send pool beside it.  A forward call waits its c1 / c2 sends before it returns and
# a replay-mode backward runs its forward (waiting those sends) before the first write of b1_send / b2_send, so the
# lifetimes never overlap; c1_recv keeps its own storage (the replay-mode backward's adjoint merge reads it).
SHARED_POOLS = (("c1_send", "b1_send"), ("c2_send", "b2_send"))
# Pools a workspace allocates on first use only: the ring exchange's hop buffers (``exchange="ring"``), which the
# a2a exchange never reads.
LAZY_POOLS = ("ring_out", "ring_in")
# The pools a workspace sizes for a forward (``backward=False``) and the ones a backward workspace adds.
FWD_POOLS = (
    "c1_send",
    "c1_recv",
    "c2_send",
    "stage_b",
    "stage_m",
    "carry",
    "eye",
) + LAZY_POOLS
BWD_POOLS = ("b1_send", "b1_recv", "b2_send", "b2_recv")


def kcp_pool_numel(
    ctx: KcpContext, *, H: int = 64, fused_walks: int = 1
) -> dict[str, int]:
    """Elements (FP32) of every pool of a rank's KCP workspace and of the call's fresh ``S_in``, by buffer name: the
    one source of the sizes ``cake_kcp`` gives its pools and fresh tensors.  ``fused_walks`` (shipped ``1``) adds one
    carry slot per crossing fragment at ``P > 1``; at ``P > 1`` ``c1_send`` also holds one staging tile per head of each
    crossing-out fragment of this rank (its ``M_f``, transposed into the send layout by the forward)."""
    H = int(H)
    hp = ctx.heads(H)
    n, P, NSEG = hp.numel, int(hp.P), int(hp.NSEG)
    n_in, n_out = int(ctx.n_in), int(ctx.n_out)
    n_carry = int(ctx.n_frag) * (P - 1) if P > 1 else 0
    if fused_walks and P > 1:
        n_carry += max(n_in, n_out)
    return {
        "c1_send": int(n["c1_send"]) + (n_out * H * TILE if P > 1 else 0),
        "c1_recv": int(n["c1_recv"]),
        "c2_send": int(n["c2_send"]),
        "S_in": int(n["c2_recv"]),
        "b1_send": max(int(n["b1_send_d"]), int(n["b1_send_dm"])),
        "b1_recv": max(int(n["b1_recv_d"]), int(n["b1_recv_dm"])),
        "b2_send": int(n["b2_send"]),
        "b2_recv": int(n["b2_recv"]),
        "stage_b": NSEG * H * TILE if P > 1 else 0,
        "stage_m": NSEG * H * TILE if P > 1 else 0,
        "carry": n_carry * H * TILE,
        "eye": TILE,
        "ring_out": max(n_in, n_out) * H * TILE,
        "ring_in": max(n_in, n_out) * H * TILE,
    }


def kcp_arena_layout(
    ctx: KcpContext,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    C: int = 8192,
) -> dict[str, int]:
    """The KCP pools of a rank's workspace (``cake_kcp.KcpWorkspace``) in bytes, as the workspace allocates them.

    ``pool.<name>``: the bytes each pool of a forward (``backward=False``) or a backward workspace needs
    (:func:`kcp_pool_numel`, 4 B per element); ``arena``: the bytes the workspace allocates -- every pool once, each
    :data:`SHARED_POOLS` pair as one storage of the larger need, the :data:`LAZY_POOLS` not at all (``ring``: the bytes
    a call with ``exchange="ring"`` adds on first use).  The saved set (``S_in``, the c1 receive tensor kept with
    ``save_transfer``, ``M_seg``, the forward's saved copy of the carry plane; :func:`kcp_saved_bytes`; the ``carry``
    pool the backward's adjoint carries use is a workspace pool) and the convolution route's halo rows
    (per-call tensors; their traffic: :func:`kcp_comm_bytes` ``halo_fwd`` / ``halo_bwd``) are never workspace pools,
    so ``save_transfer`` and ``C`` leave the workspace unchanged."""
    _check_save_mode(save_mode)
    numel = kcp_pool_numel(ctx, H=H)
    need = {
        name: numel[name] * 4 for name in FWD_POOLS + (BWD_POOLS if backward else ())
    }
    shared = {name for pair in SHARED_POOLS for name in pair}
    arena = sum(
        v for name, v in need.items() if name not in shared and name not in LAZY_POOLS
    )
    arena += sum(
        max((need[name] for name in pair if name in need), default=0)
        for pair in SHARED_POOLS
    )
    out = {f"pool.{name}": v for name, v in need.items()}
    out["ring"] = sum(need[name] for name in LAZY_POOLS)
    out["arena"] = arena
    return out


# The paths of the existing backend's plan the KCP's local stages take their records from: the forward without a
# saved set (its records read the internal ``Aqk`` / ``Akk`` rows the intra program writes; the output program is
# the KCP's own) and the checkpoint replay for both backward modes (its prep keeps the internal ``dout`` copy K2 and
# the KCP's ``dav`` read; its WY and fused-gradient records read the recomputed internal rows).
LOCAL_PATH_FWD = "fwd_nosave"
LOCAL_PATH_BWD = "replay"


def intra_writes_tables(plan, path: str = LOCAL_PATH_FWD) -> bool:
    """Whether the existing backend's intra record of ``path`` (a :class:`cake_backend.Plan`) computes the chunk
    layout from ``cu_seqlens`` and writes the tables itself (its logical vocabulary binds ``cu_seqlens``): the kernel
    backend then hands it a scratch table set so the KCP's CPU-built tables are never rewritten by a kernel."""
    from . import (
        cake_jit,
    )  # the registry literal; imported here so the metadata module stays torch-only

    record = plan.record("fwd_intra", path)
    return "cu_seqlens" in cake_jit.MODULES[record]["logical"].values()


def local_scratch_table_bytes(plan) -> int:
    """Bytes of the scratch table set of :func:`intra_writes_tables` on the arena of ``plan`` (its table buffer: the
    four chunk tables and the block table, 256 B aligned), zero where the intra record reads the tables instead."""
    return int(plan.table_words()) * 4 if intra_writes_tables(plan) else 0


def kcp_workspace_bytes(
    ctx: KcpContext,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    C: int = 8192,
) -> int:
    """Caller-held workspace bytes of one rank: the existing backend's local arena for ``(T_r, n_frag, H)``
    (``kda_train_workspace_bytes``; zero on a rank without tokens) with the intra record's scratch table set where
    it writes one (:func:`local_scratch_table_bytes`), plus the KCP arena of transient buffers
    (:func:`kcp_arena_layout`; CONSISTENCY-6, SPEC-4: saved-set tensors are counted by :func:`kcp_saved_bytes`)."""
    _check_save_mode(save_mode)
    local = 0
    if ctx.T_r > 0:
        from . import (
            cake_backend,
        )  # the existing backend's arena rule; imported here so the metadata module stays torch-only

        local = int(
            cake_backend.workspace_bytes(
                ctx.T_r,
                ctx.n_frag,
                int(H),
                backward=backward,
                save_mode=save_mode,
                sm_count=ctx.sm_count,
            )
        )
        # the scratch table set where the intra record writes one (the rule is architecture-independent; the
        # buffer size follows the same segment-count convention as the arena bytes above)
        if intra_writes_tables(
            cake_backend.plan(ctx.T_r, ctx.n_frag, int(H), ctx.sm_count)
        ):
            local += int(
                cake_backend.tables_bytes(
                    ctx.T_r, ctx.n_frag, int(H), sm_count=ctx.sm_count
                )
            )
    arena = kcp_arena_layout(
        ctx,
        H=H,
        save_mode=save_mode,
        backward=backward,
        save_transfer=save_transfer,
        C=C,
    )
    return local + arena["arena"]


# ----------------------------------------------------------------------------------------------------------------
# Private test knobs (the design note §10.1, SPEC-11)


ENABLE_TEST_KNOBS = "cake_kcp_plan.enable_test_knobs()"


def enable_test_knobs(enabled: bool = True) -> bool:
    """Opt this process into (``True``) or out of (``False``) the private ``_kcp_test_knobs`` hook of the KCP entry
    points and the test-only programs of ``build.test.json``; returns the previous setting.  A test hook: the shipped
    route never needs it, and without it no option reaches a test-only program (SPEC-11)."""
    global _TEST_KNOBS_ENABLED
    previous, _TEST_KNOBS_ENABLED = _TEST_KNOBS_ENABLED, bool(enabled)
    return previous


def test_knobs_enabled() -> bool:
    """Whether the process opted into the private test hook (:func:`enable_test_knobs`)."""
    return _TEST_KNOBS_ENABLED


def validate_test_knobs(knobs) -> dict:
    """Normalize ``_kcp_test_knobs``: ``None`` -> the public defaults ``{"precision_mode": 0, "p_rule":
    P_RULE_DEFAULT}`` (``"wave_fill1_p1tie"``); a dict is accepted only after :func:`enable_test_knobs`, holds only known keys and selects a known ``precision_mode`` (0 the hi/lo form of the TF32 walk; 1 the explicit
    three-MMA form of the R1 bitwise test; 2 plain TF32, the optional G2 cross-check) and a known ``p_rule``
    (:data:`P_RULES`; ``"wave_fill1"`` = the previous rule (the fill without the tie line), ``"wave"`` = the equal-segment
    wave rule above the chain cap alone, ``"occ"`` = the occupancy rule, ``"wave_fill"`` = the unadopted fill; read by
    :func:`build_kcp_context`).  The public API can
    therefore never select a non-zero mode or a P rule other than the shipped one."""
    if knobs is None:
        return {"precision_mode": 0, "p_rule": P_RULE_DEFAULT}
    if not test_knobs_enabled():
        raise ValueError(
            f"_kcp_test_knobs is a private test hook: it is accepted only after {ENABLE_TEST_KNOBS}"
        )
    if not isinstance(knobs, dict):
        raise TypeError(f"_kcp_test_knobs must be a dict, got {type(knobs).__name__}")
    unknown = sorted(set(knobs) - {"precision_mode", "p_rule"})
    if unknown:
        raise ValueError(
            f"_kcp_test_knobs: unknown knob(s) {unknown}; known: ['precision_mode', 'p_rule']"
        )
    mode = int(knobs.get("precision_mode", 0))
    if mode not in PRECISION_MODES:
        raise ValueError(
            f"_kcp_test_knobs: precision_mode must be one of {PRECISION_MODES}, got {mode}"
        )
    p_rule = str(knobs.get("p_rule", P_RULE_DEFAULT))
    if p_rule not in P_RULES:
        raise ValueError(
            f"_kcp_test_knobs: p_rule must be one of {P_RULES}, got {p_rule!r}"
        )
    return {"precision_mode": mode, "p_rule": p_rule}


__all__ = [
    "ALIGN",
    "CHUNK",
    "COLLECTIVES",
    "EXPLICIT_LAYOUT",
    "FLAG_CONTINUING",
    "FLAG_CROSSING_IN",
    "FLAG_CROSSING_OUT",
    "HEAD_DIM",
    "LAYOUTS",
    "MAX_CHAIN_STEPS",
    "MAX_FRAGS",
    "MAX_HEAD_CONSUMERS",
    "MAX_SEGMENTS",
    "PRECISION_MODES",
    "P_BAL_INACTIVE",
    "P_RULES",
    "P_RULE_DEFAULT",
    "SAVE_TRANSFER_DEFAULT",
    "SHARED_POOLS",
    "LAZY_POOLS",
    "FWD_POOLS",
    "BWD_POOLS",
    "STATE_TILE_BYTES",
    "SUMMARY_TILE",
    "TABLE_NAMES",
    "ENABLE_TEST_KNOBS",
    "enable_test_knobs",
    "test_knobs_enabled",
    "TILE",
    "TILE_B",
    "TILE_M",
    "KcpContext",
    "KcpFragment",
    "KcpHaloPlan",
    "KcpHeadPlan",
    "KcpPartition",
    "KcpSegmentPlan",
    "KcpSequence",
    "build_kcp_context",
    "chunk_geometry",
    "chunk_tables",
    "head_groups",
    "kcp_arena_layout",
    "kcp_comm_bytes",
    "kcp_pool_numel",
    "kcp_partition",
    "kcp_plan",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
    "near_equal",
    "owner_of_head",
    "seg_chunk_start",
    "table_layout",
    "table_words",
    "tables_bytes",
    "validate_test_knobs",
    "P1_TIE_BASE",
    "WAVE_FILL_N_FRAG_MAX",
    "WAVE_MIN_NC_MAX_FRAG",
    "WAVE_N_FRAG_MAX",
    "WAVE_RULE_10NS",
    "lpt_makespan",
    "p1_tie_steps",
    "wave_rule_cost_10ns",
    "wave_rule_cost_us",
    "wave_rule_lpt_cost_10ns",
    "wave_rule_lpt_P",
]
