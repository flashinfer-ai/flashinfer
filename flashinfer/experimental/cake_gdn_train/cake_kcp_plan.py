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

Provenance: package-local copy of ``flashinfer/experimental/cake_kda_train/cake_kcp_plan.py`` at FlashInfer
commit 13d7fcf4041635f5b7f573bb0dbab23570ae1034 (sha256 0f2d8b06bbea49df37b043fba3026a06799f0a0cba09d697e490d02bb747dfbf),
adapted for the GDN KCP route: the head count of every per-head state is the value-head count ``HV``
(``H := HV``), the plan rule is a keyword-only argument (no environment variable), and the byte queries live
in ``cake_bytes`` (GDN saved set and exchange formulas).  ``cake_gdn_train`` never imports ``cake_kda_train``;
a CPU test asserts field-identical partitions and contexts on the same inputs.

Metadata layer of the context-parallel (KCP) route of the Cake packed GDN
training backend: the partition of one packed batch of documents over the
ranks of a context-parallel group, the per-rank context (fragment table,
crossing lists, exchange splits, merge chains, the chunk / sequence / segment
tables of the existing stages, the causal-convolution halo plan) and the byte
queries.  Pure host metadata: nothing here reads the contents of a tensor or
launches a kernel; the module imports ``dataclasses``, ``torch`` and
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
of a continuing fragment, and the reverse routing of their ``dpre`` rows).  Everything that depends on the head count ``HV`` (head
ownership ``Hg = ceil(HV / W)``, ``owner(h) = h // Hg``, the split sizes of the
collectives, the merge chain tables, the segment plan ``P`` / ``NSEG`` and the
segment table) is derived on demand by ``ctx.heads(HV)`` and cached.

Segment plan rule.  ``kcp_plan_row(row_tables(part), HV, sm_count)`` (the legacy
``"row_cost"`` rule, selectable; the shipped default ``"row_cost_lpt"`` below starts from
its plan) picks the segments per fragment ``P`` of every rank (``NSEG =
n_frag * P``) from every rank's fragment table: it starts from the per-rank
``"wave_fill1"`` program (below) and switches the whole row to the ``"occ"`` or the
``"one"`` program when the crossing-aware cost model says that program is cheaper
by at least its policy's margin ``ROW_COST_MARGIN_PCT[policy]`` (``"occ"`` 2 %;
``"one"`` 0 %: the program without a compose chain, staging planes or ``P > 1``
launches takes a model tie) -- the slowest rank's sum of the LPT makespans
of the offset + transfer walks (over the crossing-out fragments at ``P = 1``, over
every fragment at ``P > 1``) and of the tape walk (every fragment), plus ``P - 1``
compose steps per wave weighted ``ROW_COST_COMPOSE_K``; constants ``WAVE_RULE_10NS``,
exact integers.  The per-rank ``"wave_fill1"`` program (``kcp_plan``): on a rank
with at most ``WAVE_N_FRAG_MAX = 3`` fragments (N_MAX) whose longest fragment
exceeds the chain cap (``NC_max_frag >= WAVE_MIN_NC_MAX_FRAG = 257`` chunks, i.e.
``P_chain >= 2``), the argmin over ``P in [P_chain, min(64, NC_max_frag)]`` of the
equal-segment wave cost
``ceil(n_frag P HV / SM) * ceil(NC_max_frag / P) * t_walk + (P - 1) * t_compose``
(smallest ``P`` on ties; the earlier ``"wave"`` rule of lever L-C); on a one-fragment
rank below the cap (``0 < NC_max_frag < 257``) the one-wave fill ``clamp(floor(SM /
HV), P_chain, min(64, NC_max_frag))`` -- the largest ``P`` whose ``P HV`` CTAs run one
CTA per SM instead of the occ ``ceil`` (:data:`WAVE_FILL_N_FRAG_MAX`; a rank with
``HV > SM`` keeps the occ ``P``); every other rank (two or three fragments below
the cap, more than three fragments, or an empty longest fragment) keeps the
earlier ``"occ"`` rule ``clamp(max(P_occ, P_chain), 1, ...)``.  ``"wave_fill1"``,
``"wave"``, ``"occ"``, ``"wave_fill"``, ``"chain_min"`` and ``"one"`` stay selectable
through the keyword-only ``p_rule`` argument of :func:`build_kcp_context` (a test
knob of the options object, never a default) for the A/B record.  ``"row_cost_lpt"``
(selectable the same way) is the row rule whose wave-regime ranks take the argmin of
the crossing-aware cost over their real fragment table, capped at the smallest ``P``
within ``ROW_COST_LPT_MARGIN_PCT`` of the optimum (``lpt_rank_P``).  Every rule is a
pure function of (the fragment tables, HV, SM count), identical on every rank.

Exchange layouts (one ``all_to_all_single`` per direction; ``s`` = one FP32
``[128, 128]`` state of one head = 16384 elements; owner ``j`` holds heads
``[j Hg, j Hg + Hg_j)`` with ``Hg_j = clamp(HV - j Hg, 0, Hg)``):

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
MAX_FRAGS = 65536  # the layout table's 16-step document search (the KDA layout table's MAX_DOCS; kept for parity)
MAX_CHAIN_STEPS = 256  # kcp_plan: no walk chain longer than 256 chunk steps
MAX_SEGMENTS = 64  # kcp_plan: at most 64 segments per fragment
P_BAL_INACTIVE = 64  # kcp_plan: P_bal until a measurement fixes t_step / t_compose
P_RULES = (
    "occ",
    "wave",
    "wave_fill1",
    "wave_fill",
    "chain_min",
    "one",
    "row_cost",
    "row_cost_lpt",
)  # P rules.  Per-rank rules (kcp_plan): "wave_fill1" = "wave" + the one-wave
# fill on one-fragment ranks below the chain cap (the shipped default before "row_cost"); "wave" = the earlier rule (lever
# L-C); "occ" = the design note §7.2 rule; "wave_fill" = the fill on every rank with n_frag <= 3 (implemented, not adopted);
# "chain_min" = the fewest segments that keep every walk chain within MAX_CHAIN_STEPS (P = P_chain: one tape walk per
# fragment below the cap, no offset / transfer / merge work beyond the chain cap's minimum); "one" = P = 1 on every rank
# (one walk chain per fragment regardless of its length -- it ignores the chain cap).
# Row rule (kcp_plan_row; "row_cost" = the selectable legacy rule, the shipped default "row_cost_lpt" starts from
# its plan): "row_cost" costs the three per-rank programs {"wave_fill1", "occ", "one"}
# over EVERY rank's fragment table with the crossing-aware LPT model (ROW_COST_*) and keeps "wave_fill1" unless "occ" or
# "one" applied to the whole row is cheaper by at least its margin ROW_COST_MARGIN_PCT[policy] ("occ" 2 %, "one" 0 %:
# the simpler program takes a model tie).  The per-rank rules stay selectable through
# the keyword-only ``p_rule`` argument (options-object test knob) for the A/B record.  Every rule is a pure function of
# (the fragment tables, HV, SM count), identical on every rank, so G3 holds under any of them.
P_RULE_DEFAULT = "row_cost_lpt"
ROW_COST_BASE = "wave_fill1"  # the per-rank program the row rule starts from
ROW_COST_POLICIES = ("occ", "one")  # the whole-row alternatives it may switch to
ROW_COST_COMPOSE_K = 2  # compose-term multiplier: the merge launches measured about twice the one-step compose model
# switch only when 100 * cost(alternative) <= (100 - margin[policy]) * cost(best so far): "occ" trades one P > 1
# program for another and keeps a 2 % guard; "one" has no compose chain, no staging planes and no P > 1 launches (and
# less memory), so it takes a model tie (margin 0: the fit of record admits the measured "one" row at model ratio 0.980)
ROW_COST_MARGIN_PCT = {"occ": 2, "one": 0}
# The row rules (every rank's table; kcp_plan_row); the per-rank kcp_plan refuses them.  "row_cost_lpt" is "row_cost" with
# one change of the base program: on a rank of the wave regime (wave_regime: 1 <= n_frag <= WAVE_N_FRAG_MAX and
# NC_max_frag >= WAVE_MIN_NC_MAX_FRAG) the equal-segment argmin of the wave line (every fragment modelled as NC_max_frag
# chunks) is replaced by lpt_rank_P: the argmin of row_cost_rank_10ns over the rank's REAL fragment table, capped at the
# smallest P within ROW_COST_LPT_MARGIN_PCT of that optimum (the measured calibration admits P = 3 on the GB300 rows of
# record; larger P are unmeasured, so a near-tie keeps the shorter compose chain).  Every other rank (D rows with >= 4
# fragments, fragments of <= 256 chunks at CP8 / CP16) keeps the "wave_fill1" program bitwise, and the whole-row "occ" /
# "one" switch stays on top.  Two gates on the B3 walk-step model walk_steps_rank_b3 (5 x the summary walks' LPT makespan +
# 3 x the tape walks': the walks of a training step that scale with the chain length) keep the rule monotone in the walks:
# a rank takes the LPT P only if its own walk steps do not increase versus the "wave_fill1" P, and the row adopts its changed
# ranks only if the slowest rank's walk steps improve by at least ROW_COST_LPT_ROW_MARGIN_PCT versus the "row_cost" vector
# (otherwise the whole row keeps that vector bitwise).  An unsegmented rank (delivered P = 1: no compose chain, the 64-wide
# fwdh / dhu kernels) never enters the LPT replacement and keeps P = 1 -- the walk-step model does not carry the per-segment
# work the first products add.  Selectable through `p_rule`; P_RULE_DEFAULT names the shipped rule.
ROW_RULES = ("row_cost", "row_cost_lpt")
ROW_COST_LPT_MARGIN_PCT = ROW_COST_MARGIN_PCT["occ"]
# The row gate's margin, calibrated on the measured GB300 rows of record (per-row ABBA of the rule against the "row_cost"
# plan at B1 and B3): the smallest value on a 0.5 % grid such that every measured row whose modelled slowest-rank walk-step
# shortening reaches it passed and every row that failed lies below it.  Rows below the margin keep "row_cost" bitwise.
ROW_COST_LPT_ROW_MARGIN_PCT = 4
# The shipped default of ``save_transfer`` (the design note §6.2) -- the merge owner
# keeps its c1 receive tensor (and the home rank its M_seg at P > 1) for the backward instead of re-sending M_f^T in b1.
# Default of the byte queries below and of ``cake_kcp.KcpOptions``; ``False`` (the earlier recompute form) stays
# selectable through the ``save_transfer`` argument; ``kcp_saved_bytes`` reports the delta.
SAVE_TRANSFER_DEFAULT = True
# Wave-aware P rule: per-CTA-step constants at one CTA per SM on GB300 (KDA, 64 heads; a GDN walk micro-benchmark refits them), in 10 ns
# units (exact integer arithmetic keeps the argmin and its ties identical on every rank).  Sources (generated benchmark files):
# t_tape 2.85 us = timeline.md "Q1 CP4 C B1" rank 3 #11 / #12 (64 CTAs x 244 steps 0.706 ms -> 2.893; 192 CTAs = 2 waves x 244
# steps 1.378 ms -> 2.824); t_xfer 3.5 us = k3_micro.md "t_step us (one chain, per wave)" tf32_hilo z-q6 3.515 / c-cp8 3.317;
# t_offset 1.2 us = timeline.md rank 3 #5 (0.581 ms / (2 waves x 244 steps) = 1.19); t_compose 20.4 us = refit.md P_bal table
# "t_compose us/step" 20.389 (map-mode c16 step).
WAVE_RULE_10NS = {"tape": 285, "xfer": 350, "offset": 120, "compose": 2040}
# N_MAX (a plan constant, part of the pure function of the fragment table, HV and SM count): the
# wave rule applies to a rank with at most this many fragments; ranks with more keep P_occ (the measured
# many-fragment D ranks lose 9-11 % at any P > 1: the compose / staging cost at P > 1 exceeds the model; D ranks
# of a multi-document pack have >= 4 fragments, two per document, so D keeps the bitwise "occ" program).
WAVE_N_FRAG_MAX = 3
# Candidate (iv): the wave rule applies only to ranks whose longest fragment exceeds the chain cap (P_chain >= 2): on
# fragments of <= 256 chunks (Q1 CP16: 244) the extra segments' compose / staging cost exceeds the shorter walks
# (measured CP16 C +5.0 % B1 / +6.4 % B3, CP16 Z +7.3 % B3 at the unconditioned wave P).
WAVE_MIN_NC_MAX_FRAG = MAX_CHAIN_STEPS + 1
# The one-wave fill rules: the "one-wave
# fill" rules -- "wave_fill1" is the shipped default since this revision, "wave_fill" stays a test knob value.  Below
# the chain cap (NC_max_frag < WAVE_MIN_NC_MAX_FRAG) the earlier "wave" rule keeps the occ P = ceil(SM / (n_frag HV)), which on the one-fragment CP16 C ranks (NC 222-256, HV = 64) launches 192 CTAs on
# 152 SMs (40 SMs host two CTAs: 148-172 step-times for 74-86-step chains); the fill rule takes the LARGEST P whose
# n_frag P HV CTAs run one CTA per SM, floor(SM / (n_frag HV)) (= 2 -> 128 CTAs, one wave of 111-128 steps, the floor's
# min_P), clamped to [P_chain, min(64, NC_max_frag)]; a rank without a one-wave P (n_frag HV > SM) keeps the occ P.
# "wave_fill1" applies it to one-fragment ranks only (the shipped rule), "wave_fill" to every rank with
# n_frag <= WAVE_N_FRAG_MAX (two-fragment HV = 64 ranks move P 2 -> 1).  Above the cap both are the shipped wave argmin.
# At HV <= 4 on 152 SMs floor(SM / HV) equals ceil(SM / HV) (38; 64-cap at HV = 1), so every HV <= 4 row keeps its program.
WAVE_FILL_N_FRAG_MAX = {"wave_fill1": 1, "wave_fill": WAVE_N_FRAG_MAX}
LAYOUTS = ("contiguous", "zigzag", "zigzag_per_document")
EXPLICIT_LAYOUT = "explicit"
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
FLAG_CROSSING_IN = 1
FLAG_CROSSING_OUT = 2
FLAG_CONTINUING = 4  # the fragment's first outputs read convolution history rows (document offset > 0)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


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


# ----------------------------------------------------------------------------------------------------------------
# Host-built integer tables on the device without a synchronizing call

UPLOAD_ALIGN_BYTES = 256  # every table starts on a 256-byte boundary of its shared buffer (vector loads, TMA)


# one host-built table on its way to the device: (values, dtype, shape); ``values`` a flat or nested list or tuple of
# ints, ``shape`` None = the sequence's own shape (a type alias; the module imports torch and dataclasses only)
TableSpec = tuple[list | tuple, torch.dtype, tuple[int, ...] | None]


def upload_tables(
    specs: dict[str, TableSpec],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Device copies of integer tables built on the host (``name -> (values, dtype, shape)``; ``values`` a flat or
    nested sequence of ints, ``shape`` ``None`` = the sequence's own shape), without a synchronizing CUDA call.

    ``torch.tensor(values, device=cuda)`` copies from pageable host memory and synchronizes the stream once per
    table (``torch.cuda.set_sync_debug_mode`` reports every one of them), so a context prepared behind queued
    device work waited for that work.  Here every table of one dtype is laid out in one pinned host buffer
    (each table at a 256-byte boundary) and copied with one asynchronous ``copy_``: a pinned non-blocking
    host-to-device copy enqueues ``cudaMemcpyAsync`` on the current stream and records the staging block on it, so
    the host never waits and the staging memory is reused only after the copy has run.  Each returned tensor is a
    contiguous view of its dtype's device buffer; the values are those of ``torch.tensor(values, dtype)``.  On a
    non-CUDA device the tables are plain CPU tensors."""
    device = torch.device(device)
    groups: dict[torch.dtype, list[tuple[str, torch.Tensor, tuple[int, ...]]]] = {}
    for name, (values, dtype, shape) in specs.items():
        cpu = torch.tensor(list(values), dtype=dtype)
        shape = tuple(cpu.shape) if shape is None else tuple(int(x) for x in shape)
        flat = cpu.reshape(-1)
        numel = 1
        for x in shape:
            numel *= x
        if flat.numel() != numel:
            raise ValueError(
                f"upload_tables: table {name!r} holds {flat.numel()} values, shape {shape} needs {numel}"
            )
        groups.setdefault(dtype, []).append((name, flat, shape))
    out: dict[str, torch.Tensor] = {}
    if device.type != "cuda":
        for items in groups.values():
            for name, flat, shape in items:
                out[name] = flat.view(shape).to(device)
        return out
    for dtype, items in groups.items():
        step = max(1, UPLOAD_ALIGN_BYTES // torch.empty((), dtype=dtype).element_size())
        offsets, total = [], 0
        for _name, flat, _shape in items:
            offsets.append(total)
            total += -(-flat.numel() // step) * step
        host = torch.zeros(
            (total,), dtype=dtype, pin_memory=True
        )  # the alignment padding is defined, never read
        for (_name, flat, _shape), off in zip(items, offsets, strict=True):
            host[off : off + flat.numel()].copy_(flat)
        buf = torch.empty((total,), dtype=dtype, device=device)
        buf.copy_(
            host, non_blocking=True
        )  # cudaMemcpyAsync; the pinned block is held until the copy has run
        for (name, flat, shape), off in zip(items, offsets, strict=True):
            out[name] = buf[off : off + flat.numel()].view(shape)
    return out


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
# CPU-built tables of the existing chunk-parallel stages (the ``table`` / ``seg_table`` programs' contract)


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
    sequence ``d`` is pseudo-sequence ``d * P + p`` (bitwise the ``kda_chunk_train_seg_table_v1`` output)."""
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
    machine each time); integer arithmetic throughout, order of equal-length groups irrelevant.  The machines are
    held as a multiset ``load -> count`` (the greedy's result depends on the loads only, not on which machine
    carries which; the filling leaves a handful of distinct loads), so a level query costs the number of distinct
    loads instead of the SM count."""
    machines = int(machines)
    if machines < 1:
        raise ValueError(f"lpt_makespan: machines must be >= 1, got {machines}")
    loads: dict[int, int] = {0: machines}
    for L, m in sorted(chains, reverse=True):
        L, m = int(L), int(m)
        if L <= 0 or m <= 0:
            continue

        def needed(
            level: int,
        ) -> int:  # jobs that raise every machine to at least `level`
            return sum(
                cnt * (-(-(level - load) // L))
                for load, cnt in loads.items()
                if load < level
            )

        lo = min(loads)
        hi = lo + (m + 1) * L  # needed(hi) >= m + 1 from the least-loaded machine alone
        while lo < hi:  # the largest level reachable with at most m jobs
            mid = (lo + hi + 1) // 2
            if needed(mid) <= m:
                lo = mid
            else:
                hi = mid - 1
        raised: dict[int, int] = {}
        placed = 0
        for load, cnt in loads.items():
            c = -(-(lo - load) // L) if load < lo else 0
            placed += cnt * c
            raised[load + c * L] = raised.get(load + c * L, 0) + cnt
        loads = raised
        rest = m - placed
        while rest:  # the leftover jobs (fewer than the machines at `lo`) go to the least-loaded machines, one each
            low = min(loads)
            take = min(rest, loads[low])
            loads[low] -= take
            if not loads[low]:
                del loads[low]
            loads[low + L] = loads.get(low + L, 0) + take
            rest -= take
    return max(loads)


def wave_rule_cost_10ns(
    n_frag: int, NC_max_frag: int, HV: int, sm_count: int, P: int
) -> int:
    """Lever L-C cost model of ``P`` segments per fragment (10 ns units, exact integers; the rule of
    record): the three walks (K1 offset, K3 transfer, K1 tape; one CTA per SM, equal segments) each take
    ``ceil(n_frag P HV / SM)`` waves of ``ceil(NC_max_frag / P)`` CTA-steps, plus ``P - 1`` K4 compose steps."""
    waves = _ceil_div(n_frag * P * HV, sm_count)
    steps = _ceil_div(NC_max_frag, P)
    walk = WAVE_RULE_10NS["tape"] + WAVE_RULE_10NS["xfer"] + WAVE_RULE_10NS["offset"]
    return waves * steps * walk + (P - 1) * WAVE_RULE_10NS["compose"]


def wave_rule_cost_us(
    n_frag: int, NC_max_frag: int, HV: int, sm_count: int, P: int
) -> float:
    return wave_rule_cost_10ns(n_frag, NC_max_frag, HV, sm_count, P) / 100.0


def _wave_rule_P(
    n_frag: int, NC_max_frag: int, HV: int, sm_count: int, p_chain: int
) -> int:
    """``argmin_P wave_rule_cost`` over ``P in [P_chain, min(64, NC_max_frag)]`` (smallest ``P`` on ties)."""
    hi = max(1, min(MAX_SEGMENTS, NC_max_frag))
    lo = min(p_chain, hi)
    best_P, best_cost = lo, None
    for P in range(lo, hi + 1):
        cost = wave_rule_cost_10ns(n_frag, NC_max_frag, HV, sm_count, P)
        if best_cost is None or cost < best_cost:
            best_P, best_cost = P, cost
    return best_P


def wave_rule_lpt_cost_10ns(
    nc_frags: "list[int]", HV: int, sm_count: int, P: int
) -> int:
    """Information form (b) (NOT the rule of record): the walks take the LPT makespan of
    ``P * HV`` chains of ``ceil(NC_f / P)`` steps per fragment ``f`` with ``NC_f`` chunks (empty fragments: none) on
    ``sm_count`` SMs, plus ``P - 1`` compose steps.  Equal fragments reproduce :func:`wave_rule_cost_10ns`."""
    P, HV = int(P), int(HV)
    chains = [(-(-int(nc) // P), P * HV) for nc in nc_frags if int(nc) > 0]
    steps = lpt_makespan(chains, sm_count) if chains else 0
    walk = WAVE_RULE_10NS["tape"] + WAVE_RULE_10NS["xfer"] + WAVE_RULE_10NS["offset"]
    return steps * walk + (P - 1) * WAVE_RULE_10NS["compose"]


def wave_rule_lpt_P(nc_frags: "list[int]", HV: int, sm_count: int) -> int:
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
        cost = wave_rule_lpt_cost_10ns(nc_frags, HV, sm_count, P)
        if best_cost is None or cost < best_cost:
            best_P, best_cost = P, cost
    return best_P


def row_tables(part: "KcpPartition") -> tuple[tuple[tuple[int, bool], ...], ...]:
    """The row rule's input: per rank the ``(chunk count, crossing_out)`` of every fragment, in local order."""
    return tuple(
        tuple((_ceil_div(f.length, CHUNK), f.crossing_out) for f in frs)
        for frs in part.fragments
    )


def row_cost_rank_10ns(
    table: "tuple[tuple[int, bool], ...]", HV: int, sm_count: int, P: int
) -> int:
    """Crossing-aware LPT cost of one rank at ``P`` segments per fragment (10 ns units, exact integers).  The offset and
    transfer walks run over the crossing-out fragments only at ``P = 1`` (a fragment that does not cross out needs no
    ``B_f`` / ``M_f``) and over every fragment at ``P > 1`` (every segment's summary feeds the compose); the tape walk runs
    over every fragment; the phases are sequential, so their LPT makespans (:func:`lpt_makespan`, ``P * HV`` chains of
    ``ceil(NC_f / P)`` steps per fragment, one CTA per SM) add; ``P - 1`` compose steps per wave of ``n_frag * HV`` chains,
    weighted :data:`ROW_COST_COMPOSE_K`.  Constants :data:`WAVE_RULE_10NS`."""
    P, HV = int(P), int(HV)
    T = WAVE_RULE_10NS
    summary = [
        (_ceil_div(nc, P), P * HV)
        for nc, crossing_out in table
        if nc > 0 and (P > 1 or crossing_out)
    ]
    tape = [(_ceil_div(nc, P), P * HV) for nc, _ in table if nc > 0]
    cost = 0
    if summary:
        cost += lpt_makespan(summary, sm_count) * (T["offset"] + T["xfer"])
    if tape:
        cost += lpt_makespan(tape, sm_count) * T["tape"]
    waves = _ceil_div(len(table) * HV, sm_count)
    return cost + (P - 1) * T["compose"] * waves * ROW_COST_COMPOSE_K


def row_cost_10ns(
    tables: "tuple[tuple[tuple[int, bool], ...], ...]",
    HV: int,
    sm_count: int,
    Ps: "tuple[int, ...]",
) -> int:
    """Row cost of a ``P`` per rank: the slowest rank's :func:`row_cost_rank_10ns` (the walks of the ranks run in parallel;
    the exchange synchronises them)."""
    return max(
        (
            row_cost_rank_10ns(table, HV, sm_count, P)
            for table, P in zip(tables, Ps, strict=True)
            if table
        ),
        default=0,
    )


def wave_regime(n_frag: int, NC_max_frag: int) -> bool:
    """The wave line's regime of :func:`kcp_plan` under ``"wave_fill1"``: ``1 <= n_frag <= WAVE_N_FRAG_MAX`` and
    ``NC_max_frag >= WAVE_MIN_NC_MAX_FRAG`` -- the only ranks whose ``P`` the ``"row_cost_lpt"`` rule may change."""
    return (
        1 <= int(n_frag) <= WAVE_N_FRAG_MAX and int(NC_max_frag) >= WAVE_MIN_NC_MAX_FRAG
    )


def walk_steps_rank_b3(
    table: "tuple[tuple[int, bool], ...]", HV: int, sm_count: int, P: int
) -> int:
    """The B3 walk-step model of one rank at ``P`` segments per fragment: ``5 * ss + 3 * ts`` with ``ss`` the LPT makespan
    (:func:`lpt_makespan`, ``P * HV`` chains of ``ceil(NC_f / P)`` steps per fragment, one CTA per SM) of the summary chains
    (the offset and transfer walks: every fragment at ``P > 1``, the crossing-out fragments at ``P = 1``) and ``ts`` that of the
    tape chains (every fragment) -- the forward's offset / transfer / tape walks, the hand-off recompute's three walks and the
    backward's adjoint offset / tape walks, i.e. the launches whose length follows the chain length.  Exact integers."""
    P, HV = int(P), int(HV)
    summary = [
        (_ceil_div(nc, P), P * HV)
        for nc, crossing_out in table
        if nc > 0 and (P > 1 or crossing_out)
    ]
    tape = [(_ceil_div(nc, P), P * HV) for nc, _ in table if nc > 0]
    ss = lpt_makespan(summary, sm_count) if summary else 0
    ts = lpt_makespan(tape, sm_count) if tape else 0
    return 5 * ss + 3 * ts


def lpt_rank_P(
    table: "tuple[tuple[int, bool], ...]",
    HV: int,
    sm_count: int,
    *,
    margin_pct: int = ROW_COST_LPT_MARGIN_PCT,
) -> int:
    """The ``"row_cost_lpt"`` rule's ``P`` of one wave-regime rank: ``argmin_P row_cost_rank_10ns(table, HV, sm_count, P)``
    over ``P in [P_chain, min(64, NC_max_frag)]`` (the wave line's range) on the rank's real fragment table, then the SMALLEST
    ``P`` whose cost is within ``margin_pct`` of that optimum (``100 * cost(P) <= (100 + margin_pct) * cost(best)``; exact
    integers; ties -> smallest ``P``).  A pure function of its arguments; ``1`` when every fragment is empty."""
    table = tuple((int(nc), bool(x)) for nc, x in table)
    nc_max = max((nc for nc, _ in table), default=0)
    if nc_max == 0:
        return 1
    hi = max(1, min(MAX_SEGMENTS, nc_max))
    lo = min(max(1, _ceil_div(nc_max, MAX_CHAIN_STEPS)), hi)
    costs = [(P, row_cost_rank_10ns(table, HV, sm_count, P)) for P in range(lo, hi + 1)]
    best = min(c for _, c in costs)
    for P, c in costs:
        if 100 * c <= (100 + int(margin_pct)) * best:
            return P
    return costs[0][0]


def kcp_plan_row(
    tables: "tuple[tuple[tuple[int, bool], ...], ...]",
    HV: int,
    sm_count: int,
    *,
    P_bal: int = P_BAL_INACTIVE,
    p_rule: str = "row_cost",
) -> tuple[KcpSegmentPlan, ...]:
    """The ``"row_cost"`` rule (selectable legacy rule; the shipped default ``"row_cost_lpt"`` starts from its plan):
    one :class:`KcpSegmentPlan` per rank from EVERY rank's fragment
    table (:func:`row_tables`), ``HV`` and the SM count.  The base program is :data:`ROW_COST_BASE` (``"wave_fill1"``)
    applied per rank; each alternative of :data:`ROW_COST_POLICIES` (``"occ"``, ``"one"``) applied to the whole row is
    costed with :func:`row_cost_10ns`, and an alternative replaces the current best only when it is cheaper and
    ``100 * cost(alternative) <= (100 - ROW_COST_MARGIN_PCT[alternative]) * cost(best)`` (exact integers; ``"occ"``
    2 %, ``"one"`` 0 %; ties keep the current best; an alternative equal to the base vector is skipped).  A pure
    function of its arguments, identical on every rank: the same topology gives the same ``P`` on every rank.  On the
    rows where it departs from ``"wave_fill1"`` the crossing states
    compose through a different number of segment products, so those rows are not bitwise against the former default.
    Every returned plan reports ``p_rule="row_cost"``.

    ``p_rule="row_cost_lpt"`` is the same rule with one change of the base program: every rank of the wave regime
    (:func:`wave_regime`) takes :func:`lpt_rank_P` -- the argmin of :func:`row_cost_rank_10ns` over its real fragment
    table, capped at the smallest ``P`` within :data:`ROW_COST_LPT_MARGIN_PCT` of the optimum -- instead of the wave
    line's equal-segment argmin; every other rank keeps its ``"wave_fill1"`` program bitwise, the ``"occ"`` / ``"one"``
    switch is unchanged, and the returned plans report ``p_rule="row_cost_lpt"``.  Two gates on the walk-step model
    :func:`walk_steps_rank_b3` keep the rule monotone in the walks: a rank takes the LPT ``P`` only if its own walk steps
    do not increase versus the ``"wave_fill1"`` ``P``, and the row adopts its changed ranks only if the slowest rank's walk
    steps improve by at least :data:`ROW_COST_LPT_ROW_MARGIN_PCT` versus the ``"row_cost"`` vector (``100 * steps(new) <=
    (100 - margin) * steps(row_cost)``; otherwise the whole row keeps that vector bitwise).  An unsegmented rank (delivered
    ``P = 1``: no compose chain, the 64-wide fwdh / dhu kernels) never enters the LPT replacement and keeps ``P = 1``.
    Where the two rules differ the changed ranks compose through a different number of segment products (not bitwise
    against ``"row_cost"``)."""
    HV, sm_count = int(HV), int(sm_count)
    if p_rule not in ROW_RULES:
        raise ValueError(
            f"kcp_plan_row: p_rule must be one of {ROW_RULES} (the row rules), got {p_rule!r}"
        )
    tables = tuple(tuple((int(nc), bool(x)) for nc, x in table) for table in tables)

    def program(rule: str) -> tuple[KcpSegmentPlan, ...]:
        return tuple(
            kcp_plan(
                len(table),
                max((nc for nc, _ in table), default=0),
                HV,
                sm_count,
                P_bal=P_bal,
                p_rule=rule,
            )
            for table in tables
        )

    base = program(ROW_COST_BASE)
    if (
        p_rule == "row_cost_lpt"
    ):  # the wave-regime ranks take the LPT argmin over their real fragment table
        today = kcp_plan_row(
            tables, HV, sm_count, P_bal=P_bal, p_rule="row_cost"
        )  # the delivered plan
        lpt = []
        for seg, table, old in zip(base, tables, today, strict=True):
            # an unsegmented rank (delivered P = 1: no compose chain, the 64-wide fwdh / dhu kernels) keeps that path
            if old.P > 1 and wave_regime(seg.n_frag, seg.NC_max_frag):
                P_new = lpt_rank_P(table, HV, sm_count)
                if P_new != seg.P and walk_steps_rank_b3(
                    table, HV, sm_count, P_new
                ) <= walk_steps_rank_b3(table, HV, sm_count, seg.P):
                    seg = dataclasses.replace(seg, P=P_new, NSEG=seg.n_frag * P_new)
            lpt.append(seg)
        base = tuple(lpt)
    base_P = tuple(seg.P for seg in base)
    best, best_cost = base, row_cost_10ns(tables, HV, sm_count, base_P)
    for rule in ROW_COST_POLICIES:
        cand = program(rule)
        cand_P = tuple(seg.P for seg in cand)
        if cand_P == base_P:
            continue
        cost = row_cost_10ns(tables, HV, sm_count, cand_P)
        margin = ROW_COST_MARGIN_PCT[rule]
        if cost < best_cost and 100 * cost <= (100 - margin) * best_cost:
            best, best_cost = cand, cost
    if (
        p_rule == "row_cost_lpt"
    ):  # the row gate: the slowest rank's walk steps must improve by the margin, else row_cost
        today_P = tuple(seg.P for seg in today)
        best_P = tuple(seg.P for seg in best)
        if any(
            P_old == 1 and P_new != 1
            for P_old, P_new in zip(today_P, best_P, strict=True)
        ):
            best = today  # the unsegmented ranks keep P = 1 whatever the whole-row switch picked
        elif best_P != today_P:

            def slowest(Ps: "tuple[int, ...]") -> int:
                return max(
                    (
                        walk_steps_rank_b3(table, HV, sm_count, P)
                        for table, P in zip(tables, Ps, strict=True)
                        if table
                    ),
                    default=0,
                )

            w_new, w_today = slowest(tuple(seg.P for seg in best)), slowest(today_P)
            if not 100 * w_new <= (100 - ROW_COST_LPT_ROW_MARGIN_PCT) * w_today:
                best = today
    return tuple(dataclasses.replace(seg, p_rule=p_rule) for seg in best)


def kcp_plan(
    n_frag: int,
    NC_max_frag: int,
    HV: int,
    sm_count: int,
    *,
    P_bal: int = P_BAL_INACTIVE,
    p_rule: str = ROW_COST_BASE,
) -> KcpSegmentPlan:
    """The per-rank KCP plan rules (``p_rule="wave_fill1"``: the one-wave fill on the earlier ``"wave"`` rule of lever
    L-C, the shipped default before the row rule :func:`kcp_plan_row` and the base program that rule starts from; the
    row rule itself is not a per-rank rule and is refused here).  The ``"wave"`` rule::

        P_occ   = ceil(SM / (n_frag * HV))                 # one transfer-walk CTA per SM
        P_chain = ceil(NC_max_frag / 256)                 # no walk chain longer than 256 steps
        P_bal   = 64 (inactive until measured)
        P_wave  = argmin over P in [P_chain, min(64, NC_max_frag)] of wave_rule_cost(P)   # smallest P on ties
        P       = P_wave                                 if n_frag <= WAVE_N_FRAG_MAX (3) and NC_max_frag >= WAVE_MIN_NC_MAX_FRAG (257)
                = clamp(max(P_occ, P_chain), 1, min(64, NC_max_frag, max(P_chain, P_bal)))   otherwise (occ)
        NSEG    = n_frag * P

    with ``wave_rule_cost(P) = ceil(n_frag P HV / SM) x ceil(NC_max_frag / P) x (t_tape + t_xfer + t_offset) +
    (P - 1) x t_compose`` (:func:`wave_rule_cost_10ns`, constants :data:`WAVE_RULE_10NS`): the three equal-segment
    walks at one CTA per SM plus the K4 compose chain.  The wave line applies only to ranks with
    ``n_frag <= WAVE_N_FRAG_MAX`` (3) whose longest fragment exceeds the chain cap (``NC_max_frag >=
    WAVE_MIN_NC_MAX_FRAG`` = 257 chunks, i.e. ``P_chain >= 2``); every other rank keeps the occ ``P``
    (many-fragment D ranks and short fragments measured slower at every ``P > 1``).  ``p_rule="occ"`` (the round-0
    §7.2 rule, selectable only through the ``p_rule`` test knob for the A/B record) uses the ``clamp`` line on every
    rank.  ``P_occ`` / ``P_chain`` / ``P_bal`` are reported under every rule.

    The one-wave fill rule (``"wave_fill1"`` shipped, ``"wave_fill"`` test knob only; :data:`WAVE_FILL_N_FRAG_MAX`)::

        P_fill  = floor(SM / (n_frag * HV))
        P       = clamp(P_fill, P_chain, min(64, NC_max_frag))   if p_rule in ("wave_fill1", "wave_fill") and
                                                                  n_frag <= WAVE_FILL_N_FRAG_MAX[p_rule] (1 | 3) and
                                                                  0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG and P_fill >= 1
                = the "wave" rule above                           otherwise

    The minimal-segment rules (test knob only, for the A/B of the one-document ranks)::

        P = P_chain                                              if p_rule == "chain_min"   (1 below the chain cap)
        P = 1                                                    if p_rule == "one"         (ignores the chain cap)

    i.e. below the chain cap the largest segment count that still runs one CTA per SM (``P_bal`` does not enter); ``p_rule="wave"`` (the earlier rule, selectable through the test knob) is the first
    block alone.

    A pure function of ``(fragment table, HV, SM count)``.  A rank without fragments has ``P = 1``, ``NSEG = 0``;
    when every fragment is empty (``NC_max_frag = 0``) the lower bound wins: ``P = 1``.
    """
    n_frag, NC_max_frag, HV, sm_count = (
        int(n_frag),
        int(NC_max_frag),
        int(HV),
        int(sm_count),
    )
    if n_frag < 0 or NC_max_frag < 0 or HV < 1 or sm_count < 1:
        raise ValueError(
            f"kcp_plan: n_frag={n_frag}, NC_max_frag={NC_max_frag} must be >= 0 and HV={HV}, sm_count={sm_count} >= 1"
        )
    if p_rule not in P_RULES or p_rule in ROW_RULES:
        raise ValueError(
            f"kcp_plan: p_rule must be one of {tuple(r for r in P_RULES if r not in ROW_RULES)} (per-rank rules; "
            f"the row rules {ROW_RULES} need every rank's table: kcp_plan_row), got {p_rule!r}"
        )
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
    p_occ = _ceil_div(sm_count, n_frag * HV)
    p_chain = max(1, _ceil_div(NC_max_frag, MAX_CHAIN_STEPS))
    fill_n_frag_max = WAVE_FILL_N_FRAG_MAX.get(p_rule)
    p_fill = sm_count // (n_frag * HV)
    if p_rule == "one":
        P = 1
    elif p_rule == "chain_min":
        P = max(1, min(p_chain, MAX_SEGMENTS, max(NC_max_frag, 1)))
    elif (
        fill_n_frag_max is not None
        and n_frag <= fill_n_frag_max
        and 0 < NC_max_frag < WAVE_MIN_NC_MAX_FRAG
        and p_fill >= 1
    ):
        P = max(p_chain, min(MAX_SEGMENTS, NC_max_frag, p_fill))
    elif (
        p_rule != "occ"
        and n_frag <= WAVE_N_FRAG_MAX
        and NC_max_frag >= WAVE_MIN_NC_MAX_FRAG
    ):
        P = (
            1
            if NC_max_frag == 0
            else _wave_rule_P(n_frag, NC_max_frag, HV, sm_count, p_chain)
        )
    else:
        upper = min(MAX_SEGMENTS, NC_max_frag, max(p_chain, int(P_bal)))
        P = max(1, min(max(p_occ, p_chain), upper))
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


def head_groups(HV: int, world_size: int) -> tuple[int, int, tuple[int, ...]]:
    """``(Hg, n_owners, Hg_r per rank)``: ``Hg = ceil(HV / W)``, owner ``r`` holds heads ``[r Hg, r Hg + Hg_r)`` with
    ``Hg_r = clamp(HV - r Hg, 0, Hg)``; ranks ``>= ceil(HV / Hg)`` own no heads (``HV < W``: owners ``0 .. HV-1``)."""
    HV, W = int(HV), int(world_size)
    if HV < 1 or W < 1:
        raise ValueError(f"head_groups: HV={HV} and world_size={W} must be >= 1")
    Hg = _ceil_div(HV, W)
    n_owners = _ceil_div(HV, Hg)
    hg_r = tuple(max(0, min(Hg, HV - r * Hg)) for r in range(W))
    return Hg, n_owners, hg_r


def owner_of_head(h: int, HV: int, world_size: int) -> int:
    Hg, _n, _hg = head_groups(HV, world_size)
    if not 0 <= h < HV:
        raise ValueError(f"owner_of_head: head {h} out of range [0, {HV})")
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
    """Everything of one rank's context that depends on the head count ``HV`` (``ctx.heads(HV)``).

    Ownership (SPEC-5): ``Hg = ceil(HV / W)``, owner ``j`` holds the compact head set ``[j Hg, j Hg + Hg_j)``,
    ``Hg_j = clamp(HV - j Hg, 0, Hg)``; ``Hg_r`` / ``head0`` are this rank's.  Every owner with heads holds ``Hg`` tile
    slots per fragment slot (``Hg_pad_by_rank``): the single partial owner (``padded_owner``, ``0 < Hg_j < Hg``, only
    when ``HV % Hg != 0``) is padded by ``pad_tiles = n_owners Hg - HV`` tiles per slot so that the kernels' two-integer
    addressing ``((h // Hg) * n + slot) * Hg + h % Hg`` is exact for every ``HV``; the padding is counted in that
    owner's splits, never read, and reported in ``pad_numel`` (elements per split; all zero for ``HV = 64``).
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
    ``dht_index_fused`` are the one-pool form of the same tables for lever L-B (``fused_walks``; the shipped
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

    HV: int
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


def _build_head_plan(ctx: KcpContext, HV: int, *, P: int | None = None) -> KcpHeadPlan:
    part = ctx.partition
    W, r = part.world_size, ctx.rank
    HV = int(HV)
    if HV < 1:
        raise ValueError(f"heads: HV must be >= 1, got {HV}")
    if HV * HEAD_DIM * CHUNK >= 2**31:
        raise ValueError(
            f"heads: HV={HV} overflows the kernels' per-chunk int32 offsets"
        )
    if ctx.R * HV * CHUNK >= 2**31:
        raise ValueError(
            f"heads: R={ctx.R} padded rows (T_r={ctx.T_r}, n_frag={ctx.n_frag}) x HV={HV} overflow the kernels' int32 "
            "row-head offsets: R * HV * 64 must stay below 2**31"
        )
    Hg, n_owners, hg_by_rank = head_groups(HV, W)
    Hg_r = hg_by_rank[r]
    # every owner with heads holds Hg tile slots per fragment slot; the single partial owner (0 < Hg_j < Hg,
    # only when HV % Hg != 0) is padded to Hg so that the kernels' two-integer addressing is exact for every HV.  The
    # padding tiles are counted in that owner's splits, never read, and reported separately (pad_numel).
    hg_pad_by_rank = tuple(Hg if x > 0 else 0 for x in hg_by_rank)
    Hg_pad_r = hg_pad_by_rank[r]
    full_groups = n_owners * Hg == HV  # no padding anywhere (validation aid)
    padded_owner = next((j for j, x in enumerate(hg_by_rank) if 0 < x < Hg), -1)
    pad_tiles = n_owners * Hg - HV
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

    # sender side: the two-integer form ((h // Hg) * n + slot) * Hg + h % Hg (exact for every HV with the padding)
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
        tuple(c1_send_tile(ctx.out_slot[f.slot], h) for h in range(HV))
        for f in mine
        if f.crossing_out
    )
    h0_tile = tuple(
        tuple(s_in_tile(ctx.in_slot[f.slot], h) for h in range(HV))
        for f in mine
        if f.crossing_in
    )
    dht_tile = tuple(
        tuple(ds_out_tile(ctx.out_slot[f.slot], h) for h in range(HV))
        for f in mine
        if f.crossing_out
    )
    b1_d_tile_d = tuple(
        tuple(b1_send_d_tile(ctx.in_slot[f.slot], h) for h in range(HV))
        for f in mine
        if f.crossing_in
    )
    b1_d_tile_dm = tuple(
        tuple(2 * b1_send_pair(ctx.in_slot[f.slot], h) + TILE_D for h in range(HV))
        for f in mine
        if f.crossing_in
    )
    b1_m_tile_dm = tuple(
        tuple(2 * b1_send_pair(ctx.in_slot[f.slot], h) + TILE_M for h in range(HV))
        for f in mine
        if f.crossing_in
    )

    # -- segment plan and the K1 / K2 / K3 segment index tables
    nc_max_frag = max((_ceil_div(f.length, CHUNK) for f in mine), default=0)
    if ctx.p_rule in ROW_RULES:  # a row rule: every rank's table, this rank's plan
        seg = kcp_plan_row(row_tables(part), HV, ctx.sm_count, p_rule=ctx.p_rule)[r]
    else:
        seg = kcp_plan(ctx.n_frag, nc_max_frag, HV, ctx.sm_count, p_rule=ctx.p_rule)
    if (
        P is not None
    ):  # the segments override of the GDN host options; the rule's plan is cached apart
        p_max = min(MAX_SEGMENTS, max(1, nc_max_frag))
        if not 1 <= int(P) <= p_max:
            raise ValueError(
                f"heads: P={P} override must be in [1, {p_max}] (MAX_SEGMENTS, chunks of the longest fragment)"
            )
        seg = dataclasses.replace(
            seg, P=int(P), NSEG=ctx.n_frag * int(P), p_rule="override"
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
    # the one-pool tables of lever L-B (fused_walks; the shipped form since an earlier revision, fused_walks=0 via the knob)
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
    # the P > 1 staging is two dense planes stage_b / stage_m of [NSEG, HV, 128, 128] each (K1 / K2 write stage_b, K3 writes
    # stage_m, both at tile_stride 1) -- counted together as one `staging` term of NSEG * HV * SUMMARY_TILE elements;
    # the carry buffer holds the dense inter-segment states
    numel["staging"] = (NSEG * HV * SUMMARY_TILE) if P > 1 else 0
    numel["carry"] = (ctx.n_frag * (P - 1) * HV * TILE) if P > 1 else 0
    numel["carry_fused"] = (
        ((extra + max(n_in, n_out)) * HV * TILE) if P > 1 else 0
    )  # lever L-B's carry plane
    dev = ctx.device
    i32, i64 = torch.int32, torch.int64
    # every table of the head plan in one asynchronous upload (upload_tables: no synchronizing call)
    tiles = {
        "out_tile": out_tile,
        "h0_tile": h0_tile,
        "dht_tile": dht_tile,
        "b1_d_tile_d": b1_d_tile_d,
        "b1_d_tile_dm": b1_d_tile_dm,
        "b1_m_tile_dm": b1_m_tile_dm,
    }
    device_tables = upload_tables(
        {
            "seg_chunk_start": (seg_table, i32, None),
            "h0_index": (h0_index, i32, None),
            "carry_index": (carry_index, i32, None),
            "dht_index": (dht_index, i32, None),
            "dcarry_index": (dcarry_index, i32, None),
            "out_index": (out_index, i32, None),
            "walk_seg_offset": (walk_offset, i32, None),
            "walk_seg_tape": (walk_tape, i32, None),
            **{
                name: ([list(x) for x in rows], i32, (len(rows), HV))
                for name, rows in tiles.items()
            },
            "fwd_chain_start": (fwd_start, i32, None),
            "fwd_chain_doc": (fwd_doc, i32, None),
            "fwd_m_off": (fwd_m, i64, None),
            "fwd_b_off": (fwd_b, i64, None),
            "fwd_out_off": (fwd_out, i64, None),
            "adj_chain_start": (adj_start, i32, None),
            "adj_chain_doc": (adj_doc, i32, None),
            "adj_d_off_d": (adj_dd, i64, None),
            "adj_d_off_dm": (adj_ddm, i64, None),
            "adj_m_off_b1": (adj_mb1, i64, None),
            "adj_m_off_c1": (adj_mc1, i64, None),
            "adj_out_off": (adj_out, i64, None),
        },
        dev,
    )
    return KcpHeadPlan(
        HV=HV,
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
    """Debug-only collective sequence counter: ``next(name)`` numbers the collectives a process issues,
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
    int32, ``hist_index`` int64).  ``heads(HV)`` derives and caches the head-dependent plan (:class:`KcpHeadPlan`).
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
    p_rule: str = P_RULE_DEFAULT  # plan rule: "row_cost_lpt" shipped ("row_cost" stays selectable); the per-rank rules via the test knob
    sequence: KcpSequence = dataclasses.field(default_factory=KcpSequence, repr=False)
    _head_plans: dict[int, KcpHeadPlan] = dataclasses.field(
        default_factory=dict, repr=False
    )
    # GDN table bundles of ``cake_tables`` (built once per context at first use, device-resident, keyed by HV)
    _caches: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def fragments(self) -> tuple[KcpFragment, ...]:
        return self.partition.fragments[self.rank]

    @property
    def ring_eligible(self) -> bool:
        """The design's ring alternative applies when every document chain has at most three fragments."""
        return self.n_d_max - 1 <= 2

    def heads(self, HV: int, *, P: int | None = None) -> KcpHeadPlan:
        """The head-dependent plan for ``HV`` heads (cached per ``HV``).  ``P`` (keyword-only; the GDN host's
        segments override) forces the segment count per fragment instead of the plan rule; overridden plans are
        cached apart from the rule's plan."""
        HV = int(HV)
        if P is not None:
            key = ("heads", HV, int(P))
            hp = self._caches.get(key)
            if hp is None:
                hp = _build_head_plan(self, HV, P=int(P))
                self._caches[key] = hp
            return hp
        hp = self._head_plans.get(HV)
        if hp is None:
            hp = _build_head_plan(self, HV)
            self._head_plans[HV] = hp
        return hp

    def prepare(self, H_qk: int, HV: int, *, P: int | None = None):
        """The device-resident GDN table bundle of this context for the head pair, built once per
        ``(HV, sm_count, P)`` key at first use (never at ``build_kcp_context``, which has no ``HV``, and
        never per call).  ``HV`` must be a multiple of ``H_qk``.  Delegates to :func:`cake_tables.prepare`."""
        from . import (
            cake_tables,
        )  # package-local; imported lazily (cake_tables imports this module)

        return cake_tables.prepare(self, H_qk, HV, P=P)

    def describe(self, HV: int | None = None) -> dict:
        """Plain-int summary for reports (optionally with the head plan of ``HV``)."""
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
        if HV is not None:
            hp = self.heads(HV)
            out["heads"] = {
                "HV": hp.HV,
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
    p_rule: str = P_RULE_DEFAULT,
) -> KcpContext:
    """The :class:`KcpContext` of ``rank`` (the KCP contract §3.1).

    ``group`` is the caller's ``torch.distributed`` process group (or ``None`` on the CPU / in a single process);
    when a group is given and the default process group is initialized, its size and this process's rank are
    checked against ``part.world_size`` / ``rank``.  ``device`` hosts the int32 tables.  ``conv_kernel_size - 1``
    history rows feed a continuing fragment's first outputs.  ``sm_count`` drives the segment plan (default: the
    CUDA device's multiprocessor count; ``1`` on a non-CUDA device = no occupancy pressure, ``P = P_chain``).
    Out-of-range problems are rejected here, before any launch: ``n_frag > 65536`` (the layout table's document
    search), int32 overflows of the kernels' offsets (``heads(HV)``).  ``p_rule`` (keyword-only; a test knob of the
    options object) selects the plan's ``P`` rule for this context (:data:`P_RULE_DEFAULT` = ``"row_cost_lpt"``;
    ``"row_cost"`` = the selectable legacy row rule of :func:`kcp_plan_row`; ``"wave_fill1"`` = its base program and
    the former default, ``"wave"`` = the
    earlier rule, ``"occ"`` = the occupancy rule, ``"wave_fill"`` = the unadopted fill on every rank with at most
    three fragments, ``"chain_min"`` / ``"one"`` = the minimal-segment programs, ``"row_cost_lpt"`` = the row rule with
    the per-rank LPT argmin on the wave-regime ranks, all kept for the A/B record); the public API never selects a rule
    other than the default.
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
    if p_rule not in P_RULES:
        raise ValueError(
            f"build_kcp_context: p_rule must be one of {P_RULES}, got {p_rule!r}"
        )
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
    flags = [
        (FLAG_CROSSING_IN if f.crossing_in else 0)
        | (FLAG_CROSSING_OUT if f.crossing_out else 0)
        | (FLAG_CONTINUING if (f.length > 0 and f.off > 0) else 0)
        for f in mine
    ]
    frag_rows = [
        [f.doc, f.off, f.length, fl] for f, fl in zip(mine, flags, strict=True)
    ]
    halo_plan = _halo_plan(part, rank, halo, local_cu)
    # every table of the context in one asynchronous upload (upload_tables: no synchronizing call)
    extra: dict[str, TableSpec] = {
        "frag_table": (frag_rows, i32, (n_frag, 4)),
        "hist_row_off": (
            [list(r) for r in halo_plan.hist_row_off],
            i32,
            (n_frag, halo),
        ),
        "tail_row_off": (
            [list(r) for r in halo_plan.tail_row_off],
            i32,
            (n_frag, halo),
        ),
        "head_row_off": (
            [[list(c) for c in r] for r in halo_plan.head_row_off],
            i32,
            (n_frag, halo, halo),
        ),
        "hist_index": (list(halo_plan.hist_index), torch.int64, None),
        "cu_seqlens_local": (list(local_cu), i32, None),
    }
    assert not set(extra) & set(tables_cpu), set(extra) & set(tables_cpu)
    uploaded = upload_tables(
        {**{name: (values, i32, None) for name, values in tables_cpu.items()}, **extra},
        device,
    )
    tables = {name: uploaded[name] for name in tables_cpu}
    frag_table = uploaded["frag_table"]
    halo_tables = {
        name: uploaded[name]
        for name in ("hist_row_off", "tail_row_off", "head_row_off", "hist_index")
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
        cu_seqlens_local=uploaded["cu_seqlens_local"],
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


__all__ = [
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
    "P_BAL_INACTIVE",
    "P_RULES",
    "P_RULE_DEFAULT",
    "SAVE_TRANSFER_DEFAULT",
    "STATE_TILE_BYTES",
    "SUMMARY_TILE",
    "TABLE_NAMES",
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
    "kcp_partition",
    "kcp_plan",
    "near_equal",
    "owner_of_head",
    "seg_chunk_start",
    "upload_tables",
    "UPLOAD_ALIGN_BYTES",
    "WAVE_FILL_N_FRAG_MAX",
    "WAVE_MIN_NC_MAX_FRAG",
    "WAVE_N_FRAG_MAX",
    "WAVE_RULE_10NS",
    "lpt_makespan",
    "wave_rule_cost_10ns",
    "wave_rule_cost_us",
    "wave_rule_lpt_cost_10ns",
    "wave_rule_lpt_P",
]
