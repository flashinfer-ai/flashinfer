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

Host backend of the Cake packed KDA chunked training kernels (SM100 / SM103 /
SM107): the host plan, the workspace arena, the saved set, the forward and
backward launch sequences and the autograd Function.

Operands (one packed batch of documents, GLM-5.3-Flash geometry): ``q, k, v``
BF16 ``[1, S, H, 128]`` (``q`` / ``k`` L2-normalized by the caller), ``g`` FP32
``[1, S, H, 128]`` (the natural-log decay gate, preactivated), ``beta`` BF16
``[1, S, H]`` (already through the sigmoid), ``cu_seqlens`` int32
``[num_docs + 1]`` on the device, ``out`` / ``dout`` BF16 like ``v``;
gradients ``dq, dk, dv, dbeta`` BF16 and ``dg`` FP32.  The leading batch axis
of 1 is optional on every operand and the outputs take the rank of the inputs.
Every operand is contiguous; a strided view is refused, never copied.

The host reads no tensor contents: the chunk tables are built on the device
from ``cu_seqlens`` -- on the segmented route up to 256 documents by the intra
program ``fwd_intra_v14`` (the block table too; no table launch) on the forward
and the replay and by ``table_v3`` where a saved-set path rebuilds them, else
by the ``table`` program (``table_v2`` on the serial route up to 256 documents;
``seg_table`` writes the segmented route's block table above) -- the
chunk-shaped programs launch at the host bound
``num_chunks_pad`` and treat ``chunk_len == 0`` as a no-op, and the persistent
programs read the real chunk count from ``seq_chunk_start[num_docs]``.  Which
variant of a stage runs and how large its grid is follow from the host
scalars (``S``, ``num_docs``, ``H``, the device's SM count and architecture)
and the path of the call (:data:`cake_jit.PATHS`) through the rules of
:mod:`.cake_jit`.  Every launch is ``cake_jit.run_module`` with a plan grid;
nothing here synchronizes or reads device memory.

Launch sequences.  Forward: table (the serial route; the segmented route above
256 documents, with ``seg_table`` after it; up to 256 documents the segmented
route's intra program writes the tables and no table runs), intra (``Aqk`` /
``Akk`` and
the GLM prep's outputs: the gated keys / queries, the gate row and the
``v_new`` pads), then on the serial route (more than eight heads)
the state recurrence, which forms the output itself and, from 32 heads, the
WY representation too (below that ``wy_fwd`` runs first, forming the gated
values from the user's ``v`` / ``beta`` token rows), and on the segmented
route (at most eight heads) ``wy_fwd``, pass 1 of the recurrence, ``seg_carry``
and pass 2 (which forms the output).  With ``save_for_backward`` (the ``fwd``
path) the recurrence and WY programs read the saved ``Aqk`` / ``Akk`` token rows
where a program for them ships and the intra program writes no internal copies;
without it (``fwd_nosave``) the internal-row forms run.  Backward: GLM prep, [intra on the
checkpoint-replay path: before the prep where the intra program writes the
tables the prep reads, after it otherwise], [WY], [the state recurrence
recompute], [dA / dv],
the adjoint state recurrence, the fused dq / dk / dg, GLM intra and, on
``sm_100a``, the gate epilogue (the GLM intra programs of ``sm_103a`` / ``sm_107a``
write the gate gradients themselves).  The bracketed launches depend on the path
and the row: from a version-2 (tape) saved set there is no recurrence
recompute; from a version-4 set carrying the segment-boundary states the
segmented recompute is pass 2 alone; the rows whose version-4 set carries the
WY representation (the paired serial 16-head rows of ``sm_100a``, the
segmented 16-head rows of ``sm_103a`` / ``sm_107a`` above 32 (document, head)
chains) launch no WY program; the serial recompute of the 64-column form from
64 heads forms the WY
representation itself (no WY launch) and leaves the transposed ``w`` tile the
matching adjoint recurrence reads; the saved-state pass 2 of the segmented
rows with at most 32 (document, head) chains does the same; the serial adjoint
recurrence accumulates ``dv`` itself, so ``dav`` computes ``dAqk`` alone there
and from 16 heads the fused gradient hosts even that (no ``dav`` launch); the
segmented adjoint recurrence accumulates ``dv`` too and the fused gradient
hosts ``dAqk`` on every segmented row (no ``dav`` launch).  The fused gradient and the GLM
intra kernel read the user's ``q`` / ``k`` / ``v`` token rows where they are;
the backward programs of the saved-set paths read the saved ``Aqk`` / ``Akk``
token rows and the user's ``dout``, the replay path's programs the internal
rows the replay recomputes (``aqk_int`` / ``akk_int`` / ``do_int``).  The two
state recurrences (``fwdh`` / ``fwdh_p3``, ``dhu``) run as the serial walk over
each document's chunk chain (four CTAs of 32 value columns when that one-wave
form fits the device, else two of 64) for more than eight heads -- except the
16-head rows of ``sm_103a`` / ``sm_107a``, which take the segmented route too --
and as the segmented three passes otherwise (:func:`cake_jit.segment_plan`):
pass 1 walks every segment of every document in parallel (the forward's from
zero and from the identity: the segment offset and transfer; the backward's
reverse walk from zero), ``seg_carry`` chains the segments' states within
each document (TF32 on the tensor cores; with at most two heads, where its
host rule accepts, the two-level carry ``seg_carry_v8`` forms the groups of
segments in parallel), pass 2 walks again
from the carried states and writes the tapes.

Memory.  A :class:`KdaWorkspace` is one arena holding only the internal
intermediates; buffers with disjoint lifetimes within one call share a slot
(:data:`FWD_SLOTS` / :data:`BWD_SLOTS` are the lifetime tables; the saved-set
paths use the backward layouts without the ``akk_int`` buffer, the replay the
full ones, and a backward workspace is sized for the larger).  Outputs, the
saved set and the tables are per-call allocations the caller owns.  A call
without a workspace allocates its arena at its start and frees it at its end;
between calls it keeps host objects only (the cached :func:`plan`, the layout
views of a workspace whose arena storage is released), no device memory.
:func:`workspace_bytes`, :func:`saved_bytes`, :func:`output_bytes` and
:func:`memory_report` are the pre-allocation queries.
``save_mode="light"`` (the default; saved set version 4) saves ``Aqk`` /
``Akk`` on the token rows (256 B per token-head), the tables, the fold's ``qg``
(256 B per internal token-head: the light backward's prep writes none), on the
rows whose light backward reads it from the set (the paired serial 16-head
rows of ``sm_100a``, the segmented 16-head rows of ``sm_103a`` / ``sm_107a``
above 32 (document, head) chains) the WY representation ``w`` / ``u`` (512 B
more; that backward launches no WY program) and, on the
segmented route when ``S >= 384 * NSEG`` (:func:`seg_states_saved`: the three
FP32 state blocks are then no larger than the ``Aqk`` / ``Akk`` set), the
forward's segment-boundary states ``hseg`` and ``seg_state``, so the
backward's recurrence recompute is pass 2 alone and its reverse carry reads
the saved transfers; shorter packs keep the token-row set and the backward
recomputes the states in its workspace (bitwise the same gradients either
way).  ``save_mode="tape"`` (saved set version 2) keeps the forward's ``h``
tape and ``v_new`` instead (768 B per token-head, plus the segment transfers
on the segmented route), so the backward skips its recurrence recompute (WY
still runs for ``w``).

Repeated calls on the same inputs give bitwise the same outputs: the kernels
reduce in a fixed order and use no atomic adds.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import threading
import types
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch

from . import cake_jit
from .cake_jit import CHUNK, HEAD_DIM, PATHS, STAGES, STAGES_BWD, STAGES_FWD

BT = CHUNK  # tokens per chunk
D = HEAD_DIM  # head dimension of q, k, v and g
MAX_DOCS = 65536  # the layout table finds a chunk's document with a 16-step search over [0, num_docs)
SCALE = D**-0.5  # the default softmax-free scale of the GLM contract
# saved set version 4 (light): {Aqk, Akk, tables, qg} + the segment-boundary states {hseg, seg_state} on the
# segmented route when the size rule holds + the WY representation {w, u} on the rows whose light backward reads
# it from the set (the paired serial 16-head rows of sm_100a, the segmented 16-head rows of sm_103a / sm_107a above
# 32 (document, head) chains: no wy launch there); versions 1 and 3 (the light sets without qg) are no longer
# produced or accepted
SAVED_VERSION = 4
SAVED_VERSION_TAPE = (
    2  # saved set version 2: + the forward's h tape and v_new (+ the segment transfers)
)
# light saves the segment states iff S >= 384 * NSEG: the three FP32 state blocks (3 * NSEG * H * 64 KiB) are then
# no larger than the Aqk / Akk set (2 * S * H * 256 B); shorter packs recompute them in the backward
SEG_SAVE_MIN_TOKENS_PER_SEG = 384
SAVE_MODES = ("light", "tape")
ALIGN = 256  # every arena sub-buffer and table starts on a 256 B boundary (TMA needs 16 B; 256 B also serves .view)
# The arena layouts: the forward's, the backward's of the saved-set paths (``bwd``) and of the checkpoint replay
# (``replay``, which keeps the internal ``akk_int`` rows), each in its light and tape form.
MODES = ("fwd", "fwd_tape", "bwd", "bwd_tape", "replay", "replay_tape")
# The lower-memory arena option (:class:`KdaArena`, ``lean``): the saved-set paths' backward layouts without the
# replay-only buffers, with the tail stages split into ``tail_passes`` chunk ranges (per-pass buffers sized for one
# range).  Not in :data:`MODES`: a lean workspace serves no replay.
LEAN_MODES = ("bwd_lean", "bwd_tape_lean")
# The device tables, in the order of the table buffer (the fifth on the segmented route only).
TABLE_NAMES = (
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seq_order",
    "seg_table",
)
# The stages each path launches, in registry order (``seg_carry`` where the segmented route launches a
# recurrence; the tables only when the saved set carries none).  The two forward paths launch the same stages:
# their records differ (the saving forward's token-row forms).
_FWD_STAGES = tuple(s for s in STAGES if s in STAGES_FWD or s == "seg_carry")
_PATH_STAGES: dict[str, tuple[str, ...]] = {
    "fwd": _FWD_STAGES,
    "fwd_nosave": _FWD_STAGES,
    "replay": tuple(
        s
        for s in STAGES
        if s in STAGES_BWD or s in ("table", "seg_table", "fwd_intra", "seg_carry")
    ),
}
for _path in ("light", "light_states", "tape"):
    _PATH_STAGES[_path] = tuple(s for s in _PATH_STAGES["replay"] if s != "fwd_intra")
del _path

# A launcher takes a record name, the logical argument values, the plan grid and the device index and runs the
# program (``cake_jit.run_module``).  The workspace holds one so a diagnostic harness can run the same composition
# against another build of the same kernels; the package itself always launches through ``cake_jit``.
Launcher = Callable[..., None]


@dataclasses.dataclass(frozen=True)
class KdaArena:
    """The lower-memory arena option of a :class:`KdaWorkspace`; off by default (``KdaArena()`` is today's arena and
    launch list).

    ``lean`` sizes a backward workspace for the saved-set paths only (layouts ``bwd_lean`` / ``bwd_tape_lean``: no
    replay layout, no internal ``dout`` copy), so the workspace serves no checkpoint replay (``chunk_kda_bwd`` with
    ``saved=None``; ``torch.utils.checkpoint`` never replays: the node's backward reads the recomputed saved set)
    and no row that dispatches the separate ``dav`` program (H 9-15, outside the contract's head counts).
    ``tail_passes = B`` (which implies ``lean``) runs the three chunk-parallel tail stages of the backward (the
    fused gradient, GLM intra and, on ``sm_100a``, the gate epilogue) once per range of ``ceil(NC_pad / B)`` chunks
    on offset views of the whole-sequence buffers, with the per-pass buffers (``dq_f, dk_f, dg_f, db_f, dAkk, dAqk``
    and, where the gate epilogue launches, ``dg_int, db_int``) sized for one range: the same kernels on the same bytes (bitwise the same gradients), ``B - 1`` more launches
    per tail stage.  The arena bytes: :func:`workspace_bytes` / :func:`memory_report` with ``arena=``; the floor
    from four passes on is 2820 B per internal token-head (plus three state blocks on the segmented route), against
    4488 at head (passes above four buy nothing).

    ``batch_tokens`` runs the forward and the backward over groups of whole, contiguous documents of at most that
    many tokens (and at most ``max_docs_per_batch`` documents; :func:`doc_batches`), one group after another, in
    an arena sized for the largest group instead of the whole pack (:meth:`Plan.batches`, :meth:`Plan.batch_plan`):
    the same kernels on the same token rows (every program reaches the tokens through the absolute ``chunk_bos``
    offsets of its group's tables), the records, the segments per document and the saved set pinned to the whole
    problem's, so the results are bitwise the whole problem's, except that an exactly-zero entry of ``u`` / ``w``
    in a chunk whose pair partner changed (the paired WY programs accumulate ``0 * partner``) may change sign.
    The calls then take ``doc_lengths``, the documents' token counts as host integers (the host reads no tensor).
    With ``tail_passes = B`` the arena is sized for the largest group (or the bound of every group) at ``B``
    passes, and each group runs its tail in the fewest passes ``b`` up to ``B`` whose lean layout at its own
    geometry fits that arena (one pass: the one-pass lean layout): each tail stage launches once per range of
    ``ceil(NC_pad / b)`` chunks instead of ``ceil(NC_pad / B)``, at the same bits.
    The option refuses the rows whose block table gives the documents proportional segment counts (at most four
    (document, head) chains on the segmented route).  The longest document bounds the arena from below
    (:func:`memory_report` with ``doc_lengths``); :func:`arena_for_budget` / :meth:`KdaWorkspace.for_budget` search
    the options for a byte budget.
    """

    lean: bool = False
    tail_passes: int = 1
    batch_tokens: int | None = None
    max_docs_per_batch: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.tail_passes, bool)
            or not isinstance(self.tail_passes, int)
            or self.tail_passes < 1
        ):
            raise ValueError(
                f"KdaArena: tail_passes must be an int >= 1, got {self.tail_passes!r}"
            )
        for name in ("batch_tokens", "max_docs_per_batch"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"KdaArena: {name} must be None or an int >= 1, got {value!r}"
                )
        if self.max_docs_per_batch is not None and self.batch_tokens is None:
            raise ValueError(
                "KdaArena: max_docs_per_batch bounds the document groups of batch_tokens; pass batch_tokens too "
                "(S, or any budget at or above the longest document, bounds the groups by their document count "
                "alone)"
            )
        if self.tail_passes > 1 and not self.lean:
            object.__setattr__(
                self, "lean", True
            )  # the tail split lives in the lean layouts

    @property
    def batched(self) -> bool:
        """Whether the calls run over document groups (``batch_tokens`` set)."""
        return self.batch_tokens is not None

    def mode(self, base: str) -> str:
        """The layout mode of ``base`` (``"bwd"`` / ``"bwd_tape"`` / the replay forms) under this option."""
        if not self.lean or base in LEAN_MODES or base.startswith("fwd"):
            return base
        if base.startswith("replay"):
            raise ValueError(
                "a lean workspace (KdaArena(lean=True)) serves no checkpoint replay (chunk_kda_bwd with "
                "saved=None): pass the saved set, or allocate the workspace with lean=False"
            )
        return base + "_lean"


# ----------------------------------------------------------------------------------------------------------------
# Host plan (pure function of the host-known scalars)


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _geometry(S: int, num_docs: int) -> tuple[int, int, int]:
    """``(NC_max, NC_pad, R)``: the chunk bound ``ceil(S / 64) + num_docs`` (at or above the real chunk count of
    every packing), its even round-up (the table length and the grid bound of the chunk-shaped programs) and the
    internal row count ``64 * NC_pad``."""
    nc_max = -(-S // BT) + num_docs
    nc_pad = nc_max + nc_max % 2
    return nc_max, nc_pad, nc_pad * BT


def _check_problem(
    S: int, num_docs: int, H: int, sm_count: int
) -> tuple[int, int, int]:
    """``(NC_max, NC_pad, R)`` of a problem within the host's limits (:func:`_geometry`); ``ValueError`` when a
    scalar is below one, ``num_docs`` exceeds :data:`MAX_DOCS` (the depth of the layout table's document search)
    or an int32 offset of the kernels would overflow: ``H * 128 * 64`` per chunk, ``R * H * 64`` over the padded
    rows of the ``[R, H, 64]`` intra-chunk buffers.  :func:`plan` and the byte queries share these checks."""
    if S < 1 or num_docs < 1 or H < 1 or sm_count < 1:
        raise ValueError(
            f"plan: S={S}, num_docs={num_docs}, H={H}, sm_count={sm_count} must all be >= 1"
        )
    if H * D * BT >= 2**31:
        raise ValueError(f"plan: H={H} overflows the kernels' per-chunk int32 offsets")
    if num_docs > MAX_DOCS:
        raise ValueError(
            f"plan: num_docs={num_docs} exceeds {MAX_DOCS}, the depth of the layout table's document search"
        )
    nc_max, nc_pad, rows = _geometry(S, num_docs)
    if rows * H * BT >= 2**31:
        raise ValueError(
            f"plan: R={rows} padded rows (S={S}, num_docs={num_docs}) x H={H} overflow the kernels' int32 "
            f"row-head offsets: R * H * 64 must stay below 2**31"
        )
    return nc_max, nc_pad, rows


@dataclasses.dataclass(frozen=True)
class Batch:
    """One document group of the batched arena option (:class:`KdaArena`, ``batch_tokens``; :meth:`Plan.batches`):
    the documents ``[d0, d1)`` of the pack (``N = d1 - d0`` of them, zero-length ones included; ``S`` tokens),
    ``c0`` the group's first chunk in the whole problem's layout (the real chunks of the documents before it),
    ``NC_pad`` the chunk bound of the group's own geometry (:func:`_geometry` of ``(S, N)``: its tables, grids and
    arena layouts) and ``NC_pad_tape`` the bound of its views of a whole-problem tape (``min(NC_pad,
    even_down(NC_pad_full - c0))``: the group's slice of a saved ``h`` / ``v_new`` ends inside them; at or above
    the group's real chunk count, since the whole problem's bound exceeds its real chunk count by at least one)."""

    d0: int
    d1: int
    S: int
    N: int
    c0: int
    NC_pad: int
    NC_pad_tape: int


def doc_batches(
    doc_lengths: Sequence[int],
    batch_tokens: int,
    max_docs_per_batch: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """The contiguous document groups ``(d0, d1)`` of a pack under the batched arena option, from host integers
    alone: greedy from the first document, a group takes the next document while its tokens stay at or under
    ``batch_tokens`` and its document count under ``max_docs_per_batch`` (``None``: unbounded).  A zero-length
    document never starts a group of its own (it joins the group before it, whatever the bounds: a group of no
    tokens would be no problem), so a pack's leading zero-length documents open its first group together with the
    first document that has tokens.  ``ValueError`` when a document alone exceeds ``batch_tokens`` (the budget's
    floor is the longest document), when a length is negative or the pack has no token."""
    lengths = tuple(int(x) for x in doc_lengths)
    budget = int(batch_tokens)
    cap = None if max_docs_per_batch is None else int(max_docs_per_batch)
    if not lengths or min(lengths) < 0:
        raise ValueError(
            f"doc_batches: doc_lengths must be non-negative ints of at least one document, got {lengths!r}"
        )
    if sum(lengths) < 1:
        raise ValueError("doc_batches: the pack has no token")
    for d, n in enumerate(lengths):
        if n > budget:
            raise ValueError(
                f"doc_batches: document {d} has {n} tokens, above batch_tokens={budget}; the budget's floor is the "
                f"longest document ({max(lengths)} tokens)"
            )
    groups: list[tuple[int, int]] = []
    d0, tokens = 0, 0
    for d, n in enumerate(lengths):
        full = tokens + n > budget or (cap is not None and d - d0 >= cap)
        if d > d0 and tokens > 0 and n > 0 and full:
            groups.append((d0, d))
            d0, tokens = d, 0
        tokens += n
    groups.append((d0, len(lengths)))
    return tuple(groups)


@dataclasses.dataclass(frozen=True)
class Plan:
    """Chunk bound, internal rows, the route of the state recurrences, the record of every stage on every path
    and its launch grids for one packed problem on one architecture.

    ``NC_max = ceil(S / 64) + num_docs`` is at or above the real chunk count of every packing of ``S`` tokens into
    ``num_docs`` documents; ``NC_pad`` rounds it up to even (the table length and the grid bound of the chunk-shaped
    programs; the pair-shaped programs cover two chunks per CTA); ``R = 64 * NC_pad`` is the internal row count.
    ``recur`` is ``"seg"`` (the segmented recurrences, ``P`` segments per document, ``NSEG = num_docs * P``
    pseudo-sequences; ``proportional`` / ``seg_arg`` are the block table's form and the walk programs' segment
    scalar, :func:`cake_jit.segment_args`) or ``"serial"`` (``P = 1``, ``NSEG = 0``); ``gsize`` is the segment
    group size of the two-level carry where the plan dispatches ``seg_carry_v8`` (:func:`cake_jit.carry_choice`),
    else 0; ``gate_fused`` whether the GLM intra program writes the gate gradients itself (``glm_intra_v14`` / ``glm_intra_v18``: no
    ``gate_epilogue`` launch, no ``dg_int`` / ``db_int`` buffers).  ``records[path]`` maps
    every stage of the path (:data:`cake_jit.PATHS`; ``light_states`` on the segmented route only) to the registry
    record its ``select_rule`` admits (``None`` for an optional stage the path or route does not launch) and
    ``grids`` every selected record to its launch grids, one per launch of the record by its stage
    (``cake_jit.launch_grid``).
    """

    S: int
    N: int  # documents (``cu_seqlens.numel() - 1``); zero-length documents count
    H: int
    sm_count: int
    arch: str
    NC_max: int
    NC_pad: int
    R: int
    recur: str
    P: int
    NSEG: int
    proportional: int
    seg_arg: int
    gsize: int
    gate_fused: bool
    records: dict[str, dict[str, str | None]]
    grids: dict[str, tuple[tuple[int, int, int], ...]]

    def dispatched(self, stage: str, path: str) -> bool:
        """Whether ``path`` launches a record of ``stage``."""
        return self._records(path).get(stage) is not None

    def _records(self, path: str) -> dict[str, str | None]:
        try:
            return self.records[path]
        except KeyError:
            if path == "light_states":
                raise ValueError(
                    "path 'light_states' exists on the segmented route only; this plan runs the serial "
                    "recurrence"
                ) from None
            raise ValueError(f"path must be one of {PATHS}, got {path!r}") from None

    def record(self, stage: str, path: str) -> str:
        records = self._records(path)
        if stage not in records:
            raise ValueError(f"stage {stage!r} is not a stage of the {path!r} path")
        name = records[stage]
        if name is None:
            raise ValueError(
                f"stage {stage!r} is not launched on the {path!r} path of this plan "
                f"({self.recur} route, {self.H} heads, {self.N} documents)"
            )
        return name

    def launches(self, stage: str, path: str) -> int:
        """How many times the stage launches its record on the path (two for the segmented adjoint recurrence's
        passes around the carry)."""
        return len(self.grids[self.record(stage, path)])

    def grid(self, stage: str, path: str, launch: int = 0) -> tuple[int, int, int]:
        return self.grids[self.record(stage, path)][launch]

    def internal_rows(self, path: str) -> bool:
        """Whether the intra program writes the internal-layout ``Aqk`` / ``Akk`` copies on ``path`` (its
        ``write_int``): a record of the path other than the intra program itself binds ``aqk_int`` or ``akk_int``.
        On the saving forward (``"fwd"``) the fused serial recurrence (``fwdh_wy_rows``), the overlap rows' WY and
        pass-2 programs (``wy_fwd_rows``, ``fwdh_p3_o_ov_rows`` / ``fwdh_p3_o_ov_id_rows``) and the paired serial rows'
        WY program and 32-column recurrence (``wy_fwd_rows``, ``fwdh_o_v2_rows``; on ``sm_107a`` the 64-column rows'
        ``wy_fwd_rows`` / ``fwdh_o_v1_rows`` too) read the saved token rows instead, so no copy is written there; the
        forward without a saved set (``"fwd_nosave"``), the other architectures' 64-column serial rows and the replay
        keep the internal-row forms and the copies."""
        return any(
            {"aqk_int", "akk_int"} & set(cake_jit.MODULES[name]["logical"].values())
            for stage, name in self._records(path).items()
            if name is not None and stage != "fwd_intra"
        )

    @property
    def light_saves_qg(self) -> bool:
        """Whether the light forward saves the fold's ``qg`` (version 4): the light backward's prep record writes
        none (``bwd_prep_g8_noqg*``: no ``qg`` among its logical names) and reads the saved tensor instead."""
        return "qg" not in self._light_prep_logical()

    @property
    def light_saves_wu(self) -> bool:
        """Whether the light forward saves the WY representation ``w`` / ``u`` (version 4): the light backward's
        prep record writes no ``kb`` (``bwd_prep_g8_noqg_nokb``: the paired serial 16-head rows of ``sm_100a`` and
        the segmented 16-head rows of ``sm_103a`` / ``sm_107a`` above 32 (document, head) chains) and neither light
        path launches a ``wy`` program; the saved tensors feed the recurrence recompute and the adjoint."""
        return "kb" not in self._light_prep_logical()

    def _light_prep_logical(self) -> set[str]:
        name = self._records("light")["bwd_prep"]
        if name is None:
            raise ValueError("the light path launches no prep record")
        return set(cake_jit.MODULES[name]["logical"].values())

    @property
    def seg_save(self) -> bool:
        """Whether the light forward of this plan saves the segment-boundary states (``hseg``, ``seg_state``):
        the segmented route is dispatched and ``S >= SEG_SAVE_MIN_TOKENS_PER_SEG * NSEG`` (:func:`seg_states_saved`);
        otherwise the backward recomputes them."""
        return seg_states_saved(self.S, self.NSEG)

    @property
    def light_path(self) -> str:
        """The path of the backward from this plan's own light saved set: ``"light_states"`` when the forward
        saves the segment-boundary states (:attr:`seg_save`), else ``"light"``."""
        return "light_states" if self.seg_save else "light"

    def stages(self, path: str) -> tuple[str, ...]:
        """The stages ``path`` launches on this plan, in registry order (the tables included; ``seg_carry`` on the
        segmented route)."""
        self._records(path)
        return tuple(
            stage
            for stage in _PATH_STAGES[path]
            if (stage == "seg_carry" and self.recur == "seg")
            or self.dispatched(stage, path)
        )

    def launch_count(self, path: str, *, tables: bool = True) -> int:
        """Programs launched by one call on ``path`` (a path name, or ``"bwd"`` for the backward from this plan's
        own light saved set, :attr:`light_path`).  A saved set carries the tables (``tables=True``): the table
        programs are not launched; ``tables=False`` counts the backward from a set whose tables the caller dropped
        (they are built again).  The forward and the replay build them where the plan dispatches a table program
        (the segmented route's intra program writes them up to 256 documents: no table launch there)."""
        if path == "bwd":
            path = self.light_path
        count = 0
        for stage in self.stages(path):
            if stage in ("table", "seg_table"):
                if (not tables) or path in ("fwd", "fwd_nosave", "replay"):
                    count += 1
            elif stage == "seg_carry":
                # the forward recurrence's carry where its pass 1 runs, the adjoint recurrence's on every
                # backward path
                count += int(self.dispatched("fwdh", path)) + int(
                    path not in ("fwd", "fwd_nosave")
                )
            else:
                count += self.launches(stage, path)
        return count

    def layout(self, mode: str, tail_passes: int = 1) -> tuple[Region, ...]:
        """The arena regions of ``mode`` (:data:`MODES`): ``"fwd"`` / ``"bwd"`` (``save_mode="light"``; the saved-set
        paths) / ``"replay"`` (the checkpoint replay, which keeps the internal ``akk_int`` rows), and their
        ``*_tape`` forms (``save_mode="tape"``: ``h`` and ``v_new`` live in the saved set); the lean layouts
        (:data:`LEAN_MODES`, :class:`KdaArena`) take their ``tail_passes``."""
        return _layout(
            self.R, self.H, self.NC_pad, self.NSEG, mode, tail_passes, self.gate_fused
        )

    def workspace_bytes(self, mode: str = "bwd", tail_passes: int = 1) -> int:
        return _mode_bytes(
            self.R, self.H, self.NC_pad, self.NSEG, mode, tail_passes, self.gate_fused
        )

    def check_arena(self, arena: KdaArena, *, backward: bool = True) -> None:
        """``ValueError`` when ``arena`` cannot serve this plan: a lean arena on a backward row that dispatches the
        separate ``dav`` program (H 9-15; the lean layouts keep no ``dv1`` and no whole-sequence ``dAqk``), a
        batched arena on a row whose block table gives the documents proportional segment counts (at most
        ``prop_max_chains`` (document, head) chains on the segmented route: a group of fewer documents would change
        every document's segments)."""
        if (
            backward
            and arena.lean
            and any(
                self.dispatched("dav", path)
                for path in self.records
                if path not in ("fwd", "fwd_nosave")
            )
        ):
            raise ValueError(
                f"KdaArena(lean=True) does not serve H={self.H}: the backward of this row launches the separate "
                "dA / dv program (dav); the lean layouts serve the contract's head counts 1, 2, 4, 8, 16, 32, 64"
            )
        if arena.batched and self.proportional:
            raise ValueError(
                f"KdaArena(batch_tokens=...) does not serve num_docs={self.N}, H={self.H}: with at most "
                f"{cake_jit.SEGMENTS['prop_max_chains']} (document, head) chains the segmented route's block table "
                "gives the documents proportional segment counts, which a group of fewer documents would change; "
                "the batched option serves the rows above that chain count and the serial route"
            )

    def arena_bytes(
        self,
        *,
        backward: bool = True,
        save_mode: str = "light",
        arena: KdaArena | None = None,
        doc_lengths: Sequence[int] | None = None,
    ) -> int:
        """The arena a :class:`KdaWorkspace` of this plan allocates: the larger of the saved-set and the replay
        layouts for a backward workspace (both paths run in it), the forward layout otherwise; under a lean
        ``arena`` the lean layout of its ``tail_passes`` (the saved-set paths only); under a batched ``arena`` the
        largest group's layout, over the groups of ``doc_lengths`` when given, else over the bound of every group
        of ``batch_tokens`` tokens (:meth:`batch_bound`)."""
        arena = KdaArena() if arena is None else arena
        self.check_arena(arena, backward=backward)
        if not arena.batched:
            return _arena_bytes(
                self.R,
                self.H,
                self.NC_pad,
                self.NSEG,
                backward=backward,
                save_mode=save_mode,
                arena=arena,
                gate_fused=self.gate_fused,
            )
        geometries = (
            [self.batch_bound(arena)]
            if doc_lengths is None
            else [
                (b.NC_pad, b.NC_pad * BT, self.batch_segments(b))
                for b in self.batches(arena, doc_lengths)
            ]
        )
        return max(
            _arena_bytes(
                rows,
                self.H,
                nc_pad,
                nseg,
                backward=backward,
                save_mode=save_mode,
                arena=arena,
                gate_fused=self.gate_fused,
            )
            for nc_pad, rows, nseg in geometries
        )

    def check_doc_lengths(self, doc_lengths: Sequence[int] | None) -> tuple[int, ...]:
        """``doc_lengths`` as a tuple of ints describing this problem (one per document in ``cu_seqlens`` order,
        none negative, summing to ``S``); ``ValueError`` otherwise, or when they are missing."""
        if doc_lengths is None:
            raise ValueError(
                "doc_lengths: the batched arena option (KdaArena(batch_tokens=...)) needs the documents' token "
                "counts as host integers (one per document in cu_seqlens order, summing to S): the host reads no "
                "tensor, so cu_seqlens cannot supply them"
            )
        lengths = tuple(int(x) for x in doc_lengths)
        if (
            len(lengths) != self.N
            or min(lengths, default=0) < 0
            or sum(lengths) != self.S
        ):
            raise ValueError(
                f"doc_lengths: {len(lengths)} documents summing to {sum(lengths)} tokens, the problem has "
                f"num_docs={self.N}, S={self.S}"
            )
        return lengths

    def batches(
        self, arena: KdaArena, doc_lengths: Sequence[int] | None
    ) -> tuple[Batch, ...]:
        """The document groups (:class:`Batch`) of this problem under a batched ``arena`` (:func:`doc_batches` of
        ``doc_lengths``, with every group's first chunk and chunk bounds)."""
        lengths = self.check_doc_lengths(doc_lengths)
        out: list[Batch] = []
        c0 = 0
        for d0, d1 in doc_batches(
            lengths, arena.batch_tokens, arena.max_docs_per_batch
        ):
            tokens, docs = sum(lengths[d0:d1]), d1 - d0
            _nc_max, nc_pad, _rows = _check_problem(tokens, docs, self.H, self.sm_count)
            left = self.NC_pad - c0
            out.append(
                Batch(d0, d1, tokens, docs, c0, nc_pad, min(nc_pad, left - left % 2))
            )
            c0 += sum(-(-n // BT) for n in lengths[d0:d1])
        return tuple(out)

    def batch_segments(self, batch: Batch) -> int:
        """The segment count of one group: ``P`` per document on the segmented route, none on the serial one."""
        return batch.N * self.P if self.recur == "seg" else 0

    def batch_bound(self, arena: KdaArena) -> tuple[int, int, int]:
        """``(NC_pad, R, NSEG)`` at or above every group's under ``arena`` for any ``doc_lengths``: the chunk bound
        of ``batch_tokens`` tokens in all the documents (a group has at most the budget's tokens and at most every
        document), capped at the whole problem's, and the whole problem's segment count."""
        tokens = min(int(arena.batch_tokens), self.S)
        nc_max = -(-tokens // BT) + self.N
        nc_pad = min(self.NC_pad, nc_max + nc_max % 2)
        return nc_pad, nc_pad * BT, self.NSEG

    def batch_plan(self, batch: Batch, *, tape: bool = False) -> Plan:
        """The plan of one document group: its own geometry (``batch.NC_pad``; with ``tape`` ``NC_pad_tape``, the
        bound of its views of a whole-problem tape), this plan's records (the whole problem's selection: every
        group runs the same programs), ``P`` segments per document with the non-proportional block table
        (``NSEG = N * P`` on the segmented route), the whole problem's two-level carry group size (its grid at the
        group's chain count: :func:`cake_jit.carry_grid`) and the grids of the records' rules at the group's
        scalars."""
        nc_pad = batch.NC_pad_tape if tape else batch.NC_pad
        nseg = self.batch_segments(batch)
        scalars = {
            "num_chunks_pad": nc_pad,
            "num_seqs": batch.N,
            "num_heads": self.H,
            "sm_count": self.sm_count,
        }
        grids = {
            name: tuple(
                cake_jit.launch_grid(
                    name, **scalars, num_segments=nseg, launch=launch, gsize=self.gsize
                )
                for launch in range(cake_jit.launches(name))
            )
            for name in self.grids
        }
        return Plan(
            S=batch.S,
            N=batch.N,
            H=self.H,
            sm_count=self.sm_count,
            arch=self.arch,
            NC_max=min(_geometry(batch.S, batch.N)[0], nc_pad),
            NC_pad=nc_pad,
            R=nc_pad * BT,
            recur=self.recur,
            P=self.P,
            NSEG=nseg,
            proportional=0,
            seg_arg=self.P if self.recur == "seg" else self.seg_arg,
            gsize=self.gsize,
            gate_fused=self.gate_fused,
            records=self.records,
            grids=grids,
        )

    def table_layout(self) -> tuple[tuple[str, int, int], ...]:
        """``(name, word offset, words)`` of the tables inside one int32 buffer, each 256 B aligned: ``chunk_bos``,
        ``chunk_len``, ``seq_chunk_start``, ``seq_order`` and, on the segmented route, the block table
        ``seg_table`` (``NSEG + 2 * num_docs + 1`` words)."""
        return _table_layout(self.NC_pad, self.N, self.NSEG)

    def table_words(self) -> int:
        _name, off, words = self.table_layout()[-1]
        return _round_up(off + words, ALIGN // 4)


@functools.cache
def _capability_arch(device_index: int) -> str | None:
    """The exported architecture of a CUDA device, ``None`` when no record targets it (a device's capability is
    fixed for the process)."""
    return cake_jit.arch_for_capability(torch.cuda.get_device_capability(device_index))


def _plan_arch(arch: str | None) -> str:
    """The architecture of a plan: ``arch``; else the current CUDA device's where it is an exported target; else
    (no CUDA device, or one no record targets: a query of the host scalars alone) the first exported architecture,
    whose records the byte queries also take without an ``arch`` (their segment count is then the
    architecture-independent one, :func:`_arch_sms`)."""
    if arch is not None:
        return arch
    if torch.cuda.is_available():
        found = _capability_arch(torch.cuda.current_device())
        if found is not None:
            return found
    return cake_jit.ARCHES[0]


def plan(S: int, num_docs: int, H: int, sm_count: int, arch: str | None = None) -> Plan:
    """The :class:`Plan` of a packed problem; reads no tensor.

    ``arch`` selects the architecture's records and, on the few-chain segmented rows, its segment count
    (:func:`cake_jit.segment_plan`; ``None``: the current CUDA device's architecture, so the plan serves a
    :class:`KdaWorkspace` on that device; without an exported device at hand the first exported architecture; the
    byte queries below take an ``arch`` of their own).  ``ValueError`` when a scalar is below one, ``num_docs``
    exceeds :data:`MAX_DOCS` (the depth of the layout table's document search) or an int32 offset of the kernels
    would overflow: ``H * 128 * 64`` per chunk, ``R * H * 64`` over the padded rows of the ``[R, H, 64]``
    intra-chunk buffers.

    The plan is a pure function of ``(S, num_docs, H, sm_count, arch)`` and is cached on them: a repeated problem
    returns the same :class:`Plan` object (its record and grid tables are shared, never modified), so the calls
    of the default entry plan a repeated shape once.
    """
    arch = _plan_arch(arch)
    if arch not in cake_jit.ARCHES:
        raise ValueError(f"plan: arch {arch!r} is not one of {cake_jit.ARCHES}")
    return _plan(int(S), int(num_docs), int(H), int(sm_count), arch)


@functools.lru_cache(maxsize=256)
def _plan(S: int, num_docs: int, H: int, sm_count: int, arch: str) -> Plan:
    """:func:`plan` of the normalized scalars (the cache; a refused problem raises every time)."""
    nc_max, nc_pad, rows = _check_problem(S, num_docs, H, sm_count)
    recur, segments, num_segments = cake_jit.segment_plan(num_docs, H, sm_count, arch)
    proportional, seg_arg = cake_jit.segment_args(num_docs, H, sm_count, arch)
    scalars = {
        "num_chunks_pad": nc_pad,
        "num_seqs": num_docs,
        "num_heads": H,
        "sm_count": sm_count,
    }
    # the saved-state path exists on the segmented route only (the serial route has no segment states)
    records = {
        path: {
            stage: cake_jit.select_record(stage, **scalars, arch=arch, path=path)
            for stage in _PATH_STAGES[path]
        }
        for path in PATHS
        if path != "light_states" or recur == "seg"
    }
    names = sorted(
        {
            name
            for by_stage in records.values()
            for name in by_stage.values()
            if name is not None
        }
    )
    grids = {
        name: tuple(
            cake_jit.launch_grid(
                name, **scalars, num_segments=num_segments, launch=launch
            )
            for launch in range(cake_jit.launches(name))
        )
        for name in names
    }
    # the two-level carry's group size where a path dispatches it (its host rule accepted the problem: the rule
    # of the record and its grid rule are the same function)
    gsize = 0
    for name in names:
        rule = cake_jit.MODULES[name]["grid_rule"]
        if rule["kind"] == "carry_groups":
            choice = cake_jit.carry_choice(
                num_docs, H, sm_count, num_segments, int(rule["s_hops"])
            )
            if choice is None:
                raise ValueError(
                    f"plan: record {name} is selected but its carry rule rejects the problem"
                )
            gsize = int(choice[0])
    return Plan(
        S=S,
        N=num_docs,
        H=H,
        sm_count=sm_count,
        arch=arch,
        NC_max=nc_max,
        NC_pad=nc_pad,
        R=rows,
        recur=recur,
        P=segments,
        NSEG=num_segments,
        proportional=proportional,
        seg_arg=seg_arg,
        gsize=gsize,
        gate_fused=records["light"]["gate_epilogue"] is None,
        records=records,
        grids=grids,
    )


def _gate_fused(arch: str | None) -> bool:
    """Whether ``arch``'s GLM intra program writes the gate gradients itself (no ``gate_epilogue`` record admits
    its rows; the layouts then carry no ``dg_int`` / ``db_int``).  ``arch=None`` (a byte query without an exported
    device at hand) takes the first exported architecture's records, as :func:`plan` does there."""
    arch = cake_jit.ARCHES[0] if arch is None else arch
    return (
        cake_jit.select_record(
            "gate_epilogue",
            num_chunks_pad=2,
            num_seqs=1,
            num_heads=1,
            sm_count=1,
            arch=arch,
            path="light",
        )
        is None
    )


def _light_prep_saves(
    num_chunks_pad: int, num_docs: int, H: int, sm_count: int, arch: str | None
) -> tuple[bool, bool]:
    """``(qg, w / u)``: the per-call tensors the light forward of the problem saves (version 4), read off the light
    backward's prep record as :attr:`Plan.light_saves_qg` / :attr:`Plan.light_saves_wu` do; ``arch=None`` (a byte
    query without an exported device at hand) takes the first exported architecture's records."""
    arch = cake_jit.ARCHES[0] if arch is None else arch
    name = cake_jit.select_record(
        "bwd_prep",
        num_chunks_pad=num_chunks_pad,
        num_seqs=num_docs,
        num_heads=H,
        sm_count=sm_count,
        arch=arch,
        path="light",
    )
    logical = set(cake_jit.MODULES[name]["logical"].values())
    return "qg" not in logical, "kb" not in logical


def seg_states_saved(S: int, NSEG: int) -> bool:
    """Whether ``save_mode="light"`` (saved set version 4) saves the segment-boundary states: the segmented route is
    dispatched (``NSEG > 0``) and ``S >= SEG_SAVE_MIN_TOKENS_PER_SEG * NSEG``, i.e. the three FP32 state blocks
    (``3 * NSEG * H * 64 KiB``: ``hseg`` and the ``seg_state`` pair) are no larger than the ``Aqk`` / ``Akk`` set
    (``2 * S * H * 256 B``).  Shorter packs keep the token-row set and the backward recomputes the states in its
    workspace (the same kernels on the same inputs: bitwise the same gradients).  A pure function of the scalars."""
    return int(NSEG) > 0 and int(S) >= SEG_SAVE_MIN_TOKENS_PER_SEG * int(NSEG)


def arena_for_budget(
    p: Plan,
    arena_budget_bytes: int,
    *,
    backward: bool = True,
    save_mode: str = "light",
    doc_lengths: Sequence[int] | None = None,
) -> KdaArena:
    """The first arena option (:class:`KdaArena`) whose arena fits ``arena_budget_bytes`` on plan ``p``: the
    default, ``lean=True``, two tail passes, four, then four passes over document groups of ``batch_tokens``
    tokens, halving from half the pack down to the longest document and last one document per group
    (``doc_lengths`` required from the batched steps on: the floor is the longest document, and a smaller budget
    could only fail inside the calls).  An option the row refuses (lean on the rows with the separate ``dav``
    program, batching on the proportional rows) is skipped; the batched steps take the lean four-pass layouts where
    the row admits them.  ``ValueError`` naming the smallest arena seen when nothing fits."""
    budget = int(arena_budget_bytes)
    smallest: int | None = None

    def fits(option: KdaArena) -> bool | None:
        nonlocal smallest
        try:
            nbytes = p.arena_bytes(
                backward=backward,
                save_mode=save_mode,
                arena=option,
                doc_lengths=doc_lengths,
            )
        except ValueError:
            return None  # the row refuses this option
        smallest = nbytes if smallest is None else min(smallest, nbytes)
        return nbytes <= budget

    lean_ok = False
    for option in (
        KdaArena(),
        KdaArena(lean=True),
        KdaArena(lean=True, tail_passes=2),
        KdaArena(lean=True, tail_passes=4),
    ):
        verdict = fits(option)
        lean_ok = lean_ok or (option.lean and verdict is not None)
        if verdict:
            return option
    if doc_lengths is None:
        raise ValueError(
            f"arena_for_budget: no unbatched arena option fits {budget} B (the smallest is {smallest} B); pass "
            "doc_lengths to search the batched option, whose floor is the longest document"
        )
    longest = max(p.check_doc_lengths(doc_lengths))
    tokens = max(longest, p.S // 2)
    passes = 4 if lean_ok else 1
    while True:
        option = KdaArena(lean=lean_ok, tail_passes=passes, batch_tokens=tokens)
        verdict = fits(option)
        if verdict:
            return option
        if verdict is None:
            raise ValueError(
                f"arena_for_budget: the row refuses the batched option and no other fits {budget} B (the smallest "
                f"is {smallest} B)"
            )
        if tokens <= longest:
            break
        tokens = max(longest, tokens // 2)
    option = KdaArena(
        lean=lean_ok, tail_passes=passes, batch_tokens=longest, max_docs_per_batch=1
    )
    if fits(option):
        return option
    raise ValueError(
        f"arena_for_budget: no arena option fits {budget} B; the floor (one document per group) is {smallest} B"
    )


# ----------------------------------------------------------------------------------------------------------------
# Arena layout: the lifetime table of every internal buffer
#
# Buffer kinds (bytes per internal token-head, R * H of them): F32 = fp32 [R, H, 128] (512), B16 = bf16 [R, H, 128]
# (256), A16 = bf16 [R, H, 64] (128), A32 = fp32 [R, H, 64] (256), V32 = fp32 [R, H] (4), TAPE = bf16 [NC_pad, H,
# 128, 128] (32768 per chunk-head = 512 per internal token-head).  The segmented route's state blocks are sized by
# the segment count, not the rows: SEG = fp32 [NSEG, H, 128, 128] (65536 B per (segment, head)); seg_b (the segment
# offsets) and seg_m (the transfers) are adjacent blocks, together the kernels' seg_state [2, NSEG, H, 128, 128];
# their sub-offsets are in state blocks and they are absent (0 B) on the serial route.
#
# Launch steps: forward 1 intra (the GLM prep's outputs and Aqk / Akk), 2 wy, 3 the state recurrence (segmented:
# pass 1, carry, pass 2), 4 the output (formed by the recurrence's last launch); backward 1 prep, (1b intra on the
# replay path), 2 wy, 3 the
# recurrence recompute (idem), 4 dav, 5 dhu (segmented: pass 1, carry, pass 2), 6 dqkg, 7 glm intra, 8 gate
# epilogue.  An occupant's lifetime is [first launch that writes it, last launch that reads it]; occupants of one
# slot have disjoint lifetimes, so no launch has an input and an output in the same slot and no launch reads a
# slot another live occupant holds.  Pad-chunk rows every consumer reads are written in the same call by their
# producer (the forward intra / the backward prep: the prep outputs and the v_new / dv2 pad rows; intra: Aqk / Akk
# pad rows; wy / dav: whole
# pairs), so nothing depends on the arena's content from a previous call, and only the per-call outputs leave the
# arena.  The fused gradient (6) and the GLM intra kernel (7) read the user's q / k / v token rows directly (TMA
# boxes, zero-filled past the pack), so the backward prep writes no chunk-layout copies of them and the backward
# layouts hold none; the backward's WY programs form the gated values from the user's v / beta token rows, so the
# prep writes no copy of them either.  The fused recomputes that form the WY representation themselves (3) leave
# the transposed w tile in the w buffer (the same bytes) and write no u.

KIND_SHAPE: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "F32": ((D,), torch.float32),
    "B16": ((D,), torch.bfloat16),
    "A16": ((BT,), torch.bfloat16),
    "A32": ((BT,), torch.float32),
    "V32": ((), torch.float32),
    "TAPE": ((D, D), torch.bfloat16),
    "SEG": ((D, D), torch.float32),
}
SEG_KINDS = ("SEG",)

Occupant = tuple[str, str, int, tuple[int, int]]
# slot -> occupants: (buffer, kind, sub-slot offset in bytes per R*H (SEG kinds: in state blocks), [first write,
# last read]).  The segmented route's blocks ride in a slot that is free during fwd_h / dhu: forward seg_b / seg_m
# [3, 3] + hseg (the carried states) [3, 3] after akk_int [1, 2]; backward seg_b [3, 3], seg_m [3, 5] (the
# transfers are read again by dhu's reverse carry at 5), hseg [3, 3], then dseg (the reverse offsets) on seg_b's
# block and dhseg (the carried adjoint states) on hseg's block at [5, 5], between kb [1, 2] and dAkk [6, 7]: three
# state blocks in either mode.  From a version-4 saved set carrying the states (light, S >= 384 * NSEG) the
# backward's seg_b / seg_m / hseg blocks stay unused: the states come from the set; the layouts are unchanged.
FWD_SLOTS: tuple[tuple[Occupant, ...], ...] = (
    (("gk", "F32", 0, (1, 3)),),
    # no vb: the forward's WY forms the gated values
    (("kb", "B16", 0, (1, 2)), ("h", "TAPE", 0, (3, 4))),
    (("kg", "B16", 0, (1, 3)),),
    (("qg", "B16", 0, (1, 4)),),
    (
        ("v_new", "B16", 0, (1, 4)),
    ),  # intra zero-fills the pad-chunk rows at 1, fwd_h writes the real rows at 3
    (("aqk_int", "A16", 0, (1, 4)),),
    (
        ("akk_int", "A16", 0, (1, 2)),
        ("seg_b", "SEG", 0, (3, 3)),
        ("seg_m", "SEG", 1, (3, 3)),
        ("hseg", "SEG", 2, (3, 3)),
    ),
    (("u", "B16", 0, (2, 3)),),
    (("w", "B16", 0, (2, 3)),),
)
# The replay layout: akk_int holds the recomputed internal Akk rows (the saved-set paths read the saved token rows
# and drop the slot: SLOTS["bwd"]); do_int is the replay prep's copy of dout (the saved-set paths pass the user's
# dout and leave the buffer unwritten); aqk_int (the replay's internal Aqk rows, read by dav at 4 and by the fused
# adjoint recurrence at 5) rides in dg_f's slot, first written at 6.
BWD_SLOTS: tuple[tuple[Occupant, ...], ...] = (
    (("gk", "F32", 0, (1, 7)),),
    (("akk_int", "A16", 0, (1, 6)),),  # written at 1b (replay path)
    (("do_int", "B16", 0, (1, 6)), ("db_int", "V32", 0, (7, 8))),
    (("v_new", "B16", 0, (1, 6)),),  # pad rows at 1, real rows at 3
    (("dv2", "B16", 0, (1, 6)),),  # pad rows at 1, real rows at 5
    (("h", "TAPE", 0, (3, 6)), ("dg_int", "F32", 0, (7, 8))),
    (("dAqk", "A32", 0, (4, 7)),),
    (("u", "B16", 0, (2, 3)), ("dh", "TAPE", 0, (5, 6))),
    (("qg", "B16", 0, (1, 5)), ("kg", "B16", 256, (1, 5)), ("dq_f", "F32", 0, (6, 7))),
    (("w", "B16", 0, (2, 5)), ("dv1", "B16", 256, (4, 5)), ("dk_f", "F32", 0, (6, 7))),
    (("dg_f", "F32", 0, (6, 7)), ("aqk_int", "A16", 0, (1, 5))),
    (
        ("kb", "B16", 0, (1, 2)),
        ("dAkk", "A32", 0, (6, 7)),
        ("seg_b", "SEG", 0, (3, 3)),
        ("seg_m", "SEG", 1, (3, 5)),
        ("hseg", "SEG", 2, (3, 3)),
        ("dseg", "SEG", 0, (5, 5)),
        ("dhseg", "SEG", 2, (5, 5)),
    ),
    (("beta_int", "V32", 0, (1, 7)),),
    (("db_f", "V32", 0, (6, 7)),),
)
# save_mode="tape": h and v_new are per-call saved tensors written by the forward (the intra program zero-fills
# v_new's pad rows there too); the backward reads them from the saved set, skips fwd_h and recomputes only w (wy,
# which also writes
# u).  On the segmented route the forward's seg_state is per-call too (its transfer half is the saved seg_m, read
# by dhu's carry); the arena's seg_b / seg_m / hseg serve only the replay path (which recomputes fwd_h).
# save_mode="light" (version 4): the forward with save_for_backward writes seg_state and hseg into per-call saved
# tensors as well when S >= 384 * NSEG (seg_states_saved), so the backward from that set runs fwd_h pass 2 alone;
# the arena's blocks then serve the replay path, the forward without save_for_backward and every shorter pack.
FWD_TAPE_SLOTS: tuple[tuple[Occupant, ...], ...] = (
    (("gk", "F32", 0, (1, 3)),),
    (("kb", "B16", 0, (1, 2)),),
    (("kg", "B16", 0, (1, 3)),),
    (("qg", "B16", 0, (1, 4)),),
    (("aqk_int", "A16", 0, (1, 4)),),
    (
        ("akk_int", "A16", 0, (1, 2)),
        ("seg_b", "SEG", 0, (3, 3)),
        ("seg_m", "SEG", 1, (3, 3)),
        ("hseg", "SEG", 2, (3, 3)),
    ),
    (("u", "B16", 0, (2, 3)),),
    (("w", "B16", 0, (2, 3)),),
)
BWD_TAPE_SLOTS: tuple[tuple[Occupant, ...], ...] = (
    (("gk", "F32", 0, (1, 7)),),
    (("akk_int", "A16", 0, (1, 6)),),
    (("do_int", "B16", 0, (1, 6)), ("db_int", "V32", 0, (7, 8))),
    (("dv2", "B16", 0, (1, 6)),),
    (("dAqk", "A32", 0, (4, 7)),),
    (
        ("u", "B16", 0, (2, 2)),
        ("dh", "TAPE", 0, (5, 6)),
        ("dg_int", "F32", 0, (7, 8)),
    ),
    (("qg", "B16", 0, (1, 5)), ("kg", "B16", 256, (1, 5)), ("dq_f", "F32", 0, (6, 7))),
    (("w", "B16", 0, (2, 5)), ("dv1", "B16", 256, (4, 5)), ("dk_f", "F32", 0, (6, 7))),
    (
        ("v_new_scratch", "B16", 0, (1, 1)),  # prep's pad-row zero-fill target (unread)
        ("dg_f", "F32", 0, (6, 7)),
        ("aqk_int", "A16", 256, (1, 5)),
    ),
    (
        ("kb", "B16", 0, (1, 2)),
        ("dAkk", "A32", 0, (6, 7)),
        ("seg_b", "SEG", 0, (3, 3)),
        ("seg_m", "SEG", 1, (3, 5)),
        ("hseg", "SEG", 2, (3, 3)),
        ("dseg", "SEG", 0, (5, 5)),
        ("dhseg", "SEG", 2, (5, 5)),
    ),
    (("beta_int", "V32", 0, (1, 7)),),
    (("db_f", "V32", 0, (6, 7)),),
)


def _without(
    slots: tuple[tuple[Occupant, ...], ...], buffer: str
) -> tuple[tuple[Occupant, ...], ...]:
    """``slots`` without the single-occupant slot of ``buffer`` (the saved-set paths' layouts drop ``akk_int``)."""
    return tuple(
        slot for slot in slots if not (len(slot) == 1 and slot[0][0] == buffer)
    )


SLOTS: dict[str, tuple[tuple[Occupant, ...], ...]] = {
    "fwd": FWD_SLOTS,
    "fwd_tape": FWD_TAPE_SLOTS,
    "bwd": _without(BWD_SLOTS, "akk_int"),
    "bwd_tape": _without(BWD_TAPE_SLOTS, "akk_int"),
    "replay": BWD_SLOTS,
    "replay_tape": BWD_TAPE_SLOTS,
}
# The lean layouts' per-pass buffer kinds: the token kinds sized for one tail range of ``R_b = 64 * ceil(NC_pad /
# tail_passes)`` internal rows instead of ``R``; their sub-offsets count in ``R_b * H`` bytes.
PASS_KINDS: dict[str, str] = {"F32p": "F32", "A32p": "A32", "V32p": "V32"}
# the per-pass set of one tail range, in packing order: the fused gradient's outputs (step 6), GLM intra's (7)
_PASS_OCCUPANTS: tuple[Occupant, ...] = (
    ("dq_f", "F32p", 0, (6, 7)),
    ("dk_f", "F32p", 0, (6, 7)),
    ("dg_f", "F32p", 0, (6, 7)),
    ("dAkk", "A32p", 0, (6, 7)),
    ("dAqk", "A32p", 0, (6, 7)),
    ("db_f", "V32p", 0, (6, 7)),
    ("dg_int", "F32p", 0, (7, 8)),
    ("db_int", "V32p", 0, (7, 8)),
)
# the gate epilogue's inputs (GLM intra's per-row gate gradients); absent from the layouts of the architectures
# whose GLM intra program writes the token-row dg / dbeta itself (``glm_intra_v14`` / ``glm_intra_v18``, no gate_epilogue launch)
_GATE_BUFFERS = ("dg_int", "db_int")
# the segmented route's backward state blocks (as in BWD_SLOTS: offsets / transfers / carried states, then the
# reverse offsets and carried adjoint states on the first and third block)
_SEG_BLOCKS_BWD: tuple[Occupant, ...] = (
    ("seg_b", "SEG", 0, (3, 3)),
    ("seg_m", "SEG", 1, (3, 5)),
    ("hseg", "SEG", 2, (3, 3)),
    ("dseg", "SEG", 0, (5, 5)),
    ("dhseg", "SEG", 2, (5, 5)),
)


def _lean_slots_single(tape: bool) -> tuple[tuple[Occupant, ...], ...]:
    """The lean layout of one tail pass: the saved-set layout (``SLOTS["bwd"]`` / ``["bwd_tape"]``) without the
    replay-only ``do_int`` copy (the saved-set paths pass the user's ``dout``; ``db_int`` takes a slot of its own)
    and with ``kb`` under the ``dh`` tape (sub-offset 256, lifetime ``[1, 3]``: the saved-state pass 2 of the rows
    with at most 32 chains, ``fwdh_p3_wy``, reads it at 3) instead of the state blocks' slot, where its declared
    ``[1, 2]`` lifetime relies on the ``light_states`` path leaving those blocks unused.  4108 B per internal
    token-head (tape: 3340) plus the state blocks."""
    out: list[tuple[Occupant, ...]] = []
    for slot in SLOTS["bwd_tape" if tape else "bwd"]:
        names = [o[0] for o in slot]
        if "do_int" in names:
            out.append((("db_int", "V32", 0, (7, 8)),))
        elif "kb" in names:
            out.append(tuple(o for o in slot if o[0] != "kb"))
        elif "dh" in names:
            out.append(tuple(slot) + (("kb", "B16", 256, (1, 3)),))
        else:
            out.append(slot)
    return tuple(out)


def _lean_slots(
    tape: bool, passes: int, R: int, H: int, NC: int, NSEG: int
) -> tuple[tuple[Occupant, ...], ...]:
    """The lean layout of ``passes >= 2`` tail passes.  Whole-sequence buffers: gk, v_new and h (light only), dv2,
    dh with u and kb under it (tape: and the prep's unread pad-row target), qg / kg, w, beta_int: 2820 B per
    internal token-head (tape 2052).  The per-pass set (:data:`_PASS_OCCUPANTS`, one tail range each) is packed
    first-fit, in order, into the bytes of the two whole-sequence slots that are dead before the tail (qg / kg: 512
    B per token-head, w: 256); what does not fit goes to an overflow slot shared with the segmented route's state
    blocks (lifetimes [3, 5] against the per-pass [6, 8]).  From four passes on everything fits: 2820 * R * H +
    3 * block, plus the ceil slack of the last range."""
    R_b = BT * -(-NC // passes)
    unit_w, unit_p = R * H, R_b * H
    whole: list[list[Occupant]] = [
        [("gk", "F32", 0, (1, 7))],
        [("dv2", "B16", 0, (1, 6))],
        [
            ("dh", "TAPE", 0, (5, 6)),
            ("u", "B16", 0, (2, 2) if tape else (2, 3)),
            ("kb", "B16", 256, (1, 3)),
        ],
        [("qg", "B16", 0, (1, 5)), ("kg", "B16", 256, (1, 5))],
        [("w", "B16", 0, (2, 5))],
        [("beta_int", "V32", 0, (1, 7))],
    ]
    if tape:
        whole[2].append(("v_new_scratch", "B16", 0, (1, 1)))
    else:
        whole.insert(1, [("v_new", "B16", 0, (1, 6))])
        whole.insert(3, [("h", "TAPE", 0, (3, 6))])
    capacity = {"qg": 512 * unit_w, "w": 256 * unit_w}
    used = {"qg": 0, "w": 0}
    extra: dict[str, list[Occupant]] = {"qg": [], "w": []}
    overflow: list[Occupant] = []
    over_used = 0
    for name, kind, _sub, life in _PASS_OCCUPANTS:
        nbytes = _kind_bytes(kind, R, H, NC, NSEG, R_b)[0]
        for host in ("qg", "w"):
            if used[host] + nbytes <= capacity[host]:
                extra[host].append((name, kind, used[host] // unit_p, life))
                used[host] += nbytes
                break
        else:
            overflow.append((name, kind, over_used // unit_p, life))
            over_used += nbytes
    slots = [tuple(group + extra.get(group[0][0], [])) for group in whole]
    if overflow or NSEG:
        slots.append(tuple(overflow) + (_SEG_BLOCKS_BWD if NSEG else ()))
    return tuple(slots)


def _slots_of(
    mode: str, tail_passes: int, R: int, H: int, NC: int, NSEG: int
) -> tuple[tuple[Occupant, ...], ...]:
    if mode in LEAN_MODES:
        tape = mode == "bwd_tape_lean"
        if tail_passes == 1:
            return _lean_slots_single(tape)
        return _lean_slots(tape, tail_passes, R, H, NC, NSEG)
    if mode not in SLOTS:
        raise ValueError(
            f"layout mode must be one of {MODES + LEAN_MODES}, got {mode!r}"
        )
    if tail_passes != 1:
        raise ValueError(
            f"layout {mode!r}: tail_passes={tail_passes} applies to the lean layouts {LEAN_MODES} only"
        )
    return SLOTS[mode]


@dataclasses.dataclass(frozen=True)
class Region:
    """One arena buffer: its slot, byte offset (a multiple of :data:`ALIGN`), size, view shape and dtype, and the
    ``[first write, last read]`` launch steps of its lifetime."""

    name: str
    slot: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    lifetime: tuple[int, int]


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _kind_bytes(
    kind: str, R: int, H: int, NC: int, NSEG: int, R_b: int | None = None
) -> tuple[int, tuple[int, ...]]:
    tail, dtype = KIND_SHAPE[PASS_KINDS.get(kind, kind)]
    if kind in PASS_KINDS:
        shape: tuple[int, ...] = (R if R_b is None else R_b, H, *tail)
    elif kind == "TAPE":
        shape = (NC, H, *tail)
    elif kind == "SEG":
        shape = (NSEG, H, *tail)
    else:
        shape = (R, H, *tail)
    n = 1
    for extent in shape:
        n *= extent
    return n * _element_size(dtype), shape


@functools.cache
def _layout(
    R: int,
    H: int,
    NC: int,
    NSEG: int,
    mode: str,
    tail_passes: int = 1,
    gate_fused: bool = False,
) -> tuple[Region, ...]:
    slots = _slots_of(mode, tail_passes, R, H, NC, NSEG)
    if gate_fused:  # no gate epilogue: its input buffers leave the layout (an emptied slot leaves too)
        slots = tuple(
            kept
            for kept in (
                tuple(o for o in slot if o[0] not in _GATE_BUFFERS) for slot in slots
            )
            if kept
        )
    block = (
        NSEG * H * D * D * 4
    )  # one fp32 [NSEG, H, 128, 128] state block of the segmented route
    R_b = BT * -(
        -NC // tail_passes
    )  # the internal rows of one tail range (the per-pass buffers' extent)
    regions: list[Region] = []
    offset = 0
    for slot, occupants in enumerate(slots):
        base = end = offset
        spans: list[tuple[str, int, int, tuple[int, int]]] = []
        for name, kind, sub, life in occupants:
            if kind in SEG_KINDS and NSEG == 0:
                continue  # the serial route has no segment state
            nbytes, shape = _kind_bytes(kind, R, H, NC, NSEG, R_b)
            unit = (
                block
                if kind in SEG_KINDS
                else (R_b * H if kind in PASS_KINDS else R * H)
            )
            start = base + sub * unit
            if start % ALIGN:
                raise ValueError(
                    f"layout {mode}: {name} at sub-offset {sub} is not {ALIGN} B aligned"
                )
            regions.append(
                Region(
                    name,
                    slot,
                    start,
                    nbytes,
                    shape,
                    KIND_SHAPE[PASS_KINDS.get(kind, kind)][1],
                    life,
                )
            )
            end = max(end, start + nbytes)
            spans.append((name, start, start + nbytes, life))
        for a in spans:  # occupants sharing bytes have disjoint lifetimes (the invariant of the tables above)
            for b in spans:
                if (
                    a is not b
                    and a[1] < b[2]
                    and b[1] < a[2]
                    and a[3][0] <= b[3][1]
                    and b[3][0] <= a[3][1]
                ):
                    raise ValueError(
                        f"layout {mode}: {a[0]} and {b[0]} overlap in bytes and in lifetime"
                    )
        offset = _round_up(end, ALIGN)
    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"layout {mode}: duplicate buffer names")
    return tuple(regions)


def _table_layout(
    NC_pad: int, N: int, NSEG: int = 0
) -> tuple[tuple[str, int, int], ...]:
    words_per_align = ALIGN // 4
    tables = [
        ("chunk_bos", NC_pad),
        ("chunk_len", NC_pad),
        ("seq_chunk_start", N + 1),
        ("seq_order", N),
    ]
    if NSEG:
        # the block table of the seg_table program: the chains' first segments [N + 1] (the carry's geometry),
        # then per chain its segment starts and its end chunk (the walk programs' geometry)
        tables.append(("seg_table", NSEG + 2 * N + 1))
    out: list[tuple[str, int, int]] = []
    off = 0
    for name, words in tables:
        out.append((name, off, words))
        off = _round_up(off + words, words_per_align)
    return tuple(out)


# ----------------------------------------------------------------------------------------------------------------
# Byte queries (host only)


def _arch_sms(sm_count: int | None, arch: str | None) -> tuple[str | None, int]:
    """``(arch, sm_count)`` of a byte query.  ``arch``: the explicit one, else the current CUDA device's where a
    device is available and is an exported target, else ``None`` (the architecture-independent few-chain segment
    count, :func:`cake_jit.segment_plan`); ``sm_count``: the explicit one, else the current device's.  So a query
    with or without the device's own SM count agrees with the plan of a workspace on that device."""
    if arch is None and torch.cuda.is_available():
        capability = tuple(
            torch.cuda.get_device_capability(torch.cuda.current_device())
        )
        arch = cake_jit.arch_for_capability(capability)
    if sm_count is not None:
        return arch, int(sm_count)
    return arch, cake_jit.sm_count(torch.cuda.current_device())


def _num_segments(num_docs: int, H: int, sm_count: int, arch: str | None) -> int:
    """``NSEG`` of the problem (zero on the serial route)."""
    return cake_jit.segment_plan(num_docs, H, sm_count, arch)[2]


def _mode_bytes(
    R: int,
    H: int,
    NC_pad: int,
    NSEG: int,
    mode: str,
    tail_passes: int = 1,
    gate_fused: bool = False,
) -> int:
    """The bytes of one arena layout (:data:`MODES` or :data:`LEAN_MODES`)."""
    return max(
        (
            r.offset + r.nbytes
            for r in _layout(R, H, NC_pad, NSEG, mode, tail_passes, gate_fused)
        ),
        default=0,
    )


def _arena_bytes(
    R: int,
    H: int,
    NC_pad: int,
    NSEG: int,
    *,
    backward: bool,
    save_mode: str,
    arena: KdaArena | None = None,
    gate_fused: bool = False,
) -> int:
    """The arena a :class:`KdaWorkspace` of the problem allocates: the larger of the saved-set and the replay
    layouts for a backward workspace (both paths run in it), the forward layout otherwise; the ``*_tape`` layouts
    under ``save_mode="tape"``; under a lean ``arena`` (:class:`KdaArena`) the lean layout of its ``tail_passes``
    alone (a lean workspace serves no replay); ``gate_fused`` drops the gate epilogue's buffers."""
    tape = save_mode == "tape"
    if not backward:
        return _mode_bytes(
            R, H, NC_pad, NSEG, "fwd_tape" if tape else "fwd", 1, gate_fused
        )
    if arena is not None and arena.lean:
        return _mode_bytes(
            R,
            H,
            NC_pad,
            NSEG,
            "bwd_tape_lean" if tape else "bwd_lean",
            arena.tail_passes,
            gate_fused,
        )
    modes = ("bwd_tape", "replay_tape") if tape else ("bwd", "replay")
    return max(_mode_bytes(R, H, NC_pad, NSEG, mode, 1, gate_fused) for mode in modes)


def workspace_bytes(
    S: int,
    num_docs: int,
    H: int,
    *,
    backward: bool = True,
    save_mode: str = "light",
    sm_count: int | None = None,
    arch: str | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> int:
    """Arena bytes a :class:`KdaWorkspace` of the problem allocates (``backward=False``: the forward's layout only;
    a backward workspace takes the larger of the saved-set paths' layout and the replay's; ``save_mode`` selects
    the tape layouts; ``arena`` the lower-memory option, :class:`KdaArena`, whose batched form sizes the arena for
    the groups of ``doc_lengths`` when given, else for the bound of every group).  A function of the host scalars
    and, through the segmented route's segment count, of
    ``sm_count`` and ``arch`` (:func:`_arch_sms`: the current device's SM count when ``sm_count`` is ``None``, the
    current device's architecture when ``arch`` is ``None``).  The outputs and the saved set are allocated per
    call: :func:`saved_bytes`, :func:`output_bytes`."""
    if save_mode not in SAVE_MODES:
        raise ValueError(
            f"workspace_bytes: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
        )
    arch, sms = _arch_sms(sm_count, arch)
    S, num_docs, H = int(S), int(num_docs), int(H)
    if arena is not None and (arena.lean or arena.batched):
        return plan(S, num_docs, H, sms, arch).arena_bytes(
            backward=backward, save_mode=save_mode, arena=arena, doc_lengths=doc_lengths
        )
    _nc_max, nc_pad, rows = _check_problem(S, num_docs, H, sms)
    nseg = _num_segments(num_docs, H, sms, arch)
    return _arena_bytes(
        rows,
        H,
        nc_pad,
        nseg,
        backward=backward,
        save_mode=save_mode,
        arena=arena,
        gate_fused=_gate_fused(arch),
    )


def tables_bytes(
    S: int,
    num_docs: int,
    H: int | None = None,
    *,
    sm_count: int | None = None,
    arch: str | None = None,
) -> int:
    """Bytes of the int32 table buffer the forward hands over in the saved set (rebuilt by the backward when
    absent); given ``H``, the segmented route's block table is included when that route is dispatched (its length
    reads ``sm_count`` and ``arch``, :func:`_arch_sms`)."""
    _nc_max, nc_pad, _rows = _geometry(int(S), int(num_docs))
    num_segments = 0
    if H is not None:
        arch, sms = _arch_sms(sm_count, arch)
        num_segments = _num_segments(int(num_docs), int(H), sms, arch)
    _name, off, words = _table_layout(nc_pad, int(num_docs), num_segments)[-1]
    return _round_up(off + words, ALIGN // 4) * 4


def saved_bytes(
    S: int,
    H: int,
    num_docs: int | None = None,
    *,
    save_mode: str = "light",
    sm_count: int | None = None,
    arch: str | None = None,
) -> int:
    """Bytes of the saved set: ``Aqk`` + ``Akk`` BF16 ``[S, H, 64]`` (``256 * S * H``), plus, when ``num_docs``
    is given, the tables and, for ``save_mode="light"`` (version 4), the fold's ``qg`` BF16 ``[R, H, 128]`` (the
    internal rows ``R = 64 * NC_pad``), on the segmented route when ``S >= 384 * NSEG`` (:func:`seg_states_saved`)
    the forward's segment-boundary states ``hseg`` FP32 ``[NSEG, H, 128, 128]`` and ``seg_state`` FP32 ``[2, NSEG,
    H, 128, 128]`` (three state blocks of ``NSEG * H * 64 KiB``; the segment count needs ``num_docs`` and
    ``sm_count`` / ``arch``, :func:`_arch_sms`) and, on the rows whose light backward reads the WY representation
    from the set (the paired serial 16-head rows of ``sm_100a`` and the segmented 16-head rows of ``sm_103a`` /
    ``sm_107a`` above 32 (document, head) chains; ``arch=None`` takes the first exported architecture's records),
    ``w`` / ``u`` BF16 ``[R, H, 128]``.  ``save_mode="tape"`` adds the internal-layout
    ``h`` BF16 ``[NC_pad, H, 128, 128]``, ``v_new`` BF16 ``[R, H, 128]`` and, on the segmented route, the forward's
    segment states FP32 ``[2, NSEG, H, 128, 128]`` whose transfer half is the saved ``seg_m`` (``num_docs``
    required) instead."""
    if save_mode not in SAVE_MODES:
        raise ValueError(
            f"saved_bytes: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
        )
    S, H = int(S), int(H)
    total = 2 * S * H * BT * 2
    if num_docs is None:
        if save_mode == "tape":
            raise ValueError("saved_bytes: save_mode='tape' needs num_docs")
        return total
    num_docs = int(num_docs)
    arch, sms = _arch_sms(sm_count, arch)
    num_segments = _num_segments(num_docs, H, sms, arch)
    total += tables_bytes(S, num_docs, H, sm_count=sms, arch=arch)
    _nc_max, nc_pad, rows = _geometry(S, num_docs)
    if save_mode == "tape":
        total += nc_pad * H * D * D * 2 + rows * H * D * 2
        total += 2 * num_segments * H * D * D * 4
        return total
    # light (version 4): the fold's qg, the segment-boundary states when the size rule holds, and w / u on the
    # rows whose light backward reads them from the set
    saves_qg, saves_wu = _light_prep_saves(nc_pad, num_docs, H, sms, arch)
    if saves_qg:
        total += rows * H * D * 2
    if saves_wu:
        total += 2 * rows * H * D * 2
    if seg_states_saved(S, num_segments):
        total += 3 * num_segments * H * D * D * 4
    return total


def output_bytes(S: int, H: int, *, backward: bool) -> int:
    """Bytes of one call's outputs: ``out`` BF16 (forward) or ``dq, dk, dv, dbeta`` BF16 + ``dg`` FP32 (backward)."""
    S, H = int(S), int(H)
    if not backward:
        return S * H * D * 2
    return 3 * S * H * D * 2 + S * H * 2 + S * H * D * 4


def lifetime_table(
    mode: str = "bwd",
    tail_passes: int = 1,
    *,
    plan: Plan | None = None,
    gate_fused: bool | None = None,
) -> list[dict[str, Any]]:
    """The layout as rows ``{slot, buffer, kind, unit, ..., first_write, last_read}``: token kinds in bytes per
    internal token-head (a tape entry of 32768 B per chunk-head spreads over the chunk's 64 rows), the segmented
    route's state blocks (``unit="seg_block"``) in bytes per (segment, head), present only when the plan
    dispatches that route.  The lean layouts (:data:`LEAN_MODES`) take ``tail_passes``; from two passes on their
    per-pass buffers (``unit="R_b*H"``: bytes per internal token-head of one tail range) are packed by the
    problem's geometry, so a ``plan`` is required there.  ``gate_fused`` (``None``: the plan's, else ``False``)
    drops the gate epilogue's ``dg_int`` / ``db_int`` (the architectures whose GLM intra program fuses it)."""
    if mode in LEAN_MODES and tail_passes > 1:
        if plan is None:
            raise ValueError(
                "lifetime_table: the multi-pass lean layouts pack the per-pass buffers by the problem's geometry; "
                "pass plan="
            )
        slots = _slots_of(mode, tail_passes, plan.R, plan.H, plan.NC_pad, plan.NSEG)
    else:
        slots = _slots_of(mode, tail_passes, 0, 1, 0, 0)
    if gate_fused is None:
        gate_fused = plan.gate_fused if plan is not None else False
    if gate_fused:
        slots = tuple(
            kept
            for kept in (
                tuple(o for o in slot if o[0] not in _GATE_BUFFERS) for slot in slots
            )
            if kept
        )
    rows: list[dict[str, Any]] = []
    for slot, occupants in enumerate(slots):
        for name, kind, sub, life in occupants:
            tail, dtype = KIND_SHAPE[PASS_KINDS.get(kind, kind)]
            n = 1
            for extent in tail:
                n *= extent
            elem = n * _element_size(dtype)
            if kind in SEG_KINDS:
                size: dict[str, Any] = {
                    "unit": "seg_block",
                    "sub_offset_blocks": sub,
                    "bytes_per_th": 0,
                    "bytes_per_seg_head": elem,
                }
            elif kind in PASS_KINDS:
                size = {"unit": "R_b*H", "sub_offset_per_th": sub, "bytes_per_th": elem}
            else:
                size = {
                    "unit": "R*H",
                    "sub_offset_per_th": sub,
                    "bytes_per_th": elem // (BT if kind == "TAPE" else 1),
                }
            rows.append(
                {
                    "slot": slot,
                    "buffer": name,
                    "kind": kind,
                    **size,
                    "first_write": life[0],
                    "last_read": life[1],
                }
            )
    return rows


def memory_report(
    S: int,
    num_docs: int,
    H: int,
    save_mode: str = "light",
    *,
    sm_count: int | None = None,
    arch: str | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bytes one problem needs beyond its inputs: the arena (forward-only and full, with the saved-set paths' and
    the replay's backward layouts apart), the saved set, the outputs and the totals of the three boundaries (arena
    + the per-call tensors live at the peak).  ``sm_count`` and ``arch`` (:func:`_arch_sms`: the current device's
    when ``None``) select the segmented route's segment count.  ``arena`` (:class:`KdaArena`) adds the configured
    arenas (``arena_bwd_configured`` / ``arena_fwd_configured``; the default's stays ``arena_bwd``), the floor of
    the lean layouts (``arena_floor``: 2820 B per internal token-head plus three state blocks) and recomputes the
    totals with the configured backward arena (the replay total keeps the default's: a lean workspace serves no
    replay).  A batched option reports the bound of every group (``arena_bwd_bound_batched``) and, with
    ``doc_lengths``, its exact groups (``batches`` as ``(d0, d1, S, N)``), ``largest_batch_rows``,
    ``batch_scratch`` (allocated per call: the largest table buffer of a group and three state blocks per segment of
    the group with the most segments; a group gathers its blocks of the saved states only where they are not one
    run of them, at most three per segment) and
    ``arena_floor_batched`` (one document per group under the lean four-pass layouts: ``2820 * R_min * H + 3 * P *
    H * 64 KiB``, ``R_min`` the internal rows of the longest document; 2052 for the tape forms)."""
    if save_mode not in SAVE_MODES:
        raise ValueError(
            f"memory_report: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
        )
    tape = save_mode == "tape"
    arch, sms = _arch_sms(sm_count, arch)
    S, num_docs, H = int(S), int(num_docs), int(H)
    arena = KdaArena() if arena is None else arena
    p = (
        plan(S, num_docs, H, sms, arch)
        if (arena.lean or arena.batched or doc_lengths is not None)
        else None
    )
    if p is not None:
        p.check_arena(arena)
    _nc_max, nc_pad, rows = _check_problem(S, num_docs, H, sms)
    recur, segments, num_segments = cake_jit.segment_plan(num_docs, H, sms, arch)
    gf = _gate_fused(arch)
    arena_fwd = _arena_bytes(
        rows,
        H,
        nc_pad,
        num_segments,
        backward=False,
        save_mode=save_mode,
        gate_fused=gf,
    )
    arena_saved = _mode_bytes(
        rows, H, nc_pad, num_segments, "bwd_tape" if tape else "bwd", 1, gf
    )
    arena_replay = _mode_bytes(
        rows, H, nc_pad, num_segments, "replay_tape" if tape else "replay", 1, gf
    )
    arena_bwd = max(arena_saved, arena_replay)
    if arena.batched:
        arena_bwd_conf = p.arena_bytes(
            backward=True, save_mode=save_mode, arena=arena, doc_lengths=doc_lengths
        )
        arena_fwd_conf = p.arena_bytes(
            backward=False, save_mode=save_mode, arena=arena, doc_lengths=doc_lengths
        )
    else:
        arena_bwd_conf = _arena_bytes(
            rows,
            H,
            nc_pad,
            num_segments,
            backward=True,
            save_mode=save_mode,
            arena=arena,
            gate_fused=gf,
        )
        arena_fwd_conf = arena_fwd
    block = num_segments * H * D * D * 4
    per_th = 2052 if tape else 2820
    arena_floor = per_th * rows * H + 3 * block
    # the batched option: the bound of every group, and with doc_lengths the exact groups, their scratch and the floor
    bound = batches = largest = scratch = floor_batched = None
    if arena.batched:
        nc_bar, rows_bar, nseg_bar = p.batch_bound(arena)
        bound = _arena_bytes(
            rows_bar,
            H,
            nc_bar,
            nseg_bar,
            backward=True,
            save_mode=save_mode,
            arena=arena,
            gate_fused=gf,
        )
        largest = rows_bar
        if doc_lengths is not None:
            groups = p.batches(arena, doc_lengths)
            batches = [(b.d0, b.d1, b.S, b.N) for b in groups]
            largest = max(b.NC_pad for b in groups) * BT
            words = 0
            for b in groups:
                layout = _table_layout(b.NC_pad, b.N, p.batch_segments(b))
                words = max(words, _round_up(layout[-1][1] + layout[-1][2], ALIGN // 4))
            segments_b = max(p.batch_segments(b) for b in groups)
            scratch = words * 4 + 3 * segments_b * H * D * D * 4
    if doc_lengths is not None:
        longest = max(p.check_doc_lengths(doc_lengths))
        nc_min = -(-longest // BT) + 1
        rows_min = BT * (nc_min + nc_min % 2)
        floor_batched = per_th * rows_min * H + 3 * (
            segments * H * D * D * 4 if recur == "seg" else 0
        )
    saved = saved_bytes(S, H, num_docs, save_mode=save_mode, sm_count=sms, arch=arch)
    out_f, out_b = output_bytes(S, H, backward=False), output_bytes(S, H, backward=True)
    # The per-call segment states in the saved set: tape two blocks (seg_state, its transfer half is the saved
    # seg_m), light three (hseg and the seg_state pair, version 4) when the size rule holds; none on the serial route.
    if recur != "seg":
        saved_blocks = 0
    elif tape:
        saved_blocks = 2
    else:
        saved_blocks = 3 if seg_states_saved(S, num_segments) else 0
    return {
        "save_mode": save_mode,
        "sm_count": sms,
        "arch": arch,
        "gate_fused": gf,
        "recur": recur,
        "segments": segments,
        "num_segments": num_segments,
        "seg_block_bytes": num_segments
        * H
        * D
        * D
        * 4,  # one fp32 state block (three per mode in the arena)
        "saved_seg_state_bytes": saved_blocks * num_segments * H * D * D * 4,
        # the light rule S >= 384 * NSEG (the tape set always carries its seg_state)
        "seg_states_saved": seg_states_saved(S, num_segments),
        "eye_bytes": D * D * 4
        if recur == "seg"
        else 0,  # the identity, outside the arena
        "arena_fwd": arena_fwd,
        "arena_bwd": arena_bwd,
        "arena_bwd_layout_saved_path": arena_saved,
        "arena_bwd_layout_replay_path": arena_replay,
        # the lower-memory arena option (KdaArena): off -> the configured arenas equal the defaults
        "lean": arena.lean,
        "tail_passes": arena.tail_passes,
        "batch_tokens": arena.batch_tokens,
        "max_docs_per_batch": arena.max_docs_per_batch,
        "arena_bwd_default": arena_bwd,
        "arena_bwd_configured": arena_bwd_conf,
        "arena_fwd_configured": arena_fwd_conf,
        "arena_floor": arena_floor,
        # the batched option: the bound of every group of batch_tokens tokens, and the exact groups of doc_lengths
        "arena_bwd_bound_batched": bound,
        "batches": batches,
        "largest_batch_rows": largest,
        "batch_scratch": scratch,
        "arena_floor_batched": floor_batched,
        "saved": saved,
        "out_fwd": out_f,
        "out_bwd": out_b,
        "fwd_total_fwd_arena": arena_fwd_conf
        + out_f
        + saved,  # a forward-only workspace
        "fwd_total_bwd_arena": arena_bwd_conf
        + out_f
        + saved,  # the same workspace serving every boundary
        "bwd_total": arena_bwd_conf + out_b,  # the saved set exists before the call
        "replay_total": arena_bwd
        + out_f
        + saved
        + out_b,  # the default arena: lean serves no replay
    }


# ----------------------------------------------------------------------------------------------------------------
# Saved set and workspace


def _alloc(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Every per-call tensor (outputs, saved set, tables) comes from here: ``torch.empty``, since every element is
    written by a kernel of the same call."""
    return torch.empty(shape, dtype=dtype, device=device)


@dataclasses.dataclass(frozen=True)
class KdaChunkSaved:
    """The forward's saved set.

    Version 4 (``save_mode="light"``, the default): ``Aqk`` / ``Akk`` BF16 ``[S, H, 64]`` on the token rows (row
    ``t`` holds the 64 chunk-local columns of its chunk; zeros above the diagonal and past the chunk end), the
    device tables ``(chunk_bos, chunk_len, seq_chunk_start, seq_order[, seg_table])``, views of one int32
    buffer (``None`` when the caller dropped them: the backward rebuilds them), the fold's ``qg`` BF16 ``[R, H,
    128]`` (internal layout; the light backward's prep writes none and its adjoint reads this one), on the
    segmented route when ``S >= 384 * NSEG`` (:func:`seg_states_saved`) the forward's segment-boundary states
    ``hseg`` FP32 ``[NSEG, H, 128, 128]`` (the carried initial state of every segment) and ``seg_state`` FP32
    ``[2, NSEG, H, 128, 128]`` (``[0]`` the segment offsets, ``[1]`` the transfers the backward's reverse carry
    reads): the backward's recurrence recompute is then pass 2 alone (both ``None`` on the serial route and for
    shorter packs, which the backward recomputes), and on the rows whose light backward reads the WY
    representation from the set (:attr:`Plan.light_saves_wu`: the paired serial 16-head rows of ``sm_100a`` and
    the segmented 16-head rows of ``sm_103a`` / ``sm_107a`` above 32 (document, head) chains) the forward's ``w``
    / ``u`` BF16 ``[R, H, 128]`` (``None`` elsewhere: the backward's ``wy`` program recomputes
    them).  Version 2 (``save_mode="tape"``) adds the forward's ``h`` tape BF16 ``[NC_pad, H, 128, 128]``,
    ``v_new`` BF16 ``[R, H, 128]`` (internal layout, pad rows zero) and, on the segmented route, ``seg_m`` FP32
    ``[NSEG, H, 128, 128]`` (``None`` on the serial route) instead, and no ``qg`` / ``w`` / ``u`` (its prep writes
    them).  Versions 1 and 3 (the light sets without ``qg``) are no longer produced or accepted.  ``S``, ``H``,
    ``num_docs`` and ``scale`` identify the problem.
    """

    version: int
    Aqk: torch.Tensor
    Akk: torch.Tensor
    tables: tuple[torch.Tensor, ...] | None
    S: int
    H: int
    num_docs: int
    scale: float
    h: torch.Tensor | None = None
    v_new: torch.Tensor | None = None
    seg_m: torch.Tensor | None = None
    # version 4 on the segmented route when S >= 384 * NSEG, else None: the carried states FP32
    # [NSEG, H, 128, 128] and the kernels' seg_state FP32 [2, NSEG, H, 128, 128] (offsets, transfers)
    hseg: torch.Tensor | None = None
    seg_state: torch.Tensor | None = None
    # version 4: the fold's qg BF16 [R, H, 128]; w / u BF16 [R, H, 128] on the rows whose light backward reads
    # the WY representation from the set, else None
    qg: torch.Tensor | None = None
    w: torch.Tensor | None = None
    u: torch.Tensor | None = None


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"the Cake KDA training kernels run on CUDA devices, not {device}"
        )
    return device.index if device.index is not None else torch.cuda.current_device()


def _seq_order(lengths: Sequence[int]) -> tuple[int, ...]:
    """The layout table's ``seq_order`` (the documents by chain rank) from host integers: by chunk count
    ``ceil(n / 64)``, most first, ties in document order.  Every program that writes the table (``table``,
    ``table_v2``, ``table_v3`` and the segmented intra program ``fwd_intra_v14``) ranks document ``d`` at
    ``#{j: nc_j > nc_d} + #{j < d: nc_j == nc_d}``, a strict order the sort reproduces: the zero-length documents
    (no chunk) rank last, in document order."""
    chunks = [-(-int(n) // BT) for n in lengths]
    return tuple(sorted(range(len(chunks)), key=lambda d: (-chunks[d], d)))


def _chain_ranks(lengths: Sequence[int]) -> tuple[int, ...]:
    """Document -> its (document, head) chain rank, the inverse of :func:`_seq_order`: the segmented route's state
    blocks are laid out by rank, ``P`` per chain (the non-proportional block table)."""
    ranks = [0] * len(lengths)
    for rank, d in enumerate(_seq_order(lengths)):
        ranks[d] = rank
    return tuple(ranks)


def _group_tail_passes(
    p: Plan,
    batch: Batch,
    nc_pad: int,
    arena: KdaArena,
    *,
    nbytes: int,
    save_mode: str,
) -> int:
    """The tail passes of one document group's backward at chunk bound ``nc_pad`` under the batched ``arena`` in an
    arena of ``nbytes``: the fewest, up to the option's ``tail_passes``, whose lean layouts at the group's geometry
    fit those bytes (the layouts the workspace's backward takes: ``bwd_tape_lean`` on a tape-mode workspace, also
    ``bwd_lean`` on a light one), else the option's (:meth:`KdaWorkspace.views` checks that one).  The arena is
    sized for the largest group at the option's passes, so a smaller group runs its tail in fewer passes; the
    split only partitions the tail's chunk ranges, so every pass count gives the same gradients bit for bit."""
    if arena.tail_passes == 1:
        return 1
    modes = ("bwd_tape_lean",) if save_mode == "tape" else ("bwd_lean", "bwd_tape_lean")
    nseg = p.batch_segments(batch)
    for passes in range(1, arena.tail_passes):
        if all(
            _mode_bytes(BT * nc_pad, p.H, nc_pad, nseg, mode, passes, p.gate_fused)
            <= nbytes
            for mode in modes
        ):
            return passes
    return arena.tail_passes


@dataclasses.dataclass(frozen=True, eq=False)
class _Packing:
    """One ``doc_lengths`` of a batched workspace (:meth:`KdaWorkspace.batches`), derived on the host from the
    lengths alone: the document groups (:meth:`Plan.batches`); on the segmented route, per group, where its state
    blocks are one run of the whole problem's the first of them (``blocks``; a group's tables rank its documents by
    the whole problem's rule, so its chains keep their relative order), else ``None`` and the group's block
    indices (``index``; ``None`` on the serial route too, which has no blocks); the real chunk count of every
    tail range of every group at each of its chunk bounds where the group runs more than one tail pass
    (:func:`_group_tail_passes`, :func:`_tail_passes`); and the device copy of these integers (``None`` until a
    call makes it; :func:`_entry_release` drops it)."""

    batches: tuple[Batch, ...]
    P: int
    blocks: tuple[int | None, ...]
    # the int32 words of the copy: the lengths, then the tail counts ((group, chunk bound) -> (first word, words))
    words: tuple[int, ...]
    tail_at: dict[tuple[int, int], tuple[int, int]]
    # the int64 words: the block indices of the groups whose blocks are not one run (group -> (first word, words))
    index: tuple[int, ...]
    index_at: dict[int, tuple[int, int]]
    words_dev: torch.Tensor | None = None
    index_dev: torch.Tensor | None = None

    def to(self, device: torch.device) -> _Packing:
        """With the device copy: from pinned host memory on the current stream without blocking (the host never
        waits for the device); the block indices only where a group needs them."""
        words = torch.tensor(self.words, dtype=torch.int32, pin_memory=True)
        index: torch.Tensor | None = None
        if self.index:
            index = torch.tensor(self.index, dtype=torch.int64, pin_memory=True)
            index = index.to(device, non_blocking=True)
        return dataclasses.replace(
            self, words_dev=words.to(device, non_blocking=True), index_dev=index
        )

    def on_host(self) -> _Packing:
        return dataclasses.replace(self, words_dev=None, index_dev=None)

    @property
    def lengths(self) -> torch.Tensor:
        """The device copy of ``doc_lengths`` (for the asynchronous check against ``cu_seqlens``)."""
        return self.words_dev[: self.batches[-1].d1]

    def tail_counts(self, group: int, nc_pad: int) -> torch.Tensor | None:
        """The real chunk counts of the group's tail ranges at chunk bound ``nc_pad`` (``None`` where the group runs
        its tail in one pass)."""
        at = self.tail_at.get((group, nc_pad))
        return None if at is None else self.words_dev[at[0] : at[0] + at[1]]

    def take(self, group: int, states: torch.Tensor) -> torch.Tensor:
        """The group's blocks of the whole problem's states ``[NSEG, ...]`` in the group's block order: a view where
        they are one run, else gathered by index."""
        first = self.blocks[group]
        if first is not None:
            return states[first : first + self.batches[group].N * self.P]
        return states.index_select(0, self.block_index(group))

    def block_index(self, group: int) -> torch.Tensor:
        at = self.index_at[group]
        return self.index_dev[at[0] : at[0] + at[1]]


def _host_packing(
    p: Plan,
    arena: KdaArena,
    lengths: tuple[int, ...],
    *,
    nbytes: int,
    save_mode: str,
) -> _Packing:
    """The host part of :class:`_Packing` for ``lengths`` (checked) on plan ``p`` under the batched ``arena``, for a
    workspace of ``nbytes`` and ``save_mode`` (the groups' tail passes: :func:`_group_tail_passes`)."""
    batches = p.batches(arena, lengths)
    ranks = _chain_ranks(lengths) if p.recur == "seg" else None
    words = list(lengths)
    tail_at: dict[tuple[int, int], tuple[int, int]] = {}
    index: list[int] = []
    index_at: dict[int, tuple[int, int]] = {}
    blocks: list[int | None] = []
    for i, b in enumerate(batches):
        # the device rule of _tail_passes, clamp(NC_real - c0, 0, ceil(NC_pad / B)), at the group's real chunks and
        # its own pass count B at each chunk bound (a group in one pass reads no counts)
        real = sum(-(-n // BT) for n in lengths[b.d0 : b.d1])
        for nc_pad in sorted({b.NC_pad, b.NC_pad_tape}):
            passes = _group_tail_passes(
                p, b, nc_pad, arena, nbytes=nbytes, save_mode=save_mode
            )
            if passes == 1:
                continue
            ranges = _tail_ranges(nc_pad, passes)
            tail_at[(i, nc_pad)] = (len(words), len(ranges))
            words += [min(max(real - c0, 0), ranges[0][1]) for c0, _n in ranges]
        if ranks is None:
            blocks.append(None)
            continue
        # the whole problem's first block of each of the group's chains, in the group's own rank order
        starts = [p.P * ranks[b.d0 + d] for d in _seq_order(lengths[b.d0 : b.d1])]
        if starts == list(range(starts[0], starts[0] + p.P * len(starts), p.P)):
            blocks.append(starts[0])
        else:
            blocks.append(None)
            index_at[i] = (len(index), p.P * len(starts))
            index += [s + j for s in starts for j in range(p.P)]
    return _Packing(
        batches, p.P, tuple(blocks), tuple(words), tail_at, tuple(index), index_at
    )


class KdaWorkspace:
    """One arena of internal-layout intermediates for one problem shape, allocated once and reused by every call.

    :func:`chunk_kda_fwd` / :func:`chunk_kda_bwd` allocate only their outputs (and the saved set with its tables).
    ``backward=False`` sizes the arena for the forward only; a backward workspace is sized for the larger of the
    saved-set paths' layout and the replay's (both run in it).  ``save_mode`` is the default of the calls on this
    workspace and sizes the arena for the tape layouts.  ``views(mode)`` are the buffer views of one layout on
    the arena (``fwd`` / ``bwd`` / ``replay`` for the light layouts); the tables are not part of the workspace
    (they travel with the saved set).  ``bytes`` is the arena's size (:func:`workspace_bytes`).  The arena is
    not initialized: every element a launch reads is written earlier in the same call (the layout note), so its
    content at allocation never reaches a result; ``zero_init=True`` fills it with zeros for diagnostics (an
    arena of 2.5 GiB takes about 0.35 ms per fill).  ``launcher`` replaces ``cake_jit.run_module`` for
    diagnostics only.  ``arena`` (:class:`KdaArena`, off by default; kept as :attr:`arena_option`, the attribute
    ``arena`` being the uint8 arena tensor) selects the lower-memory arena option: the lean backward layouts (no
    replay on this workspace), the tail split into ``tail_passes`` passes and the document groups of
    ``batch_tokens``; under a batched option ``doc_lengths`` (the documents' token counts) sizes the arena for the
    exact groups instead of the bound of every group, and the calls on the workspace take the same ``doc_lengths``.
    The attribute ``tail_passes`` is the passes of the workspace's backward tail: the option's, and in a group's
    view (:meth:`batch_view`) the group's own, the fewest up to the option's that fit the arena.
    """

    def __init__(
        self,
        plan: Plan,
        device: torch.device | str,
        *,
        backward: bool = True,
        zero_init: bool = False,
        save_mode: str = "light",
        launcher: Launcher | None = None,
        arena: KdaArena | None = None,
        doc_lengths: Sequence[int] | None = None,
    ) -> None:
        if save_mode not in SAVE_MODES:
            raise ValueError(
                f"KdaWorkspace: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
            )
        self.arena_option = KdaArena() if arena is None else arena
        plan.check_arena(self.arena_option, backward=backward)
        self.doc_lengths = (
            None if doc_lengths is None else plan.check_doc_lengths(doc_lengths)
        )
        self.plan = plan
        self.device_index = _device_index(torch.device(device))
        self.device = torch.device("cuda", self.device_index)
        self.arch = cake_jit.device_arch(self.device_index)
        if plan.arch != self.arch:
            raise ValueError(
                f"KdaWorkspace: the plan selects the records of {plan.arch}, the device is {self.arch}"
            )
        self.backward = backward
        self.save_mode = save_mode
        # the passes of this workspace's backward tail (a group's view of a batched arena takes its own)
        self.tail_passes = self.arena_option.tail_passes
        self.launcher: Launcher = cake_jit.run_module if launcher is None else launcher
        self.bytes = plan.arena_bytes(
            backward=backward,
            save_mode=save_mode,
            arena=self.arena_option,
            doc_lengths=self.doc_lengths,
        )
        self.zero_init = zero_init
        # The pad-chunk rows of every buffer a consumer reads are written in the same call (see the layout note),
        # so the content at allocation is irrelevant to the results: no fill by default (``zero_init=True`` zeros
        # the arena for diagnostics; the never-read pads are then defined too).
        self.arena = (torch.zeros if zero_init else torch.empty)(
            self.bytes, dtype=torch.uint8, device=self.device
        )
        self._views_cache: dict[str, types.SimpleNamespace] = {}
        # the eye (eye / flags) and the carry launches' epoch counter (next_epoch): one-entry lists, shared by every
        # view of this arena (with_plan, batch_view), since the flags the epochs stamp live in the shared eye
        self._eye_box: list[torch.Tensor | None] = [None]
        self._epoch: list[int] = [0]
        # the batched option's views of this arena per group geometry, and per doc_lengths its groups with their
        # host-derived state blocks and tail counts and the device copy of those integers (None until a call needs
        # it); the doc_lengths a CUDA-graph capture used, whose entries the cache keeps for the workspace's lifetime
        self._batch_views: dict[tuple[int, int, int], KdaWorkspace] = {}
        self._doc_cache: dict[tuple[int, ...], _Packing] = {}
        self._doc_graphs: set[tuple[int, ...]] = set()

    def views(self, mode: str) -> types.SimpleNamespace:
        """The buffer views of layout ``mode`` on this arena (``ValueError`` when the arena is too small for it);
        under a lean arena the backward modes map to their lean forms and the replay modes are refused."""
        mode = self.arena_option.mode(mode)
        ns = self._views_cache.get(mode)
        if ns is None:
            # the tail split shapes the lean backward layouts only; every other layout has one form
            tail = self.tail_passes if mode in LEAN_MODES else 1
            need = self.plan.workspace_bytes(mode, tail)
            if need > self.bytes:
                raise ValueError(
                    f"KdaWorkspace: layout {mode!r} of the plan (S={self.plan.S}, num_docs={self.plan.N}) needs "
                    f"{need} B, the arena has {self.bytes} B (allocated for backward={self.backward}, "
                    f"save_mode={self.save_mode!r}, arena={self.arena_option}, doc_lengths={self.doc_lengths})"
                )
            ns = types.SimpleNamespace()
            regions = {r.name: r for r in self.plan.layout(mode, tail)}
            for r in regions.values():
                setattr(
                    ns,
                    r.name,
                    self.arena[r.offset : r.offset + r.nbytes]
                    .view(r.dtype)
                    .view(r.shape),
                )
            if (
                "seg_b" in regions
            ):  # the kernels' seg_state [2, NSEG, H, 128, 128]: offsets then transfers
                sb, sm = regions["seg_b"], regions["seg_m"]
                if sm.offset != sb.offset + sb.nbytes:
                    raise ValueError(
                        f"layout {mode!r}: seg_b and seg_m must be adjacent blocks"
                    )
                ns.seg_state = (
                    self.arena[sb.offset : sm.offset + sm.nbytes]
                    .view(torch.float32)
                    .view(2, *sb.shape)
                )
            self._views_cache[mode] = ns
        return ns

    @property
    def eye(self) -> torch.Tensor:
        """FP32 ``[128, 128]`` buffer outside the arena, built as the identity: the operand the segmented walks
        take as the initial state of the transfer walk (the kernels form the identity themselves and never read
        it), and the home of the overlap pair's release flags (:attr:`flags`)."""
        eye = self._eye_box[0]
        if eye is None:
            eye = self._eye_box[0] = torch.eye(
                D, dtype=torch.float32, device=self.device
            )
        return eye

    @property
    def flags(self) -> torch.Tensor:
        """The release flags of the forward's carry || pass-2 overlap (``seg_carry_ov`` / ``fwdh_p3_o_ov*``) and of
        the two-level carry (``seg_carry_v8``): the ``uint32`` view of :attr:`eye` (16384 words; the pair uses
        ``2 * NSEG * H`` of them, one per segment, head and carry half, the two-level carry
        :func:`cake_jit.flag_words` of its grid, each stamped with the launch's epoch)."""
        return self.eye.view(torch.uint32)

    def next_epoch(self) -> int:
        """The epoch of the next carry launch under the overlap pair or the two-level carry (1, 2, ...): never
        repeats on this arena, so a flag left by an earlier launch can never equal the epoch a waiting CTA
        expects."""
        self._epoch[0] += 1
        return self._epoch[0]

    @property
    def fwd(self) -> types.SimpleNamespace:
        return self.views("fwd")

    @property
    def bwd(self) -> types.SimpleNamespace:
        return self.views("bwd")

    @classmethod
    def for_problem(
        cls,
        S: int,
        num_docs: int,
        H: int,
        device: torch.device | str,
        *,
        backward: bool = True,
        save_mode: str = "light",
        launcher: Launcher | None = None,
        arena: KdaArena | None = None,
        doc_lengths: Sequence[int] | None = None,
    ) -> KdaWorkspace:
        """A workspace for ``S`` tokens in ``num_docs`` documents with ``H`` heads on ``device``."""
        index = _device_index(torch.device(device))
        return cls(
            plan(S, num_docs, H, cake_jit.sm_count(index), cake_jit.device_arch(index)),
            device,
            backward=backward,
            save_mode=save_mode,
            launcher=launcher,
            arena=arena,
            doc_lengths=doc_lengths,
        )

    @classmethod
    def for_budget(
        cls,
        S: int,
        num_docs: int,
        H: int,
        device: torch.device | str,
        *,
        arena_budget_bytes: int,
        doc_lengths: Sequence[int] | None = None,
        backward: bool = True,
        save_mode: str = "light",
        launcher: Launcher | None = None,
    ) -> KdaWorkspace:
        """A workspace of the first arena option whose arena fits ``arena_budget_bytes`` (:func:`arena_for_budget`:
        the default, lean, two tail passes, four, then four passes over document groups halving from half the pack
        down to one document per group, which needs ``doc_lengths``); ``ValueError`` when none fits."""
        index = _device_index(torch.device(device))
        p = plan(S, num_docs, H, cake_jit.sm_count(index), cake_jit.device_arch(index))
        option = arena_for_budget(
            p,
            arena_budget_bytes,
            backward=backward,
            save_mode=save_mode,
            doc_lengths=doc_lengths,
        )
        return cls(
            p,
            device,
            backward=backward,
            save_mode=save_mode,
            launcher=launcher,
            arena=option,
            doc_lengths=doc_lengths,
        )

    def _view(self, plan: Plan) -> KdaWorkspace:
        """A view of this arena under ``plan``: the same bytes, ``eye`` / ``flags`` and epoch counter (the overlap
        pair's release flags live in the eye, stamped with the launch's epoch: views sharing the flags must share
        the counter, or a later launch's pass 2 could pass its wait on a flag an earlier one left), with a view
        cache of its own."""
        other = KdaWorkspace.__new__(KdaWorkspace)
        other.__dict__.update(self.__dict__)
        other.plan = plan
        other._views_cache = {}
        other._batch_views = {}
        other._doc_cache = {}
        other._doc_graphs = set()
        return other

    def with_plan(self, plan: Plan) -> KdaWorkspace:
        """A workspace sharing this arena under another plan of the same ``(S, num_docs, H)`` problem (its layouts
        must fit the arena when used)."""
        if (plan.S, plan.N, plan.H) != (self.plan.S, self.plan.N, self.plan.H):
            raise ValueError(
                "with_plan: the plan must describe the same (S, num_docs, H) problem"
            )
        return self._view(plan)

    def batch_view(self, batch: Batch, *, tape: bool = False) -> KdaWorkspace:
        """The view of this arena under the plan of one document group (:meth:`Plan.batch_plan`; ``tape`` selects
        the geometry of the group's views of a whole-problem tape), cached per group geometry: the group's layouts
        on the same bytes (they must fit: :meth:`views` checks), the shared ``eye`` / ``flags`` and epoch counter,
        and the group's own ``tail_passes`` (:func:`_group_tail_passes`: the fewest that fit the arena)."""
        nc_pad = batch.NC_pad_tape if tape else batch.NC_pad
        key = (batch.S, batch.N, nc_pad)
        other = self._batch_views.get(key)
        if other is None:
            other = self._view(self.plan.batch_plan(batch, tape=tape))
            other.tail_passes = _group_tail_passes(
                self.plan,
                batch,
                nc_pad,
                self.arena_option,
                nbytes=self.bytes,
                save_mode=self.save_mode,
            )
            self._batch_views[key] = other
        return other

    def batches(
        self, doc_lengths: Sequence[int] | None
    ) -> tuple[tuple[Batch, ...], torch.Tensor] | None:
        """The document groups of a call on this workspace and the device copy of ``doc_lengths`` (for the
        asynchronous check against ``cu_seqlens``), ``None`` when the option is not batched (``doc_lengths`` is then
        unused); ``ValueError`` when it is and ``doc_lengths`` is missing or mismatches the problem.  Cached per
        ``doc_lengths`` (the last few packings) with what the calls derive from them on the host (:class:`_Packing`:
        the groups' state blocks by chain rank, the tail passes' chunk counts), whose device copy rides with the
        lengths'.  The copy goes from pinned host memory on the current stream without blocking: the host never
        waits for the device.  Under CUDA-graph capture the copy must already be cached (``RuntimeError``
        otherwise): a captured copy would read a host buffer the graph does not own, so a batched call is captured
        on a caller-owned workspace that ran these ``doc_lengths`` once before the capture.  A captured graph reads
        the copy at each replay, so the cache keeps the packings a capture used for the workspace's lifetime (as
        the arena the graph also reads) and evicts only the others."""
        packing = self._packing(doc_lengths)
        return None if packing is None else (packing.batches, packing.lengths)

    def _packing(self, doc_lengths: Sequence[int] | None) -> _Packing | None:
        """:meth:`batches` with everything the calls derive from ``doc_lengths`` (:class:`_Packing`)."""
        if not self.arena_option.batched:
            return None
        key = self.plan.check_doc_lengths(doc_lengths)
        capturing = torch.cuda.is_current_stream_capturing()
        entry = self._doc_cache.get(key)
        if entry is None:
            entry = _host_packing(
                self.plan,
                self.arena_option,
                key,
                nbytes=self.bytes,
                save_mode=self.save_mode,
            )
        if entry.words_dev is None:
            if capturing:
                raise RuntimeError(
                    "KdaWorkspace: a batched arena under CUDA-graph capture needs the device copy of these "
                    "doc_lengths made before the capture: pass a caller-owned workspace and run it once with "
                    "these doc_lengths outside the capture"
                )
            entry = entry.to(self.device)
        if key not in self._doc_cache:
            # a new packing evicts the oldest of 8 cached ones no capture used (a graph reads its copy at each replay)
            free = [k for k in self._doc_cache if k not in self._doc_graphs]
            if len(free) >= 8:
                self._doc_cache.pop(free[0])
        self._doc_cache[key] = entry
        if capturing:
            self._doc_graphs.add(key)
        return entry

    def paths(self) -> tuple[str, ...]:
        """The paths this workspace serves: the two forward paths (with and without a saved set) and, with
        ``backward``, the backward paths this plan can take (``light_states`` on the segmented route only; no
        ``replay`` under a lean arena)."""
        if not self.backward:
            return ("fwd", "fwd_nosave")
        return tuple(
            path
            for path in PATHS
            if (path != "light_states" or self.plan.recur == "seg")
            and not (path == "replay" and self.arena_option.lean)
        )

    def records(self) -> dict[str, dict[str, str]]:
        """Path -> stage -> the record this plan launches there (the paths of :meth:`paths`)."""
        return {
            path: {
                stage: self.plan.record(stage, path) for stage in self.plan.stages(path)
            }
            for path in self.paths()
        }

    def load(self) -> None:
        """Load (or JIT-build) every program this plan launches on its paths; call once before timing."""
        names = {
            name for by_stage in self.records().values() for name in by_stage.values()
        }
        for name in sorted(names):
            cake_jit.load(name, self.arch)

    def run(
        self,
        stage: str,
        values: dict[str, Any],
        *,
        path: str,
        launch: int = 0,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        """Launch the record of ``stage`` on ``path`` (its ``launch``-th launch) with the logical argument
        ``values``; ``grid`` replaces the plan's grid (the tail split's per-range grids)."""
        name = self.plan.record(stage, path)
        self.launcher(
            name,
            values,
            grid=self.plan.grid(stage, path, launch) if grid is None else grid,
            device_index=self.device_index,
        )


# The workspaces of the calls made without one (the default entry), per (device, stream, problem, arena option and,
# under a batched option, doc_lengths): the plan, the layout views and the groups' plans are built once, while the
# arena storage, the eye and the device copy of doc_lengths are allocated at the start of each call and freed at
# its end (:func:`_entry_acquire` / :func:`_entry_release`).  Between calls only host objects stay; a call's arena
# is the same per-call allocation as before, from the same caching allocator on the same stream.  Calls on another
# stream take a workspace of their own, so no two queued calls share an eye.
_ENTRY_KEYS = 64
_ENTRY_PER_KEY = 2
_entry_pool: collections.OrderedDict[tuple[Any, ...], list[KdaWorkspace]] = (
    collections.OrderedDict()
)
_entry_lock = threading.Lock()


def _entry_acquire(
    S: int,
    N: int,
    H: int,
    device: torch.device,
    *,
    backward: bool,
    save_mode: str,
    arena: KdaArena | None,
    doc_lengths: Sequence[int] | None,
) -> tuple[KdaWorkspace, tuple[Any, ...] | None]:
    """The workspace of one call without a workspace argument, with its arena storage allocated, and its pool key
    (``None``: a workspace of this call alone, under CUDA-graph capture, whose device allocations belong to the
    graph; a batched arena is refused there, :meth:`KdaWorkspace.batches`)."""
    if torch.cuda.is_current_stream_capturing():
        ws = KdaWorkspace.for_problem(
            S,
            N,
            H,
            device,
            backward=backward,
            save_mode=save_mode,
            arena=arena,
            doc_lengths=doc_lengths,
        )
        return ws, None
    index = _device_index(device)
    option = KdaArena() if arena is None else arena
    lengths = (
        tuple(int(x) for x in doc_lengths)
        if option.batched and doc_lengths is not None
        else None
    )
    key = (
        index,
        torch.cuda.current_stream(index).cuda_stream,
        S,
        N,
        H,
        backward,
        save_mode,
        option,
        lengths,
    )
    with _entry_lock:
        free = _entry_pool.get(key)
        ws = free.pop() if free else None
    if ws is None:
        ws = KdaWorkspace.for_problem(
            S,
            N,
            H,
            device,
            backward=backward,
            save_mode=save_mode,
            arena=option,
            doc_lengths=doc_lengths,
        )
    else:
        if doc_lengths is not None and lengths is None:
            # checked per call, as a new workspace checks them (unused off the batched option)
            ws.plan.check_doc_lengths(doc_lengths)
        ws.arena.untyped_storage().resize_(ws.bytes)
    return ws, key


def _entry_release(ws: KdaWorkspace, key: tuple[Any, ...] | None) -> None:
    """Free the device memory of a call's workspace (arena storage, eye, the device copies of doc_lengths and of
    what derives from them: the caching allocator takes them back on the call's stream, as when a per-call tensor
    dies) and keep the rest for the next call with the same key."""
    if key is None:
        return
    ws.arena.untyped_storage().resize_(0)
    # the next call builds a fresh eye (flags 0 / the identity's 1.0f), so its epochs restart below any flag value
    ws._eye_box[0] = None
    ws._epoch[0] = 0
    ws._doc_cache = {k: packing.on_host() for k, packing in ws._doc_cache.items()}
    with _entry_lock:
        free = _entry_pool.setdefault(key, [])
        _entry_pool.move_to_end(key)
        if len(free) < _ENTRY_PER_KEY:
            free.append(ws)
        while len(_entry_pool) > _ENTRY_KEYS:
            _entry_pool.popitem(last=False)


@contextlib.contextmanager
def _workspace(
    workspace: KdaWorkspace | None,
    S: int,
    N: int,
    H: int,
    device: torch.device,
    *,
    backward: bool,
    save_mode: str | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> Iterator[KdaWorkspace]:
    """The workspace of one call: the caller's (checked against the problem) or, without one, the call's own,
    whose device memory lives for the call alone (:func:`_entry_acquire`)."""
    if workspace is None:
        ws, key = _entry_acquire(
            S,
            N,
            H,
            device,
            backward=backward,
            save_mode="light" if save_mode is None else save_mode,
            arena=arena,
            doc_lengths=doc_lengths,
        )
        try:
            yield ws
        finally:
            _entry_release(ws, key)
        return
    if arena is not None and arena != workspace.arena_option:
        raise ValueError(
            f"workspace: allocated with arena={workspace.arena_option}, the call asks for arena={arena}; pass "
            "arena=None to run with the workspace's option, or allocate a workspace with this one"
        )
    p = workspace.plan
    if (p.S, p.N, p.H) != (S, N, H):
        raise ValueError(
            f"workspace: allocated for S={p.S}, num_docs={p.N}, H={p.H}; the operands are S={S}, "
            f"num_docs={N}, H={H}"
        )
    if workspace.device != device:
        raise ValueError(
            f"workspace: on {workspace.device}, the operands are on {device}"
        )
    if backward and not workspace.backward:
        raise ValueError(
            "workspace: allocated with backward=False; the backward needs its layout"
        )
    yield workspace


# ----------------------------------------------------------------------------------------------------------------
# Operand checks (metadata only)


def _squeeze_batch(x: torch.Tensor, name: str, rank: int) -> torch.Tensor:
    if x.dim() == rank + 1:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}: the batch axis must be 1, got shape {tuple(x.shape)}"
            )
        x = x.squeeze(0)
    if x.dim() != rank:
        raise ValueError(
            f"{name}: expected {rank}-D (or a leading batch axis of 1), got shape {tuple(x.shape)}"
        )
    return x


def _check(
    x: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(x.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(
            f"{name}: must be contiguous (strides {x.stride()}); the kernels take no strided operands "
            "and the host makes no copy"
        )
    if x.device != device:
        raise ValueError(f"{name}: on {x.device}, expected {device}")


def _operands(q, k, v, g, beta, cu_seqlens):
    """The operands without their optional batch axis, the problem scalars and whether the inputs carried a batch
    axis (the outputs take the same rank)."""
    v = _squeeze_batch(v, "v", 3)
    S, H = int(v.shape[0]), int(v.shape[1])
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            f"cu_seqlens: expected a 1-D int32 tensor of num_docs + 1 >= 2 entries, got shape "
            f"{tuple(cu_seqlens.shape)}"
        )
    N = int(cu_seqlens.numel()) - 1
    device = v.device
    batched = q.dim() == 4
    q, k, g = (_squeeze_batch(x, n, 3) for x, n in ((q, "q"), (k, "k"), (g, "g")))
    beta = _squeeze_batch(beta, "beta", 2)
    _check(q, "q", (S, H, D), torch.bfloat16, device)
    _check(k, "k", (S, H, D), torch.bfloat16, device)
    _check(v, "v", (S, H, D), torch.bfloat16, device)
    _check(g, "g", (S, H, D), torch.float32, device)
    _check(beta, "beta", (S, H), torch.bfloat16, device)
    _check(cu_seqlens, "cu_seqlens", (N + 1,), torch.int32, device)
    return q, k, v, g, beta, S, N, H, batched


def _ranked(x: torch.Tensor, batched: bool) -> torch.Tensor:
    return x.unsqueeze(0) if batched else x


# ----------------------------------------------------------------------------------------------------------------
# Launch sequences


def _values(
    ws: KdaWorkspace,
    views: types.SimpleNamespace,
    tables: tuple[torch.Tensor, ...],
    **extra: Any,
) -> dict[str, Any]:
    """The logical argument values of one call: the layout's arena buffers (and the transposed views of ``w``
    the fused recomputes write), the tables, the plan scalars and the call's own tensors (``extra``)."""
    p = ws.plan
    values: dict[str, Any] = dict(vars(views))
    values.update(zip(TABLE_NAMES, tables, strict=False))
    if hasattr(views, "w"):
        values["w_t"] = views.w.view(p.NC_pad, p.H, D, BT)
        values["w_tk"] = views.w.view(p.NC_pad * D, p.H, BT)
    values.update(
        num_seqs=p.N,
        num_chunks_pad=p.NC_pad,
        num_heads=p.H,
        group=1,
        num_segments=p.NSEG,
        segments=p.P,
        seg_arg=p.seg_arg,
        proportional=p.proportional,
        # the chunk pairs the persistent pair kernels walk
        num_pairs=p.R // cake_jit.PAIR_ROWS,
    )
    values.update(extra)
    return values


def _tables(
    ws: KdaWorkspace, cu_seqlens: torch.Tensor, path: str
) -> tuple[torch.Tensor, ...]:
    """A fresh table buffer (pad entries are written as zeros): the four table views and, on the segmented route,
    the block table ``seg_table`` (``[NSEG + 2 * num_docs + 1]``: the chains' first segments in the layout table's
    document order, then per chain its segment starts and end chunk; per-document proportional segment counts
    under ``proportional``) derived from ``seq_chunk_start``, ``seq_order`` and ``segments``.  Written by the table
    program where the plan dispatches one on ``path`` (``table_v2`` / ``table`` on the serial route; ``table`` then
    ``seg_table`` on the segmented route above 256 documents; ``table_v3``, the chunk tables and the block table in
    one launch, where a saved-set path rebuilds them up to 256 documents); on the forward and the replay of the
    segmented route up to 256 documents no table program runs: the intra program ``fwd_intra_v14``, the first
    launch there, computes the layout from ``cu_seqlens`` and writes every entry of the buffer itself."""
    p = ws.plan
    buffer = _alloc((p.table_words(),), torch.int32, ws.device)
    tables = tuple(buffer[off : off + words] for _name, off, words in p.table_layout())
    if p.dispatched("table", path):
        values = {
            "cu_seqlens": cu_seqlens,
            "num_seqs": p.N,
            "num_chunks_pad": p.NC_pad,
            "segments": p.P,
            "proportional": p.proportional,
        }
        values.update(zip(TABLE_NAMES, tables, strict=False))
        ws.run("table", values, path=path)
        if p.recur == "seg" and p.dispatched("seg_table", path):
            ws.run("seg_table", values, path=path)
    return tables


def _check_tables(saved: KdaChunkSaved, ws: KdaWorkspace) -> None:
    p = ws.plan
    layout = p.table_layout()
    if saved.tables is None or len(saved.tables) != len(layout):
        raise ValueError(
            f"saved.tables: {len(saved.tables or ())} tables, the plan ({p.recur} route) has {len(layout)}"
        )
    for t, (name, _off, words) in zip(saved.tables, layout, strict=True):
        _check(t, f"saved.tables.{name}", (words,), torch.int32, ws.device)


def _assert_doc_lengths(cu_seqlens: torch.Tensor, lengths: torch.Tensor) -> None:
    """The batched calls' guard that the host's ``doc_lengths`` describe ``cu_seqlens`` (the host reads no tensor):
    a device-side comparison whose failure trips an asynchronous assertion (the next synchronization raises),
    before a group's tables could be written past the geometry the host derived.  Three tiny kernels per call."""
    torch._assert_async(
        (cu_seqlens[1:] - cu_seqlens[:-1]).eq(lengths).all(),
        "cake_kda_train: doc_lengths do not match cu_seqlens",
    )


def _fwd_intra(
    ws: KdaWorkspace,
    values: dict[str, Any],
    *,
    path: str,
    write_ext: int,
    write_int: int,
    write_prep: int,
) -> None:
    """The intra-chunk program: ``Aqk`` / ``Akk`` on the internal rows (``write_int``: where a later program of the
    path reads them, :meth:`Plan.internal_rows`) and on the token rows
    (``write_ext``), and with ``write_prep`` the GLM prep's outputs the later programs read (``gk``, ``kb``, ``kg``,
    ``qg`` and the ``v_new`` pad rows: the forward); the replay (``write_prep = 0``) has them from its own prep and
    passes the same buffers untouched.  The segmented route's record up to 256 documents (``fwd_intra_v14``) also
    computes the chunk layout from ``cu_seqlens`` and writes the chunk tables and the block table: it is the first
    launch of its call and no table program runs there."""
    ws.run(
        "fwd_intra",
        {
            **values,
            "write_ext": int(write_ext),
            "write_int": int(write_int),
            "write_prep": int(write_prep),
        },
        path=path,
    )


def _forward_recurrence(
    ws: KdaWorkspace,
    values: dict[str, Any],
    *,
    path: str,
    seg_state: torch.Tensor | None,
    hseg: torch.Tensor | None,
    write_tapes: int,
) -> None:
    """The state recurrence of ``path``: the serial walk (one launch of the ``fwdh`` stage: the forward's forms
    the output, and from 32 heads the WY representation; the backward's recompute of the 64-column form from 64
    heads forms the WY representation and leaves the transposed ``w`` tile) or the segmented passes: pass 1 (the
    ``fwdh`` stage, grid ``2 * NSEG * H``: the walk from zero gives the segment offsets, the walk from the identity
    the transfers, into ``seg_state[0]`` / ``[1]``; no tapes), seg_carry (the carried state of every segment into
    ``hseg``), pass 2 (the ``fwdh_p3`` stage, grid ``NSEG * H``: the walk from ``hseg``; the forward's forms the
    output, the saved-state backward's of the rows with at most 32 (document, head) chains the WY representation
    and the transposed ``w`` tile).  ``seg_state`` and ``hseg`` are the arena's blocks or per-call tensors (the light forward above the
    size rule saves both, the tape forward saves ``seg_state``'s transfer half).  On the ``light_states`` path
    (the backward from a version-4 saved set carrying the states) only pass 2 runs, from the saved ``hseg``; pass
    2 never writes ``seg_state`` (``write_ht = 0``).  ``write_tapes`` is 1 where the ``h`` / ``v_new`` tapes are
    needed (the backward, the tape forward).  On the forward's segmented rows whose pass-2 grid is at least 3.0
    waves of the device on ``sm_100a`` / ``sm_103a`` (2.3 on ``sm_107a``) the carry and pass 2 are the overlap pair
    ``seg_carry_ov`` / ``fwdh_p3_o_ov`` (``_id`` above two heads; on the saving forward the ``_rows`` twins, reading
    the saved ``Aqk`` token rows): pass 2 is launched with the programmatic-stream-serialization attribute and
    waits on the carry's per-segment release flags instead of the stream order.  The saving forward's fused serial
    recurrence is ``fwdh_wy_rows`` (the saved ``Akk`` / ``Aqk`` token rows), the other forward's ``fwdh_wy``; below
    the fused threshold its 32-column fused-output recurrence is ``fwdh_o_v2_rows`` (the saved ``Aqk`` rows), the
    other forward's ``fwdh_o_v2``; the 64-column form is ``fwdh_o_v1_rows`` on the saving forward of ``sm_107a`` and
    ``fwdh_o_v1`` otherwise."""
    p = ws.plan
    if p.recur != "seg":
        ws.run("fwdh", {**values, "write_tapes": int(write_tapes)}, path=path)
        return
    assert seg_state is not None and hseg is not None
    passes = {
        **values,
        "hseg": hseg,
        "eye": ws.eye,
        "flags": ws.flags,
        "seg_state": seg_state,
    }
    if p.dispatched("fwdh", path):
        ws.run(
            "fwdh", {**passes, "use_h0": 0, "write_tapes": 0, "write_ht": 1}, path=path
        )
        carry = {
            **values,
            "seg_m": seg_state[1].view(-1, D, D),
            "carry_b": seg_state[0],
            "carry_out": hseg,
            "reverse": 0,
        }
        if p.record("seg_carry", path) == "seg_carry_ov":
            # the carry || pass-2 overlap of the forward's wide rows: the carry (a plain launch) releases one
            # flag per segment, head and half, stamped with this call's epoch, after each carried state is
            # stored; pass 2, launched with the programmatic-stream-serialization attribute, starts while the
            # carry runs and spins on its segment's two flags before reading h0.  Both launches read the epoch.
            passes["epoch"] = carry["epoch"] = ws.next_epoch()
            carry["flags"] = ws.flags
        elif _two_level(ws, path):
            carry.update(_two_level_operands(ws, path, hseg))
        ws.run("seg_carry", carry, path=path)
    ws.run(
        "fwdh_p3",
        {**passes, "use_h0": 1, "write_tapes": int(write_tapes), "write_ht": 0},
        path=path,
    )


def _adjoint_recurrence(
    ws: KdaWorkspace,
    views: types.SimpleNamespace,
    values: dict[str, Any],
    *,
    path: str,
    seg_m: torch.Tensor | None,
) -> None:
    """Stage dhu: the serial reverse walk or the segmented three passes: pass 1 (the reverse walk from zero: the
    segment offsets into ``dseg``), seg_carry in reverse (the carried adjoint states into ``dhseg``, with the
    forward's transfers ``seg_m``), pass 2 (from ``dhseg``, writing the ``dh`` / ``dv2`` tapes)."""
    if ws.plan.recur != "seg":
        ws.run("dhu", values, path=path)
        return
    assert seg_m is not None
    passes = {**values, "dhseg": views.dhseg, "dseg": views.dseg}
    ws.run(
        "dhu",
        {**passes, "use_dht": 0, "write_tapes": 0, "write_ht": 1},
        path=path,
        launch=0,
    )
    carry = {
        **values,
        "seg_m": seg_m,
        "carry_b": views.dseg,
        "carry_out": views.dhseg,
        "reverse": 1,
    }
    if _two_level(ws, path):
        carry.update(_two_level_operands(ws, path, views.dhseg))
    ws.run("seg_carry", carry, path=path)
    ws.run(
        "dhu",
        {**passes, "use_dht": 1, "write_tapes": 1, "write_ht": 0},
        path=path,
        launch=1,
    )


def _two_level(ws: KdaWorkspace, path: str) -> bool:
    """Whether ``path``'s carry record is the two-level carry (its grid rule is ``carry_groups``)."""
    name = ws.plan.record("seg_carry", path)
    return cake_jit.MODULES[name]["grid_rule"]["kind"] == "carry_groups"


def _two_level_operands(
    ws: KdaWorkspace, path: str, carry_out: torch.Tensor
) -> dict[str, Any]:
    """The two-level carry's operands beyond the chain carry's: the carried states viewed as ``[NSEG * H, 128,
    128]`` tiles (the groups' products ride in them), the plan's segment group size, the release flags under a
    fresh epoch (the words a launch of the plan's grid touches must fit the flag buffer)."""
    grid = ws.plan.grid("seg_carry", path)[0]
    words = cake_jit.flag_words(grid)
    if words > ws.flags.numel():
        raise ValueError(
            f"the two-level carry's {grid} CTAs touch {words} flag words; the flag buffer has {ws.flags.numel()}"
        )
    return {
        "carry_out_tiles": carry_out.view(-1, D, D),
        "gsize": ws.plan.gsize,
        "flags": ws.flags,
        "epoch": ws.next_epoch(),
    }


def _tail_ranges(NC_pad: int, passes: int) -> tuple[tuple[int, int], ...]:
    """The ``(first chunk, length)`` ranges of ``passes`` tail passes over ``NC_pad`` chunks: ``ceil(NC_pad /
    passes)`` chunks each, the last one clipped (fewer ranges than ``passes`` when the clipped tail is empty)."""
    length = -(-int(NC_pad) // int(passes))
    return tuple(
        (c0, min(length, int(NC_pad) - c0)) for c0 in range(0, int(NC_pad), length)
    )


def _tail_passes(
    ws: KdaWorkspace,
    views: types.SimpleNamespace,
    values: dict[str, Any],
    tables: tuple[torch.Tensor, ...],
    *,
    path: str,
    h: torch.Tensor,
    nc_vec: torch.Tensor | None = None,
) -> None:
    """Steps 6-8 of the backward (the fused gradient, GLM intra and, where it launches, the gate epilogue) once per tail range
    (:func:`_tail_ranges`) of a lean workspace with ``tail_passes > 1`` (a group's view: its own).  Per range: the
    whole-sequence arena buffers (``gk``, ``v_new``, ``dv2``, ``beta_int``, the ``h`` / ``dh`` tapes) as leading-axis
    views offset by the range's first chunk, the chunk tables as the range's slices, ``seq_chunk_start`` as the
    one-entry count of the range's real chunks (``num_seqs = 0`` makes the fused gradient read that entry as its item
    bound), the lean layout's per-pass buffers as they are (sized for one range) and the token tensors whole (every
    kernel reaches them through the absolute ``chunk_bos``); the grids are the plan's rules at the range's chunk
    count.  The same kernels address the same bytes as in one pass, so the gradients are bitwise the same.
    ``nc_vec`` is the real chunk count of every range where the host has it (a batched call's groups: from
    ``doc_lengths``, :class:`_Packing`), else formed on the device."""
    p = ws.plan
    ranges = _tail_ranges(p.NC_pad, ws.tail_passes)
    if nc_vec is not None and nc_vec.numel() != len(ranges):
        raise RuntimeError(
            f"_tail_passes: {nc_vec.numel()} host chunk counts for {len(ranges)} tail ranges"
        )
    # without the host's counts (a lean call that is not batched has no doc_lengths), the real chunks of every range
    # on the device: clamp(NC_real - c0, 0, ceil(NC_pad / B)), NC_real = seq_chunk_start[N] <= NC_pad, so the
    # clipped last range needs no bound of its own; the first chunks come from an arange (the host does not wait
    # for the device)
    if nc_vec is None:
        nc_vec = torch.sub(
            tables[2][p.N : p.N + 1],
            torch.arange(
                0, p.NC_pad, ranges[0][1], dtype=torch.int32, device=tables[2].device
            ),
        ).clamp_(0, ranges[0][1])
    chunk_bos, chunk_len = tables[0], tables[1]
    for i, (c0, length) in enumerate(ranges):
        rows = BT * c0
        pass_values = {
            **values,
            "gk": values["gk"][rows:],
            "v_new": values["v_new"][rows:],
            "dv2": values["dv2"][rows:],
            "beta_int": values["beta_int"][rows:],
            "h2d": h[c0:].view(-1, D),
            "dh2d": views.dh[c0:].view(-1, D),
            "chunk_bos": chunk_bos[c0 : c0 + length],
            "chunk_len": chunk_len[c0 : c0 + length],
            "seq_chunk_start": nc_vec[i : i + 1],
            "num_seqs": 0,
            "num_chunks_pad": length,
        }
        for stage in ("dqkg", "glm_intra", "gate_epilogue"):
            if not p.dispatched(stage, path):
                continue  # the fused GLM intra program writes dg / dbeta itself
            grid = cake_jit.launch_grid(
                p.record(stage, path),
                num_chunks_pad=length,
                num_seqs=p.N,
                num_heads=p.H,
                sm_count=p.sm_count,
                num_segments=p.NSEG,
            )
            ws.run(stage, pass_values, path=path, grid=grid)


def _saved_rows(
    values: dict[str, Any],
    p: Plan,
    *,
    qg: torch.Tensor | None,
    w: torch.Tensor | None,
    u: torch.Tensor | None,
) -> None:
    """Point the call's ``qg`` / ``w`` / ``u`` (and the transposed views of ``w``) at the version-4 saved
    tensors where given."""
    if qg is not None:
        values["qg"] = qg
    if w is not None or u is not None:
        if w is None or u is None:
            raise ValueError("the saved w and u come together")
        values.update(
            w=w,
            u=u,
            w_t=w.view(p.NC_pad, p.H, D, BT),
            w_tk=w.view(p.NC_pad * D, p.H, BT),
        )


def _forward_launches(
    ws: KdaWorkspace,
    views: types.SimpleNamespace,
    cu_seqlens: torch.Tensor,
    tables: tuple[torch.Tensor, ...],
    *,
    path: str,
    write_ext: int,
    write_tapes: int,
    h: torch.Tensor,
    v_new: torch.Tensor,
    seg_state: torch.Tensor | None,
    hseg: torch.Tensor | None,
    aqk_ext: torch.Tensor,
    akk_ext: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    qg: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> None:
    """The launches of one forward on ``ws`` (the whole problem, or one document group through its view, its
    ``cu_seqlens`` slice, its tables and its tape slices; the token tensors always whole): the intra program (the
    GLM prep folded in), WY where the plan dispatches it, the state recurrence.  ``qg`` / ``w`` / ``u`` are the
    light saving forward's per-call tensors (version 4): the same writes as into the arena's slots, another
    destination, and every consumer of the call reads them there."""
    p = ws.plan
    values = _values(
        ws,
        views,
        tables,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        h=h,
        h2d=h.view(-1, D),
        v_new=v_new,
        aqk_ext=aqk_ext,
        akk_ext=akk_ext,
        out=out,
        scale=scale,
    )
    _saved_rows(values, p, qg=qg, w=w, u=u)
    _fwd_intra(
        ws,
        values,
        path=path,
        write_ext=write_ext,
        write_int=1 if p.internal_rows(path) else 0,
        write_prep=1,
    )
    if p.dispatched("wy_fwd", path):
        ws.run("wy_fwd", values, path=path)
    _forward_recurrence(
        ws, values, path=path, seg_state=seg_state, hseg=hseg, write_tapes=write_tapes
    )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float = SCALE,
    save_for_backward: bool = True,
    workspace: KdaWorkspace | None = None,
    save_mode: str | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> tuple[torch.Tensor, KdaChunkSaved | None]:
    """The forward: ``out`` BF16 like ``v`` (a fresh tensor) and the saved set (``None`` without
    ``save_for_backward``; fresh ``Aqk`` / ``Akk`` and tables and, with ``save_mode="light"`` (version 4), the
    fold's fresh ``qg``, on the segmented route when ``S >= 384 * NSEG`` the fresh segment-boundary states ``hseg``
    / ``seg_state`` the recurrence writes (shorter packs save neither and the backward recomputes them) and, on the
    rows whose light backward reads the WY representation from the set (:attr:`Plan.light_saves_wu`), the fresh
    ``w`` / ``u``; with ``save_mode="tape"`` the fresh ``h`` tape, ``v_new`` and the segment transfers instead).
    The per-call tensors replace arena slots as the destinations of the same writes: no launch is added.

    Launches (``Plan.launch_count("fwd")``): the table where the plan dispatches one, the intra-chunk program
    (which also writes the GLM prep's outputs), then on the serial route the state recurrence (which forms the
    output; ``wy_fwd`` before it below 32 heads: three or four launches), on the segmented route ``wy_fwd``, pass 1,
    ``seg_carry`` and pass 2 (five launches up to 256 documents, where the intra program ``fwd_intra_v14`` runs
    first and writes the chunk tables and the block table itself, no table launch; seven above, with ``table`` and
    ``seg_table`` before it); saving the states moves two output pointers of the recurrence from the arena to the
    saved tensors and adds no launch.  The token-row ``Aqk`` / ``Akk`` are written only with ``save_for_backward``
    (the ``fwd`` path; ``fwd_nosave`` without it): there the fused serial recurrence (``fwdh_wy_rows``), the overlap
    rows' WY and pass-2 programs (``wy_fwd_rows``, ``fwdh_p3_o_ov_rows`` / ``_id_rows``) and the paired serial rows' WY
    program and 32-column recurrence (``wy_fwd_rows``, ``fwdh_o_v2_rows``) read those rows and the intra program writes
    no internal copies (``write_int = 0``, :meth:`Plan.internal_rows`); every other forward keeps the internal-row
    programs and the copies.  The ``h`` / ``v_new`` tapes are written only in tape mode.  ``workspace``
    (:class:`KdaWorkspace`, the forward layout at least)
    is reused; without one the call allocates its arena and frees it on return (the plan and the layout views of
    a repeated problem are reused, so the host cost is close to a caller-owned workspace's: the call adds the
    arena's allocation and the eye's rebuild).  ``save_mode=None`` takes the workspace's
    default.  ``arena`` (:class:`KdaArena`) is the lower-memory arena option: the forward's layout is the same
    under every option (``None`` takes the workspace's; a differing option on a given workspace is refused).

    Under a batched option (``batch_tokens``) the call takes ``doc_lengths`` (the documents' token counts, host
    integers; checked against ``cu_seqlens`` on the device) and runs the launches above once per document group
    (:meth:`Plan.batches`) through a view of the arena sized for the group: the group's ``cu_seqlens`` slice, the
    group's tables (the intra program writes them, or the table program per group), the whole token tensors
    (absolute ``chunk_bos``), the group's slices of the tape and the group's state blocks.  With
    ``save_for_backward`` the whole problem's tables are written first by the saved-set paths' table program (the
    saved set carries them, as always), and the groups' saved states go into the whole problem's by chain rank
    (ranked on the host from ``doc_lengths`` by the tables' rule): where a group's blocks are one run of them, the
    carry writes its carried states straight into the saved ``hseg`` and its offsets / transfers are contiguous
    copies; otherwise they are scattered by index.  The outputs are bitwise the whole problem's (:class:`KdaArena`
    on the sign of exact zeros).
    """
    q, k, v, g, beta, S, N, H, batched = _operands(q, k, v, g, beta, cu_seqlens)
    with _workspace(
        workspace,
        S,
        N,
        H,
        v.device,
        backward=False,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
    ) as ws:
        out, saved = _forward(
            ws,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            save_for_backward=save_for_backward,
            save_mode=save_mode,
            doc_lengths=doc_lengths,
        )
    return _ranked(out, batched), saved


def _forward(
    ws: KdaWorkspace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    save_for_backward: bool,
    save_mode: str | None,
    doc_lengths: Sequence[int] | None,
) -> tuple[torch.Tensor, KdaChunkSaved | None]:
    """:func:`chunk_kda_fwd` on the call's workspace, the operands without their batch axis: ``out`` ``[S, H, 128]``
    and the saved set."""
    p = ws.plan
    S, N, H = p.S, p.N, p.H
    save_mode = ws.save_mode if save_mode is None else save_mode
    if save_mode not in SAVE_MODES:
        raise ValueError(
            f"chunk_kda_fwd: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
        )
    tape = save_mode == "tape" and save_for_backward
    # A tape-mode workspace sizes its arena for the tape layouts (h / v_new outside the arena), so a call on it
    # that hands no tape to a saved set allocates them per call; a light workspace holds them in the arena.  The
    # launch order and the operands are the same.
    tape_ws = (not tape) and ws.save_mode == "tape"
    mode = "fwd_tape" if (tape or tape_ws) else "fwd"
    seg = p.recur == "seg"
    # light (version 4): the segment-boundary states become per-call saved tensors when the size rule holds
    save_states = seg and save_for_backward and not tape and p.seg_save
    packing = ws._packing(doc_lengths)
    path = "fwd" if save_for_backward else "fwd_nosave"
    out = _alloc((S, H, D), torch.bfloat16, ws.device)
    aqk_ext = akk_ext = (
        None  # not written without a saved set (write_ext = 0): any BF16 tensor serves as the pointer
    )
    if save_for_backward:
        aqk_ext = _alloc((S, H, BT), torch.bfloat16, ws.device)
        akk_ext = _alloc((S, H, BT), torch.bfloat16, ws.device)
    # the whole problem's per-call tape (tape mode: the saved set owns h / v_new; a light call on a tape
    # workspace recomputes them into per-call buffers too) and its saved states (the carry's output and the
    # recurrence's offsets / transfers under the size rule; the tape's transfers)
    h = v_new = None
    if tape or tape_ws:
        h = _alloc((p.NC_pad, H, D, D), torch.bfloat16, ws.device)
        v_new = _alloc((p.R, H, D), torch.bfloat16, ws.device)
    # light (version 4): the fold's qg goes to a per-call saved tensor (the light backward's prep writes none),
    # and w / u too on the rows whose light backward reads the WY representation from the set (no wy launch there)
    light = save_for_backward and not tape
    qg_saved = w_saved = u_saved = None
    if light and p.light_saves_qg:
        qg_saved = _alloc((p.R, H, D), torch.bfloat16, ws.device)
    if light and p.light_saves_wu:
        w_saved = _alloc((p.R, H, D), torch.bfloat16, ws.device)
        u_saved = _alloc((p.R, H, D), torch.bfloat16, ws.device)
    # the groups' views of whole-problem row tensors (the tape, the saved qg / w / u) take the tape geometry
    row_slices = tape or qg_saved is not None or w_saved is not None
    seg_state = hseg = seg_m = None
    if save_states:
        seg_state = _alloc((2, p.NSEG, H, D, D), torch.float32, ws.device)
        hseg = _alloc((p.NSEG, H, D, D), torch.float32, ws.device)
    elif tape and seg:
        if packing is None:
            seg_state = _alloc((2, p.NSEG, H, D, D), torch.float32, ws.device)
            seg_m = seg_state[1]
        else:  # the groups' transfers come out of their arena blocks
            seg_m = _alloc((p.NSEG, H, D, D), torch.float32, ws.device)
    common: dict[str, Any] = {
        "path": path,
        "write_ext": 1 if save_for_backward else 0,
        "write_tapes": 1 if tape else 0,
        "out": out,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": float(scale),
    }
    if packing is None:
        a = ws.views(mode)
        tables = _tables(ws, cu_seqlens, path)
        _forward_launches(
            ws,
            a,
            cu_seqlens,
            tables,
            h=a.h if h is None else h,
            v_new=a.v_new if v_new is None else v_new,
            seg_state=(a.seg_state if seg_state is None else seg_state)
            if seg
            else None,
            hseg=(a.hseg if hseg is None else hseg) if seg else None,
            aqk_ext=a.aqk_int if aqk_ext is None else aqk_ext,
            akk_ext=a.aqk_int if akk_ext is None else akk_ext,
            qg=qg_saved,
            w=w_saved,
            u=u_saved,
            **common,
        )
    else:
        views_b = [
            (batch, ws.batch_view(batch, tape=row_slices)) for batch in packing.batches
        ]
        for _batch, ws_b in views_b:
            ws_b.views(
                mode
            )  # every group's layout fits the arena, or ValueError before any launch
        _assert_doc_lengths(cu_seqlens, packing.lengths)
        # the whole problem's tables for the saved set, by the saved-set paths' table program (the forward's intra
        # program writes the tables of its own call only: here the groups')
        tables = _tables(ws, cu_seqlens, "light") if save_for_backward else None
        # the groups' state blocks go into the whole problem's saved states by chain rank (from doc_lengths on the
        # host, :class:`_Packing`)
        scatter = seg and (save_states or seg_m is not None)
        for i, (batch, ws_b) in enumerate(views_b):
            pb = ws_b.plan
            a = ws_b.views(mode)
            cu_b = cu_seqlens[batch.d0 : batch.d1 + 1]
            tables_b = _tables(ws_b, cu_b, path)
            if tape:  # the group's slice of the whole tape (its pad chunks run into the next group's, written later)
                h_b = h[batch.c0 : batch.c0 + pb.NC_pad]
                v_new_b = v_new[BT * batch.c0 : BT * (batch.c0 + pb.NC_pad)]
            elif tape_ws:
                h_b = _alloc((pb.NC_pad, H, D, D), torch.bfloat16, ws.device)
                v_new_b = _alloc((pb.R, H, D), torch.bfloat16, ws.device)
            else:
                h_b, v_new_b = a.h, a.v_new
            # the group's slices of the whole problem's saved rows (the tape geometry: they end inside it)
            rows_b = slice(BT * batch.c0, BT * (batch.c0 + pb.NC_pad))
            # where the group's blocks are one run of the saved states, the carry writes the carried states straight
            # into the saved hseg (pass 2 reads them there); the offsets / transfers stay in the arena, since pass
            # 1 strides its two halves by the group's segment count
            first = packing.blocks[i] if scatter else None
            run = None if first is None else slice(first, first + pb.NSEG)
            _forward_launches(
                ws_b,
                a,
                cu_b,
                tables_b,
                h=h_b,
                v_new=v_new_b,
                seg_state=a.seg_state if seg else None,
                hseg=(hseg[run] if save_states and run is not None else a.hseg)
                if seg
                else None,
                aqk_ext=a.aqk_int if aqk_ext is None else aqk_ext,
                akk_ext=a.aqk_int if akk_ext is None else akk_ext,
                qg=None if qg_saved is None else qg_saved[rows_b],
                w=None if w_saved is None else w_saved[rows_b],
                u=None if u_saved is None else u_saved[rows_b],
                **common,
            )
            if run is not None:  # one run: contiguous copies of the halves
                if save_states:
                    seg_state[0, run].copy_(a.seg_state[0])
                    seg_state[1, run].copy_(a.seg_state[1])
                else:
                    seg_m[run].copy_(a.seg_state[1])
            elif scatter:  # the group's blocks scattered by index
                index = packing.block_index(i)
                if save_states:
                    hseg.index_copy_(0, index, a.hseg)
                    seg_state.index_copy_(1, index, a.seg_state)
                else:
                    seg_m.index_copy_(0, index, a.seg_state[1])
        if tape:  # the last group's slice ends inside the whole tape: no group writes the v_new pad rows after it
            last, ws_last = views_b[-1]
            v_new[BT * (last.c0 + ws_last.plan.NC_pad) :].zero_()
    saved = None
    if save_for_backward:
        if tape:
            saved = KdaChunkSaved(
                SAVED_VERSION_TAPE,
                aqk_ext,
                akk_ext,
                tables,
                S,
                H,
                N,
                float(scale),
                h=h,
                v_new=v_new,
                seg_m=seg_m,
            )
        else:
            saved = KdaChunkSaved(
                SAVED_VERSION,
                aqk_ext,
                akk_ext,
                tables,
                S,
                H,
                N,
                float(scale),
                hseg=hseg,
                seg_state=seg_state,
                qg=qg_saved,
                w=w_saved,
                u=u_saved,
            )
    return out, saved


def _backward_launches(
    ws: KdaWorkspace,
    views: types.SimpleNamespace,
    cu_seqlens: torch.Tensor,
    tables: tuple[torch.Tensor, ...],
    *,
    path: str,
    replay: bool,
    tape: bool,
    h: torch.Tensor,
    v_new: torch.Tensor,
    seg_state: torch.Tensor | None,
    hseg: torch.Tensor | None,
    seg_m: torch.Tensor | None,
    Aqk: torch.Tensor | None,
    Akk: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dout: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dg: torch.Tensor,
    dbeta: torch.Tensor,
    scale: float,
    qg: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    nc_vec: torch.Tensor | None = None,
) -> None:
    """The launches of one backward on ``ws`` (the whole problem, or one document group through its view, its
    ``cu_seqlens`` slice, its tables, its tape slices, its slices of the saved ``qg`` / ``w`` / ``u``, its blocks of
    the saved states and its tail counts ``nc_vec``; the token tensors, the saved ``Aqk`` / ``Akk`` and the
    gradients always whole): the replay's intra recompute, the prep, WY, the recurrence recompute, dA / dv, the
    adjoint recurrence and the tail (once, or per range of a lean workspace's ``tail_passes``, a group view's own),
    each where the path and the plan dispatch it.  ``seg_m`` is the forward's transfers where they do not come from
    ``seg_state[1]`` (a tape set's ``seg_m``, a group's blocks of the saved ``seg_state[1]``)."""
    p = ws.plan
    b = views
    # The operand sources of the backward programs: the saved Aqk / Akk token rows and the user's dout on the
    # saved-set paths (their records bind aqk_src / akk_src / dout), the arena's internal rows recomputed by the
    # replay (aqk_int / akk_int / do_int, the replay records' bindings).
    values = _values(
        ws,
        b,
        tables,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        dout=dout,
        h=h,
        h2d=h.view(-1, D),
        dh2d=b.dh.view(-1, D),
        v_new=v_new,
        aqk_src=b.aqk_int if replay else Aqk,
        akk_src=b.akk_int if replay else Akk,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dbeta=dbeta,
        scale=scale,
    )
    _saved_rows(values, p, qg=qg, w=w, u=u)

    # The replay recomputes the internal Aqk / Akk rows with the intra program (write_prep = 0: the prep buffers are
    # the replay prep's; the token-row pointers are unused, write_ext = 0): before the prep where the intra program
    # writes the tables the prep reads (the segmented rows up to 256 documents: no table launch), after it otherwise.
    def replay_intra() -> None:
        _fwd_intra(
            ws,
            {**values, "aqk_ext": b.akk_int, "akk_ext": b.akk_int},
            path=path,
            write_ext=0,
            write_int=1,
            write_prep=0,
        )

    intra_first = replay and not p.dispatched("table", path)
    if intra_first:
        replay_intra()
    # prep's pad-row zero-fill target: the saved v_new already carries its pads
    ws.run(
        "bwd_prep",
        {**values, "v_new": b.v_new_scratch if tape else v_new},
        path=path,
    )
    if replay and not intra_first:
        replay_intra()
    if p.dispatched("wy", path):
        ws.run("wy", values, path=path)
    if not tape:  # h / v_new come from the saved set in tape mode (wy still runs for w)
        _forward_recurrence(
            ws, values, path=path, seg_state=seg_state, hseg=hseg, write_tapes=1
        )
    if p.dispatched("dav", path):
        ws.run("dav", values, path=path)
    # the segmented dhu's reverse carry reads the forward's transfers: seg_m where given (the saved seg_m of a tape
    # set, a group's blocks of the saved transfers), else seg_state[1] (the saved one of a version-4 set with the
    # states, or the recompute's: the arena's)
    transfers = None
    if p.recur == "seg":
        transfers = (seg_m if seg_m is not None else seg_state[1]).view(-1, D, D)
    _adjoint_recurrence(ws, b, values, path=path, seg_m=transfers)
    if ws.tail_passes == 1:
        ws.run("dqkg", values, path=path)
        ws.run("glm_intra", values, path=path)
        if p.dispatched(
            "gate_epilogue", path
        ):  # not on the fused GLM intra program's architectures
            ws.run("gate_epilogue", values, path=path)
    else:
        _tail_passes(ws, b, values, tables, path=path, h=h, nc_vec=nc_vec)


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dout: torch.Tensor,
    *,
    saved: KdaChunkSaved | None = None,
    scale: float = SCALE,
    workspace: KdaWorkspace | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward: fresh ``(dq, dk, dv, dbeta, dg)`` ranked like the inputs.

    The path (:data:`cake_jit.PATHS`) follows from ``saved``: ``light`` from a version-4 set without the
    segment-boundary states (every serial problem; a segmented short pack: the full recurrence recompute),
    ``light_states`` from a version-4 set carrying them (the segmented recurrence as pass 2 alone from the saved
    ``hseg``, the adjoint carry reading the saved ``seg_state[1]``; both light paths read the saved ``qg``, and
    the rows whose set carries ``w`` / ``u`` launch no ``wy`` program), ``tape`` from a version-2 set (no recurrence
    recompute: dA / dv and the fused gradient read the saved ``h`` / ``v_new``, the segmented carry the saved
    ``seg_m``), ``replay`` with ``saved=None`` (the checkpoint replay: the table where the plan dispatches one,
    the prep and the intra program recompute the internal rows first; on the segmented rows up to 256 documents
    the intra program runs before the prep and writes the tables the prep reads).  The launches are the GLM prep,
    [intra], [WY], [the recurrence recompute], [dA / dv], the
    adjoint recurrence, the fused dq / dk / dg, GLM intra and [the gate epilogue], where the bracketed ones depend
    on the path and the row (``Plan.launch_count(path)``; the tables are rebuilt when the saved set carries
    none).  The backward writes nothing into the saved set.  ``workspace`` (:class:`KdaWorkspace` with the
    backward layout) is reused; without one the call allocates its arena and frees it on return, as the forward
    does.  ``arena``
    (:class:`KdaArena`): the lower-memory option of that per-call workspace (``None`` takes a given workspace's
    option; a differing one is refused); a lean per-call arena is sized for the saved set's version (the tape
    layouts for a version-2 set).  A lean workspace serves no replay (``saved=None``): the lean layouts and the
    tail split (``tail_passes``: steps 6-8 once per chunk range, :func:`_tail_passes`) are the saved-set paths'.

    Under a batched option (``batch_tokens``) the call takes ``doc_lengths`` (the documents' token counts, host
    integers; checked against ``cu_seqlens`` on the device) and runs the launches above once per document group
    (:meth:`Plan.batches`) through a view of the arena sized for the group: the group's ``cu_seqlens`` slice, the
    group's own tables (the table program per group: the saved set's tables describe the whole problem), the whole
    token tensors and gradients (absolute ``chunk_bos``), the group's slices of a saved tape, its state blocks
    taken from the saved states by chain rank (``light_states``, ``tape``: ranked on the host from ``doc_lengths``
    by the tables' rule; views where the group's blocks are one run of the saved ones, else gathered by index) or
    recomputed in its view (``light``, ``replay``: the replay runs group by group on a workspace that is not lean),
    and on a lean workspace its tail in the fewest passes, up to the option's, whose layout fits the arena
    (:func:`_group_tail_passes`) with their chunk counts from the host.  The gradients are bitwise the whole
    problem's (:class:`KdaArena` on the sign of exact zeros).
    """
    q, k, v, g, beta, S, N, H, batched = _operands(q, k, v, g, beta, cu_seqlens)
    dout = _squeeze_batch(dout, "dout", 3)
    _check(dout, "dout", (S, H, D), torch.bfloat16, v.device)
    option = (
        arena
        if arena is not None
        else (KdaArena() if workspace is None else workspace.arena_option)
    )
    if saved is None and option.lean:
        raise ValueError(
            "chunk_kda_bwd: a lean arena (KdaArena(lean=True"
            + (", batch_tokens=...)" if option.batched else ")")
            + ") serves no checkpoint replay (saved=None): pass the saved set, or run the replay on a workspace "
            "with lean=False"
            + (
                " (the batched option alone runs it group by group)"
                if option.batched
                else ""
            )
        )
    lean_tape = (
        option.lean and saved is not None and saved.version == SAVED_VERSION_TAPE
    )
    with _workspace(
        workspace,
        S,
        N,
        H,
        v.device,
        backward=True,
        save_mode="tape" if lean_tape else None,
        arena=arena,
        doc_lengths=doc_lengths,
    ) as ws:
        grads = _backward(
            ws,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            dout,
            saved=saved,
            scale=scale,
            doc_lengths=doc_lengths,
        )
    return tuple(_ranked(x, batched) for x in grads)


def _backward(
    ws: KdaWorkspace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dout: torch.Tensor,
    *,
    saved: KdaChunkSaved | None,
    scale: float,
    doc_lengths: Sequence[int] | None,
) -> tuple[torch.Tensor, ...]:
    """:func:`chunk_kda_bwd` on the call's workspace, the operands without their batch axis: ``(dq, dk, dv, dbeta,
    dg)``."""
    p = ws.plan
    S, N, H = p.S, p.N, p.H
    seg = p.recur == "seg"
    replay = saved is None
    tape = False
    if saved is not None:
        if saved.version not in (SAVED_VERSION, SAVED_VERSION_TAPE):
            raise ValueError(
                f"saved: version {saved.version}, this backend reads versions {SAVED_VERSION} (light) and "
                f"{SAVED_VERSION_TAPE} (tape); versions 1 and 3 (the light sets without qg) are no longer "
                "accepted"
            )
        if (saved.S, saved.H, saved.num_docs) != (S, H, N):
            raise ValueError(
                f"saved: for S={saved.S}, H={saved.H}, num_docs={saved.num_docs}; the operands are S={S}, "
                f"H={H}, num_docs={N}"
            )
        _check(saved.Aqk, "saved.Aqk", (S, H, BT), torch.bfloat16, ws.device)
        _check(saved.Akk, "saved.Akk", (S, H, BT), torch.bfloat16, ws.device)
        if saved.tables is not None:
            _check_tables(saved, ws)
        tape = saved.version == SAVED_VERSION_TAPE
        if tape:
            if saved.h is None or saved.v_new is None:
                raise ValueError("saved: a version-2 saved set needs h and v_new")
            _check(saved.h, "saved.h", (p.NC_pad, H, D, D), torch.bfloat16, ws.device)
            _check(saved.v_new, "saved.v_new", (p.R, H, D), torch.bfloat16, ws.device)
            if seg:
                if saved.seg_m is None:
                    raise ValueError(
                        "saved: a version-2 saved set of the segmented route needs seg_m"
                    )
                _check(
                    saved.seg_m,
                    "saved.seg_m",
                    (p.NSEG, H, D, D),
                    torch.float32,
                    ws.device,
                )
        elif saved.hseg is not None or saved.seg_state is not None:
            # a version-4 set carrying the segment-boundary states
            if not seg:
                raise ValueError(
                    "saved: a version-4 saved set carries segment-boundary states, but this problem runs the "
                    "serial recurrence (no segments)"
                )
            if saved.hseg is None or saved.seg_state is None:
                raise ValueError(
                    "saved: a version-4 saved set carries hseg and seg_state together or neither"
                )
            _check(
                saved.hseg, "saved.hseg", (p.NSEG, H, D, D), torch.float32, ws.device
            )
            _check(
                saved.seg_state,
                "saved.seg_state",
                (2, p.NSEG, H, D, D),
                torch.float32,
                ws.device,
            )
        if not tape:
            # a version-4 light set: the fold's qg where the light prep writes none; w / u together, exactly on
            # the rows whose light backward reads them from the set
            if p.light_saves_qg:
                if saved.qg is None:
                    raise ValueError(
                        "saved: a version-4 light saved set carries the forward's qg (the light backward's prep "
                        "writes none); this set has none"
                    )
                _check(saved.qg, "saved.qg", (p.R, H, D), torch.bfloat16, ws.device)
            if (saved.w is None) != (saved.u is None):
                raise ValueError(
                    "saved: a version-4 saved set carries w and u together or neither"
                )
            if (saved.w is not None) != p.light_saves_wu:
                raise ValueError(
                    "saved: this problem's light backward "
                    + (
                        "reads the WY representation (w / u) from the saved set, which carries none"
                        if p.light_saves_wu
                        else "recomputes the WY representation; the saved set carries w / u (another "
                        "architecture's set?)"
                    )
                )
            if saved.w is not None:
                _check(saved.w, "saved.w", (p.R, H, D), torch.bfloat16, ws.device)
                _check(saved.u, "saved.u", (p.R, H, D), torch.bfloat16, ws.device)
    # The segmented route's state sources: a version-4 saved set carrying the states (the recurrence as pass 2
    # alone) or the arena's blocks (the full recompute: a set without the states, the replay); the tape backward
    # runs no recurrence and its carry reads the saved seg_m.
    light_states = seg and not replay and not tape and saved.hseg is not None
    if replay:
        path = "replay"
    elif tape:
        path = "tape"
    else:
        path = "light_states" if light_states else "light"
    # A tape-mode workspace (arena sized for the tape layouts) recomputes h / v_new into per-call buffers
    # whenever the saved set does not carry them (a version-4 saved set, the replay), as its forward does; a
    # light one recomputes them inside the arena.  The launch order and the operands are the same.  The replay
    # layouts keep the internal akk_int rows the saved-set paths do without.
    tape_ws = (not tape) and ws.save_mode == "tape"
    mode = "bwd_tape" if (tape or tape_ws) else "bwd"
    if replay:
        mode = {"bwd": "replay", "bwd_tape": "replay_tape"}[mode]
    packing = ws._packing(doc_lengths)
    dq = _alloc((S, H, D), torch.bfloat16, ws.device)
    dk = _alloc((S, H, D), torch.bfloat16, ws.device)
    dv = _alloc((S, H, D), torch.bfloat16, ws.device)
    dg = _alloc((S, H, D), torch.float32, ws.device)
    dbeta = _alloc((S, H), torch.bfloat16, ws.device)
    # the version-4 rows the light paths read from the set (the groups of a batched option take their slices)
    light_rows = not (replay or tape)
    qg_saved = saved.qg if light_rows else None
    w_saved = saved.w if light_rows else None
    u_saved = saved.u if light_rows else None
    common: dict[str, Any] = {
        "path": path,
        "replay": replay,
        "tape": tape,
        "Aqk": None if replay else saved.Aqk,
        "Akk": None if replay else saved.Akk,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "dout": dout,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": dg,
        "dbeta": dbeta,
        "scale": float(scale),
    }
    if packing is None:
        tables = (
            _tables(ws, cu_seqlens, path)
            if (replay or saved.tables is None)
            else saved.tables
        )
        b = ws.views(mode)
        if tape:
            h, v_new = saved.h, saved.v_new
        elif tape_ws:
            h = _alloc((p.NC_pad, H, D, D), torch.bfloat16, ws.device)
            v_new = _alloc((p.R, H, D), torch.bfloat16, ws.device)
        else:
            h, v_new = b.h, b.v_new
        _backward_launches(
            ws,
            b,
            cu_seqlens,
            tables,
            h=h,
            v_new=v_new,
            seg_state=(saved.seg_state if light_states else b.seg_state)
            if seg
            else None,
            hseg=(saved.hseg if light_states else b.hseg) if seg else None,
            seg_m=saved.seg_m if (tape and seg) else None,
            qg=qg_saved,
            w=w_saved,
            u=u_saved,
            **common,
        )
    else:
        # the groups' views of whole-problem row tensors (the saved tape, the saved qg / w / u) take the tape geometry
        row_slices = tape or qg_saved is not None or w_saved is not None
        views_b = [
            (batch, ws.batch_view(batch, tape=row_slices)) for batch in packing.batches
        ]
        for _batch, ws_b in views_b:
            ws_b.views(
                mode
            )  # every group's layout fits the arena, or ValueError before any launch
        _assert_doc_lengths(cu_seqlens, packing.lengths)
        # the groups build their own tables; their blocks of the saved states come by chain rank (from doc_lengths
        # on the host, :class:`_Packing`)
        from_saved = seg and not replay and (tape or light_states)
        for i, (batch, ws_b) in enumerate(views_b):
            pb = ws_b.plan
            b = ws_b.views(mode)
            cu_b = cu_seqlens[batch.d0 : batch.d1 + 1]
            tables_b = _tables(ws_b, cu_b, path)
            if tape:  # the group's slice of the saved tape
                h_b = saved.h[batch.c0 : batch.c0 + pb.NC_pad]
                v_new_b = saved.v_new[BT * batch.c0 : BT * (batch.c0 + pb.NC_pad)]
            elif tape_ws:
                h_b = _alloc((pb.NC_pad, H, D, D), torch.bfloat16, ws.device)
                v_new_b = _alloc((pb.R, H, D), torch.bfloat16, ws.device)
            else:
                h_b, v_new_b = b.h, b.v_new
            seg_state_b = b.seg_state if seg else None
            hseg_b = b.hseg if seg else None
            seg_m_b = None
            if from_saved:  # the group's saved states: views where its blocks are one run, else gathered by index
                if tape:
                    seg_m_b = packing.take(i, saved.seg_m)
                else:
                    # pass 2 reads hseg alone and never touches its seg_state at write_ht = 0 (the arena's block
                    # serves there); the reverse carry reads the transfers as seg_m
                    hseg_b = packing.take(i, saved.hseg)
                    seg_m_b = packing.take(i, saved.seg_state[1])
            rows_b = slice(BT * batch.c0, BT * (batch.c0 + pb.NC_pad))
            _backward_launches(
                ws_b,
                b,
                cu_b,
                tables_b,
                h=h_b,
                v_new=v_new_b,
                seg_state=seg_state_b,
                hseg=hseg_b,
                seg_m=seg_m_b,
                qg=None if qg_saved is None else qg_saved[rows_b],
                w=None if w_saved is None else w_saved[rows_b],
                u=None if u_saved is None else u_saved[rows_b],
                nc_vec=packing.tail_counts(i, pb.NC_pad),
                **common,
            )
    return dq, dk, dv, dbeta, dg


# ----------------------------------------------------------------------------------------------------------------
# autograd


# The per-version extra saved tensors ChunkKDAFunction hands to autograd, in this order (absent ones are dropped
# and the presence mask travels in the context: the version-4 fields are optional independently -- qg where the
# light prep writes none, the states under the size rule, w / u on their rows).
_EXTRA_FIELDS = {
    SAVED_VERSION: ("qg", "hseg", "seg_state", "w", "u"),
    SAVED_VERSION_TAPE: ("h", "v_new", "seg_m"),
}


def _saved_extra(
    saved: KdaChunkSaved,
) -> tuple[tuple[torch.Tensor, ...], tuple[str, ...]]:
    """The present extra tensors of ``saved`` and their field names, in :data:`_EXTRA_FIELDS` order."""
    pairs = [
        (name, getattr(saved, name))
        for name in _EXTRA_FIELDS[saved.version]
        if getattr(saved, name) is not None
    ]
    return tuple(t for _name, t in pairs), tuple(name for name, _t in pairs)


class ChunkKDAFunction(torch.autograd.Function):
    """:func:`chunk_kda` as an autograd node: the forward saves the inputs and the saved set, the backward runs
    :func:`chunk_kda_bwd` from them.  Under ``torch.utils.checkpoint`` (``use_reentrant=False``) the forward is
    re-run in the backward and the saved set recomputed; no special casing.  The node keeps nothing but the saved
    tensors alive: the caller's workspace is held through a weak reference, so a workspace the caller dropped
    after the forward (a per-call one, or any workspace of a checkpointed forward whose frame is gone) is freed,
    and the backward then allocates its own (:func:`chunk_kda_bwd` without ``workspace``)."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        workspace,
        save_mode,
        arena,
        doc_lengths,
    ):
        out, saved = chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            save_for_backward=True,
            workspace=workspace,
            save_mode=save_mode,
            arena=arena,
            doc_lengths=doc_lengths,
        )
        extra, ctx.extra_fields = _saved_extra(saved)
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            saved.Aqk,
            saved.Akk,
            *saved.tables,
            *extra,
        )
        ctx.scale = scale
        # a weak reference: the node must not keep a per-call arena alive until the backward (under activation
        # checkpointing that would be one arena per layer)
        ctx.workspace_ref = None if workspace is None else weakref.ref(workspace)
        # the arena option of the backward's own workspace when the caller's is gone (KdaArena holds ints only)
        ctx.arena = (
            arena
            if arena is not None
            else (None if workspace is None else workspace.arena_option)
        )
        # the batched option's document groups (host ints) for the backward's own calls
        ctx.doc_lengths = (
            None if doc_lengths is None else tuple(int(x) for x in doc_lengths)
        )
        ctx.version, ctx.n_tables = saved.version, len(saved.tables)
        ctx.problem = (saved.S, saved.H, saved.num_docs)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, cu_seqlens, Aqk, Akk, *rest = ctx.saved_tensors
        tables, extra = tuple(rest[: ctx.n_tables]), rest[ctx.n_tables :]
        saved = KdaChunkSaved(
            ctx.version,
            Aqk,
            Akk,
            tables,
            *ctx.problem,
            ctx.scale,
            **dict(zip(ctx.extra_fields, extra, strict=True)),
        )
        if not dout.is_contiguous():
            dout = (
                dout.contiguous()
            )  # autograd may hand over an expanded or permuted gradient
        workspace = None if ctx.workspace_ref is None else ctx.workspace_ref()
        dq, dk, dv, dbeta, dg = chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            dout,
            saved=saved,
            scale=ctx.scale,
            workspace=workspace,
            arena=None if workspace is not None else ctx.arena,
            doc_lengths=ctx.doc_lengths,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None, None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float = SCALE,
    workspace: KdaWorkspace | None = None,
    save_mode: str | None = None,
    arena: KdaArena | None = None,
    doc_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """The autograd entry: ``out`` with gradients to ``q, k, v, g, beta``.

    With a ``workspace`` the forward and the backward both run in it (the backward overwrites the forward's
    intermediates, not its saved set, which autograd holds) as long as the caller keeps it: the autograd node
    holds it weakly, so a workspace created for the call and dropped after the forward is freed at once, and the
    backward allocates its own.  Without one the forward and the backward each allocate an arena for the call and
    free it on return (not zero-filled); a repeated problem reuses its plan and layout views, so the host cost per
    call is close to a caller-owned workspace's (the call adds the arena's allocation and the eye's rebuild), and
    no option (``arena``, ``doc_lengths``) makes the host wait for the device.  After a forward under ``torch.utils.checkpoint``
    (``use_reentrant=False``) the node keeps no arena and no saved tensor alive.  ``save_mode`` (``None``: the
    workspace's default, ``"light"`` without one) selects the saved set's version.  ``arena`` (:class:`KdaArena`,
    ``None``: the workspace's option, off without one) is the lower-memory arena option; the node records it, so
    the backward's own workspace (a per-call or checkpointed forward) takes the same option.  ``doc_lengths``
    (the documents' token counts, host integers) is required under a batched option and recorded by the node.
    """
    return ChunkKDAFunction.apply(
        q, k, v, g, beta, cu_seqlens, scale, workspace, save_mode, arena, doc_lengths
    )


def is_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is an exported target."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = _device_index(torch.device(device))
    return (
        cake_jit.arch_for_capability(torch.cuda.get_device_capability(index))
        is not None
    )


__all__ = [
    "ALIGN",
    "BT",
    "BWD_SLOTS",
    "BWD_TAPE_SLOTS",
    "D",
    "FWD_SLOTS",
    "FWD_TAPE_SLOTS",
    "KIND_SHAPE",
    "LEAN_MODES",
    "MAX_DOCS",
    "MODES",
    "PASS_KINDS",
    "SAVED_VERSION",
    "SAVED_VERSION_TAPE",
    "SAVE_MODES",
    "SCALE",
    "SEG_SAVE_MIN_TOKENS_PER_SEG",
    "SLOTS",
    "TABLE_NAMES",
    "ChunkKDAFunction",
    "KdaArena",
    "KdaChunkSaved",
    "KdaWorkspace",
    "Launcher",
    "Plan",
    "Region",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "is_supported",
    "lifetime_table",
    "memory_report",
    "output_bytes",
    "plan",
    "saved_bytes",
    "seg_states_saved",
    "tables_bytes",
    "workspace_bytes",
]
