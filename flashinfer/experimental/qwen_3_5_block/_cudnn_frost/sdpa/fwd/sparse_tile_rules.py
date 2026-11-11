# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SM107 index-list sparse decode's host rules: the ``tile_form`` vocabulary, the lean tile's host arithmetic and the placement /
decline rules between its two bodies.  DSL-free and import-light on purpose: the gated attention block validates and resolves a
declaration through these without importing the heuristics or the configs (which pull ``cudnn.frost.tile_dsl``); ``heuristics``,
``config_sm107`` and the sparse adapter re-export every name here unchanged."""

from __future__ import annotations

from typing import Optional, Tuple

_SPARSE_BLOCK_SIZE = 4  # tokens per block = the four rows of ONE tma_gather4 (config_sm107's own copy is the kernel's)
LEAN_SPARSE_N_KEYS = 32  # keys per sub-tile = 8 four-token blocks = one 32-bit valid word
LEAN_SPARSE_MAX_SEQLEN_Q = 4  # the serving step's rows per sequence under ONE shared list (the sparse function's own S_q <= 4 bound)
LEAN_SPARSE_MAX_SUB_TILES_PER_CTA = 8  # the sub-tile loop is unrolled at trace time (cutlass.range_constexpr); the fill rule never exceeds 7 (B 32)
LEAN_SPARSE_TOPK_MIN = 8  # top_k < 8 is ONE sub-tile = split 1: the tcgen05 tile's cell (the adapter routes it there)
LEAN_SPARSE_SERVED_MAX_S_Q = LEAN_SPARSE_MAX_SEQLEN_Q  # the step rows per sequence the ROUTE serves on the lean body = the kernel's own S_q bound (4: N_MT = ceil(S_q x G / 16) m16 tiles over ONE gathered sub-tile); the adapter and the heuristics read THIS constant, never a literal
_LEAN_N_TAIL_MAX = 2  # the S_q <= 4 bound on the appended tail blocks (the same host bound at every S_q, so the S_q axis changes no grid)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def lean_sparse_n_sub(top_k: int, block_size: int = _SPARSE_BLOCK_SIZE, n_keys: int = LEAN_SPARSE_N_KEYS) -> int:
    """The HOST bound on a (sequence, KV head)'s 32-key sub-tiles: ``ceil((top_k + n_tail_max) x block_size / n_keys)`` = 65 at
    top_k 512 (the device's real ``count + n_tail`` is smaller: the chunks past it are DEAD items that write the combine's identity)."""
    return -(-(int(top_k) + _LEAN_N_TAIL_MAX) * int(block_size) // int(n_keys))


def lean_sparse_split(top_k: int, sub_tiles_per_cta: int) -> int:
    """The lean tile's split = the chunk count ``ceil(n_sub / k)``: the grid's x extent and the combine's partial count (65 / 33 / 22 / 17 at
    top_k 512 for k = 1 / 2 / 3 / 4)."""
    return -(-lean_sparse_n_sub(top_k) // int(sub_tiles_per_cta))


# ---------------------------------------------------------------------------------------------------------------------------------
# The LEAN block-sparse DECODE tile (sm107/sparse_lean_d256_f16.py: mma.sync + cp.async, 128 threads per m16 tile, no TMEM / TMA /
# CLC) beside the tcgen05 index-list sparse core (sm107/sparse_d256_f16.py) -- the SECOND BODY of ONE function.  The sparse adapter's
# ``tile_form`` knob ("auto" / "lean" / "tcgen05") is PERFORMANCE-ONLY: the same O / LSE under either body within the fixed-order-sum
# budget (a 4 x 64-d split-K sum vs one 256-d MMA chain: never bitwise ACROSS the bodies, bitwise WITHIN each), so an autotuner may
# pick freely.  The rules below are the host reading of WHICH body a launch takes and HOW the lean body is cut; the adapter is the
# enforcement point and reads them, the gated attention block's block-sparse decode mode reads them through the adapter's vocabulary.
SPARSE_TILE_FORM_AUTO = "auto"
SPARSE_TILE_FORM_LEAN = "lean"
SPARSE_TILE_FORM_TCGEN05 = "tcgen05"
SPARSE_TILE_FORMS = (SPARSE_TILE_FORM_AUTO, SPARSE_TILE_FORM_LEAN, SPARSE_TILE_FORM_TCGEN05)
"""The ``tile_form`` vocabulary: ``"auto"`` = the placement rule on the parts of :data:`SPARSE_LEAN_TILE_DEFAULT_CCS` and the tcgen05
core everywhere else; ``"lean"`` / ``"tcgen05"`` explicit -- honoured where the body serves the declaration, a typed decline naming the
arm where it does not (never a silent fallback)."""
SPARSE_TILE_TCGEN05 = "sparse_d256_f16"
SPARSE_TILE_LEAN = "sparse_lean_d256_f16"
"""The two bodies' ``kernel_template`` names (the kernel modules' file stems under ``sdpa/fwd/kernels/sm107/``): what the adapter's
``kernel_template`` and the block's ``sparse_tile`` read back, so a route is PINNED by the kernel that compiled, never transcribed."""
LEAN_SPARSE_MAX_UNITS = 4
"""The PLACEMENT bound, in CTAs PER CHUNK: the lean tile below ``B x H_kv <= 4`` (B <= 2 on two KV heads, at EVERY served step width
S_q 1..4), the tcgen05 core above.  The unit is the lean grid's own: one CTA per (sequence, KV head, chunk of k sub-tiles) of
``128 x N_MT`` threads, the step's S_q rows riding INSIDE the CTA as ``N_MT = ceil(S_q x G / 16)`` m16 tiles of four warps each
(128 threads at S_q 1 / G 12, 512 at S_q 4 / G 16; ``config_sm107.make_cfg_d256_sparse_lean`` pins ``THREADS``) -- the grid is
``B x H_kv x split``, so S_q sizes the CTA, not the grid, and is NOT in the bound (the served rows are :func:`sparse_lean_tile_decline`'s business:
``S_q <= 4`` under one shared list).  The lean tile's advantage is the ABSENCE of the tcgen05 core's ~4.5-us per-CTA chain on a launch that is
latency-bound (one wave of small CTAs: the 212-SM class measured the kernel alone 3.34 vs 5.66 us at B 1 x 8K x S_q 1, +69 %, and the
chain 8.32 -> 5.75 us); it shrinks as the launch becomes HBM-bound, where both tiles move the same bytes and the lean form's shared
gather forces a split whose combine on a 768-block grid costs more than the tcgen05 core's lean walk at split 4 (B 32 x S_q 1: a
modeled -2 %).  The model put the crossover between 32 and 64 CTAs per chunk; the gate's bound sweep (B in 1 / 2 / 4 / 8 / 16 / 32 x S_q 1
at 8K / 32K, the S_q 4 shared-list cells at B 1 / 4 / 32, every cell the lean body vs the tcgen05 core in the same process) moved this
ONE constant to the last class that wins inside its control band, and its S_q 4 rows fix the UNIT: at the same CTAs per chunk the S_q 1
and the S_q 4 cells land on the same side (2 per chunk: B 1 wins at S_q 1 AND at S_q 4; 8 per chunk: B 4 loses at S_q 1 AND at S_q 4),
where a ``B x H_kv x S_q`` count (the first merged form of this bound, 2026-10-26) put the winning B 1 x S_q 3..4 cells on the core's side.
The gate's table by CTAs per chunk (the lean lever per step, POSITIVE = the lean body faster): 2 per chunk: B 1 x S_q 1 (the e2e
block) +6.8 % / +5.7 % / +4.8 % at 8K / 32K / 128K and B 1 x S_q 4 (the shared-list block) +6.4 % / +6.6 % / +6.8 % at 8K / 32K /
128K; 4 per chunk: B 2 x S_q 1 (the e2e block) +1.8 % / +1.3 % at 8K / 32K; 8 per chunk: B 4 x S_q 1 (the e2e block) -2.6 % /
-3.5 % / -2.0 % / -2.2 % at 2K / 8K / 32K / 128K and B 4 x S_q 4 (the shared-list block) -9.7 % / -8.8 % / -9.2 % at 8K / 32K /
128K; 16 per chunk: B 8 x S_q 1 (the e2e block) -17.2 % / -16.3 % at 8K / 32K; 32 per chunk: B 16 x S_q 1 (the e2e block) -44.1 % /
-43.6 % at 8K / 32K; 64 per chunk: B 32 x S_q 1 (the e2e block) -60.8 % / -58.7 % / -51.3 % and B 32 x S_q 4 (the shared-list
block) -36.4 % / -36.5 % / -35.7 % at 8K / 32K / 128K; a route class losing beyond its control band excludes every level from it up
-> 4."""
SPARSE_LEAN_TILE_DEFAULT_CCS: Tuple[Tuple[int, int], ...] = ((10, 7),)
"""The parts on which ``tile_form="auto"`` consults the placement rule (:func:`sparse_decode_lean_tile_selected`) at all: cc 10.7,
the one line the lean tile exists for, since the end-to-end gate measured the lean route against the tcgen05 route there (the
measured-default rule of the block's ``DECODE_PDL_DEFAULT_CCS``: a default ships with its table, in the change that measures it; the bar:
>= 1.0 us AND >= 5 % of the step at B 1 x 8K and B 1 x 32K on the row of record and on the symmetric reading with the arms built in the
other order, and no route class losing beyond its control band -- a losing class lowers :data:`LEAN_SPARSE_MAX_UNITS` to the last class
that wins).
MEASURED 2026-10-26 on an exclusive 212-SM cc 10.7 part at the 2376 MHz lock (CUDA-graph replay, one process per cell with every
arm, A/B/A x 3 x 100, the control pair within 1 %, every oracle before timing): the lean body vs the tcgen05 core in the same
block, B 1 x 8K x S_q 1 34.8 -> 32.5 us per decode step (+6.8 %, 2.22 us; +5.9 % / 1.93 us symmetric with the arms built in the
other order) and B 1 x 32K 36.6 -> 34.6 us (+5.7 %, 1.98 us; +5.6 % symmetric); B 1 x S_q 1 geomean +5.8 %, B 4 x S_q 1 -2.6 %, B
32 x S_q 1 -57.1 % past the bound; the shared-list caller block at S_q 4 (one list per sequence) B 1 +6.6 %, B 4 -9.2 %;
route-class cells losing beyond their band: B 4 x 2K x S_q 1 -2.6 %; B 4 x 8K x S_q 1 -3.5 %; B 4 x 32K x S_q 1 -2.0 %; B 4 x 128K
x S_q 1 -2.2 %; B 4 x 8K x S_q 4 -9.7 %; B 4 x 32K x S_q 4 -8.8 %; B 4 x 128K x S_q 4 -9.2 %; B 8 x 8K x S_q 1 -17.2 %; B 8 x 32K
x S_q 1 -16.3 %; B 16 x 8K x S_q 1 -44.1 %; B 16 x 32K x S_q 1 -43.6 % -> LEAN_SPARSE_MAX_UNITS 32 -> 4 (counted over B x H_kv x S_q
in the gate commit; the unit is the CTAs per chunk, B x H_kv, since the follow-up that read the same table's S_q 4 rows: the B 1 x S_q 4
win stays on the default route, the B 4 x S_q 4 loss off it).
Every other part keeps the tcgen05 core under ``"auto"`` until measured there; the explicit ``"tcgen05"`` keeps the former function on
the measured part, the explicit ``"lean"`` takes the lean tile wherever the body serves the declaration (past the bound too, and -- for
measurement -- on the Blackwell parts of :data:`SPARSE_LEAN_TILE_EXPLICIT_CCS` outside the core's record)."""
_SPARSE_LEAN_TILE_RULE_CC = (10, 7)
"""The ONE device line the lean tile was MEASURED on (the kernel's home: its traces and SASS pins are ``sm_107a``'s): the placement rule
answers False on every other part whatever the tuple above holds.  The explicit admission on the Blackwell parts is
:data:`SPARSE_LEAN_TILE_EXPLICIT_CCS`'s (below), which this rule never reads."""
SPARSE_LEAN_TILE_EXPLICIT_CCS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3))
"""The parts OUTSIDE the sparse core's record (``SparseCapabilities.sm_lo .. sm_hi`` = cc 10.7 .. 11.9) on which an EXPLICIT
``tile_form="lean"`` is admitted FOR MEASUREMENT: the Blackwell line, cc 10.0 / 10.3 (B200 / GB200, B300).  The lean body is SM100-legal
instruction for instruction (``cp.async``, ``ldmatrix``, ``mma.sync`` m16n8k16, ``bar.sync``, ``griddepcontrol``; no tcgen05 / TMA /
mbarrier / cluster / setmaxnreg), its config pins no Rubin-only size (40 KiB SMEM at k 1 under the standard carveout, 128 x N_MT threads,
REG <= 79 at k 1 on the sm_107a trace), the split arithmetic reads the device's SM count and the DSL targets the ambient device
(``sm_100a`` / ``sm_103a``; the compiled-plan cache keys on the device's compute capability), so the admission is a HOST rule, not a kernel
change; the sm100 split combine reduces the slab with the gate as it does for the dense d256 decode tile on those parts -- a lean plan's
combine FORM there is :func:`sparse_lean_combine_rule_cc`'s: on cc 10.0 the cc 10.7 split-aware rule's shape (the decode-grid kernel on the
24-96-row grids at the lean splits; :data:`SPARSE_LEAN_COMBINE_RULE_CCS`, measured 2026-10-28), on cc 10.3 the module's own forms
(unmeasured).  The AUTO default is UNMEASURED on these parts (2026-10-28: the explicit form admitted so the route can be measured on a
B200-class part): the AUTO reading keeps :data:`SPARSE_LEAN_TILE_DEFAULT_CCS` (cc 10.7) and the placement rule keeps
:data:`_SPARSE_LEAN_TILE_RULE_CC`, so ``tile_form="auto"`` / ``"tcgen05"`` on these parts resolve the tcgen05 core, which the adapter
declines by name (:func:`sparse_tile_part_decline`: 'explicit lean tile only, unmeasured').  A measurement moves a part into the DEFAULT
tuple with its table (the measured-default rule), never a transcription."""


def sparse_tile_part_decline(
    *,
    device_cc: Optional[Tuple[int, int]],
    tile_form_requested: str,
    sm_lo: int,
    sm_hi: int,
    explicit_lean_ccs: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Optional[str]:
    """Why the PART cannot serve the declared body (the arm NAMED), or ``None`` when it can -- the adapter's arch gate (after the gather4
    part floor, before the shapes), the ONE statement the gated attention block's sparse stage hears through the adapter.  ``sm_lo ..
    sm_hi`` is the sparse core's record (cc 10.7 .. 11.9: both bodies); on a part of :data:`SPARSE_LEAN_TILE_EXPLICIT_CCS` only the
    explicit ``"lean"`` is served (``"auto"`` resolves the tcgen05 core off the measured part, ``"tcgen05"`` names it: both declined with
    the explicit-lean-only text); every other part is declined with the former statement's spelling (``needs a Rubin-line GPU (cc 10.7 ..
    11.9); got cc X.Y``), which the explicit-lean-only text ends with too.  ``device_cc`` None (no reading) declines nothing here.
    ``explicit_lean_ccs`` (appended) is the part set the caller's RECORD states (``SparseCapabilities.lean_explicit_ccs``: the adapter passes
    its record's copy, which the test suite pins equal to :data:`SPARSE_LEAN_TILE_EXPLICIT_CCS`); ``None`` reads this module's constant."""
    if device_cc is None:
        return None
    cc = (int(device_cc[0]), int(device_cc[1]))
    cc_code = cc[0] * 10 + cc[1]
    if int(sm_lo) <= cc_code <= int(sm_hi):
        return None
    lo, hi = divmod(int(sm_lo), 10), divmod(int(sm_hi), 10)
    core = f"needs a Rubin-line GPU (cc {lo[0]}.{lo[1]} .. {hi[0]}.{hi[1]}); got cc {cc[0]}.{cc[1]}"
    explicit = SPARSE_LEAN_TILE_EXPLICIT_CCS if explicit_lean_ccs is None else tuple((int(c[0]), int(c[1])) for c in explicit_lean_ccs)
    if cc in explicit:
        if tile_form_requested == SPARSE_TILE_FORM_LEAN:
            return None
        return (
            f"cc {cc[0]}.{cc[1]} serves the EXPLICIT lean tile only (tile_form='lean': admitted for measurement, UNMEASURED there -- the AUTO "
            f"reading stays the measured part's, cc 10.7); tile_form={tile_form_requested!r} resolves the tcgen05 core, which {core}"
        )
    return core


SPARSE_LEAN_COMBINE_RULE_CCS: Tuple[Tuple[int, int], ...] = ((10, 0),)
"""The parts OUTSIDE the cc 10.7 line on which a LEAN plan's split-combine FORM follows the cc 10.7 split-aware rule's SHAPE
(``heuristics.decode_combine_grid_selected`` evaluated at :data:`_SPARSE_LEAN_COMBINE_RULE_SOURCE_CC` through
:func:`sparse_lean_combine_rule_cc`): the decode-grid kernel inside the combine module's bound on a small grid (<= 128 rows) from split 8
and on a larger grid from split 32, the module's own forms otherwise; never the lean group walk (a cc 10.7 form; the lean splits are
>= 9 anyway).  MEASURED 2026-10-28 on the GB200 class (cc 10.0, 152 SMs at the 1830 MHz lock; CUPTI medians over 50 cold-L2 replays of
the lean chain's combine at 24 Q heads, d 256, two independent GPU-0 cells agreeing to <= 0.1 us; the oracle green before timing):
the decode-grid kernel vs the module's own forms, us, at the lean splits 65 / 33 / 22 / 17 -- B 1 x S_q 1 (24 rows) 3.01 / 3.39 / 3.23 /
3.52 vs 8.48 / 6.78 / 8.10 / 7.14; B 1 x S_q 4 (96 rows) 3.58 / 3.74 / 3.62 / 3.55 vs 16.22 / 10.05 / 9.41 / 6.88; B 4 x S_q 1 (96 rows)
3.71 / 3.81 / 3.74 / 3.78 vs 8.38 / 6.82 / 8.16 / 7.65; B 4 x S_q 4 (384 rows) 8.83 / 8.93 / 8.80 / 8.80 vs 16.29 / 10.21 / 9.50 / 7.38 --
the cc 10.7 shape names the measured winner in 15 of the 16 cells (the miss: 384 rows at split 22, the own forms 9.50 vs 8.80).  The
measured-default rule: a part joins this tuple with its table; cc 10.3 (B300) is UNMEASURED and keeps the module's own forms, the
tcgen05 body on every part reads as before."""
_SPARSE_LEAN_COMBINE_RULE_SOURCE_CC = (10, 7)
"""The line whose split-aware combine rule the parts above borrow (``heuristics._DECODE_COMBINE_RULE_CC``, pinned equal by the test
suite: this module cannot import the heuristics, which import it)."""


def sparse_lean_combine_rule_cc(device_cc, *, lean: bool) -> Tuple[int, int]:
    """The device whose split-aware combine-form rule a sparse plan's combine entry follows: the plan's own device, except a LEAN plan on
    a part of :data:`SPARSE_LEAN_COMBINE_RULE_CCS`, which follows :data:`_SPARSE_LEAN_COMBINE_RULE_SOURCE_CC` (cc 10.7: the rule's shape,
    measured for the lean grids on that part -- the table above)."""
    cc = (int(device_cc[0]), int(device_cc[1]))
    if lean and cc in SPARSE_LEAN_COMBINE_RULE_CCS:
        return _SPARSE_LEAN_COMBINE_RULE_SOURCE_CC
    return cc


LEAN_SPARSE_DEFAULT_MAX_SUB_TILES_PER_CTA = 1
"""The cap on the DEFAULT ``sub_tiles_per_cta`` (k): 1 -- one 32-key sub-tile per CTA, the kill-line rendering (REG 66-72, 4 CTAs per SM
by registers).  The fill rule of :func:`sparse_lean_fill_sub_tiles` asks for k 2 / 3 / 4 / 7 at B 8 / B 4 x S_q 4 / B 16 / B 32 units,
but the k >= 2 renderings trace at 104-164 registers (ptxas software-pipelines the unrolled sub-tiles: 3 CTAs per SM; at 512 threads --
S_q 4 x G 16, four m-tiles -- the hardware cap 65536 / 512 = 128 binds the k >= 2 body, which then spills 7 words, while k 1 spills
nothing on any rendering: REG <= 79) and their gain over k 1 is unmeasured -- a ``-maxrregcount`` build is the gate's lever, not a default.  A caller pins k through the adapter's
``sub_tiles_per_cta`` (the block's ``sparse_sub_tiles_per_cta``); the gate's measurement raises this cap, never a transcription."""
_LEAN_FILL_OCCUPANCY_BY_S_Q = {1: 3, 2: 2}
"""The fill rule's CTAs per SM by the step's rows (MODEL, DESIGN 2.2 / 4): 3 at S_q 1 (the 72-KiB two-stage form under the standard
carveout), 2 at S_q 2 (80 KiB), 1 at S_q 3..4 (the 384-512-thread renderings, by registers)."""


def sparse_lean_splits(top_k: int) -> Tuple[int, ...]:
    """The LEAN SPLITS of ``top_k``: ``ceil(n_sub / k)`` for every k in ``1 .. min(8, n_sub - 1)`` (:func:`sparse_lean_max_sub_tiles_per_cta`)
    with ``n_sub = ceil((top_k + 2) x 4 / 32)`` (``config_sm107.lean_sparse_n_sub``: 65 at top_k 512), descending -- ``(65, 33, 22, 17, 13,
    11, 10, 9)`` at top_k 512, ``(2,)`` at top_k 8; every entry >= 2 (the lean tile always splits).  The grid's x extent and the combine's
    partial count; a pinned ``split_kv`` on the lean route must be one of them (it SELECTS k), any other value is a typed decline naming
    this list."""
    return tuple(sorted({lean_sparse_split(int(top_k), k) for k in range(1, sparse_lean_max_sub_tiles_per_cta(top_k) + 1)}, reverse=True))


def sparse_tcgen05_max_split(top_k: int) -> int:
    """The tcgen05 core's split bound at ``top_k``: ``ceil((top_k + 1) / 32)`` = the item's tile count (17 at top_k 512; a larger split has an
    empty chunk on every item).  The adapter's check and the envelope's readback read ONE copy."""
    return -(-(int(top_k) + 1) // 32)


def sparse_lean_max_sub_tiles_per_cta(top_k: int) -> int:
    """The largest k a lean launch may pin at ``top_k``: ``min(8, n_sub - 1)`` -- the sub-tile loop is unrolled at trace time
    (``config_sm107.LEAN_SPARSE_MAX_SUB_TILES_PER_CTA``), and the lean tile ALWAYS splits (its O / LSE are the fp32 partials the combine
    reduces), so k = n_sub, one chunk = split 1, is not a lean rendering: 8 at top_k 512 (n_sub 65), 1 at top_k 8 (n_sub 2)."""
    n_sub = lean_sparse_n_sub(int(top_k))
    return max(1, min(LEAN_SPARSE_MAX_SUB_TILES_PER_CTA, n_sub - 1))


def sparse_lean_tile_decline(
    *,
    s_q: int,
    group: int,
    top_k: int,
    thd: bool = False,
    bottom_right: bool = True,
    list_per_sequence: bool = True,
    seq_kv_lens_present: bool = True,
    half_dtype: bool = True,
    softmax_groups: int = 0,
) -> Optional[str]:
    """Why the LEAN tile cannot serve a declaration (the arm NAMED), or ``None`` when it can -- the ONE statement the adapter raises for
    an explicit ``tile_form="lean"`` (``NotImplementedError``) and the block's declaration repeats, and the admissibility ``"auto"`` reads
    before the placement rule.  The lean body is the DECODE form only: bottom-right rows at the END of each sequence, ONE gathered list
    per sequence -- ``list_per_sequence=True`` at ``S_q <= config_sm107.LEAN_SPARSE_SERVED_MAX_S_Q`` (4: every row of the step in one CTA,
    ``n_mt = ceil(S_q x G / 16)`` m16 tiles over one gathered sub-tile), or the per-row list at ``S_q == 1``, which IS the shared list (one
    row per sequence, the step-0 anchor the row's own position); per-row lists past one row are the core's --, the per-sequence KV length present (the visible range), dense BSHD or paged pools (never THD), bf16 / f16, a group of
    at most 16 query heads per KV head (the m16 tile's rows), ``top_k`` a multiple of 4 in [8, 512] (below 8 the list is ONE sub-tile =
    split 1: the tcgen05 core's cell), the tcgen05 core's ``softmax_groups`` knob at its default.  The epilogue gate rides the combine
    under the lean split (always >= 2), so a gate operand is served.  Every other arm (a sink, a band, log2 stats, a padded-Q trim, a pure
    list) is the record's own decline, raised before this one."""
    if thd:
        return "thd=True (packed sequences) is the tcgen05 core's arm: the lean tile has no claim-counter form -- declare tile_form='tcgen05' or 'auto'"
    if not bottom_right:
        return (
            "the lean tile serves the DECODE form only (bottom_right=True: the step's rows at the end of each sequence); a top-left / prefill "
            "declaration is the tcgen05 core's -- declare tile_form='tcgen05' or 'auto'"
        )
    if not seq_kv_lens_present:
        return (
            "the lean tile needs seq_kv_lens (the per-sequence KV length is the decode form's visible range); the tcgen05 core serves a dense "
            "bottom-right declaration without it -- pass seq_kv_lens or declare tile_form='tcgen05' / 'auto'"
        )
    if int(s_q) > LEAN_SPARSE_SERVED_MAX_S_Q:
        return (
            f"the lean tile serves S_q <= {LEAN_SPARSE_SERVED_MAX_S_Q} rows per sequence (the serving step's rows under ONE shared list: n_mt m16 "
            f"tiles over one gathered sub-tile); got S_q = {int(s_q)} -- a longer step is the tcgen05 core's (tile_form='tcgen05' / 'auto')"
        )
    if not list_per_sequence and int(s_q) != 1:
        return (
            f"per-row lists at S_q = {int(s_q)} > 1: the lean CTA gathers ONE list per sequence (the shared step-0 list, index_list_form='shared'); "
            "a per-row list is the tcgen05 core's (tile_form='tcgen05' / 'auto')"
        )
    if not half_dtype:
        return "the lean tile serves bf16 / f16 Q / K / V (the mma.sync m16n8k16 half body); the e4m3 pools arm is not wired"
    if not 1 <= int(group) <= 16:
        return f"the lean tile packs the token's query-head group on one m16 tile: H_q / H_kv must be in [1, 16], got {int(group)}"
    if int(top_k) < LEAN_SPARSE_TOPK_MIN or int(top_k) % 4 != 0:
        return (
            f"the lean tile serves top_k a multiple of 4 in [{LEAN_SPARSE_TOPK_MIN}, 512] (below {LEAN_SPARSE_TOPK_MIN} the list is one 32-key "
            f"sub-tile = split 1, the tcgen05 core's cell); got top_k = {int(top_k)}"
        )
    if int(softmax_groups or 0) != 0:
        return (
            f"softmax_groups={int(softmax_groups)} is the tcgen05 core's softmax-split knob (the lean tile's softmax is per warp on one m16 tile); "
            "leave it at 0 or declare tile_form='tcgen05'"
        )
    return None


def sparse_decode_lean_tile_selected(*, batch: int, h_kv: int, s_q: int, top_k: int, device_cc: Optional[Tuple[int, int]]) -> bool:
    """The PLACEMENT rule (DESIGN 7): True iff the lean tile is the body the route prefers for an ADMISSIBLE block-sparse decode
    launch -- the cc 10.7 line (:data:`_SPARSE_LEAN_TILE_RULE_CC`, the one part the kernel exists for), ``top_k >= 8`` (at least two
    sub-tiles: a one-sub-tile list is the tcgen05 core's) and ``B x H_kv <= LEAN_SPARSE_MAX_UNITS`` CTAs per chunk -- the lean grid's own
    unit: the step's ``s_q`` rows ride inside the CTA as m16 tiles, so ``s_q`` is NOT in the bound (it is the admissibility's,
    :func:`sparse_lean_tile_decline`: at most 4 rows under one shared list; the gate's S_q 4 rows win at B 1 and lose at B 4 exactly as
    its S_q 1 rows do) --; False above the bound (the tcgen05 core with today's split rule) and on every other part.  A dense rule from
    the configuration (the bound is the one constant the gate moves); the AUTO reading's gate on the part is :func:`sparse_tile_form_auto`'s.
    ``s_q`` stays in the signature (every caller passes the step width) and is deliberately unread here."""
    if device_cc is None or tuple(device_cc)[:2] != _SPARSE_LEAN_TILE_RULE_CC:
        return False
    if int(top_k) < LEAN_SPARSE_TOPK_MIN:
        return False
    return int(batch) * int(h_kv) <= LEAN_SPARSE_MAX_UNITS


def sparse_lean_fill_sub_tiles(*, batch: int, h_kv: int, s_q: int, group: int, top_k: int, sm_count: int) -> int:
    """DESIGN 2.2's FILL RULE for the sub-tiles per CTA, k (MODEL, information): ``units = B x H_kv`` (the S_q rows share one CTA: the
    shared list), ``k_fill = ceil(n_sub x units / (occ x SMs))`` -- one wave of resident CTAs at the rendering's CTAs per SM
    (:data:`_LEAN_FILL_OCCUPANCY_BY_S_Q`) --, ``k_bytes = ceil(S_q x G x 1 KiB / 32 KiB)`` -- the fp32 partial a CTA WRITES never exceeds
    the K + V it READS --, ``k = clamp(max(1, k_fill, k_bytes), 1, min(8, n_sub))``: 1 at B 1 / B 4 x S_q 1, 2 at B 1 x S_q 4 and B 8 x
    S_q 1, 3 at B 4 x S_q 4, 4 at B 16 x S_q 1, 7 at B 32 x S_q 1 on 212 SMs.  ``sm_count <= 0`` (no CUDA reading) gives 1.  The DEFAULT
    k is this value capped at :data:`LEAN_SPARSE_DEFAULT_MAX_SUB_TILES_PER_CTA` (:func:`sparse_lean_sub_tiles_per_cta`)."""
    k_max = sparse_lean_max_sub_tiles_per_cta(top_k)
    n_sub = lean_sparse_n_sub(int(top_k))
    if int(sm_count) <= 0:
        return 1
    units = max(1, int(batch) * int(h_kv))
    occ = _LEAN_FILL_OCCUPANCY_BY_S_Q.get(int(s_q), 1)
    k_fill = _ceil_div(n_sub * units, occ * int(sm_count))
    k_bytes = _ceil_div(int(s_q) * max(1, int(group)) * 1024, 32 * 1024)
    return max(1, min(k_max, max(k_fill, k_bytes)))


def sparse_lean_sub_tiles_per_cta(*, batch: int, h_kv: int, s_q: int, group: int, top_k: int, sm_count: int) -> int:
    """The DEFAULT sub-tiles per CTA of a lean launch: the fill rule (:func:`sparse_lean_fill_sub_tiles`) capped at
    :data:`LEAN_SPARSE_DEFAULT_MAX_SUB_TILES_PER_CTA` = 1 today -- the k >= 2 rows stay off the default until the register lever is
    measured; a pinned ``sub_tiles_per_cta`` (any k in ``1 .. min(8, n_sub)``) outranks it, the same function under every value."""
    fill = sparse_lean_fill_sub_tiles(batch=batch, h_kv=h_kv, s_q=s_q, group=group, top_k=top_k, sm_count=sm_count)
    return max(1, min(fill, LEAN_SPARSE_DEFAULT_MAX_SUB_TILES_PER_CTA, sparse_lean_max_sub_tiles_per_cta(top_k)))


def sparse_tile_form_auto(
    *, batch: int, h_kv: int, s_q: int, top_k: int, device_cc: Optional[Tuple[int, int]], admissible: bool, split_pinned: Optional[int] = None
) -> str:
    """What ``tile_form="auto"`` resolves to: :data:`SPARSE_TILE_FORM_LEAN` iff the part is in :data:`SPARSE_LEAN_TILE_DEFAULT_CCS` (the
    measured default: cc 10.7), the declaration is ``admissible`` on the lean body
    (:func:`sparse_lean_tile_decline` returned None), the placement rule selects it (:func:`sparse_decode_lean_tile_selected`) and a
    pinned split, if any, is a lean split of ``top_k`` (a caller who pinned ``split_kv`` to a tcgen05 split, or to 1 = unsplit, keeps the
    body that honours the pin: the pin is the stronger knob and the resolved form reads back); :data:`SPARSE_TILE_FORM_TCGEN05` otherwise."""
    if device_cc is None or tuple(device_cc)[:2] not in SPARSE_LEAN_TILE_DEFAULT_CCS:
        return SPARSE_TILE_FORM_TCGEN05
    if not admissible or not sparse_decode_lean_tile_selected(batch=batch, h_kv=h_kv, s_q=s_q, top_k=top_k, device_cc=device_cc):
        return SPARSE_TILE_FORM_TCGEN05
    if split_pinned is not None and int(split_pinned) not in sparse_lean_splits(top_k):
        return SPARSE_TILE_FORM_TCGEN05
    return SPARSE_TILE_FORM_LEAN


# ---------------------------------------------------------------------------------------------------------------------------------
# The fused gated e4m3 store's ROUNDING CONVENTIONS (``sm100/split_combine.compile_ptr(out_rounding=)``; the sparse adapter's ``out_rounding=``
# attribute -- an OP ATTRIBUTE of the declaration, never a knob: it changes the bytes).  "single" (the default) = ONE rounding of the fp32 gated
# value (FROST's convention); "staged" = a serving stack's gate-and-quantize order (bf16 of the merged O, the exact sigmoid as a division, bf16
# of the gated value, the fp32 multiply by div.rn(1, s), the saturating e4m3 cvt), so a provider can serve bytes identical to that stack's chain.
# The vocabulary lives HERE, in the DSL-free module, because the adapter validates it at construction, BEFORE the DSL gate (python/cudnn/AGENTS.md
# Rule 7: the kernel module is imported only after the gate), and a declaration's construction must not import ``cudnn.frost.tile_dsl`` -- the
# heuristics module does so at module level and re-exports these names unchanged (its vocabulary for the kernel module and the suites).  A third
# convention appends a name; the kernel's trailing ``quant_rounding`` constexpr appends a value.
COMBINE_OUT_ROUNDING_SINGLE = "single"
COMBINE_OUT_ROUNDING_STAGED = "staged"
COMBINE_OUT_ROUNDINGS = (COMBINE_OUT_ROUNDING_SINGLE, COMBINE_OUT_ROUNDING_STAGED)
