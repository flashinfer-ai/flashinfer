# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_score_paged`` -- the decode indexer's scorer over the PAGED compressed-key cache; its standalone selector is the host entry
``QSA/selection_paged.py``.

A decode step of a block-sparse (``QsaSpec``) layer scores each of its ``S_q`` query rows per sequence against the compressed
keys of the complete 4-token blocks the row can see, and the top-k of those scores is the row's block list.  The serving
stacks keep the compressed keys PAGED on the main block table (one compressed row per 4 tokens, ``page_size / 4`` rows per
page), so this kernel reads them there -- once, in page order, never through a dense per-sequence copy::

    pos_t   = kv_lens[b] - S_q + s                       (row s of sequence b; t = b * S_q + s)
    n_vis_t = clamp(floor((pos_t + 1) / 4), 0, n_blocks_max)
    logits[t, j] = scale * sum_h relu(qi[t, h] . kbar[b, j])        for j < n_vis_t, kbar[b, j] = cache[table[b, j // R], j % R]
    counts[t]     = n_vis_t                                          (R = page_size / 4 compressed rows per page)
    block_lens[t] = min(top_k, n_vis_t)                              (optional output)

The ``logits [T, n_blocks_max]`` fp32 + ``counts [T]`` int32 pair is the input contract of the decode top-k that follows
(the radix kernel today, a FROST top-k later): the top-k reads row ``t``'s columns ``[0, counts[t])`` -- nothing else of the
row is written here -- and pads its output with ``-1`` past ``min(top_k, counts[t])``.  The host entry
``selection_paged.qsa_select_decode_paged_forward`` (a DSL-free module, so its import path resolves on every installed DSL) pairs the
scorer with the deterministic radix top-k (the smaller block id wins a tie at the cutoff, the selected ids come back in ASCENDING
order, so two calls are bitwise) and no torch prologue: the positions and the counts are derived on device by this kernel.

**The form: a warp-level MMA over the (s, h) columns, the keys streamed by ``cp.async``.**  A decode step has 1-4 query rows per
sequence, 4 indexer heads each: the ``S_q x H_i`` (s, h) rows pad to the M = 16 of one ``mma.sync.m16n8k16`` (bf16 in, fp32
accumulate), so one MMA per 8 keys per 16 of the 128 D scores EVERY column at once -- the work is flat in ``S_q`` and the compute
term is negligible against the bytes (131 k MAC per 16 KB of keys).  The padded query tile is staged ONCE per CTA in shared memory
(``cp.async``, every thread a 16-B granule, zero-fill for the padding rows) behind the kernel's one CTA barrier and read into A
fragments; the keys stream through a per-warp ring of ``cp.async`` groups -- page part by page part, whole 128-B lines per
request, an L2 evict-first policy where the DSL's compiler serves it (the keys pass through L2 once; the logits' write-combining
lines and the table stay; the ``l2_hint`` arm below), a zero-filled copy for a dead page -- into B fragments read with
``ld.shared.v4``; relu, the head sum and the scale run on the fp32 accumulator fragment.  No TMA, no TMEM, no cluster: the first key byte is requested ONE memory round trip after the launch
(``kv_lens``, the table words and the query tile are all in flight together), nothing holds a register while it waits, and the
kernel's roofline term is the HBM bytes of the compressed rows it streams (plus the fp32 logits it writes).

Work decomposition (every constant below is a module constant or a compile key; nothing is inferred at launch):

* grid ``(ceil(n_blocks_max / blocks_per_cta), B)``, CTA = 4 warps = 128 threads; a warp owns ``keys_per_warp = 32 x
  units_per_warp`` consecutive compressed rows (``units_per_warp`` in {1, 2, 4, 8} is a performance knob: the same function;
  ``blocks_per_cta`` = 128 / 256 / 512 / 1024; the default 1 = the best point of every measured cell), streamed as n-tiles of 8
  keys, ``ring_depth`` (2 / 4 / 8, default 4: a performance knob) n-tiles in flight;
* **the K permutation** shared by A and B (so the product is unchanged): lane ``(g = lane / 4, q = lane % 4)`` reads the 16-byte
  granules ``c = q + 4 j`` (``j = 0..3``) of its row, and k-step ``kappa = 2 j + r`` takes granule ``q + 4 j``'s words ``2 r`` (logical k
  ``2q, 2q + 1``) and ``2 r + 1`` (logical k ``2q + 8, 2q + 9``): a bijection ``(kappa, k) <-> D``, so the fragment registers ARE the
  granules' words (no shuffle, no ``ldmatrix``).  A per m-tile: ``reg0 / reg1`` = rows ``g`` / ``g + 8`` at word ``2 r``, ``reg2 /
  reg3`` the same rows at word ``2 r + 1``; B per n-tile: key ``n = g``'s words ``2 r`` / ``2 r + 1``; C: lane ``(g, q)`` holds rows
  ``g`` / ``g + 8`` for keys ``2q, 2q + 1``;
* **the e4m3 READ arm** (``cache_dtype=torch.float8_e4m3fn``, the fp8 indexer cache and e4m3 queries at unit scale; sm_89+; 2026-10-21):
  the SAME kernel with the element size swapped -- a row is 128 B = 8 granules, lane ``(g, q)`` reads granules ``q + 4 j`` for ``j in
  {0, 1}`` (2 ``ld.shared.v4``), and k-step ``kappa = 2 j + r`` (4 k-steps) feeds granule ``q + 4 j``'s word ``2 r`` as the MMA k slots
  ``4q .. 4q + 3`` and word ``2 r + 1`` as ``4q + 16 .. 4q + 19`` of ``mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32`` (``D(kappa, k) = 16 (q + 4
  j) + 4 (2 r + [k >= 16]) + (k mod 4)``, a bijection onto ``[0, 128)``, shared by A and B); EVERY k-step consumes 32 B of a row in both
  forms, so the fragment indexing is dtype-free and only the step count (8 / 4), the granules per lane (4 / 2) and the row bytes (256 /
  128) move -- each a function of the element size (``selection_paged.row_bytes`` / ``k_steps`` / ``chunks_per_lane`` / ``mma_k``), never a
  second literal.  Every e4m3 x e4m3 product is exact in fp32: the arm's error is the accumulation's alone.  A copy instruction moves
  FOUR rows (lane ``l`` -> row ``4 e + (l >> 3)``, granule ``l & 7``: eight consecutive lanes per 128-B line, 2 instructions per n-tile; the
  query tile ONE ``cp.async`` per thread per m-tile), and at page 8 (``R = 2``) one instruction serves TWO page parts, so the part's
  table word is shuffled from a PER-LANE source (``4 x rows_pp x (2 e + (lane >> 4))``) and the validity check and the ``mul.wide`` page
  address are per lane; ring slot 1 KiB, SMEM 18 KiB at the defaults.  The barrier table below is UNCHANGED row by row: the arm changes
  the bytes per copy, never a participant count or a wait.  A different selection function (3 mantissa bits), chosen by the caller's
  dtype; bf16 stays the default and the reference;
* **the row map** (``selection_paged.query_row_map``): a query's heads pad to ``P = next_pow2(H_i)`` rows; ``P' = min(P, 16)``
  of them share a 16-row m-tile with ``QPT = 16 / P'`` queries, a query wider than a tile spans ``TPQ = P / 16`` tiles
  (``m_tiles = ceil(S_q x P / 16) <= 4``, the ``MAX_QUERY_ROWS`` envelope); with ``P' >= 2`` the two rows a lane holds are heads
  ``2 gi`` / ``2 gi + 1`` of ONE query (``gi = g % G``, ``G = P' / 2``) and the remaining heads sit in the ``G`` consecutive ``g``
  values = lanes ``xor 4, 8, 16``: the head sum is one in-thread add plus ``log2(G)`` butterfly shuffles (H_i 4: ONE), plus the
  in-thread sums over a query's tiles at ``P > 16``; with ``P' = 1`` the two rows are two queries and nothing is summed; a padding
  row (``h >= H_i`` inside the power of two, ``s >= S_q``) is a ZERO row of the staged tile (its copy is zero-filled);
* the prologue, every thread: ``kv_lens[b]``; the table words of the first ``2 x ring_depth`` n-tiles (per lane the word of ITS key,
  ``table[b, min(k, n_blocks_max - 1) / R]``: 1-4 distinct words per instruction, one sector); the query tile's copies (2 per m-tile,
  one ``cp.async`` group); then -- live warps only (``k_warp0 < n_vis_max``, warp-uniform) -- the first ``ring_depth`` n-tiles of
  keys (one group each); ``cp.async.wait_group`` for the query tile's group; the CTA barrier; the A fragments (8 ``ld.shared.v4``
  per m-tile per lane);
* the mainloop, live warps, in batches of ``mma_batch`` n-tiles (4 at ``units_per_warp`` 1, 2 above: the HMMA chains of a batch
  interleave) -- per n-tile: the table word of n-tile ``i + 2 x ring_depth``; ``cp.async.wait_group`` for group ``i``; ``bar.warp.sync``
  (the other lanes' copies visible, the previous slot's reads ordered before its reuse); four ``ld.shared.v4`` (the B fragments);
  the copies of n-tile ``i + ring_depth`` into the slot read one step earlier (the ring has ``ring_slots = min(ring_depth + 1,
  n_tiles)`` slots: one spare only where a mainloop copy exists); then the
  batch's ``8 x m_tiles`` MMAs per n-tile, relu per (s, h) element (fp32), the head sum, the scale; the lane with ``gi == 0`` (every
  lane at ``P' = 1``) stores query ``s``'s two adjacent keys ``8 i + 2q, + 1`` -- one ``st.global.v2.f32`` when both are visible to
  that row, ``st.global.f32`` when only the first is, nothing otherwise;
* a key copy = per page part (``min(R, 8)`` rows of one page) ONE shuffle (the part's table word from the lane that read it -- a
  constant source lane), one validity check, one ``mul.wide`` page address, then ``rows / 2`` ``cp.async.cg`` of 16 B per lane at
  distinct address registers (instruction ``e`` copies rows ``2 e`` by lanes 0-15 and ``2 e + 1`` by lanes 16-31: eight consecutive
  lanes per 128-B half, whole lines per request); a part on a ``-1`` / out-of-range page is ZERO-FILLED (``src-size 0``: nothing of the
  cache is read; the address is page 0, a real row); a row at or past the visible count or the capacity reads a real row (its table
  word was clamped to the capacity) and is never stored: its products stay in its own accumulator column;
* the op order: the MMA's fp32 accumulation over the 128 D per (s, h), ``relu``, then ``((h0 + h1) + (h2 + h3))`` and the tile sums,
  then ``x scale`` -- fixed, so two launches are bitwise; the fp64 reference's budget is met by the fp32 accumulation.

**SMEM buffer table** (`frost-kernels.md` section 5; no descriptor reads either buffer -- the swizzle's one job is the banks).  ONE
dynamic allocation sized by the DSL from the two ``cutlass.Array`` declarations (no ``smem=`` at the launch): ``sK`` at offset 0, ``sA``
at ``4 x ring_slots x 2 KiB``; total ``4 x ring_slots x 2 KiB + m_tiles x 4 KiB`` with ``ring_slots = min(ring_depth + 1, n_tiles)`` =
**36 KiB at the defaults** (units 1 = 4 n-tiles, depth 4: 4 slots; one m-tile), 44 KiB from units 2 (5 slots), 36 / 68 / 76 KiB at
depth 8 for units 1 / 2 / 4 and 8 (4 / 8 / 9 slots); every size a multiple of 2 KiB, both buffers 128-B aligned.  The e4m3 arm halves
every row: a ring slot is 1 KiB, an m-tile 2 KiB, ``sA`` at ``4 x ring_slots x 1 KiB``, **18 KiB at the defaults** (22 KiB from units 2,
18 / 34 / 38 KiB at depth 8); a 128-B row stride IS the 32-bank cycle, so there the ``c ^ (4 x (r & 1))`` XOR is what spreads the banks
(unswizzled: 2-way on every quarter-warp read); the e4m3 rows of the table are the bf16 rows at half the bytes:

=======  =====  ========================  =============================================  ==============================================  ==========================================================================
name     dtype  elems                     written by                                     read by                                         per-lane stride / swizzle
=======  =====  ========================  =============================================  ==============================================  ==========================================================================
``sK``   Int32  ``4 x ring_slots x 512``  this warp's ``cp.async.cg`` (16 B per lane:    this warp's ``ld.shared.v4``: lane ``(g, q)``   the 16-B granule ``c`` of row ``r`` sits at ``r x 256 + (c ^ (4 x (r & 1))) x 16``:
                (a per-warp ring of       lanes 0-15 row ``2 e``, 16-31 row ``2 e + 1``)  reads row ``g``'s granules ``q + 4 j``          a quarter-warp of a write (one row's eight granules) or of a read (rows ``2 m``,
                ``ring_slots`` slots of                                                                                                   ``2 m + 1``, four granules each) spans all 32 banks: ONE wavefront per quarter-
                8 rows x 256 B; 32 KiB                                                                                                    warp, the minimum for 128-bit accesses (an unswizzled layout puts the eight rows'
                at the defaults)                                                                                                          granule ``c`` on the same 16 banks: 2-way)
``sA``   Int32  ``m_tiles x 1024`` (the   every thread's ``cp.async.cg`` (2 per m-tile:  every warp's ``ld.shared.v4``: lane ``(g, q)``  the same ``c ^ (4 x (row & 1))`` layout and the same one-wavefront-per-quarter-warp
                padded query tile, 16      thread ``t`` granule ``t % 16`` of rows ``t /  reads rows ``g`` / ``g + 8``'s granules ``q +   argument (the copy's 16 consecutive lanes cover one row = two 128-B lines)
                rows x 256 B per m-tile)   16`` and ``8 + t / 16``)                       4 j``
``sK``   Int32  e4m3: ``4 x ring_slots x   lane ``l`` -> row ``4 e + (l >> 3)``, granule  lane ``(g, q)`` row ``g``'s granules ``q + 4 j``,  granule ``c`` of row ``r`` at ``r x 128 + (c ^ (4 x (r & 1))) x 16``, ``c < 8``: a
(e4m3)          256`` (slots of 8 rows x   ``l & 7`` (8 consecutive lanes per 128-B line; ``j in {0, 1}`` (2 ``ld.shared.v4``, 8 words)     quarter-warp read (rows ``2 m`` / ``2 m + 1``, granules ``q + 4 j`` and their XOR-4
                128 B; 16 KiB at the       2 instructions per n-tile; page 8: two parts                                                   twins) and a quarter-warp write (one row's 8 granules) each span the 32 banks: ONE
                defaults)                  per instruction, per-lane word)                                                                wavefront; the 128-B row stride IS the bank cycle, so the XOR is REQUIRED (2-way without)
``sA``   Int32  e4m3: ``m_tiles x 512``    ONE ``cp.async.cg`` per thread per m-tile:     lane ``(g, q)`` rows ``g`` / ``g + 8``, granules  the same layout and argument (8 consecutive lanes per row = one line)
(e4m3)          (16 rows x 128 B)          thread ``t`` granule ``t & 7`` of row ``t >> 3``  ``q + 4 j``, ``j in {0, 1}``
=======  =====  ========================  =============================================  ==============================================  ==========================================================================

**Barrier table** (`frost-kernels.md` section 4) -- no mbarrier, no TMA, no tcgen05, no cluster:

====================  ======================================  ==================================  ===================  ===============================================================================
name                  kind                                    issuing lanes                       init / expected      where
====================  ======================================  ==================================  ===================  ===============================================================================
``bar.sync 0``        the CTA barrier (``cute.arch.barrier``)  all 128 threads (4 warps x 32),     128 == the CTA size  after every thread's ``cp.async.wait_group`` of the query tile's group (a live warp
                                                              once per CTA: SUM = 128                                  allows its key groups to stay in flight, a dead warp waits for 0) and BEFORE the
                                                                                                                       per-warp live branch: every thread of every CTA reaches it exactly once
``cp.async`` groups   per-thread completion (commit / wait)    the issuing thread only              -- (no count)        the query tile: one group per thread, waited before the CTA barrier; the keys: one
                                                                                                                       group per n-tile per thread (4 copies of 16 B at bf16, 2 at e4m3), group ``i`` waited
                                                                                                                       at step ``i`` (``min(n_tiles, depth + i) - i - 1`` younger groups may stay pending)
``bar.warp.sync``     the warp barrier (``cute.arch.sync_warp``)  all 32 lanes, once per n-tile     32 == the warp       inside the warp-uniform live branch only: after the wait of group ``i``, before the
                      with memory ordering                                                                              fragment reads of slot ``i`` and the copies into the slot read at step ``i - 1``
``mma.sync.aligned``  warp-collective                          all 32 lanes, ``8 x m_tiles`` per   32                   inside the live branch only (``k_warp0 < n_vis_max``: both operands warp-uniform);
                                                              n-tile (bf16 m16n8k16; e4m3:                             the e4m3 arm's ``m16n8k32`` at half the count
                                                              ``4 x m_tiles`` of m16n8k32)
``shfl.sync.bfly``    warp-collective (mask ``0xffffffff``)    all 32 lanes, ``2 x stages`` per     32                   the same branch; the per-lane store predicates come AFTER the shuffles
                                                              query group per n-tile
``shfl.sync.idx``     warp-collective (mask ``0xffffffff``)    all 32 lanes, one per page part      32                   the same branch (the copies' table word from a constant source lane; the e4m3
                                                              (e4m3 at page 8: one per copy                            arm at page 8 shuffles a PER-LANE source: two parts per instruction, every lane
                                                              instruction, a per-lane source)                          participates)
====================  ======================================  ==================================  ===================  ===============================================================================

**The ``l2_hint`` arm** (a compile key, ``None`` by default): the key copies are ``cp.async.cg.shared.global.L2::cache_hint [dst],
[src], 16, size, policy`` with ``policy`` from ``createpolicy.fractional.L2::evict_first.b64`` where the installed CuTe DSL compiles
with CUDA ``L2_HINT_MIN_COMPILER`` = 13.4 or later, and the plain ``cp.async.cg.shared.global [dst], [src], 16, size`` below it -- the
13.3 PTX assembler bundled with the public nvidia-cutlass-dsl 4.7.0 cu13 wheel mis-encodes the hinted copy's uniform operands on sm_90 /
sm_100 (it builds the policy word and the ring base in two uniform registers and encodes all 16 hinted ``LDGSTS`` to read two OTHER
uniform registers nothing in the kernel writes: a ``Warp Illegal Instruction Parameter`` trap, ``cudaErrorIllegalInstruction``, at the
first key copy), the 12.9 assembler of the wheels' cu12 flavour emits the same encoding (a compile-only dump; the DSL binds that flavour
under a CUDA 12 driver whatever the wheel's number, a 4.8.0 wheel included, and reports it as CUDA 12.9), and the 13.4 assembler of the
4.8.0 wheel's cu13 toolchain (bound only under a CUDA 13 driver) encodes the operands it defined; the PTX is identical in all three.  The
hint is a cache-policy annotation: both arms move the same bytes into the same slots in the same order, so the logits are bitwise across
the arm (the knob cell pins it).  The compiler version is read from ``cutlass.base_dsl.version_info.CUDA_VERSION`` (present in both
wheels: the flavour the DSL bound, never the wheel's own number) at ONE seam, :func:`_bundled_cuda_version`; ``l2_hint=False`` traces
the plain copy on any DSL, ``l2_hint=True`` the hinted copy or -- below the floor -- a typed ``NotImplementedError`` (Rule 7) that names
the installed wheel and the compiler it found, states the defect as observed (never as a property of the found version) and prescribes
the cu13 flavour of 4.8.0 or later under a CUDA 13 driver, gated AFTER PDL's cc gate (a device gate before a form's gate).

PDL sites (under the ``use_pdl`` compile key, the launch attribute set with it): ``griddepcontrol.wait`` is the kernel's first
instruction -- ahead of the ``kv_lens`` / table / ``qi`` reads and so of every cache read --, ``griddepcontrol.launch_dependents``
follows the prologue's copies, BEFORE the counts and the logits are written.  That early trigger puts an obligation on the dependent
grid: a kernel launched with the programmatic-stream-serialization attribute behind this one may start while this one still runs and
MUST issue its own ``griddepcontrol.wait`` before it touches anything this kernel reads or writes (``logits``, ``counts``,
``block_lens``, and the inputs it could overwrite); the wait returns only after this grid has completed and flushed.  The standalone
selector (``qsa_select_decode_paged_forward``) launches the radix top-k WITHOUT the attribute, so the pair is inert there (the top-k starts
after the scorer completes, as any stream-ordered launch); it matters in the fused decode chain, whose nodes carry the attribute and
wait.  Without PDL (``use_pdl=False``, the standalone default) neither instruction is traced and the bf16 arm runs on any CuTe DSL
device from sm_80 (``mma.sync.m16n8k16``, ``cp.async``); the e4m3 arm needs sm_89 (``mma.sync.m16n8k32 .e4m3``) and is a typed decline
below it (:func:`compile_qsa_score_paged`).

Degenerate inputs, each with its handling: ``kv_lens[b] == 0`` (a padding sequence) -> every ``pos_t < 0``, ``n_vis = 0``: no
logits column written, ``counts = block_lens = 0``, no warp of the sequence copies a key (its ``qi`` rows and table words ARE read
-- valid memory, the round trip that keeps the critical path at two -- and its threads stage the query tile and reach the barrier);
``kv_lens[b] < S_q`` (a contract violation) -> the rows at a negative position get ``n_vis = 0`` likewise; ``n_vis`` inside a page
-> the rows past ``n_vis_max`` of a live n-tile read a real row and store nothing (their products stay in their own accumulator
column); a ``-1`` table entry (or one at or past ``num_pages``) for a VISIBLE block -> the part's copy is zero-filled: logit exactly
0, still written -- the sparse core's dead-page convention, never an out-of-bounds read; a dead page past ``n_vis_max`` may be
consulted (its word read, its copy zero-filled or a real page read) and is never stored; fewer than ``top_k`` visible blocks (the
identity region) -> ``block_lens = n_vis``, the paired top-k lists every visible id ascending then ``-1``; ``n_vis`` past
``n_blocks_max`` -> clamped to the capacity (the host checks the table's reach ``max_pages x R >= n_blocks_max``; a lane past the
capacity indexes the table at ``n_blocks_max - 1``).  Refused at the host before any launch: a table narrower than the capacity, a
page size that is not a power of two (at least 8), ``S_q x next_pow2(H_i)`` past the envelope (``MAX_QUERY_ROWS``), a head dim
other than 128, a cache whose rows are not contiguous inside a page or whose page stride is 2 GiB or more, misaligned or
non-unit-stride operands, a mixed dtype pair or one outside ``CACHE_DTYPES`` (bf16 / e4m3), and -- by name, a typed
``NotImplementedError`` -- the e4m3 pair below sm_89.

Bytes per launch (the SOL denominator): ``sum_b n_vis_max(b) x 256 B`` of compressed rows read once (128 B under the e4m3 arm);
``sum_t n_vis_t x 4 B`` of logits written; the query tile per live CTA (``m_tiles x 4 KiB``, 2 KiB at e4m3, L2-resident) and one table
word per page -- ``selection_paged.moved_bytes(..., elem_bytes=)``.

Rule 7: this module imports the DSL at import time like its neighbours, so the documented host entry lives in the DSL-free
front door ``QSA/selection_paged.py`` -- ``qsa_select_decode_paged_forward`` gates the DSL floor, the ``sm_107a``
target on cc 10.7 and the radix top-k's cc 9.0 BEFORE importing this module -- and the kernel's host-side facts (the row map,
the compile-key validator, the knob defaults, the byte model) are defined there and imported here, never the other way round
at module scope.  :func:`compile_qsa_score_paged`, the kernel-level entry a caller may reach directly, gates the floor and the
target again BEFORE any compile -- then PDL's cc 9.0, then the ``l2_hint`` arm's compiler floor --, so a too-old DSL reads as a
version decline there too; the tests skip below the floor.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mma import mma_m16n8k16_f32, mma_m16n8k32_f32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import opaque_f32_zero, opaque_i32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, st_global, st_global_v2
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

# the kernel's host-side facts live in the DSL-free front door (Rule 7: a declaration reads them without importing this module)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection_paged import (
    BLOCK_SIZE,
    CACHE_DTYPES,
    INDEX_HEAD_DIM,
    KEYS_PER_NTILE,
    MMA_M,
    NTILES_IN_FLIGHT,
    NTILES_PER_UNIT,
    THREADS,
    TOP_K_KERNEL_MAX,
    _ceil_div,
    _check_i32_1d,
    _kbar_rows_view,
    blocks_per_cta,
    cache_elem_bytes,
    k_steps,
    keys_per_warp,
    query_row_map,
    row_bytes,
    validate_params,
)

__all__ = [
    "QsaScorePagedRecipe",
    "dsl_decline_reason",
    "compile_qsa_score_paged",
    "run_qsa_score_paged",
    "L2_HINT_MIN_COMPILER",
    "l2_hint_available",
    "l2_hint_requirement_error",
    "FROST_SOURCE_DIGEST",  # read from OUTSIDE this module: the compiled-plan cache key (through ``template_key(globals(), ...)``), the tests, the run attestations
]

_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_CHUNK_BYTES = 16  # one 16-byte granule of a row: 8 bf16 (16 e4m3) = 4 words = one cp.async / ld.global.v4 / ld.shared.v4
_WORDS_PER_CHUNK = 4
# The row geometry is derived IN the kernel from the cache operand's element type at trace time through the host facts
# (selection_paged.row_bytes / k_steps): 256-B rows and 2 KiB ring slots at bf16, 128-B rows and 1 KiB slots at e4m3 -- no literal here.
_E4M3 = torch.float8_e4m3fn
_FP8_MMA_MIN_CC = (8, 9)  # mma.sync.m16n8k32 .e4m3 (the e4m3 read arm): Ada / Hopper / Blackwell / Rubin
_CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float8_e4m3fn: cutlass.Float8E4M3FN}


def _elem_bytes(elem_type) -> int:
    """Bytes per element of a cute element type (BFloat16 2, Float8E4M3FN 1): the e4m3 arm's one geometry switch, read at trace time."""
    return int(elem_type.width) // 8


L2_HINT_MIN_COMPILER = (13, 4)
"""The key copies carry the L2 evict-first hint (``cp.async.cg.shared.global.L2::cache_hint`` with a ``createpolicy`` register) only when
the installed CuTe DSL compiles with CUDA 13.4 or later.  The 13.3 PTX assembler bundled with the public nvidia-cutlass-dsl 4.7.0 wheel
mis-encodes that instruction on sm_90 / sm_100: it builds the policy word and the ring base in two uniform registers and encodes all 16
hinted ``LDGSTS`` to read two OTHER uniform registers that nothing in the kernel ever writes -- a ``Warp Illegal Instruction Parameter``
trap (``cudaErrorIllegalInstruction``) at the first key copy; the 12.9 assembler of the wheels' cu12 flavour emits the same encoding (a
compile-only dump), and the 13.4 assembler of the 4.8.0 wheel's cu13 toolchain encodes the operands it defined; the PTX is identical in
all three.  So the arm is selected at trace time: below the floor the copies are the plain ``cp.async.cg.shared.global`` (the same bytes
into the same slots, so the logits are bitwise across the arm).  The version is the DSL's bundled compiler
(``cutlass.base_dsl.version_info.CUDA_VERSION``, present in both wheels), never the wheel's own number: the DSL binds its cu12 or its
cu13 flavour by the driver's CUDA major (a 4.8.0 wheel reports 13.4 under a CUDA 13 driver and 12.9 under a CUDA 12 one), and a wheel
built with a newer toolkit keeps the hint."""


def _bundled_cuda_version() -> tuple:
    """``(major, minor)`` of the CUDA toolkit the installed CuTe DSL compiles with (its bundled ptxas: the cu12 or the cu13 flavour the
    DSL bound by the driver's CUDA major), from ``cutlass.base_dsl.version_info.CUDA_VERSION`` -- the ONE seam the ``l2_hint`` arm reads
    (``torch.version.cuda``, ``CUDA_HOME`` and the wheel's own number do not identify the DSL's compiler; Rule 7); the tests model a
    compiler by patching this function."""
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))


def l2_hint_available() -> bool:
    """Whether this DSL's compiler serves the hinted key copies (``L2_HINT_MIN_COMPILER``): the arm ``l2_hint=None`` resolves to."""
    return tuple(_bundled_cuda_version()) >= L2_HINT_MIN_COMPILER


def l2_hint_requirement_error() -> Optional[str]:
    """The typed reason ``l2_hint=True`` is refused on this DSL (Rule 7), else None: the floor, the defect as OBSERVED (the 13.3 assembler
    of the public 4.7.0 cu13 wheel natively, the cu12 flavour's 12.9 assembler in a compile-only dump -- never the found version by
    interpolation), the installed wheel with the compiler it found, and the remedy -- the cu13 flavour of 4.8.0 or later under a CUDA 13
    driver, not a bare wheel upgrade: a 4.8.0 wheel bound to its cu12 flavour already is the wheel."""
    cuda = tuple(_bundled_cuda_version())
    if cuda >= L2_HINT_MIN_COMPILER:
        return None
    _, version = cutedsl_state()
    wheel = "an unknown nvidia-cutlass-dsl build" if version is None else " ".join(version)
    need = ".".join(str(x) for x in L2_HINT_MIN_COMPILER)
    return (
        f"qsa_score_paged with l2_hint=True (the L2 evict-first hint on the key copies: cp.async.cg.shared.global.L2::cache_hint with a "
        f"createpolicy register) needs a CuTe DSL whose bundled CUDA compiler is {need} or later; below that floor the hint is not served: "
        f"the 13.3 PTX assembler of the public nvidia-cutlass-dsl 4.7.0 cu13 wheel mis-encodes those copies' uniform operands on sm_90 / "
        f"sm_100 (an illegal-instruction trap at the first key copy) and the cu12 flavour's 12.9 assembler emits the same encoding. The "
        f"installed {wheel} compiles with CUDA {cuda[0]}.{cuda[1]} -- use l2_hint=None or False (the plain copies: the same function) there, "
        f"or install 'nvidia-cutlass-dsl[cu13]>=4.8.0' and run under a CUDA 13 compatible driver (the DSL binds its cu13 toolchain only there)"
    )


def _resolve_l2_hint(l2_hint: Optional[bool]) -> bool:
    """The arm an artifact is built with: ``None`` follows the compiler (:func:`l2_hint_available`), ``False`` is the plain copy on every
    DSL, ``True`` the hinted copy -- or, below the floor, the typed decline."""
    if l2_hint is None:
        return l2_hint_available()
    if l2_hint:
        why = l2_hint_requirement_error()
        if why is not None:
            raise NotImplementedError(why)
        return True
    return False


# --------------------------------------------------------------------------------------------- the kernel


@cute.jit
def _page_word(tb_row: cutlass.Int64, k: cutlass.Int32, n_blocks_max: cutlass.Int32, rows_per_page: cutlass.Constexpr[int]) -> cutlass.Int32:
    """The table word of compressed row ``k`` (a row past the capacity indexes a real word: ``n_blocks_max - 1``)."""
    kc = k if k < n_blocks_max else n_blocks_max - cutlass.Int32(1)
    pg = kc >> cutlass.Int32(rows_per_page.bit_length() - 1)
    return ld_global(tb_row + pg.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)


@cute.jit
def _mul_wide_s32(a: cutlass.Int32, b: cutlass.Int32) -> cutlass.Int64:
    """``mul.wide.s32``: one IMAD.WIDE for a 32 x 32 -> 64-bit product (a page index x a page stride in bytes)."""
    return inline_ptx("mul.wide.s32 $0, $1, $2;", write_only_types=[cutlass.Int64], read_only_args=[a, b])


@cute.jit
def _copy_ntile(
    dst_lane: cutlass.Int64,
    src_lane_off: cutlass.Int32,
    kb_base: cutlass.Int64,
    kb_page_bytes: cutlass.Int32,
    word: cutlass.Int32,
    num_pages: cutlass.Int32,
    k_tile0: cutlass.Int32,
    policy: cutlass.Int64,
    rows_per_page: cutlass.Constexpr[int],
    rb: cutlass.Constexpr[int],
    lane_row: cutlass.Int32,
    l2_hint: cutlass.Constexpr[bool],
):
    """The n-tile's 8 compressed rows into one ring slot, PAGE PART by page part (``min(R, 8)`` rows of one page): per part ONE
    shuffle (the part's table word from the lane that read it -- a constant source lane), one validity check, one ``mul.wide``
    page address; then ``rows / rows_pi`` ``cp.async.cg`` of 16 B per lane at immediate offsets -- ``rows_pi = 512 / rb`` rows per
    instruction: at bf16 (``rb`` 256) instruction ``e`` copies rows ``2 e`` (lanes 0-15) and ``2 e + 1`` (lanes 16-31), each row's two 128-B
    halves by eight consecutive lanes (whole lines per request); at e4m3 (``rb`` 128) rows ``4 e .. 4 e + 3`` (lane ``l`` -> row ``4 e + (l
    >> 3)``, granule ``l & 7``), one 128-B line per eight lanes.  When a part is SMALLER than one instruction (e4m3 at page 8: ``R = 2``,
    two parts per 4-row instruction) each lane's part and its table word are PER LANE (``shfl.sync.idx`` from its own source lane), as
    are the validity check and the page address; ``lane_row`` (a lane's row within an instruction, ``0 .. rows_pi - 1``) selects them.
    A part on a ``-1`` / out-of-range page is ZERO-FILLED (``src-size 0``: nothing of the cache is read; the address is page 0, a
    real row, either way).  A row at or past the visible count or the capacity reads a real row (its table word was clamped to
    the capacity) and is never stored: its products stay in its own accumulator column.  ``l2_hint`` picks the copy's form (the module
    docstring's arm): the ``.L2::cache_hint`` copy carrying ``policy``, or the plain copy, which takes no policy operand."""
    rows_pi = (32 * _CHUNK_BYTES) // rb  # rows per copy instruction: 2 (bf16) / 4 (e4m3)
    rows_pp = min(rows_per_page, KEYS_PER_NTILE)
    zero_i = cutlass.Int32(0)
    if cutlass.const_expr(rows_pp >= rows_pi):
        for part in cutlass.range_constexpr(KEYS_PER_NTILE // rows_pp):
            w = cute.arch.shuffle_sync(word, 4 * part * rows_pp)  # the word of the part's first row sits in lanes 4 r .. 4 r + 3
            ok = (w >= zero_i) & (w < num_pages)
            page_c = w if ok else zero_i
            size = cutlass.Int32(_CHUNK_BYTES) if ok else zero_i
            row_off = ((k_tile0 + cutlass.Int32(part * rows_pp)) & cutlass.Int32(rows_per_page - 1)) * cutlass.Int32(rb)  # the part's first row inside its page
            src = kb_base + _mul_wide_s32(page_c, kb_page_bytes) + (row_off + src_lane_off).to(cutlass.Int64)
            dst = dst_lane + cutlass.Int64(part * rows_pp * rb)
            for e in cutlass.range_constexpr(rows_pp // rows_pi):
                # every cp.async gets its OWN address registers (an opaque offset, never base + immediate): four back-to-back LDGSTS on one
                # base register serialised on the register scoreboard under HBM latency (2x at B 32 x 128K, measured 2026-10-21)
                off_e = opaque_i32(cutlass.Int32(e * rows_pi * rb)).to(cutlass.Int64)
                if cutlass.const_expr(l2_hint):
                    inline_ptx(
                        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;",
                        write_only_types=[],
                        read_only_args=[dst + off_e, src + off_e, size, policy],
                    )
                else:
                    inline_ptx("cp.async.cg.shared.global [$0], [$1], 16, $2;", write_only_types=[], read_only_args=[dst + off_e, src + off_e, size])
    else:
        # several page parts per copy instruction (the e4m3 arm at page 8: R = 2, two 2-row parts per 4-row instruction): each lane's part
        # = e x parts_pi + lane_row / rows_pp, its table word from the per-lane source lane 4 x rows_pp x part (every lane shuffles: the
        # collective is warp-uniform, only the SOURCE is per lane), the check and the page address per lane
        parts_pi = rows_pi // rows_pp
        for e in cutlass.range_constexpr(KEYS_PER_NTILE // rows_pi):
            part = cutlass.Int32(e * parts_pi) + (lane_row >> cutlass.Int32(rows_pp.bit_length() - 1))
            w = cute.arch.shuffle_sync(word, part * cutlass.Int32(4 * rows_pp))
            ok = (w >= zero_i) & (w < num_pages)
            page_c = w if ok else zero_i
            size = cutlass.Int32(_CHUNK_BYTES) if ok else zero_i
            row_off = ((k_tile0 + part * cutlass.Int32(rows_pp)) & cutlass.Int32(rows_per_page - 1)) * cutlass.Int32(rb)
            src = kb_base + _mul_wide_s32(page_c, kb_page_bytes) + (row_off + src_lane_off).to(cutlass.Int64)
            off_e = opaque_i32(cutlass.Int32(e * rows_pi * rb)).to(cutlass.Int64)
            if cutlass.const_expr(l2_hint):
                inline_ptx(
                    "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;",
                    write_only_types=[],
                    read_only_args=[dst_lane + off_e, src, size, policy],
                )
            else:
                inline_ptx("cp.async.cg.shared.global [$0], [$1], 16, $2;", write_only_types=[], read_only_args=[dst_lane + off_e, src, size])


@cute.jit
def _read_key_frags(sK, slot_elems: cutlass.Int32, g: cutlass.Int32, qd: cutlass.Int32, rb: cutlass.Constexpr[int]):
    """Lane ``(g, q)``'s B fragments of the n-tile in the ring slot: row ``g``'s granules ``q + 4 j`` (one ``ld.shared.v4`` each,
    at the swizzled granule ``c ^ (4 x (g & 1))``) -- the 16 words the 8 k-steps consume at bf16 (``rb`` 256), the 8 words the 4
    k-steps consume at e4m3 (``rb`` 128)."""
    swz = (g & cutlass.Int32(1)) * cutlass.Int32(4)
    row = slot_elems + g * cutlass.Int32(rb // 4)
    out = []
    for j in cutlass.range_constexpr(rb // _CHUNK_BYTES // 4):
        c_swz = (qd + cutlass.Int32(4 * j)) ^ swz
        v = sK.load(row + c_swz * cutlass.Int32(_WORDS_PER_CHUNK), vector_size=4, alignment=16).to_elements()
        for x in cutlass.range_constexpr(_WORDS_PER_CHUNK):
            out.append(v[x])
    return tuple(out)


@cute.jit
def _copy_query_tile(
    sA_tile: cutlass.Int64,
    q_base: cutlass.Int64,
    q_s0: cutlass.Int64,
    q_s1: cutlass.Int64,
    t0: cutlass.Int32,
    tidx: cutlass.Int32,
    t: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    h_i: cutlass.Constexpr[int],
    rb: cutlass.Constexpr[int],
):
    """m-tile ``t`` of the padded (s, h) query tile into the CTA's SMEM staging: 16 rows x ``rb`` B by the 128 threads -- at bf16
    (``rb`` 256) two ``cp.async.cg`` instructions (thread ``tidx`` copies granule ``tidx % 16`` of rows ``tidx / 16`` and ``8 + tidx / 16``:
    sixteen consecutive lanes per row, whole lines), at e4m3 (``rb`` 128) ONE (granule ``tidx & 7`` of row ``tidx >> 3``: eight lanes per
    128-B row); a padding row (``h >= H_i`` inside the power of two, ``s >= S_q``) is ZERO-FILLED (``src-size 0``; its address is a real
    row).  The destination granule is ``c ^ (4 x (row & 1))``, the fragment reads' swizzle."""
    rm = query_row_map(s_q, h_i)
    zero_i = cutlass.Int32(0)
    s_q_i = cutlass.Int32(s_q)
    h_i_i = cutlass.Int32(h_i)
    bpe = rb // INDEX_HEAD_DIM  # 2 / 1
    granules = rb // _CHUNK_BYTES  # granules per row: 16 / 8
    rows_per_pass = THREADS // granules  # rows the 128 threads cover per instruction: 8 / 16
    c = tidx & cutlass.Int32(granules - 1)
    g_pass = tidx >> cutlass.Int32(granules.bit_length() - 1)  # the row within the pass: 0 .. rows_per_pass - 1
    for p in cutlass.range_constexpr(MMA_M // rows_per_pass):
        r = cutlass.Int32(p * rows_per_pass) + g_pass  # the padded row 0..15
        g = r & cutlass.Int32(7)  # the g row of the fragment layout (rows g / g + 8)
        half = r >> cutlass.Int32(3)
        if cutlass.const_expr(rm.tile_rows == 1):
            s_row = cutlass.Int32(t * rm.queries_per_tile) + r
            h_row = zero_i
        else:
            s_row = cutlass.Int32((t // rm.tiles_per_query) * rm.queries_per_tile) + (g >> cutlass.Int32(rm.stages))
            h_row = cutlass.Int32(16 * (t % rm.tiles_per_query)) + half + (g & cutlass.Int32(rm.group - 1)) * cutlass.Int32(2)
        valid = (s_row < s_q_i) & (h_row < h_i_i)
        s_c = s_row if s_row < s_q_i else s_q_i - cutlass.Int32(1)
        h_c = h_row if h_row < h_i_i else h_i_i - cutlass.Int32(1)
        src = (
            q_base
            + ((t0 + s_c).to(cutlass.Int64) * q_s0 + h_c.to(cutlass.Int64) * q_s1) * cutlass.Int64(bpe)
            + c.to(cutlass.Int64) * cutlass.Int64(_CHUNK_BYTES)
        )
        dst = sA_tile + (r * cutlass.Int32(rb) + (c ^ ((r & cutlass.Int32(1)) * cutlass.Int32(4))) * cutlass.Int32(_CHUNK_BYTES)).to(cutlass.Int64)
        size = cutlass.Int32(_CHUNK_BYTES) if valid else zero_i
        inline_ptx("cp.async.cg.shared.global [$0], [$1], 16, $2;", write_only_types=[], read_only_args=[dst, src, size])


@cute.jit
def _read_query_frags(sA, tile_elems: cutlass.Int32, g: cutlass.Int32, qd: cutlass.Int32, half: cutlass.Constexpr[int], rb: cutlass.Constexpr[int]):
    """Lane ``(g, q)``'s A fragments of one half (rows ``g`` / ``g + 8``) of an m-tile: granules ``q + 4 j`` of the row (one
    ``ld.shared.v4`` each at the swizzled granule ``c ^ (4 x (row & 1))``) -- the 16 words the 8 k-steps consume at bf16, the 8 words
    the 4 k-steps consume at e4m3."""
    row = cutlass.Int32(half * 8) + g
    swz = (row & cutlass.Int32(1)) * cutlass.Int32(4)
    base = tile_elems + row * cutlass.Int32(rb // 4)
    out = []
    for j in cutlass.range_constexpr(rb // _CHUNK_BYTES // 4):
        c_swz = (qd + cutlass.Int32(4 * j)) ^ swz
        v = sA.load(base + c_swz * cutlass.Int32(_WORDS_PER_CHUNK), vector_size=4, alignment=16).to_elements()
        for x in cutlass.range_constexpr(_WORDS_PER_CHUNK):
            out.append(v[x])
    return tuple(out)


@cute.kernel
def frost_qsa_score_paged(
    mQ: cute.Tensor,  # [T, H_i, D] bf16 or e4m3 (the cache dtype); token / head strides symbolic (Int64), D contiguous
    mKbar: cute.Tensor,  # [num_pages, R, D] in the same dtype; page / row strides symbolic (Int64), D contiguous
    mTable: cute.Tensor,  # [B, max_pages] int32; row stride symbolic (Int64), unit column stride
    mKvLens: cute.Tensor,  # [B] int32, contiguous
    mLogits: cute.Tensor,  # [T, n_blocks_max] fp32; row stride symbolic (Int64), unit column stride
    mCounts: cute.Tensor,  # [T] int32, contiguous
    mBlockLens: Optional[cute.Tensor],  # [T] int32, contiguous, or None (folded out)
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    l2_hint: cutlass.Constexpr[bool],
) -> None:
    """One CTA = chunk ``blockIdx.x`` of ``blocks_per_cta`` compressed rows of sequence ``blockIdx.y``, four independent warps of
    ``keys_per_warp`` rows each (module docstring); ``l2_hint`` = the key copies' form (the module docstring's arm)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
    rm = query_row_map(s_q, h_i)  # the (s, h) -> MMA-row map (host facts, compile-time)
    # the element size is the e4m3 arm's ONE geometry switch (trace time, from the cache operand's element type): 256-B rows / 8 k-steps of
    # m16n8k16 at bf16, 128-B rows / 4 k-steps of m16n8k32 at e4m3 -- every other constant below is a function of it (selection_paged)
    fp8 = cutlass.const_expr(mKbar.element_type == cutlass.Float8E4M3FN)
    bpe = _elem_bytes(mKbar.element_type)
    rb = row_bytes(bpe)  # bytes per compressed / query row
    slot_elems = KEYS_PER_NTILE * rb // 4  # one n-tile of 8 rows in the ring, in Int32 elements
    slot_bytes = KEYS_PER_NTILE * rb
    a_tile_elems = MMA_M * rb // 4  # one m-tile of the padded query tile
    a_tile_bytes = MMA_M * rb
    n_ksteps = k_steps(bpe)  # 8 / 4
    rows_pi = (32 * _CHUNK_BYTES) // rb  # rows per key copy instruction: 2 / 4
    granules = rb // _CHUNK_BYTES  # 16-B granules per row: 16 / 8
    n_tiles = NTILES_PER_UNIT * units_per_warp  # n-tiles of 8 keys per warp
    depth = ring_depth  # n-tiles of keys in flight (cp.async groups) ahead of the MMAs
    ring_slots = min(
        ring_depth + 1, n_tiles
    )  # + the slot being read (the one warp sync per n-tile orders the WAR on the slot reused next); no spare when no mainloop copy exists
    # the two SMEM buffers (declared ahead of every branch): each warp's ring of ring_slots n-tiles of 8 compressed rows, and the
    # CTA's padded (s, h) query tile (16 rows x rb B per m-tile), both written by cp.async and read as fragments
    sK = cutlass.Array(cutlass.Int32, THREADS // 32 * ring_slots * slot_elems, alignment=128, space=cutlass.AddressSpace.smem)
    sA = cutlass.Array(cutlass.Int32, rm.m_tiles * a_tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
    n_groups = rm.m_tiles // rm.tiles_per_query  # query groups per lane at P' >= 2 (one head-sum each)
    has_lens = cutlass.const_expr(mBlockLens is not None)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx & cutlass.Int32(31)
    warp = tidx >> cutlass.Int32(5)
    g = lane >> cutlass.Int32(2)  # the MMA group id: the A / C rows g and g + 8, the B key n = g
    qd = lane & cutlass.Int32(3)  # the thread in the quad: chunks q + 4 j, the C keys 2q and 2q + 1
    bidx, bidy, _ = cute.arch.block_idx()
    chunk_id = cutlass.Int32(bidx)
    b = cutlass.Int32(bidy)
    n_blocks_max = cutlass.Int32(mLogits.shape[1])
    num_pages = cutlass.Int32(mKbar.shape[0])
    four = cutlass.Int64(4)  # int32 / fp32 bytes
    b64 = b.to(cutlass.Int64)
    zero_i = cutlass.Int32(0)
    s_q_i = cutlass.Int32(s_q)
    t0 = b * s_q_i
    k_warp0 = chunk_id * cutlass.Int32(blocks_per_cta(units_per_warp)) + warp * cutlass.Int32(keys_per_warp(units_per_warp))
    # ---- the prologue: every load independent, all issued before any wait (ONE round trip ahead of the first key byte)
    kv_len = ld_global(mKvLens.iterator.toint() + b64 * four, cutlass.Int32)
    tb_row = mTable.iterator.toint() + b64 * cutlass.Int64(mTable.stride[0]) * four
    words = []  # the table word of this lane's key, per n-tile (issued two units ahead of the key loads)
    for i in cutlass.range_constexpr(min(2 * depth, n_tiles)):
        words.append(_page_word(tb_row, k_warp0 + cutlass.Int32(i * KEYS_PER_NTILE) + g, n_blocks_max, rows_per_page))
    # the query tile: every thread copies its granule of two rows per m-tile (one cp.async group for the whole tile)
    q_base = mQ.iterator.toint()
    q_s0 = cutlass.Int64(mQ.stride[0])
    q_s1 = cutlass.Int64(mQ.stride[1])
    sA_bytes = sA.data_ptr().toint()
    for t in cutlass.range_constexpr(rm.m_tiles):
        _copy_query_tile(sA_bytes + cutlass.Int64(t * a_tile_bytes), q_base, q_s0, q_s1, t0, tidx, t, s_q, h_i, rb)
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()
    # ---- the visible counts: n_vis_max (warp-uniform; row s_q - 1 sits at position kv_len - 1) and the counts write of chunk 0
    nv_max = kv_len >> cutlass.Int32(2)  # / BLOCK_SIZE (4); a negative length is clamped below either way
    nv_max = nv_max if nv_max > zero_i else zero_i
    n_vis_max = nv_max if nv_max < n_blocks_max else n_blocks_max
    if (chunk_id == zero_i) & (tidx < s_q_i):
        pos1_t = kv_len - s_q_i + tidx + cutlass.Int32(1)
        nv_t = pos1_t >> cutlass.Int32(2)
        nv_t = nv_t if nv_t > zero_i else zero_i
        nv_t = nv_t if nv_t < n_blocks_max else n_blocks_max
        row64 = (t0 + tidx).to(cutlass.Int64)
        st_global(mCounts.iterator.toint() + row64 * four, nv_t, cutlass.Int32)
        if cutlass.const_expr(has_lens):
            bl = nv_t if nv_t < top_k else top_k
            st_global(mBlockLens.iterator.toint() + row64 * four, bl, cutlass.Int32)
    live_warp = k_warp0 < n_vis_max  # WARP-UNIFORM (the warp's first key, the sequence's visible count): the mainloop's one branch
    # ---- the key stream's first `depth` n-tiles (live warps only), ONE round trip behind the queries' and the table words'
    kb_base = mKbar.iterator.toint()
    kb_s0 = cutlass.Int64(mKbar.stride[0])
    tw = [words[i] for i in range(len(words))]  # the table words (the prologue's, then the ones issued in the mainloop)
    ring_elems = warp * cutlass.Int32(ring_slots * slot_elems)  # this warp's ring, in Int32 elements
    kb_page_bytes = (kb_s0 * cutlass.Int64(bpe)).to(cutlass.Int32)  # the page stride in bytes (the host checks it fits)
    # each lane's copy position inside one copy instruction's rows_pi rows: lane_row = lane / granules (the row, 0 .. rows_pi - 1), granule =
    # lane % granules (bf16: a 2-row pair, row parity = lane / 16, granule = lane % 16; e4m3: 4 rows, lane / 8 and lane % 8) -- the odd rows'
    # granules XOR 4 in SMEM.  The SOURCE row inside a page part is a lane's row within the part: lane_row itself when a part holds whole
    # instructions, lane_row mod the part's rows when an instruction holds several parts (e4m3 at page 8).
    rows_pp = min(rows_per_page, KEYS_PER_NTILE)
    lane_row = lane >> cutlass.Int32(granules.bit_length() - 1)
    c_lane = lane & cutlass.Int32(granules - 1)
    src_lane_off = (lane_row & cutlass.Int32(min(rows_pp, rows_pi) - 1)) * cutlass.Int32(rb) + c_lane * cutlass.Int32(_CHUNK_BYTES)
    dst_lane_off = lane_row * cutlass.Int32(rb) + (c_lane ^ ((lane_row & cutlass.Int32(1)) * cutlass.Int32(4))) * cutlass.Int32(_CHUNK_BYTES)
    dst_lane = sK.data_ptr().toint() + (ring_elems * cutlass.Int32(4) + dst_lane_off).to(cutlass.Int64)  # the cp.async destination base of slot 0
    if cutlass.const_expr(l2_hint):
        # the keys stream through L2 ONCE: an evict-first policy on their copies keeps the logits' write-combining lines and the table (vLLM's
        # scorer marks its key loads the same way) -- where the DSL's compiler encodes the hinted copy (L2_HINT_MIN_COMPILER)
        policy = inline_ptx("createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", write_only_types=[cutlass.Int64], read_only_args=[])
    else:
        policy = cutlass.Int64(0)  # the plain copies take no policy operand: never read
    if live_warp:
        for i in cutlass.range_constexpr(min(depth, n_tiles)):
            _copy_ntile(
                dst_lane + cutlass.Int64((i % ring_slots) * slot_bytes),
                src_lane_off,
                kb_base,
                kb_page_bytes,
                tw[i],
                num_pages,
                k_warp0 + cutlass.Int32(i * KEYS_PER_NTILE),
                policy,
                rows_per_page,
                rb,
                lane_row,
                l2_hint,
            )
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(min(depth, n_tiles))  # this thread's query-tile copies complete (the key groups may still fly)
    else:
        cute.arch.cp_async_wait_group(0)  # a dead warp has only the query-tile group in flight
    cute.arch.barrier()  # the ONE CTA barrier: every thread's query-tile copies visible to every thread's fragment reads below
    # the A fragments from the staged tile: frag_a[t][half] = the 16 words of rows g / g + 8 of m-tile t
    frag_a = []
    for t in cutlass.range_constexpr(rm.m_tiles):
        frag_a.append(
            [_read_query_frags(sA, cutlass.Int32(t * a_tile_elems), g, qd, 0, rb), _read_query_frags(sA, cutlass.Int32(t * a_tile_elems), g, qd, 1, rb)]
        )
    if live_warp:
        # ---- the store entries of this lane: per query group the query row s, its visible count, its logits row, the storing lane
        lg_base = mLogits.iterator.toint()
        lg_s0 = cutlass.Int64(mLogits.stride[0])
        s_rows = []
        if cutlass.const_expr(rm.tile_rows == 1):
            for t in cutlass.range_constexpr(rm.m_tiles):
                for half in cutlass.range_constexpr(2):
                    s_rows.append(cutlass.Int32(t * rm.queries_per_tile + 8 * half) + g)
        else:
            for sg in cutlass.range_constexpr(n_groups):
                s_rows.append(cutlass.Int32(sg * rm.queries_per_tile) + (g >> cutlass.Int32(rm.stages)))
        store_lane = (g & cutlass.Int32(max(rm.group, 1) - 1)) == zero_i  # every lane at P' = 1 (group 0 / 1)
        ent_nv = []
        ent_addr = []
        for e in cutlass.range_constexpr(len(s_rows)):
            s_row = s_rows[e]
            in_step = s_row < s_q_i
            pos1 = kv_len - s_q_i + s_row + cutlass.Int32(1)
            nv = pos1 >> cutlass.Int32(2)
            nv = nv if nv > zero_i else zero_i
            nv = nv if nv < n_blocks_max else n_blocks_max
            nv = nv if in_step else zero_i  # a padding query row never stores
            s_c = s_row if in_step else zero_i
            ent_nv.append(nv)
            ent_addr.append(lg_base + (t0 + s_c).to(cutlass.Int64) * lg_s0 * four)
        # ---- the key stream: the rolling ring (`depth` n-tiles in flight, the table words two units ahead)
        twb = [tw[i] for i in range(len(tw))]  # the branch's own list of table words (appended below; the outer one stays as it is)
        zeros = []
        for _ in cutlass.range_constexpr(4):
            zeros.append(opaque_f32_zero())  # the accumulator seeds (a constant reaching the mma's inline ptx would ICE libNVVM)
        k_col0 = k_warp0 + qd * cutlass.Int32(2)
        mma_batch = 4 if units_per_warp == 1 else 2  # n-tiles whose HMMA chains interleave (register cost 16 per n-tile)
        for i0 in cutlass.range_constexpr(0, n_tiles, mma_batch):
            batch = list(range(i0, min(i0 + mma_batch, n_tiles)))
            kws = []
            for bi0 in cutlass.range_constexpr(len(batch)):
                i = batch[bi0]
                if cutlass.const_expr(i + 2 * depth < n_tiles):
                    twb.append(_page_word(tb_row, k_warp0 + cutlass.Int32((i + 2 * depth) * KEYS_PER_NTILE) + g, n_blocks_max, rows_per_page))
                # group i complete: min(n_tiles, depth + i) groups were committed before this point, the i + 1 oldest must be done
                cute.arch.cp_async_wait_group(min(n_tiles, depth + i) - i - 1)
                cute.arch.sync_warp()  # the copies of the OTHER lanes visible; last step's fragment reads ordered before the slot's reuse below
                kws.append(_read_key_frags(sK, ring_elems + cutlass.Int32((i % ring_slots) * slot_elems), g, qd, rb))
                if cutlass.const_expr(i + depth < n_tiles):
                    _copy_ntile(
                        dst_lane + cutlass.Int64(((i + depth) % ring_slots) * slot_bytes),
                        src_lane_off,
                        kb_base,
                        kb_page_bytes,
                        twb[i + depth],
                        num_pages,
                        k_warp0 + cutlass.Int32((i + depth) * KEYS_PER_NTILE),
                        policy,
                        rows_per_page,
                        rb,
                        lane_row,
                        l2_hint,
                    )
                    cute.arch.cp_async_commit_group()
            # the batch's MMAs: k-step outer, n-tile and m-tile inner -- `len(batch) x m_tiles` independent accumulator chains
            accs = []
            for _ in cutlass.range_constexpr(len(batch)):
                acc = []
                for _ in cutlass.range_constexpr(rm.m_tiles):
                    acc.append([zeros[0], zeros[1], zeros[2], zeros[3]])
                accs.append(acc)
            # the K permutation is dtype-free (every k-step consumes 32 B of a row): the SAME fragment indexing feeds m16n8k16 at bf16 and
            # m16n8k32 at e4m3; only the step count differs (8 / 4)
            for kappa in cutlass.range_constexpr(n_ksteps):
                j = kappa // 2
                r = kappa % 2
                for bi in cutlass.range_constexpr(len(batch)):
                    kw = kws[bi]
                    b0 = kw[4 * j + 2 * r]
                    b1 = kw[4 * j + 2 * r + 1]
                    for t in cutlass.range_constexpr(rm.m_tiles):
                        a_lo = frag_a[t][0]
                        a_hi = frag_a[t][1]
                        if cutlass.const_expr(fp8):
                            c0, c1, c2, c3 = mma_m16n8k32_f32(
                                a_lo[4 * j + 2 * r],
                                a_hi[4 * j + 2 * r],
                                a_lo[4 * j + 2 * r + 1],
                                a_hi[4 * j + 2 * r + 1],
                                b0,
                                b1,
                                accs[bi][t][0],
                                accs[bi][t][1],
                                accs[bi][t][2],
                                accs[bi][t][3],
                                ab_dtype=cutlass.Float8E4M3FN,
                            )
                        else:
                            c0, c1, c2, c3 = mma_m16n8k16_f32(
                                a_lo[4 * j + 2 * r],
                                a_hi[4 * j + 2 * r],
                                a_lo[4 * j + 2 * r + 1],
                                a_hi[4 * j + 2 * r + 1],
                                b0,
                                b1,
                                accs[bi][t][0],
                                accs[bi][t][1],
                                accs[bi][t][2],
                                accs[bi][t][3],
                                ab_dtype=cutlass.BFloat16,
                            )
                        accs[bi][t] = [c0, c1, c2, c3]
            for bi in cutlass.range_constexpr(len(batch)):
                i = batch[bi]
                acc = accs[bi]
                # relu per (s, h) element: c0 (row g, key 2q), c1 (row g, key 2q + 1), c2 (row g + 8, key 2q), c3 (row g + 8, key 2q + 1)
                relu = []
                for t in cutlass.range_constexpr(rm.m_tiles):
                    rt = []
                    for x in cutlass.range_constexpr(4):
                        rt.append(cute.math.max(acc[t][x], 0.0, ftz=True))
                    relu.append(rt)
                k0 = k_col0 + cutlass.Int32(i * KEYS_PER_NTILE)
                k1 = k0 + cutlass.Int32(1)
                # the head sums per store entry
                sums0 = []
                sums1 = []
                if cutlass.const_expr(rm.tile_rows == 1):
                    for t in cutlass.range_constexpr(rm.m_tiles):
                        sums0.append(relu[t][0])
                        sums1.append(relu[t][1])
                        sums0.append(relu[t][2])
                        sums1.append(relu[t][3])
                else:
                    for sg in cutlass.range_constexpr(n_groups):
                        v0 = relu[sg * rm.tiles_per_query][0] + relu[sg * rm.tiles_per_query][2]  # heads 2 gi and 2 gi + 1 of the query, key 2q
                        v1 = relu[sg * rm.tiles_per_query][1] + relu[sg * rm.tiles_per_query][3]  # the same, key 2q + 1
                        for ht in cutlass.range_constexpr(1, rm.tiles_per_query):
                            t = sg * rm.tiles_per_query + ht
                            v0 = v0 + (relu[t][0] + relu[t][2])
                            v1 = v1 + (relu[t][1] + relu[t][3])
                        for st in cutlass.range_constexpr(rm.stages):
                            v0 = v0 + cute.arch.shuffle_sync_bfly(v0, 4 << st)
                            v1 = v1 + cute.arch.shuffle_sync_bfly(v1, 4 << st)
                        sums0.append(v0)
                        sums1.append(v1)
                for e in cutlass.range_constexpr(len(ent_nv)):
                    nv = ent_nv[e]
                    out0 = sums0[e] * scale
                    out1 = sums1[e] * scale
                    addr = ent_addr[e] + k0.to(cutlass.Int64) * four
                    if store_lane & (k1 < nv):
                        st_global_v2(addr, (out0, out1), cutlass.Float32)
                    elif store_lane & (k0 < nv):
                        st_global(addr, out0, cutlass.Float32)


frost_qsa_score_paged.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_score_paged_launch(
    mQ: cute.Tensor,
    mKbar: cute.Tensor,
    mTable: cute.Tensor,
    mKvLens: cute.Tensor,
    mLogits: cute.Tensor,
    mCounts: cute.Tensor,
    mBlockLens: Optional[cute.Tensor],
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    n_chunks: cutlass.Int32,
    batch: cutlass.Int32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    l2_hint: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_score_paged(
        mQ, mKbar, mTable, mKvLens, mLogits, mCounts, mBlockLens, top_k, scale, h_i, s_q, rows_per_page, units_per_warp, ring_depth, use_pdl, l2_hint
    ).launch(grid=(n_chunks, batch, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)


# --------------------------------------------------------------------------------------------- compile + run

compiled_cache = {}


class QsaScorePagedRecipe(NamedTuple):
    """Build-time facts of one artifact (Rule 4: one artifact per (H_i, S_q, page size, units knob, block_lens presence, PDL, ring depth,
    the key copies' L2-hint arm, cache dtype)); the batch, the cache capacity, the table width, the page count and every stride ride in at
    runtime.  ``l2_hint`` is the arm that was BUILT (the resolved value, never ``None``)."""

    compiled: object
    h_i: int
    s_q: int
    page_size: int
    rows_per_page: int
    units_per_warp: int
    has_block_lens: bool
    use_pdl: bool
    device: int
    ring_depth: int = NTILES_IN_FLIGHT
    l2_hint: bool = True  # the arm built before the field existed: the hinted copies
    cache_dtype: object = torch.bfloat16  # appended: the (qi, kbar) pair's dtype -- bf16, or float8_e4m3fn (the e4m3 read arm)


def _fake_q(h_i: int, dtype=torch.bfloat16):
    return cute.runtime.make_fake_tensor(
        dtype=_CUTE_DTYPES[dtype],
        shape=(cute.sym_int(), int(h_i), INDEX_HEAD_DIM),
        stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1),
        assumed_align=16,
    )


def _fake_kbar(rows_per_page: int, dtype=torch.bfloat16):
    return cute.runtime.make_fake_tensor(
        dtype=_CUTE_DTYPES[dtype],
        shape=(cute.sym_int(), int(rows_per_page), INDEX_HEAD_DIM),
        stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1),
        assumed_align=16,
    )


def _fake_i32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=4)


def _fake_f32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=16)


def _fake_i32_1d():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4)


def dsl_decline_reason(device: Optional[int] = None) -> Optional[str]:
    """Rule 7: None when this DSL can build the kernel for this device, else the typed reason -- the version floor first
    (device-free), then the ``sm_107a`` target on cc 10.7.  The bf16 kernel itself (``mma.sync.m16n8k16``, ``cp.async``, shuffles, one
    CTA barrier) runs on any CuTe DSL device from sm_80; PDL adds the cc 9.0 requirement and the e4m3 read arm the sm_89 one
    (:func:`compile_qsa_score_paged`)."""
    too_old = cutedsl_requirement_error("qsa_score_paged")
    if too_old is not None:
        return too_old
    dev = current_device() if device is None else int(device)
    return cutedsl_arch_requirement_error(compute_capability(dev))


def compile_qsa_score_paged(
    *,
    h_i: int,
    s_q: int,
    page_size: int,
    units_per_warp: int = 1,
    has_block_lens: bool = True,
    use_pdl: bool = False,
    device: Optional[int] = None,
    ring_depth: int = NTILES_IN_FLIGHT,
    l2_hint: Optional[bool] = None,
    cache_dtype=torch.bfloat16,
) -> QsaScorePagedRecipe:
    """Build from SHAPES ALONE (no allocation, no launch).  The gates run BEFORE the compile, in this order (Rule 7 first): the DSL
    floor, the ``sm_107a`` target, PDL's cc 9.0, the e4m3 read arm's sm_89, then the ``l2_hint`` arm's compiler floor (a device gate
    before a form's gate) -- a decline is a ``NotImplementedError`` naming the version, the target, the device or the compiler, never an
    error from inside the DSL.  ``ring_depth`` (the n-tiles of keys in flight per warp, ``RING_DEPTH_CHOICES``) is a performance knob:
    the same function; so is ``l2_hint`` (``None`` = the hinted key copies where the DSL's compiler serves them --
    :func:`l2_hint_available` --, else the plain ones; ``True`` / ``False`` force an arm, ``True`` below the floor is the typed decline).
    ``cache_dtype`` (appended) is the ``(qi, kbar)`` pair's dtype -- ``torch.bfloat16`` (the default: the bf16 arm, from sm_80) or
    ``torch.float8_e4m3fn`` (the e4m3 READ arm: ``mma.sync.m16n8k32 .e4m3`` on 128-B rows, sm_89+ -- a typed decline naming the
    instruction below it); a different selection function, its own artifact (the bf16 keys are unchanged)."""
    global _FAKE_STREAM
    validate_params(h_i=h_i, s_q=s_q, page_size=page_size, units_per_warp=units_per_warp, ring_depth=ring_depth, l2_hint=l2_hint)
    if cache_dtype not in CACHE_DTYPES:
        raise ValueError(f"cache_dtype must be one of {CACHE_DTYPES} (the bf16 arm, or float8_e4m3fn for the e4m3 read arm), got {cache_dtype}")
    dev = current_device() if device is None else int(device)
    why = dsl_decline_reason(dev)
    if why is not None:
        raise NotImplementedError(why)
    cc = compute_capability(dev)
    if use_pdl and cc < (9, 0):
        raise NotImplementedError(
            f"use_pdl=True traces the programmatic-dependent-launch pair (griddepcontrol), an instruction of cc 9.0 and later; this device is cc {cc[0]}.{cc[1]}"
        )
    if cache_dtype == _E4M3 and tuple(cc) < _FP8_MMA_MIN_CC:
        raise NotImplementedError(
            f"cache_dtype=torch.float8_e4m3fn (the paged scorer's e4m3 read arm) traces mma.sync.m16n8k32 .e4m3, an instruction of sm_{_FP8_MMA_MIN_CC[0]}{_FP8_MMA_MIN_CC[1]} "
            f"and later (Ada / Hopper / Blackwell / Rubin); this device is cc {cc[0]}.{cc[1]} -- score a bf16 cache there"
        )
    hinted = _resolve_l2_hint(l2_hint)  # after the device gates: a forced hint below the compiler floor is the typed decline
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    rows_per_page = int(page_size) // BLOCK_SIZE
    # the memo key, append-only: the defaults keep their seven-tuple identity (the persistent cache's entries too); a non-default ring depth
    # appends it; an explicit l2_hint appends the ring depth and the arm (a default arm needs no key: it is one per process -- the DSL's
    # compiler -- and the persistent cache names the DSL version in its environment)
    key = (int(h_i), int(s_q), int(page_size), int(units_per_warp), bool(has_block_lens), bool(use_pdl), dev)
    if int(ring_depth) != NTILES_IN_FLIGHT or l2_hint is not None:
        key += (int(ring_depth),)
    if l2_hint is not None:
        key += (bool(l2_hint),)
    if cache_dtype != torch.bfloat16:
        key += (str(cache_dtype),)  # appended: the e4m3 arm is its own artifact, the bf16 keys (and their persistent-cache entries) unchanged
    if key not in compiled_cache:
        # The dict is the in-process memo; behind it sits the persistent compiled-plan cache, keyed by this file's digest
        # (``FROST_SOURCE_DIGEST`` above) + the key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_score_paged")
        with torch.cuda.device(dev):  # the compile and the DSL's load of the launchable run ON the artifact's device, whichever device is current
            compiled_cache[key] = compile_cached(
                qsa_score_paged_launch,
                _fake_q(h_i, cache_dtype),
                _fake_kbar(rows_per_page, cache_dtype),
                _fake_i32_2d(),  # the table [B, max_pages]
                _fake_i32_1d(),  # kv_lens
                _fake_f32_2d(),  # logits [T, n_blocks_max]
                _fake_i32_1d(),  # counts
                _fake_i32_1d() if has_block_lens else None,
                cutlass.Int32(0),  # top_k    ) runtime; the zeros pin the TYPE only
                cutlass.Float32(0.0),  # scale    )
                cutlass.Int32(0),  # n_chunks )
                cutlass.Int32(0),  # batch    )
                int(h_i),
                int(s_q),
                int(rows_per_page),
                int(units_per_warp),
                int(ring_depth),
                bool(use_pdl),
                bool(hinted),
                _FAKE_STREAM,
                options="--enable-tvm-ffi",
                cache_key=cache_key,
                symbol="cudnn_block_qsa_score_paged",
            )
    return QsaScorePagedRecipe(
        compiled=compiled_cache[key],
        h_i=int(h_i),
        s_q=int(s_q),
        page_size=int(page_size),
        rows_per_page=rows_per_page,
        units_per_warp=int(units_per_warp),
        has_block_lens=bool(has_block_lens),
        use_pdl=bool(use_pdl),
        device=dev,
        ring_depth=int(ring_depth),
        l2_hint=bool(hinted),
        cache_dtype=cache_dtype,
    )


def run_qsa_score_paged(
    r: QsaScorePagedRecipe,
    qi: torch.Tensor,
    kbar: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_lens: Optional[torch.Tensor],
    *,
    top_k: int,
    scale: float,
    stream,
) -> None:
    """Validate the FORM of every operand (no device read, Rule 3) and launch (Rule 1: no conversion, no allocation).

    ``qi [T, H_i, D]`` in the artifact's cache dtype (bf16, or e4m3 under the e4m3 read arm) with a unit last stride and 16-byte-aligned
    token / head strides (a column slice of a wider slab reads in place), ``T = B x s_q``; ``kbar`` the paged cache in the same dtype
    (:func:`_kbar_rows_view`); ``block_table [B, max_pages]`` int32
    with a unit column stride, ``max_pages x R >= n_blocks_max``; ``kv_lens [B]`` int32; ``logits [T, n_blocks_max]`` fp32 with a
    unit column stride and 16-byte-aligned rows; ``counts [T]`` int32; ``block_lens [T]`` int32 iff the artifact traces it;
    ``top_k`` in ``[1, 2048]``; ``scale > 0`` (the ordering is scale-free); ``stream`` a raw CUDA stream handle (an ``int`` or a
    ``cuda.CUstream``; the host entry resolves a ``torch.cuda.Stream`` to it).  The operands must live on the artifact's device
    (``r.device``, typed); the launch selects that device whichever device is current at the call and restores the caller's."""
    if not isinstance(qi, torch.Tensor) or qi.dim() != 3:
        raise ValueError(f"qi must be a [T, H_i, {INDEX_HEAD_DIM}] {r.cache_dtype} tensor, got {getattr(qi, 'shape', type(qi).__name__)}")
    if qi.dtype != r.cache_dtype:
        raise ValueError(
            f"qi must be {str(r.cache_dtype).replace('torch.', '')} for this artifact (its cache_dtype: the bf16 arm, or float8_e4m3fn for the e4m3 read "
            f"arm -- no silent cast), got {qi.dtype}"
        )
    if not qi.is_cuda:
        raise ValueError(f"qi must be a CUDA tensor (every operand is read through a device pointer), got device {qi.device}")
    device = qi.device
    t, h_i, d = (int(x) for x in qi.shape)
    if h_i != r.h_i or d != INDEX_HEAD_DIM:
        raise ValueError(f"qi must be [T, H_i={r.h_i}, {INDEX_HEAD_DIM}] (H_i is fixed per artifact), got {tuple(qi.shape)}")
    esz = cache_elem_bytes(r.cache_dtype)
    qs0, qs1, qs2 = (int(x) for x in qi.stride())
    if qs2 != 1 or (qs0 * esz) % 16 or (qs1 * esz) % 16 or qi.data_ptr() % 16:
        raise ValueError(
            f"qi needs a unit last stride and 16-byte-aligned token / head strides and base, got strides {tuple(qi.stride())}, data_ptr {qi.data_ptr():#x}"
        )
    if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2 or block_table.dtype != torch.int32:
        raise ValueError(
            f"block_table must be an int32 [B, max_pages] tensor, got {getattr(block_table, 'shape', type(block_table).__name__)} {getattr(block_table, 'dtype', '')}"
        )
    batch, max_pages = (int(x) for x in block_table.shape)
    if batch < 1 or max_pages < 1:
        raise ValueError(f"block_table must hold at least one sequence and one page column, got {tuple(block_table.shape)}")
    if int(block_table.stride(1)) != 1 or block_table.device != device:
        raise ValueError(f"block_table needs a unit column stride on {device}, got strides {tuple(block_table.stride())} on {block_table.device}")
    if t != batch * r.s_q:
        raise ValueError(
            f"qi holds T = {t} rows but block_table / kv_lens hold B = {batch} sequences of s_q = {r.s_q} rows (T must be B x s_q = {batch * r.s_q})"
        )
    _check_i32_1d(kv_lens, "kv_lens", batch, device)
    kbar_view = _kbar_rows_view(kbar, r.rows_per_page, device, dtype=r.cache_dtype)
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.dtype != torch.float32:
        raise ValueError(
            f"logits must be an fp32 [T, n_blocks_max] tensor, got {getattr(logits, 'shape', type(logits).__name__)} {getattr(logits, 'dtype', '')}"
        )
    lt, n_blocks_max = (int(x) for x in logits.shape)
    if lt != t or n_blocks_max < 1:
        raise ValueError(f"logits must be [T={t}, n_blocks_max >= 1], got {tuple(logits.shape)}")
    ls0, ls1 = (int(x) for x in logits.stride())
    if ls1 != 1 or (ls0 * 4) % 16 or logits.data_ptr() % 16 or logits.device != device:
        raise ValueError(
            f"logits needs a unit column stride, 16-byte-aligned rows and base, on {device}; got strides {tuple(logits.stride())}, data_ptr {logits.data_ptr():#x}, {logits.device}"
        )
    if max_pages * r.rows_per_page < n_blocks_max:
        raise ValueError(
            f"block_table maps max_pages x R = {max_pages} x {r.rows_per_page} = {max_pages * r.rows_per_page} compressed rows, fewer than the "
            f"n_blocks_max = {n_blocks_max} columns of logits: widen the table or shrink the logits"
        )
    _check_i32_1d(counts, "counts", t, device)
    if r.has_block_lens != (block_lens is not None):
        raise ValueError(f"this artifact was compiled {'with' if r.has_block_lens else 'without'} block_lens; the call must match (Rule 1: no silent fallback)")
    if block_lens is not None:
        _check_i32_1d(block_lens, "block_lens", t, device)
    if not 1 <= int(top_k) <= TOP_K_KERNEL_MAX:
        raise ValueError(f"top_k must be in [1, {TOP_K_KERNEL_MAX}], got {top_k}")
    if not float(scale) > 0.0:
        raise ValueError(f"scale must be positive (the ordering is scale-free), got {scale}")
    if device.index is not None and device.index != r.device:
        raise ValueError(f"this artifact was compiled for CUDA device {r.device}; the operands live on {device} (compile one per device)")
    n_chunks = _ceil_div(n_blocks_max, blocks_per_cta(r.units_per_warp))
    with torch.cuda.device(r.device):  # the launch runs in the artifact's context whichever device is current at the call
        r.compiled(
            qi,
            kbar_view,
            block_table,
            kv_lens,
            logits,
            counts,
            block_lens,
            cutlass.Int32(int(top_k)),
            cutlass.Float32(float(scale)),
            cutlass.Int32(n_chunks),
            cutlass.Int32(batch),
            cuda.CUstream(int(stream)),
        )
