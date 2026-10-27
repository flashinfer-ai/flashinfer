# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Adapter of the Rubin (SM107) index-list SPARSE d256 forward -- ``kernels/sm107/sparse_d256_f16.py``.

Frontend-only: the gated attention block's sparse stage (``gated_attention_block.api._SparseSdpa``, a geometry declared
with ``QsaSpec``; dense ``[B, S]`` or packed ``thd=True``) consumes it, and it stands alone for a caller with its own Q / K / V;
no graph form carries a block-index list, so there is NO engine row and NO manifest slot (an ``EngineSpec`` whose ``lower`` cannot run is a contract break).  The kernel's claims
live in ONE frozen record, :data:`SPARSE_CAPABILITIES`, spelled in the ``Capabilities`` vocabulary where a field exists, and
:meth:`SparseGqaFwdDslSm107.check_support` is the ENFORCEMENT point: the CuTe DSL version gate first (``sm_107a`` needs the
public 4.8.0 wheel; ``python/cudnn/AGENTS.md`` Rule 7 -- BEFORE the kernel module is imported, so a too-old DSL reads as a
version problem, never as a ``KeyError`` from inside the DSL), then every field of the record as a typed decline.  Every
arm the body does not carry yet (a sink, a band, Q-length trimming of a dense batch, log2 stats, a pure caller list) is declined
BY NAME; a later commit that lands an arm flips the record field, the config's wired-arm set and the support-matrix tracker in
the same change.  THD (packed sequences) and paged K / V pools are carried --
ONE of the two per declaration (``thd=True`` with ``paged_kv=True`` is a typed decline: the packed sequence's K / V row offset
composes with a dense tensor only, never with a page pool) -- and so is the fused epilogue gate, which composes with either
(it touches the item's Q^T slot ring and the epilogue only); see the contracts below.

The fused epilogue gate (``epilogue_gate=`` the gate OPERAND, O's ``(B, S, H_q, D)`` shape in Q's dtype at its own (batch,
seq, head) strides -- a slab column slice binds with no copy; the packed ``[1, T_q, H_q, D]`` form under THD) is served: the
kernel stages the item's gate tile into its freed Q^T slot and writes ``O * sigmoid(G)`` in place of O (``h * tanh(g / 2) + h``
on the fp32 value, the dead-row SELECT per element after it; the LSE is untouched).  Its presence is a MODULE specialization
(``TemplateParams.epilogue_gate``), so a gated and an ungated adapter are two compiled artifacts; a request without the
operand compiles the ungated kernel.

Paged K / V (``paged_kv=True``, the serving read): ``k`` / ``v`` are page POOLS ``[num_pages, H_kv, page_size, D]`` -- HND
compact, or NHD storage declared through the strides (the dense SDPA adapter's own paged contract) -- addressed through ONE
``block_table`` ``(B, max_pages)`` int32 (contiguous; the same table serves K and V), with ``seq_kv_lens`` REQUIRED as the
per-batch logical KV length (the pool has no dense extent: it is what bounds the visible range and what makes the rows of a
partially filled last page -- another sequence's tokens, or garbage -- never reach an MMA).  ``page_size`` must be a positive
multiple of 4 (a 4-token block never straddles a page, so the kernel does one table lookup per block); a ``-1`` table entry
and a page index past the table read as zero rows.  The kernel derives per operand from the strides: rows per page (page
stride / token stride), rows per head (head stride / token stride for an HND pool, 0 for an NHD pool whose head lives in the
column coordinate) and the column head stride -- one compiled kernel serves both layouts.  The pools' dtype is the
activation dtype (``kv_cache_dtypes``: bf16 / f16 with a bf16 / f16 Q); the block's paged-READ mode (``block_table`` /
``kv_lens`` through the gated attention block) lands with the decode form.

The DECODE FORM (``bottom_right=True`` + ``list_per_sequence=True`` [+ ``split_kv=S``]; the same kernel specialised by three
``TemplateParams`` arms, dense BSHD or paged pools, never THD): the step's ``S_q <= 4`` rows per sequence sit at the END of the
sequence (``bottom_right``: row ``j``'s position is ``kv_len_b - S_q + j``; a sequence shorter than the query count makes the
rows with a negative position dead), share ONE block list (``list_per_sequence``: ``block_ids`` int32 ``[B, top_k]``,
``block_lens`` ``[B]`` or None) anchored at the STEP-0 position ``kv_len_b - S_q`` -- the count comes from it and the appended tail
runs from the step-0 tail start to the row's own position (the block completed between step 0 and the row stays visible, its
own token included; the MTP shared-list semantics) -- and ``split_kv=S`` cuts every item's tiles into ``S`` chunks, each its own
work item writing fp32 partials into the caller's ``workspace``, reduced by ``sm100/split_combine.py`` into O / LSE as a second
launch on the same stream; a split with the gate operand compiles the UNGATED kernel and the GATED combine (``O *= sigmoid(G)``
on the fp32 merged value: ONE rounding, the fused epilogue's convention).  ``bottom_right=True`` is also served with per-token
lists (``[B, S_q, top_k]``: row ``j``'s list at its bottom-right position).  ``S`` is bounded by the item's maximum tile count
``ceil((top_k + 1) / 32)`` (a larger split has an empty chunk on every item); ``list_per_sequence`` needs ``bottom_right`` and
``S_q <= 4`` (the shared list's tail covers at most two blocks).

The split combine's FORM is a PLAN-TIME pin on the sparse grid ``(B, H_q, S_q)`` (appended 2026-10-21): the adapter compiles the
``sm100/split_combine`` entry with ``decode_grid=`` :func:`_sparse_combine_decode_grid` -- the dense adapter's rule
(``prepared._split_combine_decode_grid``) applied to the sparse launch's combine grid, which reduces the same split-major fp32
``(B x S, S_q, H_q, D)`` + ``(B x S, H_q, S_q)`` partial slabs the dense entry reads.  On cc 10.7 (the one line the rule was measured
on): the DECODE-GRID kernel inside the lean-grid bound at a split of at least 8 (``heuristics.decode_combine_grid_selected``: B 1 /
B 4 x 24 heads at split 8 -- the serving decode cells; flat in the partial count), the LEAN GROUP WALK on a lean-grid-overflow launch at
a split of at most 4 (``heuristics.decode_combine_lean_walk_selected``: B 32 x 24 heads x S_q 1), the module's own forms otherwise
(``DECODE_GRID_OFF`` where the rule pins them inside the bound -- B 4 x S_q 4 at split 7, any small grid below split 8 --, the AUTO entry
past the bound, which is the module's own forms there by construction); off cc 10.7 every launch keeps the module's own forms, as
before.  A PERFORMANCE pin only (the same function): the output is BITWISE the former entry's up to 16 splits -- the forms share one
arithmetic there, the combine module's order statement -- and within fp32 rounding past 16 (the decode-grid kernel merges its eight
chain partials with ``exp(m_c - m)`` factors; the final O rounds once either way).  :meth:`SparseGqaFwdDslSm107.combine_form` reads the
pinned value back.  ``use_pdl`` (appended, default False) is the programmatic-dependent-launch form of the chain's two launches -- a
PERFORMANCE knob, the same function bitwise: the sparse core launches with the programmatic-stream-serialization attribute and carries
its ``griddepcontrol`` sites (every role's wait before its first dependent global read; no explicit release -- the CTA exit triggers the
dependent grid at the core's completion, the measured placement of the kernel header's PDL SITES paragraph), and the combine's entry
takes the switch too (``compile_ptr(use_pdl=)``: every form's pair).  THD x
``use_pdl`` is a typed decline by name (:meth:`SparseGqaFwdDslSm107.check_support`; the text is :func:`pdl_decline_reason`'s, the ONE
statement of the fact that the gated attention block's declaration-time decline carries too): the packed form reads the setup launch's
metadata above every role wait (the pre-dispatch dead-CTA exit, the claim-counter scheduler), so a dependent launch of it would read
what the setup launch has not written yet; the switch's scope is the dense BSHD and paged decode chain.

TWO BODIES, ONE FUNCTION (``tile_form``, appended; a PERFORMANCE knob): the tcgen05 core (``kernels/sm107/sparse_d256_f16.py``, the
module above, ``kernel_template`` ``"sparse_d256_f16"``) and the LEAN decode tile (``kernels/sm107/sparse_lean_d256_f16.py``: ``mma.sync``
m16n8k16 + ``cp.async``, one 128-thread CTA per (sequence, KV head, chunk of ``sub_tiles_per_cta`` 32-key sub-tiles), no TMEM / TMA / CLC,
``kernel_template`` ``"sparse_lean_d256_f16"``) compute the same O / LSE -- within the fixed-order-sum budget (a 4 x 64-d split-K sum vs
one 256-d MMA chain, 32- vs 128-key tiles: never bitwise ACROSS the bodies, bitwise WITHIN each), so a caller may pin either.  The lean
body serves the DECODE form only (``bottom_right`` + the shared list at ``S_q <= config_sm107.LEAN_SPARSE_SERVED_MAX_S_Q`` = 4 -- every
row of the step in ONE CTA: ``n_mt = ceil(S_q x G / 16)`` m16 tiles of 128 threads over one gathered sub-tile, where the tcgen05 core runs
every row as its own item; the adapter builds the lean tile's own record ``config_sm107.TemplateParamsLean(decode_seqlen_q=S_q, sub_tiles_per_cta=k)``, the
compile-time S_q axis -- or the per-row list at ``S_q == 1``, which IS the shared list; per-row lists past one row are the core's --,
``seq_kv_lens`` present, dense BSHD or paged pools, bf16 / f16, G <= 16,
``top_k`` in [8, 512]) and ALWAYS splits: its ``split_kv`` is a LEAN split ``ceil(n_sub / k)`` (``heuristics.sparse_lean_splits``: 65 / 33 /
22 / 17 / ... at ``top_k`` 512) that SELECTS ``k``, the sub-tiles per CTA (a pinned ``split_kv`` must be one of them; ``split_kv`` unset =
the default ``k`` of ``heuristics.sparse_lean_sub_tiles_per_cta``, 1 today; ``sub_tiles_per_cta`` pins ``k`` directly; an explicit ``split_kv=1`` reads as UNSET on the adapter -- 1 is the constructor's default, so the two
cannot be told apart -- and under ``tile_form="auto"`` on a listed part takes the lean split: an unsplit launch there is pinned with
``tile_form="tcgen05"``, where the block's ``decode_split_kv`` has a None default and keeps the core for an explicit 1), writing the SAME fp32
split-major partial slab the tcgen05 core's split store writes, so the combine, its pinned form and the gate riding it are untouched.
``tile_form="auto"`` (the default) resolves through ``heuristics.sparse_tile_form_auto``: on the measured part
(``heuristics.SPARSE_LEAN_TILE_DEFAULT_CCS`` = cc 10.7 since the gate's table) the placement rule
``heuristics.sparse_decode_lean_tile_selected`` (``B x H_kv <= LEAN_SPARSE_MAX_UNITS`` = 4 CTAs per chunk -- the lean grid's own unit: the
step's S_q rows ride inside the CTA, so B 1 and B 2 on two KV heads take the lean body at every served S_q 1..4), the tcgen05 core on every
other part; an explicit ``"lean"`` is honoured where the body serves the declaration and a typed decline naming the arm where it does not (``heuristics.sparse_lean_tile_decline``:
THD, a prefill / top-left form, no ``seq_kv_lens``, ``S_q`` past the served rows, per-row lists past one row, a group past 16, ``top_k`` below
8, a non-default ``softmax_groups``), never a silent fallback; ``sub_tiles_per_cta`` is the lean body's pin and is refused by name where the
resolved body is the tcgen05 core.  :attr:`SparseGqaFwdDslSm107.tile_form` / :attr:`kernel_template` read the resolved body back after
``check_support`` (a route is pinned by the kernel that compiled, never transcribed); the claims record gains NO field (the function is
the same).

The device does the work (Rule 3): lengths, counts, dead items and the tail block(s) are derived on device from the position
and the per-row ``block_lens``; nothing below reads device memory.  ``execute`` validates and launches (Rule 1): no
conversion, no allocation, the caller's stream.  The kernel needs NO GMEM scratch unless split or THD
(:meth:`SparseGqaFwdDslSm107.scratch_workspace_bytes`: 0, the split partial slabs, or the THD metadata).

Two ways to bind the operands.  A standalone caller constructs with its tensors and calls ``execute()``; a caller whose
buffers exist only at execute (the block's workspace views) declares every operand as a :class:`SparseOperandDesc` -- the
shape, the strides, the dtype, the device -- and hands the tensors to ``execute(q=, k=, ...)``, where each must match its
declaration exactly (the specialization and the stride contract were validated on the declaration).  ``block_lens`` is the
one operand whose PRESENCE may differ per call: ``compile(has_block_lens=...)`` builds either variant at plan time and
``execute`` dispatches on the tensor it is handed -- never a compile on the execute path.

Stride contract (the gather map is TWO-dimensional: the tokens of EVERY batch are its rows, the token's full row its
columns): the head dim contiguous; the head stride a multiple of 64 elements (a column box start); the token stride a
multiple of 8 elements (TMA's 16-byte rule); the K / V batch stride EQUAL to ``S_kv x token stride`` (or B == 1) -- a padded
batch stride has no 2-D row coordinate; Q / O any dense (B, S, H) strides.  ``block_ids``: int32 ``[B, S_q, top_k]`` (or
``[B x S_q, top_k]``) CONTIGUOUS -- the row stride is the bulk-copy length; ``block_lens``: int32 ``[B, S_q]`` / ``[B x S_q]``
contiguous or None; ``seq_kv_lens``: int32 ``[B]`` contiguous or None (its presence is a compile-time specialization).

THD contract (``thd=True``; packed sequences, the kernel's persistent claim-counter form): Q / O ``[1, T_q, H, D]`` and K / V
``[1, T_kv, H_kv, D]`` with batch extent 1 (``T_q`` / ``T_kv`` are CAPACITIES >= 1: the live totals are device values; the
rows past them are never touched), the same per-operand stride rules; ``block_ids`` int32 ``[T_q, top_k]`` / ``[1, T_q, top_k]``
with every id RELATIVE TO ITS SEQUENCE (block ``j`` of sequence ``b`` = its tokens ``[4 j, 4 j + 4)``), ``block_lens`` ``[T_q]``
/ ``[1, T_q]`` or None, LSE ``(1, H, T_q)`` in any strides; ``seq_q_lens`` AND ``seq_kv_lens`` REQUIRED: int32 contiguous
per-sequence lengths ``[B]`` -- or cumulative ``[B + 1]`` prefix sums with ``cu_seq_q_lens`` / ``cu_seq_kv_lens`` (normalised on
device: a prefix sliced from a larger one means the same lengths) -- the sequence count ``B`` is their shape; and a
``workspace`` of :meth:`SparseGqaFwdDslSm107.scratch_workspace_bytes` bytes (16-byte aligned, on the device) for the THD
metadata the setup launch writes.  Semantics per sequence ``b`` with Q length ``S_q_b`` and KV length ``S_kv_b``: row ``pos``
sees ``n_vis = min(pos + 1, S_kv_b)`` tokens (top-left causal); ``S_kv_b = 0`` makes every row of the sequence dead (``O = 0``,
``LSE = -inf`` stored); ``S_q_b = 0`` contributes no work; a one-sequence packing computes the dense ``B = 1`` function.
Lengths are device values (Rule 3): the host validates the tensors' FORM only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

KERNEL_FILE = "sm107/sparse_d256_f16.py"
KERNEL_FILE_LEAN = "sm107/sparse_lean_d256_f16.py"  # the second body (tile_form="lean"); the vocabulary and the kernel_template names are the heuristics'
_TOPK_MIN, _TOPK_MAX = 4, 512
_BLOCK_SIZE = 4
_D = 256
_GQA_GROUP_MAX = 16
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class SparseCapabilities:
    """What the sparse core serves -- the adapter-owned claims record (no engine row: see the module docstring).  Field names
    follow ``engines.Capabilities`` where the meaning matches; the index-list fields are this record's own."""

    sm_lo: int = 107
    sm_hi: int = 119
    phase: str = "forward"
    d_shapes: frozenset = frozenset({(_D, _D)})
    d_pad_multiple: int = 0  # exact native shape only: the gather box count and the Q box are derived for d = 256
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    kv_cache_dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # the paged pools' dtype = Q's (reached through paged_kv)
    causal: bool = True  # the in-block mask: key <= the token's position (top-left, or bottom-right)
    bottom_right: bool = True  # pos = kv_len_b - S_q + row: the decode / verify rows at the END of the sequence; dense BSHD or paged, never THD
    swa: bool = False
    sink: bool = False
    padded: bool = False  # per-batch Q lengths (the dense padding mask) are not carried ...
    kv_lens: bool = True  # ... per-batch KV lengths are (seq_kv_lens: the tail follows the visible range, a 0 length is a dead row)
    # thd and paged_kv are served ONE AT A TIME -- a declaration with both is a typed decline (check_support by name, the config validator's
    # predicate, the tracker's gaps row): the two fields below read as two arms, never as their composition.
    thd: bool = True  # packed sequences: cu_seqlens / per-sequence lengths, sequence-relative block ids, the persistent claim counter
    cu_seq_len: bool = True  # ... the lengths as [B + 1] prefix sums (cu_seq_q_lens / cu_seq_kv_lens, normalised on device) -- reached through thd
    paged_kv: bool = True  # page pools [num_pages, H_kv, page_size, D] (HND compact / NHD by strides) through a (B, max_pages) table; page_size % 4 == 0
    decode: bool = (
        True  # the sparse DECODE FORM: bottom_right + list_per_sequence [+ split_kv] -- the standalone adapter, and the gated attention block's decode mode since 2026-10-14
    )
    epilogue_gate: bool = (
        True  # O * sigmoid(G) in the epilogue; G in O's shape, Q's dtype, own strides; a module specialization; composes with thd / paged_kv; under a split it rides the combine
    )
    split_kv: bool = (
        True  # split_kv = S in [2, ceil((top_k + 1) / 32)]: fp32 partials in the caller's workspace + the split combine (dense or paged, never THD)
    )
    stats_log2: bool = False
    pack_gqa: bool = True  # the work item IS the token's packed query-head group
    index_block_sizes: frozenset = frozenset({_BLOCK_SIZE})
    index_top_k_min: int = _TOPK_MIN
    index_top_k_max: int = _TOPK_MAX  # any multiple of 4 in [4, 512] is served
    index_gqa_group_min: int = 1
    index_gqa_group_max: int = _GQA_GROUP_MAX
    open_block_appended: bool = True  # the kernel appends the open tail block from the position (not a public switch)
    index_list_per_sequence: bool = (
        True  # the decode / MTP form: block_ids [B, top_k] shared by the sequence's rows, anchored at the step-0 position (needs bottom_right)
    )
    index_shared_list_max_tokens: int = 4  # S_q cap under list_per_sequence: the shared list's tail covers at most two blocks for S_q <= 4 (the MTP cap)
    cutedsl_min_version: Tuple[int, int, int] = (4, 8, 0)


SPARSE_CAPABILITIES = SparseCapabilities()

_DTYPE_CODE = {"torch.bfloat16": 2, "torch.float16": 3}
_DTYPE_PUBLIC = {"torch.bfloat16": cudnn.data_type.BFLOAT16, "torch.float16": cudnn.data_type.HALF}


@dataclass(frozen=True)
class SparseOperandDesc:
    """An operand DECLARED without storage: what a caller hands :class:`SparseGqaFwdDslSm107` at construction when its
    buffers exist only at execute (the gated attention block's workspace views).  Carries exactly what ``check_support``
    reads of a tensor -- the shape, the strides, the framework dtype (compared through ``str()``, like a tensor's) and the
    device -- and refuses to be LAUNCHED (no ``data_ptr``: ``execute`` requires the tensor, by name, never a dereference of
    nothing)."""

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: object
    device: object = None

    def stride(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.strides)

    def is_contiguous(self) -> bool:
        expect = 1
        for n, st in zip(reversed(tuple(self.shape)), reversed(tuple(self.strides))):
            if int(n) != 1 and int(st) != expect:
                return False
            expect *= max(int(n), 1)
        return True

    def data_ptr(self):
        raise ValueError(
            "sparse d256 forward (sm107): this operand was declared as a SparseOperandDesc (no storage); pass the tensor to execute(q=, k=, v=, o=, ...)"
        )


_SMS_CACHE: dict = {}


def _device_sm_count(device) -> int:
    """``multi_processor_count`` of ``device``, resolved once per device index (the query costs ~7 us; the THD grid is a
    plan-time fact of the device, so the first execute pays it and the cache serves every later call)."""
    import torch  # the framework is the caller's; imported here, never at module import

    key = getattr(device, "index", None)
    key = torch.cuda.current_device() if key is None else int(key)
    n = _SMS_CACHE.get(key)
    if n is None:
        n = int(torch.cuda.get_device_properties(key).multi_processor_count)
        _SMS_CACHE[key] = n
    return n


def _shape(t) -> Tuple[int, ...]:
    return tuple(int(x) for x in t.shape)


def _strides(t) -> Tuple[int, ...]:
    return tuple(int(x) for x in t.stride())


def _require_16b_base(name: str, t) -> None:
    """The launch binds ``t`` with ``assumed_align=16`` (a TMA tensor map's base, the bulk-copy source of an ids row), so a view that
    starts off a 16-byte boundary (a storage offset of a few elements) is refused by name here, never handed to the hardware with a
    false alignment promise.  A storage-free :class:`SparseOperandDesc` (or any shape-only stand-in without ``data_ptr``) has no
    address yet: nothing to check until execute, where every bound tensor is real."""
    if t is None or isinstance(t, SparseOperandDesc) or getattr(t, "data_ptr", None) is None:
        return
    ptr = int(t.data_ptr())
    if ptr % 16 != 0:
        raise ValueError(f"sparse d256 forward (sm107): {name} must start at a 16-byte aligned address (the TMA / bulk-copy base rule); got data_ptr=0x{ptr:x}")


def _require_cuda_device(name: str, t, expected=None, *, allow_unbound: bool = False):
    """Validate storage residency without reading its values; declarations may omit storage and its device."""
    if t is None:
        return expected
    device = getattr(t, "device", None)
    if allow_unbound and device is None and (isinstance(t, SparseOperandDesc) or getattr(t, "data_ptr", None) is None):
        return expected
    import torch

    if device is None or torch.device(device).type != "cuda":
        raise ValueError(f"sparse d256 forward (sm107): {name} must be CUDA storage; got device {device}")
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else int(index)
    if expected is not None and index != expected:
        raise ValueError(f"sparse d256 forward (sm107): {name} must be on cuda:{expected}, the execution device; got cuda:{index}")
    return index


def _paged_pool_geometry(name: str, pool, KH: int, page_size: int) -> Tuple[int, int, int, int, int]:
    """The gather map's view of one page pool ``[num_pages, H_kv, page_size, D]`` from its strides: ``(page_stride,
    token_stride, col_head_stride, rows_per_page, rows_per_head)`` in elements / rows.  The map's rows are the pool's token
    rows at the TOKEN stride (the page_size axis'), so the page stride must be a whole number of token rows and every head
    either folds into the row (HND: the head stride a multiple of the token stride, ``rows_per_head = head_stride /
    token_stride``, column head stride 0) or lives in the column (NHD: the head stride below the token stride, a multiple of
    64 elements, the heads' columns inside one token row).  A ``ValueError`` names the stride that fits neither."""
    num_pages, _KH, P, _D = _shape(pool)  # the pool's head count IS KH (the caller derived it from this shape; V's shape equals K's)
    s_page, s_head, s_tok, _ = _strides(pool)
    if s_tok % 8 != 0:
        raise ValueError(f"sparse d256 forward (sm107): {name} pool token stride must be a multiple of 8 elements (TMA's 16-byte rule); got {s_tok}")
    if s_page <= 0 or s_page % s_tok != 0:
        raise ValueError(
            f"sparse d256 forward (sm107): {name} pool page stride {s_page} must be a positive multiple of the token stride {s_tok} (whole rows per page)"
        )
    rows_per_page = s_page // s_tok
    if rows_per_page < P:
        raise ValueError(f"sparse d256 forward (sm107): {name} pool page stride {s_page} holds {rows_per_page} rows of {s_tok} but a page has {P} tokens")
    if s_head >= s_tok and s_head % s_tok == 0:
        rows_per_head, col_head = s_head // s_tok, 0  # HND: the head is a row term
        if (KH - 1) * rows_per_head + P > rows_per_page:
            raise ValueError(f"sparse d256 forward (sm107): {name} pool heads at {rows_per_head} rows apart do not fit the page's {rows_per_page} rows")
    elif s_head < s_tok:
        rows_per_head, col_head = 0, s_head  # NHD: the head is a column term
        if s_head % 64 != 0:
            raise ValueError(f"sparse d256 forward (sm107): {name} pool head stride must be a multiple of 64 elements (a 128-B column box start); got {s_head}")
        if (KH - 1) * s_head + _D > s_tok:
            raise ValueError(f"sparse d256 forward (sm107): {name} pool heads at {s_head} elements apart do not fit the token row of {s_tok}")
    else:
        raise ValueError(
            f"sparse d256 forward (sm107): {name} pool head stride {s_head} is neither a multiple of the token stride {s_tok} (HND) nor below it (NHD)"
        )
    if num_pages * rows_per_page > _INT32_MAX or (KH - 1) * col_head + _D > _INT32_MAX:
        raise NotImplementedError("sparse d256 forward (sm107): a pool extent exceeds the Int32 coordinate range")
    return s_page, s_tok, col_head, rows_per_page, rows_per_head


def _sparse_combine_decode_grid(b: int, h_q: int, s_q: int, split: int, device_cc) -> int:
    """The split combine entry's kernel FORM for a sparse plan (``split_combine.DECODE_GRID_AUTO`` / ``DECODE_GRID_OFF`` /
    ``DECODE_GRID_LEAN_WALK``): a plan-time specialization on the sparse grid ``(B, H_q, S_q)`` -- the REAL query-head count, the
    rows the combine reduces --, the split and the plan's DEVICE, never a runtime value.  The dense adapter's rule
    (``prepared._split_combine_decode_grid``) with the device passed explicitly (the sparse adapter records ``device_cc`` or reads
    its Q tensor's capability): on cc 10.7 a lean-grid-OVERFLOW launch at a split of at most
    ``heuristics.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT`` names the module's LEAN GROUP WALK (``decode_combine_lean_walk_selected``: B 32 x
    24 heads x S_q 1 at split <= 4), inside the bound the AUTO entry takes the decode-grid kernel on every sink-free launch and the
    split-aware rule pins the module's own forms where the two disagree (``decode_combine_force_lean``: a split below 8, or a mid-sized
    grid below 32 -- B 4 x S_q 4 at split 7), a deeper split past the bound keeps the AUTO entry (= the module's own forms there by
    construction).  Off cc 10.7 (the Blackwell line keeps the module's own three-way rule, measured on B200 / B300; a line the rules
    were never measured on likewise) a plan never names the walk and never compiles the AUTO entry inside the bound."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import decode_combine_force_lean, decode_combine_lean_walk_selected
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100 import split_combine

    cc = (int(device_cc[0]), int(device_cc[1]))
    if decode_combine_lean_walk_selected(b, h_q, s_q, split, device_cc=cc):
        return split_combine.DECODE_GRID_LEAN_WALK
    if decode_combine_force_lean(b, h_q, s_q, split, device_cc=cc):
        return split_combine.DECODE_GRID_OFF
    return split_combine.DECODE_GRID_AUTO


# The softmax SPLIT's module default: CUDNN_FROST_QSA_SOFTMAX_GROUPS = 0 (unset: the kernel's default = TWO 4-warp column groups of 8
# columns, 20 warps), 2 (the same, explicit) or 1 (ONE 4-warp group of 16 columns, the pre-split body).  A performance knob -- the same
# function, bitwise, at either value -- read once at import so a whole test tier renders the other form without an edit; the value
# travels in TemplateParams (the module-cache key), so no two bodies ever share a compiled-plan key.  An explicit softmax_groups=
# at construction overrides it.
SOFTMAX_GROUPS_DEFAULT = int(os.environ.get("CUDNN_FROST_QSA_SOFTMAX_GROUPS", "0") or 0)


def pdl_decline_reason(*, thd: bool) -> Optional[str]:
    """The ONE statement of the kernel fact behind the ``thd`` x ``use_pdl`` decline (``None`` = the composition is served).  The packed form's
    pre-dispatch dead-CTA exit reads the live-unit total ``meta[4B + 2]`` and its claim-counter scheduler ``meta[4B + 3]`` -- the SETUP launch's
    device values, written by the kernel launched right before the main one on the same stream -- above every role's dependent-grid wait (the
    kernel header's PDL SITES paragraph), so a dependent launch of the THD rendering would read what the setup launch has not written yet, and
    a wait above the exit would serialize every CTA behind the predecessor anyway.  :meth:`SparseGqaFwdDslSm107.check_support` raises this
    text; the gated attention block raises the same text at declaration (``thd=True`` on a ``QsaSpec`` geometry with ``use_pdl=True``) -- one
    source, pinned equal by both suites, so neither site drifts alone when the kernel's PDL scope moves."""
    if not thd:
        return None
    return (
        "thd=True with use_pdl=True is not served -- the packed form reads the setup launch's metadata (the live-unit total and the claim "
        "counter) in its pre-dispatch dead-CTA exit and in its scheduler, above every role's dependent-grid wait, so a dependent launch of it "
        "would read what the setup launch has not written yet; the programmatic-dependent-launch scope is the dense BSHD and paged decode "
        "chain -- declare use_pdl=False"
    )


class SparseGqaFwdDslSm107:
    """``O, LSE = sparse_sdpa(Q, K, V, block_ids[, block_lens][, seq_kv_lens][, block_table][, epilogue_gate])`` on cc 10.7 -- the standalone adapter.

    Construct with the operands (framework tensors: ``.shape`` / ``.stride()`` / ``.dtype`` / ``.data_ptr()`` /
    ``.device``), call :meth:`check_support` (typed declines), :meth:`compile` (one compiled artifact per specialization),
    :meth:`execute` (launch on the caller's stream).  The kernel module is imported ONLY after the DSL gate passed.
    """

    def __init__(
        self,
        *,
        q,
        k,
        v,
        o,
        block_ids,
        lse=None,
        block_lens=None,
        seq_kv_lens=None,
        top_k: int,
        block_size: int = _BLOCK_SIZE,
        scale: Optional[float] = None,
        stats_log2: bool = False,
        thd: bool = False,
        paged_kv: bool = False,
        epilogue_gate=None,
        split_kv: int = 1,
        bottom_right: bool = False,
        sink=None,
        window_left: Optional[int] = None,
        window_right: Optional[int] = None,
        seq_q_lens=None,
        list_per_sequence: bool = False,
        include_open_block: bool = True,
        device_cc: Optional[Tuple[int, int]] = None,
        page_size: int = 0,
        block_table=None,
        cu_seq_q_lens: bool = False,
        cu_seq_kv_lens: bool = False,
        workspace=None,
        softmax_groups: int = 0,
        use_pdl: bool = False,
        tile_form: str = "auto",
        sub_tiles_per_cta: Optional[int] = None,
    ):
        self.q, self.k, self.v, self.o, self.lse = q, k, v, o, lse
        # The TILE FORM (appended; a PERFORMANCE knob of the two bodies, the module docstring): "auto" / "lean" / "tcgen05" -- the
        # vocabulary is the heuristics' (one spelling for the adapter and the gated attention block); resolved by check_support (the
        # placement rule on the measured parts, the explicit body honoured or declined by name) into `tile_form` / `kernel_template`.
        # `sub_tiles_per_cta` pins the lean body's k (1 .. min(8, n_sub - 1)); it is refused by name where the resolved body is tcgen05.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import SPARSE_TILE_FORMS

        if not isinstance(tile_form, str) or tile_form not in SPARSE_TILE_FORMS:
            raise ValueError(
                f"sparse d256 forward (sm107): tile_form must be one of {SPARSE_TILE_FORMS} ('auto' = the placement rule on the measured parts, "
                f"the tcgen05 core elsewhere; 'lean' / 'tcgen05' explicit); got {tile_form!r}"
            )
        if sub_tiles_per_cta is not None and (isinstance(sub_tiles_per_cta, bool) or not isinstance(sub_tiles_per_cta, int) or sub_tiles_per_cta < 1):
            raise ValueError(
                f"sparse d256 forward (sm107): sub_tiles_per_cta must be None (the default k) or an int >= 1 (the lean tile's sub-tiles per CTA); got {sub_tiles_per_cta!r}"
            )
        self.tile_form_requested = tile_form
        self.tile_form: Optional[str] = None  # the RESOLVED body ("lean" / "tcgen05"), set by check_support
        self.sub_tiles_per_cta_requested = sub_tiles_per_cta
        self.sub_tiles_per_cta: Optional[int] = None  # the lean body's k after check_support; None on the tcgen05 body
        self.split_kv_requested = int(split_kv)  # as declared: on the lean body `split_kv` becomes the lean split check_support resolves
        # Programmatic dependent launch (appended 2026-10-21; a PERFORMANCE knob of the chain's two launches -- the same function, bitwise):
        # the launch attribute plus the griddepcontrol sites in both kernels it reaches -- the sparse core (compile(use_pdl=): every role's
        # wait before its first dependent global read; no explicit release, the CTA exit triggers the dependent at the core's completion)
        # and the split combine's entry
        # (compile_ptr(use_pdl=), every form's pair).  THD x use_pdl is check_support's typed decline (knob honoured or ineligible).
        self.use_pdl = bool(use_pdl)
        # The softmax SPLIT (appended; a PERFORMANCE knob of the kernel's softmax role, the same function bitwise): 0 = the module
        # default SOFTMAX_GROUPS_DEFAULT (itself 0 = the kernel's two 4-warp column groups), 1 / 2 explicit -- compiled in as
        # TemplateParams.qsa_softmax_groups (the module-cache key: the two renderings never share a compiled artifact).
        self.softmax_groups = int(softmax_groups) if softmax_groups else SOFTMAX_GROUPS_DEFAULT
        self.block_ids, self.block_lens, self.seq_kv_lens = block_ids, block_lens, seq_kv_lens
        # The fused epilogue gate's OPERAND (a tensor or a SparseOperandDesc; None = the ungated specialization): O's shape in
        # Q's dtype at its own (batch, seq, head) strides -- validated in check_support, compiled in as TemplateParams.epilogue_gate.
        self.gate = epilogue_gate
        self.top_k, self.block_size = int(top_k), int(block_size)
        # The softmax scale is handed to the kernel SIGNED (as scale x log2 e): the body multiplies every raw score by it BEFORE the
        # column max, so a negative attn_scale is served as is -- the dense line's TemplateParams.negate_scores (BMM1 negates Q and
        # the softmax runs at |scale|) is never raised by this adapter and the sparse body does not consume it.
        self.scale = (1.0 / math.sqrt(_D)) if scale is None else float(scale)
        self.stats_log2 = bool(stats_log2)
        # The paged read (appended): k / v are page pools, block_table the (B, max_pages) table, page_size the pool's tokens
        # per page; seq_kv_lens is then required.  Both appended arguments are refused on a dense declaration (below).
        self.paged_kv = bool(paged_kv)
        self.page_size = int(page_size)
        self.block_table = block_table
        # THD (packed sequences): the Q lengths ride ``seq_q_lens`` (REQUIRED there; a dense batch's Q-length trimming is the
        # unserved ``seq_q_lens`` arm below), the KV lengths ``seq_kv_lens``; each as per-sequence lengths [B] or cumulative
        # [B + 1] per its ``cu_seq_*_lens`` flag; the metadata lives in the caller's ``workspace``.  THD together with the paged
        # read is a typed decline in check_support (one of the two arms per declaration).
        self.thd = bool(thd)
        self.seq_q_lens = seq_q_lens
        self.cu_seq_q_lens, self.cu_seq_kv_lens = bool(cu_seq_q_lens), bool(cu_seq_kv_lens)
        self.workspace = workspace
        # The decode form's three arms (served): the bottom-right diagonal, the per-sequence shared list, the KV split -- each
        # validated in check_support against the others (never under THD; the list needs the diagonal and S_q <= 4; the split is
        # bounded by the item's tile count).  A split with a gate operand routes the gate to the combine (template_params).
        self.split_kv = int(split_kv)
        self.bottom_right = bool(bottom_right)
        self.list_per_sequence = bool(list_per_sequence)
        self.unserved = dict(
            sink=sink is not None,
            sliding_window=window_left is not None or window_right is not None,
            seq_q_lens=seq_q_lens is not None and not self.thd,
            cu_seq_lens=(self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            pure_list=not include_open_block,
            stats_log2=self.stats_log2,
        )
        self.device_cc = tuple(device_cc) if device_cc is not None else None
        self._fn = None  # the DECLARED block_lens variant (kept for the standalone path)
        self._fns = {}  # {has_block_lens: compiled fn} -- both variants may coexist (compile(has_block_lens=))
        self._combine = None  # split_kv > 1: the split combine's positional entry (+ its owner, kept alive)
        self._combine_owner = None
        self._combine_form = None  # split_kv > 1: the split_combine.DECODE_GRID_* value the combine entry was compiled with (combine_form())
        self._module = None
        self._B = self._SQ = self._SKV = self._H = self._KH = self._G = 0
        self._NSEQ = 0  # THD: the number of sequences (the lens tensors' shape); == _B on a dense batch
        self._n_ctas = 0  # THD: the occupancy-sized grid, resolved once per device at the first execute
        self._dtype_code = 0
        self._device_index = None  # pinned by declarations with a device; otherwise resolved from the current device at compile
        self._paged = None  # the paged arm's derived geometry (set by check_support): num_pages, max_pages, per-operand row terms

    # --- the claims ------------------------------------------------------------------------------------------------------

    @staticmethod
    def claims() -> SparseCapabilities:
        return SPARSE_CAPABILITIES

    # --- the enforcement point -------------------------------------------------------------------------------------------

    def _device_cc(self) -> Tuple[int, int]:
        if self.device_cc is not None:
            return self.device_cc
        import torch  # the framework is the caller's; imported here, never at module import

        _require_cuda_device("q", self.q, allow_unbound=True)
        return tuple(torch.cuda.get_device_capability(self.q.device))

    def check_support(self) -> bool:
        """Typed declines in order: the DSL gate (Rule 7, before any kernel import), the arch, the unserved arms by name, the
        dtypes, the shapes, the stride contract, the index list.  Returns True when the record serves the request."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.requirements import tma_gather4_requirement_error  # the host entry: importable below the DSL floor

        cc = self._device_cc()
        self.split_kv = self.split_kv_requested  # re-resolved below (check_support may run more than once; the pin, never the resolved value, is the input)
        msg = tma_gather4_requirement_error(cc)
        if msg is not None:
            raise NotImplementedError(f"sparse d256 forward (sm107): {msg}")
        cc_code = cc[0] * 10 + cc[1]
        if not (SPARSE_CAPABILITIES.sm_lo <= cc_code <= SPARSE_CAPABILITIES.sm_hi):
            raise NotImplementedError(f"sparse d256 forward (sm107): needs a Rubin-line GPU (cc 10.7 .. 11.9); got cc {cc[0]}.{cc[1]}")
        device = self._device_index
        for name, t in (
            ("q", self.q),
            ("k", self.k),
            ("v", self.v),
            ("o", self.o),
            ("lse", self.lse),
            ("block_ids", self.block_ids),
            ("block_lens", self.block_lens),
            ("seq_kv_lens", self.seq_kv_lens),
            ("block_table", self.block_table),
            ("seq_q_lens", self.seq_q_lens),
            ("gate", self.gate),
            ("workspace", self.workspace),
        ):
            device = _require_cuda_device(name, t, device, allow_unbound=True)
        self._device_index = device
        for name, requested in self.unserved.items():
            if requested:
                raise NotImplementedError(f"sparse d256 forward (sm107): {name} is not served by the kernel body yet (the record declines it by name)")

        # the decode form's arms against each other: none composes with THD (the packed form's position is the token's offset
        # in its sequence, its list one row per packed token, its scheduler a claim counter with no split axis); the shared
        # list is anchored at the step-0 position, so it needs the bottom-right diagonal; the split is bounded by the item's
        # tile count (ceil((top_k + 1) / 32) -- a larger split has an empty chunk on EVERY item)
        if self.split_kv < 1:
            raise ValueError(f"sparse d256 forward (sm107): split_kv must be >= 1; got {self.split_kv}")
        if self.softmax_groups not in (0, 1, 2):
            raise NotImplementedError(
                f"sparse d256 forward (sm107): softmax_groups must be 0 (the kernel default), 1 (one 4-warp column group) or 2 (the softmax split: two "
                f"4-warp groups of 8 columns); got {self.softmax_groups}"
            )
        if self.use_pdl and self.thd:
            # The packed form's ONE global read above every role's dependent-grid wait: the pre-dispatch dead-CTA exit reads the live-unit
            # total meta[4B + 2] (and the claim-counter scheduler meta[4B + 3]) -- the SETUP launch's device values, written by the kernel
            # launched right before the main one on the same stream.  A dependent launch of the THD rendering would read them before that
            # launch wrote them (the kernel header's PDL SITES paragraph), and a wait above the exit would serialize every CTA behind the
            # predecessor anyway -- so the composition is declined by name; the dense BSHD and paged chains carry the sites.  The text is
            # `pdl_decline_reason`'s: ONE statement for this site and for the gated attention block's declaration-time decline.
            raise NotImplementedError(f"sparse d256 forward (sm107): {pdl_decline_reason(thd=True)}")
        if self.thd:
            for name, requested in (("split_kv", self.split_kv > 1), ("list_per_sequence", self.list_per_sequence), ("bottom_right", self.bottom_right)):
                if requested:
                    raise NotImplementedError(
                        f"sparse d256 forward (sm107): {name} with thd=True is not served -- the packed form keeps top-left positions, one list row per "
                        "packed token and the claim-counter scheduler (no split axis); the decode form takes dense BSHD operands or paged pools"
                    )
        if self.list_per_sequence and not self.bottom_right:
            raise NotImplementedError(
                "sparse d256 forward (sm107): list_per_sequence=True needs bottom_right=True -- the shared list is anchored at the step-0 position "
                "kv_len - S_q (the decode / MTP form); a top-left per-sequence list has no serving meaning"
            )
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import sparse_tcgen05_max_split  # one copy of the core's split bound (the envelope reads the same)

        max_tiles = sparse_tcgen05_max_split(self.top_k) if _TOPK_MIN <= self.top_k <= _TOPK_MAX else 0

        # the index list's FORM (the kernel's staging is sized by it)
        if self.block_size != _BLOCK_SIZE:
            raise NotImplementedError(
                f"sparse d256 forward (sm107): block_size must be {_BLOCK_SIZE} (one gather transaction = four rows); got {self.block_size}"
            )
        if not (_TOPK_MIN <= self.top_k <= _TOPK_MAX) or self.top_k % 4 != 0:
            raise NotImplementedError(f"sparse d256 forward (sm107): top_k must be a multiple of 4 in [{_TOPK_MIN}, {_TOPK_MAX}]; got {self.top_k}")
        # the split bound is PER BODY (`_resolve_tile_form`, after the shapes: 17 = ceil((top_k + 1) / 32) on the tcgen05 core, the lean splits on the lean tile)

        # dtypes: one half dtype for Q / K / V / O, fp32 LSE
        dt = str(self.q.dtype)
        if dt not in _DTYPE_CODE:
            raise NotImplementedError(f"sparse d256 forward (sm107): Q must be bf16 or f16; got {dt}")
        for name, t in (("K", self.k), ("V", self.v), ("O", self.o)):
            if str(t.dtype) != dt:
                raise ValueError(f"sparse d256 forward (sm107): {name} dtype {t.dtype} must equal Q's {dt}")
        if self.lse is not None and str(self.lse.dtype) != "torch.float32":
            raise ValueError(f"sparse d256 forward (sm107): LSE must be float32; got {self.lse.dtype}")
        if self.gate is not None:
            # The fused epilogue gate: O's (B, S, H_q, D) shape in Q's dtype (the kernel stages it into the Q^T slot, whose
            # dtype is Q's), the head dim contiguous, the (batch, seq, head) strides TMA-expressible (16-B multiples -- the same
            # 4-D box form Q^T uses; a slab column slice at the projection's token stride qualifies).
            if str(self.gate.dtype) != dt:
                raise ValueError(f"sparse d256 forward (sm107): the epilogue gate dtype {self.gate.dtype} must equal Q's {dt}")
            if _shape(self.gate) != _shape(self.q):
                raise ValueError(
                    f"sparse d256 forward (sm107): the epilogue gate must be O-shaped (B, S_q, H_q, D) = {_shape(self.q)}; got {_shape(self.gate)}"
                )
            gst = _strides(self.gate)
            if gst[3] != 1:
                raise ValueError(f"sparse d256 forward (sm107): the epilogue gate head dim must be contiguous (stride 1); got strides {gst}")
            if any(s % 8 != 0 for s in gst[:3]):
                raise ValueError(
                    f"sparse d256 forward (sm107): the epilogue gate (batch, seq, head) strides must be multiples of 8 elements (TMA's 16-byte rule); got {gst}"
                )

        # the two wired arms are served ONE AT A TIME: under THD the gather row is the sequence's packed row cu_k[b] + 4 blk + r
        # on a dense [1, T_kv, H_kv, D] tensor, under the paged read the pool row page x rows_per_page + ...; nothing composes
        # a sequence's row offset with a page pool, so the combination is declined by name until a serving stack asks for it
        if self.thd and self.paged_kv:
            raise NotImplementedError(
                "sparse d256 forward (sm107): thd=True with paged_kv=True is not served -- the packed sequence's K / V row offset (cu_k[b]) composes "
                "with a dense [1, T_kv, H_kv, D] tensor only, never with a page pool; declare one of the two (the combination lands with its own "
                "accept cells when a serving stack asks for it)"
            )

        # the paged read's FORM: page_size a positive multiple of the block (the kernel's one-lookup-per-block premise), the
        # table and the KV lengths present; both appended arguments refused on a dense declaration (the mirror image)
        if self.paged_kv:
            if self.page_size <= 0 or self.page_size % _BLOCK_SIZE != 0:
                raise NotImplementedError(
                    f"sparse d256 forward (sm107): paged_kv needs page_size a positive multiple of {_BLOCK_SIZE} (a {_BLOCK_SIZE}-token block never "
                    f"straddles a page, one table lookup per block); got page_size={self.page_size} -- the serving stacks use multiples of 16"
                )
            if self.block_table is None:
                raise ValueError("sparse d256 forward (sm107): paged_kv needs block_table (int32 (B, max_pages): the sequence's pages in order)")
            if self.seq_kv_lens is None:
                raise ValueError(
                    "sparse d256 forward (sm107): paged_kv needs seq_kv_lens (int32 [B], the per-batch logical KV length): a page pool has no dense "
                    "extent, so the visible range and the unwritten rows of a partially filled last page can only be bounded by it"
                )
        else:
            if self.block_table is not None:
                raise ValueError("sparse d256 forward (sm107): block_table given but paged_kv=False (the table belongs to the paged read; pass paged_kv=True)")
            if self.page_size:
                raise ValueError(f"sparse d256 forward (sm107): page_size={self.page_size} given but paged_kv=False (the page size belongs to the paged read)")

        # shapes: (B, S, H, 256) BSHD -- or the packed (1, T, H, 256) under THD; one KV head serves H / KH query heads,
        # 1 <= group <= 16; paged: K / V are the pools [num_pages, H_kv, page_size, 256] (HND compact or NHD by strides) and
        # S_kv is the table's capacity
        qs, ks, vs, os_ = _shape(self.q), _shape(self.k), _shape(self.v), _shape(self.o)
        kv_form = "(num_pages, H_kv, page_size, D) -- a page pool" if self.paged_kv else "(B, S, H, D)"
        for name, s in (("Q", qs), ("K", ks), ("V", vs), ("O", os_)):
            if len(s) != 4:
                raise ValueError(f"sparse d256 forward (sm107): {name} must be rank-4 ({kv_form if name in ('K', 'V') else '(B, S, H, D)'}); got {s}")
            if s[3] != _D:
                raise NotImplementedError(f"sparse d256 forward (sm107): d_qk = d_v = {_D} exactly (no envelope); got {name} d = {s[3]}")
        B, SQ, H, _ = qs
        if vs != ks:
            raise ValueError(f"sparse d256 forward (sm107): V shape {vs} must equal K shape {ks}")
        if os_ != qs:
            raise ValueError(f"sparse d256 forward (sm107): O shape {os_} must equal Q shape {qs}")
        if self.paged_kv:
            num_pages, KH, Pk, _ = ks
            if Pk != self.page_size:
                raise ValueError(f"sparse d256 forward (sm107): the K / V pools' page axis is {Pk} tokens but page_size={self.page_size} was declared")
            bt = self.block_table
            if str(bt.dtype) != "torch.int32":
                raise ValueError(f"sparse d256 forward (sm107): block_table must be int32; got {bt.dtype}")
            bts = _shape(bt)
            if len(bts) != 2 or bts[0] != B or bts[1] < 1 or not bt.is_contiguous():
                raise ValueError(f"sparse d256 forward (sm107): block_table must be a contiguous int32 (B, max_pages) = ({B}, >= 1); got {bt.dtype} {bts}")
            max_pages = bts[1]
            SKV = max_pages * self.page_size  # the table's capacity; the visible range per batch is seq_kv_lens
        else:
            Bk, SKV, KH, _ = ks
            if Bk != B:
                raise ValueError(f"sparse d256 forward (sm107): batch mismatch: Q {qs}, K {ks}")
        n_seq = B
        if self.thd:
            if B != 1:
                raise ValueError(
                    f"sparse d256 forward (sm107): thd=True takes PACKED operands with batch extent 1 -- Q / O [1, T_q, H, D], K / V [1, T_kv, H_kv, D]; got Q {qs}, K {ks}"
                )
            if SQ < 1 or SKV < 1:
                raise ValueError(
                    f"sparse d256 forward (sm107): thd=True needs packed capacities T_q, T_kv >= 1 (a zero-extent tensor map is invalid, not empty); got T_q {SQ}, T_kv {SKV}"
                )
            n_seq = self._check_thd_lens_form()
        if self.list_per_sequence and SQ > SPARSE_CAPABILITIES.index_shared_list_max_tokens:
            raise NotImplementedError(
                f"sparse d256 forward (sm107): list_per_sequence serves S_q <= {SPARSE_CAPABILITIES.index_shared_list_max_tokens} rows per sequence (the "
                f"shared list's tail covers at most two blocks: the decode step and its MTP verify rows); got S_q = {SQ} -- pass per-token lists"
            )
        if H % KH != 0:
            raise ValueError(f"sparse d256 forward (sm107): H_q = {H} must be a multiple of H_kv = {KH}")
        G = H // KH
        if not (SPARSE_CAPABILITIES.index_gqa_group_min <= G <= SPARSE_CAPABILITIES.index_gqa_group_max):
            raise NotImplementedError(
                f"sparse d256 forward (sm107): the query-head group H_q / H_kv = {G} must be in [1, {_GQA_GROUP_MAX}] (one 16-row tile per token)"
            )
        # THE TILE FORM and the split, per body (the module docstring): resolved here, once the geometry is known
        self._resolve_tile_form(B, SQ, KH, G, cc, max_tiles)
        if self.lse is not None and _shape(self.lse) != (B, H, SQ):
            raise ValueError(f"sparse d256 forward (sm107): LSE must be (B, H_q, S_q) = {(B, H, SQ)}; got {_shape(self.lse)}")
        if B * SKV > _INT32_MAX or B * SQ * self.top_k > _INT32_MAX:
            raise NotImplementedError("sparse d256 forward (sm107): an extent exceeds the Int32 coordinate range")

        # strides
        for name, t in (("Q", self.q), ("K", self.k), ("V", self.v), ("O", self.o)):
            st = _strides(t)
            if st[3] != 1:
                raise ValueError(f"sparse d256 forward (sm107): {name} head dim must be contiguous (stride 1); got strides {st}")
        # Q is TMA-loaded (the Q^T box): its (batch, seq, head) strides are the tensor map's, encoded in 16-byte units -- the same rule the
        # gate tile follows; O is stored through generic stores and needs no such rule
        qst = _strides(self.q)
        if any(s % 8 != 0 for s in qst[:3]):
            raise ValueError(f"sparse d256 forward (sm107): Q (batch, seq, head) strides must be multiples of 8 elements (TMA's 16-byte rule); got {qst}")
        # a declared tensor (not a storage-free descriptor) is checked for its base alignment here as well as at execute
        for name, t in (("Q", self.q), ("K", self.k), ("V", self.v), ("O", self.o), ("block_ids", self.block_ids), ("the epilogue gate", self.gate)):
            _require_16b_base(name, t)
        if self.paged_kv:
            geom_k = _paged_pool_geometry("K", self.k, KH, self.page_size)
            geom_v = _paged_pool_geometry("V", self.v, KH, self.page_size)
            self._paged = (num_pages, max_pages, geom_k, geom_v)
        else:
            for name, t in (("K", self.k), ("V", self.v)):
                bs, ss, hs, _ = _strides(t)
                if (KH - 1) * hs + _D > _INT32_MAX:
                    raise NotImplementedError(f"sparse d256 forward (sm107): {name} gather columns exceed the Int32 coordinate range")
                if hs % 64 != 0:
                    raise ValueError(f"sparse d256 forward (sm107): {name} head stride must be a multiple of 64 elements (a 128-B column box start); got {hs}")
                if ss % 8 != 0:
                    raise ValueError(f"sparse d256 forward (sm107): {name} token stride must be a multiple of 8 elements (TMA's 16-byte rule); got {ss}")
                if B > 1 and bs != SKV * ss:
                    raise ValueError(
                        f"sparse d256 forward (sm107): {name} batch stride must be S_kv x the token stride ({SKV} x {ss}) for the 2-D gather map; got {bs}"
                    )

        # the index list's tensors: one row per query token, or ONE row per sequence under list_per_sequence
        ids = self.block_ids
        if str(ids.dtype) != "torch.int32":
            raise ValueError(f"sparse d256 forward (sm107): block_ids must be int32; got {ids.dtype}")
        if self.list_per_sequence:
            if _shape(ids) != (B, self.top_k):
                raise ValueError(
                    f"sparse d256 forward (sm107): list_per_sequence needs block_ids [B, top_k] = {(B, self.top_k)} (one list per sequence); got {_shape(ids)}"
                )
        elif _shape(ids) not in ((B, SQ, self.top_k), (B * SQ, self.top_k)):
            raise ValueError(f"sparse d256 forward (sm107): block_ids must be [B, S_q, top_k] = {(B, SQ, self.top_k)} or [B x S_q, top_k]; got {_shape(ids)}")
        if not ids.is_contiguous():
            raise ValueError("sparse d256 forward (sm107): block_ids must be contiguous (its row stride is the bulk-copy length)")
        if self.block_lens is not None:
            self._check_block_lens_form(self.block_lens, B, SQ, self.list_per_sequence)
        if self.seq_kv_lens is not None and not self.thd:
            kl = self.seq_kv_lens
            if str(kl.dtype) != "torch.int32" or _shape(kl) != (B,) or not kl.is_contiguous():
                raise ValueError(f"sparse d256 forward (sm107): seq_kv_lens must be a contiguous int32 [B]; got {kl.dtype} {_shape(kl)}")
        if self.thd and self.workspace is not None:
            self._check_workspace(self.workspace, n_seq)
        self._B, self._SQ, self._SKV, self._H, self._KH, self._G = B, SQ, SKV, H, KH, G
        self._NSEQ = n_seq
        self._dtype_code = _DTYPE_CODE[dt]
        return True

    def _sm_count_for_rule(self) -> int:
        """The SM count the lean k rule's fill model reads (information: the default k is capped at 1), 0 for a storage-free / non-CUDA Q."""
        dev = getattr(self.q, "device", None)
        if dev is None:
            return 0
        import torch

        try:
            d = torch.device(dev)
        except (TypeError, ValueError, RuntimeError):
            return 0
        if d.type != "cuda" or not torch.cuda.is_available():
            return 0
        return _device_sm_count(d)

    def _resolve_tile_form(self, B: int, SQ: int, KH: int, G: int, cc, max_tiles: int) -> None:
        """Which BODY serves this declaration and at which split (the module docstring's TWO BODIES paragraph).  ``tile_form="auto"`` ->
        :func:`heuristics.sparse_tile_form_auto` (the tcgen05 core unless the part is in ``SPARSE_LEAN_TILE_DEFAULT_CCS``, the lean body
        admissible, the placement rule for it and the pinned split, if any, a lean split); an explicit body as given.  The LEAN body:
        the admissibility declines by name (:func:`heuristics.sparse_lean_tile_decline`), then ``split_kv`` as a lean split that SELECTS k
        (``ceil(n_sub / k)``; unset = the default k of :func:`heuristics.sparse_lean_sub_tiles_per_cta` or the pinned ``sub_tiles_per_cta``),
        k in ``1 .. min(8, n_sub - 1)``; ``self.split_kv`` becomes the lean split and ``self.sub_tiles_per_cta`` the k.  The TCGEN05 core:
        ``split_kv`` up to ``ceil((top_k + 1) / 32)`` as before; a ``sub_tiles_per_cta`` pin is refused by name (never silently ignored)."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import sparse_tile_rules as H
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import lean_sparse_n_sub, lean_sparse_split

        split_req = self.split_kv_requested
        reason = H.sparse_lean_tile_decline(
            s_q=SQ,
            group=G,
            top_k=self.top_k,
            thd=self.thd,
            bottom_right=self.bottom_right,
            list_per_sequence=self.list_per_sequence,
            seq_kv_lens_present=self.seq_kv_lens is not None,
            half_dtype=True,  # the dtypes were validated above
            softmax_groups=self.softmax_groups,
        )
        if self.tile_form_requested == H.SPARSE_TILE_FORM_AUTO:
            form = H.sparse_tile_form_auto(
                batch=B, h_kv=KH, s_q=SQ, top_k=self.top_k, device_cc=cc, admissible=reason is None, split_pinned=split_req if split_req > 1 else None
            )
        else:
            form = self.tile_form_requested
        if form == H.SPARSE_TILE_FORM_LEAN:
            if reason is not None:
                raise NotImplementedError(f"sparse d256 forward (sm107): tile_form='lean' -- {reason}")
            n_sub = lean_sparse_n_sub(self.top_k)
            splits = H.sparse_lean_splits(self.top_k)
            k_max = H.sparse_lean_max_sub_tiles_per_cta(self.top_k)
            k_pin = self.sub_tiles_per_cta_requested
            if split_req > 1:
                if split_req not in splits:
                    raise NotImplementedError(
                        f"sparse d256 forward (sm107): split_kv={split_req} is not a lean split of top_k {self.top_k}: the lean tile's splits are {splits} "
                        f"(ceil(n_sub / k) at n_sub = {n_sub}; the split SELECTS k, the sub-tiles per CTA) -- pin one of them, or leave split_kv unset for the "
                        f"default k; the tcgen05 core (tile_form='tcgen05') takes any split in [1, {max_tiles}]"
                    )
                # the explicit k is THE k when given (ceil(n_sub / split) is not its inverse: at top_k 32 the splits of k 3 and k 4 are both 2);
                # the split must then be ITS lean split
                k = int(k_pin) if k_pin is not None else -(-n_sub // split_req)
                if k_pin is not None and lean_sparse_split(self.top_k, k) != split_req:
                    raise ValueError(
                        f"sparse d256 forward (sm107): sub_tiles_per_cta={k} and split_kv={split_req} disagree: the lean split of k {k} at top_k "
                        f"{self.top_k} is ceil({n_sub} / {k}) = {lean_sparse_split(self.top_k, k)} -- pin one of the two"
                    )
            else:
                k = (
                    int(k_pin)
                    if k_pin is not None
                    else H.sparse_lean_sub_tiles_per_cta(batch=B, h_kv=KH, s_q=SQ, group=G, top_k=self.top_k, sm_count=self._sm_count_for_rule())
                )
            if not 1 <= k <= k_max:
                raise NotImplementedError(
                    f"sparse d256 forward (sm107): sub_tiles_per_cta={k} must be in [1, {k_max}] at top_k {self.top_k} (n_sub {n_sub}: the sub-tile loop is "
                    f"unrolled at trace time, and k = n_sub would be ONE chunk = split 1 -- the lean tile always splits; a one-chunk list is the tcgen05 core's)"
                )
            self.split_kv = lean_sparse_split(self.top_k, k)
            self.sub_tiles_per_cta = k
        else:
            if self.sub_tiles_per_cta_requested is not None:
                raise NotImplementedError(
                    f"sparse d256 forward (sm107): sub_tiles_per_cta={self.sub_tiles_per_cta_requested} is the lean tile's pin (its 32-key sub-tiles per CTA), but the "
                    f"resolved tile_form is 'tcgen05' (requested {self.tile_form_requested!r}"
                    + (f": {reason}" if reason is not None else ": the placement rule / the measured-default set keep the tcgen05 core here")
                    + ") -- declare tile_form='lean' to force the lean body, or drop the pin"
                )
            if split_req > max_tiles:
                raise NotImplementedError(
                    f"sparse d256 forward (sm107): split_kv={split_req} exceeds the item's maximum tile count ceil((top_k + 1) / 32) = {max_tiles} at "
                    f"top_k={self.top_k} on the tcgen05 core -- a larger split has an empty chunk (a dead work item) on every item"
                )
            self.split_kv = split_req
            self.sub_tiles_per_cta = None
        self.tile_form = form

    @property
    def kernel_template(self) -> Optional[str]:
        """The body ``compile`` loads, by its module stem (``heuristics.SPARSE_TILE_LEAN`` / ``SPARSE_TILE_TCGEN05``): the readback a caller
        pins its route against; None before ``check_support`` resolved the form."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import SPARSE_TILE_FORM_LEAN, SPARSE_TILE_LEAN, SPARSE_TILE_TCGEN05

        if self.tile_form is None:
            return None
        return SPARSE_TILE_LEAN if self.tile_form == SPARSE_TILE_FORM_LEAN else SPARSE_TILE_TCGEN05

    @property
    def lean(self) -> bool:
        """Whether the resolved body is the lean tile (after ``check_support``)."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import SPARSE_TILE_FORM_LEAN

        return self.tile_form == SPARSE_TILE_FORM_LEAN

    @staticmethod
    def _check_block_lens_form(bl, B: int, SQ: int, per_sequence: bool = False) -> None:
        if per_sequence:
            if str(bl.dtype) != "torch.int32" or _shape(bl) != (B,) or not bl.is_contiguous():
                raise ValueError(
                    f"sparse d256 forward (sm107): list_per_sequence needs block_lens a contiguous int32 [B] = {(B,)}; got {bl.dtype} {_shape(bl)}"
                )
            return
        if str(bl.dtype) != "torch.int32" or _shape(bl) not in ((B, SQ), (B * SQ,)) or not bl.is_contiguous():
            raise ValueError(f"sparse d256 forward (sm107): block_lens must be a contiguous int32 [B, S_q] / [B x S_q]; got {bl.dtype} {_shape(bl)}")

    def _check_thd_lens_form(self) -> int:
        """THD: both length tensors present, int32, rank-1, contiguous, agreeing on the sequence count ``B`` (``[B]`` lengths or
        ``[B + 1]`` prefix sums per ``cu_seq_*_lens``).  Returns ``B``.  The VALUES are device facts the setup launch reads
        (Rule 3); nothing here dereferences them."""
        if self.seq_q_lens is None or self.seq_kv_lens is None:
            raise ValueError(
                "sparse d256 forward (sm107): thd=True needs seq_q_lens AND seq_kv_lens (int32 per-sequence lengths [B], or cumulative [B + 1] with cu_seq_q_lens / cu_seq_kv_lens)"
            )
        counts = []
        for name, t, cu in (("seq_q_lens", self.seq_q_lens, self.cu_seq_q_lens), ("seq_kv_lens", self.seq_kv_lens, self.cu_seq_kv_lens)):
            shp = _shape(t)
            if str(t.dtype) != "torch.int32" or len(shp) != 1 or not t.is_contiguous():
                raise ValueError(f"sparse d256 forward (sm107): {name} must be a contiguous int32 rank-1 tensor under thd=True; got {t.dtype} {shp}")
            counts.append(shp[0] - int(cu))
        if counts[0] < 1 or counts[0] != counts[1]:
            raise ValueError(
                f"sparse d256 forward (sm107): seq_q_lens and seq_kv_lens must describe the same B >= 1 sequences ([B] lengths, [B + 1] cumulative); "
                f"got {counts[0]} and {counts[1]} (cu_seq_q_lens={self.cu_seq_q_lens}, cu_seq_kv_lens={self.cu_seq_kv_lens})"
            )
        return counts[0]

    @staticmethod
    def _thd_meta_bytes(n_seq: int) -> int:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_META_WORDS  # the ONE layout source (words = 4 B + 4)

        return -(-(THD_META_WORDS(int(n_seq)) * 4) // 16) * 16

    def _check_workspace(self, ws, n_seq: int) -> None:
        need = self._thd_meta_bytes(n_seq) if self.thd else self._split_workspace_bytes()
        what = "thd=True" if self.thd else f"split_kv={self.split_kv}"
        if not ws.is_contiguous():
            raise ValueError(f"sparse d256 forward (sm107): the {what} workspace must be contiguous (a flat scratch span)")
        nbytes = min(
            int(ws.numel()) * int(ws.element_size()),
            int(ws.untyped_storage().nbytes()) - int(ws.storage_offset()) * int(ws.element_size()),
        )
        if nbytes < need:
            raise ValueError(f"sparse d256 forward (sm107): {what} needs a workspace of {need} bytes (scratch_workspace_bytes()); got {nbytes}")
        if int(ws.data_ptr()) % 16 != 0:
            raise ValueError(f"sparse d256 forward (sm107): the {what} workspace must be 16-byte aligned; got data_ptr=0x{int(ws.data_ptr()):x}")

    @staticmethod
    def _align16(n: int) -> int:
        return -(-int(n) // 16) * 16

    def _split_workspace_layout(self) -> Tuple[int, int, int, int]:
        """``(o_partial_off, lse_partial_off, lse_final_off, total)`` in bytes: the fp32 O partial slab ``(B x S, S_q, H, D)``
        (compact, split-major on the batch axis), the fp32 LSE partial slab ``(B x S, H, S_q)``, and -- when the caller wants no
        LSE -- a final-LSE scratch ``(B, H, S_q)`` the combine writes (its ABI always carries one); each 16-byte aligned."""
        B, SQ, H, S = self._B, self._SQ, self._H, self.split_kv
        o_off = 0
        lse_off = self._align16(o_off + B * S * SQ * H * _D * 4)
        fin_off = self._align16(lse_off + B * S * H * SQ * 4)
        total = fin_off + (self._align16(B * H * SQ * 4) if self.lse is None else 0)
        return o_off, lse_off, fin_off, total

    def _split_workspace_bytes(self) -> int:
        return self._split_workspace_layout()[3] if self.split_kv > 1 else 0

    def scratch_workspace_bytes(self) -> int:
        """Per-execute GMEM scratch beyond the operands.  Dense (BSHD or paged pools), unsplit: NONE -- the count of a row's
        list, the tail block(s) and the dead items are derived on device from the position, ``block_lens`` and ``seq_kv_lens``
        (one bounds helper per work item), and the ids rows reach SMEM by a bulk copy straight from the caller's ``block_ids``
        -- no per-sequence metadata, no index staging; the paged read walks the caller's ``block_table`` directly (no
        per-sequence descriptors); a caller that folds every engine's scratch into one workspace (the gated attention block)
        folds a 0 here.  ``split_kv > 1``: the fp32 O / LSE PARTIAL slabs the kernel writes and the combine reads (split-major,
        ``(B x S, S_q, H, D)`` + ``(B x S, H, S_q)``, 16-byte aligned; plus a final-LSE scratch when no LSE output is declared).
        THD: the int32 metadata buffer the setup launch fills (``THD_META_WORDS(B) = 4 B + 4`` words, 16-byte rounded: the
        per-sequence KV lengths, both prefix sums, the live unit total and the claim counter).  The plan is RESOLVED first
        (``check_support``): on the lean body ``split_kv`` becomes the lean split there, so a caller asking for the scratch size
        before ``check_support()`` gets the slab, never the requested split's 0."""
        if not self._G:
            self.check_support()
        if not self.thd and self.split_kv <= 1:
            return 0
        if self.thd:
            return self._thd_meta_bytes(self._NSEQ)
        return self._split_workspace_bytes()

    # --- compile / execute -----------------------------------------------------------------------------------------------

    def template_params(self):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import TemplateParams

        if not self._G:
            self.check_support()
        kw = dict(
            dtype_qkv=self._dtype_code,
            cta_mma=1,
            pack_gqa=True,
            qh_per_kh=self._G,
            qsa_block_topk=self.top_k,
            qsa_block_size=self.block_size,
            seq_kv_lens_present=self.seq_kv_lens is not None or self.thd,
            stats_log2=self.stats_log2,
            paged_kv=self.paged_kv,
            page_size=self.page_size if self.paged_kv else 0,
            thd_varlen=self.thd,
            # a split's gate rides the combine (the kernel's own gate arm is refused under a split by the config)
            epilogue_gate=self.gate is not None and self.split_kv <= 1,
            split_kv=self.split_kv,
            bottom_right=self.bottom_right,
            # the lean body is ONE rendering per list form: the shared list (its config requires it); the per-row list at ONE row per sequence
            # (admitted by _resolve_tile_form at S_q 1) IS that list -- block_ids [B, 1, top_k] / block_lens [B, 1] are the shared [B, top_k] /
            # [B] bytes, the step-0 anchor is the row's own position -- so the template is loaded in its shared form and the bound tensors need
            # no view; the tcgen05 core keeps the declared form (its per-token arm is its own rendering)
            qsa_list_per_sequence=self.list_per_sequence or self.lean,
            qsa_softmax_groups=self.softmax_groups,
        )
        if self.lean:
            # the lean tile's OWN record (config_sm107.TemplateParamsLean: the shared record plus the appended compile-time S_q axis): the
            # rendering holds every row of the step -- N_MT = ceil(S_q x G / 16) m16 tiles over ONE gathered sub-tile --, so the host entry's
            # problem_size SQ equals CFG.S_Q (its contract); the tcgen05 core keeps the shared record (no appended field: its digest and
            # compiled-plan keys are what they were)
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import TemplateParamsLean

            # the resolved k rides the record too: an explicit pin survives template lowering (ceil(n_sub / split) is not invertible) and
            # two pins are two cache identities
            return TemplateParamsLean(**kw, decode_seqlen_q=self._SQ, sub_tiles_per_cta=int(self.sub_tiles_per_cta or 0))
        return TemplateParams(**kw)

    def combine_form(self) -> int:
        """The split combine entry's kernel FORM this plan compiles (``split_combine.DECODE_GRID_AUTO`` / ``DECODE_GRID_OFF`` /
        ``DECODE_GRID_LEAN_WALK``): :func:`_sparse_combine_decode_grid` on the declared ``(B, H_q, S_q)``, the split and the device --
        a plan-time fact, read after ``check_support`` (which it runs when needed); ``DECODE_GRID_OFF`` unsplit (no combine launch)."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100 import split_combine

        if not self._G:
            self.check_support()
        if self.split_kv <= 1:
            return split_combine.DECODE_GRID_OFF
        return _sparse_combine_decode_grid(self._B, self._H, self._SQ, self.split_kv, self._device_cc())

    def compile(self, has_block_lens: Optional[bool] = None):
        """Load the template for this specialization and compile its pointer ABI (one artifact per (dtype, group, top_k,
        KV-lens presence, LSE presence, block_lens presence)).

        ``has_block_lens`` (appended) selects the ``block_lens`` variant explicitly; the default is the declared presence.  A
        caller whose ``block_lens`` is optional PER CALL compiles both variants at plan time and :meth:`execute` dispatches
        on the tensor it is handed -- plan-time keys only, never a compile on the execute path."""
        if has_block_lens is None:
            has_block_lens = self.block_lens is not None
        has_block_lens = bool(has_block_lens)
        fn = self._fns.get(has_block_lens)
        if fn is not None:
            return fn
        self.check_support()
        import torch

        # CuTe launchables belong to the compilation context. Storage-free declarations select the current device once;
        # later block_lens variants and launches must use that same device, even if the caller changes its current device.
        if self._device_index is None:
            self._device_index = int(torch.cuda.current_device())
        with torch.cuda.device(self._device_index):
            return self._compile_on_device(has_block_lens)

    def _compile_on_device(self, has_block_lens: bool):
        if self._module is None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.template_loader import load_template

            params = self.template_params()
            # the BODY (tile_form, resolved by check_support): the lean tile's template takes the same TemplateParams (its split_kv = the lean
            # split selects k) under its own tag -- two module caches, never a shared compiled artifact
            lean = self.lean
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", KERNEL_FILE_LEAN if lean else KERNEL_FILE)
            self._module = load_template(path, params, tag="sdpa_fwd_sm107_sparse_lean" if lean else "sdpa_fwd_sm107_sparse_d256")
        # Under a split the kernel's LSE is the fp32 PARTIAL (always written: it drives the combine); the caller's LSE presence
        # selects the combine's final-LSE target (the output, or the workspace scratch).
        # Their outer functools.lru_cache keys only on arguments. Bypass it so compiled_cache's device/context-aware memo
        # owns reuse: the same specialization compiled by another plan on another GPU is a different launchable.
        compile_kernel = getattr(self._module.compile, "__wrapped__", self._module.compile)
        # use_pdl: the programmatic-dependent-launch rendering of the core (its launch attribute + griddepcontrol sites; in the compile key)
        fn = compile_kernel(has_lse=(self.lse is not None) or self.split_kv > 1, has_block_lens=has_block_lens, use_pdl=self.use_pdl)
        self._fns[has_block_lens] = fn
        if has_block_lens == (self.block_lens is not None):
            self._fn = fn
        if self.split_kv > 1 and self._combine is None:
            # The split combine (sm100/split_combine.py): fp32 partials -> the caller's O dtype, ONE rounding; the gate, when
            # declared, applied to the fp32 merged value (QI01's gate-in-combine entry).  Compiled at plan time like the kernel, with
            # the kernel FORM pinned per plan (combine_form(): the decode-grid kernel / the lean group walk / the module's own forms on
            # the sparse grid (B, H_q, S_q) at this split on this device -- the dense adapter's rule) and the programmatic-dependent-
            # launch switch (use_pdl: every form of the entry carries the griddepcontrol pair; a typed decline above until the core does).
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import positional_entry
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100 import split_combine

            tag = "bf16" if self._dtype_code == 2 else "f16"
            form = self.combine_form()
            compile_combine = getattr(split_combine.compile_ptr, "__wrapped__", split_combine.compile_ptr)
            owner = compile_combine(
                dtype_o=tag,
                dtype_partial="f32",
                has_lse=True,
                stats_log2=False,
                gate=self.gate is not None,
                dtype_gate=tag if self.gate is not None else None,
                decode_grid=form,
                use_pdl=self.use_pdl,
            )
            entry = positional_entry(owner)
            if entry is None:
                raise RuntimeError("sparse d256 forward (sm107): the split combine artifact exposes no positional tvm-ffi entry")
            self._combine_owner, self._combine, self._combine_form = owner, entry, form
        return fn

    def _bind(self, name: str, given, declared, *, required: bool):
        """The tensor a launch uses for ``name``: the one handed to ``execute`` (it must match the declaration in shape,
        strides and dtype -- a mismatch is a caller bug, refused by name, never a silent re-specialization), else the
        declared one (which must then be a tensor, not a :class:`SparseOperandDesc`)."""
        if given is None:
            if declared is None:
                if required:
                    raise ValueError(f"sparse d256 forward (sm107): {name} is required at execute")
                return None
            if isinstance(declared, SparseOperandDesc):
                raise ValueError(f"sparse d256 forward (sm107): {name} was declared as a SparseOperandDesc (no storage); pass the tensor at execute({name}=)")
            return declared
        if declared is None:
            raise ValueError(f"sparse d256 forward (sm107): {name} was not declared (its presence is compiled into the specialization); construct with {name}=")
        if _shape(given) != _shape(declared) or _strides(given) != _strides(declared) or str(given.dtype) != str(declared.dtype):
            raise ValueError(
                f"sparse d256 forward (sm107): {name} at execute must match its declaration -- shape {_shape(declared)} strides {_strides(declared)} "
                f"{declared.dtype}; got shape {_shape(given)} strides {_strides(given)} {given.dtype}"
            )
        return given

    def execute(
        self,
        stream=None,
        *,
        q=None,
        k=None,
        v=None,
        o=None,
        lse=None,
        block_ids=None,
        block_lens=None,
        seq_kv_lens=None,
        block_table=None,
        seq_q_lens=None,
        workspace=None,
        gate=None,
    ):
        """Launch on ``stream`` (default: the framework's current stream).  Validation only -- no conversion, no allocation.

        The appended keyword operands BIND the launch's tensors in place of the declared ones (a caller whose buffers exist
        only at execute declares with :class:`SparseOperandDesc` and passes every tensor here); each must match its
        declaration in shape, strides and dtype exactly.  ``lse`` / ``seq_kv_lens`` / ``block_table`` / ``gate`` keep the
        declaration's presence (it is compiled in; the table exists exactly on a paged declaration, the gate is the
        ``epilogue_gate`` specialization).  ``block_lens`` may differ in PRESENCE per call: a tensor -> the ``has_block_lens``
        variant, ``None`` -> the kernel's derived default count -- the variant must have been compiled (:meth:`compile`), never
        compiled here.  THD: ``seq_q_lens`` binds like the others and ``workspace`` (the metadata buffer,
        :meth:`scratch_workspace_bytes`, 16-byte aligned) is the one operand that need not match a declaration -- any buffer of
        at least the size serves."""
        if not self._G:
            self.check_support()
        q = self._bind("q", q, self.q, required=True)
        k = self._bind("k", k, self.k, required=True)
        v = self._bind("v", v, self.v, required=True)
        o = self._bind("o", o, self.o, required=True)
        ids = self._bind("block_ids", block_ids, self.block_ids, required=True)
        lse = self._bind("lse", lse, self.lse, required=self.lse is not None)
        seq_kv_lens = self._bind("seq_kv_lens", seq_kv_lens, self.seq_kv_lens, required=self.seq_kv_lens is not None)
        table = self._bind("block_table", block_table, self.block_table, required=self.paged_kv)
        seq_q_lens = self._bind("seq_q_lens", seq_q_lens, self.seq_q_lens, required=self.thd)
        ws = None
        if self.thd:
            ws = self.workspace if workspace is None else workspace
            if ws is None or isinstance(ws, SparseOperandDesc):
                raise ValueError("sparse d256 forward (sm107): thd=True needs the metadata workspace at execute(workspace=) (scratch_workspace_bytes() bytes)")
            self._check_workspace(ws, self._NSEQ)
        gate = self._bind("gate", gate, self.gate, required=self.gate is not None)
        # every pointer handed to the kernel with assumed_align=16 starts on a 16-byte boundary (a storage-offset view is refused by name)
        for name, t in (("Q", q), ("K", k), ("V", v), ("O", o), ("block_ids", ids), ("the epilogue gate", gate)):
            _require_16b_base(name, t)
        if self.split_kv > 1:
            ws = self.workspace if workspace is None else workspace
            if ws is None or isinstance(ws, SparseOperandDesc):
                raise ValueError(
                    f"sparse d256 forward (sm107): split_kv={self.split_kv} needs the partials workspace at execute(workspace=) (scratch_workspace_bytes() bytes)"
                )
            self._check_workspace(ws, self._NSEQ)
            if self._combine is None:
                raise RuntimeError("sparse d256 forward (sm107): call compile() before execute() -- the split combine is not compiled (plan-time keys only)")
        if block_lens is not None:
            self._check_block_lens_form(block_lens, self._B, self._SQ, self.list_per_sequence)
            lens = block_lens
        elif self.block_lens is not None and not isinstance(self.block_lens, SparseOperandDesc):
            lens = self.block_lens
        else:
            lens = None  # declared as a descriptor (or not at all) and not handed over: the derived default count
        # Check declared and rebound operands alike, including every optional pointer, before any launch or pointer lowering.
        device = _require_cuda_device("q", q, self._device_index)
        for name, t in (
            ("k", k),
            ("v", v),
            ("o", o),
            ("lse", lse),
            ("block_ids", ids),
            ("block_lens", lens),
            ("seq_kv_lens", seq_kv_lens),
            ("block_table", table),
            ("seq_q_lens", seq_q_lens),
            ("gate", gate),
            ("workspace", ws),
        ):
            _require_cuda_device(name, t, device)
        fn = self._fns.get(lens is not None)
        if fn is None:
            raise RuntimeError(
                f"sparse d256 forward (sm107): call compile(has_block_lens={lens is not None}) before execute() -- the {'with' if lens is not None else 'without'}-block_lens "
                "variant is not compiled (plan-time keys only: nothing compiles on the execute path)"
            )
        import torch

        with torch.cuda.device(device):
            return self._execute_on_device(q, k, v, o, ids, lse, lens, seq_kv_lens, table, seq_q_lens, ws, gate, fn, stream)

    def _execute_on_device(self, q, k, v, o, ids, lse, lens, seq_kv_lens, table, seq_q_lens, ws, gate, fn, stream):
        import torch
        import cutlass
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import make_ptr
        import cutlass.cute as cute

        gmem = cute.AddressSpace.gmem
        half = cutlass.BFloat16 if self._dtype_code == 2 else cutlass.Float16

        def P(t, dtype, align=16):
            return None if t is None else make_ptr(dtype, t.data_ptr(), gmem, assumed_align=align)

        if stream is None:
            stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        thd_kw = dict(thd_q_lens_ptr=None, thd_kv_lens_ptr=None, thd_lens_form=None, n_ctas=None)
        if self.thd:
            # The kernel reads the metadata workspace as its ``seq_kv_lens`` tensor (the setup launch fills it from the two
            # length tensors); the grid is the resident wave (one CTA per SM, never more CTAs than packed units at capacity),
            # resolved once per device (a ~7 us query kept off the per-call path).
            if not self._n_ctas:
                self._n_ctas = max(1, min(self._SQ * self._KH, _device_sm_count(q.device)))
            thd_kw = dict(
                thd_q_lens_ptr=P(seq_q_lens, cutlass.Int32, 4),
                thd_kv_lens_ptr=P(seq_kv_lens, cutlass.Int32, 4),
                thd_lens_form=cutlass.Int32((1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)),
                n_ctas=cutlass.Int32(self._n_ctas),
            )
            meta_ptr = make_ptr(cutlass.Int32, ws.data_ptr(), gmem, assumed_align=16)
        elif seq_kv_lens is None:
            # Unread by the kernel (SEQ_KV_LENS_PRESENT = 0): the pointer slot is bound to the ids (a valid address).
            meta_ptr = P(ids, cutlass.Int32, 4)
        else:
            meta_ptr = P(seq_kv_lens, cutlass.Int32, 4)
        if self.paged_kv:
            # The kernel's (batch, seq, head) stride slots carry (page stride, TOKEN stride, COLUMN head stride) of each pool
            # and paged_geom its row terms -- all derived from the declared strides (the bound tensors match them exactly).
            num_pages, max_pages, (kp, kt, kc, k_rpp, k_rph), (vp, vt, vc, v_rpp, v_rph) = self._paged
            k_strides, v_strides = (kp, kt, kc), (vp, vt, vc)
            paged_geom = (num_pages, max_pages, k_rpp, k_rph, v_rpp, v_rph)
        else:
            k_strides, v_strides = _strides(k)[:3], _strides(v)[:3]
            paged_geom = (0, 0, 0, 0, 0, 0)  # unread on the dense arm
        if self.split_kv > 1:
            # The kernel writes the fp32 partial slabs in the workspace (split-major: batch b + s x B, compact strides); the
            # combine then reduces them into the caller's O at its strides and the LSE (the output, or the workspace scratch).
            B, SQ, H, S = self._B, self._SQ, self._H, self.split_kv
            o_off, lse_off, fin_off, _total = self._split_workspace_layout()
            base = int(ws.data_ptr())
            o_part = make_ptr(cutlass.Float32, base + o_off, gmem, assumed_align=16)
            lse_part = make_ptr(cutlass.Float32, base + lse_off, gmem, assumed_align=16)
            o_ptr, o_strides = o_part, (SQ * H * _D, H * _D, _D)
            lse_ptr, lse_strides = lse_part, (H * SQ, SQ, 1)
            kernel_gate = None  # the gate rides the combine
        else:
            o_ptr, o_strides = P(o, half), _strides(o)[:3]
            lse_ptr, lse_strides = P(lse, cutlass.Float32, 4), (_strides(lse)[:3] if lse is not None else (0, 0, 0))
            kernel_gate = gate
        common = dict(
            q_ptr=P(q, half),
            k_ptr=P(k, half),
            v_ptr=P(v, half),
            o_ptr=o_ptr,
            lse_ptr=lse_ptr,
            block_ids_ptr=P(ids, cutlass.Int32, 16),
            block_lens_ptr=P(lens, cutlass.Int32, 4),
            seq_kv_lens_ptr=meta_ptr,
            problem_size=(self._NSEQ, self._H, self._KH, self._SQ, self._SKV),
            q_strides=_strides(q)[:3],
            k_strides=k_strides,
            v_strides=v_strides,
            o_strides=o_strides,
            lse_strides=lse_strides,
            scale_softmax_log2=cutlass.Float32(self.scale * math.log2(math.e)),
            block_table_ptr=P(table, cutlass.Int32, 4),
            paged_geom=paged_geom,
            stream=stream,
        )
        if self.lean:
            # the lean body's ABI: the tcgen05 core's minus the arms it does not carry (the epilogue gate rides the combine, no THD slots);
            # the slab pointers above are the split partials (the lean tile always splits)
            fn(**common)
        else:
            fn(
                gate_ptr=P(kernel_gate, half),
                gate_strides=_strides(kernel_gate)[:3] if kernel_gate is not None else (0, 0, 0),
                **common,
                **thd_kw,
            )
        if self.split_kv > 1:
            # The second launch on the same stream: the positional combine entry (partials, O, final LSE, (B, H, S_q, D), the
            # split count, O's four BSHD strides, the LSE's three, [the gate's pointer + four BSHD strides,] the stream).
            lse_final = int(lse.data_ptr()) if lse is not None else base + fin_off
            lse_final_strides = tuple(_strides(lse)[:3]) if lse is not None else (H * SQ, SQ, 1)
            args = [base + o_off, base + lse_off, int(o.data_ptr()), lse_final, (B, H, SQ, _D), S, tuple(_strides(o)[:4]), lse_final_strides]
            if gate is not None:
                args += [int(gate.data_ptr()), tuple(_strides(gate)[:4])]
            self._combine(*args, int(stream))
