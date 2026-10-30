# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical row order of a selected block list, in ONE launch: ids DESCENDING, ``-1`` past the position rule's count.

The block-sparse indexer's selectors return a row's ``top_k`` block ids in an unspecified slot order (the compact-logits path's
fused radix top-k: the SET is reproducible, the order is not), padded with ``-1`` where a row sees fewer than ``top_k`` blocks; the
gated attention block's contract on its outputs is the CANONICAL order -- ids strictly descending, then ``-1`` -- produced today by
``torch.sort`` windows (three device-to-device copies, two elementwise passes and a radix sort per 4096-row window, a 4 KB transient
each).  This kernel is that order as one launch without a transient, and it also fixes the LIST LENGTH from the POSITION RULE rather
than from what the scorer wrote: ``counts[t]`` is row ``t``'s visible count (``floor((pos_t + 1) / 4)``, the gather kernel's
``ix_counts``), and the row comes back with exactly ``n_t = min(top_k, counts[t])`` ids followed by ``-1`` -- a DEAD row (``counts
0``) all ``-1`` whatever the scorer left in it, so a padding sequence's list is defined without a memset of the scorer's output::

    out[t, :n_t]  = sort_descending(in[t, :top_k])[:n_t]       (the DSA pads with -1 and -1 < 0 <= id, so the padding sorts last)
    out[t, n_t:]  = -1
    block_lens_out[t] = n_t                                      (when given)

In place (``out is in``) is LEGAL: a warp holds its whole row in registers before its first store and rows are disjoint; the tests
pin the in-place result bitwise against the out-of-place one.  Two launches are bitwise (no atomic decides a slot).  ``top_k`` is a
RUNTIME value -- a multiple of 4 in ``[4, 512]`` (the 16-B granule holds four ids, so every granule of a row is whole) -- read off the
tensors' shape, so ONE artifact serves every ``top_k`` and every row count; the rows' stride is symbolic (a row slice of a wider
buffer binds), the column stride unit.

Work decomposition: ONE WARP per row (``threads_per_cta / 32`` rows per CTA, grid ``ceil(T / rows_per_cta)``).  Lane ``l`` owns the
16-B granules ``l + 32 k``, ``k = 0..3``, of the 512-slot row -- four ids each, so sixteen ids per lane; a granule at or past
``top_k / 4`` is never loaded (it reads as the ``INT_MIN`` sentinel, below every id and below the ``-1`` padding) and never stored.
The 512 slots are sorted by a BITONIC NETWORK in registers over the 32 lanes: element ``i = 128 k + 4 l + e2`` (register ``e = 4 k +
e2`` of lane ``l``), the 45 compare-exchange steps of the 9 stages generated at trace time; a step whose distance is a REGISTER bit
(``j`` in ``{1, 2, 128, 256}``) is a compare-exchange between two registers of one lane, a step whose distance is a LANE bit (``j`` in
``{4, 8, 16, 32, 64}``: 25 of the 45) fetches the partner's register through ``shfl.sync.bfly`` (xor ``j / 4`` = 1 / 2 / 4 / 8 / 16)
and keeps the max or the min by its lane's position and the stage's direction bit (``(i & stage) == 0``: descending blocks; the
last stage descending throughout) -- 400 shuffles per lane, no SMEM.  Every lane of the warp reaches every shuffle: a TAIL warp
(``row >= T``) computes on a clamped in-bounds row and stores nothing (the sibling kernels' tail rule).  After the network the owner lane
of slot ``s`` stores ``-1`` for ``s >= n_t`` and the id otherwise, lane 0 stores ``n_t`` when ``block_lens_out`` is given.

**Barrier table -- EMPTY (zero rows).**  No mbarrier, no ``cp.async`` group, no CTA barrier, no TMA, no tcgen05, no cluster.  The
only cross-lane dependency is ``shfl.sync`` with the full 32-lane mask inside ONE warp (the network's 25 shuffle steps), and every
lane of every warp reaches every shuffle (the tail rule above), so no shuffle is divergent; ``SUM(issuing lanes) == init_count`` has
no row to reconcile, and the ``frost-barrier-inspect`` audit of this kernel is THIS statement (the ``qsa_compress.py`` precedent), not
an audit skipped.
**SMEM table -- EMPTY (zero rows):** no ``cutlass.Array(space=smem)``, no ``SmemAllocator``, no ``smem=`` at the launch; the row moves
register <-> global through ``ld.global.v4`` / ``st.global.v4`` (16 B per lane) and the two counts through 4-B scalar accesses.
**Per-lane stride of every global access** (the SMEM table's column, applied to global so the coalescing is justified): lane ``l``
reads / writes granules ``l + 32 k`` at ``row + 16 (l + 32 k)`` B, so each ``ld/st.global.v4`` instruction covers 512 contiguous bytes
(four whole 128-B lines), four instructions per 2-KiB row (fewer granules at a smaller ``top_k``: the live ones stay contiguous);
``counts[t]`` one lane-uniform 4-B load per warp (an L1 broadcast); ``block_lens_out[t]`` one 4-B store by lane 0.
**``fence_proxy`` -- NONE:** no async proxy exists in this kernel (no TMA, no ``cp.async.bulk``, no tcgen05, no ``st.async``); its
generic-proxy stores are consumed by later kernels in stream order (a kernel boundary is a full proxy fence), so
``frost-tile-dsl.md`` § 1 has no site here: stated, not skipped.
**Rule 7 / the target:** NO arch pin (plain LDG / STG + ``shfl.sync``: any CuTe DSL device from sm_80); ``compile_qsa_canonical_list``
gates the DSL floor first (``cutedsl_requirement_error``).  The Rubin pin of the EXTEND form is the block's, not this kernel's.

Programmatic dependent launch (``use_pdl``): the ``griddepcontrol.wait`` precedes the first dependent read (the selector's ids, the
counts) and the ``launch_dependents`` release sits right after it (the streaming-kernel placement of ``cache_write.py``).
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

from .qk_norm_rope import ACCESS_BYTES

DEFAULT_THREADS_PER_CTA = 128
"""Four warps = four rows per CTA; a latency-bound pass over a few hundred to a few thousand rows.  Re-measure per arch before changing it."""

_WARP = 32
_ROW_SLOTS = 512  # the network's width: the largest top_k the block declares (QSA_TOP_K), a power of two
_GRANULES_PER_LANE = _ROW_SLOTS // (4 * _WARP)  # 4: lane l owns granules l + 32 k
_IDS_PER_GRANULE = ACCESS_BYTES // 4  # 4 int32 ids per 16-B granule
_INT_MIN = -2147483648
TOP_K_MIN, TOP_K_MAX, TOP_K_GRANULE = 4, _ROW_SLOTS, _IDS_PER_GRANULE
_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins it with a
# compile site's key to name the persistent object (``compiled_cache``).
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def validate_canonical_shape(threads_per_cta: int) -> None:
    """A CTA holds whole warps (one warp per row).  Never an ``assert``: user-facing geometry must survive ``python -O`` and name itself."""
    if threads_per_cta < _WARP or threads_per_cta % _WARP:
        raise ValueError(f"threads_per_cta must be a positive multiple of {_WARP} (one warp per row), got {threads_per_cta}")


def check_top_k(top_k: int) -> None:
    """``top_k`` against the kernel's contract: a multiple of {TOP_K_GRANULE} in [TOP_K_MIN, TOP_K_MAX] (whole 16-B granules; the network's width)."""
    if not isinstance(top_k, int) or top_k < TOP_K_MIN or top_k > TOP_K_MAX or top_k % TOP_K_GRANULE:
        raise ValueError(f"top_k must be a multiple of {TOP_K_GRANULE} in [{TOP_K_MIN}, {TOP_K_MAX}] (whole 16-byte granules of a 512-slot row), got {top_k}")


def _bit_is_lane(bit: int) -> bool:
    """Whether bit ``bit`` of the element index ``i = 128 k + 4 l + e2`` is a LANE bit (bits 2..6) or a REGISTER bit (bits 0, 1, 7, 8)."""
    return 4 <= bit <= 64


def _reg_part(e: int) -> int:
    """The register-bit part of the element index of register ``e = 4 k + e2``: ``128 k + e2``."""
    return 128 * (e >> 2) + (e & 3)


def _reg_of_part(part: int) -> int:
    """The inverse of :func:`_reg_part`."""
    return 4 * (part >> 7) + (part & 3)


def _bit_flag(lane: cutlass.Int32, bit: int) -> cutlass.Int32:
    """``(lane & (bit / 4)) != 0`` as an Int32 0 / 1 for a lane bit of the element index."""
    shift = (bit // 4).bit_length() - 1
    return (lane >> cutlass.Int32(shift)) & cutlass.Int32(1)


@cute.jit
def _max_min(a: cutlass.Int32, b: cutlass.Int32):
    """``(max(a, b), min(a, b))`` of two ids -- the compare-exchange of one network step (a staged select, no branch)."""
    hi = a if a > b else b
    lo = b if a > b else a
    return hi, lo


@cute.jit
def _select(cond: cutlass.Boolean, x: cutlass.Int32, y: cutlass.Int32) -> cutlass.Int32:
    """``x if cond else y`` as a staged select (the network generator below is plain Python, where a ternary on a traced value is illegal)."""
    return x if cond else y


def _bitonic_descending(vals, lane: cutlass.Int32):
    """The 45-step bitonic network over the 512 elements ``i = 128 k + 4 l + e2`` held as ``vals[4 k + e2]`` on lane ``l``, in place
    on the Python list; the result is DESCENDING along ``i``.  Trace-time Python (a generator, not a jit body): the stages and the
    per-register pairing are compile-time, the per-lane flags runtime per step, every select through :func:`_max_min` /
    :func:`_select` (module docstring)."""
    stage = 2
    while stage <= _ROW_SLOTS:
        j = stage // 2
        while j >= 1:
            if stage == _ROW_SLOTS:
                desc_rt = None  # the last stage: descending throughout ((i & 512) == 0 for every i)
                desc_ct = True
            elif _bit_is_lane(stage):
                desc_rt = _bit_flag(lane, stage)  # 0 -> a descending block, 1 -> an ascending one (runtime, per lane)
                desc_ct = None
            else:
                desc_rt = None
                desc_ct = None  # per register: (reg_part & stage) == 0
            if _bit_is_lane(j):
                # a LANE step: the partner holds the same register on lane l ^ (j / 4); take the max when (lower == desc)
                upper = _bit_flag(lane, j)  # 1 when the calling lane holds the UPPER index of the pair
                for e in range(len(vals)):
                    other = cute.arch.shuffle_sync_bfly(vals[e], j // 4)
                    hi, lo = _max_min(vals[e], other)
                    if desc_ct is not None:
                        desc_e = desc_ct
                    elif desc_rt is None:
                        desc_e = (_reg_part(e) & stage) == 0
                    else:
                        desc_e = None
                    if desc_e is None:
                        # take_max = (upper == 0) == (desc_rt == 0)  <=>  (upper ^ desc_rt) == 0
                        take_max = (upper ^ desc_rt) == cutlass.Int32(0)
                    elif desc_e:
                        take_max = upper == cutlass.Int32(0)
                    else:
                        take_max = upper == cutlass.Int32(1)
                    vals[e] = _select(take_max, hi, lo)
            else:
                # a REGISTER step: the pair (e, e') of one lane with reg_part(e') = reg_part(e) ^ j, e the lower index
                for e in range(len(vals)):
                    part = _reg_part(e)
                    if part & j:
                        continue
                    e2 = _reg_of_part(part ^ j)
                    hi, lo = _max_min(vals[e], vals[e2])
                    if desc_ct is not None:
                        desc_e = desc_ct
                    elif desc_rt is None:
                        desc_e = (part & stage) == 0
                    else:
                        desc_e = None
                    if desc_e is None:
                        desc_b = desc_rt == cutlass.Int32(0)
                        vals[e] = _select(desc_b, hi, lo)
                        vals[e2] = _select(desc_b, lo, hi)
                    elif desc_e:
                        vals[e], vals[e2] = hi, lo
                    else:
                        vals[e], vals[e2] = lo, hi
            j //= 2
        stage *= 2
    return vals


@cute.kernel
def frost_qsa_canonical_list(
    mIn: cute.Tensor,  # [T, top_k] int32: the selected ids, any slot order, -1 padded; row stride symbolic, unit column stride
    mCounts: cute.Tensor,  # [T] int32: the visible count per row (the position rule), clamped in-kernel to [0, top_k]
    mOut: cute.Tensor,  # [T, top_k] int32 out (may alias mIn): ids descending, -1 past min(top_k, counts)
    mLens: Optional[cute.Tensor],  # [T] int32 out: min(top_k, counts), or None
    n_rows: cutlass.Int32,
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One warp per row (module docstring)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
        launch_dependent_grids()

    has_lens = cutlass.const_expr(mLens is not None)
    warps_per_cta = cutlass.const_expr(threads_per_cta // _WARP)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(_WARP)
    row = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(warps_per_cta) + tidx // cutlass.Int32(_WARP)
    in_row = row < n_rows
    t = row if in_row else n_rows - cutlass.Int32(1)  # the tail warp computes on the last row and stores nothing
    t64 = t.to(cutlass.Int64)

    top_k = cutlass.Int32(mIn.shape[1])
    n_gran = top_k // cutlass.Int32(_IDS_PER_GRANULE)  # the live granules of a row (top_k a multiple of 4, host-checked)
    in_base = mIn.iterator.toint() + t64 * cutlass.Int64(mIn.stride[0]) * cutlass.Int64(4)
    out_base = mOut.iterator.toint() + t64 * cutlass.Int64(mOut.stride[0]) * cutlass.Int64(4)
    sentinel = cutlass.Int32(_INT_MIN)

    # the row into registers: granule l + 32 k -> registers 4 k .. 4 k + 3; a granule past top_k reads as the sentinel
    vals = []
    for k in cutlass.range_constexpr(_GRANULES_PER_LANE):
        g = lane + cutlass.Int32(_WARP * k)
        live = g < n_gran
        g_r = g if live else n_gran - cutlass.Int32(1)  # in-bounds (n_gran >= 1)
        words = ld_global_v4(in_base + g_r.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES), cutlass.Int32)
        for e2 in cutlass.range_constexpr(_IDS_PER_GRANULE):
            vals.append(words[e2] if live else sentinel)

    vals = _bitonic_descending(vals, lane)

    # the list length from the POSITION rule, then the stores: -1 at and past it, the sorted id below it
    cnt = ld_global(mCounts.iterator.toint() + t64 * cutlass.Int64(4), cutlass.Int32)
    cnt = cnt if cnt > cutlass.Int32(0) else cutlass.Int32(0)
    n_t = cnt if cnt < top_k else top_k
    minus_one = cutlass.Int32(-1)
    for k in cutlass.range_constexpr(_GRANULES_PER_LANE):
        g = lane + cutlass.Int32(_WARP * k)
        live = g < n_gran
        slot0 = g * cutlass.Int32(_IDS_PER_GRANULE)
        outs = []
        for e2 in cutlass.range_constexpr(_IDS_PER_GRANULE):
            slot = slot0 + cutlass.Int32(e2)
            v = vals[k * _IDS_PER_GRANULE + e2]
            outs.append(v if slot < n_t else minus_one)
        if in_row & live:
            st_global_v4(out_base + g.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES), outs, cutlass.Int32)
    if cutlass.const_expr(has_lens):
        if in_row & (lane == cutlass.Int32(0)):
            st_global(mLens.iterator.toint() + t64 * cutlass.Int64(4), n_t, cutlass.Int32)


@cute.jit
def qsa_canonical_list_launch(
    ids_in: cute.Tensor,
    counts: cute.Tensor,
    ids_out: cute.Tensor,
    lens: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    n_blocks: cutlass.Int32,
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_canonical_list(ids_in, counts, ids_out, lens, n_rows, threads_per_cta, use_pdl).launch(
        grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl
    )


compiled_cache = {}


class QsaCanonicalListRecipe(NamedTuple):
    """Build-time facts of one canonical-order launch: the CTA shape, PDL, whether the per-row list length is written (Rule 4: one
    artifact per (threads, use_pdl, with_lens, device)); ``top_k`` and the row count are runtime shapes."""

    compiled: object
    threads_per_cta: int
    rows_per_cta: int
    use_pdl: bool
    with_lens: bool
    device: int = -1  # appended: the CUDA device the artifact was compiled for (part of its key); the launch selects it, operands elsewhere are refused


def _fake_ids():
    """``[T, top_k]`` int32 with a SYMBOLIC row stride and a unit column stride, 16-byte aligned (whole granules)."""
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=16)


def _fake_i32_vec():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4)


def compile_qsa_canonical_list(*, threads_per_cta: int = DEFAULT_THREADS_PER_CTA, use_pdl: bool = False, with_lens: bool = False) -> QsaCanonicalListRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch; the DSL floor is gated first (Rule 7).  ``with_lens`` traces the
    ``block_lens_out`` store (a presence switch, hence a compile key).  Every ``Constexpr`` is in the persistent key."""
    global _FAKE_STREAM
    too_old = cutedsl_requirement_error("qsa_canonical_list")
    if too_old is not None:
        raise NotImplementedError(too_old)
    validate_canonical_shape(threads_per_cta)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (int(threads_per_cta), bool(use_pdl), bool(with_lens), current_device())
    if key not in compiled_cache:
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_canonical_list")
        with torch.cuda.device(key[-1]):  # the compile and the DSL's load of the launchable run ON the artifact's device (the recipe's), whichever
            compiled_cache[key] = compile_cached(  # device is live -- a build_device() scope names a device the live context need not be
                qsa_canonical_list_launch,
                _fake_ids(),
                _fake_i32_vec(),
                _fake_ids(),
                _fake_i32_vec() if with_lens else None,
                cutlass.Int32(1),  # n_rows   ) runtime; the literals pin the TYPE only
                cutlass.Int32(1),  # n_blocks )
                int(threads_per_cta),
                bool(use_pdl),
                _FAKE_STREAM,
                options="--enable-tvm-ffi",
                cache_key=cache_key,
                symbol="cudnn_block_qsa_canonical_list",
            )
    return QsaCanonicalListRecipe(
        compiled=compiled_cache[key],
        threads_per_cta=int(threads_per_cta),
        rows_per_cta=int(threads_per_cta) // _WARP,
        use_pdl=bool(use_pdl),
        with_lens=bool(with_lens),
        device=key[-1],
    )


def _check_ids(name: str, x, device) -> None:
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.int32:
        raise ValueError(f"{name} must be a 2-D int32 [T, top_k] tensor, got {None if not isinstance(x, torch.Tensor) else f'{x.dtype} {tuple(x.shape)}'}")
    if not x.is_cuda or (device is not None and x.device != device):
        raise ValueError(f"{name} must be a CUDA tensor on {device if device is not None else 'the device'}, got device {x.device}")
    if int(x.stride(1)) != 1:
        raise ValueError(f"{name} must have a unit column stride, got strides {tuple(x.stride())}")
    if (int(x.stride(0)) * 4) % ACCESS_BYTES or x.data_ptr() % ACCESS_BYTES:
        raise ValueError(f"{name} rows must be 16-byte aligned (row stride {x.stride(0)} ids, data_ptr {x.data_ptr():#x}): whole granules")


def run_qsa_canonical_list(
    r: QsaCanonicalListRecipe,
    ids_in: torch.Tensor,
    counts: torch.Tensor,
    ids_out: torch.Tensor,
    block_lens_out: Optional[torch.Tensor] = None,
    *,
    top_k: Optional[int] = None,
    stream,
) -> None:
    """Launch.  ``ids_in`` / ``ids_out`` ``[T, top_k]`` int32 (unit column stride, 16-B aligned rows; ``ids_out`` may BE ``ids_in`` --
    the same storage at the same strides -- but never partly overlap it), ``counts [T]`` int32 contiguous, ``block_lens_out [T]`` int32
    contiguous iff the recipe traced ``with_lens`` (refused both ways); ``top_k`` (optional) must equal the tensors' width, a multiple
    of 4 in [4, 512].  Form checks only: no device read, no allocation."""
    _check_ids("ids_in", ids_in, None)
    dev = ids_in.device
    _check_ids("ids_out", ids_out, dev)
    t, k = (int(x) for x in ids_in.shape)
    if t < 1:
        raise ValueError("ids_in must hold at least one row")
    check_top_k(k)
    if top_k is not None and int(top_k) != k:
        raise ValueError(f"top_k={top_k} does not match the ids' width {k}")
    if tuple(ids_out.shape) != (t, k):
        raise ValueError(f"ids_out must be [T={t}, top_k={k}] like ids_in, got {tuple(ids_out.shape)}")
    same = ids_out.data_ptr() == ids_in.data_ptr() and ids_out.stride() == ids_in.stride()
    if not same:
        lo_i, hi_i = ids_in.data_ptr(), ids_in.data_ptr() + ((t - 1) * int(ids_in.stride(0)) + k) * 4
        lo_o, hi_o = ids_out.data_ptr(), ids_out.data_ptr() + ((t - 1) * int(ids_out.stride(0)) + k) * 4
        if lo_i < hi_o and lo_o < hi_i:
            raise ValueError("ids_out overlaps ids_in without being the same rows at the same strides: in place is legal, a partial overlap is not")
    if not isinstance(counts, torch.Tensor) or counts.dtype != torch.int32 or tuple(counts.shape) != (t,) or not counts.is_contiguous() or counts.device != dev:
        got = (
            None if not isinstance(counts, torch.Tensor) else f"{counts.dtype} {tuple(counts.shape)} contiguous={counts.is_contiguous()} device={counts.device}"
        )
        raise ValueError(f"counts must be a contiguous int32 [T={t}] tensor on {dev} (the visible count per row, the position rule), got {got}")
    if r.with_lens:
        if block_lens_out is None:
            raise ValueError("this artifact was compiled WITH the per-row list length (with_lens=True); bind block_lens_out [T] int32 (no silent fallback)")
        if (
            not isinstance(block_lens_out, torch.Tensor)
            or block_lens_out.dtype != torch.int32
            or tuple(block_lens_out.shape) != (t,)
            or not block_lens_out.is_contiguous()
            or block_lens_out.device != dev
        ):
            raise ValueError(f"block_lens_out must be a contiguous int32 [T={t}] tensor on {dev}")
    elif block_lens_out is not None:
        raise ValueError("this artifact was compiled WITHOUT the per-row list length (with_lens=False); block_lens_out would be silently ignored -- pass None")
    if dev.index is not None and dev.index != r.device:
        raise ValueError(f"this artifact was compiled for CUDA device {r.device}; the operands live on {dev} (compile one per device)")
    n_blocks = (t + r.rows_per_cta - 1) // r.rows_per_cta
    with torch.cuda.device(r.device):  # the launch runs in the artifact's context whichever device is current at the call
        r.compiled(ids_in, counts, ids_out, block_lens_out, cutlass.Int32(t), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


def build_qsa_canonical_list(
    ids_in: torch.Tensor,
    counts: torch.Tensor,
    ids_out: torch.Tensor,
    block_lens_out: Optional[torch.Tensor] = None,
    *,
    top_k: Optional[int] = None,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    stream,
) -> QsaCanonicalListRecipe:
    """Compile (cached) and run once -- the convenience form for tests.  Production callers split it: :func:`compile_qsa_canonical_list`
    at plan time, :func:`run_qsa_canonical_list` per execute."""
    r = compile_qsa_canonical_list(threads_per_cta=threads_per_cta, use_pdl=use_pdl, with_lens=block_lens_out is not None)
    run_qsa_canonical_list(r, ids_in, counts, ids_out, block_lens_out, top_k=top_k, stream=stream)
    return r


def moved_bytes(t: int, top_k: int) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number: every row read once and written once, plus the two counts."""
    return 2 * int(t) * int(top_k) * 4 + 2 * int(t) * 4


frost_qsa_canonical_list.set_name_prefix("cudnn", remove_cutlass_symbol=True)
