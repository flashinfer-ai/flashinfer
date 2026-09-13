# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""SM120 KDA prefill, decomposed: a chunk-parallel prepare and a serial recurrence.

Two device kernels issued through one compiled host entry::

    prepare
        |-- Q/K L2 normalization
        |-- gate activation and prefix scan
        +-- Kd/Qd/Ak/Aq/GTotal materialization

    recurrence
        |-- consume the chunk factors
        |-- carry a [128, 128] state in sequence order
        |-- write the output
        +-- update and store the final state

Everything this variant owns lives here: its SMEM images and swizzles, its
variant-specific inline PTX, its TMA descriptors, both device kernels, the
combined compiled entry, its ``cu_chunks``/``chunk_to_seq`` metadata, its
prepare scratch, its descriptor and call-plan caches, and its current-stream
launch.  What it shares with :mod:`.fused` comes from two sibling modules:
host mechanisms -- bounded caches, capture detection, canonical INT32 offsets,
the workspace resource slot and the ``sm_120a`` target check -- from
:mod:`.runtime`, and the device-side PTX wrappers, fragment constants and S128
geometry from :mod:`.device_common`.

Host plan and device kernels share one file because a chunk size, an SMEM
offset or a barrier id is one decision read by both sides.  Nothing in this
file imports ``fused``.

Chunk size 16, the SMEM arena offsets, the swizzles, the barrier arena, the
grid, the chunks-per-CTA policy, the rounding boundaries and the state ABI are
fixed implementation choices shared by the host plan, device code and tests.
"""

from __future__ import annotations

import os
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cuda.bindings import driver as cuda_driver
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .runtime import (
    assert_tvm_ffi_dispatched,
    BoundedDeviceCache,
    build_kernel,
    capturing,
    check_flat_output_range,
    DESCRIPTOR_BYTES,
    descriptor_cache_key,
    DK,
    DV,
    execute,
    flat_view,
    GRAPH_PINS,
    IdentityCache,
    INT32_MAX,
    KDAPrefillValidationError,
    LOG2_E,
    max_grid_dims,
    NO_VERSION,
    NORM_FLOOR,
    PlanMemo,
    PREFIX_FLOOR,
    record_stream_once,
    require_sm120a,
    sm120a_compile_options,
    STATE_ONLY_PLAN,
    tensor_version,
    TensorMapSpec,
    upload_bytes,
)
from .device_common import (
    BF16_GROUP_ELEMS,
    BF16_SEGMENTS,
    BF16_SEGMENT_ELEMS,
    BF16_SEGMENT_STRIDE,
    BT,
    F32_GROUP_ELEMS,
    F32_SEGMENTS,
    F32_SEGMENT_ELEMS,
    F32_SEGMENT_STRIDE,
    KEY_BLOCKS,
    STATE_BF16_ROWS_PER_VALUE,
    STATE_F32_ROWS_PER_VALUE,
    _s128_index,
    a_to_b,
    bf16_round,
    clear_tail_rows,
    f16_round,
    fence_tensormap_acquire,
    ldmatrix_x2,
    ldmatrix_x2_trans,
    ldmatrix_x4,
    ldmatrix_x4_trans,
    mma_16x16,
    mma_16x16_f16,
    mma_n8,
    movmatrix_b16,
    mul_bf16x2,
    pack_bf16x2,
    pack_f16x2,
    pairwise_sw32,
    raw_bf16_s128,
    raw_f32_s128,
    state_bf16_idx,
    state_x2_ptr,
    stmatrix_x2,
    stmatrix_x2_trans,
    stmatrix_x4,
    store_vec4_f32,
    store_vec8_bf16,
    sub_bf16x2,
    tma_load_3d,
    tma_store_3d,
    tma_store_commit_group,
    tma_store_wait_read,
    unpack_bf16x2,
    vec8_bf16,
    vec_at,
    vo_x2_ptr,
    warp_arrive,
)


# --------------------------------------------------------------------------
# Canonical SMEM and global index mappings
#
# These are the decomp variant's images.  The fused variant computes
# its own in its own file: the two agree on the S128 construction and
# disagree on which logical shape it is applied to, so they are not
# one helper with two callers.
# --------------------------------------------------------------------------

DV_HALF = DV // 2


# Constant coordinate permutations.
KR_AK_TOKEN_XOR = BT // 2  # 8, token-row permutation for the Ak.T image


#: Short alias for :func:`pairwise_sw32`.
pair_idx = pairwise_sw32


# ---------------------------------------------------------------------------
# Native m16n8k16 fragment maps.
#
# These are fixed by the PTX ISA and restated here so no consumer has to
# rediscover them.  ``g = lane >> 2`` and ``q = lane & 3`` throughout.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recurrence images.
#
# The recurrence reuses ``raw_bf16_s128`` unchanged for its Kd/Qd/Ak stages and
# ``pairwise_sw32`` unchanged for Aq; the three images below are new.  Every one
# is the same S128 construction -- 128-byte segments, 16-byte groups, group
# index XORed with the low bits of the segment row -- applied to a different
# logical shape, so ``_s128_index`` states it once.
# ---------------------------------------------------------------------------

#: ``Kd``/``Qd``/``Ak`` land in the existing raw BF16 image; the recurrence
#: spells it ``factor_idx`` because its rows are chunk tokens, not raw tokens.
factor_idx = raw_bf16_s128

#: The V/output half is 64 values wide, so it has one 128-byte segment per row.
VO_ROW_ELEMS = DV_HALF  # 64 BF16 = 128 B


def vo_idx(row, v_local):
    """Physical BF16 index of ``(token row, value)`` in a V/output half stage.

    ``v_local`` is in ``[0, 64)`` and is the CTA-local value, so the two DV
    halves address byte-identical stages at different global coordinates.
    """
    return _s128_index(row, v_local, VO_ROW_ELEMS, BF16_GROUP_ELEMS)


def state_f32_idx(v_local, k):
    """Physical FP32 index of ``state[v][k]`` in the conversion buffer.

    Only the external-state boundary uses this image: an FP32 initial state
    lands here and is rounded into the BF16 state, and an FP32 final state is
    widened back into it after the pipeline drains.
    """
    segment = k // F32_SEGMENT_ELEMS
    local = k - segment * F32_SEGMENT_ELEMS
    line = STATE_F32_ROWS_PER_VALUE * v_local + segment
    return _s128_index(line, local, F32_SEGMENT_ELEMS, F32_GROUP_ELEMS)


def vo_global_index(token, head, heads, dv_half, v_local):
    """Flat element index of ``out[0, token, head, 64 * dv_half + v_local]``.

    The partial-output tail stores through this map with 16-byte vectors; the
    full path never needs it, because TMA addresses the same
    element through the descriptor instead.
    """
    return (token * heads + head) * DV + dv_half * DV_HALF + v_local


# ---------------------------------------------------------------------------
# Native N=8 fragment maps.
#
# The prepare kernel only ever issues logical m16n16k16 steps, so its maps are
# written per n-block.  The recurrence's N is the eight value columns a warp
# owns, which is exactly one native MMA, so the accumulator map below drops the
# n-block term rather than pinning it to zero at every call site.
# ---------------------------------------------------------------------------

#: Value columns one compute warp owns.
WARP_VALUES = 8
#: Compute warps per CTA; ``COMPUTE_WARPS * WARP_VALUES == DV_HALF``.
COMPUTE_WARPS = DV_HALF // WARP_VALUES


def factor_a_fragment_ptr(lane, kb):
    """SMEM index lane ``lane`` addresses for a ``Kd``/``Qd`` A-operand x4 load.

    Both stages are token-major, which is the A operand's orientation already,
    so this is the plain (non-transposed) x4 map at key block ``kb``.
    """
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    col = kb * BT + 8 * (matrix_id // 2)
    return factor_idx(row, col)


def pairwise_a_fragment_ptr(lane):
    """SMEM index lane ``lane`` addresses for the ``Aq`` A-operand x4 load."""
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    col = 8 * (matrix_id // 2)
    return pairwise_sw32(row, col)


def ak_a_fragment_ptr(lane, kb):
    """SMEM index lane ``lane`` addresses for the ``Ak`` A-operand x4 trans.

    ``Ak`` is published by prepare as ``Ak.T`` with a ``token ^ 8`` row
    permutation, so the stage holds ``[token][key]`` while the MMA wants
    ``[key][token]``.  ``ldmatrix.x4.trans`` supplies the transpose, and the
    four returned registers are ``(a0, a1, a2, a3)`` directly and must not be
    permuted afterwards.
    """
    matrix_id = lane // 8
    row8 = lane - matrix_id * 8
    logical_j = (matrix_id // 2) * 8 + row8
    key = kb * BT + (matrix_id - (matrix_id // 2) * 2) * 8
    return factor_idx(logical_j ^ KR_AK_TOKEN_XOR, key)


# --------------------------------------------------------------------------
# CuTe DSL names that moved between releases
# --------------------------------------------------------------------------

#: 4.7 renamed ``make_fragment`` to ``make_rmem_tensor``; the signature
#: ``(layout_or_shape, dtype, *, loc=None, ip=None) -> Tensor`` is unchanged,
#: so one alias covers every call site.
make_rmem_tensor = getattr(cute, "make_rmem_tensor", None) or cute.make_fragment


# --------------------------------------------------------------------------
# Inline PTX the decomp kernels issue
# --------------------------------------------------------------------------


@dsl_user_op
def cp_async_16(smem_ptr, gmem_ptr, src_bytes, *, loc=None, ip=None):
    """``cp.async.cg.shared.global`` of 16 bytes.

    ``src_bytes`` is the PTX src-size operand: pass 16 for a real copy and 0 to
    have the hardware zero-fill the destination instead.  That is exactly what
    an out-of-range tail row needs, so no separate tail-clear pass is required.
    """
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None):
    """Close the current ``cp.async`` group (the non-bulk, Ampere family).

    Distinct from ``tma_store_commit_group`` (``device_common``), which closes
    a ``cp.async.bulk`` group: the two instruction families keep separate
    per-thread group FIFOs, and a wait on one says nothing about the other.
    """
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_wait_group(keep: int, *, loc=None, ip=None):
    """``cp.async.wait_group keep``: block until at most ``keep`` groups remain.

    ``keep`` is a PTX immediate, so it must be a Python int at trace time.
    Completion makes the copied bytes visible to the waiting thread only;
    publishing them to another warp still takes the mbarrier release/acquire
    pair, exactly as with any other shared-memory store.
    """
    llvm.inline_asm(
        None,
        [],
        f"cp.async.wait_group {int(keep)};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# TMA
#
# The load/store wrappers both kernels issue live in ``device_common``.
# Explicit PTX keeps transaction-barrier completion, bulk-store groups and
# proxy fences visible in the schedule.  Descriptors come from
# cuTensorMapEncodeTiled on the host and live in device memory; the kernel gets
# their addresses as Int64 scalars.  Only the publication-side fences the
# overlap needs are defined here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-kernel publication (the prepare/recurrence overlap)
#
# A flag in GMEM releases one chunk's factor slab to a *concurrently running*
# reader.  Release/acquire at gpu scope carries the LSU stores by cumulativity
# through the CTA barrier; the TMA stores additionally need their bulk groups
# *complete* (not just source-read) and an async-proxy fence before the flag.
# ---------------------------------------------------------------------------


@dsl_user_op
def tma_store_wait_all(*, loc=None, ip=None):
    """``cp.async.bulk.wait_group 0``: every bulk-store group of this thread
    is complete, written bytes included.

    The plain form, unlike ``.read`` (``tma_store_wait_read``), is a
    global-visibility guarantee, which is what a flag published to another
    kernel has to stand on.
    """
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group 0;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    """``fence.proxy.async.global``: order async-proxy (TMA) accesses with
    generic-proxy ones, so a plain flag store can publish TMA-stored bytes."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_release_gpu_u32(gmem_ptr, value, *, loc=None, ip=None):
    """``st.release.gpu.global.u32``: publish a flag at GPU scope.

    Release cumulativity covers every write the storing thread has observed,
    its own and anything a CTA barrier ordered before it, so one thread can
    publish a whole CTA's chunk.
    """
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def spin_wait_flag_ge_u32(
    gmem_ptr, target, tag: str = "f", ns: int = 128, *, loc=None, ip=None
):
    """Spin on ``ld.acquire.gpu.global.u32`` until ``*ptr >= target`` (unsigned).

    ``nanosleep`` between polls keeps the spinning warp off the issue
    ports.  The acquire on the successful poll is the consumer half of the
    ``st_release_gpu_u32`` pair.  ``tag`` must differ between two call sites
    that could land in one compiled kernel: it names the PTX branch label.
    """
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(target).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .pred p_{tag};\n\t"
        f".reg .u32 v_{tag};\n\t"
        f"spin_{tag}:\n\t"
        f"ld.acquire.gpu.global.u32 v_{tag}, [$0];\n\t"
        f"setp.lt.u32 p_{tag}, v_{tag}, $1;\n\t"
        f"@p_{tag} nanosleep.u32 {int(ns)};\n\t"
        f"@p_{tag} bra spin_{tag};\n\t"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_gpu_u32(gmem_ptr, *, loc=None, ip=None):
    """One non-blocking ``ld.acquire.gpu.global.u32``; returns the value.

    The lookahead half of the flag protocol: issued a few chunks ahead so its
    L2 round trip overlaps issue work instead of sitting on the critical path,
    where the blocking spin would put it.
    """
    from cutlass._mlir.extras import types as _T

    val = llvm.inline_asm(
        _T.IntegerType.get_signless(32),
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.acquire.gpu.global.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(val)


@dsl_user_op
def ld_relaxed_gpu_u32(gmem_ptr, *, loc=None, ip=None):
    """One ``ld.relaxed.gpu.global.u32``; returns the value, orders nothing.

    Delay the acquire until the lookahead is consumed, so the poll does not
    order the current chunk's TMA issues behind a future flag load.  A hit
    must execute ``fence_acquire_gpu`` before reading the published factors:
    control dependence and L2 reads alone do not synchronize with release.
    """
    from cutlass._mlir.extras import types as _T

    val = llvm.inline_asm(
        _T.IntegerType.get_signless(32),
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(val)


@dsl_user_op
def fence_acquire_gpu(*, loc=None, ip=None):
    """Complete the acquire pattern after a successful relaxed flag read."""
    llvm.inline_asm(
        None,
        [],
        "fence.acquire.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# --------------------------------------------------------------------------
# The prepare workspace and this variant's chunk metadata
#
# The chunk-to-sequence map and the packed factor arena exist because
# prepare is chunk-parallel; the fused variant runs one CTA per
# (sequence, head) and needs neither, so neither is shared.  What both
# variants do need -- a validated canonical INT32 ``cu_seqlens`` -- comes
# from ``runtime.canonical_offsets`` instead, and this section derives
# ``cu_chunks``/``chunk_to_seq`` from it.
# --------------------------------------------------------------------------

CHUNK = BT
DIMENSION = DK

REGION_ALIGNMENT = 256

#: 3 * 4096 (Kd, Qd, Ak) + 512 (Aq) + 512 (GTotal)
BYTES_PER_HEAD_CHUNK = (
    3 * (CHUNK * DIMENSION * 2) + (CHUNK * CHUNK * 2) + (DIMENSION * 4)
)
assert BYTES_PER_HEAD_CHUNK == 13312


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def chunks_for_lengths(lengths) -> list[int]:
    """``ceil_div(len, 16)`` per sequence."""
    return [(int(length) + CHUNK - 1) // CHUNK for length in lengths]


@dataclass(frozen=True)
class PrepareWorkspace:
    """Typed views over one packed ``uint8`` storage tensor."""

    storage: torch.Tensor
    kd: torch.Tensor
    qd: torch.Tensor
    ak: torch.Tensor
    aq: torch.Tensor
    g_total: torch.Tensor
    heads: int
    total_chunks: int
    # A decomp forward is a compound enqueue: prepare writes this workspace,
    # then recurrence consumes it.  CUDA orders individual launches submitted
    # to one stream, but two host threads can interleave those two-launch
    # sequences.  Every plan that owns this workspace therefore shares this
    # host lock and holds it until both launches have been submitted.
    launch_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    #: The overlap's ``[flag_tensor, generation]`` pair, filled on first use by
    #: :func:`_acquire_flags` and empty until then.  It lives with the
    #: workspace so every plan on the workspace shares one generation counter
    #: and a workspace the LRU drops takes its flags with it.
    pipe_flags: list = field(default_factory=list, repr=False, compare=False)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "kd": self.kd,
            "qd": self.qd,
            "ak": self.ak,
            "aq": self.aq,
            "g_total": self.g_total,
        }


def prepare_region_offsets(heads: int, total_chunks: int) -> dict[str, tuple[int, int]]:
    """Return ``{name: (byte_offset, byte_size)}`` in physical arena order."""
    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    if total_chunks < 0:
        raise ValueError(f"total_chunks must be non-negative, got {total_chunks}")

    rows = heads * total_chunks * CHUNK
    vector_bytes = rows * DIMENSION * 2
    aq_bytes = heads * total_chunks * (CHUNK * CHUNK) * 2
    gt_bytes = heads * total_chunks * DIMENSION * 4

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, size in (
        ("kd", vector_bytes),
        ("qd", vector_bytes),
        ("ak", vector_bytes),
        ("aq", aq_bytes),
        ("g_total", gt_bytes),
    ):
        cursor = _align_up(cursor, REGION_ALIGNMENT)
        offsets[name] = (cursor, size)
        cursor += size
    return offsets


def prepare_workspace_size(heads: int, total_chunks: int) -> int:
    """Total bytes of the packed prepare workspace."""
    offsets = prepare_region_offsets(heads, total_chunks)
    last_offset, last_size = offsets["g_total"]
    return _align_up(last_offset + last_size, REGION_ALIGNMENT)


def partition_prepare_workspace(
    storage: torch.Tensor, heads: int, total_chunks: int
) -> PrepareWorkspace:
    """Carve the five typed physical views out of one ``uint8`` storage."""
    if storage.dtype != torch.uint8:
        raise TypeError(f"workspace storage must be uint8, got {storage.dtype}")
    if storage.ndim != 1:
        raise ValueError("workspace storage must be 1-D")
    if not storage.is_contiguous():
        raise ValueError("workspace storage must be contiguous")

    required = prepare_workspace_size(heads, total_chunks)
    if storage.numel() < required:
        raise ValueError(
            f"workspace storage has {storage.numel()} bytes, need {required}"
        )
    # Region offsets are 256-byte aligned relative to the base, so the regions
    # are absolutely aligned only when the base is.  The CUDA caching allocator
    # always returns >=256-byte-aligned blocks, which is what the TMA
    # descriptors need; host allocations are a testing convenience and are not
    # held to that.
    if storage.is_cuda and storage.data_ptr() % REGION_ALIGNMENT:
        raise ValueError(
            f"workspace storage must be {REGION_ALIGNMENT}-byte aligned, "
            f"got offset {storage.data_ptr() % REGION_ALIGNMENT}"
        )

    offsets = prepare_region_offsets(heads, total_chunks)
    rows = total_chunks * CHUNK

    def view(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        offset, size = offsets[name]
        raw = storage[offset : offset + size]
        return raw.view(dtype).view(shape)

    return PrepareWorkspace(
        storage=storage,
        kd=view("kd", torch.bfloat16, (1, heads, rows, DIMENSION)),
        qd=view("qd", torch.bfloat16, (1, heads, rows, DIMENSION)),
        ak=view("ak", torch.bfloat16, (1, heads, rows, DIMENSION)),
        aq=view("aq", torch.bfloat16, (1, heads, total_chunks, CHUNK * CHUNK)),
        g_total=view("g_total", torch.float32, (1, heads, total_chunks, DIMENSION)),
        heads=heads,
        total_chunks=total_chunks,
    )


def allocate_prepare_workspace(
    heads: int, total_chunks: int, device: torch.device | str
) -> PrepareWorkspace:
    size = prepare_workspace_size(heads, total_chunks)
    raw = torch.empty(size + REGION_ALIGNMENT, dtype=torch.uint8, device=device)
    pad = (-raw.data_ptr()) % REGION_ALIGNMENT
    storage = raw[pad : pad + size]
    return partition_prepare_workspace(storage, heads, total_chunks)


#: One entry per (heads, total_chunks, stream) in flight.  A handful of shapes
#: on one or two streams is the normal case; the bound keeps a workload that
#: varies its chunk count from pinning every size it ever saw.
PREPARE_WORKSPACE_MAX_ENTRIES = 8

#: Reused scratch, keyed by shape *and stream*.  Allocating a fresh workspace
#: per call cost more than the two kernels themselves on short shapes.
#:
#: The stream is part of the key on purpose.  This buffer is written, not read,
#: so two forwards running concurrently on different streams must not share
#: one: unlike the descriptor caches, a wait_event on the entry would order the
#: reader against its *creation*, not against the previous writer.  Giving each
#: stream its own workspace removes the race rather than trying to order it.
_WORKSPACES = BoundedDeviceCache(
    "prepare-workspace", max_entries=PREPARE_WORKSPACE_MAX_ENTRIES
)
# ``BoundedDeviceCache`` deliberately handles device lifetime rather than
# compound get-or-create atomicity.  Serialize that compound operation here so
# two host threads using the same CUDA stream receive the same workspace and,
# critically, the same launch lock.
_WORKSPACES_CACHE_LOCK = threading.Lock()


def acquire_prepare_workspace(
    heads: int, total_chunks: int, device: torch.device | str
) -> PrepareWorkspace:
    """A workspace for this shape, reused across calls on the same stream."""
    device = torch.device(device) if isinstance(device, str) else device
    stream = (
        torch.cuda.current_stream(device).cuda_stream
        if device.type == "cuda" and torch.cuda.is_available()
        else 0
    )
    key = (heads, total_chunks, stream)
    with _WORKSPACES_CACHE_LOCK:
        hit = _WORKSPACES.get(device, key)
        if hit is not None:
            return hit
        made = allocate_prepare_workspace(heads, total_chunks, device)
        return _WORKSPACES.put(device, key, made, storages=(made.storage,))


def clear_prepare_workspaces(device: torch.device | int | None = None) -> None:
    with _WORKSPACES_CACHE_LOCK:
        _WORKSPACES.clear(device)


# ---------------------------------------------------------------------------
# Sequence metadata, cached by contents and device of ``cu_seqlens``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMetadata:
    """Canonical INT32 device metadata, plus the host copies used to validate.

    Whatever dtype the caller passes, both kernels index with INT32 on the
    device, so there is one canonical form and no second kernel
    specialization.  Prepare uses all three device tensors; the recurrence uses
    only ``cu_seqlens`` and ``cu_chunks``.
    """

    cu_seqlens: torch.Tensor  # INT32 [N + 1]
    cu_chunks: torch.Tensor  # INT32 [N + 1]
    chunk_to_seq: torch.Tensor  # INT32 [total_chunks]
    cu_seqlens_host: tuple[int, ...]
    cu_chunks_host: tuple[int, ...]
    total_chunks: int
    sequence_count: int

    def device_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.cu_seqlens, self.cu_chunks, self.chunk_to_seq)


#: 64 entries per device and 64 MiB of device payload,
#: whichever binds first.
METADATA_MAX_ENTRIES = 64
METADATA_MAX_BYTES = 64 * 1024 * 1024

_META_CONTENT = BoundedDeviceCache(
    "chunk-metadata",
    max_entries=METADATA_MAX_ENTRIES,
    max_bytes=METADATA_MAX_BYTES,
)
_META_IDENTITY = IdentityCache()


def _build_metadata(host: list[int], device: torch.device) -> ChunkMetadata:
    lengths = [b - a for a, b in zip(host, host[1:], strict=False)]
    per_sequence = chunks_for_lengths(lengths)
    cu = [0]
    for count in per_sequence:
        cu.append(cu[-1] + count)

    chunk_to_seq: list[int] = []
    for index, count in enumerate(per_sequence):
        chunk_to_seq.extend([index] * count)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return ChunkMetadata(
        cu_seqlens=i32(host),
        cu_chunks=i32(cu),
        chunk_to_seq=i32(chunk_to_seq),
        cu_seqlens_host=tuple(host),
        cu_chunks_host=tuple(cu),
        total_chunks=cu[-1],
        sequence_count=len(per_sequence),
    )


def clear_metadata_cache() -> None:
    _META_CONTENT.clear()
    _META_IDENTITY.clear()


# ==========================================================================


# --------------------------------------------------------------------------
# TMA descriptors
# --------------------------------------------------------------------------

#: 64 descriptor sets per device, LRU, for each cache.
DESCRIPTOR_MAX_ENTRIES = 64


# ---------------------------------------------------------------------------
# Shared: the encoder, the specification type, and the one geometry both
# kernels address.
# ---------------------------------------------------------------------------


def factor_slab_geometry(rows: int, heads: int, element_bytes: int = 2) -> dict:
    """The one description of the fused Kd/Qd/Ak slab, as ``(DK, rows, 3H)``.

    The three regions are the same geometry, the same dtype and exactly
    adjacent -- ``prepare_region_offsets`` aligns each to 256 bytes and every
    region size is a multiple of it, so no padding is inserted -- which makes
    them 3H planes of one tensor rather than three tensors.  Plane ``r*H + h``
    is region ``r`` (0=Kd, 1=Qd, 2=Ak) of head ``h``.

    prepare writes through this and the recurrence reads through it, and the
    fused entry passes prepare's encoded descriptor to both kernels, so the two
    sides cannot be allowed to disagree about it.  Defining it once is what
    makes that structural rather than a comment asking for care.
    """
    return dict(
        global_dims=(DK, rows, 3 * heads),
        global_stride_bytes=(DK * element_bytes, DK * rows * element_bytes),
        box_dims=(BF16_SEGMENT_ELEMS, BT, 1),
    )


# ---------------------------------------------------------------------------
# prepare: Q, K, G and the fused factor slab.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorMapSet:
    """One device buffer holding prepare's four descriptors, plus their addresses."""

    storage: torch.Tensor
    q: int
    k: int
    g: int
    #: Kd, Qd and Ak share one descriptor over 3H planes; see _factor_map.
    factor: int


def _activation_map(t: torch.Tensor, total_tokens: int, heads: int, segment_elems: int):
    """Key-major ``(DK, T_total, H)`` view of contiguous ``[1, T, H, DK]``."""
    esz = t.element_size()
    return TensorMapSpec(
        dtype=t.dtype,
        base=t.data_ptr(),
        global_dims=(DK, total_tokens, heads),
        global_stride_bytes=(heads * DK * esz, DK * esz),
        box_dims=(segment_elems, BT, 1),
    ).encode()


def _factor_map(ws_kd: torch.Tensor, rows: int, heads: int):
    """One descriptor over Kd, Qd and Ak as ``(DK, rows, 3H)``.

    Geometry comes from :func:`factor_slab_geometry`, which the recurrence's
    ``"factor"`` role uses too -- the combined entry hands *this* encoded
    descriptor to both kernels, so they cannot be allowed to disagree about it.
    Fusing the three regions also drops prepare's ``fence_tensormap_acquire``
    from three to one.
    """
    return TensorMapSpec(
        dtype=ws_kd.dtype,
        base=ws_kd.data_ptr(),
        **factor_slab_geometry(rows, heads, ws_kd.element_size()),
    ).encode()


#: Same event / LRU / lifetime contract as the recurrence cache: bounded, and a
#: cross-stream hit waits on the upload event instead of racing it.
_TENSOR_MAP_CACHE = BoundedDeviceCache(
    "prepare-descriptors", max_entries=DESCRIPTOR_MAX_ENTRIES
)


def clear_prepare_descriptor_cache() -> None:
    _TENSOR_MAP_CACHE.clear()


def build_tensor_maps(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_qd: torch.Tensor,
    ws_ak: torch.Tensor,
    total_tokens: int,
    total_chunks: int,
    heads: int,
) -> TensorMapSet:
    """Build (and cache) prepare's descriptors.

    Encoding the descriptors and copying them to the device costs far more than
    a launch, so the set is cached on the tensor addresses and shape; a caller
    that reuses its buffers hits after the first call.
    """
    rows = total_chunks * BT
    # A 128-byte S128 segment is 32 FP32 or 64 BF16, so G's box follows its
    # dtype; _encode already picks the TMA element type from it.
    g_segment_elems = (
        F32_SEGMENT_ELEMS if g.dtype is torch.float32 else BF16_SEGMENT_ELEMS
    )
    key = (
        q.data_ptr(),
        k.data_ptr(),
        g.data_ptr(),
        g.dtype,
        ws_kd.data_ptr(),
        ws_qd.data_ptr(),
        ws_ak.data_ptr(),
        total_tokens,
        rows,
        heads,
    )
    hit = _TENSOR_MAP_CACHE.get(q.device, key)
    if hit is not None:
        return hit

    blobs = [
        _activation_map(q, total_tokens, heads, BF16_SEGMENT_ELEMS),
        _activation_map(k, total_tokens, heads, BF16_SEGMENT_ELEMS),
        _activation_map(g, total_tokens, heads, g_segment_elems),
        _factor_map(ws_kd, rows, heads),
    ]
    packed = bytearray()
    for b in blobs:
        packed += b
    storage = upload_bytes(packed, q.device)
    if storage.data_ptr() % 64 != 0:
        raise RuntimeError("tensor map storage must be 64-byte aligned")

    base = storage.data_ptr()
    maps = TensorMapSet(
        storage=storage,
        q=base + 0 * DESCRIPTOR_BYTES,
        k=base + 1 * DESCRIPTOR_BYTES,
        g=base + 2 * DESCRIPTOR_BYTES,
        factor=base + 3 * DESCRIPTOR_BYTES,
    )
    return _TENSOR_MAP_CACHE.put(q.device, key, maps, (storage,))


# ---------------------------------------------------------------------------
# Recurrence: seven roles, at most seven descriptors.
#
# The specifications are plain data and need no CUDA context, so the geometry,
# the factor-plane coordinates and the exact-alias reuse count are all testable
# on the host; encoding and upload happen separately.
# ---------------------------------------------------------------------------


#: Descriptor roles, in the order they are packed into the upload buffer.
ROLES = ("factor", "aq", "gt", "v", "out", "state_in", "state_out")


def assert_factor_regions_are_fused(workspace, heads: int, total_chunks: int) -> None:
    """Refuse a workspace whose Kd/Qd/Ak are not one contiguous BF16 slab.

    This is required rather than a fallback: a temporary
    per-factor descriptor would silently change the descriptor count and the
    coordinates the kernel is compiled against.
    """
    stride = heads * total_chunks * BT * DK * 2
    base = workspace.kd.data_ptr()
    for index, name in enumerate(("qd", "ak"), start=1):
        got = getattr(workspace, name).data_ptr()
        want = base + index * stride
        if got != want:
            raise ValueError(
                f"prepare workspace region {name} is at {got}, expected "
                f"{want} ({index} x {stride} B after Kd); the recurrence's "
                "fused factor descriptor requires Kd/Qd/Ak to be adjacent"
            )


def recurrence_tensor_map_specs(
    *,
    kd_ptr: int,
    aq_ptr: int,
    gt_ptr: int,
    v_ptr: int,
    out_ptr: int,
    heads: int,
    total_tokens: int,
    total_chunks: int,
    sequences: int,
    state_in_ptr: int | None = None,
    state_out_ptr: int | None = None,
    state_dtype: torch.dtype | None = None,
) -> dict[str, TensorMapSpec]:
    """Build the descriptor specifications as plain data."""
    rows = BT * total_chunks
    specs: dict[str, TensorMapSpec] = {
        # Kd/Qd/Ak fused: 3H planes of R rows of 128 BF16 keys.  The geometry
        # is prepare's -- literally the same function -- because the fused
        # entry gives prepare's encoded descriptor to this kernel as well.
        "factor": TensorMapSpec(
            dtype=torch.bfloat16,
            base=kd_ptr,
            swizzle="128B",
            **factor_slab_geometry(rows, heads),
        ),
        # The 16x16 pairwise record prepare already wrote in its SW32 image:
        # moved verbatim, so no second swizzle is applied here.
        "aq": TensorMapSpec(
            dtype=torch.bfloat16,
            base=aq_ptr,
            global_dims=(BT * BT, total_chunks, heads),
            global_stride_bytes=(BT * BT * 2, total_chunks * BT * BT * 2),
            box_dims=(BT * BT, 1, 1),
            swizzle="NONE",
        ),
        "gt": TensorMapSpec(
            dtype=torch.float32,
            base=gt_ptr,
            global_dims=(DK, total_chunks, heads),
            global_stride_bytes=(DK * 4, total_chunks * DK * 4),
            box_dims=(DK, 1, 1),
            swizzle="NONE",
        ),
    }
    # V and out are the same geometry over [1, T, H, 128]; when they are also
    # the same storage they collapse to one descriptor by equality.
    activation = dict(
        dtype=torch.bfloat16,
        global_dims=(DV, total_tokens, heads),
        global_stride_bytes=(heads * DV * 2, DV * 2),
        box_dims=(DV_HALF, BT, 1),
        swizzle="128B",
    )
    specs["v"] = TensorMapSpec(base=v_ptr, **activation)
    specs["out"] = TensorMapSpec(base=out_ptr, **activation)

    if state_in_ptr is not None or state_out_ptr is not None:
        if state_dtype is None:
            raise ValueError("state_dtype is required when a state is present")
        state = _state_spec_fields(state_dtype, sequences * heads)
        if state_in_ptr is not None:
            specs["state_in"] = TensorMapSpec(base=state_in_ptr, **state)
        if state_out_ptr is not None:
            specs["state_out"] = TensorMapSpec(base=state_out_ptr, **state)

    for spec in specs.values():
        spec.validate()
    return specs


def _state_spec_fields(state_dtype: torch.dtype, planes: int) -> dict:
    """One full-tile instruction moves a whole state half.

    The unswizzled state address is ``v * 128 + k``, so a 128-byte S128 segment
    is 64 BF16 or 32 FP32 and the descriptor's inner mode *is* that segment.  The
    row mode then counts segments, which is why a DV half starts at row 128
    (BF16) or 256 (FP32) rather than at value 64.
    """
    if state_dtype is torch.bfloat16:
        inner, rows_per_value = BF16_SEGMENT_ELEMS, STATE_BF16_ROWS_PER_VALUE
        element_bytes = 2
    elif state_dtype is torch.float32:
        inner, rows_per_value = F32_SEGMENT_ELEMS, STATE_F32_ROWS_PER_VALUE
        element_bytes = 4
    else:
        raise ValueError(f"unsupported state dtype {state_dtype}")
    rows = rows_per_value * DV
    return dict(
        dtype=state_dtype,
        global_dims=(inner, rows, planes),
        global_stride_bytes=(inner * element_bytes, rows * inner * element_bytes),
        box_dims=(inner, rows // 2, 1),
        swizzle="128B",
    )


def unique_descriptors(specs: dict[str, TensorMapSpec]) -> list[TensorMapSpec]:
    """Distinct descriptors, in first-use order; equality *is* exact alias."""
    seen: list[TensorMapSpec] = []
    for role in ROLES:
        spec = specs.get(role)
        if spec is not None and spec not in seen:
            seen.append(spec)
    return seen


@dataclass(frozen=True)
class RecurrenceTensorMaps:
    """Device addresses of each role's descriptor, plus the backing storage."""

    storage: torch.Tensor
    addresses: dict[str, int]

    def address(self, role: str) -> int:
        """Descriptor address, or 0 for a role this launch does not use."""
        return self.addresses.get(role, 0)


def build_recurrence_tensor_maps(
    specs: dict[str, TensorMapSpec], device: torch.device
) -> RecurrenceTensorMaps:
    """Encode the distinct descriptors and upload them as one device buffer."""
    distinct = unique_descriptors(specs)
    packed = bytearray()
    for spec in distinct:
        packed += spec.encode()
    storage = upload_bytes(packed, device)
    if storage.data_ptr() % 64:
        raise RuntimeError("tensor map storage must be 64-byte aligned")

    base = storage.data_ptr()
    slot = {spec: base + i * DESCRIPTOR_BYTES for i, spec in enumerate(distinct)}
    return RecurrenceTensorMaps(
        storage=storage,
        addresses={role: slot[spec] for role, spec in specs.items()},
    )


#: 64 descriptor sets per device, LRU.  The key is every
#: field of every role's specification, so a buffer that moves, changes shape
#: or changes swizzle is a different entry rather than a stale hit.
_DESCRIPTORS = BoundedDeviceCache(
    "recurrence-descriptors", max_entries=DESCRIPTOR_MAX_ENTRIES
)


def get_recurrence_tensor_maps(
    specs: dict[str, TensorMapSpec], device: torch.device
) -> RecurrenceTensorMaps:
    """Cached :func:`build_recurrence_tensor_maps`.

    Encoding up to seven descriptors and uploading them costs far more than
    the launch itself, so a steady-state call must hit.  A hit on
    another stream waits on the upload event; it never synchronizes the host.
    """
    key = descriptor_cache_key(specs, device, ROLES)
    hit = _DESCRIPTORS.get(device, key)
    if hit is not None:
        return hit
    maps = build_recurrence_tensor_maps(specs, device)
    return _DESCRIPTORS.put(device, key, maps, (maps.storage,))


def clear_recurrence_descriptor_cache() -> None:
    _DESCRIPTORS.clear()


# --------------------------------------------------------------------------
# Recurrence host plan, arena and grid
# --------------------------------------------------------------------------

# --- Warp roles ------------------------------------------

LOAD_WARP = COMPUTE_WARPS  # 8
STORE_WARP = COMPUTE_WARPS + 1  # 9
REC_WARPS = COMPUTE_WARPS + 2  # 10
REC_THREADS = REC_WARPS * 32  # 320

#: ``min_blocks_per_mp`` launch bound of the recurrence kernel.
MIN_BLOCKS_PER_MP = 1

# --- Ring depths -----------------------------------------

INPUT_STAGES = 5
OUTPUT_STAGES = 2

# --- Input stage layout ----------------------------------

STAGE_KD = 0
STAGE_QD = 4096
STAGE_AK = 8192
STAGE_AQ = 12288
STAGE_GT = 12800
STAGE_V = 13312
INPUT_STAGE_BYTES = 15360

#: The stage's LSU share: Aq, GTotal and V arrive by ``cp.async`` from the
#: producer warp rather than by TMA.  On a full grid the per-SM TMA engine
#: paces the chunk while the LSU path sits idle beside it, so the 3 KiB that
#: fits a plain 16-byte copy goes around the engine.
INPUT_CP_ASYNC_BYTES = INPUT_STAGE_BYTES - STAGE_AQ  # Aq + GTotal + V
#: One ``arrive_and_expect_tx`` covers the six factor boxes; the ``cp.async``
#: share completes through the second IN_READY arrival instead.
INPUT_STAGE_TX_BYTES = INPUT_STAGE_BYTES - INPUT_CP_ASYNC_BYTES
#: How many chunks behind its issue the producer confirms a stage's
#: ``cp.async`` share (``cp.async.wait_group`` keep count).  Two keeps the
#: copy latency out of the producer's issue loop; the ring is five deep, so
#: the lead exists to spend.
CP_ASYNC_ARRIVE_LAG = 2
#: The two-axis predicate for the cp.async split, decided per CTA.  The TMA
#: queue is collective, so the split pays only once the grid is big enough to
#: queue it and loses on small grids; and its fixed costs (prologue, lagged
#: arrivals, smem-write contention) need enough chunks per sequence to
#: amortise.  Below either threshold the producer keeps the plain nine-box
#: TMA path.  Both values are fits to measurements on the supported target;
#: re-measure with ``benchmarks/flashinfer_benchmark.py --routine
#: recurrent_kda_prefill --backends flashinfer flashinfer-decomp
#: flashinfer-fused`` when the producer changes.
CP_ASYNC_MIN_CTAS = 24
CP_ASYNC_MIN_CHUNKS = 128

OUTPUT_STAGE_BYTES = BT * DV_HALF * 2  # 2048

# --- Arena -----------------------------------------------

SMEM_STATE = 0
STATE_BYTES = DV_HALF * DK * 2  # 16384
SMEM_INPUT = SMEM_STATE + STATE_BYTES  # 16384
SMEM_OUTPUT = SMEM_INPUT + INPUT_STAGES * INPUT_STAGE_BYTES  # 93184
REC_SMEM_BARRIERS = SMEM_OUTPUT + OUTPUT_STAGES * OUTPUT_STAGE_BYTES  # 97280

MBAR_INPUT_READY = 0
MBAR_INPUT_CONSUMED = MBAR_INPUT_READY + INPUT_STAGES * 8  # 40
MBAR_OUTPUT_READY = MBAR_INPUT_CONSUMED + INPUT_STAGES * 8  # 80
MBAR_OUTPUT_CONSUMED = MBAR_OUTPUT_READY + OUTPUT_STAGES * 8  # 96
MBAR_STATE_READY = MBAR_OUTPUT_CONSUMED + OUTPUT_STAGES * 8  # 112
BARRIER_BYTES = MBAR_STATE_READY + 8  # 120

SMEM_RAW_END = REC_SMEM_BARRIERS + BARRIER_BYTES  # 97400

#: The launch size is 97,536 B: 97,400 rounded up to **256**, not to 128
#: (which would give 97,408).  The barrier arena therefore owns a full
#: 256-byte block, leaving 3,840 B against the 101,376 B ceiling, and 97,536
#: is the number the resource gate checks.
SMEM_ALIGNMENT = 256
SMEM_DYNAMIC_BYTES = (
    (SMEM_RAW_END + SMEM_ALIGNMENT - 1) // SMEM_ALIGNMENT * SMEM_ALIGNMENT
)  # 97536

#: The FP32 external-state conversion buffer borrows the head of the pipeline
#: union.  It is live only before the input ring starts and after it drains, so
#: it does not raise the peak.
SMEM_STATE_F32 = SMEM_INPUT


# --- Arrival counts --------------------------------------

#: Mode-dependent: the plain-TMA producer makes one arrival (the expect-tx),
#: the cp.async producer two (plus the elected arrive after wait_group).
#: The kernel initialises each IN_READY as ``INPUT_READY_ARRIVALS_TMA + use_cp``.
INPUT_READY_ARRIVALS_TMA = 1
INPUT_CONSUMED_ARRIVALS = COMPUTE_WARPS  # one per compute warp, not per thread
OUTPUT_READY_ARRIVALS = COMPUTE_WARPS
OUTPUT_CONSUMED_ARRIVALS = 1
STATE_READY_ARRIVALS = 1


# ---------------------------------------------------------------------------
# Ring phase equations.
#
# Written with ``//``, ``*``, ``-`` and ``^`` only so the same bodies evaluate
# for Python ``int`` on the host and ``cutlass.Int32`` on the device.
# ---------------------------------------------------------------------------


def input_stage(chunk):
    return chunk - (chunk // INPUT_STAGES) * INPUT_STAGES


def input_generation(chunk):
    return chunk // INPUT_STAGES


def input_ready_parity(chunk):
    """Compute warps: pass once the slot's TMA has landed ``ig + 1`` times."""
    return input_generation(chunk) & 1


def input_consumed_parity(chunk):
    """Load warp: pass once all 8 compute warps have released the slot ``ig``
    times.  Generation 0 passes against the initial phase with no seeding."""
    return 1 ^ (input_generation(chunk) & 1)


def output_stage(chunk):
    return chunk - (chunk // OUTPUT_STAGES) * OUTPUT_STAGES


def output_generation(chunk):
    return chunk // OUTPUT_STAGES


def output_ready_parity(chunk):
    """Store warp: pass once all 8 compute warps have filled the slot."""
    return output_generation(chunk) & 1


def output_consumed_parity(chunk):
    """Compute warps: pass once the store warp has finished reading the slot."""
    return 1 ^ (output_generation(chunk) & 1)


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecurrencePlan:
    """Everything the recurrence launch needs, with nothing launched yet."""

    tensor_maps: object
    sequences: int
    has_state_in: bool
    has_state_out: bool
    state_dtype: torch.dtype | None


def recurrence_grid(sequences: int, heads: int) -> tuple[int, int, int]:
    """``(2N, H, 1)``: one CTA per ``(sequence, head, DV half)``, DV fastest.

    The DV half must be the fastest-varying block coordinate.  The two halves
    of one head read the *same* Kd/Qd/Ak/Aq/GTotal -- only V and the output
    differ -- and L2 serves the second reader almost entirely, so the factors
    cost one DRAM read per head rather than two.  That reuse needs the two
    halves *co-resident*, and shared memory pins the recurrence at one CTA per
    SM, so co-resident means "in the same wave", which means "adjacent".
    An order that separates the halves by ``N`` blocks (``(N, 2H, 1)``) puts
    every pair across a wave boundary once ``N >= SMs`` and reads every factor
    from DRAM twice; the loss shows up only once the grid exceeds one wave,
    which is what pins it on the wave boundary rather than the tensor shape.

    ``(2N, H, 1)`` rather than ``(2, NH, 1)`` because ``maxGridSize[1]`` is
    65,535 and ``N * H`` overflows it well inside the supported range, while
    ``2N`` sits in ``maxGridSize[0] = 2^31 - 1`` and ``H`` is small.  Rather
    than ``(2H, N, 1)`` -- which also pairs the halves -- because a wave should
    cover one head's sequences, not one sequence's heads, which measured
    slower.  That keeps the traversal order ``(N, 2H, 1)`` had; only the DV
    half moves.
    """
    return (2 * sequences, heads, 1)


def head_of(block_y: int) -> int:
    return block_y


# ---------------------------------------------------------------------------
# Host validation and launch
# ---------------------------------------------------------------------------


def checked_i32(name: str, value) -> int:
    """Reject anything that would not survive the device's INT32 indexing.

    Every derived extent is checked on the host
    before any workspace, descriptor or launch exists, rather than relying on a
    device cast to truncate silently.
    """
    number = int(value)
    if number < 0 or number > INT32_MAX:
        raise ValueError(f"{name} must fit in a non-negative INT32, got {number}")
    return number


def check_recurrence_ranges(
    *,
    total_tokens: int,
    total_chunks: int,
    sequences: int,
    heads: int,
    cu_seqlens_host: list[int],
    cu_chunks_host: list[int],
    device: torch.device,
) -> None:
    """Check every index the recurrence derives, before anything is allocated."""
    checked_i32("T_total", total_tokens)
    checked_i32("total_chunks", total_chunks)
    checked_i32("R = 16 * total_chunks", BT * total_chunks)
    checked_i32("N", sequences)
    checked_i32("H", heads)
    checked_i32("2 * H", 2 * heads)
    checked_i32("3 * H", 3 * heads)
    checked_i32("N * H", sequences * heads)
    # grid[0] = 2N is a block coordinate the kernel halves back into a
    # sequence, so it is a derived extent like any other.
    checked_i32("2 * N", 2 * sequences)
    for index, value in enumerate(cu_seqlens_host):
        checked_i32(f"cu_seqlens[{index}]", value)
    for index, value in enumerate(cu_chunks_host):
        checked_i32(f"cu_chunks[{index}]", value)
    if total_tokens:
        checked_i32("max token_base + 15", total_tokens - 1 + BT)
    if total_chunks:
        checked_i32("max 16 * gchunk + 15", BT * (total_chunks - 1) + BT - 1)
        checked_i32("max factor plane", 3 * heads - 1)
        # The flat output is bounded by its element count rather than by its
        # largest index: the DSL packs a memref extent as INT32, so it stops one
        # element before the index does.  Shared with ``fused``, which reaches
        # the same limit through the same store.
        check_flat_output_range(total_tokens, heads)

    grid = recurrence_grid(sequences, heads)
    limits = max_grid_dims(device)
    for axis, (extent, limit) in enumerate(zip(grid[:2], limits, strict=False)):
        if extent > limit:
            raise ValueError(
                f"grid[{axis}] = {extent} exceeds maxGridSize[{axis}] = {limit}"
            )


def validate_state(
    state: torch.Tensor | None,
    name: str,
    *,
    sequences: int,
    heads: int,
    device: torch.device,
) -> None:
    if state is None:
        return
    shape = (sequences, heads, DV, DK)
    if tuple(state.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(state.shape)}")
    if state.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be bfloat16 or float32, got {state.dtype}")
    if not state.is_cuda or state.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if not state.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def plan_recurrence(
    *,
    workspace,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_i32: torch.Tensor,
    cu_chunks_i32: torch.Tensor,
    cu_seqlens_host: list[int],
    cu_chunks_host: list[int],
    heads: int,
    total_tokens: int,
    total_chunks: int,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> RecurrencePlan:
    """Plan the recurrence over ``out`` and the optional final state.

    Every host-side step -- validation, descriptor encoding, stream recording
    -- without a launch: :func:`_build_plan` hands the result and prepare's
    plan to one compiled entry, so the Python/compiled boundary is crossed once
    per forward rather than twice.

    Private: :func:`run` is the public entry point.  The caller is responsible
    for the public ABI checks; what is enforced here is what the *kernel*
    depends on -- the fused factor slab, the state geometry and the INT32
    index ranges.
    """

    device = v.device
    require_sm120a(device)
    sequences = len(cu_seqlens_host) - 1

    if total_chunks == 0:
        # The caller must take the host state-only fast path
        # before any descriptor exists.  A zero-extent factor descriptor is not
        # encodable, so failing here is the honest outcome rather than a
        # silently degenerate launch.
        raise ValueError(
            "plan_recurrence requires total_chunks == 0 to be handled by the "
            "host state-only fast path"
        )

    check_recurrence_ranges(
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        sequences=sequences,
        heads=heads,
        cu_seqlens_host=cu_seqlens_host,
        cu_chunks_host=cu_chunks_host,
        device=device,
    )
    validate_state(
        initial_state, "initial_state", sequences=sequences, heads=heads, device=device
    )
    validate_state(
        final_state, "final_state", sequences=sequences, heads=heads, device=device
    )
    if initial_state is not None and final_state is not None:
        if initial_state.dtype != final_state.dtype:
            raise TypeError(
                "initial_state and final_state must share a dtype, got "
                f"{initial_state.dtype} and {final_state.dtype}"
            )
    assert_factor_regions_are_fused(workspace, heads, total_chunks)

    state_dtype = None
    if initial_state is not None:
        state_dtype = initial_state.dtype
    elif final_state is not None:
        state_dtype = final_state.dtype

    specs = recurrence_tensor_map_specs(
        kd_ptr=workspace.kd.data_ptr(),
        aq_ptr=workspace.aq.data_ptr(),
        gt_ptr=workspace.g_total.data_ptr(),
        v_ptr=v.data_ptr(),
        out_ptr=out.data_ptr(),
        heads=heads,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        sequences=sequences,
        state_in_ptr=None if initial_state is None else initial_state.data_ptr(),
        state_out_ptr=None if final_state is None else final_state.data_ptr(),
        state_dtype=state_dtype,
    )
    tensor_maps = get_recurrence_tensor_maps(specs, device)

    # Every buffer this raw launch touches is recorded on the
    # current stream, exact aliases once, so the allocator cannot hand a block
    # back while the kernel is still reading it.
    record_stream_once(
        (
            v,
            out,
            workspace.storage,
            cu_seqlens_i32,
            cu_chunks_i32,
            initial_state,
            final_state,
            tensor_maps.storage,
        ),
        torch.cuda.current_stream(device),
    )

    return RecurrencePlan(
        tensor_maps=tensor_maps,
        sequences=sequences,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        state_dtype=state_dtype,
    )


# --------------------------------------------------------------------------
# Prepare host plan, arena and residency
# --------------------------------------------------------------------------


#: Chunks per prepare CTA on the supported target.  Depth 2 is where the
#: chunk prefetch first pays (the kernel already runs near peak DRAM
#: bandwidth, so halving the CTA count is what the second chunk buys), and
#: past 2 the extra CTA-wide barriers cost more than the overlap wins.  A fit
#: to measurements; re-measure with ``benchmarks/flashinfer_benchmark.py
#: --routine recurrent_kda_prefill --backends flashinfer flashinfer-decomp
#: flashinfer-fused`` when the prepare kernel changes.
DEFAULT_PREP_CHUNKS_PER_CTA = 2

#: The optimum moves with the memory system: on a part with DRAM bandwidth to
#: spare relative to this kernel, the prefetch keeps paying past depth 2, and
#: applying the sm_120 value there costs a large factor.  Only ``(12, 0)`` is
#: a supported target (``require_sm120a`` refuses the rest); the other entries
#: exist so an experiment on those parts does not need an explicit
#: ``chunks_per_cta``.
DEFAULT_PREP_CHUNKS_PER_CTA_BY_CAPABILITY = {
    (12, 0): 2,
    (10, 0): 4,
    (9, 0): 4,
}

SUPPORTED_PREP_CHUNKS_PER_CTA = (1, 2, 3, 4)
SUPPORTED_MIN_BLOCKS_PER_SM = (1, 2, 3)


@dataclass(frozen=True)
class PrepareConfig:
    """Compile-time configuration; part of the compile cache key."""

    safe_gate: bool
    chunks_per_cta: int = DEFAULT_PREP_CHUNKS_PER_CTA
    min_blocks_per_sm: int = 1

    def __post_init__(self) -> None:
        if self.chunks_per_cta not in SUPPORTED_PREP_CHUNKS_PER_CTA:
            raise ValueError(
                f"PREP_CHUNKS_PER_CTA must be one of "
                f"{SUPPORTED_PREP_CHUNKS_PER_CTA}, got {self.chunks_per_cta}"
            )
        if self.min_blocks_per_sm not in SUPPORTED_MIN_BLOCKS_PER_SM:
            raise ValueError(
                f"MIN_BLOCKS_PER_SM must be one of "
                f"{SUPPORTED_MIN_BLOCKS_PER_SM}, got {self.min_blocks_per_sm}"
            )

    def cache_key(self, capability: tuple[int, int]) -> tuple:
        return (
            capability,
            self.safe_gate,
            torch.bfloat16,  # INV_MMA_DTYPE
            self.chunks_per_cta,
            self.min_blocks_per_sm,
        )


def default_chunks_per_cta_for(capability: tuple[int, int]) -> int:
    """Default chunks per CTA for ``capability``; the sm_120 value if unknown."""
    return DEFAULT_PREP_CHUNKS_PER_CTA_BY_CAPABILITY.get(
        (capability[0], capability[1]), DEFAULT_PREP_CHUNKS_PER_CTA
    )


def default_chunks_per_cta(device=None) -> int:
    """Default chunks per CTA for ``device``, falling back when there is no GPU."""
    if not torch.cuda.is_available():
        return DEFAULT_PREP_CHUNKS_PER_CTA
    return default_chunks_per_cta_for(torch.cuda.get_device_capability(device))


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


#: Destination buffer for ``exp2(A_log * log2e)``, one per ``(A_log, stream)``.
#:
#: This caches the *allocation*, not the computation.  Caching the result and
#: skipping the arithmetic was tried and is wrong under ``torch.cuda.graph``: a
#: cache hit during capture means the ``exp2`` is never recorded, so the graph
#: reads a frozen buffer and a later in-place update to ``A_log`` -- an
#: optimizer step -- replays with stale values and no error.  Writing into a
#: reused buffer keeps the kernel in the stream, so capture records it and
#: replay stays correct, while still avoiding the per-call allocation.
_A_LOG_EXP: dict[tuple[int, int], tuple[weakref.ref, torch.Tensor, int]] = {}


def _purge_for(key):
    def _purge(_ref, _key=key):
        _A_LOG_EXP.pop(_key, None)

    return _purge


def _a_log_exp_key(a_log: torch.Tensor) -> tuple[int, int]:
    """Tensor identity plus the stream that will consume the writable buffer."""
    stream = (
        torch.cuda.current_stream(a_log.device).cuda_stream
        if a_log.is_cuda and torch.cuda.is_available()
        else 0
    )
    return id(a_log), stream


def a_log_exp_for(
    a_log: torch.Tensor,
    log2e: float,
    *,
    out: torch.Tensor | None = None,
    key: tuple[int, int] | None = None,
) -> torch.Tensor:
    """``exp2(a_log * log2e)`` in FP32, written into a reused buffer.

    Keyed on a weak reference to ``a_log`` and on the current CUDA stream, so a
    freed parameter drops its buffers and two streams never write the same
    scratch concurrently.  ``out`` lets a cached launch plan refresh the exact
    buffer whose pointer is baked into its argument tuple, even if the cache
    was cleared or another plan replaced its entry.

    ``key`` lets a caller that already knows its stream skip recomputing it.
    :func:`_a_log_exp_key` asks the driver on every call, and neither half is
    cheap: ``torch.cuda.current_stream()`` builds a Stream object through
    several Python frames, and ``torch.cuda.is_available()`` reaches
    ``os.getenv`` by way of nvml.  That host cost is exposed on shapes whose
    kernels are too short to hide it.  A cached launch plan is pinned to one
    stream and one ``A_log`` by construction, so for that caller the key is a
    constant and belongs in the plan.
    """
    if key is None:
        key = _a_log_exp_key(a_log)
    cached = _A_LOG_EXP.get(key)
    if cached is not None:
        ref, buffer, version = cached
        if ref() is a_log:
            if out is None:
                out = buffer
            # `exp2(a_log * log2e)` is two dispatches and two kernels, mostly
            # host overhead.  It only has to run when a_log has actually
            # changed, and an optimizer step bumps the version counter.  A
            # tensor with no counter -- anything created under inference_mode
            # -- is rejected explicitly, so it recomputes rather than trusting
            # an identity it cannot check.  The sentinel compares equal to
            # itself, so testing it by value would not do that.
            #
            # Never skip inside a capture: the graph replays only what it
            # recorded, so leaving exp2 out of it means a later parameter update
            # is silently ignored on replay.
            if (
                out is buffer
                and version is not NO_VERSION
                and version == tensor_version(a_log)
                and not capturing()
            ):
                return out
        else:
            _A_LOG_EXP.pop(key, None)

    if out is None:
        out = torch.empty_like(a_log, dtype=torch.float32)

    torch.exp2(a_log.float() * log2e, out=out)
    _A_LOG_EXP[key] = (
        weakref.ref(a_log, _purge_for(key)),
        out,
        tensor_version(a_log),
    )
    return out


def clear_launch_caches() -> None:
    """Drop the ``A_log`` cache; for tests that assert on recomputation."""
    _A_LOG_EXP.clear()


@dataclass(frozen=True)
class PreparePlan:
    """Prepare's host-side work, done but not launched."""

    tensor_maps: object
    a_log_exp: torch.Tensor
    grid_x: int


def prepare_launch_plan(
    *,
    q,
    k,
    g,
    A_log,
    workspace,
    total_tokens: int,
    total_chunks: int,
    heads: int,
    config: PrepareConfig,
) -> PreparePlan:
    """Encode prepare's descriptors and derive its grid, without launching.

    :func:`_build_plan` needs these before it can hand both kernels to one
    compiled entry.  Everything here is cached on the buffer addresses, so a
    steady-state call re-encodes nothing.
    """

    return PreparePlan(
        tensor_maps=build_tensor_maps(
            q=q,
            k=k,
            g=g,
            ws_kd=workspace.kd,
            ws_qd=workspace.qd,
            ws_ak=workspace.ak,
            total_tokens=total_tokens,
            total_chunks=total_chunks,
            heads=heads,
        ),
        a_log_exp=a_log_exp_for(A_log, LOG2_E),
        grid_x=(total_chunks + config.chunks_per_cta - 1) // config.chunks_per_cta,
    )


# --------------------------------------------------------------------------
# The prepare device kernel
# --------------------------------------------------------------------------

#: The inverse is two 8x8 block inverses plus one coupling term; this is the
#: block split, not a tunable.
HALF_BT = BT // 2
PREPARE_DEVICE_THREADS = 128
PREPARE_DEVICE_WARPS = 4

BF16_SEG_ELEMS = BF16_SEGMENT_ELEMS  # 64
BF16_SEG_STRIDE = BF16_SEGMENT_STRIDE  # 1024
F32_SEG_ELEMS = F32_SEGMENT_ELEMS  # 32
F32_SEG_STRIDE = F32_SEGMENT_STRIDE  # 512

#: Q 4096 + K 4096 + G, and G is the only term that moves with its dtype.
TMA_TX_BYTES_G_FP32 = 16384
TMA_TX_BYTES_G_BF16 = 12288

# The barrier arena, as 8-byte slot indices.
MBAR_SLOT_TMA0 = 0
MBAR_SLOT_TMA1 = 1
MBAR_SLOT_K_HALF_READY = 2
MBAR_SLOT_K_FULL_READY = 3
MBAR_SLOT_RAW_RELEASED = 4
MBAR_SLOT_PAIRWISE_READY = 5


# ---------------------------------------------------------------------------
# Device index helpers (mirror the canonical index mappings above)
# ---------------------------------------------------------------------------


@cute.jit
def kr_ak_idx(token, dim):
    return raw_bf16_s128(token ^ 8, dim)


@cute.jit
def prepare_pair_idx(row, col):
    storage_col = col ^ 8
    byte_offset = 2 * (row * 16 + storage_col)
    return (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2


@cute.jit
def warp_row_sum_8(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce the eight lanes that cooperate on one token row."""
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=4))
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=2))
    return value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=1))


# ---------------------------------------------------------------------------
# Fragment helpers
# ---------------------------------------------------------------------------


@cute.jit
def load_a_fragment(smem, token_base, dim_base, lane):
    """A-operand ``ldmatrix.x4``."""
    matrix_id = lane // 8
    row = token_base + (lane % 8) + 8 * (matrix_id % 2)
    col = dim_base + 8 * (matrix_id // 2)
    return ldmatrix_x4(smem + raw_bf16_s128(row, col))


@cute.jit
def load_b_fragment(smem, dim_base, lane):
    """B-operand ``ldmatrix.x4``.

    The row half is keyed on bit 4 of the lane, not bit 3.
    """
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (lane // 16)
    col = dim_base + 8 * (matrix_id % 2)
    return ldmatrix_x4(smem + raw_bf16_s128(row, col))


@cute.jit
def load_pairwise_a_fragment(smem, lane):
    """Load a 16x16 pairwise tile in A layout."""
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (matrix_id % 2)
    col = 8 * (matrix_id // 2)
    return ldmatrix_x4(smem + prepare_pair_idx(row, col))


@cute.jit
def a_to_a_transposed(a0, a1, a2, a3):
    """A layout -> A layout of the transpose; registers 1 and 2 swap."""
    return (
        movmatrix_b16(a0),
        movmatrix_b16(a2),
        movmatrix_b16(a1),
        movmatrix_b16(a3),
    )


@cute.jit
def acc_to_a_fragment(c):
    """Register-local accumulator -> A-layout pack."""
    return (
        pack_bf16x2(c[0], c[1]),
        pack_bf16x2(c[2], c[3]),
        pack_bf16x2(c[4], c[5]),
        pack_bf16x2(c[6], c[7]),
    )


@cute.jit
def acc_to_a_fragment_f16(c):
    """As :func:`acc_to_a_fragment`, packing FP16 instead of BF16.

    ``movmatrix.b16`` in :func:`a_to_b` is dtype-blind, so the A->B step is
    shared with the BF16 path unchanged.
    """
    return (
        pack_f16x2(c[0], c[1]),
        pack_f16x2(c[2], c[3]),
        pack_f16x2(c[4], c[5]),
        pack_f16x2(c[6], c[7]),
    )


@cute.jit
def ZERO8():
    """Eight zeroed FP32 accumulator slots."""
    z = cutlass.Float32(0.0)
    return (z, z, z, z, z, z, z, z)


@cute.jit
def kk_half(lhs_ptr, ki_ptr, lane, half, c):
    """``lhs @ Ki.T`` over the four K=16 phases of head-dimension half ``half``.

    Split so warp 0 can start K blocks 0-3 on ``k_half_ready`` and only wait
    for ``k_full_ready`` before blocks 4-7.
    """
    for j in cutlass.range_constexpr(4):
        d0 = (half * 4 + j) * 16
        a0, a1, a2, a3 = load_a_fragment(lhs_ptr, 0, d0, lane)
        b0, b1, b2, b3 = load_b_fragment(ki_ptr, d0, lane)
        c = mma_16x16((a0, a1, a2, a3), (b0, b1, b2, b3), c)
    return c


@cute.jit
def kk_over_dk(lhs_ptr, ki_ptr, lane):
    """``lhs @ Ki.T`` over all eight K=16 phases of the head dimension."""
    return kk_half(lhs_ptr, ki_ptr, lane, 1, kk_half(lhs_ptr, ki_ptr, lane, 0, ZERO8()))


@cute.jit
def prepare_stmatrix_coord(lane):
    """Row/col of the 16-byte row segment lane ``lane`` feeds to stmatrix.x4.

    Same quadrant order as the A fragment.
    """
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    col = 8 * (matrix_id // 2)
    return row, col


@cute.jit
def acc_coord(lane, slot):
    """``(row, col)`` of accumulator slot ``slot`` in ``[0, 8)``."""
    n_block = slot // 4
    reg = slot - n_block * 4
    row = (lane // 4) + 8 * (reg // 2)
    col = 8 * n_block + 2 * (lane % 4) + (reg % 2)
    return row, col


@cute.jit
def issue_chunk_tma(
    desc_q,
    desc_k,
    desc_g,
    p_q,
    p_k,
    p_g,
    mbar,
    token_base,
    head,
    G_FP32: cutlass.Constexpr,
):
    """Issue one chunk's Q/K/G as TMA boxes.

    Two Q, two K, and G in four boxes at FP32 or two at BF16, all against
    ``mbar``: 4096 + 4096 + 8192 or 4096 bytes.  A single elected lane issues
    the lot, which is what lets the prefetch be confined to one warp.

    Coordinates are ``(segment * segment_elems, token_base, head)`` against the
    key-major descriptors.
    """
    for seg in cutlass.range_constexpr(BF16_SEGMENTS):
        c0 = seg * BF16_SEG_ELEMS
        tma_load_3d(p_q + seg * BF16_SEG_STRIDE, desc_q, mbar, c0, token_base, head)
        tma_load_3d(p_k + seg * BF16_SEG_STRIDE, desc_k, mbar, c0, token_base, head)
    if cutlass.const_expr(G_FP32):
        for seg in cutlass.range_constexpr(F32_SEGMENTS):
            c0 = seg * F32_SEG_ELEMS
            tma_load_3d(p_g + seg * F32_SEG_STRIDE, desc_g, mbar, c0, token_base, head)
    else:
        for seg in cutlass.range_constexpr(BF16_SEGMENTS):
            c0 = seg * BF16_SEG_ELEMS
            tma_load_3d(p_g + seg * BF16_SEG_STRIDE, desc_g, mbar, c0, token_base, head)


@cute.jit
def load_beta_stage(gbeta, smem_beta, stage, token_base, valid_rows, head, lane, heads):
    """Activate one chunk's 16 beta logits into beta stage ``stage``.

    ``smem_beta`` is double-buffered so that this strided column read out of
    ``[T, H]`` -- which cannot coalesce, one sector per token -- is issued a
    full chunk before the values are needed, instead of stalling the whole CTA
    at a barrier behind its DRAM latency.
    """
    if lane < BT:
        bv = cutlass.Float32(0.0)
        if lane < valid_rows:
            logit = cutlass.Float32(gbeta[(token_base + lane) * heads + head])
            half = cutlass.Float32(0.5)
            # Stored as FP32, unrounded.  The activated value has exactly two
            # consumers: the strict-lower scale, which is an FP32 multiply, and
            # the AINV column scale, which packs to BF16 itself.  Rounding here
            # would be invisible to the second and only lossy to the first, at
            # the cost of two extra conversions per lane.
            bv = (
                cutlass.Float32(cute.math.tanh(logit * half, fastmath=True)) * half
                + half
            )
        smem_beta[stage * BT + lane] = bv


@cute.jit
def vec4_f32(ptr, idx):
    """Load 4 contiguous FP32 (16 bytes) into a register fragment."""
    frag = make_rmem_tensor(4, cutlass.Float32)
    cute.autovec_copy(vec_at(ptr, idx, 4), frag)
    return frag


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def prepare_kernel(
    gq: cute.Tensor,
    gk: cute.Tensor,
    gg: cute.Tensor,
    gbeta: cute.Tensor,
    ga_log_exp: cute.Tensor,
    gdt: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    gchunk_to_seq: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_ak: cute.Tensor,
    ws_aq: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_flag: cute.Tensor,
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_factor: cutlass.Int64,
    SCALE: cutlass.Float32,
    GATE_SCALE_LOG2: cutlass.Float32,
    TOTAL_CHUNKS: cutlass.Int32,
    heads: cutlass.Int32,
    gen: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
    PIPE: cutlass.Constexpr,
    DEFER: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    # Overlap mode swaps the grid axes so issue order is breadth-first across
    # heads (all heads' chunk j before chunk j+1): the recurrence CTAs all
    # chase one narrow write frontier instead of head H-1 idling to the end.
    if cutlass.const_expr(PIPE):
        head = bidx
    else:
        head = bidy

    alloc = cutlass.utils.SmemAllocator()
    p_kd = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    # Deliberately not aliased onto the raw stages.  At CPC == 1 the raw Q and
    # K stages are dead after the Kd/Ki/Qd loop, so Qd could overwrite Q and
    # Ki could overwrite K without added synchronization ((my_token, d0) ->
    # thread is a bijection), freeing enough SMEM for four resident CTAs.  It
    # measured slower: the kernel already sits near peak DRAM bandwidth, so
    # extra warps have nothing to issue and the added CTAs only lengthen every
    # CTA-wide rendezvous.
    p_qd = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_q = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_k = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_g = alloc.allocate_array(cutlass.Float32, BT * DK)
    p_ki = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_ainv = alloc.allocate_array(cutlass.BFloat16, BT * BT)
    p_gamma = alloc.allocate_array(cutlass.BFloat16, DK)
    # ``smem_beta`` is 128 bytes: two 16-float stages, so a chunk's beta is
    # loaded and activated one chunk ahead of its use.
    p_beta = alloc.allocate_array(cutlass.Float32, 2 * BT)
    p_bar = alloc.allocate_array(cutlass.Int64, 6)
    p_qk = alloc.allocate_array(cutlass.BFloat16, BT * BT)

    # Pointers feed ldmatrix; the flat tensor views give dynamic scalar access.
    smem_g = cute.make_tensor(p_g, cute.make_layout(BT * DK))
    # A BF16 view of the same 8192-byte stage.  With BF16 ``G`` the raw input
    # is the 4096-byte Q/K image living in its first half; the stage is then
    # overwritten in full by FP32 ``exp_g``, which is why the arena cannot
    # simply shrink and why the two must not overlap in time (see the extra
    # barrier before the writeback below).  Unused when ``G`` is FP32.
    p_g_bf16 = cute.recast_ptr(p_g, dtype=cutlass.BFloat16)
    smem_g_bf16 = cute.make_tensor(p_g_bf16, cute.make_layout(BT * DK))
    # Trace-time selection: which view the raw G load and tail-clear address.
    p_g_raw = p_g if cutlass.const_expr(G_FP32) else p_g_bf16
    G_TX = TMA_TX_BYTES_G_FP32 if cutlass.const_expr(G_FP32) else TMA_TX_BYTES_G_BF16
    smem_ainv = cute.make_tensor(p_ainv, cute.make_layout(BT * BT))
    smem_gamma = cute.make_tensor(p_gamma, cute.make_layout(DK))
    smem_beta = cute.make_tensor(p_beta, cute.make_layout(2 * BT))
    smem_qk = cute.make_tensor(p_qk, cute.make_layout(BT * BT))

    # The barrier arena.  The two input barriers alternate by
    # chunk parity so that reusing one two chunks later toggles its phase.
    mbar_tma0 = p_bar + MBAR_SLOT_TMA0
    mbar_tma1 = p_bar + MBAR_SLOT_TMA1
    mbar_k_half = p_bar + MBAR_SLOT_K_HALF_READY
    mbar_k_full = p_bar + MBAR_SLOT_K_FULL_READY
    mbar_raw_released = p_bar + MBAR_SLOT_RAW_RELEASED
    mbar_pairwise = p_bar + MBAR_SLOT_PAIRWISE_READY
    if warp_id == 0:
        if lane == 0:
            cute.arch.mbarrier_init(mbar_tma0, 1)
            cute.arch.mbarrier_init(mbar_tma1, 1)
            cute.arch.mbarrier_init(mbar_k_half, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_k_full, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_raw_released, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_pairwise, 1)
            cute.arch.mbarrier_init_fence()
            fence_tensormap_acquire(desc_q)
            fence_tensormap_acquire(desc_k)
            fence_tensormap_acquire(desc_g)
            # One acquire, not three: Kd/Qd/Ak are 3H planes of one map.
            fence_tensormap_acquire(desc_factor)
    # Nothing may arrive or wait before the
    # initialized state is published.
    cute.arch.barrier()

    if cutlass.const_expr(PIPE):
        cta_chunk_base = bidy * CPC
    else:
        cta_chunk_base = bidx * CPC
    my_chunks = TOTAL_CHUNKS - cta_chunk_base
    if my_chunks > CPC:
        my_chunks = cutlass.Int32(CPC)

    dt_value = cutlass.Float32(gdt[head * DK + tidx])
    alog = cutlass.Float32(ga_log_exp[head])
    row_in_warp = lane // 8
    lane_in_row = lane % 8
    my_token = warp_id * 4 + row_in_warp
    q4 = lane % 4

    # Prologue: start the first chunk's copies before entering the loop.
    seq0 = cutlass.Int32(gchunk_to_seq[cta_chunk_base])
    lc0 = cta_chunk_base - cutlass.Int32(gcu_chunks[seq0])
    tb0 = cutlass.Int32(gcu_seqlens[seq0]) + lc0 * BT
    vr0 = cutlass.Int32(gcu_seqlens[seq0 + 1]) - tb0
    if vr0 > BT:
        vr0 = cutlass.Int32(BT)
    if warp_id == 0:
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_tma0, G_TX)
            issue_chunk_tma(
                desc_q,
                desc_k,
                desc_g,
                p_q,
                p_k,
                p_g_raw,
                mbar_tma0,
                tb0,
                head,
                G_FP32,
            )
        load_beta_stage(gbeta, smem_beta, 0, tb0, vr0, head, lane, heads)

    # DEFER publishes chunk lc-1 at the top of chunk lc: on long chains the
    # bulk-store drain hides in the next input wait instead of stalling
    # prepare's production pace.  Short chains publish same-chunk (the drain
    # is cheap there and the earlier flag is worth more to the consumer).
    # ``pipe_decisions`` chooses.
    prev_flag_idx = cutlass.Int32(-1)
    for lc in cutlass.range_constexpr(CPC):
        if lc < my_chunks:
            gchunk = cta_chunk_base + lc
            seq = cutlass.Int32(gchunk_to_seq[gchunk])
            local_c = gchunk - cutlass.Int32(gcu_chunks[seq])
            seq_start = cutlass.Int32(gcu_seqlens[seq])
            seq_end = cutlass.Int32(gcu_seqlens[seq + 1])
            token_base = seq_start + local_c * BT
            valid_rows = seq_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            # ---- wait for this chunk's input TMA, publish the stage -----
            # the two barrier objects alternate, so each is
            # reused every other chunk and its phase toggles then.
            beta_stage = lc & 1
            tma_wait_phase = (lc >> 1) & 1
            if cutlass.const_expr(lc & 1):
                cute.arch.mbarrier_wait(mbar_tma1, tma_wait_phase)
            else:
                cute.arch.mbarrier_wait(mbar_tma0, tma_wait_phase)

            # A short chunk sits mid-tensor, so TMA loaded the
            # next sequence's rows rather than zeros.  Clear them before the
            # barrier that publishes the stage.
            if valid_rows < BT:
                clear_tail_rows(
                    p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32, PREPARE_DEVICE_THREADS
                )
            if cutlass.const_expr(PIPE and DEFER):
                if lane == 0:
                    tma_store_wait_all()
                    fence_proxy_async_global()
            cute.arch.barrier()
            if cutlass.const_expr(PIPE and DEFER):
                if warp_id == 0:
                    if lane == 0:
                        if prev_flag_idx >= 0:
                            st_release_gpu_u32(ws_flag.iterator + prev_flag_idx, gen)
                prev_flag_idx = head * TOTAL_CHUNKS + gchunk

            # ---- gate prefix --------------------------
            gate_regs = [cutlass.Float32(0.0) for _ in range(BT)]
            for r in cutlass.range_constexpr(BT):
                if cutlass.const_expr(G_FP32):
                    raw = smem_g[raw_f32_s128(r, tidx)]
                else:
                    raw = cutlass.Float32(smem_g_bf16[raw_bf16_s128(r, tidx)])
                inc = cutlass.Float32(0.0)
                if r < valid_rows:
                    x = raw + dt_value
                    if cutlass.const_expr(SAFE_GATE):
                        half = cutlass.Float32(0.5)
                        sig = (
                            cutlass.Float32(
                                cute.math.tanh(alog * x * half, fastmath=True)
                            )
                            * half
                            + half
                        )
                        inc = GATE_SCALE_LOG2 * sig
                    else:
                        sp = cutlass.Float32(0.0)
                        if x > cutlass.Float32(20.0):
                            sp = x * cutlass.Float32(LOG2_E)
                        else:
                            sp = cutlass.Float32(
                                cute.math.log2(
                                    cutlass.Float32(1.0)
                                    + cutlass.Float32(cute.math.exp(x, fastmath=True)),
                                    fastmath=True,
                                )
                            )
                        inc = -alog * sp
                gate_regs[r] = inc

            acc = cutlass.Float32(0.0)
            for p in cutlass.range_constexpr(BT // 2):
                r0 = p * 2
                g0 = gate_regs[r0]
                g1 = gate_regs[r0 + 1]
                prefix0 = acc + g0
                prefix1 = acc + (g0 + g1)
                gate_regs[r0] = prefix0
                gate_regs[r0 + 1] = prefix1
                acc = prefix1

            for r in cutlass.range_constexpr(BT):
                pv = gate_regs[r]
                if pv < cutlass.Float32(PREFIX_FLOOR):
                    pv = cutlass.Float32(PREFIX_FLOOR)
                gate_regs[r] = cutlass.Float32(cute.math.exp2(pv, fastmath=True))

            gamma = gate_regs[BT - 1]
            ws_gt[(head * TOTAL_CHUNKS + gchunk) * DK + tidx] = gamma
            smem_gamma[tidx] = gamma.to(cutlass.BFloat16)
            if cutlass.const_expr(not G_FP32):
                # The FP32 exp_g image about to be written spans all 8192 bytes
                # of the stage; the BF16 raw G it overwrites occupied only the
                # first 4096, in a different index map.  A thread's write can
                # therefore land on a raw element another thread has not read
                # yet, so the read loop above must be complete CTA-wide first.
                # With FP32 G the two maps coincide and each thread rewrites
                # exactly the addresses it read, so no barrier is needed.
                cute.arch.barrier()
            for r in cutlass.range_constexpr(BT):
                smem_g[raw_f32_s128(r, tidx)] = gate_regs[r]
            cute.arch.barrier()

            # ---- norm + Kd/Ki/Qd --------------------
            # Every SMEM and global access below is 16 bytes wide.  The 8
            # features a lane owns are contiguous and 16-byte aligned under
            # raw_bf16_s128, and exp_g is read as 2 x float4 because 8
            # consecutive FP32 are two separate 4-element runs whose order
            # depends on the token parity.
            q_ss = cutlass.Float32(0.0)
            k_ss = cutlass.Float32(0.0)
            for h in cutlass.range_constexpr(2):
                d0 = 64 * h + 8 * lane_in_row
                fq = vec8_bf16(p_q, raw_bf16_s128(my_token, d0))
                fk = vec8_bf16(p_k, raw_bf16_s128(my_token, d0))
                for i in cutlass.range_constexpr(8):
                    qv = cutlass.Float32(fq[i])
                    kv = cutlass.Float32(fk[i])
                    q_ss = q_ss + qv * qv
                    k_ss = k_ss + kv * kv
            q_ss = warp_row_sum_8(q_ss)
            k_ss = warp_row_sum_8(k_ss)

            q_inv = cutlass.Float32(0.0)
            k_inv = cutlass.Float32(0.0)
            if my_token < valid_rows:
                qf = q_ss
                if qf < cutlass.Float32(NORM_FLOOR):
                    qf = cutlass.Float32(NORM_FLOOR)
                kf = k_ss
                if kf < cutlass.Float32(NORM_FLOOR):
                    kf = cutlass.Float32(NORM_FLOOR)
                q_inv = cutlass.Float32(cute.math.rsqrt(qf, fastmath=True))
                k_inv = cutlass.Float32(cute.math.rsqrt(kf, fastmath=True))

            s16 = bf16_round(SCALE)
            for h in cutlass.range_constexpr(2):
                d0 = 64 * h + 8 * lane_in_row
                bidx = raw_bf16_s128(my_token, d0)
                fq = vec8_bf16(p_q, bidx)
                fk = vec8_bf16(p_k, bidx)
                fg0 = vec4_f32(p_g, raw_f32_s128(my_token, d0))
                fg1 = vec4_f32(p_g, raw_f32_s128(my_token, d0 + 4))

                o_kd = make_rmem_tensor(8, cutlass.BFloat16)
                o_ki = make_rmem_tensor(8, cutlass.BFloat16)
                o_qd = make_rmem_tensor(8, cutlass.BFloat16)
                for j in cutlass.range_constexpr(4):
                    i = 0 + j
                    eg = fg0[j]
                    kv = cutlass.Float32(fk[i]) * k_inv
                    kd_v = bf16_round(bf16_round(kv) * bf16_round(eg))
                    ki_v = bf16_round(kv * cutlass.Float32(cute.arch.rcp_approx(eg)))
                    qv = bf16_round(cutlass.Float32(fq[i]) * q_inv)
                    qd_v = bf16_round(bf16_round(qv * bf16_round(eg)) * s16)
                    o_kd[i] = kd_v.to(cutlass.BFloat16)
                    o_ki[i] = ki_v.to(cutlass.BFloat16)
                    o_qd[i] = qd_v.to(cutlass.BFloat16)

                for j in cutlass.range_constexpr(4):
                    i = 4 + j
                    eg = fg1[j]
                    kv = cutlass.Float32(fk[i]) * k_inv
                    kd_v = bf16_round(bf16_round(kv) * bf16_round(eg))
                    ki_v = bf16_round(kv * cutlass.Float32(cute.arch.rcp_approx(eg)))
                    qv = bf16_round(cutlass.Float32(fq[i]) * q_inv)
                    qd_v = bf16_round(bf16_round(qv * bf16_round(eg)) * s16)
                    o_kd[i] = kd_v.to(cutlass.BFloat16)
                    o_ki[i] = ki_v.to(cutlass.BFloat16)
                    o_qd[i] = qd_v.to(cutlass.BFloat16)

                store_vec8_bf16(p_kd, bidx, o_kd)
                store_vec8_bf16(p_ki, bidx, o_ki)
                store_vec8_bf16(p_qd, bidx, o_qd)
                # Signal each Kd/Ki half the moment it is in SMEM.  Kd and Qd
                # leave for global by TMA later, read straight out of these
                # same images, so there is no separate global store here.
                if cutlass.const_expr(h == 0):
                    warp_arrive(mbar_k_half, lane)
                else:
                    warp_arrive(mbar_k_full, lane)

            # This warp is done with raw Q/K/exp-g and with Qd; the raw
            # stages may be overwritten once all four arrive.
            warp_arrive(mbar_raw_released, lane)
            phase = lc & 1

            # Next chunk's beta, split into an early load and a late
            # activation (as the fused variant's ``inverse_role`` does): the
            # sixteen-lane column read out of [T, H] cannot coalesce, so W1
            # issues the load at the top of its tail and activates at the
            # bottom, with the whole tail covering the DRAM latency.  The
            # carriers are predefined here because a name first assigned
            # inside a dynamic branch becomes an unbalanced branch output at
            # trace time.  Carried as the raw BF16 scalar, not FP32:
            # converting at the load site puts a CVT right behind the LDG,
            # and that CVT is a consumer, so the whole DRAM latency would land
            # there.  The convert belongs at the activation site below.
            beta_logit = cutlass.BFloat16(0.0)
            beta_do = cutlass.Int32(0)
            beta_nvr = cutlass.Int32(0)
            next_tb = cutlass.Int32(0)
            next_vr = cutlass.Int32(0)
            # The next chunk's coordinates are a two-deep chain of scalar
            # global loads (chunk_to_seq, then cu_chunks/cu_seqlens); read at
            # the issue site they would stall W1 for two round trips.  Issued
            # here, the whole gate + norm phase covers them.
            if cutlass.const_expr(lc + 1 < CPC):
                if warp_id == 1:
                    nchunk_top = gchunk + 1
                    if nchunk_top < TOTAL_CHUNKS:
                        nseq_top = cutlass.Int32(gchunk_to_seq[nchunk_top])
                        nlc_top = nchunk_top - cutlass.Int32(gcu_chunks[nseq_top])
                        next_tb = cutlass.Int32(gcu_seqlens[nseq_top]) + nlc_top * BT
                        next_vr = cutlass.Int32(gcu_seqlens[nseq_top + 1]) - next_tb
                        if next_vr > BT:
                            next_vr = cutlass.Int32(BT)

            # ---- warp 0: KK -> L -> AINV ------------------------------
            if warp_id == 0:
                cute.arch.mbarrier_wait(mbar_k_half, phase)
                c = kk_half(p_kd, p_ki, lane, 0, ZERO8())
                cute.arch.mbarrier_wait(mbar_k_full, phase)
                c = kk_half(p_kd, p_ki, lane, 1, c)
                masked = [cutlass.Float32(0.0) for _ in range(8)]
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = cutlass.Float32(0.0)
                    if row > col:
                        v = c[slot] * smem_beta[beta_stage * BT + row]
                    masked[slot] = v

                # Blockwise 8x8 inverse.  With D the block diagonal of L and
                # A21 its one off-diagonal block:
                #
                #   Binv = (I - D)(I + D^2)(I + D^4)   exact, blocks nilpotent
                #                                      at 8, so 3 factors
                #   AINV = Binv - Binv @ A21 @ Binv    exact, (Binv @ A21)^2 = 0
                #
                # Six MMAs, and it never forms a power above D^4 of an 8x8
                # block, whereas a 16x16 Neumann chain would build L^8 of the
                # full matrix only to cancel it away.  Operands are FP16, not
                # BF16.
                d = [cutlass.Float32(0.0) for _ in range(8)]
                a21 = [cutlass.Float32(0.0) for _ in range(8)]
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    if (row < HALF_BT) == (col < HALF_BT):
                        d[slot] = masked[slot]
                    elif row >= HALF_BT:
                        a21[slot] = masked[slot]

                dp = acc_to_a_fragment_f16(tuple(d))
                pows = []
                for _ in cutlass.range_constexpr(2):
                    pb = a_to_b((dp[0], dp[1], dp[2], dp[3]))
                    sq = mma_16x16_f16(
                        (dp[0], dp[1], dp[2], dp[3]),
                        (pb[0], pb[1], pb[2], pb[3]),
                        ZERO8(),
                    )
                    dp = acc_to_a_fragment_f16(sq)
                    pows.append(dp)

                # binv = (I - D) (I + D^2) (I + D^4), as a running accumulator:
                # every D^k here is a power of a strict-lower matrix and so has
                # a zero diagonal, which makes R(I + D^k) == I + R(D^k) and
                # therefore MMA(x, I + D^k) == x + MMA(x, D^k).  Materializing
                # I + D^k instead would cost a second pack per step in the
                # issue-bound warp-0 region.
                binv = [cutlass.Float32(0.0) for _ in range(8)]
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    eye = cutlass.Float32(0.0)
                    if row == col:
                        eye = cutlass.Float32(1.0)
                    binv[slot] = eye - f16_round(d[slot])

                for step in cutlass.range_constexpr(2):
                    lhs = acc_to_a_fragment_f16(tuple(binv))
                    pf = pows[step]
                    pb = a_to_b((pf[0], pf[1], pf[2], pf[3]))
                    prod = mma_16x16_f16(
                        (lhs[0], lhs[1], lhs[2], lhs[3]),
                        (pb[0], pb[1], pb[2], pb[3]),
                        ZERO8(),
                    )
                    for slot in cutlass.range_constexpr(8):
                        binv[slot] = f16_round(binv[slot]) + prod[slot]

                # x21 = -(Binv @ A21) @ Binv, written into the lower-left block.
                bf = acc_to_a_fragment_f16(tuple(binv))
                af = acc_to_a_fragment_f16(tuple(a21))
                ab = a_to_b((af[0], af[1], af[2], af[3]))
                t1 = mma_16x16_f16(
                    (bf[0], bf[1], bf[2], bf[3]), (ab[0], ab[1], ab[2], ab[3]), ZERO8()
                )
                tf = acc_to_a_fragment_f16(t1)
                bb = a_to_b((bf[0], bf[1], bf[2], bf[3]))
                x21 = mma_16x16_f16(
                    (tf[0], tf[1], tf[2], tf[3]), (bb[0], bb[1], bb[2], bb[3]), ZERO8()
                )

                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = binv[slot]
                    if row >= HALF_BT and col < HALF_BT:
                        v = -x21[slot]
                    smem_ainv[prepare_pair_idx(row, col)] = bf16_round(v).to(
                        cutlass.BFloat16
                    )
                # This arrival publishes AINV and also proves warp 0 is done
                # reading the Kd stage, which is what lets warps 1 and 3
                # overwrite their Kd half with Ak.
                warp_arrive(mbar_pairwise, lane)

            # ---- warp 2: causal QK, staged through SMEM ---------------
            if warp_id == 2:
                cute.arch.mbarrier_wait(mbar_k_full, phase)
                c = kk_over_dk(p_qd, p_ki, lane)
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = cutlass.Float32(0.0)
                    if row >= col:
                        v = c[slot]
                    smem_qk[prepare_pair_idx(row, col)] = bf16_round(v).to(
                        cutlass.BFloat16
                    )
                # smem_qk is produced and consumed by this warp alone.
                cute.arch.sync_warp()

            # ---- warp 1: release the raw stages, then prefetch ----------
            # One elected lane issues the whole 16 KB by TMA, so only warp 1
            # stops for the prefetch.
            if warp_id == 1:
                cute.arch.mbarrier_wait(mbar_raw_released, phase)
                # Publish the ordinary Kd stores to the async
                # shared proxy before the TMA engine reads them, and converge
                # the warp so lane 0 speaks for all 32 lanes' writes.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(desc_factor, p_kd, 0, gchunk * BT, head)
                    tma_store_commit_group()
                if lc + 1 < CPC:
                    if lc + 1 < my_chunks:
                        ntb = next_tb
                        nvr = next_vr
                        if lane == 0:
                            if cutlass.const_expr((lc + 1) & 1):
                                cute.arch.mbarrier_arrive_and_expect_tx(mbar_tma1, G_TX)
                                issue_chunk_tma(
                                    desc_q,
                                    desc_k,
                                    desc_g,
                                    p_q,
                                    p_k,
                                    p_g_raw,
                                    mbar_tma1,
                                    ntb,
                                    head,
                                    G_FP32,
                                )
                            else:
                                cute.arch.mbarrier_arrive_and_expect_tx(mbar_tma0, G_TX)
                                issue_chunk_tma(
                                    desc_q,
                                    desc_k,
                                    desc_g,
                                    p_q,
                                    p_k,
                                    p_g_raw,
                                    mbar_tma0,
                                    ntb,
                                    head,
                                    G_FP32,
                                )
                        # Issue the beta logit read here; consume it at the
                        # end of the chunk.  The register carries it across
                        # the Ki-gamma work, the pairwise wait and the Ak
                        # build, which is what buys the load its distance.
                        beta_do = cutlass.Int32(1)
                        beta_nvr = nvr
                        if lane < BT:
                            if lane < nvr:
                                beta_logit = gbeta[(ntb + lane) * heads + head]
            elif warp_id == 3:
                # Warp 3 issues none of the prefetch, but it still acquires
                # ``mbar_raw_released``: that is what makes the other warps'
                # Ki stores visible to its Ak path below, without relying on a
                # chained release/acquire through warp 0.
                cute.arch.mbarrier_wait(mbar_raw_released, phase)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    # Kd segment 1 plus both Qd segments, one group.
                    tma_store_3d(
                        desc_factor,
                        p_kd + BF16_SEG_STRIDE,
                        BF16_SEG_ELEMS,
                        gchunk * BT,
                        head,
                    )
                    tma_store_3d(desc_factor, p_qd, 0, gchunk * BT, heads + head)
                    tma_store_3d(
                        desc_factor,
                        p_qd + BF16_SEG_STRIDE,
                        BF16_SEG_ELEMS,
                        gchunk * BT,
                        heads + head,
                    )
                    tma_store_commit_group()

            # ---- AINV_beta, then Aq (warp 2) and Ak (warps 1 and 3) ---
            # ``mbar_pairwise`` is a plain release/acquire pair on smem_ainv,
            # whose only writer is warp 0.  The Ki that warps 1 and 3 read
            # below is covered without leaning on chained visibility: warp 2
            # acquired ``mbar_k_full`` directly, and warps 1 and 3 acquired
            # ``mbar_raw_released``, on which every warp arrives after its
            # ``mbar_k_full`` arrival and therefore after its Ki stores.
            if warp_id != 0:
                bst = beta_stage * BT
                if warp_id == 2:
                    cute.arch.mbarrier_wait(mbar_pairwise, phase)
                    ai = load_pairwise_a_fragment(p_ainv, lane)
                    blo = pack_bf16x2(
                        smem_beta[bst + 2 * q4], smem_beta[bst + 2 * q4 + 1]
                    )
                    bhi = pack_bf16x2(
                        smem_beta[bst + 2 * q4 + 8], smem_beta[bst + 2 * q4 + 9]
                    )
                    ab0 = mul_bf16x2(ai[0], blo)
                    ab1 = mul_bf16x2(ai[1], blo)
                    ab2 = mul_bf16x2(ai[2], bhi)
                    ab3 = mul_bf16x2(ai[3], bhi)
                    bb = a_to_b((ab0, ab1, ab2, ab3))
                    qk_a = load_pairwise_a_fragment(p_qk, lane)
                    aq = mma_16x16(
                        (qk_a[0], qk_a[1], qk_a[2], qk_a[3]),
                        (bb[0], bb[1], bb[2], bb[3]),
                        ZERO8(),
                    )
                    base = (head * TOTAL_CHUNKS + gchunk) * (BT * BT)
                    # Stage through SMEM so the global store is one contiguous
                    # 16-byte run per lane.  Direct stores cannot be:
                    # prepare_pair_idx swaps the column halves (col ^ 8), so
                    # one store instruction only ever fills half of each row's
                    # 32 bytes, spraying 128 B over 8 sectors, and L1 does not
                    # merge across instructions, so twice the sectors leave
                    # the SM.  Staged, the L1->L2 write traffic equals the
                    # output's native size.
                    #
                    # smem_qk is free here: warp 2 owns it alone and has just
                    # consumed it into qk_a, so this costs no shared memory.
                    cute.arch.sync_warp()
                    aqf = acc_to_a_fragment(aq)
                    srow, scol = prepare_stmatrix_coord(lane)
                    stmatrix_x4(
                        p_qk + prepare_pair_idx(srow, scol),
                        aqf[0],
                        aqf[1],
                        aqf[2],
                        aqf[3],
                    )
                    cute.arch.sync_warp()
                    cute.autovec_copy(
                        vec8_bf16(p_qk, lane * 8),
                        vec_at(ws_aq.iterator, base + lane * 8, 8),
                    )

                else:
                    # The AINV-independent half of Ak, done while W0 is
                    # still solving: Ki is covered by the warp's own
                    # ``mbar_raw_released`` acquire and gamma by the exp_g
                    # barrier, so the four Ki-gamma B operands can be built
                    # before the pairwise wait instead of after it.  The
                    # instruction sequence per tile is unchanged, so the
                    # output stays bit-identical; only the wait shrinks.
                    tile_base = 0
                    if warp_id == 3:
                        tile_base = 4
                    kbs = []
                    for t in cutlass.range_constexpr(4):
                        d0 = (tile_base + t) * 16
                        ki0, ki1, ki2, ki3 = load_a_fragment(p_ki, 0, d0, lane)
                        gl = pack_bf16x2(
                            cutlass.Float32(smem_gamma[d0 + 2 * q4]),
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 1]),
                        )
                        gh = pack_bf16x2(
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 8]),
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 9]),
                        )
                        kbs.append(
                            a_to_b(
                                (
                                    mul_bf16x2(ki0, gl),
                                    mul_bf16x2(ki1, gl),
                                    mul_bf16x2(ki2, gh),
                                    mul_bf16x2(ki3, gh),
                                ),
                            )
                        )
                    cute.arch.mbarrier_wait(mbar_pairwise, phase)
                    ai = load_pairwise_a_fragment(p_ainv, lane)
                    blo = pack_bf16x2(
                        smem_beta[bst + 2 * q4], smem_beta[bst + 2 * q4 + 1]
                    )
                    bhi = pack_bf16x2(
                        smem_beta[bst + 2 * q4 + 8], smem_beta[bst + 2 * q4 + 9]
                    )
                    ab0 = mul_bf16x2(ai[0], blo)
                    ab1 = mul_bf16x2(ai[1], blo)
                    ab2 = mul_bf16x2(ai[2], bhi)
                    ab3 = mul_bf16x2(ai[3], bhi)
                    # ``mbar_pairwise`` above proved warp 0 is done reading
                    # the Kd stage; this waits for the warp's own Kd TMA store
                    # to have finished *reading* its half, which is the other
                    # half of the condition for overwriting it.  The warp
                    # synchronization joins the two before any lane writes.
                    if lane == 0:
                        tma_store_wait_read(0)
                    cute.arch.sync_warp()

                    at0, at1, at2, at3 = a_to_a_transposed(ab0, ab1, ab2, ab3)
                    for t in cutlass.range_constexpr(4):
                        d0 = (tile_base + t) * 16
                        kb = kbs[t]
                        akc = mma_16x16(
                            (at0, at1, at2, at3), (kb[0], kb[1], kb[2], kb[3]), ZERO8()
                        )
                        # Publish Ak.T with stmatrix.x4 through
                        # the row-^8 image, into the Kd stage that is now dead.
                        # Warp 1 owns d < 64 -> bytes [0,2048), warp 3 the rest.
                        akf = acc_to_a_fragment(akc)
                        srow, scol = prepare_stmatrix_coord(lane)
                        stmatrix_x4(
                            p_kd + kr_ak_idx(srow, d0 + scol),
                            akf[0],
                            akf[1],
                            akf[2],
                            akf[3],
                        )

                    # Each store warp moves its own Ak segment: the warp that
                    # produced the half is the warp that ships it, so no CTA
                    # barrier or re-read by other warps is needed.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        seg = tile_base // 4
                        tma_store_3d(
                            desc_factor,
                            p_kd + seg * BF16_SEG_STRIDE,
                            seg * BF16_SEG_ELEMS,
                            gchunk * BT,
                            2 * heads + head,  # Ak is plane region 2
                        )
                        tma_store_commit_group()
                        # The source stage cannot be reused
                        # until the store has read it.
                        tma_store_wait_read(0)

            # The deferred half of the beta split: activate and publish the
            # logit W1 loaded at the top of its tail.  Storing here still
            # precedes the recycle barrier, which publishes it to the next
            # chunk's readers.
            if cutlass.const_expr(lc + 1 < CPC):
                if warp_id == 1:
                    if beta_do != 0:
                        if lane < BT:
                            bv = cutlass.Float32(0.0)
                            if lane < beta_nvr:
                                half = cutlass.Float32(0.5)
                                bv = (
                                    cutlass.Float32(
                                        cute.math.tanh(
                                            cutlass.Float32(beta_logit) * half,
                                            fastmath=True,
                                        )
                                    )
                                    * half
                                    + half
                                )
                            smem_beta[((lc + 1) & 1) * BT + lane] = bv

            # Same-chunk publication: the chunk's bulk stores must be complete
            # (not just source-read) and proxy-fenced before the recycle
            # barrier orders everything ahead of the flag.  Deferring the flag
            # by a chunk is free on prepare's side but charges a consumer that
            # has caught up with the frontier a whole publication period, which
            # is why short chains publish here rather than at the top of the
            # next chunk.
            if cutlass.const_expr(PIPE and not DEFER):
                if lane == 0:
                    tma_store_wait_all()
                    fence_proxy_async_global()

            # Chunk recycle.
            cute.arch.barrier()

            if cutlass.const_expr(PIPE and not DEFER):
                if warp_id == 0:
                    if lane == 0:
                        st_release_gpu_u32(
                            ws_flag.iterator + (head * TOTAL_CHUNKS + gchunk),
                            gen,
                        )

    if cutlass.const_expr(PIPE and DEFER):
        if lane == 0:
            tma_store_wait_all()
            fence_proxy_async_global()
        cute.arch.barrier()
        if warp_id == 0:
            if lane == 0:
                if prev_flag_idx >= 0:
                    st_release_gpu_u32(ws_flag.iterator + prev_flag_idx, gen)


def _flat(t: torch.Tensor):
    """Flat CuTe view, cached on the address (see :func:`~.runtime.flat_view`)."""
    return flat_view(t, align=16)


# --------------------------------------------------------------------------
# The recurrence device kernel
# --------------------------------------------------------------------------


# Barrier slots as Int64 indices into the arena.
BAR_IN_READY = MBAR_INPUT_READY // 8  # 0
BAR_IN_CONSUMED = MBAR_INPUT_CONSUMED // 8  # 5
BAR_OUT_READY = MBAR_OUTPUT_READY // 8  # 10
BAR_OUT_CONSUMED = MBAR_OUTPUT_CONSUMED // 8  # 12
BAR_STATE = MBAR_STATE_READY // 8  # 14

#: One 128-byte S128 segment of a factor tile, in BF16 elements.
FACTOR_SEGMENT_ELEMS = BF16_SEGMENT_STRIDE  # 1024

#: 16-byte tasks covering the whole state half, for the conversion and clear
#: passes.  1024 tasks over 320 threads is four rounds with a bound check; this
#: runs once per kernel, not per chunk.
STATE_TASKS = DV_HALF * (DK // 8)  # 1024
STATE_TASK_ROUNDS = (STATE_TASKS + REC_THREADS - 1) // REC_THREADS  # 4


# ---------------------------------------------------------------------------
# Small DSL helpers
# ---------------------------------------------------------------------------


@cute.jit
def zero_acc4():
    z = cutlass.Float32(0.0)
    return (z, z, z, z)


# ---------------------------------------------------------------------------
# State entry and exit
# ---------------------------------------------------------------------------


@cute.jit
def issue_factor_boxes(p16, desc_factor, mbar, stage16, factor_row, heads, head):
    """The six Kd/Qd/Ak boxes of one stage, from the elected lane.

    Shared by both producer modes so they cannot drift: the cp.async mode
    sends only these six by TMA, the small-grid mode sends these six plus
    Aq, GTotal and V.
    """
    for factor in cutlass.range_constexpr(3):
        plane = factor * heads + head
        dst = stage16 + (STAGE_KD // 2) + factor * (4096 // 2)
        for segment in cutlass.range_constexpr(BF16_SEGMENTS):
            tma_load_3d(
                p16 + (dst + segment * FACTOR_SEGMENT_ELEMS),
                desc_factor,
                mbar,
                segment * BF16_SEGMENT_ELEMS,
                factor_row,
                plane,
            )


@cute.jit
def clear_state(p_state, tidx):
    """Zero the persistent BF16 state with every CTA thread."""
    zeros = make_rmem_tensor(8, cutlass.BFloat16)
    for i in cutlass.range_constexpr(8):
        zeros[i] = cutlass.BFloat16(0.0)
    for rep in cutlass.range_constexpr(STATE_TASK_ROUNDS):
        task = tidx + rep * REC_THREADS
        if task < STATE_TASKS:
            v_local = task // (DK // 8)
            k0 = (task - v_local * (DK // 8)) * 8
            store_vec8_bf16(p_state, state_bf16_idx(v_local, k0), zeros)


@cute.jit
def state_f32_to_bf16(p_state, p_state_f32, tidx):
    """Round the FP32 landing buffer into the BF16 persistent state.

    Even an FP32 external state crosses this boundary, because the state
    carried between chunks is BF16 by contract.
    """
    for rep in cutlass.range_constexpr(STATE_TASK_ROUNDS):
        task = tidx + rep * REC_THREADS
        if task < STATE_TASKS:
            v_local = task // (DK // 8)
            k0 = (task - v_local * (DK // 8)) * 8
            # The FP32 image groups four elements, so eight keys are two
            # separate aligned float4 runs whose order depends on the row.
            lo = vec4_f32(p_state_f32, state_f32_idx(v_local, k0))
            hi = vec4_f32(p_state_f32, state_f32_idx(v_local, k0 + 4))
            out = make_rmem_tensor(8, cutlass.BFloat16)
            for i in cutlass.range_constexpr(4):
                out[i] = lo[i].to(cutlass.BFloat16)
                out[4 + i] = hi[i].to(cutlass.BFloat16)
            store_vec8_bf16(p_state, state_bf16_idx(v_local, k0), out)


@cute.jit
def state_bf16_to_f32(p_state, p_state_f32, tidx):
    """Widen the BF16 state into the FP32 buffer an FP32 final state stores."""
    for rep in cutlass.range_constexpr(STATE_TASK_ROUNDS):
        task = tidx + rep * REC_THREADS
        if task < STATE_TASKS:
            v_local = task // (DK // 8)
            k0 = (task - v_local * (DK // 8)) * 8
            src = vec8_bf16(p_state, state_bf16_idx(v_local, k0))
            lo = make_rmem_tensor(4, cutlass.Float32)
            hi = make_rmem_tensor(4, cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                lo[i] = cutlass.Float32(src[i])
                hi[i] = cutlass.Float32(src[4 + i])
            store_vec4_f32(p_state_f32, state_f32_idx(v_local, k0), lo)
            store_vec4_f32(p_state_f32, state_f32_idx(v_local, k0 + 4), hi)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def recurrence_kernel(
    gout: cute.Tensor,
    gv: cute.Tensor,
    gaq: cute.Tensor,
    ggt: cute.Tensor,
    gflag: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    desc_factor: cutlass.Int64,
    desc_aq: cutlass.Int64,
    desc_gt: cutlass.Int64,
    desc_v: cutlass.Int64,
    desc_out: cutlass.Int64,
    desc_state_in: cutlass.Int64,
    desc_state_out: cutlass.Int64,
    heads: cutlass.Int32,
    total_chunks: cutlass.Int32,
    rec_ctas: cutlass.Int32,
    gen: cutlass.Int32,
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
    PIPE: cutlass.Constexpr,
    GATE_ACQ: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    # x = 2 * seq + dv_half, y = head (recurrence_grid): the DV half is the
    # fastest-varying coordinate, so the two halves of a head are adjacent
    # blocks and share a wave, which is what lets L2 serve the second one's
    # factor loads.  Everything coarser keeps the order (N, 2H, 1) had.
    seq = bidx >> 1
    dv_half = bidx - seq * 2
    head = bidy

    # --- fixed arena ------------------------------------
    # One allocation of exactly SMEM_DYNAMIC_BYTES, so the launch parameter is
    # that same number and every address is a compile-time constant offset.
    alloc = cutlass.utils.SmemAllocator()
    p_base = alloc.allocate(SMEM_DYNAMIC_BYTES, 1024)
    p16 = cute.recast_ptr(p_base, dtype=cutlass.BFloat16)
    p32 = cute.recast_ptr(p_base, dtype=cutlass.Float32)
    p64 = cute.recast_ptr(p_base, dtype=cutlass.Int64)

    p_state = p16 + (SMEM_STATE // 2)
    p_state_f32 = p32 + (SMEM_STATE_F32 // 4)
    p_bar = p64 + (REC_SMEM_BARRIERS // 8)

    # --- sequence coordinates ---------------------------
    # Read before the barrier init because the IN_READY arrival count is
    # mode-dependent (below), and the mode needs ``num_chunks``.
    token_start = cutlass.Int32(gcu_seqlens[seq])
    token_end = cutlass.Int32(gcu_seqlens[seq + 1])
    chunk_base = cutlass.Int32(gcu_chunks[seq])
    num_chunks = cutlass.Int32(gcu_chunks[seq + 1]) - chunk_base
    state_plane = seq * heads + head

    # Whether this CTA's producer runs the cp.async split (see the producer
    # role).  Both axes are needed: the TMA queue it goes around is
    # collective, so small grids never queue, and the split's fixed costs
    # need chunks to amortise.  ``num_chunks`` is this sequence's own count,
    # so a ragged varlen batch decides per CTA.
    use_cp = cutlass.Int32(0)
    if rec_ctas >= CP_ASYNC_MIN_CTAS:
        if num_chunks >= CP_ASYNC_MIN_CHUNKS:
            use_cp = cutlass.Int32(1)

    if warp_id == 0:
        if lane == 0:
            for s in cutlass.range_constexpr(INPUT_STAGES):
                # Two arrivals in cp.async mode (the expect-tx, then the
                # elected arrive after wait_group); one in plain-TMA mode, so
                # that mode pays no second-arrival tax.
                cute.arch.mbarrier_init(
                    p_bar + BAR_IN_READY + s,
                    INPUT_READY_ARRIVALS_TMA + use_cp,
                )
                cute.arch.mbarrier_init(
                    p_bar + BAR_IN_CONSUMED + s, INPUT_CONSUMED_ARRIVALS
                )
            for s in cutlass.range_constexpr(OUTPUT_STAGES):
                cute.arch.mbarrier_init(
                    p_bar + BAR_OUT_READY + s, OUTPUT_READY_ARRIVALS
                )
                cute.arch.mbarrier_init(
                    p_bar + BAR_OUT_CONSUMED + s, OUTPUT_CONSUMED_ARRIVALS
                )
            cute.arch.mbarrier_init(p_bar + BAR_STATE, STATE_READY_ARRIVALS)
            cute.arch.mbarrier_init_fence()
            fence_tensormap_acquire(desc_factor)
            fence_tensormap_acquire(desc_aq)
            fence_tensormap_acquire(desc_gt)
            fence_tensormap_acquire(desc_v)
            fence_tensormap_acquire(desc_out)
            if cutlass.const_expr(HAS_STATE_IN):
                fence_tensormap_acquire(desc_state_in)
            if cutlass.const_expr(HAS_STATE_OUT):
                fence_tensormap_acquire(desc_state_out)
    cute.arch.barrier()

    # --- initial state ----------------------------------
    if cutlass.const_expr(HAS_STATE_IN):
        if cutlass.const_expr(STATE_FP32):
            if warp_id == 0:
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        p_bar + BAR_STATE, DV_HALF * DK * 4
                    )
                    tma_load_3d(
                        p_state_f32,
                        desc_state_in,
                        p_bar + BAR_STATE,
                        0,
                        dv_half * (STATE_F32_ROWS_PER_VALUE * DV_HALF),
                        state_plane,
                    )
            cute.arch.mbarrier_wait(p_bar + BAR_STATE, 0)
            state_f32_to_bf16(p_state, p_state_f32, tidx)
        else:
            if warp_id == 0:
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        p_bar + BAR_STATE, DV_HALF * DK * 2
                    )
                    tma_load_3d(
                        p_state,
                        desc_state_in,
                        p_bar + BAR_STATE,
                        0,
                        dv_half * (STATE_BF16_ROWS_PER_VALUE * DV_HALF),
                        state_plane,
                    )
            cute.arch.mbarrier_wait(p_bar + BAR_STATE, 0)
    else:
        clear_state(p_state, tidx)
    cute.arch.barrier()

    # --- roles ---------------------------------
    if warp_id < COMPUTE_WARPS:
        v_base = warp_id * WARP_VALUES
        # The state lives in registers for the whole kernel, read from shared
        # memory once here and written back once after the chunk loop.  Sixteen
        # packed BF16 registers per lane hold this warp's [128 key, 8 value]
        # slice, which is warp-private -- ``v_base`` is ``warp_id *
        # WARP_VALUES`` and no warp ever addresses another's columns -- so
        # nothing here needs a cross-warp exchange.
        #
        # It replaces 24 shared-memory instructions per chunk per warp (eight
        # ``ldmatrix_x2`` in pass 1, eight ``ldmatrix_x2_trans`` and eight
        # ``stmatrix_x2_trans`` in pass 2) with 16 ``movmatrix``.  The
        # ``.trans`` read is the C view, which is the layout pass 2 accumulates
        # in; pass 1 wants the B operand, which is the same 8x8 transpose per
        # register and so one ``movmatrix`` each.
        h_state: tuple = ()
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            lo, hi = ldmatrix_x2_trans(p_state + state_x2_ptr(lane, kb, v_base))
            h_state = h_state + (lo, hi)

        for c in range(num_chunks):
            in_stage = input_stage(c)
            out_stage = output_stage(c)
            stage16 = (SMEM_INPUT // 2) + in_stage * (INPUT_STAGE_BYTES // 2)
            p_kd = p16 + (stage16 + STAGE_KD // 2)
            p_qd = p16 + (stage16 + STAGE_QD // 2)
            p_ak = p16 + (stage16 + STAGE_AK // 2)
            p_aq = p16 + (stage16 + STAGE_AQ // 2)
            p_v = p16 + (stage16 + STAGE_V // 2)
            smem_gt = cute.make_tensor(
                p32
                + ((SMEM_INPUT + STAGE_GT) // 4 + in_stage * (INPUT_STAGE_BYTES // 4)),
                cute.make_layout(DK),
            )
            p_out = p16 + ((SMEM_OUTPUT // 2) + out_stage * (OUTPUT_STAGE_BYTES // 2))

            token_base = token_start + c * BT
            valid_rows = token_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            cute.arch.mbarrier_wait(
                p_bar + BAR_IN_READY + in_stage, input_ready_parity(c)
            )

            # ---- pass 1: X = Kd @ H and O = Qd @ H --------------------------
            # One state B fragment per key block feeds both MMAs, and nothing
            # writes the state until pass 2, so the fragment can be dropped
            # immediately after use.
            acc_x = zero_acc4()
            acc_o = zero_acc4()
            for kb in cutlass.range_constexpr(KEY_BLOCKS):
                # C -> B is a within-tile 8x8 transpose, one instruction per
                # packed register, and the fragment is fresh rather than
                # aliased onto ``h_state``: aliasing an MMA operand onto the
                # persistent state registers makes the compiler spill them.
                b = (
                    movmatrix_b16(h_state[2 * kb]),
                    movmatrix_b16(h_state[2 * kb + 1]),
                )
                acc_x = mma_n8(
                    ldmatrix_x4(p_kd + factor_a_fragment_ptr(lane, kb)), b, acc_x
                )
                acc_o = mma_n8(
                    ldmatrix_x4(p_qd + factor_a_fragment_ptr(lane, kb)), b, acc_o
                )

            # ---- residual -------------------------
            # Both halves of a packed C register are the same token row, so the
            # tail mask is one predicate per register.  An invalid row selects an
            # exact packed BF16 zero instead of subtracting a V that belongs to
            # the next sequence.

            v_lo, v_hi = ldmatrix_x2(p_v + vo_x2_ptr(lane, v_base))
            row_lo = lane // 4
            res_lo = cutlass.Int32(0)
            res_hi = cutlass.Int32(0)
            if row_lo < valid_rows:
                res_lo = sub_bf16x2(v_lo, pack_bf16x2(acc_x[0], acc_x[1]))
            if row_lo + 8 < valid_rows:
                res_hi = sub_bf16x2(v_hi, pack_bf16x2(acc_x[2], acc_x[3]))
            # C layout -> B layout: transpose each packed 8x8 quadrant in place.
            res_b = (movmatrix_b16(res_lo), movmatrix_b16(res_hi))

            # ---- O += Aq @ R, into the same FP32 accumulator ----------------
            acc_o = mma_n8(
                ldmatrix_x4(p_aq + pairwise_a_fragment_ptr(lane)), res_b, acc_o
            )

            # ---- publish the output before the state update -----------------
            cute.arch.mbarrier_wait(
                p_bar + BAR_OUT_CONSUMED + out_stage, output_consumed_parity(c)
            )
            stmatrix_x2(
                p_out + vo_x2_ptr(lane, v_base),
                pack_bf16x2(acc_o[0], acc_o[1]),
                pack_bf16x2(acc_o[2], acc_o[3]),
            )
            warp_arrive(p_bar + BAR_OUT_READY + out_stage, lane)

            # ---- pass 2: H_next = Diag(GTotal) H + Ak @ R -------------------
            # The same addresses as pass 1, read with ``.trans`` so the state
            # arrives already in the accumulator's layout.  Each warp writes
            # only its own value columns, and pass 1 is complete, so the update
            # is safe in place.
            next_state: tuple = ()
            for kb in cutlass.range_constexpr(KEY_BLOCKS):
                h0, h1 = unpack_bf16x2(h_state[2 * kb])
                h2, h3 = unpack_bf16x2(h_state[2 * kb + 1])
                decay_lo = cutlass.Float32(smem_gt[kb * BT + row_lo])
                decay_hi = cutlass.Float32(smem_gt[kb * BT + row_lo + 8])
                acc_s = (
                    decay_lo * h0,
                    decay_lo * h1,
                    decay_hi * h2,
                    decay_hi * h3,
                )
                acc_s = mma_n8(
                    ldmatrix_x4_trans(p_ak + ak_a_fragment_ptr(lane, kb)),
                    res_b,
                    acc_s,
                )
                next_state = next_state + (
                    pack_bf16x2(acc_s[0], acc_s[1]),
                    pack_bf16x2(acc_s[2], acc_s[3]),
                )
            h_state = next_state

            warp_arrive(p_bar + BAR_IN_CONSUMED + in_stage, lane)

        # The final-state path below reads the state from shared memory with all
        # 320 threads, so the compute warps publish their registers first.  The
        # converging barrier after this branch is what orders it.
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            stmatrix_x2_trans(
                p_state + state_x2_ptr(lane, kb, v_base),
                h_state[2 * kb],
                h_state[2 * kb + 1],
            )

    elif warp_id == LOAD_WARP:
        # Lane 0's flag watermark and this head's flag row (head-major layout,
        # so consecutive chunks are consecutive words and the lookahead walks
        # a cache line).  Defined before the loop: pytree rule.
        fw = cutlass.Int32(0)
        lk_val = cutlass.Int32(0)
        lk_idx = cutlass.Int32(-1)
        flag_head_base = head * total_chunks
        for c in range(num_chunks):
            in_stage = input_stage(c)
            cute.arch.mbarrier_wait(
                p_bar + BAR_IN_CONSUMED + in_stage, input_consumed_parity(c)
            )

            # Overlap mode: this chunk's factor slab must have been published
            # by the concurrently running prepare before any of the nine boxes
            # (or the cp.async granules) reads it.  Both paths acquire the
            # publication: the blocking spin uses an acquire load, while a
            # relaxed lookahead hit needs an acquire fence before consumption.
            # The warp sync then carries lane 0's acquire to the cp.async lanes.
            if cutlass.const_expr(PIPE):
                if lane == 0:
                    if lk_idx == fw:
                        if lk_val >= gen:
                            if cutlass.const_expr(not GATE_ACQ):
                                fence_acquire_gpu()
                            fw = fw + 1
                    if fw <= c:
                        spin_wait_flag_ge_u32(
                            gflag.iterator + (flag_head_base + chunk_base + c),
                            gen,
                            "rec",
                            32,
                        )
                        fw = c + 1
                    if fw < num_chunks:
                        if fw <= c + 8:
                            lk_idx = fw
                            # Tight-chase shapes (many chunks, many CTAs) run
                            # the acquire flavour, whose issue stall paces the
                            # chase; elsewhere the relaxed poll defers its
                            # acquire fence until the lookahead is consumed.
                            # ``pipe_decisions`` chooses.
                            if cutlass.const_expr(GATE_ACQ):
                                lk_val = ld_acquire_gpu_u32(
                                    gflag.iterator + (flag_head_base + chunk_base + fw)
                                )
                            else:
                                lk_val = ld_relaxed_gpu_u32(
                                    gflag.iterator + (flag_head_base + chunk_base + fw)
                                )
                cute.arch.sync_warp()

            gchunk = chunk_base + c
            token_base = token_start + c * BT
            factor_row = gchunk * BT
            stage16 = SMEM_INPUT // 2 + in_stage * (INPUT_STAGE_BYTES // 2)
            mbar = p_bar + BAR_IN_READY + in_stage
            if use_cp == 0:
                # Small-grid mode: all nine boxes by TMA.  On a small grid the
                # TMA path is not queued, and the cp.async split only costs
                # there: its shared-memory writes contend with the compute
                # warps' ldmatrix traffic in pass 1 and pass 2.  IN_READY was
                # initialised with one arrival in this mode, so the expect-tx
                # is its only arrival and the transaction bytes gate the
                # phase.
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, INPUT_STAGE_BYTES)
                    issue_factor_boxes(
                        p16, desc_factor, mbar, stage16, factor_row, heads, head
                    )
                    tma_load_3d(
                        p16 + (stage16 + STAGE_AQ // 2),
                        desc_aq,
                        mbar,
                        0,
                        gchunk,
                        head,
                    )
                    tma_load_3d(
                        p16 + (stage16 + STAGE_GT // 2),
                        desc_gt,
                        mbar,
                        0,
                        gchunk,
                        head,
                    )
                    tma_load_3d(
                        p16 + (stage16 + STAGE_V // 2),
                        desc_v,
                        mbar,
                        dv_half * DV_HALF,
                        token_base,
                        head,
                    )
            else:
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, INPUT_STAGE_TX_BYTES)
                    # Six factor boxes, 12,288 bytes on the transaction
                    # count.  Aq, GTotal and V go by ``cp.async`` below: on a
                    # full grid the TMA path queues collectively and is this
                    # kernel's pacer, while the LSU path sits idle beside it,
                    # so the 3 KiB that fits a plain 16-byte copy goes around
                    # the queue.
                    issue_factor_boxes(
                        p16, desc_factor, mbar, stage16, factor_row, heads, head
                    )
                # Aq: 512 B, linear in both images, one granule a lane.
                cp_async_16(
                    p16 + (stage16 + STAGE_AQ // 2 + lane * 8),
                    gaq.iterator
                    + ((head * total_chunks + gchunk) * (BT * BT) + lane * 8),
                    16,
                )
                # GTotal: 512 B of linear FP32, one granule a lane.
                cp_async_16(
                    p16 + (stage16 + STAGE_GT // 2 + lane * 8),
                    ggt.iterator + ((head * total_chunks + gchunk) * DK + lane * 4),
                    16,
                )
                # V: 2 KiB, four granules a lane.  The vo image is
                # 128B-swizzled and ``vo_idx`` carries the swizzle, exactly
                # as the store warp's read of it does; the global side reuses
                # the out tensor's index map because V and out share their
                # geometry.  Rows past the sequence end zero-fill through
                # src-size 0, which is the contract the TMA box honoured for
                # them.
                for rep in cutlass.range_constexpr(4):
                    task = lane + rep * 32
                    row = task // 8
                    col8 = (task - row * 8) * 8
                    srcb = cutlass.Int32(16)
                    if token_base + row >= token_end:
                        srcb = cutlass.Int32(0)
                    cp_async_16(
                        p16 + (stage16 + STAGE_V // 2 + vo_idx(row, col8)),
                        gv.iterator
                        + vo_global_index(token_base + row, head, heads, dv_half, col8),
                        srcb,
                    )
                cp_async_commit_group()
                # The stage's second IN_READY arrival, CP_ASYNC_ARRIVE_LAG
                # chunks behind its issue: the wait proves the lagged chunk's
                # cp.async share landed, the elected arrive publishes it, and
                # the lag keeps the copy latency out of this loop.  The
                # five-deep ring exists to spend on exactly this.
                if c >= CP_ASYNC_ARRIVE_LAG:
                    cp_async_wait_group(CP_ASYNC_ARRIVE_LAG)
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(
                            p_bar + BAR_IN_READY + input_stage(c - CP_ASYNC_ARRIVE_LAG)
                        )
        # Drain the arrivals the lag still owes (cp.async mode only; the
        # small-grid mode arrived in-loop).  ``idx`` can start negative when
        # the sequence is shorter than the lag, hence the guard.
        if use_cp != 0:
            cp_async_wait_group(0)
            cute.arch.sync_warp()
            if lane == 0:
                for k in cutlass.range_constexpr(CP_ASYNC_ARRIVE_LAG):
                    idx = num_chunks - CP_ASYNC_ARRIVE_LAG + k
                    if idx >= 0:
                        cute.arch.mbarrier_arrive(
                            p_bar + BAR_IN_READY + input_stage(idx)
                        )

    else:
        for c in range(num_chunks):
            out_stage = output_stage(c)
            p_out = p16 + ((SMEM_OUTPUT // 2) + out_stage * (OUTPUT_STAGE_BYTES // 2))
            token_base = token_start + c * BT
            valid_rows = token_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            cute.arch.mbarrier_wait(
                p_bar + BAR_OUT_READY + out_stage, output_ready_parity(c)
            )

            if valid_rows == BT:
                # The full path: the stage is released only after the store
                # has read it, not when it is merely committed.  A TMA store
                # rather than the warp's own vector stores: the latter add
                # 2 KiB of shared-memory reads per chunk that contend with the
                # compute warps' ldmatrix traffic, which costs on small grids
                # and gains nothing on full ones.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(desc_out, p_out, dv_half * DV_HALF, token_base, head)
                    tma_store_commit_group()
                    tma_store_wait_read(0)
                    cute.arch.mbarrier_arrive(p_bar + BAR_OUT_CONSUMED + out_stage)
            else:
                # Tail path: the whole warp stores 16-byte vectors, and no TMA,
                # fence, commit or wait-group is involved.
                for rep in cutlass.range_constexpr(4):
                    task = lane + rep * 32
                    if task < valid_rows * 8:
                        row = task // 8
                        vec = task - row * 8
                        frag = vec8_bf16(p_out, vo_idx(row, vec * 8))
                        store_vec8_bf16(
                            gout.iterator,
                            vo_global_index(
                                token_base + row, head, heads, dv_half, vec * 8
                            ),
                            frag,
                        )
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(p_bar + BAR_OUT_CONSUMED + out_stage)

    # --- final state handoff ----------------------------
    # All three roles converge here: the loads are issued, the last state update
    # is written, and every output store has completed its ``wait_group.read``.
    cute.arch.barrier()
    if cutlass.const_expr(HAS_STATE_OUT):
        if cutlass.const_expr(STATE_FP32):
            state_bf16_to_f32(p_state, p_state_f32, tidx)
            cute.arch.barrier()
            if warp_id == STORE_WARP:
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(
                        desc_state_out,
                        p_state_f32,
                        0,
                        dv_half * (STATE_F32_ROWS_PER_VALUE * DV_HALF),
                        state_plane,
                    )
                    tma_store_commit_group()
                    tma_store_wait_read(0)
        else:
            if warp_id == STORE_WARP:
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(
                        desc_state_out,
                        p_state,
                        0,
                        dv_half * (STATE_BF16_ROWS_PER_VALUE * DV_HALF),
                        state_plane,
                    )
                    tma_store_commit_group()
                    tma_store_wait_read(0)


#: PIPE=False launches never touch the flag slot, but the entry still
#: marshals a tensor there; one cached element per device does.
_DUMMY_FLAGS: dict = {}


def _dummy_flag(device):
    t = _DUMMY_FLAGS.get(device)
    if t is None:
        t = torch.zeros(4, dtype=torch.int32, device=device)
        _DUMMY_FLAGS[device] = t
    return t


# --------------------------------------------------------------------------
# One compiled host entry that launches both kernels
# --------------------------------------------------------------------------

# The DSL's AST preprocessor replays this module's module-level imports into
# the tracing scope when it compiles the jit below.  Both kernels and every
# helper they reach are either defined in this file or imported by name at
# the top of it (third-party modules, ``runtime`` and ``device_common``), so
# there is nothing else for it to resolve.


@cute.jit
def _fwd_entry(
    # --- prepare inputs ---
    gq: cute.Tensor,
    gk: cute.Tensor,
    gg: cute.Tensor,
    gbeta: cute.Tensor,
    ga_log_exp: cute.Tensor,
    gdt: cute.Tensor,
    # --- shared metadata: packed once, used by both ---
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    gchunk_to_seq: cute.Tensor,
    # --- workspace ---
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_ak: cute.Tensor,
    ws_aq: cute.Tensor,
    ws_gt: cute.Tensor,
    # --- recurrence output, and its cp.async-fed inputs ---
    gout: cute.Tensor,
    gv: cute.Tensor,
    # --- overlap-mode chunk flags (prepare releases, recurrence acquires) ---
    gflag: cute.Tensor,
    # --- descriptors; desc_factor is shared, prepare writes it and the
    #     recurrence reads it ---
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_factor: cutlass.Int64,
    desc_aq: cutlass.Int64,
    desc_gt: cutlass.Int64,
    desc_v: cutlass.Int64,
    desc_out: cutlass.Int64,
    desc_state_in: cutlass.Int64,
    desc_state_out: cutlass.Int64,
    # --- scalars ---
    scale: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    total_chunks: cutlass.Int32,
    heads: cutlass.Int32,
    rec_ctas: cutlass.Int32,
    prep_grid_x: cutlass.Int32,
    rec_grid_x: cutlass.Int32,
    rec_grid_y: cutlass.Int32,
    gen: cutlass.Int32,
    stream,
    stream_b,
    # --- specializations ---
    SAFE_GATE: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
    PIPE: cutlass.Constexpr,
    GATE_ACQ: cutlass.Constexpr,
    DEFER: cutlass.Constexpr,
):
    """Both launches, in order.

    Off/serial modes pass the same stream twice and ordering is the caller's
    stream: the recurrence reads the factors prepare writes, and same-stream
    launches are already ordered.  Dual mode passes a high-priority side
    stream as ``stream_b`` and the per-chunk flag ring carries the ordering
    instead; both launches still ride ONE compiled crossing.
    """
    if cutlass.const_expr(PIPE):
        pgx = heads
        pgy = prep_grid_x
    else:
        pgx = prep_grid_x
        pgy = heads

    prepare_kernel(
        gq,
        gk,
        gg,
        gbeta,
        ga_log_exp,
        gdt,
        gcu_seqlens,
        gcu_chunks,
        gchunk_to_seq,
        ws_kd,
        ws_qd,
        ws_ak,
        ws_aq,
        ws_gt,
        gflag,
        desc_q,
        desc_k,
        desc_g,
        desc_factor,
        scale,
        gate_scale_log2,
        total_chunks,
        heads,
        gen,
        SAFE_GATE,
        CPC,
        G_FP32,
        PIPE,
        DEFER,
    ).launch(
        # Overlap mode issues breadth-first across heads (see prepare_kernel).
        grid=(pgx, pgy, 1),
        block=(PREPARE_DEVICE_THREADS, 1, 1),
        stream=stream,
    )

    recurrence_kernel(
        gout,
        gv,
        ws_aq,
        ws_gt,
        gflag,
        gcu_seqlens,
        gcu_chunks,
        desc_factor,
        desc_aq,
        desc_gt,
        desc_v,
        desc_out,
        desc_state_in,
        desc_state_out,
        heads,
        total_chunks,
        rec_ctas,
        gen,
        HAS_STATE_IN,
        HAS_STATE_OUT,
        STATE_FP32,
        PIPE,
        GATE_ACQ,
    ).launch(
        grid=(rec_grid_x, rec_grid_y, 1),
        block=(REC_THREADS, 1, 1),
        smem=SMEM_DYNAMIC_BYTES,
        min_blocks_per_mp=MIN_BLOCKS_PER_MP,
        # Off/serial pass the same stream twice; dual passes the high-priority
        # side stream, so both launches ride ONE compiled crossing.
        stream=stream_b,
    )


def entry_kernel_name(key: tuple) -> str:
    """The persistent cache's specialization name for a combined-entry key.

    Every compile-time parameter and nothing else, spelled so that a directory
    listing answers "which specializations did this run build?".
    """
    (
        safe_gate,
        chunks_per_cta,
        g_fp32,
        state_in,
        state_out,
        state_fp32,
        pipe_on,
        gate_acq,
        defer_pub,
    ) = key
    parts = [
        "safegate" if safe_gate else "rawgate",
        f"cpc{int(chunks_per_cta)}",
        "gfp32" if g_fp32 else "gbf16",
        "si" if state_in else "nosi",
        "so" if state_out else "noso",
        "statefp32" if state_fp32 else "statebf16",
    ]
    if not pipe_on:
        parts.append("nopipe")
    else:
        parts.append("pipe")
        parts.append("acq" if gate_acq else "rlx")
        parts.append("defer" if defer_pub else "samechunk")
    return "decomp_" + "_".join(parts)


#: Keyed by CUDA device and specialization; heads and grids are runtime values.
_FWD_ENTRY_CACHE: dict = {}


def clear_fwd_cache() -> None:
    _FWD_ENTRY_CACHE.clear()


class FwdCall:
    """A packed argument tuple and the compiled entry that takes it.

    Everything in ``args`` is either a scalar fixed by the shape or a device
    tensor whose address the caller keeps alive, so a repeated call with the
    same buffers can skip the whole host path and go straight to ``run``.  The
    one thing that must not be frozen is ``a_log_exp``: its *buffer* is reused
    but its contents are recomputed per call, since ``A_log`` is a parameter an
    optimizer updates between steps.  ``run`` therefore refreshes it and only
    then launches; freezing it would make a captured graph replay stale
    values (see ``_A_LOG_EXP``).
    """

    __slots__ = (
        "args",
        "compiled",
        "a_log",
        "a_log_exp",
        "launch_lock",
        "gen_ref",
        "gen_slot",
        "dual_ctx",
        "_a_log_key",
        "_keepalive",
    )

    def __init__(
        self,
        args,
        compiled,
        a_log,
        a_log_exp,
        launch_lock,
        keepalive=(),
        gen_ref=None,
        gen_slot=None,
        dual_ctx=None,
    ):
        self.args = args
        # Dual mode: (caller torch stream, recurrence torch stream) plus two
        # reusable events created here.  prepare rides the caller stream, so
        # its ordering is implicit; only the recurrence side needs the pair.
        if dual_ctx is not None:
            caller, s_b = dual_ctx
            dual_ctx = (caller, s_b, torch.cuda.Event(), torch.cuda.Event())
        self.dual_ctx = dual_ctx
        # The overlap's flag generation: a [tensor, counter] pair shared by
        # every plan on the same flag buffer, bumped under the launch lock so
        # two plans never reuse a generation value.
        self.gen_ref = gen_ref
        self.gen_slot = gen_slot
        self.compiled = compiled
        self.a_log = a_log
        self.a_log_exp = a_log_exp
        self.launch_lock = launch_lock
        # This plan is pinned to one stream -- the ``PlanMemo`` keys on it, so
        # a plan is never replayed from another stream -- and to
        # one ``A_log``, which it holds a reference to.  Its ``_A_LOG_EXP`` key
        # is therefore a constant; recomputing it per call is driver and
        # ``os.getenv`` traffic that short shapes cannot hide (see
        # :func:`a_log_exp_for`).
        self._a_log_key = None if a_log is None else _a_log_exp_key(a_log)
        # The descriptor addresses in ``args`` are raw integers into buffers
        # owned by the descriptor caches.  Clearing one of those caches -- which
        # a test does deliberately, and eviction does on its own -- would free
        # the storage and leave this plan pointing at nothing.  Holding the
        # owners here keeps a cached plan self-sufficient.
        self._keepalive = keepalive

    def run(self) -> None:
        # The lock spans the refresh plus the whole compiled host entry.  The
        # latter returns only after prepare and recurrence have both been
        # submitted, so another host thread cannot enqueue a prepare between
        # this call's prepare and recurrence while sharing the workspace.
        with self.launch_lock:
            # Callers that pass a pre-refreshed buffer without its source
            # tensor still need the workspace lock; only the refresh is
            # conditional.
            if self.a_log is not None:
                a_log_exp_for(
                    self.a_log, LOG2_E, out=self.a_log_exp, key=self._a_log_key
                )
            if self.gen_ref is not None:
                gen = self.gen_ref[1] + 1
                reset = capturing()
                if gen > INT32_MAX:
                    # The consumers compare the flags against ``gen`` as INT32,
                    # so the counter restarts from a cleared buffer rather than
                    # wrapping.
                    gen = 1
                    reset = True
                self.gen_ref[1] = gen
                self.args[self.gen_slot] = cutlass.Int32(gen)
                if reset:
                    # A replay re-issues the captured ``gen`` as a constant, so
                    # the flags the previous replay left at that value would
                    # satisfy every consumer wait before prepare has published.
                    # Recording the reset here puts a memset node ahead of both
                    # kernels in the graph.  Eager calls skip it: the monotonic
                    # generation already distinguishes one call from the next.
                    self.gen_ref[0].zero_()
            if self.dual_ctx is None:
                self.compiled(*self.args)
            else:
                caller, s_b, ev_root, ev_b = self.dual_ctx
                ev_root.record(caller)
                s_b.wait_event(ev_root)
                self.compiled(*self.args)
                ev_b.record(s_b)
                caller.wait_event(ev_b)


def launch_fwd(
    *,
    q,
    k,
    g,
    beta,
    v,
    a_log_exp,
    dt_bias,
    cu_seqlens,
    cu_chunks,
    chunk_to_seq,
    workspace,
    out,
    prep_tmaps,
    rec_tmaps,
    scale,
    gate_scale_log2,
    total_chunks,
    heads,
    prep_grid_x,
    rec_grid_x,
    rec_grid_y,
    safe_gate,
    chunks_per_cta,
    g_fp32,
    has_state_in,
    has_state_out,
    state_fp32,
    a_log=None,
    build_only: bool = False,
):
    """Pack once, cross the boundary once, launch both.

    With ``build_only`` the packed :class:`FwdCall` is returned instead of being
    run, so :func:`_build_plan` can cache it and skip the host path on the next
    call with the same buffers.
    """
    # The tensors' device, not the current one: a plan for a ``cuda:1`` tensor
    # built while ``cuda:0`` is current must not bake in a ``cuda:0`` stream.
    caller = torch.cuda.current_stream(out.device)
    stream = cuda_driver.CUstream(caller.cuda_stream)

    mode = _PIPE_MODE
    if mode == "dual" and capturing():
        # The supported capture flow warms the plan eagerly first, so a plan
        # built here is already off the documented path.  Keep it single-stream
        # rather than have the side stream join a capture it was not warmed
        # on.  An eagerly built dual plan does replay inside a graph; see
        # ``FwdCall.run`` for the flag reset that makes that safe.
        mode = ""
    if mode and prep_grid_x > max_grid_dims(out.device)[1]:
        # PIPE swaps prepare's chunk-group axis into grid.y, whose limit is
        # smaller than grid.x's. Keep the ordinary layout when it would not fit.
        mode = ""
    mode, gate_acq, defer_pub = pipe_decisions(
        mode,
        heads=heads,
        total_chunks=total_chunks,
        rec_ctas=rec_grid_x * rec_grid_y,
        sm_count=_sm_count(out.device),
    )
    pipe_mode = mode
    pipe_on = pipe_mode in ("serial", "dual")
    flag_ent = (
        _acquire_flags(workspace, out.device)
        if pipe_on
        else [_dummy_flag(out.device), 0]
    )

    args = [
        flat_view(q),
        flat_view(k),
        flat_view(g),
        flat_view(beta),
        flat_view(a_log_exp),
        flat_view(dt_bias),
        flat_view(cu_seqlens),
        flat_view(cu_chunks),
        flat_view(chunk_to_seq),
        flat_view(workspace.kd),
        flat_view(workspace.qd),
        flat_view(workspace.ak),
        flat_view(workspace.aq),
        flat_view(workspace.g_total),
        flat_view(out),
        flat_view(v),
        flat_view(flag_ent[0]),
        cutlass.Int64(prep_tmaps.q),
        cutlass.Int64(prep_tmaps.k),
        cutlass.Int64(prep_tmaps.g),
        # One descriptor, shared: prepare writes the factor slab through it and
        # the recurrence reads through it.  The two encodings were already
        # identical -- (DK, rows, 3H), 128B swizzle -- so this is the reuse and
        # not a coincidence.
        cutlass.Int64(prep_tmaps.factor),
        # The recurrence keys its descriptors by role and returns 0 for a role
        # this launch does not use, which is how the optional state maps work.
        cutlass.Int64(rec_tmaps.address("aq")),
        cutlass.Int64(rec_tmaps.address("gt")),
        cutlass.Int64(rec_tmaps.address("v")),
        cutlass.Int64(rec_tmaps.address("out")),
        cutlass.Int64(rec_tmaps.address("state_in")),
        cutlass.Int64(rec_tmaps.address("state_out")),
        cutlass.Float32(scale),
        cutlass.Float32(gate_scale_log2),
        cutlass.Int32(total_chunks),
        cutlass.Int32(heads),
        cutlass.Int32(rec_grid_x * rec_grid_y),
        cutlass.Int32(prep_grid_x),
        cutlass.Int32(rec_grid_x),
        cutlass.Int32(rec_grid_y),
        cutlass.Int32(flag_ent[1]),
        stream,
        # Dual: the recurrence's high-priority side stream; otherwise the
        # caller stream again, which keeps one compiled entry per mode key.
        cuda_driver.CUstream(_rec_stream(out.device).cuda_stream)
        if pipe_mode == "dual"
        else stream,
    ]
    gen_slot = len(args) - 3
    key = (
        bool(safe_gate),
        int(chunks_per_cta),
        bool(g_fp32),
        bool(has_state_in),
        bool(has_state_out),
        bool(state_fp32),
        bool(pipe_on),
        bool(gate_acq),
        bool(defer_pub),
    )
    cache_key = (out.device.index, *key)
    compiled = _FWD_ENTRY_CACHE.get(cache_key)
    if compiled is None:
        # ``--enable-tvm-ffi`` is a *compile option*, not the
        # ``CUTE_DSL_ENABLE_TVM_FFI`` env var.  The env var makes the runtime
        # reject every argument that is not int/float/bool or does not expose
        # ``__tvm_ffi_object__``, and it never gets that far: the DSL casts an
        # argument to its annotated Numeric type before the check, so a native
        # int arrives as ``cutlass.Int64`` anyway.
        #
        # It is not optional here either way -- the persistent cache reloads
        # its artifacts with ``enable_tvm_ffi=True``.
        def _compile():
            # Subscript, not ``options=``: the keyword form silently drops
            # EnableTVMFFI and hands back a ctypes-marshalled callable.
            compiled = cute.compile[sm120a_compile_options()](_fwd_entry, *args, *key)
            return assert_tvm_ffi_dispatched(compiled, entry_kernel_name(key))

        compiled = build_kernel(
            entry_kernel_name(key),
            _compile,
            device=out.device,
            key_files=(__file__,),
        )
        _FWD_ENTRY_CACHE[cache_key] = compiled
    call = FwdCall(
        args,
        compiled,
        a_log,
        a_log_exp,
        workspace.launch_lock,
        keepalive=(prep_tmaps, rec_tmaps, workspace, a_log_exp),
        gen_ref=flag_ent if pipe_on else None,
        gen_slot=gen_slot,
        dual_ctx=(caller, _rec_stream(out.device)) if pipe_mode == "dual" else None,
    )
    if build_only:
        return call
    call.run()
    return None


#: ``FLASHINFER_KDA_PIPE`` resolves once at import and is off unless set.
#: ``"dual"`` overlaps prepare and recurrence on two streams.  Its consumer
#: CTAs spin on flags that a concurrently running kernel publishes, so forward
#: progress depends on prepare staying resident, which :func:`pipe_decisions`
#: guards with a heuristic rather than a hardware guarantee; that is why the
#: overlap is opt-in.  ``"serial"`` runs the flag machinery on one stream
#: without the overlap, isolating the flags' cost from the overlap's benefit.
#: Any other value is off.
_PIPE_MODE = os.environ.get("FLASHINFER_KDA_PIPE", "").strip().lower()
if _PIPE_MODE not in ("serial", "dual"):
    _PIPE_MODE = ""


#: SM counts of the CC 12.0 parts on which the acquire-flavoured lookahead
#: measured faster than the relaxed one, and only at ``rec_ctas >= 64`` on long
#: chains; extrapolating it to smaller grids lost and on larger parts the two
#: were level.  Keyed on SM count like ``AUTO_PROFILES``: a fit for a part, not
#: a property of the kernel.  Elsewhere a relaxed poll acquires through a
#: fence when consumed; the choice changes placement, never synchronization.
ACQUIRE_LOOKAHEAD_SM_COUNTS = frozenset({110})


def pipe_decisions(mode, *, heads, total_chunks, rec_ctas, sm_count, guard=True):
    """The overlap's predicate family, in one testable place.

    Returns ``(mode, gate_acq, defer_pub)`` with ``mode`` possibly downgraded
    to off.  Every boundary below is a fit to measurements, not a model, and
    should be re-measured with ``benchmarks/flashinfer_benchmark.py --routine
    recurrent_kda_prefill --backends flashinfer flashinfer-decomp
    flashinfer-fused`` when either kernel changes:

    * liveness: spinning recurrence CTAs inherit SMs as prepare CTAs retire
      and never give them back, so a recurrence grid that fills the device
      deadlocks; dual requires ``rec_ctas <= sm_count - 8``.
    * eligibility: dual pays only where there is something to hide --
      ``cps >= 256`` chains with ``4 <= rec_ctas <= 64`` (the chase regime)
      or ``32 <= cps <= 64`` with ``24 <= rec_ctas <= 64`` (the resident-slab
      regime).  The gap around ``cps ~ 128`` is where mid-length prepare
      stretches under SM partitioning by more than the recurrence can cover.
      Very short shapes lose to the fixed per-call cost; ``heads == 1`` has
      too little prepare to hide.

      Both regimes cap at ``rec_ctas`` 64 because larger recurrence grids
      measured as losses: the grid takes nearly every SM and prepare starves,
      the liveness failure in its milder form.
    * flavour and placement: long chains take the acquire lookahead (its
      issue stall paces the tight chase) and the one-chunk deferred
      publication (the store drain hides in the next input wait); short
      chains take a relaxed poll with an acquire fence at consumption and
      same-chunk publication (the earlier flag is worth more than the drain
      costs there).
    """
    cps = total_chunks // max(rec_ctas // (2 * heads), 1) if heads else 0
    if mode == "dual":
        live = rec_ctas <= sm_count - 8
        eligible = (cps >= 256 and 4 <= rec_ctas <= 64) or (
            32 <= cps <= 64 and 24 <= rec_ctas <= 64
        )
        if not live or (guard and not eligible):
            mode = ""
    pipe_on = mode in ("serial", "dual")
    gate_acq = (
        pipe_on
        and rec_ctas >= 64
        and cps >= 256
        and sm_count in ACQUIRE_LOOKAHEAD_SM_COUNTS
    )
    defer_pub = pipe_on and cps >= 256
    return mode, gate_acq, defer_pub


_SM_COUNT: dict = {}


def _sm_count(device):
    v = _SM_COUNT.get(device)
    if v is None:
        v = torch.cuda.get_device_properties(device).multi_processor_count
        _SM_COUNT[device] = v
    return v


def _acquire_flags(workspace: PrepareWorkspace, device) -> list:
    """The overlap's ``[flag_tensor, generation]`` pair for ``workspace``.

    One flag per (head, chunk).  Two live workspaces can run concurrently on
    two streams and must not share a generation counter, which is why the pair
    is the workspace's own; every plan on the workspace shares it and bumps
    the generation under the workspace's launch lock.
    """
    flags = workspace.pipe_flags
    if not flags:
        flags.extend(
            (
                torch.zeros(
                    workspace.total_chunks * workspace.heads,
                    dtype=torch.int32,
                    device=device,
                ),
                0,
            )
        )
    return flags


#: The recurrence's side stream, one per device, at high priority: at equal
#: priority the work distributor drains prepare's pending CTAs first (launch
#: order) and the recurrence only becomes resident in prepare's tail wave.
#: prepare rides the caller stream, which is both the liveness argument
#: (submitted first) and half the event traffic.
_REC_STREAMS: dict = {}
_REC_STREAMS_LOCK = threading.Lock()


def _rec_stream(device):
    """The side stream for ``device``, created on first use.

    Keyed on the tensors' device rather than the current one: a plan for a
    ``cuda:1`` input built while ``cuda:0`` is current must not launch its
    recurrence on a ``cuda:0`` stream.
    """
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    stream = _REC_STREAMS.get(index)
    if stream is None:
        with _REC_STREAMS_LOCK:
            stream = _REC_STREAMS.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=index, priority=-1)
                _REC_STREAMS[index] = stream
    return stream


# --------------------------------------------------------------------------
# The host path
#
# Everything between the backend ABI and the compiled entry -- the chunk
# tables, the factor arena, descriptor encoding, argument marshalling -- is a
# pure function of the tensors' addresses, shapes, dtypes and versions plus two
# floats.  A caller workspace additionally changes ownership and addresses of
# metadata and writable scratch, so both cache layers distinguish its stable
# identity.  In a serving loop none of those change between steps, which is why
# it is all cached: two layers, cheapest first, and a third that is not a cache
# at all but the caller's workspace.
#
# 1. ``PlanMemo.fast_path`` compares object identity against the previous call.
#    Same object means same address, shape, dtype, device and contiguity, so
#    only in-place mutation is left to check.
# 2. ``PlanMemo.get`` is an LRU on the full identity key, for callers that
#    alternate between a few buffer sets.
# 3. A :class:`~.runtime.SM120PrefillResources` supplied by the caller owns the
#    metadata, the arena and the descriptors for the lifetime of a CUDA graph.
#    Replay never re-enters Python, so nothing could renew an LRU position;
#    binding them to a workspace is what gives them a lifetime that outlives
#    this module's caches.
#
# The stream is in both keys.  A compiled entry bakes the ``CUstream`` it was
# built on into its argument tuple, so replaying a plan built on another stream
# would launch work correctly ordered against the wrong stream.  It is also
# what makes a capture -- which runs on its own stream -- miss and rebuild
# rather than replay the default stream's plan.
# --------------------------------------------------------------------------


#: This variant's call memo; see :class:`~.runtime.PlanMemo`.
_MEMO = PlanMemo()


def clear_caches() -> None:
    """Drop every cache this variant owns."""
    _MEMO.clear()
    clear_prepare_workspaces()
    clear_metadata_cache()
    clear_prepare_descriptor_cache()
    clear_recurrence_descriptor_cache()
    clear_launch_caches()
    clear_fwd_cache()


# --------------------------------------------------------------------------
# Chunk tables and the factor arena.
#
# Both are derived from the canonical offsets ``runtime`` already validated, so
# neither re-reads the device.  When the caller supplies a workspace they are
# built into it once, at eager warmup, and frozen: a captured graph records the
# addresses, and growing or reallocating either afterwards would leave replay
# reading a stale pointer.
# --------------------------------------------------------------------------


def chunk_tables(offsets, device, resources=None) -> ChunkMetadata:
    """``cu_chunks`` and ``chunk_to_seq`` for these offsets.

    Without a workspace this is the process-wide content-keyed cache, which is
    right for eager use.  With one, the tables live in the workspace at a
    fixed address for as long as it does.
    """
    host = list(offsets.host)
    per_sequence = chunks_for_lengths(offsets.lengths)
    cu = [0]
    for count in per_sequence:
        cu.append(cu[-1] + count)
    chunk_to_seq: list[int] = []
    for index, count in enumerate(per_sequence):
        chunk_to_seq.extend([index] * count)
    total_chunks = cu[-1]

    if resources is None:
        if capturing():
            # Without a workspace there is nowhere to put tables that a replay
            # can read, so this path would allocate -- and the error torch
            # raises for that ("Cannot copy between CPU and CUDA tensors during
            # CUDA graph capture") names the symptom rather than the cause.
            raise RuntimeError(
                "CUDA graph capture of this backend needs a caller-owned "
                "RecurrentKDAPrefillWorkspace, warmed eagerly on the capture "
                "stream with the same tensors; capturing without one would "
                "have to allocate the chunk tables inside the graph"
            )
        return _build_metadata(host, device)

    signature = (tuple(host), tuple(cu))
    cached = resources.chunk_signature_matches(signature)
    if cached is not None:
        return cached

    if capturing():
        raise RuntimeError(
            "CUDA graph capture cannot build this variant's chunk tables; warm "
            "the workspace with one eager call on the same offsets first"
        )

    cu_seqlens_i32 = resources.ensure_capacity("cu_seqlens_i32", len(host), torch.int32)
    cu_chunks_i32 = resources.ensure_capacity("cu_chunks_i32", len(cu), torch.int32)
    chunk_to_seq_i32 = resources.ensure_capacity(
        "chunk_to_seq_i32", max(len(chunk_to_seq), 1), torch.int32
    )[: len(chunk_to_seq)]
    cu_seqlens_i32.copy_(torch.tensor(host, dtype=torch.int32))
    cu_chunks_i32.copy_(torch.tensor(cu, dtype=torch.int32))
    if chunk_to_seq:
        chunk_to_seq_i32.copy_(torch.tensor(chunk_to_seq, dtype=torch.int32))

    meta = ChunkMetadata(
        cu_seqlens=cu_seqlens_i32,
        cu_chunks=cu_chunks_i32,
        chunk_to_seq=chunk_to_seq_i32,
        cu_seqlens_host=tuple(host),
        cu_chunks_host=tuple(cu),
        total_chunks=total_chunks,
        sequence_count=len(per_sequence),
    )
    resources.freeze_chunk_tables(signature, meta)
    return meta


def factor_arena(heads: int, total_chunks: int, device, resources=None):
    """The packed prepare workspace for this shape.

    ``acquire_prepare_workspace`` keys on ``(heads, total_chunks, stream)``
    because the arena is *written*, not read: two forwards on different streams
    must not share one, and a ``wait_event`` on the entry would order the
    reader against its creation rather than against the previous writer.  A
    caller-supplied workspace is already per-stream by construction, so it
    holds exactly one.
    """
    if resources is None:
        return acquire_prepare_workspace(heads, total_chunks, device)
    return resources.scratch_arena(
        (heads, total_chunks),
        lambda: allocate_prepare_workspace(heads, total_chunks, device),
    )


# --------------------------------------------------------------------------
# The entry point.
# --------------------------------------------------------------------------


def run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    info,
    offsets,
    resources=None,
    safe_gate: bool = True,
) -> Any:
    """Launch the decomposed prefill, writing ``out`` and ``final_state``.

    Returns the plan it ran, or :data:`STATE_ONLY_PLAN`, which the facade
    memoizes together with :func:`execute`.

    ``info`` and ``offsets`` come from :mod:`.runtime`: the facade validates
    and canonicalizes once, so this is not a second validation pass.  What is
    checked here is what *this schedule* depends on and the fused one does not
    -- the fused factor slab, the state geometry and the INT32 index ranges.

    ``safe_gate=False`` is refused rather than ignored: this variant does not
    implement the unbounded gate, and accepting the argument would report a
    numerical configuration the launch never used.
    """
    if not safe_gate:
        raise KDAPrefillValidationError(
            "the decomp variant does not support safe_gate=False; use the fused variant"
        )

    tensors = (
        q,
        k,
        v,
        g,
        beta,
        out,
        A_log,
        dt_bias,
        initial_state,
        final_state,
        cu_seqlens,
    )

    scalars = (float(scale), float(lower_bound))
    plan = _MEMO.fast_path(q.device, tensors, scalars, resources)
    if plan is None:
        key = _MEMO.identity(q.device, tensors, scalars, resources)
        # Everything from here to the launch is the miss path, and it is taken
        # under one lock: two threads building concurrently share a factor
        # arena, a descriptor cache and a DSL compiler session, and none of the
        # three tolerates it.
        with _MEMO.build_lock:
            plan = _MEMO.get(key)
            if plan is None:
                plan = _build_plan(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    out=out,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    lower_bound=lower_bound,
                    initial_state=initial_state,
                    final_state=final_state,
                    info=info,
                    offsets=offsets,
                    resources=resources,
                )
                _MEMO.remember_plan(key, plan)
            _MEMO.remember(q.device, tensors, scalars, resources, plan)

    execute(plan, initial_state, final_state)
    return plan


def _build_plan(
    *,
    q,
    k,
    v,
    g,
    beta,
    out,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    initial_state,
    final_state,
    info,
    offsets,
    resources,
):
    """The full host path: tables, arena, descriptors, argument marshalling."""
    device = q.device
    require_sm120a(device)

    if info.total_tokens == 0:
        return STATE_ONLY_PLAN

    # Fixed ``[B, T, ...]`` reshapes to packed ``[1, B * T, ...]`` as a view.
    # ``reshape`` on a contiguous tensor never copies, which matters: a copy of
    # ``out`` would silently drop the caller's writes.
    if info.input_mode == "fixed":
        pq, pk, pv, pg, pout, pbeta = (
            t.reshape(1, t.shape[0] * t.shape[1], *t.shape[2:])
            for t in (q, k, v, g, out, beta)
        )
    else:
        pq, pk, pv, pg, pout, pbeta = q, k, v, g, out, beta

    meta = chunk_tables(offsets, device, resources)
    if meta.total_chunks == 0:
        return STATE_ONLY_PLAN

    heads = info.heads
    total_tokens = pq.shape[1]
    check_recurrence_ranges(
        total_tokens=total_tokens,
        total_chunks=meta.total_chunks,
        sequences=meta.sequence_count,
        heads=heads,
        cu_seqlens_host=list(meta.cu_seqlens_host),
        cu_chunks_host=list(meta.cu_chunks_host),
        device=device,
    )

    workspace = factor_arena(heads, meta.total_chunks, device, resources)
    config = PrepareConfig(
        safe_gate=True,
        # Follow the measured per-architecture optimum.  The dataclass default
        # pins every device to the sm_120 value.
        chunks_per_cta=default_chunks_per_cta(device),
    )

    # Do both kernels' host-side work, then hand them to ONE compiled entry.
    # Two separate launches crossed the Python/compiled boundary twice per
    # forward and rebuilt their argument tuples each time, which is per-launch
    # rather than per-token and therefore flat in T.
    recurrence_plan = plan_recurrence(
        workspace=workspace,
        v=pv,
        out=pout,
        cu_seqlens_i32=meta.cu_seqlens,
        cu_chunks_i32=meta.cu_chunks,
        cu_seqlens_host=list(meta.cu_seqlens_host),
        cu_chunks_host=list(meta.cu_chunks_host),
        heads=heads,
        total_tokens=total_tokens,
        total_chunks=meta.total_chunks,
        initial_state=initial_state,
        final_state=final_state,
    )
    prep = prepare_launch_plan(
        q=pq,
        k=pk,
        g=pg,
        A_log=A_log,
        workspace=workspace,
        total_tokens=total_tokens,
        total_chunks=meta.total_chunks,
        heads=heads,
        config=config,
    )
    rec_grid = recurrence_grid(recurrence_plan.sequences, heads)
    call = launch_fwd(
        a_log=A_log,
        build_only=True,
        q=pq,
        k=pk,
        g=pg,
        beta=pbeta,
        v=pv,
        a_log_exp=prep.a_log_exp,
        dt_bias=dt_bias,
        cu_seqlens=meta.cu_seqlens,
        cu_chunks=meta.cu_chunks,
        chunk_to_seq=meta.chunk_to_seq,
        workspace=workspace,
        out=pout,
        prep_tmaps=prep.tensor_maps,
        rec_tmaps=recurrence_plan.tensor_maps,
        scale=float(scale),
        gate_scale_log2=float(lower_bound) * LOG2_E,
        total_chunks=meta.total_chunks,
        heads=heads,
        prep_grid_x=prep.grid_x,
        rec_grid_x=rec_grid[0],
        rec_grid_y=rec_grid[1],
        safe_gate=True,
        chunks_per_cta=config.chunks_per_cta,
        g_fp32=pg.dtype is torch.float32,
        has_state_in=recurrence_plan.has_state_in,
        has_state_out=recurrence_plan.has_state_out,
        state_fp32=recurrence_plan.state_dtype is torch.float32,
    )

    if resources is not None:
        # Replay never re-enters Python, so everything the capture recorded has
        # to stay alive at its captured address for the workspace's lifetime.
        resources.pin(
            call,
            call.compiled,
            prep.tensor_maps,
            recurrence_plan.tensor_maps,
            workspace,
            prep.a_log_exp,
            meta,
            offsets,
            offsets.canonical,
            offsets.source,
        )
    elif capturing():
        # A capture without a workspace has nowhere to put the pins, so they go
        # to the process-wide table.  Deliberately for the process lifetime:
        # an eviction that left a replayed graph reading a dangling device
        # pointer fails far from its cause and only sometimes.
        GRAPH_PINS.pin(
            (id(call), id(workspace)),
            call,
            call.compiled,
            prep.tensor_maps,
            recurrence_plan.tensor_maps,
            workspace,
            meta,
            offsets,
        )
    return call


__all__ = [
    "ChunkMetadata",
    "PrepareConfig",
    "PrepareWorkspace",
    "chunk_tables",
    "clear_caches",
    "default_chunks_per_cta",
    "factor_arena",
    "recurrence_grid",
    "execute",
    "run",
]
