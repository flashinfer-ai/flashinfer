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


"""SM120 KDA prefill, fused: one kernel does prepare and recurrence together.

A 512-thread CTA per (sequence, head), with the warp roles, SMEM arena and
barrier schedule the sections below define.  Where :mod:`.decomp` materializes
chunk factors to a workspace and reads them back, this variant keeps them in
shared memory and never leaves the kernel.

Everything this variant owns lives here: its CTA topology, its SMEM images and
swizzles, its inline PTX, its TMA descriptors, the device kernel, the compiled
entry, its descriptor and call-plan caches, and its current-stream launch.
Only the mechanisms it shares with :mod:`.decomp` come from :mod:`.runtime`.

The two variants are not folded together and do not import one another.  Their
layouts, PTX helpers and descriptor encodings diverged far enough that a shared
spelling would be a shared name over two different meanings -- the fused
``pairwise_a_ptr`` and the decomp ``pairwise_a_fragment_ptr`` address different
images -- and the plan is explicit that a same-named helper whose
specialization, rounding or barrier ownership differs stays in its variant.

The 512-thread CTA topology, the warp role ownership, the SMEM arena, the gate
numeric boundaries and the state/output alias contract are fixed implementation
choices shared by the host plan, device code and tests.
"""

import ctypes
import os
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .runtime import (
    GRAPH_PINS,
    SM120_CODE_TARGET,
    BoundedDeviceCache,
    KDAPrefillValidationError,
    assert_tvm_ffi_dispatched,
    build_kernel,
    capturing,
    check_flat_output_range,
    current_stream_ptr,
    flat_view,
    is_exact_alias,
    max_grid_dims,
    resource_cache_token,
    require_sm120a,
    sm120a_compile_options,
    tensor_identity,
    tensor_version,
)


# --------------------------------------------------------------------------
# Section 1: CTA topology, SMEM arena and barrier ids
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

#: The kernel guards on this exactly; other capabilities do not fall back here.
DEVICE_CC = (12, 0)

#: Production code target.  ``setmaxnreg`` requires an architecture-specific
#: Blackwell target, so plain ``sm_120`` cannot build a releasable kernel
#:.  This string is part of the compile cache key.
CODE_TARGET = "sm_120a"

# ---------------------------------------------------------------------------
# Problem geometry
# ---------------------------------------------------------------------------

BT = 16
DK = 128
DV = 128

# ---------------------------------------------------------------------------
# CTA shape and warp ownership
# ---------------------------------------------------------------------------

THREADS = 512
WARPS = 16

#: W0-W3 gate/norm/materialize, and produce Ak.T.
PREPARE_WARPS = 4
#: W4-W11, each permanently owning 16 value columns of the recurrent state.
RECURRENCE_WARPS = 8
RECURRENCE_WARP0 = 4
#: Value columns one recurrence warp owns; ``RECURRENCE_WARPS * 16 == DV``.
WARP_VALUES = DV // RECURRENCE_WARPS  # 16

#: Service warps.
TMA_WARP = 12
QK_WARP = 13
IO_WARP = 14
INV_WARP = 15

#: Key dimensions one prepare warp owns: ``[32 * p, 32 * p + 32)``.
PREPARE_WARP_DIMS = DK // PREPARE_WARPS  # 32

#: ``m16n8k16`` steps of one ``[16, 128] @ [128, *]`` reduction.
KEY_BLOCKS = DK // BT  # 8

# ---------------------------------------------------------------------------
# Warpgroup register budgets -- REQUESTED, NOT GRANTED
#
# These four numbers describe the redistribution the design asks for.  It does
# not currently happen -- but not for the reason recorded here earlier, and the
# real reason is addressable.
#
# The DSL is not at fault.  Dumped at all three levels, on 4.3 and 4.7 alike,
# the MLIR carries three ``setmaxregister`` ops and the PTX carries three
# ``setmaxnreg`` instructions with exactly the immediates below -- 120, 64, 160.
# They vanish between PTX and SASS, and ptxas says why:
#
#     (C7508) Potential Performance Loss: 'setmaxnreg' ignored;
#             unable to determine register count at entry.
#
# Not "unsupported on this target".  sm_120a *does* have the instruction; it is
# spelled ``USETMAXREG.DEALLOC.CTAPOOL`` / ``USETMAXREG.TRY_ALLOC.CTAPOOL``
# rather than Hopper's ``SETMAXNREG``, which is why an earlier search of this
# kernel's SASS for the Hopper mnemonic returned a confident zero.  Feed ptxas
# a minimal kernel with ``.maxnreg 128`` and both forms appear in the SASS.
#
# What our PTX lacks is that entry bound: it carries ``.reqntid`` from the block
# size and no ``.maxnreg``, so ptxas cannot compute the post-dec/inc budget and
# drops the request.  See the ``min_blocks_per_mp`` note below -- that is the
# one knob the DSL exposes that would supply it, and the measurement recorded
# there was taken while ``setmaxnreg`` was silently a no-op, so it does not yet
# say what the pair of them does together.
#
# Until the bound is supplied, every warp gets ``LAUNCH_MAXNREG`` and nothing
# else.  The consequence is not cosmetic: a recurrence warp holds 96 registers
# of state (h32's 64 plus h16's 32) inside 128, leaving ~32 to work in, and
# ptxas spills four of them once per chunk.  Those four instructions are 0.1% of
# the instruction count and 93.8% of the L1TEX sector requests, because local
# memory is per-thread and a warp's 32 lanes land on 32 separate sectors.
#
# Raising these constants alone does nothing -- three variants that
# redistributed the 1,024 registers the budget leaves free measured
# byte-identical spill, which is exactly what a dropped instruction predicts.
WG0_MAXNREG = 120  # W0-W3   prepare
WG1_MAXNREG = 160  # W4-W7   recurrence
WG2_MAXNREG = 160  # W8-W11  recurrence
WG3_MAXNREG = 64  # W12-W15 TMA, QK/Aq, I/O, KK/inverse

#: Registers the launch reserves per thread before redistribution.
LAUNCH_MAXNREG = 128

#: What the requested split would have cost, kept only so the test that pins
#: it against the pool keeps meaning something.  The *actual* allocation is
#: uniform: ``THREADS * LAUNCH_MAXNREG == 512 * 128 == 65,536``, the whole file.
CTA_REGISTER_BUDGET = (
    4 * WG0_MAXNREG + 4 * WG1_MAXNREG + 4 * WG2_MAXNREG + 4 * WG3_MAXNREG
) * 32

#: The allocation that actually happens, every warp alike.
CTA_REGISTERS_ACTUAL = THREADS * LAUNCH_MAXNREG
CTA_REGISTER_POOL = 65536

# ---------------------------------------------------------------------------
# SMEM arena
# ---------------------------------------------------------------------------

MAIN_SLOTS = 3
MAIN_SLOT_BYTES = 16384
V_STAGES = 3
V_STAGE_BYTES = 4096

MAIN_OFFSET = 0
V_STAGE_OFFSET = MAIN_SLOTS * MAIN_SLOT_BYTES  # 49152
CONTROL_OFFSET = V_STAGE_OFFSET + V_STAGES * V_STAGE_BYTES  # 61440
CONTROL_BYTES = 1024

DYNAMIC_SMEM_BYTES = CONTROL_OFFSET + CONTROL_BYTES  # 62464

#: SM120's per-CTA opt-in maximum.  The arena deliberately leaves 38,912 B
#: unused rather than adding a fourth main_slot.
SM120_SMEM_LIMIT = 101376

#: The design target and the acceptance result alike -- but deliberately *not*
#: passed to the launch as ``min_blocks_per_mp``.
#:
#: ``__launch_bounds__(512, 1)`` tells ptxas it may use at most 65,536 / 512 =
#: 128 registers per thread.  Measured on GB202: constraining it costs 6,457
#: local-memory instructions against 240 without, so it stays off.
#:
#: The original reasoning here was that the bound would defeat
#: ``setmaxnreg.inc``.  Read again, that has it backwards: an entry register
#: count is the *precondition* for ``setmaxnreg`` rather than an obstacle to it,
#: and this is the only knob the DSL offers that supplies one.
#:
#: That pairing has now been measured, and it settles the question against the
#: split.  With ``min_blocks_per_mp=1`` the PTX gains ``.minnctapersm``, ptxas
#: stops reporting C7508, and three ``USETMAXREG.*.CTAPOOL`` appear in the SASS:
#: the manual 120/64/160 split is honoured, exactly as designed.  It is also
#: 1.23x to 1.72x slower, median 1.62x, on seven shapes, bit-identical output,
#: 68/68 GPU tests passing.
#:
#: The SASS says why.  ``setmaxnreg`` moves a *runtime* hardware budget, but
#: ptxas assigns register *numbers* statically, once, for a kernel body that all
#: sixteen warps enter.  It must therefore satisfy the smallest warpgroup, and
#: WG3 asks for 64 -- so the whole kernel is compiled into 61 registers and
#: spills, 241 LDL and 148 STL against zero without the bound.  The recurrence
#: warps' extra registers exist at runtime and are unreachable, because nothing
#: was ever numbered above R61.
#:
#: This is the real obstacle, and it is structural rather than a missing flag:
#: warp-specialized register redistribution needs ptxas to compile per-role
#: register footprints, which one function entered by every role does not give
#: it.  The bound stays off.
#:
#: Worth noting separately: the shipped build does not spill at all.  DSL 4.7
#: compiles this kernel to R123 with zero LDL and zero STL, 4.3 to R125 with one
#: LDL and three STL.  The spill this note used to be about is gone.
#:
#: Dropping the bound costs no occupancy.  62,464 B of dynamic shared memory
#: already limits this kernel to one CTA per SM on its own, which NCU confirms
#: (``launch__occupancy_limit_shared_mem == 1``).
MIN_BLOCKS_PER_MP = 1

# ---------------------------------------------------------------------------
# Main main_slot phases, relative to ``16384 * main_slot``
# ---------------------------------------------------------------------------

SLOT_Q = 0  # Qraw -> Qd -> O
SLOT_K = 4096  # Kraw -> Kd
SLOT_G_LO = 8192  # raw G / E lower -> Ki -> Ak.T
SLOT_G_HI = 12288  # raw G / E upper -> factor records

#: Aliases naming the phase a consumer actually addresses.
SLOT_QD = SLOT_Q
SLOT_OUT = SLOT_Q
SLOT_KD = SLOT_K
SLOT_KI = SLOT_G_LO
SLOT_AKT = SLOT_G_LO

#: Factor records, written only once all four prepare warps have captured E.
SLOT_AINV_BETA = 12288  # [16, 16] BF16 SW32, 512 B
SLOT_AQ = 12800  # [16, 16] BF16 SW32, 512 B
SLOT_GTOTAL = 13312  # [128]    FP32,      512 B
SLOT_RESERVED = 13824  # production must not touch [13824, 16384)

# ---------------------------------------------------------------------------
# Control arena, relative to CONTROL_OFFSET
# ---------------------------------------------------------------------------

#: 128 B per main_slot: 16 FP32 q_inv then 16 FP32 k_inv.
SCRATCH_OFFSET = 0
SCRATCH_SLOT_BYTES = 128
SCRATCH_K_INV = 64

BARRIER_OFFSET = 384

#: ``(relative byte offset, arrival count)``; ``None`` marks a transaction
#: barrier, which is initialized with an arrival count of 1 and completed by
#: ``arrive_expect_tx``.  Every event but ``state_io`` has one object per main_slot.
BAR_QKG_READY = 384
BAR_V_READY = 408
BAR_MATERIALIZED = 432
BAR_QK_DONE = 456
BAR_AINV_READY = 480
BAR_AQ_READY = 504
BAR_AK_READY = 528
BAR_PROJECTION_DONE = 552
BAR_R_FORMED = 576
BAR_OUTPUT_READY = 600
BAR_OUTPUT_READ_DONE = 624
BAR_STATE_DONE = 648
BAR_STATE_IO = 672
BAR_RESERVED = 680

#: Arrival counts.  Transaction barriers take 1.
ARRIVALS_TX = 1
ARRIVALS_PREPARE = PREPARE_WARPS  # 4
ARRIVALS_RECURRENCE = RECURRENCE_WARPS  # 8
ARRIVALS_SINGLE = 1

#: ``(name, byte offset, arrivals, per-main_slot)`` .3 table order.
BARRIER_TABLE = (
    ("qkg_ready", BAR_QKG_READY, ARRIVALS_TX, True),
    ("v_ready", BAR_V_READY, ARRIVALS_TX, True),
    ("materialized", BAR_MATERIALIZED, ARRIVALS_PREPARE, True),
    ("qk_done", BAR_QK_DONE, ARRIVALS_SINGLE, True),
    ("ainv_ready", BAR_AINV_READY, ARRIVALS_SINGLE, True),
    ("aq_ready", BAR_AQ_READY, ARRIVALS_SINGLE, True),
    ("ak_ready", BAR_AK_READY, ARRIVALS_PREPARE, True),
    ("projection_done", BAR_PROJECTION_DONE, ARRIVALS_RECURRENCE, True),
    ("r_formed", BAR_R_FORMED, ARRIVALS_RECURRENCE, True),
    ("output_ready", BAR_OUTPUT_READY, ARRIVALS_RECURRENCE, True),
    ("output_read_done", BAR_OUTPUT_READ_DONE, ARRIVALS_SINGLE, True),
    ("state_done", BAR_STATE_DONE, ARRIVALS_RECURRENCE, True),
    ("state_io", BAR_STATE_IO, ARRIVALS_TX, False),
)

#: The prepare group's named barrier: ``bar.sync 1, 128``.
PREPARE_BARRIER_ID = 1
PREPARE_BARRIER_THREADS = PREPARE_WARPS * 32  # 128

# ---------------------------------------------------------------------------
# TMA transaction bytes
# ---------------------------------------------------------------------------

#: Q and K are two 4096-byte boxes each; G is 8192 B at FP32 and 4096 B at BF16.
QKG_TX_BYTES_G_FP32 = 4096 + 4096 + 8192  # 16384
QKG_TX_BYTES_G_BF16 = 4096 + 4096 + 4096  # 12288
V_TX_BYTES = 4096

#: BF16 boundary state is 2 x 16 KiB; FP32 is 4 x 16 KiB one window at a time.
STATE_BF16_TX_BYTES = 32768
STATE_F32_WINDOW_TX_BYTES = 16384
STATE_F32_WINDOWS = 4

# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------

LOG2_E = 1.4426950408889634
PREFIX_FLOOR = -126.0
NORM_FLOOR = 1.0e-24
SOFTPLUS_CUT = 20.0

#: ``lower_bound`` is validated against this closed interval, not compiled in.
LOWER_BOUND_RANGE = (-5.0, 0.0)

# ---------------------------------------------------------------------------
# Three-main_slot phase algebra
# ---------------------------------------------------------------------------


def main_slot(chunk: int) -> int:
    """Main main_slot and V stage index of chunk ``chunk``.

    Spelled without ``%`` so the identical expression evaluates for a Python
    ``int`` on the host and for a ``cutlass.Int32`` on the device.
    """
    return chunk - MAIN_SLOTS * (chunk // MAIN_SLOTS)


def generation(chunk: int) -> int:
    """How many times chunk ``chunk``'s main_slot has been used before it."""
    return chunk // MAIN_SLOTS


def ready_parity(chunk: int) -> int:
    """Phase a *consumer* waits on for chunk ``chunk``'s events.

    ``mbarrier_wait(bar, p)`` passes when the barrier's current phase differs
    from ``p``, so one wait per generation alternates 0, 1, 0, ...
    """
    return (chunk // MAIN_SLOTS) & 1


def reuse_parity(chunk: int) -> int:
    """Phase a *producer* waits on before overwriting chunk ``chunk``'s main_slot.

    Generation 0 yields 1, which passes immediately against a freshly
    initialized phase-0 barrier.  That is what lets W12 run one uniform loop
    from ``c = 0`` with no prologue special case.
    """
    return 1 ^ ((chunk // MAIN_SLOTS) & 1)


def chunks_for_seqlen(seqlen: int) -> int:
    return (seqlen + BT - 1) // BT


def grid(sequences: int, heads: int) -> tuple[int, int, int]:
    """One CTA per ``(sequence, head)``."""
    return (sequences, heads, 1)


# --------------------------------------------------------------------------
# Section 2: SMEM images and fragment maps
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S128 image parameters
# ---------------------------------------------------------------------------

#: A 128-byte segment is 64 BF16 or 32 FP32; a 16-byte group is 8 or 4.
BF16_SEGMENT_ELEMS = 64
BF16_GROUP_ELEMS = 8
BF16_SEGMENTS = DK // BF16_SEGMENT_ELEMS  # 2
BF16_SEGMENT_STRIDE = BT * BF16_SEGMENT_ELEMS  # 1024 elements

F32_SEGMENT_ELEMS = 32
F32_GROUP_ELEMS = 4
F32_SEGMENTS = DK // F32_SEGMENT_ELEMS  # 4
F32_SEGMENT_STRIDE = BT * F32_SEGMENT_ELEMS  # 512 elements

#: Constant coordinate permutations.  A constant XOR of a *coordinate* is not
#: expressible as a CuTe ``Swizzle`` (which only folds higher offset bits into
#: lower ones), so these stay explicit transforms outside the layout objects.
AK_TOKEN_XOR = BT // 2  # 8
PAIRWISE_COL_XOR = 8
PAIRWISE_ROW_STRIDE = BT

#: Rows of the state image one value spans, by element width.
STATE_BF16_ROWS_PER_VALUE = DK // BF16_SEGMENT_ELEMS  # 2
STATE_F32_ROWS_PER_VALUE = DK // F32_SEGMENT_ELEMS  # 4

#: Value columns one FP32 boundary window carries.
STATE_F32_WINDOW_VALUES = 32

SWIZZLE_S128_BYTES = (3, 4, 3)
SWIZZLE_SW32_BYTES = (1, 4, 3)


# ---------------------------------------------------------------------------
# Section 4.1-4.5: SMEM images
# ---------------------------------------------------------------------------


def raw_bf16_s128(row, dim):
    """BF16 element index of logical ``(row, dim)`` in a ``[16, 128]`` tile.

    The image behind Q, K, V, Qd, Kd, Ki and O.  Byte-unit
    swizzle ``Swizzle<3, 4, 3>``.
    """
    segment = dim // BF16_SEGMENT_ELEMS
    local = dim - segment * BF16_SEGMENT_ELEMS
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return (
        segment * BF16_SEGMENT_STRIDE
        + row * BF16_SEGMENT_ELEMS
        + (group ^ (row & 7)) * BF16_GROUP_ELEMS
        + inner
    )


def raw_f32_s128(row, dim):
    """FP32 element index of logical ``(row, dim)``.

    Raw ``G`` lands here.  ``E`` overwrites the same region but no longer at the
    same addresses -- see :func:`gate_e_f32` -- which is why the rendezvous
    between the gate's reads and its stores is now unconditional.
    """
    segment = dim // F32_SEGMENT_ELEMS
    local = dim - segment * F32_SEGMENT_ELEMS
    group = local // F32_GROUP_ELEMS
    inner = local - group * F32_GROUP_ELEMS
    return (
        segment * F32_SEGMENT_STRIDE
        + row * F32_SEGMENT_ELEMS
        + (group ^ (row & 7)) * F32_GROUP_ELEMS
        + inner
    )


def pairwise_sw32(row, col):
    """BF16 element index of a ``[16, 16]`` pairwise tile.

    Carries ``AinvBeta`` and ``Aq``.  The ``col ^ 8`` term is a coordinate
    permutation applied before the SW32 swizzle.
    """
    storage_col = col ^ PAIRWISE_COL_XOR
    byte0 = 2 * (PAIRWISE_ROW_STRIDE * row + storage_col)
    return (byte0 ^ (((byte0 >> 7) & 1) << 4)) // 2


def ak_t_s128(token, key):
    """BF16 element index of ``Ak.T[token, key]``.

    ``Ak`` is logically ``[K, token] = [128, 16]`` and is published transposed
    with a ``token ^ 8`` row permutation.  That permutation is what lets the
    recurrence recover the ``[key, token]`` A operand with a single
    ``ldmatrix.x4.trans`` and no register shuffle afterwards.
    """
    return raw_bf16_s128(token ^ AK_TOKEN_XOR, key)


def state_bf16_idx(v, k):
    """BF16 element index of external state ``[V, K]`` element ``(v, k)``.

    ``v`` is the CTA-global value in ``[0, 128)``: the two
    16 KiB halves are main slots 0 and 1, which are adjacent, so one index
    function spans both.  The unswizzled address is ``v * 128 + k`` -- physical
    ``[V, K]`` row-major and logical ``[K, V]`` column-major at once, which is
    what lets an external ``[V, K]`` state land by TMA with no transpose and
    still be read as ``H[K, V]``.
    """
    segment = k // BF16_SEGMENT_ELEMS
    local = k - segment * BF16_SEGMENT_ELEMS
    line = STATE_BF16_ROWS_PER_VALUE * v + segment
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return line * BF16_SEGMENT_ELEMS + (group ^ (line & 7)) * BF16_GROUP_ELEMS + inner


def state_f32_window_idx(v_local, k):
    """FP32 element index within one 32-value boundary window.

    ``v_local`` is in ``[0, 32)``; window ``w`` carries
    values ``[32 * w, 32 * w + 32)`` and occupies main slot 0 alone.
    """
    segment = k // F32_SEGMENT_ELEMS
    local = k - segment * F32_SEGMENT_ELEMS
    line = STATE_F32_ROWS_PER_VALUE * v_local + segment
    group = local // F32_GROUP_ELEMS
    inner = local - group * F32_GROUP_ELEMS
    return line * F32_SEGMENT_ELEMS + (group ^ (line & 7)) * F32_GROUP_ELEMS + inner


def gtotal_idx(k):
    """``GTotal`` is a contiguous FP32 ``[128]`` record with no swizzle."""
    return k


# ---------------------------------------------------------------------------
# Section 4.6: H register layout
# ---------------------------------------------------------------------------


def h32_idx(kb, nb, reg):
    """Flat index of FP32 master-state register ``(kb, nb, reg)``, 64 per lane."""
    return (kb * 2 + nb) * 4 + reg


def h16_idx(kb, nb, reg):
    """Flat index of packed BF16 state register ``(kb, nb, reg)``, 32 per lane."""
    return (kb * 2 + nb) * 2 + reg


def h32_coord(lane, kb, nb, reg, v_base):
    """Logical ``(k, v)`` of FP32 state register ``(kb, nb, reg)``.

    The native C map of an ``m16n8k16`` tile whose rows are keys and whose
    columns are the eight values at ``v_base + 8 * nb``.
    """
    g = lane >> 2
    q = lane & 3
    k = kb * BT + g + 8 * (reg >> 1)
    v = v_base + 8 * nb + 2 * q + (reg & 1)
    return (k, v)


def h16_pair_coords(lane, kb, nb, reg, v_base):
    """The two ``(k, v)`` coordinates packed into H16 register ``reg``.

    H16 register ``reg`` holds ``pack(c[2 * reg], c[2 * reg + 1])``, which is
    one key row and two adjacent values.
    """
    return (
        h32_coord(lane, kb, nb, 2 * reg, v_base),
        h32_coord(lane, kb, nb, 2 * reg + 1, v_base),
    )


def h_b_coords(lane, kb, nb, reg, v_base):
    """The two ``(k, v)`` coordinates of B register ``reg`` after ``movmatrix``.

    ``movmatrix.sync.aligned.m8n8.trans.b16`` on each of the
    two packed H16 registers is a bitwise C-to-B conversion, so no rounding
    happens here and the projection reads the state the MMA's B operand wants.
    """
    g = lane >> 2
    q = lane & 3
    k = kb * BT + 2 * q + 8 * reg
    v = v_base + 8 * nb + g
    return ((k, v), (k + 1, v))


# ---------------------------------------------------------------------------
# Transposed recurrence (H as the MMA A operand)
#
# The projection consumes H as a B operand, and a C-to-B conversion is a
# cross-lane transpose -- ``movmatrix`` per register, the largest single
# consumer of the LSU pipe.  C-to-A, by contrast, keeps the M axis and is a
# pure within-lane repack, which ``pack_a_bf16`` already does for free.
#
# Transposing the whole recurrence moves H into the A slot: the accumulator
# becomes ``(M = value, N = key)``, so ``pack`` alone lands the A fragment and
# the transpose disappears.  Every other operand becomes a B read from SMEM,
# which is a pointer-map change at identical instruction count.
# ---------------------------------------------------------------------------

#: Key columns one transposed accumulator tile carries; 16 tiles span DK.
HT_TILE_KEYS = 8
HT_TILES = DK // HT_TILE_KEYS  # 16


def h32t_idx(kt, reg):
    """Flat index of transposed FP32 state register ``(kt, reg)``, 64 per lane."""
    return kt * 4 + reg


def h16t_idx(kt, reg):
    """Flat index of transposed packed BF16 register ``(kt, reg)``, 32 per lane."""
    return kt * 2 + reg


def h32t_coord(lane, kt, reg, v_base):
    """Logical ``(k, v)`` of transposed FP32 state register ``(kt, reg)``.

    The native C map of an ``m16n8k16`` tile whose rows are *values* and whose
    columns are the eight keys at ``8 * kt`` -- the transpose of
    :func:`h32_coord`.
    """
    g = lane >> 2
    q = lane & 3
    v = v_base + g + 8 * (reg >> 1)
    k = HT_TILE_KEYS * kt + 2 * q + (reg & 1)
    return (k, v)


def h16t_pair_coords(lane, kt, reg, v_base):
    """The two ``(k, v)`` coordinates packed into transposed H16 register."""
    return (
        h32t_coord(lane, kt, 2 * reg, v_base),
        h32t_coord(lane, kt, 2 * reg + 1, v_base),
    )


def h_a_reg(j, r):
    """H16 register holding A-fragment register ``r`` of key block ``j``.

    The whole point of the transpose: this is a *selection*, not a conversion.
    Tiles ``2j`` and ``2j + 1`` supply K = 16, and their four packed registers
    already sit in A-fragment order, so the four are simply ``4j`` to ``4j + 3``.
    """
    return 4 * j + r


#: GTotal values one lane takes in the cooperative load, and the tiles they
#: then serve by shuffle.
GTOTAL_GROUP = DK // 32  # 4


def gtotal_group_key(lane, group):
    """Key whose GTotal ``lane`` holds after the cooperative load of ``group``."""
    return 32 * group + lane


def gtotal_shuffle_source(lane, kt, half):
    """Lane holding the GTotal that ``lane`` needs for tile ``kt``, half ``half``.

    The transposed accumulator puts key on the column axis, so a lane's pair is
    fixed by ``q = lane & 3`` and no pair is reused across tiles -- 32 values
    per lane where the un-transposed form needed 16, all of it redundant, since
    the warp still reads the same 128 unique values.  Loading them once
    cooperatively and shuffling costs four fully coalesced loads instead of
    sixteen scattered ones.

    The source *register* is warp-uniform by construction (each group is one
    register); only the source lane varies, which is what makes the exchange
    expressible at all.
    """
    return 8 * (kt - GTOTAL_GROUP * (kt // GTOTAL_GROUP)) + 2 * (lane & 3) + half


def ak_b_ptr(lane, j):
    """``Ak.T`` ``ldmatrix.x4.trans`` producing the ``[token, key]`` B operand.

    :func:`ak_a_ptr` delivers the A operand the un-transposed recurrence wants;
    the transposed one needs Ak as B, and the image is contiguous in key while
    a B register wants two adjacent *tokens*, hence ``.trans``.
    """
    r = lane >> 3
    token = lane - 8 * r
    return raw_bf16_s128((token + 8 * (r & 1)) ^ AK_TOKEN_XOR, BT * j + 8 * (r >> 1))


def state_x2t_ptr(lane, kt, value_base):
    """Boundary-state ``ldmatrix.x2`` / ``stmatrix.x2``, transposed recurrence.

    No ``.trans``, unlike :func:`state_x2_ptr`.  The image is physically
    ``[V, K]`` and the transposed C tile wants one value and two adjacent keys,
    which is the image's own contiguous direction -- the transpose that the
    un-transposed recurrence needed here disappears with it.
    """
    matrix = (lane // 8) & 1
    v = value_base + (lane - (lane // 8) * 8) + 8 * matrix
    return state_bf16_idx(v, HT_TILE_KEYS * kt)


def pairwise_b_ptr(lane):
    """``Aq.T`` / ``AinvBeta.T`` ``ldmatrix.x4`` producing the MMA B operand.

    No ``.trans``: a B register wants two adjacent ``k`` at one ``n``, which in
    ``Aq.T`` is two adjacent *columns* of ``Aq`` at one row, and the pairwise
    image is contiguous in column.  Only the addressed rows differ from
    :func:`pairwise_a_ptr`.
    """
    m = lane >> 3
    return pairwise_sw32((lane - 8 * m) + 8 * (m >> 1), 8 * (m & 1))


def vo_x2t_ptr(lane, value_base, nb):
    """V ``ldmatrix.x2.trans`` / O ``stmatrix.x2.trans``, transposed recurrence.

    Eight tokens each supply eight contiguous values; ``.trans`` turns that into
    rows of values and columns of tokens, which is the transposed C tile's
    shape.  ``nb`` selects the eight tokens.
    """
    matrix = (lane // 8) & 1
    row = lane - (lane // 8) * 8
    return raw_bf16_s128(HT_TILE_KEYS * nb + row, value_base + 8 * matrix)


# ---------------------------------------------------------------------------
# Section 9.1: native m16n8k16 fragment coordinates
# ---------------------------------------------------------------------------


def mma_a_coords(lane, reg):
    """Logical ``(row, k)`` of the two 16-bit halves in A register ``reg``."""
    g = lane >> 2
    q = lane & 3
    row = g + 8 * (reg & 1)
    k = 2 * q + 8 * (reg >> 1)
    return ((row, k), (row, k + 1))


def mma_b_coords(lane, reg):
    """Logical ``(k, n)`` of the two 16-bit halves in B register ``reg``."""
    g = lane >> 2
    q = lane & 3
    k = 2 * q + 8 * reg
    return ((k, g), (k + 1, g))


def mma_c_coord(lane, reg, n_base=0):
    """Logical ``(row, n)`` of FP32 accumulator register ``reg``."""
    g = lane >> 2
    q = lane & 3
    return (g + 8 * (reg >> 1), n_base + 2 * q + (reg & 1))


def mma_c16_coord(lane, slot):
    """``(row, col)`` of slot ``slot`` in a logical N=16 accumulator.

    Slots 0-3 are the low N=8 half and 4-7 the high half, matching the order
    two native MMAs write them.
    """
    nb = slot // 4
    return mma_c_coord(lane, slot - nb * 4, 8 * nb)


# ---------------------------------------------------------------------------
# Section 9.2: SMEM pointer maps
#
# ``factor_a_ptr`` and ``ki_b_ptr`` are NOT the same lane map.  The A map takes
# its row half from ``m & 1`` and its column half from ``m >> 1``; the B map
# takes the row half from lane bit 4 and the column half from lane bit 3.  Using
# the A map with a ``.trans`` modifier in place of the B map silently
# transposes the operand, which is why the two are separate functions and why
# the plan pins an exhaustive probe on the difference.
# ---------------------------------------------------------------------------


def factor_a_ptr(lane, kb):
    """Qd/Kd ``ldmatrix.x4`` A operand; also the matching ``stmatrix.x4``."""
    m = lane // 8
    row = (lane - m * 8) + 8 * (m & 1)
    col = BT * kb + 8 * (m >> 1)
    return raw_bf16_s128(row, col)


def ki_b_ptr(lane, kb):
    """Ki ``ldmatrix.x4`` (no ``.trans``) producing the MMA B operand."""
    m = lane // 8
    row = (lane - m * 8) + 8 * (lane // 16)
    col = BT * kb + 8 * (m & 1)
    return raw_bf16_s128(row, col)


def pairwise_a_ptr(lane):
    """AinvBeta / Aq ``ldmatrix.x4`` A operand."""
    m = lane // 8
    row = (lane - m * 8) + 8 * (m & 1)
    col = 8 * (m >> 1)
    return pairwise_sw32(row, col)


#: ``stmatrix.x4`` of a pairwise tile addresses the same 16 rows as the load.
pairwise_store_ptr = pairwise_a_ptr


def ak_a_ptr(lane, kb):
    """Ak.T ``ldmatrix.x4.trans`` producing the ``[key, token]`` A operand."""
    m = lane // 8
    logical_t = 8 * (m >> 1) + (lane - m * 8)
    key = BT * kb + 8 * (m & 1)
    return raw_bf16_s128(logical_t ^ AK_TOKEN_XOR, key)


def ak_store_ptr(lane, key_base):
    """``stmatrix.x4`` of one ``[16, 16]`` Ak.T tile at ``key_base``."""
    m = lane // 8
    token = (lane - m * 8) + 8 * (m & 1)
    key = key_base + 8 * (m >> 1)
    return ak_t_s128(token, key)


def vo_x2_ptr(lane, value_base):
    """V ``ldmatrix.x2`` / O ``stmatrix.x2`` over a ``[16, 8]`` value block.

    Non-transposed in both directions: the stage is token-major and the C
    tile's rows are tokens, so the load and the store share this map.
    """
    matrix = (lane // 8) & 1
    token = (lane - (lane // 8) * 8) + 8 * matrix
    return raw_bf16_s128(token, value_base)


def state_x2_ptr(lane, kb, value_base):
    """Boundary-state ``ldmatrix.x2.trans`` / ``stmatrix.x2.trans`` map.

    The state image is physically ``[V, K]``; ``.trans`` turns those 16 rows
    into the ``[K, V]`` C tile the registers hold, so the prologue load and the
    epilogue store use one map and differ only in instruction direction.  Lanes
    16-31 are ignored by an x2 copy.
    """
    matrix = (lane // 8) & 1
    v = value_base + (lane - (lane // 8) * 8)
    key = BT * kb + 8 * matrix
    return state_bf16_idx(v, key)


def state_f32_window_reg_coord(lane, kb, nb, reg, local_v_base):
    """``(v_local, k)`` an FP32 boundary window read/writes for H32 ``reg``.

    the FP32 boundary has no matrix-copy path, so
    each lane addresses its four accumulator elements individually through
    :func:`state_f32_window_idx`.
    """
    g = lane >> 2
    q = lane & 3
    k = BT * kb + g + 8 * (reg >> 1)
    v_local = local_v_base + 8 * nb + 2 * q + (reg & 1)
    return (v_local, k)


# ---------------------------------------------------------------------------
# Section 3.5: prepare-warp ownership
# ---------------------------------------------------------------------------


def gate_owner_dim(warp, lane):
    """Key dimension whose 16 gate values and ``GTotal`` lane ``lane`` owns."""
    return 32 * warp + lane


def norm_row(warp, lane):
    """Token row lane ``lane`` of prepare warp ``warp`` helps normalize."""
    return 4 * warp + (lane // 8)


def norm_dims(lane):
    """The 16 feature indices a norm lane squares, in accumulation order.

    The order is part of the contract: the FP32 sum is not reassociated, so a
    different traversal is a different number.
    """
    lane_in_row = lane & 7
    return [8 * lane_in_row + i for i in range(8)] + [
        64 + 8 * lane_in_row + i for i in range(8)
    ]


def materialize_coords(warp, lane, j, nb, r):
    """``(t0, t1, key)`` of materializer B register ``2 * nb + r``.

    The materializer does *not* keep the gate stage's
    "lane ``l`` owns dimension ``32p + l``" mapping: it forms the MMA **B**
    fragment of Qd/Kd/Ki/Kr directly, so its lane coordinates are the B map's.
    The four ``(nb, r)`` combinations cover one ``[16, 16]`` tile exactly once,
    and ``j`` selects which of the warp's two key tiles.
    """
    g = lane >> 2
    q = lane & 3
    t0 = 2 * q + 8 * r
    key = 32 * warp + BT * j + 8 * nb + g
    return (t0, t0 + 1, key)


#: Elements of the E image one vector access covers, and the groups per row.
E_GROUP_ELEMS = 4
E_GROUPS = BT // E_GROUP_ELEMS  # 4


def gate_e_group_xor(key):
    """Token-group permutation of ``key`` in the E image.

    Two *separated* key bit pairs, not one.  The store's phase holds eight
    consecutive keys and the load's holds four keys eight apart, and a
    permutation driven by ``key & 3`` alone leaves whichever of the two it does
    not address two-way conflicted.  Folding ``key >> 2`` in as well is what
    makes both conflict-free at once.
    """
    return (key ^ (key >> 2)) & 3


def gate_e_f32(key, token):
    """FP32 element index of E at ``(key, token)`` in a ``[128, 16]`` tile.

    E is *transposed* relative to the raw G image it overwrites: raw G arrives
    from TMA as ``[token, key]`` because that is what TMA lands, but both sides
    of E's exchange walk along ``token`` -- the gate owns one key and writes
    sixteen tokens, the materializer wants two adjacent tokens at one key.  A
    ``[key, token]`` image makes each side contiguous, turning sixteen scalar
    stores into four vector ones and sixteen scalar loads into eight pair
    loads.

    The permutation above is a constant XOR of a *coordinate*, like
    :data:`AK_TOKEN_XOR`, not a CuTe ``Swizzle``.  ``token``'s low two bits are
    left alone so a token pair ``(t, t + 1)`` with even ``t`` stays adjacent,
    which is what the materializer's pair load requires.

    Total size is ``128 * 16 * 4 = 8192`` bytes, exactly ``SLOT_G_LO``: the
    conflict-free form needs no padding.
    """
    group = token // E_GROUP_ELEMS
    inner = token - group * E_GROUP_ELEMS
    return BT * key + E_GROUP_ELEMS * (group ^ gate_e_group_xor(key)) + inner


def materialize_b_ptr(warp, lane, j):
    """Raw Q/K ``ldmatrix.x4.trans`` delivering :func:`materialize_coords`.

    The materializer's four registers are the native B fragment of the warp's
    ``[16, 16]`` key tile ``j``: register ``2 * nb + r`` holds tokens
    ``2 * (lane & 3) + 8 * r`` and ``+1`` at key ``8 * nb + (lane >> 2)``.  A
    B fragment over a ``[token, key]`` image is what ``.trans`` produces, so
    the whole tile is one instruction per factor rather than eight scalar
    loads per register.

    The eight elements each addressed row contributes are keys ``base`` to
    ``base + 7`` of one token, and :func:`raw_bf16_s128` is contiguous in
    ``dim`` inside an eight-aligned group, so the S128 swizzle is absorbed
    entirely into the row address -- unlike :func:`ak_a_ptr`, no coordinate
    XOR is needed.
    """
    reg = lane >> 3
    row = lane - 8 * reg
    nb = reg >> 1
    r = reg - 2 * nb
    return raw_bf16_s128(8 * r + row, 32 * warp + BT * j + 8 * nb)


def kr_b_coords(warp, lane, j, nb, r):
    """The two ``(token, key)`` coordinates of Kr B register ``2 * nb + r``.

    Identical to :func:`materialize_coords`; named separately because plan
    Section 9.3 pins the Kr B-fragment mapping as its own contract -- Kr is the
    one factor that is never published to SMEM, so this map is the only
    definition of where its values live.
    """
    t0, t1, key = materialize_coords(warp, lane, j, nb, r)
    return ((t0, key), (t1, key))


# ---------------------------------------------------------------------------
# Section 9.3: fixed register permutations
# ---------------------------------------------------------------------------

#: A-layout -> B-layout of the same matrix: transpose each 8x8 quadrant.
MOVMATRIX_A_TO_B = (0, 1, 2, 3)

#: A-layout -> A-layout of the transpose: the off-diagonal quadrants exchange.
MOVMATRIX_A_TO_AT = (0, 2, 1, 3)

#: C-layout -> B-layout of a packed 16x8 tile (H16 and the residual R).
MOVMATRIX_C_TO_B = (0, 1)

#: Quadrant order of the packed A fragment the inverse chain works in
#:: top-left, bottom-left, top-right, bottom-right.
INVERSE_FRAG_QUADRANTS = ("TL", "BL", "TR", "BR")


# ---------------------------------------------------------------------------
# Global element indices
# ---------------------------------------------------------------------------


def out_global_index(token, head, heads, dim):
    """Flat element index of ``out[0, token, head, dim]``.

    Only the tail output path needs this; a full chunk is stored by TMA, which
    addresses the same elements through its descriptor.
    """
    return (token * heads + head) * DV + dim


def beta_global_index(token, head, heads):
    """Flat element index of ``beta[0, token, head]``."""
    return token * heads + head


# ---------------------------------------------------------------------------
# CuTe layout objects (TMA descriptors only)
# ---------------------------------------------------------------------------


def make_cute_layouts():
    """Build the three composed layouts the images above describe.

    Imported lazily so host-only consumers -- the layout tests, the reference
    and the validation matrix -- do not need the CUTLASS DSL installed.
    """
    import cutlass.cute as cute

    raw_bf16 = cute.make_composed_layout(
        cute.make_swizzle(*SWIZZLE_S128_BYTES),
        0,
        cute.make_layout(
            (BT, (BF16_SEGMENT_ELEMS, BF16_SEGMENTS)),
            stride=(BF16_SEGMENT_ELEMS, (1, BF16_SEGMENT_STRIDE)),
        ),
    )
    raw_f32 = cute.make_composed_layout(
        cute.make_swizzle(*SWIZZLE_S128_BYTES),
        0,
        cute.make_layout(
            (BT, (F32_SEGMENT_ELEMS, F32_SEGMENTS)),
            stride=(F32_SEGMENT_ELEMS, (1, F32_SEGMENT_STRIDE)),
        ),
    )
    pairwise = cute.make_composed_layout(
        cute.make_swizzle(*SWIZZLE_SW32_BYTES),
        0,
        cute.make_layout((BT, BT), stride=(PAIRWISE_ROW_STRIDE, 1)),
    )
    return raw_bf16, raw_f32, pairwise


# --------------------------------------------------------------------------
# Section 3: inline PTX the fused kernel issues
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Matrix copies
# ---------------------------------------------------------------------------


def _ldmatrix(count: str, trans: str, smem_ptr, num: int, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    outs = ", ".join(f"${i}" for i in range(num))
    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.IntegerType.get_signless(32)] * num),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 {{{outs}}}, [${num}];",
        ",".join(["=r"] * num) + ",r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(
                _T.IntegerType.get_signless(32), struct, [i], loc=loc, ip=ip
            )
        )
        for i in range(num)
    )


@dsl_user_op
def ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` -> four b32.

    Qd/Kd A operands, the Ki B operand and the pairwise tiles.  Which of those
    it produces is entirely a property of the pointer map it is given.
    """
    return _ldmatrix(".x4", "", smem_ptr, 4, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix...x4.trans``: Ak.T -> the ``[key, token]`` Ak A operand."""
    return _ldmatrix(".x4", ".trans", smem_ptr, 4, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix...x2``: one ``[16, 8]`` V tile.  Only lanes 0-15 address."""
    return _ldmatrix(".x2", "", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix...x2.trans``: the boundary state's C-layout view.

    The physical state image is ``[V, K]``; reading down its columns is what
    turns it into the ``[K, V]`` accumulator layout H32/H16 are held in.
    """
    return _ldmatrix(".x2", ".trans", smem_ptr, 2, loc=loc, ip=ip)


def _stmatrix(count: str, trans: str, smem_ptr, regs, *, loc=None, ip=None):
    ins = ", ".join(f"${i + 1}" for i in range(len(regs)))
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            *[cutlass.Int32(r).ir_value(loc=loc, ip=ip) for r in regs],
        ],
        f"stmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 [$0], {{{ins}}};",
        ",".join(["r"] * (len(regs) + 1)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmatrix_x4(smem_ptr, r0, r1, r2, r3, *, loc=None, ip=None):
    """``stmatrix...x4``: Qd/Kd/Ki publication, AinvBeta, Aq and Ak.T."""
    _stmatrix(".x4", "", smem_ptr, (r0, r1, r2, r3), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x4_trans(smem_ptr, r0, r1, r2, r3, *, loc=None, ip=None):
    """``stmatrix...x4.trans``: publish a B-layout fragment to the A image.

    The transpose the store performs is per 8x8 tile with the register order
    unchanged, which is exactly :func:`~...kernel.a_to_b`'s
    ``MOVMATRIX_A_TO_B = (0, 1, 2, 3)``.  Using this instead costs four
    ``movmatrix`` less per factor and lands the same image; the equality is
    checked against the host store model in the layout tests.
    """
    _stmatrix(".x4", ".trans", smem_ptr, (r0, r1, r2, r3), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x2(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix...x2``: the output tile."""
    _stmatrix(".x2", "", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x2_trans(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix...x2.trans``: exactly inverts :func:`ldmatrix_x2_trans`."""
    _stmatrix(".x2", ".trans", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def fresh_b32(value, *, loc=None, ip=None):
    """A byte-identity ``prmt.b32``: same value, a *different* register.

    This computes nothing.  It exists to keep the MMA's operand register
    distinct from the persistent H16 state register, and removing it is a
    measured 4% regression, not a cleanup.

    The transposed recurrence hands H16 straight to the MMA as its A operand.
    Without an intervening value, ptxas must satisfy the MMA's register
    constraints on registers that are live across the entire kernel, and with
    96 of a recurrence warp's 160 registers already pinned to h32/h16 it
    resolves that by spilling -- measured, ``LDL`` rises 6.3x and the kernel
    slows by 4.0%.  Standing this instruction in the same place keeps spill
    unchanged and gains 4.2%.

    ``prmt`` rather than ``mov`` because ptxas folds an identity move; the
    byte-permute survives, which the opcode counts confirm.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "prmt.b32 $0, $1, $1, 0x3210;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def movmatrix_b16(value, *, loc=None, ip=None):
    """``movmatrix.sync.aligned.m8n8.trans.b16``: transpose one packed 8x8.

    Bit-exact: this moves 16-bit lanes between registers and never rounds, which
    is what makes the H16 C-to-B conversion free of a second rounding boundary.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Tensor Core
# ---------------------------------------------------------------------------


def _mma_m16n8k16(
    kind: str, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None
):
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.F32Type.get()] * 4),
        [
            cutlass.Int32(a0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a3).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c3).ir_value(loc=loc, ip=ip),
        ],
        f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(4)
    )


@dsl_user_op
def mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``: every BF16 MMA."""
    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``...f32.f16.f16.f32``: the blockwise inverse's twelve MMAs, and only those."""
    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Packed conversion and arithmetic
# ---------------------------------------------------------------------------


def _pack(instr: str, lo, hi, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            f"{instr} $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """``cvt.rn.bf16x2.f32``; ``lo`` lands in the low half."""
    return _pack("cvt.rn.bf16x2.f32", lo, hi, loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(lo, hi, *, loc=None, ip=None):
    """``cvt.rn.f16x2.f32``; ``lo`` lands in the low half.

    The inverse chain's pack.  it must not go through BF16
    first -- the three extra significand bits are the point.
    """
    return _pack("cvt.rn.f16x2.f32", lo, hi, loc=loc, ip=ip)


def _binary_packed(instr: str, a, b, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            f"{instr} $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_bf16x2(a, b, *, loc=None, ip=None):
    """``mul.rn.bf16x2``: two BF16 products, each rounded once."""
    return _binary_packed("mul.rn.bf16x2", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_bf16x2(a, b, *, loc=None, ip=None):
    """``sub.rn.bf16x2``: the residual ``BF16(V - X)`` for two tokens at once."""
    return _binary_packed("sub.rn.bf16x2", a, b, loc=loc, ip=ip)


@dsl_user_op
def unpack_bf16x2(value, *, loc=None, ip=None):
    """Widen a packed BF16 pair to two FP32, low half first."""
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.F32Type.get()] * 2),
        [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi;"
        "  mov.b32 {lo, hi}, $2;"
        "  cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(2)
    )


@dsl_user_op
def rcp_approx_ftz(x, *, loc=None, ip=None):
    """``rcp.approx.ftz.f32``: used for ``1 / E`` when forming Ki, and nowhere else.

    ``.ftz`` flushes subnormal *inputs* to zero, so ``E < 2^-126`` would return
    ``+Inf`` and ``0 * Inf`` would poison a whole KK row.  The
    :data:`PREFIX_FLOOR` clamp on the gate
    prefix is what keeps ``E`` at or above the smallest normal, which is why
    that clamp is not optional.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        llvm.inline_asm(
            _T.F32Type.get(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Barriers
# ---------------------------------------------------------------------------


@dsl_user_op
def mbarrier_test_wait_parity(mbar_ptr, phase, *, loc=None, ip=None):
    """``mbarrier.test_wait.parity.acquire.cta``: a single non-blocking probe.

    the design allows this in the first-ready loop and forbids
    ``try_wait``, which may suspend the warp for up to 10,000,000 ns.  The
    result only *selects* a branch; the chosen branch still performs the real
    32-lane acquire, because a broadcast predicate gives the other 31 lanes no
    visibility of the producer's shared-memory stores.

    Branch-free on purpose: an inline-asm label would have to be unique per
    expansion, and ``selp`` costs less than getting that wrong.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Boolean(
        cutlass.Int32(
            llvm.inline_asm(
                _T.IntegerType.get_signless(32),
                [
                    mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
                    cutlass.Int32(phase).ir_value(loc=loc, ip=ip),
                ],
                "{ .reg .pred p;"
                "  mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
                " p, [$1], $2;"
                "  selp.b32 $0, 1, 0, p; }",
                "=r,r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )
        != cutlass.Int32(0)
    )


# ---------------------------------------------------------------------------
# Register redistribution
# ---------------------------------------------------------------------------


#: DSL 4.7 renamed these and deprecated the old spellings; 4.3 only has the old
#: ones.  Resolve once at import so the tree builds on both without emitting a
#: DeprecationWarning per traced call site under 4.7.  Both spellings lower to
#: the same op -- which on ``sm_120a`` is dropped entirely, see
#: Section 1.
_REG_DEC = (
    getattr(cute.arch, "setmaxregister_decrease", None)
    or cute.arch.warpgroup_reg_dealloc
)
_REG_INC = (
    getattr(cute.arch, "setmaxregister_increase", None) or cute.arch.warpgroup_reg_alloc
)


@dsl_user_op
def setmaxnreg_dec(count: int, *, loc=None, ip=None):
    """``setmaxnreg.dec.sync.aligned.u32``.

    A warpgroup instruction: every warp of the four must reach this with the
    same immediate.  The caller is responsible for that -- nothing here can
    check it, and a per-role call is silently wrong rather than an error.
    """
    _REG_DEC(count, loc=loc, ip=ip)


@dsl_user_op
def setmaxnreg_inc(count: int, *, loc=None, ip=None):
    """``setmaxnreg.inc.sync.aligned.u32``; see :func:`setmaxnreg_dec`."""
    _REG_INC(count, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# TMA
# ---------------------------------------------------------------------------


@dsl_user_op
def fence_tensormap_acquire(desc_addr, *, loc=None, ip=None):
    """``fence.proxy.tensormap::generic.acquire.gpu [desc], 128``.

    The descriptor is written by the host and read by the TMA unit through a
    different proxy, so the kernel must acquire it before first use.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip)],
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_3d(smem_ptr, desc_addr, mbar_ptr, c0, c1, c2, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.3d`` box, global to shared.

    ``shared::cta``, not ``shared::cluster``: SM120 has no thread block
    clusters.  Completion is reported to ``mbar_ptr`` as transaction bytes, so
    the consumer waits on the mbarrier rather than on a commit group.
    """
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5}], [$2];",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_3d(desc_addr, smem_ptr, c0, c1, c2, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.3d`` box, shared to global.

    Stores carry no mbarrier; they are tracked with bulk commit groups, and
    :func:`tma_store_wait_read` is what releases the source SMEM.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [$0, {$2, $3, $4}], [$1];",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_commit_group(*, loc=None, ip=None):
    """``cp.async.bulk.commit_group``."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_wait_read(keep: int, *, loc=None, ip=None):
    """``cp.async.bulk.wait_group.read``: a source-reuse guarantee.

    It does not claim the store is globally visible, only that the TMA engine
    has finished *reading* the SMEM it came from -- which is exactly the
    condition for recycling the slot.
    """
    llvm.inline_asm(
        None,
        [],
        f"cp.async.bulk.wait_group.read {keep};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# --------------------------------------------------------------------------
# Section 4: TMA descriptors
# --------------------------------------------------------------------------

DESCRIPTOR_BYTES = 128
DESCRIPTOR_ALIGN = 64
GLOBAL_BASE_ALIGN = 16

#: the design fixes this order; the launch path indexes uploads by it.
DESCRIPTOR_ROLES = ("q", "k", "g", "v", "out", "state_in", "state_out")

_TMA_DTYPES = {
    torch.bfloat16: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    torch.float32: "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
}


@dataclass(frozen=True)
class TensorMapSpec:
    """One descriptor's complete encoder configuration.

    Only the fields below vary between roles; everything else in the encoder
    call is fixed by the design and is not representable as a difference.
    """

    dtype: torch.dtype
    base: int
    global_dims: tuple[int, ...]
    global_stride_bytes: tuple[int, ...]
    box_dims: tuple[int, ...]
    #: Present so the cache key and the equality test cover them, even though
    #: the contract pins all three.
    element_strides: tuple[int, ...] = (1, 1, 1)
    interleave: str = "NONE"
    swizzle: str = "128B"
    l2_promotion: str = "128B"
    oob_fill: str = "NONE"

    @property
    def rank(self) -> int:
        return len(self.global_dims)

    @property
    def element_size(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    @property
    def box_bytes(self) -> int:
        n = 1
        for b in self.box_dims:
            n *= b
        return n * self.element_size

    def key(self, device) -> tuple:
        """Full cache key: device plus every encoder field."""
        index = device.index if isinstance(device, torch.device) else device
        return (
            index,
            self.base,
            self.dtype,
            self.rank,
            self.global_dims,
            self.global_stride_bytes,
            self.box_dims,
            self.element_strides,
            self.interleave,
            self.swizzle,
            self.l2_promotion,
            self.oob_fill,
        )

    def validate(self) -> None:
        """Check what the driver will not.

        The driver rejects some of this itself, but with an error that names a
        parameter index rather than a role, and it accepts a misaligned global
        base outright -- the corruption from that shows up as wrong numbers in
        one head, far from here.
        """
        if self.rank != 3:
            raise ValueError(f"TMA rank must be 3, got {self.rank}")
        if self.base % GLOBAL_BASE_ALIGN:
            raise ValueError(
                f"TMA global base must be {GLOBAL_BASE_ALIGN}-byte aligned, "
                f"got {self.base:#x}"
            )
        if any(d <= 0 for d in self.global_dims):
            raise ValueError(
                f"TMA global dims must be positive, got {self.global_dims}"
            )
        if len(self.global_stride_bytes) != self.rank - 1:
            raise ValueError(
                "TMA global strides cover dimensions 1..rank-1 only, got "
                f"{len(self.global_stride_bytes)} for rank {self.rank}"
            )
        for s in self.global_stride_bytes:
            if s <= 0 or s % GLOBAL_BASE_ALIGN:
                raise ValueError(
                    f"TMA global strides must be positive and "
                    f"{GLOBAL_BASE_ALIGN}-byte aligned, got {self.global_stride_bytes}"
                )
        if len(self.box_dims) != self.rank:
            raise ValueError(f"TMA box must have rank {self.rank}")
        if any(not 1 <= b <= 256 for b in self.box_dims):
            raise ValueError(
                f"TMA box extents must be in [1, 256], got {self.box_dims}"
            )
        inner_bytes = self.box_dims[0] * self.element_size
        if inner_bytes != 128:
            raise ValueError(
                f"a 128B-swizzled inner box must be exactly 128 bytes, "
                f"got {inner_bytes}"
            )
        if self.dtype not in _TMA_DTYPES:
            raise ValueError(f"unsupported TMA element type {self.dtype}")

    def encode(self) -> bytes:
        """Encode this spec as its 128 raw descriptor bytes."""
        import cuda.bindings.driver as drv

        self.validate()
        return _encode_tiled(drv, self)


def _encode_tiled(drv, spec: TensorMapSpec) -> bytes:
    err, tmap = drv.cuTensorMapEncodeTiled(
        getattr(drv.CUtensorMapDataType, _TMA_DTYPES[spec.dtype]),
        spec.rank,
        spec.base,
        [drv.cuuint64_t(d) for d in spec.global_dims],
        [drv.cuuint64_t(s) for s in spec.global_stride_bytes],
        [drv.cuuint32_t(b) for b in spec.box_dims],
        [drv.cuuint32_t(e) for e in spec.element_strides],
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    # cuda-python wraps the descriptor, so take its address via getPtr().
    return bytes(ctypes.string_at(tmap.getPtr(), DESCRIPTOR_BYTES))


# ---------------------------------------------------------------------------
# Section 6.1: activations
# ---------------------------------------------------------------------------


def activation_spec(t: torch.Tensor, total_tokens: int, heads: int) -> TensorMapSpec:
    """Key-major ``(128, T_total, H)`` view of contiguous ``[1, T, H, 128]``.

    A 128-byte S128 segment is 64 BF16 or 32 FP32, so the box's inner extent
    follows the dtype: Q/K/V/out and BF16 ``G`` take two boxes per tile, FP32
    ``G`` four.
    """
    esz = t.element_size()
    segment = BF16_SEGMENT_ELEMS if esz == 2 else F32_SEGMENT_ELEMS
    return TensorMapSpec(
        dtype=t.dtype,
        base=t.data_ptr(),
        global_dims=(DK, total_tokens, heads),
        global_stride_bytes=(heads * DK * esz, DK * esz),
        box_dims=(segment, BT, 1),
    )


def activation_segments(dtype: torch.dtype) -> int:
    """Boxes one ``[16, 128]`` tile takes: 2 at BF16, 4 at FP32."""
    return 2 if dtype is torch.bfloat16 else 4


def activation_segment_elems(dtype: torch.dtype) -> int:
    return BF16_SEGMENT_ELEMS if dtype is torch.bfloat16 else F32_SEGMENT_ELEMS


# ---------------------------------------------------------------------------
# Section 6.2: state boundary
# ---------------------------------------------------------------------------


def state_spec(t: torch.Tensor, sequences: int, heads: int) -> TensorMapSpec:
    """External ``[N, H, 128, 128]`` state as ``(inner, lines, plane)``.

    The third dimension flattens ``(seq, head)`` into ``plane = seq * H + head``.
    The second counts 128-byte *lines*, not values: one value spans two BF16 or
    four FP32 segments, which is exactly the ``state_bf16_idx`` /
    ``state_f32_window_idx`` row index.  That is what lets the external
    ``[V, K]`` image land by TMA with no transpose and still be read as
    ``H[K, V]`` by the MMA.

    One descriptor serves both the prologue load and the epilogue store; the
    direction is decided by the PTX instruction alone.
    """
    esz = t.element_size()
    if esz == 2:
        segment, rows_per_value = BF16_SEGMENT_ELEMS, STATE_BF16_ROWS_PER_VALUE
    else:
        segment, rows_per_value = F32_SEGMENT_ELEMS, STATE_F32_ROWS_PER_VALUE
    lines = rows_per_value * DK
    return TensorMapSpec(
        dtype=t.dtype,
        base=t.data_ptr(),
        global_dims=(segment, lines, sequences * heads),
        global_stride_bytes=(segment * esz, DK * DK * esz),
        box_dims=(segment, lines // state_calls(t.dtype), 1),
    )


def state_calls(dtype: torch.dtype) -> int:
    """TMA calls one state plane takes: 2 at BF16 (16 KiB each), 4 at FP32."""
    return 2 if dtype is torch.bfloat16 else 4


def state_box_rows(dtype: torch.dtype) -> int:
    """Line rows one state box covers: 128 either way, at 16 KiB per box."""
    rows_per_value = (
        STATE_BF16_ROWS_PER_VALUE
        if dtype is torch.bfloat16
        else STATE_F32_ROWS_PER_VALUE
    )
    return rows_per_value * DK // state_calls(dtype)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorMapUpload:
    """A device buffer of descriptors plus each role's address.

    ``storage`` is kept as a strong reference: the addresses below are raw
    device pointers, and letting the tensor die would leave the kernel reading
    freed memory on the next launch.
    """

    storage: torch.Tensor
    addresses: dict[str, int]
    specs: dict[str, TensorMapSpec]

    def address(self, role: str) -> int:
        """Descriptor address of ``role``, or 0 when the role is absent.

        Zero is the kernel's "no descriptor" sentinel: an absent state has no
        ``fence.proxy.tensormap`` and no copy, both under ``const_expr``
        branches, so the address is never dereferenced.
        """
        return self.addresses.get(role, 0)


def build_upload(specs: dict[str, TensorMapSpec], device) -> TensorMapUpload:
    """Encode, deduplicate and upload the descriptors for one launch.

    Deduplication is by full spec equality, which is exactly the design's
    rule: ``v is out`` and an exact state in/out alias produce identical specs
    and share one descriptor, while anything that differs in a single encoder
    field gets its own.  Direction is still decided by the PTX instruction, so
    sharing a descriptor between a load and a store is safe.
    """
    blobs: list[bytes] = []
    slot_of: dict[TensorMapSpec, int] = {}
    role_slot: dict[str, int] = {}
    for role in DESCRIPTOR_ROLES:
        spec = specs.get(role)
        if spec is None:
            continue
        slot = slot_of.get(spec)
        if slot is None:
            slot = len(blobs)
            slot_of[spec] = slot
            blobs.append(spec.encode())
        role_slot[role] = slot

    if not blobs:
        empty = torch.empty(0, dtype=torch.uint8, device=device)
        return TensorMapUpload(storage=empty, addresses={}, specs=dict(specs))

    packed = bytearray()
    for blob in blobs:
        packed += blob
    storage = torch.frombuffer(bytes(packed), dtype=torch.uint8).clone().to(device)
    base = storage.data_ptr()
    if base % DESCRIPTOR_ALIGN:
        raise RuntimeError(
            f"TMA descriptor storage must be {DESCRIPTOR_ALIGN}-byte aligned, "
            f"got {base:#x}"
        )
    addresses = {
        role: base + slot * DESCRIPTOR_BYTES for role, slot in role_slot.items()
    }
    return TensorMapUpload(storage=storage, addresses=addresses, specs=dict(specs))


# --------------------------------------------------------------------------
# Section 5: the fused device kernel
# --------------------------------------------------------------------------

#: Barrier arena indices, in 8-byte units from the start of dynamic SMEM.
_BAR = CONTROL_OFFSET // 8
BAR_QKG = _BAR + BAR_QKG_READY // 8
BAR_V = _BAR + BAR_V_READY // 8
BAR_MAT = _BAR + BAR_MATERIALIZED // 8
BAR_QKD = _BAR + BAR_QK_DONE // 8
BAR_AINV = _BAR + BAR_AINV_READY // 8
BAR_AQ = _BAR + BAR_AQ_READY // 8
BAR_AK = _BAR + BAR_AK_READY // 8
BAR_PROJ = _BAR + BAR_PROJECTION_DONE // 8
BAR_RFORM = _BAR + BAR_R_FORMED // 8
BAR_OUTR = _BAR + BAR_OUTPUT_READY // 8
BAR_OUTD = _BAR + BAR_OUTPUT_READ_DONE // 8
BAR_STATED = _BAR + BAR_STATE_DONE // 8
BAR_STATEIO = _BAR + BAR_STATE_IO // 8

#: Element strides of one main slot / V stage, by view width.
SLOT16 = MAIN_SLOT_BYTES // 2  # 8192 BF16
SLOT32 = MAIN_SLOT_BYTES // 4  # 4096 FP32
VSTAGE16 = V_STAGE_BYTES // 2  # 2048 BF16
V_BASE16 = V_STAGE_OFFSET // 2  # 24576
CTRL32 = CONTROL_OFFSET // 4  # 15360

#: Quadrant slot pairs of an N=16 accumulator: TL, BL, TR, BR (plan Sec. 2.2).
QUAD_TL = (0, 1)
QUAD_BL = (2, 3)
QUAD_TR = (4, 5)
QUAD_BR = (6, 7)
DIAGONAL_SLOTS = QUAD_TL + QUAD_BR


# ---------------------------------------------------------------------------
# Small DSL helpers
# ---------------------------------------------------------------------------


@cute.jit
def vec_at(ptr, idx, elems):
    """``ptr + idx`` as an ``elems``-long tensor, keeping 16-byte access width.

    ``Pointer.__add__`` drops the alignment attribute for any dynamic offset --
    even ``ptr + 8 * dyn``, since it does not reason about the multiplier -- and
    ``autovec_copy`` honours that attribute.  Without the divisibility claim
    every 8-element BF16 access lowers to eight scalar ones.  Each index
    reaching this helper is a multiple of ``elems`` by construction, which the
    layout tests check exhaustively.
    """
    return cute.make_tensor(
        ptr + cute.assume(cutlass.Int32(idx), divby=elems), cute.make_layout(elems)
    )


@cute.jit
def vec8_bf16(ptr, idx):
    frag = cute.make_rmem_tensor(8, cutlass.BFloat16)
    cute.autovec_copy(vec_at(ptr, idx, 8), frag)
    return frag


@cute.jit
def zero_vec8_bf16(ptr, idx):
    zeros = cute.make_rmem_tensor(8, cutlass.BFloat16)
    for i in cutlass.range_constexpr(8):
        zeros[i] = cutlass.BFloat16(0.0)
    cute.autovec_copy(zeros, vec_at(ptr, idx, 8))


@cute.jit
def store_vec4_f32(ptr, idx, values, base):
    """``values[base:base + 4]`` to ``ptr + idx`` as one 16-byte store."""
    vec = cute.make_rmem_tensor(4, cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        vec[i] = values[base + i]
    cute.autovec_copy(vec, vec_at(ptr, idx, 4))


@cute.jit
def load_vec2_f32(ptr, idx):
    """Two adjacent FP32 at ``ptr + idx`` as one 8-byte load."""
    frag = cute.make_rmem_tensor(2, cutlass.Float32)
    cute.autovec_copy(vec_at(ptr, idx, 2), frag)
    return frag


@cute.jit
def zero_vec4_f32(ptr, idx):
    zeros = cute.make_rmem_tensor(4, cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        zeros[i] = cutlass.Float32(0.0)
    cute.autovec_copy(zeros, vec_at(ptr, idx, 4))


@cute.jit
def bf16_round(x):
    """R16: ``cvt.rn.bf16.f32`` widened back to FP32."""
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


@cute.jit
def f16_round(x):
    """R_F16: FP32 -> FP16 -> FP32, the inverse chain's boundary."""
    return x.to(cutlass.Float16).to(cutlass.Float32)


@cute.jit
def zero4():
    z = cutlass.Float32(0.0)
    return (z, z, z, z)


@cute.jit
def zero8():
    z = cutlass.Float32(0.0)
    return (z, z, z, z, z, z, z, z)


@cute.jit
def mma_n8(a, b, c):
    """One native ``m16n8k16``: A is four registers, B two, C four."""
    return mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])


@cute.jit
def mma_16x16(a, b, c):
    """One logical ``m16n16k16``: the N=8 low half, then the high half.

    the design fixes that order and forbids any rounding between the two
    halves or inside the K reduction.
    """
    n0 = mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[2], b[3], c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


@cute.jit
def mma_16x16_f16(a, b, c):
    """:func:`mma_16x16` with FP16 operands: the twelve inverse MMAs."""
    n0 = mma_m16n8k16_f16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_f16(a[0], a[1], a[2], a[3], b[2], b[3], c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


@cute.jit
def a_to_b(f):
    """A-layout -> B-layout of the same matrix; register order unchanged."""
    return (
        movmatrix_b16(f[0]),
        movmatrix_b16(f[1]),
        movmatrix_b16(f[2]),
        movmatrix_b16(f[3]),
    )


@cute.jit
def a_to_at(f):
    """A-layout -> A-layout of the transpose; registers 1 and 2 swap.

    The one place a register swap is allowed: it is a matrix
    transpose, not a relabelling.
    """
    return (
        movmatrix_b16(f[0]),
        movmatrix_b16(f[2]),
        movmatrix_b16(f[1]),
        movmatrix_b16(f[3]),
    )


@cute.jit
def pack_a_bf16(c):
    """N=16 FP32 accumulator -> BF16 A-layout fragment."""
    return (
        pack_bf16x2(c[0], c[1]),
        pack_bf16x2(c[2], c[3]),
        pack_bf16x2(c[4], c[5]),
        pack_bf16x2(c[6], c[7]),
    )


@cute.jit
def pack_a_f16(c):
    """N=16 FP32 accumulator -> FP16 A-layout fragment (``pack16``)."""
    return (
        pack_f16x2(c[0], c[1]),
        pack_f16x2(c[2], c[3]),
        pack_f16x2(c[4], c[5]),
        pack_f16x2(c[6], c[7]),
    )


@cute.jit
def warp_arrive(mbar, lane):
    """One arrival per warp on a warp-counted barrier.

    ``mbarrier.arrive`` counts per thread, so 32 lanes arriving would overshoot
    a count of 4 or 8.  The warp synchronization before the elected arrival is
    what makes the other 31 lanes' shared-memory stores visible to whoever
    observes it; the arrival itself carries release semantics at CTA scope.
    """
    cute.arch.sync_warp()
    if lane == 0:
        cute.arch.mbarrier_arrive(mbar)


@cute.jit
def clamp_rows(token_end, token_base):
    """``min(16, token_end - token_base)``."""
    v = token_end - token_base
    if v > BT:
        v = cutlass.Int32(BT)
    return v


@cute.jit
def prepare_rendezvous():
    """The prepare group's named barrier, ``bar.sync 1, 128``.

    W0-W3 are the only group that uses it and they materialize one chunk at a
    time, so all four of a chunk's rendezvous points reuse the same id in
    sequence rather than needing four mbarriers.
    """
    cute.arch.barrier(
        barrier_id=PREPARE_BARRIER_ID,
        number_of_threads=PREPARE_BARRIER_THREADS,
    )


# ---------------------------------------------------------------------------
# W15: blockwise inverse
# ---------------------------------------------------------------------------


@cute.jit
def blockwise_inverse(l_acc, lane):
    """``(I + L)^-1`` from a strict-lower FP32 accumulator, as a BF16 A fragment.

    Twelve native FP16 MMAs.  ``l_acc`` holds BF16-rounded strict-lower values
    in the N=16 accumulator layout; the result is ``Ainv`` in A layout with an
    exactly zero top-right quadrant.

    Both update steps must be a product against an exact-zero C followed by a
    *separate* FP32 scalar add of the re-rounded seed.  Passing the seed as the
    MMA's C operand saves an instruction and changes the result, so plan
    Section 2.2 forbids it.  ``B0 = I - D`` likewise comes from the widened
    BF16 diagonal of ``L``, not from the already-FP16-rounded ``D``.
    """
    zero_r = cutlass.Int32(0)

    # D is the block *diagonal* of L -- the two 8x8 blocks -- and A21hat its
    # single lower-left block.  Keeping the other quadrants exactly zero is
    # what makes the chain's nilpotency argument hold.
    d_f16 = (
        pack_f16x2(l_acc[0], l_acc[1]),
        zero_r,
        zero_r,
        pack_f16x2(l_acc[6], l_acc[7]),
    )
    a21_f16 = (zero_r, pack_f16x2(l_acc[2], l_acc[3]), zero_r, zero_r)

    d2_f16 = pack_a_f16(mma_16x16_f16(d_f16, a_to_b(d_f16), zero8()))

    b0 = [cutlass.Float32(0.0) for _ in range(8)]
    for s in cutlass.range_constexpr(8):
        if cutlass.const_expr(s in DIAGONAL_SLOTS):
            row, col = mma_c16_coord(lane, s)
            eye = cutlass.Float32(0.0)
            if row == col:
                eye = cutlass.Float32(1.0)
            b0[s] = f16_round(eye - l_acc[s])

    b1_prod = mma_16x16_f16(pack_a_f16(tuple(b0)), a_to_b(d2_f16), zero8())
    b1 = [b0[s] + b1_prod[s] for s in range(8)]

    d4_f16 = pack_a_f16(mma_16x16_f16(d2_f16, a_to_b(d2_f16), zero8()))

    binv_prod = mma_16x16_f16(pack_a_f16(tuple(b1)), a_to_b(d4_f16), zero8())
    binv = [f16_round(b1[s]) + binv_prod[s] for s in range(8)]

    # Binv is block diagonal by construction, so a diagonal-only fragment is
    # exact rather than a truncation.
    binv_packed = pack_a_f16(tuple(binv))
    binv_f16 = (binv_packed[0], zero_r, zero_r, binv_packed[3])

    t1 = mma_16x16_f16(binv_f16, a_to_b(a21_f16), zero8())
    # T1 = Binv @ A21hat is lower-left only.  Negating before the FP16 round is
    # exact either way; doing it here keeps the pack to one instruction.
    t1n_f16 = (zero_r, pack_f16x2(-t1[2], -t1[3]), zero_r, zero_r)
    x21 = mma_16x16_f16(t1n_f16, a_to_b(binv_f16), zero8())

    return (
        pack_bf16x2(binv[0], binv[1]),
        pack_bf16x2(x21[2], x21[3]),
        zero_r,
        pack_bf16x2(binv[6], binv[7]),
    )


# ---------------------------------------------------------------------------
# W0-W3: prepare
# ---------------------------------------------------------------------------


@cute.jit
def clear_tail_rows(p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32: cutlass.Constexpr):
    """Zero the invalid rows of the raw stages.

    TMA only zero-fills coordinates outside the *tensor*, and a short chunk
    sits mid-tensor: the rows past ``valid_rows`` hold the next sequence's
    tokens, so the copy faithfully loaded real data there.  The task map
    matches the loads' 16-byte width so the writes stay one vector wide.
    """
    for rep in cutlass.range_constexpr(2):
        task = tidx + rep * PREPARE_BARRIER_THREADS
        row = task // BT
        d0 = (task - row * BT) * 8
        if row >= valid_rows:
            idx = raw_bf16_s128(row, d0)
            zero_vec8_bf16(p_q, idx)
            zero_vec8_bf16(p_k, idx)

    if cutlass.const_expr(G_FP32):
        for rep in cutlass.range_constexpr(4):
            task = tidx + rep * PREPARE_BARRIER_THREADS
            row = task // 32
            d0 = (task - row * 32) * 4
            if row >= valid_rows:
                zero_vec4_f32(p_g_raw, raw_f32_s128(row, d0))
    else:
        for rep in cutlass.range_constexpr(2):
            task = tidx + rep * PREPARE_BARRIER_THREADS
            row = task // BT
            d0 = (task - row * BT) * 8
            if row >= valid_rows:
                zero_vec8_bf16(p_g_raw, raw_bf16_s128(row, d0))


@cute.jit
def gate_column(
    smem_e,
    smem_g_bf16,
    dim,
    dt_value,
    a_val,
    gate_scale_log2,
    valid_rows,
    G_FP32: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
):
    """The 16 gate values of one key dimension, as ``E = exp2(clamped prefix)``.

    Three things here are contract, not style: the
    invalid-row mask selects an exact ``+0.0`` *increment* -- zeroing raw ``G``
    is not enough, because ``dt_bias`` still produces a non-zero increment; the
    scan uses the pairwise bracketing rather than a plain running sum; and the
    clamp happens after the complete scan and before ``exp2``.
    """
    regs = [cutlass.Float32(0.0) for _ in range(BT)]
    for r in cutlass.range_constexpr(BT):
        if cutlass.const_expr(G_FP32):
            raw = smem_e[raw_f32_s128(r, dim)]
        else:
            raw = cutlass.Float32(smem_g_bf16[raw_bf16_s128(r, dim)])
        inc = cutlass.Float32(0.0)
        if r < valid_rows:
            x = raw + dt_value
            if cutlass.const_expr(SAFE_GATE):
                half = cutlass.Float32(0.5)
                sig = (
                    cutlass.Float32(cute.math.tanh(a_val * x * half, fastmath=True))
                    * half
                    + half
                )
                inc = gate_scale_log2 * sig
            else:
                sp = cutlass.Float32(0.0)
                if x > cutlass.Float32(SOFTPLUS_CUT):
                    sp = x * cutlass.Float32(LOG2_E)
                else:
                    sp = cutlass.Float32(
                        cute.math.log2(
                            cutlass.Float32(1.0)
                            + cutlass.Float32(cute.math.exp(x, fastmath=True)),
                            fastmath=True,
                        )
                    )
                inc = -a_val * sp
        regs[r] = inc

    acc = cutlass.Float32(0.0)
    for p in cutlass.range_constexpr(BT // 2):
        g0 = regs[2 * p]
        g1 = regs[2 * p + 1]
        first = acc + g0
        second = acc + (g0 + g1)
        regs[2 * p] = first
        regs[2 * p + 1] = second
        acc = second

    for r in cutlass.range_constexpr(BT):
        pv = regs[r]
        if pv < cutlass.Float32(PREFIX_FLOOR):
            pv = cutlass.Float32(PREFIX_FLOOR)
        regs[r] = cutlass.Float32(cute.math.exp2(pv, fastmath=True))
    return regs


@cute.jit
def row_sum_8(value):
    """Reduce the eight lanes that cooperate on one token row.

    the design pins the butterfly offsets to 4 -> 2 -> 1 and each step to
    a single FP32 add.  A tree or vector reduction reassociates, which is a
    different number.
    """
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=4))
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=2))
    return value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=1))


@cute.jit
def prepare_head(
    c,
    p16,
    p32,
    p64,
    tidx,
    dim,
    dt_value,
    a_val,
    gate_scale_log2,
    token_start,
    token_end,
    G_FP32: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
):
    """Chunk ``c``'s gate column: wait for raw Q/K/G, clear its tail, scan it.

    This is the stage the pipeline moves.  It is everything from ``qkg_ready``
    up to and including ``gate_column`` -- 51 MUFU, the longest dependent chain
    in prepare -- and it depends on nothing the previous chunk produces, which
    is what makes it hoistable ahead of the previous chunk's ``ainv_ready``
    wait.  It returns the whole column because ``e_col`` is the only thing the
    rest of the chunk needs from it; ``gt_owned`` is its last element.
    """
    s = main_slot(c)
    pr = ready_parity(c)
    s16 = s * SLOT16
    s32 = s * SLOT32
    valid_rows = clamp_rows(token_end, token_start + c * BT)

    p_q = p16 + s16
    p_k = p16 + (s16 + SLOT_K // 2)
    p_g_bf16 = p16 + (s16 + SLOT_G_LO // 2)
    p_e = p32 + (s32 + SLOT_G_LO // 4)

    smem_e = cute.make_tensor(p_e, cute.make_layout(BT * DK))
    smem_g_bf16 = cute.make_tensor(p_g_bf16, cute.make_layout(BT * DK))
    p_g_raw = p_e if cutlass.const_expr(G_FP32) else p_g_bf16

    cute.arch.mbarrier_wait(p64 + (BAR_QKG + s), pr)
    if valid_rows < BT:
        clear_tail_rows(p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32)
    # Rendezvous 1: raw operands are readable only once every tail row of
    # every stage has been cleared.
    prepare_rendezvous()

    return gate_column(
        smem_e,
        smem_g_bf16,
        dim,
        dt_value,
        a_val,
        gate_scale_log2,
        valid_rows,
        G_FP32,
        SAFE_GATE,
    )


@cute.jit
def prepare_body(
    c,
    e_col,
    p16,
    p32,
    p64,
    lane,
    p,
    dim,
    norm_token,
    lane_in_row,
    q_reg,
    packed_scale,
    token_start,
    token_end,
):
    """Publish E, normalize, and materialize Qd/Kd/Ki; hand Kr to the tail.

    Everything here belongs to chunk ``c`` alone and ends at
    ``materialized``.  ``kr_b`` is returned rather than stored because Kr is
    the one factor that never reaches SMEM -- it stays in the B layout it was
    built in until the tail multiplies it by AinvBeta.
    """
    s = main_slot(c)
    s16 = s * SLOT16
    s32 = s * SLOT32
    valid_rows = clamp_rows(token_end, token_start + c * BT)

    p_q = p16 + s16
    p_k = p16 + (s16 + SLOT_K // 2)
    p_ki = p16 + (s16 + SLOT_KI // 2)
    p_e = p32 + (s32 + SLOT_G_LO // 4)
    smem_gt = cute.make_tensor(p32 + (s32 + SLOT_GTOTAL // 4), cute.make_layout(DK))
    scratch = cute.make_tensor(
        p32 + (CTRL32 + s * (SCRATCH_SLOT_BYTES // 4)), cute.make_layout(32)
    )
    gt_owned = e_col[BT - 1]

    # Rendezvous 2: one lane's E store can land on a raw element another
    # lane has not read yet.  This used to be skipped for FP32 G, where the
    # E map and the raw map coincided and each lane rewrote exactly what it
    # read.  E is transposed now, so the two maps no longer coincide for
    # either width and the barrier is unconditional -- one `bar.sync 1, 128`
    # per chunk, bought for twenty fewer shared accesses per warp.
    prepare_rendezvous()
    for grp in cutlass.range_constexpr(E_GROUPS):
        store_vec4_f32(
            p_e,
            gate_e_f32(dim, E_GROUP_ELEMS * grp),
            e_col,
            E_GROUP_ELEMS * grp,
        )

    # --- norm: four token rows per warp, eight lanes per row ---------------
    q_ss = cutlass.Float32(0.0)
    k_ss = cutlass.Float32(0.0)
    for h in cutlass.range_constexpr(2):
        base = raw_bf16_s128(norm_token, 64 * h + 8 * lane_in_row)
        fq = vec8_bf16(p_q, base)
        fk = vec8_bf16(p_k, base)
        for i in cutlass.range_constexpr(8):
            qv = cutlass.Float32(fq[i])
            kv = cutlass.Float32(fk[i])
            q_ss = q_ss + qv * qv
            k_ss = k_ss + kv * kv
    q_ss = row_sum_8(q_ss)
    k_ss = row_sum_8(k_ss)
    # All eight lanes hold the reduced value; only the row's lane 0 stores.
    if lane_in_row == 0:
        q_inv = cutlass.Float32(0.0)
        k_inv = cutlass.Float32(0.0)
        if norm_token < valid_rows:
            qf = q_ss
            if qf < cutlass.Float32(NORM_FLOOR):
                qf = cutlass.Float32(NORM_FLOOR)
            kf = k_ss
            if kf < cutlass.Float32(NORM_FLOOR):
                kf = cutlass.Float32(NORM_FLOOR)
            q_inv = cutlass.Float32(cute.math.rsqrt(qf, fastmath=True))
            k_inv = cutlass.Float32(cute.math.rsqrt(kf, fastmath=True))
        scratch[norm_token] = q_inv
        scratch[BT + norm_token] = k_inv

    # Rendezvous 3: E and the norm scratch are published together.  The
    # norm's row owners are not the dimension owners that read them back,
    # which is exactly why q_inv/k_inv need shared scratch at all.
    prepare_rendezvous()

    # --- capture E at the materializer's own coordinates -------------------
    e_cap = [cutlass.Float32(0.0) for _ in range(16)]
    for j in cutlass.range_constexpr(2):
        for nb in cutlass.range_constexpr(2):
            for r in cutlass.range_constexpr(2):
                t0, _t1, key = materialize_coords(p, lane, j, nb, r)
                idx = ((j * 2 + nb) * 2 + r) * 2
                # ``t0`` is even and ``t1 = t0 + 1``, so the pair never
                # straddles a group boundary and one load covers both.
                pair = load_vec2_f32(p_e, gate_e_f32(key, t0))
                e_cap[idx] = pair[0]
                e_cap[idx + 1] = pair[1]
    # t0/t1 depend only on (q, r), so eight scalar reads cover all tiles.
    q_inv_t = [
        scratch[2 * q_reg],
        scratch[2 * q_reg + 1],
        scratch[2 * q_reg + 8],
        scratch[2 * q_reg + 9],
    ]
    k_inv_t = [
        scratch[BT + 2 * q_reg],
        scratch[BT + 2 * q_reg + 1],
        scratch[BT + 2 * q_reg + 8],
        scratch[BT + 2 * q_reg + 9],
    ]

    # Rendezvous 4: every prepare warp now holds its whole E set in
    # registers, so E's bytes may be overwritten by Ki and factor records.
    prepare_rendezvous()
    smem_gt[dim] = gt_owned

    kr_b = [cutlass.Int32(0) for _ in range(8)]
    for j in cutlass.range_constexpr(2):
        kd_b = [cutlass.Int32(0) for _ in range(4)]
        ki_b = [cutlass.Int32(0) for _ in range(4)]
        qd_b = [cutlass.Int32(0) for _ in range(4)]
        # The whole [16, 16] tile in one instruction per factor: the
        # materializer's coordinates *are* the native B fragment, which is
        # what `.trans` delivers over a [token, key] image.  Both tiles are
        # read before this iteration's factor stores overwrite them, and
        # the two j values address disjoint key halves, so hoisting the
        # loads above the stores is safe for j = 1 as well.
        k_raw = ldmatrix_x4_trans(p_k + materialize_b_ptr(p, lane, j))
        q_raw = ldmatrix_x4_trans(p_q + materialize_b_ptr(p, lane, j))
        for nb in cutlass.range_constexpr(2):
            for r in cutlass.range_constexpr(2):
                # Only the key is still needed as a coordinate -- the two
                # token rows are now implicit in the fragment the tile load
                # delivered -- but it comes from the same map, because the
                # shuffle below depends on it agreeing with the gate's
                # ownership.
                key = materialize_coords(p, lane, j, nb, r)[2]
                idx = ((j * 2 + nb) * 2 + r) * 2
                e0 = e_cap[idx]
                e1 = e_cap[idx + 1]
                qi0 = q_inv_t[2 * r]
                qi1 = q_inv_t[2 * r + 1]
                ki0 = k_inv_t[2 * r]
                ki1 = k_inv_t[2 * r + 1]

                # `cvt.f32.bf16` is the same widening the scalar load's
                # Float32() did, and it is exact, so the products below are
                # the same FP32 numbers as before.
                kv0, kv1 = unpack_bf16x2(k_raw[2 * nb + r])
                kn0 = kv0 * ki0
                kn1 = kv1 * ki1
                e16 = pack_bf16x2(e0, e1)
                kd_b[2 * nb + r] = mul_bf16x2(pack_bf16x2(kn0, kn1), e16)
                ki_b[2 * nb + r] = pack_bf16x2(
                    kn0 * rcp_approx_ftz(e0), kn1 * rcp_approx_ftz(e1)
                )
                qv0, qv1 = unpack_bf16x2(q_raw[2 * nb + r])
                qn16 = pack_bf16x2(qv0 * qi0, qv1 * qi1)
                qd_b[2 * nb + r] = mul_bf16x2(mul_bf16x2(qn16, e16), packed_scale)

                # GTotal of ``key`` belongs to its gate owner, which is a
                # different lane than the one materializing this element.
                gt16 = bf16_round(
                    cutlass.Float32(cute.arch.shuffle_sync(gt_owned, key - 32 * p))
                )
                kr_b[4 * j + 2 * nb + r] = mul_bf16x2(
                    ki_b[2 * nb + r], pack_bf16x2(gt16, gt16)
                )

        # B layout -> A layout rides the store: `stmatrix.x4.trans`
        # transposes each 8x8 tile with the register order unchanged, which
        # is what the three `a_to_b` calls used to spend twelve movmatrix
        # doing.  Kr alone stays in the B layout it was built in and never
        # reaches SMEM.
        store_at = factor_a_ptr(lane, 2 * p + j)
        stmatrix_x4_trans(p_q + store_at, qd_b[0], qd_b[1], qd_b[2], qd_b[3])
        stmatrix_x4_trans(p_k + store_at, kd_b[0], kd_b[1], kd_b[2], kd_b[3])
        stmatrix_x4_trans(p_ki + store_at, ki_b[0], ki_b[1], ki_b[2], ki_b[3])

    warp_arrive(p64 + (BAR_MAT + s), lane)
    return kr_b


@cute.jit
def prepare_tail(c, kr_b, p16, p64, lane, p):
    """``Ak.T = AinvBeta.T * Kr``, once W15 publishes the inverse.

    The wait here is the 7.5% window the pipeline exists to fill: whatever the
    caller schedules between ``prepare_body`` and this call runs while W15 is
    still inverting.
    """
    s = main_slot(c)
    pr = ready_parity(c)
    p_ki = p16 + (s * SLOT16 + SLOT_KI // 2)
    p_ainvb = p16 + (s * SLOT16 + SLOT_AINV_BETA // 2)

    cute.arch.mbarrier_wait(p64 + (BAR_AINV + s), pr)
    ainvb_t = a_to_at(ldmatrix_x4(p_ainvb + pairwise_a_ptr(lane)))
    for tile in cutlass.range_constexpr(2):
        acc = mma_16x16(
            ainvb_t,
            (
                kr_b[4 * tile],
                kr_b[4 * tile + 1],
                kr_b[4 * tile + 2],
                kr_b[4 * tile + 3],
            ),
            zero8(),
        )
        frag = pack_a_bf16(acc)
        if cutlass.const_expr(tile == 0):
            # Ki dies for W13 at qk_done and for W15 at ainv_ready; both
            # are required before the first byte of Ak.T lands on it.  This
            # is usually already satisfied, but the wait may not be elided
            # on that assumption.
            cute.arch.mbarrier_wait(p64 + (BAR_QKD + s), pr)
        stmatrix_x4(
            p_ki + ak_store_ptr(lane, 32 * p + BT * tile),
            frag[0],
            frag[1],
            frag[2],
            frag[3],
        )
    warp_arrive(p64 + (BAR_AK + s), lane)


@cute.jit
def prepare_role(
    ga_log,
    gdt,
    p16,
    p32,
    p64,
    tidx,
    lane,
    warp_id,
    head,
    token_start,
    token_end,
    num_chunks,
    scale,
    gate_scale_log2,
    G_FP32: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
):
    """Gate, normalize, materialize Qd/Kd/Ki, retain Kr, then publish Ak.T.

    A prepare warp uses three different ownerships inside one chunk, and they
    do not compose into "warp p owns 32 dimensions": the gate is one dimension
    per lane, the norm is one token row per eight lanes, and the materializer
    works in the MMA **B** fragment's coordinates.  Conflating them is the
    easiest way to get this stage subtly wrong.

    The chunk is split into ``prepare_head`` / ``prepare_body`` /
    ``prepare_tail`` so the head can be software-pipelined one chunk ahead of
    the tail's ``ainv_ready`` wait; see the comment on the loop below.
    """
    p = warp_id
    dim = gate_owner_dim(p, lane)
    dt_value = cutlass.Float32(gdt[head * DK + dim])

    # ``a = exp2(A_log * log2e)``, computed once by lane 0 and broadcast.
    a_seed = cutlass.Float32(0.0)
    if lane == 0:
        a_seed = cutlass.Float32(
            cute.math.exp2(
                cutlass.Float32(ga_log[head]) * cutlass.Float32(LOG2_E),
                fastmath=True,
            )
        )
    a_val = cutlass.Float32(cute.arch.shuffle_sync(a_seed, 0))

    s16_scale = bf16_round(scale)
    packed_scale = pack_bf16x2(s16_scale, s16_scale)

    norm_token = norm_row(p, lane)
    lane_in_row = lane & 7
    q_reg = lane & 3

    # --- the software pipeline -------------------------------------------
    #
    # ``head(c + 1)`` is issued before ``tail(c)``, so the gate's exp2 chain
    # for the next chunk runs while W15 is still inverting this one.  The
    # shape is the textbook one -- prologue, ``n - 1`` steady-state
    # iterations, epilogue -- and deliberately contains no runtime predicate:
    # an earlier attempt guarded the hoisted head with
    # ``if c + 1 < num_chunks``, and that dynamic ``if`` was the entire source
    # of its trouble.  Drawing the loop boundary here needs none.
    #
    # ``e_col`` (16 FP32) is the only value carried across the iteration
    # boundary, and ``kr_b`` (8 packed pairs) the only one live across the
    # hoisted head.  Both fit: ~82 registers are free at the ``ainv_ready``
    # wait, because Qd/Kd/Ki have just gone to SMEM.
    #
    # The guard is required and is not the predicate that caused the earlier
    # trouble.  A packed varlen batch may contain a zero-length sequence, and
    # such a block runs with ``num_chunks == 0``.  Every other role spells its
    # work as ``for c in range(num_chunks)``, which is simply empty then; this
    # role no longer does, because the prologue sits outside the loop.  An
    # unguarded prologue waits on ``qkg_ready`` for a chunk W12's own empty
    # loop will never produce, and W0-W3 hang there forever.  ``last`` would
    # also be -1 and index a slot that does not exist.
    #
    # Unlike ``if c + 1 < num_chunks`` inside the loop, this carries nothing
    # across the branch: the whole pipeline is on one side and the block has
    # no work at all on the other.
    if num_chunks > 0:
        e_col = prepare_head(
            cutlass.Int32(0),
            p16,
            p32,
            p64,
            tidx,
            dim,
            dt_value,
            a_val,
            gate_scale_log2,
            token_start,
            token_end,
            G_FP32,
            SAFE_GATE,
        )
        for c in range(num_chunks - 1):
            kr_b = prepare_body(
                c,
                e_col,
                p16,
                p32,
                p64,
                lane,
                p,
                dim,
                norm_token,
                lane_in_row,
                q_reg,
                packed_scale,
                token_start,
                token_end,
            )
            e_col = prepare_head(
                c + 1,
                p16,
                p32,
                p64,
                tidx,
                dim,
                dt_value,
                a_val,
                gate_scale_log2,
                token_start,
                token_end,
                G_FP32,
                SAFE_GATE,
            )
            prepare_tail(c, kr_b, p16, p64, lane, p)

        last = num_chunks - 1
        kr_b = prepare_body(
            last,
            e_col,
            p16,
            p32,
            p64,
            lane,
            p,
            dim,
            norm_token,
            lane_in_row,
            q_reg,
            packed_scale,
            token_start,
            token_end,
        )
        prepare_tail(last, kr_b, p16, p64, lane, p)


# ---------------------------------------------------------------------------
# W12: TMA producer
# ---------------------------------------------------------------------------


@cute.jit
def tma_role(
    p16,
    p32,
    p64,
    lane,
    head,
    token_start,
    num_chunks,
    desc_q,
    desc_k,
    desc_g,
    desc_v,
    G_FP32: cutlass.Constexpr,
):
    """Prefetch Q/K/G and V three chunks ahead, against independent barriers.

    Generation 0's reuse parity is 1, which passes immediately on a freshly
    initialized phase-0 barrier, so this is one uniform loop from ``c = 0``
    with no prologue special case: the first three iterations fall straight
    through and only ``c >= 3`` is throttled by the release of slot ``c - 3``.

    All three release conditions are required and none subsumes another:
    ``r_formed`` frees the V stage, ``output_read_done`` the output half of the
    main slot, and ``state_done`` the Ak.T and GTotal records.

    Q/K/G and V share this issue point but not their barrier: making V's
    visibility depend on ``qkg_ready`` would stall the residual behind factors
    it does not need.
    """
    qkg_tx = QKG_TX_BYTES_G_FP32 if cutlass.const_expr(G_FP32) else QKG_TX_BYTES_G_BF16
    for c in range(num_chunks):
        s = main_slot(c)
        rp = reuse_parity(c)
        if lane == 0:
            cute.arch.mbarrier_wait(p64 + (BAR_RFORM + s), rp)
            cute.arch.mbarrier_wait(p64 + (BAR_OUTD + s), rp)
            cute.arch.mbarrier_wait(p64 + (BAR_STATED + s), rp)

            token_base = token_start + c * BT
            s16 = s * SLOT16
            p_q = p16 + s16
            p_k = p16 + (s16 + SLOT_K // 2)
            mbar = p64 + (BAR_QKG + s)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, qkg_tx)
            for seg in cutlass.range_constexpr(2):
                c0 = seg * BF16_SEGMENT_ELEMS
                off = seg * BF16_SEGMENT_STRIDE
                tma_load_3d(p_q + off, desc_q, mbar, c0, token_base, head)
                tma_load_3d(p_k + off, desc_k, mbar, c0, token_base, head)
            if cutlass.const_expr(G_FP32):
                p_g = p32 + (s * SLOT32 + SLOT_G_LO // 4)
                for seg in cutlass.range_constexpr(4):
                    tma_load_3d(
                        p_g + seg * F32_SEGMENT_STRIDE,
                        desc_g,
                        mbar,
                        seg * F32_SEGMENT_ELEMS,
                        token_base,
                        head,
                    )
            else:
                p_g = p16 + (s16 + SLOT_G_LO // 2)
                for seg in cutlass.range_constexpr(2):
                    tma_load_3d(
                        p_g + seg * BF16_SEGMENT_STRIDE,
                        desc_g,
                        mbar,
                        seg * BF16_SEGMENT_ELEMS,
                        token_base,
                        head,
                    )

            vbar = p64 + (BAR_V + s)
            p_v = p16 + (V_BASE16 + s * VSTAGE16)
            cute.arch.mbarrier_arrive_and_expect_tx(vbar, V_TX_BYTES)
            for seg in cutlass.range_constexpr(2):
                tma_load_3d(
                    p_v + seg * BF16_SEGMENT_STRIDE,
                    desc_v,
                    vbar,
                    seg * BF16_SEGMENT_ELEMS,
                    token_base,
                    head,
                )


# ---------------------------------------------------------------------------
# W13: QK and Aq
# ---------------------------------------------------------------------------


@cute.jit
def qk_role(p16, p64, lane, num_chunks):
    """Causal ``QK = tril(Qd @ Ki.T)``, then ``Aq = QK @ AinvBeta``.

    ``QK_A`` stays in registers across the wait for ``ainv_ready``; staging it
    through SMEM would need a region the arena does not have.  ``qk_done`` is
    released as soon as ``Ki`` has been read, long before ``Aq`` exists,
    because W0-W3 need it to start overwriting ``Ki`` with ``Ak.T``.
    """
    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        s16 = s * SLOT16
        p_qd = p16 + s16
        p_ki = p16 + (s16 + SLOT_KI // 2)
        p_ainvb = p16 + (s16 + SLOT_AINV_BETA // 2)
        p_aq = p16 + (s16 + SLOT_AQ // 2)

        cute.arch.mbarrier_wait(p64 + (BAR_MAT + s), pr)
        acc = zero8()
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            acc = mma_16x16(
                ldmatrix_x4(p_qd + factor_a_ptr(lane, kb)),
                ldmatrix_x4(p_ki + ki_b_ptr(lane, kb)),
                acc,
            )
        masked = [cutlass.Float32(0.0) for _ in range(8)]
        for slot in cutlass.range_constexpr(8):
            row, col = mma_c16_coord(lane, slot)
            v = cutlass.Float32(0.0)
            if row >= col:
                v = acc[slot]
            masked[slot] = bf16_round(v)
        qk_a = pack_a_bf16(tuple(masked))
        warp_arrive(p64 + (BAR_QKD + s), lane)

        cute.arch.mbarrier_wait(p64 + (BAR_AINV + s), pr)
        ainvb_b = a_to_b(ldmatrix_x4(p_ainvb + pairwise_a_ptr(lane)))
        frag = pack_a_bf16(mma_16x16(qk_a, ainvb_b, zero8()))
        stmatrix_x4(p_aq + pairwise_store_ptr(lane), frag[0], frag[1], frag[2], frag[3])
        warp_arrive(p64 + (BAR_AQ + s), lane)


# ---------------------------------------------------------------------------
# W15: beta, KK and the blockwise inverse
# ---------------------------------------------------------------------------


@cute.jit
def inverse_role(
    gbeta, p16, p64, lane, head, heads, token_start, token_end, num_chunks
):
    """Activate beta, form ``L``, invert ``I + L`` and publish ``AinvBeta``.

    Beta gets no shared memory: each lane holds at most one activated FP32
    value and every row or column access is a ``shuffle_sync``.  An invalid
    token's activated beta must be an exact ``+0.0`` -- it scales a whole
    strict-lower row, so anything else leaks the tail into ``Ainv``.
    """
    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        s16 = s * SLOT16
        p_kd = p16 + (s16 + SLOT_KD // 2)
        p_ki = p16 + (s16 + SLOT_KI // 2)
        p_ainvb = p16 + (s16 + SLOT_AINV_BETA // 2)
        token_base = token_start + c * BT
        valid_rows = clamp_rows(token_end, token_base)

        # Issued before the wait: this is a strided column read out of [T, H]
        # that cannot coalesce, so it should be in flight during the KK loop.
        beta_owned = cutlass.Float32(0.0)
        if lane < BT:
            if lane < valid_rows:
                logit = cutlass.Float32(
                    gbeta[beta_global_index(token_base + lane, head, heads)]
                )
                half = cutlass.Float32(0.5)
                beta_owned = (
                    cutlass.Float32(cute.math.tanh(logit * half, fastmath=True)) * half
                    + half
                )

        cute.arch.mbarrier_wait(p64 + (BAR_MAT + s), pr)
        acc = zero8()
        for kb in cutlass.range_constexpr(KEY_BLOCKS):
            acc = mma_16x16(
                ldmatrix_x4(p_kd + factor_a_ptr(lane, kb)),
                ldmatrix_x4(p_ki + ki_b_ptr(lane, kb)),
                acc,
            )

        # A lane's eight accumulator slots occupy only two distinct rows, so
        # two shuffles cover the whole strict-lower scale.
        g_reg = lane >> 2
        q_reg = lane & 3
        beta_lo_row = cutlass.Float32(cute.arch.shuffle_sync(beta_owned, g_reg))
        beta_hi_row = cutlass.Float32(cute.arch.shuffle_sync(beta_owned, g_reg + 8))
        l_acc = [cutlass.Float32(0.0) for _ in range(8)]
        for slot in cutlass.range_constexpr(8):
            row, col = mma_c16_coord(lane, slot)
            brow = beta_lo_row if cutlass.const_expr((slot & 3) < 2) else beta_hi_row
            v = cutlass.Float32(0.0)
            if row > col:
                v = bf16_round(acc[slot] * brow)
            l_acc[slot] = v

        ainv_a = blockwise_inverse(tuple(l_acc), lane)

        beta_lo = pack_bf16x2(
            cutlass.Float32(cute.arch.shuffle_sync(beta_owned, 2 * q_reg)),
            cutlass.Float32(cute.arch.shuffle_sync(beta_owned, 2 * q_reg + 1)),
        )
        beta_hi = pack_bf16x2(
            cutlass.Float32(cute.arch.shuffle_sync(beta_owned, 2 * q_reg + 8)),
            cutlass.Float32(cute.arch.shuffle_sync(beta_owned, 2 * q_reg + 9)),
        )
        stmatrix_x4(
            p_ainvb + pairwise_store_ptr(lane),
            mul_bf16x2(ainv_a[0], beta_lo),
            mul_bf16x2(ainv_a[1], beta_lo),
            mul_bf16x2(ainv_a[2], beta_hi),
            mul_bf16x2(ainv_a[3], beta_hi),
        )
        warp_arrive(p64 + (BAR_AINV + s), lane)


# ---------------------------------------------------------------------------
# W14: output store
# ---------------------------------------------------------------------------


@cute.jit
def io_role(
    gout, p16, p64, lane, head, heads, token_start, token_end, num_chunks, desc_out
):
    """Publish each chunk's output, then release the slot.

    A partial chunk must not go out by TMA: the box is 16 rows wide, and in a
    packed sequence the rows past ``valid_rows`` belong to the *next* sequence,
    so a full-tile store would overwrite output another CTA owns.

    ``output_read_done`` is released only after the store has *read* its source
    -- ``wait_group.read``, not merely ``commit`` -- because that is the
    condition for W12 to overwrite the slot.
    """
    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        p_out = p16 + s * SLOT16
        token_base = token_start + c * BT
        valid_rows = clamp_rows(token_end, token_base)

        cute.arch.mbarrier_wait(p64 + (BAR_OUTR + s), pr)
        if valid_rows == BT:
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                for seg in cutlass.range_constexpr(2):
                    tma_store_3d(
                        desc_out,
                        p_out + seg * BF16_SEGMENT_STRIDE,
                        seg * BF16_SEGMENT_ELEMS,
                        token_base,
                        head,
                    )
                tma_store_commit_group()
                tma_store_wait_read(0)
        else:
            for rep in cutlass.range_constexpr(8):
                task = lane + rep * 32
                if task < valid_rows * BT:
                    row = task // BT
                    d0 = (task - row * BT) * 8
                    frag = vec8_bf16(p_out, raw_bf16_s128(row, d0))
                    cute.autovec_copy(
                        frag,
                        vec_at(
                            gout.iterator,
                            out_global_index(token_base + row, head, heads, d0),
                            8,
                        ),
                    )
            cute.arch.sync_warp()
        warp_arrive(p64 + (BAR_OUTD + s), lane)


# ---------------------------------------------------------------------------
# W4-W11: recurrence
# ---------------------------------------------------------------------------


@cute.jit
def h_branch(p_akt, smem_gt, h32, h16, res_a, lane, bar_ak, bar_state, parity):
    """``H.T = H.T Diag(GTotal) + R.T @ Ak.T``, in place, tile by tile.

    The transposed form: the accumulator's rows are values and its columns are
    keys, so ``pack_bf16x2`` alone leaves H16 in the A-fragment order the
    projection wants and no ``movmatrix`` is needed anywhere.

    GTotal now indexes the *column* axis, so a lane needs the two adjacent keys
    of each tile rather than two rows shared across both value blocks -- 32
    values per lane instead of 16.  They are loaded as scalars, not as a pair:
    the live range is what matters here, not the instruction count, and a
    16-byte or 8-byte load imposes a register-pair alignment this warp has
    already been measured unable to afford.

    The blocking wait is the real acquire: the first-ready probe only *chose*
    this branch, and a broadcast predicate gives the other 31 lanes no
    visibility of the Ak.T stores they are about to read.
    """
    cute.arch.mbarrier_wait(bar_ak, parity)
    # GTotal cooperatively: one fully coalesced scalar load per 32 keys covers
    # four tiles for the whole warp -- four wavefronts per chunk instead of one
    # per tile -- and a shuffle hands each lane the two it holds.  The source
    # register is warp-uniform (there is only one), which is what makes the
    # exchange expressible; only the source lane varies.
    gt_grp = [cutlass.Float32(0.0) for _ in range(GTOTAL_GROUP)]
    for i in cutlass.range_constexpr(GTOTAL_GROUP):
        gt_grp[i] = cutlass.Float32(smem_gt[gtotal_group_key(lane, i)])
    for j in cutlass.range_constexpr(KEY_BLOCKS):
        ak_b = ldmatrix_x4_trans(p_akt + ak_b_ptr(lane, j))
        for nb in cutlass.range_constexpr(2):
            kt = 2 * j + nb
            grp = kt // GTOTAL_GROUP
            gt0 = cutlass.Float32(
                cute.arch.shuffle_sync(gt_grp[grp], gtotal_shuffle_source(lane, kt, 0))
            )
            gt1 = cutlass.Float32(
                cute.arch.shuffle_sync(gt_grp[grp], gtotal_shuffle_source(lane, kt, 1))
            )
            i0 = h32t_idx(kt, 0)
            acc = mma_n8(
                res_a,
                (ak_b[2 * nb], ak_b[2 * nb + 1]),
                (
                    gt0 * h32[i0],
                    gt1 * h32[i0 + 1],
                    gt0 * h32[i0 + 2],
                    gt1 * h32[i0 + 3],
                ),
            )
            for r in cutlass.range_constexpr(4):
                h32[i0 + r] = acc[r]
            j0 = h16t_idx(kt, 0)
            h16[j0] = pack_bf16x2(acc[0], acc[1])
            h16[j0 + 1] = pack_bf16x2(acc[2], acc[3])
    warp_arrive(bar_state, lane)


@cute.jit
def o_branch(
    p_out, p_aq, o_acc, res_a, lane, v_base, bar_aq, bar_proj, bar_out, parity
):
    """``O.T = H.T @ Qd.T + R.T @ Aq.T``, into the projection's own FP32 bank.

    ``projection_done`` is waited on here not because ``Aq`` needs it, but
    because the output overwrites ``Qd``: it transitively proves every
    recurrence warp and W13 have finished reading that region.

    The store is ``.trans``: the accumulator is ``[value, token]`` while the
    output image is token-major, and the transpose rides the store for free.
    """
    cute.arch.mbarrier_wait(bar_aq, parity)
    cute.arch.mbarrier_wait(bar_proj, parity)
    aq_b = ldmatrix_x4(p_aq + pairwise_b_ptr(lane))
    for nb in cutlass.range_constexpr(2):
        acc = mma_n8(res_a, (aq_b[2 * nb], aq_b[2 * nb + 1]), o_acc[nb])
        stmatrix_x2_trans(
            p_out + vo_x2t_ptr(lane, v_base, nb),
            pack_bf16x2(acc[0], acc[1]),
            pack_bf16x2(acc[2], acc[3]),
        )
    warp_arrive(bar_out, lane)


@cute.jit
def recurrence_role(
    p16, p32, p64, h32, h16, lane, rec_id, token_start, token_end, num_chunks
):
    """Projection, residual, then the O and H branches in first-ready order.

    Transposed throughout: the accumulators are ``[value, token]`` and H is the
    MMA's A operand, so no C-to-B conversion happens anywhere in this warp.
    """
    v_base = rec_id * WARP_VALUES
    q = lane & 3

    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        s16 = s * SLOT16
        s32 = s * SLOT32
        p_out = p16 + s16
        p_qd = p_out
        p_kd = p16 + (s16 + SLOT_KD // 2)
        p_akt = p16 + (s16 + SLOT_AKT // 2)
        p_aq = p16 + (s16 + SLOT_AQ // 2)
        p_v = p16 + (V_BASE16 + s * VSTAGE16)
        smem_gt = cute.make_tensor(p32 + (s32 + SLOT_GTOTAL // 4), cute.make_layout(DK))
        token_base = token_start + c * BT
        valid_rows = clamp_rows(token_end, token_base)

        cute.arch.mbarrier_wait(p64 + (BAR_MAT + s), pr)

        # --- projection: H.T is the A operand, straight out of H16 ----------
        # `h_a_reg` is a selection, not a conversion: key block j's A fragment
        # is h16[4j..4j+3] already.  `fresh_b32` computes nothing -- it keeps
        # the MMA's operand off the persistent state registers, which is worth
        # 8 percentage points; see its docstring.
        x_acc = [zero4(), zero4()]
        o_acc = [zero4(), zero4()]
        for j in cutlass.range_constexpr(KEY_BLOCKS):
            kd_b = ldmatrix_x4(p_kd + ki_b_ptr(lane, j))
            qd_b = ldmatrix_x4(p_qd + ki_b_ptr(lane, j))
            h_a = (
                fresh_b32(h16[h_a_reg(j, 0)]),
                fresh_b32(h16[h_a_reg(j, 1)]),
                fresh_b32(h16[h_a_reg(j, 2)]),
                fresh_b32(h16[h_a_reg(j, 3)]),
            )
            for nb in cutlass.range_constexpr(2):
                x_acc[nb] = mma_n8(h_a, (kd_b[2 * nb], kd_b[2 * nb + 1]), x_acc[nb])
                o_acc[nb] = mma_n8(h_a, (qd_b[2 * nb], qd_b[2 * nb + 1]), o_acc[nb])
        warp_arrive(p64 + (BAR_PROJ + s), lane)

        # --- residual -------------------------------------------------------
        # A packed register now holds one value and *two adjacent tokens*, so
        # the tail is no longer one predicate per register: the two halves can
        # straddle `valid_rows`.  The mask is applied after the subtract and
        # leaves an exact packed zero on an invalid token, which is what the
        # contract requires -- V is not tail-cleared in prepare, so its rows
        # past `valid_rows` hold the next sequence's tokens.
        cute.arch.mbarrier_wait(p64 + (BAR_V + s), pr)
        res_a = [cutlass.Int32(0) for _ in range(4)]
        for nb in cutlass.range_constexpr(2):
            v_lo, v_hi = ldmatrix_x2_trans(p_v + vo_x2t_ptr(lane, v_base, nb))
            t0 = 8 * nb + 2 * q
            mask = cutlass.Int32(0)
            if t0 < valid_rows:
                mask = mask | cutlass.Int32(0x0000FFFF)
            if t0 + 1 < valid_rows:
                mask = mask | cutlass.Int32(0xFFFF0000)
            res_a[2 * nb] = (
                cutlass.Int32(sub_bf16x2(v_lo, pack_bf16x2(x_acc[nb][0], x_acc[nb][1])))
                & mask
            )
            res_a[2 * nb + 1] = (
                cutlass.Int32(sub_bf16x2(v_hi, pack_bf16x2(x_acc[nb][2], x_acc[nb][3])))
                & mask
            )
        warp_arrive(p64 + (BAR_RFORM + s), lane)

        # --- first-ready branch selection -------------
        # The two branches share no data, so whichever factor lands first can
        # run.  Only lane 0 probes, with TEST rather than TRY -- a try-wait
        # would suspend the warp for up to 10 ms and turn a scheduling hint
        # into a stall -- and the result is broadcast so the branch stays
        # warp-uniform.  H wins a tie: it is the next chunk's projection
        # dependency, while O can still overlap W14's store.
        bar_ak = p64 + (BAR_AK + s)
        bar_aq = p64 + (BAR_AQ + s)
        bar_proj = p64 + (BAR_PROJ + s)
        sel = cutlass.Int32(0)
        while sel == 0:
            flags = cutlass.Int32(0)
            if lane == 0:
                if mbarrier_test_wait_parity(bar_ak, pr):
                    flags = flags + 1
                if mbarrier_test_wait_parity(bar_aq, pr):
                    if mbarrier_test_wait_parity(bar_proj, pr):
                        flags = flags + 2
            flags = cutlass.Int32(cute.arch.shuffle_sync(flags, 0))
            if (flags & 1) != 0:
                sel = cutlass.Int32(1)
            elif (flags & 2) != 0:
                sel = cutlass.Int32(2)

        bar_state = p64 + (BAR_STATED + s)
        bar_out = p64 + (BAR_OUTR + s)
        if sel == 1:
            h_branch(p_akt, smem_gt, h32, h16, res_a, lane, bar_ak, bar_state, pr)
            o_branch(
                p_out,
                p_aq,
                o_acc,
                res_a,
                lane,
                v_base,
                bar_aq,
                bar_proj,
                bar_out,
                pr,
            )
        else:
            o_branch(
                p_out,
                p_aq,
                o_acc,
                res_a,
                lane,
                v_base,
                bar_aq,
                bar_proj,
                bar_out,
                pr,
            )
            h_branch(p_akt, smem_gt, h32, h16, res_a, lane, bar_ak, bar_state, pr)


# ---------------------------------------------------------------------------
# State boundary
#
# The prologue and epilogue are mirrored into the recurrence and service
# halves.  Each pair issues the SAME number of ``cta_barrier`` calls -- 1 / 1 /
# 5 for an absent / BF16 / FP32 initial state and 1 / 2 / 9 for the final one
# -- which is what makes a CTA-wide barrier legal from two divergent program
# points.  Changing one half without the other deadlocks the CTA.
# ---------------------------------------------------------------------------


@cute.jit
def state_window_view(p32):
    """Main slot 0 as a flat FP32 window: 4096 elements, exactly 16 KiB."""
    return cute.make_tensor(p32, cute.make_layout(SLOT32))


@cute.jit
def load_state_window(h32, h16, p32, lane, local_v_base):
    """Read one FP32 boundary window into H32, then derive H16.

    The FP32 image has no matrix-copy path -- ``ldmatrix`` is 16-bit only -- so
    each lane addresses its four accumulator elements individually.
    """
    window = state_window_view(p32)
    for kt in cutlass.range_constexpr(HT_TILES):
        i0 = h32t_idx(kt, 0)
        for r in cutlass.range_constexpr(4):
            k, v_local = h32t_coord(lane, kt, r, local_v_base)
            h32[i0 + r] = window[state_f32_window_idx(v_local, k)]
        j0 = h16t_idx(kt, 0)
        h16[j0] = pack_bf16x2(h32[i0], h32[i0 + 1])
        h16[j0 + 1] = pack_bf16x2(h32[i0 + 2], h32[i0 + 3])


@cute.jit
def store_state_window(h32, p32, lane, local_v_base):
    """Write H32 back into one FP32 boundary window."""
    window = state_window_view(p32)
    for kt in cutlass.range_constexpr(HT_TILES):
        i0 = h32t_idx(kt, 0)
        for r in cutlass.range_constexpr(4):
            k, v_local = h32t_coord(lane, kt, r, local_v_base)
            window[state_f32_window_idx(v_local, k)] = h32[i0 + r]


@cute.jit
def state_prologue_recurrence(
    h32,
    h16,
    p16,
    p32,
    p64,
    warp_id,
    lane,
    v_base,
    HAS_STATE_IN: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    """Bring the external state into registers (step 3 of the design)."""
    if cutlass.const_expr(not HAS_STATE_IN):
        for i in cutlass.range_constexpr(64):
            h32[i] = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(32):
            h16[i] = cutlass.Int32(0)
    elif cutlass.const_expr(STATE_FP32):
        # Four 32-value windows through main slot 0, one at a time: two
        # recurrence warps own each, and no two windows are ever live together.
        for w in cutlass.range_constexpr(STATE_F32_WINDOWS):
            cute.arch.mbarrier_wait(p64 + BAR_STATEIO, w & 1)
            if warp_id == RECURRENCE_WARP0 + 2 * w:
                load_state_window(h32, h16, p32, lane, 0)
            if warp_id == RECURRENCE_WARP0 + 2 * w + 1:
                load_state_window(h32, h16, p32, lane, WARP_VALUES)
            cute.arch.barrier()
    else:
        # One 32 KiB transaction over main slots 0 and 1; ``state_bf16_idx``
        # spans both because they are adjacent.
        cute.arch.mbarrier_wait(p64 + BAR_STATEIO, 0)
        for kt in cutlass.range_constexpr(HT_TILES):
            c0, c1 = ldmatrix_x2(p16 + state_x2t_ptr(lane, kt, v_base))
            j0 = h16t_idx(kt, 0)
            h16[j0] = c0
            h16[j0 + 1] = c1
            f0, f1 = unpack_bf16x2(c0)
            f2, f3 = unpack_bf16x2(c1)
            i0 = h32t_idx(kt, 0)
            h32[i0] = f0
            h32[i0 + 1] = f1
            h32[i0 + 2] = f2
            h32[i0 + 3] = f3
    cute.arch.barrier()


@cute.jit
def state_prologue_service(
    p16,
    p32,
    p64,
    warp_id,
    lane,
    state_plane,
    desc_state_in,
    HAS_STATE_IN: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    """The service half of :func:`state_prologue_recurrence`: W14 issues the TMA."""
    if cutlass.const_expr(not HAS_STATE_IN):
        pass
    elif cutlass.const_expr(STATE_FP32):
        for w in cutlass.range_constexpr(STATE_F32_WINDOWS):
            if warp_id == IO_WARP:
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        p64 + BAR_STATEIO, STATE_F32_WINDOW_TX_BYTES
                    )
                    tma_load_3d(
                        p32,
                        desc_state_in,
                        p64 + BAR_STATEIO,
                        0,
                        128 * w,
                        state_plane,
                    )
            cute.arch.mbarrier_wait(p64 + BAR_STATEIO, w & 1)
            cute.arch.barrier()
    else:
        if warp_id == IO_WARP:
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    p64 + BAR_STATEIO, STATE_BF16_TX_BYTES
                )
                for h in cutlass.range_constexpr(2):
                    tma_load_3d(
                        p16 + h * SLOT16,
                        desc_state_in,
                        p64 + BAR_STATEIO,
                        0,
                        128 * h,
                        state_plane,
                    )
        cute.arch.mbarrier_wait(p64 + BAR_STATEIO, 0)
    cute.arch.barrier()


@cute.jit
def state_epilogue_recurrence(
    h32,
    h16,
    p16,
    p32,
    warp_id,
    lane,
    v_base,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    """Stage the register state back out."""
    cute.arch.barrier()
    if cutlass.const_expr(HAS_STATE_OUT):
        if cutlass.const_expr(STATE_FP32):
            for w in cutlass.range_constexpr(STATE_F32_WINDOWS):
                if warp_id == RECURRENCE_WARP0 + 2 * w:
                    store_state_window(h32, p32, lane, 0)
                if warp_id == RECURRENCE_WARP0 + 2 * w + 1:
                    store_state_window(h32, p32, lane, WARP_VALUES)
                cute.arch.barrier()
                cute.arch.barrier()
        else:
            for kt in cutlass.range_constexpr(HT_TILES):
                j0 = h16t_idx(kt, 0)
                stmatrix_x2(
                    p16 + state_x2t_ptr(lane, kt, v_base),
                    h16[j0],
                    h16[j0 + 1],
                )
            cute.arch.barrier()


@cute.jit
def state_epilogue_service(
    p16,
    p32,
    warp_id,
    lane,
    state_plane,
    desc_state_out,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    """The service half of :func:`state_epilogue_recurrence`.

    The CTA barrier before the fence is what orders the recurrence warps'
    ordinary stores; the fence then publishes them to the async proxy for the
    TMA engine, which reads through a different one.
    """
    cute.arch.barrier()
    if cutlass.const_expr(HAS_STATE_OUT):
        if cutlass.const_expr(STATE_FP32):
            for w in cutlass.range_constexpr(STATE_F32_WINDOWS):
                cute.arch.barrier()
                if warp_id == IO_WARP:
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        tma_store_3d(desc_state_out, p32, 0, 128 * w, state_plane)
                        tma_store_commit_group()
                        # The next window overwrites main slot 0, so the source
                        # read has to be complete before the barrier below.
                        tma_store_wait_read(0)
                cute.arch.barrier()
        else:
            cute.arch.barrier()
            if warp_id == IO_WARP:
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    for h in cutlass.range_constexpr(2):
                        tma_store_3d(
                            desc_state_out,
                            p16 + h * SLOT16,
                            0,
                            128 * h,
                            state_plane,
                        )
                    tma_store_commit_group()
                    tma_store_wait_read(0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def fused_kda_kernel(
    gout: cute.Tensor,
    gbeta: cute.Tensor,
    ga_log: cute.Tensor,
    gdt: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_v: cutlass.Int64,
    desc_out: cutlass.Int64,
    desc_state_in: cutlass.Int64,
    desc_state_out: cutlass.Int64,
    scale: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    heads: cutlass.Int32,
    G_FP32: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    seq, head, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    # --- fixed arena ------------------------------------
    # One allocation of exactly DYNAMIC_SMEM_BYTES, so the launch parameter is
    # the plan's number and every region base is a compile-time constant.
    alloc = cutlass.utils.SmemAllocator()
    p_base = alloc.allocate(DYNAMIC_SMEM_BYTES, 1024)
    p16 = cute.recast_ptr(p_base, dtype=cutlass.BFloat16)
    p32 = cute.recast_ptr(p_base, dtype=cutlass.Float32)
    p64 = cute.recast_ptr(p_base, dtype=cutlass.Int64)

    if warp_id == TMA_WARP:
        if lane == 0:
            for s in cutlass.range_constexpr(MAIN_SLOTS):
                cute.arch.mbarrier_init(p64 + (BAR_QKG + s), ARRIVALS_TX)
                cute.arch.mbarrier_init(p64 + (BAR_V + s), ARRIVALS_TX)
                cute.arch.mbarrier_init(p64 + (BAR_MAT + s), ARRIVALS_PREPARE)
                cute.arch.mbarrier_init(p64 + (BAR_QKD + s), ARRIVALS_SINGLE)
                cute.arch.mbarrier_init(p64 + (BAR_AINV + s), ARRIVALS_SINGLE)
                cute.arch.mbarrier_init(p64 + (BAR_AQ + s), ARRIVALS_SINGLE)
                cute.arch.mbarrier_init(p64 + (BAR_AK + s), ARRIVALS_PREPARE)
                cute.arch.mbarrier_init(p64 + (BAR_PROJ + s), ARRIVALS_RECURRENCE)
                cute.arch.mbarrier_init(p64 + (BAR_RFORM + s), ARRIVALS_RECURRENCE)
                cute.arch.mbarrier_init(p64 + (BAR_OUTR + s), ARRIVALS_RECURRENCE)
                cute.arch.mbarrier_init(p64 + (BAR_OUTD + s), ARRIVALS_SINGLE)
                cute.arch.mbarrier_init(p64 + (BAR_STATED + s), ARRIVALS_RECURRENCE)
            cute.arch.mbarrier_init(p64 + BAR_STATEIO, ARRIVALS_TX)
            cute.arch.mbarrier_init_fence()
            fence_tensormap_acquire(desc_q)
            fence_tensormap_acquire(desc_k)
            fence_tensormap_acquire(desc_g)
            fence_tensormap_acquire(desc_v)
            fence_tensormap_acquire(desc_out)
            if cutlass.const_expr(HAS_STATE_IN):
                fence_tensormap_acquire(desc_state_in)
            if cutlass.const_expr(HAS_STATE_OUT):
                fence_tensormap_acquire(desc_state_out)
    # Nothing may arrive on or wait for a barrier before its state is published.
    cute.arch.barrier()

    # --- warpgroup register redistribution --------------
    # setmaxnreg is a warpgroup instruction: all four warps of a group must
    # execute the same action with the same immediate at the same point.  The
    # decreases have to land -- and be observed CTA-wide -- before any increase,
    # or the pool is momentarily oversubscribed.
    if warp_id < PREPARE_WARPS:
        setmaxnreg_dec(WG0_MAXNREG)
    if warp_id >= TMA_WARP:
        setmaxnreg_dec(WG3_MAXNREG)
    cute.arch.barrier()
    if warp_id >= RECURRENCE_WARP0:
        if warp_id < TMA_WARP:
            setmaxnreg_inc(WG1_MAXNREG)
    cute.arch.barrier()

    # --- sequence coordinates ---------------------------
    token_start = cutlass.Int32(gcu_seqlens[seq])
    token_end = cutlass.Int32(gcu_seqlens[seq + 1])
    num_chunks = (token_end - token_start + (BT - 1)) // BT
    state_plane = seq * heads + head

    # A zero-length sequence enters no chunk loop but still runs the state
    # prologue and epilogue, so its final state is its initial one.
    if warp_id >= RECURRENCE_WARP0 and warp_id < TMA_WARP:
        rec_id = warp_id - RECURRENCE_WARP0
        v_base = rec_id * WARP_VALUES
        # Declared here, not in the common region: their 96 values would
        # otherwise be live across the service warps' code and ptxas could not
        # prove the 64-register budget for WG3.
        h32 = cute.make_rmem_tensor(64, cutlass.Float32)
        h16 = cute.make_rmem_tensor(32, cutlass.Int32)
        state_prologue_recurrence(
            h32,
            h16,
            p16,
            p32,
            p64,
            warp_id,
            lane,
            v_base,
            HAS_STATE_IN,
            STATE_FP32,
        )
        recurrence_role(
            p16,
            p32,
            p64,
            h32,
            h16,
            lane,
            rec_id,
            token_start,
            token_end,
            num_chunks,
        )
        state_epilogue_recurrence(
            h32, h16, p16, p32, warp_id, lane, v_base, HAS_STATE_OUT, STATE_FP32
        )
    else:
        state_prologue_service(
            p16,
            p32,
            p64,
            warp_id,
            lane,
            state_plane,
            desc_state_in,
            HAS_STATE_IN,
            STATE_FP32,
        )
        if warp_id < PREPARE_WARPS:
            prepare_role(
                ga_log,
                gdt,
                p16,
                p32,
                p64,
                tidx,
                lane,
                warp_id,
                head,
                token_start,
                token_end,
                num_chunks,
                scale,
                gate_scale_log2,
                G_FP32,
                SAFE_GATE,
            )
        elif warp_id == TMA_WARP:
            tma_role(
                p16,
                p32,
                p64,
                lane,
                head,
                token_start,
                num_chunks,
                desc_q,
                desc_k,
                desc_g,
                desc_v,
                G_FP32,
            )
        elif warp_id == QK_WARP:
            qk_role(p16, p64, lane, num_chunks)
        elif warp_id == IO_WARP:
            io_role(
                gout,
                p16,
                p64,
                lane,
                head,
                heads,
                token_start,
                token_end,
                num_chunks,
                desc_out,
            )
        else:
            inverse_role(
                gbeta,
                p16,
                p64,
                lane,
                head,
                heads,
                token_start,
                token_end,
                num_chunks,
            )
        state_epilogue_service(
            p16,
            p32,
            warp_id,
            lane,
            state_plane,
            desc_state_out,
            HAS_STATE_OUT,
            STATE_FP32,
        )


# --------------------------------------------------------------------------
# Section 6: compiled entry, descriptor cache and launch
# --------------------------------------------------------------------------

#: Devices already proven to be CC 12.0.  ``get_device_capability`` is a driver


@dataclass(frozen=True)
class KernelKey:
    """the design's ordered compile key.  Fields may not be merged."""

    device: int
    device_cc: tuple[int, int]
    code_target: str
    g_dtype: torch.dtype
    safe_gate: bool
    state_dtype: torch.dtype | None
    has_initial_state: bool
    has_final_state: bool
    input_mode: str

    def __post_init__(self) -> None:
        has_state = self.has_initial_state or self.has_final_state
        if has_state != (self.state_dtype is not None):
            raise ValueError(
                "state_dtype must be present exactly when a state is present: "
                f"initial={self.has_initial_state}, final={self.has_final_state}, "
                f"dtype={self.state_dtype}"
            )
        if self.input_mode not in ("fixed", "packed"):
            raise ValueError(f"unknown input mode {self.input_mode!r}")


_KERNEL_CACHE: dict[KernelKey, object] = {}
_DESCRIPTOR_CACHE = BoundedDeviceCache("kda-descriptors")


def clear_kernel_caches() -> None:
    """Drop the compile and descriptor caches.

    Not the whole variant: :func:`clear_caches` below does that.  This is the
    device-side half, which a test that wants a cold compile needs on its own.
    """
    _KERNEL_CACHE.clear()
    _DESCRIPTOR_CACHE.clear()


def kernel_cache_size() -> int:
    return len(_KERNEL_CACHE)


def descriptor_cache_stats(device):
    return _DESCRIPTOR_CACHE.stats(device)


def build_descriptors(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    out: torch.Tensor,
    initial_state: torch.Tensor | None,
    final_state: torch.Tensor | None,
    total_tokens: int,
    sequences: int,
    heads: int,
) -> TensorMapUpload:
    """Encode (or reuse) the seven The descriptors."""
    specs: dict[str, TensorMapSpec] = {
        "q": activation_spec(q, total_tokens, heads),
        "k": activation_spec(k, total_tokens, heads),
        "g": activation_spec(g, total_tokens, heads),
        "v": activation_spec(v, total_tokens, heads),
        "out": activation_spec(out, total_tokens, heads),
    }
    if initial_state is not None:
        specs["state_in"] = state_spec(initial_state, sequences, heads)
    if final_state is not None:
        specs["state_out"] = state_spec(final_state, sequences, heads)

    device = q.device
    key = tuple(
        (role, specs[role].key(device)) for role in DESCRIPTOR_ROLES if role in specs
    )
    hit = _DESCRIPTOR_CACHE.get(device, key)
    if hit is not None:
        return hit
    if capturing():
        raise RuntimeError(
            "CUDA graph capture requires a TMA descriptor cache hit; run one "
            "eager fwd with the same tensors before capturing"
        )
    upload = build_upload(specs, device)
    return _DESCRIPTOR_CACHE.put(device, key, upload, (upload.storage,))


def descriptor_cache_key(specs: dict[str, TensorMapSpec], device) -> tuple:
    """The key :func:`build_descriptors` would use.  For capture pre-checks."""
    return tuple(
        (role, specs[role].key(device)) for role in DESCRIPTOR_ROLES if role in specs
    )


#: The traced entry, built on first use.  ``@cute.jit`` decoration is not free
#: and the result is a constant, so rebuilding it per call was pure waste.
_ENTRY: tuple | None = None


#: Whether to hand ptxas an entry register count, which is the precondition for
#: the manual W0-W15 register split to survive to SASS at all.
#:
#: ``min_blocks_per_mp=1`` compiles to ``__launch_bounds__(512, 1)``, from which
#: ptxas derives 65,536 / 512 = 128 registers per thread at entry.  That is not
#: the final allocation -- it is the baseline the three ``setmaxnreg``
#: instructions move away from, down to 120 and 64 and up to 160.  Without it
#: ptxas reports ``(C7508) 'setmaxnreg' ignored; unable to determine register
#: count at entry`` and holds every warp at 128, which is the uniform split the
#: design does not want.
#:
#: Off by default until the pairing is measured: the bound acting *alone*, with
#: the redistribution still dropped, was measured at 6,457 local-memory
#: instructions against 240 (see :mod:`config`).  Set ``KDA_LAUNCH_BOUNDS=1`` to
#: build the other variant.
_LAUNCH_BOUNDS = (
    {"min_blocks_per_mp": 1} if os.environ.get("KDA_LAUNCH_BOUNDS") == "1" else {}
)


def _entry_module():
    """Import the device kernel lazily and build the entry once.

    Host-only consumers -- validation, metadata, the layout tests -- must not
    need the CUTLASS DSL installed, and importing it costs seconds, so this
    stays out of module import.  The memo is what keeps it off the hot path.

    This is also why this module must not carry ``from __future__ import
    annotations``.  ``cutlass`` is a local of this function, not a module
    global, so ``_fused_entry``'s parameter annotations below can only be
    resolved as real objects at ``def`` time.  Postponed annotations would
    leave them as the strings ``"cutlass.Int64"`` and friends, and the DSL
    resolves a jit function's signature with ``inspect.get_annotations(...,
    eval_str=True)``, which evaluates them against this module's globals --
    where ``cutlass`` is not bound.  DSL 4.3 never looked; 4.7 does, and every
    GPU test failed with ``NameError: name 'cutlass' is not defined`` raised
    from inside ``inspect``.
    """
    global _ENTRY
    if _ENTRY is not None:
        return _ENTRY
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def _fused_entry(
        gout,
        gbeta,
        ga_log,
        gdt,
        gcu_seqlens,
        desc_q: cutlass.Int64,
        desc_k: cutlass.Int64,
        desc_g: cutlass.Int64,
        desc_v: cutlass.Int64,
        desc_out: cutlass.Int64,
        desc_state_in: cutlass.Int64,
        desc_state_out: cutlass.Int64,
        scale: cutlass.Float32,
        gate_scale_log2: cutlass.Float32,
        heads: cutlass.Int32,
        grid_x: cutlass.Int32,
        grid_y: cutlass.Int32,
        stream,
        G_FP32: cutlass.Constexpr,
        SAFE_GATE: cutlass.Constexpr,
        HAS_STATE_IN: cutlass.Constexpr,
        HAS_STATE_OUT: cutlass.Constexpr,
        STATE_FP32: cutlass.Constexpr,
    ):
        fused_kda_kernel(
            gout,
            gbeta,
            ga_log,
            gdt,
            gcu_seqlens,
            desc_q,
            desc_k,
            desc_g,
            desc_v,
            desc_out,
            desc_state_in,
            desc_state_out,
            scale,
            gate_scale_log2,
            heads,
            G_FP32,
            SAFE_GATE,
            HAS_STATE_IN,
            HAS_STATE_OUT,
            STATE_FP32,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(THREADS, 1, 1),
            smem=DYNAMIC_SMEM_BYTES,
            stream=stream,
            **_LAUNCH_BOUNDS,
        )

    _ENTRY = (cutlass, cute, _fused_entry)
    return _ENTRY


@dataclass(frozen=True)
class LaunchPlan:
    """Everything a repeated launch needs, minus the launch itself.

    The whole host path is a pure function of the tensors' addresses, shapes,
    dtypes and the two scalars, so on a training loop that reuses its buffers
    this can be built once and replayed.  ``descriptors`` is held by strong
    reference: the argument tuple carries raw device addresses into it, and
    letting the upload die would leave the kernel reading freed memory.

    ``stream`` is baked into ``args``, which is why the cache above this must
    key on it -- replaying a plan from another stream would launch against the
    wrong one, correctly ordered against the wrong work.
    """

    key: KernelKey
    descriptors: TensorMapUpload
    compiled: Any
    args: tuple

    def run(self) -> None:
        self.compiled(*self.args)


def prepare_launch(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens_i32: torch.Tensor,
    scale: float,
    lower_bound: float,
    sequences: int,
    heads: int,
    total_tokens: int,
    initial_state: torch.Tensor | None,
    final_state: torch.Tensor | None,
    state_dtype: torch.dtype | None,
    safe_gate: bool,
    input_mode: str,
    stream=None,
) -> LaunchPlan:
    """Compile (or reuse) and assemble everything one fused forward needs.

    Does not launch.  Splitting this from the launch is what lets the caller
    memoize it; it also keeps the graph-capture guards -- which must run on a
    cold cache -- on the path that actually builds things.
    """
    import cuda.bindings.driver as cuda_driver

    require_sm120a(q.device)
    # ``out_global_index`` feeds ``vec_at``, which casts to ``Int32``, and the
    # DSL packs the flat view's extent as INT32 as well -- so the shape is
    # bounded here, before anything is allocated.  This half had no such check
    # at all, and the wrapped index is not an error but a negative offset.
    check_flat_output_range(total_tokens, heads)
    cutlass, _cute, entry = _entry_module()

    if stream is None:
        # The tensors' device, not the current one.  torch.cuda.current_stream()
        # with no argument returns the *current* device's stream, so with the
        # inputs on cuda:1 and cuda:0 current it hands a device-0 handle to a
        # device-1 launch -- invalid resource handle at best, silently wrong
        # stream ordering at worst, and neither is visible at the call site.
        stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    descriptors = build_descriptors(
        q=q,
        k=k,
        v=v,
        g=g,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        total_tokens=total_tokens,
        sequences=sequences,
        heads=heads,
    )

    grid_x, grid_y, _ = grid(sequences, heads)
    # grid_y is the head count, and nothing upstream bounds it against the
    # device.  The decomposed variant checks its own grid (recurrence.py); this
    # one did not, so a call with more heads than maxGridSize[1] -- 65535 on
    # every current part, against 2^31-1 on axis 0 -- reached the driver and
    # failed there.  A short, stateless call at H = 65536 fits in memory
    # perfectly well, so nothing else would have stopped it.

    limits = max_grid_dims(q.device)
    for axis, (extent, limit) in enumerate(zip((grid_x, grid_y), limits, strict=False)):
        if extent > limit:
            raise ValueError(
                f"grid[{axis}] = {extent} exceeds maxGridSize[{axis}] = {limit}"
            )
    # ``lower_bound * log2e`` is computed once in Python double and cast once,
    # so the device never repeats the conversion and lower_bound stays out of
    # the compile key.
    args = (
        flat_view(out),
        flat_view(beta),
        flat_view(A_log),
        flat_view(dt_bias),
        flat_view(cu_seqlens_i32),
        cutlass.Int64(descriptors.address("q")),
        cutlass.Int64(descriptors.address("k")),
        cutlass.Int64(descriptors.address("g")),
        cutlass.Int64(descriptors.address("v")),
        cutlass.Int64(descriptors.address("out")),
        cutlass.Int64(descriptors.address("state_in")),
        cutlass.Int64(descriptors.address("state_out")),
        cutlass.Float32(scale),
        cutlass.Float32(lower_bound * LOG2_E),
        cutlass.Int32(heads),
        cutlass.Int32(grid_x),
        cutlass.Int32(grid_y),
        stream,
    )
    constexprs = (
        g.dtype is torch.float32,
        bool(safe_gate),
        initial_state is not None,
        final_state is not None,
        state_dtype is torch.float32,
    )

    key = KernelKey(
        device=(
            q.device.index
            if q.device.index is not None
            else torch.cuda.current_device()
        ),
        device_cc=DEVICE_CC,
        code_target=CODE_TARGET,
        g_dtype=g.dtype,
        safe_gate=bool(safe_gate),
        state_dtype=state_dtype,
        has_initial_state=initial_state is not None,
        has_final_state=final_state is not None,
        input_mode=input_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        if capturing():
            raise RuntimeError(
                "CUDA graph capture cannot compile; run one eager fwd with the "
                "same KernelKey before capturing"
            )

        def _compile():
            import cutlass.cute as cute

            # Subscript, not ``options=``: the keyword form silently drops
            # EnableTVMFFI and hands back a ctypes-marshalled callable.
            compiled = cute.compile[sm120a_compile_options()](entry, *args, *constexprs)
            return assert_tvm_ffi_dispatched(compiled, kernel_name(key))

        # The specialization name carries every compile-time parameter and
        # nothing that varies per call: a tensor address or a runtime shape
        # here would be a cache that never hits.
        compiled = build_kernel(
            kernel_name(key), _compile, device=q.device, key_files=(__file__,)
        )
        _KERNEL_CACHE[key] = compiled
    return LaunchPlan(key=key, descriptors=descriptors, compiled=compiled, args=args)


def launch_fused(**kwargs) -> tuple[KernelKey, TensorMapUpload, object]:
    """Prepare and launch in one step.  Used by tests and one-shot callers."""
    plan = prepare_launch(**kwargs)
    plan.run()
    return plan.key, plan.descriptors, plan.compiled


# --------------------------------------------------------------------------
# Code target and specialization naming.
#
# The kernel is written against ``sm_120a``'s architecture-specific instruction
# set, so the target is a contract rather than a coincidence of whatever the
# DSL detected.  It is stated as an explicit compile option, never by writing
# ``CUTE_DSL_ARCH``; ``runtime.build_kernel`` refuses to write an artifact if
# the persistent cache resolves a different target than the one asked for.
# --------------------------------------------------------------------------


def code_target() -> str:
    """The code target this variant compiles for.  For reports and cache keys."""
    return SM120_CODE_TARGET


def kernel_name(key: "KernelKey") -> str:
    """The persistent cache's specialization name for ``key``.

    Every compile-time parameter and nothing else.  Deliberately readable: it
    is the name of a directory entry someone will have to recognize when asking
    which specializations a run built.
    """
    if key.state_dtype is None:
        state = "nostate"
    elif key.state_dtype is torch.float32:
        state = "statefp32"
    else:
        state = "statebf16"
    return "fused_" + "_".join(
        (
            "gfp32" if key.g_dtype is torch.float32 else "gbf16",
            "safegate" if key.safe_gate else "rawgate",
            state,
            "si" if key.has_initial_state else "nosi",
            "so" if key.has_final_state else "noso",
            key.input_mode,
        )
    )


# --------------------------------------------------------------------------
# Section 7: the host path
#
# Structurally the same idea as the decomposed variant's: an LRU on a
# tensor- and workspace-identity key, an object-identity fast path in front of
# it, a state-only shortcut, and the stream in the key because a plan bakes its
# ``CUstream`` into the argument tuple.
#
# The two are not folded together, and the difference is load-bearing rather
# than stylistic.  This half carries ``safe_gate``, which the other refuses
# outright; it has one descriptor set where the other has two plus a shared
# factor slab; and it has no chunk tables at all, because a fused CTA owns a
# whole (sequence, head) and never needs a chunk-to-sequence map.  A merged
# "call plan cache" would have to carry the union of those and branch on the
# variant at every step, which is the same code with an extra way to be wrong.
#
# Three things are deliberately not cached, because caching them would be wrong
# rather than merely slow:
#
# * the zero-token state copy, which is real work the caller expects on every
#   call -- the plan records that it is a state-only call and redoes it;
# * anything derived from tensor *contents*;
# * plans across streams, for the reason above.
# --------------------------------------------------------------------------

#: One entry per distinct set of buffers a caller uses; a serving loop that
#: reuses its activations needs exactly one.

#: 16 is a ceiling on *how many rotating buffer sets stay fast*, not a memory
#: budget, and lowering it does not trade speed for memory -- it buys nothing
#: until the workload exceeds it and then costs everything.  Measured on the
#: 110-SM part at ``[1, 1024, 8, 128]``, rotating N buffer sets:
#:
#: ===  ==================  ==================
#: cap  N = 4               N = 8
#: ===  ==================  ==================
#: 16   101 us / 40.6 MiB   116 us / 72.7 MiB
#: 4    105 us / 40.6 MiB   7145 us / 40.6 MiB
#: 2    7670 us / 24.6 MiB  7546 us / 24.6 MiB
#: ===  ==================  ==================
#:
#: Below the cap the retention is identical whatever the cap is; at the cap the
#: hit becomes a rebuild, and a rebuild is ~7.3 ms against a ~100 us hit.  A
#: deployment that needs the memory back should reduce how many buffer sets it
#: rotates, or call ``clear_kda_prefill_sm120_caches()``; shrinking this number
#: only moves the same allocation into a 70x slower path.
CALL_PLAN_MAX_ENTRIES = 16
_CALL_PLANS: OrderedDict = OrderedDict()

#: The previous call's tensors, versions, scalars, workspace, stream and plan. Strong
#: references, one slot: keying on ``id()`` would be unsound, since a freed
#: tensor's id can be reused by a different one and the plan would then be
#: handed to the wrong buffers, and weak references would cost a dereference
#: per tensor to save pinning a single call's worth.
_LAST: tuple | None = None

#: Marks "this call has no tokens", so the zero-token path is reached on a plan
#: hit without re-deriving the metadata that proves it.
_STATE_ONLY = object()

#: Serializes the cache-miss path: plan construction, descriptor encoding and
#: the compile behind it.
#:
#: The launch lock below is not enough on its own, and the reason is worth
#: recording. Four host threads issuing the same shape share one factor arena
#: and one launch lock, so their *enqueues* interleave correctly -- but a cold
#: build is not an enqueue. It encodes descriptors, writes shared scratch and
#: calls into the CuTe DSL compiler, none of which is re-entrant: the legacy
#: implementation, given the same four threads, does not produce wrong numbers
#: so much as raise ``_fwd_entry() requires a code object with 0 free vars`` and
#: "Please start a session before accessing session data" from inside the DSL.
#: Measured here, the first thread's output was wrong in 1515 of 32768 elements
#: while the other three were exact -- the signature of the builder racing its
#: own build.
#:
#: A module-level lock is the right shape for that. It is taken only on a miss,
#: so a serving loop that reuses its buffers never sees it, and the two cache
#: layers in front of it are checked before it is acquired.
_BUILD_LOCK = threading.RLock()


def clear_caches() -> None:
    """Drop every cache this variant owns."""
    global _LAST
    _CALL_PLANS.clear()
    _LAST = None
    clear_kernel_caches()


def call_plan_stats() -> dict:
    return {"plans": len(_CALL_PLANS), "last_call_warm": _LAST is not None}


def _identity(device, tensors, scale, lower_bound, safe_gate, resources) -> tuple:
    # The stream of the *inputs'* device: ``prepare_launch`` bakes
    # ``torch.cuda.current_stream(q.device)`` into the plan's argument tuple,
    # so a key built from the current device's stream would let two streams on
    # the input's device share one entry and reuse the first one's plan.
    return (
        tuple(tensor_identity(t) for t in tensors),
        float(scale),
        float(lower_bound),
        bool(safe_gate),
        resource_cache_token(resources),
        current_stream_ptr(device),
    )


def _fast_path(device, tensors, scale, lower_bound, safe_gate, resources):
    """The previous call's plan if this call is identical to it, else ``None``."""
    if _LAST is None:
        return None
    (
        last_tensors,
        last_versions,
        last_scalars,
        last_resources,
        last_stream,
        plan,
    ) = _LAST
    if last_scalars != (float(scale), float(lower_bound), bool(safe_gate)):
        return None
    if last_resources is not resource_cache_token(resources):
        return None
    if last_stream != current_stream_ptr(device):
        return None
    if len(last_tensors) != len(tensors):
        return None
    for ref, tensor in zip(last_tensors, tensors, strict=True):
        if ref is None:
            if tensor is not None:
                return None
        elif ref() is not tensor:
            return None
    for tensor, version in zip(tensors, last_versions, strict=True):
        if tensor is not None and tensor_version(tensor) != version:
            return None
    return plan


def _remember(device, tensors, scale, lower_bound, safe_gate, resources, plan) -> None:
    global _LAST
    # Weak, like the plan LRU above: this entry outlives the call, and strong
    # references to q, k, v, g and out would hold one whole activation set off
    # the caching allocator until the next call replaced it.
    _LAST = (
        tuple(None if t is None else weakref.ref(t) for t in tensors),
        tuple(None if t is None else tensor_version(t) for t in tensors),
        (float(scale), float(lower_bound), bool(safe_gate)),
        resource_cache_token(resources),
        current_stream_ptr(device),
        plan,
    )


def _state_only(initial_state, final_state) -> None:
    """The zero-token path.

    No kernel launches and no zero-extent descriptor is built: a TensorMap with
    a zero global dimension is invalid, and there is nothing for it to describe.
    """
    if final_state is None:
        return
    if initial_state is None:
        final_state.zero_()
        return
    if is_exact_alias(initial_state, final_state):
        return
    final_state.copy_(initial_state)


def _packed(t: torch.Tensor | None) -> torch.Tensor | None:
    """Fixed ``[B, T, ...]`` -> packed ``[1, B * T, ...]``, as a view.

    ``reshape`` on a contiguous tensor never copies, which matters here: a copy
    of ``out`` would silently drop the caller's writes.
    """
    if t is None:
        return None
    return t.reshape(1, t.shape[0] * t.shape[1], *t.shape[2:])


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
) -> None:
    """Launch the fused prefill, writing ``out`` and ``final_state`` in place.

    ``info`` and ``offsets`` come from :mod:`.runtime`: the facade validates and
    canonicalizes once, so this is not a second validation pass.
    """
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

    plan = _fast_path(q.device, tensors, scale, lower_bound, safe_gate, resources)
    if plan is None:
        key = _identity(q.device, tensors, scale, lower_bound, safe_gate, resources)
        # See ``_BUILD_LOCK``: the miss path is not re-entrant, and the two
        # cache layers in front of it mean a warm caller never reaches it.
        with _BUILD_LOCK:
            plan = _CALL_PLANS.get(key)
            if plan is not None:
                _CALL_PLANS.move_to_end(key)
            else:
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
                    cu_seqlens=cu_seqlens,
                    info=info,
                    offsets=offsets,
                    resources=resources,
                    safe_gate=safe_gate,
                )
                _CALL_PLANS[key] = plan
                while len(_CALL_PLANS) > CALL_PLAN_MAX_ENTRIES:
                    _CALL_PLANS.popitem(last=False)
            _remember(q.device, tensors, scale, lower_bound, safe_gate, resources, plan)

    execute(plan, initial_state, final_state)
    return plan


def execute(plan, initial_state, final_state) -> None:
    """Run an already-resolved plan.

    Split out so the facade, which has its own memo on the same tensor
    identities, does not have to repeat the comparison this module's fast path
    would do to find the same plan again. Two layers each walking eleven
    tensors cost about 6 us per call -- nothing against a 1 ms kernel, and a
    third of the whole call at B=1 T=16 H=4.
    """
    if plan is _STATE_ONLY:
        # Real work the caller expects on every call, so a hit redoes it.
        _state_only(initial_state, final_state)
        return
    plan.run()


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
    cu_seqlens,
    info,
    offsets,
    resources,
    safe_gate,
):
    """The full host path: canonicalize, encode descriptors, marshal arguments.

    Everything that must run on a cold cache -- and therefore every
    graph-capture guard -- lives here.
    """
    require_sm120a(q.device)

    if info.total_tokens == 0:
        return _STATE_ONLY

    if info.input_mode == "fixed":
        pq, pk, pv, pg = (_packed(t) for t in (q, k, v, g))
        pbeta = _packed(beta)
        pout = _packed(out)
    else:
        if capturing() and cu_seqlens.dtype is not torch.int32:
            # Even a previously converted INT64 tensor is refused *without a
            # workspace*: replay would reuse the canonical buffer without
            # re-validating the source, so the two could drift apart with
            # nothing to detect it.  With a workspace the canonical buffer is
            # the workspace's own and replay copies into it, which is what
            # makes INT64 packed capture supportable at all.
            if resources is None or resources.cu_seqlens_i32 is None:
                raise RuntimeError(
                    "CUDA graph capture of INT64 packed offsets needs an "
                    "explicit RecurrentKDAPrefillWorkspace warmed on the same "
                    f"offsets; got {cu_seqlens.dtype} with no workspace"
                )
        pq, pk, pv, pg, pbeta, pout = q, k, v, g, beta, out

    if offsets.sequences != info.sequences:
        raise KDAPrefillValidationError(
            f"cu_seqlens describes {offsets.sequences} sequences but the state "
            f"shapes describe {info.sequences}"
        )

    cu_seqlens_i32 = _stable_offsets(offsets, resources)

    plan = prepare_launch(
        q=pq,
        k=pk,
        v=pv,
        g=pg,
        beta=pbeta,
        out=pout,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens_i32=cu_seqlens_i32,
        scale=float(scale),
        lower_bound=float(lower_bound),
        sequences=info.sequences,
        heads=info.heads,
        total_tokens=info.total_tokens,
        initial_state=initial_state,
        final_state=final_state,
        state_dtype=info.state_dtype,
        safe_gate=bool(safe_gate),
        input_mode=info.input_mode,
    )

    if resources is not None:
        # Replay never re-enters Python, so everything the capture recorded has
        # to stay alive at its captured address for the workspace's lifetime.
        resources.pin(
            plan,
            plan.compiled,
            plan.descriptors,
            plan.descriptors.storage,
            offsets,
            offsets.canonical,
            offsets.source,
            cu_seqlens_i32,
        )
    elif capturing():
        # A capture without a workspace has nowhere to put the pins, so they go
        # to the process-wide table.  Deliberately for the process lifetime: an
        # eviction that left a replayed graph reading a dangling device pointer
        # fails far from its cause and only sometimes.
        GRAPH_PINS.pin(
            (plan.key, id(plan.descriptors)),
            plan.compiled,
            plan.descriptors,
            plan.descriptors.storage,
            offsets,
            offsets.canonical,
            offsets.source,
        )
    return plan


def _stable_offsets(offsets, resources):
    """The INT32 offsets the kernel reads, at an address replay can rely on.

    Without a workspace this is whatever ``runtime`` canonicalized. With one,
    eager warmup copies into a fixed-address workspace buffer that remains
    valid for graph replay; the documented capture contract keeps offset values
    unchanged for that graph's lifetime.
    """
    if resources is None:
        return offsets.canonical
    buffer = resources.ensure_capacity(
        "cu_seqlens_i32", offsets.canonical.numel(), torch.int32
    )
    buffer.copy_(offsets.canonical)
    return buffer


def dynamic_smem_bytes() -> int:
    """The launch's dynamic shared-memory size, for host-side resource checks."""
    return DYNAMIC_SMEM_BYTES


__all__ = [
    "CALL_PLAN_MAX_ENTRIES",
    "KernelKey",
    "LaunchPlan",
    "call_plan_stats",
    "clear_caches",
    "clear_kernel_caches",
    "code_target",
    "dynamic_smem_bytes",
    "kernel_name",
    "execute",
    "run",
]
