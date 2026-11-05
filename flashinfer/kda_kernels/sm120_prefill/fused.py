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
swizzles, its variant-specific inline PTX, its TMA descriptors, the device
kernel, the compiled entry, its descriptor and call-plan caches, and its
current-stream launch.  What it shares with :mod:`.decomp` comes from two
sibling modules: host mechanisms from :mod:`.runtime`, and the device-side PTX
wrappers, fragment constants and S128 geometry from :mod:`.device_common`.

The two variants do not import one another.  A same-named helper whose
specialization, rounding or barrier ownership differs between them -- the fused
``pairwise_a_ptr`` and the decomp ``pairwise_a_fragment_ptr`` address different
images -- stays in its variant rather than becoming one name over two meanings.

The 512-thread CTA topology, the warp role ownership, the SMEM arena, the gate
numeric boundaries and the state/output alias contract are fixed implementation
choices shared by the host plan, device code and tests.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
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
    KDAPrefillValidationError,
    LOG2_E,
    max_grid_dims,
    NORM_FLOOR,
    PlanMemo,
    PREFIX_FLOOR,
    require_sm120a,
    SM120_CODE_TARGET,
    sm120a_compile_options,
    STATE_ONLY_PLAN,
    TensorMapSpec,
)
from .device_common import (
    BF16_SEGMENT_ELEMS,
    BF16_SEGMENT_STRIDE,
    BT,
    F32_GROUP_ELEMS,
    F32_SEGMENT_ELEMS,
    F32_SEGMENT_STRIDE,
    KEY_BLOCKS,
    STATE_BF16_ROWS_PER_VALUE,
    STATE_F32_ROWS_PER_VALUE,
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
    mma_c_coord,
    mma_n8,
    movmatrix_b16,
    mul_bf16x2,
    pack_bf16x2,
    pack_f16x2,
    pairwise_sw32,
    raw_bf16_s128,
    raw_f32_s128,
    state_bf16_idx,
    stmatrix_x2,
    stmatrix_x2_trans,
    stmatrix_x4,
    store_vec8_bf16,
    sub_bf16x2,
    tma_load_3d,
    tma_store_3d,
    tma_store_commit_group,
    tma_store_wait_read,
    unpack_bf16x2,
    vec8_bf16,
    vec_at,
    warp_arrive,
)


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

#: The kernel guards on this exactly; other capabilities do not fall back here.
DEVICE_CC = (12, 0)

# ---------------------------------------------------------------------------
# CTA shape and warp ownership
# ---------------------------------------------------------------------------

THREADS = 512
WARPS = 16

#: W0-W3 gate, normalize and materialize Qd/Kd/Ki.
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


# ---------------------------------------------------------------------------
# Warpgroup register budgets
#
# The ``setmaxnreg`` immediates the kernel issues after barrier init.  They are
# requested, not granted: the PTX carries no ``.maxnreg`` entry bound, so ptxas
# cannot compute the post-dec/inc budget and drops the instructions.  Every
# warp therefore runs on the uniform launch budget of 128 registers per thread
# (``THREADS * 128`` is the whole register file), and a recurrence warp keeps
# its 96 registers of state (h32's 64 plus h16's 32) inside that.
# ---------------------------------------------------------------------------
WG0_MAXNREG = 120  # W0-W3   prepare
WG1_MAXNREG = 160  # W4-W11  recurrence
WG3_MAXNREG = 64  # W12-W15 TMA, QK/Aq, I/O, KK/inverse


# ---------------------------------------------------------------------------
# SMEM arena
# ---------------------------------------------------------------------------

#: Four: with three, the TMA producer sat on the slot-free wait while prepare
#: waited on ``qkg_ready`` -- it was short of a slot to write into, not of
#: bandwidth.  The fourth slot fits in the arena the CTA already owns.
MAIN_SLOTS = 4
MAIN_SLOT_BYTES = 16384

#: Must equal :data:`MAIN_SLOTS`.  ``tma_role`` and ``recurrence_role`` index
#: the V stage with ``main_slot(c)``, the *main* slot index, so a shorter V
#: ring is an out-of-bounds read rather than a smaller ring.
V_STAGES = MAIN_SLOTS
V_STAGE_BYTES = 4096

V_STAGE_OFFSET = MAIN_SLOTS * MAIN_SLOT_BYTES  # 65536
CONTROL_OFFSET = V_STAGE_OFFSET + V_STAGES * V_STAGE_BYTES  # 81920
CONTROL_BYTES = 1024

DYNAMIC_SMEM_BYTES = CONTROL_OFFSET + CONTROL_BYTES  # 82944


# ---------------------------------------------------------------------------
# Main slot phases, relative to ``MAIN_SLOT_BYTES * main_slot(c)``
# ---------------------------------------------------------------------------

SLOT_K = 4096  # Kraw -> Kd
SLOT_G_LO = 8192  # raw G / E lower -> Ki -> Ak.T

SLOT_KD = SLOT_K
SLOT_KI = SLOT_G_LO
SLOT_AKT = SLOT_G_LO

#: Factor records, written only once all four prepare warps have captured E.
SLOT_AINV_BETA = 12288  # [16, 16] BF16 SW32, 512 B
SLOT_AQ = 12800  # [16, 16] BF16 SW32, 512 B
SLOT_GTOTAL = 13312  # [128]    FP32,      512 B

# ---------------------------------------------------------------------------
# Control arena, relative to CONTROL_OFFSET.  It holds the mbarriers, from
# offset 0; the rest of it is unused.
# ---------------------------------------------------------------------------

#: One mbarrier per main slot per event, eight bytes each.  Derived from the
#: slot count so that raising ``MAIN_SLOTS`` cannot overlap one event's
#: barriers with the next event's.
_BAR_STRIDE = MAIN_SLOTS * 8  # 32

#: Byte offset of each event's barrier group, relative to ``CONTROL_OFFSET``.
#: Every event but ``state_io`` has one mbarrier per main slot.  ``qkg_ready``,
#: ``v_ready`` and ``state_io`` are transaction barriers: arrival count 1,
#: completed by ``arrive_expect_tx``.  Group index 3 is unused.
BAR_QKG_READY = 0 * _BAR_STRIDE
BAR_V_READY = 1 * _BAR_STRIDE
BAR_MATERIALIZED = 2 * _BAR_STRIDE
BAR_AINV_READY = 4 * _BAR_STRIDE
BAR_AQ_READY = 5 * _BAR_STRIDE
BAR_AK_READY = 6 * _BAR_STRIDE
BAR_PROJECTION_DONE = 7 * _BAR_STRIDE
BAR_R_FORMED = 8 * _BAR_STRIDE
BAR_OUTPUT_READY = 9 * _BAR_STRIDE
BAR_OUTPUT_READ_DONE = 10 * _BAR_STRIDE
BAR_STATE_DONE = 11 * _BAR_STRIDE
BAR_STATE_IO = 12 * _BAR_STRIDE

#: Arrival counts.  Transaction barriers take 1.
ARRIVALS_TX = 1
ARRIVALS_PREPARE = PREPARE_WARPS  # 4
ARRIVALS_RECURRENCE = RECURRENCE_WARPS  # 8
ARRIVALS_SINGLE = 1

#: ``ainv_ready`` counts two: W15 arrives when AinvBeta exists, W13 when it has
#: finished reading ``Ki``.  The Ak.T stores need *both* before they may
#: overwrite Ki, so one barrier carries both conditions.
ARRIVALS_AINV_READY = 2

#: ``ak_ready`` counts two: W13 and W14 publish two Ak.T strips each.  The 2/2
#: split is also an SMSP split; putting all four strips on one warp slowed
#: every warp sharing its SMSP.  See ``qk_role`` and ``io_role``.
ARRIVALS_AK = 2


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

SOFTPLUS_CUT = 20.0

#: ``lower_bound`` is validated against this closed interval, not compiled in.
LOWER_BOUND_RANGE = (-5.0, 0.0)

# ---------------------------------------------------------------------------
# Slot and phase algebra
# ---------------------------------------------------------------------------


def main_slot(chunk: int) -> int:
    """Main slot and V stage index of chunk ``chunk``.

    Spelled without ``%`` so the identical expression evaluates for a Python
    ``int`` on the host and for a ``cutlass.Int32`` on the device.
    """
    return chunk - MAIN_SLOTS * (chunk // MAIN_SLOTS)


def generation(chunk: int) -> int:
    """How many times chunk ``chunk``'s main slot has been used before it."""
    return chunk // MAIN_SLOTS


def ready_parity(chunk: int) -> int:
    """Phase a *consumer* waits on for chunk ``chunk``'s events.

    ``mbarrier_wait(bar, p)`` passes when the barrier's current phase differs
    from ``p``, so one wait per generation alternates 0, 1, 0, ...
    """
    return (chunk // MAIN_SLOTS) & 1


def reuse_parity(chunk: int) -> int:
    """Phase a *producer* waits on before overwriting chunk ``chunk``'s main slot.

    Generation 0 yields 1, which passes immediately against a freshly
    initialized phase-0 barrier.  That is what lets W12 run one uniform loop
    from ``c = 0`` with no prologue special case.
    """
    return 1 ^ ((chunk // MAIN_SLOTS) & 1)


def grid(sequences: int, heads: int) -> tuple[int, int, int]:
    """One CTA per ``(sequence, head)``."""
    return (sequences, heads, 1)


# --------------------------------------------------------------------------
# SMEM images and fragment maps
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S128 image parameters
# ---------------------------------------------------------------------------


#: Constant coordinate permutations.  A constant XOR of a *coordinate* is not
#: expressible as a CuTe ``Swizzle`` (which only folds higher offset bits into
#: lower ones), so these stay explicit transforms outside the layout objects.
AK_TOKEN_XOR = BT // 2  # 8


# ---------------------------------------------------------------------------
# SMEM images
# ---------------------------------------------------------------------------


def ak_t_s128(token, key):
    """BF16 element index of ``Ak.T[token, key]``.

    ``Ak`` is logically ``[K, token] = [128, 16]`` and is published transposed
    with a ``token ^ 8`` row permutation.  That permutation is what lets the
    recurrence recover the ``[key, token]`` A operand with a single
    ``ldmatrix.x4.trans`` and no register shuffle afterwards.
    """
    return raw_bf16_s128(token ^ AK_TOKEN_XOR, key)


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


# ---------------------------------------------------------------------------
# H register layout
# ---------------------------------------------------------------------------


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
    columns are the eight keys at ``8 * kt``.  Each lane carries two adjacent
    keys for each of two values separated by eight rows.
    """
    g = lane >> 2
    q = lane & 3
    v = v_base + g + 8 * (reg >> 1)
    k = HT_TILE_KEYS * kt + 2 * q + (reg & 1)
    return (k, v)


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

    The transposed recurrence needs Ak as B.  The image is contiguous in key,
    while a B register wants two adjacent *tokens*, hence ``.trans``.
    """
    r = lane >> 3
    token = lane - 8 * r
    return raw_bf16_s128((token + 8 * (r & 1)) ^ AK_TOKEN_XOR, BT * j + 8 * (r >> 1))


def state_x2t_ptr(lane, kt, value_base):
    """Boundary-state ``ldmatrix.x2`` / ``stmatrix.x2``, transposed recurrence.

    No ``.trans``: the image is physically
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
# Native m16n8k16 fragment coordinates
# ---------------------------------------------------------------------------


def mma_c16_coord(lane, slot):
    """``(row, col)`` of slot ``slot`` in a logical N=16 accumulator.

    Slots 0-3 are the low N=8 half and 4-7 the high half, matching the order
    two native MMAs write them.
    """
    nb = slot // 4
    return mma_c_coord(lane, slot - nb * 4, 8 * nb)


# ---------------------------------------------------------------------------
# SMEM pointer maps
#
# ``factor_a_ptr`` and ``ki_b_ptr`` are NOT the same lane map.  The A map takes
# its row half from ``m & 1`` and its column half from ``m >> 1``; the B map
# takes the row half from lane bit 4 and the column half from lane bit 3.  Using
# the A map with a ``.trans`` modifier in place of the B map silently
# transposes the operand, which is why the two are separate functions.
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


def ak_store_ptr(lane, key_base):
    """``stmatrix.x4`` of one ``[16, 16]`` Ak.T tile at ``key_base``."""
    m = lane // 8
    token = (lane - m * 8) + 8 * (m & 1)
    key = key_base + 8 * (m >> 1)
    return ak_t_s128(token, key)


# ---------------------------------------------------------------------------
# Prepare-warp ownership
# ---------------------------------------------------------------------------


def gate_owner_dim(warp, lane):
    """Key dimension whose 16 gate values and ``GTotal`` lane ``lane`` owns."""
    return 32 * warp + lane


def norm_row(warp, lane):
    """Token row lane ``lane`` of prepare warp ``warp`` helps normalize."""
    return 4 * warp + (lane // 8)


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
    entirely into the row address.  Raw Q/K need no token-coordinate XOR,
    unlike the Ak.T image.
    """
    reg = lane >> 3
    row = lane - 8 * reg
    nb = reg >> 1
    r = reg - 2 * nb
    return raw_bf16_s128(8 * r + row, 32 * warp + BT * j + 8 * nb)


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


# --------------------------------------------------------------------------
# Inline PTX the fused kernel issues
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Matrix copies
# ---------------------------------------------------------------------------


@dsl_user_op
def fresh_b32(value, *, loc=None, ip=None):
    """A byte-identity ``prmt.b32``: same value, a *different* register.

    This computes nothing.  It keeps the MMA's operand register distinct from
    the persistent H16 state register: the transposed recurrence hands H16
    straight to the MMA as its A operand, and without an intervening value
    ptxas must satisfy the MMA's register constraints on registers that are
    live across the entire kernel, which -- with 96 registers already pinned
    to h32/h16 -- it resolves by spilling.  Removing this is a measured
    regression, not a cleanup.

    ``prmt`` rather than ``mov`` because ptxas folds an identity move; the
    byte-permute survives.
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


# ---------------------------------------------------------------------------
# Packed conversion and arithmetic
# ---------------------------------------------------------------------------


@dsl_user_op
def rcp_approx_ftz(x, *, loc=None, ip=None):
    """``rcp.approx.ftz.f32``: used for ``1 / E`` when forming Ki, and nowhere else.

    ``.ftz`` flushes subnormal *inputs* to zero, so ``E < 2^-126`` would return
    ``+Inf`` and ``0 * Inf`` would poison a whole KK row.  The
    :data:`PREFIX_FLOOR` clamp on the gate prefix is what keeps ``E`` at or
    above the smallest normal, which is why that clamp is not optional.
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

    Used in the first-ready loop in place of ``try_wait``, which may suspend
    the warp for an implementation-defined time limit.  The result only
    *selects* a branch; the chosen branch still performs the real 32-lane
    acquire, because a broadcast predicate gives the other 31 lanes no
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


#: Newer DSL releases renamed these and deprecated the old spellings; older
#: releases only have the old names.  Resolve once at import so both build
#: without a DeprecationWarning per traced call site.  Both spellings lower to
#: the same op, which ptxas currently drops -- see the ``WG*_MAXNREG`` note.
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


# --------------------------------------------------------------------------
# TMA descriptors
# --------------------------------------------------------------------------

DESCRIPTOR_ALIGN = 64

#: This order is fixed; the launch path indexes uploads by it.
DESCRIPTOR_ROLES = ("q", "k", "g", "v", "out", "state_in", "state_out")


# ---------------------------------------------------------------------------
# Activations
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


# ---------------------------------------------------------------------------
# State boundary
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

    def address(self, role: str) -> int:
        """Descriptor address of ``role``, or 0 when the role is absent.

        Zero is the kernel's "no descriptor" sentinel: an absent state has no
        ``fence.proxy.tensormap`` and no copy, both under ``const_expr``
        branches, so the address is never dereferenced.
        """
        return self.addresses.get(role, 0)


def build_upload(specs: dict[str, TensorMapSpec], device) -> TensorMapUpload:
    """Encode, deduplicate and upload the descriptors for one launch.

    Deduplication is by full spec equality: ``v is out`` and an exact state
    in/out alias produce identical specs and share one descriptor, while
    anything that differs in a single encoder field gets its own.  Direction
    is still decided by the PTX instruction, so sharing a descriptor between a
    load and a store is safe.
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
        return TensorMapUpload(storage=empty, addresses={})

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
    return TensorMapUpload(storage=storage, addresses=addresses)


# --------------------------------------------------------------------------
# The fused device kernel
# --------------------------------------------------------------------------

#: Barrier arena indices, in 8-byte units from the start of dynamic SMEM.
_BAR = CONTROL_OFFSET // 8
BAR_QKG = _BAR + BAR_QKG_READY // 8
BAR_V = _BAR + BAR_V_READY // 8
BAR_MAT = _BAR + BAR_MATERIALIZED // 8
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
V_BASE16 = V_STAGE_OFFSET // 2  # 32768

#: Quadrant slot pairs of an N=16 accumulator: TL, BL, TR, BR.
QUAD_TL = (0, 1)
QUAD_BR = (6, 7)
DIAGONAL_SLOTS = QUAD_TL + QUAD_BR


# ---------------------------------------------------------------------------
# Small DSL helpers
# ---------------------------------------------------------------------------


@cute.jit
def load_vec4_f32(ptr, idx):
    """Four adjacent FP32 at ``ptr + idx`` as one 16-byte load."""
    frag = cute.make_rmem_tensor(4, cutlass.Float32)
    cute.autovec_copy(vec_at(ptr, idx, 4), frag)
    return frag


@cute.jit
def store_vec8_bf16_from_f32(ptr, idx, values):
    """Eight FP32 rounded to BF16 and written as one 16-byte store."""
    frag = cute.make_rmem_tensor(8, cutlass.BFloat16)
    for i in cutlass.range_constexpr(8):
        frag[i] = values[i].to(cutlass.BFloat16)
    store_vec8_bf16(ptr, idx, frag)


@cute.jit
def zero4():
    z = cutlass.Float32(0.0)
    return (z, z, z, z)


@cute.jit
def zero8():
    z = cutlass.Float32(0.0)
    return (z, z, z, z, z, z, z, z)


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
    MMA's C operand saves an instruction and changes the result, so it is
    forbidden here.  ``B0 = I - D`` likewise comes from the widened
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

    Three things here are contract, not style: the invalid-row mask selects
    an exact ``+0.0`` *increment* -- zeroing raw ``G`` is not enough, because
    ``dt_bias`` still produces a non-zero increment; the scan uses the pairwise
    bracketing rather than a plain running sum; and the clamp happens after
    the complete scan and before ``exp2``.
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

    The butterfly offsets are pinned to 4 -> 2 -> 1 and each step to
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

    Everything from ``qkg_ready`` up to and including ``gate_column``, the
    longest dependent chain in prepare.  It returns the whole column because
    ``e_col`` is the only thing the rest of the chunk needs from it;
    ``gt_owned`` is its last element.
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
        clear_tail_rows(
            p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32, PREPARE_BARRIER_THREADS
        )
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
    s16_scale,
    token_start,
    token_end,
):
    """Publish E row-major, then normalize and materialize in one ownership.

    The lane that reduces a token row also normalizes it, decays it and
    publishes it, so ``q_inv``/``k_inv`` never leave a register and the raw
    Q/K operands are read from shared memory once.  The cost is E, which must
    be readable as ``(token, 16 features)`` and so uses the row-major image
    raw ``G`` uses -- sixteen scalar stores from the gate's column ownership
    rather than four vector ones.

    ``kn`` and ``qn`` stay FP32 into the rounding; nothing is published
    between the norm and the decay.
    """
    s = main_slot(c)
    s16 = s * SLOT16
    s32 = s * SLOT32
    valid_rows = clamp_rows(token_end, token_start + c * BT)

    p_q = p16 + s16
    p_k = p16 + (s16 + SLOT_K // 2)
    p_ki = p16 + (s16 + SLOT_KI // 2)
    p_e = p32 + (s32 + SLOT_G_LO // 4)
    smem_e = cute.make_tensor(p_e, cute.make_layout(BT * DK))
    smem_gt = cute.make_tensor(p32 + (s32 + SLOT_GTOTAL // 4), cute.make_layout(DK))
    gt_owned = e_col[BT - 1]

    # Rendezvous 2: E overwrites raw G's bytes.
    prepare_rendezvous()

    # E row-major, at raw G's own addresses.  The gate owns one key and holds
    # sixteen tokens, so this is sixteen scalar stores; the consumer below owns
    # one token and sixteen keys and reads four vectors.
    for r in cutlass.range_constexpr(BT):
        smem_e[raw_f32_s128(r, dim)] = e_col[r]

    # Rendezvous 3: E is published and every read below is another warp's.
    prepare_rendezvous()

    # --- norm, in the ownership that will also materialize -----------------
    # Both halves stay in registers for the materialize step below: sixteen
    # registers per lane.
    fq0 = vec8_bf16(p_q, raw_bf16_s128(norm_token, 8 * lane_in_row))
    fk0 = vec8_bf16(p_k, raw_bf16_s128(norm_token, 8 * lane_in_row))
    fq1 = vec8_bf16(p_q, raw_bf16_s128(norm_token, 64 + 8 * lane_in_row))
    fk1 = vec8_bf16(p_k, raw_bf16_s128(norm_token, 64 + 8 * lane_in_row))
    q_ss = cutlass.Float32(0.0)
    k_ss = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(8):
        qv = cutlass.Float32(fq0[i])
        kv = cutlass.Float32(fk0[i])
        q_ss = q_ss + qv * qv
        k_ss = k_ss + kv * kv
    for i in cutlass.range_constexpr(8):
        qv = cutlass.Float32(fq1[i])
        kv = cutlass.Float32(fk1[i])
        q_ss = q_ss + qv * qv
        k_ss = k_ss + kv * kv
    q_ss = row_sum_8(q_ss)
    k_ss = row_sum_8(k_ss)
    # Every lane of the row computes the reciprocal: eight redundant MUFU are
    # cheaper than a store, a 128-thread rendezvous and eight scalar reads.
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

    # --- E for this lane's own sixteen features ----------------------------
    e0 = load_vec4_f32(p_e, raw_f32_s128(norm_token, 8 * lane_in_row))
    e1 = load_vec4_f32(p_e, raw_f32_s128(norm_token, 8 * lane_in_row + 4))
    e2 = load_vec4_f32(p_e, raw_f32_s128(norm_token, 64 + 8 * lane_in_row))
    e3 = load_vec4_f32(p_e, raw_f32_s128(norm_token, 64 + 8 * lane_in_row + 4))

    # Rendezvous 4: E is in registers everywhere, its bytes may become Ki.
    prepare_rendezvous()

    # GTotal only after rendezvous 4: SLOT_GTOTAL is 13,312 and E spans
    # 8,192 to 16,384, so this store lands *inside* E and may not happen
    # while another warp is still reading it.
    smem_gt[dim] = gt_owned
    qd = [cutlass.Float32(0.0) for _ in range(16)]
    kd = [cutlass.Float32(0.0) for _ in range(16)]
    ki = [cutlass.Float32(0.0) for _ in range(16)]
    for half in cutlass.range_constexpr(2):
        fq = fq0 if cutlass.const_expr(half == 0) else fq1
        fk = fk0 if cutlass.const_expr(half == 0) else fk1
        lo = e0 if cutlass.const_expr(half == 0) else e2
        hi = e1 if cutlass.const_expr(half == 0) else e3
        for i in cutlass.range_constexpr(8):
            ev = lo[i] if cutlass.const_expr(i < 4) else hi[i - 4]
            e16 = bf16_round(ev)
            kn = cutlass.Float32(fk[i]) * k_inv
            qn = cutlass.Float32(fq[i]) * q_inv
            idx = 8 * half + i
            kd[idx] = bf16_round(bf16_round(kn) * e16)
            ki[idx] = bf16_round(kn * rcp_approx_ftz(ev))
            qd[idx] = bf16_round(bf16_round(bf16_round(qn) * e16) * s16_scale)
    store_vec8_bf16_from_f32(p_q, raw_bf16_s128(norm_token, 8 * lane_in_row), qd[0:8])
    store_vec8_bf16_from_f32(p_k, raw_bf16_s128(norm_token, 8 * lane_in_row), kd[0:8])
    store_vec8_bf16_from_f32(p_ki, raw_bf16_s128(norm_token, 8 * lane_in_row), ki[0:8])
    store_vec8_bf16_from_f32(
        p_q, raw_bf16_s128(norm_token, 64 + 8 * lane_in_row), qd[8:16]
    )
    store_vec8_bf16_from_f32(
        p_k, raw_bf16_s128(norm_token, 64 + 8 * lane_in_row), kd[8:16]
    )
    store_vec8_bf16_from_f32(
        p_ki, raw_bf16_s128(norm_token, 64 + 8 * lane_in_row), ki[8:16]
    )
    warp_arrive(p64 + (BAR_MAT + s), lane)


@cute.jit
def build_kr(p16, p32, lane, s16, s32, p):
    """Kr for strip ``p`` -- ``Ki * Diag(exp2 GTotal)`` -- from published data.

    Needs only what ``materialized`` guarantees: Ki and GTotal are
    ``prepare_body``'s stores, and nothing touches the Ki bytes before the
    Ak.T stores, which wait for ``ainv_ready``.  Building it ahead of that
    wait made no measurable difference, so the caller runs it wherever reads
    simplest.

    The caller passes ``p`` as a Python constant; four lanes share each key
    and shared memory broadcasts the read.
    """
    p_ki = p16 + (s16 + SLOT_KI // 2)
    smem_gt = cute.make_tensor(p32 + (s32 + SLOT_GTOTAL // 4), cute.make_layout(DK))
    kr_b = [cutlass.Int32(0) for _ in range(8)]
    for j in cutlass.range_constexpr(2):
        ki_r = ldmatrix_x4_trans(p_ki + materialize_b_ptr(p, lane, j))
        for nb in cutlass.range_constexpr(2):
            for r in cutlass.range_constexpr(2):
                key = materialize_coords(p, lane, j, nb, r)[2]
                gt16 = bf16_round(cutlass.Float32(smem_gt[key]))
                kr_b[4 * j + 2 * nb + r] = mul_bf16x2(
                    ki_r[2 * nb + r], pack_bf16x2(gt16, gt16)
                )
    return (
        kr_b[0],
        kr_b[1],
        kr_b[2],
        kr_b[3],
        kr_b[4],
        kr_b[5],
        kr_b[6],
        kr_b[7],
    )


@cute.jit
def akt_store(p16, lane, s16, p, ainvb_t, kr_b):
    """``Ak.T = AinvBeta.T @ Kr`` for strip ``p``, over the Ki bytes.

    This step needs ``ainv_ready``: AinvBeta is W15's solve output, and the
    stores overwrite Ki, which the barrier's two arrivals (W15 published, W13
    has read) make safe.  Strips are disjoint by key -- ``ak_t_s128`` permutes
    tokens, never keys -- so W13 (strips 0, 1) and W14 (strips 2, 3) need no
    ordering against each other.
    """
    p_ki = p16 + (s16 + SLOT_KI // 2)
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
        stmatrix_x4(
            p_ki + ak_store_ptr(lane, 32 * p + BT * tile),
            frag[0],
            frag[1],
            frag[2],
            frag[3],
        )


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
    """Gate, normalize, and materialize Qd/Kd/Ki.

    A prepare warp uses two different ownerships inside one chunk, and they
    do not compose into "warp p owns 32 dimensions": the gate is one key
    dimension per lane, while the norm and the materialize step are one token
    row per eight lanes.  Conflating them is the easiest way to get this stage
    subtly wrong.

    Ak.T is not published here: W13 and W14 do that (:func:`akt_store`), so
    this warp never waits on ``ainv_ready``.
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

    norm_token = norm_row(p, lane)
    lane_in_row = lane & 7

    # A straight loop is the pipeline: ``head(c)`` waits only on TMA, which
    # runs MAIN_SLOTS chunks ahead, and nothing here consumes ``ainv_ready``
    # (the Ak.T tail runs on W13/W14 as :func:`akt_store`), so there is no
    # wait left for a software-pipelined skew to hide.
    for c in range(num_chunks):
        e_col = prepare_head(
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
            G_FP32,
            SAFE_GATE,
        )
        prepare_body(
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
            s16_scale,
            token_start,
            token_end,
        )


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
    """Prefetch Q/K/G and V ``MAIN_SLOTS`` chunks ahead, on independent barriers.

    Generation 0's reuse parity is 1, which passes immediately on a freshly
    initialized phase-0 barrier, so this is one uniform loop from ``c = 0``
    with no prologue special case: the first ``MAIN_SLOTS`` iterations fall
    straight through and only ``c >= MAIN_SLOTS`` is throttled by the release
    of slot ``c - MAIN_SLOTS``.

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
def qk_role(p16, p32, p64, lane, num_chunks):
    """Causal ``QK = tril(Qd @ Ki.T)``, then Ak.T strips 0-1, then ``Aq``.

    ``QK_A`` stays in registers across the wait for ``ainv_ready``; staging it
    through SMEM would need a region the arena does not have.  The arrival
    after the QK loop says ``Ki`` has been read -- the license (together with
    W15's own arrival) for this same warp to come back and overwrite ``Ki``
    with ``Ak.T``.

    Ak.T before Aq, not after: ``h_branch`` consumes Ak.T and feeds
    ``state_done``, which gates W12's slot reuse four chunks out, while Aq
    only feeds the O branch and W14's store.  The recurrence's first-ready
    probe takes whichever lands first, and this order lands the one with the
    longer consequence chain first.
    """
    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        s16 = s * SLOT16
        s32 = s * SLOT32
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
        # Ki has been read.  This is ``ainv_ready``'s second arrival, not a
        # barrier of its own: the Ak.T stores need this *and* W15's AinvBeta,
        # so one barrier counting two says exactly that.
        warp_arrive(p64 + (BAR_AINV + s), lane)

        cute.arch.mbarrier_wait(p64 + (BAR_AINV + s), pr)
        # Strips 0 and 1; W14 carries 2 and 3.  The 2/2 split is also an SMSP
        # split: concentrating all four strips' ldmatrix/stmatrix bursts on one
        # warp slowed the prepare and recurrence warps sharing its SMSP.  Kr is
        # built per strip, eight registers at a time.
        ainvb_t = a_to_at(ldmatrix_x4(p_ainvb + pairwise_a_ptr(lane)))
        for strip in cutlass.range_constexpr(2):
            kr_b = build_kr(p16, p32, lane, s16, s32, strip)
            akt_store(p16, lane, s16, strip, ainvb_t, kr_b)
        warp_arrive(p64 + (BAR_AK + s), lane)
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

    The logit is loaded one whole chunk early.  It is a strided column read
    out of ``[T, H]`` -- sixteen lanes, one sector each, nothing coalesces --
    and when the inputs do not sit in L2 it is a full DRAM latency.  On such a
    shape W15 is the laggard, so its ``materialized`` wait is too short to
    hide that latency, and the exposed part lands on the ``ainv_ready`` chain
    that W13, the Ak.T strips and both recurrence branches sit behind.
    Hiding a DRAM load takes a chunk of distance, not a barrier of distance:
    chunk ``c + 1``'s load is issued before chunk ``c``'s activation and lands
    behind ``c``'s entire KK + solve + wait.  One FP32 register carries it
    across.

    The last iteration re-reads its own chunk's logit (the ``nc`` clamp)
    rather than predicating the load away: a dead in-bounds load is cheaper
    than a dynamic guard on the hoisted load.
    """
    logit_next = cutlass.Float32(0.0)
    if num_chunks > 0:
        if lane < BT:
            if lane < clamp_rows(token_end, token_start):
                logit_next = cutlass.Float32(
                    gbeta[beta_global_index(token_start + lane, head, heads)]
                )
    for c in range(num_chunks):
        s = main_slot(c)
        pr = ready_parity(c)
        s16 = s * SLOT16
        p_kd = p16 + (s16 + SLOT_KD // 2)
        p_ki = p16 + (s16 + SLOT_KI // 2)
        p_ainvb = p16 + (s16 + SLOT_AINV_BETA // 2)
        token_base = token_start + c * BT
        valid_rows = clamp_rows(token_end, token_base)

        logit_own = logit_next
        beta_owned = cutlass.Float32(0.0)
        if lane < BT:
            if lane < valid_rows:
                half = cutlass.Float32(0.5)
                beta_owned = (
                    cutlass.Float32(cute.math.tanh(logit_own * half, fastmath=True))
                    * half
                    + half
                )

        # The next chunk's logit, issued right here on purpose: it lands
        # behind this chunk's whole KK + solve, which is what hides it.
        nc = c + 1
        if nc >= num_chunks:
            nc = num_chunks - 1
        next_base = token_start + nc * BT
        logit_next = cutlass.Float32(0.0)
        if lane < BT:
            if lane < clamp_rows(token_end, next_base):
                logit_next = cutlass.Float32(
                    gbeta[beta_global_index(next_base + lane, head, heads)]
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
    gout, p16, p32, p64, lane, head, heads, token_start, token_end, num_chunks, desc_out
):
    """Ak.T strips 2-3 at each iteration's top, then chunk ``c``'s output.

    Because strips(c) run after store(c - 1), this warp's ``ak_ready`` arrival
    trails W13's.  The alternatives do no better: all four strips on W13
    concentrates the burst on one SMSP; a one-chunk skew is this same
    instruction sequence with the loop boundary redrawn; and a two-chunk skew
    blocks the store path behind a future chunk's ``ainv_ready``.  The late
    arrival is ring slack the recurrence's first-ready probe re-absorbs.

    A partial chunk must not go out by TMA: the box is 16 rows wide, and in a
    packed sequence the rows past ``valid_rows`` belong to the *next* sequence,
    so a full-tile store would overwrite output another CTA owns.

    ``output_read_done`` is released only after the store has *read* its source
    -- ``wait_group.read``, not merely ``commit`` -- because that is the
    condition for W12 to overwrite the slot.
    """
    for c in range(num_chunks):
        io_strips(p16, p32, p64, lane, c)
        io_store_one(
            gout, p16, p64, lane, head, heads, token_start, token_end, c, desc_out
        )


@cute.jit
def io_strips(p16, p32, p64, lane, c):
    """W14's half of Ak.T -- strips 2 and 3 of chunk ``c``."""
    s = main_slot(c)
    pr = ready_parity(c)
    s16 = s * SLOT16
    s32 = s * SLOT32
    p_ainvb = p16 + (s16 + SLOT_AINV_BETA // 2)
    cute.arch.mbarrier_wait(p64 + (BAR_AINV + s), pr)
    ainvb_t = a_to_at(ldmatrix_x4(p_ainvb + pairwise_a_ptr(lane)))
    for strip in cutlass.range_constexpr(2):
        kr_b = build_kr(p16, p32, lane, s16, s32, strip + 2)
        akt_store(p16, lane, s16, strip + 2, ainvb_t, kr_b)
    warp_arrive(p64 + (BAR_AK + s), lane)


@cute.jit
def io_store_one(
    gout, p16, p64, lane, head, heads, token_start, token_end, c, desc_out
):
    """Wait for chunk ``c``'s output, publish it, release the slot."""
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

    GTotal indexes the *column* axis, so a lane needs the two adjacent keys
    of each tile rather than two rows shared across both value blocks -- 32
    values per lane instead of 16.  They are loaded as scalars, not as a pair:
    the live range is what matters here, not the instruction count, and a
    16-byte or 8-byte load imposes a register-pair alignment this
    register-bound warp cannot afford.

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
        # the MMA's operand off the persistent state registers; see its
        # docstring.
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
        # A packed register holds one value and *two adjacent tokens*, so the
        # tail is not one predicate per register: the two halves can
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
        # may suspend the warp and turn a scheduling hint into a stall -- and
        # the result is broadcast so the branch stays warp-uniform.  O wins a
        # tie: O-first fires `output_ready` mid-iteration, so W14's store and
        # the Ak.T strips gated behind it start earlier.  H-first would not
        # help the next chunk's projection, since proj(c + 1) starts only
        # after the whole iteration retires, whichever half ran first.
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
            if (flags & 2) != 0:
                sel = cutlass.Int32(2)
            elif (flags & 1) != 0:
                sel = cutlass.Int32(1)

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
    """Bring the external state into registers."""
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
    # that same number and every region base is a compile-time constant.
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
                cute.arch.mbarrier_init(p64 + (BAR_AINV + s), ARRIVALS_AINV_READY)
                cute.arch.mbarrier_init(p64 + (BAR_AQ + s), ARRIVALS_SINGLE)
                cute.arch.mbarrier_init(p64 + (BAR_AK + s), ARRIVALS_AK)
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
    # execute the same action with the same immediate at the same point, and
    # the decreases must be observed CTA-wide before any increase.  ptxas
    # currently drops the request; see the ``WG*_MAXNREG`` note.
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
        # Declared here, not in the common region, so their 96 values are not
        # live across the service warps' code.
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
            qk_role(p16, p32, p64, lane, num_chunks)
        elif warp_id == IO_WARP:
            io_role(
                gout,
                p16,
                p32,
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
# Compiled entry, descriptor cache and launch
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelKey:
    """The ordered compile key.  Fields may not be merged."""

    device: int
    device_cc: tuple[int, int]
    code_target: str
    g_dtype: torch.dtype
    safe_gate: bool
    state_dtype: torch.dtype | None
    has_initial_state: bool
    has_final_state: bool

    def __post_init__(self) -> None:
        has_state = self.has_initial_state or self.has_final_state
        if has_state != (self.state_dtype is not None):
            raise ValueError(
                "state_dtype must be present exactly when a state is present: "
                f"initial={self.has_initial_state}, final={self.has_final_state}, "
                f"dtype={self.state_dtype}"
            )


_KERNEL_CACHE: dict[KernelKey, object] = {}
_DESCRIPTOR_CACHE = BoundedDeviceCache("kda-descriptors")


def clear_kernel_caches() -> None:
    """Drop the compile and descriptor caches.

    Not the whole variant: :func:`clear_caches` below does that.  This is the
    device-side half, which a test that wants a cold compile needs on its own.
    """
    _KERNEL_CACHE.clear()
    _DESCRIPTOR_CACHE.clear()


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
    """Encode (or reuse) the TMA descriptors for one launch."""
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
    key = descriptor_cache_key(specs, device, DESCRIPTOR_ROLES)
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


#: The traced entry, built on first use.  ``@cute.jit`` decoration is not free
#: and the result is a constant, so it is built once.
_ENTRY: tuple | None = None


def _entry_module():
    """Build the traced entry once and memoize it.

    The DSL resolves a jit function's signature from its parameter
    annotations, so ``_fused_entry``'s ``cutlass.*`` annotations are written
    as real objects rather than postponed strings.
    """
    global _ENTRY
    if _ENTRY is not None:
        return _ENTRY

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

    ``stream`` is baked into ``args``, which is why the call-plan cache keys
    on it -- replaying a plan from another stream would launch against the
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
    # bounded here, before anything is allocated; a wrapped index is not an
    # error but a negative offset.
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
    # device: maxGridSize[1] is 65535 where axis 0 allows 2^31-1, and a short,
    # stateless call at H = 65536 fits in memory perfectly well, so this is
    # the only check that stops it before the driver.
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
        code_target=SM120_CODE_TARGET,
        g_dtype=g.dtype,
        safe_gate=bool(safe_gate),
        state_dtype=state_dtype,
        has_initial_state=initial_state is not None,
        has_final_state=final_state is not None,
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
        )
    )


# --------------------------------------------------------------------------
# The host path
#
# An LRU of launch plans on a tensor- and workspace-identity key, an
# object-identity fast path in front of it, a state-only shortcut, and the
# stream in the key because a plan bakes its ``CUstream`` into the argument
# tuple.  It is separate from the decomposed variant's cache: this one carries
# ``safe_gate``, has a single descriptor set and needs no chunk tables, since
# a fused CTA owns a whole (sequence, head).
#
# Three things are deliberately not cached, because caching them would be wrong
# rather than merely slow:
#
# * the zero-token state copy, which is real work the caller expects on every
#   call -- the plan records that it is a state-only call and redoes it;
# * anything derived from tensor *contents*;
# * plans across streams, for the reason above.
# --------------------------------------------------------------------------


#: This variant's call memo; see :class:`~.runtime.PlanMemo`.
_MEMO = PlanMemo()


def clear_caches() -> None:
    """Drop every cache this variant owns."""
    _MEMO.clear()
    clear_kernel_caches()


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
) -> Any:
    """Launch the fused prefill, writing ``out`` and ``final_state`` in place.

    ``info`` and ``offsets`` come from :mod:`.runtime`: the facade validates and
    canonicalizes once, so this is not a second validation pass.  Returns the
    plan it ran, or :data:`STATE_ONLY_PLAN`, which the facade memoizes together
    with :func:`execute`.
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

    scalars = (float(scale), float(lower_bound), bool(safe_gate))
    plan = _MEMO.fast_path(q.device, tensors, scalars, resources)
    if plan is None:
        key = _MEMO.identity(q.device, tensors, scalars, resources)
        # See ``PlanMemo.build_lock``: the miss path is not re-entrant, and
        # the two memo levels in front of it mean a warm caller never reaches it.
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
                    cu_seqlens=cu_seqlens,
                    info=info,
                    offsets=offsets,
                    resources=resources,
                    safe_gate=safe_gate,
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
        return STATE_ONLY_PLAN

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
    "KernelKey",
    "LaunchPlan",
    "clear_caches",
    "clear_kernel_caches",
    "code_target",
    "dynamic_smem_bytes",
    "kernel_name",
    "execute",
    "run",
]
