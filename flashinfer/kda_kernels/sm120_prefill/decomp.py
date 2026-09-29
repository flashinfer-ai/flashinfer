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
inline PTX, its TMA descriptors, both device kernels, the combined compiled
entry, its ``cu_chunks``/``chunk_to_seq`` metadata, its prepare scratch, its
descriptor and call-plan caches, and its current-stream launch.  Only the
mechanisms this variant shares with :mod:`.fused` -- bounded caches, capture
detection, canonical INT32 offsets, the workspace resource slot and the
``sm_120a`` target check -- come from :mod:`.runtime`.

The single file is deliberate.  The sections below were nine modules and the
split cost more than it bought: a chunk size, an SMEM offset and a barrier id
are one decision each, read by both the host plan and the device kernel, and
holding them apart made every one of them an import.  What does NOT belong
here is anything :mod:`.fused` also needs, which is why ``runtime.py`` exists
and why nothing in this file imports ``fused``.

Chunk size 16, the SMEM arena offsets, the swizzles, the barrier arena, the
grid, the chunks-per-CTA policy, the rounding boundaries and the state ABI are
fixed implementation choices shared by the host plan, device code and tests.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cuda.bindings import driver as cuda_driver
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .runtime import (
    GRAPH_PINS,
    NO_VERSION,
    BoundedDeviceCache,
    IdentityCache,
    KDAPrefillValidationError,
    assert_tvm_ffi_dispatched,
    build_kernel,
    capturing,
    check_flat_output_range,
    current_stream_ptr,
    flat_view,
    is_exact_alias,
    max_grid_dims,
    record_stream_once,
    resource_cache_token,
    require_sm120a,
    sm120a_compile_options,
    tensor_identity,
    tensor_version,
    upload_bytes,
)


# --------------------------------------------------------------------------
# Section 1: canonical SMEM and global index mappings
#
# These are the decomp variant's images.  The fused variant computes
# its own in its own file: the two agree on the S128 construction and
# disagree on which logical shape it is applied to, so they are not
# one helper with two callers.
# --------------------------------------------------------------------------

BT = 16
DK = 128
#: Value dimension.  Equal to ``DK`` for the shapes this backend supports, but
#: the recurrence addresses the two independently -- a CTA owns all ``DK`` keys
#: and only ``DV_HALF`` of the values.
DV = 128
DV_HALF = DV // 2

# raw_bf16_s128: 2 segments x 128 bytes, 8-element (16 byte) groups.
BF16_SEGMENT_ELEMS = 64
BF16_GROUP_ELEMS = 8
BF16_SEGMENTS = DK // BF16_SEGMENT_ELEMS
BF16_SEGMENT_STRIDE = BT * BF16_SEGMENT_ELEMS  # 1024 elements
BF16_ROW_XOR_MASK = BF16_SEGMENT_ELEMS // BF16_GROUP_ELEMS - 1  # 7

# raw_f32_s128: 4 segments x 128 bytes, 4-element (16 byte) groups.
F32_SEGMENT_ELEMS = 32
F32_GROUP_ELEMS = 4
F32_SEGMENTS = DK // F32_SEGMENT_ELEMS
F32_SEGMENT_STRIDE = BT * F32_SEGMENT_ELEMS  # 512 elements
F32_ROW_XOR_MASK = F32_SEGMENT_ELEMS // F32_GROUP_ELEMS - 1  # 7

# Constant coordinate permutations.
KR_AK_TOKEN_XOR = BT // 2  # 8, token-row permutation for the Ak.T image
PAIRWISE_COL_XOR = 8  # column permutation of the 16x16 pairwise image
PAIRWISE_ROW_STRIDE = 16

# Byte sizes of one staged tile.
RAW_BF16_STAGE_ELEMS = BT * DK  # 2048 BF16 -> 4096 bytes
RAW_F32_STAGE_ELEMS = BT * DK  # 2048 FP32 -> 8192 bytes
PAIRWISE_STAGE_ELEMS = BT * BT  # 256 BF16 -> 512 bytes


def raw_bf16_s128(token, dim):
    """Physical BF16 element index of logical ``(token, dim)``.

    Used for raw Q, raw K, ``Ki``, ``Kd`` and ``Qd``.  ``Kd``/``Qd`` carry no
    feature permutation, so they share this image with the
    raw stages and with ``Ki``.
    """
    segment = dim // BF16_SEGMENT_ELEMS
    local = dim - segment * BF16_SEGMENT_ELEMS
    group = local // BF16_GROUP_ELEMS
    inner = local - group * BF16_GROUP_ELEMS
    return (
        segment * BF16_SEGMENT_STRIDE
        + token * BF16_SEGMENT_ELEMS
        + (group ^ (token & BF16_ROW_XOR_MASK)) * BF16_GROUP_ELEMS
        + inner
    )


def raw_f32_s128(token, dim):
    """Physical FP32 element index of logical ``(token, dim)``.

    The single FP32 image in the kernel: TMA destination for raw ``G`` and,
    after the gate scan, in-place storage for FP32 ``exp_g`` at the very same
    addresses.
    """
    segment = dim // F32_SEGMENT_ELEMS
    local = dim - segment * F32_SEGMENT_ELEMS
    group = local // F32_GROUP_ELEMS
    inner = local - group * F32_GROUP_ELEMS
    return (
        segment * F32_SEGMENT_STRIDE
        + token * F32_SEGMENT_ELEMS
        + (group ^ (token & F32_ROW_XOR_MASK)) * F32_GROUP_ELEMS
        + inner
    )


def kr_ak_bf16_s128(token, dim):
    """Physical BF16 element index used to publish ``Ak.T``.

    The ``token ^ 8`` permutation is what produces the ``ws_ak[c, j ^ 8, d]``
    global image.  Because the swizzle is applied at row ``token ^ 8``, the two
    2048-byte halves of the stage still map to feature ranges ``[0,64)`` and
    ``[64,128)``, which is what lets warps 1 and 3 recycle the halves
    independently.
    """
    return raw_bf16_s128(token ^ KR_AK_TOKEN_XOR, dim)


def pairwise_sw32(row, col):
    """Physical BF16 element index of a 16x16 pairwise tile element."""
    storage_col = col ^ PAIRWISE_COL_XOR
    byte_offset = 2 * (row * PAIRWISE_ROW_STRIDE + storage_col)
    return (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2


# Backwards-compatible alias used by the workspace pack/unpack helpers and by
# the design's ``ws_aq[c, pair_idx(i, j)]``.
pair_idx = pairwise_sw32


# ---------------------------------------------------------------------------
# Native m16n8k16 fragment maps.
#
# These are fixed by the PTX ISA.  They are restated here so no consumer has to
# rediscover them and so the unit tests can enumerate them.  ``g = lane >> 2``
# and ``q = lane & 3`` throughout.
# ---------------------------------------------------------------------------


def mma_a_coords(lane, reg):
    """Logical ``(row, col)`` of the two BF16 halves in A register ``reg``."""
    g = lane >> 2
    q = lane & 3
    row = g + 8 * (reg & 1)
    col = 2 * q + 8 * (reg >> 1)
    return ((row, col), (row, col + 1))


def mma_b_coords(lane, reg):
    """Logical ``(k, n)`` of the two BF16 halves in B register ``reg``."""
    g = lane >> 2
    q = lane & 3
    k = 2 * q + 8 * reg
    return ((k, g), (k + 1, g))


def mma_c_coord(lane, n_block, reg):
    """Logical ``(row, col)`` of FP32 accumulator register ``reg``."""
    g = lane >> 2
    q = lane & 3
    row = g + 8 * (reg >> 1)
    col = 8 * n_block + 2 * q + (reg & 1)
    return (row, col)


def ldmatrix_a_coord(lane):
    """Row/col of the 16-byte row segment lane ``lane`` feeds to an A load.

    The row half is keyed
    on **bit 3** of the lane.
    """
    matrix_id = lane >> 3
    row = (lane & 7) + (8 if (matrix_id & 1) else 0)
    col = 8 if (matrix_id >> 1) else 0
    return (row, col)


def ldmatrix_b_coord(lane):
    """Row/col of the 16-byte row segment lane ``lane`` feeds to a B load.

    The row half is keyed
    on **bit 4** of the lane, unlike the A rule; using the A rule here silently
    transposes the operand.
    """
    matrix_id = lane >> 3
    row = (lane & 7) + (8 if (lane >> 4) else 0)
    col = 8 if (matrix_id & 1) else 0
    return (row, col)


def stmatrix_coord(lane):
    """Row/col of the 16-byte row segment lane ``lane`` feeds to an x4 store.

    Same tile/row convention as ``ldmatrix.x4`` and the same quadrant order as
    the **A** fragment.
    """
    matrix_id = lane >> 3
    row = (lane & 7) + (8 if (matrix_id & 1) else 0)
    col = 8 if (matrix_id >= 2) else 0
    return (row, col)


#: ``movmatrix`` register order for converting an A-layout fragment into the B
#: layout of the *same* matrix: transpose each 8x8 quadrant, identity order.
MOVMATRIX_A_TO_B = (0, 1, 2, 3)

#: ``movmatrix`` register order for converting an A-layout fragment into the
#: A layout of the *transposed* matrix: quadrants (1,0) and (0,1) exchange, so
#: registers 1 and 2 swap.
MOVMATRIX_A_TO_AT = (0, 2, 1, 3)


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
VO_STAGE_ELEMS = BT * VO_ROW_ELEMS  # 1024 BF16 = 2048 B

#: Physical rows of one value in the state image, by element width.  The state
#: is stored ``[V, K]`` and read logically as ``[K, V]``,
#: so one value spans 128 keys = 2 BF16 or 4 FP32 128-byte segments.
STATE_BF16_ROWS_PER_VALUE = DK // BF16_SEGMENT_ELEMS  # 2
STATE_F32_ROWS_PER_VALUE = DK // F32_SEGMENT_ELEMS  # 4
STATE_BF16_ELEMS = DV_HALF * DK  # 8192 BF16 = 16 KiB
STATE_F32_ELEMS = DV_HALF * DK  # 8192 FP32 = 32 KiB


def _s128_index(row, column, segment_elems, group_elems):
    """S128 element index of ``column`` within 128-byte row ``row``.

    ``row`` here is the *segment* row -- the unit the swizzle XOR keys on -- not
    a logical matrix row.  The three recurrence images differ only in what they
    call a segment row and how they split a logical coordinate into one.
    """
    group = column // group_elems
    inner = column - group * group_elems
    return (
        row * segment_elems
        + (group ^ (row & (segment_elems // group_elems - 1))) * group_elems
        + inner
    )


def vo_idx(row, v_local):
    """Physical BF16 index of ``(token row, value)`` in a V/output half stage.

    ``v_local`` is in ``[0, 64)`` and is the CTA-local value, so the two DV
    halves address byte-identical stages at different global coordinates.
    """
    return _s128_index(row, v_local, VO_ROW_ELEMS, BF16_GROUP_ELEMS)


def state_bf16_idx(v_local, k):
    """Physical BF16 index of logical ``H[k][v]`` in the persistent state.

    The unswizzled address function is ``state_idx(k, v) = v * 128 + k`` (plan
    Section 12.2 Q1): physical ``[V, K]`` row-major *and* logical ``[K, V]``
    column-major at once, which is what lets an external ``[V, K]`` state half
    land by TMA with no transpose and still be read as ``H[K, V]`` by the MMA.
    """
    segment = k // BF16_SEGMENT_ELEMS
    local = k - segment * BF16_SEGMENT_ELEMS
    line = STATE_BF16_ROWS_PER_VALUE * v_local + segment
    return _s128_index(line, local, BF16_SEGMENT_ELEMS, BF16_GROUP_ELEMS)


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


def gt_idx(k):
    """``GTotal`` is a contiguous FP32 ``[128]`` record with no swizzle."""
    return k


def state_half_rows(dv_half, *, fp32: bool = False) -> int:
    """First tensor-map row of DV half ``dv_half``.

    The state descriptors encode each 128-byte segment as the tensor map's
    inner mode, so the half's coordinate is a row index and not a value index.
    """
    per_value = STATE_F32_ROWS_PER_VALUE if fp32 else STATE_BF16_ROWS_PER_VALUE
    return dv_half * per_value * DV_HALF


def vo_global_index(token, head, heads, dv_half, v_local):
    """Flat element index of ``out[0, token, head, 64 * dv_half + v_local]``.

    The partial-output tail stores through this map with 16-byte vectors (plan
    Section 8.2); the full path never needs it, because TMA addresses the same
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


def mma_n8_c_coord(lane, reg):
    """``(row, col)`` of accumulator register ``reg`` in a native N=8 MMA."""
    return mma_c_coord(lane, 0, reg)


def state_x2_ptr(lane, kb, v_base):
    """SMEM index lane ``lane`` addresses for a state 16x8 ``ldmatrix.x2``.

    The same 16 addresses serve both state reads .2: without
    ``.trans`` they produce the MMA **B** operand of ``Kd @ H`` and ``Qd @ H``,
    and with ``.trans`` they produce the **C** view the state update needs.  The
    two passes therefore differ only in one instruction modifier, which is why
    the second pass can reload from SMEM instead of keeping eight fragments
    live.  Lanes 16-31 are ignored by an x2 copy.
    """
    matrix = (lane // 8) & 1
    row = lane - (lane // 8) * 8
    return state_bf16_idx(v_base + row, kb * BT + 8 * matrix)


def vo_x2_ptr(lane, v_base):
    """SMEM index lane ``lane`` addresses for a V/output 16x8 ``ldmatrix.x2``.

    Non-transposed in both directions: the stage is token-major and the C tile's
    rows are tokens, so the loaded V and the stored output share this map.
    """
    matrix = (lane // 8) & 1
    row = (lane - (lane // 8) * 8) + 8 * matrix
    return vo_idx(row, v_base)


def packed_c_reg_coords(lane, reg):
    """Logical ``(row, col)`` of the two halves of packed C register ``reg``.

    A 16x8 C tile is four FP32 accumulator registers, but only two b32 registers
    once rounded to BF16, and an x2 matrix copy moves exactly those two.  Packed
    register ``reg`` is the pair ``(2 * reg, 2 * reg + 1)``, which is one C row
    and two adjacent columns -- so the ``ldmatrix``/``stmatrix`` halves line up
    with the accumulator without any lane shuffle.

    This is the register map for the V load, the output store, and the
    ``.trans`` state read alike: :func:`vo_x2_ptr` and :func:`state_x2_ptr`
    already absorb the orientation difference, so only the instruction's
    ``.trans`` modifier changes between them.
    """
    return (mma_n8_c_coord(lane, 2 * reg), mma_n8_c_coord(lane, 2 * reg + 1))


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
    four returned registers are ``(a0, a1, a2, a3)`` directly -- plan
    Section 7.1 forbids permuting them afterwards.
    """
    matrix_id = lane // 8
    row8 = lane - matrix_id * 8
    logical_j = (matrix_id // 2) * 8 + row8
    key = kb * BT + (matrix_id - (matrix_id // 2) * 2) * 8
    return factor_idx(logical_j ^ KR_AK_TOKEN_XOR, key)


#: ``movmatrix`` register order converting the BF16 residual from the C layout
#: it is computed in to the B layout the Aq/Ak MMAs consume: transpose each
#: packed 8x8 quadrant in place, keeping the register index.
MOVMATRIX_C_TO_B = (0, 1)


#: ``cute.make_swizzle`` takes the design's **byte**-unit parameters, not
#: the element-unit ones.  A composed layout's swizzle is applied to the byte
#: offset, so both S128 images spell as ``Swizzle<3,4,3>`` regardless of element
#: width -- ``<3,3,3>`` (BF16) and ``<3,2,3>`` (FP32) are the element-unit
#: spellings of the *same* images and are wrong as arguments here.  Checked
#: against the arithmetic formulas in ``test_layouts``.
SWIZZLE_S128_BYTES = (3, 4, 3)
SWIZZLE_SW32_BYTES = (1, 4, 3)


def make_cute_layouts():
    """Build the three CuTe composed layouts .

    ``raw_bf16`` and ``raw_f32`` are what ``make_tiled_tma_atom`` consumes, so
    they must reproduce :func:`raw_bf16_s128` and :func:`raw_f32_s128` exactly.

    ``pairwise`` carries the SW32 swizzle only.  :func:`pairwise_sw32` also
    permutes the column by ``^ PAIRWISE_COL_XOR``, which is not expressible as
    a CuTe layout, so this object is *not* a complete model of that image.  No
    TMA descriptor uses it: the design has warp 2 store ``Aq`` directly.

    Imported lazily so that host-only consumers (workspace helpers, layout unit
    tests) do not need the CUTLASS DSL installed.
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
# Section 2: CuTe DSL names that moved between releases
#
# An import rather than a module-level getattr: the DSL's AST
# preprocessor replays a traced module's imports into the tracing
# scope and nothing else.
# --------------------------------------------------------------------------

#: 4.7 renamed ``make_fragment`` to ``make_rmem_tensor``; the signature
#: ``(layout_or_shape, dtype, *, loc=None, ip=None) -> Tensor`` is unchanged,
#: so one alias covers every call site.
make_rmem_tensor = getattr(cute, "make_rmem_tensor", None) or cute.make_fragment


# --------------------------------------------------------------------------
# Section 3: inline PTX the decomp kernels issue
# --------------------------------------------------------------------------


@dsl_user_op
def ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` -> four b32 registers."""
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        llvm.StructType.get_literal([_T.IntegerType.get_signless(32)] * 4),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
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
        for i in range(4)
    )


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
def ldmatrix_x2(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x2.shared.b16`` -> two b32 registers.

    the state's MMA **B** operand and the V tile
    both load through this.  Only lanes 0-15 supply addresses.
    """
    return _ldmatrix(".x2", "", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`` -> two b32 registers.

    Same addresses as :func:`ldmatrix_x2`, read down the memory columns instead
    of across its rows.  That is what turns the physical ``[V, K]`` state into
    the logical ``[K, V]`` C tile the state update accumulates into.
    """
    return _ldmatrix(".x2", ".trans", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_trans(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`` -> four b32 registers.

    The ``Ak`` A operand: prepare publishes ``Ak.T`` with a ``token ^ 8`` row
    permutation, and the pointer map of ``layouts.ak_a_fragment_ptr`` plus this
    instruction produce the logical ``[key, token]`` fragment with no register
    permutation afterwards.
    """
    return _ldmatrix(".x4", ".trans", smem_ptr, 4, loc=loc, ip=ip)


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
def stmatrix_x2(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x2.shared.b16``: the output half store."""
    _stmatrix(".x2", "", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x2_trans(smem_ptr, r0, r1, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x2.trans.shared.b16``: the state write-back.

    Exactly inverts :func:`ldmatrix_x2_trans` against the same pointer map, so
    the state update reads and writes the identical 16x8 block of one warp's
    value columns.
    """
    _stmatrix(".x2", ".trans", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x4(smem_ptr, r0, r1, r2, r3, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x4.shared.b16`` from four b32 registers."""

    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r3).ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def movmatrix_b16(value, *, loc=None, ip=None):
    """``movmatrix.sync.aligned.m8n8.trans.b16``: transpose one 8x8 b16 tile."""
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
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``."""

    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``."""

    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def pack_bf16x2(lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None):
    """Round two FP32 values to BF16 and pack them into one b32 register.

    ``cvt.rn.bf16x2.f32 d, hi, lo`` places ``lo`` in the low half.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f16x2(lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None):
    """Round two FP32 values to FP16 and pack them into one b32 register.

    The FP16 twin of :func:`pack_bf16x2`, with the same half ordering:
    ``cvt.rn.f16x2.f32 d, hi, lo`` places ``lo`` in the low half.  Used by the
    inverse chain, whose operands are FP16 regardless of the kernel's input
    dtype -- FP16's 10-bit significand against BF16's 7 is worth 4-8x there,
    and the chain is the one stage where the extra bits survive.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_bf16x2(a, b, *, loc=None, ip=None):
    """Packed BF16 multiply of two b32 registers, rounding each product."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sub_bf16x2(a, b, *, loc=None, ip=None):
    """Packed BF16 subtract of two b32 registers, rounding each difference.

    the design builds the residual with this rather than with
    FP32 arithmetic: both operands are already BF16, so the exact difference is
    representable and a single ``sub.rn`` reproduces the contract's
    ``BF16(V - X)`` boundary for two values at once.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "sub.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2(value, *, loc=None, ip=None):
    """Widen a packed BF16 pair to two FP32, low half first.

    The state's decay term is ``FP32(GTotal) * FP32(H_bf16)``,
    so the reloaded BF16 state has to reach the FP32 accumulator; this is the
    widening, and it recovers nothing the entry rounding already discarded.
    """
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
def cvt_bf16x2_to_f16x2(value, *, loc=None, ip=None):
    """Re-round a packed BF16 pair through FP16 (for an FP16 inverse chain)."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 lo, hi; .reg .f32 flo, fhi;"
            "  mov.b32 {lo, hi}, $1;"
            "  cvt.f32.bf16 flo, lo; cvt.f32.bf16 fhi, hi;"
            "  cvt.rn.f16x2.f32 $0, fhi, flo; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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


# ---------------------------------------------------------------------------
# TMA
#
# The public wheel's TMA path (cpasync.make_tiled_tma_atom + cute.copy)
# compiles but its copy never completes on sm_120, and the API this is
# written against lowers to MLIR ops the wheel does not ship.  The instruction
# itself is fine on this hardware, so issue it directly.  Descriptors come from
# cuTensorMapEncodeTiled on the host and live in device memory; the kernel gets
# their addresses as Int64 scalars.
# ---------------------------------------------------------------------------


@dsl_user_op
def fence_tensormap_acquire(desc_addr, *, loc=None, ip=None):
    """Publish a host-written tensor map to the tensormap proxy.

    The descriptor is written by the host and read by the TMA unit through a
    different proxy, so the kernel has to acquire it before first use.
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

    ``shared::cta`` rather than ``shared::cluster``: sm_120 has no thread block
    clusters, and this is the form that works.  Completion is
    reported to ``mbar_ptr`` as transaction bytes, so the consumer waits on the
    mbarrier rather than on a commit group.
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
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [$0], [$1, {$3, $4, $5}], [$2];",
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

    Stores have no mbarrier; they are tracked with bulk commit groups, and the
    ``read`` wait below is what releases the source SMEM for reuse.
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
    """Close the current bulk-store group."""
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
    """Wait until at most ``keep`` bulk-store groups still hold their source.

    ``.read`` is a source-SMEM reuse guarantee, not a claim that the store is
    globally visible (step 8 of the design).
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
# Section 4: the prepare workspace and this variant's chunk metadata
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


def total_chunks_for_lengths(lengths) -> int:
    """Exact total chunk count; zero-length sequences contribute zero chunks."""
    return sum(chunks_for_lengths(lengths))


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

#: Reused scratch, keyed by shape *and stream*.  ``fwd`` allocated a fresh
#: workspace on every call, which measured 35 us against 20 us of actual kernel
#: time -- the allocation cost more than the work.
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

    whatever dtype the caller passes, both kernels index with
    INT32 on the device, so there is one canonical form and no second kernel
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


def chunk_metadata(cu_seqlens: torch.Tensor) -> ChunkMetadata:
    """Build (or reuse) the canonical INT32 metadata for ``cu_seqlens``.

    A content miss copies ``cu_seqlens`` to the host, which synchronizes; that is
    unavoidable, because the chunk counts are not knowable otherwise.  The
    identity cache remembers only the bounded cache key -- same tensor object,
    unmutated -- so every device-payload hit still performs the bounded cache's
    stream ordering and updates its LRU position.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [N + 1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")

    device = cu_seqlens.device
    cached_key = _META_IDENTITY.get(cu_seqlens)
    if cached_key is not None:
        cached = _META_CONTENT.get(device, cached_key)
        if cached is not None:
            return cached

    host = cu_seqlens.detach().to("cpu", torch.int64).tolist()
    if host[0] != 0:
        raise ValueError(f"cu_seqlens[0] must be 0, got {host[0]}")
    if any(b < a for a, b in zip(host, host[1:], strict=False)):
        raise ValueError("cu_seqlens must be non-decreasing")

    key = (tuple(host),)
    meta = _META_CONTENT.get(device, key)
    if meta is None:
        meta = _build_metadata(host, device)
        _META_CONTENT.put(device, key, meta, meta.device_tensors())
    _META_IDENTITY.put(cu_seqlens, key)
    return meta


def metadata_cache_stats(device: torch.device | int):
    return _META_CONTENT.stats(device)


def clear_metadata_cache() -> None:
    _META_CONTENT.clear()
    _META_IDENTITY.clear()


# ==========================================================================


# --------------------------------------------------------------------------
# Section 5: TMA descriptors
# --------------------------------------------------------------------------

#: Elements per 128-byte S128 segment, by element size.

DESCRIPTOR_BYTES = 128
#: Q, K, G, Kd, Qd, Ak.
NUM_TENSOR_MAPS = 6

#: 64 descriptor sets per device, LRU, for each cache.
DESCRIPTOR_MAX_ENTRIES = 64

#: Swizzle names accepted by :func:`encode_tensor_map`.  The recurrence needs
#: ``"NONE"`` for the two 512-byte records that prepare already laid out (plan
#: Section 12.2 Q2); everything else lands in the S128 image.
SWIZZLES = ("128B", "NONE")


# ---------------------------------------------------------------------------
# Shared: the encoder, the specification type, and the one geometry both
# kernels address.
# ---------------------------------------------------------------------------


def encode_tensor_map(
    dtype: torch.dtype,
    base_ptr: int,
    global_dim,
    strides_bytes,
    box_dim,
    *,
    swizzle: str = "128B",
) -> bytes:
    """Encode one ``cuTensorMapEncodeTiled`` descriptor as 128 raw bytes.

    Shared by prepare and the recurrence.  Every map fixes ``interleave=NONE``,
    unit element strides, 128-byte L2 promotion and ``OOB_FILL_NONE``; only the
    dtype, geometry and swizzle vary.
    """
    import cuda.bindings.driver as drv

    if dtype is torch.bfloat16:
        tma_dtype = drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    elif dtype is torch.float32:
        tma_dtype = drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    else:
        raise ValueError(f"unsupported TMA element type {dtype}")
    if swizzle not in SWIZZLES:
        raise ValueError(f"swizzle must be one of {SWIZZLES}, got {swizzle!r}")

    swizzle_enum = (
        drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle == "128B"
        else drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    )

    rank = len(global_dim)
    err, tmap = drv.cuTensorMapEncodeTiled(
        tma_dtype,
        rank,
        base_ptr,
        [drv.cuuint64_t(d) for d in global_dim],
        [drv.cuuint64_t(s) for s in strides_bytes],
        [drv.cuuint32_t(b) for b in box_dim],
        [drv.cuuint32_t(1)] * rank,
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_enum,
        drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    # cuda-python wraps the descriptor, so take its address via getPtr().
    return bytes(ctypes.string_at(tmap.getPtr(), DESCRIPTOR_BYTES))


@dataclass(frozen=True)
class TensorMapSpec:
    """Everything ``cuTensorMapEncodeTiled`` is given, and nothing else.

    Two roles that produce equal specs are the same descriptor.  ``role`` is
    deliberately absent from the comparison; it is carried alongside.
    """

    dtype: torch.dtype
    base_ptr: int
    global_dim: tuple[int, ...]
    global_stride_bytes: tuple[int, ...]
    box_dim: tuple[int, ...]
    swizzle: str

    def validate(self) -> None:
        """Check the alignment rules a wrong descriptor would otherwise hide."""
        element_bytes = 2 if self.dtype is torch.bfloat16 else 4
        if self.base_ptr % 16:
            raise ValueError(
                f"TMA global base must be 16-byte aligned, got {self.base_ptr}"
            )
        inner_bytes = self.box_dim[0] * element_bytes
        limit = 128 if self.swizzle == "128B" else 512
        if inner_bytes % 16 or inner_bytes > limit:
            raise ValueError(
                f"TMA inner box is {inner_bytes} B; must be a multiple of 16 "
                f"and at most {limit} for swizzle {self.swizzle}"
            )
        for extent in self.box_dim:
            if not 1 <= extent <= 256:
                raise ValueError(f"TMA box extent {extent} outside [1, 256]")
        for stride in self.global_stride_bytes:
            if stride % 16:
                raise ValueError(f"TMA global stride {stride} is not 16-byte aligned")
        for dim, box in zip(self.global_dim, self.box_dim, strict=True):
            if dim <= 0:
                raise ValueError(f"TMA global dim {dim} must be positive")
            if box > dim and box != self.box_dim[1]:
                # A box may exceed a degenerate outer dim only through the row
                # mode, which the tail path relies on; the inner and plane modes
                # must fit.
                raise ValueError(f"TMA box {box} exceeds global dim {dim}")

    def encode(self) -> bytes:
        return encode_tensor_map(
            self.dtype,
            self.base_ptr,
            self.global_dim,
            self.global_stride_bytes,
            self.box_dim,
            swizzle=self.swizzle,
        )


def _spec_fields(geometry: dict) -> dict:
    """Rename :func:`factor_slab_geometry`'s keys for :class:`TensorMapSpec`.

    The encoder takes ``strides_bytes``; the spec dataclass calls the same
    field ``global_stride_bytes``.  One adapter beats two copies of the tuple.
    """
    return dict(
        global_dim=geometry["global_dim"],
        global_stride_bytes=geometry["strides_bytes"],
        box_dim=geometry["box_dim"],
    )


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
        global_dim=(DK, rows, 3 * heads),
        strides_bytes=(DK * element_bytes, DK * rows * element_bytes),
        box_dim=(BF16_SEGMENT_ELEMS, BT, 1),
    )


# ---------------------------------------------------------------------------
# prepare: Q, K, G and the fused factor slab.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorMapSet:
    """One device buffer holding all six descriptors, plus their addresses."""

    storage: torch.Tensor
    q: int
    k: int
    g: int
    #: Kd, Qd and Ak share one descriptor over 3H planes; see _factor_map.
    factor: int


def _encode(
    dtype: torch.dtype, base_ptr: int, global_dim, strides_bytes, box_dim
) -> bytes:
    """Prepare's S128-only shorthand for :func:`encode_tensor_map`."""
    return encode_tensor_map(dtype, base_ptr, global_dim, strides_bytes, box_dim)


def _activation_map(t: torch.Tensor, total_tokens: int, heads: int, segment_elems: int):
    """Key-major ``(DK, T_total, H)`` view of contiguous ``[1, T, H, DK]``."""
    esz = t.element_size()
    return _encode(
        t.dtype,
        t.data_ptr(),
        global_dim=(DK, total_tokens, heads),
        strides_bytes=(heads * DK * esz, DK * esz),
        box_dim=(segment_elems, BT, 1),
    )


def _factor_map(ws_kd: torch.Tensor, rows: int, heads: int):
    """One descriptor over Kd, Qd and Ak as ``(DK, rows, 3H)``.

    The three regions are the same geometry, the same dtype and exactly
    adjacent -- ``prepare_region_offsets`` aligns each to 256 bytes and every
    region size is a multiple of it, so no padding is inserted -- which makes
    them 3H planes of one tensor rather than three tensors.  Plane ``r*H + h``
    is region ``r`` (0=Kd, 1=Qd, 2=Ak) of head ``h``.

    Geometry comes from :func:`factor_slab_geometry`, which the recurrence's
    ``"factor"`` role uses too -- the fused ``fwd`` entry hands *this* encoded
    descriptor to both kernels, so they cannot be allowed to disagree about it.
    Fusing the three regions also drops prepare's per-chunk
    ``fence_tensormap_acquire`` from three to one.
    """
    return _encode(
        ws_kd.dtype,
        ws_kd.data_ptr(),
        **factor_slab_geometry(rows, heads, ws_kd.element_size()),
    )


#: the design requires the prepare cache to carry the same event / LRU /
#: lifetime contract as the recurrence one; a plain dict keyed on addresses grew
#: without bound and never ordered a cross-stream hit against its own upload.
_TENSOR_MAP_CACHE = BoundedDeviceCache(
    "prepare-descriptors", max_entries=DESCRIPTOR_MAX_ENTRIES
)


def prepare_descriptor_cache_stats(device):
    return _TENSOR_MAP_CACHE.stats(device)


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
    """Build (and cache) the six The descriptors.

    Encoding six descriptors and copying them to the device costs far more than
    a launch, so the set is cached on the tensor addresses and shape.  The
    benchmark reuses its buffers, so this is a hit after the first call.
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
# recurrence (the design-7.4): seven roles, at most seven descriptors.
#
# The specifications are plain data and need no CUDA context, so the geometry,
# the factor-plane coordinates and the exact-alias reuse count are all testable
# on the host; encoding and upload happen separately.
# ---------------------------------------------------------------------------


#: Descriptor roles, in the order they are packed into the upload buffer.
ROLES = ("factor", "aq", "gt", "v", "out", "state_in", "state_out")

#: Plane order inside the fused factor descriptor.
FACTOR_PLANES = ("kd", "qd", "ak")


def factor_plane(factor: str, head: int, heads: int) -> int:
    """``Kd -> head``, ``Qd -> H + head``, ``Ak -> 2H + head``."""
    return FACTOR_PLANES.index(factor) * heads + head


def assert_factor_regions_are_fused(workspace, heads: int, total_chunks: int) -> None:
    """Refuse a workspace whose Kd/Qd/Ak are not one contiguous BF16 slab.

    the design requires this rather than a fallback: a temporary
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
    """Build the the design / 7.3 specifications as plain data."""
    rows = BT * total_chunks
    specs: dict[str, TensorMapSpec] = {
        # Kd/Qd/Ak fused: 3H planes of R rows of 128 BF16 keys.  The geometry
        # is prepare's -- literally the same function -- because the fused
        # entry gives prepare's encoded descriptor to this kernel as well.
        "factor": TensorMapSpec(
            dtype=torch.bfloat16,
            base_ptr=kd_ptr,
            swizzle="128B",
            **_spec_fields(factor_slab_geometry(rows, heads)),
        ),
        # The 16x16 pairwise record prepare already wrote in its SW32 image:
        # moved verbatim, so no second swizzle is applied here.
        "aq": TensorMapSpec(
            dtype=torch.bfloat16,
            base_ptr=aq_ptr,
            global_dim=(BT * BT, total_chunks, heads),
            global_stride_bytes=(BT * BT * 2, total_chunks * BT * BT * 2),
            box_dim=(BT * BT, 1, 1),
            swizzle="NONE",
        ),
        "gt": TensorMapSpec(
            dtype=torch.float32,
            base_ptr=gt_ptr,
            global_dim=(DK, total_chunks, heads),
            global_stride_bytes=(DK * 4, total_chunks * DK * 4),
            box_dim=(DK, 1, 1),
            swizzle="NONE",
        ),
    }
    # V and out are the same geometry over [1, T, H, 128]; when they are also
    # the same storage they collapse to one descriptor by equality.
    activation = dict(
        dtype=torch.bfloat16,
        global_dim=(DV, total_tokens, heads),
        global_stride_bytes=(heads * DV * 2, DV * 2),
        box_dim=(DV_HALF, BT, 1),
        swizzle="128B",
    )
    specs["v"] = TensorMapSpec(base_ptr=v_ptr, **activation)
    specs["out"] = TensorMapSpec(base_ptr=out_ptr, **activation)

    if state_in_ptr is not None or state_out_ptr is not None:
        if state_dtype is None:
            raise ValueError("state_dtype is required when a state is present")
        state = _state_spec_fields(state_dtype, sequences * heads)
        if state_in_ptr is not None:
            specs["state_in"] = TensorMapSpec(base_ptr=state_in_ptr, **state)
        if state_out_ptr is not None:
            specs["state_out"] = TensorMapSpec(base_ptr=state_out_ptr, **state)

    for spec in specs.values():
        spec.validate()
    return specs


def _state_spec_fields(state_dtype: torch.dtype, planes: int) -> dict:
    """one full-tile instruction moves a whole state half.

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
        global_dim=(inner, rows, planes),
        global_stride_bytes=(inner * element_bytes, rows * inner * element_bytes),
        box_dim=(inner, rows // 2, 1),
        swizzle="128B",
    )


def state_box_bytes(state_dtype: torch.dtype) -> int:
    """Transaction bytes of one state-half TMA (16 KiB BF16, 32 KiB FP32)."""
    fields = _state_spec_fields(state_dtype, 1)
    element_bytes = 2 if state_dtype is torch.bfloat16 else 4
    return fields["box_dim"][0] * fields["box_dim"][1] * element_bytes


def unique_descriptors(specs: dict[str, TensorMapSpec]) -> list[TensorMapSpec]:
    """Distinct descriptors, in first-use order; equality *is* exact alias."""
    seen: list[TensorMapSpec] = []
    for role in ROLES:
        spec = specs.get(role)
        if spec is not None and spec not in seen:
            seen.append(spec)
    return seen


def descriptor_count(specs: dict[str, TensorMapSpec]) -> int:
    return len(unique_descriptors(specs))


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


def descriptor_cache_key(specs: dict[str, TensorMapSpec]) -> tuple:
    return tuple((role, specs[role]) for role in ROLES if role in specs)


def get_recurrence_tensor_maps(
    specs: dict[str, TensorMapSpec], device: torch.device
) -> RecurrenceTensorMaps:
    """Cached :func:`build_recurrence_tensor_maps`.

    Encoding seven descriptors and copying 896 bytes to the device costs far
    more than the launch itself, so a steady-state call must hit.  A hit on
    another stream waits on the upload event; it never synchronizes the host.
    """
    key = descriptor_cache_key(specs)
    hit = _DESCRIPTORS.get(device, key)
    if hit is not None:
        return hit
    maps = build_recurrence_tensor_maps(specs, device)
    return _DESCRIPTORS.put(device, key, maps, (maps.storage,))


def recurrence_descriptor_cache_stats(device: torch.device | int):
    return _DESCRIPTORS.stats(device)


def clear_recurrence_descriptor_cache() -> None:
    _DESCRIPTORS.clear()


# --------------------------------------------------------------------------
# Section 6: recurrence host plan, arena and grid
# --------------------------------------------------------------------------

# --- Warp roles ------------------------------------------

LOAD_WARP = COMPUTE_WARPS  # 8
STORE_WARP = COMPUTE_WARPS + 1  # 9
REC_WARPS = COMPUTE_WARPS + 2  # 10
REC_THREADS = REC_WARPS * 32  # 320
#: Value columns one compute warp owns for the whole kernel.

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

#: One ``arrive_and_expect_tx`` covers all nine loads of a stage.
INPUT_STAGE_TX_BYTES = INPUT_STAGE_BYTES
#: 2 Kd + 2 Qd + 2 Ak + 1 Aq + 1 GTotal + 1 V.
INPUT_TMA_INSTRUCTIONS = 9

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
NUM_BARRIERS = BARRIER_BYTES // 8  # 15

SMEM_RAW_END = REC_SMEM_BARRIERS + BARRIER_BYTES  # 97400

#: the design fixes the launch size at 97,536 B and calls it the 128-byte
#: alignment of 97,400.  Rounding 97,400 up to 128 is 97,408; 97,536 is the
#: round-up to **256**, which is what the plan's own two other statements say --
#: "120 B barrier and the trailing alignment together take 256 B" and "3,840 B
#: remain against the 101,376 B ceiling".  The barrier arena therefore owns a
#: full 256-byte block, and 97,536 is the number the resource gate checks.
SMEM_ALIGNMENT = 256
SMEM_DYNAMIC_BYTES = (
    (SMEM_RAW_END + SMEM_ALIGNMENT - 1) // SMEM_ALIGNMENT * SMEM_ALIGNMENT
)  # 97536

#: The FP32 external-state conversion buffer borrows the head of the pipeline
#: union.  It is live only before the input ring starts and after it drains, so
#: it does not raise the peak.
SMEM_STATE_F32 = SMEM_INPUT
STATE_F32_BYTES = DV_HALF * DK * 4  # 32768

#: CC 12.0 per-CTA shared memory, and the driver's own per-CTA overhead.  The
#: launch parameter excludes the driver's share; residency does not.
REC_SMEM_PER_CTA_BYTES = 99 * 1024  # 101376
REC_SMEM_PER_SM_BYTES = 100 * 1024  # 102400
REC_DRIVER_SMEM_PER_BLOCK_BYTES = 1024
MIN_BLOCKS_PER_MP = 1

# --- Arrival counts --------------------------------------

INPUT_READY_ARRIVALS = 1  # completed by transaction bytes
INPUT_CONSUMED_ARRIVALS = COMPUTE_WARPS  # one per compute warp, not per thread
OUTPUT_READY_ARRIVALS = COMPUTE_WARPS
OUTPUT_CONSUMED_ARRIVALS = 1
STATE_READY_ARRIVALS = 1


def smem_regions() -> dict[str, tuple[int, int]]:
    """``{name: (offset, bytes)}`` for every region of the fixed arena."""
    regions: dict[str, tuple[int, int]] = {"state": (SMEM_STATE, STATE_BYTES)}
    for stage in range(INPUT_STAGES):
        regions[f"input{stage}"] = (
            SMEM_INPUT + stage * INPUT_STAGE_BYTES,
            INPUT_STAGE_BYTES,
        )
    for stage in range(OUTPUT_STAGES):
        regions[f"output{stage}"] = (
            SMEM_OUTPUT + stage * OUTPUT_STAGE_BYTES,
            OUTPUT_STAGE_BYTES,
        )
    regions["barriers"] = (REC_SMEM_BARRIERS, BARRIER_BYTES)
    return regions


def input_stage_base(stage: int) -> int:
    return SMEM_INPUT + stage * INPUT_STAGE_BYTES


def output_stage_base(stage: int) -> int:
    return SMEM_OUTPUT + stage * OUTPUT_STAGE_BYTES


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
    differ -- and L2 serves the second reader entirely: at H=64 the kernel's
    DRAM read is 572.9 MB against 570.4 MB of unique bytes, where re-reading
    the factors per half would be 1006.6 MB.  That reuse needs the two halves
    *co-resident*, and shared memory pins the recurrence at one CTA per SM, so
    co-resident means "in the same wave", which means "adjacent".

    The old ``(N, 2H, 1)`` issued blocks in the order ``seq + N * (2 * head +
    dv_half)``, separating the halves of a head by ``N`` blocks.  Past ``2N >
    SMs`` some pairs straddle a wave boundary and past ``N >= SMs`` all of them
    do: the first wave is every ``dv_half = 0`` CTA and the second is every
    ``dv_half = 1`` CTA, so every factor is read from DRAM twice.  Measured on
    a 188-SM SM120 device at T=2048, holding the work fixed and only
    refactoring ``P = N * H``:

        P=188 as 188x1 -> 573.7 us, L2 hit 0.07%, DRAM read 745.4 MB
        P=188 as  94x2 -> 382.1 us, L2 hit 37.7%, DRAM read 425.1 MB

    and at P=94 -- one wave, where the halves land together either way -- all
    four factorizations of P agree to within 3%, which is what makes the wave
    boundary and not the tensor shape the cause.

    ``(2N, H, 1)`` rather than ``(2, NH, 1)`` because ``maxGridSize[1]`` is
    65,535 and ``N * H`` overflows it well inside the supported range, while
    ``2N`` sits in ``maxGridSize[0] = 2^31 - 1`` and ``H`` is small.  Rather
    than ``(2H, N, 1)`` -- which also pairs the halves -- because a wave should
    cover one head's sequences, not one sequence's heads: the latter costs 4%
    at B=4, H=96 (197.7 us against 205.2 us).  That keeps the traversal order
    ``(N, 2H, 1)`` already had; only the DV half moves.
    """
    return (2 * sequences, heads, 1)


def dv_half_of(block_x: int) -> int:
    return block_x & 1


def seq_of(block_x: int) -> int:
    return block_x >> 1


def head_of(block_y: int) -> int:
    return block_y


# ---------------------------------------------------------------------------
# Host validation and launch
# ---------------------------------------------------------------------------

INT32_MAX = 2**31 - 1


def checked_i32(name: str, value) -> int:
    """Reject anything that would not survive the device's INT32 indexing.

    the design requires every derived extent to be checked on the host
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


def launch_recurrence(
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
    stream=None,
    plan_only: bool = False,
) -> RecurrencePlan | None:
    """Run the recurrence in place over ``out`` and the optional final state.

    With ``plan_only``, do every host-side step -- validation, descriptor
    encoding, stream recording -- and return the result instead of launching.
    ``fwd`` uses that to hand both kernels to one compiled entry, so the
    Python/compiled boundary is crossed once per forward rather than twice.

    Private: the design keeps ``fwd`` as the only public entry point.  The
    caller is responsible for the public ABI checks; what is enforced here is
    what the *kernel* depends on -- the fused factor slab, the state geometry
    and the INT32 index ranges.
    """

    device = v.device
    require_sm120a(device)
    sequences = len(cu_seqlens_host) - 1

    if total_chunks == 0:
        # the caller must take the host state-only fast path
        # before any descriptor exists.  A zero-extent factor descriptor is not
        # encodable, so failing here is the honest outcome rather than a
        # silently degenerate launch.
        raise ValueError(
            "launch_recurrence requires total_chunks == 0 to be handled by the "
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

    # every buffer this raw launch touches is recorded on the
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

    plan = RecurrencePlan(
        tensor_maps=tensor_maps,
        sequences=sequences,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        state_dtype=state_dtype,
    )
    if plan_only:
        return plan

    launch_recurrence_device(
        out=out,
        cu_seqlens_i32=cu_seqlens_i32,
        cu_chunks_i32=cu_chunks_i32,
        tensor_maps=tensor_maps,
        heads=heads,
        sequences=sequences,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        state_dtype=state_dtype,
        stream=stream,
    )
    return None


# --------------------------------------------------------------------------
# Section 7: prepare host plan, arena and residency
# --------------------------------------------------------------------------

PREP_WARPS = 4
PREP_THREADS = PREP_WARPS * 32

#: CPC 2 is the measured optimum on sm_120, the supported target:
#:
#:   fixed_h96   1: 971.20us   2: 960.00us   3: 965.12us   4: 974.94us
#:   fixed_h64   1: 630.11us   2: 623.84us   3: 631.17us   4: 633.79us
#:
#: Depth 2 is where the chunk prefetch first pays -- DRAM utilisation 90.51%
#: to 91.27%, CTA count halved -- and past 2 the extra CTA-wide barriers cost
#: more than the overlap wins.
DEFAULT_PREP_CHUNKS_PER_CTA = 2

#: The optimum moves with the memory system, so the default follows the device.
#: sm_120 saturates DRAM at ~90%, where extra chunks buy nothing and only add
#: barriers.  sm_90 (H20) sits at ~40%, where the prefetch has room to pay:
#:
#:   H20 fixed_h64   1: 903.9   2: 550.4   3: 506.6   4: 486.3   5: 508.1us
#:   H20 fixed_h96   2: 808.9   3: 741.0   4: 722.2   5: 783.6   6: 1100us
#:
#: 4 is the peak on both H20 cases and 5 is already past it; by 8 instruction
#: issue has collapsed under the unrolled chunk loop.
#:
#: sm_100 (B200) behaves like sm_90, not like sm_120 -- it also has bandwidth
#: to spare relative to this kernel, so the prefetch keeps paying past depth 2:
#:
#:   B200 fixed_h96   1: 1112.5   2: 560.4   3: 449.0   4: 424.2us
#:   B200 fixed_h64   1:  662.9   2: 361.8   3: 300.4   4: 288.2us
#:
#: Falling back to the sm_120 value of 2 there cost 1.32x on h96 and 1.26x on
#: h64, which is a large part of why the first B200 measurements were *slower*
#: than FlashKDA -- a default tuned for one memory system silently applied to
#: another.
#:
#: sm_90 and sm_100 are experiments and not targets (see target.py); their
#: entries here only make those measurements reproducible without an explicit
#: chunks_per_cta, and nothing in the test matrix exercises them.
DEFAULT_PREP_CHUNKS_PER_CTA_BY_CAPABILITY = {
    (12, 0): 2,
    (10, 0): 4,
    (9, 0): 4,
}

#: log2(e); the gate is evaluated in the log2 domain.
_LOG2_E = 1.4426950408889634

SUPPORTED_PREP_CHUNKS_PER_CTA = (1, 2, 3, 4)
SUPPORTED_MIN_BLOCKS_PER_SM = (1, 2, 3)

# --- SMEM arena byte offsets -------------------------------
# Layout when Ki and QK have their own stages, i.e. CPC > 1.
SMEM_KD_THEN_AK = 0
SMEM_QD = 4096
SMEM_Q_RAW = 8192
SMEM_K_RAW = 12288
SMEM_GATE_EXPG = 16384
SMEM_KI = 24576
SMEM_AINV = 28672
SMEM_GAMMA_BF16 = 29184
SMEM_BETA_ACT = 29440
PREP_SMEM_BARRIERS = 29568
SMEM_QK = 29616
SMEM_ARENA_BYTES = 30128

#: Six 64-bit mbarriers inside the 48-byte barrier arena.
PREP_MBAR_TMA0 = PREP_SMEM_BARRIERS + 0
PREP_MBAR_TMA1 = PREP_SMEM_BARRIERS + 8
PREP_MBAR_K_HALF_READY = PREP_SMEM_BARRIERS + 16
PREP_MBAR_K_FULL_READY = PREP_SMEM_BARRIERS + 24
PREP_MBAR_RAW_RELEASED = PREP_SMEM_BARRIERS + 32
PREP_MBAR_PAIRWISE_READY = PREP_SMEM_BARRIERS + 40

#: Expected TMA load bytes per chunk: Q 4096 + K 4096 + G 8192.
TMA_LOAD_BYTES_PER_CHUNK = 16384
#: Same with the opt-in BF16 G: only G moves, 8192 -> 4096.  The SMEM stage
#: stays 8192 bytes either way, because it is overwritten by FP32 ``exp_g``.
TMA_LOAD_BYTES_PER_CHUNK_G_BF16 = 12288

#: CC 12.0 shared-memory limits.
PREP_SMEM_PER_SM_BYTES = 100 * 1024
PREP_SMEM_PER_CTA_BYTES = 99 * 1024
#: Static SMEM does not need the ``cudaFuncSetAttribute`` opt-in below 48 KiB.
STATIC_SMEM_LIMIT_BYTES = 48 * 1024

#: The driver adds 1024 bytes per CTA on top of the arena (NCU: "Driver Shared
#: Memory Per Block 1.02 Kbyte"), so the residency thresholds are on
#: arena + PREP_DRIVER_SMEM_PER_BLOCK_BYTES, not on the arena alone.  A fourth
#: resident CTA therefore needs the arena at 102400/4 - 1024 = 24,576 B.
#:
#: Aliasing Ki onto the dead raw Q stage and QK onto raw K was tried and
#: reverted: it reaches 25,520 B, still 944 B short of four CTAs, and cost
#: 0.3-2.0% because Ki and Q sharing an address stops the compiler reordering
#: the Kd/Ki/Qd loop.  Getting to 24,576 needs a structural change, not another
#: alias -- gamma is live through that loop and beta from the prologue, so
#: neither can move onto a raw stage.
PREP_DRIVER_SMEM_PER_BLOCK_BYTES = 1024
SMEM_FOR_FOUR_CTAS = PREP_SMEM_PER_SM_BYTES // 4 - PREP_DRIVER_SMEM_PER_BLOCK_BYTES

#: A four-CTA arena is reachable -- at CPC == 1, Qd over raw Q plus Ki over raw
#: K frees 8192 B and lands here, with NCU confirming four resident CTAs and
#: 32.9% achieved occupancy.  It was measured and not kept: it is 0.24-2.36%
#: slower, because the kernel is at ~90% of peak DRAM and the extra warps have
#: nothing to issue, while every CTA-wide rendezvous gets longer.  Occupancy is
#: not this kernel's constraint; bytes are.
SMEM_ARENA_BYTES_FOUR_CTA = SMEM_ARENA_BYTES - 2 * 4096


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
            torch.bfloat16,  # INV_MMA_DTYPE, the design
            self.chunks_per_cta,
            self.min_blocks_per_sm,
        )


def default_chunks_per_cta_for(capability: tuple[int, int]) -> int:
    """Measured optimum for an architecture; the sm_120 value if unknown."""
    return DEFAULT_PREP_CHUNKS_PER_CTA_BY_CAPABILITY.get(
        (capability[0], capability[1]), DEFAULT_PREP_CHUNKS_PER_CTA
    )


def default_chunks_per_cta(device=None) -> int:
    """Measured optimum for ``device``, falling back when there is no GPU."""
    if not torch.cuda.is_available():
        return DEFAULT_PREP_CHUNKS_PER_CTA
    return default_chunks_per_cta_for(torch.cuda.get_device_capability(device))


def max_resident_ctas_from_smem(arena_bytes: int | None = None) -> int:
    """SMEM-side residency ceiling.

    Counts the driver's own per-CTA shared memory, without which this
    overestimates: 25,520 B looks like four CTAs and measures three.
    """
    if arena_bytes is None:
        arena_bytes = SMEM_ARENA_BYTES
    return PREP_SMEM_PER_SM_BYTES // (arena_bytes + PREP_DRIVER_SMEM_PER_BLOCK_BYTES)


def uses_dynamic_smem(arena_bytes: int = SMEM_ARENA_BYTES) -> bool:
    """Whether the arena needs the >48 KiB dynamic-SMEM opt-in."""
    return arena_bytes > STATIC_SMEM_LIMIT_BYTES


# ---------------------------------------------------------------------------
# Device-side index helpers.
#
# Written with ``//``, ``*``, ``+``, ``-`` and ``^`` only, so the same bodies
# evaluate for Python ``int`` on the host and ``cutlass.Int32`` on the device.
# They mirror Section 1 exactly.
# ---------------------------------------------------------------------------

raw_bf16_index = raw_bf16_s128
raw_f32_index = raw_f32_s128
kr_ak_index = kr_ak_bf16_s128
pairwise_index = pairwise_sw32


# ---------------------------------------------------------------------------
# Key-major global views
# ---------------------------------------------------------------------------


def activation_layout_strides(total_tokens: int, heads: int):
    """Strides of the CuTe view of contiguous ``[1, T_total, H, 128]``."""
    return (1, DK * heads, DK, DK * total_tokens * heads)


def workspace_layout_strides(total_chunks: int, heads: int):
    """Strides of the CuTe view of contiguous ``[1, H, total_chunks*16, 128]``."""
    rows = total_chunks * BT
    return (1, DK, DK * rows, DK * rows * heads)


def prepare_grid(total_chunks: int, heads: int, chunks_per_cta: int):
    """``(grid.x, grid.y, grid.z)``; the host must not launch with grid.x == 0."""
    return ((total_chunks + chunks_per_cta - 1) // chunks_per_cta, heads, 1)


def chunks_in_cta(total_chunks: int, block_x: int, chunks_per_cta: int) -> int:
    """CTA-uniform loop bound ``my_chunks``."""
    base = block_x * chunks_per_cta
    return max(0, min(chunks_per_cta, total_chunks - base))


def tma_slot_and_phase(local_chunk: int) -> tuple[int, int, int, int]:
    """``(tma_slot, tma_wait_phase, compute_wait_phase, beta_stage)``."""
    return (
        local_chunk & 1,
        (local_chunk >> 1) & 1,
        local_chunk & 1,
        local_chunk & 1,
    )


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
    ``os.getenv`` by way of nvml.  Measured on a 188-SM SM120 device, that pair
    cost 5.0-5.5 us per call on the four shapes whose kernels are too short to
    hide it -- 23.2 us to 28.7 us on ``fk-fixed-h32-t256`` -- and nothing at
    all above ``multiwave``, where the launch queue absorbs it.  A cached launch
    plan is pinned to one stream and one ``A_log`` by construction, so for that
    caller the key is a constant and belongs in the plan.
    """
    if key is None:
        key = _a_log_exp_key(a_log)
    cached = _A_LOG_EXP.get(key)
    if cached is not None:
        ref, buffer, version = cached
        if ref() is a_log:
            if out is None:
                out = buffer
            # `exp2(a_log * log2e)` is two dispatches and two kernels, measured
            # at 9.3 us against 2.2 us of device time.  It only has to run when
            # a_log has actually changed, and an optimizer step bumps the
            # version counter.  A tensor with no counter -- anything created
            # under inference_mode -- is rejected explicitly, so it recomputes
            # rather than trusting an identity it cannot check.  The sentinel
            # compares equal to itself, so testing it by value would not do
            # that.
            #
            # Never skip inside a capture: the graph replays only what it
            # recorded, so leaving exp2 out of it means a later parameter update
            # is silently ignored on replay.  That exact bug shipped once.
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

    ``fwd`` needs these before it can hand both kernels to one compiled entry.
    Everything here is cached on the buffer addresses, so a steady-state call
    re-encodes nothing.
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
        a_log_exp=a_log_exp_for(A_log, _LOG2_E),
        grid_x=(total_chunks + config.chunks_per_cta - 1) // config.chunks_per_cta,
    )


def launch_prepare(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    workspace: PrepareWorkspace,
    cu_seqlens: torch.Tensor,
    config: PrepareConfig | None = None,
) -> None:
    """Write the five prepare outputs into ``workspace`` in place."""
    require_sm120a(q.device)
    config = config or PrepareConfig(
        safe_gate=True, chunks_per_cta=default_chunks_per_cta(q.device)
    )

    meta = chunk_metadata(cu_seqlens)
    if meta.total_chunks == 0:
        # return without launching; grid.x == 0 is invalid.
        return

    launch_prepare_device(
        q=q,
        k=k,
        g=g,
        beta=beta,
        a_log_exp=a_log_exp_for(A_log, LOG2_E),
        dt_bias=dt_bias,
        # Narrowed once when the metadata was built.  ``cu_seqlens.to(int32)``
        # here would allocate on every call whenever the caller holds INT64,
        # which is the common case.
        cu_seqlens=meta.cu_seqlens,
        cu_chunks=meta.cu_chunks,
        chunk_to_seq=meta.chunk_to_seq,
        workspace=workspace,
        scale=float(scale),
        lower_bound=float(lower_bound),
        heads=q.shape[2],
        total_tokens=q.shape[1],
        total_chunks=meta.total_chunks,
        config=config,
    )


# --------------------------------------------------------------------------
# Section 8: the prepare device kernel
# --------------------------------------------------------------------------

#: The inverse is computed as two 8x8 block inverses plus one coupling term
#:; this is the block split, not a tunable.
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

PREFIX_FLOOR = -126.0
NORM_SS_FLOOR = 1.0e-24
LOG2_E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Device index helpers (mirror Section 1)
# ---------------------------------------------------------------------------


@cute.jit
def raw_bf16_idx(token, dim):
    seg = dim // BF16_SEG_ELEMS
    local = dim - seg * BF16_SEG_ELEMS
    group = local // 8
    inner = local - group * 8
    return (
        seg * BF16_SEG_STRIDE
        + token * BF16_SEG_ELEMS
        + (group ^ (token & 7)) * 8
        + inner
    )


@cute.jit
def raw_f32_idx(token, dim):
    seg = dim // F32_SEG_ELEMS
    local = dim - seg * F32_SEG_ELEMS
    group = local // 4
    inner = local - group * 4
    return (
        seg * F32_SEG_STRIDE + token * F32_SEG_ELEMS + (group ^ (token & 7)) * 4 + inner
    )


@cute.jit
def raw_g_idx(token, dim, G_FP32: cutlass.Constexpr):
    """SMEM index of ``G[token, dim]``, in whichever image its dtype implies.

    FP32 ``G`` gets its own 4-segment S128 image; BF16 ``G`` is byte-identical
    in shape to Q and K, so it simply reuses theirs.  Both are 128-byte
    segments -- only the element count per segment differs.
    """
    if cutlass.const_expr(G_FP32):
        return raw_f32_idx(token, dim)
    return raw_bf16_idx(token, dim)


@cute.jit
def kr_ak_idx(token, dim):
    return raw_bf16_idx(token ^ 8, dim)


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


@cute.jit
def bf16_round(x: cutlass.Float32) -> cutlass.Float32:
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


def f16_round(x: cutlass.Float32) -> cutlass.Float32:
    """Round through FP16, the inverse chain's operand dtype (Section 8.2)."""
    return x.to(cutlass.Float16).to(cutlass.Float32)


# ---------------------------------------------------------------------------
# Fragment helpers
# ---------------------------------------------------------------------------


@cute.jit
def load_a_fragment(smem, token_base, dim_base, lane):
    """A-operand ``ldmatrix.x4``."""
    matrix_id = lane // 8
    row = token_base + (lane % 8) + 8 * (matrix_id % 2)
    col = dim_base + 8 * (matrix_id // 2)
    return ldmatrix_x4(smem + raw_bf16_idx(row, col))


@cute.jit
def load_b_fragment(smem, dim_base, lane):
    """B-operand ``ldmatrix.x4``.

    The row half is keyed on bit 4 of the lane, not bit 3.
    """
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (lane // 16)
    col = dim_base + 8 * (matrix_id % 2)
    return ldmatrix_x4(smem + raw_bf16_idx(row, col))


@cute.jit
def load_pairwise_a_fragment(smem, lane):
    """Load a 16x16 pairwise tile in A layout."""
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (matrix_id % 2)
    col = 8 * (matrix_id // 2)
    return ldmatrix_x4(smem + prepare_pair_idx(row, col))


@cute.jit
def a_to_b(a0, a1, a2, a3):
    """A layout -> B layout of the same matrix; identity register order."""
    return (
        movmatrix_b16(a0),
        movmatrix_b16(a1),
        movmatrix_b16(a2),
        movmatrix_b16(a3),
    )


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
def mma_16x16(a0, a1, a2, a3, b0, b1, b2, b3, c):
    """One logical m16n16k16 step: two native N=8 MMAs."""
    n0 = mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_bf16(a0, a1, a2, a3, b2, b3, c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


@cute.jit
def mma_16x16_f16(a0, a1, a2, a3, b0, b1, b2, b3, c):
    """``mma_16x16`` with FP16 operands, for the inverse (Section 8.2).

    Same shape, same accumulator width, same two native N=8 issues -- only the
    operand dtype differs, so this costs exactly what the BF16 form costs.
    """
    n0 = mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_f16(a0, a1, a2, a3, b2, b3, c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


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
    for ``k_full_ready`` before blocks 4-7 (step 5 of the design).
    """
    for j in cutlass.range_constexpr(4):
        d0 = (half * 4 + j) * 16
        a0, a1, a2, a3 = load_a_fragment(lhs_ptr, 0, d0, lane)
        b0, b1, b2, b3 = load_b_fragment(ki_ptr, d0, lane)
        c = mma_16x16(a0, a1, a2, a3, b0, b1, b2, b3, c)
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
    the lot, which is the whole point of TMA here -- the ``cp.async`` version
    this replaces needed 64 threads to move the same bytes, so it could not be
    confined to one warp the way step 6 of the design describes.

    Coordinates are ``(segment * segment_elems, token_base, head)`` against the
    key-major descriptors of Section 7.1.
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
def clear_tail_rows(p_q, p_k, p_g, valid_rows, tidx, G_FP32: cutlass.Constexpr):
    """Zero the invalid rows of the raw stages.

    TMA only zero-fills coordinates outside the *tensor*, and a short chunk sits
    mid-tensor: the rows past ``valid_rows`` belong to the next sequence in the
    packed layout, so TMA faithfully loads real data there.  The ``cp.async``
    version this replaces got the zeros for free by passing src-size 0.

    Same 16-byte task map as the loads, so the writes stay one vector wide.
    """
    for rep in cutlass.range_constexpr(2):
        slot = tidx + rep * PREPARE_DEVICE_THREADS
        row = slot // 16
        d0 = (slot - row * 16) * 8
        if row >= valid_rows:
            zero8 = make_rmem_tensor(8, cutlass.BFloat16)
            for i in cutlass.range_constexpr(8):
                zero8[i] = cutlass.BFloat16(0.0)
            bidx = raw_bf16_idx(row, d0)
            store_vec8_bf16(p_q, bidx, zero8)
            store_vec8_bf16(p_k, bidx, zero8)

    if cutlass.const_expr(G_FP32):
        for rep in cutlass.range_constexpr(4):
            slot = tidx + rep * PREPARE_DEVICE_THREADS
            row = slot // 32
            d0 = (slot - row * 32) * 4
            if row >= valid_rows:
                zero4 = make_rmem_tensor(4, cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    zero4[i] = cutlass.Float32(0.0)
                cute.autovec_copy(zero4, vec_at(p_g, raw_f32_idx(row, d0), 4))
    else:
        # BF16 G is the same 4096-byte image as Q and K, so it takes the same
        # two-rep, 16-byte task map rather than the FP32 four-rep one.
        for rep in cutlass.range_constexpr(2):
            slot = tidx + rep * PREPARE_DEVICE_THREADS
            row = slot // 16
            d0 = (slot - row * 16) * 8
            if row >= valid_rows:
                zero8 = make_rmem_tensor(8, cutlass.BFloat16)
                for i in cutlass.range_constexpr(8):
                    zero8[i] = cutlass.BFloat16(0.0)
                store_vec8_bf16(p_g, raw_bf16_idx(row, d0), zero8)


@cute.jit
def warp_arrive(mbar, lane):
    """One arrival per warp on a 4-warp mbarrier.

    ``mbarrier.arrive`` counts per thread, so a whole warp arriving would
    overshoot an arrival count of ``PREPARE_DEVICE_WARPS``.  The warp synchronization before
    the elected arrival is what makes the other 31 lanes' shared-memory stores
    visible to whoever observes the arrival; the arrival itself carries release
    semantics at CTA scope for the electing lane.
    """
    cute.arch.sync_warp()
    if lane == 0:
        cute.arch.mbarrier_arrive(mbar)


@cute.jit
def load_beta_stage(gbeta, smem_beta, stage, token_base, valid_rows, head, lane, heads):
    """Activate one chunk's 16 beta logits into beta stage ``stage``.

    the design double-buffers ``smem_beta_act`` so that this strided
    column read out of ``[T, H]`` -- which cannot coalesce, one sector per
    token -- is issued a full chunk before the values are needed, instead of
    stalling the whole CTA at a barrier behind its DRAM latency.
    """
    if lane < BT:
        bv = cutlass.Float32(0.0)
        if lane < valid_rows:
            logit = cutlass.Float32(gbeta[(token_base + lane) * heads + head])
            half = cutlass.Float32(0.5)
            # Stored as FP32, unrounded.  The activated value has exactly two
            # consumers: the strict-lower scale, which is an FP32 multiply, and
            # the AINV column scale, which packs to BF16 itself.  Rounding here
            # is invisible to the second and only lossy to the first, at the
            # cost of two F2F conversions per lane (measured).
            bv = (
                cutlass.Float32(cute.math.tanh(logit * half, fastmath=True)) * half
                + half
            )
        smem_beta[stage * BT + lane] = bv


@cute.jit
def vec_at(ptr, idx, elems):
    """``ptr + idx`` as an ``elems``-long tensor, keeping the pointer alignment.

    ``Pointer.__add__`` lowers the pointer's ``alignment`` attribute to one
    element whenever the offset is dynamic -- even for ``ptr + 8 * dyn``, since
    it does not reason about the multiplier.  ``autovec_copy`` honours that
    attribute, so without this every 8-element BF16 access lowers to eight
    ``STG.E.U16`` / ``STS.U16`` instructions instead of one 128-bit access.

    Every index reaching this helper is a multiple of ``elems`` by
    construction: ``raw_bf16_s128`` and ``raw_f32_s128`` only add multiples of
    the 8- or 4-element group once ``dim`` is group-aligned (their ``inner``
    term is then zero), the workspace row bases are multiples of ``DK``, and
    the Section 7.4 ``Aq`` pair starts are even.  Stating that divisor costs no
    instructions -- it is a compile-time constraint, not a round-up.
    """
    return cute.make_tensor(
        ptr + cute.assume(cutlass.Int32(idx), divby=elems), cute.make_layout(elems)
    )


@cute.jit
def vec8_bf16(ptr, idx):
    """Load 8 contiguous BF16 (16 bytes) into a register fragment."""
    frag = make_rmem_tensor(8, cutlass.BFloat16)
    cute.autovec_copy(vec_at(ptr, idx, 8), frag)
    return frag


@cute.jit
def vec4_f32(ptr, idx):
    """Load 4 contiguous FP32 (16 bytes) into a register fragment."""
    frag = make_rmem_tensor(4, cutlass.Float32)
    cute.autovec_copy(vec_at(ptr, idx, 4), frag)
    return frag


@cute.jit
def store_vec8_bf16(ptr, idx, frag):
    """Store an 8-element BF16 fragment as one 16-byte access."""
    cute.autovec_copy(frag, vec_at(ptr, idx, 8))


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
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_factor: cutlass.Int64,
    SCALE: cutlass.Float32,
    GATE_SCALE_LOG2: cutlass.Float32,
    TOTAL_CHUNKS: cutlass.Int32,
    heads: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    head = bidy

    alloc = cutlass.utils.SmemAllocator()
    p_kd = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    # Four resident CTAs are reachable and were measured: at CPC == 1 the raw
    # Q and K stages are dead after the Kd/Ki/Qd loop, so Qd can overwrite Q
    # and Ki can overwrite K -- correct without added synchronization, since
    # (my_token, d0) -> thread is a bijection and each thread rewrites exactly
    # what it read.  That frees 8192 B, giving 21,936 B and occupancy_limit
    # shared_mem == 4, achieved occupancy 32.9%.
    #
    # It measures SLOWER everywhere: +0.24% h96 FP32, +1.58% h96 BF16, +1.64%
    # h64 FP32, +2.36% h64 BF16, with barrier stall roughly doubling (2.837 ->
    # 4.945) and warps_eligible essentially flat (0.158 -> 0.164).  Occupancy
    # was never the constraint: the kernel sits at ~90% of peak DRAM, so extra
    # warps have nothing to issue and four CTAs merely lengthen every CTA-wide
    # rendezvous.  Kept unaliased.
    p_qd = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_q = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_k = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_g = alloc.allocate_array(cutlass.Float32, BT * DK)
    p_ki = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_ainv = alloc.allocate_array(cutlass.BFloat16, BT * BT)
    p_gamma = alloc.allocate_array(cutlass.BFloat16, DK)
    # the design gives smem_beta_act 128 bytes: two 16-float stages, so a
    # chunk's beta is loaded and activated one chunk ahead of its use.
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
    # step 1 of the design: nothing may arrive or wait before the
    # initialized state is published.
    cute.arch.barrier()

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

            # Section 7.3: a short chunk sits mid-tensor, so TMA loaded the
            # next sequence's rows rather than zeros.  Clear them before the
            # barrier that publishes the stage.
            if valid_rows < BT:
                clear_tail_rows(p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32)
            cute.arch.barrier()

            # ---- gate prefix --------------------------
            gate_regs = [cutlass.Float32(0.0) for _ in range(BT)]
            for r in cutlass.range_constexpr(BT):
                if cutlass.const_expr(G_FP32):
                    raw = smem_g[raw_f32_idx(r, tidx)]
                else:
                    raw = cutlass.Float32(smem_g_bf16[raw_bf16_idx(r, tidx)])
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
                smem_g[raw_f32_idx(r, tidx)] = gate_regs[r]
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
                fq = vec8_bf16(p_q, raw_bf16_idx(my_token, d0))
                fk = vec8_bf16(p_k, raw_bf16_idx(my_token, d0))
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
                if qf < cutlass.Float32(NORM_SS_FLOOR):
                    qf = cutlass.Float32(NORM_SS_FLOOR)
                kf = k_ss
                if kf < cutlass.Float32(NORM_SS_FLOOR):
                    kf = cutlass.Float32(NORM_SS_FLOOR)
                q_inv = cutlass.Float32(cute.math.rsqrt(qf, fastmath=True))
                k_inv = cutlass.Float32(cute.math.rsqrt(kf, fastmath=True))

            s16 = bf16_round(SCALE)
            for h in cutlass.range_constexpr(2):
                d0 = 64 * h + 8 * lane_in_row
                bidx = raw_bf16_idx(my_token, d0)
                fq = vec8_bf16(p_q, bidx)
                fk = vec8_bf16(p_k, bidx)
                fg0 = vec4_f32(p_g, raw_f32_idx(my_token, d0))
                fg1 = vec4_f32(p_g, raw_f32_idx(my_token, d0 + 4))

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
                # step 5 of the design: signal each Kd/Ki half the moment it
                # is in SMEM.  Kd and Qd leave for global by TMA later, read
                # straight out of these same images (Section 7.4), so there is
                # no separate global store here any more.
                if cutlass.const_expr(h == 0):
                    warp_arrive(mbar_k_half, lane)
                else:
                    warp_arrive(mbar_k_full, lane)

            # This warp is done with raw Q/K/exp-g and with Qd (Section 12.3
            # step 5); the raw stages may be overwritten once all four arrive.
            warp_arrive(mbar_raw_released, lane)
            phase = lc & 1

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

                # Blockwise 8x8 inverse.  With D the block
                # diagonal of L and A21 its one off-diagonal block:
                #
                #   Binv = (I - D)(I + D^2)(I + D^4)   exact, blocks nilpotent
                #                                      at 8, so 3 factors
                #   AINV = Binv - Binv @ A21 @ Binv    exact, (Binv @ A21)^2 = 0
                #
                # Six MMAs, the same count as the 16x16 Neumann chain this
                # replaces, and one fewer pack/round stage -- but it never
                # forms a power above D^4 of an 8x8 block, where the Neumann
                # chain built L^8 of the full matrix and cancelled it away.
                # Operands are FP16, not BF16 (Section 8.2).
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
                    pb = a_to_b(dp[0], dp[1], dp[2], dp[3])
                    sq = mma_16x16_f16(
                        dp[0], dp[1], dp[2], dp[3], pb[0], pb[1], pb[2], pb[3], ZERO8()
                    )
                    dp = acc_to_a_fragment_f16(sq)
                    pows.append(dp)

                # binv = (I - D) (I + D^2) (I + D^4), as a running accumulator:
                # every D^k here is a power of a strict-lower matrix and so has
                # a zero diagonal, which makes R(I + D^k) == I + R(D^k) and
                # therefore MMA(x, I + D^k) == x + MMA(x, D^k).  Materializing
                # I + D^k instead would cost a second pack per step in the
                # issue-bound warp-0 region -- measured at +0.85% on fixed_h96.
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
                    pb = a_to_b(pf[0], pf[1], pf[2], pf[3])
                    prod = mma_16x16_f16(
                        lhs[0],
                        lhs[1],
                        lhs[2],
                        lhs[3],
                        pb[0],
                        pb[1],
                        pb[2],
                        pb[3],
                        ZERO8(),
                    )
                    for slot in cutlass.range_constexpr(8):
                        binv[slot] = f16_round(binv[slot]) + prod[slot]

                # x21 = -(Binv @ A21) @ Binv, written into the lower-left block.
                bf = acc_to_a_fragment_f16(tuple(binv))
                af = acc_to_a_fragment_f16(tuple(a21))
                ab = a_to_b(af[0], af[1], af[2], af[3])
                t1 = mma_16x16_f16(
                    bf[0], bf[1], bf[2], bf[3], ab[0], ab[1], ab[2], ab[3], ZERO8()
                )
                tf = acc_to_a_fragment_f16(t1)
                bb = a_to_b(bf[0], bf[1], bf[2], bf[3])
                x21 = mma_16x16_f16(
                    tf[0], tf[1], tf[2], tf[3], bb[0], bb[1], bb[2], bb[3], ZERO8()
                )

                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = binv[slot]
                    if row >= HALF_BT and col < HALF_BT:
                        v = -x21[slot]
                    smem_ainv[prepare_pair_idx(row, col)] = bf16_round(v).to(
                        cutlass.BFloat16
                    )
                # step 7 of the design.  This arrival publishes AINV and
                # also proves warp 0 is done reading smem_kd_then_ak, which is
                # what lets warps 1 and 3 overwrite their Kd half with Ak.
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
            # step 6 of the design, now literal: one elected lane issues
            # the whole 16 KB, so only warp 1 stops for the prefetch.  The
            # cp.async version needed 64 threads and had to borrow warp 3 too.
            if warp_id == 1:
                cute.arch.mbarrier_wait(mbar_raw_released, phase)
                # publish the ordinary Kd stores to the async
                # shared proxy before the TMA engine reads them, and converge
                # the warp so lane 0 speaks for all 32 lanes' writes.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(desc_factor, p_kd, 0, gchunk * BT, head)
                    tma_store_commit_group()
                if lc + 1 < CPC:
                    if lc + 1 < my_chunks:
                        nchunk = gchunk + 1
                        nseq = cutlass.Int32(gchunk_to_seq[nchunk])
                        nlc = nchunk - cutlass.Int32(gcu_chunks[nseq])
                        ntb = cutlass.Int32(gcu_seqlens[nseq]) + nlc * BT
                        nvr = cutlass.Int32(gcu_seqlens[nseq + 1]) - ntb
                        if nvr > BT:
                            nvr = cutlass.Int32(BT)
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
                        load_beta_stage(
                            gbeta,
                            smem_beta,
                            (lc + 1) & 1,
                            ntb,
                            nvr,
                            head,
                            lane,
                            heads,
                        )
            elif warp_id == 3:
                # Warp 3 no longer issues any of the prefetch, but it still
                # acquires raw_operands_released: that is what makes the other
                # warps' Ki stores visible to its Ak path below, without
                # relying on a chained release/acquire through warp 0.
                cute.arch.mbarrier_wait(mbar_raw_released, phase)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    # Section 7.4: Kd segment 1 plus both Qd segments, one group.
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
            # pairwise_ready is a plain release/acquire pair on smem_ainv, whose
            # only writer is warp 0.  The Ki that warps 1 and 3 read below is
            # covered without leaning on chained visibility: warp 2 acquired
            # k_full_ready directly, and warps 1 and 3 acquired
            # raw_operands_released, on which every warp arrives after its
            # k_full_ready arrival and therefore after its Ki stores.
            if warp_id != 0:
                cute.arch.mbarrier_wait(mbar_pairwise, phase)
                ai = load_pairwise_a_fragment(p_ainv, lane)
                bst = beta_stage * BT
                blo = pack_bf16x2(smem_beta[bst + 2 * q4], smem_beta[bst + 2 * q4 + 1])
                bhi = pack_bf16x2(
                    smem_beta[bst + 2 * q4 + 8], smem_beta[bst + 2 * q4 + 9]
                )
                ab0 = mul_bf16x2(ai[0], blo)
                ab1 = mul_bf16x2(ai[1], blo)
                ab2 = mul_bf16x2(ai[2], bhi)
                ab3 = mul_bf16x2(ai[3], bhi)

                if warp_id == 2:
                    bb = a_to_b(ab0, ab1, ab2, ab3)
                    qk_a = load_pairwise_a_fragment(p_qk, lane)
                    aq = mma_16x16(
                        qk_a[0],
                        qk_a[1],
                        qk_a[2],
                        qk_a[3],
                        bb[0],
                        bb[1],
                        bb[2],
                        bb[3],
                        ZERO8(),
                    )
                    base = (head * TOTAL_CHUNKS + gchunk) * (BT * BT)
                    # the design.  Stage through SMEM so the global
                    # store is one contiguous 16-byte run per lane.  Direct
                    # stores cannot be: prepare_pair_idx swaps the column halves
                    # (col ^ 8), so one store instruction only ever fills half
                    # of each row's 32 bytes, spraying 128 B over 8 sectors.
                    # The four instructions did tile the 512 B exactly, but L1
                    # does not merge across instructions, so 32 sectors left
                    # the SM where 16 would do.  Measured, fixed_h96: STG 8 ->
                    # 5 instructions per chunk, L1 request 48 -> 32 sectors,
                    # and the kernel's whole L1->L2 write becomes 654.31 MB
                    # against a native output of 49,152 x 13,312 = 654,311,424
                    # B -- exactly 1.000x.
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
                    # Section 12.3 step 8: pairwise_ready above proved warp 0 is
                    # done reading smem_kd_then_ak; this waits for the warp's own
                    # Kd TMA store to have finished *reading* its half, which is
                    # the other half of the condition for overwriting it.  The
                    # warp synchronization joins the two before any lane writes.
                    if lane == 0:
                        tma_store_wait_read(0)
                    cute.arch.sync_warp()

                    at0, at1, at2, at3 = a_to_a_transposed(ab0, ab1, ab2, ab3)
                    tile_base = 0
                    if warp_id == 3:
                        tile_base = 4
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
                        kb = a_to_b(
                            mul_bf16x2(ki0, gl),
                            mul_bf16x2(ki1, gl),
                            mul_bf16x2(ki2, gh),
                            mul_bf16x2(ki3, gh),
                        )
                        akc = mma_16x16(
                            at0, at1, at2, at3, kb[0], kb[1], kb[2], kb[3], ZERO8()
                        )
                        # publish Ak.T with stmatrix.x4 through
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

                    # Section 7.4: each store warp moves its own Ak segment.
                    # This is what removes the CTA barrier and the 128-thread
                    # re-read the vector-store version needed -- the warp that
                    # produced the half is the warp that ships it.
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
                        # Section 12.3 step 8: the source stage cannot be reused
                        # until the store has read it.
                        tma_store_wait_read(0)

            # Chunk recycle (Section 12.3 step 9).
            cute.arch.barrier()


@cute.jit
def _prepare_entry(
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
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_factor: cutlass.Int64,
    scale: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    total_chunks: cutlass.Int32,
    heads: cutlass.Int32,
    grid_x: cutlass.Int32,
    stream,
    SAFE_GATE: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
):
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
        desc_q,
        desc_k,
        desc_g,
        desc_factor,
        scale,
        gate_scale_log2,
        total_chunks,
        heads,
        SAFE_GATE,
        CPC,
        G_FP32,
    ).launch(
        grid=(grid_x, heads, 1),
        block=(PREPARE_DEVICE_THREADS, 1, 1),
        stream=stream,
    )


_PREPARE_DEVICE_CACHE: dict = {}


def _flat(t: torch.Tensor):
    """Flat CuTe view, cached on the address (see :func:`~.runtime.flat_view`)."""
    return flat_view(t, align=16)


def launch_prepare_device(
    *,
    q,
    k,
    g,
    beta,
    a_log_exp,
    dt_bias,
    cu_seqlens,
    cu_chunks,
    chunk_to_seq,
    workspace,
    scale,
    lower_bound,
    heads,
    total_tokens,
    total_chunks,
    config,
):
    import cuda.bindings.driver as cuda_driver

    grid_x = (total_chunks + config.chunks_per_cta - 1) // config.chunks_per_cta
    # The tensors' device, not the current one; see fused/launch.py.
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    # Cached on the buffer addresses, so a steady-state launch does not pay for
    # encoding descriptors.
    tmaps = build_tensor_maps(
        q=q,
        k=k,
        g=g,
        ws_kd=workspace.kd,
        ws_qd=workspace.qd,
        ws_ak=workspace.ak,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        heads=heads,
    )

    args = (
        _flat(q),
        _flat(k),
        _flat(g),
        _flat(beta),
        _flat(a_log_exp),
        _flat(dt_bias),
        _flat(cu_seqlens),
        _flat(cu_chunks),
        _flat(chunk_to_seq),
        _flat(workspace.kd),
        _flat(workspace.qd),
        _flat(workspace.ak),
        _flat(workspace.aq),
        _flat(workspace.g_total),
        cutlass.Int64(tmaps.q),
        cutlass.Int64(tmaps.k),
        cutlass.Int64(tmaps.g),
        cutlass.Int64(tmaps.factor),
        cutlass.Float32(scale),
        cutlass.Float32(lower_bound * LOG2_E),
        cutlass.Int32(total_chunks),
        cutlass.Int32(heads),
        cutlass.Int32(grid_x),
        stream,
    )
    g_fp32 = g.dtype is torch.float32
    # Device is part of the in-process key because a loaded callable is tied to
    # its CUDA context. Heads and grid remain runtime values.
    key = (
        q.device.index,
        bool(config.safe_gate),
        int(config.chunks_per_cta),
        g_fp32,
    )
    compiled = _PREPARE_DEVICE_CACHE.get(key)
    if compiled is None:
        with torch.cuda.device(q.device):
            compiled = cute.compile(
                _prepare_entry,
                *args,
                bool(config.safe_gate),
                int(config.chunks_per_cta),
                g_fp32,
            )
        _PREPARE_DEVICE_CACHE[key] = compiled
    compiled(*args)


# --------------------------------------------------------------------------
# Section 9: the recurrence device kernel
# --------------------------------------------------------------------------

# Absolute imports only: the DSL's AST preprocessor re-executes this module's
# import list when it traces a ``@cute.jit`` body, and it cannot resolve a
# relative ``from . import x``.

KEY_BLOCKS = DK // BT  # 8

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
def store_vec4_f32(ptr, idx, frag):
    cute.autovec_copy(frag, vec_at(ptr, idx, 4))


@cute.jit
def zero_acc4():
    z = cutlass.Float32(0.0)
    return (z, z, z, z)


@cute.jit
def mma_n8(a, b, c):
    """One native ``m16n8k16``: A is four registers, B two, C four."""
    return mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])


# ---------------------------------------------------------------------------
# State entry and exit
# ---------------------------------------------------------------------------


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

    even an FP32 external state crosses this boundary, because
    the state carried between chunks is BF16 by contract.
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
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
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
    # the plan's number and every address is a compile-time constant offset.
    alloc = cutlass.utils.SmemAllocator()
    p_base = alloc.allocate(SMEM_DYNAMIC_BYTES, 1024)
    p16 = cute.recast_ptr(p_base, dtype=cutlass.BFloat16)
    p32 = cute.recast_ptr(p_base, dtype=cutlass.Float32)
    p64 = cute.recast_ptr(p_base, dtype=cutlass.Int64)

    p_state = p16 + (SMEM_STATE // 2)
    p_state_f32 = p32 + (SMEM_STATE_F32 // 4)
    p_bar = p64 + (REC_SMEM_BARRIERS // 8)

    if warp_id == 0:
        if lane == 0:
            for s in cutlass.range_constexpr(INPUT_STAGES):
                cute.arch.mbarrier_init(p_bar + BAR_IN_READY + s, INPUT_READY_ARRIVALS)
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

    # --- sequence coordinates ---------------------------
    token_start = cutlass.Int32(gcu_seqlens[seq])
    token_end = cutlass.Int32(gcu_seqlens[seq + 1])
    chunk_base = cutlass.Int32(gcu_chunks[seq])
    num_chunks = cutlass.Int32(gcu_chunks[seq + 1]) - chunk_base
    state_plane = seq * heads + head

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
        # in; pass 1 wants the B operand and gets it with one ``movmatrix`` per
        # register.  ``tests/test_recurrence_layouts.py`` enumerates that
        # equality, 64 of 64.
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

            # The nine IKET ranges below -- five here, two on the producer, two
            # on the store warp -- are emitted only under the research build; see
            # the research tracing hooks.  They split a chunk into stall and
            # work for each role, which is what NCU cannot show.
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
                # aliased onto ``h_state``: aliasing an MMA operand onto
                # persistent state registers is what made engine's equivalent
                # probe spill.
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
        for c in range(num_chunks):
            in_stage = input_stage(c)
            cute.arch.mbarrier_wait(
                p_bar + BAR_IN_CONSUMED + in_stage, input_consumed_parity(c)
            )

            if lane == 0:
                gchunk = chunk_base + c
                token_base = token_start + c * BT
                factor_row = gchunk * BT
                stage16 = SMEM_INPUT // 2 + in_stage * (INPUT_STAGE_BYTES // 2)
                mbar = p_bar + BAR_IN_READY + in_stage
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, INPUT_STAGE_TX_BYTES)
                # Nine instructions, one completion barrier, 15,360 bytes.
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
                tma_load_3d(
                    p16 + (stage16 + STAGE_AQ // 2),
                    desc_aq,
                    mbar,
                    0,
                    gchunk,
                    head,
                )
                tma_load_3d(
                    p32
                    + (
                        (SMEM_INPUT + STAGE_GT) // 4
                        + in_stage * (INPUT_STAGE_BYTES // 4)
                    ),
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
                # The full path: the stage is released only after
                # the store has read it, not when it is merely committed.
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


@cute.jit
def _recurrence_entry(
    gout: cute.Tensor,
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
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    stream,
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    recurrence_kernel(
        gout,
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
        HAS_STATE_IN,
        HAS_STATE_OUT,
        STATE_FP32,
    ).launch(
        grid=(grid_x, grid_y, 1),
        block=(REC_THREADS, 1, 1),
        smem=SMEM_DYNAMIC_BYTES,
        min_blocks_per_mp=MIN_BLOCKS_PER_MP,
        stream=stream,
    )


#: Per-device compile cache. H, T, N and grid dimensions remain runtime values.
_RECURRENCE_DEVICE_CACHE: dict = {}


def clear_compile_cache() -> None:
    _RECURRENCE_DEVICE_CACHE.clear()


def launch_recurrence_device(
    *,
    out: torch.Tensor,
    cu_seqlens_i32: torch.Tensor,
    cu_chunks_i32: torch.Tensor,
    tensor_maps,
    heads: int,
    sequences: int,
    has_state_in: bool,
    has_state_out: bool,
    state_dtype: torch.dtype | None,
    stream=None,
) -> None:
    """Compile (or reuse) and launch the recurrence kernel."""
    import cuda.bindings.driver as cuda_driver

    if stream is None:
        # The tensors' device, not the current one; see fused/launch.py.
        stream = cuda_driver.CUstream(torch.cuda.current_stream(out.device).cuda_stream)

    state_fp32 = state_dtype is torch.float32
    grid = recurrence_grid(sequences, heads)
    args = (
        _flat(out),
        _flat(cu_seqlens_i32),
        _flat(cu_chunks_i32),
        cutlass.Int64(tensor_maps.address("factor")),
        cutlass.Int64(tensor_maps.address("aq")),
        cutlass.Int64(tensor_maps.address("gt")),
        cutlass.Int64(tensor_maps.address("v")),
        cutlass.Int64(tensor_maps.address("out")),
        cutlass.Int64(tensor_maps.address("state_in")),
        cutlass.Int64(tensor_maps.address("state_out")),
        cutlass.Int32(heads),
        cutlass.Int32(grid[0]),
        cutlass.Int32(grid[1]),
        stream,
    )
    key = (
        out.device.index,
        torch.cuda.get_device_capability(out.device),
        bool(has_state_in),
        bool(has_state_out),
        state_dtype,
    )
    compiled = _RECURRENCE_DEVICE_CACHE.get(key)
    if compiled is None:
        with torch.cuda.device(out.device):
            compiled = cute.compile(
                _recurrence_entry,
                *args,
                bool(has_state_in),
                bool(has_state_out),
                bool(state_fp32),
            )
        _RECURRENCE_DEVICE_CACHE[key] = compiled
    compiled(*args)


# --------------------------------------------------------------------------
# Section 10: one compiled host entry that launches both kernels
# --------------------------------------------------------------------------

# The DSL's AST preprocessor replays this module's module-level imports
# into the tracing scope when it compiles the jit below.  Both kernels and
# every helper they reach now live in this one file, so there is nothing
# left for it to resolve except the third-party imports at the top and the
# handful of names from ``runtime``.


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
    # --- recurrence output ---
    gout: cute.Tensor,
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
    prep_grid_x: cutlass.Int32,
    rec_grid_x: cutlass.Int32,
    rec_grid_y: cutlass.Int32,
    stream,
    # --- specializations ---
    SAFE_GATE: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
    HAS_STATE_IN: cutlass.Constexpr,
    HAS_STATE_OUT: cutlass.Constexpr,
    STATE_FP32: cutlass.Constexpr,
):
    """Both launches, in order, on one stream.

    Ordering is the caller's stream, not an event: recurrence reads the factors
    prepare writes, and same-stream launches are already ordered. An optional
    second stream measured slower because co-resident overlap needs a flag ring.
    """
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
        desc_q,
        desc_k,
        desc_g,
        desc_factor,
        scale,
        gate_scale_log2,
        total_chunks,
        heads,
        SAFE_GATE,
        CPC,
        G_FP32,
    ).launch(
        grid=(prep_grid_x, heads, 1),
        block=(PREPARE_DEVICE_THREADS, 1, 1),
        stream=stream,
    )

    recurrence_kernel(
        gout,
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
        HAS_STATE_IN,
        HAS_STATE_OUT,
        STATE_FP32,
    ).launch(
        grid=(rec_grid_x, rec_grid_y, 1),
        block=(REC_THREADS, 1, 1),
        smem=SMEM_DYNAMIC_BYTES,
        min_blocks_per_mp=MIN_BLOCKS_PER_MP,
        stream=stream,
    )


def entry_kernel_name(key: tuple) -> str:
    """The persistent cache's specialization name for a combined-entry key.

    Every compile-time parameter and nothing else, spelled so that a directory
    listing answers "which specializations did this run build?".
    """
    safe_gate, chunks_per_cta, g_fp32, state_in, state_out, state_fp32 = key
    return "decomp_" + "_".join(
        (
            "safegate" if safe_gate else "rawgate",
            f"cpc{int(chunks_per_cta)}",
            "gfp32" if g_fp32 else "gbf16",
            "si" if state_in else "nosi",
            "so" if state_out else "noso",
            "statefp32" if state_fp32 else "statebf16",
        )
    )


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
    then launches -- freezing it is precisely how the result cache broke under
    graph capture once already.
    """

    __slots__ = (
        "args",
        "compiled",
        "a_log",
        "a_log_exp",
        "gate_scale_log2",
        "launch_lock",
        "_a_log_key",
        "_keepalive",
    )

    def __init__(
        self,
        args,
        compiled,
        a_log,
        a_log_exp,
        gate_scale_log2,
        launch_lock,
        keepalive=(),
    ):
        self.args = args
        self.compiled = compiled
        self.a_log = a_log
        self.a_log_exp = a_log_exp
        self.gate_scale_log2 = gate_scale_log2
        self.launch_lock = launch_lock
        # This plan is pinned to one stream -- ``_fwd_identity`` keys on it, and
        # replaying a plan from another stream is already refused -- and to one
        # ``A_log``, which it holds a reference to.  Its ``_A_LOG_EXP`` key is
        # therefore a constant, and recomputing it per call cost 5.0-5.5 us of
        # driver and ``os.getenv`` traffic on shapes short enough to notice.
        # See :func:`a_log_exp_for` for the measurement.

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
            # Internal profiling callers may pass an already-refreshed buffer
            # without its source tensor.  They still need the workspace lock;
            # only the refresh is conditional.
            if self.a_log is not None:
                a_log_exp_for(
                    self.a_log, LOG2_E, out=self.a_log_exp, key=self._a_log_key
                )
            self.compiled(*self.args)


def launch_fwd(
    *,
    q,
    k,
    g,
    beta,
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
    run, so ``fwd`` can cache it and skip the host path on the next call with
    the same buffers.
    """
    # The tensors' device, not the current one; see fused/launch.py.
    stream = cuda_driver.CUstream(torch.cuda.current_stream(out.device).cuda_stream)

    args = (
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
        cutlass.Int32(prep_grid_x),
        cutlass.Int32(rec_grid_x),
        cutlass.Int32(rec_grid_y),
        stream,
    )
    key = (
        bool(safe_gate),
        int(chunks_per_cta),
        bool(g_fp32),
        bool(has_state_in),
        bool(has_state_out),
        bool(state_fp32),
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
        gate_scale_log2,
        workspace.launch_lock,
        keepalive=(prep_tmaps, rec_tmaps, workspace, a_log_exp),
    )
    if build_only:
        return call
    call.run()
    return None


# --------------------------------------------------------------------------
# Section 11: the host path
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
# 1. :func:`_fast_path` compares object identity against the previous call.
#    Same object means same address, shape, dtype, device and contiguity, so
#    only in-place mutation is left to check.
# 2. :data:`_CALL_PLANS` is an LRU on the full identity key, for callers that
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

#: One entry per distinct set of buffers a caller uses; a serving loop that
#: reuses its activations needs exactly one.  Bounded so a workload cycling
#: through many buffers cannot pin every plan it ever built.
#:
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
#: handed to the wrong buffers.
_LAST: tuple | None = None

#: Marks "this call has no chunks at all", so the zero-token path is reached on
#: a plan hit without re-deriving the metadata that proves it.
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
    clear_prepare_workspaces()
    clear_metadata_cache()
    clear_prepare_descriptor_cache()
    clear_recurrence_descriptor_cache()
    clear_launch_caches()
    clear_compile_cache()
    clear_fwd_cache()


def call_plan_stats() -> dict:
    return {"plans": len(_CALL_PLANS), "last_call_warm": _LAST is not None}


def _identity(device, tensors, scale, lower_bound, resources) -> tuple:
    # The stream of the *inputs'* device: ``launch_recurrence`` bakes
    # ``torch.cuda.current_stream(q.device)`` into the plan's argument tuple,
    # so a key built from the current device's stream would let two streams on
    # the input's device share one entry and reuse the first one's plan.
    return (
        tuple(tensor_identity(t) for t in tensors),
        float(scale),
        float(lower_bound),
        resource_cache_token(resources),
        current_stream_ptr(device),
    )


def _fast_path(device, tensors, scale, lower_bound, resources):
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
    if last_scalars != (float(scale), float(lower_bound)):
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


def _remember(device, tensors, scale, lower_bound, resources, plan) -> None:
    global _LAST
    # Weak, like the plan LRU above: this entry outlives the call, and strong
    # references to q, k, v, g and out would hold one whole activation set off
    # the caching allocator until the next call replaced it.
    _LAST = (
        tuple(None if t is None else weakref.ref(t) for t in tensors),
        tuple(None if t is None else tensor_version(t) for t in tensors),
        (float(scale), float(lower_bound)),
        resource_cache_token(resources),
        current_stream_ptr(device),
        plan,
    )


def _remember_plan(key, value) -> None:
    _CALL_PLANS[key] = value
    while len(_CALL_PLANS) > CALL_PLAN_MAX_ENTRIES:
        _CALL_PLANS.popitem(last=False)


def _state_only(initial_state, final_state) -> None:
    """No chunks, so nothing but the state ABI is left.

    Nothing here may allocate a workspace, encode a descriptor, upload metadata
    or consult the compile cache.  It is real work the caller expects on every
    call, so a plan hit redoes it rather than skipping it.
    """
    if final_state is None:
        return
    if initial_state is None:
        final_state.zero_()
        return
    if is_exact_alias(initial_state, final_state):
        return
    final_state.copy_(initial_state)


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
) -> None:
    """Launch the decomposed prefill, writing ``out`` and ``final_state``.

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

    plan = _fast_path(q.device, tensors, scale, lower_bound, resources)
    if plan is None:
        key = _identity(q.device, tensors, scale, lower_bound, resources)
        # Everything from here to the launch is the miss path, and it is taken
        # under one lock: two threads building concurrently share a factor
        # arena, a descriptor cache and a DSL compiler session, and none of the
        # three tolerates it.
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
                    info=info,
                    offsets=offsets,
                    resources=resources,
                )
                _remember_plan(key, plan)
            _remember(q.device, tensors, scale, lower_bound, resources, plan)

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
    info,
    offsets,
    resources,
):
    """The full host path: tables, arena, descriptors, argument marshalling."""
    device = q.device
    require_sm120a(device)

    if info.total_tokens == 0:
        return _STATE_ONLY

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
        return _STATE_ONLY

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
    recurrence_plan = launch_recurrence(
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
        plan_only=True,
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
    "CALL_PLAN_MAX_ENTRIES",
    "ChunkMetadata",
    "PrepareConfig",
    "PrepareWorkspace",
    "call_plan_stats",
    "chunk_tables",
    "clear_caches",
    "default_chunks_per_cta",
    "factor_arena",
    "recurrence_grid",
    "execute",
    "run",
]
