# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous epilogue for the fused fc1+fc2 swap-AB MegaMoE kernel.

Component boundaries use ``TensorWithContract`` to keep per-thread RMEM layout
semantics explicit at the handoff between transpose, SwiGLU, quantize, and fc2
store components.
"""

import dataclasses
from typing import Any, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .contract import (
    Contract,
    FunctionMapping,
    Space,
    TensorWithContract,
    assert_contract_equivalent,
)
from .fc1_fc2_fuse_sched import BlockPhase
from common.megamoe_constants import Nvfp4BlockSize
from src.token_comm import TokenSrcMetadata

Fc1GateUpInterleave = 16
EpilogueTokenTile = 64
Fc1EpilogueOutputTile = 64
WarpThreadCount = 32
EpiWarpCount = 4

# Done-counter publish batching (``epi_flag_batch``, passed from the runner):
# amortize the device-scope fence (``fence_acq_rel_gpu``) over this many
# publishing task tiles.  1 == per-tile baseline.  The per-tile
# ``task_tile_boundary_bar`` already orders every epilogue warp's stores at CTA
# scope, so the batched fence + relaxed reductions run on a single (tidx==0)
# thread after that barrier.  A batch is flushed on every fc1<->fc2 phase switch
# (see run loop) so it never straddles a phase boundary -- this keeps the fc1
# done-counter (which feeds back into the fc2 MMA) from deadlocking.


# =============================================================================
# Module-local helpers
# =============================================================================


@dsl_user_op
def _red_add_relaxed_sys_v2_bf16x2(
    addr,
    val0_packed_bf16x2,
    val1_packed_bf16x2,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.relaxed.sys.global.add.v2.bf16x2 [addr], {v0, v1};``.

    Used by the fc2 REDG path to atomic-add 4 bf16 cells.  Inline asm is
    used because cuTeDSL has no vector-form ``red.v2.bf16x2`` surface; the
    operands are packed bf16x2 bit patterns carried in 32-bit registers.
    """
    llvm.inline_asm(
        None,
        [
            addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            val0_packed_bf16x2.ir_value(loc=loc, ip=ip),
            val1_packed_bf16x2.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.sys.global.add.noftz.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.release.gpu.add.s32`` to a GMEM int32 location.

    Publishes fc1 task-tile completion after the caller has flushed the fc1
    output stores.  Single-thread helper; caller guards the thread predicate.
    """
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_relaxed_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue ``red.relaxed.gpu.add.s32`` to a GMEM int32 location.

    Relaxed-ordering variant of :func:`_red_add_release_gpu_s32`: drops the
    release memory-ordering semantics so the reduction carries no fence.  Use
    only where a separate fence already orders the prior stores.  Single-thread
    helper; caller guards the thread predicate.
    """
    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_relaxed_gpu_s32_addr(
    addr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """``red.relaxed.gpu.add.s32`` to a GMEM int32 given its raw int64 address.

    Same as :func:`_red_add_relaxed_gpu_s32` but takes the target address as an
    ``Int64`` value (e.g. ``(counter.iterator + off).toint()``) instead of a
    pointer, so a batch of pre-computed done-counter targets can be replayed
    from an RMEM scratch buffer under a single device fence.
    """
    llvm.inline_asm(
        None,
        [
            addr.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_bulk_s2g(
    dst_gmem,
    src_smem,
    size_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue non-tensor descriptor-free ``cp.async.bulk`` SMEM->GMEM.

    cuTeDSL does expose ``cpasync.CopyBulkS2GOp`` / ``cute.copy`` for this
    instruction family, but that abstraction bakes the transfer size into
    the copy atom / static tensor layout: CuteNvGPU lowers it as an
    ``arch.copy.SM90.bulk_copy_s2g`` op whose ``size`` is an ``I32Attr``.
    The fc2 UBLK epilogue needs a runtime byte count for the hidden-tail
    row (still 16B-aligned, but not necessarily the full 128-hidden row).
    Using the cute copy atom would silently encode the wrong semantic
    contract, so keep the raw PTX here until the dialect grows a dynamic-size
    descriptor-free bulk-copy op.

    This helper only issues the instruction.  The caller owns
    ``cp_async_bulk_commit_group`` so copy and reduce bulk paths share the
    same group boundary.
    """
    # with cute.arch.elect_one():
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_reduce_async_bulk_add_noftz_bf16_s2g(
    dst_gmem,
    src_smem,
    size_bytes,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue non-tensor ``cp.reduce.async.bulk`` for BF16 add.

    cuTeDSL currently exposes descriptor-free ``CopyBulkS2GOp`` but not the
    matching descriptor-free reduce atom.  Keep the fallback local to this
    epilogue path so the rest of the bulk pipeline can still share the same
    tensor/layout front-end.
    """
    # with cute.arch.elect_one():
    llvm.inline_asm(
        None,
        [
            dst_gmem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Region tag
# =============================================================================


class Region:
    """Codegen-time region tag for a 16x32 sub-region within a 32x32 tile."""

    Top = 0
    Bottom = 1


# =============================================================================
# TmemTranspose16x32
# =============================================================================


class _TmemTranspose16x32Core:
    """Contract-naive physical implementation of the 16x32 -> 32x16 TMEM
    in-place transpose.  Shared by:

      - ``TmemTranspose16x32``       : fc1 epi codomain naming
                                       (``intermediate_output_idx``);
                                       elements are fp32 (swiglu fold output).
      - ``TmemTranspose16x32Packed`` : fc2 epi codomain naming
                                       (``hidden_pair_idx``); elements are
                                       32-bit packed ``(bf16, bf16)`` pairs.

    The (lane_idx, elem_idx) physical distribution is identical for both
    subclasses -- the underlying tcgen05 atoms are 32-bit element atoms,
    agnostic to whether each 32-bit slot holds an fp32 or a packed bf16x2.
    Only the codomain semantic names differ, expressed via the subclass's
    ``InputContract`` / ``OutputContract`` class attributes.

    Per-thread RMEM coordinate convention (used by both subclasses' contracts):

      - ``lane_idx`` -- warp lane id (= thread index within warp), in [0, 32).
      - ``elem_idx`` -- per-thread reg index, in [0, 16).

    Subclasses MUST override these two class attributes:
      ``InputContract``  -- (lane_idx, elem_idx) -> codomain mapping after
                            R1.Load (or after ``reg_tensor`` is fed in for
                            skip-R1.Load mode).
      ``OutputContract`` -- (lane_idx, elem_idx) -> codomain mapping after
                            ``r4_perm`` has run all four rounds.

    The Core's ``__init__`` reads ``self.InputContract`` / ``self.OutputContract``
    via Python's normal MRO attribute lookup; the subclass's overrides take
    precedence at construction time.
    """

    # Subclasses MUST override these.
    InputContract: Contract
    OutputContract: Contract

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(
        rmem: cute.Tensor, num_regs: int, offset: int = 0
    ) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset,
            cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),)),
        )

    @staticmethod
    def load_subtile_raw_acc(
        tmem_subtile_tensor: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """LDTM the entire 32-lane x 64-col raw acc region of one epi
        subtile into 4 independent (16,) fp32 RMEM tensors.

        Used by the overlap-acc unroll path in
        ``_run_fc{1,2}_task_tile`` to extract all raw acc data of the
        first 2 subtiles up front, so that the acc TMEM can be released
        to the next mma right after the first subtile's 4 LDTMs (instead
        of waiting for a full subtile body to complete).

        ``tmem_subtile_tensor`` is the (32 lanes, 64 cols) view onto a
        single epi subtile's acc TMEM region (already offset by
        ``warp_lane_offset + acc_stage_col_offset + subtile_col_offset``;
        see ``SwapABSwigluFp4Epilogue._subtile_local_tmem_tensor``).

        Returns a 4-tuple of (16,) fp32 RMEM tensors, each carrying
        the (lane_idx, elem_idx) -> codomain distribution described by
        ``TmemTranspose16x32.InputContract`` /
        ``TmemTranspose16x32Packed.InputContract`` (physically identical
        for fc1 and fc2, only codomain semantic names differ):

          [0] gate_lo / first-half top   -- subtile cols 0..31, lanes 0..15
          [1] up_lo   / first-half bot   -- subtile cols 0..31, lanes 16..31
          [2] raw_top / second-half top  -- subtile cols 32..63, lanes 0..15
          [3] raw_bot / second-half bot  -- subtile cols 32..63, lanes 16..31

        4 atom calls of ``Ld16x64bOp(Repetition.x16) Float32`` -- the
        same atom currently used by the per-subtile entry LDTM in
        ``_run_fc1_subtile`` and by ``second_t.r1_load`` /
        ``Fc2AccLoadAndPack`` per-half LDTMs.  Caller is expected to
        wrap each output in ``TensorWithContract`` with
        ``TmemTranspose16x32{,Packed}.InputContract`` before handing
        them downstream.
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            _TmemTranspose16x32Core._io_dtype,
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride

        # 4 source 16-lane x 32-col views over the (32, 64) subtile region:
        #   first  half (cols 0..31): top  lanes 0..15  / bot lanes 16..31
        #   second half (cols 32..63): top lanes 0..15  / bot lanes 16..31
        # All offsets are Python ints (compile-time const) so cute can
        # const-fold them and infer the correct (>= 8 B / 2 col) ptr
        # alignment that the LDTM atom requires.  Using ``cutlass.Int32``
        # offsets here would wrap them as SSA values that cute treats as
        # alignment-unknown, tripping the atom's verifier.
        first_top_view = cute.make_tensor(
            ptr,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_top_view = cute.make_tensor(
            ptr + 32,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off,
            _TmemTranspose16x32Core._tmem_layout(16, 32),
        )

        first_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        first_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)

        cute.copy(
            atom_ld16x64,
            first_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            first_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(first_bot, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_top_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_bot_view,
            _TmemTranspose16x32Core._rmem_copy_view(second_bot, 16),
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(
        self,
        tmem_ptr,
        region: int,
        reg_tensor: Optional[TensorWithContract] = None,
    ) -> None:
        half_lane_off = 16 * self._TmemRowStride
        if region == Region.Top:
            src_ptr = tmem_ptr
            dst_ptr = tmem_ptr
        elif region == Region.Bottom:
            src_ptr = tmem_ptr + half_lane_off
            dst_ptr = tmem_ptr + 16
        else:
            raise ValueError("region must be Region.Top or Region.Bottom")

        self.region = region

        self._tmem_src_full = cute.make_tensor(src_ptr, self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr, self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr, self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(
            dst_ptr + half_lane_off, self._tmem_layout(16, 16)
        )

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8),
            self._io_dtype,
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16),
            self._io_dtype,
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
            self._io_dtype,
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4),
            self._io_dtype,
        )

        self._src_regs = cute.make_rmem_tensor((16,), self._io_dtype)
        output_tensor = cute.make_rmem_tensor((16,), self._io_dtype)
        self.output = TensorWithContract(
            tensor=output_tensor,
            contract=self.OutputContract,
        )

        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            assert_contract_equivalent(
                reg_tensor.contract,
                self.InputContract,
                context=f"{type(self).__name__} skip-R1.Load reg_tensor",
            )
            for r in range(16):
                self._src_regs[r] = reg_tensor.tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r1_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(
            self._atom_st16x128,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_src_full,
        )

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(
            self._atom_ld16x64,
            self._tmem_src_full,
            self._rmem_copy_view(self._src_regs, 16),
        )

    def r2_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self._src_regs, 16),
            self._tmem_dst_full,
        )

    # -- R3 ------------------------------------------------------------------

    def r3_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r3_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r3_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_dst_full,
        )

    # -- R4 ------------------------------------------------------------------

    def r4_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r4_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r4_perm(self) -> None:
        for r in range(16):
            self.output.tensor[r] = self._src_regs[self._PermR4[r]]

    def r4_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output.tensor, 16),
            self._tmem_dst_full,
        )


class TmemTranspose16x32(_TmemTranspose16x32Core):
    """fc1 epi 16x32 -> 32x16 TMEM in-place transpose.

    Contract summary:
      - input : ``token_idx = elem_idx * 2 + ((lane_idx // 2) % 2)``
      - output: ``token_idx = lane_idx``
    The second codomain axis is ``intermediate_output_idx``.
    """

    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "intermediate_output_idx"), (32, 16))

    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "intermediate_output_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }
        ),
    )
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": lane_idx,
                "intermediate_output_idx": elem_idx,
            }
        ),
    )


class TmemTranspose16x32Packed(_TmemTranspose16x32Core):
    """fc2 epi 16x32 -> 32x16 TMEM in-place transpose, 32-bit packed
    bf16x2 elements.

    Same physical atom sequence as ``TmemTranspose16x32``; codomain is
    ``(token_idx, hidden_pair_idx)`` and each slot holds one packed bf16x2.
    """

    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))

    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "hidden_pair_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }
        ),
    )
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": lane_idx,
                "hidden_pair_idx": elem_idx,
            }
        ),
    )


# =============================================================================
# TmemTranspose32x32Inplace
# =============================================================================


class TmemTranspose32x32Inplace:
    """fc1 epi 32x32 in-place TMEM transpose: two ``TmemTranspose16x32``
    sub-instances (``top`` = lanes 0..15, ``bot`` = lanes 16..31).

    Optional ``reg_tensor_top`` / ``reg_tensor_bot`` enable skip-R1.Load mode
    for both halves; they must be provided or omitted together.
    """

    def __init__(
        self,
        tmem_ptr,
        reg_tensor_top: Optional[TensorWithContract] = None,
        reg_tensor_bot: Optional[TensorWithContract] = None,
    ) -> None:
        if (reg_tensor_top is None) != (reg_tensor_bot is None):
            raise ValueError(
                "TmemTranspose32x32Inplace: reg_tensor_top and reg_tensor_bot "
                "must be provided or omitted together (both halves either "
                "skip-R1.Load or do R1.Load)."
            )
        self.top = TmemTranspose16x32(tmem_ptr, Region.Top, reg_tensor=reg_tensor_top)
        self.bot = TmemTranspose16x32(
            tmem_ptr, Region.Bottom, reg_tensor=reg_tensor_bot
        )


# =============================================================================
# SwigluCompute
# =============================================================================


class SwigluCompute:
    """Element-wise SwiGLU fold over a configurable reg range
    (packed_f32x2 path).

    SwigluCompute does NOT have a fixed ``InputContract``: the caller
    determines the input distribution by what it hands in.  At
    construction time we validate that ``gate.contract`` and ``up.contract``
    are equal -- the fold is element-wise, so they must share the same
    physical (lane_idx, elem_idx) -> (logical) distribution; only the
    semantic label differs (gate slice vs up slice).

    ``self.output`` inherits the input contract: the fold is element-wise,
    so the output has the same (lane_idx, elem_idx) -> physical mapping
    as the input.  Only the codomain semantic label changes (intermediate
    input slot -> intermediate output slot); since both labels are
    logically the same axis, the contract object is reused as-is.

    ``fold(start, end)`` writes ``self.output.tensor[start:end]`` only.
    The caller may invoke ``fold`` with disjoint ranges to disperse SwiGLU's
    MUFU traffic across surrounding transpose STTM boundaries.
    """

    _Log2E = 1.4426950408889634

    def __init__(
        self,
        gate: TensorWithContract,
        up: TensorWithContract,
        alpha,
    ) -> None:
        assert_contract_equivalent(
            gate.contract,
            up.contract,
            context="SwigluCompute gate/up contract",
        )

        self._gate = gate.tensor
        self._up = up.tensor
        self._alpha = alpha

        output_tensor = cute.make_rmem_tensor((16,), cutlass.Float32)
        self.output = TensorWithContract(
            tensor=output_tensor,
            contract=gate.contract,
        )

    def fold(self, start: int = 0, end: int = 16) -> None:
        """Fold pairs ``(i, i+1)`` for ``i in range(start, end, 2)``::

            out[i, i+1] = (alpha^2) * up[i, i+1] * gate[i, i+1] *
                          sigmoid(alpha * gate[i, i+1])
            sigmoid(x)  = rcp(1 + exp2(-x * log2(e)))

        Reassociated to put the FMUL2 first so the inner mul collapses to
        one FMUL2 per pair.  ``mul``/``add`` go through packed_f32x2;
        ``exp2``/``rcp_approx`` run scalar but on adjacent pairs ptxas
        back-to-backs them on the MUFU pipe.

        ``start`` / ``end`` must be pair-aligned.
        """
        alpha_f32 = cutlass.Float32(self._alpha)
        neg_alpha_log2e = alpha_f32 * cutlass.Float32(-self._Log2E)
        neg_alpha_log2e_pair = (neg_alpha_log2e, neg_alpha_log2e)
        alpha_sq = alpha_f32 * alpha_f32
        alpha_sq_pair = (alpha_sq, alpha_sq)
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))

        out = self.output.tensor
        for i in range(start, end, 2):
            ug = cute.arch.mul_packed_f32x2(
                (self._up[i], self._up[i + 1]),
                (self._gate[i], self._gate[i + 1]),
            )

            neg_g_log2e = cute.arch.mul_packed_f32x2(
                (self._gate[i], self._gate[i + 1]), neg_alpha_log2e_pair
            )
            exp_pair = (
                cute.math.exp2(neg_g_log2e[0], fastmath=True),
                cute.math.exp2(neg_g_log2e[1], fastmath=True),
            )
            one_plus_exp = cute.arch.add_packed_f32x2(exp_pair, one_pair)
            sigmoid_pair = (
                cute.arch.rcp_approx(one_plus_exp[0]),
                cute.arch.rcp_approx(one_plus_exp[1]),
            )

            ug_sig = cute.arch.mul_packed_f32x2(ug, sigmoid_pair)
            out_pair = cute.arch.mul_packed_f32x2(ug_sig, alpha_sq_pair)

            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]


# =============================================================================
# PostSwigluHalf
# =============================================================================


class PostSwigluHalf:
    """Per-half SwiGLU finalize: topk-weight broadcast mul + gen_sf + quantize
    at construction, then ``stg_sfc`` / ``r2s`` as later atomic actions.

    The post-transpose contract gives each thread one token's 16 values, so
    topk broadcast is one scalar multiply across local regs.  The kernel applies
    topk weights before NVFP4 quantization so the fc2 mainloop reads already
    weighted fc1 output.
    """

    # InputContract: explicit definition (NOT an alias of
    # ``TmemTranspose16x32.OutputContract``).  Two distinct Contract objects
    # carrying the same mapping function let the construct-time
    # ``assert_contract_equivalent(swiglu.contract, self.InputContract)``
    # check actually exercise the equivalence comparison: if anyone later
    # mutates the upstream OutputContract without updating this one, the
    # mismatch is caught at codegen time.  Aliasing would silently
    # short-circuit the validation.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "intermediate_output_idx"), (32, 16))
    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": lane_idx,
                "intermediate_output_idx": elem_idx,
            }
        ),
    )

    _Nvfp4RcpLimit = 1.0 / 6.0  # 1 / max abs of Float4E2M1FN (= 6.0)
    _Fp32Max = 3.40282346638528859812e38

    def __init__(
        self,
        swiglu: TensorWithContract,
        *,
        sC: cute.Tensor,
        gSFC: cute.Tensor,
        preloaded_topk_weight: cutlass.Float32,
        warp_idx,
        norm_const,
        sf_vec_size: int,
        half_idx: int,
        token_idx,
        thread_in_warp,
        intermediate_downproj_idx,
        intermediate_downproj,
        cga_cluster_tile_intermediate_downproj: int,
    ) -> None:
        assert_contract_equivalent(
            swiglu.contract,
            self.InputContract,
            context="PostSwigluHalf swiglu input",
        )

        self._sC = sC
        self._gSFC = gSFC
        self._warp_idx = warp_idx
        self._sf_vec_size = sf_vec_size

        self._half_idx = half_idx
        self._token_idx = token_idx
        self._thread_in_warp = thread_in_warp
        self._intermediate_downproj_idx = intermediate_downproj_idx
        self._intermediate_downproj = intermediate_downproj
        self._cga_cluster_tile_intermediate_downproj = (
            cga_cluster_tile_intermediate_downproj
        )

        self._sfc_reg, self._scaled_regs = self._gen_sfc_quantize(
            swiglu.tensor, norm_const, preloaded_topk_weight
        )

    def _gen_sfc_quantize(self, swiglu_rmem: cute.Tensor, norm_const, topk_weight):
        """Compute SFC + pre-quantized scaled fp32 regs in RMEM."""
        sfc_reg = cute.make_rmem_tensor((1,), cutlass.Float8E4M3FN)
        weighted_regs = cute.make_rmem_tensor((16,), cutlass.Float32)
        scaled_regs = cute.make_rmem_tensor((16,), cutlass.Float32)

        # Path A: multiply topk before NVFP4 quantize.
        topk_pair = (topk_weight, topk_weight)
        for i in range(0, 16, 2):
            w0, w1 = cute.arch.mul_packed_f32x2(
                (cutlass.Float32(swiglu_rmem[i]), cutlass.Float32(swiglu_rmem[i + 1])),
                topk_pair,
            )
            weighted_regs[i] = w0
            weighted_regs[i + 1] = w1

        # Step 1: absmax over the weighted regs.
        absmax = cutlass.Float32(0.0)
        for i in range(16):
            v = weighted_regs[i]
            abs_v = cute.arch.fmax(v, -v)
            absmax = cute.arch.fmax(absmax, abs_v)

        sfc_fp32 = absmax * cutlass.Float32(self._Nvfp4RcpLimit) * norm_const
        sfc_reg[0] = sfc_fp32.to(cutlass.Float8E4M3FN)
        sfc_fp32_rt = cutlass.Float32(sfc_reg[0])

        acc_scale = norm_const * cute.arch.rcp_approx(sfc_fp32_rt)
        acc_scale = cute.arch.fmin(acc_scale, cutlass.Float32(self._Fp32Max))
        mask = cute.arch.fmin(sfc_fp32_rt * cutlass.Float32(1e30), cutlass.Float32(1.0))
        acc_scale = acc_scale * mask

        acc_scale_pair = (acc_scale, acc_scale)
        for i in range(0, 16, 2):
            s0, s1 = cute.arch.mul_packed_f32x2(
                (weighted_regs[i], weighted_regs[i + 1]),
                acc_scale_pair,
            )
            scaled_regs[i] = s0
            scaled_regs[i + 1] = s1

        return sfc_reg, scaled_regs

    @cute.jit
    def stg_sfc(self) -> None:
        """RMEM -> GMEM: 1 fp8 SFC byte for this token/SF block.

        Skip when the warp's intermediate_downproj position is past the
        valid bound; corresponding fp4 is TMA-OOB-fill-0 on fc2 data leg.

        ``self._intermediate_downproj`` is one of three flavors:

          * Python int that is a multiple of
            ``_cga_cluster_tile_intermediate_downproj``: the predicate is
            statically True; const_expr collapses the entire branch into
            an unconditional STG (no runtime cmp / branch).

          * Python int that is NOT statically aligned (static_expert_shape
            path with e.g. ``intermediate_downproj == 96`` and
            ``cga_cluster_tile_intermediate_downproj == 64``): the
            second branch runs a runtime predicate against a const SSA
            (the Python int folds to an immediate cmp).

          * Int32 SSA (static_expert_shape is None, i.e. dynamic-shape
            mode): the second branch runs a full runtime cmp.  The
            ``isinstance`` short-circuit prevents the first branch from
            trying to ``%`` an SSA value at trace time.
        """
        if cutlass.const_expr(
            isinstance(self._intermediate_downproj, int)
            and self._intermediate_downproj
            % self._cga_cluster_tile_intermediate_downproj
            == 0
        ):
            self._gSFC[self._token_idx, self._intermediate_downproj_idx, 0] = (
                self._sfc_reg[0]
            )
            return

        # Runtime predicate.  Works for both the unaligned-static and the
        # dynamic-shape paths; subtile size assumption (1 fp8 / warp /
        # subtile) is unchanged from the previous implementation.
        if self._intermediate_downproj_idx < self._intermediate_downproj:
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=8
                ),
                self._sfc_reg,
                cute.composition(
                    self._gSFC[self._token_idx, self._intermediate_downproj_idx, None],
                    self._sfc_reg.shape,
                ),
            )

    def r2s(self, subtile_idx) -> None:
        """RMEM -> SMEM: per-thread STS.64 of 16 fp4 to ``sC[subtile_idx]``."""
        fp4_regs = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
        fp4_vec = self._scaled_regs.load().to(cutlass.Float4E2M1FN)
        fp4_regs.store(fp4_vec)

        sC_stage = cute.slice_(self._sC, (None, None, subtile_idx))
        token_coord = self._thread_in_warp + 32 * self._half_idx
        sC_thread_row = cute.local_tile(
            sC_stage,
            (1, 16),
            (token_coord, self._warp_idx),
        )

        copy_atom_64b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float4E2M1FN,
            num_bits_per_copy=64,
        )
        cute.copy(
            copy_atom_64b,
            cute.coalesce(fp4_regs),
            cute.coalesce(sC_thread_row),
        )


# =============================================================================
# Fc2AccLoadAndPack
# =============================================================================


class Fc2AccLoadAndPack:
    """fc2 epi: LDTM x 2 + cvt.rn.bf16x2.f32 fuse + pair packing.

    Output is a ``TensorWithContract`` matching
    ``TmemTranspose16x32Packed.InputContract``; each 32-bit slot stores one
    bf16x2 pair ``(hidden_i, hidden_i + 16)``.
    """

    # OutputContract: explicit definition (NOT an alias of
    # ``TmemTranspose16x32Packed.InputContract``).  Two distinct Contract
    # objects carrying the same mapping function let the construct-time
    # equivalence check inside ``TmemTranspose16x32Packed.__init__`` (when
    # given ``reg_tensor=self.output``) actually exercise the check: if
    # either side's mapping drifts, the codegen-time assertion catches it.
    # Aliasing would silently short-circuit the validation.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))
    OutputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": elem_idx * 2 + ((lane_idx // 2) % 2),
                "hidden_pair_idx": (lane_idx % 2) * 8 + lane_idx // 4,
            }
        ),
    )

    _io_dtype = cutlass.Float32  # 32-bit slot dtype for downstream atoms
    _TmemRowStride = _TmemTranspose16x32Core._TmemRowStride

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return _TmemTranspose16x32Core._tmem_layout(num_lanes, num_cols)

    @staticmethod
    def _rmem_copy_view(
        rmem: cute.Tensor, num_regs: int, offset: int = 0
    ) -> cute.Tensor:
        return _TmemTranspose16x32Core._rmem_copy_view(rmem, num_regs, offset)

    def __init__(
        self,
        tmem_ptr=None,
        *,
        preload_acc: Optional[Tuple[cute.Tensor, cute.Tensor]] = None,
        fc2_output_dtype: Type[cutlass.Numeric],
    ) -> None:
        """Create the packed-bf16x2 RMEM view from TMEM or preloaded acc."""
        if (tmem_ptr is None) == (preload_acc is None):
            raise ValueError(
                "Fc2AccLoadAndPack: exactly one of tmem_ptr / preload_acc "
                "must be provided (LDTM mode vs skip-LDTM preload mode)."
            )

        # Gather top/bottom hidden halves into one 32-reg fp32 vector.
        acc_full = cute.make_rmem_tensor((32,), self._io_dtype)
        if preload_acc is None:
            # Two LDTMs cover top and bottom hidden halves.
            atom_ld16x64 = cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
                self._io_dtype,
            )
            tmem_top_view = cute.make_tensor(
                tmem_ptr,
                self._tmem_layout(16, 32),
            )
            tmem_bot_view = cute.make_tensor(
                tmem_ptr + 16 * self._TmemRowStride,
                self._tmem_layout(16, 32),
            )
            cute.copy(
                atom_ld16x64,
                tmem_top_view,
                self._rmem_copy_view(acc_full, 16, offset=0),
            )
            cute.copy(
                atom_ld16x64,
                tmem_bot_view,
                self._rmem_copy_view(acc_full, 16, offset=16),
            )
        else:
            top_reg, bot_reg = preload_acc
            for i in range(16):
                acc_full[i] = top_reg[i]
                acc_full[i + 16] = bot_reg[i]

        # Interleave so bf16x2 pairs become (hidden_i, hidden_i+16).
        reordered_fp32 = cute.make_rmem_tensor((32,), self._io_dtype)
        for i in range(16):
            reordered_fp32[2 * i] = acc_full[i]
            reordered_fp32[2 * i + 1] = acc_full[i + 16]

        # Bulk cast lets NVVM form cvt.rn.bf16x2.f32 for adjacent pairs.
        packed_bf16 = cute.make_rmem_tensor((32,), fc2_output_dtype)
        packed_bf16.store(reordered_fp32.load().to(fc2_output_dtype))

        # Recast 32 bf16 elements as 16 32-bit slots for downstream atoms.
        packed_fp32 = cute.recast_tensor(packed_bf16, self._io_dtype)

        # Contract matches what the packed transpose expects as input.
        self.output = TensorWithContract(
            tensor=packed_fp32,
            contract=self.OutputContract,
        )


# =============================================================================
# Fc2 return tile -- full CTA-token-tile return routing
# =============================================================================


class Fc2ReturnTokenContract:
    """Physical token-issue contract for one full fc2 CTA token tile.

    Subclasses answer only the local token coordinate:

        (epi_tidx, token_iter) -> token_idx_within_cta_token_tile

    ``pool_token_global`` is deliberately outside the contract; it is derived
    by ``Fc2ReturnTile`` as ``cta_token_tile_start + token_idx``.  This keeps
    the contract reusable across task tiles and prevents return routing from
    depending on per-tile row offsets.
    """

    num_token_iters: int
    contract: Contract

    @cute.jit
    def token_idx_within_cta_tile(self, epi_tidx, token_iter):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class TransposeStgReturnTokenContract(Fc2ReturnTokenContract):
    """STG.256 token issue: each epi thread owns one token per 32-row chunk."""

    cta_token_count: int

    def __post_init__(self) -> None:
        if self.cta_token_count % 32 != 0:
            raise ValueError(
                "TransposeStgReturnTokenContract requires cta_token_count "
                f"divisible by 32, got {self.cta_token_count}."
            )
        domain = Space(
            ("epi_tidx", "token_iter"),
            (EpiWarpCount * WarpThreadCount, self.cta_token_count // 32),
        )
        codomain = Space(("token_idx_within_cta_token_tile",), (self.cta_token_count,))
        contract = Contract(
            domain=domain,
            codomain=codomain,
            mapping=FunctionMapping(
                lambda epi_tidx, token_iter: {
                    "token_idx_within_cta_token_tile": token_iter * 32
                    + (epi_tidx % 32),
                }
            ),
        )
        object.__setattr__(self, "num_token_iters", self.cta_token_count // 32)
        object.__setattr__(self, "contract", contract)

    @cute.jit
    def token_idx_within_cta_tile(self, epi_tidx, token_iter):
        lane_idx = epi_tidx % cutlass.Int32(32)
        return token_iter * cutlass.Int32(32) + lane_idx


@dataclasses.dataclass(frozen=True)
class TransposeRedgReturnTokenContract(Fc2ReturnTokenContract):
    """REDG token issue for transpose epilogue.

    For each 64-token subtile, one lane issues 8 token metadata lookups:
    4 for the first half and 4 for the second half.  The hidden REDG pair
    loop is intentionally not part of this token contract.
    """

    cta_token_count: int

    def __post_init__(self) -> None:
        if self.cta_token_count % EpilogueTokenTile != 0:
            raise ValueError(
                "TransposeRedgReturnTokenContract requires cta_token_count "
                f"divisible by {EpilogueTokenTile}, got {self.cta_token_count}."
            )
        num_iters = (self.cta_token_count // EpilogueTokenTile) * 8
        domain = Space(
            ("epi_tidx", "token_iter"),
            (EpiWarpCount * WarpThreadCount, num_iters),
        )
        codomain = Space(("token_idx_within_cta_token_tile",), (self.cta_token_count,))
        contract = Contract(
            domain=domain,
            codomain=codomain,
            mapping=FunctionMapping(
                lambda epi_tidx, token_iter: {
                    "token_idx_within_cta_token_tile": (token_iter // 8)
                    * EpilogueTokenTile
                    + ((token_iter % 8) // 4) * 32
                    + ((epi_tidx % 32) // 4)
                    + ((token_iter % 4) * 8),
                }
            ),
        )
        object.__setattr__(self, "num_token_iters", num_iters)
        object.__setattr__(self, "contract", contract)

    @cute.jit
    def token_idx_within_cta_tile(self, epi_tidx, token_iter):
        lane_idx = epi_tidx % cutlass.Int32(32)
        subtile_idx = token_iter // cutlass.Int32(8)
        iter_in_subtile = token_iter % cutlass.Int32(8)
        return (
            subtile_idx * cutlass.Int32(EpilogueTokenTile)
            + (iter_in_subtile // cutlass.Int32(4)) * cutlass.Int32(32)
            + lane_idx // cutlass.Int32(4)
            + (iter_in_subtile % cutlass.Int32(4)) * cutlass.Int32(8)
        )


@dataclasses.dataclass(frozen=True)
class BulkReturnTokenContract(Fc2ReturnTokenContract):
    """UBLK token issue: each epi thread owns one row per 128-token group."""

    cta_token_count: int

    def __post_init__(self) -> None:
        if self.cta_token_count % 128 != 0:
            raise ValueError(
                "BulkReturnTokenContract requires cta_token_count divisible "
                f"by 128, got {self.cta_token_count}."
            )
        num_iters = self.cta_token_count // 128
        domain = Space(
            ("epi_tidx", "token_iter"),
            (EpiWarpCount * WarpThreadCount, num_iters),
        )
        codomain = Space(("token_idx_within_cta_token_tile",), (self.cta_token_count,))
        contract = Contract(
            domain=domain,
            codomain=codomain,
            mapping=FunctionMapping(
                lambda epi_tidx, token_iter: {
                    "token_idx_within_cta_token_tile": token_iter * 128
                    + ((epi_tidx % 32) // 8) * 32
                    + (epi_tidx // 32) * 8
                    + (epi_tidx % 8),
                }
            ),
        )
        object.__setattr__(self, "num_token_iters", num_iters)
        object.__setattr__(self, "contract", contract)

    @cute.jit
    def token_idx_within_cta_tile(self, epi_tidx, token_iter):
        warp_idx = epi_tidx // cutlass.Int32(32)
        lane_idx = epi_tidx % cutlass.Int32(32)
        return (
            token_iter * cutlass.Int32(128)
            + (lane_idx // cutlass.Int32(8)) * cutlass.Int32(32)
            + warp_idx * cutlass.Int32(8)
            + lane_idx % cutlass.Int32(8)
        )


@dataclasses.dataclass(frozen=True)
class Fc2ReturnTile:
    """Full CTA-token-tile return-side view for fc2 output.

    The tile owns push-based return routing for both lean direct mode and
    MegaMoE metadata-driven mode.  Store paths provide only a token contract;
    this object decides whether metadata is prefetched for the full CTA tile
    or loaded just in time by ``resolve``.
    """

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    cta_token_tile_start: Any = None
    valid_token_row_end: Any = None
    reduce_topk_in_kernel: bool = False
    token_contract: Fc2ReturnTokenContract = None
    prefetch: bool = True

    def __post_init__(self) -> None:
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2ReturnTile: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (lean / direct mode) or both non-None (MegaMoE / "
                "indirect mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                f"peer_rank_ptr_mapper={'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )
        if self.reduce_topk_in_kernel and self.metadata is None:
            raise ValueError(
                "Fc2ReturnTile: reduce_topk_in_kernel=True is only valid in "
                "indirect mode (metadata + peer_rank_ptr_mapper both set).  The "
                "lean / direct path has no topk axis to reduce."
            )
        if self.token_contract is None:
            raise ValueError("Fc2ReturnTile requires a token_contract.")
        if self.cta_token_tile_start is None:
            raise ValueError("Fc2ReturnTile requires cta_token_tile_start.")
        if self.valid_token_row_end is None:
            raise ValueError("Fc2ReturnTile requires valid_token_row_end.")

    @cute.jit
    def prefetch_for_epi_thread(self, epi_tidx):
        if cutlass.const_expr(not self.prefetch):
            return None

        token_count: cutlass.Constexpr[int] = self.token_contract.num_token_iters
        pool_tokens = cute.make_rmem_tensor((token_count,), cutlass.Int32)
        dst_ranks = cute.make_rmem_tensor((token_count,), cutlass.Int32)
        dst_tokens = cute.make_rmem_tensor((token_count,), cutlass.Int32)
        topks = cute.make_rmem_tensor((token_count,), cutlass.Int32)
        valid = cute.make_rmem_tensor((token_count,), cutlass.Int32)

        for token_iter in cutlass.range_constexpr(token_count):
            token_offset = self.token_contract.token_idx_within_cta_tile(
                epi_tidx,
                cutlass.Int32(token_iter),
            )
            pool_token = self.cta_token_tile_start + token_offset
            pool_tokens[token_iter] = pool_token
            valid[token_iter] = cutlass.Int32(0)
            dst_ranks[token_iter] = cutlass.Int32(0)
            if cutlass.const_expr(self.metadata is None):
                dst_tokens[token_iter] = pool_token
            else:
                dst_tokens[token_iter] = cutlass.Int32(0)
            topks[token_iter] = cutlass.Int32(0)

            if pool_token < self.valid_token_row_end:
                valid[token_iter] = cutlass.Int32(1)
                if cutlass.const_expr(self.metadata is not None):
                    md = TokenSrcMetadata.load(
                        self.metadata.iterator.toint()
                        + cutlass.Int64(pool_token)
                        * cutlass.Int64(TokenSrcMetadata.nbytes)
                    )
                    dst_ranks[token_iter] = md.src_rank
                    dst_tokens[token_iter] = md.src_token
                    if cutlass.const_expr(not self.reduce_topk_in_kernel):
                        topks[token_iter] = md.src_topk

        return Fc2ReturnTilePrefetch(
            tile=self,
            pool_tokens=pool_tokens,
            dst_ranks=dst_ranks,
            dst_tokens=dst_tokens,
            topks=topks,
            valid=valid,
        )

    @cute.jit
    def resolve(self, epi_tidx, token_iter, prefetch):
        if cutlass.const_expr(prefetch is not None):
            return prefetch.row(token_iter), prefetch.is_valid(token_iter)

        token_offset = self.token_contract.token_idx_within_cta_tile(
            epi_tidx,
            token_iter,
        )
        pool_token = self.cta_token_tile_start + token_offset
        valid = pool_token < self.valid_token_row_end
        if cutlass.const_expr(self.metadata is None):
            return cute.slice_(self.tensor, (pool_token, 0, None)), valid

        dst_rank = cutlass.Int32(0)
        dst_token = cutlass.Int32(0)
        topk = cutlass.Int32(0)
        if valid:
            md = TokenSrcMetadata.load(
                self.metadata.iterator.toint()
                + cutlass.Int64(pool_token) * cutlass.Int64(TokenSrcMetadata.nbytes)
            )
            dst_rank = md.src_rank
            dst_token = md.src_token
            if cutlass.const_expr(not self.reduce_topk_in_kernel):
                topk = md.src_topk
        local_row = cute.slice_(self.tensor, (dst_token, topk, None))
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator,
            dst_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout), valid


@dataclasses.dataclass(frozen=True)
class Fc2ReturnTilePrefetch:
    """Per-epi-thread full-tile materialized return metadata."""

    tile: Fc2ReturnTile
    pool_tokens: cute.Tensor
    dst_ranks: cute.Tensor
    dst_tokens: cute.Tensor
    topks: cute.Tensor
    valid: cute.Tensor

    @cute.jit
    def is_valid(self, token_iter):
        return self.valid[token_iter] != cutlass.Int32(0)

    @cute.jit
    def row(self, token_iter) -> cute.Tensor:
        if cutlass.const_expr(self.tile.metadata is None):
            return cute.slice_(
                self.tile.tensor,
                (self.pool_tokens[token_iter], 0, None),
            )

        local_row = cute.slice_(
            self.tile.tensor,
            (self.dst_tokens[token_iter], self.topks[token_iter], None),
        )
        peer_iter = self.tile.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator,
            self.dst_ranks[token_iter],
        )
        return cute.make_tensor(peer_iter, local_row.layout)


# =============================================================================
# Fc2UnpackPermuteStg
# =============================================================================


class Fc2UnpackPermuteStg:
    """fc2 epi RMEM -> GMEM dispatcher: STG.256 (default) or topk-collapsing
    REDG (form B), const_expr-switched on the full-tile return view.

    Input contract maps ``lane_idx`` to token and ``elem_idx`` to hidden
    pair (one packed bf16x2 = 2 bf16 per 32-bit slot).  This is the
    output contract of ``TmemTranspose16x32Packed`` -- the per-half fc2
    transpose hands the packed regs to this class verbatim.

    The destination row is resolved through ``Fc2ReturnTilePrefetch``:

      * **STG.256 mode (default)**: 2 x STG.256 (32 B each = 16 bf16)
        per thread; each warp lane lands its 32 hidden in 64 B onto a
        unique destination row.  No cross-thread coalescing across
        lanes (each lane targets a distinct token).

      * **REDG mode**: STTM-then-2xLDTM-then-8xREDG.v2.bf16x2 per
        thread (8 B atomic-add per call).  The STTM+LDTM shuffle
        (see below) puts 4 consecutive lanes on the SAME token row's
        contiguous 32 B segment, so the warp-wide
        ``red.global.add.v2.bf16x2`` traffic naturally coalesces
        into sector-aligned 32 B atomic transactions.

    REDG-path register reshuffle -- post-LDTM contract
    ===================================================

    Source TMEM: after STTM ``St32x32b(Repetition.x16)``, this warp's
    32 token rows x 32 hidden cells (packed as 32 lanes x 16
    bf16x2 cols) sit in a contiguous 32-lane x 16-col TMEM slab.

    The slab is then read back through TWO calls of
    ``Ld16x256b(Repetition.x2)``:

      iter 0: source = (TMEM lanes  0..15, cols 0..15)  -- first 16 tokens
      iter 1: source = (TMEM lanes 16..31, cols 0..15)  -- second 16 tokens

    For each call (Rep.x2 = the 4-reg image-1 16dp pattern in cols 0..7
    followed by the same pattern in cols 8..15, yielding 8 reg/thread),
    the per-thread register layout is:

      tmem_lane_in_ldtm = (lane_idx // 4) + 8 * ((elem_idx // 2) % 2)
      tmem_col          = (lane_idx %  4) * 2 + (elem_idx % 2)
                                              + 8 * (elem_idx // 4)

    -- captured as ``_RedgPerLdtmContract`` below.  Concretely, the 8 regs split into FOUR adjacent
    8 B pairs, each pair = 2 contiguous bf16x2 cells along the hidden
    axis of a single token row:

      pair 0 = (Rx+0, Rx+1) -> row (lane//4),     cols (lane%4)*2 .. +1
      pair 1 = (Rx+2, Rx+3) -> row (lane//4)+8,   cols (lane%4)*2 .. +1
      pair 2 = (Rx+4, Rx+5) -> row (lane//4),     cols (lane%4)*2+8 .. +9
      pair 3 = (Rx+6, Rx+7) -> row (lane//4)+8,   cols (lane%4)*2+8 .. +9

    Coalescing invariant (the whole reason we do this shuffle):
    for each fixed (ldtm_iter, pair_idx) the four lanes {T_{4k+0..3}}
    sweep the SAME token row's 8 cells (= 32 B) with stride (2 cells
    = 8 B) per lane.  When all 4 lanes simultaneously fire
    ``red.global.add.v2.bf16x2`` (8 B each), the memory subsystem
    coalesces into one sector-aligned 32 B atomic transaction.

    Per-thread REDG count per subtile-half: 2 ldtm_iter * 4 reg-pair =
    8 REDG.v2.bf16x2 = 64 B (same byte budget as the STG.256-mode
    path's 2 x STG.256).

    Per-thread metadata LDG count: 4 distinct token rows
    (= ``(lane//4) + {0, 8, 16, 24}``); no cross-thread shuffle, every
    thread LDGs its own copy (4 lanes per token row redundantly read
    the same 3 uint32s -- acceptable now, would be the first thing to
    optimise if metadata LDG ever shows up as a profile bottleneck).
    """

    # Direct-path input contract -- shared by both modes, both consume
    # ``TmemTranspose16x32Packed.OutputContract``-shaped RMEM.
    _domain = Space(("lane_idx", "elem_idx"), (32, 16))
    _codomain = Space(("token_idx", "hidden_pair_idx"), (32, 16))
    InputContract = Contract(
        domain=_domain,
        codomain=_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "token_idx": lane_idx,
                "hidden_pair_idx": elem_idx,
            }
        ),
    )

    # REDG-path per-LDTM RMEM contract: the (lane_idx, elem_idx) ->
    # (TMEM lane within the LDTM source slab, TMEM col within the slab)
    # distribution that ONE ``Ld16x256b(Repetition.x2)`` call produces
    # off a 16-lane x 16-col TMEM slab.  See the class docstring for
    # the pair semantics + coalescing invariant.
    _redg_per_ldtm_domain = Space(("lane_idx", "elem_idx"), (32, 8))
    _redg_per_ldtm_codomain = Space(("tmem_lane_in_ldtm", "tmem_col"), (16, 16))
    _RedgPerLdtmContract = Contract(
        domain=_redg_per_ldtm_domain,
        codomain=_redg_per_ldtm_codomain,
        mapping=FunctionMapping(
            lambda lane_idx, elem_idx: {
                "tmem_lane_in_ldtm": (lane_idx // 4) + 8 * ((elem_idx // 2) % 2),
                "tmem_col": (lane_idx % 4) * 2 + (elem_idx % 2) + 8 * (elem_idx // 4),
            }
        ),
    )

    # Per-warp hidden tile width (warp w handles hidden [w*32, w*32+32)).
    _HiddenPerWarp = 32
    # Bits per STG.256 store.
    _StgBitsPerCopy = 256
    # REDG-path layout constants.
    _RedgLdtmIterCount = 2  # 2 x Ld16x256b(Repetition.x2)
    _RedgRegPerLdtm = 8  # reg/thread per LDTM call
    _RedgRegPairsPerLdtm = 4  # = _RedgRegPerLdtm / 2
    _RedgBf16PerPair = 4  # 8 B pair = 2 bf16x2 = 4 bf16

    def __init__(
        self,
        packed: TensorWithContract,
        *,
        return_prefetch: Fc2ReturnTilePrefetch,
        token_iter,
        warp_idx,
        epi_tidx,
        valid_hidden,
        tile_hidden_idx,
        hidden_tile_size: int,
        needs_hidden_predicate: bool,
        fc2_output_dtype: Type[cutlass.Numeric],
        # REDG-only: the (32 lanes, 64 cols) TMEM view onto this warp's
        # acc subtile region (i.e. the ``tmem_subtile_tensor`` the caller
        # built via ``_subtile_local_tmem_tensor``).  We carve a
        # 32-lane x 16-col STTM+LDTM reshuffle slab out of it based on
        # ``half_idx`` (cols 0..15 for half 0, cols 32..47 for half 1 --
        # matching the per-half ``tmem_first_ptr`` / ``tmem_second_ptr``
        # split inside ``_run_fc2_subtile``).  The slab is overwritten
        # by this class's STTM and consumed by its two LDTM reads, both
        # inside the same task tile body -- the next mma (and any acc
        # release) only fires after this body returns, so the reuse is
        # race-free (see ``_run_fc2_subtile`` comments).  Required iff
        # ``return_prefetch.tile.reduce_topk_in_kernel=True``; rejected if
        # that flag is False AND a non-None value is passed (to catch
        # confused call-sites).
        tmem_subtile_scratch: Optional[cute.Tensor] = None,
    ) -> None:
        assert_contract_equivalent(
            packed.contract,
            self.InputContract,
            context="Fc2UnpackPermuteStg packed input",
        )
        if cutlass.const_expr(return_prefetch.tile.reduce_topk_in_kernel):
            if tmem_subtile_scratch is None:
                raise ValueError(
                    "Fc2UnpackPermuteStg: tmem_subtile_scratch is required "
                    "when return_prefetch.tile.reduce_topk_in_kernel=True "
                    "(needs the (32, 64) subtile TMEM view to carve a "
                    "STTM+LDTM reshuffle slab out of)."
                )
        else:
            if tmem_subtile_scratch is not None:
                raise ValueError(
                    "Fc2UnpackPermuteStg: tmem_subtile_scratch must be None "
                    "when reduce_topk_in_kernel=False."
                )

        self._return_prefetch = return_prefetch
        self._token_iter = token_iter
        self._warp_idx = warp_idx
        self._epi_tidx = epi_tidx
        self._lane_idx = epi_tidx % cutlass.Int32(32)
        self._valid_hidden = valid_hidden
        self._tile_hidden_base = hidden_tile_size * tile_hidden_idx
        self._needs_hidden_predicate = needs_hidden_predicate
        self._fc2_output_dtype = fc2_output_dtype

        if cutlass.const_expr(return_prefetch.tile.reduce_topk_in_kernel):
            self._init_redg(packed, tmem_subtile_scratch)
        else:
            self._init_direct(packed)

    # ------------------------------------------------------------------
    # STG.256 mode -- original direct path
    # ------------------------------------------------------------------

    def _init_direct(self, packed: TensorWithContract) -> None:
        """Direct-path setup: unpack + permute packed bf16x2 RMEM back
        to natural hidden order and pre-resolve the destination row.

        Direct STG.256 mode: resolve one destination row per lane.
        """
        self._token_row_global = self._return_prefetch.pool_tokens[self._token_iter]
        self._token_row_1d = self._return_prefetch.row(self._token_iter)
        self._token_valid = self._return_prefetch.is_valid(self._token_iter)

        # -- Step 2: recast (16, Float32) view back to (32, fc2_output_dtype)
        packed_bf16 = cute.recast_tensor(packed.tensor, self._fc2_output_dtype)

        # -- Step 3: unpack + permute back to natural hidden order ---------
        natural = cute.make_rmem_tensor((32,), self._fc2_output_dtype)
        for h in range(16):
            natural[h] = packed_bf16[2 * h]  # lo lane = hidden h
            natural[h + 16] = packed_bf16[2 * h + 1]  # hi lane = hidden h+16

        # -- Step 4: cache for stg() ---------------------------------------
        self._bf16_regs = natural

    @cute.jit
    def _stg_direct(self) -> None:
        """RMEM -> GMEM: 2 x STG.256 of 16 bf16 each.

        Slices:
          - STG #1: ``natural[0 : 16]``  -> hidden cols ``[w*32, w*32+16)``
          - STG #2: ``natural[16 : 32]`` -> hidden cols ``[w*32+16, w*32+32)``
          (where ``w = warp_idx``).

        The destination ``(hidden,)`` row view was pre-resolved in
        ``_init_direct`` and cached as ``self._token_row_1d``; this
        body just slices it at the per-warp / per-STG hidden offset
        and fires STG.256.  No LDG / no peer arithmetic here.
        """
        copy_atom_256b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._fc2_output_dtype,
            num_bits_per_copy=self._StgBitsPerCopy,
        )

        token_row_1d = self._token_row_1d

        # Every STG.256 below is gated by hand-rolled predicates +
        # hand-computed offsets:
        #   * ``hidden_base`` = ``tile_hidden_base + warp_hidden_base +
        #     stg_idx*16`` indexed into ``token_row_1d``, a hand-resolved
        #     FULL ``(hidden,)`` row from the full-tile return prefetch
        #     (token / topk resolved, hidden NOT shifted).  This is how the
        #     ``hidden > cta_tile_m`` bug landed -- the base lean path used
        #     a caller-shifted ``_gOut_subtile`` so the stg body only saw
        #     subtile-local offsets, but the MegaMoE rewrite dropped that
        #     shift on the floor.
        #   * ``token_row_global < valid_token_row_end`` is the (always-on)
        #     row predicate that filters out per-expert padding rows whose
        #     destination ``src_token`` metadata is uninitialised.  Without
        #     it last-task-tile padding STGs corrupt ``combine_output[0]``
        #     (or any row whose ``src_token`` slot is 0 / stale).
        # The whole offset + predicate chain should be redone properly:
        # either a ``TensorWithContract`` that pins the per-tile hidden /
        # token / peer axes (and the valid range) by construction, or a
        # tile-level cute tensor the caller hands down already sliced down
        # to the valid (token, hidden) window.  Until then this is glue.
        if self._token_valid:
            warp_hidden_base = self._warp_idx * self._HiddenPerWarp

            for stg_idx in range(2):
                reg_view = cute.make_tensor(
                    self._bf16_regs.iterator + stg_idx * 16,
                    cute.make_layout((((16,), 1),), stride=(((1,), 0),)),
                )

                hidden_base = self._tile_hidden_base + warp_hidden_base + stg_idx * 16

                gOut_thread_row = cute.local_tile(
                    token_row_1d,
                    (16,),
                    (hidden_base // 16,),
                )

                aligned_row_iter = cute.make_ptr(
                    gOut_thread_row.element_type,
                    gOut_thread_row.iterator.toint(),
                    AddressSpace.gmem,
                    assumed_align=32,
                )
                gOut_thread_row = cute.make_tensor(
                    aligned_row_iter, gOut_thread_row.layout
                )

                if cutlass.const_expr(self._needs_hidden_predicate):
                    if hidden_base < self._valid_hidden:
                        cute.copy(
                            copy_atom_256b,
                            cute.coalesce(reg_view),
                            cute.coalesce(gOut_thread_row),
                        )
                else:
                    cute.copy(
                        copy_atom_256b,
                        cute.coalesce(reg_view),
                        cute.coalesce(gOut_thread_row),
                    )

    # ------------------------------------------------------------------
    # REDG mode -- STTM + 2x LDTM + 8x REDG.v2.bf16x2
    # ------------------------------------------------------------------

    def _init_redg(
        self,
        packed: TensorWithContract,
        tmem_subtile_scratch: cute.Tensor,
    ) -> None:
        """REDG-path setup: fire STTM(St32x32b.x16) immediately so its
        latency overlaps with the upcoming 4 metadata LDGs + 2 LDTMs.

        Also pre-resolves the 4 destination rows (one per distinct
        token this lane holds across both LDTM iterations) so the
        metadata LDGs are issued early too.  Row resolution + STTM
        are independent (the row resolution reads ``metadata`` GMEM +
        the in-param-bank ``peer_rank_ptr_mapper``; STTM writes TMEM), so the
        two latency chains run in parallel.

        ``tmem_subtile_scratch`` is the caller's (32 lanes, 64 cols)
        ``tmem_subtile_tensor`` view onto this warp's acc TMEM subtile
        region.  We carve a 32-lane x 16-col slab out of it at column
        offset ``32 * half_idx`` -- matching the per-half base used by
        the upstream in-place transpose (``tmem_first_ptr`` at col 0,
        ``tmem_second_ptr`` at col 32).  By the time this body runs,
        the transpose has already finished reading the slab back into
        RMEM (``r4_perm`` was the last consumer), so STTM-overwriting
        the slab is race-free; the next mma cannot reclaim the subtile
        until acc release fires at task-tile boundary, well after both
        per-half ``stg()`` calls have completed.
        """
        # Carve the per-half 32x16 slab.  Col stride in the
        # ``_tmem_layout`` is 1 (see ``_TmemTranspose16x32Core``), so
        # a per-col pointer offset of ``32 * half_idx`` matches the
        # ``tmem_second_ptr = tmem_first_ptr + 32`` arithmetic in
        # ``_run_fc2_subtile``.
        redg_iter_in_subtile = self._token_iter % cutlass.Int32(8)
        half_idx = redg_iter_in_subtile // cutlass.Int32(4)
        scratch_iter = tmem_subtile_scratch.iterator + cute.assume(
            32 * half_idx,
            divby=32,
        )
        # Per-LDTM source views: each is 16 lanes x 16 cols, addressing
        # rows 0..15 (iter 0) and rows 16..31 (iter 1) of the slab.
        # The 16-lane offset in TMEM cell units = 16 *
        # ``_TmemTranspose16x32Core._TmemRowStride``.
        ldtm_half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride
        self._redg_ldtm_src_views = (
            cute.make_tensor(
                scratch_iter,
                _TmemTranspose16x32Core._tmem_layout(16, 16),
            ),
            cute.make_tensor(
                scratch_iter + ldtm_half_lane_off,
                _TmemTranspose16x32Core._tmem_layout(16, 16),
            ),
        )

        # -- Natural-order permute + STTM to TMEM scratch -----------------
        packed_bf16 = cute.recast_tensor(packed.tensor, self._fc2_output_dtype)
        natural = cute.make_rmem_tensor((32,), self._fc2_output_dtype)
        for h in range(16):
            natural[h] = packed_bf16[2 * h]
            natural[h + 16] = packed_bf16[2 * h + 1]
        natural_fp32 = cute.recast_tensor(natural, cutlass.Float32)

        sttm_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16),
            cutlass.Float32,
        )
        rmem_view = cute.make_tensor(
            natural_fp32.iterator,
            cute.make_layout((((16,), 1),), stride=(((1,), 0),)),
        )
        sttm_dst_view = cute.make_tensor(
            scratch_iter,
            _TmemTranspose16x32Core._tmem_layout(32, 16),
        )
        cute.copy(sttm_atom, rmem_view, sttm_dst_view)

        # -- Pre-resolve the 4 destination rows ----------------------------
        # Lane t (in half) holds 4 distinct token rows across the two
        # LDTM iterations:
        #
        #   ldtm_iter 0:  token_in_half = (lane % 32 // 4) + {0, 8}
        #   ldtm_iter 1:  token_in_half = (lane % 32 // 4) + {16, 24}
        #
        # The 4 lanes {T_{4k+0..3}} all share the SAME 4 tokens (one
        # for each (iter, pair_parity) slot), so 4 lanes redundantly
        # LDG the same metadata + ``peer_rank_ptr_mapper.map`` redirect; no shuffle,
        # simple but suboptimal (acceptable per design decision).
        #
        # Per-half token base = 32 * half_idx (NOT lane-dependent --
        # ``lane // 4 + {0, 8, 16, 24}`` covers all 32 token slots
        # within the half).
        # Per-lane full-tile token rows + matching dest-row tensors.
        self._redg_token_row_globals = tuple(
            self._return_prefetch.pool_tokens[self._token_iter + cutlass.Int32(i)]
            for i in range(self._RedgRegPairsPerLdtm)
        )
        self._redg_token_rows_1d = tuple(
            self._return_prefetch.row(self._token_iter + cutlass.Int32(i))
            for i in range(self._RedgRegPairsPerLdtm)
        )
        self._redg_token_valid = tuple(
            self._return_prefetch.is_valid(self._token_iter + cutlass.Int32(i))
            for i in range(self._RedgRegPairsPerLdtm)
        )

    @cute.jit
    def _stg_redg(self) -> None:
        """TMEM scratch -> RMEM -> GMEM REDG path.

        For each of 2 LDTM iterations, this body:

          1. Calls ``Ld16x256b(Repetition.x2)`` on the per-iter
             16-lane x 16-col TMEM slab.  Each call produces 8
             reg/thread following ``_RedgPerLdtmContract`` (see the
             class docstring for the (lane, elem) -> (tmem_lane, tmem_col)
             mapping and the reg-pair semantics).
          2. For each of 4 reg-pairs, emits 1
             ``red.global.add.v2.bf16x2`` call (= 2 packed bf16x2 = 4
             bf16 = 8 B atomic-add) onto the destination row
             corresponding to that pair's (token, hidden_seg) slot.

        Total per thread per subtile-half: 2 LDTM + 8 REDG.v2.bf16x2
        = 64 B written = same byte budget as the STG.256-mode path.

        Predicates: a single per-token-row valid predicate gates the
        REDGs targetting that row.  Hidden predicate (when the warp's
        hidden segment is past ``valid_hidden``) is checked once per
        pair (every pair = 1 hidden segment of 4 contiguous bf16).
        """
        ldtm_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2),
            cutlass.Float32,
        )

        warp_hidden_base = self._warp_idx * self._HiddenPerWarp

        # Per-pair token-row global + dest-row picker.  Pairs 0/2 share
        # token A = base_in_half + 0/16; pairs 1/3 share token B = +8/+24.
        # Combined into a (iter, pair_idx_in_iter) -> token_slot table
        # so the inner loop can index it directly.
        #
        # pair_idx_in_iter -> token_slot in self._redg_token_rows_1d:
        #   iter 0 pair 0: slot 0  (base + 0)
        #   iter 0 pair 1: slot 1  (base + 8)
        #   iter 0 pair 2: slot 0
        #   iter 0 pair 3: slot 1
        #   iter 1 pair 0: slot 2  (base + 16)
        #   iter 1 pair 1: slot 3  (base + 24)
        #   iter 1 pair 2: slot 2
        #   iter 1 pair 3: slot 3
        pair_to_token_slot = (0, 1, 0, 1)

        # Address-side per-pair hidden offset in bf16 elements:
        #   pair 0/1 -> low  segment, starting at  hidden_pair_col * 2
        #               (= (lane%4) * 2 * 2 = (lane%4) * 4 bf16 from
        #               the warp's hidden base)
        #   pair 2/3 -> high segment, +16 bf16 from low
        # Computed once outside the inner loop because lane-relative.
        lane_quad = self._lane_idx % cutlass.Int32(4)
        # Per-pair low-end hidden offset within the warp's 32-hidden
        # segment, in bf16 element units.
        per_pair_hidden_in_warp_lo = (
            lane_quad * cutlass.Int32(4),  # pair 0
            lane_quad * cutlass.Int32(4),  # pair 1
            lane_quad * cutlass.Int32(4) + cutlass.Int32(16),  # pair 2
            lane_quad * cutlass.Int32(4) + cutlass.Int32(16),  # pair 3
        )

        for ldtm_iter in cutlass.range_constexpr(self._RedgLdtmIterCount):
            regs = cute.make_rmem_tensor(
                (self._RedgRegPerLdtm,),
                cutlass.Float32,
            )
            cute.copy(
                ldtm_atom,
                self._redg_ldtm_src_views[ldtm_iter],
                cute.make_tensor(
                    regs.iterator,
                    cute.make_layout(
                        (((self._RedgRegPerLdtm,), 1),),
                        stride=(((1,), 0),),
                    ),
                ),
            )
            # ``regs`` is 8 fp32 slots; each slot's 32-bit bit-pattern
            # is one packed bf16x2 (= 2 bf16 along hidden), inherited
            # from ``Fc2AccLoadAndPack`` upstream.  We pass the fp32
            # ir.Value straight into the bf16x2 REDG inline asm -- PTX
            # constraint ``r`` only cares about the 32-bit register
            # payload, see ``_red_add_relaxed_sys_v2_bf16x2`` docstring.

            for pair_idx in cutlass.range_constexpr(self._RedgRegPairsPerLdtm):
                token_slot = pair_to_token_slot[pair_idx] + (2 if ldtm_iter == 1 else 0)
                token_row_1d = self._redg_token_rows_1d[token_slot]
                token_valid = self._redg_token_valid[token_slot]
                hidden_in_warp_lo = per_pair_hidden_in_warp_lo[pair_idx]
                hidden_base = (
                    cutlass.Int32(self._tile_hidden_base)
                    + cutlass.Int32(warp_hidden_base)
                    + hidden_in_warp_lo
                )

                # One v2.bf16x2 REDG = the pair's full 8 B payload (4
                # bf16) in a single PTX op.  4 consecutive lanes hold
                # adjacent (hidden_base + lane_quad * 4 ..
                # + (lane_quad+1)*4) ranges of the same row, which the
                # memory subsystem coalesces into a sector-aligned 32 B
                # atomic transaction (the whole reason we did the
                # STTM+LDTM shuffle).  Pair occupies fp32 regs
                # ``[pair_idx*2, pair_idx*2 + 1]`` -- each fp32 slot is
                # one packed bf16x2 cell along the hidden axis.
                if token_valid:
                    addr = token_row_1d.iterator + hidden_base
                    val0 = cutlass.Float32(regs[pair_idx * 2])
                    val1 = cutlass.Float32(regs[pair_idx * 2 + 1])
                    if cutlass.const_expr(self._needs_hidden_predicate):
                        if hidden_base < self._valid_hidden:
                            _red_add_relaxed_sys_v2_bf16x2(
                                addr,
                                val0,
                                val1,
                            )
                    else:
                        _red_add_relaxed_sys_v2_bf16x2(
                            addr,
                            val0,
                            val1,
                        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    @cute.jit
    def stg(self) -> None:
        """Const_expr dispatch to the STG.256 (default) or REDG path."""
        if cutlass.const_expr(self._return_prefetch.tile.reduce_topk_in_kernel):
            self._stg_redg()
        else:
            self._stg_direct()


# =============================================================================
# SwapABSwigluFp4Epilogue
# =============================================================================


class SwapABSwigluFp4Epilogue:
    """Autonomous epilogue for the swap-AB SwiGLU NVFP4 kernel.

    ``run()`` is the single entry point the kernel calls inside the epi
    warp body.  The kernel's responsibility is reduced to:

      - allocate / free TMEM and build ``acc_tensor``
      - construct the AB / acc pipelines
      - obtain the scheduler consumer

    Everything else (acc consumer state, task-tile loop, overlap rotation,
    early release, TMA store commit / drain, per-subtile dispatch) lives
    inside this class.
    """

    # Per-subtile rotated-leader sync constants.
    _SubtileBarIdBase = 4

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        sf_vec_size: int,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        fc2_output_dtype: Type[cutlass.Numeric],
        non_ubulk_fc2_store: bool,
        in_kernel_fc2_reduce: bool,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: int = 1,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        allow_overlap_acc: bool = True,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if fc1_output_dtype is not cutlass.Float4E2M1FN:
            raise NotImplementedError(
                "SwapABSwigluFp4Epilogue currently assumes fc1 output in "
                f"sC is NVFP4 Float4E2M1FN; got {fc1_output_dtype}. "
                "Changing this dtype requires redesigning the fixed 8KB "
                "shared epilogue scratch layout."
            )
        if token_back_by_dispatch and not non_ubulk_fc2_store:
            raise ValueError(
                "token_back_by_dispatch=True requires non_ubulk_fc2_store=True; "
                "bulk fc2 store is incompatible with dispatch-warp token back "
                "(STG is strictly more efficient for that pipeline)."
            )
        if token_back_by_dispatch:
            in_kernel_fc2_reduce = False
        self._fc2_use_bulk = not non_ubulk_fc2_store
        self._reduce_topk_in_kernel = in_kernel_fc2_reduce
        self._token_back_by_dispatch = token_back_by_dispatch
        # Done-counter publish batch granularity (per-tile == 1).  Plain Python
        # int -> usable as a compile-time shape / range_constexpr bound below.
        self._epi_flag_batch = max(1, int(epi_flag_batch))
        self._fc2_output_dtype = fc2_output_dtype

        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self._sf_vec_size = sf_vec_size
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids

        atom_thr_size = 2 if use_2cta_instrs else 1
        self._cta_tile_m = mma_tiler_mnk[0] // atom_thr_size
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if self._fc2_use_bulk:
            if static_expert_shape is None:
                raise NotImplementedError(
                    "fc2_ublkcp/fc2_ublkredg currently require "
                    "static_expert_shape so the 128-hidden bulk tail is "
                    "known at codegen time."
                )
            if static_expert_shape[2] % 8 != 0:
                raise NotImplementedError(
                    "fc2_ublkcp/fc2_ublkredg currently require hidden to be "
                    f"a multiple of 8 BF16 elements (16B bulk alignment); "
                    f"got hidden={static_expert_shape[2]}."
                )
        if (
            static_expert_shape is not None
            and static_expert_shape[2] % (self._cta_tile_m * cluster_shape_mn[0]) == 0
        ):
            self._fc2_hidden_needs_predicate: bool = False
        else:
            self._fc2_hidden_needs_predicate: bool = True

        # K-padding gate for fc1 epi SF writes; see PostSwigluHalf.stg_sfc.
        # ``cga_cluster_tile_intermediate_downproj`` is the CGA-level
        # alignment unit on the intermediate_downproj axis; when
        # ``intermediate_downproj`` is an integer multiple of it the
        # predicate is statically True and elided.
        self._cga_cluster_tile_intermediate_downproj: int = (
            self._cta_tile_m // 2
        ) * cluster_shape_mn[0]
        if static_expert_shape is not None:
            intermediate_downproj = static_expert_shape[1] // 2
            if intermediate_downproj % sf_vec_size != 0:
                raise NotImplementedError(
                    f"intermediate_downproj ({intermediate_downproj}) must "
                    f"be a multiple of sf_vec_size ({sf_vec_size}); sub-SF-"
                    f"block K-masking is not implemented."
                )
            self._intermediate_downproj: Optional[int] = intermediate_downproj
        else:
            self._intermediate_downproj: Optional[int] = None

        self._epi_tile = (EpilogueTokenTile, Fc1EpilogueOutputTile)
        self._subtile_cnt = self._cta_tile_n // EpilogueTokenTile

        self._overlapping_accum = allow_overlap_acc and (
            self._cta_tile_n == EpiWarpCount * EpilogueTokenTile
        )

        self._num_acc_stage = 2
        self._num_acc_pipeline_stages = (
            1 if self._overlapping_accum else self._num_acc_stage
        )

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = (
            self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128
        )
        self._num_sf_tmem_cols = self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage - (
            self._num_sf_tmem_cols if self._overlapping_accum else 0
        )

        self._iter_acc_early_release = (
            self._num_sf_tmem_cols + EpilogueTokenTile - 1
        ) // EpilogueTokenTile

    # -- Codegen-time queries (read by kernel) --------------------------------

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def overlapping_accum(self) -> bool:
        return self._overlapping_accum

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self._num_acc_pipeline_stages

    @property
    def num_acc_stage(self) -> int:
        return self._num_acc_stage

    @property
    def iter_acc_early_release(self) -> int:
        return self._iter_acc_early_release

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    @property
    def num_sf_tmem_cols(self) -> int:
        return self._num_sf_tmem_cols

    @property
    def num_sfa_tmem_cols(self) -> int:
        return self._num_sfa_tmem_cols

    @property
    def num_sfb_tmem_cols(self) -> int:
        return self._num_sfb_tmem_cols

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self._num_accumulator_tmem_cols

    # -- sC SMEM layout queries -----------------------------------------------

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self._epi_tile,
            n_stages,
        )

    @property
    def smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype, self.smem_layout_one_stage)

    @property
    def num_fc1_c_stages(self) -> int:
        # sC is a fixed 8KB epilogue scratch.  fc1 views it as four 2KB
        # NVFP4 C stages; fc2 UBLK views the same bytes as one 32x128 BF16
        # scratch tile.
        return 4

    @staticmethod
    def _fc2_bulk_smem_layout() -> cute.Layout:
        return cute.make_layout((32, 128), stride=(128, 1))

    def fc2_bulk_scratch_view(self, smem_fc1_output_buffer: cute.Tensor) -> cute.Tensor:
        scratch_iter = cute.recast_ptr(
            smem_fc1_output_buffer.iterator,
            dtype=self._fc2_output_dtype,
        )
        return cute.make_tensor(scratch_iter, self._fc2_bulk_smem_layout())

    # -- Subtile-local TMEM view helper ---------------------------------------

    def _subtile_local_tmem_tensor(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
        acc_stage_col_offset,
    ) -> cute.Tensor:
        """Build a (32 lanes, 64 cols) cute.Tensor view onto one epi
        subtile's per-warp acc TMEM region.

        Owns the per-warp lane offset, per-stage col offset (overlap-acc
        phase aware), and per-subtile col offset arithmetic.  Returned
        tensor is what ``_run_fc{1,2}_subtile`` and
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` consume.

        ``cute.assume(divby=16)`` is applied here once -- callees can
        derive ``+32`` first/second-half ptrs from the returned tensor's
        iterator without re-asserting alignment (16-aligned base + 32 is
        still 16-aligned).

        ``divby=16`` (instead of 64) so the assume holds even under
        ``overlapping_accum=True`` with phase=1, where
        ``acc_stage_col_offset = phase * (256 - num_sf_tmem_cols) = 208``
        (when ``num_sf_tmem_cols = 48``) and ``208 % 64 = 16``.
        ``divby=16`` still satisfies the downstream alignment check that
        ``cute.assume`` exists to bypass; the codegen optimisation
        difference between ``divby=16`` and ``divby=64`` is negligible
        for this offset arithmetic.
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTokenTile
        total = (warp_lane_off << 16) + acc_stage_col_offset + subtile_col_off
        subtile_ptr = base + cute.assume(total, divby=16)
        return cute.make_tensor(
            subtile_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTokenTile),
        )

    # -- Subtile-level TMA store cmd issue ------------------------------------

    @staticmethod
    @dsl_user_op
    @cute.jit
    def tma_store_fc1_output(
        warp_idx,
        sC,
        subtile_idx,
        tma_atom_fc1_output,
        g_fc1_output_subtile_view: cute.Tensor,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """Per-subtile fence + rotated-leader TMA store cmd issue.

        Subtile-level operation -- it is not per-half, so it lives on the
        epilogue rather than on ``PostSwigluHalf``.  All 4 epi warps call
        this; ``subtile_idx`` doubles as the sC stage index AND drives both
        the leader-warp choice and the NamedBarrier id::

            leader_warp_idx = subtile_idx              (warp s leads sC[s])
            subtile_bar_id  = _SubtileBarIdBase + subtile_idx

        Each subtile owns its own bar id, so producer warps fire-and-forget
        arrive on this bar and race ahead into the next subtile without
        phase-mismatch on a shared bar.  The leader does ``arrive_and_wait``
        and issues the bulk-tensor store; the other 3 warps only ``arrive``.
        No commit / acquire here -- task-tile-boundary commit + drain lives
        inside ``run()``.

        ``@staticmethod @dsl_user_op @cute.jit``: jit is required for the
        ``if warp_idx == leader_warp_idx`` runtime conditional to lower to
        scf.if; making it a free-form static keeps live-locals serialization
        simple (only cute-native types in scope).
        """
        cute.arch.fence_proxy("async.shared", space="cta")
        sC_stage = cute.slice_(sC, (None, None, subtile_idx))
        g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view, (None, None, 0))
        bSG_sC, bSG_g_fc1_output = cpasync.tma_partition(
            tma_atom_fc1_output,
            0,
            cute.make_layout(1),
            cute.group_modes(sC_stage, 0, 2),
            cute.group_modes(g_fc1_output_2d, 0, 2),
        )

        # Warp 0 issues all fc1 TMA stores so the task-tile commit/wait covers
        # every subtile before the release-add to fc1_done_counter.
        leader_warp_idx = cutlass.Int32(0)
        subtile_bar_id = subtile_idx + cutlass.Int32(
            SwapABSwigluFp4Epilogue._SubtileBarIdBase
        )
        subtile_bar = pipeline.NamedBarrier(
            barrier_id=subtile_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if warp_idx == leader_warp_idx:
            subtile_bar.arrive_and_wait()
            cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g_fc1_output)
        else:
            subtile_bar.arrive()

    # -- Full task-tile loop --------------------------------------------------

    @cute.jit
    def run(
        self,
        # ── Acc TMEM + acc pipeline (shared, both phases) ────────────────
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        # ── Sched ────────────────────────────────────────────────────────
        sched_consumer,
        sched_ext,
        # ── fc1 (Linear1 phase) outputs ──────────────────────────────────
        # NVFP4 packed output staged through SMEM and dispatched by TMA
        # bulk store.
        smem_fc1_output_buffer: cute.Tensor,  # sC SMEM (subtile_cnt slots)
        tma_atom_fc1_output: cute.CopyAtom,  # TMA store atom for fc1 NVFP4 out
        gmem_fc1_output: cute.Tensor,  # GMEM target for the TMA store
        gmem_fc1_output_sf: cute.Tensor,  # GMEM fp8 SFC, written by per-thread STG
        # ── topk weights (applied before fc1-output NVFP4 quantize) ─────
        gmem_topk_scores: cute.Tensor,
        # ── fc2 (Linear2 phase) output ───────────────────────────────────
        # MoE-domain ``(token_max, topk, hidden)`` output.
        gmem_fc2_output: cute.Tensor,
        # ── fc1 -> fc2 release-acquire signal ──────────────────────────
        # GMEM int32 counter, 1D shape (max_token_block_per_rank,). Indexed by
        # ``cumulative_token_block_count + tile_n_idx`` as carried by
        # ``SwapABSwigluFp4Fc12WorkTileInfo``.
        gmem_fc1_done_counter: cute.Tensor,
        # ── Per-warp / per-thread ────────────────────────────────────────
        warp_idx: int,
        tidx,
        # ── Epi-side runtime scaling (fc1 only) ─────────────────────────
        alpha,
        norm_const,
        # ── MegaMoE-only routing bundle (Optional) ──────────────────────
        # None = direct local output; non-None = metadata-driven peer output.
        token_comm_args=None,
    ) -> None:
        """Run the full fc1+fc2-fused epilogue task-tile loop.

        After each task tile, the next scheduler tile is consumed before the
        previous fc1 TMA stores are drained and the fc1_done_counter is
        release-added.  That ordering overlaps scheduler latency while still
        publishing fc1 output before any fc2 tile can pass its spin.
        """
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages
        )
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=32 * len(self._epilogue_warp_ids),
        )

        # gmem_fc2_output is MoE-domain (token_max, topk, hidden); valid
        # hidden cols = axis 2 size.
        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[2])

        # In-bound fc1 tile count (= ext_fc2_spin_threshold mirror).
        # gmem_fc1_output is (tokens_sum_padded, intermediate_downproj);
        # tile step (post-swiglu) = cta_tile_m // 2.
        intermediate_downproj_tile_count = (
            gmem_fc1_output.shape[1] + (self._cta_tile_m // 2) - 1
        ) // (self._cta_tile_m // 2)

        # Init=1 (= reverse): under overlapping_accum the first task tile
        # walks subtiles N-1, 0, 1, ..., N-2 so the rightmost subtile
        # (containing the staggered overlap region cols) is processed first
        # and its TMEM cols are released to the next phase's mma immediately.
        # Shared across phases -- fc2 inherits the same overlap rotation.
        # Constexpr-elided under non-overlap.
        is_odd_turn = cutlass.Int32(1)

        # Done-counter publish batch scratch: accumulate up to epi_batch
        # reduction target addresses (fc1 or fc2, unified) and flush them under
        # one device fence.  Only tidx==0 reads/writes the buffer; epi_pend_n is
        # kept warp-uniform (updated under uniform conditions) so all epilogue
        # threads agree on flush points.  epi_batch is a plain Python int, usable
        # as a compile-time tensor shape / range_constexpr bound.
        epi_batch = self._epi_flag_batch
        epi_pend_n = cutlass.Int32(0)
        epi_pend_is_fc1 = cutlass.Int32(0)
        epi_pend_addr = cute.make_rmem_tensor((epi_batch,), cutlass.Int64)

        def _flush_done_batch(_pend_n, _pend_addr, _tidx, _batch):
            # tidx==0 publishes the accumulated batch: ONE device fence (the
            # preceding task_tile_boundary_bar already ordered every epilogue
            # warp's stores at CTA scope) then one relaxed reduction per
            # recorded target.  Returns the reset count.  cuTeDSL forbids
            # closures over ANY local (even a Python int), so the buffer / tid /
            # batch-size are all passed explicitly.
            if _tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                for _b in cutlass.range_constexpr(0, _batch, 1):
                    if cutlass.Int32(_b) < _pend_n:
                        _red_add_relaxed_gpu_s32_addr(_pend_addr[_b], cutlass.Int32(1))
            return cutlass.Int32(0)

        work_tile_info = sched_consumer.consume_work()
        while work_tile_info.is_valid_tile:
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                self._run_fc1_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    gmem_fc1_output_sf=gmem_fc1_output_sf,
                    gmem_topk_scores=gmem_topk_scores,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    alpha=alpha,
                    norm_const=norm_const,
                )
            else:
                self._run_fc2_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    sched_ext=sched_ext,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    gmem_fc2_output=gmem_fc2_output,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    token_comm_args=token_comm_args,
                )
            iket.range_pop()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            cur_fc1_counter_slot = (
                work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
            )
            # Only valid fc1 N-tiles signal the buffer ready. ``tile_m_idx`` in swap_ab is the
            # ``intermediate_downproj_tile_idx``.
            cur_intermediate_downproj_tile_in_bound = (
                work_tile_info.tile_m_idx < intermediate_downproj_tile_count
            )
            cur_fc2_expert_idx = work_tile_info.expert_idx

            acc_consumer_state.advance()
            if cutlass.const_expr(self._overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            work_tile_info = sched_consumer.consume_work()

            iket.range_push("epi_flag")
            # Drain fc1 TMA stores locally (per-tile).  The device-scope fence
            # that publishes them is batched with the done-counter reduction
            # below -- the fence orders fc1 TMA and fc2 STG stores alike, so we
            # only record the reduction target here.
            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            task_tile_boundary_bar.arrive_and_wait()

            # Publish completion.  fc1_done_counter feeds back into the fc2 MMA
            # (whose acc the epilogue later consumes), so a batch must never
            # straddle an fc1<->fc2 phase boundary: when the publishing phase
            # flips, flush the accumulated batch FIRST so the next phase never
            # waits on a flag still trapped in the buffer (else deadlock).
            # Within a phase, batch the device fence + relaxed reductions across
            # up to epi_batch tiles.  epi_pend_n / epi_pend_is_fc1 are kept
            # warp-uniform so every epilogue thread agrees on flush points.
            if cur_was_linear1:
                if cur_intermediate_downproj_tile_in_bound:
                    if epi_pend_n > cutlass.Int32(0):
                        if epi_pend_is_fc1 == cutlass.Int32(0):
                            epi_pend_n = _flush_done_batch(
                                epi_pend_n, epi_pend_addr, tidx, epi_batch
                            )
                    if tidx == 0:
                        epi_pend_addr[epi_pend_n] = (
                            gmem_fc1_done_counter.iterator + cur_fc1_counter_slot
                        ).toint()
                    epi_pend_is_fc1 = cutlass.Int32(1)
                    epi_pend_n = epi_pend_n + cutlass.Int32(1)
                    if epi_pend_n == cutlass.Int32(epi_batch):
                        epi_pend_n = _flush_done_batch(
                            epi_pend_n, epi_pend_addr, tidx, epi_batch
                        )
            else:
                if cutlass.const_expr(self._token_back_by_dispatch):
                    if epi_pend_n > cutlass.Int32(0):
                        if epi_pend_is_fc1 == cutlass.Int32(1):
                            epi_pend_n = _flush_done_batch(
                                epi_pend_n, epi_pend_addr, tidx, epi_batch
                            )
                    if tidx == 0:
                        epi_pend_addr[epi_pend_n] = (
                            token_comm_args.fc2_done_counter.iterator
                            + cur_fc2_expert_idx
                        ).toint()
                    epi_pend_is_fc1 = cutlass.Int32(0)
                    epi_pend_n = epi_pend_n + cutlass.Int32(1)
                    if epi_pend_n == cutlass.Int32(epi_batch):
                        epi_pend_n = _flush_done_batch(
                            epi_pend_n, epi_pend_addr, tidx, epi_batch
                        )
            iket.range_pop()

        # Tail flush: publish any leftover (< epi_batch) accumulated targets with
        # one final device fence.  Each leftover tile's stores were already
        # ordered by its per-tile boundary barrier inside the loop.
        if epi_pend_n > cutlass.Int32(0):
            epi_pend_n = _flush_done_batch(epi_pend_n, epi_pend_addr, tidx, epi_batch)

    # -- Per-phase task-tile dispatch ------------------------------------------

    @cute.jit
    def _run_fc1_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
    ) -> None:
        """Linear1 task-tile body."""
        real_fc1_output, _ = sched_ext.get_gmem_tensor(
            "c",
            gmem_fc1_output,
            work_tile_info,
        )
        real_fc1_output_sf, _ = sched_ext.get_gmem_tensor(
            "sfc",
            gmem_fc1_output_sf,
            work_tile_info,
        )
        # Shifted topk view; downstream indexing is expert-local.
        real_topk_scores, _ = sched_ext.get_gmem_tensor(
            "topk",
            gmem_topk_scores,
            work_tile_info,
        )

        iket.range_push("fc1_epi_wait")
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_pop()
        iket.range_push("fc1_epi")

        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(acc_consumer_state.phase) * (
                256 - self._num_sf_tmem_cols
            )
        else:
            acc_stage_col_offset = (
                cutlass.Int32(acc_consumer_state.index) * self._cta_tile_n
            )

        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        subtile_cnt = self._subtile_cnt

        # Overlap path preloads two subtiles before releasing acc TMEM.
        unroll_tile_cnt = 2 if cutlass.const_expr(self._overlapping_accum) else 0
        remain_subtile_cnt = subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (
                cutlass.Int32(subtile_cnt) - is_odd_turn
            ) % cutlass.Int32(subtile_cnt)
            subtile_idx_second = (
                cutlass.Int32(subtile_cnt + 1) - is_odd_turn
            ) % cutlass.Int32(subtile_cnt)

            tmem_subtile_first = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_first,
                warp_idx,
                acc_stage_col_offset,
            )
            tmem_subtile_second = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_second,
                warp_idx,
                acc_stage_col_offset,
            )

            # Always preload before unconditional acc release.
            preload_subtile_first = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_first
            )

            # Release acc to next MMA unconditionally.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            preload_subtile_second = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_second
            )

            # Both unrolled subtiles borrow tmem_subtile_second as workspace.
            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(64) < valid_tokens:
                    self._run_fc1_subtile(
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_subtile_second,
                        real_fc1_output=real_fc1_output,
                        real_fc1_output_sf=real_fc1_output_sf,
                        real_topk_scores=real_topk_scores,
                        work_tile_info=work_tile_info,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        warp_idx=warp_idx,
                        tidx=tidx,
                        alpha=alpha,
                        norm_const=norm_const,
                        preload_acc=preload_pair[i],
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = (
                    cutlass.Int32(real_i + subtile_cnt) - is_odd_turn
                ) % cutlass.Int32(subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(64) < valid_tokens:
                self._run_fc1_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=self._subtile_local_tmem_tensor(
                        tmem_acc_tensor,
                        subtile_idx,
                        warp_idx,
                        acc_stage_col_offset,
                    ),
                    real_fc1_output=real_fc1_output,
                    real_fc1_output_sf=real_fc1_output_sf,
                    real_topk_scores=real_topk_scores,
                    work_tile_info=work_tile_info,
                    smem_fc1_output_buffer=smem_fc1_output_buffer,
                    tma_atom_fc1_output=tma_atom_fc1_output,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    alpha=alpha,
                    norm_const=norm_const,
                )

        # Non-overlap-path release: at the natural task-tile boundary.
        if cutlass.const_expr(not self._overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_fc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        sched_ext,
        smem_fc1_output_buffer: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
    ) -> None:
        """fc2 (Linear2) task-tile body.

        Mirrors ``_run_fc1_task_tile``'s shape -- same acc_pipeline
        wait/release lifecycle, same subtile loop with overlap rotation
        and valid_tokens early-exit.  The transpose path uses
        ``_run_fc2_subtile`` per subtile; the UBLK path delegates the
        task-tile body to ``_run_fc2_bulk_task_tile``.

        Path A: fc2 epi takes no topk weight (the per-token scalar was
        already pre-multiplied into the swiglu fp32 output by the upstream
        fc1 ``PostSwigluHalf``).  The default transpose path is LDTM + cvt
        + pack + transpose + unpack-permute + STG.256.  The UBLK path uses
        LDTM + BF16 scratch staging + descriptor-free bulk copy/reduce.

        ``gmem_fc2_output`` is the MoE-domain ``(token_max, topk, hidden)``
        tensor; return routing is owned by ``Fc2ReturnTile`` at full CTA
        token tile granularity.
        """
        metadata_u32 = None
        peer_rank_ptr_mapper = None
        if cutlass.const_expr(token_comm_args is None or self._token_back_by_dispatch):
            if cutlass.const_expr(self._reduce_topk_in_kernel):
                raise ValueError(
                    "in_kernel_fc2_reduce requires token_comm metadata + a "
                    "peer_rank_ptr_mapper and does not coexist with "
                    "token_back_by_dispatch; STG to a local fc2 workspace "
                    "instead."
                )
        else:
            metadata_u32 = cute.recast_tensor(
                token_comm_args.token_src_metadata,
                cutlass.Uint32,
            )
            peer_rank_ptr_mapper = token_comm_args.peer_rank_ptr_mapper

        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        subtile_cnt = self._subtile_cnt
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
        )
        valid_token_row_end = task_tile_data_row_start + valid_tokens
        thread_in_warp = tidx % 32

        if cutlass.const_expr(self._fc2_use_bulk):
            token_contract = BulkReturnTokenContract(self._cta_tile_n)
        elif cutlass.const_expr(self._reduce_topk_in_kernel):
            token_contract = TransposeRedgReturnTokenContract(self._cta_tile_n)
        else:
            token_contract = TransposeStgReturnTokenContract(self._cta_tile_n)

        return_tile = Fc2ReturnTile(
            tensor=gmem_fc2_output,
            metadata=metadata_u32,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            cta_token_tile_start=task_tile_data_row_start,
            valid_token_row_end=valid_token_row_end,
            reduce_topk_in_kernel=self._reduce_topk_in_kernel,
            token_contract=token_contract,
        )
        epi_tidx = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp

        acc_ready = cutlass.Boolean(False)
        if not work_tile_info.peek_ready:
            acc_pipeline.consumer_wait(acc_consumer_state)
            acc_ready = cutlass.Boolean(True)
        return_prefetch = return_tile.prefetch_for_epi_thread(epi_tidx)

        iket.range_push("fc2_epi_wait")
        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        iket.range_pop()
        iket.range_push("fc2_epi")

        if cutlass.const_expr(self._overlapping_accum):
            acc_stage_col_offset = cutlass.Int32(acc_consumer_state.phase) * (
                256 - self._num_sf_tmem_cols
            )
        else:
            acc_stage_col_offset = (
                cutlass.Int32(acc_consumer_state.index) * self._cta_tile_n
            )

        if cutlass.const_expr(self._fc2_use_bulk):
            self._run_fc2_bulk_task_tile(
                work_tile_info=work_tile_info,
                tmem_acc_tensor=tmem_acc_tensor,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                is_odd_turn=is_odd_turn,
                smem_fc1_output_buffer=smem_fc1_output_buffer,
                return_prefetch=return_prefetch,
                valid_tokens=valid_tokens,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                acc_stage_col_offset=acc_stage_col_offset,
            )
            return

        # Same overlap-acc structure as fc1; downstream subtile body differs.
        unroll_tile_cnt = 2 if cutlass.const_expr(self._overlapping_accum) else 0
        remain_subtile_cnt = subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (
                cutlass.Int32(subtile_cnt) - is_odd_turn
            ) % cutlass.Int32(subtile_cnt)
            subtile_idx_second = (
                cutlass.Int32(subtile_cnt + 1) - is_odd_turn
            ) % cutlass.Int32(subtile_cnt)

            tmem_subtile_first = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_first,
                warp_idx,
                acc_stage_col_offset,
            )
            tmem_subtile_second = self._subtile_local_tmem_tensor(
                tmem_acc_tensor,
                subtile_idx_second,
                warp_idx,
                acc_stage_col_offset,
            )

            preload_subtile_first = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_first
            )

            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            preload_subtile_second = _TmemTranspose16x32Core.load_subtile_raw_acc(
                tmem_subtile_second
            )

            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(64) < valid_tokens:
                    self._run_fc2_subtile(
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_subtile_second,
                        return_prefetch=return_prefetch,
                        work_tile_info=work_tile_info,
                        valid_hidden=valid_hidden,
                        warp_idx=warp_idx,
                        tidx=tidx,
                        preload_acc=preload_pair[i],
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = (
                    cutlass.Int32(real_i + subtile_cnt) - is_odd_turn
                ) % cutlass.Int32(subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)

            if subtile_idx * cutlass.Int32(64) < valid_tokens:
                self._run_fc2_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=self._subtile_local_tmem_tensor(
                        tmem_acc_tensor,
                        subtile_idx,
                        warp_idx,
                        acc_stage_col_offset,
                    ),
                    return_prefetch=return_prefetch,
                    work_tile_info=work_tile_info,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                )

        if cutlass.const_expr(not self._overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_fc2_bulk_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn,
        smem_fc1_output_buffer: cute.Tensor,
        return_prefetch: Fc2ReturnTilePrefetch,
        valid_tokens,
        valid_hidden,
        warp_idx: int,
        tidx,
        acc_stage_col_offset,
    ) -> None:
        """Run one fc2 task tile through the UBLK bulk-copy epilogue path."""
        scratch = self.fc2_bulk_scratch_view(smem_fc1_output_buffer)
        subtile_cnt = self._subtile_cnt

        if cutlass.const_expr(self._overlapping_accum):
            first_subtile_idx = (
                cutlass.Int32(subtile_cnt) - is_odd_turn
            ) % cutlass.Int32(subtile_cnt)
            if first_subtile_idx * cutlass.Int32(64) >= valid_tokens:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

        scratch_lifetime_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )

        for i in cutlass.range(subtile_cnt, unroll=1):
            if cutlass.const_expr(self._overlapping_accum):
                subtile_idx = (
                    cutlass.Int32(i + subtile_cnt) - is_odd_turn
                ) % cutlass.Int32(subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(i)

            release_after_ldtm = cutlass.const_expr(
                self._overlapping_accum
            ) and i == cutlass.Int32(0)
            if subtile_idx * cutlass.Int32(64) < valid_tokens:
                self._run_fc2_bulk_subtile(
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=self._subtile_local_tmem_tensor(
                        tmem_acc_tensor,
                        subtile_idx,
                        warp_idx,
                        acc_stage_col_offset,
                    ),
                    scratch=scratch,
                    return_prefetch=return_prefetch,
                    work_tile_info=work_tile_info,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    release_after_ldtm=release_after_ldtm,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
                # Drain each UBLK group before the fixed 8KB scratch is reused
                # by the next subtile or task tile.
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                scratch_lifetime_bar.arrive_and_wait()

        if cutlass.const_expr(not self._overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    # -- Per-subtile dispatch -------------------------------------------------

    @cute.jit
    def _run_fc1_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        *,
        preload_acc: Optional[
            Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        ] = None,
    ) -> None:
        """Run one fc1 epi subtile with contract-backed component wiring.

        ``tmem_subtile_tensor`` is a (32 lanes, 64 cols) view onto the
        per-warp acc TMEM region (or, in the overlap-acc unroll path,
        onto a *workspace* subtile's region that this subtile is
        borrowing for its in-place transpose STTMs).  Caller-built via
        ``_subtile_local_tmem_tensor``.

        ``preload_acc`` (4-tuple of (16,) fp32 RMEM tensors) is
        non-None **only** in the overlap-acc unroll path -- it carries
        the raw acc data that has already been LDTM'd out by
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` *before* the
        acc TMEM was released to the next mma.  Tuple ordering:

            preload_acc[0]: gate_lo  (subtile cols 0..31, lanes 0..15)
            preload_acc[1]: up_lo    (subtile cols 0..31, lanes 16..31)
            preload_acc[2]: raw_top  (subtile cols 32..63, lanes 0..15)
            preload_acc[3]: raw_bot  (subtile cols 32..63, lanes 16..31)

        When provided, the per-subtile entry LDTM x 2 (gate / up) is
        skipped and ``second_t`` is constructed in skip-R1.Load mode
        with raw_top / raw_bot fed through ``reg_tensor_top`` /
        ``reg_tensor_bot``; the rest of the body (transpose rounds,
        SwiGLU folds, post-half quantize, R2S, TMA cmd) is identical.

        When ``preload_acc is None`` the body matches the original
        sequential-per-subtile path bit-for-bit -- fine-grained ILP
        interleaving of transpose rounds, SwiGLU folds, and post-SwiGLU
        tasks is preserved exactly.
        """
        # -- Per-half TMEM ptrs derived from the (32, 64) subtile view ------
        #
        # ``tmem_subtile_tensor.iterator`` is the (lane 0, col 0) corner
        # of this subtile's 64-col region, already 16-aligned by the
        # ``cute.assume(divby=16)`` inside ``_subtile_local_tmem_tensor``.
        # +32 second-half offset uses a Python int (compile-time const)
        # so cute const-folds it into the ptr and propagates the 16-col
        # alignment to the LDTM/STTM atoms; a ``cutlass.Int32(32)`` here
        # would be an SSA value that cute treats as alignment-unknown,
        # tripping the LDTM atom's "tmem aligned at >= 2 cols" verifier.
        tmem_first_ptr = tmem_subtile_tensor.iterator
        tmem_second_ptr = tmem_first_ptr + 32

        # -- Per-subtile GMEM views and per-half meta -------------------------

        subtile_token_tile_idx = (
            work_tile_info.tile_n_idx * (self._cta_tile_n // EpilogueTokenTile)
            + subtile_idx
        )
        g_fc1_output_subtile_view = cute.local_tile(
            real_fc1_output,
            (EpilogueTokenTile, Fc1EpilogueOutputTile, 1),
            (subtile_token_tile_idx, work_tile_info.tile_m_idx, 0),
        )

        thread_in_warp = tidx % 32
        subtile_token_start = (
            work_tile_info.tile_n_idx * self._cta_tile_n
            + subtile_idx * EpilogueTokenTile
        )
        token_left = subtile_token_start + thread_in_warp
        token_right = token_left + 32
        intermediate_downproj_idx = (
            work_tile_info.tile_m_idx * (self._cta_tile_m // 2)
            + warp_idx * Nvfp4BlockSize
        )

        # Resolve ``intermediate_downproj`` for the PostSwigluHalf SFC
        # predicate.
        if cutlass.const_expr(self._intermediate_downproj is not None):
            intermediate_downproj_value = self._intermediate_downproj
        else:
            intermediate_downproj_value = real_fc1_output.shape[1]

        # -- Pre-LDG topk weights ---------------------------------------------
        #
        # Hoisted ahead of the LDTM x 2 below so the per-thread topk-weight
        # GMEM round trip overlaps with the entire LDTM + transpose +
        # quantize pipeline.  The PostSwigluHalf instances at the end of
        # this subtile receive these as ``preloaded_topk_weight`` instead
        # of running their own LDG inside __init__ -- saving the LDG +
        # cvt latency from the critical path.
        #
        # Each thread reads exactly 2 fp32: ``token_left`` for the
        # left-half post (covering tokens 0..31 of this subtile) and
        # ``token_right`` for the right-half post (tokens 32..63).
        # ``real_topk_scores`` is the per-expert-shifted view of the
        # global topk_scores 1D tensor (produced by
        # ``SwapABSwigluFp4Fc12SchedExtension.get_gmem_tensor("topk", ...)``).
        # Indexing it with the EXPERT-LOCAL token coord matches the SFC
        # write-side coord convention.
        topk_left = cutlass.Float32(real_topk_scores[token_left])
        topk_right = cutlass.Float32(real_topk_scores[token_right])

        # -- raw f_gate / f_up: either LDTM here or take from preload_acc ----
        #
        # Both paths produce RMEM tensors that, by definition, carry
        # ``TmemTranspose16x32.InputContract`` -- the (lane_idx, elem_idx)
        # -> (token_idx, intermediate_output_idx) distribution that
        # ``Ld16x64bOp(Repetition.x16) Float32`` LDTM produces, and that
        # ``load_subtile_raw_acc`` (using the very same atom) reproduces.
        # Wrap them as TensorWithContract so they cross the boundary into
        # SwigluCompute under the contract-backed handoff rule.
        if cutlass.const_expr(preload_acc is None):
            f_gate_tensor = cute.make_rmem_tensor((16,), cutlass.Float32)
            f_up_tensor = cute.make_rmem_tensor((16,), cutlass.Float32)
            atom_ld16x64 = cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
                cutlass.Float32,
            )
            tmem_gate_view = cute.make_tensor(
                tmem_first_ptr,
                TmemTranspose16x32._tmem_layout(16, 32),
            )
            tmem_up_view = cute.make_tensor(
                tmem_first_ptr + (16 << 16),
                TmemTranspose16x32._tmem_layout(16, 32),
            )
            cute.copy(
                atom_ld16x64,
                tmem_gate_view,
                TmemTranspose16x32._rmem_copy_view(f_gate_tensor, 16),
            )
            cute.copy(
                atom_ld16x64,
                tmem_up_view,
                TmemTranspose16x32._rmem_copy_view(f_up_tensor, 16),
            )
        else:
            f_gate_tensor = preload_acc[0]
            f_up_tensor = preload_acc[1]

        f_gate = TensorWithContract(
            tensor=f_gate_tensor,
            contract=TmemTranspose16x32.InputContract,
        )
        f_up = TensorWithContract(
            tensor=f_up_tensor,
            contract=TmemTranspose16x32.InputContract,
        )

        # -- SwiGLU fold first half (0..8 now; 8..16 dispersed below) --------

        first_swiglu = SwigluCompute(gate=f_gate, up=f_up, alpha=alpha)
        first_swiglu.fold(0, 8)

        # -- Second 32x32 in-place transpose (R4 has no STTM) ----------------
        #
        # preload_acc=None: standard path (R1.Load LDTMs from TMEM).
        # preload_acc!=None: skip-R1.Load mode -- raw_top / raw_bot have
        # already been LDTM'd out by ``load_subtile_raw_acc`` and are
        # being fed in via reg_tensor_{top,bot}.  R1.Store then writes
        # them into the borrowed workspace cols (= ``tmem_second_ptr``,
        # which under the unroll path points to the *workspace* subtile's
        # second-half cols, NOT this subtile's own second-half cols).

        if cutlass.const_expr(preload_acc is None):
            second_t = TmemTranspose32x32Inplace(tmem_second_ptr)
            second_t.bot.r1_load()
            second_t.top.r1_load()
        else:
            second_t = TmemTranspose32x32Inplace(
                tmem_second_ptr,
                reg_tensor_top=TensorWithContract(
                    tensor=preload_acc[2],
                    contract=TmemTranspose16x32.InputContract,
                ),
                reg_tensor_bot=TensorWithContract(
                    tensor=preload_acc[3],
                    contract=TmemTranspose16x32.InputContract,
                ),
            )
            # skip-R1.Load: r1_load is a no-op inside _TmemTranspose16x32Core,
            # so we don't call it here.
        second_t.bot.r1_perm()
        second_t.top.r1_perm()
        second_t.bot.r1_store()
        second_t.top.r1_store()

        second_t.bot.r2_load()
        second_t.top.r2_load()
        second_t.top.r2_store()
        second_t.bot.r2_store()

        second_t.top.r3_load_top()
        second_t.top.r3_load_bot()
        second_t.bot.r3_load_top()
        second_t.bot.r3_load_bot()
        first_swiglu.fold(8, 16)
        second_t.top.r3_perm()
        second_t.bot.r3_perm()
        second_t.top.r3_store()
        second_t.bot.r3_store()

        second_t.top.r4_load_top()
        second_t.top.r4_load_bot()
        second_t.bot.r4_load_top()
        second_t.bot.r4_load_bot()
        second_t.top.r4_perm()
        second_t.bot.r4_perm()

        # -- SwiGLU fold second half (0..8 now; 8..16 dispersed below) ------
        #
        # second_t.top.output / second_t.bot.output carry
        # ``TmemTranspose16x32.OutputContract``; SwigluCompute validates
        # gate/up contracts are equal at construction time.

        second_swiglu = SwigluCompute(
            gate=second_t.top.output,
            up=second_t.bot.output,
            alpha=alpha,
        )
        second_swiglu.fold(0, 8)

        # -- First 16x32 transpose (skip-R1.Load; R4 has no STTM) -----------
        #
        # ``first_swiglu.output.contract == TmemTranspose16x32.InputContract``
        # (SwigluCompute inherits from f_gate's contract, which IS
        # ``InputContract``).  TmemTranspose16x32.__init__ validates the
        # reg_tensor contract against InputContract.

        first_t = TmemTranspose16x32(
            tmem_first_ptr,
            Region.Top,
            reg_tensor=first_swiglu.output,
        )
        first_t.r1_perm()
        first_t.r1_store()
        first_t.r2_load()
        first_t.r2_store()
        first_t.r3_load_top()
        first_t.r3_load_bot()
        second_swiglu.fold(8, 16)
        first_t.r3_perm()
        first_t.r3_store()
        first_t.r4_load_top()
        first_t.r4_load_bot()

        # gen_sf + quantize for the right half (second_swiglu.output is
        # ready; first_t.r4_perm hasn't run yet, so left half waits below).
        # The per-thread topk-weight LDG was hoisted to the subtile prologue
        # (``topk_right`` is now a ready fp32 register); PostSwigluHalf
        # consumes it via ``preloaded_topk_weight`` and pre-multiplies it
        # into the swiglu fp32 values BEFORE NVFP4 quantize (Path A).
        # ``token_right`` is the EXPERT-LOCAL token coord (same coord
        # that indexes ``real_fc1_output_sf`` for the SFC write).
        post_right = PostSwigluHalf(
            second_swiglu.output,
            sC=smem_fc1_output_buffer,
            gSFC=real_fc1_output_sf,
            warp_idx=warp_idx,
            norm_const=norm_const,
            sf_vec_size=Nvfp4BlockSize,
            half_idx=1,
            token_idx=token_right,
            thread_in_warp=thread_in_warp,
            preloaded_topk_weight=topk_right,
            intermediate_downproj_idx=intermediate_downproj_idx,
            intermediate_downproj=intermediate_downproj_value,
            cga_cluster_tile_intermediate_downproj=(
                self._cga_cluster_tile_intermediate_downproj
            ),
        )

        first_t.r4_perm()

        # -- Quant + R2S + STG SFC, right then left half ---------------------
        #
        # No per-subtile-entry barrier wait: each subtile owns its own bar
        # id (``subtile_bar_id``), so producer warps can fire-and-forget
        # arrive on the previous subtile's bar and immediately start the
        # next subtile's R2S without throttle.  Phase correctness is
        # guaranteed by single-arrive-per-warp-per-bar within a task tile.

        post_right.stg_sfc()

        post_left = PostSwigluHalf(
            first_t.output,
            sC=smem_fc1_output_buffer,
            gSFC=real_fc1_output_sf,
            warp_idx=warp_idx,
            norm_const=norm_const,
            sf_vec_size=Nvfp4BlockSize,
            half_idx=0,
            token_idx=token_left,
            thread_in_warp=thread_in_warp,
            preloaded_topk_weight=topk_left,
            intermediate_downproj_idx=intermediate_downproj_idx,
            intermediate_downproj=intermediate_downproj_value,
            cga_cluster_tile_intermediate_downproj=(
                self._cga_cluster_tile_intermediate_downproj
            ),
        )
        post_left.stg_sfc()

        post_right.r2s(subtile_idx)
        post_left.r2s(subtile_idx)

        # -- TMA store C cmd issue (per-subtile; task-tile commit lives in
        #    the run() body, not here) -------------------------------------

        SwapABSwigluFp4Epilogue.tma_store_fc1_output(
            warp_idx,
            smem_fc1_output_buffer,
            subtile_idx,
            tma_atom_fc1_output,
            g_fc1_output_subtile_view,
        )

    @cute.jit
    def _load_fc2_bulk_raw_acc(
        self,
        tmem_subtile_tensor: cute.Tensor,
    ) -> cute.Tensor:
        """LDTM one 64-token fc2 subtile as raw fp32 accumulators.

        Swap-AB makes the TMEM data path row the hidden coordinate and the
        TMEM column the token coordinate.  ``Ld32x32b.x64`` therefore gives
        each lane one hidden row and 64 consecutive token registers.
        """
        raw_regs = cute.make_rmem_tensor((64,), cutlass.Float32)
        atom_ld32x32_x64 = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
            cutlass.Float32,
        )
        cute.copy(
            atom_ld32x32_x64,
            tmem_subtile_tensor,
            _TmemTranspose16x32Core._rmem_copy_view(raw_regs, 64),
        )
        return raw_regs

    @cute.jit
    def _store_fc2_bulk_token_group_to_smem(
        self,
        raw_regs: cute.Tensor,
        scratch: cute.Tensor,
        *,
        token_group_idx: int,
        warp_idx: int,
        tidx,
    ) -> None:
        """F2FP + warp-collective STS for one 32-token group.

        For token iteration t, lanes 0..31 of each epi warp collectively
        store hidden-contiguous BF16 values into ``scratch[t, hidden]``.
        """
        lane_idx = tidx % 32
        warp_hidden_base = cutlass.Int32(warp_idx * 32)

        for token_i in cutlass.range_constexpr(32):
            src_reg = token_i + 32 * token_group_idx
            scratch[token_i, warp_hidden_base + lane_idx] = raw_regs[src_reg].to(
                self._fc2_output_dtype
            )

    @cute.jit
    def _issue_fc2_bulk_token_group(
        self,
        scratch: cute.Tensor,
        return_prefetch: Fc2ReturnTilePrefetch,
        *,
        subtile_idx,
        token_group_idx: int,
        tile_hidden_idx,
        hidden_tile_size: int,
        valid_hidden,
        warp_idx,
        tidx,
    ) -> None:
        """Issue one 32-token x 128-hidden scratch tile via UBLKCP/UBLKREDG."""
        pair_idx = subtile_idx // cutlass.Int32(2)
        issue_group = (subtile_idx % cutlass.Int32(2)) * cutlass.Int32(
            2
        ) + cutlass.Int32(token_group_idx)
        lane_idx = tidx % 32
        lane_group = lane_idx // cutlass.Int32(8)
        lane_in_group = lane_idx % cutlass.Int32(8)
        scratch_row = cutlass.Int32(warp_idx * 8) + lane_in_group
        slot = pair_idx
        hidden_base = cutlass.Int32(tile_hidden_idx * hidden_tile_size)
        # Metadata is prefetched once per CTA tile: warp W owns tokens
        # [8*W, 8*W+8) within each active 32-token group.  All epi warps
        # issue their own 8 row-bulk operations for the active lane group.
        if lane_group == issue_group:
            if cutlass.const_expr(self._fc2_hidden_needs_predicate):
                if hidden_base < valid_hidden:
                    copy_elems = cutlass.Int32(128)
                    if hidden_base + cutlass.Int32(128) > valid_hidden:
                        copy_elems = valid_hidden - hidden_base
                    self._issue_fc2_bulk_row(
                        scratch,
                        return_prefetch,
                        slot=slot,
                        scratch_row=scratch_row,
                        hidden_base=hidden_base,
                        copy_bytes=copy_elems * cutlass.Int32(2),
                    )
            else:
                self._issue_fc2_bulk_row(
                    scratch,
                    return_prefetch,
                    slot=slot,
                    scratch_row=scratch_row,
                    hidden_base=hidden_base,
                    copy_bytes=cutlass.Int32(256),
                )

    @cute.jit
    def _issue_fc2_bulk_row(
        self,
        scratch: cute.Tensor,
        return_prefetch: Fc2ReturnTilePrefetch,
        *,
        slot,
        scratch_row,
        hidden_base,
        copy_bytes,
    ) -> None:
        src_row = cute.slice_(scratch, (scratch_row, None))
        token_row_1d = return_prefetch.row(slot)
        dst_ptr = token_row_1d.iterator + hidden_base
        dst_ptr = cute.make_ptr(
            token_row_1d.element_type,
            dst_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=16,
        )
        if return_prefetch.is_valid(slot):
            if cutlass.const_expr(self._reduce_topk_in_kernel):
                _cp_reduce_async_bulk_add_noftz_bf16_s2g(
                    dst_ptr,
                    src_row.iterator,
                    copy_bytes,
                )
            else:
                _cp_async_bulk_s2g(
                    dst_ptr,
                    src_row.iterator,
                    copy_bytes,
                )

    @cute.jit
    def _run_fc2_bulk_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        scratch: cute.Tensor,
        return_prefetch: Fc2ReturnTilePrefetch,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        *,
        release_after_ldtm,
        acc_pipeline,
        acc_consumer_state,
    ) -> None:
        """Run one fc2 UBLK subtile.

        Both 32-token groups are LDTM'd before any SMEM reuse, so overlap
        mode can release accumulator TMEM immediately after this prologue.
        The two groups then reuse the same 32x128 BF16 scratch tile.
        """
        raw_acc = self._load_fc2_bulk_raw_acc(tmem_subtile_tensor)

        if release_after_ldtm:
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

        task_tile_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )

        # token 0..31 of this 64-token subtile
        self._store_fc2_bulk_token_group_to_smem(
            raw_acc,
            scratch,
            token_group_idx=0,
            warp_idx=warp_idx,
            tidx=tidx,
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        task_tile_bar.arrive_and_wait()
        self._issue_fc2_bulk_token_group(
            scratch,
            return_prefetch,
            subtile_idx=subtile_idx,
            token_group_idx=0,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            valid_hidden=valid_hidden,
            warp_idx=warp_idx,
            tidx=tidx,
        )
        cute.arch.cp_async_bulk_commit_group()

        # token 32..63 of this 64-token subtile
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.sync_warp()
        task_tile_bar.arrive_and_wait()
        self._store_fc2_bulk_token_group_to_smem(
            raw_acc,
            scratch,
            token_group_idx=1,
            warp_idx=warp_idx,
            tidx=tidx,
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        task_tile_bar.arrive_and_wait()
        self._issue_fc2_bulk_token_group(
            scratch,
            return_prefetch,
            subtile_idx=subtile_idx,
            token_group_idx=1,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            valid_hidden=valid_hidden,
            warp_idx=warp_idx,
            tidx=tidx,
        )
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _run_fc2_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        return_prefetch: Fc2ReturnTilePrefetch,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        *,
        preload_acc: Optional[
            Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        ] = None,
    ) -> None:
        """Run one fc2 epi subtile: LDTM + cvt + pack + transpose + STG.

        Per fc2 epi subtile (64 token x cta_tile_m hidden):

          - 4 epi warp share the subtile.  Per warp = 32 hidden x 64 token.
          - Split into first/second halves (32 token left / 32 token right);
            each half is 32 hidden x 32 token (= one warp-region of acc TMEM).

        Per-half pipeline:

          1. ``Fc2AccLoadAndPack``           : LDTM x 2 + cvt.rn.bf16x2.f32 +
                                               pack pairs into (16, Float32)
                                               packed-bf16x2 RMEM tensor
                                               carrying TmemTranspose16x32-
                                               Packed.InputContract.
          2. ``TmemTranspose16x32Packed``    : 4-round in-place transpose
                                               (skip-R1.Load mode -- the
                                               packed tensor is fed directly
                                               via reg_tensor=).  Output
                                               carries TmemTranspose16x32-
                                               Packed.OutputContract.
          3. ``Fc2UnpackPermuteStg``         : recast packed view back to
                                               (32, BFloat16), permute to
                                               natural hidden order, fire
                                               2 x STG.256 to GMEM out2.

        first half (token 0..31) and second half (token 32..63) are
        sequential; we let the compiler interleave LDTMs / STTMs / STGs
        across the two halves for ILP rather than driving an explicit
        cross-half action ordering (fc2 has no SwiGLU coupling between
        halves, so a simpler structure is fine here).

        ``tmem_subtile_tensor`` (32 hidden lanes, 64 token cols): per-warp
        acc TMEM region of this fc2 subtile (or, in the overlap-acc unroll
        path, a workspace subtile's region this subtile is borrowing).
        Caller-built via ``_subtile_local_tmem_tensor``.

        ``preload_acc`` (4-tuple of (16,) fp32 RMEM tensors): non-None
        only in the overlap-acc unroll path.  Carries the raw acc data
        of this subtile already LDTM'd out by
        ``_TmemTranspose16x32Core.load_subtile_raw_acc`` *before* the
        acc TMEM was released.  Tuple ordering (matches fc1):

            preload_acc[0]: first-half top  (cols 0..31, hidden lanes 0..15)
            preload_acc[1]: first-half bot  (cols 0..31, hidden lanes 16..31)
            preload_acc[2]: second-half top (cols 32..63, hidden lanes 0..15)
            preload_acc[3]: second-half bot (cols 32..63, hidden lanes 16..31)

        When provided, ``Fc2AccLoadAndPack`` for each half is constructed
        with ``preload_acc=(top, bot)`` instead of ``tmem_ptr=...``,
        skipping the per-half LDTM x 2.  The downstream
        ``TmemTranspose16x32Packed`` (already skip-R1.Load) and
        ``Fc2UnpackPermuteStg`` are unchanged.
        """
        # -- Per-half TMEM ptrs derived from the (32, 64) subtile view ------
        # Same Python-int alignment propagation convention as fc1.
        tmem_first_ptr = tmem_subtile_tensor.iterator
        tmem_second_ptr = tmem_first_ptr + 32

        thread_in_warp = tidx % 32
        epi_tidx = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp

        # ``redg_subtile_scratch``: the same (32 lanes, 64 cols) view
        # the caller already built for the upstream transpose
        # (= ``tmem_subtile_tensor`` from ``_subtile_local_tmem_tensor``).
        # In REDG mode ``Fc2UnpackPermuteStg`` carves a per-half
        # 32-lane x 16-col STTM+LDTM reshuffle slab out of it at col
        # offset ``32 * half_idx``, matching the per-half tmem_*_ptr
        # split used by the transpose.  Race-free reasoning:
        #   * tile_n=256 (overlap-acc unroll path): in-place transpose
        #     runs on the NEXT subtile's TMEM region, so the per-half
        #     slab here belongs to that next-subtile region and the
        #     next acc mma does not touch it until after the epilogue
        #     body returns.
        #   * tile_n=128: transpose runs on the CURRENT subtile's TMEM
        #     region, but acc_pipeline.consumer_release is hoisted to
        #     AFTER the full per-subtile loop (see
        #     ``_run_fc2_task_tile``), so the next mma is gated by the
        #     same task-tile-boundary fence the REDG STTM has already
        #     cleared by the time release fires.
        if cutlass.const_expr(return_prefetch.tile.reduce_topk_in_kernel):
            redg_subtile_scratch = tmem_subtile_tensor
        else:
            redg_subtile_scratch = None

        # -- First half (token 0..31): LDTM + pack + transpose + STG -------
        if cutlass.const_expr(preload_acc is None):
            first_pack = Fc2AccLoadAndPack(
                tmem_first_ptr,
                fc2_output_dtype=self._fc2_output_dtype,
            )
        else:
            first_pack = Fc2AccLoadAndPack(
                preload_acc=(preload_acc[0], preload_acc[1]),
                fc2_output_dtype=self._fc2_output_dtype,
            )
        first_t = TmemTranspose16x32Packed(
            tmem_first_ptr,
            Region.Top,
            reg_tensor=first_pack.output,
        )
        first_t.r1_perm()
        first_t.r1_store()
        first_t.r2_load()
        first_t.r2_store()
        first_t.r3_load_top()
        first_t.r3_load_bot()
        first_t.r3_perm()
        first_t.r3_store()
        first_t.r4_load_top()
        first_t.r4_load_bot()
        first_t.r4_perm()

        first_stg = Fc2UnpackPermuteStg(
            first_t.output,
            return_prefetch=return_prefetch,
            fc2_output_dtype=self._fc2_output_dtype,
            token_iter=subtile_idx * cutlass.Int32(8)
            if cutlass.const_expr(return_prefetch.tile.reduce_topk_in_kernel)
            else subtile_idx * cutlass.Int32(2),
            warp_idx=warp_idx,
            epi_tidx=epi_tidx,
            valid_hidden=valid_hidden,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            needs_hidden_predicate=self._fc2_hidden_needs_predicate,
            tmem_subtile_scratch=redg_subtile_scratch,
        )
        first_stg.stg()

        # -- Second half (token 32..63): same pipeline -----------------------
        if cutlass.const_expr(preload_acc is None):
            second_pack = Fc2AccLoadAndPack(
                tmem_second_ptr,
                fc2_output_dtype=self._fc2_output_dtype,
            )
        else:
            second_pack = Fc2AccLoadAndPack(
                preload_acc=(preload_acc[2], preload_acc[3]),
                fc2_output_dtype=self._fc2_output_dtype,
            )
        second_t = TmemTranspose16x32Packed(
            tmem_second_ptr,
            Region.Top,
            reg_tensor=second_pack.output,
        )
        second_t.r1_perm()
        second_t.r1_store()
        second_t.r2_load()
        second_t.r2_store()
        second_t.r3_load_top()
        second_t.r3_load_bot()
        second_t.r3_perm()
        second_t.r3_store()
        second_t.r4_load_top()
        second_t.r4_load_bot()
        second_t.r4_perm()

        second_stg = Fc2UnpackPermuteStg(
            second_t.output,
            return_prefetch=return_prefetch,
            fc2_output_dtype=self._fc2_output_dtype,
            token_iter=(
                subtile_idx * cutlass.Int32(8) + cutlass.Int32(4)
                if cutlass.const_expr(return_prefetch.tile.reduce_topk_in_kernel)
                else subtile_idx * cutlass.Int32(2) + cutlass.Int32(1)
            ),
            warp_idx=warp_idx,
            epi_tidx=epi_tidx,
            valid_hidden=valid_hidden,
            tile_hidden_idx=work_tile_info.tile_m_idx,
            hidden_tile_size=self._cta_tile_m,
            needs_hidden_predicate=self._fc2_hidden_needs_predicate,
            tmem_subtile_scratch=redg_subtile_scratch,
        )
        second_stg.stg()
