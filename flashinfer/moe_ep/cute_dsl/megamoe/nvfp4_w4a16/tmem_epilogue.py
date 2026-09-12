# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 epilogue data movement derived from the swapped MegaMoE implementation.

Only the 32-bit TMEM transpose, BF16 alpha/cast/reorder, and local-or-peer STG
path are retained. There is no activation quantization or FC1/FC2 core base.
"""

import dataclasses
from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int64
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import AddressSpace

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    Contract,
    FunctionMapping,
    Space,
    eval_function_mapping,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import TokenSrcMetadata
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import SymBufferDeviceBase


class TmemTranspose16x32:
    """16x32 -> 32x16 TMEM in-place transpose from register inputs.

    The per-thread RMEM ``(lane_idx, elem_idx) -> (tmem_dp, tmem_col)`` mapping
    is fixed by the underlying atom sequence and is identical for fc1 (each slot
    is an fp32 swiglu-fold value, ``tmem_col`` = intermediate-output index) and
    fc2 (each slot is a packed bf16x2, ``tmem_col`` = hidden-pair index).  Only
    the ``tmem_col`` semantic name differs between the two uses; the physical
    distribution below is the single source of truth.

    Input distribution -- what each (lane_idx, elem_idx) reg holds on entry
    (fed through ``reg_tensor`` after LDTM or FC1 activation):

        tmem_dp  = elem_idx * 2 + (lane_idx // 2) % 2          # in [0, 32)
        tmem_col = (lane_idx % 2) * 8 + lane_idx // 4          # in [0, 16)

    Output distribution -- after all four rounds, the 32-dp x 16-col result has
    each lane owning one full dp-row of 16 cols:

        tmem_dp  = lane_idx                                    # in [0, 32)
        tmem_col = elem_idx                                    # in [0, 16)
    """

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((TmemTranspose16x32._TmemRowStride, 1), 0),),
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
        """Load four (16,) FP32 fragments from one 32-lane x 64-column subtile.

        Returns first-half top/bottom, then second-half top/bottom in the
        input register distribution documented on this class.
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
            TmemTranspose16x32._io_dtype,
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * TmemTranspose16x32._TmemRowStride

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
            TmemTranspose16x32._tmem_layout(16, 32),
        )
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off,
            TmemTranspose16x32._tmem_layout(16, 32),
        )
        second_top_view = cute.make_tensor(
            ptr + 32,
            TmemTranspose16x32._tmem_layout(16, 32),
        )
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off,
            TmemTranspose16x32._tmem_layout(16, 32),
        )

        first_top = cute.make_rmem_tensor((16,), TmemTranspose16x32._io_dtype)
        first_bot = cute.make_rmem_tensor((16,), TmemTranspose16x32._io_dtype)
        second_top = cute.make_rmem_tensor((16,), TmemTranspose16x32._io_dtype)
        second_bot = cute.make_rmem_tensor((16,), TmemTranspose16x32._io_dtype)

        cute.copy(
            atom_ld16x64,
            first_top_view,
            TmemTranspose16x32._rmem_copy_view(first_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            first_bot_view,
            TmemTranspose16x32._rmem_copy_view(first_bot, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_top_view,
            TmemTranspose16x32._rmem_copy_view(second_top, 16),
        )
        cute.copy(
            atom_ld16x64,
            second_bot_view,
            TmemTranspose16x32._rmem_copy_view(second_bot, 16),
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(
        self,
        tmem_ptr,
        reg_tensor: cute.Tensor,
    ) -> None:
        half_lane_off = 16 * self._TmemRowStride
        self._tmem_src_full = cute.make_tensor(tmem_ptr, self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(tmem_ptr, self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(tmem_ptr, self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(
            tmem_ptr + half_lane_off, self._tmem_layout(16, 16)
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
        # ``output`` is a bare (16,) RMEM fragment; its (lane_idx, elem_idx)
        # distribution after all four rounds is the transpose output mapping
        # documented on ``TmemTranspose16x32``.
        self.output = cute.make_rmem_tensor((16,), self._io_dtype)

        # FP32 values and packed BF16 pairs both occupy sixteen 32-bit registers.
        if cutlass.const_expr(reg_tensor.element_type.width != 32):
            raise TypeError(
                f"{type(self).__name__} reg_tensor must be a 32-bit element "
                f"type (fp32 or packed bf16x2), got element type "
                f"{reg_tensor.element_type} (width {reg_tensor.element_type.width})."
            )
        if cutlass.const_expr(cute.size(reg_tensor) != 16):
            raise ValueError(
                f"{type(self).__name__} reg_tensor must hold exactly 16 "
                f"elements, got {cute.size(reg_tensor)}."
            )
        for r in range(16):
            self._src_regs[r] = reg_tensor[r]

    def r1_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(
            self._atom_st16x128,
            self._rmem_copy_view(self.output, 16),
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
            self.output[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(
            self._atom_st32x32,
            self._rmem_copy_view(self.output, 16),
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
            self.output[r] = self._src_regs[self._PermR4[r]]

    def from_r1_perm_until_last_store(self) -> cute.Tensor:
        self.r1_perm()
        self.r1_store()
        self.r2_load()
        self.r2_store()
        self.r3_load_top()
        self.r3_load_bot()
        self.r3_perm()
        self.r3_store()
        self.r4_load_top()
        self.r4_load_bot()
        self.r4_perm()
        return self.output


@dataclasses.dataclass(frozen=True)
class W4A16EpiArgs:
    # Per-expert FP32 scales.
    fc1_alpha: cute.Tensor
    fc2_alpha: cute.Tensor


class EpilogueContext:
    """Immutable loop-invariant context; dynamic values stay explicit arguments."""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen_", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after __init__ "
                f"(cannot set {name!r})."
            )
        object.__setattr__(self, name, value)

    def _freeze(self):
        object.__setattr__(self, "_frozen_", True)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        assert len(values) == 0
        return self


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    """BF16 STG destinations, either local pool rows or per-token peer metadata."""

    metadata: Optional[cute.Tensor]
    token_bases: Optional[cutlass.Int32]
    base_outputs: cute.Tensor
    hidden_base_this_cta_tile: Union[cutlass.Int32, int]
    peer_rank_ptr_mapper: Optional[SymBufferDeviceBase]
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden_this_cta_tile: Union[cutlass.Int32, int]
    output_mappings: Contract
    epi_tid: cutlass.Int32
    dst_ptrs: Optional[cute.Tensor] = None
    valid: Optional[cute.Tensor] = None

    def __post_init__(self):
        if (self.metadata is None) == (self.token_bases is None):
            raise ValueError(
                "Fc2OutputRouter requires exactly one of metadata or a token base."
            )
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputRouter requires peer_rank_ptr_mapper iff metadata is set."
            )

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        # Only the metadata (comm) path prefetches a pointer array: its
        # metadata-derived address has long-latency LDGs worth issuing early.
        # The local (no-comm) path computes its affine address on demand in
        # get_dst() -- no array, hence no runtime-indexed local-memory spill.
        if cutlass.const_expr(self.metadata is None):
            return self
        iter_axis = self.output_mappings.domain.names.index("iter_idx")
        copy_iters: cutlass.Constexpr[int] = self.output_mappings.domain.sizes[
            iter_axis
        ]

        # Adjacent 16-BF16 stores share a token and one aligned 32-hidden
        # region. H%32 makes their hidden validity identical, including tails.
        valid = cute.make_rmem_tensor((copy_iters // 2,), cutlass.Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters // 2,), cutlass.Int64)

        for pair_idx in cutlass.range_constexpr(copy_iters // 2):
            coord = eval_function_mapping(
                self.output_mappings,
                epi_tid=self.epi_tid,
                iter_idx=2 * pair_idx,
            )
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])

            valid[pair_idx] = cutlass.Int32(0)
            dst_ptrs[pair_idx] = cutlass.Int64(0)

            token_valid = token_in_tile < self.valid_tokens_this_cta_tile
            hidden_valid = hidden_in_tile < cutlass.Int32(
                self.valid_hidden_this_cta_tile
            )
            if token_valid and hidden_valid:
                valid[pair_idx] = cutlass.Int32(1)
                md = TokenSrcMetadata.load(
                    self.metadata.iterator.toint()
                    + Int64(token_in_tile) * Int64(TokenSrcMetadata.nbytes)
                )
                dst_rank = md.src_rank
                dst_token = md.src_token
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                dst_topk = md.src_topk
                # Int64 token coord: domain_offset on (token, topk, hidden)
                # computes dst_token*K*H, which overflows int32 once T*K*H > 2^31.
                dst_ptrs[pair_idx] = self.peer_rank_ptr_mapper.ptr_map_to_rank(
                    cute.domain_offset(
                        (Int64(dst_token), dst_topk, dst_hidden), self.base_outputs
                    ).iterator,
                    dst_rank,
                    byte_align=32,
                ).toint()

        return dataclasses.replace(
            self,
            dst_ptrs=dst_ptrs,
            valid=valid,
        )

    @cute.jit
    def get_data_dst(
        self,
        iter_idx: Union[int, cutlass.Int32],
    ) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue DATA destination: gmem pointer + validity predicate.

        The router owns ``data_output`` and supplies a 32-byte-aligned pointer
        for the caller's 256-bit vector store.
        """
        if cutlass.const_expr(self.metadata is None):
            # no-comm: on-demand affine address (no prefetched array). The
            # invariant base hoists out of the caller's loop via CSE; a
            # constexpr iter folds the per-issue offset into the store.
            coord = eval_function_mapping(
                self.output_mappings,
                epi_tid=self.epi_tid,
                iter_idx=iter_idx,
            )
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            pred = cutlass.Int32(0)
            addr = cutlass.Int64(0)
            if (
                token_in_tile < self.valid_tokens_this_cta_tile
                and hidden_in_tile < cutlass.Int32(self.valid_hidden_this_cta_tile)
            ):
                pred = cutlass.Int32(1)
                dst_tokens = self.token_bases + token_in_tile
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                # Int64 token coord: dst_tokens*K*H overflows int32 once T*K*H > 2^31.
                addr = self.base_outputs[
                    Int64(dst_tokens), None, dst_hidden
                ].iterator.toint()
        else:
            # Preserve the hidden stride; compact BF16 rows fold this to 32B.
            addr = self.dst_ptrs[iter_idx // 2] + (
                Int64(iter_idx % 2) * Int64(self.base_outputs.stride[2]) * Int64(32)
            )
            pred = self.valid[iter_idx // 2]
        ptr = cute.make_ptr(
            self.base_outputs.element_type,
            addr,
            AddressSpace.gmem,
            assumed_align=32,
        )
        return ptr, pred


@cute.jit
def fc2_f2fp(
    *tensors,
    alpha_val: cutlass.Float32,
) -> cute.Tensor:
    reorder_dtype = cutlass.BFloat16
    total_size = 0
    for t in tensors:
        total_size += cute.size(t)
    converted_acc = cute.make_rmem_tensor((total_size,), reorder_dtype)
    elems_processed = 0
    for t in tensors:
        current_tensor_size = cute.size(t)
        dst = cute.make_tensor(
            converted_acc.iterator + elems_processed,
            cute.make_layout((current_tensor_size,)),
        )
        if cutlass.const_expr(current_tensor_size % 2 != 0):
            raise ValueError("fc2_f2fp expects even elements for each input tensor.")
        scaled = cute.make_rmem_tensor((current_tensor_size,), cutlass.Float32)
        for i in cutlass.range_constexpr(0, current_tensor_size, 2):
            # scaled[i] = t[i] * alpha_val
            s0, s1 = cute.arch.mul_packed_f32x2(
                (t[i], t[i + 1]), (alpha_val, alpha_val)
            )
            scaled[i] = s0
            scaled[i + 1] = s1
        dst.store(scaled.load().to(reorder_dtype))
        elems_processed += current_tensor_size
    return converted_acc


@cute.jit
def fc2_stg_post_f2fp_reorder(
    *,
    casted: cute.Tensor,  # (subtile_cnt,)
    tmem_subtile_view: cute.Tensor,  # (epi_tile_m, epi_tile_n)
):
    if cutlass.const_expr(cute.size(casted) != 64):
        raise NotImplementedError(
            "fc2 stg pass expects 64 fp32 regs in total before store reorder."
        )

    # casted (flat 64) = [h0_top, h0_bot, h1_top, h1_bot], 16 bf16 each.
    #
    # gather: interleave (top[i], bot[i]) -> bf16x2 slot i, both halves at once.
    #   read casted through (t, hidden, half) -> casted[t*16 + hidden + half*32]
    #   in (t fastest) order -> [top0,bot0,top1,bot1,...] per half = packed bf16x2.
    # scatter: de-interleave the transposed natural-hidden regs back to the
    #   STG pre-store order (token = lane + 32*(vid//32), hidden = vid % 32).
    gather_top_bot_map = ((2, 16, 2), (16, 1, 32))
    scatter_top_bot_map = ((16, 2, 2), (2, 1, 32))
    dtype = cutlass.BFloat16

    packed = cute.make_rmem_tensor((64,), dtype)
    cute.autovec_copy(
        cute.composition(
            casted,
            cute.make_layout((gather_top_bot_map[0],), stride=(gather_top_bot_map[1],)),
        ),
        packed,
    )
    packed_i32 = cute.recast_tensor(packed, cutlass.Float32)  # (32,): 16 i32 per half

    # Reuse the 32-bit transpose: each i32 slot carries one packed bf16x2 pair.
    token_0_32_pre_scatter_back = TmemTranspose16x32(
        tmem_subtile_view.iterator,
        reg_tensor=cute.composition(packed_i32, (16,)),
    ).from_r1_perm_until_last_store()
    token_32_64_pre_scatter_back = TmemTranspose16x32(
        tmem_subtile_view.iterator + 32,
        reg_tensor=cute.composition(cute.domain_offset(16, packed_i32), (16,)),
    ).from_r1_perm_until_last_store()
    cute.autovec_copy(
        token_0_32_pre_scatter_back, cute.zipped_divide(packed_i32, (16,))[None, 0]
    )
    cute.autovec_copy(
        token_32_64_pre_scatter_back, cute.zipped_divide(packed_i32, (16,))[None, 1]
    )
    out = cute.make_rmem_tensor((64,), dtype)
    cute.autovec_copy(
        cute.composition(
            packed,
            cute.make_layout(
                (scatter_top_bot_map[0],), stride=(scatter_top_bot_map[1],)
            ),
        ),
        out,
    )
    return out


@cute.jit
def fc2_stg_store_function(
    *,
    subtile: cute.Tensor,  # BF16 fragment in store order.
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
):
    data_subtile = subtile
    stg_width_elems: cutlass.Constexpr[int] = min(
        32, 256 // data_subtile.element_type.width
    )
    stg_bits: cutlass.Constexpr[int] = stg_width_elems * data_subtile.element_type.width
    copy_atom_vec = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Int32,
        num_bits_per_copy=stg_bits,
    )
    elems_per_thread: cutlass.Constexpr[int] = cute.size(data_subtile)
    if cutlass.const_expr(elems_per_thread % stg_width_elems != 0):
        raise ValueError(
            "fc2 STG store requires pre-store elems per thread to be divisible "
            f"by STG issue width, got {elems_per_thread} and {stg_width_elems}."
        )

    iters_per_subtile: cutlass.Constexpr[int] = elems_per_thread // stg_width_elems
    copy_src = cute.zipped_divide(data_subtile, (stg_width_elems,))
    single_copy_layout = cute.make_layout(((stg_width_elems, 1),), stride=((1, 0),))
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(iters_per_subtile)
    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            src_i = cute.make_tensor(
                copy_src[None, local_iter].iterator, single_copy_layout
            )
            dst_i = cute.make_tensor(dst_ptr, single_copy_layout)
            cute.copy(
                copy_atom_vec,
                cute.recast_tensor(src_i, cutlass.Int32),
                cute.recast_tensor(dst_i, cutlass.Int32),
            )


def make_bf16_fc2_store_mapping(*, cta_token_tile_size, cta_hidden_tile_size):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    elems_per_stg = 16
    stgs_per_hidden32 = 2
    return Contract(
        domain=Space(
            ("epi_tid", "iter_idx"),
            (128, stgs_per_hidden32 * cta_token_tile_size // 32),
        ),
        codomain=Space(
            ("token_in_cta_tile", "hidden_in_cta_tile"),
            (cta_token_tile_size, cta_hidden_tile_size),
        ),
        mapping=FunctionMapping(
            lambda epi_tid, iter_idx: {
                "token_in_cta_tile": epi_tid % 32 + iter_idx // stgs_per_hidden32 * 32,
                "hidden_in_cta_tile": (iter_idx % stgs_per_hidden32) * elems_per_stg
                + epi_tid // 32 * 32,
            }
        ),
    )
