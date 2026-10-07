# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Sched extension for the fused fc1+fc2 GLU MXFP8 (non-swapAB) kernel."""

from typing import List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from cutlass._mlir import ir

from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF
from moe_nvfp4_swapab.custom_ext import (
    PeekReadyBit,
    PhaseBits,
    PhaseMask,
    SwapABSwigluFp4Fc12SchedExtension,
    SwapABSwigluFp4Fc12WorkTileInfo,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from moe_nvfp4_swapab.moe_utils import rewrite_tensor_shape, spin_wait
from moe_nvfp4_swapab.moe_persistent_scheduler import MoEWorkTileInfo


# =============================================================================
# MXFP8 WorkTileInfo (non-swapAB)
# =============================================================================


class GluMxFp8WorkTileInfo(MoEWorkTileInfo):
    """WorkTileInfo for the MXFP8 non-swapAB kernel.

    Inherits directly from ``MoEWorkTileInfo`` (not from the swap-AB subclass)
    because the MXFP8 field layout is distinct: slot 6 holds
    ``valid_tokens_in_cta_cluster_tile`` — a packed Int32 (high 16 bits =
    per-CTA tile count, low 16 bits = cluster-level count).

    Class-level constant ``_cluster_m`` must be set by
    ``GluMxFp8Fc12SchedExtension.__init__`` before JIT compilation.
    """

    TotalFields = 8
    _cluster_m: int = 1

    def __init__(
        self,
        expert_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        cumulative_data_physical_row: Int32,
        cumulative_sf_physical_row: Int32,
        cumulative_token_block_count: Int32,
        # Packed token counts: high 16 bits = per-CTA valid_tokens_in_cta_tile,
        #                      low  16 bits = valid_tokens_in_cluster_tile.
        valid_tokens_in_cta_cluster_tile: Int32,
        phase_and_peek: Int32,
        # Transient scheduler field — carried through MLIR but NOT written to SMEM.
        # fc1_counter_index: cluster_token_block_idx (intra-expert cluster token-block index).
        fc1_counter_index: Int32,
    ):
        super().__init__(expert_idx, tile_m_idx, tile_n_idx, cumulative_data_physical_row)
        self.cumulative_data_physical_row = self.k_tile_cnt
        self.cumulative_sf_physical_row = cumulative_sf_physical_row
        self.cumulative_token_block_count = cumulative_token_block_count
        self.valid_tokens_in_cta_cluster_tile = valid_tokens_in_cta_cluster_tile
        self.phase_and_peek = phase_and_peek
        self.fc1_counter_index = fc1_counter_index

    @property
    def phase(self) -> Int32:
        return self.phase_and_peek & Int32(PhaseMask)

    @property
    def peek_ready(self):
        return ((self.phase_and_peek >> Int32(PhaseBits)) & Int32(1)) != Int32(0)

    @property
    def valid_tokens_in_cta_tile(self) -> Int32:
        return self.valid_tokens_in_cta_cluster_tile >> Int32(16)

    @property
    def valid_tokens_in_cluster_tile(self) -> Int32:
        return self.valid_tokens_in_cta_cluster_tile & Int32(0xFFFF)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = super().__extract_mlir_values__()                                   # [0..3]
        values.extend(extract_mlir_values(self.cumulative_sf_physical_row))          # [4]
        values.extend(extract_mlir_values(self.cumulative_token_block_count))        # [5]
        values.extend(extract_mlir_values(self.valid_tokens_in_cta_cluster_tile))    # [6]
        values.extend(extract_mlir_values(self.phase_and_peek))                      # [7]
        values.extend(extract_mlir_values(self.fc1_counter_index))                   # [8]
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GluMxFp8WorkTileInfo":
        assert len(values) == 9
        return type(self)(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            cumulative_data_physical_row=new_from_mlir_values(
                self.cumulative_data_physical_row, [values[3]]
            ),
            cumulative_sf_physical_row=new_from_mlir_values(
                self.cumulative_sf_physical_row, [values[4]]
            ),
            cumulative_token_block_count=new_from_mlir_values(
                self.cumulative_token_block_count, [values[5]]
            ),
            valid_tokens_in_cta_cluster_tile=new_from_mlir_values(
                self.valid_tokens_in_cta_cluster_tile, [values[6]]
            ),
            phase_and_peek=new_from_mlir_values(self.phase_and_peek, [values[7]]),
            fc1_counter_index=new_from_mlir_values(self.fc1_counter_index, [values[8]]),
        )

    def to_rmem(self) -> cute.Tensor:
        rmem = cute.make_rmem_tensor((self.TotalFields,), cutlass.Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.k_tile_cnt          # = cumulative_data_physical_row
        rmem[4] = self.cumulative_sf_physical_row
        rmem[5] = self.cumulative_token_block_count
        rmem[6] = self.valid_tokens_in_cta_cluster_tile
        rmem[7] = self.phase_and_peek
        return rmem

    @classmethod
    def from_rmem(cls, rmem: cute.Tensor) -> "GluMxFp8WorkTileInfo":
        return cls(
            expert_idx=rmem[0],  # type: ignore[arg-type]
            tile_m_idx=rmem[1],  # type: ignore[arg-type]
            tile_n_idx=rmem[2],  # type: ignore[arg-type]
            cumulative_data_physical_row=rmem[3],  # type: ignore[arg-type]
            cumulative_sf_physical_row=rmem[4],  # type: ignore[arg-type]
            cumulative_token_block_count=rmem[5],  # type: ignore[arg-type]
            valid_tokens_in_cta_cluster_tile=rmem[6],  # type: ignore[arg-type]
            phase_and_peek=rmem[7],  # type: ignore[arg-type]
            fc1_counter_index=rmem[1] // cutlass.Int32(cls._cluster_m),  # type: ignore[arg-type]
        )


class GluMxFp8Fc12SchedExtension(SwapABSwigluFp4Fc12SchedExtension):
    """Sched extension for the fused fc1+fc2 GLU MXFP8 kernel.
    """

    WorkTileInfo = GluMxFp8WorkTileInfo

    def __init__(
        self,
        sf_vec_size: int,
        fc1_done_counter_ptr: Pointer,
        fc2_spin_threshold: Union[int, Int32],
        fc1_ready_counter_ptr: Optional[Pointer] = None,
        cluster_m: int = 1,
    ):
        super().__init__(
            sf_vec_size=sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter_ptr,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_ptr=fc1_ready_counter_ptr,
        )
        self.cluster_m = cluster_m
        GluMxFp8WorkTileInfo._cluster_m = cluster_m

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GluMxFp8Fc12SchedExtension":
        base = super().__new_from_mlir_values__(values)
        result = type(self).__new__(type(self))
        result.workspace = base.workspace
        result.sf_vec_size = base.sf_vec_size
        result.fc1_done_counter_ptr = base.fc1_done_counter_ptr
        result.fc2_spin_threshold = base.fc2_spin_threshold
        result.fc1_ready_counter_ptr = base.fc1_ready_counter_ptr
        result.cluster_m = self.cluster_m
        return result

    @cute.jit
    def enrich_work_tile_info(
        self,
        base_work: GluMxFp8WorkTileInfo,
    ) -> GluMxFp8WorkTileInfo:
        """MXFP8 (non-swapAB) override: use cluster-level slot for FC1 arrival counter.

        The swap-AB extension uses ``tile_n_idx`` directly for the
        dispatch->fc1 ready-counter slot.  For NVFP4 swap-AB that is correct
        because N is the token direction.  For MXFP8 non-swap-AB, M is the
        token direction, so the FC1 peek must index by
        ``cumulative_token_block_count + tile_m_idx // cluster_m``
        to match the cluster-granular slot that dispatch_pull increments.

        The scheduler packs per-CTA and cluster-level token counts into
        ``valid_tokens_in_cta_cluster_tile``; this function passes it through
        unchanged and uses the ``valid_tokens_in_cluster_tile`` property for the
        FC1 peek threshold.
        """
        is_valid = base_work.is_valid_tile
        new_phase_and_peek = base_work.phase_and_peek

        if is_valid:
            is_fc1 = base_work.phase == Int32(int(BlockPhase.Linear1))
            is_fc2 = base_work.phase == Int32(int(BlockPhase.Linear2))

            # FC1 arrival peek: cluster-granular slot = cumul + tile_m_idx // cluster_m.
            if cutlass.const_expr(self.fc1_ready_counter_ptr is not None):
                if is_fc1:
                    fc1_counter_slot = (
                        base_work.cumulative_token_block_count
                        + base_work.tile_m_idx // Int32(self.cluster_m)
                    )
                    counter_ptr = self.fc1_ready_counter_ptr + fc1_counter_slot
                    peek_ready = spin_wait(
                        counter_ptr,
                        lambda v: v >= base_work.valid_tokens_in_cluster_tile,
                        peek_only=True,
                    )
                    peek_bit = Int32(0)
                    if peek_ready:
                        peek_bit = Int32(PeekReadyBit)
                    new_phase_and_peek = base_work.phase_and_peek | peek_bit

            # FC2 peek: MXFP8 TMA-A always spins (peek_ready never checked)
            if is_fc2:
                fc2_counter_slot = (
                    base_work.cumulative_token_block_count + base_work.fc1_counter_index
                )
                counter_ptr = self.fc1_done_counter_ptr + fc2_counter_slot
                peek_ready = spin_wait(
                    counter_ptr,
                    lambda v: v >= self.fc2_spin_threshold,
                    peek_only=True,
                )
                peek_bit = Int32(0)
                if peek_ready:
                    peek_bit = Int32(PeekReadyBit)
                new_phase_and_peek = base_work.phase_and_peek | peek_bit

        return GluMxFp8WorkTileInfo(
            expert_idx=base_work.expert_idx,
            tile_m_idx=base_work.tile_m_idx,
            tile_n_idx=base_work.tile_n_idx,
            cumulative_data_physical_row=base_work.cumulative_data_physical_row,
            cumulative_sf_physical_row=base_work.cumulative_sf_physical_row,
            cumulative_token_block_count=base_work.cumulative_token_block_count,
            valid_tokens_in_cta_cluster_tile=base_work.valid_tokens_in_cta_cluster_tile,
            phase_and_peek=new_phase_and_peek,
            fc1_counter_index=base_work.fc1_counter_index,
        )

    @cute.jit
    def prefetch_for_expert(self, expert_idx: Int32) -> None:
        """No-op on subclass for the same reason as ``enrich_work_tile_info``."""
        pass

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: SwapABSwigluFp4Fc12WorkTileInfo,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """Phase-invariant GMEM slice for the operands."""
        expert_idx = work_tile_info.expert_idx
        data_token_offset = work_tile_info.cumulative_data_physical_row
        sf_token_offset = work_tile_info.cumulative_sf_physical_row

        shape = gmem_tensor_in_moe_view.shape
        stride = gmem_tensor_in_moe_view.stride
        c1 = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        if cutlass.const_expr(tensor_name == "fc1_activation"):
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc1_weight"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc1_activation_sf"):
            real = cute.domain_offset(
                (sf_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride)
            )
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc1_weight_sf"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride)
            )
            return (real, None)

        elif cutlass.const_expr(tensor_name == "c"):
            # Raw fc1 accumulator output (gate+up FP32, pre-SwiGLU).
            # Token-indexed like "d" but distinct tensor with intermediate_gateup columns.
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "d"):
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))  # type: ignore[index]
            return (real, None)

        elif cutlass.const_expr(tensor_name == "sfd"):
            real = cute.domain_offset(
                (sf_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            per_expert_shape = (shape[0], shape[1], c1)  # type: ignore[index]
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride)
            )
            return (real, None)

        elif cutlass.const_expr(tensor_name == "topk"):
            real = cute.domain_offset(
                (data_token_offset,), gmem_tensor_in_moe_view
            )
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_activation"):
            # fc2 A-side = fc1_output (token-indexed, same formula as "b")
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_activation_sf"):
            # fc2 SFA = fc1_output_sf (token-indexed sf)
            real = cute.domain_offset(
                (sf_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride)
            )
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_weight"):
            # fc2 B-side = fc2_weight (expert-indexed)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_weight_sf"):
            # fc2 SFB = fc2_weight_sf (expert-indexed)
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(
                real.iterator, cute.make_layout(sf_layout.shape, stride=stride)
            )
            return (real, None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")
