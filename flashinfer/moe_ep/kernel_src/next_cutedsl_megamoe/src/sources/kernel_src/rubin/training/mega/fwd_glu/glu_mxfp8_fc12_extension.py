# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Scheduling adapter for the MXFP8 GLU FC12 kernel."""

import dataclasses
from typing import ClassVar, List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ......helpers.dsl_helpers import spin_peek, spin_wait
from .....schedulers.fc12_mapping import BlockPhase, NonSwapAbFc12WorkTileInfo, peek_ready_bit


# Forward GLU tensor roles: FC1 activation is token-indexed (M), FC1 weight is
# expert-indexed; "d"/"sfd" are the FC1 fp8 output + E8M0 plane, "c" the raw gate/up.
TensorRole = Literal[
    "fc1_activation",
    "fc1_weight",
    "fc1_activation_sf",
    "fc1_weight_sf",
    "c",
    "d",
    "sfd",
    "topk",
    "fc2_activation",
    "fc2_activation_sf",
    "fc2_weight",
    "fc2_weight_sf",
]


@cute.jit
def _rewrite_tensor_shape(tensor: cute.Tensor, new_shape: Tuple) -> cute.Tensor:
    return cute.make_tensor(tensor.iterator, cute.make_layout(new_shape, stride=tensor.stride))


@dataclasses.dataclass(frozen=True)
class GluMxFp8Fc12SchedExtension:
    """Kernel-owned work-tile preparation and GMEM view adapter (non-swap MXFP8)."""

    work_tile_type: ClassVar[type] = NonSwapAbFc12WorkTileInfo

    sf_vec_size: int
    fc1_done_counter_pointer: Pointer
    fc2_spin_threshold: Int32
    fc1_ready_counter_pointer: Optional[Pointer] = None
    cluster_m: int = 1

    def __post_init__(self) -> None:
        if self.sf_vec_size <= 0:
            raise ValueError(f"sf_vec_size must be positive, got {self.sf_vec_size}.")
        if self.cluster_m <= 0:
            raise ValueError(f"cluster_m must be positive, got {self.cluster_m}.")
        object.__setattr__(self, "fc2_spin_threshold", Int32(self.fc2_spin_threshold))

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.fc1_done_counter_pointer))
        values.extend(extract_mlir_values(self.fc2_spin_threshold))
        if self.fc1_ready_counter_pointer is not None:
            values.extend(extract_mlir_values(self.fc1_ready_counter_pointer))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GluMxFp8Fc12SchedExtension":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return result

        fc1_done_counter_pointer = rebuild(self.fc1_done_counter_pointer)
        fc2_spin_threshold = rebuild(self.fc2_spin_threshold)
        fc1_ready_counter_pointer = (
            rebuild(self.fc1_ready_counter_pointer) if self.fc1_ready_counter_pointer is not None else None
        )
        if value_index != len(values):
            raise ValueError(
                f"GluMxFp8Fc12SchedExtension MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter_pointer,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=fc1_ready_counter_pointer,
            cluster_m=self.cluster_m,
        )

    @cute.jit
    def _counter_slot(self, work_tile: NonSwapAbFc12WorkTileInfo) -> Int32:
        # Cluster-granular token-block slot: dispatch_pull increments one counter per
        # cluster-level token block, so M-direction tiles fold by cluster_m.
        return work_tile.cumulative_token_block_count + work_tile.tile_m_idx // Int32(self.cluster_m)

    @cute.jit
    def prepare_work_tile(self, work_tile: NonSwapAbFc12WorkTileInfo) -> NonSwapAbFc12WorkTileInfo:
        """Pack kernel readiness observations into the published tile flags."""
        phase_and_flags = work_tile.phase_and_flags
        if work_tile.is_valid_tile:
            counter_slot = self._counter_slot(work_tile)
            is_fc1 = work_tile.phase == Int32(BlockPhase.Linear1)
            is_fc2 = work_tile.phase == Int32(BlockPhase.Linear2)

            if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
                if is_fc1:
                    counter_pointer = self.fc1_ready_counter_pointer + counter_slot
                    peek_flag = Int32(0)
                    if spin_peek(counter_pointer, lambda value: value >= work_tile.valid_tokens_in_cluster_tile):
                        peek_flag = Int32(peek_ready_bit)
                    phase_and_flags = work_tile.phase_and_flags | peek_flag

            if is_fc2:
                counter_pointer = self.fc1_done_counter_pointer + counter_slot
                peek_flag = Int32(0)
                if spin_peek(counter_pointer, lambda value: value >= self.fc2_spin_threshold):
                    peek_flag = Int32(peek_ready_bit)
                phase_and_flags = work_tile.phase_and_flags | peek_flag

        return NonSwapAbFc12WorkTileInfo(
            expert_idx=work_tile.expert_idx,
            tile_m_idx=work_tile.tile_m_idx,
            tile_n_idx=work_tile.tile_n_idx,
            cumulative_data_physical_row=work_tile.cumulative_data_physical_row,
            cumulative_sf_physical_row=work_tile.cumulative_sf_physical_row,
            cumulative_token_block_count=work_tile.cumulative_token_block_count,
            valid_tokens_in_cta_cluster_tile=work_tile.valid_tokens_in_cta_cluster_tile,
            phase_and_flags=phase_and_flags,
        )

    @cute.jit
    def wait_for_input(self, work_tile: NonSwapAbFc12WorkTileInfo) -> None:
        """Wait until this FC1 input tile's cluster-level token count has arrived."""
        if cutlass.const_expr(self.fc1_ready_counter_pointer is not None):
            counter_pointer = self.fc1_ready_counter_pointer + self._counter_slot(work_tile)
            spin_wait(
                counter_pointer,
                lambda value: value >= work_tile.valid_tokens_in_cluster_tile,
                peek_status=work_tile.peek_ready,
            )

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: TensorRole,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info: NonSwapAbFc12WorkTileInfo,
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
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc1_weight"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc1_activation_sf"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc1_weight_sf"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "c"):
            # Raw fc1 accumulator output (gate+up FP32, pre-SwiGLU): token-indexed.
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "d"):
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "sfd"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "topk"):
            real = cute.domain_offset((data_token_offset,), gmem_tensor_in_moe_view)
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_activation"):
            real = cute.domain_offset((data_token_offset, 0, 0), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc2_activation_sf"):
            real = cute.domain_offset((sf_token_offset, 0, 0), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "fc2_weight"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            return (_rewrite_tensor_shape(real, (shape[0], shape[1], c1)), None)

        elif cutlass.const_expr(tensor_name == "fc2_weight_sf"):
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            per_expert_shape = (shape[0], shape[1], c1)
            sf_layout = tile_atom_to_shape_SF(per_expert_shape, sf_vec_size)
            real = cute.make_tensor(real.iterator, cute.make_layout(sf_layout.shape, stride=stride))
            return (real, None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")


__all__ = ["GluMxFp8Fc12SchedExtension", "TensorRole"]
