# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Sched extension for the fused fc1+fc2 GLU BF16 kernel."""

from typing import List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Int32
from cutlass._mlir import ir

from moe_nvfp4_swapab.custom_ext import GluMxFp8Fc12SchedExtension
from moe_nvfp4_swapab.moe_utils import rewrite_tensor_shape


class GluBf16Fc12SchedExtension(GluMxFp8Fc12SchedExtension):
    """Sched extension for the fused fc1+fc2 GLU BF16 kernel.

    Inherits the scheduling logic from the parent: the cluster-granular FC1
    ready-counter peek and the FC2 done-counter peek in
    ``enrich_work_tile_info`` are operand-set independent (M is the token
    direction; slot = ``cumulative_token_block_count + tile_m_idx //
    cluster_m``).

    Overrides construction and ``get_gmem_tensor`` for the BF16 operand set:
    seven data tensors (``fc1_activation`` / ``fc1_weight`` / ``c`` / ``d`` /
    ``topk`` / ``fc2_activation`` / ``fc2_weight``), no scale-factor planes.
    """

    def __init__(
        self,
        fc1_done_counter_ptr: Pointer,
        fc2_spin_threshold: Union[int, Int32],
        fc1_ready_counter_ptr: Optional[Pointer] = None,
        cluster_m: int = 1,
    ):
        # ``sf_vec_size`` is a required parent ctor arg; the BF16 operand set
        # never requests a scale-factor tensor, so the stored value is never
        # read.
        super().__init__(
            sf_vec_size=32,
            fc1_done_counter_ptr=fc1_done_counter_ptr,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_ptr=fc1_ready_counter_ptr,
            cluster_m=cluster_m,
        )

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GluBf16Fc12SchedExtension":
        base = super().__new_from_mlir_values__(values)
        result = GluBf16Fc12SchedExtension.__new__(GluBf16Fc12SchedExtension)
        result.workspace = base.workspace
        result.sf_vec_size = base.sf_vec_size
        result.fc1_done_counter_ptr = base.fc1_done_counter_ptr
        result.fc2_spin_threshold = base.fc2_spin_threshold
        result.fc1_ready_counter_ptr = base.fc1_ready_counter_ptr
        result.cluster_m = base.cluster_m
        return result

    @cute.jit
    def get_gmem_tensor(
        self,
        tensor_name: str,
        gmem_tensor_in_moe_view: cute.Tensor,
        work_tile_info,
    ) -> Tuple[cute.Tensor, Optional[Pointer]]:
        """Phase-invariant GMEM slice for the BF16 operands.

        Token-indexed tensors are offset by the expert's cumulative physical
        row; expert-indexed weights are offset by the expert index on the L
        mode.
        """
        expert_idx = work_tile_info.expert_idx
        data_token_offset = work_tile_info.cumulative_data_physical_row

        shape = gmem_tensor_in_moe_view.shape
        c1 = cutlass.Int32(1)

        if cutlass.const_expr(
            tensor_name in ("fc1_activation", "c", "d", "fc2_activation")
        ):
            # Token-indexed data tensors (activation, raw gate+up C, fc1
            # output D / fc2 A-side reload of it, fc2 output).
            real = cute.domain_offset(
                (data_token_offset, 0, 0), gmem_tensor_in_moe_view
            )
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name in ("fc1_weight", "fc2_weight")):
            # Expert-indexed weights.
            real = cute.domain_offset((0, 0, expert_idx), gmem_tensor_in_moe_view)
            real = rewrite_tensor_shape(real, (shape[0], shape[1], c1))
            return (real, None)

        elif cutlass.const_expr(tensor_name == "topk"):
            real = cute.domain_offset(
                (data_token_offset,), gmem_tensor_in_moe_view
            )
            return (real, None)

        raise ValueError(f"Unknown tensor_name: {tensor_name!r}.")
