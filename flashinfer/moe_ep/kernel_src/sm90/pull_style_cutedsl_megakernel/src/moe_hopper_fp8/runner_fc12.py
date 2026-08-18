# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Host driver for the MegaMoE FP8 GLU fused fc1+fc2 kernel.
"""


import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch

## TODO: currently some common modules are located in moe_nvfp4_swapab,
## which will be moved to common package later. These paths dependency
## could be removed once the modules are moved.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
_NVFP4_DIR = os.path.join(_PARENT_DIR, "moe_nvfp4_swapab")
for _p in (_PARENT_DIR, _NVFP4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_nvfp4_swapab.runner_fc12_common import (
    ProblemDesc,
    ImplDesc as _BaseImplDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_hopper_fp8.epilogue_fp8_swapab import (
    SwapABTileMChoices,
    SwapABTokenTileNChoices,
)
from moe_hopper_fp8.epilogue_fp8 import (
    NonSwapTileMChoices,
    NonSwapTileNChoices,
)
from moe_nvfp4_swapab.runner_common import (
    swiglu_fold_interleave,
)
from common.megamoe_constants import (
    Fp8BlockScaleK,
    Fp8Fc2ActivationScaleK,
    Fp8GateUpInterleave,
    Fp8WeightScaleBlockK,
    Fp8WeightScaleBlockN,
)
from common.host_utils import (
    kind_data_dtype,
    kind_sf_vec_size,
)
from moe_hopper_fp8.hopper_moe_utils import (
    FP8_ACCUM_MODE_CHOICES,
    FP8_KIND_CHOICES,
    FP8_SCALE_MODE_CHOICES,
    Fp8PerTensorTargetAmax,
    create_fp8_tensor,
    compute_fp8_per_tensor_output_dequant_scale_from_absmax,
    dequantize_fp8_per_token_block,
    fp8_block_scaled_reference_mm,
    fp8_kind_to_cutlass_dtype,
    fp8_per_tensor_wgmma_reference_mm,
    make_constant_block_scale,
    make_fp8_per_tensor_dequant_scale,
    quantize_fp8_per_token_block,
    quantize_fp8_weight_block_nk,
)


# =============================================================================
# FP8 tester
# =============================================================================


class ImplDesc(_BaseImplDesc):
    """Hopper FP8 impl descriptor with the swap-AB short-N specializations."""

    def _validate_mma_cta_mode(self, m: int) -> None:
        if self.use_2cta_instrs:
            raise ValueError(
                "Hopper FP8 only supports 1CTA MMA; "
                f"got mma_tiler_m={m}, use_2cta_instrs=True."
            )

    def __post_init__(self) -> None:
        m, n, k = self.mma_tiler_mnk
        non_swap_geometry = (
            m in NonSwapTileMChoices and n in NonSwapTileNChoices
        )
        if n not in SwapABTokenTileNChoices and not non_swap_geometry:
            raise ValueError(
                "Hopper FP8 mma_tiler_mnk must use swap-AB N in "
                f"{SwapABTokenTileNChoices} or a non-swap geometry in "
                f"M={NonSwapTileMChoices}, N={NonSwapTileNChoices}; "
                f"got ({m}, {n}, {k})."
            )

        # The shared NVFP4/MXFP8 descriptor intentionally keeps its original
        # N choices. Validate all common fields through it using an equivalent
        # supported N, then restore this Hopper-only compile-time N. This
        # temporary value never becomes the physical token padding.
        original_tiler = self.mma_tiler_mnk
        if n < 64:
            self.mma_tiler_mnk = (m, 64, k)
        try:
            super().__post_init__()
        finally:
            self.mma_tiler_mnk = original_tiler


class SwigluFp8Fc12Tester(Fc12TesterBase):
    """FP8 host-side input/reference/launch/validation driver."""

    def __init__(
        self,
        problem: ProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        fp8_scale_mode: str = "per_tensor",
        fp8_accum_mode: str = "1xacc",
        swap_ab: bool = False,
    ) -> None:
        self.swap_ab = swap_ab
        super().__init__(problem, impl, misc)
        if fp8_scale_mode not in FP8_SCALE_MODE_CHOICES:
            raise ValueError(
                f"fp8_scale_mode must be one of {FP8_SCALE_MODE_CHOICES}, "
                f"got {fp8_scale_mode!r}."
            )
        self.fp8_scale_mode = fp8_scale_mode
        if fp8_accum_mode not in FP8_ACCUM_MODE_CHOICES:
            raise ValueError(
                f"fp8_accum_mode must be one of {FP8_ACCUM_MODE_CHOICES}, "
                f"got {fp8_accum_mode!r}."
            )
        self.fp8_accum_mode = fp8_accum_mode
        # Non-swap publicly supports M=64; the retained M=128 implementation
        # is intentionally disabled. Swap-AB uses weight-M=128/256.
        m, n, _k = impl.mma_tiler_mnk
        valid_geometry = (
            m in SwapABTileMChoices and n in SwapABTokenTileNChoices
            if self.swap_ab
            else m in NonSwapTileMChoices and n in NonSwapTileNChoices
        )
        if not valid_geometry or impl.use_2cta_instrs:
            raise ValueError(
                "Hopper FP8 fused fc12 geometry does not match swap_ab="
                f"{self.swap_ab}; got "
                f"mma_tiler_mnk={impl.mma_tiler_mnk}, "
                f"use_2cta_instrs={impl.use_2cta_instrs}."
            )
        self.fc1_activation_dequant_scale: Optional[torch.Tensor] = None
        self.fc1_weight_dequant_scale: Optional[torch.Tensor] = None
        self.fc2_activation_dequant_scale: Optional[torch.Tensor] = None
        self.fc2_weight_dequant_scale: Optional[torch.Tensor] = None
        self.fc1_activation_block_scale: Optional[torch.Tensor] = None
        self.fc1_weight_block_scale: Optional[torch.Tensor] = None
        self.fc2_activation_block_scale: Optional[torch.Tensor] = None
        self.fc2_weight_block_scale: Optional[torch.Tensor] = None
        self._ref_fc1_raw_fp32_per_expert: List[Optional[torch.Tensor]] = []

    @property
    def _epilogue_token_tile(self) -> int:
        # A non-empty FC1 tail issues one full physical token tile: M for the
        # native layout and N after swapping A/B.
        return (
            self.impl.mma_tiler_mnk[1]
            if self.swap_ab
            else self.impl.mma_tiler_mnk[0]
        )

    # ------------------------------------------------------------------
    # Kind hooks: input / output tensor creation
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # FP8: 2D (token_max, hidden) -- epilogue_fp8.py uses shape[1].
        return (data_total_rows, self.problem.hidden)

    def generate_inputs(self) -> None:
        super().generate_inputs()
        if self.fp8_scale_mode != "blockwise" or self.activation is None:
            return
        if (
            self.fc1_activation_block_scale is None
            or self.fc1_weight_block_scale is None
            or self.fc2_weight_block_scale is None
        ):
            raise RuntimeError("blockwise generate_inputs did not create block scales.")
        self.activation_sf = self.fc1_activation_block_scale
        self.fc1_weight_sf = self.fc1_weight_block_scale
        self.fc2_weight_sf = self.fc2_weight_block_scale
        device = self.activation.device
        self.fc1_activation_dequant_scale = torch.ones(
            (1,), dtype=torch.float32, device=device
        )
        self.fc1_weight_dequant_scale = torch.ones(
            (self.problem.experts,), dtype=torch.float32, device=device
        )
        self.fc2_activation_dequant_scale = torch.ones(
            (1,), dtype=torch.float32, device=device
        )
        self.fc2_weight_dequant_scale = torch.ones(
            (self.problem.experts,), dtype=torch.float32, device=device
        )

    def _check_blockwise_problem_shape(self) -> None:
        problem = self.problem
        intermediate_downproj = problem.intermediate // 2
        checks = (
            ("hidden", problem.hidden, Fp8BlockScaleK),
            ("intermediate", problem.intermediate, Fp8WeightScaleBlockN),
            ("intermediate_downproj", intermediate_downproj, Fp8Fc2ActivationScaleK),
            ("intermediate_downproj", intermediate_downproj, Fp8WeightScaleBlockK),
        )
        for name, value, divisor in checks:
            if value % divisor != 0:
                raise ValueError(
                    f"blockwise FP8 requires {name}={value} divisible by {divisor}."
                )

    def _create_blockwise_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        intermediate_downproj = intermediate // 2
        experts = problem.experts
        data_dtype = kind_data_dtype(problem.kind)
        self._check_blockwise_problem_shape()

        if self.misc.perf_run:
            self.activation = create_fp8_tensor(
                (data_total_rows, hidden),
                data_dtype,
                perf_run=self.misc.perf_run,
            )
            self.fc1_activation_block_scale = make_constant_block_scale(
                data_dtype, (data_total_rows, hidden // Fp8BlockScaleK)
            )

            fc1_weight_nk = create_fp8_tensor(
                (experts, intermediate, hidden),
                data_dtype,
                perf_run=self.misc.perf_run,
            )
            self.fc1_weight = fc1_weight_nk.permute(0, 2, 1)
            self.fc1_weight_block_scale = make_constant_block_scale(
                data_dtype,
                (
                    experts,
                    intermediate // Fp8WeightScaleBlockN,
                    hidden // Fp8WeightScaleBlockK,
                ),
            )

            fc2_weight_nk = create_fp8_tensor(
                (experts, hidden, intermediate_downproj),
                data_dtype,
                perf_run=self.misc.perf_run,
            )
            self.fc2_weight = fc2_weight_nk.permute(0, 2, 1)
            self.fc2_weight_block_scale = make_constant_block_scale(
                data_dtype,
                (
                    experts,
                    hidden // Fp8WeightScaleBlockN,
                    intermediate_downproj // Fp8WeightScaleBlockK,
                ),
            )
            return

        activation_src = create_fp8_tensor(
            (data_total_rows, hidden),
            data_dtype,
            perf_run=self.misc.perf_run,
            return_fp8=False,
        )
        (
            self.activation,
            self.fc1_activation_block_scale,
        ) = quantize_fp8_per_token_block(
            activation_src,
            data_dtype,
            block_k=Fp8BlockScaleK,
            target_amax=Fp8PerTensorTargetAmax,
        )

        fc1_weight_nk_parts = []
        fc1_weight_scale_parts = []
        for _expert_idx in range(experts):
            weight_src = create_fp8_tensor(
                (intermediate, hidden),
                data_dtype,
                perf_run=self.misc.perf_run,
                return_fp8=False,
            )
            weight_q, weight_scale = quantize_fp8_weight_block_nk(
                weight_src,
                data_dtype,
                block_n=Fp8WeightScaleBlockN,
                block_k=Fp8WeightScaleBlockK,
                target_amax=Fp8PerTensorTargetAmax,
            )
            fc1_weight_nk_parts.append(weight_q)
            fc1_weight_scale_parts.append(weight_scale)
        self.fc1_weight = torch.stack(fc1_weight_nk_parts, dim=0).permute(0, 2, 1)
        self.fc1_weight_block_scale = torch.stack(fc1_weight_scale_parts, dim=0)

        fc2_weight_nk_parts = []
        fc2_weight_scale_parts = []
        for _expert_idx in range(experts):
            weight_src = create_fp8_tensor(
                (hidden, intermediate_downproj),
                data_dtype,
                perf_run=self.misc.perf_run,
                return_fp8=False,
            )
            weight_q, weight_scale = quantize_fp8_weight_block_nk(
                weight_src,
                data_dtype,
                block_n=Fp8WeightScaleBlockN,
                block_k=Fp8WeightScaleBlockK,
                target_amax=Fp8PerTensorTargetAmax,
            )
            fc2_weight_nk_parts.append(weight_q)
            fc2_weight_scale_parts.append(weight_scale)
        self.fc2_weight = torch.stack(fc2_weight_nk_parts, dim=0).permute(0, 2, 1)
        self.fc2_weight_block_scale = torch.stack(fc2_weight_scale_parts, dim=0)

    def _create_per_tensor_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts
        data_dtype = kind_data_dtype(problem.kind)

        # -- activation: (data_total_rows, hidden) fp8, hidden stride-1 --
        self.activation = create_fp8_tensor(
            (data_total_rows, hidden),
            data_dtype,
            perf_run=self.misc.perf_run,
        )
        self.fc1_activation_dequant_scale = make_fp8_per_tensor_dequant_scale(
            self.activation
        )

        # -- fc1_weight: (experts, intermediate, hidden) -> permute ->
        #    (experts, hidden, intermediate), hidden stride-1 --
        self.fc1_weight = create_fp8_tensor(
            (experts, intermediate, hidden),
            data_dtype,
            perf_run=self.misc.perf_run,
        ).permute(0, 2, 1)
        self.fc1_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
            self.fc1_weight, reduce_dims=(1, 2)
        )

        # -- fc2_weight: (experts, hidden, inter//2) -> permute ->
        #    (experts, inter//2, hidden), inter//2 stride-1 --
        self.fc2_weight = create_fp8_tensor(
            (experts, hidden, intermediate // 2),
            data_dtype,
            perf_run=self.misc.perf_run,
        ).permute(0, 2, 1)
        self.fc2_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
            self.fc2_weight, reduce_dims=(1, 2)
        )

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        if self.fp8_scale_mode == "blockwise":
            self._create_blockwise_input_data_tensors(data_total_rows)
        else:
            self._create_per_tensor_input_data_tensors(data_total_rows)

    def _init_topk_scores(self, data_total_rows: int) -> None:
        """FP8 fc12 does not apply topk weighting; use 1.0 on valid rows."""
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        self.topk_scores = torch.zeros(
            (data_total_rows,), dtype=torch.float32, device="cuda",
        )
        for e in range(self.problem.experts):
            v_e = valid_tokens[e]
            if v_e == 0:
                continue
            phys = data_offsets[e]
            self.topk_scores[phys : phys + v_e] = 1.0

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        # 0xFF byte fill: bf16/fp16 0xFFFF = NaN -- kernel output overwriting
        # valid rows is easy to distinguish from "kernel never touched this
        # row".  FP8 stores a flat 2D ``(token_max, hidden)`` output.
        problem = self.problem
        hidden = problem.hidden
        fc2_output_bytes = torch.full(
            (data_total_rows, hidden * problem.fc2_output_dtype.itemsize),
            0xFF,
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output = fc2_output_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, hidden
        )

    # ------------------------------------------------------------------
    # Kind hooks: reference compute
    # ------------------------------------------------------------------

    def _quantize_fc2_activation_per_tensor(
        self, swiglu: torch.Tensor, fc2_activation_dequant_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_dtype = kind_data_dtype(self.problem.kind)
        fc2_activation_scale = fc2_activation_dequant_scale.to(
            device=swiglu.device, dtype=torch.float32
        )
        fc1_q = (swiglu / fc2_activation_scale).to(data_dtype)
        sf_cols = swiglu.shape[1] // kind_sf_vec_size(self.problem.kind)
        fc1_sf = torch.ones(
            (swiglu.shape[0], sf_cols),
            dtype=self.activation_sf.dtype,
            device=swiglu.device,
        )
        return fc1_q, fc1_sf

    def compute_reference(self) -> None:
        """FP8 fused fc1+fc2 reference dispatch."""
        if self.fp8_scale_mode == "blockwise":
            self._compute_reference_blockwise()
        else:
            self._compute_reference_per_tensor()

    def _compute_reference_per_tensor(self) -> None:
        """FP8 per-tensor fused fc1+fc2 reference with explicit fp32 dequant scales."""
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return

        problem = self.problem
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        data_total_rows = data_offsets[-1]
        data_dtype = kind_data_dtype(problem.kind)
        gate_up_clamp = getattr(problem, "gate_up_clamp", None)

        ref_bytes = torch.zeros(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output_ref = ref_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, problem.hidden
        )

        self._ref_fc1_q_per_expert = [None] * problem.experts
        self._ref_fc1_raw_sf_per_expert = [None] * problem.experts
        swiglu_per_expert: List[Optional[torch.Tensor]] = [None] * problem.experts
        swiglu_absmax = torch.zeros((), dtype=torch.float32, device=self.activation.device)

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for expert_idx in range(problem.experts):
                v_e = valid_tokens[expert_idx]
                if v_e == 0:
                    continue

                d_start = data_offsets[expert_idx]
                act_slice = self.activation[d_start : d_start + v_e]
                if self.misc.ref_compute_graph == "deepgemm":
                    fc1_fp32 = fp8_per_tensor_wgmma_reference_mm(
                        act_slice,
                        self.fc1_weight[expert_idx],
                        accum_mode=self.fp8_accum_mode,
                        k_chunk=self.impl.mma_tiler_mnk[2],
                    )
                    fc1_fp32 = (
                        fc1_fp32
                        * self.fc1_activation_dequant_scale[0]
                        * self.fc1_weight_dequant_scale[expert_idx]
                    )
                else:
                    act_fp32 = (
                        act_slice.to(torch.float32)
                        * self.fc1_activation_dequant_scale[0]
                    )
                    w1_fp32 = (
                        self.fc1_weight[expert_idx].to(torch.float32)
                        * self.fc1_weight_dequant_scale[expert_idx]
                    )
                    fc1_fp32 = act_fp32 @ w1_fp32

                swiglu = swiglu_fold_interleave(
                    fc1_fp32,
                    Fp8GateUpInterleave,
                    gate_up_clamp=gate_up_clamp,
                )
                swiglu_per_expert[expert_idx] = swiglu
                if swiglu.numel() > 0:
                    swiglu_absmax = torch.maximum(
                        swiglu_absmax, swiglu.to(torch.float32).abs().amax()
                    )

            fc2_activation_dequant_scale = (
                compute_fp8_per_tensor_output_dequant_scale_from_absmax(
                    swiglu_absmax,
                    data_dtype,
                    device=self.activation.device,
                )
            )
            self.fc2_activation_dequant_scale = fc2_activation_dequant_scale

            for expert_idx in range(problem.experts):
                v_e = valid_tokens[expert_idx]
                swiglu = swiglu_per_expert[expert_idx]
                if v_e == 0 or swiglu is None:
                    continue

                d_start = data_offsets[expert_idx]
                fc1_q, fc1_sf = self._quantize_fc2_activation_per_tensor(
                    swiglu, fc2_activation_dequant_scale
                )
                if self.misc.ref_compute_graph == "deepgemm":
                    fc2_fp32 = fp8_per_tensor_wgmma_reference_mm(
                        fc1_q,
                        self.fc2_weight[expert_idx],
                        accum_mode=self.fp8_accum_mode,
                        k_chunk=self.impl.mma_tiler_mnk[2],
                    )
                    fc2_fp32 = (
                        fc2_fp32
                        * fc2_activation_dequant_scale[0]
                        * self.fc2_weight_dequant_scale[expert_idx]
                    )
                    fc2_out = fc2_fp32.to(problem.fc2_output_dtype)
                else:
                    fc1_dq = (
                        fc1_q.to(torch.float32)
                        * fc2_activation_dequant_scale[0]
                    )
                    w2_fp32 = (
                        self.fc2_weight[expert_idx].to(torch.float32)
                        * self.fc2_weight_dequant_scale[expert_idx]
                    )
                    fc2_fp32 = fc1_dq @ w2_fp32
                    fc2_out = fc2_fp32.to(problem.fc2_output_dtype)
                self.fc2_output_ref[d_start : d_start + v_e] = fc2_out

                self._ref_fc1_q_per_expert[expert_idx] = fc1_q
                self._ref_fc1_raw_sf_per_expert[expert_idx] = fc1_sf
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    def _compute_reference_blockwise(self) -> None:
        """DeepGEMM-style blockwise FP8 fused fc1+fc2 reference."""
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return
        if (
            self.activation_sf is None
            or self.fc1_weight_sf is None
            or self.fc2_weight_sf is None
        ):
            raise RuntimeError("blockwise reference requires FP32 block scales.")

        problem = self.problem
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        data_total_rows = data_offsets[-1]
        intermediate_downproj = problem.intermediate // 2
        gate_up_clamp = getattr(problem, "gate_up_clamp", None)

        ref_bytes = torch.zeros(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            dtype=torch.uint8, device="cuda",
        )
        self.fc2_output_ref = ref_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, problem.hidden
        )
        self.fc2_activation_block_scale = torch.ones(
            (data_total_rows, intermediate_downproj // Fp8Fc2ActivationScaleK),
            dtype=torch.float32,
            device=self.activation.device,
        )
        self._ref_fc1_q_per_expert = [None] * problem.experts
        self._ref_fc1_raw_sf_per_expert = [None] * problem.experts
        self._ref_fc1_raw_fp32_per_expert = [None] * problem.experts

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for expert_idx in range(problem.experts):
                v_e = valid_tokens[expert_idx]
                if v_e == 0:
                    continue

                d_start = data_offsets[expert_idx]
                act_slice = self.activation[d_start : d_start + v_e]
                act_scale = self.activation_sf[d_start : d_start + v_e]

                fc1_fp32 = fp8_block_scaled_reference_mm(
                    act_slice,
                    self.fc1_weight[expert_idx],
                    act_scale,
                    self.fc1_weight_sf[expert_idx],
                    a_scale_block_k=Fp8BlockScaleK,
                    b_scale_block_n=Fp8WeightScaleBlockN,
                    b_scale_block_k=Fp8WeightScaleBlockK,
                )
                swiglu = swiglu_fold_interleave(
                    fc1_fp32,
                    Fp8GateUpInterleave,
                    gate_up_clamp=gate_up_clamp,
                )
                self._ref_fc1_raw_fp32_per_expert[expert_idx] = swiglu
                fc2_act_fp8, fc2_act_scale = quantize_fp8_per_token_block(
                    swiglu,
                    kind_data_dtype(problem.kind),
                    block_k=Fp8Fc2ActivationScaleK,
                    # Both kernel variants reuse one FP32 reciprocal per block.
                    use_reciprocal_multiply=True,
                )
                self.fc2_activation_block_scale[
                    d_start : d_start + v_e
                ] = fc2_act_scale

                fc2_fp32 = fp8_block_scaled_reference_mm(
                    fc2_act_fp8,
                    self.fc2_weight[expert_idx],
                    fc2_act_scale,
                    self.fc2_weight_sf[expert_idx],
                    a_scale_block_k=Fp8Fc2ActivationScaleK,
                    b_scale_block_n=Fp8WeightScaleBlockN,
                    b_scale_block_k=Fp8WeightScaleBlockK,
                )
                self.fc2_output_ref[d_start : d_start + v_e] = fc2_fp32.to(
                    problem.fc2_output_dtype
                )
                self._ref_fc1_q_per_expert[expert_idx] = fc2_act_fp8
                self._ref_fc1_raw_sf_per_expert[expert_idx] = fc2_act_scale
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    # ------------------------------------------------------------------
    # Kind hooks: kernel + validation
    # ------------------------------------------------------------------

    def _instantiate_kernel(self, common_kwargs: dict):
        if self.swap_ab:
            from moe_hopper_fp8.kernel_fp8_glu_fc12_swapab import (
                Sm90SwapABSwigluFp8Fc12Kernel as Kernel,
            )
        else:
            from moe_hopper_fp8.kernel_fp8_glu_fc12 import (
                Sm90SwigluFp8Fc12Kernel as Kernel,
            )

        return Kernel(
            **common_kwargs,
            ab_dtype=fp8_kind_to_cutlass_dtype(self.problem.kind),
            fp8_scale_mode=self.fp8_scale_mode,
            fp8_accum_mode=self.fp8_accum_mode,
            gate_up_clamp=self.problem.gate_up_clamp,
        )

    def _ensure_fp8_per_tensor_scale_tensors(self) -> None:
        device = self.activation.device
        if self.fc1_activation_dequant_scale is None:
            self.fc1_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device=device
            )
        if self.fc1_weight_dequant_scale is None:
            self.fc1_weight_dequant_scale = torch.ones(
                (self.problem.experts,), dtype=torch.float32, device=device
            )
        if self.fc2_activation_dequant_scale is None:
            self.fc2_activation_dequant_scale = torch.ones(
                (1,), dtype=torch.float32, device=device
            )
        if self.fc2_weight_dequant_scale is None:
            self.fc2_weight_dequant_scale = torch.ones(
                (self.problem.experts,), dtype=torch.float32, device=device
            )

    def _extra_kernel_runtime_kwargs(self, to_cute) -> dict:
        self._ensure_fp8_per_tensor_scale_tensors()
        return {
            "fc1_activation_dequant_scale": to_cute(
                self.fc1_activation_dequant_scale, assumed_align=4
            ),
            "fc1_weight_dequant_scale": to_cute(
                self.fc1_weight_dequant_scale, assumed_align=4
            ),
            "fc2_activation_dequant_scale": to_cute(
                self.fc2_activation_dequant_scale, assumed_align=4
            ),
            "fc2_weight_dequant_scale": to_cute(
                self.fc2_weight_dequant_scale, assumed_align=4
            ),
        }

    def _partition_workspace(self, counter_token_tile: int):
        if self.swap_ab:
            counter_token_tile = self.impl.mma_tiler_mnk[1]
        if self.fp8_scale_mode != "blockwise":
            return super()._partition_workspace(counter_token_tile)

        problem = self.problem
        data_dtype = kind_data_dtype(problem.kind)
        experts = problem.experts
        intermediate_downproj = problem.intermediate // 2
        data_total_rows = int(self.data_physical_offsets[-1])

        fc1_output_byte_count = data_total_rows * intermediate_downproj
        fc1_output_sf_cols = intermediate_downproj // Fp8Fc2ActivationScaleK
        fc1_output_sf_byte_count = data_total_rows * fc1_output_sf_cols * 4
        counter_slots_upper = (
            (data_total_rows + counter_token_tile - 1) // counter_token_tile
            + experts
        )
        fc1_done_counter_byte_count = counter_slots_upper * 4

        ws = self.workspace
        offset = 0

        fc1_output_torch = (
            ws[offset : offset + fc1_output_byte_count]
            .view(torch.uint8)
            .view(data_dtype)
            .reshape(data_total_rows, intermediate_downproj)
        )
        offset += fc1_output_byte_count

        fc1_output_sf_torch = (
            ws[offset : offset + fc1_output_sf_byte_count]
            .view(torch.uint8)
            .view(torch.float32)
            .reshape(data_total_rows, fc1_output_sf_cols)
        )
        offset += fc1_output_sf_byte_count

        fc1_done_counter_torch = (
            ws[offset : offset + fc1_done_counter_byte_count]
            .view(torch.int32)
        )
        offset += fc1_done_counter_byte_count

        if self.impl.load_balance_mode == "atomic_counter":
            load_balance_counter_torch = ws[offset : offset + 4].view(torch.int32)
        else:
            load_balance_counter_torch = None

        return (
            fc1_output_torch,
            fc1_output_sf_torch,
            fc1_done_counter_torch,
            load_balance_counter_torch,
        )

    def _fc2_tolerance(self) -> Tuple[float, float]:
        # BF16 output: minimum representable relative error = 1/128 ≈ 0.78%
        # (1 BF16 ULP at any value v is v/128, which always exceeds
        # rtol=1e-5 × v regardless of magnitude).  With {0.5, 1.0} input
        # scales the fc2 output reaches ±3K; at that scale 1 BF16 ULP = 8
        # while 1e-5 × 3K = 0.03 — a 267× gap.  1e-2 covers 1 BF16 ULP
        # at all magnitudes (1/128 ≈ 0.78% < 1%) without masking real bugs
        # (genuine GEMM errors are O(1) relative, not 0.78%).
        return 1e-5, 1e-2

    def _validate_fc1_phase(self) -> None:
        """Compare kernel-written fc1 workspace to reference per expert.

        Uses ``compare_and_report_mismatches`` rather than the NVFP4 detail
        loop (the fp4 nibble decode in that loop is meaningless for fp8).
        """
        if (
            self._ws_fc1_output_torch is None
            or self._ws_fc1_output_sf_torch is None
            or not self._ref_fc1_q_per_expert
        ):
            print("[fc1 phase ablation] skipped (workspace or ref not populated)")
            return

        from common.host_utils import compare_and_report_mismatches

        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets

        print("\n" + "=" * 60)
        print("[DEBUG fc1] compare_and_report_mismatches per expert:")
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref_q = self._ref_fc1_q_per_expert[e]
            if v_e == 0 or ref_q is None:
                continue
            kq = self._ws_fc1_output_torch[doff[e] : doff[e] + v_e]
            if self.fp8_scale_mode == "blockwise":
                ref_sf = self._ref_fc1_raw_sf_per_expert[e]
                ref_raw = self._ref_fc1_raw_fp32_per_expert[e]
                if ref_sf is None:
                    continue
                ksf = self._ws_fc1_output_sf_torch[doff[e] : doff[e] + v_e]
                kfp32 = dequantize_fp8_per_token_block(
                    kq,
                    ksf,
                    block_k=Fp8Fc2ActivationScaleK,
                )
                if ref_raw is None:
                    ref_fp32 = dequantize_fp8_per_token_block(
                        ref_q,
                        ref_sf,
                        block_k=Fp8Fc2ActivationScaleK,
                    )
                else:
                    ref_fp32 = ref_raw
            else:
                fc2_activation_scale = self.fc2_activation_dequant_scale[0]
                kfp32 = kq.to(torch.float32) * fc2_activation_scale
                ref_fp32 = ref_q.to(torch.float32) * fc2_activation_scale
            ### TODO: replace to silent check when kernel stable.
            compare_and_report_mismatches(
                kfp32.cpu(),
                ref_fp32.cpu(),
                name=f"fc1_expert{e}",
                atol=5e-2, rtol=5e-2, max_mismatches=5,
            )
        print("=" * 60)


# =============================================================================
# CLI entry point
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """argparse setup for the FP8 fused fc12 path."""
    parser = argparse.ArgumentParser(
        description="MoE FP8 GLU fused fc1+fc2 SwiGLU (host-ready harness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_fc12_arguments(parser)
    parser.set_defaults(
        mma_tiler_mnk="64,128,128",
        cluster_shape_mnk="1,1,1",
        use_2cta_instrs=False,
    )

    # -- FP8 Problem --
    parser.add_argument(
        "--kind", type=str, default="fp8_e4m3",
        choices=list(FP8_KIND_CHOICES),
        help=(
            "FP8 element format. Legacy mxfp8_* names are accepted "
            "as aliases."
        ),
    )
    parser.add_argument(
        "--flag_batch", type=int, default=1,
        help="dispatch_pull release-flag batch size; 1 == per-token "
        "baseline, larger amortizes the device fence over more tokens.",
    )
    parser.add_argument(
        "--gate_up_clamp", type=float, default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
    )
    parser.add_argument(
        "--fp8_scale_mode", type=str, default="per_tensor",
        choices=list(FP8_SCALE_MODE_CHOICES),
        help="FP8 scale interpretation: scalar per-tensor or DeepGEMM-style blockwise.",
    )
    parser.add_argument(
        "--fp8_accum_mode", type=str, default="1xacc",
        choices=list(FP8_ACCUM_MODE_CHOICES),
        help="Per-tensor WGMMA accumulation mode; ignored by blockwise scaling.",
    )
    parser.add_argument(
        "--swap_ab", action="store_true",
        help="Use the Hopper weight-as-A M128/M256xN swap-AB kernel.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    problem = ProblemDesc(
        tokens_after_topk=args.tokens_after_topk,
        experts=args.experts,
        balance_route=args.balance_route,
        hidden=args.hidden,
        intermediate=args.intermediate,
        simulate_ep=args.simulate_ep,
        fc2_output_dtype=args.fc2_output_dtype,
        kind=args.kind,
        gate_up_clamp=args.gate_up_clamp,
    )

    mma_tiler_mnk = parse_tuple(args.mma_tiler_mnk)
    if args.swap_ab and mma_tiler_mnk == (64, 128, 128):
        mma_tiler_mnk = (256, 32, 128)

    impl = ImplDesc(
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        flag_batch=args.flag_batch,
    )

    misc = MiscDesc(
        perf_run=args.perf_run,
        skip_ref_check=args.skip_ref_check,
        run_target_kernel_only=args.run_target_kernel_only,
        enable_debug_checks=args.enable_debug_checks,
        ref_compute_graph=args.ref_compute_graph,
        enable_iket=args.enable_iket,
        seed=args.seed,
        verbose=args.verbose,
        perf_warmup=args.perf_warmup,
        perf_iters=args.perf_iters,
    )

    tester = SwigluFp8Fc12Tester(
        problem,
        impl,
        misc,
        fp8_scale_mode=args.fp8_scale_mode,
        fp8_accum_mode=args.fp8_accum_mode,
        swap_ab=args.swap_ab,
    )
    tester.run()


if __name__ == "__main__":
    main()
    exit(0)
