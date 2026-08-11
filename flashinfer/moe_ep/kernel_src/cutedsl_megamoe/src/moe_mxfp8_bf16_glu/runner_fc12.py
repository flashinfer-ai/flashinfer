# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host driver for the lean MXFP8-weight/BF16-activation fused FC12 path.

The shared :class:`ProblemDesc` deliberately remains ``kind="bf16"`` because
the activation, FC1 hand-off, and final output all follow the BF16 contract.
``weight_kind`` is kept separately and controls only the E4M3/E5M2 weight
storage.  This avoids teaching the common runner that a single ``kind`` can
describe two different operand dtypes.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Literal, Optional, Tuple

import torch

from common.host_utils import mxfp8_quantize_per_block_32_row
from moe_bf16_glu.runner_common import TrainingImplDesc
from moe_bf16_glu.runner_fc12 import SwigluBf16Fc12Tester
from moe_nvfp4_swapab.runner_common import assemble_raw_scales_stacked_expert
from moe_nvfp4_swapab.runner_fc12_common import (
    Fc12TesterBase,
    MiscDesc,
    ProblemDesc,
    add_common_fc12_arguments,
    parse_tuple,
)


WeightKind = Literal["mxfp8_bf16_e4m3", "mxfp8_bf16_e5m2"]

_WEIGHT_DTYPE_BY_KIND = {
    "mxfp8_bf16_e4m3": torch.float8_e4m3fn,
    "mxfp8_bf16_e5m2": torch.float8_e5m2,
}


class Mxfp8Bf16Fc12Tester(SwigluBf16Fc12Tester):
    """Lean FC12 tester with MXFP8 weights and BF16 activations/handoff."""

    def __init__(
        self,
        problem: ProblemDesc,
        impl: TrainingImplDesc,
        misc: MiscDesc,
        *,
        weight_kind: WeightKind,
    ) -> None:
        # Do not call SwigluBf16Fc12Tester.__init__: that implementation locks
        # the dense BF16 kernel to N=256. Mixed supports N128 TMEM and the two
        # validated N256 SMEM/overlap geometries.
        Fc12TesterBase.__init__(self, problem, impl, misc)
        if problem.kind != "bf16":
            raise ValueError(
                "the mixed runner uses ProblemDesc(kind='bf16') for its "
                f"activation/handoff contract; got {problem.kind!r}."
            )
        if problem.fc2_output_dtype is not torch.bfloat16:
            raise ValueError("mixed FC12 supports BF16 output only.")
        if problem.tokens_after_topk < 0:
            raise ValueError(
                "tokens_after_topk must be non-negative; "
                f"got {problem.tokens_after_topk}."
            )
        if problem.gate_up_clamp is not None and (
            not math.isfinite(problem.gate_up_clamp)
            or problem.gate_up_clamp < 0.0
        ):
            raise ValueError(
                "gate_up_clamp must be finite and non-negative when set; "
                f"got {problem.gate_up_clamp!r}."
            )
        if weight_kind not in _WEIGHT_DTYPE_BY_KIND:
            raise ValueError(
                f"weight_kind must be one of {tuple(_WEIGHT_DTYPE_BY_KIND)}, "
                f"got {weight_kind!r}."
            )
        m, n, k = impl.mma_tiler_mnk
        supported_mma_tilers = (
            (256, 128, 128),
            (256, 256, 128),
        )
        if (m, n, k) not in supported_mma_tilers:
            raise ValueError(
                "mixed FC12 requires mma_tiler_mnk in "
                f"{supported_mma_tilers}; "
                f"got {impl.mma_tiler_mnk}."
            )
        if not impl.use_2cta_instrs:
            raise ValueError(
                "mixed FC12 requires use_2cta_instrs=True."
            )
        if impl.cluster_shape_mnk != (2, 1, 1):
            raise ValueError(
                "mixed FC12 requires cluster_shape_mnk=(2,1,1); "
                f"got {impl.cluster_shape_mnk}."
            )
        if impl.generate_c:
            raise ValueError("mixed FC12 does not support the optional raw-C tensor.")
        if not impl.non_ubulk_fc2_store:
            raise ValueError("mixed FC12 requires the direct FC2 store path.")
        if impl.token_back_mode != "epi_warps":
            raise ValueError(
                "mixed FC12 does not enable token-back warps; "
                "use token_back_mode='epi_warps'."
            )
        if (
            problem.hidden <= 0
            or problem.intermediate <= 0
            or problem.hidden % 32 != 0
            or (problem.intermediate // 2) % 32 != 0
        ):
            raise ValueError(
                "hidden and intermediate/2 must be positive multiples of "
                "the MXFP8 per-32 scale block."
            )

        self.weight_kind: WeightKind = weight_kind
        self.weight_torch_dtype = _WEIGHT_DTYPE_BY_KIND[weight_kind]

    def _validate_host_tensor_contracts(self) -> None:
        """Fail before CuTeDSL tracing when a public tensor ABI is malformed."""

        required = {
            "activation": self.activation,
            "fc1_weight": self.fc1_weight,
            "fc1_weight_sf": self.fc1_weight_sf,
            "fc2_weight": self.fc2_weight,
            "fc2_weight_sf": self.fc2_weight_sf,
            "fc2_output": self.fc2_output,
            "topk_scores": self.topk_scores,
            "offs": self.offs,
        }
        missing = [name for name, tensor in required.items() if tensor is None]
        if missing:
            raise RuntimeError(
                "generate_inputs must populate tensors before launch; missing "
                + ", ".join(missing)
            )

        activation = self.activation
        fc1_weight = self.fc1_weight
        fc1_weight_sf = self.fc1_weight_sf
        fc2_weight = self.fc2_weight
        fc2_weight_sf = self.fc2_weight_sf
        fc2_output = self.fc2_output
        topk_scores = self.topk_scores
        offs = self.offs
        assert activation is not None
        assert fc1_weight is not None
        assert fc1_weight_sf is not None
        assert fc2_weight is not None
        assert fc2_weight_sf is not None
        assert fc2_output is not None
        assert topk_scores is not None
        assert offs is not None

        e = self.problem.experts
        h = self.problem.hidden
        i = self.problem.intermediate
        rows = activation.shape[0]
        expected = {
            "activation": (rows, h),
            "fc1_weight": (e, h, i),
            "fc2_weight": (e, i // 2, h),
            "fc2_output": (rows, 1, h),
            "topk_scores": (rows,),
            "offs": (e,),
        }
        tensors = {
            "activation": activation,
            "fc1_weight": fc1_weight,
            "fc2_weight": fc2_weight,
            "fc2_output": fc2_output,
            "topk_scores": topk_scores,
            "offs": offs,
        }
        for name, shape in expected.items():
            actual_shape = tuple(tensors[name].shape)
            if actual_shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {actual_shape}."
                )

        expected_dtypes = {
            "activation": torch.bfloat16,
            "fc1_weight": self.weight_torch_dtype,
            "fc1_weight_sf": torch.float8_e8m0fnu,
            "fc2_weight": self.weight_torch_dtype,
            "fc2_weight_sf": torch.float8_e8m0fnu,
            "fc2_output": torch.bfloat16,
            "topk_scores": torch.float32,
            "offs": torch.int32,
        }
        for name, dtype in expected_dtypes.items():
            if required[name].dtype is not dtype:
                raise TypeError(
                    f"{name} must have dtype {dtype}, "
                    f"got {required[name].dtype}."
                )

        def _sf_flat_elements(output_features: int, reduction: int) -> int:
            padded_rows = ((output_features + 127) // 128) * 128
            scale_cols = reduction // 32
            padded_cols = ((scale_cols + 3) // 4) * 4
            return padded_rows * padded_cols

        expected_sf_shapes = {
            "fc1_weight_sf": (e, _sf_flat_elements(i, h)),
            "fc2_weight_sf": (e, _sf_flat_elements(h, i // 2)),
        }
        for name, shape in expected_sf_shapes.items():
            tensor = required[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, "
                    f"got {tuple(tensor.shape)}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")

        if activation.stride(1) != 1:
            raise ValueError("activation hidden/K dimension must be stride-1.")
        if fc1_weight.stride(1) != 1:
            raise ValueError("fc1_weight hidden/K dimension must be stride-1.")
        if fc2_weight.stride(1) != 1:
            raise ValueError(
                "fc2_weight intermediate-down/K dimension must be stride-1."
            )
        if fc2_output.stride(2) != 1:
            raise ValueError("fc2_output hidden dimension must be stride-1.")
        if not topk_scores.is_contiguous() or not offs.is_contiguous():
            raise ValueError("topk_scores and offs must be contiguous.")

    def run_kernel(self) -> None:
        self._validate_host_tensor_contracts()
        super().run_kernel()

    @property
    def _epilogue_token_tile(self) -> int:
        from moe_mxfp8_bf16_glu.epilogue_mxfp8_bf16 import (
            EpilogueTokenTile,
        )

        return EpilogueTokenTile

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # The shared swap-AB FC2 output router consumes MoE-domain
        # (token, topk, hidden).  Lean FC12 has topk=1.
        return (data_total_rows, 1, self.problem.hidden)

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        shape = self._fc2_output_shape(data_total_rows)
        byte_shape = (*shape[:-1], shape[-1] * torch.bfloat16.itemsize)
        output_bytes = torch.full(
            byte_shape, 0xFF, dtype=torch.uint8, device="cuda"
        )
        self.fc2_output = output_bytes.view(torch.bfloat16).reshape(shape)

    def _quantize_weight_rows(
        self, shape: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Return physical K-major weights plus public atom-swizzled scales.

        ``shape`` is logical ``(experts, output_features, K)``.  The returned
        weight has physical host shape ``(experts, K, output_features)`` so K
        is stride-1.  Each scale plane is swizzled independently, then stacked
        on the expert axis, matching ``tile_atom_to_shape_SF((M,K,L), 32)``.
        """

        experts, output_features, k = shape
        logical_weights: List[torch.Tensor] = []
        raw_scales: List[torch.Tensor] = []
        for _expert in range(experts):
            if self.misc.perf_run:
                source = torch.randn(
                    (output_features, k), device="cuda", dtype=torch.float32
                )
            else:
                # Small values keep both FP8 kinds away from saturation while
                # still exercising non-unit E8M0 scales and BF16 rounding.
                source = (
                    torch.randn(
                        (output_features, k),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.25
                )
            quantized, raw_scale = mxfp8_quantize_per_block_32_row(
                source, self.weight_torch_dtype
            )
            logical_weights.append(quantized)
            raw_scales.append(raw_scale)

        logical = torch.stack(logical_weights, dim=0)
        physical = logical.transpose(1, 2)
        swizzled = assemble_raw_scales_stacked_expert(raw_scales)
        return physical, swizzled, raw_scales

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        self.activation = self._create_bf16_tensor(
            (data_total_rows, problem.hidden)
        )
        (
            self.fc1_weight,
            self.fc1_weight_sf,
            self.raw_fc1_weight_sf_list,
        ) = self._quantize_weight_rows(
            (problem.experts, problem.intermediate, problem.hidden)
        )
        (
            self.fc2_weight,
            self.fc2_weight_sf,
            self.raw_fc2_weight_sf_list,
        ) = self._quantize_weight_rows(
            (
                problem.experts,
                problem.hidden,
                problem.intermediate // 2,
            )
        )

    def _generate_inputs_skeleton(
        self,
        valid_tokens_per_expert: List[int],
        data_total_rows: int,
    ) -> None:
        """Allocate correctly typed placeholders for compile/perf-only runs."""

        super()._generate_inputs_skeleton(
            valid_tokens_per_expert, data_total_rows
        )
        e = self.problem.experts
        i = self.problem.intermediate
        h = self.problem.hidden
        fp8 = self.weight_torch_dtype
        # Match the real quantization path: expose physical (E, K, M)
        # shapes while retaining stride-1 along K for swap-AB operand A.
        self.fc1_weight = torch.empty(
            (e, i, h), dtype=fp8, device="cuda"
        ).transpose(1, 2)
        self.fc2_weight = torch.empty(
            (e, h, i // 2), dtype=fp8, device="cuda"
        ).transpose(1, 2)

        # Build one correctly padded/swizzled empty SF plane to obtain its
        # public expert stride.  Values are irrelevant in compile-only mode.
        sf1_raw = [
            torch.ones(
                (i, h // 32),
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )
            for _ in range(e)
        ]
        sf2_raw = [
            torch.ones(
                (h, (i // 2) // 32),
                dtype=torch.float8_e8m0fnu,
                device="cuda",
            )
            for _ in range(e)
        ]
        self.raw_fc1_weight_sf_list = sf1_raw
        self.raw_fc2_weight_sf_list = sf2_raw
        self.fc1_weight_sf = assemble_raw_scales_stacked_expert(sf1_raw)
        self.fc2_weight_sf = assemble_raw_scales_stacked_expert(sf2_raw)

    def _partition_workspace(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Partition the BF16 workspace using swap-AB token-block indexing."""

        problem = self.problem
        data_total_rows = int(self.data_physical_offsets[-1])
        intermediate_downproj = problem.intermediate // 2
        counter_token_tile = self.impl.mma_tiler_mnk[1]
        counter_slots_upper = (
            (data_total_rows + counter_token_tile - 1) // counter_token_tile
            + problem.experts
        )

        fc1_bytes = data_total_rows * intermediate_downproj * 2
        counter_bytes = counter_slots_upper * 4
        ws = self.workspace
        offset = 0
        fc1_output = (
            ws[offset : offset + fc1_bytes]
            .view(torch.bfloat16)
            .reshape(data_total_rows, intermediate_downproj)
        )
        offset += fc1_bytes
        done_counter = ws[offset : offset + counter_bytes].view(torch.int32)
        offset += counter_bytes
        load_balance_counter = None
        if self.impl.load_balance_mode == "atomic_counter":
            load_balance_counter = ws[offset : offset + 4].view(torch.int32)
        return fc1_output, done_counter, load_balance_counter

    def _instantiate_kernel(self, common_kwargs: dict):
        import cutlass

        from moe_mxfp8_bf16_glu.kernel_mxfp8_bf16_glu_fc12 import (
            Sm100SwapABMxfp8Bf16Fc12Kernel,
        )

        kernel = Sm100SwapABMxfp8Bf16Fc12Kernel(
            **common_kwargs,
            transform_buffer=getattr(
                self.impl, "transform_buffer", "tmem"
            ),
            accumulator_overlap=getattr(
                self.impl, "accumulator_overlap", False
            ),
            transform_k_tile=getattr(
                self.impl, "transform_k_tile", 128
            ),
            epi_flag_batch=self.impl.epi_flag_batch,
            gate_up_clamp=self.problem.gate_up_clamp,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
        )
        self._kernel_c_dtype = cutlass.BFloat16
        return kernel

    def _extra_runtime_kwargs(self) -> dict:
        import cutlass.torch as cutlass_torch

        def _to_cute(tensor: torch.Tensor):
            result = cutlass_torch.from_dlpack(tensor, assumed_align=16)
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return result.mark_layout_dynamic(leading_dim=leading_dim)

        return {
            "fc1_weight_sf": _to_cute(self.fc1_weight_sf),
            "fc2_weight_sf": _to_cute(self.fc2_weight_sf),
        }

    def compute_reference(self) -> None:
        if self.activation is None or self.offs is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")
        if self.misc.skip_ref_check:
            return
        self._validate_host_tensor_contracts()

        from moe_bf16_glu.mega_reference_bf16 import (
            _DenseGemmReferenceLauncher,
            reference_expert_fc12,
        )
        from moe_mxfp8_bf16_glu.mega_reference_mxfp8_bf16 import (
            mxfp8_weight_from_swizzled_to_bf16,
        )
        from moe_mxfp8_bf16_glu.epilogue_mxfp8_bf16 import (
            Fc1GateUpInterleave,
        )

        problem = self.problem
        total_rows = self.data_physical_offsets[-1]
        if getattr(self, "_ref_mm", None) is None:
            self._ref_mm = _DenseGemmReferenceLauncher(
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )

        self.fc2_output_ref = torch.zeros(
            (total_rows, problem.hidden),
            dtype=problem.fc2_output_dtype,
            device="cuda",
        )
        self._ref_fc1_q_per_expert = [None] * problem.experts
        self._ref_fc1_gateup_per_expert = [None] * problem.experts
        for expert in range(problem.experts):
            valid = self.valid_tokens_per_expert[expert]
            if valid == 0:
                continue
            start = self.data_physical_offsets[expert]
            act = self.activation[start : start + valid]
            topk = self.topk_scores[start : start + valid]
            sf1_flat = self.fc1_weight_sf[expert].reshape(-1)
            sf2_flat = self.fc2_weight_sf[expert].reshape(-1)

            # The mixed mainloop first expands each FP8/E8M0 weight block to
            # BF16, then feeds that BF16 operand into tcgen05.  Mirror the
            # transform exactly, but run both reference GEMMs through the
            # repository's dense tcgen05 launcher.  This preserves the
            # project's exact GPU-oracle contract and removes any ambiguity
            # from a host matmul's potentially different reduction order.
            fc1_weight_bf16 = mxfp8_weight_from_swizzled_to_bf16(
                self.fc1_weight[expert],
                sf1_flat,
            )
            fc2_weight_bf16 = mxfp8_weight_from_swizzled_to_bf16(
                self.fc2_weight[expert],
                sf2_flat,
            )
            fc2_fp32, fc1_bf16, fc1_fp32 = reference_expert_fc12(
                ref_mm=self._ref_mm,
                act=act,
                fc1_weight=fc1_weight_bf16,
                fc2_weight=fc2_weight_bf16,
                intermediate=problem.intermediate,
                hidden=problem.hidden,
                gate_up_interleave=Fc1GateUpInterleave,
                gate_up_clamp=problem.gate_up_clamp,
                topk_weights=topk,
                ref_compute_graph=self.misc.ref_compute_graph,
            )
            self._ref_fc1_q_per_expert[expert] = fc1_bf16
            self._ref_fc1_gateup_per_expert[expert] = fc1_fp32.to(
                torch.bfloat16
            )
            fc2_fp32 = self._apply_topk_post_fc2(fc2_fp32, topk)
            self.fc2_output_ref[start : start + valid] = fc2_fp32.to(
                problem.fc2_output_dtype
            )

    def _validate_padding_sentinels(self) -> None:
        """Ensure predicated stores did not touch per-expert padding rows."""

        if self.fc2_output is None or self._ws_fc1_output_torch is None:
            raise RuntimeError(
                "padding validation requires a completed kernel launch."
            )

        fc2_bytes = self.fc2_output.view(torch.uint8)
        fc1_bytes = self._ws_fc1_output_torch.view(torch.uint8)
        for expert, valid in enumerate(self.valid_tokens_per_expert):
            pad_start = self.data_physical_offsets[expert] + valid
            pad_end = self.data_physical_offsets[expert + 1]
            if pad_start >= pad_end:
                continue
            fc2_padding = fc2_bytes[pad_start:pad_end]
            fc1_padding = fc1_bytes[pad_start:pad_end]
            fc2_bad = int((fc2_padding != 0xFF).sum().item())
            fc1_bad = int((fc1_padding != 0).sum().item())
            if fc2_bad or fc1_bad:
                raise AssertionError(
                    f"expert {expert} padding rows [{pad_start}, {pad_end}) "
                    f"were modified: fc2_bad_bytes={fc2_bad}, "
                    f"fc1_bad_bytes={fc1_bad}."
                )

    def validate(self) -> None:
        super().validate()
        self._validate_padding_sentinels()

    def _print_layout_info(self) -> None:
        super()._print_layout_info()
        for name in ("fc1_weight_sf", "fc2_weight_sf"):
            tensor = getattr(self, name)
            print(
                f"{name}: shape={tuple(tensor.shape)} "
                f"stride={tensor.stride()} dtype={tensor.dtype}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lean fused FC12: MXFP8 weights + BF16 activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_fc12_arguments(parser)
    parser.set_defaults(
        kind="mxfp8_bf16_e4m3",
        mma_tiler_mnk="256,128,128",
        cluster_shape_mnk="2,1,1",
        use_2cta_instrs=True,
    )
    parser.add_argument(
        "--kind",
        choices=tuple(_WEIGHT_DTYPE_BY_KIND),
        default="mxfp8_bf16_e4m3",
        help="MXFP8 weight encoding; activations/handoff remain BF16.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="Optional asymmetric SwiGLU gate/up clamp.",
    )
    parser.add_argument(
        "--transform_buffer",
        choices=("smem", "tmem"),
        default="tmem",
        help="Storage backing the transformed-A pipeline.",
    )
    parser.add_argument(
        "--accumulator_overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the two physical accumulator stages overlap in TMEM.",
    )
    parser.add_argument(
        "--transform_k_tile",
        type=int,
        choices=(64, 128),
        default=128,
        help="Internal transform/MMA K tile; raw TMA K remains MNK.K.",
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
        kind="bf16",
        gate_up_clamp=args.gate_up_clamp,
    )
    impl = TrainingImplDesc(
        mma_tiler_mnk=parse_tuple(args.mma_tiler_mnk),
        cluster_shape_mnk=parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        generate_c=False,
        use_stg_fc1=False,
    )
    impl.transform_buffer = args.transform_buffer
    impl.accumulator_overlap = args.accumulator_overlap
    impl.transform_k_tile = args.transform_k_tile
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
    tester = Mxfp8Bf16Fc12Tester(
        problem, impl, misc, weight_kind=args.kind
    )
    tester.run()


if __name__ == "__main__":
    main()
