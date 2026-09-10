# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host driver for the SM120 MXFP8 swap-AB fused fc1+fc2 kernel."""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch

# Ensure absolute package imports work when this file is run as a script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from moe_sm120_mxfp8_swapab.runner_fc12_common import (
    ProblemDesc,
    ImplDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_sm120_mxfp8_swapab.runner_common import (
    from_blocked,
    dequant_block_scale_to_fp32,
)
from common.host_utils import (
    compare_and_report_mismatches,
    kind_data_dtype,
    kind_sf_vec_size,
    mxfp8_quantize_per_block_32_row,
)


class Sm120SwapABSwigluMxfp8Fc12Tester(Fc12TesterBase):
    """MXFP8 host-side input/reference/launch/validation driver."""

    @property
    def _epilogue_token_tile(self) -> int:
        return self.impl.mma_tiler_mnk[1]

    def _create_fp8_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Create finite FP8 input data for correctness or perf runs."""
        if self.misc.perf_run:
            n = 1
            for s in shape:
                n *= s
            if dtype == torch.float8_e4m3fn:
                idx = torch.randint(0, 254, (n,), device="cuda")
                flat_bytes = torch.where(idx < 127, idx, idx + 1).to(torch.uint8)
            elif dtype == torch.float8_e5m2:
                idx = torch.randint(0, 248, (n,), device="cuda")
                flat_bytes = torch.where(idx < 124, idx, idx + 4).to(torch.uint8)
            else:
                raise ValueError(f"Unsupported fp8 dtype: {dtype}")
            return flat_bytes.view(dtype).reshape(shape)

        fp32 = torch.zeros(shape, dtype=torch.float32, device="cuda")
        rand = torch.rand(shape, device="cuda")
        fp32[rand < 0.10] = 1.0
        fp32[(rand >= 0.10) & (rand < 0.20)] = -1.0
        return fp32.to(dtype)

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # Keep the swap-AB MegaMoE storage contract: topk=1 axis is present
        # even though the lean fc12 path only writes one output row per token.
        return (data_total_rows, 1, self.problem.hidden)

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        data_dtype = kind_data_dtype(problem.kind)

        self.activation = self._create_fp8_tensor(
            (data_total_rows, problem.hidden), data_dtype
        )
        # Store RHS operands in the same K-major view used by the skeleton path:
        # logical B[N, K] is physically generated as contiguous (N, K), then
        # exposed to the kernel as (K, N) with K stride-1.
        self.fc1_weight = self._create_fp8_tensor(
            (problem.experts, problem.intermediate, problem.hidden), data_dtype
        ).permute(0, 2, 1)
        self.fc2_weight = self._create_fp8_tensor(
            (problem.experts, problem.hidden, problem.intermediate // 2), data_dtype
        ).permute(0, 2, 1)

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        problem = self.problem
        fc2_output_bytes = torch.full(
            (data_total_rows, problem.hidden * problem.fc2_output_dtype.itemsize),
            0xFF,
            dtype=torch.uint8,
            device="cuda",
        )
        self.fc2_output = fc2_output_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, 1, problem.hidden
        )

    def _quantize_fc1(
        self, swiglu: torch.Tensor, norm_const_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return mxfp8_quantize_per_block_32_row(
            swiglu, kind_data_dtype(self.problem.kind)
        )

    def _apply_topk_post_fc2(
        self, fc2_fp32: torch.Tensor, topk_slice: torch.Tensor
    ) -> torch.Tensor:
        if self.misc.ref_compute_graph == "transformers":
            return fc2_fp32 * topk_slice.unsqueeze(-1)
        return fc2_fp32

    def _instantiate_kernel(self, common_kwargs: dict):
        import cutlass

        kind_to_cutlass_dtype = {
            "mxfp8_e4m3": cutlass.Float8E4M3FN,
            "mxfp8_e5m2": cutlass.Float8E5M2,
        }
        kernel_kwargs = dict(
            **common_kwargs,
            ab_dtype=kind_to_cutlass_dtype[self.problem.kind],
            fc2_output_dtype=cutlass.BFloat16,
            non_ubulk_fc2_store=self.impl.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=self.impl.in_kernel_fc2_reduce,
            gate_up_clamp=self.problem.gate_up_clamp,
            epi_flag_batch=self.impl.epi_flag_batch,
        )
        from moe_sm120_mxfp8_swapab.kernel_fc12 import (
            Sm120SwapABSwigluMxfp8Fc12Kernel,
        )

        return Sm120SwapABSwigluMxfp8Fc12Kernel(**kernel_kwargs)

    def _fc2_tolerance(self) -> Tuple[float, float]:
        if self.problem.kind == "mxfp8_e5m2":
            # E5M2's two-bit mantissa makes the QMMA-vs-reference accumulation
            # order visible around zero even after the BF16 output conversion.
            return 2e-3, 1e-2
        return 1e-5, 1e-2

    def _validate_fc1_phase(self) -> None:
        if (
            self._ws_fc1_output_torch is None
            or self._ws_fc1_output_sf_torch is None
            or not self._ref_fc1_q_per_expert
        ):
            print("[fc1 phase ablation] skipped (workspace or ref not populated)")
            return

        sf_vec_size = kind_sf_vec_size(self.problem.kind)
        k_sf = (self.problem.intermediate // 2) // sf_vec_size
        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets
        soff = self.sf_physical_offsets

        print("\n" + "=" * 60)
        print("[DEBUG fc1] MXFP8 fc1 hand-off per expert:")
        for e in range(self.problem.experts):
            v_e = valid[e]
            if v_e == 0:
                continue
            ref_q = self._ref_fc1_q_per_expert[e]
            ref_sf = self._ref_fc1_raw_sf_per_expert[e]
            if ref_q is None or ref_sf is None:
                continue

            kernel_q = self._ws_fc1_output_torch[doff[e] : doff[e] + v_e]
            kernel_sf_swizzled = self._ws_fc1_output_sf_torch[soff[e] : soff[e + 1]]
            kernel_sf = from_blocked(kernel_sf_swizzled.contiguous().view(-1), v_e, k_sf)
            kernel_fp32 = dequant_block_scale_to_fp32(
                kernel_q, kernel_sf, sf_vec_size, None
            )
            ref_fp32 = dequant_block_scale_to_fp32(ref_q, ref_sf, sf_vec_size, None)
            print(f"[expert {e}] data rows={v_e}")
            compare_and_report_mismatches(
                kernel_fp32.cpu(),
                ref_fp32.cpu(),
                name=f"fc1_e{e}_dequant",
                atol=5e-2,
                rtol=5e-2,
                max_mismatches=5,
            )
        print("=" * 60)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE SM120 MXFP8 Swap-AB fused fc1+fc2 SwiGLU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_fc12_arguments(parser)
    parser.set_defaults(mma_tiler_mnk="64,64,128", cluster_shape_mnk="1,1,1")

    parser.add_argument(
        "--kind",
        type=str,
        default="mxfp8_e4m3",
        choices=["mxfp8_e4m3", "mxfp8_e5m2"],
        help="MXFP8 element format.",
    )
    parser.add_argument(
        "--flag_batch",
        type=int,
        default=1,
        help="dispatch_pull release-flag batch size.",
    )
    parser.add_argument(
        "--epi_flag_batch",
        type=str,
        default="1,1",
        help="(fc1,fc2) done-counter publish batch in comma form.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="Optional DeepSeek-V4 SwiGLU clamp.",
    )
    parser.add_argument(
        "--use_bulk_fc2_store",
        action="store_true",
        default=False,
        help="Use bulk fc2 store path instead of STG.",
    )
    parser.add_argument(
        "--in_kernel_fc2_reduce",
        action="store_true",
        default=False,
        help="Reduce topk in-kernel via fc2 atomic add.",
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
        gate_up_clamp=args.gate_up_clamp,
        kind=args.kind,
    )

    impl = ImplDesc(
        mma_tiler_mnk=parse_tuple(args.mma_tiler_mnk),
        cluster_shape_mnk=parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=False,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        non_ubulk_fc2_store=not args.use_bulk_fc2_store,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        flag_batch=args.flag_batch,
        epi_flag_batch=parse_tuple(args.epi_flag_batch),
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
    )

    tester = Sm120SwapABSwigluMxfp8Fc12Tester(problem, impl, misc)
    tester.run()


if __name__ == "__main__":
    main()
