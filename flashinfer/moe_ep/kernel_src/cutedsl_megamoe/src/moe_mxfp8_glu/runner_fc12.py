# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Host driver for the MegaMoE MXFP8 GLU fused fc1+fc2 kernel.
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
    ImplDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_nvfp4_swapab.runner_common import (
    from_blocked,
    dequant_block_scale_to_fp32,
)
from common.host_utils import (
    kind_data_dtype,
    kind_sf_vec_size,
    mxfp8_quantize_per_block_32,
)


# =============================================================================
# MXFP8 tester
# =============================================================================


class SwigluMxfp8Fc12Tester(Fc12TesterBase):
    """MXFP8 host-side input/reference/launch/validation driver."""

    def __init__(
        self,
        problem: ProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
    ) -> None:
        super().__init__(problem, impl, misc)
        # MXFP8 currently only supports the (M=256, N=256) mma tile with
        # 2-CTA instructions.
        m, n, _k = impl.mma_tiler_mnk
        if (m, n) != (256, 256) or not impl.use_2cta_instrs:
            raise ValueError(
                "MXFP8 fused fc12 currently only supports mma_tiler (M, N) = "
                "(256, 256) with use_2cta_instrs=True (the only validated "
                f"config); got mma_tiler_mnk={impl.mma_tiler_mnk}, "
                f"use_2cta_instrs={impl.use_2cta_instrs}."
            )

    # ------------------------------------------------------------------
    # FP8 tensor creation
    # ------------------------------------------------------------------

    def _create_fp8_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create an FP8 (e4m3 / e5m2) tensor.

        Two modes (selected by ``misc.perf_run``):
          - correctness: sparse {0, +1, -1} (80 % zeros / 10 % +1 / 10 % -1).
          - perf: random finite fp8 bytes (timing only); ``randint`` skips
            dtype-specific NaN/Inf encodings (e4m3: 0x7F/0xFF;
            e5m2: 0x7C-0x7F / 0xFC-0xFF).
        """
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

    # ------------------------------------------------------------------
    # Kind hooks: input / output tensor creation
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # MXFP8: 2D (token_max, hidden) -- epilogue_mxfp8.py uses shape[1].
        return (data_total_rows, self.problem.hidden)

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts
        data_dtype = kind_data_dtype(problem.kind)

        # -- activation: (data_total_rows, hidden) fp8, hidden stride-1 --
        self.activation = self._create_fp8_tensor((data_total_rows, hidden), data_dtype)

        # -- fc1_weight: (experts, intermediate, hidden) -> permute ->
        #    (experts, hidden, intermediate), hidden stride-1 --
        self.fc1_weight = self._create_fp8_tensor(
            (experts, intermediate, hidden), data_dtype
        ).permute(0, 2, 1)

        # -- fc2_weight: (experts, hidden, inter//2) -> permute ->
        #    (experts, inter//2, hidden), inter//2 stride-1 --
        self.fc2_weight = self._create_fp8_tensor(
            (experts, hidden, intermediate // 2), data_dtype
        ).permute(0, 2, 1)

    def _init_topk_scores(self, data_total_rows: int) -> None:
        """MXFP8 fc12 does not apply topk weighting; use 1.0 on valid rows."""
        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        self.topk_scores = torch.zeros(
            (data_total_rows,),
            dtype=torch.float32,
            device="cuda",
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
        # row".  MXFP8 stores a flat 2D ``(token_max, hidden)`` output.
        problem = self.problem
        hidden = problem.hidden
        fc2_output_bytes = torch.full(
            (
                data_total_rows,
                hidden
                * torch.tensor([], dtype=problem.fc2_output_dtype).element_size(),
            ),
            0xFF,
            dtype=torch.uint8,
            device="cuda",
        )
        self.fc2_output = fc2_output_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, hidden
        )

    # ------------------------------------------------------------------
    # Kind hooks: reference compute
    # ------------------------------------------------------------------

    def _quantize_fc1(
        self, swiglu: torch.Tensor, norm_const_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_dtype = kind_data_dtype(self.problem.kind)
        return mxfp8_quantize_per_block_32(swiglu, data_dtype)

    # ------------------------------------------------------------------
    # Kind hooks: kernel + validation
    # ------------------------------------------------------------------

    def _instantiate_kernel(self, common_kwargs: dict):
        import cutlass
        from moe_mxfp8_glu.kernel_mxfp8_glu_fc12 import Sm100SwigluMxfp8Fc12Kernel

        _kind_to_cutlass_dtype = {
            "mxfp8_e4m3": cutlass.Float8E4M3FN,
            "mxfp8_e5m2": cutlass.Float8E5M2,
        }
        return Sm100SwigluMxfp8Fc12Kernel(
            **common_kwargs,
            ab_dtype=_kind_to_cutlass_dtype[self.problem.kind],
            gate_up_clamp=self.problem.gate_up_clamp,
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

        sf_vec_size = kind_sf_vec_size(self.problem.kind)
        K_sf = (self.problem.intermediate // 2) // sf_vec_size
        valid = self.valid_tokens_per_expert
        doff = self.data_physical_offsets
        soff = self.sf_physical_offsets

        print("\n" + "=" * 60)
        print("[DEBUG fc1] compare_and_report_mismatches per expert:")
        for e in range(self.problem.experts):
            v_e = valid[e]
            ref_q = self._ref_fc1_q_per_expert[e]
            ref_sf = self._ref_fc1_raw_sf_per_expert[e]
            if v_e == 0 or ref_q is None or ref_sf is None:
                continue
            kq = self._ws_fc1_output_torch[doff[e] : doff[e] + v_e]
            ksf_sw = self._ws_fc1_output_sf_torch[soff[e] : soff[e + 1]]
            ksf = from_blocked(ksf_sw.contiguous().view(-1), v_e, K_sf)
            kfp32 = dequant_block_scale_to_fp32(kq, ksf, sf_vec_size, None)
            ref_fp32 = dequant_block_scale_to_fp32(ref_q, ref_sf, sf_vec_size, None)
            ### TODO: replace to silent check when kernel stable.
            compare_and_report_mismatches(
                kfp32.cpu(),
                ref_fp32.cpu(),
                name=f"fc1_expert{e}",
                atol=5e-2,
                rtol=5e-2,
                max_mismatches=5,
            )
        print("=" * 60)


# =============================================================================
# CLI entry point
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    """argparse setup for the MXFP8 fused fc12 path."""
    parser = argparse.ArgumentParser(
        description="MoE MXFP8 GLU fused fc1+fc2 SwiGLU (host-ready harness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_fc12_arguments(parser)

    # -- MXFP8-only Problem --
    parser.add_argument(
        "--kind",
        type=str,
        default="mxfp8_e4m3",
        choices=["mxfp8_e4m3", "mxfp8_e5m2"],
        help="MXFP8 element format: mxfp8_e4m3 (default) or mxfp8_e5m2.",
    )
    parser.add_argument(
        "--flag_batch",
        type=int,
        default=1,
        help="dispatch_pull release-flag batch size; 1 == per-token "
        "baseline, larger amortizes the device fence over more tokens.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="DeepSeek-V4 swiglu_limit: clamp gate/up pre-activations before SiLU.",
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

    impl = ImplDesc(
        mma_tiler_mnk=parse_tuple(args.mma_tiler_mnk),
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
    )

    tester = SwigluMxfp8Fc12Tester(problem, impl, misc)
    tester.run()


if __name__ == "__main__":
    main()
    exit(0)
