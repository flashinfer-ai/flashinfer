# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host driver for the MegaMoE NVFP4 swap-AB fused fc1+fc2 kernel."""

import argparse
import os
import sys
from dataclasses import dataclass, field  # noqa: F401
from typing import List, Literal, Optional, Tuple  # noqa: F401

import numpy as np  # noqa: F401
import torch

# Ensure absolute package imports work when this file is run as a script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Re-export the shared descriptors so legacy importers (e.g. mega_runner.py)
# continue to find them at ``moe_nvfp4_swapab.runner_fc12``.
from moe_nvfp4_swapab.runner_fc12_common import (
    ProblemDesc,
    ImplDesc,
    MiscDesc,
    Fc12TesterBase,
    add_common_fc12_arguments,
    parse_tuple,
)
from moe_nvfp4_swapab.epilogue_refactor import SwapABSwigluFp4Epilogue
from moe_nvfp4_swapab.runner_common import (
    Nvfp4DataDtype,
    dequant_block_scale_to_fp32,
    from_blocked,
    nvfp4_quantize_per_block_16,
)
from common.host_utils import (
    kind_data_dtype,
    kind_sf_vec_size,
)


# =============================================================================
# NVFP4 tester
# =============================================================================


class SwapABSwigluFp4Fc12Tester(Fc12TesterBase):
    """NVFP4 host-side input/reference/launch/validation driver."""

    @property
    def _epilogue_token_tile(self) -> int:
        # Sourced from the NVFP4 epilogue class (C.8), not a runner hardcode.
        return SwapABSwigluFp4Epilogue._EpilogueTokenTileSize

    # ------------------------------------------------------------------
    # FP4 tensor creation
    # ------------------------------------------------------------------

    def _create_fp4_tensor(
        self, logical_shape: Tuple[int, ...], packed_dim: int = -1
    ) -> torch.Tensor:
        """Create a ``float4_e2m1fn_x2`` tensor.

        Two modes (selected by ``misc.perf_run``):
          - correctness: sparse nibbles from {0x0, 0x2, 0xA} (= {0, +1, -1}),
            80 % zeros / 10 % +1 / 10 % -1.  Keeps GEMM absmax low enough
            that SwiGLU output stays below fp8e4m3fn SF saturation (= 448).
          - perf: random uint8 bytes (FP4 has no NaN; every byte is a valid
            packed pair).  Used for timing.

        ``packed_dim`` is the logical dim that becomes stride-1 (i.e. halved
        in the byte buffer).  The size along that dim must be even.
        """
        ndim = len(logical_shape)
        packed_dim = packed_dim % ndim
        if logical_shape[packed_dim] % 2 != 0:
            raise ValueError(
                f"packed_dim {packed_dim} size ({logical_shape[packed_dim]}) must be even."
            )

        if self.misc.perf_run:
            elem_cnt = 1
            for s in logical_shape:
                elem_cnt *= s
            byte_cnt = elem_cnt // 2
            flat = torch.randint(0, 256, (byte_cnt,), dtype=torch.uint8, device="cuda")

            shape_reordered = list(logical_shape)
            need_perm = packed_dim != ndim - 1
            if need_perm:
                shape_reordered[packed_dim], shape_reordered[-1] = (
                    shape_reordered[-1],
                    shape_reordered[packed_dim],
                )
            shape_reordered[-1] //= 2
            tensor = flat.view(Nvfp4DataDtype).reshape(shape_reordered)
            if need_perm:
                perm = list(range(ndim))
                perm[packed_dim], perm[-1] = perm[-1], perm[packed_dim]
                tensor = tensor.permute(perm)
            return tensor

        # -- Correctness mode --
        rand_u8 = torch.randint(0, 100, logical_shape, dtype=torch.uint8, device="cuda")
        nibbles = torch.zeros_like(rand_u8)  # default 0 (nibble 0x0)
        nibbles.masked_fill_((rand_u8 >= 80) & (rand_u8 < 90), 0x2)  # 10 %: +1
        nibbles.masked_fill_(rand_u8 >= 90, 0xA)  # 10 %: -1

        need_perm = packed_dim != ndim - 1
        if need_perm:
            perm_to_last = list(range(ndim))
            perm_to_last[packed_dim], perm_to_last[-1] = (
                perm_to_last[-1],
                perm_to_last[packed_dim],
            )
            nibbles = nibbles.permute(perm_to_last).contiguous()

        even = nibbles[..., ::2]
        odd = nibbles[..., 1::2]
        packed_uint8 = (odd << 4) | even
        tensor = packed_uint8.view(Nvfp4DataDtype)

        if need_perm:
            inv_perm = list(range(ndim))
            inv_perm[packed_dim], inv_perm[-1] = inv_perm[-1], inv_perm[packed_dim]
            tensor = tensor.permute(inv_perm)
        return tensor

    # ------------------------------------------------------------------
    # Kind hooks: input / output tensor creation
    # ------------------------------------------------------------------

    def _fc2_output_shape(self, data_total_rows: int) -> Tuple[int, ...]:
        # MoE-domain layout ``(token_max, topk, hidden)``: lean fc1+fc2
        # collapses topk=1 (codegen-time const), storage is row-major and
        # therefore byte-equivalent to the legacy ``(M, H)`` form -- the
        # second axis only carries kernel-side stride metadata used by the
        # fc2 return tile to address one output cell per pool token row
        # (topk index always 0 in lean mode).  Validation / display paths
        # squeeze axis 1 to recover ``(M, H)`` for comparison.
        return (data_total_rows, 1, self.problem.hidden)

    def _create_input_data_tensors(self, data_total_rows: int) -> None:
        problem = self.problem
        hidden = problem.hidden
        intermediate = problem.intermediate
        experts = problem.experts
        data_dtype = kind_data_dtype(problem.kind)

        # -- activation --
        if data_total_rows > 0:
            self.activation = self._create_fp4_tensor(
                (data_total_rows, hidden), packed_dim=-1
            )
        else:
            self.activation = torch.empty(
                (0, hidden // 2),
                dtype=torch.uint8,
                device="cuda",
            ).view(data_dtype)

        # -- fc1_weight --
        self.fc1_weight = self._create_fp4_tensor(
            (experts, hidden, intermediate), packed_dim=1
        )

        # -- fc2_weight --
        self.fc2_weight = self._create_fp4_tensor(
            (experts, intermediate // 2, hidden), packed_dim=1
        )

        # -- per-expert fc1/fc2 alpha (always random for the NVFP4 tester,
        #    mirroring mega_runner): real-value rescale folded into the
        #    epilogue, fed to both the kernel runtime and the reference core.
        self.fc1_alpha = (
            torch.randint(1, 5, (experts,), device="cuda").to(torch.float32) * 0.5
        )
        self.fc2_alpha = (
            torch.randint(1, 5, (experts,), device="cuda").to(torch.float32) * 0.5
        )

    def _alloc_fc2_output(self, data_total_rows: int) -> None:
        # 0xFF byte fill: bf16 0xFFFF = NaN, fp16 0xFFFF = NaN -- kernel
        # output overwriting valid rows is easy to distinguish from "kernel
        # never touched this row".  Storage shape carries the lean
        # ``topk=1`` axis demanded by the MoE-domain fc2 return tile contract;
        # byte layout is identical to the legacy 2D form.
        problem = self.problem
        hidden = problem.hidden
        fc2_output_bytes = torch.full(
            (data_total_rows, hidden * problem.fc2_output_dtype.itemsize),
            0xFF,
            dtype=torch.uint8,
            device="cuda",
        )
        self.fc2_output = fc2_output_bytes.view(problem.fc2_output_dtype).reshape(
            data_total_rows, 1, hidden
        )

    # ------------------------------------------------------------------
    # Kind hooks: reference compute
    # ------------------------------------------------------------------

    def _quantize_fc1(
        self, swiglu: torch.Tensor, norm_const_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_quantize_per_block_16(swiglu, norm_const_val)

    def _apply_topk_post_fc2(
        self, fc2_fp32: torch.Tensor, topk_slice: torch.Tensor
    ) -> torch.Tensor:
        # HF transformers semantics: topk weight applied AFTER fc2 GEMM.
        if self.misc.ref_compute_graph == "transformers":
            return fc2_fp32 * topk_slice.unsqueeze(-1)
        return fc2_fp32

    # ------------------------------------------------------------------
    # Kind hooks: kernel + validation
    # ------------------------------------------------------------------

    def _instantiate_kernel(self, common_kwargs: dict):
        import cutlass
        from moe_nvfp4_swapab.kernel_fc12 import Sm100SwapABSwigluFp4Fc12Kernel

        return Sm100SwapABSwigluFp4Fc12Kernel(
            **common_kwargs,
            fc2_output_dtype=cutlass.BFloat16,
            non_ubulk_fc2_store=self.impl.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=self.impl.in_kernel_fc2_reduce,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            gate_up_clamp=self.problem.gate_up_clamp,
            epi_flag_batch=self.impl.epi_flag_batch,
        )

    def _fc2_tolerance(self) -> Tuple[float, float]:
        return 5e-2, 5e-2

    def _validate_fc1_phase(self) -> None:
        """fc1 NVFP4 hand-off ablation: dequant kernel vs ref and compare values.

        The kernel's fc1 nibble *byte packing* need not match the host
        ``nvfp4_quantize`` packing, so a raw byte compare is meaningless; instead
        dequantize both sides (``from_blocked`` un-swizzles the kernel SF) and
        compare fp32 values.  With a bit-exact reference (fc1 also runs the real
        blockscaled GEMM) any remaining diff is just fp4 RTNE near bin
        boundaries, so this should be tiny.
        """
        if (
            self._ws_fc1_output_torch is None
            or self._ws_fc1_output_sf_torch is None
            or not self._ref_fc1_q_per_expert
        ):
            print("[fc1 phase ablation] skipped (workspace or ref not populated)")
            return

        intermediate_downproj = self.problem.intermediate // 2
        sf_vec_size = kind_sf_vec_size(self.problem.kind)
        K_sf = intermediate_downproj // sf_vec_size  # raw SF cols per token

        valid_tokens = self.valid_tokens_per_expert
        data_offsets = self.data_physical_offsets
        sf_offsets = self.sf_physical_offsets

        print("=" * 60)
        print("[fc1 phase ablation] kernel vs ref fc1 hand-off (dequant fp32)")
        print(
            f"{'expert':>6} {'v_e':>6} {'max_diff':>12} "
            f"{'mean_diff':>12} {'n_bad@5e-2':>12} {'%bad':>8}"
        )

        total_n = 0
        total_bad = 0
        overall_max = 0.0
        for e in range(self.problem.experts):
            v_e = valid_tokens[e]
            if v_e == 0:
                continue
            ref_q = self._ref_fc1_q_per_expert[e]
            ref_sf = self._ref_fc1_raw_sf_per_expert[e]
            if ref_q is None or ref_sf is None:
                continue
            d_start = data_offsets[e]
            sf_start = sf_offsets[e]
            sf_end = sf_offsets[e + 1]

            kernel_q = self._ws_fc1_output_torch[d_start : d_start + v_e]
            kernel_sf_raw = from_blocked(
                self._ws_fc1_output_sf_torch[sf_start:sf_end].contiguous().view(-1),
                v_e,
                K_sf,
            )
            kernel_fp32 = dequant_block_scale_to_fp32(
                kernel_q, kernel_sf_raw, sf_vec_size, None
            )
            ref_fp32 = dequant_block_scale_to_fp32(ref_q, ref_sf, sf_vec_size, None)

            diff = (kernel_fp32 - ref_fp32).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            n = diff.numel()
            n_bad = int((diff > 5e-2).sum().item())
            total_n += n
            total_bad += n_bad
            overall_max = max(overall_max, max_d)
            print(
                f"{e:>6d} {v_e:>6d} {max_d:>12.4g} {mean_d:>12.4g} "
                f"{n_bad:>12d} {n_bad / max(n, 1) * 100:>7.3f}%"
            )

        print(
            f"{'TOTAL':>6} {'':>6} {overall_max:>12.4g} {'':>12} "
            f"{total_bad:>12d} {total_bad / max(total_n, 1) * 100:>7.3f}%"
        )
        print("=" * 60)


# =============================================================================
# CLI entry point
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    """argparse setup for the NVFP4 fused fc12 path."""
    parser = argparse.ArgumentParser(
        description="MoE NVFP4 Swap-AB fused fc1+fc2 SwiGLU (host-ready harness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_fc12_arguments(parser)

    parser.add_argument(
        "--flag_batch",
        type=int,
        default=1,
        help="dispatch_pull release-flag batch size; 1 == per-token "
        "baseline, larger amortizes the device fence over more tokens.",
    )
    parser.add_argument(
        "--epi_flag_batch",
        type=str,
        default="1,1",
        help="(fc1,fc2) done-counter publish batch in comma form like "
        "--mma_tiler_mnk (warp-cooperative, each in 1..32). E.g. 1,4.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="DeepSeek-V4 swiglu_limit: asymmetric clamp on the real gate/up "
        "pre-activations (gate<=limit, |up|<=limit) before SiLU. "
        "Omitted/None disables it; must be non-negative.",
    )
    parser.add_argument(
        "--use_bulk_fc2_store",
        action="store_true",
        default=False,
        help="Use bulk (cp.async.bulk) fc2 store path instead of STG.256.",
    )
    parser.add_argument(
        "--in_kernel_fc2_reduce",
        action="store_true",
        default=False,
        help="Reduce topk in-kernel via fc2 atomic add (lean runner forbids this).",
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
        kind="nvfp4",
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

    tester = SwapABSwigluFp4Fc12Tester(problem, impl, misc)
    tester.run()


if __name__ == "__main__":
    main()
    exit(0)
