# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FP4 NVF4 FC1 BatchedGemm TS kernel.

FC1 FP4 combines:
  - TMA gather4 routing for activations (routeAct=tma)
  - FP4 E2M1 data with E4M3 scale factors
  - Block-scaled MMA (MXF4NVF4)
  - SwiGLU activation in epilogue

Covers variant table rows 0-23 (FC1_LL, tile_k=256 subset).

Run with: pytest tests/prims_ts/test_batched_gemm_fp4_fc1.py -v
Requires: CUDA GPU with SM100A+ (B200 Blackwell)
"""

import pytest

import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
    ActKind,
    BatchMode,
    BiasType,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="FP4 cvt (cvt.e2m1x2.f32) requires Blackwell sm_100+",
    ),
]

# FP4 FC1 LowLatency base config (rows 0-23).
FP4_FC1_LL = dict(
    route_act=1,
    act_kind=1,  # FC1, TMA route, SwiGLU
    route_sfs_act=0,  # NONE: SF uses padded R128c4 layout (default, working)
    sf_layout_b=1,
    cluster_m=1,
    tile_m=128,
    mma_m=128,
    mma_k=64,
    dtype_a=int(DType.E2M1),
    dtype_b=int(DType.E2M1),
    dtype_c=int(DType.BF16),
    sf_bits=8,
    epilogue_regs=128,
    mma_regs=48,
    load_regs=48,
    load_sf_regs=48,
    copy_sf_regs=48,
    workid_regs=48,
    padding_regs=48,
    gather_regs=48,
    use_unroll_loop_2x_for_mma=0,
    use_max_tmem_overlap=0,
    batch_mode=int(BatchMode.BATCH_M),
    transpose_mma_output=0,
)


def _run_fp4_fc1(
    *,
    tile_n,
    tile_k=256,
    pipeline_stages=5,
    tile_scheduler=0,
    num_experts=2,
    num_tokens=256,
    top_k=1,
    swap_ab=False,
    act_kind=1,
    problem_n=None,
    route_sfs_act=0,
    cluster_m=1,
    mma_m=128,
    epi_tile_n=None,
    sf_layout_b=1,
    unroll=0,
    bias_type=0,
    gemm1_clamp_limit_value=2.0,
    scale_c_value=1.0,
    scale_gate_value=1.0,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    if epi_tile_n is None:
        epi_tile_n = tile_n
    num_stages_tmem_acc = 2 if tile_scheduler == 1 else 1

    base = {
        k: v
        for k, v in FP4_FC1_LL.items()
        if k
        not in (
            "act_kind",
            "route_sfs_act",
            "cluster_m",
            "mma_m",
            "sf_layout_b",
            "use_unroll_loop_2x_for_mma",
            "batch_mode",
            "transpose_mma_output",
        )
    }
    result = reference_check(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        problem_n=problem_n,
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=epi_tile_n,
        tile_k=tile_k,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        tile_scheduler=tile_scheduler,
        num_stages_tmem_acc=num_stages_tmem_acc,
        batch_mode=int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
        transpose_mma_output=int(swap_ab),
        act_kind=act_kind,
        route_sfs_act=route_sfs_act,
        cluster_m=cluster_m,
        mma_m=mma_m,
        sf_layout_b=sf_layout_b,
        use_unroll_loop_2x_for_mma=unroll,
        bias_type=bias_type,
        has_gemm1_clamp_limit=int(act_kind == int(ActKind.SWIGLU)),
        scale_c_value=scale_c_value,
        scale_gate_value=scale_gate_value,
        gemm1_clamp_limit_value=gemm1_clamp_limit_value,
        **base,
    )
    assert result, f"FP4 FC1 failed: tile_n={tile_n}, tile_k={tile_k}, swap={swap_ab}"


class TestFp4Fc1Validation:
    def test_validate_tile8_k256(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=256,
            **uniform_pipeline_stage_overrides(5),
            num_stages_tmem_acc=1,
            **FP4_FC1_LL,
        )

    def test_implicit_routed_sfb_warp_layout_matches_generated_k512(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            ActKind,
            BatchMode,
            RouteImpl,
            TileScheduler,
            compute_warp_layout,
            make_config,
        )

        common = dict(
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            route_act=int(RouteImpl.TMA),
            route_sfs_act=int(RouteImpl.NONE),
            act_kind=int(ActKind.SWIGLU),
            tile_m=128,
            tile_k=512,
            mma_k=64,
            dtype_a=int(DType.E2M1),
            dtype_b=int(DType.E2M1),
            dtype_c=int(DType.E2M1),
            **uniform_pipeline_stage_overrides(5),
            num_stages_tmem_acc=2,
            tile_scheduler=int(TileScheduler.PERSISTENT),
            use_early_exit=1,
        )

        tile8 = make_config(tile_n=8, mma_n=8, mma_m=128, cluster_m=1, **common)
        compute_warp_layout(tile8)
        assert tile8.num_load_b_warps == 2
        assert tile8.num_load_sfb_warps == 2
        assert tile8.num_padding_warps == 3
        assert tile8.threads_per_cta == 20 * 32

        tile16 = make_config(tile_n=16, mma_n=16, mma_m=256, cluster_m=2, **common)
        compute_warp_layout(tile16)
        assert tile16.num_load_b_warps == 2
        assert tile16.num_load_sfb_warps == 4
        assert tile16.num_padding_warps == 1
        assert tile16.threads_per_cta == 20 * 32


class TestFp4Fc1GPU:
    """FP4 FC1 GPU: TMA gather4 + FP4 MMA + SwiGLU."""

    def test_tile8_noact_noswap(self):
        _run_fp4_fc1(tile_n=8, swap_ab=False, act_kind=0)

    def test_tile8_swiglu_noswap(self):
        _run_fp4_fc1(tile_n=8, swap_ab=False, act_kind=1)

    def test_tile8_swiglu_bias_m(self):
        _run_fp4_fc1(tile_n=8, swap_ab=False, act_kind=1, bias_type=1)

    def test_tile8_swiglu_clamp(self):
        _run_fp4_fc1(
            tile_n=8,
            swap_ab=False,
            act_kind=1,
            gemm1_clamp_limit_value=0.5,
        )

    def test_tile8_swiglu_global_scale_gate(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 8
        tile_k = 256
        base = {k: v for k, v in FP4_FC1_LL.items() if k != "act_kind"}
        common = dict(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            act_kind=1,
            return_output=True,
            **base,
        )
        ok_base, out_base = reference_check(
            scale_c_value=1.0, scale_gate_value=1.0, **common
        )
        ok_scaled, out_scaled = reference_check(
            scale_c_value=1.0, scale_gate_value=0.5, **common
        )
        assert ok_base and ok_scaled
        assert (out_base.float() - out_scaled.float()).abs().max().item() > 0.0

    def test_tile16_swiglu_noswap(self):
        _run_fp4_fc1(tile_n=16, swap_ab=False, act_kind=1)

    def test_tile8_noact_swap(self):
        _run_fp4_fc1(tile_n=8, swap_ab=True, act_kind=0, pipeline_stages=9)

    def test_tile8_swiglu_swap(self):
        _run_fp4_fc1(tile_n=8, swap_ab=True, act_kind=1, pipeline_stages=9)

    def test_tile8_swiglu_swap_epilogue_quant(self):
        """FC1 can return packed E2M1 C plus R128c4 E4M3 output SF."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 8
        tile_k = 256
        base = {
            k: v
            for k, v in FP4_FC1_LL.items()
            if k
            not in (
                "act_kind",
                "dtype_c",
                "batch_mode",
                "transpose_mma_output",
            )
        }
        common = dict(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(9),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            act_kind=1,
            return_output=True,
            seed=123,
            **base,
        )
        ok_bf16, out_bf16 = reference_check(
            dtype_c=int(DType.BF16),
            **common,
        )
        ok_fp4, out_fp4 = reference_check(
            dtype_c=int(DType.E2M1),
            **common,
        )
        assert ok_bf16 and ok_fp4
        diff = (out_fp4.float() - out_bf16.float()).abs()
        tolerance = max(0.75, out_bf16.float().abs().max().item() * 0.35)
        assert diff.max().item() < tolerance

    def test_tile8_swiglu_swap_epilogue_quant_per_token_sfb(self):
        """NVFP4 FC1 applies per-token SFB before bias in the E2M1 epilogue."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 8
        tile_k = 512
        base = {
            k: v
            for k, v in FP4_FC1_LL.items()
            if k
            not in (
                "act_kind",
                "batch_mode",
                "dtype_c",
                "sf_layout_b",
                "transpose_mma_output",
            )
        }
        assert reference_check(
            num_experts=2,
            num_tokens=16,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            act_kind=int(ActKind.SWIGLU),
            dtype_c=int(DType.E2M1),
            sf_layout_b=int(SfLayout.LINEAR),
            sf_layout_c=int(SfLayout.R8c4),
            use_global_scales=1,
            use_per_token_sf_b=1,
            per_token_sf_dtype=int(DType.FP32),
            bias_type=int(BiasType.M),
            **base,
        )

    def test_tile32_k512_swap_epilogue_quant_second_kbox(self):
        """Regression for the second FP4 K-box B descriptor stride."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        base = {
            k: v
            for k, v in FP4_FC1_LL.items()
            if k
            not in (
                "act_kind",
                "batch_mode",
                "dtype_c",
                "epilogue_regs",
                "sf_layout_b",
                "transpose_mma_output",
            )
        }
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            problem_n=768,
            problem_k=1024,
            tile_n=32,
            mma_n=32,
            epi_tile_n=32,
            tile_k=512,
            **uniform_pipeline_stage_overrides(4),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            act_kind=1,
            dtype_c=int(DType.E2M1),
            epilogue_regs=160,
            sf_layout_b=int(SfLayout.LINEAR),
            use_global_scales=1,
            seed=123,
            **base,
        )
        assert result, "FP4 FC1 tileN32/tileK512 second K-box regression failed"

    def test_tile8_4experts(self):
        _run_fp4_fc1(tile_n=8, num_experts=4, num_tokens=512)

    def test_tile8_swiglu_swap_nonrounded_topk(self):
        _run_fp4_fc1(
            tile_n=8,
            num_experts=3,
            num_tokens=130,
            top_k=2,
            swap_ab=True,
            act_kind=1,
            pipeline_stages=9,
        )


class TestFp4Fc1SfGather:
    """FP4 FC1 with SF gather for the routed activation's scale factors."""

    _BASE = {k: v for k, v in FP4_FC1_LL.items() if k != "route_sfs_act"}

    def test_sf_tma_gather4_blocked_at_k256(self):
        """SF TMA gather4 requires sf_k >= 32 for the routed operand."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        with pytest.raises(ValueError, match="SF TMA gather4 requires sf_k >= 32"):
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                tile_n=8,
                mma_n=8,
                epi_tile_n=8,
                tile_k=256,
                **uniform_pipeline_stage_overrides(5),
                num_stages_tmem_acc=1,
                route_sfs_act=1,  # TMA — blocked at tile_k=256
                **self._BASE,
            )

    def test_sf_tma_gather4_k512_validation(self):
        """SF TMA gather4 at tile_k=512: validation passes."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=512,
            **uniform_pipeline_stage_overrides(5),
            num_stages_tmem_acc=1,
            route_sfs_act=1,  # TMA gather4 — OK at tile_k=512
            **self._BASE,
        )

    def test_sf_tma_gather4_k512_gpu(self):
        """SF TMA gather4 at tile_k=512: GPU correctness."""
        base = {
            k: v
            for k, v in self._BASE.items()
            if k not in ("act_kind", "batch_mode", "transpose_mma_output")
        }
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=512,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            batch_mode=int(BatchMode.BATCH_M),
            transpose_mma_output=0,
            act_kind=0,
            route_sfs_act=1,  # TMA gather4
            **base,
        )
        assert result, "FP4 FC1 SF TMA gather4 at tile_k=512 failed"

    def test_sf_ldgsts_gather_validation(self):
        """SF LDGSTS gather: schedule validation passes."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=256,
            **uniform_pipeline_stage_overrides(5),
            num_stages_tmem_acc=1,
            route_sfs_act=2,  # LDGSTS
            **self._BASE,
        )

    def test_sf_ldgsts_gather_gpu(self):
        """SF LDGSTS gather at tile_k=256: GPU correctness."""
        base = {
            k: v
            for k, v in self._BASE.items()
            if k not in ("act_kind", "batch_mode", "transpose_mma_output")
        }
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=256,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            batch_mode=int(BatchMode.BATCH_M),
            transpose_mma_output=0,
            act_kind=0,
            route_sfs_act=2,  # LDGSTS
            **base,
        )
        assert result, "FP4 FC1 SF LDGSTS gather at tile_k=256 failed"

    def test_sf_ldgsts_gather_swap_gpu(self):
        """SwapAB SF LDGSTS gather: covers routed SFB + separate CopySf."""
        _run_fp4_fc1(
            tile_n=8,
            tile_k=256,
            pipeline_stages=9,
            route_sfs_act=2,
            swap_ab=True,
            act_kind=0,
        )

    def test_sf_tma_gather4_swap_k512_gpu(self):
        """SwapAB SF TMA gather4 at tile_k=512: covers routed SFB via TMA."""
        _run_fp4_fc1(
            tile_n=8,
            tile_k=512,
            pipeline_stages=5,
            route_sfs_act=1,  # TMA gather4
            swap_ab=True,
            act_kind=0,
        )


class TestFp4Fc1HT:
    """FP4 FC1 HighThroughput: 2-CTA cluster with routed activation SF."""

    def test_ht_tile64_k256_ldgsts_validation(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        base = {
            k: v
            for k, v in TestFp4Fc1SfGather._BASE.items()
            if k
            not in ("cluster_m", "mma_m", "sf_layout_b", "use_unroll_loop_2x_for_mma")
        }
        for unroll in (0,):
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                tile_n=64,
                mma_n=64,
                epi_tile_n=32,
                tile_k=256,
                **uniform_pipeline_stage_overrides(6),
                tile_scheduler=1,
                num_stages_tmem_acc=2,
                route_sfs_act=2,
                cluster_m=2,
                mma_m=256,
                sf_layout_b=2,
                use_unroll_loop_2x_for_mma=unroll,
                **base,
            )

    def test_ht_tile64_k256_ldgsts_gpu(self):
        """2-CTA compact SFB STTM path that previously hung without debug prints."""
        for unroll in (0,):
            _run_fp4_fc1(
                tile_n=64,
                tile_k=256,
                pipeline_stages=6,
                tile_scheduler=1,
                route_sfs_act=2,
                cluster_m=2,
                mma_m=256,
                epi_tile_n=32,
                sf_layout_b=2,
                swap_ab=True,
                unroll=unroll,
            )

    def test_ht_tile128_k256_ldgsts_validation(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        base = {
            k: v
            for k, v in TestFp4Fc1SfGather._BASE.items()
            if k not in ("cluster_m", "mma_m", "sf_layout_b")
        }
        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=128,
            mma_n=128,
            epi_tile_n=32,
            tile_k=256,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=1,
            num_stages_tmem_acc=2,
            route_sfs_act=2,
            cluster_m=2,
            mma_m=256,
            sf_layout_b=2,
            **base,
        )

    def test_ht_tile128_k256_ldgsts_gpu(self):
        """2-CTA HT routed-SF LDGSTS path that previously produced NaNs."""
        _run_fp4_fc1(
            tile_n=128,
            tile_k=256,
            pipeline_stages=6,
            tile_scheduler=1,
            route_sfs_act=2,
            cluster_m=2,
            mma_m=256,
            epi_tile_n=32,
            sf_layout_b=2,
            swap_ab=True,
        )

    def test_ht_tile256_k256_ldgplussts_rejected(self):
        """LDG+STS route-SF parsing is explicit but not implemented in TS."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            RouteImpl,
            SfLayout,
        )
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        with pytest.raises(ValueError, match="LDG_PLUS_STS"):
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                batch_mode=int(BatchMode.BATCH_N),
                transpose_mma_output=1,
                route_act=int(RouteImpl.TMA),
                route_sfs_act=int(RouteImpl.LDG_PLUS_STS),
                act_kind=1,
                tile_m=128,
                tile_n=256,
                tile_k=256,
                epi_tile_m=128,
                epi_tile_n=64,
                mma_m=256,
                mma_n=256,
                mma_k=64,
                cluster_m=2,
                dtype_a=int(DType.E2M1),
                dtype_b=int(DType.E2M1),
                dtype_c=int(DType.BF16),
                sf_bits=8,
                **uniform_pipeline_stage_overrides(4),
                tile_scheduler=1,
                num_stages_tmem_acc=1,
                sf_layout_b=int(SfLayout.R128c4),
                use_max_tmem_overlap=1,
                epilogue_regs=144,
                mma_regs=88,
                load_regs=32,
                load_sf_regs=40,
                workid_regs=88,
                padding_regs=88,
            )


class TestFp4Fc1SfbTmaRouteCluster2:
    """The two routed-SFB (routeSfsAct=tma) cluster_m=2 benchmark configs.

    These are the swapAB FC1 SwiGLU shapes that motivated routing operand B's
    scale factors through TMA gather4 (Linear SMEM SFB) under a 2-CTA cluster,
    with FP4 (E2M1) output. They also exercise the Kaiming-init + block-scaled
    quantization regime, without which the SwiGLU clamp amplifies accumulator
    cancellation into sparse ref-check failures on the larger (256-token) shape.

    num_experts is reduced from the benchmark's 64 to keep the test light; the
    routed-SFB feature and the cancellation scenario depend on token count and
    cluster shape, not expert count.
    """

    @staticmethod
    def _base():
        return dict(
            num_experts=8,
            top_k=8,
            problem_n=2048,
            problem_k=7168,
            seed=42,
            act_kind=int(ActKind.SWIGLU),
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            bias_type=int(BiasType.M),
            has_gemm1_clamp_limit=1,
            cluster_m=2,
            dtype_a=int(DType.E2M1),
            dtype_b=int(DType.E2M1),
            dtype_c=int(DType.E2M1),
            epi_tile_n=32,
            tile_k=512,
            tile_m=128,
            mma_m=256,
            mma_k=64,
            route_act=int(RouteImpl.TMA),
            route_sfs_act=int(RouteImpl.TMA),
            sf_layout_a=int(SfLayout.R128c4),
            sf_layout_b=int(SfLayout.LINEAR),
            sf_layout_c=int(SfLayout.R8c4),
            tile_scheduler=int(TileScheduler.PERSISTENT),
            use_global_scales=1,
            use_clc_fast_drain=0,
            use_early_exit=1,
            use_max_tmem_overlap=0,
            use_tma_oob_opt=1,
            epilogue_regs=160,
            mma_regs=48,
            load_regs=48,
            load_sf_regs=48,
            copy_sf_regs=48,
            workid_regs=48,
            padding_regs=48,
        )

    def test_tile_n32_num_tokens4(self):
        """Case 1: tile_n=32, num_tokens=4 (small shape)."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

        result = reference_check(
            num_tokens=4,
            tile_n=32,
            mma_n=32,
            **uniform_pipeline_stage_overrides(5, tmem_acc_stages=2),
            **self._base(),
        )
        assert result, "routed-SFB tma cluster2 tile_n=32 (4 tokens) failed"

    def test_tile_n64_num_tokens256(self):
        """Case 2: tile_n=64, num_tokens=256 (large shape; cancellation-prone)."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

        result = reference_check(
            num_tokens=256,
            tile_n=64,
            mma_n=64,
            **uniform_pipeline_stage_overrides(4, tmem_acc_stages=2),
            **self._base(),
        )
        assert result, "routed-SFB tma cluster2 tile_n=64 (256 tokens) failed"
