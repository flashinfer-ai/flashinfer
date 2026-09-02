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

"""Tests for BF16×BF16 FC1 BatchedGemm TS kernel with LDGSTS gather.

Run with: pytest tests/prims_ts/test_batched_gemm_bf16_fc1.py -v
Requires: CUDA GPU with SM100+ (B200 Blackwell)
"""

import pytest

import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
    BatchMode,
    DType,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="kernels require Blackwell sm_100+ (skip on sm_120a)",
    ),
]

# Base FC1: LDGSTS gather, non-swapAB. No warp layout fields (kernel computes them).
BF16_FC1_BASE = dict(
    batch_mode=int(BatchMode.BATCH_M),
    transpose_mma_output=0,
    route_act=2,
    tile_scheduler=0,
    act_kind=0,
    sf_layout_b=1,
    cluster_m=1,
    tile_m=128,
    mma_m=128,
    mma_k=16,
    dtype_a=int(DType.BF16),
    dtype_b=int(DType.BF16),
    dtype_c=int(DType.BF16),
    epilogue_regs=160,
    mma_regs=48,
    load_regs=48,
    gather_regs=48,
    padding_regs=48,
    use_unroll_loop_2x_for_mma=0,
    use_max_tmem_overlap=0,
)


class TestBf16Fc1GatherValidation:
    def test_validate_gather_noswap_tile16(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            **{
                k: v
                for k, v in BF16_FC1_BASE.items()
                if k not in ("batch_mode", "transpose_mma_output")
            },
        )
        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            **BF16_FC1_BASE,
        )


class TestBf16Fc1TmaRoute:
    """FC1 with routeAct=tma: activations loaded via TMA gather4."""

    TMA_CFG = {**BF16_FC1_BASE, "route_act": 1}

    def _run(
        self,
        *,
        tile_n=16,
        num_experts=2,
        num_tokens=256,
        top_k=1,
        swap_ab=False,
        act_kind=0,
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **self.TMA_CFG,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
            "act_kind": act_kind,
        }
        result = reference_check(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=top_k,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )
        assert result

    def test_tma_route_noswap(self):
        self._run(swap_ab=False)

    def test_tma_route_swap(self):
        self._run(swap_ab=True)

    def test_tma_route_swiglu_noswap(self):
        self._run(swap_ab=False, act_kind=1)

    def test_tma_route_swiglu_noswap_returns_compact_output(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        ok, output = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            return_output=True,
            **uniform_pipeline_stage_overrides(4),
            **{**self.TMA_CFG, "act_kind": 1},
        )

        assert ok
        assert output.shape == (256, 8)

    def test_tma_route_swiglu_swap(self):
        self._run(swap_ab=True, act_kind=1)

    def test_tma_route_4experts(self):
        self._run(num_experts=4, num_tokens=512)

    def test_tma_route_nonrounded_topk(self):
        self._run(num_experts=3, num_tokens=130, top_k=2, swap_ab=False)


class TestBf16Fc1ActivationValidation:
    def _validate(self, *, act_kind: int, swap_ab: bool) -> None:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        cfg = {
            **BF16_FC1_BASE,
            "act_kind": act_kind,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
        }
        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )

    def test_validate_geglu_noswap_tile16(self):
        self._validate(act_kind=2, swap_ab=False)

    def test_validate_geglu_swap_tile16(self):
        self._validate(act_kind=2, swap_ab=True)

    def test_validate_relu_eltwise_noswap_tile16(self):
        self._validate(act_kind=3, swap_ab=False)

    def test_validate_relu_eltwise_swap_tile16(self):
        self._validate(act_kind=3, swap_ab=True)

    def test_validate_none_eltwise_noswap_tile16(self):
        self._validate(act_kind=0, swap_ab=False)

    def test_validate_none_eltwise_swap_tile16(self):
        self._validate(act_kind=0, swap_ab=True)


class TestBf16Fc1GatherGPU:
    def _run(
        self,
        *,
        tile_n=16,
        pipeline_stages=4,
        num_experts=1,
        num_tokens=128,
        top_k=1,
        swap_ab=False,
        cluster_m=1,
        problem_k=None,
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC1_BASE,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
            "cluster_m": cluster_m,
        }
        if cluster_m >= 2:
            cfg["mma_m"] = 256
            cfg["tile_scheduler"] = 1
            cfg["num_stages_tmem_acc"] = 2
            # num_gather_warps computed by kernel's compute_warp_layout
        result = reference_check(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=top_k,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            problem_k=problem_k,
            **uniform_pipeline_stage_overrides(pipeline_stages),
            **cfg,
        )
        assert result, (
            f"FC1 gather failed: tile_n={tile_n}, swap={swap_ab}, cluster={cluster_m}"
        )

    def test_gather_noswap_tile8(self):
        self._run(tile_n=8, swap_ab=False)

    def test_gather_noswap_tile16(self):
        self._run(tile_n=16, swap_ab=False)

    def test_gather_noswap_tile32(self):
        self._run(tile_n=32, swap_ab=False)

    def test_gather_swap_tile8(self):
        self._run(tile_n=8, swap_ab=True)

    def test_gather_swap_tile16(self):
        self._run(tile_n=16, swap_ab=True)

    def test_gather_noswap_4experts(self):
        self._run(tile_n=16, num_experts=4, num_tokens=512, swap_ab=False)

    def test_gather_swap_4experts(self):
        self._run(tile_n=16, num_experts=4, num_tokens=512, swap_ab=True)

    def test_gather_noswap_nonrounded_topk(self):
        self._run(tile_n=16, num_experts=3, num_tokens=130, top_k=2, swap_ab=False)

    def test_gather_swap_nonrounded_topk(self):
        self._run(tile_n=16, num_experts=3, num_tokens=130, top_k=2, swap_ab=True)

    def test_gather_noswap_2cta_tile64(self):
        self._run(
            tile_n=64,
            swap_ab=False,
            cluster_m=2,
            num_experts=2,
            num_tokens=256,
            pipeline_stages=4,
        )

    def test_gather_swap_2cta_tile64(self):
        self._run(
            tile_n=64,
            swap_ab=True,
            cluster_m=2,
            num_experts=2,
            num_tokens=256,
            pipeline_stages=4,
        )

    @pytest.mark.parametrize("swap_ab", [False, True])
    def test_gather_2cta_multistage_wrap(self, swap_ab):
        """Cover two full turns of the four-stage 2-CTA LDGSTS pipeline."""
        self._run(
            tile_n=64,
            swap_ab=swap_ab,
            cluster_m=2,
            num_experts=1,
            num_tokens=64,
            problem_k=1024,
            pipeline_stages=4,
        )


class TestBf16Fc1SwiGLU:
    def _run_swiglu(
        self,
        *,
        tile_n=16,
        pipeline_stages=4,
        num_experts=1,
        num_tokens=128,
        swap_ab=False,
        has_gather=False,
        gemm1_clamp_limit_value=2.0,
        bias_type=0,
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC1_BASE,
            "act_kind": 1,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
            "has_gemm1_clamp_limit": 1,
            "bias_type": bias_type,
        }
        if not has_gather:
            cfg["route_act"] = 0  # NONE — no gather
        result = reference_check(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=1,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            gemm1_clamp_limit_value=gemm1_clamp_limit_value,
            **uniform_pipeline_stage_overrides(pipeline_stages),
            **cfg,
        )
        assert result

    def test_swiglu_noswap_tile8(self):
        self._run_swiglu(tile_n=8, swap_ab=False)

    def test_swiglu_noswap_tile16(self):
        self._run_swiglu(tile_n=16, swap_ab=False)

    def test_swiglu_swap_tile8(self):
        self._run_swiglu(tile_n=8, swap_ab=True)

    def test_swiglu_swap_tile16(self):
        self._run_swiglu(tile_n=16, swap_ab=True)

    def test_swiglu_gather_noswap(self):
        self._run_swiglu(
            tile_n=16, swap_ab=False, has_gather=True, num_experts=2, num_tokens=256
        )

    def test_swiglu_gather_swap(self):
        self._run_swiglu(
            tile_n=16, swap_ab=True, has_gather=True, num_experts=2, num_tokens=256
        )

    def test_swiglu_clamp_bias_noswap(self):
        self._run_swiglu(
            tile_n=16,
            swap_ab=False,
            gemm1_clamp_limit_value=0.01,
            bias_type=1,
        )


class TestBf16Fc1GeGLU:
    def _run_geglu(self, *, tile_n=16, swap_ab=False, has_gather=False):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC1_BASE,
            "act_kind": 2,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
        }
        if not has_gather:
            cfg["route_act"] = 0  # NONE — no gather
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )
        assert result

    def test_geglu_noswap(self):
        self._run_geglu(swap_ab=False)

    def test_geglu_swap(self):
        self._run_geglu(swap_ab=True)

    def test_geglu_gather(self):
        self._run_geglu(swap_ab=False, has_gather=True)


class TestBf16Fc1ReLU2:
    def _run_relu2(self, *, tile_n=16, swap_ab=False, has_gather=False):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC1_BASE,
            "act_kind": 3,
            "batch_mode": int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
            "transpose_mma_output": int(swap_ab),
        }
        if not has_gather:
            cfg["route_act"] = 0  # NONE — no gather
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )
        assert result

    def test_relu2_noswap(self):
        self._run_relu2(swap_ab=False)

    def test_relu2_swap(self):
        self._run_relu2(swap_ab=True)

    def test_relu2_gather(self):
        self._run_relu2(swap_ab=False, has_gather=True)


class TestBf16Fc1HighThroughput:
    HT_CFG = {
        **BF16_FC1_BASE,
        "route_act": 1,
        "cluster_m": 2,
        "mma_m": 256,
        "tile_scheduler": 1,
        "num_stages_tmem_acc": 2,
        "batch_mode": int(BatchMode.BATCH_N),
        "transpose_mma_output": 1,
        "epilogue_regs": 168,
        "mma_regs": 96,
        "load_regs": 96,
        "gather_regs": 96,
        "padding_regs": 96,
        "workid_regs": 96,
    }

    @pytest.mark.timeout(240)
    def test_ht_generated_rows(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        for tile_n, tile_k, pipeline_stages in (
            (64, 128, 4),
            (128, 64, 6),
        ):
            for act_kind in (1, 3):
                for unroll in (0,):
                    cfg = {
                        **self.HT_CFG,
                        "act_kind": act_kind,
                        "use_unroll_loop_2x_for_mma": unroll,
                    }
                    result = reference_check(
                        num_experts=2,
                        num_tokens=256,
                        top_k=1,
                        problem_n=256,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        mma_n=tile_n,
                        epi_tile_n=tile_n,
                        **uniform_pipeline_stage_overrides(pipeline_stages),
                        **cfg,
                    )
                    assert result

    @pytest.mark.timeout(120)
    def test_ht_tma_cluster_dynamic_k1024(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **self.HT_CFG,
            "act_kind": 1,
            "bias_type": 0,
            "has_gemm1_clamp_limit": 1,
            "route_sfs_act": 0,
            "use_clc_fast_drain": 0,
            "use_early_exit": 1,
            "use_tma_oob_opt": 1,
            "use_tma_store": 1,
        }
        result = reference_check(
            num_experts=32,
            num_tokens=1,
            top_k=8,
            problem_n=4096,
            problem_k=1024,
            tile_n=64,
            tile_k=128,
            mma_n=64,
            epi_tile_n=64,
            **uniform_pipeline_stage_overrides(5),
            **cfg,
        )
        assert result
