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

"""Tests for FP4 NVF4 (E2M1 data, E4M3 scale factors) FC2 BatchedGemm TS kernel.

Covers variant table rows 31-54 (FC2_LL) and 55-58 (FC2_HT):
  tile_n ∈ {8, 16, 32}, tile_k ∈ {256, 512}, stages ∈ {5, 9},
  static/persistent, 1-CTA/2-CTA.

Run with: pytest tests/prims_ts/test_batched_gemm_fp4_fc2.py -v
Requires: CUDA GPU with SM100A+ (B200 Blackwell)
"""

import pytest

import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
    BatchMode,
    BiasType,
    DType,
    SfLayout,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))), reason="FP4 cvt (cvt.e2m1x2.f32) requires Blackwell sm_100+"
    ),
]

# FP4 FC2 LowLatency base config (rows 31-54).
# tmem_sfa/sfb_cols = tile_k / mma_k * 4 = num_kblocks * 4
FP4_FC2_LL = dict(
    route_act=0,
    act_kind=0,
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
    use_unroll_loop_2x_for_mma=0,
    use_max_tmem_overlap=0,
    batch_mode=int(BatchMode.BATCH_M),
    transpose_mma_output=0,
)

def _run_fp4_fc2(
    *,
    tile_n,
    tile_k=256,
    pipeline_stages=5,
    tile_scheduler=0,
    cluster_m=1,
    num_experts=2,
    num_tokens=256,
    top_k=1,
    bias_type=0,
    use_pdl=0,
    use_tma_store=0,
    sf_layout_a=None,
    sf_layout_b=None,
    scale_c_value=1.0,
    scale_gate_value=1.0,
):
    """Helper: run FP4 FC2 reference check."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    num_stages_tmem_acc = 2 if tile_scheduler == 1 else 1
    cfg = {**FP4_FC2_LL}
    cfg["cluster_m"] = cluster_m
    if cluster_m >= 2:
        cfg["mma_m"] = 256
    if sf_layout_a is not None:
        cfg["sf_layout_a"] = sf_layout_a
    if sf_layout_b is not None:
        cfg["sf_layout_b"] = sf_layout_b

    result = reference_check(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=tile_n,
        tile_k=tile_k,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        tile_scheduler=tile_scheduler,
        num_stages_tmem_acc=num_stages_tmem_acc,
        bias_type=bias_type,
        use_pdl=use_pdl,
        use_tma_store=use_tma_store,
        scale_c_value=scale_c_value,
        scale_gate_value=scale_gate_value,
        **cfg,
    )
    assert result, (
        "FP4 FC2 failed: "
        f"tile_n={tile_n}, tile_k={tile_k}, stages={pipeline_stages}, "
        f"scheduler={tile_scheduler}, cluster_m={cluster_m}"
    )

class TestFp4Fc2Validation:

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
            **FP4_FC2_LL,
        )

    def test_validate_tile16_k512(self):
        """tile_k=512 uses separate CopySf plus 256-K TMA boxes."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        tile_n = 16
        tile_k = 512
        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(5),
            num_stages_tmem_acc=1,
            **FP4_FC2_LL,
        )

        with pytest.raises(ValueError, match="sf_layout_a=8x4"):
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                tile_n=tile_n,
                mma_n=tile_n,
                epi_tile_n=tile_n,
                tile_k=tile_k,
                **uniform_pipeline_stage_overrides(5),
                num_stages_tmem_acc=1,
                sf_layout_a=int(SfLayout.R8c4),
                **FP4_FC2_LL,
            )

    def test_reject_b_sf_8x4_tile128(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        tile_n = 128
        tile_k = 256
        with pytest.raises(ValueError, match="sf_layout_b=8x4"):
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                tile_n=tile_n,
                mma_n=tile_n,
                epi_tile_n=tile_n,
                tile_k=tile_k,
                **uniform_pipeline_stage_overrides(5),
                num_stages_tmem_acc=1,
                **FP4_FC2_LL,
            )

class TestFp4Fc2LLStatic:
    """FP4 FC2 LowLatency static scheduler (rows 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53)."""

    def test_tile8_k256_s5(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=5)

    def test_tile8_k256_s5_tma_store(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=5, use_tma_store=1)

    def test_tile8_k256_s5_b_sf_8x4(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=5)

    def test_tile8_k256_s5_bias_m(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=5, bias_type=1)

    def test_tile8_k256_s5_global_scale_c(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 8
        tile_k = 256
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
            return_output=True,
            **FP4_FC2_LL,
        )
        ok_base, out_base = reference_check(scale_c_value=1.0, **common)
        ok_scaled, out_scaled = reference_check(scale_c_value=2.0, **common)
        assert ok_base and ok_scaled
        torch.testing.assert_close(
            out_scaled.float(),
            out_base.float() * 2.0,
            rtol=0.02,
            atol=0.25,
        )

    def test_tile8_k256_s5_tma_oob_opt_partial_m_tile(self):
        _run_fp4_fc2(
            tile_n=8,
            tile_k=256,
            pipeline_stages=5,
            num_experts=1,
            num_tokens=160,
        )

    def test_tile8_k256_s5_nonrounded_topk_expanded_layout(self):
        _run_fp4_fc2(
            tile_n=8,
            tile_k=256,
            pipeline_stages=5,
            num_experts=3,
            num_tokens=130,
            top_k=2,
        )

    def test_tile8_k256_s5_early_exit_overestimated_token_grid(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 8
        tile_k = 256
        result = reference_check(
            num_experts=4,
            num_tokens=160,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            use_early_exit=1,
            **FP4_FC2_LL,
        )
        assert result

    def test_tile16_k256_s5(self):
        _run_fp4_fc2(tile_n=16, tile_k=256, pipeline_stages=5)

    def test_tile32_k256_s5(self):
        _run_fp4_fc2(tile_n=32, tile_k=256, pipeline_stages=5)

    def test_tile8_k512_s5(self):
        _run_fp4_fc2(tile_n=8, tile_k=512, pipeline_stages=5, num_tokens=128)

    def test_tile8_k512_s5_per_token_sfb(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **FP4_FC2_LL,
            "sf_layout_a": int(SfLayout.R128c4),
            "sf_layout_b": int(SfLayout.R8c4),
            "sf_layout_c": int(SfLayout.R8c4),
            "batch_mode": int(BatchMode.BATCH_N),
            "transpose_mma_output": 1,
            "use_global_scales": 1,
            "use_per_token_sf_b": 1,
            "per_token_sf_dtype": int(DType.FP32),
            "bias_type": int(BiasType.M),
            "use_tma_store": 1,
        }

        assert reference_check(
            num_experts=2,
            num_tokens=16,
            top_k=1,
            tile_n=8,
            mma_n=8,
            epi_tile_n=8,
            tile_k=512,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=0,
            num_stages_tmem_acc=1,
            **cfg,
        )

    def test_tile16_k512_s5_cluster2(self):
        _run_fp4_fc2(tile_n=16, tile_k=512, pipeline_stages=5, cluster_m=2)

    def test_tile32_k512_s5_cluster2(self):
        _run_fp4_fc2(tile_n=32, tile_k=512, pipeline_stages=5, cluster_m=2)

    def test_tile8_k256_s9(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=9)

class TestFp4Fc2LLPersistent:
    """FP4 FC2 LowLatency persistent (rows 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54)."""


    def test_tile8_k256_persistent(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=5, tile_scheduler=1)

    def test_tile8_k512_persistent(self):
        _run_fp4_fc2(
            tile_n=8,
            tile_k=512,
            pipeline_stages=5,
            tile_scheduler=1,
            num_tokens=128,
        )

    def test_tile8_k256_s9_persistent(self):
        _run_fp4_fc2(tile_n=8, tile_k=256, pipeline_stages=9, tile_scheduler=1)

    def test_tile8_k256_persistent_tma_store(self):
        _run_fp4_fc2(
            tile_n=8,
            tile_k=256,
            pipeline_stages=5,
            tile_scheduler=1,
            use_tma_store=1,
        )

    def test_tile16_k256_persistent(self):
        _run_fp4_fc2(tile_n=16, tile_k=256, pipeline_stages=5, tile_scheduler=1)

    def test_tile16_k512_persistent_cluster2(self):
        _run_fp4_fc2(
            tile_n=16,
            tile_k=512,
            pipeline_stages=5,
            tile_scheduler=1,
            cluster_m=2,
        )

    def test_tile16_k256_persistent_pdl(self):
        _run_fp4_fc2(
            tile_n=16,
            tile_k=256,
            pipeline_stages=5,
            tile_scheduler=1,
            use_pdl=1,
        )

    def test_tile16_k256_persistent_early_exit(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        tile_n = 16
        tile_k = 256
        result = reference_check(
            num_experts=4,
            num_tokens=160,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=1,
            num_stages_tmem_acc=2,
            use_early_exit=1,
            **FP4_FC2_LL,
        )
        assert result

    def test_tile32_k512_persistent_cluster2(self):
        _run_fp4_fc2(
            tile_n=32,
            tile_k=512,
            pipeline_stages=5,
            tile_scheduler=1,
            cluster_m=2,
        )

class TestFp4Fc2MultiExpert:
    """FP4 FC2 with multiple experts."""

    def test_4experts_tile16(self):
        _run_fp4_fc2(tile_n=16, tile_k=256, num_experts=4, num_tokens=512)

class TestFp4Fc2HT:
    """FP4 FC2 HighThroughput: 2-CTA cluster (rows 55-58)."""

    pytestmark = [pytest.mark.timeout(120)]

    def _run_ht(
        self,
        *,
        tile_n,
        tile_k=256,
        pipeline_stages=4,
        use_unroll_loop_2x_for_mma=0,
        num_experts=2,
        num_tokens=256,
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            SfLayout,
        )
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **FP4_FC2_LL,
            "cluster_m": 2,
            "mma_m": 256,
            "sf_layout_b": int(SfLayout.R8c4 if tile_n <= 64 else SfLayout.R128c4),
            "sf_layout_c": int(SfLayout.R8c4 if tile_n <= 64 else SfLayout.R128c4),
            "batch_mode": int(BatchMode.BATCH_N),
            "transpose_mma_output": 1,
            "use_unroll_loop_2x_for_mma": use_unroll_loop_2x_for_mma,
        }

        result = reference_check(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=1,
            tile_n=tile_n,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            tile_k=tile_k,
            **uniform_pipeline_stage_overrides(pipeline_stages),
            tile_scheduler=1,
            num_stages_tmem_acc=2,
            **cfg,
        )
        assert result, f"FP4 FC2 HT failed: tile_n={tile_n}"

    def test_ht_tile32_k256(self):
        self._run_ht(tile_n=32)

    @pytest.mark.parametrize("unroll", [0])
    @pytest.mark.timeout(240)
    def test_ht_tile64_k512(self, unroll):
        self._run_ht(
            tile_n=64,
            tile_k=512,
            use_unroll_loop_2x_for_mma=unroll,
            num_tokens=128,
        )

    @pytest.mark.parametrize("unroll", [0])
    @pytest.mark.timeout(240)
    def test_ht_tile128_k256(self, unroll):
        self._run_ht(
            tile_n=128,
            pipeline_stages=5,
            use_unroll_loop_2x_for_mma=unroll,
        )

    @pytest.mark.timeout(240)
    def test_ht_tile256_k256(self):
        """2-CTA HT256 fused-UTCCP path with tile_n=256 TMEM overlap."""
        from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
            SfLayout,
        )
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            batch_mode=int(BatchMode.BATCH_N),
            transpose_mma_output=1,
            route_act=0,
            act_kind=0,
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
            **uniform_pipeline_stage_overrides(5),
            tile_scheduler=1,
            num_stages_tmem_acc=1,
            sf_layout_b=int(SfLayout.R128c4),
            use_max_tmem_overlap=1,
            epilogue_regs=48,
            mma_regs=48,
            load_regs=48,
            load_sf_regs=48,
            workid_regs=48,
            padding_regs=48,
        )
        assert result, "FP4 FC2 HT256 fused-UTCCP failed"
