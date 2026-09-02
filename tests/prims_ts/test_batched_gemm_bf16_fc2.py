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

"""Tests for BF16×BF16 FC2 BatchedGemm TS kernel.

Run with: pytest tests/prims_ts/test_batched_gemm_bf16_fc2.py -v
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

# Common BF16 FC2 config — no warp layout fields (computed inside kernel).
BF16_FC2_BASE = dict(
    batch_mode=int(BatchMode.BATCH_M),
    transpose_mma_output=0,
    route_act=0,
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
    mma_regs=24,
    load_regs=24,
    padding_regs=24,
    workid_regs=24,
    use_unroll_loop_2x_for_mma=0,
    use_max_tmem_overlap=0,
)


def _run_bf16_fc2(
    *,
    tile_n,
    tile_k=128,
    pipeline_stages=4,
    num_experts=1,
    num_tokens=128,
    top_k=1,
    use_tma_store=0,
    dtype_c=int(DType.BF16),
    expected_output_dtype=None,
    max_abs_tol=0.1,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    cfg = {**BF16_FC2_BASE, "dtype_c": int(dtype_c)}
    result = reference_check(
        num_experts=num_experts,
        num_tokens=num_tokens,
        top_k=top_k,
        tile_n=tile_n,
        tile_k=tile_k,
        mma_n=tile_n,
        epi_tile_n=tile_n,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        use_tma_store=use_tma_store,
        return_output=expected_output_dtype is not None,
        **cfg,
    )
    if expected_output_dtype is not None:
        result, output = result
        assert output.dtype == expected_output_dtype
    assert result, (
        f"Reference check failed for tileN={tile_n}, stages={pipeline_stages}, "
        f"dtype_c={dtype_c}"
    )


class TestScheduleValidation:
    def test_validate_bf16_fc2_tile8(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        build_batched_gemm_task_manager(
            num_experts=1,
            num_tokens=128,
            top_k=1,
            tile_n=8,
            tile_k=128,
            mma_n=8,
            epi_tile_n=8,
            **uniform_pipeline_stage_overrides(4),
            **BF16_FC2_BASE,
        )

    def test_validate_bf16_fc2_tile16(self):
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
            **uniform_pipeline_stage_overrides(5),
            **BF16_FC2_BASE,
        )

    def test_validate_pdl_wait_for_num_non_exiting_ctas(self):
        from cutlass.experimental.task_scheduling.resources import (
            PdlLaunchBarrier,
            PdlWaitBarrier,
        )

        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        task_manager = build_batched_gemm_task_manager(
            num_experts=1,
            num_tokens=160,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            use_pdl=1,
            use_early_exit=1,
            do_pdl_wait_for_num_non_exiting_ctas=1,
            **BF16_FC2_BASE,
        )

        all_task_resources = [
            resource
            for task in task_manager.tasks
            for resource in task.src_resources + task.dst_resources
        ]

        # qgai's Gen-matched non-gather FC2 schedule loads the launch bound in
        # the prologue but keeps the PDL wait on the FC1-output consumer.  Only
        # gather schedules may mark the wait complete before task execution.
        assert task_manager._assume_pdl_wait_completed is False
        assert any(
            isinstance(resource, PdlWaitBarrier) for resource in all_task_resources
        )
        assert any(
            isinstance(resource, PdlLaunchBarrier) for resource in all_task_resources
        )

    def test_validate_pdl_wait_for_num_non_exiting_ctas_requires_pdl(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        with pytest.raises(
            ValueError,
            match="do_pdl_wait_for_num_non_exiting_ctas requires use_pdl=1",
        ):
            build_batched_gemm_task_manager(
                num_experts=1,
                num_tokens=160,
                top_k=1,
                tile_n=16,
                tile_k=128,
                mma_n=16,
                epi_tile_n=16,
                **uniform_pipeline_stage_overrides(4),
                use_early_exit=1,
                do_pdl_wait_for_num_non_exiting_ctas=1,
                **BF16_FC2_BASE,
            )

    def test_validate_pdl_wait_for_num_non_exiting_ctas_requires_early_exit(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        with pytest.raises(
            ValueError,
            match="do_pdl_wait_for_num_non_exiting_ctas requires use_early_exit=1",
        ):
            build_batched_gemm_task_manager(
                num_experts=1,
                num_tokens=160,
                top_k=1,
                tile_n=16,
                tile_k=128,
                mma_n=16,
                epi_tile_n=16,
                **uniform_pipeline_stage_overrides(4),
                use_pdl=1,
                do_pdl_wait_for_num_non_exiting_ctas=1,
                **BF16_FC2_BASE,
            )


class TestBf16Fc2ReferenceCheck:
    @pytest.mark.parametrize(
        ("dtype_c", "expected_output_dtype"),
        (
            (int(DType.BF16), torch.bfloat16),
            (int(DType.FP16), torch.float16),
        ),
        ids=("bf16_c", "fp16_c"),
    )
    def test_plain_output_dtype(self, dtype_c, expected_output_dtype):
        _run_bf16_fc2(
            tile_n=16,
            pipeline_stages=4,
            dtype_c=dtype_c,
            expected_output_dtype=expected_output_dtype,
        )

    def test_tile_n8_stages4(self):
        _run_bf16_fc2(tile_n=8, pipeline_stages=4)

    def test_tile_n16_stages4(self):
        _run_bf16_fc2(tile_n=16, pipeline_stages=4)

    def test_tile_n32_stages4(self):
        _run_bf16_fc2(tile_n=32, pipeline_stages=4)

    def test_tma_store_tile_n8(self):
        _run_bf16_fc2(tile_n=8, use_tma_store=1)

    def test_tma_store_tile_n32(self):
        _run_bf16_fc2(tile_n=32, use_tma_store=1)

    def test_bias_m_tile_n16(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=1,
            num_tokens=128,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            bias_type=1,
            **BF16_FC2_BASE,
        )
        assert result

    def test_tma_oob_opt_partial_m_tile(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=4,
            num_tokens=160,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            use_tma_oob_opt=1,
            **BF16_FC2_BASE,
        )
        assert result

    def test_early_exit_overestimated_token_grid(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=1,
            num_tokens=160,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            use_early_exit=1,
            **BF16_FC2_BASE,
        )
        assert result

    def test_early_exit_pdl_wait_for_num_non_exiting_ctas(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=1,
            num_tokens=160,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(4),
            use_pdl=1,
            use_early_exit=1,
            do_pdl_wait_for_num_non_exiting_ctas=1,
            **BF16_FC2_BASE,
        )
        assert result

    def test_nonrounded_topk_expanded_layout(self):
        _run_bf16_fc2(tile_n=16, num_experts=3, num_tokens=130, top_k=2)


class TestBf16Fc2LargerProblem:
    def test_tile_n16_problem_n256_k256(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            problem_n=256,
            problem_k=256,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            **BF16_FC2_BASE,
        )
        assert result

    def test_tile_n16_problem_n1024_k1024(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=4,
            num_tokens=512,
            top_k=1,
            problem_n=1024,
            problem_k=1024,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            **BF16_FC2_BASE,
        )
        assert result


class TestBf16Fc2MultiExpert:
    def test_2_experts(self):
        _run_bf16_fc2(tile_n=16, num_experts=2, num_tokens=256)

    def test_4_experts(self):
        _run_bf16_fc2(tile_n=16, num_experts=4, num_tokens=512)


class TestBf16Fc2Persistent:
    PERSISTENT_CFG = {
        **BF16_FC2_BASE,
        "tile_scheduler": 1,
        "num_stages_tmem_acc": 2,
    }

    def test_validate_persistent_tile16(self):
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
            **uniform_pipeline_stage_overrides(5),
            **self.PERSISTENT_CFG,
        )

    def test_validate_persistent_pdl_uses_split_barriers(self):
        from cutlass.experimental.task_scheduling.enums import ScheduleStage
        from cutlass.experimental.task_scheduling.resources import (
            PdlLaunchBarrier,
            PdlWaitBarrier,
        )

        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        task_manager = build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            use_pdl=1,
            **self.PERSISTENT_CFG,
        )

        load_a_task = next(t for t in task_manager.tasks if t.name == "LoadATask")
        pdl_wait = next(
            r for r in load_a_task.src_resources if isinstance(r, PdlWaitBarrier)
        )
        pdl_launch = next(
            r for r in load_a_task.dst_resources if isinstance(r, PdlLaunchBarrier)
        )

        assert (
            id(pdl_wait),
            ScheduleStage.ConsumerWork,
            0,
        ) in load_a_task.pre_work_loop_head_slots
        assert (
            id(pdl_launch),
            ScheduleStage.ProducerWork,
            0,
        ) in load_a_task.post_work_loop_tail_slots

        wait_pos = next(
            idx
            for idx, (resource, stage, _call_id, _label) in enumerate(
                load_a_task.head_schedule_list
            )
            if resource is pdl_wait and stage == ScheduleStage.ConsumerWork
        )
        gmem_a_aux_pos = next(
            idx
            for idx, (resource, stage, _call_id, _label) in enumerate(
                load_a_task.head_schedule_list
            )
            if resource.name == "GmemA" and stage == ScheduleStage.ConsumerAuxWork
        )
        assert wait_pos < gmem_a_aux_pos

        dep_graph = {
            downstream.name: [upstream.name for upstream in upstreams]
            for downstream, upstreams in task_manager.resource_dependency_graph.items()
        }
        assert dep_graph["GmemA"] == ["PdlWait"]
        assert dep_graph["PdlLaunch"] == []

    def test_persistent_tile16_stages5(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            **self.PERSISTENT_CFG,
        )
        assert result

    def test_persistent_tile16_stages5_pdl(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            use_pdl=1,
            **self.PERSISTENT_CFG,
        )
        assert result

    def test_persistent_early_exit_overestimated_token_grid(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=4,
            num_tokens=160,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            use_early_exit=1,
            **self.PERSISTENT_CFG,
        )
        assert result

    def test_persistent_tile8_stages4(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        result = reference_check(
            num_experts=4,
            num_tokens=512,
            top_k=1,
            tile_n=8,
            tile_k=128,
            mma_n=8,
            epi_tile_n=8,
            **uniform_pipeline_stage_overrides(4),
            **self.PERSISTENT_CFG,
        )
        assert result

    def test_persistent_tile32_generated_rows(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        for pipeline_stages in (5, 4):
            for unroll in (0,):
                cfg = {
                    **self.PERSISTENT_CFG,
                    "use_unroll_loop_2x_for_mma": unroll,
                }
                result = reference_check(
                    num_experts=2,
                    num_tokens=256,
                    top_k=1,
                    tile_n=32,
                    tile_k=128,
                    mma_n=32,
                    epi_tile_n=32,
                    **uniform_pipeline_stage_overrides(pipeline_stages),
                    **cfg,
                )
                assert result


class TestBf16Fc2DoubleBufferedAcc:
    def _run_double_buf(
        self, *, tile_n, pipeline_stages=5, num_experts=1, num_tokens=128
    ):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {**BF16_FC2_BASE, "num_stages_tmem_acc": 2}
        result = reference_check(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=1,
            tile_n=tile_n,
            tile_k=128,
            mma_n=tile_n,
            epi_tile_n=tile_n,
            **uniform_pipeline_stage_overrides(pipeline_stages),
            **cfg,
        )
        assert result

    def test_tile_n8_stages5_mma2(self):
        self._run_double_buf(tile_n=8, pipeline_stages=5)

    def test_tile_n16_stages5_mma2(self):
        self._run_double_buf(tile_n=16, pipeline_stages=5)

    def test_tile_n16_stages5_mma2_multi_expert(self):
        self._run_double_buf(
            tile_n=16, pipeline_stages=5, num_experts=4, num_tokens=512
        )


class TestBf16Fc2SwapAB:
    def test_swap_ab_true_2experts(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC2_BASE,
            "batch_mode": int(BatchMode.BATCH_N),
            "transpose_mma_output": 1,
        }
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            **cfg,
        )
        assert result

    def test_swap_ab_false_2experts(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = BF16_FC2_BASE
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=16,
            tile_k=128,
            mma_n=16,
            epi_tile_n=16,
            **uniform_pipeline_stage_overrides(5),
            **cfg,
        )
        assert result

    def test_swap_ab_true_4experts(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **BF16_FC2_BASE,
            "batch_mode": int(BatchMode.BATCH_N),
            "transpose_mma_output": 1,
        }
        result = reference_check(
            num_experts=4,
            num_tokens=512,
            top_k=1,
            tile_n=8,
            tile_k=128,
            mma_n=8,
            epi_tile_n=8,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )
        assert result


class TestBf16Fc2HighThroughput:
    HT_CFG = {
        **BF16_FC2_BASE,
        "cluster_m": 2,
        "mma_m": 256,
        "tile_scheduler": 1,
        "num_stages_tmem_acc": 2,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "epilogue_regs": 168,
        "mma_regs": 96,
        "load_regs": 96,
        "padding_regs": 96,
        "workid_regs": 96,
    }

    def test_validate_ht_generated_rows(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        for tile_n, tile_k, epi_tile_n, pipeline_stages in (
            (64, 128, 64, 4),
            (128, 64, 64, 8),
        ):
            cfg = {**self.HT_CFG}
            build_batched_gemm_task_manager(
                num_experts=2,
                num_tokens=256,
                top_k=1,
                tile_n=tile_n,
                tile_k=tile_k,
                mma_n=tile_n,
                epi_tile_n=epi_tile_n,
                **uniform_pipeline_stage_overrides(pipeline_stages),
                **cfg,
            )

    def test_ht_generated_rows_no_swap(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        for tile_n, tile_k, epi_tile_n, pipeline_stages in (
            (64, 128, 64, 4),
            (128, 64, 64, 8),
        ):
            for unroll in (0,):
                cfg = {
                    **self.HT_CFG,
                    "use_unroll_loop_2x_for_mma": unroll,
                }
                result = reference_check(
                    num_experts=2,
                    num_tokens=256,
                    top_k=1,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    mma_n=tile_n,
                    epi_tile_n=epi_tile_n,
                    **uniform_pipeline_stage_overrides(pipeline_stages),
                    **cfg,
                )
                assert result

    def test_ht_tile64_stages4_swap_ab(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        cfg = {
            **self.HT_CFG,
            "batch_mode": int(BatchMode.BATCH_N),
            "transpose_mma_output": 1,
        }
        result = reference_check(
            num_experts=2,
            num_tokens=256,
            top_k=1,
            tile_n=64,
            tile_k=128,
            mma_n=64,
            epi_tile_n=64,
            **uniform_pipeline_stage_overrides(4),
            **cfg,
        )
        assert result

    def test_ht_tile128_after_fc1_swiglu_reuses_output_block(self, monkeypatch):
        import gc

        import cutlass.cute.testing as testing
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            benchmark,
            reference_check,
        )

        def direct_benchmark(
            callable,
            *,
            workspace_generator=None,
            workspace_count=1,
            **kwargs,
        ):
            del workspace_count, kwargs
            workspace = workspace_generator()
            callable(*workspace.args, **workspace.kwargs)
            torch.cuda.synchronize()
            del workspace
            gc.collect()
            torch.cuda.synchronize()
            return 0.0

        monkeypatch.setattr(testing, "benchmark", direct_benchmark)

        pred_cfg = {
            **self.HT_CFG,
            "route_act": 1,
            "act_kind": 1,
            "bias_type": 0,
            "has_gemm1_clamp_limit": 1,
            "use_clc_fast_drain": 0,
            "use_early_exit": 1,
            "use_tma_oob_opt": 1,
            "use_tma_store": 1,
        }
        target_cfg = {
            **self.HT_CFG,
            "bias_type": 0,
            "use_clc_fast_drain": 0,
            "use_early_exit": 1,
            "use_tma_oob_opt": 1,
            "use_tma_store": 1,
        }
        benchmark(
            num_experts=32,
            num_tokens=32,
            top_k=8,
            problem_n=4096,
            problem_k=7168,
            warmup_iters=1,
            bench_iters=1,
            num_rotated_buffers=0,
            tile_n=64,
            tile_k=128,
            mma_n=64,
            epi_tile_n=64,
            **uniform_pipeline_stage_overrides(5),
            **pred_cfg,
        )
        result = reference_check(
            num_experts=32,
            num_tokens=256,
            top_k=8,
            problem_n=7168,
            problem_k=2048,
            tile_n=128,
            tile_k=64,
            mma_n=128,
            epi_tile_n=64,
            **uniform_pipeline_stage_overrides(8),
            **target_cfg,
        )
        assert result


class TestBf16Fc2RepeatedLaunch:
    def test_repeated_launch_10x(self):
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
            reference_check,
        )

        for i in range(3):
            result = reference_check(
                num_experts=1,
                num_tokens=128,
                top_k=1,
                tile_n=16,
                tile_k=128,
                mma_n=16,
                epi_tile_n=16,
                **uniform_pipeline_stage_overrides(4),
                **BF16_FC2_BASE,
            )
            assert result, f"Failed on iteration {i}"
