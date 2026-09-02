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

"""B200 correctness for compact DeepSeek-FP8 scales converted in TMEM."""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

pytestmark = [
    pytest.mark.xdist_group("isolated_cuda"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
        reason="native MXFP8 MMA requires SM100A+",
    ),
]


def _native_fc2_config(*, tile_k=128):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        SfLayout,
        TileScheduler,
        uniform_pipeline_stage_overrides,
    )

    return {
        "batch_mode": int(BatchMode.BATCH_N),
        "transpose_mma_output": 1,
        "dtype_a": int(DType.MXE4M3),
        "dtype_b": int(DType.MXE4M3),
        "dtype_c": int(DType.BF16),
        "tile_m": 128,
        "tile_n": 8,
        "tile_k": tile_k,
        "mma_m": 128,
        "mma_n": 8,
        "mma_k": 32,
        "epi_tile_n": 8,
        "sf_layout_a": int(SfLayout.R128c4),
        "sf_layout_b": int(SfLayout.R8c4),
        **uniform_pipeline_stage_overrides(2 if tile_k == 512 else 4),
        "num_stages_tmem_acc": 2,
        "tile_scheduler": int(TileScheduler.STATIC),
        "use_mxfp8_deepseek_fp8": 1,
    }


def _native_fc1_config(*, tile_k=128):
    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_deepseek_fp8_moe_tactic,
    )

    assert tile_k in (128, 256, 512)
    pair = map_trtllm_deepseek_fp8_moe_tactic(
        [-1, -1],
        num_tokens=256,
        top_k=8,
        num_local_experts=256,
        use_mxfp8_backed_dsfp8=True,
    )
    cfg = dict(pair.fc1.cfg.kwargs)
    cfg["tile_k"] = tile_k
    return cfg


def test_tmem_expansion_asymmetric_k256_multiple_weight_blocks_and_rows():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=9,
        top_k=2,
        problem_n=256,
        problem_k=256,
        seed=20260731,
        output_guard_elements=257,
        **_native_fc2_config(),
    )


def test_native_fc1_routed_scales_and_mxfp8_output_scales():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=9,
        top_k=2,
        problem_n=256,
        problem_k=256,
        seed=20260901,
        repeat_launches=3,
        **_native_fc1_config(),
    )


def test_native_expansion_tile_k256_consumes_two_distinct_k128_blocks():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=9,
        top_k=2,
        problem_n=256,
        problem_k=512,
        seed=20260801,
        output_guard_elements=257,
        **_native_fc2_config(tile_k=256),
    )


def test_native_expansion_tile_k256_handles_one_k128_block_tail():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=9,
        top_k=2,
        problem_n=256,
        problem_k=384,
        seed=20260804,
        output_guard_elements=257,
        **_native_fc2_config(tile_k=256),
    )


def test_native_tmem_expansion_tile_k512_consumes_four_k128_blocks():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=9,
        top_k=2,
        problem_n=256,
        problem_k=1024,
        seed=20260902,
        output_guard_elements=257,
        **_native_fc2_config(tile_k=512),
    )
