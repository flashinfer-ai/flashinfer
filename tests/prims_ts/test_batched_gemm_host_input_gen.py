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

"""Correctness coverage for host-generated BatchedGemm TS inputs."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
)
from flashinfer.utils import is_sm100a_supported

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
        reason="BatchedGemm TS requires Blackwell sm_100+",
    ),
]


_COMMON = {
    "batch_mode": int(BatchMode.BATCH_N),
    "transpose_mma_output": 1,
    "tile_scheduler": int(TileScheduler.PERSISTENT),
    "use_shuffled_matrix": 1,
    "umma_cta_count": 2,
    "cluster_dim_x": 2,
    "tile_n": 128,
    "mma_m": 256,
    "mma_n": 128,
    "mma_k": 64,
    "num_stages_tmem_acc": 2,
    "use_tma_store": 1,
    "use_unroll_loop_2x_for_mma": 1,
    "use_early_exit": 1,
    "use_clc_fast_drain": 1,
    "use_work_throttle": 0,
    "use_pdl": 1,
}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                **_COMMON,
                "problem_n": 256,
                "problem_k": 512,
                "act_kind": int(ActKind.SWIGLU),
                "route_act": int(RouteImpl.LDGSTS),
                "route_sfs_act": int(RouteImpl.LDGSTS),
                "sf_layout_b": int(SfLayout.LINEAR),
                "sf_layout_c": int(SfLayout.R128c4),
                "dtype_c": int(DType.E2M1),
                "tile_k": 512,
                "epi_tile_n": 32,
                "num_stages_a": 3,
                "num_stages_b": 3,
                "num_stages_smem_sfa": 3,
                "num_stages_smem_sfb": 3,
                "num_stages_tmem_sfa": 0,
                "num_stages_tmem_sfb": 0,
                "use_tma_oob_opt": 0,
                "num_load_a_warps": 1,
                "num_load_b_warps": 4,
                "num_load_sfa_warps": 1,
                "num_load_sfb_warps": 4,
                "load_b_regs": 128,
                "load_sfb_regs": 128,
                "fuse_operand_sf_loads": 1,
                "fuse_sf_copy_to_mma": 1,
            },
            id="fc1-ldgsts-nvfp4-swiglu",
        ),
        pytest.param(
            {
                **_COMMON,
                "problem_n": 256,
                "problem_k": 256,
                "act_kind": int(ActKind.NONE),
                "route_sfs_act": int(RouteImpl.NONE),
                "sf_layout_b": int(SfLayout.R128c4),
                "sf_layout_c": int(SfLayout.R128c4),
                "dtype_c": int(DType.BF16),
                "tile_k": 256,
                "epi_tile_n": 128,
                "num_stages_a": 6,
                "num_stages_b": 6,
                "num_stages_smem_sfa": 6,
                "num_stages_smem_sfb": 6,
                "num_stages_tmem_sfa": 6,
                "num_stages_tmem_sfb": 6,
                "num_stages_c_smem": 1,
                "num_load_a_warps": 1,
                "num_load_b_warps": 1,
                "num_load_sfa_warps": 1,
                "num_load_sfb_warps": 1,
            },
            id="fc2-r128c4-bf16",
        ),
    ],
)
def test_host_input_generation(config):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=1,
        num_tokens=128,
        top_k=1,
        input_device="cpu",
        **config,
    )
