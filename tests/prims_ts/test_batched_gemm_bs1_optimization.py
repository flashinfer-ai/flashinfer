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

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    DType,
    RouteImpl,
    TileScheduler,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    build_batched_gemm_task_manager,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _cuda_kernel_ops
from flashinfer.prims_ts.moe.config_mapper import (
    _DType,
    _make_json_moe_config_pair,
    valid_prims_ts_nvfp4_moe_tactics,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import is_sm100a_supported


def _find_bs1_ldgsts_persistent_pair(*, enable_pdl=False):
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        num_tokens=1,
        top_k=8,
        num_local_experts=256,
    )

    for tactic in tactics:
        pair = _make_json_moe_config_pair(
            tile_n=tactic[0],
            moe_config_index=tactic[1],
            activation_type=int(ActivationType.Swiglu),
            dtype_a=int(_DType.E2M1),
            dtype_b=int(_DType.E2M1),
            fc1_dtype_c=int(_DType.E2M1),
            fc2_dtype_c=int(_DType.BF16),
            dtype_label="NVFP4xNVFP4",
            enable_pdl=enable_pdl,
        )
        if pair is None:
            continue
        fc1 = pair.fc1.cfg.kwargs
        fc2 = pair.fc2.cfg.kwargs
        if (
            fc1["tile_n"] == 8
            and fc1["tile_k"] == 512
            and fc1["cluster_m"] == 1
            and fc1["route_act"] == int(RouteImpl.LDGSTS)
            and fc1["route_sfs_act"] == int(RouteImpl.LDGSTS)
            and fc1["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc1["use_clc_fast_drain"] == 1
            and fc1["use_work_throttle"] == 1
            and fc2["tile_n"] == 8
            and fc2["tile_k"] == 512
            and fc2["cluster_m"] == 1
            and fc2["num_stages_a"] == 4
            and fc2["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc2["use_clc_fast_drain"] == 0
            and fc2["use_work_throttle"] == 0
        ):
            return pair

    return None


def test_nvfp4_search_contains_bs1_ldgsts_persistent_pair():
    assert _find_bs1_ldgsts_persistent_pair() is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 PrimsTS kernels require Blackwell SM100A+",
)
@pytest.mark.parametrize(
    (
        "stage",
        "problem_n",
        "problem_k",
        "num_experts",
        "top_k",
        "early_exit_max_token_ctas",
        "dtype_c_override",
        "seed",
    ),
    (
        pytest.param(
            "fc1", 4096, 7168, 16, 8, 16, int(DType.BF16), 42, id="fc1-mainloop"
        ),
        pytest.param("fc1", 256, 512, 2, 1, 4, None, 123, id="fc1-quantized-output"),
        pytest.param("fc2", 7168, 2048, 16, 8, 16, None, 42, id="fc2"),
    ),
)
def test_bs1_deepseek_persistent_pair_gpu_correctness(
    stage,
    problem_n,
    problem_k,
    num_experts,
    top_k,
    early_exit_max_token_ctas,
    dtype_c_override,
    seed,
):
    """Exercise the BS=1 tactic, including skipped persistent work tiles."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = _find_bs1_ldgsts_persistent_pair()
    assert pair is not None
    cfg = dict(pair.fc1.cfg.kwargs if stage == "fc1" else pair.fc2.cfg.kwargs)
    if dtype_c_override is not None:
        # The standalone random-input reference overstates E2M1 quantization
        # error for DeepSeek's large K. Use BF16 here to isolate the exact FC1
        # LDGSTS/persistent mainloop; the second FC1 row covers its real E2M1
        # output epilogue with the same tile and pipeline schedule.
        cfg["dtype_c"] = dtype_c_override

    assert reference_check(
        num_experts=num_experts,
        num_tokens=1,
        top_k=top_k,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=seed,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
        **cfg,
    )


def test_bs1_persistent_early_exit_pdl_wait_completes_before_task_graph():
    from cutlass.experimental.task_scheduling.resources import PdlWaitBarrier

    pair = _find_bs1_ldgsts_persistent_pair(enable_pdl=True)
    assert pair is not None

    task_manager = build_batched_gemm_task_manager(
        num_experts=256,
        num_tokens=1,
        top_k=8,
        early_exit_max_token_ctas=8,
        verbose=False,
        **pair.fc1.cfg.kwargs,
    )
    all_task_resources = [
        resource
        for task in task_manager.tasks
        for resource in task.src_resources + task.dst_resources
    ]

    assert task_manager._assume_pdl_wait_completed is True
    assert not any(
        isinstance(resource, PdlWaitBarrier) for resource in all_task_resources
    )


def test_cuda_kernel_search_handles_nested_mlir_regions():
    def make_op(name, *children):
        regions = []
        if children:
            regions.append(
                SimpleNamespace(blocks=[SimpleNamespace(operations=list(children))])
            )
        operation = SimpleNamespace(name=name, regions=regions)
        return SimpleNamespace(operation=operation)

    kernel = make_op("cuda.kernel")
    nested = make_op("test.level2", kernel)
    top_level = make_op("test.level1", nested)
    module = SimpleNamespace(operation=make_op("builtin.module", top_level).operation)

    assert list(_cuda_kernel_ops(module)) == [kernel]
