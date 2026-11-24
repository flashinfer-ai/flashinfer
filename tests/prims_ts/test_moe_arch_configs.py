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

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    build_batched_gemm_task_manager,
)
from flashinfer.prims_ts.moe.config_mapper import (
    _expanded_prims_ts_json_configs,
    map_trtllm_bf16_moe_tactic,
    map_trtllm_fp8_per_tensor_moe_tactic,
    map_trtllm_mxfp4_mxfp8_moe_tactic,
    map_trtllm_nvfp4_moe_tactic,
)
from flashinfer.tllm_enums import ActivationType


def test_default_mapper_resource_follows_runtime_sm_arch(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        batched_gemm_run, "is_prims_ts_arch_available", lambda major, minor: True
    )

    configs = _expanded_prims_ts_json_configs()
    assert any(cfg.options.get("mma_k") == 128 for cfg in configs)
    assert not any(
        cfg.options.get("dtype_a") == "e2m1" and cfg.options.get("mma_k") == 64
        for cfg in configs
    )


@pytest.mark.parametrize(
    ("mapper", "expected_mma_k"),
    (
        (map_trtllm_nvfp4_moe_tactic, 128),
        (map_trtllm_fp8_per_tensor_moe_tactic, 64),
        (map_trtllm_mxfp4_mxfp8_moe_tactic, 64),
    ),
)
def test_rubin_mapper_materializes_2x_mma_k(mapper, expected_mma_k):
    pair = mapper([-1, -1], sm_arch=107)
    for gemm in (pair.fc1, pair.fc2):
        assert gemm.cfg.kwargs["sm_arch"] == 107
        assert gemm.cfg.kwargs["mma_k"] == expected_mma_k
        validate_config(gemm.cfg.build())
        build_batched_gemm_task_manager(
            num_experts=2,
            num_tokens=128,
            top_k=1,
            verbose=False,
            **gemm.cfg.kwargs,
        )


def test_blackwell_mapper_never_exposes_rubin_mma_k():
    configs = _expanded_prims_ts_json_configs(100)
    assert all(cfg.options.get("mma_k") != 128 for cfg in configs)
    assert all(
        cfg.options.get("mma_k") != 64
        for cfg in configs
        if cfg.options.get("dtype_a") in ("e4m3", "mxe2m1", "mxe4m3")
        and not cfg.options.get("use_deepseek_fp8", False)
    )


def test_silu_reuses_relu2_row_with_non_gated_activation():
    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        sm_arch=100,
        activation_type=int(ActivationType.Silu),
    )
    fc1_cfg = pair.fc1.cfg.build()
    assert fc1_cfg.act_kind == int(ActKind.SILU)
    assert fc1_cfg.has_eltwise_act
    assert not fc1_cfg.has_gated_epilogue
