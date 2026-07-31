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

from flashinfer.fused_moe.backends.prims_ts.fp4_op import _resolve_routing_inputs
from flashinfer.fused_moe.shared.inputs import RoutingInputMode
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    DType,
    RouteImpl,
    TileScheduler,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.moe.config_mapper import (
    _DType,
    _make_json_moe_config_pair,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_nvfp4_moe_tactics,
)
from flashinfer.prims_ts.moe.runner import _split_token_tile_metadata
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import is_sm100a_supported


def _find_bs1_ldgsts_persistent_pair():
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


def test_from_logits_uses_bf16_routed_weight_storage():
    logits = torch.empty((4, 16), dtype=torch.float32)
    hidden_states = torch.empty((4, 32), dtype=torch.bfloat16)

    resolved_logits, resolved_ids, resolved_weights = _resolve_routing_inputs(
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        topk_ids=None,
        topk_weights=None,
        hidden_states=hidden_states,
    )

    assert resolved_logits is logits
    assert resolved_ids.dtype == torch.int32
    assert resolved_weights.dtype == torch.bfloat16


def test_nvfp4_tile256_pair_reuses_metadata_for_tile128_fc1():
    pair = map_trtllm_nvfp4_moe_tactic(
        [256, 0],
        num_tokens=8192,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert fc1.tile_n == 128
    assert fc1.metadata_tile_n == 256
    assert fc2.tile_n == 256
    assert fc2.metadata_tile_n == 256
    assert fc1.num_stages_a == 3
    assert fc2.num_stages_a == 4


def test_split_token_tile_metadata_reuses_input_tensors():
    tile_idx = torch.arange(4, dtype=torch.int32)
    mn_limit = torch.arange(4, dtype=torch.int32)
    num_non_exiting_ctas = torch.ones(1, dtype=torch.int32)

    result = _split_token_tile_metadata(
        tile_idx=tile_idx,
        mn_limit=mn_limit,
        num_non_exiting_ctas=num_non_exiting_ctas,
        source_tile_n=256,
        target_tile_n=128,
    )

    assert result[0] is tile_idx
    assert result[1] is mn_limit
    assert result[2] is num_non_exiting_ctas


def test_validation_rejects_nonintegral_metadata_tile_ratio():
    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=128,
        metadata_tile_n=192,
    )

    with pytest.raises(ValueError, match="metadata_tile_n must be a positive multiple"):
        validate_config(cfg)
