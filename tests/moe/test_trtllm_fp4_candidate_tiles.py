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

"""Input row count must not prune valid small tiles from FP4 autotuning."""

import pytest
import torch

from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
)
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize(
    "dtype_act,dtype_weights",
    [
        pytest.param(DtypeTrtllmGen.E2m1, DtypeTrtllmGen.E2m1, id="nvfp4"),
        pytest.param(DtypeTrtllmGen.MxE4m3, DtypeTrtllmGen.MxE2m1, id="mxfp8-mxfp4"),
        pytest.param(DtypeTrtllmGen.Bfloat16, DtypeTrtllmGen.MxE2m1, id="bf16-mxfp4"),
    ],
)
@pytest.mark.parametrize("num_tokens", [128, 512])
@pytest.mark.parametrize("num_local_experts", [4, 16])
def test_fp4_candidate_tiles_include_small_tiles(
    dtype_act, dtype_weights, num_tokens, num_local_experts
):
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("Requires the SM10x TRT-LLM MoE backend")

    # Synthetic dimensions shared with existing small-shape MoE correctness
    # tests. Candidate enumeration must not treat all input rows as local work.
    module = get_trtllm_moe_sm100_module().moe_op
    tactics = module.trtllm_get_valid_moe_configs(
        dtype_act,
        dtype_weights,
        Fp8QuantizationType.NoneFp8,
        2,  # top_k
        1024,  # hidden_size
        1024,  # hidden_size_output
        1024,  # intermediate_size
        num_local_experts,
        ActivationType.Swiglu.value,
        True,  # use_shuffled_weight
        WeightLayout.MajorK.value,
        False,  # use_per_token_scaling
        num_tokens,
        False,  # has_gemm1_lora_delta
    )
    tiles = {int(tactic[0]) for tactic in tactics}
    required_tiles = {8, 16, 32, 64}
    if dtype_act != DtypeTrtllmGen.Bfloat16:
        required_tiles.update({128, 256})
    assert required_tiles <= tiles, f"Missing supported tiles: {required_tiles - tiles}"
