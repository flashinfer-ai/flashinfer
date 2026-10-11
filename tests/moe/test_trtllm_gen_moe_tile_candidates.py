"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tile_N candidates offered to the autotuner by the TRT-LLM-gen fused MoE
launchers under expert parallelism.

``trtllm_get_valid_moe_configs`` takes both the local and the global expert
count. With ``num_experts == num_local_experts`` the candidates are the tiles
around ``num_tokens * top_k / num_local_experts`` (unchanged behaviour). With
``num_experts > num_local_experts`` they also include the tile covering
``num_tokens * top_k / num_experts`` (the per-expert row count when every
received row carries a global expert id) when that tile is below every local
candidate.
"""

import pytest
import torch

from flashinfer import ActivationType, RoutingMethodType, autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import WeightLayout, trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.backends.trtllm import MoERunner as TrtllmMoERunner
from flashinfer.fused_moe.core import Fp8QuantizationType, get_trtllm_moe_sm100_module
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
from flashinfer.tllm_enums import DtypeTrtllmGen
from flashinfer.utils import get_compute_capability

_BASE_TILES = (8, 16, 32, 64, 128)

# (dtype_act, dtype_weights, fp8_quantization_type, use_shuffled_weight,
#  weight_layout, tile ladder of the launcher that serves this combination).
_LAUNCHERS = {
    "Bf16": (
        DtypeTrtllmGen.Bfloat16,
        DtypeTrtllmGen.Bfloat16,
        Fp8QuantizationType.NoneFp8,
        True,
        WeightLayout.BlockMajorK,
        _BASE_TILES,
    ),
    "Fp8PerTensor": (
        DtypeTrtllmGen.E4m3,
        DtypeTrtllmGen.E4m3,
        Fp8QuantizationType.PerTensorFp8,
        True,
        WeightLayout.MajorK,
        _BASE_TILES,
    ),
    "Fp8PerChannel": (
        DtypeTrtllmGen.E4m3,
        DtypeTrtllmGen.E4m3,
        Fp8QuantizationType.PerChannelFp8,
        True,
        WeightLayout.MajorK,
        _BASE_TILES,
    ),
    "DeepSeekFp8": (
        DtypeTrtllmGen.E4m3,
        DtypeTrtllmGen.E4m3,
        Fp8QuantizationType.DeepSeekFp8,
        False,
        WeightLayout.MajorK,
        _BASE_TILES,
    ),
    "MxFp8": (
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxE4m3,
        Fp8QuantizationType.MxFp8,
        True,
        WeightLayout.MajorK,
        _BASE_TILES + (256,),
    ),
    "MxInt4": (
        DtypeTrtllmGen.Bfloat16,
        DtypeTrtllmGen.MxInt4,
        Fp8QuantizationType.NoneFp8,
        True,
        WeightLayout.BlockMajorK,
        _BASE_TILES,
    ),
    "NvFP4xNvFP4": (
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.E2m1,
        Fp8QuantizationType.NoneFp8,
        True,
        WeightLayout.MajorK,
        _BASE_TILES + (192, 256),
    ),
    "MxFP4xMxFP8": (
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.MxE2m1,
        Fp8QuantizationType.NoneFp8,
        True,
        WeightLayout.MajorK,
        _BASE_TILES + (192, 256),
    ),
    "MxFP4xBf16": (
        DtypeTrtllmGen.Bfloat16,
        DtypeTrtllmGen.MxE2m1,
        Fp8QuantizationType.NoneFp8,
        True,
        WeightLayout.MajorK,
        (8, 16, 32, 64),
    ),
}

_HIDDEN_SIZE = 4096
_INTERMEDIATE_SIZE = 2048

# (num_tokens, top_k, num_local_experts, num_experts): every row a rank
# received under expert parallelism, e.g. 4 ranks x 256 tokens over 128 experts.
_EP_SHAPES = [
    pytest.param(4 * 32, 8, 32, 128, id="E128-EP4-32tok"),
    pytest.param(4 * 256, 8, 32, 128, id="E128-EP4-256tok"),
    pytest.param(8 * 32, 8, 16, 128, id="E128-EP8-32tok"),
    pytest.param(8 * 256, 8, 32, 256, id="E256-EP8-256tok"),
    pytest.param(16 * 128, 8, 24, 384, id="E384-EP16-128tok"),
]


def _require_sm100_family():
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("TRT-LLM-gen fused MoE requires an SM100-family GPU.")


def _ladder(name: str) -> tuple:
    return _LAUNCHERS[name][5]


def _reference_tiles(ladder, num_tokens, top_k, num_experts) -> set:
    """Mirror of computeSelectedTileN's autotuner window: the first tile covering
    the mean rows per expert, plus its -2 / -1 / +1 / +2 neighbours on the ladder."""
    avg = num_tokens * top_k / num_experts
    center = next((i for i, t in enumerate(ladder) if t >= avg), len(ladder) - 1)
    return set(ladder[max(center - 2, 0) : center + 3])


def _reference_ep_tiles(
    ladder, num_tokens, top_k, num_local_experts, num_experts
) -> set:
    """Mirror of computeCandidateTileN: the local window plus the tile covering
    the global estimate when that tile is below the whole local window."""
    local = _reference_tiles(ladder, num_tokens, top_k, num_local_experts)
    avg = num_tokens * top_k / num_experts
    covering = next((t for t in ladder if t >= avg), ladder[-1])
    return local | {covering} if covering < min(local) else local


def _offered_tiles(
    name, num_tokens, top_k, num_local_experts, num_experts, lora=False
) -> set:
    dtype_act, dtype_weights, fp8_type, shuffled, layout, _ = _LAUNCHERS[name]
    moe_op = get_trtllm_moe_sm100_module().moe_op
    tactics = moe_op.trtllm_get_valid_moe_configs(
        dtype_act,
        dtype_weights,
        fp8_type,
        top_k,
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        _INTERMEDIATE_SIZE,
        num_local_experts,
        num_experts,
        ActivationType.Swiglu.value,
        shuffled,
        layout.value,
        False,  # use_per_token_scaling
        num_tokens,
        lora,  # has_gemm1_lora_delta
    )
    return {int(t[0]) for t in tactics}


@pytest.mark.parametrize("name", list(_LAUNCHERS))
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 512, 4096])
def test_tile_candidates_unchanged_without_expert_parallelism(name, num_tokens):
    """num_experts == num_local_experts offers exactly the local-estimate tiles."""
    _require_sm100_family()
    top_k, num_experts = 8, 128
    offered = _offered_tiles(name, num_tokens, top_k, num_experts, num_experts)
    assert offered == _reference_tiles(_ladder(name), num_tokens, top_k, num_experts)


@pytest.mark.parametrize("name", list(_LAUNCHERS))
@pytest.mark.parametrize("num_tokens,top_k,num_local_experts,num_experts", _EP_SHAPES)
def test_tile_candidates_under_expert_parallelism(
    name, num_tokens, top_k, num_local_experts, num_experts
):
    """num_experts > num_local_experts keeps every local-estimate tile and adds
    at most one: the tile covering the global estimate, only if it is smaller
    than every local-estimate tile."""
    _require_sm100_family()
    ladder = _ladder(name)
    local_only = _offered_tiles(
        name, num_tokens, top_k, num_local_experts, num_local_experts
    )
    offered = _offered_tiles(name, num_tokens, top_k, num_local_experts, num_experts)

    assert local_only == _reference_tiles(ladder, num_tokens, top_k, num_local_experts)
    assert offered >= local_only
    assert offered == _reference_ep_tiles(
        ladder, num_tokens, top_k, num_local_experts, num_experts
    )
    global_avg = num_tokens * top_k / num_experts
    covering = next((t for t in ladder if t >= global_avg), ladder[-1])
    assert covering in offered
    assert offered - local_only <= {covering}
    assert all(t < min(local_only) for t in offered - local_only)


@pytest.mark.parametrize("name", ["NvFP4xNvFP4", "MxFP4xMxFP8"])
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("num_local_experts,num_experts", [(32, 128), (32, 256)])
def test_tile_candidates_with_lora_never_add_tile_192(
    name, num_tokens, num_local_experts, num_experts
):
    """With a gemm1 LoRA delta the 192 tile has no kernel, so offering it fails
    the whole query. Expert parallelism must not add it where the local-estimate
    set does not already contain it."""
    _require_sm100_family()
    top_k = 8
    ladder = _ladder(name)
    local_ref = _reference_tiles(ladder, num_tokens, top_k, num_local_experts)
    if 192 in local_ref:
        pytest.skip("192 is already offered without expert parallelism.")
    offered = _offered_tiles(
        name, num_tokens, top_k, num_local_experts, num_experts, lora=True
    )
    assert 192 not in offered
    assert offered == _reference_ep_tiles(
        ladder, num_tokens, top_k, num_local_experts, num_experts
    )


def test_autotuner_queries_global_expert_count(monkeypatch):
    """The FP4 runner passes the global expert count, so tuning an
    expert-parallel call sees the small tiles."""
    _require_sm100_family()
    seen = []
    original = TrtllmMoERunner.get_valid_tactics

    def _spy(self, inputs, profile):
        tactics = original(self, inputs, profile)
        num_rows = MoeRunnerInputs.from_list(inputs).hidden_states.shape[0]
        seen.append((self.num_experts, num_rows, {int(t[0]) for t in tactics}))
        return tactics

    monkeypatch.setattr(TrtllmMoERunner, "get_valid_tactics", _spy)
    TrtllmMoERunner.valid_tactics_dict.clear()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()

    device = torch.device("cuda:0")
    num_experts, num_local_experts, top_k = 256, 32, 8
    hidden_size, intermediate_size, num_tokens = 1024, 512, 256
    routing_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    routing_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    # Weight contents are irrelevant here: only the offered tactics are checked.
    gemm1_weights = torch.empty(
        num_local_experts,
        intermediate_size * 2,
        hidden_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    gemm1_weights_scale = torch.ones(
        num_local_experts,
        intermediate_size * 2,
        hidden_size // 2 // 16,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    gemm2_weights = torch.empty(
        num_local_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    gemm2_weights_scale = torch.ones(
        num_local_experts,
        hidden_size,
        intermediate_size // 2 // 16,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    with autotune(tune_mode=True):
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=8,
            topk_group=4,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_local_experts,
            routed_scaling_factor=2.5,
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
            do_finalize=True,
            output=output,
            tune_max_num_tokens=num_tokens,
        )

    assert all(n == num_experts for n, _, _ in seen)
    offered = [tiles for _, rows, tiles in seen if rows == num_tokens]
    assert offered, "the autotuner did not query the FP4 runner at 256 rows"
    # 256 rows x top-8 over 32 local experts estimate 64 rows per expert, whose
    # candidates start at tile 16; over 256 global experts the estimate is 8 rows,
    # which brings in tile 8.
    assert all(8 in tiles for tiles in offered)
