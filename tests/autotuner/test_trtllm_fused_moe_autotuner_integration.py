"""Integration tests for TRTLLM fused MoE launcher with autotuner."""

import contextlib
import pytest
import torch

from flashinfer import autotune, RoutingMethodType
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability
from .utils import reset_autotuner

TUNE_MAX = 8192


def _prepare_bf16_moe_weights(num_experts, intermediate_size, hidden_size, device):
    """Prepare shuffled BF16 weights in BlockMajorK layout."""
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout

    gemm1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
    )
    g1_shuffled, g2_shuffled = [], []
    for i in range(num_experts):
        g1_shuffled.append(
            convert_to_block_layout(
                shuffle_matrix_a(gemm1[i].view(torch.uint8), 64), 128
            )
        )
        g2_shuffled.append(
            convert_to_block_layout(
                shuffle_matrix_a(gemm2[i].view(torch.uint8), 64), 128
            )
        )
    return (
        torch.stack(g1_shuffled).view(torch.bfloat16),
        torch.stack(g2_shuffled).view(torch.bfloat16),
    )


def _overwrite_cached_tactic_for_op(custom_op: str, new_tactic):
    """Overwrite cached tactics for one op key."""
    tuner = AutoTuner.get()
    updated = 0
    for key, (runner_id, _tactic, profile) in list(tuner.profiling_cache.items()):
        if key[0] == custom_op:
            tuner.profiling_cache[key] = (runner_id, new_tactic, profile)
            updated += 1
    assert updated > 0, f"No autotuner cache entries found for {custom_op}"


def _tune_bf16_moe_once(
    *,
    device,
    tune_num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    gemm1_weights,
    gemm2_weights,
    tune_max: int,
):
    from flashinfer.fused_moe import trtllm_bf16_moe, WeightLayout

    routing_tune = torch.rand(
        tune_num_tokens, num_experts, device=device, dtype=torch.bfloat16
    )
    hidden_tune = torch.randn(
        tune_num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    with autotune(tune_mode=True):
        trtllm_bf16_moe(
            routing_logits=routing_tune,
            routing_bias=None,
            hidden_states=hidden_tune,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize.value,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            tune_max_num_tokens=tune_max,
        )


def _run_bf16_moe_infer(
    *,
    device,
    infer_num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    gemm1_weights,
    gemm2_weights,
    tune_max: int,
):
    from flashinfer.fused_moe import trtllm_bf16_moe, WeightLayout

    routing_infer = torch.rand(
        infer_num_tokens, num_experts, device=device, dtype=torch.bfloat16
    )
    hidden_infer = torch.randn(
        infer_num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    output = trtllm_bf16_moe(
        routing_logits=routing_infer,
        routing_bias=None,
        hidden_states=hidden_infer,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        tune_max_num_tokens=tune_max,
    )
    assert output.shape[0] == infer_num_tokens
    assert output.isfinite().all(), "Output should be finite"


def _compute_selected_tile_n_base_element(
    num_tokens: int, top_k: int, num_experts: int
) -> int:
    """Compute the base element used by computeSelectedTileN(num_tokens) to filter tile_N candidates."""
    from flashinfer.fused_moe.utils import next_positive_power_of_2

    return min(next_positive_power_of_2(int(num_tokens * top_k / num_experts)), 256)


def _make_tile_bias(target_tile_n: int):
    """Return a profiling stub that favours *target_tile_n* over every other tile."""

    def _bias(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        tile_n = -1
        with contextlib.suppress(BaseException):
            tile_n = tactic[0]
        return 1.0 if tile_n == target_tile_n else 5.0

    return _bias


def _require_sm100():
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("This test requires an SM100/SM103 (Blackwell) GPU.")


@pytest.mark.parametrize(
    "top_k,num_experts",
    [
        (2, 8),
        (4, 16),
        (8, 128),  # Qwen3-VL-MoE-like (text)
        (10, 128),  # Routing kernel currently supports top_k <= 10
    ],
)
@pytest.mark.parametrize(
    "tune_num_tokens,infer_num_tokens",
    [
        (256, 500),
    ],
)
def test_bf16_moe_all_supported_tile_n_inference_succeed(
    monkeypatch,
    top_k: int,
    num_experts: int,
    tune_num_tokens: int,
    infer_num_tokens: int,
):
    """SM100 BF16 integration: Test that MoE works when given any supported tileN value,
    including values filtered out by computeSelectedTileN for the given inference num tokens.
    """
    from flashinfer.fused_moe.utils import last_positive_power_of_2

    _require_sm100()
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    hidden_size = 1024
    intermediate_size = 1024

    gemm1_weights, gemm2_weights = _prepare_bf16_moe_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    tile_n_base_autotune_cache = _compute_selected_tile_n_base_element(
        tune_num_tokens, top_k, num_experts
    )
    tile_n_base_inference = _compute_selected_tile_n_base_element(
        infer_num_tokens, top_k, num_experts
    )
    assert tile_n_base_autotune_cache < tile_n_base_inference, (
        "Test setup error: autotuning tile_N base element should be smaller than inference tile_N base element to trigger the intended scenario"
    )
    assert last_positive_power_of_2(infer_num_tokens) == tune_num_tokens

    supported_tile_n_values = [8, 16, 32, 64, 128]
    for tile_n in supported_tile_n_values:
        reset_autotuner()
        monkeypatch.setattr(
            AutoTuner, "_profile_single_kernel", _make_tile_bias(tile_n)
        )
        _tune_bf16_moe_once(
            device=device,
            tune_num_tokens=tune_num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            tune_max=TUNE_MAX,
        )

        tuner = AutoTuner.get()
        assert len(tuner.profiling_cache) > 0, (
            "Autotuner cache should be populated after tuning"
        )

        _run_bf16_moe_infer(
            device=device,
            infer_num_tokens=infer_num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            tune_max=TUNE_MAX,
        )


@pytest.mark.parametrize(
    "invalid_tactic",
    [
        [4096, 0],  # unsupported tile_N
        [32, 10_000_000],  # invalid config index
        [32],  # malformed tactic payload
    ],
)
def test_bf16_moe_invalid_tactic_raises_runtime_error(monkeypatch, invalid_tactic):
    """SM100 integration: invalid tactics should fail."""
    _require_sm100()
    reset_autotuner()
    device = torch.device("cuda:0")

    hidden_size, intermediate_size = 1024, 1024
    top_k, num_experts = 4, 16
    tune_num_tokens, infer_num_tokens = 256, 500

    gemm1_weights, gemm2_weights = _prepare_bf16_moe_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", _make_tile_bias(32))
    _tune_bf16_moe_once(
        device=device,
        tune_num_tokens=tune_num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        tune_max=TUNE_MAX,
    )

    _overwrite_cached_tactic_for_op("flashinfer::trtllm_bf16_moe", invalid_tactic)
    with pytest.raises(RuntimeError):
        _run_bf16_moe_infer(
            device=device,
            infer_num_tokens=infer_num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            tune_max=TUNE_MAX,
        )
