"""Integration tests for TRTLLM fused MoE launcher with autotuner."""

import contextlib
import pytest
import torch

from flashinfer import autotune, RoutingMethodType
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import (
    get_compute_capability,
    next_positive_power_of_2,
    last_positive_power_of_2,
)
from flashinfer.fused_moe.utils import make_random_topk_ids
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


@pytest.mark.parametrize("routed", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_experts", [16])
@pytest.mark.parametrize("top_k", [4])
def test_fp4_moe_autotune(
    routed: bool,
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Smoke test: trtllm_fp4_block_scale_moe (routing_logits path) and
    trtllm_fp4_block_scale_routed_moe (pre-computed topk_ids path) must not crash
    during autotuning.
    """
    _require_sm100()
    reset_autotuner()
    device = torch.device("cuda:0")

    from flashinfer.fused_moe import (
        trtllm_fp4_block_scale_moe,
        trtllm_fp4_block_scale_routed_moe,
    )

    hidden_size = 3072
    intermediate_size = 3072

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    gemm1_weights = torch.empty(
        num_experts,
        intermediate_size * 2,
        hidden_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    gemm1_weights_scale = torch.empty(
        num_experts,
        intermediate_size * 2,
        hidden_size // 2 // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    gemm2_weights = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    gemm2_weights_scale = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size // 2 // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    common_kwargs = dict(
        routing_bias=None,
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
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        output=output,
        tune_max_num_tokens=1,
    )

    with autotune(tune_mode=True):
        if routed:
            topk_ids = make_random_topk_ids(num_experts, num_tokens, top_k, device)
            topk_weights = torch.ones(
                num_tokens, top_k, dtype=torch.bfloat16, device=device
            )
            packed_topk_ids = (topk_ids.to(torch.int32) << 16) | topk_weights.view(
                torch.int16
            ).to(torch.int32)
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed_topk_ids,
                **common_kwargs,
            )
        else:
            routing_logits = torch.rand(
                num_tokens, num_experts, dtype=torch.bfloat16, device=device
            )
            trtllm_fp4_block_scale_moe(
                routing_logits=routing_logits,
                **common_kwargs,
            )


@pytest.mark.parametrize("routed", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("num_experts", [16])
@pytest.mark.parametrize("top_k", [4])
def test_fp8_moe_autotune(
    routed: bool,
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Smoke test: trtllm_fp8_block_scale_moe (routing_logits path) and
    trtllm_fp8_block_scale_routed_moe (pre-computed topk_ids path) must not crash
    during autotuning.
    """
    _require_sm100()
    reset_autotuner()
    device = torch.device("cuda:0")

    from flashinfer.fused_moe import (
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_block_scale_routed_moe,
    )

    hidden_size = 512
    intermediate_size = 512

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    hidden_states_scale = torch.ones(
        hidden_size // 128, num_tokens, dtype=torch.float32, device=device
    )
    gemm1_weights = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    gemm2_weights = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        dtype=torch.float32,
        device=device,
    )

    common_kwargs = dict(
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        tune_max_num_tokens=1,
    )

    with autotune(tune_mode=True):
        if routed:
            topk_ids = make_random_topk_ids(num_experts, num_tokens, top_k, device)
            topk_weights = torch.ones(
                num_tokens, top_k, dtype=torch.bfloat16, device=device
            )
            packed_topk_ids = (topk_ids.to(torch.int32) << 16) | topk_weights.view(
                torch.int16
            ).to(torch.int32)
            trtllm_fp8_block_scale_routed_moe(
                topk_ids=packed_topk_ids,
                **common_kwargs,
            )
        else:
            routing_logits = torch.rand(
                num_tokens, num_experts, dtype=torch.bfloat16, device=device
            )
            trtllm_fp8_block_scale_moe(
                routing_logits=routing_logits,
                **common_kwargs,
            )


@pytest.mark.parametrize("num_fused_shared_experts", [1, 2])
def test_fp8_block_scale_moe_fused_shared_experts_autotune(num_fused_shared_experts):
    """Fused shared experts must agree with the autotuner on the fused dimensions.

    Regression guard for the valid-tactic enumeration path: ``get_valid_tactics``
    must query ``trtllm_get_valid_moe_configs`` with the fused (routed + shared)
    ``top_k`` / ``local_num_experts`` so the chosen tactic passes
    ``prepare_moe()``'s ``effectiveTopK`` / ``effectiveLocalExperts`` validation.
    Shared experts only flow through the DeepSeekV3 routing path. Smoke test:
    must not crash during autotuning.
    """
    _require_sm100()
    reset_autotuner()
    device = torch.device("cuda:0")

    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    num_experts = 256  # routed experts
    top_k = 8
    n_group = 8
    topk_group = 4
    hidden_size = 1024
    intermediate_size = 512
    num_tokens = 8
    # Weight tensors carry the routed + fused shared experts in the expert dim.
    total_experts = num_experts + num_fused_shared_experts

    routing_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    routing_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    hidden_states_scale = torch.ones(
        hidden_size // 128, num_tokens, dtype=torch.float32, device=device
    )
    gemm1_weights = (
        torch.randn(total_experts, 2 * intermediate_size, hidden_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm1_weights_scale = torch.ones(
        total_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    gemm2_weights = (
        torch.randn(total_experts, hidden_size, intermediate_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm2_weights_scale = torch.ones(
        total_experts,
        hidden_size // 128,
        intermediate_size // 128,
        dtype=torch.float32,
        device=device,
    )

    with autotune(tune_mode=True):
        trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=2.5,
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
            num_fused_shared_experts=num_fused_shared_experts,
            tune_max_num_tokens=8,
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


def test_fp8_moe_inference_loop_no_lru_cache_leak():
    """Regression test: repeated FP8 MoE inference with varying token counts must not
    cause unbounded AutoTuner._find_nearest_profile lru_cache growth.

    Root cause of the leak: _make_tuning_config (called on every inference call)
    constructs a fresh TuningConfig containing a new lambda closure each time.
    DynamicTensorSpec hashes callables by object identity (id()), so each fresh
    construction yields a new lru_cache key regardless of whether the shape changed.
    The result is one new cache entry per inference call instead of one per unique
    bucket mapping.

    After the fix, cache growth should be bounded by the number of distinct token
    bucket values, not by the number of inference calls.
    """
    import tracemalloc

    _require_sm100()
    reset_autotuner()
    device = torch.device("cuda:0")

    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

    num_experts = 16
    top_k = 4
    hidden_size = 512
    intermediate_size = 512
    tune_max_num_tokens = 8192
    # Use enough iterations to make unbounded growth obvious while keeping the
    # test fast; 200 calls across ~20 distinct bucket values is sufficient.
    N = 200

    gemm1_weights = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    gemm2_weights = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        dtype=torch.float32,
        device=device,
    )

    def _make_fp8_inputs(num_tokens: int):
        hidden_states = (
            torch.randn(num_tokens, hidden_size, device=device)
            .clamp(-1, 1)
            .to(torch.float8_e4m3fn)
        )
        # DeepSeekFp8 scale layout: [hidden_size // 128, num_tokens]
        hidden_states_scale = torch.ones(
            hidden_size // 128, num_tokens, dtype=torch.float32, device=device
        )
        routing_logits = torch.rand(
            num_tokens, num_experts, dtype=torch.bfloat16, device=device
        )
        return hidden_states, hidden_states_scale, routing_logits

    common_kwargs = dict(
        routing_bias=None,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        tune_max_num_tokens=tune_max_num_tokens,
    )

    # Tune once at a representative token count before the measurement window.
    h, hs, r = _make_fp8_inputs(128)
    with autotune(tune_mode=True):
        trtllm_fp8_block_scale_moe(
            routing_logits=r,
            hidden_states=h,
            hidden_states_scale=hs,
            **common_kwargs,
        )
    torch.cuda.synchronize()

    cache_before = AutoTuner._find_nearest_profile.cache_info().currsize
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Inference loop: random token counts in [1, 512] to cover many distinct
    # bucket values while keeping tensor allocation minimal.
    torch.manual_seed(0)
    token_counts = torch.randint(1, 513, (N,)).tolist()
    distinct_counts = len(set(token_counts))

    for num_tokens in token_counts:
        h, hs, r = _make_fp8_inputs(num_tokens)
        trtllm_fp8_block_scale_moe(
            routing_logits=r,
            hidden_states=h,
            hidden_states_scale=hs,
            **common_kwargs,
        )

    torch.cuda.synchronize()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    cache_after = AutoTuner._find_nearest_profile.cache_info().currsize
    cache_growth = cache_after - cache_before

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    allocated_kb = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024

    print(
        f"\n{N} inference calls, {distinct_counts} distinct token counts. "
        f"lru_cache grew by {cache_growth} entries ({allocated_kb:.1f} KB Python)."
    )

    # The cache must grow by at most one entry per distinct token bucket mapping —
    # a small constant determined by the tuning bucket grid, not by call count.
    # tune_max_num_tokens=8192 with get_hybrid_num_tokens_buckets produces ~20
    # buckets; allow 2× headroom.
    max_expected_cache_growth = 40
    assert cache_growth <= max_expected_cache_growth, (
        f"lru_cache grew by {cache_growth} entries across {N} inference calls "
        f"({distinct_counts} distinct token counts). "
        f"Expected growth bounded by tuning bucket count (~{max_expected_cache_growth}). "
        f"TuningConfig is likely being reconstructed on every call — "
        f"move it to a module-level or class-level singleton."
    )
