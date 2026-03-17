"""Integration tests for TRTLLM MoE launcher fallback and wrapper contracts."""

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability

TUNE_MAX = 8192


def _reset_autotuner() -> AutoTuner:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    tuner.is_tuning_mode = False
    return tuner


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
            routing_method_type=1,  # Renormalize
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
        routing_method_type=1,  # Renormalize
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        tune_max_num_tokens=tune_max,
    )
    assert output.shape[0] == infer_num_tokens
    assert output.isfinite().all(), "Output should be finite"


def _require_sm100():
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("This test requires an SM100/SM103 (Blackwell) GPU.")


@pytest.mark.parametrize("fp8_quantization_type", ["DeepSeekFp8", "MxFp8"])
def test_fp8_block_scale_moe_deepseek_contract_args(monkeypatch, fp8_quantization_type):
    """Contract test: wrapper forwards DeepSeek-style routing/group arguments unchanged."""
    from flashinfer.fused_moe import core as moe_core

    captured = {}

    def fake_trtllm_fp8_block_scale_moe(self, *args):
        captured["args"] = args
        hidden_states = args[4]
        return [hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)]

    fake_module = type(
        "FakeMoeModule",
        (),
        {"trtllm_fp8_block_scale_moe": fake_trtllm_fp8_block_scale_moe},
    )()
    monkeypatch.setattr(moe_core, "get_trtllm_moe_sm100_module", lambda: fake_module)

    seq_len = 8
    hidden_size = 128
    intermediate_size = 256
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    routed_scaling_factor = 2.5
    tune_max_num_tokens = TUNE_MAX

    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
    routing_bias = torch.randn(num_experts, dtype=torch.float32)
    hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)
    hidden_states_scale = torch.ones(hidden_size // 128, seq_len, dtype=torch.float32)
    gemm1_weights = torch.empty(1, dtype=torch.float32)
    gemm1_weights_scale = torch.empty(1, dtype=torch.float32)
    gemm2_weights = torch.empty(1, dtype=torch.float32)
    gemm2_weights_scale = torch.empty(1, dtype=torch.float32)

    quant_type = getattr(moe_core.Fp8QuantizationType, fp8_quantization_type)
    output = moe_core.trtllm_fp8_block_scale_moe(
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
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=2,  # DeepSeekV3 routing mode
        use_shuffled_weight=True,
        weight_layout=moe_core.WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=True,
        tune_max_num_tokens=tune_max_num_tokens,
        fp8_quantization_type=quant_type,
    )

    assert output.shape == hidden_states.shape
    assert "args" in captured, (
        "Expected wrapper to call trtllm_fp8_block_scale_moe backend"
    )

    args = captured["args"]
    assert args[12] == top_k
    assert args[13] == n_group
    assert args[14] == topk_group
    assert args[18] == routed_scaling_factor
    assert args[19] == 2
    assert args[20] is True
    assert args[21] == int(moe_core.WeightLayout.BlockMajorK)
    assert args[24] == tune_max_num_tokens
    assert args[25] == quant_type


def test_fp4_block_scale_moe_contract_args(monkeypatch):
    """Contract test: FP4 wrapper forwards core MoE routing/kernel args unchanged."""
    from flashinfer.fused_moe import core as moe_core

    captured = {}

    def fake_trtllm_fp4_block_scale_moe(self, *args):
        captured["args"] = args
        hidden_states = args[4]
        return [hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)]

    fake_module = type(
        "FakeMoeModule",
        (),
        {"trtllm_fp4_block_scale_moe": fake_trtllm_fp4_block_scale_moe},
    )()
    monkeypatch.setattr(moe_core, "get_trtllm_moe_sm100_module", lambda: fake_module)

    seq_len = 8
    hidden_size = 128
    intermediate_size = 256
    num_experts = 128
    top_k = 8
    tune_max_num_tokens = TUNE_MAX

    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
    hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)
    hidden_states_scale = torch.ones(seq_len, hidden_size // 16, dtype=torch.float32)
    gemm1_weights = torch.empty(1, dtype=torch.uint8)
    gemm1_weights_scale = torch.empty(1, dtype=torch.float32)
    gemm2_weights = torch.empty(1, dtype=torch.uint8)
    gemm2_weights_scale = torch.empty(1, dtype=torch.float32)

    output = moe_core.trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
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
        routing_method_type=1,
        do_finalize=True,
        enable_pdl=True,
        activation_type=moe_core.ActivationType.Swiglu.value,
        output=None,
        tune_max_num_tokens=tune_max_num_tokens,
    )

    assert output[0].shape == hidden_states.shape
    assert "args" in captured, (
        "Expected wrapper to call trtllm_fp4_block_scale_moe backend"
    )

    args = captured["args"]
    assert args[18] == num_experts
    assert args[19] == top_k
    assert args[20] is None
    assert args[21] is None
    assert args[22] == intermediate_size
    assert args[24] == num_experts
    assert args[26] == 1
    assert args[27] is True
    assert args[31] == tune_max_num_tokens


def test_bf16_moe_qwen35_contract_args(monkeypatch):
    """Contract test: BF16 wrapper forwards Qwen3.5-style ungrouped routing args unchanged."""
    from flashinfer.fused_moe import core as moe_core

    captured = {}

    def fake_trtllm_bf16_moe(self, *args):
        captured["args"] = args
        hidden_states = args[4]
        return [hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)]

    fake_module = type("FakeMoeModule", (), {"trtllm_bf16_moe": fake_trtllm_bf16_moe})()
    monkeypatch.setattr(moe_core, "get_trtllm_moe_sm100_module", lambda: fake_module)

    seq_len = 8
    hidden_size = 128
    intermediate_size = 1024
    num_experts = 512
    top_k = 10
    tune_max_num_tokens = TUNE_MAX

    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32)
    hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)
    gemm1_weights = torch.empty(1, dtype=torch.bfloat16)
    gemm2_weights = torch.empty(1, dtype=torch.bfloat16)

    output = moe_core.trtllm_bf16_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
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
        routing_method_type=0,
        use_shuffled_weight=True,
        weight_layout=moe_core.WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=True,
        tune_max_num_tokens=tune_max_num_tokens,
    )

    assert output.shape == hidden_states.shape
    assert "args" in captured, "Expected wrapper to call trtllm_bf16_moe backend"

    args = captured["args"]
    assert args[7] == num_experts
    assert args[8] == top_k
    assert args[9] is None
    assert args[10] is None
    assert args[11] == intermediate_size
    assert args[13] == num_experts
    assert args[16] is True
    assert args[17] == int(moe_core.WeightLayout.BlockMajorK)
    assert args[20] == tune_max_num_tokens


@pytest.mark.parametrize(
    "top_k,num_experts",
    [
        (2, 8),
        (4, 16),
        (8, 128),  # Qwen3-VL-MoE-like (text)
        (10, 128),  # Routing kernel currently supports top_k <= 10
    ],
)
def test_bf16_moe_tile_mismatch_no_crash_after_fix(monkeypatch, top_k, num_experts):
    """SM100 BF16 integration: cached mismatched tile falls back without crash."""
    from flashinfer.fused_moe.utils import last_positive_power_of_2

    _require_sm100()
    _reset_autotuner()
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    hidden_size = 1024
    intermediate_size = 1024

    gemm1_weights, gemm2_weights = _prepare_bf16_moe_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    tune_num_tokens = 256
    infer_num_tokens = 500
    assert last_positive_power_of_2(infer_num_tokens) == tune_num_tokens

    tune_max = TUNE_MAX

    def bias_tile_32(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        """Force tile_N=32 to be selected as the 'fastest' tactic."""
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == 32 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", bias_tile_32)
    _tune_bf16_moe_once(
        device=device,
        tune_num_tokens=tune_num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        tune_max=tune_max,
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
        tune_max=tune_max,
    )


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
    "seed,stale_tactic",
    [
        (123, [4096, 0]),  # unsupported tile_N
        (124, [32, 10_000_000]),  # invalid config index
        (125, [32]),  # malformed tactic payload
    ],
)
def test_bf16_moe_stale_cached_tactic_falls_back_no_crash(
    monkeypatch, top_k, num_experts, seed, stale_tactic
):
    """SM100 integration: stale cache payloads should fall back without crashing."""
    _require_sm100()
    _reset_autotuner()
    torch.manual_seed(seed)
    device = torch.device("cuda:0")

    hidden_size, intermediate_size = 1024, 1024
    tune_num_tokens, infer_num_tokens = 256, 500
    tune_max = TUNE_MAX

    gemm1_weights, gemm2_weights = _prepare_bf16_moe_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    def bias_tile_32(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == 32 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", bias_tile_32)
    _tune_bf16_moe_once(
        device=device,
        tune_num_tokens=tune_num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        tune_max=tune_max,
    )

    _overwrite_cached_tactic_for_op("flashinfer::trtllm_bf16_moe", stale_tactic)
    _run_bf16_moe_infer(
        device=device,
        infer_num_tokens=infer_num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        tune_max=tune_max,
    )
