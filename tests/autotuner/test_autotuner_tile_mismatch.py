"""
Regression test for the tile_N mismatch bug (RuntimeError: unordered_map::at).

The bug is in the C++ MoE kernel launcher (trtllm_fused_moe_kernel_launcher.cu).
During autotuning, the Python autotuner profiles kernels using bucketed
num_tokens (e.g. 256) and caches the best tactic [tile_N, config].
During inference, the C++ launcher calls computeSelectedTileN with the
*actual* num_tokens (e.g. 500) to build launchers_map — a subset of
supported tile sizes. When the actual num_tokens differs from the
bucketed value, computeSelectedTileN can produce a different tile subset,
and the cached tile_N may not exist in launchers_map, causing
launchers_map.at(tile_N) to throw.

The C++ fix adds a fallback when the cached tile_N is missing:
  if (launchers_map.find(tile_N) == launchers_map.end()) { fallback }

The fix to _find_nearest_profile (propagating the bucketed value to all
linked dimensions) is what exposed this latent C++ bug. Before that fix,
only the first linked dimension was bucketed during inference, so the
profile key never matched the tuning-time key — the autotuner always
returned the fallback tactic (-1) and the C++ fallback path was taken.
After the fix, cache hits occur and the cached tile_N reaches the C++
launcher, where the missing guard causes the crash. Both fixes are needed:
the Python fix makes the autotuner cache work correctly for MoE, and the
C++ fix makes the launcher handle tile_N mismatches gracefully.
"""

import math

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TuningConfig,
    TunableRunner,
)
from flashinfer.fused_moe.utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from flashinfer.utils import get_compute_capability


# ---------------------------------------------------------------------------
# Helper: Python mirror of C++ computeSelectedTileN
# (csrc/trtllm_fused_moe_kernel_launcher.cu, lines 85-107)
# ---------------------------------------------------------------------------
def _next_power_of_two(n: float) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


def compute_selected_tile_n(
    supported_tiles: list[int],
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
) -> set[int]:
    """Python equivalent of C++ computeSelectedTileN."""
    avg = num_tokens * top_k / num_local_experts
    tile = max(supported_tiles[0], min(supported_tiles[-1], _next_power_of_two(avg)))
    try:
        idx = supported_tiles.index(tile)
    except ValueError:
        # tile not in supported_tiles, find closest
        idx = min(
            range(len(supported_tiles)),
            key=lambda i: abs(supported_tiles[i] - tile),
        )
    selected = {supported_tiles[idx]}
    if idx + 1 < len(supported_tiles):
        selected.add(supported_tiles[idx + 1])
    if idx + 2 < len(supported_tiles):
        selected.add(supported_tiles[idx + 2])
    if idx > 0:
        selected.add(supported_tiles[idx - 1])
    return selected


# ---------------------------------------------------------------------------
# TileAwareDummyRunner: returns different valid tactics depending on shapes
# ---------------------------------------------------------------------------
class TileAwareDummyRunner(TunableRunner):
    """Runner whose valid tactics depend on input num_tokens, mimicking
    the real trtllm-gen MoE runner where computeSelectedTileN filters tiles.

    Each tactic is a list [tile_N, config_idx] matching the real MoE convention.
    """

    SUPPORTED_TILES = [8, 16, 32, 64]  # FP4 base tiles (no 128/256 for bf16 act)

    def __init__(self, top_k: int = 8, num_local_experts: int = 64):
        self.top_k = top_k
        self.num_local_experts = num_local_experts

    def get_valid_tactics(self, inputs, profile):
        num_tokens = inputs[0].shape[0]
        selected = compute_selected_tile_n(
            self.SUPPORTED_TILES, num_tokens, self.top_k, self.num_local_experts
        )
        tactics = []
        for tile in sorted(selected):
            for cfg in range(2):  # 2 configs per tile
                tactics.append([tile, cfg])
        return tactics

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


# ---------------------------------------------------------------------------
# Shared MoE-like tuning config (mirrors the real one)
# ---------------------------------------------------------------------------
TUNE_MAX = 4096

MOE_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            input_idx=(0,),
            dim_idx=(0,),
            gen_tuning_buckets=get_last_power_of_2_num_tokens_buckets(TUNE_MAX, 1),
            map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), TUNE_MAX),
        ),
    ),
)


def _reset_autotuner() -> AutoTuner:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    tuner.is_tuning_mode = False
    return tuner


# ===================================================================
# Tests
# ===================================================================


def test_tile_set_differs_between_bucketed_and_actual_num_tokens():
    """Prove that computeSelectedTileN can return different tile sets
    for the bucketed num_tokens vs. the actual num_tokens that maps to
    the same autotuner bucket.

    This is the precondition for the unordered_map::at crash.
    """
    top_k = 8
    num_experts = 64

    # With the base SUPPORTED_TILES=[8,16,32,64] and top_k=8, num_experts=64,
    # both bucketed=1024 and actual=1624 clamp to max tile 64, so the sets match.
    # Use a wider tile range that DOES produce a mismatch:
    tiles_small = [8, 16, 32, 64, 128]
    actual = 500
    bucket = last_positive_power_of_2(actual)  # 256
    assert bucket == 256

    tiles_bucket = compute_selected_tile_n(tiles_small, bucket, top_k, num_experts)
    tiles_actual = compute_selected_tile_n(tiles_small, actual, top_k, num_experts)

    # bucket=256: avg=256*8/64=32, nextPow2=32 → idx=2, selected={16,32,64,128}
    # actual=500: avg=500*8/64=62.5, nextPow2=64 → idx=3, selected={32,64,128}
    assert tiles_bucket != tiles_actual, (
        f"Expected tile sets to differ for bucket={bucket} vs actual={actual}, "
        f"got {tiles_bucket} vs {tiles_actual}"
    )

    # A tile_N valid for the bucket may not be valid for the actual
    only_in_bucket = tiles_bucket - tiles_actual
    assert len(only_in_bucket) > 0, (
        "Expected at least one tile valid for bucketed but not for actual shapes"
    )


def test_autotuner_returns_cached_tactic_for_different_actual_shape(monkeypatch):
    """The autotuner returns a cached tactic when num_tokens differs from
    the tuning shape but maps to the same bucket.

    Before the C++ fix, if the cached tile_N was not in
    computeSelectedTileN(actual_num_tokens), this caused
    RuntimeError: unordered_map::at.
    """
    tuner = _reset_autotuner()
    runner = TileAwareDummyRunner(top_k=8, num_local_experts=64)
    hidden_size = 256

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        # Make the tactic with tile_N=16 the "best" (lowest time)
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == 16 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    # --- Phase 1: Tune with bucketed num_tokens ---
    # The autotuner generates profiles for all buckets.
    # For bucket=256, avg=256*8/64=32, tiles include {16,32,64,...},
    # so tile_N=16 is valid and will be selected as best.
    tune_inputs = [torch.empty((256, hidden_size), dtype=torch.float32)]
    with autotune(tune_mode=True):
        _, tuned_tactic = tuner.choose_one(
            "test_tile_mismatch", [runner], MOE_TUNING_CONFIG, tune_inputs
        )

    assert isinstance(tuned_tactic, list)
    assert tuned_tactic[0] == 16, f"Expected tile_N=16, got {tuned_tactic}"

    # --- Phase 2: Inference with different actual num_tokens ---
    # actual=500 maps to bucket=256 (same bucket) so we get a cache hit.
    infer_inputs = [torch.empty((500, hidden_size), dtype=torch.float32)]
    _, infer_tactic = tuner.choose_one(
        "test_tile_mismatch", [runner], MOE_TUNING_CONFIG, infer_inputs
    )

    # The autotuner returns the CACHED tactic from tuning (tile_N=16).
    assert infer_tactic == tuned_tactic, "Expected cache hit returning the tuned tactic"

    # --- Verify the mismatch ---
    # For actual=500: avg=500*8/64=62.5, nextPow2=64, tiles={32,64,128}
    # tile_N=16 is NOT in this set → would crash without the C++ fix.
    actual_tiles = compute_selected_tile_n(
        TileAwareDummyRunner.SUPPORTED_TILES + [128],
        500,
        runner.top_k,
        runner.num_local_experts,
    )
    assert tuned_tactic[0] not in actual_tiles, (
        f"tile_N={tuned_tactic[0]} should NOT be in actual tiles {actual_tiles} — "
        "this is the mismatch that caused the C++ unordered_map::at crash"
    )


def test_autotuner_cache_miss_returns_fallback_for_unseen_bucket():
    """When num_tokens maps to a bucket that was never tuned,
    the autotuner correctly returns the fallback tactic=-1."""
    tuner = _reset_autotuner()
    runner = TileAwareDummyRunner()
    hidden_size = 256

    # No tuning done — inference should always get fallback
    inputs = [torch.empty((1624, hidden_size), dtype=torch.float32)]
    _, tactic = tuner.choose_one("test_no_tune", [runner], MOE_TUNING_CONFIG, inputs)
    assert tactic == -1, "Expected fallback tactic when no tuning was done"


def test_different_actual_tokens_same_bucket_get_same_cached_tactic(monkeypatch):
    """Multiple actual num_tokens that map to the same bucket should all
    receive the same cached tactic — confirming the autotuner uses the
    bucketed profile, not the actual shapes, for cache lookup."""
    tuner = _reset_autotuner()
    runner = TileAwareDummyRunner(top_k=8, num_local_experts=64)
    hidden_size = 128

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == 32 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    tune_inputs = [torch.empty((512, hidden_size), dtype=torch.float32)]
    with autotune(tune_mode=True):
        _, tuned_tactic = tuner.choose_one(
            "test_same_bucket", [runner], MOE_TUNING_CONFIG, tune_inputs
        )

    assert tuned_tactic[0] == 32

    # All these map to bucket 512 via last_positive_power_of_2
    for actual in [513, 600, 700, 800, 900, 1000, 1023]:
        assert last_positive_power_of_2(actual) == 512
        infer_inputs = [torch.empty((actual, hidden_size), dtype=torch.float32)]
        _, tactic = tuner.choose_one(
            "test_same_bucket", [runner], MOE_TUNING_CONFIG, infer_inputs
        )
        assert tactic == tuned_tactic, (
            f"Expected cached tactic {tuned_tactic} for num_tokens={actual}, got {tactic}"
        )


# ===================================================================
# SM100 integration test — exercises the real C++ launcher
# ===================================================================


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


def test_bf16_moe_tile_mismatch_no_crash_after_fix(monkeypatch):
    """SM100 integration test: tune BF16 MoE with one num_tokens, then
    infer with a different num_tokens that maps to the same autotuner
    bucket but selects different C++ tiles.

    Before the C++ fix this raised ``RuntimeError: unordered_map::at``.
    After the fix the launcher falls back to a default tile gracefully.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("This test requires an SM100/SM103 (Blackwell) GPU.")

    from flashinfer.fused_moe import trtllm_bf16_moe, WeightLayout

    _reset_autotuner()
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    # Model dimensions — keep small for speed
    num_experts = 8
    top_k = 2
    hidden_size = 1024
    intermediate_size = 1024

    # Prepare weights once (they don't depend on num_tokens)
    gemm1_weights, gemm2_weights = _prepare_bf16_moe_weights(
        num_experts, intermediate_size, hidden_size, device
    )

    # --- Choose num_tokens that trigger the tile mismatch ---
    # BF16 MoE supported tiles: {8, 16, 32, 64, 128}
    # tune_num_tokens=256 (bucket): avg=256*2/8=64 → tiles={32,64,128}
    # infer_num_tokens=500 (bucket=256): avg=500*2/8=125 → tiles={64,128}
    # A tactic with tile_N=32 is valid for bucket but NOT for actual.
    tune_num_tokens = 256
    infer_num_tokens = 500
    assert last_positive_power_of_2(infer_num_tokens) == tune_num_tokens

    tune_max = 4096

    def bias_tile_32(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        """Force tile_N=32 to be selected as the 'fastest' tactic."""
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == 32 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", bias_tile_32)

    # --- Phase 1: Tune with tune_num_tokens ---
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
            routing_method_type=1,  # Renormalize (TopK/Default not supported for BF16)
            use_shuffled_weight=True,
            weight_layout=WeightLayout.BlockMajorK,
            tune_max_num_tokens=tune_max,
        )

    # Verify the autotuner populated its cache during tuning
    tuner = AutoTuner.get()
    assert len(tuner.profiling_cache) > 0, (
        "Autotuner cache should be populated after tuning"
    )

    # --- Phase 2: Inference with infer_num_tokens (same bucket, different tiles) ---
    # Without the C++ fix, this would crash with RuntimeError: unordered_map::at
    # because tile_N=32 is not in computeSelectedTileN(500, 2, 8) = {64, 128}.
    routing_infer = torch.rand(
        infer_num_tokens, num_experts, device=device, dtype=torch.bfloat16
    )
    hidden_infer = torch.randn(
        infer_num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )

    # This call should NOT crash (with the C++ fix it falls back to a valid tile)
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
