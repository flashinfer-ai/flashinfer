"""Regression tests for MoE autotuner tile/config mismatch handling."""

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


class TileAwareDummyRunner(TunableRunner):
    """Dummy runner whose valid tactics depend on num_tokens.
    returns different valid tactics depending on shapes."""

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


TUNE_MAX = 8192

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


def _find_bucket_actual_tile_mismatch_case(
    *,
    supported_tiles: list[int],
    top_k: int,
    num_local_experts: int,
    tune_max: int = TUNE_MAX,
):
    """Find (bucket, actual, tuned_tile) where tuned_tile is valid only for bucket."""
    for actual in range(2, tune_max + 1):
        bucket = min(last_positive_power_of_2(actual), tune_max)
        if bucket == actual:
            continue
        tiles_bucket = compute_selected_tile_n(
            supported_tiles, bucket, top_k, num_local_experts
        )
        tiles_actual = compute_selected_tile_n(
            supported_tiles, actual, top_k, num_local_experts
        )
        only_in_bucket = sorted(tiles_bucket - tiles_actual)
        if only_in_bucket:
            return bucket, actual, only_in_bucket[0]
    return None


@pytest.mark.parametrize(
    "top_k,num_experts",
    [
        (2, 8),
        (4, 16),
        (8, 64),
        (8, 128),  # Qwen3-VL-MoE-like (text) routing fanout/expert count
        (10, 512),  # Qwen3.5-MoE-like (text) routing fanout/expert count
    ],
)
def test_tile_set_differs_between_bucketed_and_actual_num_tokens(top_k, num_experts):
    """Bucketed and actual shapes in the same bucket can still pick different tiles."""
    tiles = [8, 16, 32, 64, 128]
    case = _find_bucket_actual_tile_mismatch_case(
        supported_tiles=tiles, top_k=top_k, num_local_experts=num_experts
    )
    assert case is not None, (
        f"No mismatch case found for top_k={top_k}, experts={num_experts}"
    )
    bucket, actual, _ = case
    tiles_bucket = compute_selected_tile_n(tiles, bucket, top_k, num_experts)
    tiles_actual = compute_selected_tile_n(tiles, actual, top_k, num_experts)

    assert tiles_bucket != tiles_actual, (
        f"Expected tile sets to differ for bucket={bucket} vs actual={actual}, "
        f"got {tiles_bucket} vs {tiles_actual}"
    )

    only_in_bucket = tiles_bucket - tiles_actual
    assert len(only_in_bucket) > 0, (
        "Expected at least one tile valid for bucketed but not for actual shapes"
    )


@pytest.mark.parametrize(
    "top_k,num_experts,hidden_size",
    [
        (2, 8, 128),
        (4, 16, 128),
        (8, 64, 256),
        (12, 128, 1024),
        (
            8,
            256,
            7168,
        ),  # DeepSeek-v3-like MoE dimensions (lightweight autotuner-only path)
        (8, 128, 4096),  # Qwen3-VL-MoE-like text dimensions (autotuner-only path)
        (10, 512, 4096),  # Qwen3.5-MoE-like text dimensions (autotuner-only path)
    ],
)
def test_autotuner_returns_cached_tactic_for_different_actual_shape(
    monkeypatch, top_k, num_experts, hidden_size
):
    """Cache lookup uses bucketed profile, not raw runtime num_tokens."""
    tuner = _reset_autotuner()
    runner = TileAwareDummyRunner(top_k=top_k, num_local_experts=num_experts)
    case = _find_bucket_actual_tile_mismatch_case(
        supported_tiles=runner.SUPPORTED_TILES,
        top_k=top_k,
        num_local_experts=num_experts,
    )
    assert case is not None, (
        f"No mismatch case found for top_k={top_k}, experts={num_experts}"
    )
    tune_bucket, infer_actual, tuned_tile = case

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        # Make the chosen mismatch tile "best" so autotuner caches it.
        tile_n = tactic[0] if isinstance(tactic, list) else -1
        return 1.0 if tile_n == tuned_tile else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    tune_inputs = [torch.empty((tune_bucket, hidden_size), dtype=torch.float32)]
    with autotune(tune_mode=True):
        _, tuned_tactic = tuner.choose_one(
            "test_tile_mismatch", [runner], MOE_TUNING_CONFIG, tune_inputs
        )

    assert isinstance(tuned_tactic, list)
    assert tuned_tactic[0] == tuned_tile, (
        f"Expected tile_N={tuned_tile} for top_k={top_k}, experts={num_experts}, "
        f"got {tuned_tactic}"
    )

    assert last_positive_power_of_2(infer_actual) == tune_bucket
    infer_inputs = [torch.empty((infer_actual, hidden_size), dtype=torch.float32)]
    _, infer_tactic = tuner.choose_one(
        "test_tile_mismatch", [runner], MOE_TUNING_CONFIG, infer_inputs
    )

    assert infer_tactic == tuned_tactic, "Expected cache hit returning the tuned tactic"

    actual_tiles = compute_selected_tile_n(
        TileAwareDummyRunner.SUPPORTED_TILES,
        infer_actual,
        runner.top_k,
        runner.num_local_experts,
    )
    assert tuned_tactic[0] not in actual_tiles, (
        f"tile_N={tuned_tactic[0]} should NOT be in actual tiles {actual_tiles} — "
        "this is the mismatch that caused the C++ unordered_map::at crash"
    )


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

    for actual in [513, 600, 700, 800, 900, 1000, 1023]:
        assert last_positive_power_of_2(actual) == 512
        infer_inputs = [torch.empty((actual, hidden_size), dtype=torch.float32)]
        _, tactic = tuner.choose_one(
            "test_same_bucket", [runner], MOE_TUNING_CONFIG, infer_inputs
        )
        assert tactic == tuned_tactic, (
            f"Expected cached tactic {tuned_tactic} for num_tokens={actual}, got {tactic}"
        )
