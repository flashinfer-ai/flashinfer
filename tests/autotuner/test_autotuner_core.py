import random

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    TuningConfig,
    TunableRunner,
)
from flashinfer.fused_moe.utils import last_positive_power_of_2
from .utils import reset_autotuner


def _moe_input_shapes(
    num_tokens: int,
    hidden_size: int = 4096,
    num_experts: int = 256,
    top_k: int = 8,
    hidden_states_scale_width: int = 32,
) -> tuple[torch.Size, ...]:
    # num_tokens is the first dimension of all tensors
    return (
        torch.Size([num_tokens, hidden_size]),
        torch.Size([num_tokens, num_experts]),
        torch.Size([num_tokens, top_k]),
        torch.Size([num_tokens, top_k]),
        torch.Size([num_tokens, hidden_size]),
        torch.Size([num_tokens, hidden_states_scale_width]),
    )


class DummyRunner(TunableRunner):
    def __init__(self, valid_tactics=(0, 1, 2)):
        self.valid_tactics = valid_tactics

    def get_valid_tactics(self, inputs, profile):
        return self.valid_tactics

    def forward(self, inputs, tactic: int = -1, do_preparation: bool = False, **kwargs):
        return inputs[0]


def test_find_nearest_profile_passthrough_without_specs():
    """No dynamic/constraint specs should keep shape values unchanged."""
    shapes = (torch.Size([3, 5]), torch.Size([7, 11, 13]))
    out = AutoTuner._find_nearest_profile(shapes, TuningConfig())
    assert out == ((3, 5), (7, 11, 13))


@pytest.mark.parametrize(
    "leading_dim,expected_bucket",
    [
        (1000, 512),
        (1024, 1024),
        (4000, 2048),
        (4096, 4096),
        (8000, 4096),
        (8192, 8192),
        (10000, 8192),
    ],
)
def test_find_nearest_profile_dynamic_and_constraint(leading_dim, expected_bucket):
    """Dynamic dim is bucketized and constrained dim is replaced by wildcard -1."""
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(512, 1024, 2048, 4096, 8192),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                input_idx=1,
                dim_idx=2,
                infer_shape=lambda shapes: shapes[0][0] // 2,
            ),
        ),
    )
    shapes = (torch.Size([leading_dim, 8]), torch.Size([10, 9, 6]))
    out = AutoTuner._find_nearest_profile(shapes, tuning_config)
    assert out == ((expected_bucket, 8), (10, 9, -1))


@pytest.mark.parametrize(
    "num_tokens,expected_bucket",
    [
        (1024, 1024),
        (4096, 4096),
        (8192, 8192),
    ],
)
def test_find_nearest_profile_single_tensor_bucketization_exact_powers(
    num_tokens, expected_bucket
):
    """Exact power-of-two mapping is validated on one tensor (no linked-dim semantics)."""
    gen_tuning_buckets = (512, 1024, 2048, 4096, 8192)
    input_shape = (_moe_input_shapes(num_tokens)[0],)

    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=gen_tuning_buckets,
                map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), 8192),
            ),
        )
    )
    nearest = AutoTuner._find_nearest_profile(input_shape, config)
    assert nearest[0][0] == expected_bucket
    assert nearest[0][1:] == input_shape[0][1:]


@pytest.mark.parametrize(
    "num_tokens,expected_bucket",
    [
        (1000, 512),
        (1024, 1024),
        (4000, 2048),
        (4096, 4096),
        (8000, 4096),
        (8192, 8192),
        (10000, 8192),
    ],
)
def test_find_nearest_profile_moe_shared_num_tokens_axis(num_tokens, expected_bucket):
    """MoE linked tensors should all map num_tokens together to one bucket."""
    gen_tuning_buckets = (512, 1024, 2048, 4096, 8192, 16384)
    shapes: tuple[torch.Size, ...] = _moe_input_shapes(num_tokens)

    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                # MoE test input has 6 tensors:
                # output, routing_logits, topk_ids, expert_weights, hidden_states, hidden_states_scale.
                # They all share num_tokens on dim 0, so we link indices (0..5) to dim_idx=0.
                input_idx=(0, 1, 2, 3, 4, 5),
                dim_idx=(0, 0, 0, 0, 0, 0),
                gen_tuning_buckets=gen_tuning_buckets,
                map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), 8192),
            ),
        )
    )
    nearest = AutoTuner._find_nearest_profile(shapes, config)
    assert all(shape[0] == expected_bucket for shape in nearest)
    for nearest_shape, original_shape in zip(nearest, shapes, strict=True):
        assert nearest_shape[1:] == original_shape[1:]


def test_find_nearest_profile_moe_same_bucket_same_profile():
    """MoE inputs mapping to the same bucket should share an identical profile."""
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                # Same MoE linkage as above: all 6 tensors share num_tokens on dim 0.
                input_idx=(0, 1, 2, 3, 4, 5),
                dim_idx=(0, 0, 0, 0, 0, 0),
                gen_tuning_buckets=(512, 1024, 2048, 4096, 8192),
                map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), 8192),
            ),
        )
    )
    p1 = AutoTuner._find_nearest_profile(_moe_input_shapes(1000), config)
    p2 = AutoTuner._find_nearest_profile(_moe_input_shapes(1023), config)
    assert p1 == p2


def test_find_nearest_profile_maps_all_linked_dims():
    """One logical dynamic axis should update every linked tensor/dimension."""
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2),
                dim_idx=(0, 1, 2),
                gen_tuning_buckets=(16, 32, 48, 64),
                map_to_tuning_buckets=lambda x: ((x + 15) // 16) * 16,
            ),
        )
    )
    shapes = (torch.Size([33, 4]), torch.Size([2, 33, 9]), torch.Size([5, 6, 33, 7]))
    out = AutoTuner._find_nearest_profile(shapes, tuning_config)
    assert out == ((48, 4), (2, 48, 9), (5, 6, 48, 7))


@pytest.mark.parametrize(
    "shape_a,shape_b,expected_equal",
    [
        (torch.Size([130, 16]), torch.Size([200, 16]), True),
        (torch.Size([130, 16]), torch.Size([300, 16]), False),
        (torch.Size([1000, 16]), torch.Size([1024, 16]), False),
        (torch.Size([4000, 16]), torch.Size([4096, 16]), False),
        (torch.Size([8000, 16]), torch.Size([8192, 16]), False),
    ],
)
def test_get_cache_key_bucketization(shape_a, shape_b, expected_equal):
    """Cache keys should match only when bucketized nearest profiles match."""
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(64, 128, 256),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        )
    )
    runner = DummyRunner()
    key1 = AutoTuner._get_cache_key("dummy", runner, (shape_a,), config)
    key2 = AutoTuner._get_cache_key("dummy", runner, (shape_b,), config)
    assert (key1 == key2) is expected_equal


def test_search_cache_hit_and_miss():
    """search_cache should report miss before seeding and hit after seeding."""
    tuner = reset_autotuner()
    config = TuningConfig()
    runner = DummyRunner()
    shapes = (torch.Size([8, 16]),)

    miss = tuner.search_cache("dummy", [runner], shapes, config)
    assert miss == (False, 0, -1, None)

    key = AutoTuner._get_cache_key("dummy", runner, shapes, config)
    tuner.profiling_cache[key] = (0, 1, None)
    hit = tuner.search_cache("dummy", [runner], shapes, config)
    assert hit == (True, 0, 1, None)


def test_search_cache_preserving_leading_dims_hits_while_flattened_misses(monkeypatch):
    """Shape-preserving reshape keeps cache-hit behavior; full flatten can change bucket/key."""
    tuner = reset_autotuner()
    runner = DummyRunner()
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                # In MoE-style kernels, leading dim represents num_tokens.
                # Keep one dynamic tensor here so this test isolates layout effects
                # (and does not depend on known linked-dim mapping bugs).
                input_idx=(0,),
                dim_idx=(0,),
                # Only cache the current bucket; this makes alternative layouts
                # map to a miss when their nearest bucket differs.
                gen_tuning_buckets=lambda x: (last_positive_power_of_2(x),),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        )
    )

    # MoE semantic shape: [num_tokens, hidden_size].
    m, n = 1000, 256
    preserve_layout_inputs = [torch.empty((m, n), dtype=torch.float32)]

    # Flattening destroys the num_tokens axis and changes autotuner's shape key.
    flattened_layout_shapes = (torch.Size([m * n]),)

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        return {0: 5.0, 1: 1.0, 2: 3.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    with autotune(tune_mode=True):
        tuner.choose_one("dummy_layout", [runner], config, preserve_layout_inputs)

    # Search with shape that preserves num_tokens as dim0 -> expected cache hit.
    preserved_hit, _, _, _ = tuner.search_cache(
        "dummy_layout",
        [runner],
        tuple(t.shape for t in preserve_layout_inputs),
        config,
    )

    # Search with flattened shape (num_tokens lost) -> expected cache miss.
    flattened_hit, _, _, _ = tuner.search_cache(
        "dummy_layout",
        [runner],
        flattened_layout_shapes,
        config,
    )

    assert preserved_hit is True
    assert flattened_hit is False


def test_choose_one_inference_uses_cache_or_fallback():
    """Inference path should use cached tactic when present, else fallback -1."""
    tuner = reset_autotuner()
    runner = DummyRunner()
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    # No cache -> fallback.
    chosen_runner, tactic = tuner.choose_one("dummy", [runner], config, inputs)
    assert chosen_runner is runner
    assert tactic == -1

    # Seed cache -> cache hit.
    key = AutoTuner._get_cache_key("dummy", runner, (inputs[0].shape,), config)
    tuner.profiling_cache[key] = (0, 2, None)
    chosen_runner, tactic = tuner.choose_one("dummy", [runner], config, inputs)
    assert chosen_runner is runner
    assert tactic == 2


def test_choose_one_tuning_selects_best_tactic_and_populates_cache(monkeypatch):
    """Tuning path should select lowest-profile-time tactic and cache result."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2))
    inputs = [torch.empty((16, 32), dtype=torch.float32)]
    config = TuningConfig()

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        return {0: 5.0, 1: 1.0, 2: 3.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    with autotune(tune_mode=True):
        chosen_runner, tactic = tuner.choose_one("dummy_tune", [runner], config, inputs)

    assert chosen_runner is runner
    assert tactic == 1
    assert len(tuner.profiling_cache) >= 1
    assert tuner.stats.tuned_op_total_configs["dummy_tune"] >= 1
    assert tuner.stats.tuned_op_successful_configs["dummy_tune"] >= 1


def test_prepare_input_tensors_reuses_static_and_recreates_dynamic():
    """Profiles apply constraints, dynamic inputs are recreated, static inputs are reused."""
    tuner = reset_autotuner()
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(8, 16),
                map_to_tuning_buckets=lambda x: x,
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                input_idx=0,
                dim_idx=1,
                infer_shape=lambda shapes: shapes[0][0] // 2,
            ),
        ),
    )
    inputs = [
        torch.empty((12, 99), dtype=torch.float32),
        torch.empty((2, 3), dtype=torch.float32),
    ]
    profiles = tuner._generate_optimization_profiles(config, inputs)
    assert len(profiles) == 2
    assert profiles[0].get_opt_shapes()[0] == (8, 4)
    assert profiles[1].get_opt_shapes()[0] == (16, 8)

    prepared = tuner._prepare_input_tensors(profiles[0], inputs)

    assert tuple(prepared[0].shape) == (8, 4)
    assert prepared[0] is not inputs[0]
    assert prepared[1] is inputs[1]


class TileTacticDummyRunner(TunableRunner):
    def __init__(self, supported_tiles: tuple[int, ...], num_tactics_per_tile: int = 2):
        self.supported_tiles = supported_tiles
        self.num_tactics_per_tile = num_tactics_per_tile

    def get_valid_tactics(self, inputs, profile):
        tactics = []
        for tile in sorted(self.supported_tiles):
            for cfg in range(self.num_tactics_per_tile):
                tactics.append([tile, cfg])
        return tactics

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def test_choose_one_different_infer_tokens_same_bucket_get_same_cached_tactic(
    monkeypatch,
):
    """Multiple actual num_tokens that map to the same bucket should all
    receive the same cached tactic - confirming the autotuner uses the
    bucketed profile, not the actual shapes, for cache lookup."""
    tuner = reset_autotuner()
    runner = TileTacticDummyRunner(supported_tiles=(8, 16, 32, 64))
    hidden_size = 128
    bucket_start = 512
    bucket_end = 1024
    tuning_buckets = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    )
    tune_max = max(tuning_buckets)

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        """When num_tokens is in the bucket [512, 1024):
            return a low score (indicating good performance) if tile_n=32 and cfg=1
        When num_tokens < 512:
            return a low score if tile_n=16 and cfg=1
        When num_tokens >= 1024:
            return a low score if tile_n=64 and cfg=1
         For all other tile_n and cfg combinations, return a high score (indicating bad performance).
        """
        if isinstance(tactic, list):
            tile_n = tactic[0]
            tactic_cfg = tactic[1]
        else:
            tile_n = -1
            tactic_cfg = -1
        num_tokens = prof_inputs[0].shape[0]
        if num_tokens < bucket_start:
            target_tile_n = 16
        elif bucket_start <= num_tokens < bucket_end:
            target_tile_n = 32
        else:
            target_tile_n = 64
        return 1.0 if tile_n == target_tile_n and tactic_cfg == 1 else 5.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    tune_inputs = [torch.empty((bucket_start, hidden_size), dtype=torch.float32)]
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=tuning_buckets,
                map_to_tuning_buckets=lambda x: min(
                    last_positive_power_of_2(x), tune_max
                ),
            ),
        ),
    )
    with autotune(tune_mode=True):
        tuner.choose_one("test_same_bucket", [runner], tuning_config, tune_inputs)

    num_tokens_with_expected_tactic_list = [
        (random.randrange(bucket_start, bucket_end), [32, 1]) for _ in range(3)
    ]
    num_tokens_with_expected_tactic_list += [
        (random.randrange(0, bucket_start), [16, 1]) for _ in range(3)
    ]
    num_tokens_with_expected_tactic_list += [
        (random.randrange(bucket_end, bucket_end * 2), [64, 1]) for _ in range(3)
    ]
    for actual, expected_tactic in num_tokens_with_expected_tactic_list:
        infer_inputs = [torch.empty((actual, hidden_size), dtype=torch.float32)]
        _, tactic = tuner.choose_one(
            "test_same_bucket", [runner], tuning_config, infer_inputs
        )
        assert tactic == expected_tactic, (
            f"Expected cached tactic {expected_tactic} for num_tokens={actual}, got {tactic}"
        )
