import random
import tracemalloc
from unittest.mock import MagicMock, patch

import pytest
import torch

import flashinfer.fused_moe.core as core_mod
from flashinfer import autotune
from flashinfer.autotuner.initializers import autotuner_initializer_randn
from flashinfer.fused_moe.core import MoeRunnerInputs, _moe_topk_ids_init
from flashinfer.fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    make_hybrid_bucket_mapper,
)
from flashinfer.mla._core import (
    _build_mla_decode_tuning_config,
    _mla_decode_tuning_config,
)
from flashinfer.tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
from flashinfer.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    TuningConfig,
    TunableRunner,
    make_bucket_mapper,
    round_to_nearest_bucket,
)

from flashinfer.utils import last_positive_power_of_2

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

    tune_inputs = [
        [torch.empty((bucket_start // 2, hidden_size), dtype=torch.float32)],
        [torch.empty((bucket_start, hidden_size), dtype=torch.float32)],
        [torch.empty((bucket_end, hidden_size), dtype=torch.float32)],
    ]
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
        for inputs in tune_inputs:
            tuner.choose_one("test_same_bucket", [runner], tuning_config, inputs)

    num_tokens_with_expected_tactic_list = [
        (random.randrange(bucket_start, bucket_end), [32, 1]) for _ in range(3)
    ]
    num_tokens_with_expected_tactic_list += [
        (random.randrange(1, bucket_start), [16, 1]) for _ in range(3)
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


# ---------------------------------------------------------------------------
# Tests for custom tuning buckets and round_up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,expected",
    [
        (50, 100),
        (100, 100),
        (150, 100),
        (199, 100),
        (200, 200),
        (350, 200),
        (500, 500),
        (999, 500),
        (1000, 1000),
        (2000, 1000),
    ],
)
def test_round_to_nearest_bucket_floor(x, expected):
    """round_to_nearest_bucket with round_map=False floors to largest bucket <= x."""
    buckets = [100, 200, 500, 1000]
    assert round_to_nearest_bucket(x, buckets, round_map=False) == expected


@pytest.mark.parametrize(
    "x,expected",
    [
        (50, 100),
        (100, 100),
        (101, 200),
        (150, 200),
        (200, 200),
        (201, 500),
        (350, 500),
        (500, 500),
        (501, 1000),
        (999, 1000),
        (1000, 1000),
        (2000, 1000),
    ],
)
def test_round_to_nearest_bucket_ceil(x, expected):
    """round_to_nearest_bucket with round_map=True ceils to smallest bucket >= x."""
    buckets = [100, 200, 500, 1000]
    assert round_to_nearest_bucket(x, buckets, round_map=True) == expected


def test_make_bucket_mapper_floor():
    """make_bucket_mapper with round_map=False returns a floor mapper."""
    mapper = make_bucket_mapper((1000, 500, 200, 100), round_map=False)
    assert mapper(350) == 200
    assert mapper(500) == 500
    assert mapper(999) == 500
    assert mapper(50) == 100


def test_make_bucket_mapper_ceil():
    """make_bucket_mapper with round_map=True returns a ceil mapper."""
    mapper = make_bucket_mapper((1000, 500, 200, 100), round_map=True)
    assert mapper(350) == 500
    assert mapper(500) == 500
    assert mapper(501) == 1000
    assert mapper(50) == 100


@pytest.mark.parametrize(
    "leading_dim,expected_bucket",
    [
        (50, 100),
        (100, 100),
        (150, 100),
        (250, 200),
        (500, 500),
        (750, 500),
        (1000, 1000),
        (1500, 1000),
    ],
)
def test_find_nearest_profile_custom_buckets(leading_dim, expected_bucket):
    """Custom non-power-of-2 buckets with floor rounding."""
    custom_buckets = (100, 200, 500, 1000)
    mapper = make_bucket_mapper(custom_buckets, round_map=False)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=custom_buckets,
                map_to_tuning_buckets=mapper,
            ),
        ),
    )
    shapes = (torch.Size([leading_dim, 8]),)
    out = AutoTuner._find_nearest_profile(shapes, tuning_config)
    assert out[0][0] == expected_bucket
    assert out[0][1] == 8


@pytest.mark.parametrize(
    "leading_dim,expected_bucket",
    [
        (50, 100),
        (100, 100),
        (150, 200),
        (250, 500),
        (500, 500),
        (750, 1000),
        (1000, 1000),
        (1500, 1000),
    ],
)
def test_find_nearest_profile_round_up(leading_dim, expected_bucket):
    """Custom non-power-of-2 buckets with ceil rounding."""
    custom_buckets = (100, 200, 500, 1000)
    mapper = make_bucket_mapper(custom_buckets, round_map=True)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=custom_buckets,
                map_to_tuning_buckets=mapper,
            ),
        ),
    )
    shapes = (torch.Size([leading_dim, 8]),)
    out = AutoTuner._find_nearest_profile(shapes, tuning_config)
    assert out[0][0] == expected_bucket


def test_autotune_context_custom_buckets(monkeypatch):
    """autotune(tuning_buckets=...) overrides measurement points for choose_one."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2))
    inputs = [torch.empty((350, 32), dtype=torch.float32)]

    # Default config uses power-of-2 buckets from spec
    default_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(256, 512, 1024),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        ),
    )

    profiled_shapes = []

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        profiled_shapes.append(prof_inputs[0].shape[0])
        return {0: 5.0, 1: 1.0, 2: 3.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, tuning_buckets=(100, 200, 500)):
        tuner.choose_one("custom_buckets_test", [runner], default_config, inputs)

    # Profiles should have been generated at the custom bucket points, not the
    # original power-of-2 points.
    unique_shapes = sorted(set(profiled_shapes))
    assert unique_shapes == [100, 200, 500]


def test_autotune_context_round_up(monkeypatch):
    """autotune(round_up=True) uses ceil rounding for cache lookup."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1))

    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(128, 256, 512),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        ),
    )

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        n = prof_inputs[0].shape[0]
        if n == 256:
            return {0: 1.0, 1: 5.0}[tactic]  # tactic 0 wins at 256
        return {0: 5.0, 1: 1.0}[tactic]  # tactic 1 wins elsewhere

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    # Tune with round_up
    tune_inputs = [torch.empty((256, 32), dtype=torch.float32)]
    with autotune(tune_mode=True, round_up=True):
        tuner.choose_one("round_up_test", [runner], config, tune_inputs)

    # Inference: 200 should round UP to 256 (not down to 128)
    infer_inputs = [torch.empty((200, 32), dtype=torch.float32)]
    with autotune(tune_mode=False, round_up=True):
        _, tactic = tuner.choose_one("round_up_test", [runner], config, infer_inputs)

    assert tactic == 0, f"Expected tactic 0 (bucket 256 via round_up), got {tactic}"


def test_autotune_context_both_overrides(monkeypatch):
    """autotune with both custom buckets and round_up=True."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1))

    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(256, 512, 1024),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        ),
    )

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        n = prof_inputs[0].shape[0]
        if n == 300:
            return {0: 1.0, 1: 5.0}[tactic]
        return {0: 5.0, 1: 1.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    tune_inputs = [torch.empty((300, 16), dtype=torch.float32)]
    with autotune(tune_mode=True, tuning_buckets=(100, 300, 600), round_up=True):
        tuner.choose_one("both_overrides_test", [runner], config, tune_inputs)

    # 250 rounds UP to 300 with custom buckets
    infer_inputs = [torch.empty((250, 16), dtype=torch.float32)]
    with autotune(tune_mode=False, tuning_buckets=(100, 300, 600), round_up=True):
        _, tactic = tuner.choose_one(
            "both_overrides_test", [runner], config, infer_inputs
        )

    assert tactic == 0, f"Expected tactic 0 (bucket 300 via round_up), got {tactic}"


def test_autotune_context_restores_overrides():
    """Overrides are cleared when autotune() context exits."""
    tuner = reset_autotuner()

    assert tuner._override_tuning_buckets is None
    assert tuner._override_round_up is False

    with autotune(tune_mode=False, tuning_buckets=(100, 200), round_up=True):
        assert tuner._override_tuning_buckets == (100, 200)
        assert tuner._override_round_up is True

    assert tuner._override_tuning_buckets is None
    assert tuner._override_round_up is False


def test_choose_one_with_custom_buckets_selects_best_tactic(monkeypatch):
    """Full choose_one flow with custom buckets: profile, cache, retrieve."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2))

    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(512, 1024),
                map_to_tuning_buckets=last_positive_power_of_2,
            ),
        ),
    )

    def fake_profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kw):
        n = prof_inputs[0].shape[0]
        if n <= 200:
            return {0: 3.0, 1: 1.0, 2: 5.0}[tactic]  # tactic 1 best for small
        elif n <= 400:
            return {0: 1.0, 1: 5.0, 2: 3.0}[tactic]  # tactic 0 best for medium
        else:
            return {0: 5.0, 1: 3.0, 2: 1.0}[tactic]  # tactic 2 best for large

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    custom_buckets = (100, 300, 500)
    tune_inputs = [torch.empty((500, 64), dtype=torch.float32)]
    with autotune(tune_mode=True, tuning_buckets=custom_buckets):
        tuner.choose_one("custom_select_test", [runner], config, tune_inputs)

    # Inference with custom buckets (floor rounding):
    # 150 -> bucket 100 -> tactic 1
    # 350 -> bucket 300 -> tactic 0
    # 450 -> bucket 300 -> tactic 0
    # 600 -> bucket 500 -> tactic 2
    test_cases = [
        (150, 1),
        (350, 0),
        (450, 0),
        (600, 2),
    ]
    for actual_n, expected_tactic in test_cases:
        infer_inputs = [torch.empty((actual_n, 64), dtype=torch.float32)]
        with autotune(tune_mode=False, tuning_buckets=custom_buckets):
            _, tactic = tuner.choose_one(
                "custom_select_test", [runner], config, infer_inputs
            )
        assert tactic == expected_tactic, (
            f"n={actual_n}: expected tactic {expected_tactic}, got {tactic}"
        )


# ---------------------------------------------------------------------------
# Tests for None / optional input tensors
# ---------------------------------------------------------------------------


def test_prepare_input_tensors_none_input_preserved():
    """None inputs (e.g. routing_logits in non-routed MoE) should pass through without crashing."""
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
    )
    # Second input is None -- this used to blow up with AttributeError on .dtype/.shape
    inputs = [
        torch.empty((12, 64), dtype=torch.float32),
        None,
    ]
    profiles = tuner._generate_optimization_profiles(config, inputs)
    assert len(profiles) == 2

    prepared = tuner._prepare_input_tensors(profiles[0], inputs)
    assert prepared[0] is not inputs[0]  # dynamic -> recreated
    assert prepared[1] is None  # None stays None


@pytest.mark.parametrize(
    "non_tensor",
    [torch.bfloat16, None],
    ids=["dtype", "none"],
)
def test_prepare_input_tensors_with_batches_preserves_non_tensor(
    monkeypatch, non_tensor
):
    """Cold-L2 batches clone tensors while preserving scalar and optional inputs."""
    tuner = reset_autotuner()
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda: 4)
    inputs = [torch.ones(1), non_tensor]

    batches = tuner._prepare_input_tensors_with_batches(
        inputs, TuningConfig(use_cold_l2_cache=True)
    )

    assert batches[0] is inputs
    assert len(batches) > 1
    for batch in batches[1:]:
        assert batch[0] is not inputs[0]
        torch.testing.assert_close(batch[0], inputs[0])
        assert batch[1] is non_tensor


def test_choose_one_with_none_input_no_crash():
    """choose_one inference path should not crash when an input tensor is None."""
    tuner = reset_autotuner()
    runner = DummyRunner()
    inputs = [
        torch.empty((4, 8), dtype=torch.float32),
        None,  # optional tensor, e.g. routing_logits
        torch.empty((4, 2), dtype=torch.int64),
    ]
    config = TuningConfig()

    # Inference path (no tuning) -- should fall through to fallback without blowing up.
    chosen_runner, tactic = tuner.choose_one(
        "none_input_smoke", [runner], config, inputs
    )
    assert chosen_runner is runner
    assert tactic == -1


# ---------------------------------------------------------------------------
# Tests: skip_ops
# ---------------------------------------------------------------------------


def test_skip_ops_prevents_profiling(monkeypatch):
    """Skipped ops should return fallback immediately without profiling."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2))
    inputs = [torch.empty((16, 32), dtype=torch.float32)]
    config = TuningConfig()

    profile_calls = []

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        profile_calls.append(tactic)
        return 1.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, skip_ops={"skip_me"}):
        chosen_runner, tactic = tuner.choose_one("skip_me", [runner], config, inputs)

    assert chosen_runner is runner
    assert tactic == -1
    assert len(profile_calls) == 0


def test_skip_ops_does_not_affect_other_ops(monkeypatch):
    """Non-skipped ops should still be profiled normally."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1))
    inputs = [torch.empty((16, 32), dtype=torch.float32)]
    config = TuningConfig()

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        return {0: 5.0, 1: 1.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, skip_ops={"some_other_op"}):
        chosen_runner, tactic = tuner.choose_one("tune_me", [runner], config, inputs)

    assert chosen_runner is runner
    assert tactic == 1  # best tactic selected via profiling


def test_skip_ops_nested_union(monkeypatch):
    """Nested autotune contexts should union their skip_ops sets."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0,))
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        return 1.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, skip_ops={"op_a"}):
        # op_a should be skipped
        _, tactic_a = tuner.choose_one("op_a", [runner], config, inputs)
        assert tactic_a == -1

        with autotune(tune_mode=True, skip_ops={"op_b"}):
            # Both op_a and op_b should be skipped in inner context
            _, tactic_a2 = tuner.choose_one("op_a", [runner], config, inputs)
            assert tactic_a2 == -1
            _, tactic_b = tuner.choose_one("op_b", [runner], config, inputs)
            assert tactic_b == -1

        # After inner context exits, only op_a should still be skipped
        _, tactic_a3 = tuner.choose_one("op_a", [runner], config, inputs)
        assert tactic_a3 == -1


def test_skip_ops_returns_first_runner():
    """Skipped ops should always return runners[0], even with multiple runners."""
    tuner = reset_autotuner()
    runner_a = DummyRunner(valid_tactics=(0,))
    runner_b = DummyRunner(valid_tactics=(1,))
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    with autotune(tune_mode=True, skip_ops={"multi_runner_op"}):
        chosen, tactic = tuner.choose_one(
            "multi_runner_op", [runner_a, runner_b], config, inputs
        )

    assert chosen is runner_a
    assert tactic == -1


def test_skip_ops_empty_set_is_noop(monkeypatch):
    """skip_ops=set() should not skip anything."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1))
    inputs = [torch.empty((16, 32), dtype=torch.float32)]
    config = TuningConfig()

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        return {0: 5.0, 1: 1.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, skip_ops=set()):
        chosen, tactic = tuner.choose_one("should_tune", [runner], config, inputs)

    assert tactic == 1  # profiled and selected best


def test_skip_ops_nested_inner_op_resumes_after_exit(monkeypatch):
    """op_b added by inner context should be profiled again after inner exits."""
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0,))
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    profile_calls = []

    def fake_profile(
        self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs
    ):
        profile_calls.append(1)
        return 1.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    with autotune(tune_mode=True, skip_ops={"op_a"}):
        with autotune(tune_mode=True, skip_ops={"op_b"}):
            _, tactic_b = tuner.choose_one("op_b", [runner], config, inputs)
            assert tactic_b == -1  # skipped in inner

        # After inner exits, op_b should be profiled
        profile_calls.clear()
        _, tactic_b2 = tuner.choose_one("op_b", [runner], config, inputs)
        assert len(profile_calls) > 0  # was profiled


def test_skip_ops_does_not_pollute_cache():
    """Skipped ops should not create entries in profiling_cache."""
    tuner = reset_autotuner()
    runner = DummyRunner()
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    cache_before = len(tuner.profiling_cache)

    with autotune(tune_mode=True, skip_ops={"no_cache_op"}):
        tuner.choose_one("no_cache_op", [runner], config, inputs)

    assert len(tuner.profiling_cache) == cache_before


def test_skip_ops_restored_after_context():
    """skip_ops should be fully cleared after context exits."""
    tuner = reset_autotuner()
    runner = DummyRunner()
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    config = TuningConfig()

    with autotune(tune_mode=False, skip_ops={"some_op"}):
        _, tactic = tuner.choose_one("some_op", [runner], config, inputs)
        assert tactic == -1

    # After context, skip_ops should be empty — op goes through normal path
    assert tuner._effective_skip_ops == frozenset()


def _build_num_tokens_tuning_config(mapper):
    """Build a TuningConfig that buckets dim 0 of input 0 using *mapper*.

    Two configs built with the *same* ``mapper`` object are equal and hash-equal
    (see ``DynamicTensorSpec.__hash__``/``__eq__``), so they collapse to a single
    ``_find_nearest_profile`` lru_cache key.  Built with distinct ``mapper``
    objects (e.g. fresh lambdas) they are distinct keys.
    """
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(512, 1024, 2048, 4096, 8192),
                map_to_tuning_buckets=mapper,
            ),
        ),
    )


def test_find_nearest_profile_cache_dedups_equivalent_configs():
    """Regression test for the _find_nearest_profile lru_cache memory leak.

    ``_find_nearest_profile`` is ``@lru_cache(maxsize=None)`` keyed on
    ``(shapes, tuning_config)``.  ``TuningConfig`` hashes and compares its
    ``DynamicTensorSpec`` via the *identity* of the ``map_to_tuning_buckets``
    callable.  Production callers (e.g. fused MoE) rebuild a ``TuningConfig`` on
    every inference call; as long as ``map_to_tuning_buckets`` is a *stable*
    callable, every rebuilt-but-equivalent config collapses to the same cache
    key and the cache stays bounded.

    This test holds the input shape FIXED and rebuilds an equivalent config on
    every iteration, then asserts the cache does NOT grow.  Contrast with
    ``test_find_nearest_profile_cache_grows_with_fresh_callable`` below, which
    shows the unbounded growth when the callable identity changes per call —
    the exact bug this guards against.
    """
    AutoTuner._find_nearest_profile.cache_clear()

    shapes = ((1024, 128),)

    # Warm up with one equivalent config so the single expected entry exists.
    AutoTuner._find_nearest_profile(
        shapes, _build_num_tokens_tuning_config(last_positive_power_of_2)
    )
    cache_before = AutoTuner._find_nearest_profile.cache_info().currsize

    # Rebuild an *equivalent* config on every call, same shape every time.
    # With a stable callable these all map to one cache key -> no growth.
    N = 5_000
    for _ in range(N):
        config = _build_num_tokens_tuning_config(last_positive_power_of_2)
        AutoTuner._find_nearest_profile(shapes, config)

    cache_growth = AutoTuner._find_nearest_profile.cache_info().currsize - cache_before
    AutoTuner._find_nearest_profile.cache_clear()

    assert cache_growth == 0, (
        f"Cache grew by {cache_growth} entries across {N} calls with equivalent "
        "configs for a fixed shape. Equivalent TuningConfigs must collapse to a "
        "single cache key — a per-call lambda/closure for map_to_tuning_buckets "
        "reintroduces the unbounded-growth leak."
    )


def test_find_nearest_profile_cache_grows_with_fresh_callable():
    """Negative control: a fresh callable identity per call leaks one entry/call.

    This documents the failure mode that
    ``test_find_nearest_profile_cache_dedups_equivalent_configs`` guards against
    and proves the methodology is sound (the cache genuinely *can* grow per call).
    A new ``lambda`` each iteration gives each config a distinct cache key even
    though the shape and bucketing logic are identical, so the cache grows by
    exactly N — the original memory leak.
    """
    AutoTuner._find_nearest_profile.cache_clear()

    shapes = ((1024, 128),)

    # Warm up with one fresh-lambda config.
    AutoTuner._find_nearest_profile(
        shapes, _build_num_tokens_tuning_config(lambda x: last_positive_power_of_2(x))
    )
    cache_before = AutoTuner._find_nearest_profile.cache_info().currsize

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Fresh lambda each iteration -> distinct cache key each call -> leak.
    N = 5_000
    for _ in range(N):
        config = _build_num_tokens_tuning_config(lambda x: last_positive_power_of_2(x))
        AutoTuner._find_nearest_profile(shapes, config)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    cache_growth = AutoTuner._find_nearest_profile.cache_info().currsize - cache_before
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    allocated_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    AutoTuner._find_nearest_profile.cache_clear()

    assert cache_growth == N, (
        f"Expected {N} new cache entries (one per fresh callable), got {cache_growth}."
    )
    assert allocated_bytes > 0, "Expected Python allocation growth from the leak"

    print(
        f"\nFresh-callable leak: cache grew by {cache_growth} entries, "
        f"Python allocations grew by {allocated_bytes / 1024:.1f} KB "
        f"({allocated_bytes / N:.0f} B/call)."
    )


def _build_moe_style_tuning_config(topk_ids_initializer):
    """Build a config with tensor_initializers present, MoE-style.
    Mimics ``_make_tuning_config`` in fused_moe/core.py.
    """
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1),
                dim_idx=(0, 0),
                gen_tuning_buckets=get_hybrid_num_tokens_buckets(8192, 1),
                map_to_tuning_buckets=make_hybrid_bucket_mapper(8192),
                tensor_initializers=[
                    autotuner_initializer_randn,
                    topk_ids_initializer,
                ],
            ),
        ),
    )


def test_find_nearest_profile_cache_dedups_moe_config_with_initializers():
    """Regression test: rebuilt MoE-style configs with tensor_initializers
    must collapse to a single cache entry.
    """
    # The factory must return the identical object for the same expert count.
    assert _moe_topk_ids_init(128) is _moe_topk_ids_init(128)

    AutoTuner._find_nearest_profile.cache_clear()
    shapes = ((1024, 4096), (1024, 8))

    AutoTuner._find_nearest_profile(
        shapes, _build_moe_style_tuning_config(_moe_topk_ids_init(128))
    )
    cache_before = AutoTuner._find_nearest_profile.cache_info().currsize

    N = 1_000
    for _ in range(N):
        config = _build_moe_style_tuning_config(_moe_topk_ids_init(128))
        AutoTuner._find_nearest_profile(shapes, config)

    cache_growth = AutoTuner._find_nearest_profile.cache_info().currsize - cache_before
    AutoTuner._find_nearest_profile.cache_clear()

    assert cache_growth == 0, (
        f"Cache grew by {cache_growth} entries across {N} rebuilds of an "
        "equivalent MoE-style config with a fixed shape."
    )


def test_make_tuning_config_reuses_topk_ids_initializer():
    """_make_tuning_config must return configs whose topk_ids initializer is the
    same object across calls for the same num_experts.
    """
    fn = core_mod.get_trtllm_moe_sm100_module
    fn.cache_clear()
    try:
        mock_module = MagicMock()
        mock_module.get_library_path.return_value = "/tmp/fake.so"
        with (
            patch.object(
                core_mod,
                "gen_trtllm_gen_fused_moe_sm100_module",
                return_value=mock_module,
            ),
            patch.object(core_mod, "setup_cubin_loader"),
        ):
            MoERunner = core_mod.get_trtllm_moe_sm100_module().MoERunner

        runner = MoERunner(
            top_k=8,
            num_local_experts=128,
            dtype_act=DtypeTrtllmGen.Bfloat16,
            dtype_weights=DtypeTrtllmGen.Bfloat16,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=4096,
            intermediate_size=14336,
            num_experts=128,
        )
        moe_inputs = MoeRunnerInputs(
            output=torch.empty((8, 4096)),
            routing_logits=None,
            topk_ids=torch.zeros((8, 8), dtype=torch.int32),
            expert_weights=None,
            hidden_states=torch.empty((8, 4096)),
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )

        config_a = runner._make_tuning_config(moe_inputs)
        config_b = runner._make_tuning_config(moe_inputs)

        spec_a = config_a.dynamic_tensor_specs[0]
        spec_b = config_b.dynamic_tensor_specs[0]
        topk_idx = MoeRunnerInputs.idx("topk_ids")
        init_a = spec_a.tensor_initializers[spec_a.input_idx.index(topk_idx)]
        init_b = spec_b.tensor_initializers[spec_b.input_idx.index(topk_idx)]

        assert init_a is init_b, (
            "_make_tuning_config returned a different topk_ids initializer object "
            "on each call. It must reuse _moe_topk_ids_init(num_experts) so that "
            "rebuilt TuningConfigs collapse to the same _find_nearest_profile "
            "lru_cache key — a per-call closure reintroduces the memory leak."
        )
    finally:
        fn.cache_clear()


def test_find_nearest_profile_cache_grows_with_fresh_closure_initializer():
    """Negative control: a fresh initializer closure per call leaks one
    entry per call DESPITE equal hashes.
    """
    AutoTuner._find_nearest_profile.cache_clear()
    shapes = ((1024, 4096), (1024, 8))

    def make_fresh_closure():
        def _init(s, dt, dev):
            return None

        return _init

    ref_config = _build_moe_style_tuning_config(make_fresh_closure())
    other_config = _build_moe_style_tuning_config(make_fresh_closure())
    assert hash(ref_config) == hash(other_config), "hashes should match"
    assert ref_config != other_config, "equality should fail on fresh closures"

    AutoTuner._find_nearest_profile(shapes, ref_config)
    cache_before = AutoTuner._find_nearest_profile.cache_info().currsize

    N = 1_000
    for _ in range(N):
        config = _build_moe_style_tuning_config(make_fresh_closure())
        AutoTuner._find_nearest_profile(shapes, config)

    cache_growth = AutoTuner._find_nearest_profile.cache_info().currsize - cache_before
    AutoTuner._find_nearest_profile.cache_clear()

    assert cache_growth == N, (
        f"Expected {N} new cache entries (one per fresh closure), got {cache_growth}."
    )


def _call_build_mla_decode_tuning_config():
    """Call _build_mla_decode_tuning_config with fresh (equivalent) tensors.

    runner_names is restricted to trtllm-gen so bucket computation stays
    host-only (no SM count query), letting the test run without a GPU.
    """
    num_pages, page_size, head_dim = 128, 32, 576
    return _build_mla_decode_tuning_config(
        kv_cache=torch.empty((num_pages, page_size, head_dim)),
        block_tables=torch.zeros((8, 64), dtype=torch.int32),
        workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        runner_names=("trtllm-gen",),
        q_len=4,
        num_heads=128,
        kv_lora_rank=512,
        max_seq_len=1024,
        device=torch.device("cpu"),
    )


def test_mla_decode_tuning_config_is_memoized():
    """Equivalent MLA-decode dispatcher calls must reuse one TuningConfig object.

    A fresh config per call embeds two fresh initializer closures; those hash
    identically to but never compare equal with previous ones, so each call
    would leak one _find_nearest_profile cache entry and lookups would scan the
    whole collision chain (observed as GC gen-2 pause growth and decaying
    decode throughput in serving).
    """
    _mla_decode_tuning_config.cache_clear()
    try:
        config_a = _call_build_mla_decode_tuning_config()
        config_b = _call_build_mla_decode_tuning_config()

        assert config_a is config_b, (
            "_build_mla_decode_tuning_config returned a different TuningConfig "
            "object for equivalent arguments. It must memoize on "
            "(buckets, num_pages, profile_seq_len) so rebuilt configs collapse to "
            "the same _find_nearest_profile lru_cache key — a per-call config "
            "reintroduces the memory leak."
        )
    finally:
        _mla_decode_tuning_config.cache_clear()


def test_find_nearest_profile_cache_dedups_mla_decode_config():
    """Regression test: rebuilt MLA-decode configs must collapse to a single
    _find_nearest_profile cache entry.
    """
    AutoTuner._find_nearest_profile.cache_clear()
    _mla_decode_tuning_config.cache_clear()
    try:
        # [query, block_tables, seq_lens, out] as passed by the MLA dispatcher.
        shapes = ((8, 4, 128, 576), (8, 64), (8,), (8, 4, 128, 512))

        AutoTuner._find_nearest_profile(shapes, _call_build_mla_decode_tuning_config())
        cache_before = AutoTuner._find_nearest_profile.cache_info().currsize

        N = 1_000
        for _ in range(N):
            AutoTuner._find_nearest_profile(
                shapes, _call_build_mla_decode_tuning_config()
            )

        cache_growth = (
            AutoTuner._find_nearest_profile.cache_info().currsize - cache_before
        )

        assert cache_growth == 0, (
            f"Cache grew by {cache_growth} entries across {N} rebuilds of an "
            "equivalent MLA-decode tuning config with a fixed shape."
        )
    finally:
        AutoTuner._find_nearest_profile.cache_clear()
        _mla_decode_tuning_config.cache_clear()
