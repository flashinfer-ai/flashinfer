"""CPU-only layout tests for the balanced PrimsTS MLA host planner."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.attention.prims_ts import _balanced_scheduler as scheduler_module
from flashinfer.attention.prims_ts._balanced_gate import (
    BALANCED_MLA_GATE_THRESHOLDS,
    EXPECTED_MAX_SEQ_LEN_ENV,
    EXPECTED_MEAN_SEQ_LEN_ENV,
    should_use_prims_ts_balanced_mla,
)
from flashinfer.attention.prims_ts._balanced_plan import BalancedMLADecodePlan
from flashinfer.attention.prims_ts._balanced_scheduler import (
    B200_BALANCED_COST_MODEL_ID,
    B200_BF16_2CTA_COST,
    BalancedCostModel,
    balanced_cost_bucket,
    build_balanced_schedule,
    require_balanced_cost_model_calibration,
    select_b200_balanced_cost_model,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.constants import (
    balanced_partial_capacity,
    balanced_reducer_capacity,
    balanced_work_descriptor_capacity,
)


def _request_ranges(schedule):
    ranges = defaultdict(list)
    for descriptor in schedule.descriptors:
        ranges[descriptor.request_idx].append(
            (
                descriptor.local_split_idx,
                descriptor.k_tile_start,
                descriptor.k_tile_start + descriptor.k_tile_count,
            )
        )
    return ranges


@pytest.mark.parametrize(
    "num_heads,dtype_name,min_batch,min_tokens",
    [(*key, *thresholds) for key, thresholds in BALANCED_MLA_GATE_THRESHOLDS.items()],
)
def test_production_gate_enforces_all_three_admission_terms(
    num_heads, dtype_name, min_batch, min_tokens
):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float8_e4m3fn
    expected_mean = (min_tokens + min_batch - 1) // min_batch
    expected_max = 2 * expected_mean
    kwargs = {
        "batch_size": min_batch,
        "num_heads": num_heads,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_kv_len": expected_max,
        "max_seq_len_q": 1,
        "q_dtype": dtype,
        "kv_dtype": dtype,
        "expected_mean_seq_len": expected_mean,
        "expected_max_seq_len": expected_max,
    }

    assert should_use_prims_ts_balanced_mla(**kwargs)
    assert not should_use_prims_ts_balanced_mla(
        **{**kwargs, "batch_size": min_batch - 1}
    )
    assert not should_use_prims_ts_balanced_mla(
        **{
            **kwargs,
            "expected_mean_seq_len": max(1, (min_tokens - 1) // min_batch),
        }
    )
    assert not should_use_prims_ts_balanced_mla(
        **{**kwargs, "expected_max_seq_len": 2 * expected_mean - 1}
    )


def test_production_gate_rejects_uniform_and_short_ragged_regressions():
    common = {
        "num_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_seq_len_q": 1,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
    }

    # Every uniform regression has concentration one, independent of scale.
    assert not should_use_prims_ts_balanced_mla(
        **common,
        batch_size=128,
        max_kv_len=131_072,
        expected_mean_seq_len=131_072,
        expected_max_seq_len=131_072,
    )
    # The final sweep's BF16 2CTA ragged-2K/B8 loss has concentration 2.91,
    # but only 5,632 expected tokens, below this row's 8,192-token floor.
    assert not should_use_prims_ts_balanced_mla(
        **common,
        batch_size=8,
        max_kv_len=2_048,
        expected_mean_seq_len=704,
        expected_max_seq_len=2_048,
    )


def test_production_gate_uses_environment_declarations(monkeypatch):
    monkeypatch.setenv(EXPECTED_MEAN_SEQ_LEN_ENV, "4096")
    monkeypatch.setenv(EXPECTED_MAX_SEQ_LEN_ENV, "8192")
    kwargs = {
        "batch_size": 2,
        "num_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_kv_len": 8192,
        "max_seq_len_q": 1,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
    }

    assert should_use_prims_ts_balanced_mla(**kwargs)
    # Explicit arguments take precedence over process-wide declarations.
    assert not should_use_prims_ts_balanced_mla(
        **kwargs,
        expected_mean_seq_len=4096,
        expected_max_seq_len=4096,
    )


def test_production_gate_requires_consistent_declarations(monkeypatch):
    monkeypatch.delenv(EXPECTED_MEAN_SEQ_LEN_ENV, raising=False)
    monkeypatch.delenv(EXPECTED_MAX_SEQ_LEN_ENV, raising=False)
    kwargs = {
        "batch_size": 2,
        "num_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_kv_len": 8192,
        "max_seq_len_q": 1,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
    }

    with pytest.raises(ValueError, match=EXPECTED_MEAN_SEQ_LEN_ENV):
        should_use_prims_ts_balanced_mla(**kwargs)
    with pytest.raises(ValueError, match=EXPECTED_MAX_SEQ_LEN_ENV):
        should_use_prims_ts_balanced_mla(
            **kwargs,
            expected_mean_seq_len=4096,
        )
    with pytest.raises(ValueError, match="greater than or equal"):
        should_use_prims_ts_balanced_mla(
            **kwargs,
            expected_mean_seq_len=4096,
            expected_max_seq_len=2048,
        )


def test_production_gate_defaults_unsupported_shapes_without_declarations(
    monkeypatch,
):
    monkeypatch.delenv(EXPECTED_MEAN_SEQ_LEN_ENV, raising=False)
    monkeypatch.delenv(EXPECTED_MAX_SEQ_LEN_ENV, raising=False)
    common = {
        "batch_size": 128,
        "num_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_kv_len": 131_072,
        "max_seq_len_q": 1,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
    }

    assert not should_use_prims_ts_balanced_mla(**{**common, "max_seq_len_q": 2})
    assert not should_use_prims_ts_balanced_mla(**{**common, "num_heads": 8})
    assert not should_use_prims_ts_balanced_mla(
        **{**common, "kv_dtype": torch.float8_e4m3fn}
    )


def test_production_gate_caps_expected_max_at_plan_capacity():
    assert not should_use_prims_ts_balanced_mla(
        batch_size=2,
        num_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_kv_len=6144,
        max_seq_len_q=1,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        expected_mean_seq_len=4096,
        expected_max_seq_len=131_072,
    )


def _assert_device_plan_matches_reference(plan, schedule):
    torch.cuda.synchronize()
    descriptor_count = len(schedule.descriptors)

    expected_descriptors = []
    for descriptor in schedule.descriptors:
        split_count = schedule.split_counts[descriptor.request_idx]
        split_info = 0
        if split_count > 1:
            split_begin = schedule.split_begins[descriptor.request_idx]
            split_info = (
                1
                | ((split_begin + descriptor.local_split_idx) << 1)
                | (split_count << 13)
                | (split_begin << 20)
            )
        expected_descriptors.append(
            [
                descriptor.request_idx,
                descriptor.k_tile_start,
                descriptor.k_tile_start + descriptor.k_tile_count,
                split_info,
            ]
        )
    assert plan.last_descriptor_count == descriptor_count
    assert plan.last_target_piece_tiles == schedule.target_piece_tiles
    assert (
        plan.work_descriptors[:descriptor_count].cpu().tolist() == expected_descriptors
    )
    assert plan.partition_offsets.cpu().tolist() == list(schedule.partition_offsets)
    expected_combine_requests = [
        request_idx
        for request_idx, split_count in enumerate(schedule.split_counts)
        if split_count > 1
    ]
    assert plan.last_combine_request_count == len(expected_combine_requests)
    assert plan.num_combine_descriptors.cpu().tolist() == [
        len(expected_combine_requests)
    ]
    expected_combine_descriptors = [
        [
            request_idx,
            schedule.split_counts[request_idx],
            schedule.split_begins[request_idx],
            0,
        ]
        for request_idx in expected_combine_requests
    ]
    assert plan.combine_descriptors[
        : len(expected_combine_requests)
    ].cpu().tolist() == (expected_combine_descriptors)


def test_empty_schedule_has_replay_stable_shape():
    schedule = build_balanced_schedule([0, 0, 0], num_partitions=4)

    assert schedule.descriptors == ()
    assert schedule.partition_offsets == (0, 0, 0, 0, 0)
    assert schedule.split_counts == (0, 0, 0)
    assert schedule.split_begins == (0, 0, 0)
    assert schedule.partition_costs == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "batch_size,num_partitions,expected",
    [
        (128, 74, (202, 148, 74)),
        (32, 74, (106, 106, 32)),
    ],
)
def test_balanced_capacities_separate_work_partials_and_reducer_slots(
    batch_size, num_partitions, expected
):
    assert (
        balanced_work_descriptor_capacity(batch_size, num_partitions),
        balanced_partial_capacity(batch_size, num_partitions),
        balanced_reducer_capacity(batch_size, num_partitions),
    ) == expected


def test_inactive_slots_preserve_active_plus_partition_capacity_proof():
    seq_lens = [131072, 65536, 32768] + [0] * 125
    num_partitions = 74
    schedule = build_balanced_schedule(seq_lens, num_partitions=num_partitions)

    active_count = sum(seq_len > 0 for seq_len in seq_lens)
    split_partial_count = sum(
        split_count for split_count in schedule.split_counts if split_count > 1
    )
    combine_count = sum(split_count > 1 for split_count in schedule.split_counts)
    assert len(schedule.descriptors) <= active_count + num_partitions
    assert split_partial_count <= balanced_partial_capacity(
        len(seq_lens), num_partitions
    )
    assert combine_count <= balanced_reducer_capacity(len(seq_lens), num_partitions)


@pytest.mark.parametrize(
    "seq_lens,num_partitions,expected",
    [
        ([0, 0], 8, "single"),
        ([131072], 74, "single"),
        ([128] * 8, 74, "sparse_uniform_small"),
        ([512] * 7 + [2048], 74, "legacy"),
        ([512] * 7 + [131072], 74, "sparse_small"),
        ([512] * 127 + [16384], 74, "legacy"),
        ([512] * 127 + [131072], 74, "sparse"),
        ([128] * 64, 74, "sparse"),
        ([32768] * 32, 74, "dense_small"),
        ([8192] * 128, 74, "dense_large"),
    ],
)
def test_cost_bucket_uses_replay_visible_workload(seq_lens, num_partitions, expected):
    assert balanced_cost_bucket(seq_lens, num_partitions=num_partitions) == expected


def test_b200_cost_lookup_normalizes_runtime_policy_names():
    bucket, cost = select_b200_balanced_cost_model(
        [512] * 7 + [131072],
        kernel_family="throughput_2cta",
        dtype_name="e4m3",
        num_partitions=74,
    )

    assert bucket == "sparse_small"
    assert cost == BalancedCostModel(1000, 0)

    uniform_bucket, uniform_cost = select_b200_balanced_cost_model(
        [8192] * 8,
        kernel_family="throughput_2cta",
        dtype_name="bf16",
        num_partitions=74,
    )
    uniform_schedule = build_balanced_schedule(
        [8192] * 8,
        num_partitions=74,
        cost=uniform_cost,
    )
    assert uniform_bucket == "sparse_uniform_small"
    assert uniform_schedule.target_piece_tiles == 8


def test_balanced_calibration_registry_requires_exact_device_identity():
    calibration = require_balanced_cost_model_calibration(
        device_name="NVIDIA B200",
        compute_capability=(10, 0),
        multi_processor_count=148,
        device_index=0,
    )
    assert calibration.model_id == B200_BALANCED_COST_MODEL_ID

    for identity in (
        ("NVIDIA B300", (10, 3), 148),
        ("NVIDIA B200", (10, 0), 132),
        ("NVIDIA B200-SXM", (10, 0), 148),
    ):
        with pytest.raises(
            NotImplementedError,
            match=r"bench_prims_ts_balanced_mla_cost_model.py.*add the measured",
        ):
            require_balanced_cost_model_calibration(
                device_name=identity[0],
                compute_capability=identity[1],
                multi_processor_count=identity[2],
                device_index=7,
            )


def test_balanced_device_plan_rejects_uncalibrated_device_before_allocation(
    monkeypatch,
):
    properties = SimpleNamespace(
        name="Uncalibrated Accelerator",
        major=10,
        minor=0,
        multi_processor_count=148,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("an uncalibrated plan must fail before allocation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(
        NotImplementedError,
        match=r"will not use an uncalibrated fallback",
    ):
        BalancedMLADecodePlan(
            batch_size=1,
            num_partitions=1,
            device=torch.device("cuda:7"),
        )


def test_native_scheduler_has_no_obsolete_cost_lookup_or_gqa_routing():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "csrc/prims_balanced_mla_scheduler.cu").read_text()
    header = (repo_root / "csrc/prims_balanced_mla_scheduler.cuh").read_text()

    for obsolete in (
        "BalancedCostFamily",
        "hasBalancedCostModel",
        "hostLookupCost",
        "useCostOverride",
        "costPerBlockOverride",
    ):
        assert obsolete not in source
        assert obsolete not in header


def test_ragged_schedule_covers_every_k_tile_once():
    seq_lens = [131072, 8193, 4096, 257, 1, 0, 65537, 2048]
    schedule = build_balanced_schedule(seq_lens, num_partitions=74)
    ranges = _request_ranges(schedule)

    assert len(schedule.descriptors) <= len(seq_lens) + 74
    assert schedule.partition_offsets[0] == 0
    assert schedule.partition_offsets[-1] == len(schedule.descriptors)
    assert list(schedule.partition_offsets) == sorted(schedule.partition_offsets)

    for request_idx, seq_len in enumerate(seq_lens):
        expected_tiles = (seq_len + 127) // 128
        request_ranges = sorted(ranges[request_idx])
        assert schedule.split_counts[request_idx] == len(request_ranges)
        if expected_tiles == 0:
            assert request_ranges == []
            continue
        assert [entry[0] for entry in request_ranges] == list(
            range(len(request_ranges))
        )
        cursor = 0
        for _, begin, end in request_ranges:
            assert begin == cursor
            assert end > begin
            cursor = end
        assert cursor == expected_tiles


def test_concentrated_work_is_split_and_load_balanced():
    schedule = build_balanced_schedule(
        [131072] + [128] * 7,
        num_partitions=16,
    )

    assert schedule.split_counts[0] > 1
    assert max(schedule.partition_costs) < (
        B200_BF16_2CTA_COST.fixed_piece_cost
        + 1024 * B200_BF16_2CTA_COST.cost_per_k_tile
    )


def test_mixed_uniform_shorts_use_lpt_across_all_partitions():
    cost = BalancedCostModel(cost_per_k_tile=1000, fixed_piece_cost=0)

    schedule = build_balanced_schedule(
        [1024] * 64 + [2048] * 64,
        num_partitions=74,
        cost=cost,
    )

    assert schedule.target_piece_tiles == 24
    assert schedule.split_counts == (1,) * 128
    assert max(schedule.partition_costs) == 24_000
    assert all(value > 0 for value in schedule.partition_costs)


def test_cost_search_scores_emitted_short_then_long_placement():
    cost = BalancedCostModel(
        cost_per_k_tile=1000,
        fixed_piece_cost=5000,
        split_piece_cost=3000,
    )

    schedule = build_balanced_schedule(
        [257 * 128, 16 * 128],
        num_partitions=4,
        cost=cost,
    )

    # Targets 78 and 86 both emit a 94-us modeled makespan. The exact
    # placement scorer therefore applies the established coarser-target tie
    # break instead of preferring 78 from a different global-LPT placement.
    assert schedule.target_piece_tiles == 86
    assert max(schedule.partition_costs) == 94_000


def test_equal_length_ties_are_deterministic():
    first = build_balanced_schedule([2048] * 16, num_partitions=8)
    second = build_balanced_schedule([2048] * 16, num_partitions=8)

    assert first == second
    for partition_idx in range(8):
        begin, end = first.partition_offsets[partition_idx : partition_idx + 2]
        request_indices = [
            descriptor.request_idx for descriptor in first.descriptors[begin:end]
        ]
        assert request_indices == sorted(request_indices)


@pytest.mark.parametrize(
    "seq_lens,num_partitions,k_tile_tokens",
    [([-1], 1, 128), ([1], 0, 128), ([1], 1, 0)],
)
def test_invalid_inputs_are_rejected(seq_lens, num_partitions, k_tile_tokens):
    with pytest.raises(ValueError):
        build_balanced_schedule(
            seq_lens,
            num_partitions=num_partitions,
            k_tile_tokens=k_tile_tokens,
        )


def test_reference_rejects_unrepresentable_derived_target():
    int32_max = 2**31 - 1
    cost = BalancedCostModel(
        cost_per_k_tile=1,
        fixed_piece_cost=0,
        split_fixed_cost=int32_max,
        split_piece_cost=int32_max,
    )

    with pytest.raises(OverflowError, match="target piece tiles exceeds int32"):
        build_balanced_schedule([128], num_partitions=1, cost=cost)


def test_split_penalty_changes_target_without_breaking_coverage():
    expensive_splits = BalancedCostModel(
        cost_per_k_tile=1,
        fixed_piece_cost=1,
        reducer_fixed_cost=10_000,
        reducer_piece_cost=10_000,
    )
    schedule = build_balanced_schedule(
        [128 * 32, 128 * 31],
        num_partitions=8,
        cost=expensive_splits,
    )

    assert schedule.split_counts == (1, 1)
    assert sum(descriptor.k_tile_count for descriptor in schedule.descriptors) == 63


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_plan_handles_int32_max_sequence_length():
    int32_max = 2**31 - 1
    cost = BalancedCostModel(cost_per_k_tile=1, fixed_piece_cost=0)
    expected = build_balanced_schedule(
        [int32_max],
        num_partitions=1,
        cost=cost,
    )
    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=1,
        device=torch.device("cuda"),
        cost=cost,
    )

    plan.replan([int32_max])

    assert expected.descriptors[0].k_tile_count == (int32_max + 127) // 128
    _assert_device_plan_matches_reference(plan, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_plan_rejects_unrepresentable_derived_target():
    int32_max = 2**31 - 1
    cost = BalancedCostModel(
        cost_per_k_tile=1,
        fixed_piece_cost=0,
        split_fixed_cost=int32_max,
        split_piece_cost=int32_max,
    )
    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=1,
        device=torch.device("cuda"),
        cost=cost,
    )

    with pytest.raises(Exception, match="target piece tiles exceeds int32"):
        plan.replan([128])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_forced_plan_enforces_active_plus_partition_limit():
    plan = BalancedMLADecodePlan(
        batch_size=8,
        num_partitions=4,
        device=torch.device("cuda"),
        cost=BalancedCostModel(cost_per_k_tile=1000, fixed_piece_cost=0),
    )

    with pytest.raises(Exception, match=r"active \+ P limit"):
        plan._replan_forced_target([1024, 1024, 0, 0, 0, 0, 0, 0], 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
@pytest.mark.parametrize("target", (1, 2, 3))
def test_native_forced_single_partition_plan_never_marks_a_split(target):
    """A P=1 clamp must write public output rather than an orphan partial."""

    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=1,
        device=torch.device("cuda"),
        cost=BalancedCostModel(
            cost_per_k_tile=1000,
            fixed_piece_cost=0,
            split_fixed_cost=100_000,
            split_piece_cost=100_000,
        ),
    )

    plan._replan_forced_target([256], target)

    assert plan.last_descriptor_count == 1
    assert plan.last_target_piece_tiles == target
    assert plan.last_combine_request_count == 0
    assert plan.work_descriptors[0].cpu().tolist() == [0, 0, 2, 0]
    assert plan.partition_offsets.cpu().tolist() == [0, 1]
    assert plan.num_combine_descriptors.cpu().tolist() == [0]


def test_reference_single_partition_clamp_does_not_charge_split_penalty():
    cost = BalancedCostModel(
        cost_per_k_tile=1000,
        fixed_piece_cost=3000,
        split_fixed_cost=100_000,
        split_piece_cost=200_000,
    )
    partitions = [[]]
    partition_costs = [0]

    scheduler_module._equal_split_assign(
        partitions,
        partition_costs,
        request_indices=[0],
        k_tiles=[2],
        target_piece_tiles=1,
        cost=cost,
        charge_split_penalty=True,
    )

    assert partition_costs == [5000]
    assert len(partitions[0]) == 1
    assert partitions[0][0].num_pieces == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_device_plan_allocates_compact_partial_and_reducer_capacities():
    plan = BalancedMLADecodePlan(
        batch_size=128,
        num_partitions=74,
        device=torch.device("cuda"),
        cost=B200_BF16_2CTA_COST,
    )

    assert plan.descriptor_capacity == 202
    assert plan.partial_capacity == 148
    assert plan.reducer_capacity == 74
    assert plan.work_descriptors.shape == (202, 4)
    assert plan.combine_descriptors.shape == (74, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_device_plan_replans_without_changing_addresses():
    plan = BalancedMLADecodePlan(
        batch_size=4,
        num_partitions=8,
        device=torch.device("cuda"),
        cost=B200_BF16_2CTA_COST,
    )
    addresses = (
        plan.work_descriptors.data_ptr(),
        plan.partition_offsets.data_ptr(),
        plan.combine_descriptors.data_ptr(),
        plan.num_combine_descriptors.data_ptr(),
    )

    first = build_balanced_schedule([131072, 128, 4096, 2048], num_partitions=8)
    plan.replan([131072, 128, 4096, 2048])
    _assert_device_plan_matches_reference(plan, first)

    second = build_balanced_schedule([128, 32768, 256, 8192], num_partitions=8)
    plan.replan([128, 32768, 256, 8192])
    torch.cuda.synchronize()
    assert addresses == (
        plan.work_descriptors.data_ptr(),
        plan.partition_offsets.data_ptr(),
        plan.combine_descriptors.data_ptr(),
        plan.num_combine_descriptors.data_ptr(),
    )
    _assert_device_plan_matches_reference(plan, second)

    # Reusing pinned staging immediately must not race the prior nonblocking
    # host-to-device refill.
    plan.replan([131072, 128, 4096, 2048])
    plan.replan([128, 32768, 256, 8192])
    _assert_device_plan_matches_reference(plan, second)

    with pytest.raises(ValueError, match="non-negative int32"):
        plan.replan([-1, 32768, 256, 8192])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_cost_search_matches_equal_piece_reference():
    """Score the quotient/remainder piece sizes emitted by both planners."""

    cost = BalancedCostModel(cost_per_k_tile=1000, fixed_piece_cost=0)
    seq_lens = [2 * 128, 7 * 128]
    expected = build_balanced_schedule(
        seq_lens,
        num_partitions=3,
        cost=cost,
    )
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=3,
        device=torch.device("cuda"),
        cost=cost,
    )

    plan.replan(seq_lens)

    assert expected.target_piece_tiles == 6
    assert expected.split_counts == (1, 2)
    assert sorted(
        descriptor.k_tile_count
        for descriptor in expected.descriptors
        if descriptor.request_idx == 1
    ) == [3, 4]
    _assert_device_plan_matches_reference(plan, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_cost_search_scores_emitted_placement():
    cost = BalancedCostModel(
        cost_per_k_tile=1000,
        fixed_piece_cost=5000,
        split_piece_cost=3000,
    )
    seq_lens = [257 * 128, 16 * 128]
    expected = build_balanced_schedule(
        seq_lens,
        num_partitions=4,
        cost=cost,
    )
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=4,
        device=torch.device("cuda"),
        cost=cost,
    )

    plan.replan(seq_lens)

    assert expected.target_piece_tiles == 86
    _assert_device_plan_matches_reference(plan, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_native_cost_search_charges_fixed_only_reducer_model():
    """Treat any explicit reducer model as an additive critical-path cost."""

    cost = BalancedCostModel(
        cost_per_k_tile=1000,
        fixed_piece_cost=0,
        reducer_fixed_cost=10_000,
        reducer_piece_cost=0,
    )
    seq_lens = [4 * 128]
    expected = build_balanced_schedule(
        seq_lens,
        num_partitions=4,
        cost=cost,
    )
    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=4,
        device=torch.device("cuda"),
        cost=cost,
    )

    plan.replan(seq_lens)

    assert expected.target_piece_tiles == 4
    assert expected.split_counts == (1,)
    _assert_device_plan_matches_reference(plan, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_device_plan_selects_calibrated_cost_bucket_on_b200():
    if "B200" not in torch.cuda.get_device_name().upper():
        pytest.skip("B200 calibration is device-specific")
    seq_lens = [512] * 7 + [131072]
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=74,
        device=torch.device("cuda"),
        cost=None,
        kernel_family="throughput_2cta",
        dtype_name="bf16",
    )
    _, cost = select_b200_balanced_cost_model(
        seq_lens,
        kernel_family="2cta",
        dtype_name="bf16",
        num_partitions=74,
    )

    plan.replan(seq_lens)
    expected = build_balanced_schedule(seq_lens, num_partitions=74, cost=cost)

    assert plan.last_cost_bucket == "sparse_small"
    assert plan.cost == cost
    info = plan.cost_model_info
    assert info["model_id"] == B200_BALANCED_COST_MODEL_ID
    assert info["source"] == "calibrated-registry"
    assert info["device_name"] == "NVIDIA B200"
    assert info["compute_capability"] == (10, 0)
    assert info["multi_processor_count"] == 148
    assert info["kernel_family"] == "throughput_2cta"
    assert info["dtype"] == "bf16"
    assert info["bucket"] == "sparse_small"
    assert info["parameters"] == cost
    with pytest.raises(TypeError):
        info["bucket"] = "dense_large"
    _assert_device_plan_matches_reference(plan, expected)
