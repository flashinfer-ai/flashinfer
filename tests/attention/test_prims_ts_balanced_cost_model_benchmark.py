"""Regression coverage for the balanced MLA cost-model benchmark harness."""

from __future__ import annotations

import copy

import pytest
import torch

from benchmarks import bench_prims_ts_balanced_mla_cost_model as benchmark
from flashinfer.attention.prims_ts._balanced_plan import BalancedMLADecodePlan
from flashinfer.attention.prims_ts._balanced_scheduler import BalancedCostModel
from flashinfer.attention.prims_ts._balanced_scheduler import B200_BF16_2CTA_COST


def test_cost_model_source_fingerprint_tracks_contents(tmp_path):
    source = tmp_path / "scheduler.py"
    source.write_text("first\n", encoding="utf-8")
    first = benchmark._hash_source_files(tmp_path, (source,))

    source.write_text("second\n", encoding="utf-8")
    second = benchmark._hash_source_files(tmp_path, (source,))

    assert first != second


def test_measurement_source_fingerprint_excludes_fit_only_edits(tmp_path):
    source = tmp_path / "benchmark.py"
    source.write_text(
        "MEASUREMENT = 1\n"
        "FIT = 1\n"
        "def measure():\n    return MEASUREMENT\n"
        "def fit():\n    return FIT\n",
        encoding="utf-8",
    )

    def fingerprint():
        return benchmark._hash_python_symbols(
            source,
            symbols=frozenset({"measure"}),
            constants=frozenset({"MEASUREMENT"}),
        )

    original = fingerprint()
    source.write_text(
        "MEASUREMENT = 1\n"
        "FIT = 2\n"
        "def measure():\n    return MEASUREMENT\n"
        "def fit():\n    return FIT * 2\n",
        encoding="utf-8",
    )
    assert fingerprint() == original

    source.write_text(
        "MEASUREMENT = 2\n"
        "FIT = 2\n"
        "def measure():\n    return MEASUREMENT + 1\n"
        "def fit():\n    return FIT * 2\n",
        encoding="utf-8",
    )
    assert fingerprint() != original


@pytest.mark.parametrize("changed_section", ("hardware", "software", "source"))
def test_cost_model_resume_rejects_identity_mismatch(changed_section: str):
    expected = {
        "schema_version": 2,
        "measurement_method": "native-forced-v1",
        "hardware": {"uuid": "gpu-a"},
        "software": {"torch": "current"},
        "source": {"sha256": "current"},
    }
    existing = copy.deepcopy(expected)
    existing[changed_section][next(iter(existing[changed_section]))] = "stale"

    with pytest.raises(ValueError, match=r"identity does not match"):
        benchmark.validate_resume_signature(
            [{"record": "metadata", "signature": existing}], expected
        )


def test_cost_model_resume_rejects_legacy_artifact_without_identity():
    expected = {
        "schema_version": 2,
        "measurement_method": "native-forced-v1",
    }

    with pytest.raises(ValueError, match=r"identity does not match"):
        benchmark.validate_resume_signature(
            [{"record": "metadata", "signature": {"seed": 7}}], expected
        )


def test_cost_model_resume_accepts_only_exact_single_cohort():
    expected = {
        "schema_version": 2,
        "measurement_method": "native-forced-v1",
        "hardware": {"uuid": "gpu-a"},
        "software": {"torch": "current"},
        "source": {"sha256": "current"},
    }

    assert not benchmark.validate_resume_signature([], expected)
    assert benchmark.validate_resume_signature(
        [{"record": "metadata", "signature": copy.deepcopy(expected)}], expected
    )

    with pytest.raises(ValueError, match=r"metadata is missing"):
        benchmark.validate_resume_signature(
            [{"record": "measurement", "case": "stale"}], expected
        )

    with pytest.raises(ValueError, match=r"identity does not match"):
        benchmark.validate_resume_signature(
            [
                {"record": "metadata", "signature": copy.deepcopy(expected)},
                {"record": "metadata", "signature": copy.deepcopy(expected)},
            ],
            expected,
        )


def test_refit_generation_reuses_measurements_but_refreshes_fit_and_validation():
    old_identity = {"sha256": "fit-v1"}
    new_identity = {"sha256": "fit-v2"}
    bucket = "dense_large"
    records = [
        {"record": "measurement", "case": "case-a", "latency_us": 10.0},
        {
            "record": "fit",
            "fit_generation": 0,
            "fit_identity": old_identity,
            "family": "2cta",
            "dtype": "bf16",
            "cost_bucket": bucket,
            "method": benchmark.fit_method(bucket),
        },
        {
            "record": "validation",
            "fit_generation": 0,
            "fit_identity": old_identity,
            "family": "2cta",
            "dtype": "bf16",
            "cost_bucket": bucket,
            "case": "case-a",
            "method": benchmark.validation_method(bucket),
        },
    ]

    assert (
        benchmark.select_fit_generation(
            records,
            requested_generation=None,
            current_fit_identity=old_identity,
        )
        == 0
    )
    with pytest.raises(ValueError, match="--refit-generation 1"):
        benchmark.select_fit_generation(
            records,
            requested_generation=None,
            current_fit_identity=new_identity,
        )

    generation = benchmark.select_fit_generation(
        records,
        requested_generation=1,
        current_fit_identity=new_identity,
    )
    assert generation == 1
    assert benchmark.completed_fit_records(records, generation) == {}
    assert benchmark.completed_validation_keys(records, generation) == set()
    assert [record for record in records if record["record"] == "measurement"] == [
        {"record": "measurement", "case": "case-a", "latency_us": 10.0}
    ]

    records.extend(
        [
            {
                "record": "fit",
                "fit_generation": generation,
                "fit_identity": new_identity,
                "family": "2cta",
                "dtype": "bf16",
                "cost_bucket": bucket,
                "method": benchmark.fit_method(bucket),
            },
            {
                "record": "validation",
                "fit_generation": generation,
                "fit_identity": new_identity,
                "family": "2cta",
                "dtype": "bf16",
                "cost_bucket": bucket,
                "case": "case-a",
                "method": benchmark.validation_method(bucket),
            },
        ]
    )
    assert list(benchmark.completed_fit_records(records, generation)) == [
        ("2cta", "bf16", bucket)
    ]
    assert benchmark.completed_validation_keys(records, generation) == {
        ("2cta", "bf16", "case-a")
    }
    assert (
        benchmark.select_fit_generation(
            records,
            requested_generation=None,
            current_fit_identity=new_identity,
        )
        == generation
    )

    with pytest.raises(ValueError, match="newer than"):
        benchmark.select_fit_generation(
            records,
            requested_generation=0,
            current_fit_identity=new_identity,
        )


def test_refit_generation_rejects_incomplete_measurement_cohort(tmp_path):
    case = benchmark.CalibrationCase("missing", (128,), 128)

    with pytest.raises(ValueError, match="complete measurement cohort"):
        benchmark.collect_measurements(
            output_path=tmp_path / "artifact.jsonl",
            records=[],
            families=("2cta",),
            dtype_names=("bf16",),
            cases=(case,),
            device=torch.device("cpu"),
            seed=1,
            warmups=1,
            iterations=1,
            trials=1,
            quick=True,
            refit_only=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_forced_target_stages_current_packed_descriptor_abi():
    plan = BalancedMLADecodePlan(
        batch_size=2,
        num_partitions=4,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
    )

    stats = benchmark.stage_forced_target(plan, (10 * 128, 2 * 128), target=3)

    assert stats is not None
    assert stats["descriptor_count"] == 5
    assert stats["combine_request_count"] == 1
    assert stats["placement_cost_model"] == {
        "cost_per_k_tile": 1000,
        "fixed_piece_cost": 0,
        "split_fixed_cost": 0,
        "split_piece_cost": 0,
        "reducer_fixed_cost": 0,
        "reducer_piece_cost": 0,
    }
    assert stats["schedule_sha256"] == benchmark.schedule_sha256(
        stats["schedule_snapshot"]
    )
    descriptors = plan.work_descriptors[:5].cpu().tolist()
    request_zero = sorted(
        (row for row in descriptors if row[0] == 0), key=lambda row: row[1]
    )
    assert [(row[1], row[2]) for row in request_zero] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]
    split_info = [row[3] for row in request_zero]
    assert [value & 1 for value in split_info] == [1, 1, 1, 1]
    assert [(value >> 1) & 0xFFF for value in split_info] == [0, 1, 2, 3]
    assert [(value >> 13) & 0x7F for value in split_info] == [4, 4, 4, 4]
    assert plan.combine_descriptors[0].cpu().tolist() == [0, 4, 0, 0]
    assert plan.num_combine_descriptors.cpu().tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_forced_target_uses_production_lpt_short_placement():
    """Keep forced calibration placement identical to the native scheduler."""

    num_short_requests = 127
    plan = BalancedMLADecodePlan(
        batch_size=num_short_requests + 1,
        num_partitions=74,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
    )
    seq_lens = (4 * 128,) * num_short_requests + (1024 * 128,)

    stats = benchmark.stage_forced_target(plan, seq_lens, target=128)

    assert stats is not None
    offsets = plan.partition_offsets.cpu().tolist()
    descriptors = plan.work_descriptors[: plan.last_descriptor_count].cpu().tolist()
    short_partitions = set()
    long_partitions = set()
    for partition_idx, (begin, end) in enumerate(
        zip(offsets, offsets[1:], strict=False)
    ):
        for descriptor in descriptors[begin:end]:
            (
                short_partitions
                if descriptor[0] < num_short_requests
                else long_partitions
            ).add(partition_idx)
    assert short_partitions == set(range(74))
    assert long_partitions == set(range(53, 61))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_selected_target_and_native_forced_target_emit_identical_schedule():
    """Permit paired-timing reuse only for byte-equivalent native schedules."""

    seq_lens = (512,) * 127 + (131072,)
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=74,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
    )
    plan.replan(seq_lens)
    selected_target = plan.last_target_piece_tiles
    selected_schedule = benchmark.schedule_signature(plan)
    selected_snapshot = benchmark.schedule_snapshot(plan)
    reference_snapshot = benchmark.schedule_snapshot_from_reference(
        benchmark.build_balanced_schedule(
            seq_lens,
            num_partitions=plan.num_partitions,
            k_tile_tokens=plan.k_tile_tokens,
            cost=plan.cost,
        )
    )

    stats = benchmark.stage_forced_target(plan, seq_lens, selected_target)

    assert stats is not None
    assert benchmark.schedule_signature(plan) == selected_schedule
    assert stats["schedule_snapshot"] == selected_snapshot == reference_snapshot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA plan buffers")
def test_same_forced_target_can_emit_different_cost_model_placements():
    """A target is not a complete measurement or schedule identity."""

    seq_tiles = (10,) * 72 + (6, 4, 4, 3) + (400,)
    seq_lens = tuple(value * 128 for value in seq_tiles)
    long_request = len(seq_lens) - 1
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=74,
        device=torch.device("cuda"),
        cost=B200_BF16_2CTA_COST,
    )

    bootstrap = benchmark.stage_forced_target(plan, seq_lens, target=200)
    assert bootstrap is not None
    plan.cost = BalancedCostModel(1000, 0)
    flat = benchmark.stage_forced_target(plan, seq_lens, target=200)
    assert flat is not None

    def request_partitions(snapshot, request_idx):
        result = set()
        descriptors = snapshot["work_descriptors"]
        offsets = snapshot["partition_offsets"]
        for partition, (begin, end) in enumerate(
            zip(offsets, offsets[1:], strict=False)
        ):
            if any(row[0] == request_idx for row in descriptors[begin:end]):
                result.add(partition)
        return result

    assert bootstrap["schedule_sha256"] != flat["schedule_sha256"]
    assert request_partitions(bootstrap["schedule_snapshot"], long_request) == {
        0,
        1,
    }
    assert request_partitions(flat["schedule_snapshot"], long_request) == {72, 73}

    restored = benchmark.stage_recorded_schedule(
        plan, seq_lens, {**bootstrap, "target_tiles": 200}
    )
    assert restored["schedule_sha256"] == bootstrap["schedule_sha256"]
    assert plan.cost == B200_BF16_2CTA_COST


def test_observed_latency_is_keyed_by_schedule_not_target():
    rows = [
        {"target_tiles": 200, "schedule_sha256": "schedule-a", "latency_us": 10.0},
        {"target_tiles": 200, "schedule_sha256": "schedule-b", "latency_us": 25.0},
    ]

    assert benchmark.observed_schedule_latency(rows, "schedule-a") == 10.0
    assert benchmark.observed_schedule_latency(rows, "schedule-b") == 25.0
    assert benchmark.observed_schedule_latency(rows, "unmeasured") is None


@pytest.mark.arch_blackwell
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="balanced PrimsTS MLA requires SM100 or SM103",
)
@pytest.mark.parametrize("dtype_name", ("bf16", "fp8"))
def test_cost_model_benchmark_measures_forced_graph_against_reference(dtype_name: str):
    case = benchmark.CalibrationCase("smoke", (4096,), 4096)
    runtime = benchmark.make_runtime(
        case,
        family="2cta",
        dtype_name=dtype_name,
        device=torch.device("cuda"),
        seed=20260911,
    )
    assert runtime.wrapper._plan_state is not None
    plan = runtime.wrapper._plan_state.balanced_plan
    assert plan is not None

    stats = benchmark.stage_forced_target(plan, case.seq_lens, target=8)
    assert stats is not None
    assert stats["max_split_count"] == 4
    runtime.graph.replay()
    torch.cuda.synchronize()

    max_abs_diff = benchmark.check_output(runtime)
    assert max_abs_diff < (2e-2 if dtype_name == "fp8" else 5e-3)
