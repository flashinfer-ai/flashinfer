"""Tests for balanced PrimsTS MLA gating and CUDA schedulers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.attention.prims_ts.balanced_scheduler import (
    cost_model as scheduler_module,
    tune_cost_model as cost_model_tuner,
)
from flashinfer.attention.prims_ts.balanced_scheduler.gate import (
    BALANCED_MLA_GATE_THRESHOLDS,
    EXPECTED_MAX_SEQ_LEN_ENV,
    EXPECTED_MEAN_SEQ_LEN_ENV,
    should_use_prims_ts_balanced_mla,
)
from flashinfer.attention.prims_ts.balanced_scheduler.cost_model import (
    B200_BALANCED_COST_MODEL_ID,
    B200_BF16_2CTA_COST,
    BalancedCostModel,
    balanced_cost_bucket,
    require_balanced_cost_model_calibration,
    select_b200_balanced_cost_model,
)
from flashinfer.attention.prims_ts.balanced_scheduler.plan import (
    BalancedMLADecodePlan,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.helpers.constants import (
    balanced_partial_capacity,
    balanced_reducer_capacity,
    balanced_work_descriptor_capacity,
)


def _make_tuner_identity_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create the minimum source tree required by measurement fingerprinting."""

    root = tmp_path / "repo"
    tuner_path = (
        root / "flashinfer/attention/prims_ts/balanced_scheduler/tune_cost_model.py"
    )
    tuner_path.parent.mkdir(parents=True)
    tuner_path.write_text(
        Path(cost_model_tuner.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "flashinfer/attention/prims_ts/runtime.py").write_text(
        "MEASUREMENT_RUNTIME_VERSION = 1\n",
        encoding="utf-8",
    )
    csrc_root = root / "csrc/prims_ts"
    csrc_root.mkdir(parents=True)
    for name in (
        "balanced_mla_plan.cu",
        "balanced_mla_scheduler_device.cu",
        "balanced_mla_scheduler.cuh",
    ):
        (csrc_root / name).write_text(f"// {name}\n", encoding="utf-8")
    return root, tuner_path


def test_measurement_fingerprint_ignores_fit_only_tuner_edits(tmp_path):
    root, tuner_path = _make_tuner_identity_tree(tmp_path)
    measurement_before = cost_model_tuner.measurement_source_identity(root)
    fit_before = cost_model_tuner.fit_source_identity(root)

    source = tuner_path.read_text(encoding="utf-8")
    edited = source.replace(
        'FIT_METHOD = "tail_aware_selection_regret_v8_schedule_identity"',
        'FIT_METHOD = "test_fit_only_change"',
        1,
    )
    assert edited != source
    tuner_path.write_text(edited, encoding="utf-8")

    assert cost_model_tuner.measurement_source_identity(root) == measurement_before
    assert cost_model_tuner.fit_source_identity(root) != fit_before


def test_measurement_fingerprint_tracks_measurement_tuner_edits(tmp_path):
    root, tuner_path = _make_tuner_identity_tree(tmp_path)
    before = cost_model_tuner.measurement_source_identity(root)

    source = tuner_path.read_text(encoding="utf-8")
    edited = source.replace("PAGE_SIZE = 32", "PAGE_SIZE = 64", 1)
    assert edited != source
    tuner_path.write_text(edited, encoding="utf-8")
    after = cost_model_tuner.measurement_source_identity(root)

    # The tuner is absent from the whole-file digest, but its selected
    # measurement definitions retain their independent fingerprint.
    assert after["sha256"] == before["sha256"]
    assert after["file_count"] == before["file_count"]
    assert after["measurement_harness_sha256"] != before["measurement_harness_sha256"]


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


def _assert_device_plan_covers_requests(plan, seq_lens):
    """Check the descriptor ABI without requiring a particular placement order."""

    metadata = plan.synchronize_device_metadata()
    descriptor_count, _target, combine_count, status, _bucket = metadata
    assert status == 0
    descriptors = plan.work_descriptors[:descriptor_count].cpu().tolist()
    offsets = plan.partition_offsets.cpu().tolist()
    assert offsets[0] == 0
    assert offsets[-1] == descriptor_count
    assert offsets == sorted(offsets)

    request_descriptors = defaultdict(list)
    for partition in range(plan.num_partitions):
        for descriptor in descriptors[offsets[partition] : offsets[partition + 1]]:
            request, start, end, split_info = descriptor
            assert 0 <= request < len(seq_lens)
            assert 0 <= start < end
            request_descriptors[request].append((start, end, split_info))

    expected_split_begins = {}
    split_cursor = 0
    for request, seq_len in enumerate(seq_lens):
        expected_tiles = (seq_len + plan.k_tile_tokens - 1) // plan.k_tile_tokens
        pieces = sorted(request_descriptors[request])
        if expected_tiles == 0:
            assert pieces == []
            continue
        assert pieces[0][0] == 0
        assert pieces[-1][1] == expected_tiles
        assert all(
            left[1] == right[0] for left, right in zip(pieces, pieces[1:], strict=False)
        )
        if len(pieces) == 1:
            assert pieces[0][2] == 0
            continue
        expected_split_begins[request] = split_cursor
        global_indices = set()
        for _start, _end, split_info in pieces:
            assert split_info & 1
            global_indices.add((split_info >> 1) & 0xFFF)
            assert ((split_info >> 13) & 0x7F) == len(pieces)
            assert ((split_info >> 20) & 0xFFF) == split_cursor
        assert global_indices == set(range(split_cursor, split_cursor + len(pieces)))
        split_cursor += len(pieces)

    combine_descriptors = plan.combine_descriptors[:combine_count].cpu().tolist()
    assert plan.num_combine_descriptors.cpu().tolist() == [combine_count]
    assert [row[0] for row in combine_descriptors] == list(expected_split_begins)
    for request, num_pieces, split_begin, reserved in combine_descriptors:
        assert num_pieces == len(request_descriptors[request])
        assert split_begin == expected_split_begins[request]
        assert reserved == 0


def _modeled_emitted_makespan(plan: BalancedMLADecodePlan, cost: BalancedCostModel):
    descriptors = plan.work_descriptors[: plan.last_descriptor_count].cpu().tolist()
    offsets = plan.partition_offsets.cpu().tolist()
    additive_combine = bool(cost.reducer_fixed_cost or cost.reducer_piece_cost)
    max_pieces = 0
    loads = []
    for partition in range(plan.num_partitions):
        load = 0
        for _request, start, end, split_info in descriptors[
            offsets[partition] : offsets[partition + 1]
        ]:
            is_split = split_info & 1
            pieces = (split_info >> 13) & 0x7F if is_split else 1
            max_pieces = max(max_pieces, pieces)
            blocks = end - start
            charged_blocks = (
                (blocks + plan.num_insts_kv - 1) // plan.num_insts_kv
            ) * plan.num_insts_kv
            load += cost.fixed_piece_cost + charged_blocks * cost.cost_per_k_tile
            if is_split and not additive_combine:
                load += cost.split_piece_cost + cost.split_fixed_cost // pieces
        loads.append(load)
    makespan = max(loads, default=0)
    if additive_combine and max_pieces >= 2:
        makespan += cost.reducer_fixed_cost + cost.reducer_piece_cost * max_pieces
    return makespan


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
            match=r"balanced_scheduler.*tune_cost_model.*add the measured",
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
    with pytest.raises(NotImplementedError, match="uncalibrated fallback"):
        BalancedMLADecodePlan(
            batch_size=1,
            num_partitions=1,
            device=torch.device("cuda:7"),
        )


def test_only_cuda_scheduler_implementations_remain():
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "csrc/prims_ts/balanced_mla_scheduler.cu").exists()
    assert not hasattr(scheduler_module, "build_balanced_schedule")
    jit_source = (
        repo_root / "flashinfer/attention/prims_ts/balanced_scheduler/jit.py"
    ).read_text()
    assert "balanced_mla_scheduler_device.cu" in jit_source
    assert '"balanced_mla_scheduler.cu"' not in jit_source


_REQUIRES_CUDA_SCHEDULER = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="balanced scheduler tests require a CUDA device",
)


def _snapshot(plan):
    return (
        plan.last_descriptor_count,
        plan.last_target_piece_tiles,
        plan.last_combine_request_count,
        plan.work_descriptors[: plan.last_descriptor_count].cpu().tolist(),
        plan.partition_offsets.cpu().tolist(),
        plan.combine_descriptors[: plan.last_combine_request_count].cpu().tolist(),
    )


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize("scheduler", ("exact", "optimized"))
@pytest.mark.parametrize(
    "seq_lens,num_partitions,cost",
    (
        ([0] * 8, 4, BalancedCostModel(1000, 0)),
        ([2048] * 16, 4, BalancedCostModel(1000, 0)),
        ([131072, 0, 4096, 128, 0, 32768], 8, B200_BF16_2CTA_COST),
        ([2 * 128, 7 * 128], 3, BalancedCostModel(1000, 0)),
        (
            [257 * 128, 16 * 128],
            4,
            BalancedCostModel(1000, 5000, split_piece_cost=3000),
        ),
    ),
)
def test_cuda_scheduler_policies_preserve_descriptor_semantics(
    scheduler, seq_lens, num_partitions, cost
):
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=num_partitions,
        device=torch.device("cuda"),
        cost=cost,
        max_seq_len=max(seq_lens, default=0),
    )
    device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    plan.schedule_device(device_seq_lens, scheduler=scheduler)

    _assert_device_plan_covers_requests(plan, seq_lens)


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize("scheduler", ("exact", "optimized"))
def test_cuda_scheduler_splits_on_complete_cta_work_units(scheduler):
    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=4,
        device=torch.device("cuda"),
        k_tile_tokens=128,
        num_insts_kv=2,
        cost=BalancedCostModel(1000, 0),
        max_seq_len=9 * 128,
    )
    seq_lens = torch.tensor([9 * 128], dtype=torch.int32, device="cuda")

    plan.schedule_device(
        seq_lens,
        scheduler=scheduler,
        forced_target_piece_tiles=3,
    )

    assert plan.last_target_piece_tiles == 4
    descriptors = sorted(
        plan.work_descriptors[: plan.last_descriptor_count].cpu().tolist()
    )
    assert [descriptor[:3] for descriptor in descriptors] == [
        [0, 0, 4],
        [0, 4, 8],
        [0, 8, 9],
    ]
    _assert_device_plan_covers_requests(plan, [9 * 128])


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize("scheduler", ("exact", "optimized"))
@pytest.mark.parametrize("num_insts_kv", (1, 2))
@pytest.mark.parametrize(
    "cost",
    (
        BalancedCostModel(1000, 7000, 11000, 13000),
        BalancedCostModel(1000, 7000, 0, 0, 19000, 1700),
    ),
)
def test_cuda_scheduler_reports_emitted_modeled_makespan(scheduler, num_insts_kv, cost):
    seq_lens = [131072, 65536, 32768, 8192, 1024, 256, 0]
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=8,
        device=torch.device("cuda"),
        num_insts_kv=num_insts_kv,
        cost=BalancedCostModel(1000, 0),
        evaluation_cost=cost,
        max_seq_len=max(seq_lens),
    )

    plan.schedule_device(
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        scheduler=scheduler,
        estimate_cost=True,
    )

    assert plan.last_predicted_cost == _modeled_emitted_makespan(plan, cost)


@_REQUIRES_CUDA_SCHEDULER
def test_optimized_scheduler_is_default_and_policies_are_named():
    seq_lens = [131072, 8192, 257, 0]
    plan = BalancedMLADecodePlan(
        batch_size=4,
        num_partitions=8,
        device=torch.device("cuda"),
        cost=B200_BF16_2CTA_COST,
        max_seq_len=131072,
    )
    device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    plan.schedule_device(device_seq_lens)
    default_snapshot = _snapshot(plan)
    plan.schedule_device(device_seq_lens, scheduler="exact")
    _assert_device_plan_covers_requests(plan, seq_lens)
    plan.schedule_device(device_seq_lens, scheduler="optimized")

    assert _snapshot(plan) == default_snapshot
    with pytest.raises(ValueError, match="'exact' or 'optimized'"):
        plan.schedule_device(device_seq_lens, scheduler="equivalent")


@_REQUIRES_CUDA_SCHEDULER
def test_optimized_scheduler_rejects_more_than_fold_thread_partitions():
    seq_lens = [128]
    accepted_plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=256,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=128,
    )
    plan = BalancedMLADecodePlan(
        batch_size=1,
        num_partitions=257,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=128,
    )
    device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    accepted_plan.schedule_device(device_seq_lens, scheduler="optimized")
    _assert_device_plan_covers_requests(accepted_plan, seq_lens)

    with pytest.raises(RuntimeError, match=r"supports at most\s+256\s+partitions"):
        plan.schedule_device(device_seq_lens, scheduler="optimized")

    plan.schedule_device(device_seq_lens, scheduler="exact")
    _assert_device_plan_covers_requests(plan, seq_lens)


def _fold_dynamic_shared_bytes(batch_size: int, num_partitions: int) -> int:
    block_bytes = (batch_size * 4 + 7) & ~7
    return block_bytes + (batch_size + 2 * num_partitions) * 8


@_REQUIRES_CUDA_SCHEDULER
def test_optimized_scheduler_opts_in_to_large_dynamic_shared_memory():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    default_bytes = properties.shared_memory_per_block
    opt_in_bytes = properties.shared_memory_per_block_optin
    if opt_in_bytes <= default_bytes + 8192:
        pytest.skip("CUDA device has no useful dynamic shared-memory opt-in range")

    num_partitions = 4
    batch_size = 1
    while (
        _fold_dynamic_shared_bytes(batch_size, num_partitions) <= default_bytes + 4096
    ):
        batch_size += 1
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    seq_lens[0] = 128
    plan = BalancedMLADecodePlan(
        batch_size=batch_size,
        num_partitions=num_partitions,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=128,
    )

    plan.schedule_device(seq_lens, scheduler="optimized")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.schedule_device(seq_lens, validate=False, scheduler="optimized")
    graph.replay()

    assert plan.partition_offsets[-1].item() == 1


@_REQUIRES_CUDA_SCHEDULER
def test_optimized_scheduler_rejects_unsupported_dynamic_shared_memory():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    opt_in_bytes = properties.shared_memory_per_block_optin
    num_partitions = 4
    batch_size = 1
    while _fold_dynamic_shared_bytes(batch_size, num_partitions) <= opt_in_bytes:
        batch_size += 1
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    seq_lens[0] = 128
    plan = BalancedMLADecodePlan(
        batch_size=batch_size,
        num_partitions=num_partitions,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=128,
    )

    with pytest.raises(
        RuntimeError, match=r"permits at most\s+\d+\s+total opt-in bytes"
    ):
        plan.schedule_device(seq_lens, scheduler="optimized")


@_REQUIRES_CUDA_SCHEDULER
def test_device_plan_allocates_compact_capacities():
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


@_REQUIRES_CUDA_SCHEDULER
def test_cuda_scheduler_reuses_stable_output_addresses():
    plan = BalancedMLADecodePlan(
        batch_size=4,
        num_partitions=8,
        device=torch.device("cuda"),
        cost=B200_BF16_2CTA_COST,
        max_seq_len=131072,
    )
    addresses = (
        plan.work_descriptors.data_ptr(),
        plan.partition_offsets.data_ptr(),
        plan.combine_descriptors.data_ptr(),
        plan.num_combine_descriptors.data_ptr(),
    )
    for seq_lens in ([131072, 128, 4096, 2048], [128, 32768, 256, 8192]):
        device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        plan.schedule_device(device_seq_lens)
        _assert_device_plan_covers_requests(plan, seq_lens)
        assert addresses == (
            plan.work_descriptors.data_ptr(),
            plan.partition_offsets.data_ptr(),
            plan.combine_descriptors.data_ptr(),
            plan.num_combine_descriptors.data_ptr(),
        )


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize("scheduler", ("exact", "optimized"))
def test_cuda_scheduler_runs_inside_cuda_graph(scheduler):
    plan = BalancedMLADecodePlan(
        batch_size=4,
        num_partitions=8,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=131072,
    )
    live_seq_lens = torch.tensor(
        [131072, 128, 4096, 2048], dtype=torch.int32, device="cuda"
    )
    plan.schedule_device(live_seq_lens, scheduler=scheduler)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.schedule_device(live_seq_lens, validate=False, scheduler=scheduler)

    replay_lengths = [128, 32768, 256, 8192]
    live_seq_lens.copy_(torch.tensor(replay_lengths, dtype=torch.int32, device="cuda"))
    graph.replay()

    _assert_device_plan_covers_requests(plan, replay_lengths)


@_REQUIRES_CUDA_SCHEDULER
def test_cuda_scheduler_failure_publishes_empty_safe_schedule():
    plan = BalancedMLADecodePlan(
        batch_size=2,
        num_partitions=4,
        device=torch.device("cuda"),
        cost=BalancedCostModel(1000, 0),
        max_seq_len=4096,
    )
    invalid_lengths = torch.tensor([-1, 8192], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="sequence length"):
        plan.schedule_device(invalid_lengths)

    assert plan.partition_offsets.cpu().tolist() == [0] * 5
    assert plan.num_combine_descriptors.cpu().tolist() == [0]


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize("target", (1, 2, 3))
def test_forced_single_partition_plan_never_marks_a_split(target):
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
        max_seq_len=256,
    )
    seq_lens = torch.tensor([256], dtype=torch.int32, device="cuda")

    plan.schedule_device(seq_lens, forced_target_piece_tiles=target)

    assert plan.last_descriptor_count == 1
    assert plan.last_target_piece_tiles == target
    assert plan.last_combine_request_count == 0
    assert plan.work_descriptors[0].cpu().tolist() == [0, 0, 2, 0]
    assert plan.partition_offsets.cpu().tolist() == [0, 1]


@_REQUIRES_CUDA_SCHEDULER
@pytest.mark.parametrize(
    "seq_lens,cost,expected_target",
    (
        (
            [2048] * 5
            + [3072] * 3
            + [10112] * 12
            + [30080] * 24
            + [50048] * 17
            + [100096] * 2
            + [110080],
            BalancedCostModel(1000, 64000),
            144,
        ),
        (
            [4096] * 24
            + [8192] * 13
            + [16384] * 13
            + [32768] * 7
            + [65536] * 4
            + [131072] * 3,
            BalancedCostModel(1000, 64000),
            141,
        ),
        (
            [4096] * 11 + [8192] * 6 + [16384] * 6 + [32768] * 4 + [65536] * 5,
            BalancedCostModel(1000, 0, split_fixed_cost=35244),
            86,
        ),
        (
            [4096] * 10 + [8192] * 7 + [16384] * 9 + [32768] * 5 + [65536],
            BalancedCostModel(1000, 1000),
            64,
        ),
    ),
)
def test_optimized_scheduler_corrects_wave_boundaries(seq_lens, cost, expected_target):
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=74,
        device=torch.device("cuda"),
        cost=cost,
        max_seq_len=131072,
    )
    device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    plan.schedule_device(device_seq_lens, scheduler="optimized")

    assert plan.last_target_piece_tiles == expected_target
    _assert_device_plan_covers_requests(plan, seq_lens)


@_REQUIRES_CUDA_SCHEDULER
def test_device_scheduler_selects_calibrated_bucket_from_live_lengths():
    if "B200" not in torch.cuda.get_device_name().upper():
        pytest.skip("B200 calibration is device-specific")
    seq_lens = [512] * 7 + [131072]
    plan = BalancedMLADecodePlan(
        batch_size=len(seq_lens),
        num_partitions=74,
        device=torch.device("cuda"),
        kernel_family="throughput_2cta",
        dtype_name="bf16",
        max_seq_len=131072,
    )

    plan.schedule_device(torch.tensor(seq_lens, dtype=torch.int32, device="cuda"))

    assert plan.last_cost_bucket == "sparse_small"
    assert plan.cost_model_info["bucket"] == "sparse_small"
    _assert_device_plan_covers_requests(plan, seq_lens)
