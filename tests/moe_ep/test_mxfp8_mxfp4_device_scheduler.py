# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host simulation and source-shape checks for the device scheduler."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


_HERE = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/cute_dsl/megamoe/mxfp8_mxfp4"
)
_SPEC = spec_from_file_location(
    "mxfp8_mxfp4_scheduler_contract", _HERE / "scheduler.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_HOST = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HOST
_SPEC.loader.exec_module(_HOST)


def _config(**overrides):
    values = dict(
        expert_token_counts=(65, 0, 17),
        gate_up_size=512,
        hidden_size=256,
        cta_tile_m=32,
        cta_tile_n=128,
        cluster_shape_mn=(2, 1),
        group_hint=5,
        token_padding_block=64,
        sf_padding_block=128,
        num_persistent_clusters=3,
        fc2_ready_threshold=2,
    )
    values.update(overrides)
    return _HOST.SchedulerConfig(**values)


def _trace(claimer, scheduler):
    return [record.to_words() for record in scheduler.records_for_claimer(claimer)]


@pytest.mark.parametrize("clusters", [3, 64])
def test_static_and_atomic_claim_traces_cover_the_same_device_work(clusters) -> None:
    """Model the device claim paths against the stable WorkRecord contract."""
    scheduler = _HOST.PersistentFc1Fc2Scheduler(
        _config(num_persistent_clusters=clusters)
    )
    static_records = []
    for cluster_id in range(scheduler.config.num_persistent_clusters):
        static_records.extend(
            _trace(
                _HOST.StaticTileClaimer(
                    cluster_id=cluster_id,
                    cluster_count=scheduler.config.num_persistent_clusters,
                ),
                scheduler,
            )
        )
    atomic_records = _trace(_HOST.AtomicTileClaimer(), scheduler)
    assert sorted(static_records) == sorted(atomic_records)

    # Expert 2 follows differently padded data/SF rows and two token blocks
    # from expert 0. This is the block-32 scale-row invariant the device
    # scheduler must preserve when rewinding FC1 -> FC2 and opening a group.
    expert_2 = next(words for words in atomic_records if words[0] == 2)
    assert expert_2[3:6] == (128, 128, 2)


def test_every_cluster_cta_decodes_one_claim_with_shared_phase_and_offsets() -> None:
    scheduler = _HOST.PersistentFc1Fc2Scheduler(_config(expert_token_counts=(65,)))
    for claim in range(scheduler.cluster_tile_count):
        records = [
            scheduler.record_for(claim, cta_coord_mn=(cta_token, 0))
            for cta_token in range(2)
        ]
        assert len({r.expert_idx for r in records}) == 1
        assert len({r.phase for r in records}) == 1
        assert len({r.cumulative_data_physical_row for r in records}) == 1
        assert len({r.cumulative_sf_physical_row for r in records}) == 1
        assert records[1].tile_n_idx == records[0].tile_n_idx + 1


@pytest.mark.parametrize("order", ["expert", "longest_first"])
@pytest.mark.parametrize("mode", ["static", "atomic_counter", "alternating"])
def test_scheduler_params_optional_fields_roundtrip(mode, order, monkeypatch):
    """Exercise the actual marshaling methods with strict lightweight values.

    Stand-ins reject None so a lost optional-field guard or inconsistent
    extraction order fails without requiring a GPU.
    Device compilation remains covered by the GPU integration suite.
    """
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import device_scheduler

    class DynamicInt(int):
        pass

    class Value:
        def __init__(self, name):
            self.name = name

    def extract(value):
        assert value is not None
        return [value]

    def rebind(prototype, values):
        assert prototype is not None and len(values) == 1
        return values[0]

    monkeypatch.setattr(device_scheduler, "Int32", DynamicInt)
    monkeypatch.setattr(device_scheduler, "extract_mlir_values", extract)
    monkeypatch.setattr(device_scheduler, "new_from_mlir_values", rebind)
    params = device_scheduler.DeviceSchedulerParams(
        expert_token_sizes=Value("counts"),
        expert_count=DynamicInt(3),
        gate_up_size=512,
        hidden_size=256,
        cta_tile_tokens=64,
        cta_tile_features=128,
        cluster_shape_token_feature=(1, 1),
        group_hint=5,
        token_padding_block=64,
        sf_padding_block=128,
        fc1_ready_counter_ptr=Value("dispatch"),
        fc1_done_counter_ptr=Value("fc1_done"),
        fc2_ready_threshold=4,
        load_balance_mode=mode,
        claim_order=order,
        load_balance_counter_ptr=Value("claim") if mode != "static" else None,
    )
    values = params.__extract_mlir_values__()
    rebound = params.__new_from_mlir_values__(values)
    assert vars(rebound) == vars(params)
    # Replacement values must actually propagate instead of closing over the
    # old Python object. This matters across MLIR region boundaries.
    replacements = [Value(f"new_{index}") for index in range(len(values))]
    rebound = params.__new_from_mlir_values__(replacements)
    assert rebound.expert_token_sizes is replacements[1]
    assert rebound.fc1_ready_counter_ptr is replacements[2]
