# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only launch-policy and tuning-contract tests for PCIe IPC AG/RS."""

import pytest
import torch

from flashinfer.comm._pcie_ipc_collective_tuning import (
    PcieIpcCollectiveTuningState,
    candidate_configs,
    config_to_tactic,
    default_cache_path,
    tactic_to_config,
)
from flashinfer.comm._pcie_ipc_ag_rs_topology import (
    _RankLinks,
    decide_pcie_ipc_ag_rs_topology,
)
from flashinfer.comm.pcie_ipc_ag import _TUNING_SPEC as AG_TUNING_SPEC
from flashinfer.comm.pcie_ipc_ag_policy import (
    PcieIpcAllGatherVariant,
    get_pcie_ipc_all_gather_launch_config,
)
from flashinfer.comm.pcie_ipc_rs import _TUNING_SPEC as RS_TUNING_SPEC
from flashinfer.comm.pcie_ipc_rs_policy import (
    PcieIpcReduceScatterVariant,
    get_pcie_ipc_reduce_scatter_launch_config,
)
from flashinfer.comm._pcie_ipc_workspace import _PcieIpcWorkspace


def test_policy_uses_payload_bytes_at_variant_thresholds() -> None:
    ag_threshold = 256 * 1024
    for payload_bytes, expected_variant in (
        (ag_threshold - 16, PcieIpcAllGatherVariant.FLAT_PUSH),
        (ag_threshold, PcieIpcAllGatherVariant.RECURSIVE_DOUBLING),
    ):
        ag_16bit = get_pcie_ipc_all_gather_launch_config(
            4, payload_bytes // 2, element_size=2
        )
        ag_fp32 = get_pcie_ipc_all_gather_launch_config(
            4, payload_bytes // 4, element_size=4
        )
        assert ag_16bit == ag_fp32
        assert ag_16bit.variant == expected_variant

    rs_threshold = 128 * 1024
    for payload_bytes, expected_variant in (
        (rs_threshold - 16, PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK),
        (rs_threshold, PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC),
    ):
        rs_16bit = get_pcie_ipc_reduce_scatter_launch_config(
            8,
            payload_bytes // 2,
            element_size=2,
            ordered_4plus4=True,
        )
        rs_fp32 = get_pcie_ipc_reduce_scatter_launch_config(
            8,
            payload_bytes // 4,
            element_size=4,
            ordered_4plus4=True,
        )
        assert rs_16bit == rs_fp32
        assert rs_16bit.variant == expected_variant

    assert get_pcie_ipc_all_gather_launch_config(2, 3, element_size=4) is None
    assert get_pcie_ipc_reduce_scatter_launch_config(2, 7, element_size=2) is None
    for element_size in (1, 8, 16):
        assert (
            get_pcie_ipc_all_gather_launch_config(2, 16, element_size=element_size)
            is None
        )
        assert (
            get_pcie_ipc_reduce_scatter_launch_config(2, 16, element_size=element_size)
            is None
        )


def test_workspace_constructor_jointly_checks_collective_identity(monkeypatch) -> None:
    group = object()
    collective_name = "PCIe IPC all-gather"

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def fake_all_gather_object(gathered, local, group=None) -> None:
        assert local["collective"] == collective_name
        gathered[0] = dict(local)
        gathered[1] = {**local, "collective": "PCIe IPC reduce-scatter"}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(ValueError, match="collective"):
        _PcieIpcWorkspace(
            group=group,
            max_numel=8,
            dtype=torch.bfloat16,
            max_blocks=1,
            collective_name=collective_name,
            module_getter=lambda: None,
            workspace_size_name="unused_workspace_size",
            init_name="unused_init",
            dispose_name="unused_dispose",
        )


@pytest.mark.parametrize("element_size", [2, 4])
@pytest.mark.parametrize("offset", [-16, 0, 16])
def test_rs_flat_seed_threshold_does_not_apply_to_tp4(
    element_size: int, offset: int
) -> None:
    # This is a seed regression check, not a claim about measured performance.
    payload_bytes = 4 * 1024 * 1024 + offset
    tp2 = get_pcie_ipc_reduce_scatter_launch_config(
        2, payload_bytes // element_size, element_size=element_size
    )
    tp4 = get_pcie_ipc_reduce_scatter_launch_config(
        4, payload_bytes // element_size, element_size=element_size
    )
    expected_tp2 = (
        PcieIpcReduceScatterVariant.FLAT_ONE_PACK
        if offset < 0
        else PcieIpcReduceScatterVariant.FLAT_CYCLIC
    )
    assert tp2.variant == expected_tp2
    assert tp4.variant == PcieIpcReduceScatterVariant.FLAT_ONE_PACK


def test_policy_admits_only_world_size_specific_variants() -> None:
    assert (
        get_pcie_ipc_all_gather_launch_config(2, 8).variant
        == PcieIpcAllGatherVariant.RECURSIVE_DOUBLING
    )
    assert (
        get_pcie_ipc_all_gather_launch_config(4, 8).variant
        == PcieIpcAllGatherVariant.FLAT_PUSH
    )
    assert (
        get_pcie_ipc_all_gather_launch_config(
            8, 256 * 1024, ordered_4plus4=True
        ).variant
        == PcieIpcAllGatherVariant.COPY_ENGINE
    )
    assert (
        get_pcie_ipc_reduce_scatter_launch_config(2, 8).variant
        == PcieIpcReduceScatterVariant.FLAT_ONE_PACK
    )
    assert (
        get_pcie_ipc_reduce_scatter_launch_config(8, 8, ordered_4plus4=True).variant
        == PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK
    )
    assert get_pcie_ipc_reduce_scatter_launch_config(8, 8) is None


@pytest.mark.parametrize("payload_bytes", [4 * 1024, 8 * 1024 * 1024])
@pytest.mark.parametrize(
    "spec,world_size,ordered_4plus4,expected_variants",
    [
        (
            AG_TUNING_SPEC,
            2,
            False,
            {PcieIpcAllGatherVariant.RECURSIVE_DOUBLING},
        ),
        (
            AG_TUNING_SPEC,
            4,
            False,
            {
                PcieIpcAllGatherVariant.FLAT_PUSH,
                PcieIpcAllGatherVariant.RECURSIVE_DOUBLING,
            },
        ),
        (
            AG_TUNING_SPEC,
            8,
            False,
            {PcieIpcAllGatherVariant.RECURSIVE_DOUBLING},
        ),
        (
            AG_TUNING_SPEC,
            8,
            True,
            {
                PcieIpcAllGatherVariant.RECURSIVE_DOUBLING,
                PcieIpcAllGatherVariant.COPY_ENGINE,
            },
        ),
        (
            RS_TUNING_SPEC,
            2,
            False,
            {
                PcieIpcReduceScatterVariant.FLAT_CYCLIC,
                PcieIpcReduceScatterVariant.FLAT_ONE_PACK,
            },
        ),
        (
            RS_TUNING_SPEC,
            4,
            False,
            {
                PcieIpcReduceScatterVariant.FLAT_CYCLIC,
                PcieIpcReduceScatterVariant.FLAT_ONE_PACK,
            },
        ),
        (RS_TUNING_SPEC, 8, False, set()),
        (
            RS_TUNING_SPEC,
            8,
            True,
            {
                PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC,
                PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK,
            },
        ),
    ],
)
def test_tuning_explores_all_admitted_variants_across_seed_thresholds(
    spec, world_size, ordered_4plus4, expected_variants, payload_bytes
) -> None:
    configs = candidate_configs(
        spec,
        world_size=world_size,
        shard_numel=payload_bytes // 2,
        element_size=2,
        max_blocks=3,
        ordered_4plus4=ordered_4plus4,
        blocks=(0, 1, 3, 4),
        threads=(64, 512, 1024),
    )
    assert {config.variant for config in configs} == expected_variants
    # A smaller workspace capacity and the AG/RS kernel's thread limit must
    # constrain every candidate, including alternatives to the default seed.
    assert all(0 < config.blocks <= 3 and config.threads <= 512 for config in configs)
    assert all(
        config.threads == 32
        for config in configs
        if spec is AG_TUNING_SPEC
        and config.variant == PcieIpcAllGatherVariant.COPY_ENGINE
    )


def test_candidate_enumeration_and_tactic_round_trip_are_dtype_generic() -> None:
    kwargs = dict(
        spec=RS_TUNING_SPEC,
        world_size=2,
        max_blocks=64,
        blocks=(1, 2),
        threads=(64,),
    )
    configs_16bit = candidate_configs(shard_numel=1024, element_size=2, **kwargs)
    configs_fp32 = candidate_configs(shard_numel=512, element_size=4, **kwargs)
    assert configs_16bit == configs_fp32
    for spec, configs in (
        (RS_TUNING_SPEC, configs_fp32),
        (
            AG_TUNING_SPEC,
            candidate_configs(
                AG_TUNING_SPEC,
                world_size=4,
                shard_numel=512,
                element_size=4,
                max_blocks=64,
                blocks=(1,),
                threads=(64,),
            ),
        ),
    ):
        assert configs
        for config in configs:
            assert tactic_to_config(spec, config_to_tactic(config)) == config


class _FakeWorkspace:
    world_size = 2
    max_blocks = 64
    max_numel = 4096

    def __init__(
        self,
        dtype: torch.dtype,
        *,
        ordered_4plus4: bool = False,
        placement: str = "placement-a",
    ) -> None:
        self.dtype = dtype
        self.ordered_4plus4 = ordered_4plus4
        self.placement_fingerprint = placement


def test_tuning_cache_key_separates_dtype_and_shape() -> None:
    state = object.__new__(PcieIpcCollectiveTuningState)
    state.spec = AG_TUNING_SPEC

    state.workspace = _FakeWorkspace(torch.bfloat16)
    bf16 = state._cache_key(torch.empty((1, 8), dtype=torch.bfloat16))
    state.workspace = _FakeWorkspace(torch.float16)
    fp16 = state._cache_key(torch.empty((1, 8), dtype=torch.float16))
    state.workspace = _FakeWorkspace(torch.float32)
    fp32 = state._cache_key(torch.empty((1, 8), dtype=torch.float32))
    other_shape = state._cache_key(torch.empty((2, 4), dtype=torch.float32))
    state.workspace = _FakeWorkspace(torch.float32, placement="placement-b")
    other_placement = state._cache_key(torch.empty((1, 8), dtype=torch.float32))
    state.workspace = _FakeWorkspace(torch.float32, ordered_4plus4=True)
    ordered = state._cache_key(torch.empty((1, 8), dtype=torch.float32))

    assert len({bf16, fp16, fp32, other_shape, other_placement, ordered}) == 6
    assert default_cache_path(
        AG_TUNING_SPEC, 2, torch.bfloat16, "placement-a"
    ) != default_cache_path(AG_TUNING_SPEC, 2, torch.float16, "placement-a")
    assert default_cache_path(
        AG_TUNING_SPEC, 2, torch.float32, "placement-a"
    ) != default_cache_path(AG_TUNING_SPEC, 2, torch.float32, "placement-b")


def _topology(islands: tuple[int, ...]):
    uuids = [f"GPU-{rank}" for rank in range(8)]
    return [
        _RankLinks(
            rank=rank,
            hostname="host",
            device_uuid=uuids[rank],
            peer_system={
                uuids[peer]: islands[rank] != islands[peer]
                for peer in range(8)
                if peer != rank
            },
        )
        for rank in range(8)
    ]


def test_tp8_topology_requires_ordered_four_plus_four_ranks() -> None:
    accepted = decide_pcie_ipc_ag_rs_topology(_topology((0, 0, 0, 0, 1, 1, 1, 1)))
    interleaved = decide_pcie_ipc_ag_rs_topology(_topology((0, 1, 0, 1, 0, 1, 0, 1)))
    incomplete_topology = _topology((0, 0, 0, 0, 1, 1, 1, 1))
    del incomplete_topology[0].peer_system["GPU-1"]
    incomplete = decide_pcie_ipc_ag_rs_topology(incomplete_topology)
    errored_topology = _topology((0, 0, 0, 0, 1, 1, 1, 1))
    errored_topology[0].pair_errors["GPU-1"] = "probe failed"
    errored = decide_pcie_ipc_ag_rs_topology(errored_topology)

    assert accepted.ordered_4plus4
    assert not interleaved.ordered_4plus4
    assert not incomplete.ordered_4plus4
    assert not errored.ordered_4plus4
    assert accepted.placement_fingerprint != interleaved.placement_fingerprint


def test_workspace_configuration_is_read_only() -> None:
    properties = (
        "group",
        "world_size",
        "dtype",
        "max_numel",
        "max_blocks",
        "ordered_4plus4",
        "placement_fingerprint",
    )
    assert all(getattr(_PcieIpcWorkspace, name).fset is None for name in properties)
