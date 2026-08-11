"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Properties of the PCIe IPC autotuning layer that need no GPU.

Everything here guards a failure that is silent on hardware: a tactic that
cannot be persisted, a cache entry reused for the wrong workspace, a verdict
reduced with the wrong operator. The multi-GPU tests can only observe the
consequences, and one of the consequences is a hang.
"""

import json

import pytest
import torch

from flashinfer.autotuner import _json_to_tactic, _tactic_to_json, make_bucket_mapper
from flashinfer.comm.pcie_ipc_policy import (
    MAX_BLOCKS,
    IpcLaunchConfig,
    IpcVariant,
    _is_launchable,
    get_pcie_ipc_launch_config,
)
from flashinfer.comm.pcie_ipc_topology import PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR
from flashinfer.comm import pcie_ipc_tuning as tuning

_WORLD_SIZES = (2, 4, 8)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_candidates_are_pure_and_launchable(world_size: int) -> None:
    """Every rank must derive the same candidate list, and none may be rejected.

    The autotuner profiles candidates collectively with no timeout, so a list
    that differs by one element between ranks deadlocks on the first timing
    reduction. And a candidate the launcher rejects raises on the calling rank
    only, leaving its peers spinning.
    """
    first = tuning.candidate_tactics(world_size)
    assert first == tuning.candidate_tactics(world_size)
    assert len(first) == len(set(first)), "candidates must be distinct"
    for tactic in first:
        config = tuning.tactic_to_config(tactic)
        assert _is_launchable(world_size, config, MAX_BLOCKS), tactic


def test_candidate_rejections_match_the_documented_rules() -> None:
    """The only configurations excluded are the ones the header cannot dispatch."""
    grid = [
        (int(v), b, t)
        for v in IpcVariant
        for b in tuning.TUNE_BLOCKS
        for t in tuning.TUNE_THREADS
    ]

    # World size 8: the block-partitioned kernel needs blocks % 4 == 0.
    rejected = set(grid) - set(tuning.candidate_tactics(8))
    assert rejected == {
        (int(IpcVariant.STAGED), b, t)
        for b in tuning.TUNE_BLOCKS
        if b % 4 != 0
        for t in tuning.TUNE_THREADS
    }

    # World size 4: no FLAT_STAGED, and threads must be at least world_size
    # (which every entry in the grid already satisfies).
    rejected4 = set(grid) - set(tuning.candidate_tactics(4))
    assert all(t[0] == int(IpcVariant.FLAT_STAGED) for t in rejected4)

    # World size 2: only the two variants that name a TP2 kernel.
    for tactic in tuning.candidate_tactics(2):
        assert tactic[0] in (int(IpcVariant.UNSTAGED), int(IpcVariant.STAGED))


def test_the_grid_can_express_the_shipping_table() -> None:
    """Tuning must be able to reach the table, not merely fall back to it.

    Tactic -1 always reproduces the table, but if the grid cannot name the
    table's own configuration then the tuner can never explore its
    neighbourhood -- it can only accept or reject it wholesale. Both tables use
    values that are not powers of two, so a powers-of-two grid would silently
    have this property.
    """
    missing = []
    for profile in (PROFILE_ROOTCPLX, PROFILE_SWITCHPAIR):
        for world_size, hidden in ((2, 2048), (4, 4096), (8, 6144)):
            for batch in range(1, 129):
                config = get_pcie_ipc_launch_config(profile, world_size, hidden, batch)
                if config is None:
                    continue
                tactic = tuning.config_to_tactic(config)
                if tactic not in tuning.candidate_tactics(world_size):
                    missing.append((profile, world_size, batch, config))
    assert not missing, f"grid cannot express these table entries: {missing[:5]}"


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_tactics_survive_the_persistence_round_trip(world_size: int) -> None:
    """A tactic that cannot be serialised fails at save time, after the whole run.

    ``_tactic_to_json`` passes anything that is not a scalar or an iterable
    through unchanged, so a dataclass reaches ``json.dump`` and raises there --
    at the end of tuning, with every measurement already discarded.
    """
    for tactic in tuning.candidate_tactics(world_size):
        encoded = json.loads(json.dumps(_tactic_to_json(tactic)))
        assert _json_to_tactic(encoded) == tactic
    assert _json_to_tactic(json.loads(json.dumps(_tactic_to_json(-1)))) == -1


def test_tactic_codec_round_trips_and_rejects_nonsense() -> None:
    for world_size in _WORLD_SIZES:
        for tactic in tuning.candidate_tactics(world_size):
            assert tuning.config_to_tactic(tuning.tactic_to_config(tactic)) == tactic
    with pytest.raises(ValueError):
        tuning.tactic_to_config((0, 1))
    with pytest.raises(ValueError):
        tuning.tactic_to_config((99, 1, 128))


def test_resolve_falls_back_to_the_table() -> None:
    """Every way a tactic can be unusable ends at the table, not at an exception.

    The autotuner does not check that a cached tactic can implement the shape
    it is reused for, so a stale entry has to be caught here. Raising instead
    would take down one rank mid-collective and leave the rest spinning.
    """
    table = IpcLaunchConfig(32, 128, IpcVariant.UNSTAGED)
    resolve = tuning.resolve_tuned_config

    assert resolve(table, tuning.TABLE_TACTIC, 8, MAX_BLOCKS) is table
    assert resolve(table, None, 8, MAX_BLOCKS) is table
    # Malformed.
    assert resolve(table, (1, 2), 8, MAX_BLOCKS) is table
    assert resolve(table, "nonsense", 8, MAX_BLOCKS) is table
    assert resolve(table, (99, 1, 128), 8, MAX_BLOCKS) is table
    # Stale: tuned against a larger workspace than this one was built with.
    assert resolve(table, (0, 128, 128), 8, 32) is table
    # Stale: a variant this world size does not dispatch.
    assert resolve(table, (int(IpcVariant.FLAT_STAGED), 1, 128), 4, MAX_BLOCKS) is table
    # Usable.
    assert resolve(table, (int(IpcVariant.STAGED_RING), 2, 256), 8, MAX_BLOCKS) == (
        IpcLaunchConfig(2, 256, IpcVariant.STAGED_RING)
    )


def test_cache_key_extras_separate_every_workspace_dimension() -> None:
    """The autotuner's own key is only the bucketed shapes.

    Without these, a TP4 and a TP8 entry at the same shape share one slot, a
    configuration measured on one fabric is reused on the other, and a cache
    written for one workspace size is applied to another.
    """
    base = dict(
        world_size=8,
        profile=PROFILE_ROOTCPLX,
        max_blocks=128,
        max_numel=6144 * 128,
        dtype=torch.bfloat16,
    )
    reference = tuning.cache_key_extras(**base)
    assert reference == tuning.cache_key_extras(**base), "must be deterministic"
    assert isinstance(hash(reference), int), "extras must be hashable"

    for field, other in (
        ("world_size", 4),
        ("profile", PROFILE_SWITCHPAIR),
        ("max_blocks", 32),
        ("max_numel", 6144 * 64),
        ("dtype", torch.float16),
    ):
        assert tuning.cache_key_extras(**{**base, field: other}) != reference, field


def test_cache_key_extras_are_synthesis_invariant() -> None:
    """The tuner keys on synthesized tensors and looks up with real ones.

    Anything derived from tensor *content* would make those two disagree and
    every lookup would miss.
    """
    real = torch.zeros(4, 6144, dtype=torch.bfloat16)
    synthetic = tuning.small_int_initializer(
        (4, 6144), torch.bfloat16, torch.device("cpu")
    )
    common = dict(
        world_size=8, profile=PROFILE_ROOTCPLX, max_blocks=128, max_numel=6144 * 128
    )
    assert tuning.cache_key_extras(dtype=real.dtype, **common) == (
        tuning.cache_key_extras(dtype=synthetic.dtype, **common)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_synthesized_inputs_reduce_exactly(dtype: torch.dtype) -> None:
    """Zero-tolerance verification is only valid if the group sum is exact.

    The kernels sum in a different order than NCCL, so an inexact sum would
    show up as a mismatch and the gate would reject every candidate.
    """
    max_sum = (tuning.INIT_MAX_VALUE - 1) * max(_WORLD_SIZES)
    assert max_sum <= 256, "must stay inside bfloat16's exact-integer range"

    values = torch.arange(0, tuning.INIT_MAX_VALUE, dtype=torch.int32).to(dtype)
    for world_size in _WORLD_SIZES:
        summed = (values.float() * world_size).to(dtype)
        assert torch.equal(summed.float(), values.float() * world_size)

    synthetic = tuning.small_int_initializer((8, 64), dtype, torch.device("cpu"))
    assert synthetic.dtype is dtype
    assert int(synthetic.float().min()) >= 0
    assert int(synthetic.float().max()) < tuning.INIT_MAX_VALUE
    assert (synthetic.float() == synthetic.float().round()).all()


def test_verdict_reduces_with_max_over_wrong() -> None:
    """The verdict must be a group decision, and MAX over "was wrong" is it.

    Corruption in this protocol is not uniform: a rank can be clean while its
    peers are wrong. A rank-local verdict, or a SUM, lets a clean rank keep a
    candidate its peers rejected -- and the ranks then profile different
    candidate sets, which deadlocks the autotuner's timing reduction on the
    first divergence.
    """
    calls = []
    real_dist = tuning.dist
    try:
        tuning.dist = type(
            "D",
            (),
            {
                "all_reduce": staticmethod(
                    lambda tensor, op=None, group=None: calls.append(op)
                ),
                "ReduceOp": real_dist.ReduceOp,
            },
        )
        tuning.reduce_verdict(torch.zeros(3, dtype=torch.int32), None)
    finally:
        tuning.dist = real_dist

    assert calls, "the verdict must be reduced, not decided locally"
    assert all(op is real_dist.ReduceOp.MAX for op in calls), (
        f"a wrong-flag verdict must reduce with MAX (logical OR), got {calls}"
    )


def test_tuning_config_is_shared_and_its_mapper_is_stable() -> None:
    """The lookup and the search must map a shape through the identical mapper.

    ``_find_nearest_profile`` is memoised on the spec, so a freshly built
    mapper on every call would both miss the cache and grow it without bound.
    """
    assert tuning.pcie_ipc_tuning_config() is tuning.pcie_ipc_tuning_config()
    buckets = (1, 2, 4)
    assert make_bucket_mapper(buckets, round_map=False) is (
        make_bucket_mapper(buckets, round_map=False)
    )
    config = tuning.pcie_ipc_tuning_config()
    (spec,) = config.dynamic_tensor_specs
    assert spec.input_idx == (0,) and spec.dim_idx == (0,), (
        "only the batch dimension may bucket; hidden must stay exact in the key"
    )
    assert config.constraint_specs == (), (
        "a constraint dim is stored as -1, which would erase hidden from the key"
    )


def test_batch_buckets_never_exceed_the_batch_they_stand_for() -> None:
    """Floor semantics: a bucket is always a batch that was actually measured.

    Rounding up would apply a configuration measured at 64 to a batch of 33.
    """
    mapper = make_bucket_mapper(tuning.TUNE_BATCHES, round_map=False)
    for batch in range(1, 257):
        bucket = mapper(batch)
        assert bucket in tuning.TUNE_BATCHES
        assert bucket <= batch or batch < min(tuning.TUNE_BATCHES)


def test_workspace_capacity_filters_buckets() -> None:
    """A bucket larger than the workspace would raise inside the collective."""
    assert tuning.tuned_batches_for(6144, tuning.TUNE_BATCHES, 6144 * 8) == (
        1,
        2,
        4,
        8,
    )
    assert tuning.tuned_batches_for(6144, tuning.TUNE_BATCHES, 6144 * 128) == (
        tuning.TUNE_BATCHES
    )


def test_custom_op_name_is_stable() -> None:
    """It is baked into every persisted cache key; renaming it orphans the file."""
    assert tuning.PCIE_IPC_CUSTOM_OP == "flashinfer::pcie_ipc_all_reduce"
    assert tuning.PCIE_IPC_TUNE_VERSION == 1


def test_pack_config_is_injective_over_the_candidate_space() -> None:
    """The cross-rank agreement check compares packed configurations."""
    packed = {}
    for world_size in _WORLD_SIZES:
        for tactic in tuning.candidate_tactics(world_size):
            config = tuning.tactic_to_config(tactic)
            key = tuning.pack_config(config)
            assert packed.setdefault(key, config) == config
