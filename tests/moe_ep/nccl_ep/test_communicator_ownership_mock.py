"""Check ownership and teardown order for mirrored and adopted communicators."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetAlgoKnobAllocator,
    FleetAlgoKnobNcclEpConfig,
    FleetParams,
)
from flashinfer.moe_ep.backends.split.comm.nccl_ep import fleet as module


def make_fleet(pointer=None, knobs=()):
    return module.NcclEpFleet(
        BootstrapConfig(rank=0, world_size=4, nccl_comm=pointer),
        FleetParams(num_experts=8, max_tokens_per_rank=4, token_hidden_size=2048),
        knobs,
    )


@pytest.mark.parametrize("adopted", [False, True])
def test_destroy_respects_comm_owner(
    fake_nccl_ep, bypass_build_checks, monkeypatch, adopted
):
    events = []
    comm = SimpleNamespace(destroy=Mock(side_effect=lambda: events.append("comm")))
    monkeypatch.setattr(module, "_resolve_comm", lambda bootstrap: comm)
    fleet = make_fleet(123 if adopted else None)
    monkeypatch.setattr(fleet._group, "destroy", lambda: events.append("group"))
    window = SimpleNamespace(close=lambda: events.append("window"))
    fleet._dispatch_windows["tokens"] = (None, None, window, None)
    fleet.destroy()
    fleet.destroy()
    assert events == (["group", "window"] if adopted else ["group", "window", "comm"])


def test_topology_replacement_releases_only_owned_comms(
    fake_nccl_ep, bypass_build_checks, monkeypatch
):
    first, adopted, last = [SimpleNamespace(destroy=Mock()) for _ in range(3)]
    values = iter([first, adopted, last])
    monkeypatch.setattr(module, "_resolve_comm", lambda bootstrap: next(values))
    fleet = make_fleet()
    fleet.update_topology(BootstrapConfig(rank=0, world_size=4, nccl_comm=123))
    first.destroy.assert_called_once()
    fleet.update_topology(BootstrapConfig(rank=0, world_size=4))
    adopted.destroy.assert_not_called()
    fleet.destroy()
    last.destroy.assert_called_once()


@pytest.mark.parametrize("adopted", [False, True])
def test_failed_group_creation_releases_owned_comm(
    fake_nccl_ep, bypass_build_checks, monkeypatch, adopted
):
    comm = SimpleNamespace(destroy=Mock())
    monkeypatch.setattr(module, "_resolve_comm", lambda bootstrap: comm)

    def fail(*args, **kwargs):
        raise RuntimeError("group creation failed")

    monkeypatch.setattr(fake_nccl_ep.Group, "create", fail)
    with pytest.raises(RuntimeError, match="group creation"):
        make_fleet(123 if adopted else None)
    assert comm.destroy.call_count == (0 if adopted else 1)


@pytest.mark.parametrize("adopted", [False, True])
@pytest.mark.parametrize("failure", ["options", "provider_config"])
def test_invalid_replacement_preserves_live_group(
    fake_nccl_ep, bypass_build_checks, monkeypatch, adopted, failure
):
    allocator = FleetAlgoKnobAllocator(torch_caching=True)
    knobs = [allocator, FleetAlgoKnobNcclEpConfig(num_topk=2)]
    fleet = make_fleet(123 if adopted else None, knobs)
    group, bootstrap, comm = fleet.group, fleet.bootstrap, fleet._comm
    callbacks, old_knobs = fleet._alloc_trampolines, fleet._fleet_knobs
    group_destroy = Mock()
    monkeypatch.setattr(group, "destroy", group_destroy)
    fleet._hot_cache["live"] = object()
    hot = dict(fleet._hot_cache)
    config_type = fake_nccl_ep.GroupConfig

    def checked_config(**kwargs):
        if kwargs.get("num_topk") == 3:
            raise ValueError("provider rejected candidate configuration")
        return config_type(**kwargs)

    monkeypatch.setattr(fake_nccl_ep, "GroupConfig", checked_config)
    invalid = FleetAlgoKnobNcclEpConfig(num_topk=9 if failure == "options" else 3)
    reason = (
        "num_topk"
        if failure == "options"
        else "provider rejected candidate configuration"
    )
    with pytest.raises(ValueError, match=reason):
        fleet.update_topology(
            BootstrapConfig(rank=0, world_size=2), [allocator, invalid]
        )
    group_destroy.assert_not_called()
    comm.destroy.assert_not_called()
    assert fleet.group is group and fleet.bootstrap is bootstrap
    assert fleet._fleet_knobs is old_knobs and fleet._hot_cache == hot
    assert fleet._alloc_trampolines is callbacks
    assert not fleet._destroyed and fleet._owns_comm is not adopted

    fleet.update_topology(BootstrapConfig(rank=0, world_size=4), knobs)
    group_destroy.assert_called_once()
    assert fleet.group is not group and not fleet._hot_cache
    assert fleet._alloc_trampolines is callbacks
    assert fleet.group.config.alloc.alloc_fn == group.config.alloc.alloc_fn
    fleet.destroy()
