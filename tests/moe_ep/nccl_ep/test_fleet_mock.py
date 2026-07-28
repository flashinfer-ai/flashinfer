"""Host-only unit tests for NcclEpFleet (fake ``nccl.ep``, no GPU comm).

Uses the shared ``fake_nccl_ep`` / ``bypass_build_checks`` fixtures from
``conftest.py`` (recording stand-ins for ``Group`` / ``Tensor`` / the config
dataclasses / the enums, plus fakes for ``nccl.core.Communicator`` and
``nccl.ep.interop.torch.get_nccl_comm_from_group``).
"""

from __future__ import annotations

import logging

import pytest


def _fleet_params(**overrides):
    from flashinfer.moe_ep.config import FleetParams

    kwargs = dict(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
    )
    kwargs.update(overrides)
    return FleetParams(**kwargs)


def test_fleet_builds_group_config(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    params = _fleet_params(algorithm=EpAlgorithm.LOW_LATENCY)
    bootstrap = BootstrapConfig(world_size=4, rank=0)

    fleet = NcclEpFleet(bootstrap, params)

    assert len(fake_nccl_ep._log["groups"]) == 1
    cfg = fake_nccl_ep._log["groups"][0].config
    assert cfg.algorithm == fake_nccl_ep.Algorithm.LOW_LATENCY
    assert cfg.num_experts == 8
    assert cfg.max_dispatch_tokens_per_rank == 128
    assert cfg.max_token_bytes == 7168 * 2
    assert fleet.group is fake_nccl_ep._log["groups"][0]


def test_handle_create_uses_expert_major_and_int64_topk(
    fake_nccl_ep, bypass_build_checks
):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _fleet_params())

    topk_ids = torch.zeros(16, 2, dtype=torch.int32)
    weights = torch.ones(16, 2, dtype=torch.float32)
    fleet.create_handle(
        HandleParams(topk_ids=topk_ids),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )

    fake_handle = fake_nccl_ep._log["handles"][-1]
    assert fake_handle.layout == fake_nccl_ep.Layout.EXPERT_MAJOR
    assert fake_handle.topk_idx.buffer.dtype == torch.int64


def test_handle_create_uses_rank_major_layout(fake_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        HandleParams,
    )
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    params = _fleet_params(
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.RANK_MAJOR,
    )
    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), params)

    topk_ids = torch.zeros(16, 2, dtype=torch.int32)
    weights = torch.ones(16, 2, dtype=torch.float32)
    fleet.create_handle(
        HandleParams(topk_ids=topk_ids),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )

    fake_handle = fake_nccl_ep._log["handles"][-1]
    assert fake_handle.layout == fake_nccl_ep.Layout.RANK_MAJOR
    assert fake_handle.topk_idx.buffer.dtype == torch.int64


def test_fleet_params_rejects_rank_major_under_ht():
    from flashinfer.moe_ep.config import EpAlgorithm, EpLayout

    with pytest.raises(ValueError):
        _fleet_params(
            algorithm=EpAlgorithm.HIGH_THROUGHPUT,
            layout=EpLayout.RANK_MAJOR,
        )


# --------------------------------------------------------- HT clamp (8192 cap)


def test_clamp_ht_max_tokens_is_noop_for_ll_and_within_cap():
    from flashinfer.moe_ep.config import EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import (
        _clamp_ht_max_tokens,
    )

    ll = _fleet_params(max_tokens_per_rank=16384, algorithm=EpAlgorithm.LOW_LATENCY)
    assert _clamp_ht_max_tokens(ll) is ll

    ht_small = _fleet_params(
        max_tokens_per_rank=8192, algorithm=EpAlgorithm.HIGH_THROUGHPUT
    )
    assert _clamp_ht_max_tokens(ht_small) is ht_small


def test_clamp_ht_max_tokens_clamps_and_warns(caplog):
    from flashinfer.moe_ep.config import EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import (
        _HT_MAX_SUPPORTED_TOKENS_PER_RANK,
        _clamp_ht_max_tokens,
    )

    ht = _fleet_params(max_tokens_per_rank=16384, algorithm=EpAlgorithm.HIGH_THROUGHPUT)
    with caplog.at_level(
        logging.WARNING, logger="flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet"
    ):
        clamped = _clamp_ht_max_tokens(ht)
    assert clamped.max_tokens_per_rank == _HT_MAX_SUPPORTED_TOKENS_PER_RANK
    assert clamped is not ht
    assert any("clamping" in r.getMessage() for r in caplog.records)


def test_fleet_clamps_ht_params_and_group_config_agree(
    fake_nccl_ep, bypass_build_checks
):
    """The stored params AND the GroupConfig must both see the clamped value —
    the handle sizes its recv buffers from ``fleet.params``, so a mismatch
    would desynchronize buffer sizes from the transport budget."""
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    params = _fleet_params(
        max_tokens_per_rank=16384, algorithm=EpAlgorithm.HIGH_THROUGHPUT
    )
    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), params)

    assert fleet.params.max_tokens_per_rank == 8192
    cfg = fake_nccl_ep._log["groups"][0].config
    assert cfg.max_dispatch_tokens_per_rank == 8192
    assert cfg.max_recv_tokens_per_rank == 8192 * 4


# --------------------------------------------------------------- alloc config


def test_group_config_has_no_alloc_without_knob(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _fleet_params())

    cfg = fake_nccl_ep._log["groups"][0].config
    assert "alloc" not in cfg.kwargs


def test_allocator_knob_explicit_addresses(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.algo_knobs import FleetAlgoKnobAllocator
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    knob = FleetAlgoKnobAllocator(alloc_fn=0x1234, free_fn=0x5678, context=0x9ABC)
    NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _fleet_params(), [knob])

    alloc = fake_nccl_ep._log["groups"][0].config.alloc
    assert isinstance(alloc, fake_nccl_ep.AllocConfig)
    assert alloc.alloc_fn == 0x1234
    assert alloc.free_fn == 0x5678
    assert alloc.context == 0x9ABC


def test_allocator_knob_torch_caching_installs_trampolines(
    fake_nccl_ep, bypass_build_checks
):
    from flashinfer.moe_ep.algo_knobs import FleetAlgoKnobAllocator
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    knob = FleetAlgoKnobAllocator(torch_caching=True)
    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _fleet_params(), [knob])

    alloc = fake_nccl_ep._log["groups"][0].config.alloc
    assert isinstance(alloc, fake_nccl_ep.AllocConfig)
    assert alloc.alloc_fn and alloc.free_fn  # real C-callable addresses
    # The keepalive anchor is load-bearing: NCCL-EP holds the raw pointers, so
    # GC'ing the trampolines while the Group lives is a C-side use-after-free.
    assert fleet._alloc_trampolines is not None
    assert len(fleet._alloc_trampolines) == 2


# --------------------------------------------------------------- _resolve_comm


def test_resolve_comm_adopts_existing_nccl_comm(fake_nccl_ep):
    """nccl_comm set → wrap-without-own via Communicator(ptr=...), and no
    fresh communicator is bootstrapped."""
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import _resolve_comm

    comm = _resolve_comm(BootstrapConfig(world_size=4, rank=0, nccl_comm=0xDEAD))

    assert isinstance(comm, fake_nccl_ep._core.Communicator)
    assert comm.ptr == 0xDEAD
    assert fake_nccl_ep._log["comm_from_group"] == []


def test_resolve_comm_mirrors_bootstrap_process_group(fake_nccl_ep):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import _resolve_comm

    sentinel = object()
    _resolve_comm(
        BootstrapConfig(world_size=4, rank=0, process_group=sentinel)  # type: ignore[arg-type]
    )

    assert fake_nccl_ep._log["comm_from_group"] == [sentinel]
    assert fake_nccl_ep._core.Communicator.instances == []
