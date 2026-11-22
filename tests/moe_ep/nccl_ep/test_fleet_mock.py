"""Host-only unit tests for NcclEpFleet (fake ``nccl.ep``, no GPU comm).

Uses the shared ``fake_nccl_ep`` / ``bypass_build_checks`` fixtures from
``conftest.py`` (recording stand-ins for ``Group`` / ``Tensor`` / the config
dataclasses / the enums, plus fakes for ``nccl.core.Communicator`` and
``nccl.ep.interop.torch.get_nccl_comm_from_group``).
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def modern_nccl_ep(fake_nccl_ep):
    import enum

    from packaging.version import Version

    fake_nccl_ep.get_lib_version = lambda: Version("0.3.0")
    fake_nccl_ep.HtFlow = enum.IntEnum(
        "HtFlow", {"AUTO": 0, "PUSH_PULL": 1, "PULL_PUSH": 2}
    )
    fake_nccl_ep.MetaExchangePolicy = enum.IntEnum(
        "MetaExchangePolicy", {"AUTO": 0, "REQUIRE_FUSED": 1, "FORCE_SCAN": 2}
    )
    return fake_nccl_ep


@pytest.mark.parametrize(
    "modern,algorithm,policy,message",
    [
        (False, "HIGH_THROUGHPUT", None, "EP 0.3"),
        (True, "LOW_LATENCY", None, "HIGH_THROUGHPUT"),
        (True, "HIGH_THROUGHPUT", "require_fused", "early received-token"),
    ],
)
def test_eager_ht_admission(
    fake_nccl_ep, bypass_build_checks, modern, algorithm, policy, message
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetAlgoKnobNcclEpConfig,
    )
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    if modern:
        fake_nccl_ep.get_lib_version = lambda: "0.3.0"
    with pytest.raises(ValueError, match=message):
        NcclEpFleet(
            BootstrapConfig(world_size=4, rank=0),
            _fleet_params(algorithm=EpAlgorithm[algorithm]),
            [FleetAlgoKnobNcclEpConfig(eager_recv=True, meta_exchange_policy=policy)],
        )
    assert fake_nccl_ep._log["groups"] == []


def test_eager_ht_requires_boolean():
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig

    with pytest.raises(ValueError, match="eager_recv must be a bool"):
        FleetAlgoKnobNcclEpConfig(eager_recv=1)


@pytest.mark.parametrize("hidden", [3072, 3584])
def test_release_ll_hidden_sizes(modern_nccl_ep, bypass_build_checks, hidden):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0), _fleet_params(token_hidden_size=hidden)
    )
    assert modern_nccl_ep._log["groups"][-1].config.max_token_bytes == hidden * 2


@pytest.mark.parametrize("hidden", [3072, 3584])
def test_legacy_ll_hidden_sizes_remain_guarded(
    fake_nccl_ep, bypass_build_checks, hidden
):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    with pytest.raises(ValueError, match="admits the LL sizes"):
        NcclEpFleet(
            BootstrapConfig(world_size=4, rank=0),
            _fleet_params(token_hidden_size=hidden),
        )
    assert fake_nccl_ep._log["groups"] == []


def test_release_ht_controls_preserve_provider_flow(
    modern_nccl_ep, bypass_build_checks
):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(algorithm=EpAlgorithm.HIGH_THROUGHPUT),
        [
            FleetAlgoKnobNcclEpConfig(
                num_topk=4,
                max_num_sms=8,
                dispatch_num_sms=4,
                combine_num_sms=6,
                meta_exchange_policy="force_scan",
            )
        ],
    )
    cfg = fleet.group.config
    assert (
        cfg.num_topk,
        cfg.max_num_sms,
        cfg.dispatch_num_sms,
        cfg.combine_num_sms,
    ) == (4, 8, 4, 6)
    assert "ht_flow" not in cfg.kwargs
    assert cfg.meta_exchange_policy == modern_nccl_ep.MetaExchangePolicy.FORCE_SCAN


@pytest.mark.parametrize("flow", ["auto", "push_pull", "pull_push"])
def test_release_ht_flow_selection(modern_nccl_ep, bypass_build_checks, flow):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(algorithm=EpAlgorithm.HIGH_THROUGHPUT),
        [FleetAlgoKnobNcclEpConfig(num_topk=4, ht_flow=flow)],
    )
    assert fleet.group.config.ht_flow == getattr(modern_nccl_ep.HtFlow, flow.upper())


def test_ll_rejects_ht_flow(modern_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    with pytest.raises(ValueError, match="only to HIGH_THROUGHPUT"):
        NcclEpFleet(
            BootstrapConfig(world_size=4, rank=0),
            _fleet_params(),
            [FleetAlgoKnobNcclEpConfig(ht_flow="pull_push")],
        )
    assert modern_nccl_ep._log["groups"] == []


def test_ht_flow_rejects_unknown_value():
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig

    with pytest.raises(ValueError, match="ht_flow must"):
        FleetAlgoKnobNcclEpConfig(ht_flow="unknown")


@pytest.mark.parametrize(
    "options",
    [
        {"dispatch_num_sms": 4},
        {"combine_num_sms": 4},
        {"meta_exchange_policy": "auto"},
        {"ht_flow": "pull_push"},
    ],
)
def test_new_controls_reject_legacy_provider(
    fake_nccl_ep, bypass_build_checks, options
):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    with pytest.raises(ValueError, match="requires nccl-extensions 0.2"):
        NcclEpFleet(
            BootstrapConfig(world_size=4, rank=0),
            _fleet_params(algorithm=EpAlgorithm.HIGH_THROUGHPUT),
            [FleetAlgoKnobNcclEpConfig(**options)],
        )
    assert fake_nccl_ep._log["groups"] == []


def test_legacy_shared_controls(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(),
        [FleetAlgoKnobNcclEpConfig(num_topk=2, max_num_sms=4)],
    )
    assert fleet.group.config.num_topk == 2
    assert fleet.group.config.max_num_sms == 4
    assert "ht_flow" not in fleet.group.config.kwargs


def test_fused_metadata_counter_contract(modern_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep import (
        FleetAlgoKnobNcclEpConfig,
        HandleAlgoKnobNumReceivedTokens,
    )
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm, HandleParams
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(algorithm=EpAlgorithm.HIGH_THROUGHPUT),
        [FleetAlgoKnobNcclEpConfig(num_topk=2, meta_exchange_policy="require_fused")],
    )
    with pytest.raises(ValueError, match="received-token counters"):
        fleet.create_handle(
            HandleParams(topk_ids=torch.zeros(16, 2, dtype=torch.int64)),
            [HandleAlgoKnobNumReceivedTokens(torch.zeros(1, dtype=torch.int32))],
        )
    assert not fleet._any_handle_created
    assert modern_nccl_ep._log["handles"] == []


def test_group_topk_bound_checked_before_handle(modern_nccl_ep, bypass_build_checks):
    import torch

    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(),
        [FleetAlgoKnobNcclEpConfig(num_topk=1)],
    )
    with pytest.raises(ValueError, match="handle top-k exceeds"):
        fleet.create_handle(
            HandleParams(topk_ids=torch.zeros(16, 2, dtype=torch.int64))
        )
    assert not fleet._any_handle_created


def test_mask_update_keeps_host_input_unchanged_until_copy_finishes(
    fake_nccl_ep, bypass_build_checks, recording_ffi, monkeypatch
):
    import torch

    fleet = _ft_fleet()
    host = torch.empty(4, dtype=torch.int32)
    monkeypatch.setattr(fleet, "_ft_bufs", lambda: (None, host))
    trace = []

    class PendingCopy:
        def record(self, stream):
            assert stream == 0
            self.expected = host.clone()
            trace.append("copy")

        def synchronize(self):
            assert torch.equal(host, self.expected), "host mask changed during its copy"
            trace.append("wait")

    monkeypatch.setattr(torch.cuda, "Event", PendingCopy)
    monkeypatch.setattr(
        "flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet.torch_stream",
        lambda stream: stream,
    )
    fleet.set_active_mask([1, 0, 1, 1])
    fleet.set_active_mask([1, 1, 0, 1])
    assert trace == ["copy", "wait", "copy"]
    fleet.destroy()
    assert trace == ["copy", "wait", "copy", "wait"]


def test_readmission_requires_a_handle_in_the_replacement_group(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    import torch

    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams

    fleet = _ft_fleet()
    handle = fleet.create_handle(
        HandleParams(topk_ids=torch.arange(8, dtype=torch.int64).reshape(4, 2))
    )
    handle.destroy()
    fleet.update_topology(BootstrapConfig(world_size=4, rank=0))
    with pytest.raises(RuntimeError, match="requires at least one handle"):
        fleet.clear_faults(readmit=True)
    assert recording_ffi.calls == []


@pytest.mark.parametrize(
    "field", ["num_topk", "max_num_sms", "dispatch_num_sms", "combine_num_sms"]
)
@pytest.mark.parametrize("value", [-1, True])
def test_nccl_group_control_validation(field, value):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig

    with pytest.raises(ValueError, match="nonnegative integer"):
        FleetAlgoKnobNcclEpConfig(**{field: value})


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


# ------------------------------------------------------------ fault tolerance


class _RecordingFfi:
    """Stand-in for the ctypes shim; records the exact calls the Fleet makes."""

    available = True
    missing: tuple = ()

    def __init__(self):
        self.calls: list = []

    def mask_query(self, group, dev_ptr, stream):
        self.calls.append(("query", dev_ptr, stream))

    def mask_update(self, group, host_ptr, stream):
        self.calls.append(("update", host_ptr, stream))

    def mask_clean(self, group, stream):
        self.calls.append(("clean", stream))

    def get_async_error(self, group):
        self.calls.append(("get_async_error",))
        return True

    def error_clear(self, group):
        self.calls.append(("error_clear",))


@pytest.fixture
def recording_ffi(monkeypatch):
    ffi = _RecordingFfi()
    monkeypatch.setattr(
        "flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet.mask_ffi", lambda: ffi
    )
    return ffi


def _ft_fleet(**knob_kwargs):
    from flashinfer.moe_ep.algo_knobs import FleetAlgoKnobFaultTolerance
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    return NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _fleet_params(),
        [FleetAlgoKnobFaultTolerance(**knob_kwargs)],
    )


def test_ft_knob_sets_enable_mask_and_timeout(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    _ft_fleet(timeout_ms=5000)
    cfg = fake_nccl_ep._log["groups"][0].config
    assert cfg.enable_mask is True
    assert cfg.timeout_ns == 5000 * 1_000_000  # ms -> ns


def test_no_ft_knob_leaves_mask_fields_unset(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _fleet_params())
    cfg = fake_nccl_ep._log["groups"][0].config
    assert "enable_mask" not in cfg.kwargs
    assert "timeout_ns" not in cfg.kwargs
    assert fleet.supports_fault_tolerance is False


def test_zero_timeout_leaves_transport_default(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    _ft_fleet(timeout_ms=0)
    cfg = fake_nccl_ep._log["groups"][0].config
    assert cfg.enable_mask is True
    assert "timeout_ns" not in cfg.kwargs  # 0 = library default (~100 s)


def test_disabled_knob_is_inert(fake_nccl_ep, bypass_build_checks):
    fleet = _ft_fleet(enabled=False)
    cfg = fake_nccl_ep._log["groups"][0].config
    assert "enable_mask" not in cfg.kwargs
    assert fleet.supports_fault_tolerance is False


def test_unavailable_ffi_fails_at_construction(
    fake_nccl_ep, bypass_build_checks, monkeypatch
):
    """Better to fail when the Fleet is built than at the first real fault."""
    from flashinfer.moe_ep.errors import MoEEpFaultToleranceUnsupportedError

    class _Unavailable:
        available = False
        missing = ("ncclEpMaskQuery",)

    monkeypatch.setattr(
        "flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet.mask_ffi",
        lambda: _Unavailable(),
    )
    with pytest.raises(MoEEpFaultToleranceUnsupportedError, match="ncclEpMaskQuery"):
        _ft_fleet()


def test_query_uses_device_buffer(fake_nccl_ep, bypass_build_checks, recording_ffi):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the staging buffers")
    fleet = _ft_fleet()
    out = fleet.query_active_mask()
    assert out.dtype == torch.int32 and out.numel() == 4 and out.is_cuda
    kind, ptr, _ = recording_ffi.calls[-1]
    assert kind == "query" and ptr == out.data_ptr()


def test_update_uses_pinned_host_buffer_and_reuses_it(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    """ncclEpMaskUpdate is stream-ordered, so the host source must be pinned
    and Fleet-owned; a fresh pageable buffer each call would be a
    use-after-write race."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the staging buffers")
    fleet = _ft_fleet()
    fleet.set_active_mask([1, 1, 0, 1])
    fleet.set_active_mask([1, 0, 0, 1])
    ptrs = [c[1] for c in recording_ffi.calls if c[0] == "update"]
    assert len(ptrs) == 2 and ptrs[0] == ptrs[1]  # same buffer reused
    _, host = fleet._ft_bufs()
    assert host.is_pinned() and not host.is_cuda
    assert host.tolist() == [1, 0, 0, 1]
    assert fleet.active_mask_epoch == 2


def test_set_active_mask_rejects_masking_self(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the staging buffers")
    fleet = _ft_fleet()
    with pytest.raises(ValueError, match="cannot mask itself"):
        fleet.set_active_mask([0, 1, 1, 1])  # rank 0 masking rank 0


def test_query_fault_reads_host_flag(fake_nccl_ep, bypass_build_checks, recording_ffi):
    assert _ft_fleet().query_fault() is True
    assert ("get_async_error",) in recording_ffi.calls


def test_clear_faults_without_readmit_only_clears_the_flag(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    fleet = _ft_fleet()
    fleet.clear_faults()
    assert recording_ffi.calls == [("error_clear",)]


def test_clear_faults_readmit_requires_a_handle(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    """ncclEpMaskClean asserts on the LL buffer the first handle allocates;
    without this guard the process would SIGABRT from C."""
    fleet = _ft_fleet()
    with pytest.raises(RuntimeError, match="at least one handle"):
        fleet.clear_faults(readmit=True)


def test_clear_faults_readmit_cleans_then_clears(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    """Order matters: MaskClean does NOT clear the async flag."""
    import torch

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import EpLayout, HandleParams

    fleet = _ft_fleet()
    fleet.create_handle(
        HandleParams(topk_ids=torch.zeros(4, 2, dtype=torch.int32)),
        [HandleAlgoKnobTopKWeights(weights=torch.ones(4, 2))],
    )
    assert fleet.params.layout is EpLayout.EXPERT_MAJOR
    with pytest.warns(RuntimeWarning, match="RANK_MAJOR"):
        fleet.clear_faults(readmit=True)
    kinds = [c[0] for c in recording_ffi.calls]
    assert kinds[-2:] == ["clean", "error_clear"]


def test_ft_buffers_resize_on_update_topology(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the staging buffers")
    from flashinfer.moe_ep.config import BootstrapConfig

    fleet = _ft_fleet()
    assert fleet.query_active_mask().numel() == 4
    fleet.update_topology(BootstrapConfig(world_size=8, rank=0))
    assert fleet.query_active_mask().numel() == 8


def test_query_fault_rejects_graph_capture(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    """query_fault() is a HOST read, so capture cannot record it at all.

    Called inside a capture region it would return the capture-time answer and
    freeze the branch taken on it into the graph forever -- a graph that
    ignores faults because none had happened when it was recorded. That is
    quieter than the stream-ordered calls, hence the guard.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    fleet = _ft_fleet()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), pytest.raises(RuntimeError, match="host-side read"):
        fleet.query_fault()


def test_capture_guard_reason_is_per_operation(
    fake_nccl_ep, bypass_build_checks, recording_ffi
):
    """The rationale differs per call; a single blanket message was wrong."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    fleet = _ft_fleet()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with pytest.raises(RuntimeError, match="baked into the captured"):
            fleet.set_active_mask([1, 1, 0, 1])
        with pytest.raises(RuntimeError, match="consumed on the host"):
            fleet.query_active_mask()


def test_build_fixture_preserves_cuda_capture_detection(monkeypatch, request):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    request.getfixturevalue("bypass_build_checks")
    assert torch.cuda.is_current_stream_capturing()
