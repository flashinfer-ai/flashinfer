"""Host contracts for W4A16 preferred/fallback cluster launches."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4 import frontend
from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4.autotune import (
    _cluster_candidates,
    bf16_nvfp4_candidates,
)
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import tuner


def _config(**knobs):
    return frontend.MegaMoEBf16Nvfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=128,
        num_topk=2,
        num_total_experts=4,
        hidden=768,
        intermediate=384,
        **knobs,
    )


@pytest.mark.parametrize("fallback", [None, [2, 1, 1]])
def test_four_cta_geometry_accepts_json_and_matches_tuner(fallback):
    knobs = dict(cluster_shape_mnk=[4, 1, 1], fallback_cluster_shape_mnk=fallback)
    config = _config(**knobs)
    assert config.cluster_shape_mnk == (4, 1, 1)
    assert config.fallback_cluster_shape_mnk == (
        None if fallback is None else (2, 1, 1)
    )
    assert tuner.is_valid_bf16_nvfp4(knobs)


@pytest.mark.parametrize(
    "knobs",
    [
        dict(cluster_shape_mnk=(2, 1, 1), fallback_cluster_shape_mnk=(1, 1, 1)),
        dict(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(1, 1, 1)),
        dict(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(2, 2, 1)),
        dict(cluster_shape_mnk=(2, 2, 1)),
        dict(
            cluster_shape_mnk=(4, 1, 1),
            mma_tiler_mnk=(128, 64, 256),
            use_2cta_instrs=False,
        ),
    ],
)
def test_invalid_cluster_geometry_is_rejected_before_allocation(knobs):
    with pytest.raises(ValueError, match="W4A16"):
        _config(**knobs)
    assert not tuner.is_valid_bf16_nvfp4(knobs)


@pytest.mark.parametrize(
    "occupancy,expected",
    [
        ({4: 32, 2: 74}, (37, 32, 10)),
        ({4: 37, 2: 74}, (37, 37, 0)),
        ({4: 4, 2: 9}, (4, 4, 0)),
    ],
)
def test_launch_uses_gpc_occupancy_and_whole_logical_tiles(
    monkeypatch, occupancy, expected
):
    monkeypatch.setattr(frontend.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        frontend, "_max_active_clusters", lambda device, size: occupancy[size]
    )
    wrapper = frontend.MegaMoEBf16Nvfp4Frontend(
        _config(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(2, 1, 1))
    )
    assert wrapper._cluster_launch_configuration() == expected
    logical, preferred, fallback = expected
    assert 4 * logical == 4 * preferred + 2 * fallback
    assert 4 * logical <= 2 * occupancy[2]


def test_uniform_two_cta_launch_preserves_existing_capacity(monkeypatch):
    monkeypatch.setattr(frontend.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        frontend.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=148),
    )
    wrapper = frontend.MegaMoEBf16Nvfp4Frontend(_config())
    assert wrapper._cluster_launch_configuration() == (74, None, None)


@pytest.mark.parametrize("fallback", [None, (2, 1, 1)])
def test_unavailable_preferred_cluster_rejects_zero_grid(monkeypatch, fallback):
    monkeypatch.setattr(frontend.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        frontend, "_max_active_clusters", lambda device, size: {4: 0, 2: 4}[size]
    )
    wrapper = frontend.MegaMoEBf16Nvfp4Frontend(
        _config(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=fallback)
    )
    with pytest.raises(ValueError, match="unavailable on this device partition"):
        wrapper._cluster_launch_configuration()


def test_zero_fallback_capacity_constructs_uniform_kernel(monkeypatch):
    config = _config(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(2, 1, 1))
    wrapper = frontend.MegaMoEBf16Nvfp4Frontend(config)
    monkeypatch.setattr(wrapper, "_cluster_launch_configuration", lambda: (37, 37, 0))
    monkeypatch.setattr(frontend, "ensure_not_capturing", lambda _: None)
    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(BFloat16=object()))
    captured = {}

    class ConstructionCaptured(Exception):
        pass

    def capture_kernel(**kwargs):
        captured.update(kwargs)
        # Stop before allocation/compilation so this remains a CPU host test.
        raise ConstructionCaptured

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4.megamoe_kernel",
        SimpleNamespace(Sm100W4A16MegaMoEKernel=capture_kernel),
    )
    with pytest.raises(ConstructionCaptured):
        wrapper._ensure_compiled(None)
    assert captured["cluster_shape_mnk"] == (4, 1, 1)
    assert captured["fallback_cluster_shape_mnk"] is None
    assert captured["preferred_cluster_count"] is None
    assert captured["fallback_cluster_count"] is None
    assert captured["group_hint"] == 37
    assert wrapper.config.fallback_cluster_shape_mnk == (2, 1, 1)


def test_uniform_candidates_clear_previous_mixed_geometry():
    mixed_config = _config(
        cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(2, 1, 1)
    )
    for candidate in bf16_nvfp4_candidates():
        assert tuner.is_valid_bf16_nvfp4_for_config(mixed_config, candidate)
        if candidate["cluster_shape_mnk"] != (2, 1, 1):
            continue
        uniform_config = tuner.with_knobs(mixed_config, candidate)
        assert uniform_config.cluster_shape_mnk == (2, 1, 1)
        assert uniform_config.fallback_cluster_shape_mnk is None


@pytest.mark.parametrize(
    "enable_in_kernel,uniform_count,original_count,total_count",
    [(False, 16, 28, 34), (True, 24, 42, 51)],
)
def test_auto_catalog_preserves_uniform_and_adds_mixed_tactics(
    enable_in_kernel, uniform_count, original_count, total_count
):
    candidates = bf16_nvfp4_candidates(enable_in_kernel_fc2_reduce=enable_in_kernel)
    assert len(candidates) == total_count
    assert len({tuple(sorted(c.items())) for c in candidates}) == total_count
    uniform = candidates[:uniform_count]
    mixed = candidates[uniform_count:]
    assert all(c["cluster_shape_mnk"] == (2, 1, 1) for c in uniform)
    assert all(c["fallback_cluster_shape_mnk"] is None for c in uniform)
    assert {c["mma_tiler_mnk"] for c in uniform} == {
        (128, 32, 256),
        (256, 32, 256),
        (256, 64, 256),
        (256, 128, 256),
    }
    # Each M256 uniform tactic has one mixed counterpart and identical math,
    # scheduler, grouping, flags, return path and reduction permission.
    assert candidates[uniform_count:original_count] == [
        {
            **candidate,
            "cluster_shape_mnk": (4, 1, 1),
            "fallback_cluster_shape_mnk": (2, 1, 1),
        }
        for candidate in uniform
        if candidate["mma_tiler_mnk"][0] == 256
    ]
    assert {c["mma_tiler_mnk"] for c in mixed} == {
        (256, 32, 256),
        (256, 64, 256),
        (256, 128, 256),
    }
    # Append only the smaller mixed groups; preserve every prior entry and
    # change no math, return mode or reduction permission on the new suffix.
    assert candidates[original_count:] == [
        {
            **candidate,
            "cluster_shape_mnk": (4, 1, 1),
            "fallback_cluster_shape_mnk": (2, 1, 1),
            "group_hint": 256,
        }
        for candidate in uniform
        if candidate["mma_tiler_mnk"][0] == 256 and candidate["group_hint"] == 512
    ]
    assert {c["group_hint"] for c in uniform} == {512, 1024}
    assert {c["group_hint"] for c in mixed} == {256, 512, 1024}
    assert {c["token_back_mode"] for c in mixed} == {
        "epi_warps",
        "reuse_dispatch_warps",
    }
    assert all(c["use_2cta_instrs"] for c in mixed)
    assert all(tuner.is_valid_bf16_nvfp4(c) for c in candidates)
    reduced = [c for c in candidates if c["in_kernel_fc2_reduce"]]
    assert len(reduced) == (17 if enable_in_kernel else 0)
    assert all(c["token_back_mode"] == "reuse_dispatch_warps" for c in reduced)
    if enable_in_kernel:
        assert [c for c in candidates if not c["in_kernel_fc2_reduce"]] == (
            bf16_nvfp4_candidates()
        )


@pytest.mark.parametrize("enable_in_kernel,uniform_count", [(False, 16), (True, 24)])
def test_auto_zero_preferred_capacity_keeps_uniform_order(
    monkeypatch, enable_in_kernel, uniform_count
):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(frontend, "_max_active_clusters", lambda device, size: 0)
    candidates = bf16_nvfp4_candidates(enable_in_kernel_fc2_reduce=enable_in_kernel)
    with pytest.warns(RuntimeWarning, match="four-CTA clusters are unavailable"):
        assert _cluster_candidates(candidates, _config()) == candidates[:uniform_count]


@pytest.mark.parametrize("peer_status", [(1, 1), (1, 0), (0, 0)])
def test_auto_cluster_capability_agrees_across_ep_group(monkeypatch, peer_status):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(frontend, "_max_active_clusters", lambda device, size: 33)
    tensor = torch.tensor
    # Exercise the real reduction inputs without a CUDA allocation or process group.
    monkeypatch.setattr(
        torch, "tensor", lambda data, **kwargs: tensor(data, dtype=kwargs["dtype"])
    )
    ep_group = object()
    reductions = []

    def all_reduce(status, *, op, group):
        assert status.tolist() == [1, 1]
        assert op == dist.ReduceOp.MIN
        assert group is ep_group
        status.copy_(tensor(peer_status))
        reductions.append(True)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    candidates = bf16_nvfp4_candidates()
    if not peer_status[0]:
        with pytest.raises(RuntimeError, match="occupancy query failed on an EP rank"):
            _cluster_candidates(candidates, _config(), ep_group)
    elif not peer_status[1]:
        with pytest.warns(RuntimeWarning, match="four-CTA clusters are unavailable"):
            assert (
                _cluster_candidates(candidates, _config(), ep_group) == candidates[:16]
            )
    else:
        assert _cluster_candidates(candidates, _config(), ep_group) is candidates
    assert len(reductions) == 1


def test_auto_inherited_cluster4_is_filtered_and_uniform_needs_no_probe(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    probes = []

    def occupancy(device, size):
        probes.append((device, size))
        return 0

    monkeypatch.setattr(frontend, "_max_active_clusters", occupancy)
    mixed = _config(cluster_shape_mnk=(4, 1, 1), fallback_cluster_shape_mnk=(2, 1, 1))
    with pytest.raises(ValueError, match="No W4A16 autotune candidates remain"):
        _cluster_candidates([{}], mixed)
    assert probes == [(0, 4)]
    uniform = bf16_nvfp4_candidates()[:16]
    assert _cluster_candidates(uniform, mixed) is uniform
    assert probes == [(0, 4)]


def test_auto_cluster_query_failure_is_not_a_zero_occupancy_result(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    query_error = RuntimeError("occupancy query broke")

    def occupancy(device, size):
        raise query_error

    monkeypatch.setattr(frontend, "_max_active_clusters", occupancy)
    with pytest.raises(RuntimeError, match="occupancy query failed") as raised:
        _cluster_candidates(bf16_nvfp4_candidates(), _config())
    assert raised.value.__cause__ is query_error
