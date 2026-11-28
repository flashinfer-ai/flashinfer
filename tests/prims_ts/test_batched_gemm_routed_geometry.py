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

"""Native SFA across M fragments and fixed or flexible multi-atom CGAs."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_batched_gemm_task_manager,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    RouteImpl,
    TileScheduler,
    compute_launch_grid,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.routed_config import routed_options


CGAS = [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)]


def options(tile_m, cga, scheduler):
    return routed_options(
        collective_tile_m=tile_m,
        cluster_shape=cga,
        scheduler=scheduler,
        max_active_clusters=2,
    )


def tma_native_options(cga, warps=8, *, tile_m=512):
    from tests.prims_ts.test_batched_gemm_routed_e32 import (
        routed_e32_config,
        routed_e32_dense_config,
    )

    assert tile_m in (256, 512)
    cfg = (
        routed_e32_dense_config(RouteImpl.TMA)
        if tile_m == 512
        else routed_e32_config(RouteImpl.TMA, warps, 4)
    )
    cfg.update(
        cluster_dim_x=cga[0],
        cluster_dim_y=cga[1],
        num_load_a_warps=warps,
        num_load_sfa_warps=4,
        routed_native_sfa=1,
        load_a_regs=48,
    )
    for name in (
        "mma_regs",
        "load_regs",
        "load_sf_regs",
        "copy_sf_regs",
        "sync_regs",
        "workid_regs",
        "padding_regs",
    ):
        cfg[name] = 48
    return cfg


def flexible_tma_native_options(
    cga=(4, 4), *, tile_m=512, specialize=True, local_barrier=True, a_stages=5
):
    cfg = tma_native_options(cga, tile_m=tile_m)
    cfg.update(
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
        dense_specialize_tma_cga=int(specialize),
        dense_local_barrier=int(local_barrier),
        num_stages_a=a_stages,
    )
    return cfg


@pytest.mark.parametrize("cga", CGAS[1:])
@pytest.mark.parametrize(
    "specialize,local_barrier", [(False, False), (True, False), (True, True)]
)
def test_native_tma_flexible_host_schedule(cga, specialize, local_barrier):
    opts = flexible_tma_native_options(
        cga, specialize=specialize, local_barrier=local_barrier
    )
    cfg = make_config(**opts)
    if not specialize:
        with pytest.raises(
            ValueError, match="partitioned routed TMA A requires static CGA"
        ):
            validate_config(cfg, problem_mnk=(32768, 1280, 2304))
        return
    validate_config(cfg, problem_mnk=(32768, 1280, 2304))
    assert cfg.fallback_cluster_shape == (2, 1, 1)
    _build_batched_gemm_task_manager(verbose=False, **opts)


@pytest.mark.parametrize("tile_m", [256, 512])
def test_native_tma_flexible_host_rejects_fused_ldgsts(tile_m):
    opts = options(tile_m, (4, 4), "clc")
    opts.update(fallback_cluster_dim_x=2, fallback_cluster_dim_y=1)
    with pytest.raises(ValueError, match="routed_native_sfa"):
        validate_config(make_config(**opts), problem_mnk=(32768, 1280, 2304))


@pytest.mark.parametrize("tile_m", [256, 512])
def test_native_tma_flexible_host_rejects_static_queue(tile_m):
    opts = flexible_tma_native_options(tile_m=tile_m)
    opts.update(
        tile_scheduler=int(TileScheduler.STATIC_PERSISTENT),
        num_stages_workid=2,
        static_persistent_clusters=2,
    )
    # Static queue consumer counts still describe the preferred physical CGA.
    with pytest.raises(ValueError, match="routed_native_sfa"):
        validate_config(make_config(**opts), problem_mnk=(32768, 1280, 2304))


@pytest.mark.parametrize("cga", CGAS[1:])
@pytest.mark.parametrize(
    "specialize,local_barrier", [(False, False), (True, False), (True, True)]
)
def test_native_tma_tile256_flexible_host_schedule(cga, specialize, local_barrier):
    opts = flexible_tma_native_options(
        cga, tile_m=256, specialize=specialize, local_barrier=local_barrier
    )
    cfg = make_config(**opts)
    validate_config(cfg, problem_mnk=(16384, 1280, 2304))
    assert not cfg.uses_routed_weight_pipeline
    assert not cfg.partitions_routed_tma_a
    assert cfg.uses_ldgsts_sf_relay and not cfg.fuse_operand_sf_loads
    assert cfg.fallback_cluster_shape == (2, 1, 1)
    _build_batched_gemm_task_manager(verbose=False, **opts)


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("cga", CGAS[1:])
def test_native_tma_tile256_flexible_ring_wrap(monkeypatch, cga):
    from tests.prims_ts.test_batched_gemm_routed_e32 import check_routed_e32

    opts = flexible_tma_native_options(cga, tile_m=256)
    opts["routed_sfa_first"] = 1
    check_routed_e32(
        monkeypatch,
        opts,
        num_tokens=257,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize("specialize", [False, True])
def test_native_tma_tile256_flexible_independent_rings(monkeypatch, specialize):
    from tests.prims_ts.test_batched_gemm_routed_e32 import check_routed_e32

    opts = flexible_tma_native_options(
        tile_m=256, specialize=specialize, local_barrier=specialize, a_stages=4
    )
    opts.update(
        routed_sfa_first=1,
        num_stages_b=6,
        num_stages_smem_sfa=3,
        num_stages_smem_sfb=4,
    )
    check_routed_e32(
        monkeypatch,
        opts,
        num_tokens=257,
        problem_n=1280,
        problem_k=2816,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("cga", CGAS)
def test_native_tma_host_separate_loaders(cga, warps):
    opts = tma_native_options(cga, warps)
    cfg = make_config(**opts)
    assert cfg.has_tma_route and not cfg.has_gather
    assert cfg.uses_ldgsts_sf_relay
    assert not cfg.fuse_operand_sf_loads
    assert not cfg.shares_ldgsts_a_sfa_pipeline
    assert cfg.tmem_sfa_cols == 8
    _build_batched_gemm_task_manager(verbose=False, **opts)


def test_native_tma_host_rejects_fused_tile256_allocator():
    opts = options(256, (2, 1), "clc")
    opts.update(
        route_act=int(RouteImpl.TMA),
        fuse_operand_sf_loads=0,
        fused_a_single_commit=0,
        dense_partition_tma_loads=0,
    )
    # The fixed allocator's payload names and completion barriers describe
    # fused LDGSTS A/SFA, not separate TMA-A and native-SFA resources.
    with pytest.raises(ValueError, match="routed_low_smem"):
        validate_config(make_config(**opts), problem_mnk=(512, 1280, 2304))


def install_sfa_first(monkeypatch):
    """Legacy comparison-script shim; kernel tests use routed_sfa_first."""
    from cutlass.experimental.task_scheduling.memory import SmemAllocator

    original_layout = SmemAllocator.compute_layout

    def sfa_first_layout(allocator):
        # Match the native-SFA placement of the measured TMA-A tile512 preset.
        allocator._allocations.sort(
            key=lambda allocation: not allocation.name.endswith("_sf_ldg")
        )
        original_layout(allocator)

    monkeypatch.setattr(SmemAllocator, "compute_layout", sfa_first_layout)


def check_native_tma(monkeypatch, cga, *, num_tokens=1031, problem_n=1280):
    opts = tma_native_options(cga)
    opts["routed_sfa_first"] = 1
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    result = runner.reference_check(
        num_experts=3,
        num_tokens=num_tokens,
        top_k=2,
        problem_n=problem_n,
        problem_k=2304,
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **opts,
    )
    assert result and result.max_abs == 0


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "cga,specialize,local_barrier,a_stages,sfa_first",
    [
        ((4, 4), True, True, 5, True),
        ((4, 4), True, True, 4, False),
        ((4, 4), True, False, 5, True),
        ((4, 1), True, True, 5, True),
        ((2, 4), True, True, 5, True),
    ],
)
def test_native_tma_flexible_ring_wrap(
    monkeypatch, cga, specialize, local_barrier, a_stages, sfa_first
):
    from tests.prims_ts.test_batched_gemm_routed_e32 import check_routed_e32

    opts = flexible_tma_native_options(
        cga, specialize=specialize, local_barrier=local_barrier, a_stages=a_stages
    )
    opts["routed_sfa_first"] = int(sfa_first)
    # Ragged M/N, two M fragments, ring wrap, multiple persistent waves,
    # and both physical CGAs. A4/SFA5 keeps independent relay stage indices.
    check_routed_e32(
        monkeypatch,
        opts,
        num_tokens=2051,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
def test_native_tma_flexible_long_k(monkeypatch):
    from tests.prims_ts.test_batched_gemm_routed_e32 import check_routed_e32

    # Match the historical correctness shape; 31 K tiles repeatedly wrap
    # the independent four-stage A and five-stage native-SFA relay rings.
    check_routed_e32(
        monkeypatch,
        flexible_tma_native_options(a_stages=4),
        num_tokens=2051,
        problem_n=1280,
        problem_k=7936,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("cga", CGAS)
def test_native_tma_ragged_ring_wrap(monkeypatch, cga):
    check_native_tma(monkeypatch, cga)


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("cga", [(2, 1), (4, 4)])
def test_native_tma_clc_reuse(monkeypatch, cga):
    # 126 two-CTA clusters / 18 sixteen-CTA clusters exceed one 216-SM
    # resident wave while keeping only three small expert matrices.
    check_native_tma(monkeypatch, cga, num_tokens=4097, problem_n=1792)


@pytest.mark.parametrize("scheduler", ["static", "clc"])
@pytest.mark.parametrize("cga", CGAS)
@pytest.mark.parametrize("tile_m", [256, 512])
def test_native_geometry_host_schedule(tile_m, cga, scheduler):
    # Execute the real task/barrier checks, including the per-rank queue
    # census. Larger X has multiple MMA leaders, larger Y has multiple N tiles.
    _build_batched_gemm_task_manager(verbose=False, **options(tile_m, cga, scheduler))


@pytest.mark.parametrize("experts,n", [(256, 4096), (384, 6144)])
@pytest.mark.parametrize("cga", CGAS)
def test_native_tile512_host_route_padding_and_grid(experts, n, cga):
    cfg = make_config(**options(512, cga, "clc"))
    layout = runner._make_token_layout(
        num_tokens=512,
        num_experts=experts,
        top_k=experts,
        tile_size=cfg.token_tile_size,
        cluster_dim_in_token=cfg.token_cluster_dim,
    )
    assert layout.tokens_per_expert == [512] * experts
    assert layout.total_padded_tokens == experts * (512 if cga[0] == 2 else 1024)
    assert layout.num_token_cgas == experts
    expected_clusters = experts * (n // (256 * cga[1]))
    args = dict(
        problem_m=layout.total_padded_tokens, problem_n=n, early_exit_max_token_ctas=0
    )
    assert compute_launch_grid(cfg, **args) == (*cga, expected_clusters)
    static_cfg = make_config(**options(512, cga, "static"))
    assert compute_launch_grid(static_cfg, **args) == (*cga, 2)


@pytest.mark.parametrize("cluster_x", [2, 4])
@pytest.mark.parametrize("repeated", [False, True])
def test_native_fixture_host_route_padding(cluster_x, repeated):
    import torch
    from flashinfer.prims_ts.batched_gemm.tools.routed import routed_inputs

    index = torch.arange(3 * 512, dtype=torch.int32)
    routes = (index * 17 + index // 512) % 512
    if repeated:
        routes.fill_(511)

    class Fixture:
        def load(self, name):
            assert name == "route"
            return routes

    with routed_inputs(runner, Fixture()):
        layout = runner._make_token_layout(
            num_tokens=512,
            num_experts=3,
            top_k=3,
            tile_size=256,
            cluster_dim_in_token=cluster_x,
        )
    for expert, start in enumerate(layout.padded_starts[:-1]):
        assert (
            layout.expanded_to_token[start : start + 512]
            == routes[expert * 512 : (expert + 1) * 512].tolist()
        )
        padding = 256 * cluster_x - 512
        assert (
            layout.expanded_to_token[start + 512 : start + 512 + padding]
            == [0] * padding
        )
        assert (
            layout.expanded_to_expert[start + 512 : start + 512 + padding]
            == [-1] * padding
        )


@pytest.mark.parametrize("cluster_x", [2, 4])
@pytest.mark.parametrize(
    "invalid", ["float", "int64", "short", "long", "negative", "too_large"]
)
def test_native_fixture_host_rejects_invalid_routes(cluster_x, invalid):
    import torch
    from types import SimpleNamespace
    from flashinfer.prims_ts.batched_gemm.tools.routed import routed_inputs

    routes = torch.zeros(3 * 512, dtype=torch.int32)
    if invalid in ("float", "int64"):
        routes = routes.to(torch.float32 if invalid == "float" else torch.int64)
    elif invalid == "short":
        routes = routes[:-1]
    elif invalid == "long":
        routes = torch.cat((routes, routes[:1]))
    else:
        routes[-1] = -1 if invalid == "negative" else 512
    original = runner._make_token_layout
    with (
        pytest.raises(ValueError, match="route fixture"),
        routed_inputs(runner, SimpleNamespace(load=lambda name: routes)),
    ):
        runner._make_token_layout(
            num_tokens=512,
            num_experts=3,
            top_k=3,
            tile_size=256,
            cluster_dim_in_token=cluster_x,
        )
    assert runner._make_token_layout is original


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("y", [0, 1])
def test_sfb_host_release_reaches_both_partition_loaders(monkeypatch, partitioned, y):
    from dataclasses import replace
    from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga
    from tests.prims_ts.python_body import python_body

    cfg = make_config(**options(512, (2, 2), "clc"))
    cfg = replace(
        cfg, dense_split_tma_pipeline=0, dense_partition_tma_loads=int(partitioned)
    )
    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(cga.cutlass, "const_expr", lambda value: value)
    monkeypatch.setattr(cga.cutlass, "range_constexpr", range)
    monkeypatch.setattr(cga, "active_cga_coords", lambda _: (2, 2, 2 * y, 0, 0, y))
    for name in (
        "_rank_bit",
        "_atom_mask",
        "_row_mask",
        "_project_payload_to_barrier_owners",
        "cga_policy_mask",
        "input_tma_release",
        "input_tma_payload_mask",
    ):
        monkeypatch.setattr(cga, name, python_body(getattr(cga, name)))

    payload, owners, release, arrivals = python_body(cga.cga_tma_masks)(cfg, "sfb")
    # Both partition loaders acquire their own empty barrier. Every one must
    # receive the MMA release, including the odd CTA in the second Y atom.
    loader_ranks = [2 * y, 2 * y + 1] if partitioned else [2 * y]
    assert release == sum(1 << rank for rank in loader_ranks)
    assert payload == 3 << (2 * y)
    assert owners == 1 << (2 * y)
    assert arrivals == 1


def check(monkeypatch, tile_m, cga, scheduler, *, empty=False):
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    result = runner.reference_check(
        num_experts=8 if empty else 3,
        num_tokens=1 if empty else 1031,
        top_k=1 if empty else 2,
        problem_n=1280,  # Five N tiles leave a partial 2-/4-wide CGA.
        problem_k=2304,  # Nine K tiles wrap both five- and eight-stage rings.
        routing="uniform",
        repeat_launches=3,
        output_guard_elements=256,
        **options(tile_m, cga, scheduler),
    )
    assert result and result.max_abs == 0


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("scheduler", ["static", "clc"])
@pytest.mark.parametrize("cga", CGAS)
@pytest.mark.parametrize("tile_m", [256, 512])
def test_native_geometry_ragged_ring_wrap(monkeypatch, tile_m, cga, scheduler):
    check(monkeypatch, tile_m, cga, scheduler)


@pytest.mark.arch_rubin
@pytest.mark.timeout(180)
@pytest.mark.parametrize("scheduler", ["static", "clc"])
@pytest.mark.parametrize("cga", [(2, 4), (4, 4)])
def test_native_tile512_empty_experts(monkeypatch, cga, scheduler):
    check(monkeypatch, 512, cga, scheduler, empty=True)
