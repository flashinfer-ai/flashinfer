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

"""E32 routed loaders and completion relays for tile256 and tile512."""

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    CtaRasterOrder,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    WeightLayout,
    make_config,
)

pytestmark = [pytest.mark.arch_rubin, pytest.mark.timeout(180)]


def routed_e32_config(route_a, warps_a=4, warps_sf=4, *, wide_cga=False):
    # Keep the E32 arithmetic, weight layout, and N traversal. Start with
    # separate routed resources: the tuned dense pipeline forbids routing,
    # and the LDGSTS activation proxy currently requires equal A/B depths.
    return dict(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=int(route_a),
        route_sfs_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        act_kind=int(ActKind.NONE),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        weight_layout=int(WeightLayout.BlockMajorK),
        umma_cta_count=2,
        cluster_dim_x=4 if wide_cga else 2,
        cluster_dim_y=4 if wide_cga else 1,
        fallback_cluster_dim_x=2 if wide_cga else 0,
        fallback_cluster_dim_y=1,
        tile_n=256,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        mma_k=128,
        epi_tile_n=64,
        # The LDGSTS SFA relay needs another control warp. With eight A or
        # SFA warps, the resulting 24-warp CTA cannot compile a 64-register
        # TMEM load (ptxas needs >=82 registers; the CTA limit is 80).
        tmem_ldst_max_num_regs=(32 if max(warps_a, warps_sf) == 8 else 64),
        num_stages_a=5,
        num_stages_b=5,
        num_stages_smem_sfa=5,
        num_stages_smem_sfb=5,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=1,
        num_stages_tmem_sfb=1,
        num_stages_c_smem=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        use_tma_oob_opt=0,
        use_tma_store=1,
        use_max_tmem_overlap=0,
        use_early_exit=0,
        use_pdl=0,
        use_global_scales=0,
        use_unroll_loop_2x_for_mma=0,
        fuse_sf_copy_to_mma=1,
        num_load_a_warps=warps_a,
        num_load_sfa_warps=warps_sf,
        epilogue_regs=128,
        mma_regs=64,
        load_regs=64,
        load_sf_regs=64,
        copy_sf_regs=64,
        sync_regs=64,
        workid_regs=64,
        padding_regs=64,
    )


def check_routed_e32(monkeypatch, options, *, full_sf_range=False, **shape):
    if not full_sf_range:

        def bounded_sf(shape, *, cfg, device, layout_kind):
            return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

        monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    resolved = make_config(**options)
    assert resolved.warp_layout.num_load_a_warps == options["num_load_a_warps"]
    assert resolved.warp_layout.num_load_sfa_warps == options["num_load_sfa_warps"]
    assert resolved.warp_layout.num_load_a_warps > 1
    assert resolved.warp_layout.num_load_sfa_warps > 1
    result = runner.reference_check(
        num_experts=32,
        top_k=8,
        repeat_launches=3,
        output_guard_elements=256,
        **shape,
        **options,
    )
    assert result
    if not full_sf_range:
        assert result.max_abs == 0


def routed_e32_tile512_config(route_a, warps_a=4, warps_sf=4, *, wide_cga=False):
    options = routed_e32_config(route_a, warps_a, warps_sf, wide_cga=wide_cga)
    # The packed eight-warp output leaves 16 control warps for LDGSTS 4/4.
    # Donate their unused registers to the epilogue; A/SFA/B/SFB retain
    # independent tasks. Matching LDGSTS A/SFA rings share barriers without
    # enabling dense TMA load fusion.
    options.update(
        dense_m_repeats=2,
        dense_b_reuse=1,
        dense_rolled_mma=1,
        dense_async_sf=1,
        dense_rolled_epilogue=1,
        dense_preload_epilogue=1,
        dense_dkg_store_order=1,
        dense_epilogue_warps=8,
        epilogue_regs=144,
        tmem_ldst_max_num_regs=64,
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
        options[name] = 40
    if route_a == RouteImpl.LDGSTS and warps_a == 4:
        # The A loader occupies a full warpgroup and keeps route indices
        # live across K. Its register budget can differ from other controls.
        options["load_a_regs"] = 64
    return options


def routed_e32_dense_config(route_a, warps=4, *, wide_cga=False):
    """Retain dense weight scheduling and cache policies with routed A/SFA."""
    options = routed_e32_tile512_config(route_a, warps, warps, wide_cga=wide_cga)
    options.update(
        dense_split_tma_pipeline=1,
        dense_fuse_tma_loads=1,
        dense_partition_tma_loads=1,
        dense_specialize_tma_cga=int(wide_cga),
        dense_local_barrier=int(wide_cga),
        dense_retain_a=1,
        dense_a_l2_promotion=256 if route_a == RouteImpl.TMA else 0,
        dense_streaming_c=1,
        num_stages_b=6,
        num_stages_smem_sfb=6,
        num_stages_workid=1,
        use_work_throttle=0,
        load_a_regs=64,
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
        options[name] = 64
    return options


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize(
    "warps,wide_cga,k,sfa_stages",
    [(4, False, 256, 5), (4, False, 7936, 3), (4, True, 1792, 5), (2, True, 2304, 5)],
)
def test_routed_e32_retains_dense_optimizations(
    monkeypatch, route_a, warps, wide_cga, k, sfa_stages
):
    options = routed_e32_dense_config(route_a, warps, wide_cga=wide_cga)
    options["num_stages_smem_sfa"] = sfa_stages
    check_routed_e32(
        monkeypatch,
        options,
        num_tokens=2051 if wide_cga else 1543,
        problem_n=1280 if wide_cga else 256,
        problem_k=k,
        routing="uniform",
        require_mixed_cga_runtime=wide_cga,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize("cluster_y", [2, 4])
def test_routed_e32_dense_fixed_cga_tail_and_ring_wrap(monkeypatch, route_a, cluster_y):
    options = routed_e32_dense_config(route_a)
    options.update(cluster_dim_y=cluster_y, num_stages_smem_sfa=3)
    check_routed_e32(
        monkeypatch,
        options,
        num_tokens=1027,
        problem_n=768,
        problem_k=2304,
        routing="uniform",
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_dense_empty_experts(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_dense_config(route_a),
        num_tokens=17,
        problem_n=256,
        problem_k=256,
        routing="zipf:2.0",
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_dense_full_scale_range(monkeypatch, route_a):
    monkeypatch.setenv("BATCHED_GEMM_TS_RANDOMIZE_SF", "all")
    check_routed_e32(
        monkeypatch,
        routed_e32_dense_config(route_a),
        num_tokens=129,
        problem_n=512,
        problem_k=1792,
        full_sf_range=True,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_smoke(monkeypatch, route_a):
    # >384 routes/expert exercises both M fragments in both CTAs, including
    # the partial final fragment. K256 also checks an underfilled load ring.
    check_routed_e32(
        monkeypatch,
        routed_e32_tile512_config(route_a),
        num_tokens=1543,
        problem_n=256,
        problem_k=256,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize("warps_a,warps_sf", [(2, 2), (4, 4)])
def test_routed_e32_tile512_ring_wrap(monkeypatch, route_a, warps_a, warps_sf):
    options = routed_e32_tile512_config(route_a, warps_a, warps_sf)
    if warps_a == 2:
        # The two-warp gather shares its warpgroup with MMA and B loading.
        for name in (
            "mma_regs",
            "load_regs",
            "load_sf_regs",
            "sync_regs",
            "workid_regs",
            "padding_regs",
        ):
            options[name] = 64
    check_routed_e32(
        monkeypatch,
        options,
        num_tokens=1543,
        problem_n=768,
        problem_k=1792,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_mixed_cga(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_tile512_config(route_a, wide_cga=True),
        num_tokens=2051,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_empty_experts(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_tile512_config(route_a),
        num_tokens=17,
        problem_n=256,
        problem_k=256,
        routing="zipf:2.0",
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_independent_sf_rings(monkeypatch, route_a):
    options = routed_e32_tile512_config(route_a)
    options.update(num_stages_smem_sfa=3, num_stages_smem_sfb=4)
    check_routed_e32(
        monkeypatch,
        options,
        num_tokens=1543,
        problem_n=256,
        problem_k=2816,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_full_scale_range(monkeypatch, route_a):
    monkeypatch.setenv("BATCHED_GEMM_TS_RANDOMIZE_SF", "all")
    check_routed_e32(
        monkeypatch,
        routed_e32_tile512_config(route_a),
        full_sf_range=True,
        num_tokens=1543,
        problem_n=256,
        problem_k=1536,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_tile512_long_k(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_tile512_config(route_a),
        num_tokens=1543,
        problem_n=256,
        problem_k=7936,
    )


def test_routed_e32_tile512_tma_six_weight_stages(monkeypatch):
    options = routed_e32_tile512_config(RouteImpl.TMA)
    options.update(num_stages_b=6, num_stages_smem_sfb=6)
    check_routed_e32(
        monkeypatch,
        options,
        num_tokens=1543,
        problem_n=256,
        problem_k=7936,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize("warps_a,warps_sf", [(2, 2), (4, 4), (8, 4), (4, 8)])
def test_routed_e32_loader_warps(monkeypatch, route_a, warps_a, warps_sf):
    check_routed_e32(
        monkeypatch,
        routed_e32_config(route_a, warps_a, warps_sf),
        num_tokens=259,
        problem_n=768,
        problem_k=1792,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_short_k_empty_experts(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_config(route_a),
        num_tokens=17,
        problem_n=256,
        problem_k=256,
        routing="zipf:2.0",
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_mixed_cga_multiple_waves(monkeypatch, route_a):
    check_routed_e32(
        monkeypatch,
        routed_e32_config(route_a, wide_cga=True),
        num_tokens=2051,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        require_mixed_cga_runtime=True,
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
def test_routed_e32_full_scale_range(monkeypatch, route_a):
    monkeypatch.setenv("BATCHED_GEMM_TS_RANDOMIZE_SF", "all")
    check_routed_e32(
        monkeypatch,
        routed_e32_config(route_a),
        num_tokens=129,
        problem_n=512,
        problem_k=1536,
        full_sf_range=True,
    )
