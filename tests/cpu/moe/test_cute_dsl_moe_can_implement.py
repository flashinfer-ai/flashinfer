"""Directed host-side checks for blockscaled MoE kernel contracts.

The finalize epilogue limits its bulk transfer to the remaining output columns,
including for cluster-padding CTAs. The gemm1 SFC store is still unpredicated,
so only gemm1 must reject configurations that leave a partial N tile. These are
pure classmethod checks -- no GPU work.
"""

import pytest

cutlass = pytest.importorskip("cutlass")

from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
    BlockScaledContiguousGatherGroupedGemmKernel,
)
from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)


def _finalize_ok(
    n,
    mma_tiler_mn,
    cluster_shape_mn,
    *,
    a_dtype=cutlass.Float4E2M1FN,
    b_dtype=cutlass.Float4E2M1FN,
    sf_dtype=cutlass.Float8E4M3FN,
    sf_vec_size=16,
    k=512,
):
    return Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        out_dtype=cutlass.BFloat16,
        final_scale_dtype=cutlass.Float32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=1024,
        n=n,
        k=k,
        l=8,
        a_major="k",
        b_major="k",
        out_major="n",
    )


def _gemm1_ok(
    n,
    mma_tiler_mn,
    cluster_shape_mn,
    *,
    a_dtype=cutlass.Float4E2M1FN,
    b_dtype=cutlass.Float4E2M1FN,
    sf_dtype=cutlass.Float8E4M3FN,
    sf_vec_size=16,
    c_dtype=cutlass.Float4E2M1FN,
    k=512,
):
    return BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=1024,
        n=n,
        k=k,
        l=8,
        a_major="k",
        b_major="k",
        c_major="n",
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "n,mma,cluster,expect",
    [
        # A padding CTA has no remaining columns and skips its row transfer.
        (256, (128, 256), (1, 2), True),
        # 2 exact tiles / cluster_n=2 -> exact cluster tiling: fine.
        (512, (128, 256), (1, 2), True),
        # A partial tile transfers only columns 256..383.
        (384, (128, 256), (1, 2), True),
        # The same tail handling applies without an N cluster.
        (384, (128, 256), (1, 1), True),
        # 3 exact 128-tiles, no cluster: fine.
        (384, (128, 128), (1, 1), True),
        (384, (128, 192), (1, 1), True),
        (256, (128, 192), (1, 1), True),
        (128, (128, 256), (1, 1), True),
    ],
)
def test_finalize_n_tiling(n, mma, cluster, expect):
    assert _finalize_ok(n, mma, cluster) is expect


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "n,mma,expect",
    [
        # The scale-factor store requires exact tiling under mma_n=256.
        (384, (128, 256), False),
        # Exact tiling: fine.
        (512, (128, 256), True),
        (384, (128, 128), True),
    ],
)
def test_gemm1_n_tiling_guard(n, mma, expect):
    # gemm1 requires cluster_n == 1 (enforced independently).
    assert _gemm1_ok(n, mma, (1, 1)) is expect


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", [_gemm1_ok, _finalize_ok])
def test_w4a8_dtype_contract(op):
    kwargs = dict(
        a_dtype=cutlass.Float8E4M3FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        k=128,
    )
    if op is _gemm1_ok:
        kwargs["c_dtype"] = cutlass.Float8E4M3FN
    assert op(256, (128, 128), (1, 1), **kwargs)
    assert not op(
        256,
        (128, 128),
        (1, 1),
        **{
            **kwargs,
            "a_dtype": cutlass.Float4E2M1FN,
            "b_dtype": cutlass.Float8E4M3FN,
        },
    )
    assert not op(
        256,
        (128, 128),
        (1, 1),
        **{**kwargs, "sf_dtype": cutlass.Float8E4M3FN},
    )
    assert not op(256, (128, 128), (1, 1), **{**kwargs, "sf_vec_size": 16})
    assert not op(256, (128, 128), (1, 1), **{**kwargs, "k": 64})


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mma,n,expect",
    [
        ((128, 64), 128, True),
        ((128, 192), 384, True),
        ((128, 64), 192, False),
        ((128, 192), 256, True),
    ],
)
def test_w4a8_finalize_n_tiling(mma, n, expect):
    assert (
        _finalize_ok(
            n,
            mma,
            (1, 1),
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float4E2M1FN,
            sf_dtype=cutlass.Float8E8M0FNU,
            sf_vec_size=32,
            k=128,
        )
        is expect
    )


@pytest.mark.cpu_only
def test_every_rubin_cluster_covers_one_routing_tile():
    """Every enumerated Rubin tactic maps one cluster to one routing tile.

    ``moe_sort`` leaves tile-metadata entries past ``num_non_exiting_tiles``
    uninitialized, and both GEMMs bound their reads by that exact count. A
    cluster spanning several routing tiles (a 1-CTA MMA with
    ``cluster_shape_m > 1``) would need the count rounded up to a multiple of
    ``cluster_shape_m`` for barrier uniformity, which widens those reads past
    what routing wrote, so such a tactic must never reach the autotuner. A
    2-CTA MMA (``mma_inst_shape[0] == 256``) with ``cluster_shape_m == 2``
    covers exactly one routing tile.
    """
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        rubin_moe_cluster_covers_one_tile,
    )
    from flashinfer.fused_moe.cute_dsl.tuner import ALL_RUBIN_MOE_TACTICS

    assert ALL_RUBIN_MOE_TACTICS, "Rubin tactic list is empty -- nothing asserted"
    for tile_size, gemm1_tactic, gemm2_tactic in ALL_RUBIN_MOE_TACTICS:
        for name, (mma_tiler, mma_inst_shape, cluster_shape_mn, _) in (
            ("gemm1", gemm1_tactic),
            ("gemm2", gemm2_tactic),
        ):
            assert rubin_moe_cluster_covers_one_tile(
                tile_size, mma_tiler, mma_inst_shape, cluster_shape_mn
            ), (
                f"{name} tactic {(mma_tiler, mma_inst_shape, cluster_shape_mn)} "
                f"at tile_size={tile_size} spans several routing tiles"
            )


@pytest.mark.cpu_only
def test_rubin_gemm1_offers_two_cta_tile_256():
    """tile_size=256 offers the 2-CTA FC1 MMA with cluster (2, 1).

    At tile_size=128 a 2-CTA cluster would cover two routing tiles, so it
    is not offered there.
    """
    from flashinfer.fused_moe.cute_dsl.tuner import get_rubin_gemm1_valid_tactics

    two_cta = [t for t in get_rubin_gemm1_valid_tactics(256) if t[1][0] == 256]
    assert sorted(t[0] for t in two_cta) == [(256, 128, 256), (256, 256, 256)]
    assert all(t[2] == (2, 1) for t in two_cta)
    assert all(t[1][0] == 128 for t in get_rubin_gemm1_valid_tactics(128))


@pytest.mark.cpu_only
def test_rubin_gemm2_offers_two_cta_tile_256():
    """tile_size=256 offers the 2-CTA finalize MMA with cluster (2, 1)."""
    from flashinfer.fused_moe.cute_dsl.tuner import get_rubin_gemm2_valid_tactics

    two_cta = [t for t in get_rubin_gemm2_valid_tactics(256) if t[1][0] == 256]
    assert sorted(t[0] for t in two_cta) == [(256, 128, 256), (256, 256, 256)]
    assert all(t[2] == (2, 1) for t in two_cta)
    assert all(t[1][0] == 128 for t in get_rubin_gemm2_valid_tactics(128))


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile_size", [128, 256])
def test_rubin_gemm2_offers_cluster_n_2(tile_size):
    """Every 1-CTA finalize MMA is offered with cluster (1, 1) and (1, 2)."""
    from flashinfer.fused_moe.cute_dsl.tuner import get_rubin_gemm2_valid_tactics

    tactics = get_rubin_gemm2_valid_tactics(tile_size)
    one_cta = {(t[0], t[1]) for t in tactics if t[1][0] == 128}
    assert one_cta, f"no 1-CTA finalize tactic at tile_size={tile_size}"
    for mma_tiler, mma_inst_shape in one_cta:
        clusters = {
            t[2] for t in tactics if (t[0], t[1]) == (mma_tiler, mma_inst_shape)
        }
        assert clusters == {(1, 1), (1, 2)}, (mma_tiler, mma_inst_shape, clusters)
    # A 2-CTA finalize keeps cluster (2, 1) only.
    assert all(t[2] == (2, 1) for t in tactics if t[1][0] == 256)
