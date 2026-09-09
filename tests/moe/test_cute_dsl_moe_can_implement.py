"""Directed host-side checks for the gh #3957 N-tail handling.

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


def _finalize_ok(n, mma_tiler_mn, cluster_shape_mn):
    return Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        out_dtype=cutlass.BFloat16,
        final_scale_dtype=cutlass.Float32,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=1024,
        n=n,
        k=512,
        l=8,
        a_major="k",
        b_major="k",
        out_major="n",
    )


def _gemm1_ok(n, mma_tiler_mn, cluster_shape_mn):
    return BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        c_dtype=cutlass.Float4E2M1FN,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        m=1024,
        n=n,
        k=512,
        l=8,
        a_major="k",
        b_major="k",
        c_major="n",
    )


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
    ],
)
def test_finalize_n_tiling(n, mma, cluster, expect):
    assert _finalize_ok(n, mma, cluster) is expect


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


def test_every_rubin_tactic_is_single_cta():
    """No enumerated Rubin tactic may use a multi-CTA cluster.

    ``moe_sort`` leaves tile-metadata entries past ``num_non_exiting_tiles``
    uninitialized, and both GEMMs bound their reads by that exact count. A
    ``cluster_shape_m > 1`` tactic would need the count rounded up to a
    multiple of ``cluster_shape_m`` for barrier uniformity, which widens
    those reads past what routing wrote. Neither the rounding nor the
    matching buffer initialization is implemented, so such a tactic must
    never reach the autotuner.
    """
    from flashinfer.fused_moe.cute_dsl.tuner import ALL_RUBIN_MOE_TACTICS

    assert ALL_RUBIN_MOE_TACTICS, "Rubin tactic list is empty -- nothing asserted"
    for tile_size, gemm1_tactic, gemm2_tactic in ALL_RUBIN_MOE_TACTICS:
        _, _, gemm1_cluster_shape_mn, _ = gemm1_tactic
        _, _, gemm2_cluster_shape_mn, _ = gemm2_tactic
        assert gemm1_cluster_shape_mn[0] == 1, (
            f"gemm1 tactic {gemm1_tactic} at tile_size={tile_size} uses "
            f"cluster_shape_m={gemm1_cluster_shape_mn[0]}"
        )
        assert gemm2_cluster_shape_mn[0] == 1, (
            f"gemm2 tactic {gemm2_tactic} at tile_size={tile_size} uses "
            f"cluster_shape_m={gemm2_cluster_shape_mn[0]}"
        )
