"""Directed host-side checks for the gh #3957 can_implement N-tile guards.

The CuteDSL MoE epilogues store full CTA-tile rows with no column predicate
(finalize: raw-pointer ``cp.reduce.async.bulk`` scatter; gemm1: unpredicated
SFC ``autovec_copy``), so ``can_implement`` must reject any tactic whose
N-tiling leaves a partial CTA tile or a cluster-padding CTA along N.
These are pure classmethod checks -- no GPU work.
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
        ab_dtype=cutlass.Float4E2M1FN,
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
        ab_dtype=cutlass.Float4E2M1FN,
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
        # The observed gh #3957 combo: 1 N-tile but 2 CTAs/cluster along N ->
        # the padding CTA bulk-reduces one full tile past the output columns.
        (256, (128, 256), (1, 2), False),
        # 2 exact tiles / cluster_n=2 -> exact cluster tiling: fine.
        (512, (128, 256), (1, 2), True),
        # Partial N-tile (384 = 1.5 tiles): the second CTA writes columns
        # 256..511 of a 384-wide output. Must reject even though
        # ceil_div(384,256)=2 divides cluster_n=2 (the draft guard's miss).
        (384, (128, 256), (1, 2), False),
        # Same partial tile without any cluster: still an overrun.
        (384, (128, 256), (1, 1), False),
        # 3 exact 128-tiles, no cluster: fine.
        (384, (128, 128), (1, 1), True),
    ],
)
def test_finalize_n_tiling_guard(n, mma, cluster, expect):
    assert _finalize_ok(n, mma, cluster) is expect


@pytest.mark.parametrize(
    "n,mma,expect",
    [
        # Partial N-tile under mma_n=256: the unpredicated SFC store overruns.
        (384, (128, 256), False),
        # Exact tiling: fine.
        (512, (128, 256), True),
        (384, (128, 128), True),
    ],
)
def test_gemm1_n_tiling_guard(n, mma, expect):
    # gemm1 requires cluster_n == 1 (enforced independently).
    assert _gemm1_ok(n, mma, (1, 1)) is expect
