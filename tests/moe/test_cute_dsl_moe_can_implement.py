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
