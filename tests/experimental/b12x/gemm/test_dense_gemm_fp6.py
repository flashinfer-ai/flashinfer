from __future__ import annotations

import cutlass

from b12x._lib.utils import mxfp6_tile_k
from b12x._lib.dense_gemm import DenseGemmKernel


def test_dense_gemm_can_implement_mxfp6_e3m2() -> None:
    assert DenseGemmKernel.can_implement(
        cutlass.Float6E3M2FN,
        cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        n=128,
        k=128,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )


def test_dense_gemm_can_implement_mxfp6_e2m3() -> None:
    assert DenseGemmKernel.can_implement(
        cutlass.Float6E2M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        n=256,
        k=256,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )


def test_dense_gemm_rejects_mxfp6_with_sf_vec_size_16() -> None:
    assert not DenseGemmKernel.can_implement(
        cutlass.Float6E3M2FN,
        cutlass.Float8E4M3FN,
        sf_vec_size=16,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        n=128,
        k=128,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )


def test_dense_gemm_rejects_mxfp6_k_not_multiple_of_tile_k() -> None:
    assert not DenseGemmKernel.can_implement(
        cutlass.Float6E3M2FN,
        cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        n=128,
        k=129,
        l=1,
        a_major="k",
        b_major="k",
        c_major="n",
    )
    assert mxfp6_tile_k() == 128
