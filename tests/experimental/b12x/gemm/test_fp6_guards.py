from __future__ import annotations

# TODO(port): several tests below exercised the retired pre-facade b12x.integration.tp_moe
# host helpers (_mxfp6_cutlass_dtypes / _mxfp6_fmt_pair / _normalize_fp6_source_format)
# and the FP6 micro-kernel predicate (is_supported_mxfp6), none of which has an
# upstream b12x equivalent yet. Those test bodies are kept as reference
# behind pytest.skip gates; rewrite them against b12x.moe.fused_moe once
# the w6a8_mx run path lands.

import cutlass
import pytest

from b12x._lib.utils import mxfp6_packed_k_bytes, mxfp6_tile_k
from b12x._lib.dense_gemm import DenseGemmKernel


def test_dense_gemm_can_implement_mixed_fp6_requires_uniform_ab_dtype() -> None:
    """Dense GEMM uses one ab_dtype for both operands; mixed formats are MoE-only."""
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


def test_moe_w6a8_mxfp6_format_dtypes() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    from b12x.integration import tp_moe  # retired pre-facade API (module TODO)

    act, wt, sf = tp_moe._mxfp6_cutlass_dtypes("mxfp6_w6a8")
    assert act is cutlass.Float8E4M3FN
    assert wt is cutlass.Float6E2M3FN
    assert sf is cutlass.Float8E8M0FNU
    assert tp_moe._mxfp6_fmt_pair("mxfp6_w6a8") == ("e4m3", "e2m3")


def test_moe_default_mxfp6_mixed_format_dtypes() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    from b12x.integration import tp_moe  # retired pre-facade API (module TODO)

    act, wt, sf = tp_moe._mxfp6_cutlass_dtypes("mxfp6_default")
    assert act is cutlass.Float6E3M2FN
    assert wt is cutlass.Float6E2M3FN
    assert sf is cutlass.Float8E8M0FNU


def test_moe_uniform_mxfp6_e2m3_source_format() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    from b12x.integration import tp_moe  # retired pre-facade API (module TODO)

    act, wt, _ = tp_moe._mxfp6_cutlass_dtypes("mxfp6_e2m3")
    assert act is cutlass.Float6E2M3FN
    assert wt is cutlass.Float6E2M3FN


def test_micro_kernel_rejects_mxfp6() -> None:
    pytest.skip("pending: FP6 micro kernel port (is_supported_mxfp6 not upstream)")
    from b12x.moe._shared.kernels.silu import MoEMicroKernelSilu

    assert MoEMicroKernelSilu.is_supported_mxfp6(1, 128, 128, 8, 4) is False


def test_w6a6_packed_k_bytes_matches_tile_k() -> None:
    k = mxfp6_tile_k()
    assert mxfp6_packed_k_bytes(k) == (k * 3) // 4
    assert k % 128 == 0


def test_normalize_fp6_source_format_rejects_unknown() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    from b12x.integration import tp_moe  # retired pre-facade API (module TODO)

    with pytest.raises(ValueError, match="source_format must be"):
        tp_moe._normalize_fp6_source_format("not_a_format")
