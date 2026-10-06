"""Tests for native packed-FP6 B streaming (``dense_gemm(b_packed=True)``).

The packed mode TMA-streams the 3:4-packed weight ``(N, 3K/4, L)`` and expands
to byte-containers in smem; with identical activation codes, scales, and alpha
it must be BIT-IDENTICAL to the validated ``b_preexpanded`` path — only the B
streaming format differs.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _gemm_operands(m: int, n: int, k: int, source_format: str):
    """Quantized operands for one case, mirroring ``dense_fp6_linear_expanded``."""
    from b12x._lib.fp6 import SF_VEC_SIZE_FP6, as_grouped_mxfp6_scale_view
    from b12x.quantization.mxfp6.fp6_dense_weights import (
        _TILE,
        _quantize_matrix_fp6_bytes,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    f8_max = float(torch.finfo(torch.float8_e4m3fn).max)

    w_bf16 = (torch.randn(n, k, device=device) * 0.1).to(torch.bfloat16)
    fp6w = quantize_dense_weight_to_fp6(w_bf16, source_format=source_format)
    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    a_amax = (
        torch.linalg.vector_norm(x, ord=float("inf"))
        .to(torch.float32)
        .clamp_min_(1e-6)
    )
    a_gs = (f8_max * 28.0 / a_amax).reshape(1)
    m_pad = ((m + _TILE - 1) // _TILE) * _TILE
    x_pad = torch.empty(m_pad, k, dtype=torch.bfloat16, device=device)
    x_pad[:m].copy_(x)
    a_codes, a_scale = _quantize_matrix_fp6_bytes(x_pad, fp6w.fmt, a_gs)
    a_sf = as_grouped_mxfp6_scale_view(a_scale.view(1, -1), m_pad, k)
    common = dict(
        alpha=torch.reciprocal(a_gs * fp6w.global_scale),
        ab_dtype=fp6w.ab_dtype,
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=SF_VEC_SIZE_FP6,
        c_dtype="bfloat16",
        a_preexpanded=True,
    )
    return fp6w, (a_codes[:m].unsqueeze(-1), a_sf), fp6w.scale_view(), common


@cuda_required
@pytest.mark.parametrize(
    "m,n,k,source_format",
    [
        (1, 6144, 512, "e2m3"),  # decode tile (heuristic (16,64)), 4 K-tiles
        (1, 6144, 512, "e3m2"),  # same tile, other FP6 sub-format
        (3, 6144, 512, "e2m3"),  # MTP verify shape, same tile
        (5, 6144, 512, "e2m3"),  # small-M shape, same decode tile
        (128, 256, 256, "e2m3"),  # small square, 2 K-tiles
        (128, 6144, 512, "e2m3"),  # prefill-ish wide-N tile
    ],
)
def test_b_packed_matches_b_preexpanded(m, n, k, source_format):
    from b12x._lib.dense_gemm import dense_gemm

    fp6w, a_operand, b_sf, common = _gemm_operands(m, n, k, source_format)

    y_ref = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        a_operand,
        (fp6w.expanded_weight(), b_sf),
        out=y_ref,
        b_preexpanded=True,
        **common,
    )
    y_pk = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        a_operand,
        (fp6w.packed.unsqueeze(-1), b_sf),
        out=y_pk,
        b_packed=True,
        **common,
    )
    torch.testing.assert_close(y_pk, y_ref, rtol=0.0, atol=0.0)


@cuda_required
def test_b_packed_rejects_bad_shapes():
    from b12x._lib.dense_gemm import dense_gemm

    fp6w, a_operand, b_sf, common = _gemm_operands(1, 256, 256, "e2m3")
    y = torch.empty((1, 256, 1), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="mutually exclusive"):
        dense_gemm(
            a_operand,
            (fp6w.packed.unsqueeze(-1), b_sf),
            out=y,
            b_packed=True,
            b_preexpanded=True,
            **common,
        )
    with pytest.raises(ValueError, match="3K/4"):
        dense_gemm(
            a_operand,
            (fp6w.expanded_weight(), b_sf),  # wrong: expanded shape with b_packed
            out=y,
            b_packed=True,
            **common,
        )


@cuda_required
def test_dense_gemm_rejects_misaligned_row_scale():
    from b12x._lib.dense_gemm import dense_gemm

    fp6w, a_operand, b_sf, common = _gemm_operands(1, 256, 256, "e2m3")
    out = torch.empty((1, 256, 1), device="cuda", dtype=torch.bfloat16)
    storage = torch.ones(2, device="cuda", dtype=torch.bfloat16)
    row_scale = storage[1:]
    assert row_scale.is_contiguous() and row_scale.data_ptr() % 16 != 0

    with pytest.raises(ValueError, match="row_scale"):
        dense_gemm(
            a_operand,
            (fp6w.expanded_weight(), b_sf),
            out=out,
            b_preexpanded=True,
            row_scale=row_scale,
            **common,
        )


@cuda_required
@pytest.mark.parametrize("missing", ["x", "scale"])
def test_dense_gemm_requires_fused_inputs_as_a_pair(missing):
    from b12x._lib.dense_gemm import dense_gemm

    fp6w, a_operand, b_sf, common = _gemm_operands(1, 256, 256, "e2m3")
    out = torch.empty((1, 256, 1), device="cuda", dtype=torch.bfloat16)
    x = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="provided together"):
        dense_gemm(
            a_operand,
            (fp6w.expanded_weight(), b_sf),
            out=out,
            b_preexpanded=True,
            x_bf16=None if missing == "x" else x,
            w_gscale=None if missing == "scale" else scale,
            **common,
        )
@cuda_required
@pytest.mark.parametrize("m", [1, 3, 128])
def test_linear_autodetects_packed_weight(m):
    """``dense_fp6_linear_expanded`` must route a packed weight via b_packed.

    The serving path passes :meth:`FP6DenseWeight.gemm_weight` (packed for
    wide-N layers) into one shared entry point; format detection from the K
    extent must yield bit-identical results to the expanded path.
    """
    from b12x.quantization.mxfp6.fp6_dense_weights import (
        dense_fp6_linear_expanded,
        quantize_dense_weight_to_fp6,
    )

    torch.manual_seed(0)
    n, k = 6144, 512
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    fp6w = quantize_dense_weight_to_fp6(w_bf16)
    x = (torch.randn(m, k, device="cuda") * 0.1).to(torch.bfloat16)

    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)
    y_exp = dense_fp6_linear_expanded(x, fp6w.expanded_weight(), *args)
    y_pk = dense_fp6_linear_expanded(x, fp6w.packed, *args)
    torch.testing.assert_close(y_pk, y_exp, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="matches neither"):
        dense_fp6_linear_expanded(x, fp6w.packed[:, :-4], *args)


@cuda_required
def test_gemm_weight_selection_threshold(monkeypatch):
    """``gemm_weight`` picks packed iff out_features >= PACKED_GEMM_MIN_N.

    Regression for the selective serving wiring: below the threshold the
    expanded view is returned (and cached); above it the raw packed codes are
    returned, no expansion is materialized, and the end-to-end linear stays
    bit-identical to the expanded path.
    """
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(0)
    n, k = 6144, 512
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    x = (torch.randn(1, k, device="cuda") * 0.1).to(torch.bfloat16)

    fp6w = fdw.quantize_dense_weight_to_fp6(w_bf16)
    monkeypatch.setattr(fdw, "PACKED_GEMM_MIN_N", n + 1)
    assert not fp6w.use_packed_gemm
    assert fp6w.gemm_weight().shape[1] == k
    y_exp = fdw.dense_fp6_linear(x, fp6w)

    fp6w_pk = fdw.quantize_dense_weight_to_fp6(w_bf16)
    monkeypatch.setattr(fdw, "PACKED_GEMM_MIN_N", n)
    assert fp6w_pk.use_packed_gemm
    assert fp6w_pk.gemm_weight().shape[1] == 3 * k // 4
    assert fp6w_pk._packed_expanded is None  # no expanded copy materialized
    y_pk = fdw.dense_fp6_linear(x, fp6w_pk)

    torch.testing.assert_close(y_pk, y_exp, rtol=0.0, atol=0.0)


@cuda_required
def test_custom_op_accepts_packed_weight():
    """The opaque ``b12x::fp6_dense_linear`` op works with a packed weight arg."""
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from b12x.quantization.mxfp6.fp6_dense_weights import quantize_dense_weight_to_fp6

    torch.manual_seed(0)
    n, k = 6144, 512
    w_bf16 = (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    fp6w = quantize_dense_weight_to_fp6(w_bf16)
    x = (torch.randn(1, k, device="cuda") * 0.1).to(torch.bfloat16)

    args = (fp6w.scale_storage, fp6w.global_scale, fp6w.fmt, n, k)
    y_exp = torch.ops.b12x.fp6_dense_linear(x, fp6w.expanded_weight(), *args)
    y_pk = torch.ops.b12x.fp6_dense_linear(x, fp6w.packed, *args)
    torch.testing.assert_close(y_pk, y_exp, rtol=0.0, atol=0.0)
