"""Tests for the ``b12x::fp6_dense_linear`` torch custom op.

Validates that the opaque op matches the eager :func:`dense_fp6_linear` path
bit-for-bit and that ``torch.compile(fullgraph=True)`` traces a model using the
op without graph breaks — the property vLLM's VLLM_COMPILE mode relies on.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _make_weight(n: int = 256, k: int = 256):
    from b12x.quantization.mxfp6.fp6_dense_weights import quantize_dense_weight_to_fp6

    torch.manual_seed(0)
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    return quantize_dense_weight_to_fp6(w_bf16)


def _op_args(w, plan):
    return (
        w.expanded_weight(),
        w.scale_storage,
        w.global_scale,
        w.fmt,
        w.out_features,
        w.in_features,
        plan.handle,
        w.act_fmt,
    )


@contextmanager
def _prepared_execution(w, m: int):
    from b12x.preparation import PreparedCall, PreparationSession
    from b12x.quantization.mxfp6 import Mxfp6DenseQuery, plan
    from b12x.quantization.mxfp6.fp6_dense_weights import _DENSE_PER_ROW_GS

    declaration = plan(Mxfp6DenseQuery(
        max_tokens=m, in_features=w.in_features, out_features=w.out_features,
        weight_format=w.fmt, activation_format=w.act_fmt,
        weight_storage="expanded", global_scale_kind="multiplier",
        output_mode="functional", per_row_global_scale=_DENSE_PER_ROW_GS,
    ))
    source = torch.zeros((m, w.in_features), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, w.out_features), dtype=torch.bfloat16, device="cuda")
    request = declaration.request(
        name=f"fp6-op-{m}",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                source, w.expanded_weight(), w.scale_storage, w.global_scale,
                out=output.unsqueeze(-1),
            ),
            output=output, owners=(source, output),
        ),
    )
    with PreparationSession(device=source.device, autotune=False) as session:
        session.prepare((request,))
        yield declaration


@cuda_required
def test_custom_op_matches_eager():
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from b12x.quantization.mxfp6.fp6_dense_weights import dense_fp6_linear

    w = _make_weight()
    for m in (1, 3, 128):
        x = torch.randn(m, w.in_features, dtype=torch.bfloat16, device="cuda")
        ref = dense_fp6_linear(x, w)
        with _prepared_execution(w, m) as plan:
            got = torch.ops.b12x.fp6_dense_linear(x, *_op_args(w, plan))
            assert got.shape == (m, w.out_features)
            assert got.dtype == torch.bfloat16
            torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


@cuda_required
def test_small_m_matches_padded_rows():
    """The small-M decode tile must agree with the (reference-validated) m=128 path.

    Rows are independent in the GEMM, so y(x[:m])[i] must be bit-identical to
    y(x)[i]. The activation amax is pinned to row 0 so every slice quantizes
    with the same global scale; m=1 hits the decode tile (heuristic (16,64) at
    this N), m=3 the MTP verify shape, m=5 a non-tile-multiple m.
    """
    from b12x.quantization.mxfp6.fp6_dense_weights import dense_fp6_linear

    w = _make_weight()
    x = torch.randn(128, w.in_features, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 8.0  # amax lives in row 0 -> identical a_gs for all slices
    y_full = dense_fp6_linear(x, w)
    for m in (1, 3, 5):
        y_small = dense_fp6_linear(x[:m], w)
        torch.testing.assert_close(y_small, y_full[:m], rtol=0.0, atol=0.0)


@cuda_required
def test_bytes_quantizer_matches_packed_expand():
    """``emit="bytes"`` must equal the packed quantizer's output, just unpacked.

    The bytes mode shares the entire quantization path (UE8M0 scale, cvt) with
    the packed mode and differs only in the store packing, so codes must match
    ``expand_mxfp6_packed_to_bytes(packed)`` bit-for-bit, and scales must be
    identical.
    """
    from b12x._lib.fp6 import expand_mxfp6_packed_to_bytes
    from b12x.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6,
        _quantize_matrix_fp6_bytes,
    )

    torch.manual_seed(0)
    m, k = 128, 256
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    f8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    a_gs = (f8_max * 28.0 / x.abs().amax().to(torch.float32)).reshape(1)
    for fmt in ("e3m2", "e2m3"):
        packed, scale_packed = _quantize_matrix_fp6(x, fmt, a_gs)
        codes, scale_bytes = _quantize_matrix_fp6_bytes(x, fmt, a_gs)
        assert codes.shape == (m, k)
        expected = expand_mxfp6_packed_to_bytes(packed, k)
        torch.testing.assert_close(codes, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(scale_bytes, scale_packed, rtol=0.0, atol=0.0)


@cuda_required
def test_custom_op_compile_fullgraph():
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401

    w = _make_weight()
    with _prepared_execution(w, 1) as plan:
        args = _op_args(w, plan)

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.b12x.fp6_dense_linear(x, *args)

        x = torch.randn(1, w.in_features, dtype=torch.bfloat16, device="cuda")
        eager = fn(x)
        compiled = torch.compile(fn, fullgraph=True)
        got = compiled(x)
        torch.testing.assert_close(got, eager, rtol=0.0, atol=0.0)


@cuda_required
def test_custom_op_fake_tensor_shape():
    import b12x.quantization.mxfp6.fp6_dense_op  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode

    w = _make_weight()
    with _prepared_execution(w, 7) as plan:
        args = _op_args(w, plan)
        with FakeTensorMode() as mode:
            fx = mode.from_tensor(
                torch.empty(7, w.in_features, dtype=torch.bfloat16),
            )
            fargs = [
                mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
            out = torch.ops.b12x.fp6_dense_linear(fx, *fargs)
            assert tuple(out.shape) == (7, w.out_features)
            assert out.dtype == torch.bfloat16
