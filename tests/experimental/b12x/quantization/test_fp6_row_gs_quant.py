"""Exactness tests for large-M per-row FP6 activation quantization."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_TILE = 128


def _host_row_gs(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row recipe from ``dense_fp6_linear_expanded``."""
    from b12x._lib.fp6 import mx_gs_numerator

    amax = x.abs().amax(dim=1, keepdim=True).float()
    # f64 divide + f32 cast == div.rn.f32 (see the host-path comments).
    gs = (mx_gs_numerator(fmt) / amax.clamp_min_(1e-6).double()).float()
    inv_gs = (1.0 / gs.double()).float().to(torch.bfloat16)
    return gs, inv_gs


@cuda_required
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2", "e4m3"])
# k=2048 puts the row scan past one strided pass, so the multi-iteration
# reduction and its tail elements are covered as well as the single-pass case.
@pytest.mark.parametrize("k", [512, 2048])
def test_row_gs_kernel_matches_host_recipe(k, fmt):
    from b12x.quantization.mxfp6.fp6_row_gs import compile_fp6_row_gs

    torch.manual_seed(0)
    m = 256
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.1).to(torch.bfloat16)
    # Pin row amaxes at boundary positions, including the final element, and
    # include an all-zero row for the same clamp path as padded rows.
    x[0, k - 1] = 4.0
    x[7] = 0.0
    x[m - 1, 0] = -2.625  # borderline divisor exercising div.rn vs torch div
    w_gs = torch.tensor([0.5], dtype=torch.float32, device=device)

    gs_ref, inv_gs_ref = _host_row_gs(x, fmt)

    launch = compile_fp6_row_gs(m, k, fmt=fmt)
    gs_out = torch.empty(m, dtype=torch.float32, device=device)
    inv_gs_out = torch.empty(m, dtype=torch.bfloat16, device=device)
    alpha_out = torch.empty(1, dtype=torch.float32, device=device)
    launch(x, w_gs, gs_out, inv_gs_out, alpha_out)

    torch.testing.assert_close(gs_out, gs_ref.view(-1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(inv_gs_out, inv_gs_ref.view(-1), rtol=0.0, atol=0.0)
    # alpha = 1/(1*w_gs): div.rn.f32 == torch.reciprocal (both correctly rounded).
    alpha_ref = torch.reciprocal(
        torch.ones(1, dtype=torch.float32, device=device) * w_gs
    )
    torch.testing.assert_close(alpha_out, alpha_ref, rtol=0.0, atol=0.0)


@cuda_required
@pytest.mark.parametrize("fmt", ["e2m3", "e3m2", "e4m3"])
@pytest.mark.parametrize("m", [128, 256])
def test_per_row_quantizer_matches_host_chain(m, fmt):
    """Fused two-kernel path == host pre-scale + unit-gs TMA quantization."""
    from b12x.quantization.mxfp6.fp6_dense_weights import (
        _quantize_matrix_fp6_bytes,
        _quantize_matrix_fp6_bytes_per_row,
    )

    torch.manual_seed(1)
    k = 512
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.3).to(torch.bfloat16)
    x[3] = 0.0  # zero row: clamp path, mirrors quantizer padding rows
    w_gs = torch.tensor([0.75], dtype=torch.float32, device=device)

    # Independent host formulation.
    gs_ref, inv_gs_ref = _host_row_gs(x, fmt)
    x_pre = (x.float() * gs_ref).to(torch.bfloat16)
    unit = torch.ones(1, dtype=torch.float32, device=device)
    codes_ref, scale_ref = _quantize_matrix_fp6_bytes(x_pre, fmt, unit)
    alpha_ref = torch.reciprocal(unit * w_gs)

    codes, scale, alpha, inv_gs = _quantize_matrix_fp6_bytes_per_row(x, fmt, w_gs)

    torch.testing.assert_close(codes, codes_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        scale.view(-1), scale_ref.view(-1), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(alpha, alpha_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inv_gs, inv_gs_ref.view(-1), rtol=0.0, atol=0.0)


@cuda_required
@pytest.mark.parametrize("m", [17, 128, 130, 256])
def test_large_m_linear_fused_vs_host_chain_bit_exact(m, monkeypatch):
    """``dense_fp6_linear`` output is bit-identical fused vs host chain.

    m=17 covers the small-M/large-M crossover boundary, m=130 the padded
    (m % 128 != 0) case where zero padding rows flow through the fused
    RowGsKernel clamp path.
    """
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(2)
    w = fdw.quantize_dense_weight_to_fp6(
        torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    )
    x = torch.randn(m, w.in_features, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(fdw, "_PER_ROW_IN_KERNEL", False)
    y_host = fdw.dense_fp6_linear(x, w)
    monkeypatch.setattr(fdw, "_PER_ROW_IN_KERNEL", True)
    y_fused = fdw.dense_fp6_linear(x, w)

    torch.testing.assert_close(y_fused, y_host, rtol=0.0, atol=0.0)


@cuda_required
def test_per_row_compile_guards():
    from b12x.quantization.mxfp6 import compile_bf16_to_fp6_tma
    from b12x.quantization.mxfp6.fp6_row_gs import compile_fp6_row_gs

    with pytest.raises(ValueError, match="emit='bytes'"):
        compile_bf16_to_fp6_tma(128, 512, fmt="e3m2", emit="packed", per_row=True)
    with pytest.raises(AssertionError, match="multiple of 128"):
        compile_fp6_row_gs(128, 96)


@cuda_required
def test_row_gs_launch_rejects_misaligned_input():
    from b12x.quantization.mxfp6.fp6_row_gs import compile_fp6_row_gs

    m, k = 128, 512
    storage = torch.empty(m * k + 1, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(m, k)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    w_gs = torch.ones(1, dtype=torch.float32, device="cuda")
    gs = torch.empty(m, dtype=torch.float32, device="cuda")
    inv_gs = torch.empty(m, dtype=torch.bfloat16, device="cuda")
    alpha = torch.empty(1, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="bf16_input"):
        compile_fp6_row_gs(m, k)(x, w_gs, gs, inv_gs, alpha)


@cuda_required
def test_per_row_tma_launch_validates_scale_shape():
    from b12x.quantization.mxfp6 import (
        allocate_bf16_to_fp6_tma_outputs,
        compile_bf16_to_fp6_tma,
    )

    m, k = 128, 512
    x = torch.zeros((m, k), dtype=torch.bfloat16, device="cuda")
    wrong_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    out = allocate_bf16_to_fp6_tma_outputs(m, k, emit="bytes")

    with pytest.raises(ValueError, match="global_scale"):
        compile_bf16_to_fp6_tma(m, k, fmt="e4m3", emit="bytes", per_row=True)(
            x,
            wrong_scale,
            out.packed_a_flat,
            out.scale_flat,
        )
