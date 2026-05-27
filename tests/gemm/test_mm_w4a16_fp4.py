# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the W4A16 FP4 GEMM API.

The cross-backend contract tests sweep over every supported backend
listed in :data:`ALL_BACKENDS`.  When a new backend (cute-dsl, ...) is
added, append its identifier to that list and the full numeric /
behaviour grid runs against it automatically.

Inputs are bf16 throughout; fp16 activations are out of scope for the
current API.
"""

import pytest
import torch

import flashinfer
from flashinfer.gemm.gemm_w4a16 import (
    _dequantize_w4a16_fp4_torch,
    mm_w4a16_fp4,
    prepare_w4a16_fp4_weights,
)


# =============================================================================
# Backend + shape grids
# =============================================================================


# Backends covered by the cross-backend contract tests.  Real backends
# (cute-dsl, ...) get appended here as they land.
ALL_BACKENDS = ["torch"]


# (M, N, K) grid for numerical correctness.  Biased toward small M
# (W4A16's primary use case is decode-shaped GEMMs), with a few medium
# M values to exercise the M-tiling path.  All N are multiples of 128
# so the 128x4 SF swizzle has full (not padded) trailing blocks.
PROBLEM_SIZES = [
    # tiny: smoke / minimum valid shapes
    (1, 128, 128),
    (1, 256, 512),
    (4, 256, 512),
    (16, 256, 256),
    # mid: typical decode at a few model widths
    (1, 1024, 1024),
    (4, 1024, 1024),
    (16, 1024, 1024),
    (64, 1024, 1024),
    # large: realistic model-layer N/K, sweep of M
    (1, 4096, 4096),
    (4, 4096, 4096),
    (16, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
]


# Single mid-size shape used by the secondary contract tests
# (alpha=None, out_dtype override, preallocated out, K-mismatch).  They
# don't need to sweep the full numeric grid -- they only check that the
# behaviour is consistent across backends.
SMOKE_MNK = (16, 1024, 1024)


# Tolerance for the bf16 output vs the fp32-accum reference.  Torch
# backend passes this trivially (it runs the same fp32 path as the
# reference), real backends with different intra-kernel accumulation
# orders pick up at most ~1 bf16 ULP of difference.
ATOL = 5e-3
RTOL = 5e-3


# =============================================================================
# Helpers
# =============================================================================


def _make_random_fp4_weights(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a random matrix to NVFP4 + return (mat2, b_fp4, b_sf, alpha).

    Mirrors the canonical caller pattern: a user does
    ``b_fp4, b_sf = flashinfer.nvfp4_quantize(mat2, g_b, ...)`` and pairs
    that with ``alpha = 1 / g_b``.
    """
    mat2 = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    g_b = (448 * 6) / mat2.float().abs().nan_to_num().max()
    b_fp4, b_sf = flashinfer.nvfp4_quantize(
        mat2,
        g_b,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    alpha = torch.tensor([1.0 / g_b.item()], device=device, dtype=torch.float32)
    return mat2, b_fp4, b_sf, alpha


# =============================================================================
# Backend dispatch
# =============================================================================


def test_unknown_backend_raises_on_prepare():
    """Calling prepare with an unknown backend is an error."""
    device = torch.device("cuda")
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(64, 128, device)
    with pytest.raises(ValueError, match="Unknown backend"):
        prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend="not-a-real-backend")


def test_unknown_backend_raises_on_compute():
    """Calling compute with an unknown backend is an error."""
    device = torch.device("cuda")
    a = torch.randn((4, 128), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(64, 128, device)
    with pytest.raises(ValueError, match="Unknown backend"):
        mm_w4a16_fp4(a, b_fp4, b_sf, alpha, backend="not-a-real-backend")


# =============================================================================
# Cross-backend numerical / behaviour contract
# =============================================================================


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("m,n,k", PROBLEM_SIZES)
def test_backend_matches_handwritten_dequant_matmul(backend, m, n, k):
    """Backend output must match a hand-rolled fp32 dequant + matmul.

    Reference = ``(a.float() @ dequant(b).T).to(bf16)``.  Every backend
    is expected to produce numerically equivalent output (up to ~1 bf16
    ULP).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = mm_w4a16_fp4(a, b_p, sf_p, alpha_p, backend=backend)

    weight_fp32 = _dequantize_w4a16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    ref = (a.float() @ weight_fp32.T).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_alpha_none_equals_alpha_one(backend):
    """alpha=None must produce identical output to alpha=tensor([1.0])."""
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, _ = _make_random_fp4_weights(n, k, device)

    b1, sf1, a1 = prepare_w4a16_fp4_weights(
        b_fp4,
        b_sf,
        torch.ones(1, device=device, dtype=torch.float32),
        backend=backend,
    )
    out_one = mm_w4a16_fp4(a, b1, sf1, a1, backend=backend)

    b0, sf0, a0 = prepare_w4a16_fp4_weights(b_fp4, b_sf, None, backend=backend)
    out_none = mm_w4a16_fp4(a, b0, sf0, a0, backend=backend)

    torch.testing.assert_close(out_none, out_one, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_out_dtype_override(backend):
    """out_dtype kwarg controls return dtype independently of a.dtype."""
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = mm_w4a16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out_dtype=torch.float16,
    )
    assert out.dtype == torch.float16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_preallocated_out(backend):
    """Caller-provided out tensor is written in place."""
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    out_ptr_before = out.data_ptr()
    returned = mm_w4a16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out=out,
    )
    assert returned.data_ptr() == out_ptr_before
    ref = mm_w4a16_fp4(a, b_p, sf_p, alpha_p, backend=backend)
    torch.testing.assert_close(returned, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_shape_mismatch_raises(backend):
    """K of a must match K inferred from prepared b."""
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    a_wrong_k = torch.randn((m, k * 2), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        mm_w4a16_fp4(a_wrong_k, b_p, sf_p, alpha_p, backend=backend)


# =============================================================================
# Dispatcher-level input validation
# =============================================================================
#
# These checks fire before any backend-specific code runs, so they're
# not parametrized over backend.


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float16])
def test_a_dtype_must_be_bfloat16(bad_dtype):
    """Only bfloat16 activations are supported (fp16 deferred)."""
    device = torch.device("cuda")
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(64, 128, device)
    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend="torch")
    a_bad = torch.randn((4, 128), device=device, dtype=bad_dtype)
    with pytest.raises(TypeError):
        mm_w4a16_fp4(a_bad, b_p, sf_p, alpha_p, backend="torch")


def test_b_dtype_must_be_uint8_in_prepare():
    """Prepare rejects non-uint8 B."""
    device = torch.device("cuda")
    b_bad = torch.zeros((64, 64), device=device, dtype=torch.int32)
    b_descale = torch.zeros((4096,), device=device, dtype=torch.uint8)
    with pytest.raises(TypeError):
        prepare_w4a16_fp4_weights(b_bad, b_descale, None, backend="torch")


def test_alpha_dtype_must_be_float32():
    """Prepare rejects non-fp32 alpha."""
    device = torch.device("cuda")
    _, b_fp4, b_sf, _ = _make_random_fp4_weights(64, 128, device)
    alpha_bad = torch.ones(1, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha_bad, backend="torch")
