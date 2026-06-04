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
    _CUDNN_W4A16_MIN_BACKEND_VERSION,
    _dequantize_w4a16_fp4_torch,
    mm_w4a16_fp4,
    prepare_w4a16_fp4_weights,
)
from flashinfer.utils import get_compute_capability


# =============================================================================
# Backend + shape grids
# =============================================================================


# Backends covered by the cross-backend contract tests.  Real backends
# (cute-dsl, ...) get appended here as they land.
ALL_BACKENDS = ["torch", "cudnn", "cute-dsl"]


# Each backend consumes a specific SF layout (see ``is_sf_swizzled`` on
# ``mm_w4a16_fp4``): the torch reference unswizzles internally so it needs
# the canonical 128x4-swizzled SF, while cuDNN requires a non-swizzled
# (linear) SF.  The cute-DSL backend takes a fully bespoke layout produced
# by ``prepare_w4a16_fp4_weights`` and ignores the flag.
# ``prepare_w4a16_fp4_weights`` already emits the right layout per backend;
# this maps the matching ``is_sf_swizzled`` flag.
_IS_SF_SWIZZLED = {"torch": True, "cudnn": False, "cute-dsl": False}


def _sf_swizzled(backend: str) -> bool:
    return _IS_SF_SWIZZLED[backend]


def _skip_if_backend_unavailable(backend: str) -> None:
    """Skip the current test if ``backend`` can't run on this device.

    Gates on compute-capability support (via the ``@backend_requirement``
    introspection) plus runtime availability that the cc check doesn't
    cover -- notably the cuDNN W4A16 minimum backend version.
    """
    device = torch.device("cuda")
    cc = get_compute_capability(device)
    cc_number = cc[0] * 10 + cc[1]
    if not mm_w4a16_fp4.is_backend_supported(backend, cc_number):
        pytest.skip(f"{backend} not supported on compute capability {cc_number}")
    if backend == "cudnn":
        try:
            import cudnn
        except ImportError:
            pytest.skip("cuDNN not available")
        if cudnn.backend_version() < _CUDNN_W4A16_MIN_BACKEND_VERSION:
            pytest.skip(
                f"cuDNN W4A16 needs backend >= {_CUDNN_W4A16_MIN_BACKEND_VERSION}, "
                f"found {cudnn.backend_version()}"
            )


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


# Tolerance for the backend output vs the fp32-accum reference.
#
# * ``torch`` runs the exact same fp32 dequant + matmul as the reference, so
#   it matches to a very tight tolerance.
# * ``cudnn`` dequantizes the FP4 weight to bf16 (to feed the bf16 tensor-core
#   matmul; accumulation is still fp32), so its weight carries ~1 bf16 ULP of
#   rounding vs the fp32 reference.  That shows up as up to ~2^-7 (= 0.0078)
#   relative error on a small fraction of output elements, which exceeds the
#   torch tolerance but is expected for a true W4A16 (bf16-activation) kernel.
ATOL = 5e-3
RTOL = 5e-3

_BACKEND_TOL = {
    "torch": {"atol": 5e-3, "rtol": 5e-3},
    # cute-dsl, like cudnn, dequantizes the FP4 weight to bf16 before the
    # tensor-core matmul, so it carries ~1 bf16 ULP of weight rounding vs
    # the fp32 reference -- same tolerance budget as cudnn.
    "cudnn": {"atol": 1.5e-2, "rtol": 1.5e-2},
    "cute-dsl": {"atol": 1.5e-2, "rtol": 1.5e-2},
}


def _tol(backend: str) -> dict:
    return _BACKEND_TOL.get(backend, {"atol": ATOL, "rtol": RTOL})


def _assert_close_to_reference(out: torch.Tensor, ref: torch.Tensor, backend: str):
    """Compare a backend's output against the fp32-accurate reference.

    The ``torch`` backend runs the exact same fp32 dequant + matmul as the
    reference, so it is checked elementwise to a tight tolerance.  The
    ``cudnn`` / ``cute-dsl`` backends dequantize the FP4 weight to bf16
    before a bf16 tensor-core matmul, so a handful of large-magnitude
    output elements carry bf16 rounding that exceeds any sane elementwise
    bound at large K.  For those we validate the relative L2-norm of the
    error (catches decorrelation *and* scale/alpha bugs -- cosine alone is
    scale-invariant) together with cosine similarity, mirroring
    ``tests/gemm/test_mm_fp4.py``.
    """
    if backend == "torch":
        torch.testing.assert_close(out, ref, **_tol(backend))
        return
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
    rel_l2 = (torch.linalg.vector_norm(out_f - ref_f) / ref_norm).item()
    cos = torch.nn.functional.cosine_similarity(out_f, ref_f, dim=0).item()
    assert rel_l2 < 2e-2, f"{backend}: relative L2 error {rel_l2:.4f} exceeds 2e-2"
    assert cos > 0.999, f"{backend}: cosine similarity {cos:.6f} below 0.999"


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
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = mm_w4a16_fp4(
        a, b_p, sf_p, alpha_p, backend=backend, is_sf_swizzled=_sf_swizzled(backend)
    )

    weight_fp32 = _dequantize_w4a16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    ref = (a.float() @ weight_fp32.T).to(torch.bfloat16)

    _assert_close_to_reference(out, ref, backend)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_alpha_none_equals_alpha_one(backend):
    """alpha=None must produce identical output to alpha=tensor([1.0])."""
    _skip_if_backend_unavailable(backend)
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
    out_one = mm_w4a16_fp4(
        a, b1, sf1, a1, backend=backend, is_sf_swizzled=_sf_swizzled(backend)
    )

    b0, sf0, a0 = prepare_w4a16_fp4_weights(b_fp4, b_sf, None, backend=backend)
    out_none = mm_w4a16_fp4(
        a, b0, sf0, a0, backend=backend, is_sf_swizzled=_sf_swizzled(backend)
    )

    torch.testing.assert_close(out_none, out_one, **_tol(backend))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_out_dtype_override(backend):
    """out_dtype kwarg controls return dtype independently of a.dtype."""
    _skip_if_backend_unavailable(backend)
    if backend == "cute-dsl":
        # The cute-dsl kernel's MMA path requires out_dtype == a.dtype, so
        # it cannot emit fp16 from a bf16 activation (see _compute_cute_dsl).
        pytest.skip("cute-dsl requires out_dtype == a.dtype")
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
        is_sf_swizzled=_sf_swizzled(backend),
    )
    assert out.dtype == torch.float16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_preallocated_out(backend):
    """Caller-provided out tensor is written in place."""
    _skip_if_backend_unavailable(backend)
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
        is_sf_swizzled=_sf_swizzled(backend),
    )
    assert returned.data_ptr() == out_ptr_before
    ref = mm_w4a16_fp4(
        a, b_p, sf_p, alpha_p, backend=backend, is_sf_swizzled=_sf_swizzled(backend)
    )
    torch.testing.assert_close(returned, ref, **_tol(backend))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_shape_mismatch_raises(backend):
    """K of a must match K inferred from prepared b."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    _, b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_w4a16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    a_wrong_k = torch.randn((m, k * 2), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        mm_w4a16_fp4(
            a_wrong_k,
            b_p,
            sf_p,
            alpha_p,
            backend=backend,
            is_sf_swizzled=_sf_swizzled(backend),
        )


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
