# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 x FP4 GEMM API ``mm_bf16_fp4``."""

import pytest
import torch

import flashinfer
from flashinfer import mm_bf16_fp4, prepare_bf16_fp4_weights
from flashinfer.autotuner import autotune
from flashinfer.gemm.gemm_bf16_fp4 import (
    _CUDNN_BF16_FP4_MIN_BACKEND_VERSION,
    _unswizzle_sf_128x4,
)
from flashinfer.utils import get_compute_capability


# E2M1 (FP4) value table, signed (codes 0-7 positive, 8-15 negative), matching
# ``flashinfer.nvfp4_quantize``.
_E2M1_VALUES_FP32 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def _dequantize_bf16_fp4_torch(b, b_descale, alpha, n, k, block_size):
    """PyTorch implementation of swizzled nvfp4 dequantization to fp32."""
    device = b.device
    k_sf = k // block_size
    lut = torch.tensor(_E2M1_VALUES_FP32, dtype=torch.float32, device=device)
    b_int = b.to(torch.int64)
    codes = torch.stack([b_int & 0xF, (b_int >> 4) & 0xF], dim=-1).reshape(n, k)
    values = lut[codes]
    sf = _unswizzle_sf_128x4(b_descale, n, k_sf).view(torch.float8_e4m3fn)
    sf_expanded = sf.to(torch.float32).repeat_interleave(block_size, dim=1)
    weight = values * sf_expanded
    if alpha is not None:
        weight = weight * alpha.to(torch.float32)
    return weight


# =============================================================================
# Backend + shape grids
# =============================================================================


# Backends covered by the cross-backend contract tests.  New backends get
# appended here as they land.
ALL_BACKENDS = ["cudnn", "cute-dsl"]


def _skip_if_backend_unavailable(backend: str) -> None:
    """Skip the current test if ``backend`` can't run on this device."""
    device = torch.device("cuda")
    cc = get_compute_capability(device)
    cc_number = cc[0] * 10 + cc[1]
    if not mm_bf16_fp4.is_backend_supported(backend, cc_number):
        pytest.skip(f"{backend} not supported on compute capability {cc_number}")
    if backend == "cudnn":
        try:
            import cudnn
        except ImportError:
            pytest.skip("cuDNN not available")
        if cudnn.backend_version() < _CUDNN_BF16_FP4_MIN_BACKEND_VERSION:
            pytest.skip(
                f"cuDNN bf16 x fp4 needs backend >= {_CUDNN_BF16_FP4_MIN_BACKEND_VERSION}, "
                f"found {cudnn.backend_version()}"
            )


def _skip_if_compute_capability_unsupported() -> None:
    """Skip the current test if no bf16 x fp4 backend supports this device."""
    cc = get_compute_capability(torch.device("cuda"))
    cc_number = cc[0] * 10 + cc[1]
    if not mm_bf16_fp4.is_backend_supported("cudnn", cc_number):
        pytest.skip(f"mm_bf16_fp4 not supported on compute capability {cc_number}")


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
    # ---- non-power-of-2 shapes (N, K multiples of 64; mixed tile_K=64/128) ----
    # small M (decode), odd M and odd N/K
    (1, 192, 192),
    (2, 320, 256),
    (3, 448, 320),
    (5, 576, 192),
    (7, 704, 256),
    (11, 832, 384),
    (13, 960, 512),
    (17, 1088, 576),
    (6, 1216, 640),
    (9, 1344, 768),
    (1, 2112, 1024),
    (4, 2688, 1344),
    (15, 1600, 896),
    # mid M
    (48, 192, 1024),
    (96, 320, 768),
    (100, 576, 1152),
    (127, 704, 960),
    (192, 832, 1024),
    (200, 1088, 1280),
    (250, 1216, 1024),
    (160, 2560, 2112),
    (96, 3072, 1344),
    (48, 1856, 1536),
    # large M (prefill)
    (384, 1088, 1024),
    (500, 1344, 1152),
    (768, 2112, 2048),
    (1000, 1600, 1536),
    (1500, 2560, 1024),
    (2048, 2688, 2688),
    (3000, 1088, 768),
    (1024, 4160, 2048),
    (640, 6144, 1024),
    (2000, 3200, 1024),
    # skinny / wide extremes
    (1, 5120, 256),
    (7, 4160, 192),
    (13, 11008, 128),
    (64, 2112, 2112),
    (256, 832, 1344),
    (333, 1600, 640),
    (17, 3072, 3072),
]

# Default shape for API sanity tests.
# (alpha=None, out_dtype override, preallocated out, K-mismatch).
SMOKE_MNK = (16, 1024, 1024)

ATOL = 1.5e-2
RTOL = 1.5e-2


def _assert_close_to_reference(out: torch.Tensor, ref: torch.Tensor, backend: str):
    """Compare a backend's output against the fp32-accurate reference."""
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a random matrix to NVFP4 + return (b_fp4, b_sf, alpha).

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
    return b_fp4, b_sf, alpha


# =============================================================================
# Cross-backend numerical / behaviour contract
# =============================================================================


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("m,n,k", PROBLEM_SIZES)
def test_backend_matches_handwritten_dequant_matmul(auto_tuning, backend, m, n, k):
    """Backend output must match a hand-rolled fp32 dequant + matmul.

    Reference = ``(a.float() @ dequant(b).T).to(bf16)``.  Every backend
    is expected to produce numerically equivalent output (up to ~1 bf16
    ULP).  Run with ``auto_tuning`` both off (fallback tactic) and on
    (so the autotuner's selected tactic is exercised too).
    """
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    with autotune(auto_tuning):
        out = mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend=backend)

    weight_fp32 = _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    ref = (a.float() @ weight_fp32.T).to(torch.bfloat16)

    _assert_close_to_reference(out, ref, backend)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_backend_alpha_none_equals_alpha_one(auto_tuning, backend):
    """alpha=None must produce identical output to alpha=tensor([1.0])."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, _ = _make_random_fp4_weights(n, k, device)

    b1, sf1, a1 = prepare_bf16_fp4_weights(
        b_fp4,
        b_sf,
        torch.ones(1, device=device, dtype=torch.float32),
        backend=backend,
    )
    b0, sf0, a0 = prepare_bf16_fp4_weights(b_fp4, b_sf, None, backend=backend)
    with autotune(auto_tuning):
        out_one = mm_bf16_fp4(a, b1, sf1, a1, backend=backend)
        out_none = mm_bf16_fp4(a, b0, sf0, a0, backend=backend)

    torch.testing.assert_close(out_none, out_one, atol=ATOL, rtol=RTOL)


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
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = mm_bf16_fp4(
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
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    out_ptr_before = out.data_ptr()
    returned = mm_bf16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out=out,
    )
    assert returned.data_ptr() == out_ptr_before
    ref = mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend=backend)
    torch.testing.assert_close(returned, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_shape_mismatch_raises(backend):
    """K of a must match K inferred from prepared b."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    a_wrong_k = torch.randn((m, k * 2), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        mm_bf16_fp4(
            a_wrong_k,
            b_p,
            sf_p,
            alpha_p,
            backend=backend,
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
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_fp4, b_sf, alpha = _make_random_fp4_weights(64, 128, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    a_bad = torch.randn((4, 128), device=device, dtype=bad_dtype)
    with pytest.raises(TypeError):
        mm_bf16_fp4(a_bad, b_p, sf_p, alpha_p, backend="cute-dsl")


def test_b_dtype_must_be_uint8_in_prepare():
    """Prepare rejects non-uint8 B."""
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_bad = torch.zeros((64, 64), device=device, dtype=torch.int32)
    b_descale = torch.zeros((4096,), device=device, dtype=torch.uint8)
    with pytest.raises(TypeError):
        prepare_bf16_fp4_weights(b_bad, b_descale, None, backend="cute-dsl")


def test_alpha_dtype_must_be_float32():
    """Prepare rejects non-fp32 alpha."""
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_fp4, b_sf, _ = _make_random_fp4_weights(64, 128, device)
    alpha_bad = torch.ones(1, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        prepare_bf16_fp4_weights(b_fp4, b_sf, alpha_bad, backend="cute-dsl")
