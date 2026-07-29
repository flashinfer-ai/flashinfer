"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cute_dsl import is_cute_dsl_available


def _is_sm12x_supported():
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(),
    reason="Requires SM120/SM121 GPU with CUDA 12.8+",
)

pytestmark = [cute_dsl_available, sm120_required]


def _reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x.float(), w.float())


def _check(y: torch.Tensor, ref: torch.Tensor):
    # f32 accumulation in a different reduction order than cuBLAS: results
    # match the fp32 reference to bf16 rounding, not bitwise.
    torch.testing.assert_close(y.float(), ref, rtol=1.6e-2, atol=1e-3)


@pytest.mark.parametrize("m", [1, 2, 4, 7, 8])
@pytest.mark.parametrize(
    "n,k",
    [
        (96, 5120),  # real target-model projection
        (64, 2048),  # real target-model projection
        (1, 1024),  # degenerate single-column output
        (128, 2048),  # routing-window edge (BF16_GEMV_MAX_N)
    ],
)
def test_bf16_gemv_kernel_path(m, n, k, monkeypatch):
    from flashinfer import bf16_gemv
    from flashinfer.gemm import gemm_bf16_gemv as mod

    def no_fallback(*args, **kwargs):
        raise AssertionError("kernel-eligible shape took the cuBLAS fallback")

    monkeypatch.setattr(mod, "_cublas_fallback", no_fallback)
    torch.manual_seed(42)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    y = bf16_gemv(x, w)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    _check(y, _reference(x, w))


@pytest.mark.parametrize("m", [9, 16, 64])
def test_bf16_gemv_fallback_m_above_max(m):
    """m > SMALL_M_MAX must take the cuBLAS fallback and stay correct."""
    from flashinfer import bf16_gemv
    from flashinfer.gemm.gemm_bf16_gemv import BF16_GEMV_SMALL_M_MAX

    assert m > BF16_GEMV_SMALL_M_MAX
    torch.manual_seed(0)
    x = torch.randn(m, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(96, 2048, dtype=torch.bfloat16, device="cuda") * 0.05
    y = bf16_gemv(x, w)
    # The fallback is F.linear itself, so it must match bitwise.
    torch.testing.assert_close(y, F.linear(x, w), rtol=0, atol=0)


def test_bf16_gemv_fallback_odd_k():
    """K not a multiple of 8 must take the cuBLAS fallback."""
    from flashinfer import bf16_gemv

    torch.manual_seed(0)
    x = torch.randn(2, 2052, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(64, 2052, dtype=torch.bfloat16, device="cuda") * 0.05
    y = bf16_gemv(x, w)
    torch.testing.assert_close(y, F.linear(x, w), rtol=0, atol=0)


def test_bf16_gemv_noncontiguous_x():
    """A non-contiguous x is made contiguous before the kernel launch."""
    from flashinfer import bf16_gemv

    torch.manual_seed(0)
    x_wide = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    x = x_wide[:, ::2]  # (4, 2048), non-contiguous
    assert not x.is_contiguous()
    w = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda") * 0.05
    y = bf16_gemv(x, w)
    _check(y, _reference(x.contiguous(), w))


def test_bf16_gemv_out_param():
    from flashinfer import bf16_gemv

    torch.manual_seed(0)
    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda") * 0.05
    out = torch.empty(2, 64, dtype=torch.bfloat16, device="cuda")
    y = bf16_gemv(x, w, out=out)
    assert y is out
    _check(out, _reference(x, w))


def test_bf16_gemv_empty_n_falls_back():
    from flashinfer import bf16_gemv

    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(0, 2048, dtype=torch.bfloat16, device="cuda")
    y = bf16_gemv(x, w)
    assert y.shape == (2, 0)


def test_bf16_gemv_empty_k_falls_back():
    """K=0 is a valid degenerate linear and must not reach kernel compilation."""
    from flashinfer import bf16_gemv

    x = torch.empty(2, 0, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(64, 0, dtype=torch.bfloat16, device="cuda")
    y = bf16_gemv(x, w)
    torch.testing.assert_close(y, F.linear(x, w), rtol=0, atol=0)


def test_bf16_gemv_misaligned_out_falls_back():
    """A contiguous but 16-byte-misaligned out must take the cuBLAS fallback."""
    from flashinfer import bf16_gemv

    torch.manual_seed(0)
    m, n, k = 2, 64, 2048
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    buf = torch.empty(m * n + 8, dtype=torch.bfloat16, device="cuda")
    out = buf[1 : 1 + m * n].view(m, n)
    assert out.is_contiguous() and out.data_ptr() % 16 != 0
    y = bf16_gemv(x, w, out=out)
    assert y is out
    # Match the fallback op on an identically misaligned buffer bitwise:
    # matmul picks a different (slightly less accurate) path for unaligned
    # outputs, so a plain fp32-reference check would be flaky here.
    buf2 = torch.empty(m * n + 8, dtype=torch.bfloat16, device="cuda")
    expected = torch.matmul(x, w.t(), out=buf2[1 : 1 + m * n].view(m, n))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_bf16_gemv_rejects_bad_out():
    from flashinfer import bf16_gemv

    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    bad_shape = torch.empty(64, 2, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="out"):
        bf16_gemv(x, w, out=bad_shape)
    bad_dtype = torch.empty(2, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="out"):
        bf16_gemv(x, w, out=bad_dtype)


def test_bf16_gemv_window():
    """The routing window matches the measured win region boundaries."""
    from flashinfer.gemm.gemm_bf16_gemv import bf16_gemv_window

    # Real projection shapes: full m range at N <= 64, m <= 4 above it.
    assert bf16_gemv_window(1, 64, 2048)
    assert bf16_gemv_window(8, 64, 2048)
    assert bf16_gemv_window(4, 96, 5120)
    assert not bf16_gemv_window(8, 96, 5120)
    assert bf16_gemv_window(4, 128, 2048)
    assert not bf16_gemv_window(5, 128, 2048)
    # Outside: wide N, short K, m out of kernel range.
    assert not bf16_gemv_window(2, 256, 2048)
    assert not bf16_gemv_window(2, 64, 512)
    assert not bf16_gemv_window(9, 64, 2048)
    assert not bf16_gemv_window(0, 64, 2048)


def test_bf16_gemv_rejects_bad_dtype():
    from flashinfer import bf16_gemv

    x = torch.randn(2, 2048, dtype=torch.float16, device="cuda")
    w = torch.randn(64, 2048, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="bfloat16"):
        bf16_gemv(x, w)


def test_bf16_gemv_rejects_k_mismatch():
    from flashinfer import bf16_gemv

    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="mismatch"):
        bf16_gemv(x, w)


def test_bf16_gemv_precompile_and_cuda_graph():
    """precompile all decode-m variants, then capture and replay a graph."""
    from flashinfer import bf16_gemv, precompile_bf16_gemv
    from flashinfer.gemm.gemm_bf16_gemv import BF16_GEMV_SMALL_M_MAX

    torch.manual_seed(7)
    n, k = 64, 2048
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    precompile_bf16_gemv(w)

    for m in (1, BF16_GEMV_SMALL_M_MAX):
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bf16_gemv(x, w, out=out)

        x.copy_(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
        g.replay()
        torch.cuda.synchronize()
        _check(out, _reference(x, w))


def test_bf16_gemv_uncompiled_shape_during_capture_falls_back():
    """A never-seen (m, n, k) inside graph capture must not JIT mid-capture."""
    from flashinfer import bf16_gemv

    torch.manual_seed(7)
    n, k = 32, 1024  # deliberately not precompiled elsewhere in this file
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    x = torch.randn(3, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(3, n, dtype=torch.bfloat16, device="cuda")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        bf16_gemv(x, w, out=out)

    x.copy_(torch.randn(3, k, dtype=torch.bfloat16, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    _check(out, _reference(x, w))
