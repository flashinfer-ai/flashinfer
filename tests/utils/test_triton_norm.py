"""
Copyright (c) 2024 by FlashInfer team.

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

triton = pytest.importorskip("triton")

from flashinfer.triton.kernels.norm import rms_norm_kernel  # noqa: E402
from flashinfer.triton.norm import (  # noqa: E402
    _SINGLE_PASS_MAX_HIDDEN,
    rms_norm,
    rms_norm_add_residual,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Triton rms_norm requires a CUDA device",
    ),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        reason="The supported Triton version requires SM80 or newer (bf16 cases "
        "also need Ampere+); Turing/SM75 is out of range",
    ),
]


def torch_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # Match the kernel: cast to the narrow dtype only after the weight multiply.
    x = (x * w.float()).to(orig_dtype)
    return x


def _general_kernel_rms_norm(x, w, eps):
    """Invoke the general two-pass kernel directly (bypassing the fast-path
    dispatch) to obtain the reference the fast path must match bit-for-bit."""
    b, n = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n)
    num_warps = max(8, min(32, block_size // 256))
    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=None,
        r_ptr=None,
        r_stride=0,
        w_ptr=w,
        o_ptr=out,
        o_stride=out.stride(0),
        o_scale_ptr=None,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=False,
        HAS_OUT_SCALE=False,
        HAS_OUTPUT=True,
        HAS_RESIDUAL=False,
        num_warps=num_warps,
    )
    return out


@pytest.mark.parametrize("batch_size", [1, 4, 16, 128, 4096])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 3072, 4096, 5120, 7168, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_fastpath(batch_size, hidden_size, dtype):
    """The single-load fast path is numerically correct and bit-identical to
    the general two-pass kernel."""
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    out = torch.empty_like(x)
    rms_norm(x, w, out, eps)

    # Correctness against a plain torch reference.
    ref = torch_rms_norm(x, w, eps)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    # Bit-exact equivalence with the general kernel it replaces.
    general = _general_kernel_rms_norm(x, w, eps)
    assert torch.equal(out, general), "fast path diverged from general kernel"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rms_norm_large_hidden_fallback(dtype):
    """Hidden dims above the fast-path cap fall back to the general kernel and
    stay correct."""
    torch.manual_seed(0)
    eps = 1e-6
    hidden_size = _SINGLE_PASS_MAX_HIDDEN * 2  # 16384, above the cap
    x = torch.randn(8, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    out = torch.empty_like(x)
    rms_norm(x, w, out, eps)

    ref = torch_rms_norm(x, w, eps)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("which", ["x", "out", "weight"])
def test_triton_rms_norm_noncontiguous_rejected(which):
    """A non-contiguous hidden dimension (inner stride != 1) is rejected: both
    kernels index the hidden dim with a bare element offset and would otherwise
    read the wrong elements."""
    torch.manual_seed(0)
    eps = 1e-6
    dtype = torch.float16
    batch_size, hidden_size = 16, 4096

    def strided_2d():
        # Slice out every other column so the inner stride is 2.
        base = torch.randn(batch_size, hidden_size * 2, dtype=dtype, device="cuda")
        return base[:, ::2]

    def strided_1d():
        base = torch.randn(hidden_size * 2, dtype=dtype, device="cuda")
        return base[::2]

    x = (
        strided_2d()
        if which == "x"
        else torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    )
    out = strided_2d() if which == "out" else torch.empty_like(x)
    w = (
        strided_1d()
        if which == "weight"
        else torch.randn(hidden_size, dtype=dtype, device="cuda")
    )

    assert x.stride(1) != 1 or out.stride(1) != 1 or w.stride(0) != 1
    with pytest.raises(ValueError):
        rms_norm(x, w, out, eps)


def test_triton_rms_norm_residual_unaffected():
    """The fused residual path is untouched by the fast-path dispatch."""
    torch.manual_seed(0)
    eps = 1e-6
    batch_size, hidden_size = 16, 4096
    dtype = torch.float16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    # Reference: r = r + x; out = rmsnorm(r)
    r_ref = (residual.float() + x.float()).to(dtype)
    out_ref = torch_rms_norm(r_ref, w, eps)

    x_work = x.clone()
    r_work = residual.clone()
    out = torch.empty_like(x)
    rms_norm_add_residual(x_work, r_work, w, eps, x_out=out)

    torch.testing.assert_close(r_work, r_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_triton_rms_norm_fastpath(16, 4096, torch.float16)
