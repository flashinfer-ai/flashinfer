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

from flashinfer.gemm import w8a16_fp8_gemm, w8a16_fp8_gemm_workspace_bytes
from flashinfer.utils import get_compute_capability


@pytest.fixture(autouse=True)
def _needs_hopper_or_blackwell():
    """The op has Hopper (sm_90a) and Blackwell (sm_100a) builds only."""
    if get_compute_capability(torch.device("cuda"))[0] not in (9, 10):
        pytest.skip("w8a16_fp8_gemm needs Hopper or Blackwell")


def _quantize(w32: torch.Tensor):
    scale = (w32.abs().amax() / 448.0).to(torch.float32).reshape(1)
    return (w32 / scale).to(torch.float8_e4m3fn), scale


def _cast_copy_reference(act, w_fp8, scale, bias):
    """TRT-LLM W8A16FP8LinearMethod.apply, verbatim semantics."""
    weight = w_fp8.to(act.dtype).mul_(scale.to(act.dtype))
    return F.linear(act, weight, bias)


@pytest.mark.parametrize("m", [1, 7, 128, 1024])
@pytest.mark.parametrize(
    "n,k", [(1024, 5120), (5120, 1024), (640, 25600), (1280, 1536)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
def test_w8a16_fp8_gemm(m, n, k, dtype, bias):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.float32).to(dtype)
    w32 = torch.randn(n, k, device=dev, dtype=torch.float32) * 0.02
    w_fp8, scale = _quantize(w32)
    b = torch.randn(n, device=dev, dtype=torch.float32).to(dtype) if bias else None

    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=b)
    assert out.shape == (m, n)
    assert out.dtype == dtype

    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_out = _cast_copy_reference(act, w_fp8, scale, b)

    # Both the kernel and the cast-copy path differ from the fp32 reference
    # only by 16-bit rounding / accumulation order; the kernel must not be
    # worse than the cast-copy path by more than a small margin.
    kernel_err = (out.float() - ref32).abs().max().item()
    cast_err = (cast_out.float() - ref32).abs().max().item()
    floor = 2e-3 * ref32.abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert kernel_err <= max(2.0 * cast_err, floor), (kernel_err, cast_err, floor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_every_fp8_value(dtype):
    """Every finite e4m3 byte (denormals included) must dequantize exactly.

    act = identity over K, so out[m, n] = scale * weight[n, m] with no accumulation
    error beyond the final 16-bit rounding; the weight table enumerates all 254
    finite encodings (the two NaN bytes are excluded)."""
    dev = torch.device("cuda")
    K = 256
    codes = torch.arange(256, dtype=torch.int32, device=dev)
    codes = codes[
        (codes & 0x7F) != 0x7F
    ]  # drop the two e4m3fn NaN encodings (254 left)
    n = 8
    table = torch.zeros(n, K, dtype=torch.uint8, device=dev)
    for row in range(n):
        table[row, : codes.numel()] = codes.roll(row * 31).to(torch.uint8)
    w_fp8 = table.view(torch.float8_e4m3fn)
    scale = torch.tensor([3.0], dtype=torch.float32, device=dev)
    act = torch.eye(K, device=dev, dtype=torch.float32).to(dtype)
    out = w8a16_fp8_gemm(act, w_fp8, scale)
    ref = (w_fp8.float() * scale).T.to(dtype)
    assert torch.equal(out, ref), (out.float() - ref.float()).abs().max()


def test_w8a16_fp8_gemm_3d_and_out():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(2, 33, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(512, 256, device=dev) * 0.02)
    out = torch.empty(2, 33, 512, device=dev, dtype=torch.bfloat16)
    ret = w8a16_fp8_gemm(act, w_fp8, scale, out=out)
    assert ret is out
    ref = act.float().reshape(-1, 256) @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float().reshape(-1, 512), ref, atol=5e-2, rtol=2e-2)
    # A non-contiguous out would be silently replaced by a copy: rejected instead.
    with pytest.raises(ValueError, match="contiguous"):
        w8a16_fp8_gemm(
            act,
            w_fp8,
            scale,
            out=torch.empty(2, 512, 33, device=dev, dtype=torch.bfloat16).transpose(
                1, 2
            ),
        )
    with pytest.raises(ValueError, match="out must be"):
        w8a16_fp8_gemm(
            act,
            w_fp8,
            scale,
            out=torch.empty(66, 512, device=dev, dtype=torch.bfloat16),
        )


def test_w8a16_fp8_gemm_caller_workspace():
    """A caller-owned workspace (any dtype, oversized) gives the same result as the
    allocated one; an undersized one is rejected before launch.  The split-K variant
    is forced so the call needs a workspace on both architectures."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    algo = {9: "128x128_splitk", 10: "2sm256x128_splitk"}.get(major)
    if algo is None:
        pytest.skip("w8a16_fp8_gemm needs Hopper or Blackwell")
    m, n, k = 256, 1024, 5120
    act = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    need = w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo)
    assert need > 0
    assert (
        w8a16_fp8_gemm_workspace_bytes(m, n, k, dev, algo) == need
    )  # pure in its inputs
    ref = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo)
    ws = torch.full(
        (need + 1000,), 0x7F, dtype=torch.uint8, device=dev
    )  # garbage, oversized
    out = w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws)
    assert torch.equal(out, ref)
    ws16 = torch.empty((need // 2 + 8,), dtype=torch.bfloat16, device=dev)
    assert torch.equal(
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws16), ref
    )
    with pytest.raises(ValueError, match="workspace too small"):
        w8a16_fp8_gemm(act, w_fp8, scale, algo=algo, workspace=ws[: need - 4])
    # A single-pass shape ignores the buffer entirely (0 bytes needed).
    plain = {9: "128x256", 10: "2sm256x128"}[major]
    assert w8a16_fp8_gemm_workspace_bytes(4096, 4096, 5120, dev, plain) == 0
    tiny = torch.empty(1, dtype=torch.uint8, device=dev)
    big_act = torch.randn(4096, k, device=dev, dtype=torch.bfloat16)
    w2, s2 = _quantize(torch.randn(4096, k, device=dev) * 0.02)
    assert torch.equal(
        w8a16_fp8_gemm(big_act, w2, s2, algo=plain, workspace=tiny),
        w8a16_fp8_gemm(big_act, w2, s2, algo=plain),
    )


def test_w8a16_fp8_gemm_rejects_bad_inputs():
    dev = torch.device("cuda")
    act = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(32, 64, device=dev))
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act.float(), w_fp8, scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8.to(torch.bfloat16), scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act[:, :32], w_fp8, scale)


@pytest.mark.parametrize("n,k", [(1024, 5128), (1028, 5120)])
def test_w8a16_fp8_gemm_rejects_misaligned(n, k):
    """K % 16 and N % 8 are rejected upfront with a clear message (no silent fallback)."""
    dev = torch.device("cuda")
    act = torch.randn(64, k, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev))
    with pytest.raises(ValueError, match="multiple of"):
        w8a16_fp8_gemm(act, w_fp8, scale)


def test_w8a16_fp8_gemm_rejects_retired_hand_algo():
    """The hand-written Hopper kernel was retired in version H7; its name is gone."""
    dev = torch.device("cuda")
    act = torch.randn(16, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(128, 256, device=dev))
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8, scale, algo="hand")


_ALGOS_SM100 = [
    "2sm256x128",
    "2sm256x192",
    "2sm256x256",
    "1sm128x128",
    "2sm256x128_splitk",
    "2sm256x128_streamk",
    "2sm256x160",
]
_ALGOS_SM90 = [
    "128x256",
    "128x192",
    "128x160",
    "128x128",
    "64x256",
    "128x256_streamk",
    "128x192_streamk",
    "64x128",
    "128x128_splitk",
    "128x256_splitk",
]


@pytest.mark.parametrize("algo", _ALGOS_SM100 + _ALGOS_SM90)
@pytest.mark.parametrize(
    "m,n,k", [(1024, 1024, 5120), (200, 1024, 5120), (4096, 128, 5120)]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_fp8_gemm_algo(algo, m, n, k, bias, dtype):
    """Every kernel variant reachable through ``algo`` must agree with the cast-copy
    reference; a variant of another architecture is rejected."""
    dev = torch.device("cuda")
    major = get_compute_capability(dev)[0]
    mine = algo in (_ALGOS_SM100 if major == 10 else _ALGOS_SM90 if major == 9 else [])
    torch.manual_seed(0)
    act = torch.randn(m, k, device=dev, dtype=dtype)
    w_fp8, scale = _quantize(torch.randn(n, k, device=dev) * 0.02)
    b = torch.randn(n, device=dev, dtype=dtype) if bias else None
    if not mine:
        with pytest.raises(ValueError):
            w8a16_fp8_gemm(act, w_fp8, scale, bias=b, algo=algo)
        return
    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=b, algo=algo)
    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_err = (
        (_cast_copy_reference(act, w_fp8, scale, b).float() - ref32).abs().max().item()
    )
    err = (out.float() - ref32).abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert err <= max(2.0 * cast_err, 2e-3 * ref32.abs().max().item()), (
        algo,
        err,
        cast_err,
    )
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8, scale, algo="nope")
