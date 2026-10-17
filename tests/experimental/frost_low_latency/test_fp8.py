# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from flashinfer import bmm_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 7)),
    reason="frost-low-latency is qualified on SM100/SM107",
)


def inputs(m=4, k=512, n=64):
    a = torch.randn(1, m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(1, n, k, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
    sa = torch.tensor([0.25], device="cuda")
    sb = torch.tensor([1.5], device="cuda")
    return a, b, sa, sb


def run(a, b, sa, sb, out=None):
    return bmm_fp8(a, b, sa, sb, torch.bfloat16, out=out, backend="frost-low-latency")


def check(y, a, b, sa, sb):
    ref = (a.double() @ b.double()) * (sa * sb).double()
    torch.testing.assert_close(y, ref.to(torch.bfloat16), rtol=0.008, atol=0.001)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("k,n", [(512, 8), (1536, 136), (6144, 5120)])
def test_shapes(m, k, n):
    a, b, sa, sb = inputs(m, k, n)
    out = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
    assert run(a, b, sa, sb, out) is out
    check(out, a, b, sa, sb)


def test_rebinding_and_graph():
    a, b, sa, sb = inputs()
    run(a, b, sa, sb)
    a2, b2, sa2, sb2 = inputs()
    sa2.fill_(0.75)
    sb2.fill_(0.125)
    out = torch.empty((1, 4, 64), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(a2, b2, sa2, sb2, out)
    torch.cuda.synchronize()
    check(out, a2, b2, sa2, sb2)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(a2, b2, sa2, sb2, out)
    # Replay must read all current tensor contents, including both scales/weights.
    a2.copy_(a)
    b2.copy_(b)
    sa2.fill_(0.5)
    sb2.fill_(2.0)
    out.fill_(float("nan"))
    g.replay()
    check(out, a2, b2, sa2, sb2)


def test_full_finite_range_and_zero_scale():
    a, b, sa, sb = inputs(1, 512, 256)
    values = torch.arange(256, device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    values = torch.nan_to_num(values.float()).to(torch.float8_e4m3fn)
    a.copy_(values.repeat(2).reshape_as(a))
    b.transpose(1, 2).copy_(values.reshape(1, 256, 1).expand(1, 256, 512))
    check(run(a, b, sa, sb), a, b, sa, sb)
    sa.zero_()
    assert torch.count_nonzero(run(a, b, sa, sb)) == 0


@pytest.mark.parametrize(
    "case", ["m", "k", "n", "batch", "scale", "layout", "alignment", "out"]
)
def test_rejects_unsupported(case):
    a, b, sa, sb = inputs()
    out = None
    if case == "m":
        a = a.repeat(1, 17, 1)
    elif case == "k":
        a, b = (
            a[:, :, :256].contiguous(),
            b[:, :256, :].transpose(1, 2).contiguous().transpose(1, 2),
        )
    elif case == "n":
        b = b[:, :, :63]
    elif case == "batch":
        a, b = a.repeat(2, 1, 1), b.transpose(1, 2).repeat(2, 1, 1).transpose(1, 2)
    elif case == "scale":
        sa = sa.repeat(2)
    elif case == "layout":
        b = b.contiguous()
    elif case == "alignment":
        a = torch.empty(a.numel() + 1, device="cuda", dtype=a.dtype)[1:].reshape_as(a)
    elif case == "out":
        out = torch.empty((1, 4, 128), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    with pytest.raises((ValueError, RuntimeError)):
        run(a, b, sa, sb, out)


def test_explicit_only(monkeypatch):
    from flashinfer.gemm.gemm_base import _heuristic_func_bmm_fp8

    a, b, sa, sb = inputs()
    for enabled in ("0", "1"):
        monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", enabled)
        candidates = _heuristic_func_bmm_fp8(
            ["cublas", "frost-low-latency"], a, b, sa, sb, torch.bfloat16
        )
        assert "frost-low-latency" not in candidates


def test_all_e4m3_product_pairs():
    """Check the packed-product precision claim before any reduction can cancel errors."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
    from flashinfer.experimental.frost_low_latency.fp8 import fp8x4_product_to_float4

    @cute.kernel
    def product_kernel(x: cute.Tensor, w: cute.Tensor, y: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        i = block * 128 + tid
        values = fp8x4_product_to_float4(x[i], w[i])
        for j in cutlass.range_constexpr(4):
            y[i * 4 + j] = values[j]

    @cute.jit
    def launch(x, w, y, stream):
        product_kernel(x, w, y).launch(
            grid=(128, 1, 1), block=(128, 1, 1), stream=stream
        )

    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    x = codes.repeat_interleave(256)
    w = codes.repeat(256)
    out = torch.empty(65536, device="cuda", dtype=torch.float32)
    args = (x.view(torch.uint32), w.view(torch.uint32), out)
    fn = cute.compile(
        launch,
        *(from_dlpack(t, assumed_align=4) for t in args),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    fn(*args)
    reference = (
        x.view(torch.float8_e4m3fn).float() * w.view(torch.float8_e4m3fn).float()
    )
    torch.testing.assert_close(out, reference, atol=0, rtol=0, equal_nan=True)
