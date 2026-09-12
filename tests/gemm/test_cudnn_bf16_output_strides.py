"""Regression for fixed-shape cuDNN BF16 graphs with caller-owned strided output."""

import pytest
import torch

import flashinfer
from flashinfer.gemm import gemm_base


@pytest.mark.parametrize("rank", [2, 3])
def test_fixed_shape_cudnn_bf16_output_layouts_and_changed_graph(monkeypatch, rank):
    if not torch.cuda.is_available() or not gemm_base.CUDNN_AVAILABLE:
        pytest.skip("cuDNN CUDA execution required")
    cc = torch.cuda.get_device_capability()
    api = flashinfer.bmm_bf16 if rank == 3 else flashinfer.mm_bf16
    if not api.is_backend_supported("cudnn", cc[0] * 10 + cc[1]):
        pytest.skip("cuDNN BF16 GEMM not supported on this device")
    # This is also the route used after an override-shape tactic falls back.
    monkeypatch.setattr(gemm_base, "_is_cudnn_override_shape_available", lambda: False)
    torch.manual_seed(42241 + rank)
    groups, m, n, k = (8 if rank == 3 else 1), 2, 128, 256
    base_a = torch.randn(m, groups, k, device="cuda", dtype=torch.bfloat16)
    a3 = base_a.transpose(0, 1)
    weight = torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16)
    a, b = (a3, weight.transpose(1, 2)) if rank == 3 else (a3[0], weight[0].T)
    # Alternate declarations at identical shape to exercise graph cache identity.
    for layout in ("contiguous", "interleaved", "padded", "contiguous"):
        if layout == "contiguous":
            storage = torch.full(
                (groups, m, n), 17, device="cuda", dtype=torch.bfloat16
            )
            out3 = storage
        elif layout == "interleaved":
            storage = torch.full(
                (m, groups, n), 17, device="cuda", dtype=torch.bfloat16
            )
            out3 = storage.transpose(0, 1)
        else:
            storage = torch.full(
                (groups, m, n * 2), 17, device="cuda", dtype=torch.bfloat16
            )
            out3 = storage[..., :n]
        out = out3 if rank == 3 else out3[0]

        def run():
            return api(a, b, out=out, backend="cudnn")

        def verify():
            expected = torch.matmul(a.double(), b.double()).bfloat16()
            torch.testing.assert_close(out, expected, rtol=0.01, atol=0.005)
            if layout == "padded":
                torch.testing.assert_close(
                    storage[..., n:],
                    torch.full_like(storage[..., n:], 17),
                    rtol=0,
                    atol=0,
                )

        run()
        verify()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(2):
            base_a.copy_(torch.randn_like(base_a))
            weight.copy_(torch.randn_like(weight))
            graph.replay()
            verify()
