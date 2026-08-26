# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flashinfer
from flashinfer.gemm.gemm_dual_bf16_weight import (
    _dual_bf16_weight_gemm_kernel_kind,
)


def _is_exact_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm100(), reason="dual BF16 weight GEMM requires exact SM100"
)


def _make_inputs(m: int, n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32)
    weight_high, weight_low = flashinfer.prepare_dual_bf16_weights(weight)
    return a, weight_high, weight_low


def _reference(a, weight_high, weight_low, out_dtype):
    high = torch.mm(a.float(), weight_high.float().T)
    low = torch.mm(a.float(), weight_low.float().T)
    return (high + low / 256.0).to(out_dtype)


@pytest.mark.parametrize(
    "m,n,k,expected_kind",
    [
        (32, 64, 128, 1),
        (32, 193, 128, 2),
        (256, 193, 256, 0),
        (257, 64, 256, 1),
        (512, 193, 256, 2),
        (1024, 64, 256, 2),
    ],
)
def test_dispatch_boundaries(m, n, k, expected_kind):
    kind = _dual_bf16_weight_gemm_kernel_kind(m, n, k, "cuda")
    workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, "cuda")
    assert kind == expected_kind
    assert (workspace_size > 0) == (expected_kind == 0)


@pytest.mark.parametrize(
    "m,n,k,expected_kind",
    [
        # K=128 bypasses split-K. These cases exercise the smallest legal K,
        # minimum dimensions, 1SM's N=64 requirement, and the 2SM N tail.
        (1, 1, 128, 2),
        (1, 64, 128, 1),
        (1, 65, 128, 2),
        # M=256 is the inclusive split-K boundary when K has >=2 tiles.
        (256, 65, 256, 0),
        # M=257 is the first non-split row. Divisible N selects 1SM; an N tail
        # uses the compatible 2SM path.
        (257, 64, 128, 1),
        (257, 65, 128, 2),
        # M=1024 is the inclusive lower boundary of the normal 2SM policy.
        (1023, 64, 128, 1),
        (1024, 64, 128, 2),
        # Exercise both sides of the 2SM N-tile and M-tile switches.
        (1024, 128, 128, 2),
        (1024, 129, 128, 2),
        (1025, 64, 128, 2),
        (4095, 64, 128, 2),
        (4096, 64, 128, 2),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_correctness(m, n, k, expected_kind, out_dtype):
    a, weight_high, weight_low = _make_inputs(m, n, k)
    assert _dual_bf16_weight_gemm_kernel_kind(m, n, k, a.device) == expected_kind
    output = flashinfer.mm_bf16_dual_weight(
        a, weight_high, weight_low, out_dtype=out_dtype
    )
    reference = _reference(a, weight_high, weight_low, out_dtype)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


def test_prepare_weights():
    weight = torch.randn(17, 256, dtype=torch.float32)
    weight_high, weight_low = flashinfer.prepare_dual_bf16_weights(weight)
    reconstructed = weight_high.float() + weight_low.float() / 256.0

    assert weight_high.dtype == torch.bfloat16
    assert weight_low.dtype == torch.bfloat16
    assert weight_high.is_contiguous()
    assert weight_low.is_contiguous()
    assert (weight - reconstructed).abs().max() < (
        weight - weight_high.float()
    ).abs().max()


def test_split_k_workspace_reuse_and_out():
    m, n, k = 128, 193, 256
    a, weight_high, weight_low = _make_inputs(m, n, k)
    workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, a.device)
    workspace = torch.empty(workspace_size, device=a.device, dtype=torch.uint8)
    out = torch.empty(m, n, device=a.device, dtype=torch.float32)

    for _ in range(3):
        returned = flashinfer.mm_bf16_dual_weight(
            a,
            weight_high,
            weight_low,
            out_dtype=torch.float32,
            out=out,
            workspace_buffer=workspace,
        )
        assert returned is out

    reference = _reference(a, weight_high, weight_low, torch.float32)
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)


def test_rejects_small_workspace():
    m, n, k = 128, 193, 256
    a, weight_high, weight_low = _make_inputs(m, n, k)
    workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, a.device)
    workspace = torch.empty(workspace_size - 1, device=a.device, dtype=torch.uint8)

    with pytest.raises(ValueError, match="workspace_buffer is too small"):
        flashinfer.mm_bf16_dual_weight(
            a, weight_high, weight_low, workspace_buffer=workspace
        )


def test_rejects_misaligned_workspace():
    m, n, k = 128, 193, 256
    a, weight_high, weight_low = _make_inputs(m, n, k)
    workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, a.device)
    storage = torch.empty(workspace_size + 1, device=a.device, dtype=torch.uint8)
    workspace = storage[1:]

    with pytest.raises(Exception, match="16-byte aligned"):
        flashinfer.mm_bf16_dual_weight(
            a, weight_high, weight_low, workspace_buffer=workspace
        )


def test_multistream_all_dispatch_paths():
    cases = [
        (128, 65, 256, 0),
        (257, 64, 128, 1),
        (257, 65, 128, 2),
    ]
    inputs = [_make_inputs(m, n, k, seed=i) for i, (m, n, k, _) in enumerate(cases)]
    workspaces = [
        torch.empty(
            max(
                flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, "cuda"),
                1,
            ),
            device="cuda",
            dtype=torch.uint8,
        )
        for m, n, k, _ in cases
    ]
    outputs = [
        torch.empty(m, n, device="cuda", dtype=torch.float32) for m, n, _, _ in cases
    ]
    streams = [torch.cuda.Stream() for _ in cases]

    for stream, case, case_inputs, workspace, output in zip(
        streams, cases, inputs, workspaces, outputs, strict=True
    ):
        m, n, k, expected_kind = case
        assert _dual_bf16_weight_gemm_kernel_kind(m, n, k, "cuda") == expected_kind
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flashinfer.mm_bf16_dual_weight(
                *case_inputs,
                out_dtype=torch.float32,
                out=output,
                workspace_buffer=workspace,
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)

    for case_inputs, output in zip(inputs, outputs, strict=True):
        reference = _reference(*case_inputs, torch.float32)
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "m,n,k,expected_kind",
    [
        (128, 65, 256, 0),
        (257, 64, 128, 1),
        (257, 65, 128, 2),
    ],
)
def test_cuda_graph_all_dispatch_paths(m, n, k, expected_kind):
    a, weight_high, weight_low = _make_inputs(m, n, k)
    assert _dual_bf16_weight_gemm_kernel_kind(m, n, k, a.device) == expected_kind
    workspace_size = flashinfer.dual_bf16_weight_gemm_workspace_size(m, n, k, a.device)
    workspace = torch.empty(max(workspace_size, 1), device=a.device, dtype=torch.uint8)
    output = torch.empty(m, n, device=a.device, dtype=torch.float32)

    def run():
        flashinfer.mm_bf16_dual_weight(
            a,
            weight_high,
            weight_low,
            out_dtype=torch.float32,
            out=output,
            workspace_buffer=workspace,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    reference = _reference(a, weight_high, weight_low, torch.float32)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


def test_validation():
    a, weight_high, weight_low = _make_inputs(8, 64, 256)

    with pytest.raises(ValueError, match="identical"):
        flashinfer.mm_bf16_dual_weight(a, weight_high, weight_low[:, :128].contiguous())
    with pytest.raises(TypeError, match="out_dtype"):
        flashinfer.mm_bf16_dual_weight(
            a, weight_high, weight_low, out_dtype=torch.float16
        )
    with pytest.raises(ValueError, match="contiguous"):
        flashinfer.mm_bf16_dual_weight(
            a[:, ::2], weight_high[:, ::2], weight_low[:, ::2]
        )
