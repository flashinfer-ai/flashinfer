# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM12x blockscaled single-row GEMM, including retained CUDA graph inputs."""

import math

import pytest
import torch

from flashinfer.gemm import gemm_fp8_nt_blockscaled
from flashinfer.utils import is_sm12x_supported


pytestmark = pytest.mark.node_exclusive


@pytest.fixture(autouse=True)
def require_sm12x():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("SM12x only")


def _inputs(n, k, mode, dtype=torch.float8_e4m3fn):
    a = (torch.randn((1, k), device="cuda") / 4).to(dtype)
    b = (torch.randn((n, k), device="cuda") / math.sqrt(k)).to(dtype)
    # Nonuniform, non-power-of-two scales exercise each K and N scale block.
    sa = torch.rand((1, k // 128), device="cuda") + 0.25
    sb = torch.rand((n // 128, k // 128), device="cuda") + 0.25
    if mode == "MN":
        sa, sb = sa.T.contiguous(), sb.T.contiguous()
    return a, b, sa, sb


def _reference(a, b, sa, sb, mode):
    if mode == "MN":
        sa, sb = sa.T, sb.T
    a_dequant = a.float() * sa.repeat_interleave(128, dim=1)
    b_dequant = b.float() * sb.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    # A single-row dot product avoids TF32 settings affecting the reference.
    return (a_dequant * b_dequant).sum(dim=1).unsqueeze(0)


@pytest.mark.parametrize("n,k", [(128, 128), (1024, 256), (2048, 4096), (14336, 2048)])
@pytest.mark.parametrize("mode", ["K", "MN"])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_out", [False, True])
def test_blockscaled_single_row(n, k, mode, input_dtype, out_dtype, use_out):
    torch.manual_seed(9556)
    inputs = _inputs(n, k, mode, input_dtype)
    ref = _reference(*inputs, mode)
    out = (
        torch.full((1, n), float("nan"), device="cuda", dtype=out_dtype)
        if use_out
        else None
    )
    result = gemm_fp8_nt_blockscaled(
        *inputs, scale_major_mode=mode, out=out, out_dtype=out_dtype
    )
    assert result.shape == (1, n) and result.dtype == out_dtype
    if use_out:
        assert result is out
    torch.testing.assert_close(result.float(), ref, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("mode", ["K", "MN"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_blockscaled_single_row_graph_replay(mode, out_dtype):
    torch.manual_seed(9556)
    n, k = 1024, 512
    inputs = _inputs(n, k, mode)
    out = torch.empty((1, n), device="cuda", dtype=out_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemm_fp8_nt_blockscaled(*inputs, scale_major_mode=mode, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gemm_fp8_nt_blockscaled(*inputs, scale_major_mode=mode, out=out)
    for _ in range(4):
        # Keep captured addresses but replace both operands and both scale tensors.
        for dst, src in zip(inputs, _inputs(n, k, mode), strict=True):
            dst.copy_(src)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            out.float(), _reference(*inputs, mode), atol=1e-3, rtol=1e-2
        )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 64, 256),
        (1, 192, 256),
        (2, 128, 256),
        (32, 128, 256),
        (129, 128, 256),
        (1, 128, 64),
        (1, 128, 192),
    ],
)
def test_blockscaled_other_sm12x_tails_still_raise(m, n, k):
    a = torch.zeros((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros((n, k), device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.ones((math.ceil(m / 128), math.ceil(k / 128)), device="cuda")
    sb = torch.ones((math.ceil(n / 128), math.ceil(k / 128)), device="cuda")
    out = torch.full((m, n), 123.0, device="cuda", dtype=torch.bfloat16)
    # RuntimeError is the native launcher rejection; Python backend validation
    # may reject earlier with ValueError (the independent #9642 change).
    with pytest.raises((RuntimeError, ValueError)):
        gemm_fp8_nt_blockscaled(a, b, sa, sb, scale_major_mode="K", out=out)
        torch.cuda.synchronize()
    assert bool((out == 123).all())
