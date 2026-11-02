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

SVDQuant with dynamic NVFP4 4over6 activations on SM12x:
- the quantization part of nvfp4_4over6_dynamic_quantize_lora_down_sm120 (packed values, scale buffer, global
  scale, alpha) is bit-identical to nvfp4_4over6_dynamic_quantize_sm120 on every LoRA-down path;
- down matches an FP32 reference within BF16 rounding;
- nvfp4_4over6_svdquant_linear_sm120 equals mm_nvfp4_svdquant on those operands bit for bit.
"""

import pytest
import torch

from flashinfer import (
    NVFP44Over6Config,
    fp4_quantize,
    mm_nvfp4_svdquant,
    nvfp4_4over6_dynamic_quantize_lora_down_sm120,
    nvfp4_4over6_dynamic_quantize_sm120,
    nvfp4_4over6_svdquant_linear_sm120,
)
from flashinfer.quantization.nvfp4_4over6_sm120 import (
    _default_rcp_quant_range,
    get_nvfp4_4over6_quantize_sm120_module,
)
from flashinfer.utils import get_compute_capability

DEVICE = torch.device("cuda")
MSE_256 = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")


def _skip_unless_sm12x():
    if not torch.cuda.is_available() or get_compute_capability(DEVICE)[0] != 12:
        pytest.skip("requires an SM12x GPU")


def make_input(m: int, k: int, seed: int, kind: str = "tails") -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(m, k, device=DEVICE, generator=g)
    spread = (
        torch.randn(m, k // 16, 1, device=DEVICE, generator=g)
        .repeat_interleave(16, 1)
        .reshape(m, k)
    )
    if kind == "tails":
        x = x * torch.exp(spread * 0.5)
    elif kind == "wide":
        x = x * torch.exp(spread * 4.0)
        x[m // 2] = 0.0
    return x.bfloat16().contiguous()


def make_lora(rank: int, k: int, n: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    lora_down = (
        (torch.randn(rank, k, device=DEVICE, generator=g) * k**-0.5)
        .bfloat16()
        .contiguous()
    )
    lora_up = (
        (torch.randn(n, rank, device=DEVICE, generator=g) * 0.05)
        .bfloat16()
        .contiguous()
    )
    return lora_down, lora_up


def make_weight(n: int, k: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(n, k, device=DEVICE, generator=g) * k**-0.5).bfloat16()
    w_gs = (448.0 * 6.0 / w.abs().amax().float()).reshape(1)
    w_fp4, w_sf = fp4_quantize(w, w_gs, 16, False, True, nvfp4_4over6=None)
    return (
        w_fp4.view(torch.uint8).reshape(n, k // 2),
        w_sf.view(torch.uint8).reshape(-1),
        (1.0 / w_gs),
    )


def check_quantize_lora_down(x, lora_down, wgs, x_q, sf, down, gs, alpha):
    ref_q, ref_sf, ref_gs, ref_alpha = nvfp4_4over6_dynamic_quantize_sm120(
        x, wgs, MSE_256
    )
    assert torch.equal(gs, ref_gs) and torch.equal(alpha, ref_alpha)
    assert torch.equal(x_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(
        sf.view(torch.uint8).reshape(-1), ref_sf.view(torch.uint8).reshape(-1)
    )
    ref_down = (x.float() @ lora_down.float().t()) / alpha
    # one BF16 rounding on the fused paths, two (GEMM output, then the division) on the unfused GEMM path
    torch.testing.assert_close(
        down.float(), ref_down, rtol=1e-2, atol=1e-2 * ref_down.abs().max().item()
    )


@pytest.mark.parametrize("m", [1, 3, 8, 16, 37, 64, 129, 1024])
@pytest.mark.parametrize("k", [256, 2048, 4096, 12288])
@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("path", [-1, 0, 1, 2])
def test_quantize_lora_down_paths(m, k, rank, path):
    _skip_unless_sm12x()
    nblk = m * k // 16
    if path == 1 and not (m <= 8 and nblk <= 2048 and k <= 16384):
        pytest.skip("outside the small fused path")
    if path == 2 and not (rank == 64 and m <= 64 and k % 256 == 0 and k <= 16384):
        pytest.skip("outside the cooperative fused path")
    _, _, _, op = get_nvfp4_4over6_quantize_sm120_module()
    x = make_input(m, k, 13 * m + k)
    lora_down, _ = make_lora(rank, k, 32, m + rank)
    wgs = torch.tensor([0.0123], device=DEVICE)
    gs, alpha = torch.empty(1, device=DEVICE), torch.empty(1, device=DEVICE)
    x_q, sf, down = op(
        x, lora_down, gs, alpha, wgs, _default_rcp_quant_range(MSE_256), 2, path
    )
    check_quantize_lora_down(x, lora_down, wgs, x_q, sf, down, gs, alpha)


@pytest.mark.parametrize("m,k", [(1, 4096), (5, 2048), (64, 4096), (300, 12288)])
def test_quantize_lora_down_public(m, k):
    _skip_unless_sm12x()
    x = make_input(m, k, 7, "wide")
    lora_down, _ = make_lora(64, k, 32, 3)
    wgs = torch.tensor([2.5], device=DEVICE)
    x_q, sf, down, gs, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
        x.reshape(1, m, k), lora_down, wgs
    )
    assert x_q.shape == (1, m, k // 2) and down.shape == (1, m, 64)
    check_quantize_lora_down(
        x, lora_down, wgs, x_q.reshape(m, -1), sf, down.reshape(m, -1), gs, alpha
    )


@pytest.mark.parametrize("m", [1, 8, 37, 64, 512])
@pytest.mark.parametrize("n,k", [(512, 2048), (4096, 4096)])
@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_matches_mm(m, n, k, rank, with_bias):
    _skip_unless_sm12x()
    x = make_input(m, k, m * 5 + n)
    w_fp4, w_sf, wgs = make_weight(n, k, n + k)
    lora_down, lora_up = make_lora(rank, k, n, rank + k)
    bias = (torch.randn(n, device=DEVICE) * 0.1).bfloat16() if with_bias else None
    y = nvfp4_4over6_svdquant_linear_sm120(
        x, w_fp4, w_sf, wgs, lora_down, lora_up, bias=bias, backend="cutlass-sm120"
    )
    x_q, sf, down, _, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
        x, lora_down, wgs
    )
    ref = mm_nvfp4_svdquant(
        x_q, w_fp4, sf, w_sf, alpha, down, lora_up, bias=bias, backend="cutlass-sm120"
    )
    assert y.shape == (m, n)
    assert torch.equal(y, ref)


def test_linear_auto_backend_and_leading_dims():
    _skip_unless_sm12x()
    m, n, k = 24, 512, 2048
    x = make_input(m, k, 1)
    w_fp4, w_sf, wgs = make_weight(n, k, 2)
    lora_down, lora_up = make_lora(64, k, n, 3)
    y = nvfp4_4over6_svdquant_linear_sm120(
        x.reshape(2, 12, k), w_fp4, w_sf, wgs, lora_down, lora_up
    )
    assert y.shape == (2, 12, n)
    x_q, sf, down, _, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
        x, lora_down, wgs
    )
    ref = mm_nvfp4_svdquant(x_q, w_fp4, sf, w_sf, alpha, down, lora_up)
    assert torch.equal(y.reshape(m, n), ref)


@pytest.mark.parametrize("m", [1, 64, 1024])
def test_linear_cuda_graph_replay(m):
    """Replays see new activations: the dynamic scale, alpha and down follow the input."""
    _skip_unless_sm12x()
    n, k = 512, 4096
    x = make_input(m, k, 9)
    w_fp4, w_sf, wgs = make_weight(n, k, 10)
    lora_down, lora_up = make_lora(64, k, n, 11)
    run = lambda: nvfp4_4over6_svdquant_linear_sm120(  # noqa: E731
        x, w_fp4, w_sf, wgs, lora_down, lora_up, backend="cutlass-sm120"
    )
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = run()
    for seed, scale in ((12, 1.0), (13, 1e3), (14, 1e-3)):
        x.copy_(make_input(m, k, seed, "wide").float().mul(scale).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        x_q, sf, down, _, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
            x, lora_down, wgs
        )
        ref = mm_nvfp4_svdquant(
            x_q, w_fp4, sf, w_sf, alpha, down, lora_up, backend="cutlass-sm120"
        )
        assert torch.equal(y, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
