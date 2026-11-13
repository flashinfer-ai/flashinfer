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

The SM12x 4over6 quantizers must be bit-identical to fp4_quantize with the same recipe: packed values and the
whole swizzled scale buffer (padding included), for every kernel variant the dispatcher can pick.
"""

import pytest
import torch

from flashinfer import (
    NVFP44Over6Config,
    fp4_quantize,
    nvfp4_4over6_dynamic_quantize_sm120,
    nvfp4_4over6_quantize_sm120,
)
from flashinfer.quantization.nvfp4_4over6_sm120 import (
    _sf_numel,
    get_nvfp4_4over6_quantize_sm120_module,
)
from flashinfer.utils import get_compute_capability

DEVICE = torch.device("cuda")


def _skip_unless_sm12x():
    if not torch.cuda.is_available() or get_compute_capability(DEVICE)[0] != 12:
        pytest.skip("requires an SM12x GPU")


def make_input(m: int, k: int, kind: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(m, k, device=DEVICE, generator=g)
    spread = (
        torch.randn(m, k // 16, 1, device=DEVICE, generator=g)
        .repeat_interleave(16, 1)
        .reshape(m, k)
    )
    if kind == "tails":  # per-block magnitude spread
        x = x * torch.exp(spread * 0.5)
    elif (
        kind == "wide"
    ):  # blocks across many binades, all-zero blocks and an all-zero row
        x = x * torch.exp(spread * 4.0)
        zero = (
            torch.rand(m, k // 16, 1, device=DEVICE, generator=g)
            .repeat_interleave(16, 1)
            .reshape(m, k)
            < 0.05
        )
        x = x.masked_fill(zero, 0.0)
        x[m // 2] = 0.0
    elif kind == "tiny":
        x = x * 1e-30
    elif kind == "huge":
        x = x * 1e30
    return x.bfloat16().contiguous()


def global_scale_for(
    x: torch.Tensor, config: NVFP44Over6Config, factor: float = 1.0
) -> torch.Tensor:
    amax = x.abs().amax().float().clamp(min=1e-12)
    return (6.0 * config.e4m3_max * factor / amax).reshape(1).contiguous()


def reference(x: torch.Tensor, gs: torch.Tensor, config: NVFP44Over6Config):
    x_q, sf = fp4_quantize(x, gs, 16, False, True, nvfp4_4over6=config)
    return x_q.view(torch.uint8).reshape(x.shape[0], -1), sf.view(torch.uint8).reshape(
        -1
    )


def assert_same(out, ref, m: int, k: int):
    x_q, sf = out
    ref_q, ref_sf = ref
    assert torch.equal(x_q.view(torch.uint8).reshape(m, k // 2), ref_q)
    assert sf.numel() == _sf_numel(m, k) == ref_sf.numel()
    assert torch.equal(sf.view(torch.uint8).reshape(-1), ref_sf)


MSE_256 = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")
MSE_448 = NVFP44Over6Config(e4m3_max=448, err_mode="MSE")


@pytest.mark.parametrize("m", [1, 3, 16, 127, 129, 300, 1024])
@pytest.mark.parametrize("k", [16, 48, 80, 2048, 2064, 5120])
@pytest.mark.parametrize("kind", ["normal", "tails", "wide", "tiny", "huge"])
@pytest.mark.parametrize("config", [MSE_256, MSE_448], ids=["e4m3_256", "e4m3_448"])
def test_static_matches_fp4_quantize(m, k, kind, config):
    _skip_unless_sm12x()
    x = make_input(m, k, kind, 7919 * m + k)
    gs = global_scale_for(x, config)
    assert_same(
        nvfp4_4over6_quantize_sm120(x, gs, nvfp4_4over6=config),
        reference(x, gs, config),
        m,
        k,
    )


@pytest.mark.parametrize("factor", [0.25, 0.5, 2.0, 4.0])
def test_static_other_global_scales(factor):
    """Scales that over- or under-fill the E4M3 range (block scales saturate or go subnormal)."""
    _skip_unless_sm12x()
    m, k = 257, 4096
    x = make_input(m, k, "wide", 11)
    gs = global_scale_for(x, MSE_256, factor)
    assert_same(nvfp4_4over6_quantize_sm120(x, gs), reference(x, gs, MSE_256), m, k)


# (unroll, v8, order) of every static kernel variant
STATIC_VARIANTS = [
    (1, False, 0),
    (1, True, 0),
    (2, True, 0),
    (11, False, 0),
    (11, True, 0),
    (11, True, 1),
    (12, True, 0),
    (21, False, 0),
    (31, True, -1),
]


@pytest.mark.parametrize("m,k", [(1, 2048), (37, 4096), (300, 5120), (4096, 2048)])
@pytest.mark.parametrize("variant", STATIC_VARIANTS)
@pytest.mark.parametrize("ctas_per_sm", [0, 1, 8])
def test_static_variants(m, k, variant, ctas_per_sm):
    _skip_unless_sm12x()
    module = get_nvfp4_4over6_quantize_sm120_module()[0]
    unroll, v8, order = variant
    x = make_input(m, k, "tails", m + k)
    gs = global_scale_for(x, MSE_256)
    x_q = torch.empty((m, k // 2), dtype=torch.uint8, device=DEVICE)
    sf = torch.empty((_sf_numel(m, k),), dtype=torch.uint8, device=DEVICE)
    module.nvfp4_4over6_quantize_sm120(
        x, gs, x_q, sf, unroll, ctas_per_sm, v8, 1, order
    )
    assert_same((x_q, sf), reference(x, gs, MSE_256), m, k)


def dynamic_reference(x: torch.Tensor, wgs: torch.Tensor, config: NVFP44Over6Config):
    """amax -> deq -> global scale / alpha in fp32 (round to nearest), then fp4_quantize."""
    rcp_q = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(
        6.0 * config.e4m3_max, dtype=torch.float32
    )
    deq = x.abs().amax().float().clamp(min=1e-12) * rcp_q.to(DEVICE)
    gs = (torch.ones(1, device=DEVICE) / deq).reshape(1)
    alpha = (deq * wgs).reshape(1)
    return reference(x, gs, config), gs, alpha


def assert_dynamic(out, x, wgs, config):
    m, k = x.shape
    x_q, sf, gs, alpha = out
    ref, ref_gs, ref_alpha = dynamic_reference(x, wgs, config)
    assert torch.equal(gs, ref_gs) and torch.equal(alpha, ref_alpha)
    assert_same((x_q, sf), ref, m, k)


@pytest.mark.parametrize("m", [1, 3, 16, 127, 129, 300, 1024, 4096])
@pytest.mark.parametrize("k", [16, 80, 2048, 2064, 5120])
@pytest.mark.parametrize("kind", ["normal", "wide", "tiny"])
def test_dynamic_matches_reference(m, k, kind):
    _skip_unless_sm12x()
    x = make_input(m, k, kind, 31 * m + k)
    wgs = torch.tensor([0.37], device=DEVICE)
    assert_dynamic(nvfp4_4over6_dynamic_quantize_sm120(x, wgs), x, wgs, MSE_256)


# (small_threads, hold, ctas_per_sm, v8, order); small_threads 2000 + n selects the lane-pair small kernel
DYNAMIC_VARIANTS = [
    (128, 0, 0, True, 0),
    (256, 0, 0, False, 0),
    (1024, 0, 0, True, 0),
    (2128, 0, 0, False, 0),
    (2512, 0, 0, False, 0),
    (-1, 1, 0, True, 0),
    (-1, 2, 2, True, 0),
    (-1, 4, 2, True, 0),
    (-1, 4, 1, False, 1),
    (-1, 2, 1, True, 1),
]


@pytest.mark.parametrize("m,k", [(1, 2048), (16, 4096), (300, 5120), (16384, 2048)])
@pytest.mark.parametrize("variant", DYNAMIC_VARIANTS)
def test_dynamic_variants(m, k, variant):
    _skip_unless_sm12x()
    small_threads, hold, ctas_per_sm, v8, order = variant
    nblk = m * k // 16
    if small_threads > 2000 and 2 * nblk > 8 * (small_threads - 2000):
        pytest.skip("input too large for the lane-pair small path")
    if 0 < small_threads < 2000 and nblk > 8 * small_threads:
        pytest.skip("input too large for the small path")
    module = get_nvfp4_4over6_quantize_sm120_module()[0]
    x = make_input(m, k, "tails", m * 3 + k)
    wgs = torch.tensor([1.5], device=DEVICE)
    gs, alpha = torch.empty(1, device=DEVICE), torch.empty(1, device=DEVICE)
    scratch = torch.zeros(8192, device=DEVICE)
    x_q = torch.empty((m, k // 2), dtype=torch.uint8, device=DEVICE)
    sf = torch.empty((_sf_numel(m, k),), dtype=torch.uint8, device=DEVICE)
    rcp_q = float(torch.tensor(1.0) / torch.tensor(1536.0))
    module.nvfp4_4over6_dynamic_quantize_sm120(
        x,
        gs,
        alpha,
        wgs,
        rcp_q,
        scratch,
        x_q,
        sf,
        small_threads,
        hold,
        ctas_per_sm,
        v8,
        2,
        order,
    )
    assert_dynamic((x_q, sf, gs, alpha), x, wgs, MSE_256)


@pytest.mark.parametrize("m,k", [(1, 4096), (64, 4096), (2048, 5120)])
def test_dynamic_cuda_graph_replay(m, k):
    """Replays see new input values (the scratch slots are rewritten by every launch)."""
    _skip_unless_sm12x()
    x = make_input(m, k, "tails", 5)
    wgs = torch.ones(1, device=DEVICE)
    gs, alpha = torch.empty(1, device=DEVICE), torch.empty(1, device=DEVICE)
    nvfp4_4over6_dynamic_quantize_sm120(
        x, wgs, global_scale_out=gs, alpha_out=alpha
    )  # warm up / build
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nvfp4_4over6_dynamic_quantize_sm120(
            x, wgs, global_scale_out=gs, alpha_out=alpha
        )
    for seed, scale in ((6, 1.0), (7, 1e3), (8, 1e-3)):
        x.copy_(make_input(m, k, "wide", seed).float().mul(scale).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        assert_dynamic(out, x, wgs, MSE_256)


def test_leading_dims_and_pdl_off():
    _skip_unless_sm12x()
    x = make_input(2 * 37, 4096, "tails", 3)
    gs = global_scale_for(x, MSE_256)
    x_q, sf = nvfp4_4over6_quantize_sm120(x.reshape(2, 37, 4096), gs, enable_pdl=False)
    assert x_q.shape == (2, 37, 2048)
    assert_same((x_q.reshape(-1, 2048), sf), reference(x, gs, MSE_256), 2 * 37, 4096)


def test_rejects_unsupported_recipes():
    _skip_unless_sm12x()
    x = make_input(4, 64, "normal", 1)
    gs = global_scale_for(x, MSE_256)
    with pytest.raises(ValueError):
        nvfp4_4over6_quantize_sm120(
            x, gs, nvfp4_4over6=NVFP44Over6Config(e4m3_max=256, err_mode="MAE")
        )
    with pytest.raises(ValueError):
        nvfp4_4over6_quantize_sm120(
            x,
            gs,
            nvfp4_4over6=NVFP44Over6Config(
                e4m3_max=256, err_mode="MSE", err_use_fast_math=True
            ),
        )
    with pytest.raises(ValueError):
        nvfp4_4over6_quantize_sm120(x.half(), gs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
