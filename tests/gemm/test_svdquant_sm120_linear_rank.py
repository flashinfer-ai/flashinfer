"""Rank-aware dispatch and complete-linear validation for SM120 CUTLASS."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer import autotune, nvfp4_quantize, svdquant_linear
from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.utils import get_compute_capability

from .test_nvfp4_svdquant_gemm import (
    _mm_fp4_residual,
    _nvfp4_quantize_128x4,
    _sqnr_db,
)


@pytest.mark.parametrize("rank", [32, 64])
def test_linear_runners_load_the_requested_rank(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    requested_ranks: list[int] = []

    def get_module(lora_rank: int = 32) -> SimpleNamespace:
        requested_ranks.append(lora_rank)
        return SimpleNamespace()

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_nvfp4_svdquant_sm120_module", get_module
    )

    svdquant_sm120_cutlass._sm120_linear_runners(
        False, torch.device("cuda:0"), 129, 256, 256, rank
    )

    assert requested_ranks
    assert set(requested_ranks) == {rank}


@pytest.mark.parametrize("rank", [0, 48, 96, 128])
def test_linear_rejects_unsupported_ranks_before_loading_kernels(rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        svdquant_linear(
            torch.empty((1, 32), dtype=torch.bfloat16),
            torch.empty((32, 16), dtype=torch.uint8),
            torch.empty(128, dtype=torch.uint8),
            torch.ones(1),
            torch.ones(32, dtype=torch.bfloat16),
            torch.empty((32, rank), dtype=torch.bfloat16),
            torch.empty((32, rank), dtype=torch.bfloat16),
            torch.ones(1),
            enable_pdl=False,
            backend="cutlass-sm120",
        )


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize(
    "shape", [(129, 256, 256), (537, 5376, 7168), (64, 3072, 3072)]
)
@pytest.mark.parametrize("use_bias,enable_pdl", [(False, False), (True, True)])
def test_linear_rank_autotune_and_graph_replay(
    rank: int, shape: tuple[int, int, int], use_bias: bool, enable_pdl: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires SM120")
    if get_compute_capability(torch.device("cuda")) != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    pqs = (1 + 0.3 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_sf = (2688.0 / (x * pqs).float().abs().amax()).reshape(1)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    weight_fp4, weight_sf, weight_global_sf = _nvfp4_quantize_128x4(weight)
    alpha = (1.0 / (global_sf * weight_global_sf)).reshape(1)
    lora_a = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    l2t = (pqs[:, None] * lora_a.T).contiguous()
    lora_b = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16) / rank**0.25
    l1 = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if use_bias else None

    def linear() -> torch.Tensor:
        return svdquant_linear(
            x,
            weight_fp4,
            weight_sf.reshape(-1),
            alpha,
            pqs,
            l2t,
            l1,
            global_sf,
            bias=bias,
            enable_pdl=enable_pdl,
            backend="cutlass-sm120",
        )

    def reference() -> torch.Tensor:
        xq, x_sf = nvfp4_quantize((x * pqs).to(torch.bfloat16), global_sf)
        residual = _mm_fp4_residual(
            xq, weight_fp4, x_sf, weight_sf, alpha, backend="cutlass"
        )
        down = torch.mm(x, l2t)
        expected = residual + down.float() @ lora_b.float().T
        return expected if bias is None else expected + bias.float()

    expected = reference()
    eager = linear()
    assert _sqnr_db(expected, eager.float()) > 40.0

    with autotune(True):
        tuned = linear()
    assert _sqnr_db(expected, tuned.float()) > 40.0

    cached = linear()
    assert _sqnr_db(expected, cached.float()) > 40.0
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear()

    x.copy_(torch.randn_like(x) / k**0.25)
    l2t.copy_((pqs[:, None] * torch.randn_like(lora_a).T / k**0.25).contiguous())
    lora_b.copy_(torch.randn_like(lora_b) / rank**0.25)
    l1.copy_((lora_b.float() / alpha).bfloat16())
    expected_fresh = reference()
    graph.replay()
    torch.cuda.synchronize()

    assert captured.shape == (m, n)
    assert captured.dtype == torch.bfloat16
    assert _sqnr_db(expected_fresh, captured.float()) > 40.0
