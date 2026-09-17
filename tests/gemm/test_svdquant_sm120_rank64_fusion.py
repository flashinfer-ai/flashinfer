"""Rank-64 fused preprocessing admission and CUDA Graph correctness."""

import pytest
import torch

from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm import svdquant_sm120_routes as routes

from .test_nvfp4_svdquant_gemm import _sqnr_db
from .test_svdquant_sm120_linear_rank import (
    test_linear_rank_autotune_and_graph_replay as _check_linear_graph_replay,
)


@pytest.mark.parametrize("m,k", [(64, 3072), (537, 7168), (2048, 3072), (4096, 12288)])
def test_rank64_offers_fused_preprocessing(m: int, k: int) -> None:
    assert backend._sm120_fused_linear_supported(m, k, 64)
    variants = routes.sm120_producer_variants(m, k, 64)
    assert variants
    assert {family for family, _, _ in variants} == {
        routes.SM120_FAMILY_LARGE_M,
        routes.SM120_FAMILY_SMALL_M,
    }
    for variant, entry in enumerate(variants):
        assert routes.sm120_decode_producer_variant(m, k, variant, 64) == entry
        assert not routes.sm120_variant_packs_l2t(m, k, variant, 64)


def test_rank64_fusion_invalidates_unfused_route_cache() -> None:
    assert routes.sm120_linear_route_abi_version(64, 3072, 64) > 10
    assert routes.sm120_linear_route_abi_version(64, 3072, 32) == 10


@pytest.mark.parametrize(
    "m,k,variant",
    [
        (m, k, variant)
        for m, k in [(64, 3072), (537, 7168), (2048, 3072), (4096, 12288)]
        for variant in range(routes.sm120_producer_variant_count(m, k, 64))
    ],
)
def test_rank64_fused_prefix_observes_graph_weight_updates(
    m: int, k: int, variant: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260917)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    pqs = (1 + 0.2 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_scale = (2688 / (x.float() * pqs.float()).abs().amax()).reshape(1)
    l2t = torch.randn((k, 64), device="cuda", dtype=torch.bfloat16)
    xq = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sf = torch.zeros(
        ((m + 127) // 128 * 128 * (k // 16),), device="cuda", dtype=torch.uint8
    )
    down = torch.empty((m, 64), device="cuda", dtype=torch.bfloat16)
    expected_xq = torch.empty_like(xq)
    expected_sf = torch.zeros_like(sf)
    module = backend.get_nvfp4_svdquant_sm120_module(64)
    family, tiling, address_policy = routes.sm120_decode_producer_variant(
        m, k, variant, 64
    )

    def prefix() -> None:
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            x, pqs, global_scale, l2t, xq, sf, down, family, *tiling, address_policy
        )

    def check() -> None:
        module.nvfp4_quantize_smooth(
            x, pqs, global_scale, expected_xq, expected_sf, False
        )
        reference = x.float() @ l2t.float()
        assert torch.equal(xq, expected_xq)
        assert torch.equal(sf, expected_sf)
        for rank_column in range(64):
            assert (
                _sqnr_db(reference[:, rank_column], down[:, rank_column].float()) > 40
            )

    prefix()
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefix()

    for _ in range(2):
        x.copy_(torch.randn_like(x))
        l2t.copy_(torch.randn_like(l2t))
        down.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.parametrize("shape", [(64, 3072, 3072), (537, 5376, 7168)])
def test_rank64_complete_linear_with_fused_runner_only(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int]
) -> None:
    def fused_runners(
        enable_pdl: bool, device: torch.device, m: int, n: int, k: int, rank: int
    ) -> list[backend.TunableRunner]:
        return [backend._sm120_fused_linear_runner(enable_pdl, device, rank)]

    monkeypatch.setattr(backend, "_cached_sm120_linear_runners", fused_runners)
    monkeypatch.setattr(backend, "_SM120_LINEAR_DISPATCH_CACHE", {})
    backend.AutoTuner.get().clear_cache()
    try:
        _check_linear_graph_replay(64, shape, True, True)
    finally:
        backend.AutoTuner.get().clear_cache()
