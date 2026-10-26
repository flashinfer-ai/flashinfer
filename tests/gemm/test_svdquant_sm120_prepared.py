"""Prepared SVDQuant weights retain captured addresses across weight updates."""

from contextlib import nullcontext

import pytest
import torch

from flashinfer import prepare_svdquant_l2t
from flashinfer.gemm import svdquant_sm120_cutlass as backend


@pytest.fixture(autouse=True)
def clear_layout_caches():
    backend._SM120_PREPARED_L2T.clear()
    backend._SM120_KMAJOR_L2T_CACHE.clear()
    backend._SM120_M537_PACKED_L2T_CACHE.clear()
    yield
    backend._SM120_PREPARED_L2T.clear()
    backend._SM120_KMAJOR_L2T_CACHE.clear()
    backend._SM120_M537_PACKED_L2T_CACHE.clear()


@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("inference", [False, True])
def test_reprepare_updates_existing_layout(rank: int, inference: bool) -> None:
    # Given: a consumer retains the layout prepared for an existing weight.
    with torch.inference_mode() if inference else nullcontext():
        source = torch.zeros((3072, rank), dtype=torch.bfloat16)
        backend.prepare_sm120_l2t(source)
        read = (
            backend._cached_sm120_m537_l2t
            if rank == 32
            else backend._cached_sm120_kmajor_l2t
        )
        packed = read(source)
        expected = packed.clone().fill_(1)

        # When: the weight is changed and prepared again.
        source.fill_(1)
        backend.prepare_sm120_l2t(source)

        # Then: the retained layout sees the update without a new address.
        torch.testing.assert_close(packed, expected)


@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("inference", [False, True])
def test_prepared_graph_observes_reprepare(rank: int, inference: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # Given: the graph reads a prepared layout, with the source kept alive.
    with torch.inference_mode() if inference else nullcontext():
        source = torch.zeros((3072, rank), device="cuda", dtype=torch.bfloat16)
        prepare_svdquant_l2t(source)
        read = (
            backend._cached_sm120_m537_l2t
            if rank == 32
            else backend._cached_sm120_kmajor_l2t
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = read(source).clone()

        # When: reprepare follows an in-place update outside capture.
        source.fill_(1)
        prepare_svdquant_l2t(source)
        graph.replay()

        # Then: replay observes the current weight in the original buffer.
        torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize("rank", [32, 128])
def test_preparing_inside_capture_is_rejected(rank: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    source = torch.zeros((3072, rank), device="cuda", dtype=torch.bfloat16)
    prepare_svdquant_l2t(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="capture"):
            prepare_svdquant_l2t(source)
        result = source.clone()
    graph.replay()
    torch.testing.assert_close(result, source)


@pytest.mark.parametrize("family,tiling", [(0, (768, 160, 64)), (5, (128, 64, 204))])
def test_dynamic_shared_memory_opt_in_on_each_device(
    family: int, tiling: tuple[int, int, int]
) -> None:
    devices = [
        index
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index) == (12, 0)
    ]
    if len(devices) < 2:
        pytest.skip("requires two SM120 GPUs")
    # Given: one loaded module is used on two devices in the same process.
    module = backend.get_nvfp4_svdquant_sm120_module(128)
    for index in devices[:2]:
        with torch.cuda.device(index):
            m, k, rank = 17, 256, 128
            x = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
            l2t = torch.ones((k, rank), device="cuda", dtype=torch.bfloat16)
            pqs = torch.ones(k, device="cuda", dtype=torch.bfloat16)
            scale = torch.ones(1, device="cuda")
            xq = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
            sf = torch.empty(128 * (k // 16), device="cuda", dtype=torch.uint8)
            down = torch.empty((m, rank), device="cuda", dtype=torch.bfloat16)

            # When: the same dynamic-shared-memory kernel runs on each device.
            module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
                x, pqs, scale, l2t, xq, sf, down, family, *tiling, 0
            )

            # Then: each launch succeeds and computes the matrix product.
            torch.testing.assert_close(down, torch.full_like(down, k))
