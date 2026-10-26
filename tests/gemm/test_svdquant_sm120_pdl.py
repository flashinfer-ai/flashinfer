"""Deferred PDL consumers see fresh quantize output on every graph replay."""

import pytest
import torch

from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm import svdquant_sm120_routes as routes


@pytest.mark.parametrize("capture", [False, True])
def test_quantize_handoff_workspace_is_isolated(capture: bool) -> None:
    """Independent streams or captures must not overwrite each other's counter."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    workspaces = []
    for index in range(2):
        context = (
            torch.cuda.graph(graphs[index])
            if capture
            else torch.cuda.stream(streams[index])
        )
        with context:
            workspace = backend._sm120_linear_workspace(16, torch.device("cuda"), 128)
            workspace.fill_(index + 1)
        workspaces.append(workspace)
    if capture:
        # Both captures use PyTorch's shared capture stream, but replay elsewhere.
        for graph, stream in zip(graphs, streams, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for index, workspace in enumerate(workspaces):
        torch.testing.assert_close(workspace, torch.full_like(workspace, index + 1))


@pytest.mark.parametrize("m,k", [(17, 3072), (200, 4096), (64, 11008)])
@pytest.mark.parametrize("extra_workspace", [0, 8])
def test_deferred_prefix_matches_serialized_replays(
    m: int, k: int, extra_workspace: int
) -> None:
    """Replayed PDL output must match serialization after inputs and scratch change."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # Given: changing inputs and each deferred small-M producer configuration.
    torch.manual_seed(1)
    n, rank = 128, 128
    module = backend.get_nvfp4_svdquant_sm120_module(rank)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    pqs = torch.ones(k, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    l2t = torch.randn((k, rank), device="cuda", dtype=torch.bfloat16) * 0.01
    l1 = torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.01
    weight = torch.full((n, k // 2), 0x22, device="cuda", dtype=torch.uint8)
    weight_sf = torch.full((n * k // 16,), 0x38, device="cuda", dtype=torch.uint8)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    xq = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sf = torch.empty(
        ((m + 127) // 128 * 128 * k // 16,), device="cuda", dtype=torch.uint8
    )
    down = torch.empty((m, rank), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    packed = backend._pack_sm120_kmajor_l2t(l2t)
    row = module.nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank)
    size = module.nvfp4_svdquant_gemm_workspace_size(m, n, k, row)
    workspace = torch.empty(size + extra_workspace, device="cuda", dtype=torch.uint8)

    for family, tiling, policy in routes.sm120_producer_variants(m, k, rank):
        if (
            family
            not in (routes.SM120_FAMILY_SMALL_M, routes.SM120_FAMILY_SMALL_M_SPLIT)
            or policy != 1
        ):
            continue

        def run(enable_pdl: bool) -> None:
            """Run a fixed producer/GEMM pair with only dependency policy varying."""
            module.nvfp4_svdquant_linear_sm120(
                x,
                pqs,
                scale,
                packed,
                weight,
                weight_sf,
                scale,
                l1,
                bias,
                xq,
                sf,
                down,
                output,
                workspace,
                row,
                enable_pdl,
                family,
                *tiling,
                policy,
            )

        run(True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(True)

        # When: repeated replays follow updates and overwritten scratch buffers.
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            run(False)
            expected = output.clone()
            xq.fill_(0xFF)
            sf.fill_(0xFF)
            down.fill_(float("nan"))
            output.fill_(float("nan"))
            graph.replay()

            # Then: both the explicit handoff and exact-workspace fallback agree.
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph.reset()
