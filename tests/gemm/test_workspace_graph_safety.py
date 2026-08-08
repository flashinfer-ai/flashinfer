"""Workspace growth must never invalidate storage a captured CUDA graph references.

Captured graph nodes hold the workspace's raw device address as a baked kernel
argument. Growing a workspace by freeing-and-reallocating (``resize_()`` /
cache-entry replacement) leaves every previously captured graph writing into
memory the allocator may reuse (silent corruption) or unmap (Xid 31 MMU fault
at replay). Both tests drive pre-existing library entry points and fail on the
unsafe growth behavior without needing a GPU fault.
"""

import pytest
import torch

from flashinfer.utils import _cache_buf, _get_cache_buf, get_compute_capability

SMALL = 8 * 1024 * 1024
LARGE = 16 * 1024 * 1024
SENTINEL = 0x5A


def _capture_fill(buf: torch.Tensor) -> torch.cuda.CUDAGraph:
    """Capture a graph whose only node writes SENTINEL through buf's data_ptr."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buf.fill_(0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buf.fill_(SENTINEL)
    return graph


def test_cache_buf_growth_keeps_graph_referenced_storage_alive():
    """Growing a cached workspace must not free the old storage.

    A captured graph plays the role of any captured kernel holding the old
    buffer's address. With free-on-growth semantics the freed block is
    reused by the fresh allocations below and replay corrupts them.
    """
    name = "test_ws_graph_safety_canary"
    device = torch.device("cuda:0")
    buf = _get_cache_buf(name, SMALL, device)
    graph = _capture_fill(buf)

    grown = _get_cache_buf(name, LARGE, device)
    assert grown.size(0) >= LARGE
    del buf

    probes = [torch.zeros(SMALL, dtype=torch.uint8, device=device) for _ in range(3)]
    graph.replay()
    torch.cuda.synchronize()
    for probe in probes:
        assert not (probe == SENTINEL).any(), (
            "graph replay wrote into memory reused by a new allocation: "
            "workspace growth freed storage a captured graph still references"
        )
    _cache_buf.pop((name, device), None)


def test_cudnn_fp8_gemm_workspace_growth_preserves_baked_pointer():
    """A cuDNN fp8 GEMM whose plan outgrows the workspace must not free it.

    Mirrors the production failure: graphs capture the workspace address,
    then a later execution selects a plan needing more workspace than the
    buffer holds. The library must satisfy the demand without invalidating
    the caller-visible tensor (the address captured graphs reference), and
    the GEMM result must still be correct.
    """
    cudnn = pytest.importorskip("cudnn")
    from flashinfer import bmm_fp8
    from flashinfer.gemm.gemm_base import (
        _cudnn_gemm_fp8,
        _get_cudnn_workspace_size,
        _torch_data_type_to_cudnn_data_type,
        build_cudnn_gemm_fp8_graph,
    )

    device = torch.device("cuda:0")
    major, minor = get_compute_capability(device)
    if not bmm_fp8.is_backend_supported("cudnn", major * 10 + minor):
        pytest.skip(
            f"cuDNN fp8 GEMM not supported on sm{major}{minor}; skipping fp8 graph test"
        )
    b, m, n, k = 1, 48, 80, 64
    a = (torch.randn(b, m, k, device=device) * 0.1).to(torch.float8_e4m3fn)
    mat2 = (
        (torch.randn(b, n, k, device=device) * 0.1)
        .to(torch.float8_e4m3fn)
        .transpose(-2, -1)
    )
    a_scale = torch.ones(1, 1, 1, device=device, dtype=torch.float32)
    b_scale = torch.ones(1, 1, 1, device=device, dtype=torch.float32)
    out = torch.empty(b, m, n, device=device, dtype=torch.bfloat16)

    plan_graph = build_cudnn_gemm_fp8_graph(
        a.shape,
        a.stride(),
        mat2.shape,
        mat2.stride(),
        _torch_data_type_to_cudnn_data_type(a.dtype),
        _torch_data_type_to_cudnn_data_type(mat2.dtype),
        _torch_data_type_to_cudnn_data_type(out.dtype),
        device,
        policy=cudnn.build_plan_policy.ALL,
    )
    workspace = torch.empty(1024, dtype=torch.uint8, device=device)
    tactic = next(
        (
            t
            for t in range(plan_graph.get_execution_plan_count())
            if _get_cudnn_workspace_size(plan_graph, t) > workspace.numel()
        ),
        None,
    )
    if tactic is None:
        pytest.skip("no cuDNN plan at this shape needs more than 1KiB workspace")

    graph = _capture_fill(workspace)
    ptr_before = workspace.data_ptr()

    _cudnn_gemm_fp8(workspace, a, mat2, a_scale, b_scale, out, out.dtype, tactic)
    torch.cuda.synchronize()

    assert workspace.data_ptr() == ptr_before and workspace.numel() == 1024, (
        "workspace growth invalidated the storage captured graphs reference"
    )

    graph.replay()
    torch.cuda.synchronize()
    assert (workspace == SENTINEL).all(), "baked pointer no longer writable"

    reference = torch.bmm(a.float(), mat2.float())
    cos = torch.nn.functional.cosine_similarity(
        reference.reshape(-1), out.float().reshape(-1), dim=0
    )
    assert cos > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
