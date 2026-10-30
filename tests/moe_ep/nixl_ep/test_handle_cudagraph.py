"""CUDA graph routing updates without a NIXL transport dependency."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("index_bits", [32, 64])
def test_routing_updates_replay_from_static_input(index_bits):
    """A captured routing copy must observe each new input without a host update."""
    import torch

    from flashinfer.moe_ep.backends.split.comm.nixl_ep.handle import NixlEpHandle
    from flashinfer.moe_ep.config import HandleParams

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    index_dtype = torch.int32 if index_bits == 32 else torch.int64
    fleet = SimpleNamespace(_nixl_ep=SimpleNamespace(topk_idx_t=index_dtype))
    ids = torch.zeros(8, 2, dtype=torch.int64, device="cuda")
    handle = NixlEpHandle(fleet, HandleParams(topk_ids=ids))
    observed = torch.empty_like(ids, dtype=index_dtype)
    bound_ptr = handle._topk_ids.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        handle.update(HandleParams(topk_ids=ids))
        # Read the tensor passed to low_latency_dispatch, with real GPU work.
        observed.copy_(handle._topk_ids)
    for value in (1, 5, 2):
        ids.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(observed, ids.to(index_dtype))
    assert handle._topk_ids.data_ptr() == bound_ptr
    del graph
    handle.destroy()
