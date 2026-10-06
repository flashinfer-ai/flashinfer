"""Decode quantization workspace tests for CUDA-graph capture."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _clear_registries(fdw):
    fdw._QUANT_SCRATCH.clear()
    fdw._CAPTURE_ASSIGNED.clear()
    fdw._CAPTURE_CLAIMED.clear()


@pytest.fixture(autouse=True)
def _isolate_scratch_registries():
    """Restore the process-wide scratch registries after each test.

    These tests clear buckets other tests (and any warm serving path in the
    same interpreter) rely on, so the clears must not outlive the module.
    """
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    saved = (
        dict(fdw._QUANT_SCRATCH),
        dict(fdw._CAPTURE_ASSIGNED),
        {key: list(value) for key, value in fdw._CAPTURE_CLAIMED.items()},
    )
    # Registry-size assertions require an empty isolated state.
    _clear_registries(fdw)
    try:
        yield
    finally:
        for registry, snapshot in zip(
            (fdw._QUANT_SCRATCH, fdw._CAPTURE_ASSIGNED, fdw._CAPTURE_CLAIMED),
            saved,
            strict=True,
        ):
            registry.clear()
            registry.update(snapshot)


@cuda_required
def test_capture_claims_eager_bucket_per_stream(monkeypatch):
    """Fake-capture unit test of the claim logic (no real graph needed)."""
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    device = torch.device("cuda")
    _clear_registries(fdw)
    m_pad, k = 128, 512

    # Prewarm one bucket on each stream that will participate in capture.
    eager_entry = fdw._small_m_quant_scratch(m_pad, k, device)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        side_entry = fdw._small_m_quant_scratch(m_pad, k, device)
    assert len(fdw._QUANT_SCRATCH) == 2

    # "Capture" on the same stream: must claim the eager bucket, not allocate.
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    claimed = fdw._small_m_quant_scratch(m_pad, k, device)
    assert claimed is eager_entry
    assert len(fdw._QUANT_SCRATCH) == 2
    # Repeated capture calls on one stream use the same stable address.
    assert fdw._small_m_quant_scratch(m_pad, k, device) is eager_entry

    # The second capture stream claims its own prewarmed bucket.
    with torch.cuda.stream(side):
        other = fdw._small_m_quant_scratch(m_pad, k, device)
        assert other is side_entry
        assert other is not eager_entry
        assert fdw._small_m_quant_scratch(m_pad, k, device) is other

    # Claimed graph storage is reserved. Later eager work gets a replacement
    # instead of racing a replay that retains the old pointer.
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )
    eager_replacement = fdw._small_m_quant_scratch(m_pad, k, device)
    assert eager_replacement is not eager_entry
    assert fdw._CAPTURE_ASSIGNED
    torch.cuda.synchronize()


@cuda_required
def test_capture_refuses_unplanned_quant_scratch(monkeypatch):
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarmed scratch bucket"):
        fdw._small_m_quant_scratch(128, 512, torch.device("cuda"))


@cuda_required
def test_decode_graph_replay_reuses_bucket_and_is_bit_exact():
    """Real capture: no zero-fills recorded for the quant scratch, bit-exact replay."""
    import b12x.quantization.mxfp6.fp6_dense_weights as fdw

    torch.manual_seed(0)
    _clear_registries(fdw)
    n, k = 6144, 512
    w = fdw.quantize_dense_weight_to_fp6(
        (torch.randn(n, k, device="cuda") * 0.1).to(torch.bfloat16)
    )
    x_static = (torch.randn(1, k, device="cuda") * 0.1).to(torch.bfloat16)

    # Eager warm passes (compiles kernels, creates buckets) — one on the
    # default stream, one on the capture-warmup side stream, mirroring the
    # serving flow (load-time warm + pre-capture eager runs).
    y_ref = fdw.dense_fp6_linear(x_static, w)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fdw.dense_fp6_linear(x_static, w)
    torch.cuda.current_stream().wait_stream(side)
    buckets_before = len(fdw._QUANT_SCRATCH)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = fdw.dense_fp6_linear(x_static, w)

    # Capture claimed an existing eager bucket: assignment recorded, and the
    # assigned buffers alias one of the retained eager entries.
    assert fdw._CAPTURE_ASSIGNED, "capture did not claim a bucket"
    assert len(fdw._QUANT_SCRATCH) == buckets_before
    assigned = next(iter(fdw._CAPTURE_ASSIGNED.values()))
    eager_ptrs = {
        e[0].packed_a_storage.data_ptr() for e in fdw._QUANT_SCRATCH.values()
    }
    assert assigned[0].packed_a_storage.data_ptr() in eager_ptrs

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y_static, y_ref, rtol=0.0, atol=0.0)

    # Replay tracks new contents in the static input. The eager comparison uses
    # separate scratch because graph-claimed storage remains reserved.
    x_static.copy_(
        (torch.randn(1, k, device="cuda") * 0.2).to(torch.bfloat16)
    )
    graph.replay()
    torch.cuda.synchronize()
    y_replay = y_static.clone()
    y_eager = fdw.dense_fp6_linear(x_static, w)
    torch.testing.assert_close(y_replay, y_eager, rtol=0.0, atol=0.0)
