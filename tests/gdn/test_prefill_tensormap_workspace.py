"""GDN prefill tensor-map workspace: %nsmid probe under CUDA graph capture (#9484).

The workspace holds one 128-byte tensor map per ``%smid``, so it is sized by
``%nsmid``, which a device probe reads. The first prefill call in a process may
be made under ``torch.cuda.graph(...)``; the probe must not break that capture.
"""

import warnings

import pytest
import torch

from flashinfer.gdn_kernels.delta_rule_dsl import helpers
from flashinfer.gdn_kernels.delta_rule_dsl.helpers import (
    get_device_nsmid,
    probe_device_nsmid,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import is_sm90a_supported, is_sm12x_supported

from .reference_delta_rule import exclusive_cumsum


def _skip_unless_nsmid_backend():
    device = torch.device("cuda")
    if not (is_sm90a_supported(device) or is_sm12x_supported(device)):
        pytest.skip("GDN prefill workspace sized by %nsmid on SM90 / SM12x only")


def test_nsmid_probe_under_capture():
    _skip_unless_nsmid_backend()
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    eager = probe_device_nsmid(device)
    assert eager >= sm_count

    x = torch.zeros(16, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = probe_device_nsmid(device)
        x.add_(1)
    assert captured == eager

    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 2))


def test_probe_failure_fallback_is_not_undersized(monkeypatch):
    """A failed probe must not size the workspace below the SM count (#9484)."""
    _skip_unless_nsmid_backend()
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    def broken_probe(_device):
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(helpers, "probe_device_nsmid", broken_probe)
    get_device_nsmid.cache_clear()
    try:
        with pytest.warns(RuntimeWarning, match="probe failed"):
            slots = get_device_nsmid(device)
        assert slots >= max(sm_count, helpers._NSMID_FALLBACK_SLOTS)
        # Cached: the failed probe is not retried (or re-warned) on later calls.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_device_nsmid(device) == slots
    finally:
        get_device_nsmid.cache_clear()


@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_first_prefill_call_under_capture(qkv_factory, dtype, use_cp):
    _skip_unless_nsmid_backend()
    device = torch.device("cuda")
    torch.manual_seed(0)

    seq_lens = [64, 128, 512]
    num_q_heads, num_k_heads, num_v_heads, head_size = 16, 16, 32, 128
    total, num_seqs = sum(seq_lens), len(seq_lens)
    num_heads = max(num_q_heads, num_v_heads)
    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seq_lens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64)
        alpha = torch.rand(total, num_heads)
        beta = torch.rand(total, num_heads)

    def run(o, state):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            head_size**-0.5,
            None,
            True,
            cu_seq_lens,
            use_qk_l2norm_in_kernel=False,
            output=o,
            output_state=state,
            use_cp=use_cp,
            max_seqlen=max(seq_lens),
        )

    def outputs():
        o = torch.full(
            (total, num_heads, head_size), float("nan"), dtype=dtype, device=device
        )
        state = torch.full(
            (num_seqs, num_heads, head_size, head_size), float("nan"), device=device
        )
        return o, state

    graph_o, graph_state = outputs()
    get_device_nsmid.cache_clear()
    graph = torch.cuda.CUDAGraph()
    # A probe failure would take the multi_processor_count fallback and still
    # pass the checks below without MPS, so the fallback warning fails the test.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with torch.cuda.graph(graph):
            run(graph_o, graph_state)
    assert get_device_nsmid(device) == probe_device_nsmid(device)
    assert get_device_nsmid(device) >= (
        torch.cuda.get_device_properties(device).multi_processor_count
    )

    graph.replay()
    eager_o, eager_state = outputs()
    run(eager_o, eager_state)
    torch.cuda.synchronize()
    assert torch.equal(graph_o, eager_o)
    assert torch.equal(graph_state, eager_state)
