"""GDN prefill tensor-map workspace: %nsmid probe under CUDA graph capture (#9484).

The workspace holds one 128-byte tail tensor map per ``%smid``, so it is sized by
``%nsmid``, which a device probe reads. The first prefill call in a process may
be made under ``torch.cuda.graph(...)``; the probe must not break that capture.
"""

import pytest
import torch

from flashinfer.gdn_kernels.delta_rule_dsl import helpers
from flashinfer.gdn_kernels.delta_rule_dsl.helpers import (
    get_device_nsmid,
    probe_device_nsmid,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import _cache_buf, is_sm90a_supported, is_sm12x_supported

from .reference_delta_rule import blockwise_delta_rule, exclusive_cumsum

# Tail descriptors are written only for a non-final sequence whose length is not
# a multiple of the 64-token tile, so these lengths exercise the workspace.
SEQ_LENS = [93, 130, 512]
REPLAYS = 3


def _skip_unless_nsmid_backend():
    device = torch.device("cuda")
    if not (is_sm90a_supported(device) or is_sm12x_supported(device)):
        pytest.skip("GDN prefill workspace sized by %nsmid on SM90 / SM12x only")


def _tensormap_workspace(device, use_cp):
    """The cached workspace that this wrapper (arch, CP or not) indexes by %smid."""
    if not use_cp:
        name = "gdn_prefill_tensormaps"
    elif is_sm12x_supported(device):
        name = "gdn_cp_sm120_prefill_tensormaps"
    else:
        name = "gdn_cp_prefill_tensormaps"
    index = torch.cuda.current_device() if device.index is None else device.index
    bufs = [
        buf
        for (key, _), buf in _cache_buf.items()
        if key == name and buf.device.index == index
    ]
    assert len(bufs) == 1, f"expected one {name} buffer on cuda:{index}"
    return bufs[0]


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


def test_probe_failure_raises_once_compiled(monkeypatch):
    """A failed probe raises instead of sizing the workspace from a guess (#9484)."""
    _skip_unless_nsmid_backend()
    device = torch.device("cuda")
    calls = []

    def broken_probe(_device):
        calls.append(_device)
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(helpers, "probe_device_nsmid", broken_probe)
    helpers._cached_nsmid_probe.cache_clear()
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="cannot size the GDN prefill"):
                get_device_nsmid(device)
        assert len(calls) == 1  # the failure is cached, not re-probed per call
    finally:
        helpers._cached_nsmid_probe.cache_clear()


@pytest.mark.parametrize("use_cp", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_first_prefill_call_under_capture(qkv_factory, dtype, use_cp):
    _skip_unless_nsmid_backend()
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_q_heads, num_k_heads, num_v_heads, head_size = 16, 16, 32, 128
    total, num_seqs = sum(SEQ_LENS), len(SEQ_LENS)
    num_heads = max(num_q_heads, num_v_heads)
    scale = head_size**-0.5

    def make_inputs():
        with device:
            q, k, v = qkv_factory(
                SEQ_LENS, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
            )
            k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
            return q, k, v, torch.rand(total, num_heads), torch.rand(total, num_heads)

    q, k, v, alpha, beta = make_inputs()
    cu_seq_lens = torch.tensor(
        exclusive_cumsum(SEQ_LENS), dtype=torch.int64, device=device
    )

    def run(o, state):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            scale,
            None,
            True,
            cu_seq_lens,
            use_qk_l2norm_in_kernel=False,
            output=o,
            output_state=state,
            use_cp=use_cp,
            max_seqlen=max(SEQ_LENS),
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
    helpers._cached_nsmid_probe.cache_clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(graph_o, graph_state)
    assert get_device_nsmid(device) == probe_device_nsmid(device)
    assert get_device_nsmid(device) >= (
        torch.cuda.get_device_properties(device).multi_processor_count
    )

    if dtype == torch.bfloat16:
        tol_o, tol_state = dict(atol=1e-2, rtol=1e-2), dict(atol=5e-3, rtol=1e-3)
    else:
        tol_o, tol_state = dict(atol=2e-3, rtol=1e-3), dict(atol=1e-3, rtol=1e-4)

    workspace = _tensormap_workspace(device, use_cp)
    for replay in range(REPLAYS):
        if replay:  # the graph must follow new input contents
            for dst, src in zip((q, k, v, alpha, beta), make_inputs(), strict=True):
                dst.copy_(src)
        graph_o.fill_(float("nan"))
        graph_state.fill_(float("nan"))
        graph.replay()

        eager_o, eager_state = outputs()
        workspace.zero_()
        run(eager_o, eager_state)
        torch.cuda.synchronize()
        assert workspace.any(), "no tail tensor map was written"
        assert torch.equal(graph_o, eager_o)
        assert torch.equal(graph_state, eager_state)

        ref_o, ref_state = blockwise_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            SEQ_LENS,
            scale_factor=scale,
            alpha=alpha,
            beta=beta,
            state_dtype=torch.float32,
        )
        torch.testing.assert_close(eager_o, ref_o.to(dtype), **tol_o)
        torch.testing.assert_close(
            eager_state.transpose(-1, -2), ref_state.float(), **tol_state
        )
