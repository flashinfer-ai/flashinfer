"""Real MUSA Inductor coverage for the native Simple-STP graph boundary."""

import os

import pytest
import torch

from flashinfer.mamba.selective_state_update import selective_state_update


TEST_DEVICE = "musa" if hasattr(torch.version, "musa") else "cpu"


def _native_inputs(seed):
    torch.manual_seed(seed)
    device = torch.device(TEST_DEVICE)
    heads, dim, dstate, groups, slots = 64, 64, 128, 8, 8
    state = torch.randn(slots, heads, dim, dstate, device=device, dtype=torch.float16)
    x = torch.randn(1, heads, dim, device=device, dtype=torch.bfloat16)
    dt_head = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    dt = dt_head[None, :, None].expand(1, heads, dim)
    a_head = -torch.rand(heads, device=device, dtype=torch.float32) - 1
    A = a_head[:, None, None].expand(heads, dim, dstate)
    B = torch.randn(1, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(heads, 1, device=device, dtype=torch.float32).expand(heads, dim)
    dt_bias = torch.randn(heads, 1, device=device, dtype=torch.float32).expand(heads, dim)
    z = torch.randn_like(x)
    src = torch.tensor([1], device=device, dtype=torch.int32)
    dst = torch.tensor([6], device=device, dtype=torch.int32)
    out = torch.empty_like(x)
    return state, x, dt, A, B, C, D, dt_bias, z, src, dst, out


def _step(state, x, dt, A, B, C, D, dt_bias, z, src, dst, out):
    return selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        pad_slot_id=-1,
        out=out,
        algorithm="simple",
        backend="flashinfer",
    )


@pytest.mark.skipif(TEST_DEVICE != "musa", reason="MUSA Inductor test")
@pytest.mark.skipif(
    os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1",
    reason="native extension is opt-in",
)
def test_native_ssu_inductor_fullgraph_replay_and_mutation():
    eager_inputs = _native_inputs(2201)
    eager_state = eager_inputs[0].clone()
    eager_args = (eager_state, *eager_inputs[1:])
    eager_out = _step(*eager_args)

    compiled = torch.compile(_step, backend="inductor", fullgraph=True, dynamic=False)
    first_inputs = _native_inputs(2201)
    first_out = compiled(*first_inputs)
    torch.musa.synchronize()
    torch.testing.assert_close(first_out, eager_out, rtol=0.05, atol=0.05)
    torch.testing.assert_close(first_inputs[0][6], eager_state[6], rtol=0.05, atol=0.05)
    torch.testing.assert_close(first_inputs[-1], first_out, rtol=0, atol=0)
    torch.testing.assert_close(first_inputs[0][1], eager_state[1], rtol=0, atol=0)

    second_inputs = _native_inputs(2202)
    # Keep shapes/static guards identical while changing the caller-owned slot
    # metadata. This catches a graph replay that accidentally captured the
    # first source/destination indices as constants.
    second_inputs[9].copy_(torch.tensor([2], device=second_inputs[9].device))
    second_inputs[10].copy_(torch.tensor([5], device=second_inputs[10].device))
    second_state_before = second_inputs[0].clone()
    second_eager_state = second_state_before.clone()
    second_eager_args = (second_eager_state, *second_inputs[1:])
    second_eager_out = _step(*second_eager_args)
    second_out = compiled(*second_inputs)
    torch.musa.synchronize()
    torch.testing.assert_close(second_out, second_eager_out, rtol=0.05, atol=0.05)
    torch.testing.assert_close(
        second_inputs[0][5], second_eager_state[5], rtol=0.05, atol=0.05
    )
    torch.testing.assert_close(second_inputs[-1], second_out, rtol=0, atol=0)
    torch.testing.assert_close(second_inputs[0][2], second_state_before[2], rtol=0, atol=0)

    explanation = torch._dynamo.explain(_step)(*(_native_inputs(2203)))
    assert explanation.graph_break_count == 0
    assert any(
        "flashinfer_musa.simple_stp" in str(node.target)
        for graph in explanation.graphs
        for node in graph.graph.nodes
    )


@pytest.mark.skipif(TEST_DEVICE != "musa", reason="MUSA graph test")
@pytest.mark.skipif(
    os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1",
    reason="native extension is opt-in",
)
def test_native_ssu_musa_graph_updates_existing_buffers():
    from flashinfer.mamba.musa_ssu_native import preload_musa_simple_stp

    preload_musa_simple_stp()
    compiled = torch.compile(_step, backend="inductor", fullgraph=True, dynamic=False)
    static = _native_inputs(2210)
    compiled(*static)
    torch.musa.synchronize()

    capture_stream = torch.musa.Stream()
    capture_stream.wait_stream(torch.musa.current_stream())
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        captured_output = compiled(*static)
    torch.musa.current_stream().wait_stream(capture_stream)

    updated = _native_inputs(2211)
    updated[9].fill_(2)
    updated[10].fill_(5)
    expected_state = updated[0].clone()
    expected_output = _step(expected_state, *updated[1:]).clone()
    # Preserve each static tensor's address and tied stride layout. Copying
    # directly to an expanded view would write overlapping storage.
    for index in (0, 1, 4, 5, 8, 9, 10):
        static[index].copy_(updated[index])
    static[2][..., 0].copy_(updated[2][..., 0])
    static[3][:, 0, 0].copy_(updated[3][:, 0, 0])
    for index in (6, 7):
        static[index][:, 0].copy_(updated[index][:, 0])
    static[-1].fill_(float("nan"))

    graph.replay()
    torch.musa.synchronize()
    torch.testing.assert_close(captured_output, expected_output, rtol=0.05, atol=0.05)
    torch.testing.assert_close(static[-1], expected_output, rtol=0.05, atol=0.05)
    torch.testing.assert_close(static[0][5], expected_state[5], rtol=0.05, atol=0.05)
    torch.testing.assert_close(static[0][2], updated[0][2], rtol=0, atol=0)
    torch.testing.assert_close(static[0][6], updated[0][6], rtol=0, atol=0)
