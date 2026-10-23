# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 recurrence checks for the experimental CuTe FlashQLA port."""

import pytest
import torch
from flashinfer.utils import get_compute_capability


def _reference(q, k, v, alpha, beta, initial, lengths, scale):
    """Independent token recurrence in FP64, state laid out as [H,V,K]."""
    heads = v.shape[1]
    q = q.double().repeat_interleave(heads // q.shape[1], dim=1)
    k = k.double().repeat_interleave(heads // k.shape[1], dim=1)
    v = v.double()
    outputs, states = [], []
    offset = 0
    for seq, length in enumerate(lengths):
        state = initial[seq].double().clone()
        for token in range(offset, offset + length):
            state = state * alpha[token, :, None, None].double()
            prediction = torch.einsum("hvk,hk->hv", state, k[token])
            delta = beta[token, :, None].double() * (v[token] - prediction)
            state = state + delta[:, :, None] * k[token, :, None, :]
            outputs.append(scale * torch.einsum("hvk,hk->hv", state, q[token]))
        states.append(state)
        offset += length
    output = torch.stack(outputs) if outputs else v.new_empty((0, heads, 128))
    return output, torch.stack(states)


def _assert_relative(actual, expected):
    assert actual.shape == expected.shape
    diff = torch.linalg.vector_norm(actual.double() - expected)
    norm = torch.linalg.vector_norm(expected)
    error = diff / norm if norm.item() else diff
    assert torch.isfinite(error) and error.item() <= 0.02


@pytest.mark.parametrize(
    "dtype,lengths,cp,norm,vfirst,initial_kind,index_dtype,gate_dtype",
    [
        (torch.bfloat16, (65,), False, False, True, "fp32", torch.int32, torch.float32),
        (torch.float16, (65,), True, True, False, "none", torch.int64, torch.float32),
        (
            torch.bfloat16,
            (33, 65),
            True,
            True,
            False,
            "fp32",
            torch.int64,
            torch.bfloat16,
        ),
        (
            torch.float16,
            (33, 65),
            False,
            False,
            True,
            "native",
            torch.int32,
            torch.float16,
        ),
        (torch.bfloat16, (128,), True, False, True, "none", torch.int32, torch.float32),
        (torch.bfloat16, (65,), True, False, False, "fp16", torch.int64, torch.float32),
    ],
)
@pytest.mark.parametrize("consumer_v_tile", [None, 64, 128])
@torch.inference_mode()
def test_qla_cute_forward(
    dtype,
    lengths,
    cp,
    norm,
    vfirst,
    initial_kind,
    index_dtype,
    gate_dtype,
    consumer_v_tile,
):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("CuTe FlashQLA forward requires SM120")
    from flashinfer.experimental.gdn_sm120_cute._api import (
        chunk_gated_delta_rule,
        build_plan,
    )

    torch.manual_seed(96)
    total = sum(lengths)
    hq, hv = 2, 6
    q, k = [
        torch.randn(1, total, hq, 128, device="cuda", dtype=dtype) for _ in range(2)
    ]
    if not norm:
        q, k = [
            torch.nn.functional.normalize(x.float(), dim=-1).to(dtype) for x in (q, k)
        ]
    v = torch.randn(1, total, hv, 128, device="cuda", dtype=dtype)
    g = (
        torch.nn.functional.logsigmoid(torch.randn(1, total, hv, device="cuda")) / 16
    ).to(gate_dtype)
    beta = torch.rand_like(g)
    h = None
    if initial_kind != "none":
        state_dtype = (
            torch.float16
            if initial_kind == "fp16"
            else (dtype if initial_kind == "native" else torch.float32)
        )
        h = torch.randn(len(lengths), hv, 128, 128, device="cuda", dtype=state_dtype)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=index_dtype
    )
    plan = build_plan(lengths, hv, q.device, index_dtype, False)
    if cp:
        segments = []
        ranges = []
        offset = 0
        for seq, length in enumerate(lengths):
            first = len(segments)
            end = offset + length
            for token in range(offset, end, 32):
                segments.append(
                    (token, min(token + 32, end), seq, int(token + 32 >= end))
                )
            ranges.append((first, len(segments)))
            offset = end
        plan.segments = torch.tensor(segments, device=q.device, dtype=index_dtype)
        plan.ranges = torch.tensor(ranges, device=q.device, dtype=index_dtype)
        plan.use_cp = True
    actual, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h,
        output_final_state=True,
        use_qk_l2norm_in_kernel=norm,
        cu_seqlens=cu,
        state_v_first=vfirst,
        plan=plan,
        consumer_v_tile=consumer_v_tile,
    )
    if norm:
        q, k = [
            (
                x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            for x in (q, k)
        ]
    initial = (
        torch.zeros(len(lengths), hv, 128, 128, device=q.device, dtype=torch.float64)
        if h is None
        else (h if vfirst else h.transpose(-1, -2)).double()
    )
    expected, expected_final = _reference(
        q[0], k[0], v[0], g[0].double().exp(), beta[0], initial, lengths, 128**-0.5
    )
    _assert_relative(actual[0], expected)
    _assert_relative(final if vfirst else final.transpose(-1, -2), expected_final)


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("consumer_v_tile", [None, 64, 128])
@torch.inference_mode()
def test_qla_cute_strides_and_graph(cp, consumer_v_tile):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("CuTe FlashQLA forward requires SM120")
    from flashinfer.experimental.gdn_sm120_cute._api import (
        chunk_gated_delta_rule,
        build_plan,
    )

    torch.manual_seed(12)
    q, k = [
        torch.randn(1, 65, 2, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
        for _ in range(2)
    ]
    v = torch.randn(1, 65, 6, 256, device="cuda", dtype=q.dtype)[..., ::2]
    g = torch.full((1, 65, 12), -0.05, device="cuda")[..., ::2]
    beta = torch.rand_like(g)
    h = torch.randn(1, 6, 128, 128, device="cuda").transpose(-1, -2)
    cu = torch.tensor([0, -1, 65, -1], device="cuda", dtype=torch.int64)[::2]
    plan = build_plan((65,), 6, q.device, torch.int64, False)
    if cp:
        plan.segments = torch.tensor(
            [[0, 32, 0, 0], [32, 65, 0, 1]], device=q.device, dtype=torch.int64
        )
        plan.ranges = torch.tensor([[0, 2]], device=q.device, dtype=torch.int64)
        plan.use_cp = True
    kwargs = dict(
        initial_state=h,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        state_v_first=False,
        plan=plan,
        consumer_v_tile=consumer_v_tile,
    )
    strided = chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    kwargs["initial_state"] = h.contiguous()
    kwargs["cu_seqlens"] = cu.contiguous()
    eager = chunk_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        **kwargs,
    )
    for a, b in zip(strided, eager, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Keep plan alive for the captured graph; no GPU metadata read in capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chunk_gated_delta_rule(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            **kwargs,
        )
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    with torch.cuda.graph(graph, stream=stream):
        captured = chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, eager, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("batch,hq,hv", [(2, 16, 48), (4, 8, 24)])
@torch.inference_mode()
def test_qla_cute_qualified_auto_graph(batch, hq, hv):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("qualified auto consumer requires SM120")
    if torch.cuda.get_device_properties("cuda").multi_processor_count != 156:
        pytest.skip("qualified auto consumer is measured on 156-SM hardware")
    from flashinfer.experimental.gdn_sm120_cute._api import (
        chunk_gated_delta_rule,
        build_plan,
    )

    torch.manual_seed(314)
    length = 2048
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(batch, length, hq, 128, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(batch, length, hv, 128, device="cuda", dtype=q.dtype)
    g = torch.full((batch, length, hv), -0.05, device="cuda")
    beta = torch.rand_like(g)
    initial = torch.randn(batch, hv, 128, 128, device="cuda")
    plan = build_plan([length] * batch, hv, q.device, torch.int32, False)
    kwargs = dict(
        initial_state=initial,
        output_final_state=True,
        state_v_first=True,
        plan=plan,
    )
    expected = chunk_gated_delta_rule(q, k, v, g, beta, consumer_v_tile=128, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["precomputed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("full_first", [True, False])
@torch.inference_mode()
def test_full_chunks_cache_switch(backend, dtype, index_dtype, full_first):
    """Equal tensor/segment shapes must not alias full/tail compile entries."""
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("CuTe full-chunk specialization requires SM120")
    from flashinfer.experimental.gdn_sm120_cute._api import build_plan
    from flashinfer.experimental.gdn_sm120_cute.preprocess import (
        precompute,
    )
    from flashinfer.experimental.gdn_sm120_cute.consumer_v128 import (
        fused_forward as precomputed,
    )

    torch.manual_seed(634232)
    total, hq, hv = 160, 2, 6
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(total, hq, 128, device="cuda"), dim=-1
        ).to(dtype)
        for _ in range(2)
    ]
    v = torch.randn(total, hv, 128, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(total, hv, device="cuda")) / 16
    g[:, -1] = 0
    beta = torch.rand_like(g)
    initial = torch.randn(2, hv, 128, 128, device="cuda")
    order = (True, False, True) if full_first else (False, True, False)
    for full in order:
        lengths = (64, 96) if full else (65, 95)
        plan = build_plan(lengths, hv, q.device, index_dtype, False)
        if backend == "precomputed":
            a, prefix = precompute(k, beta, g, plan.indices)
            forward = precomputed
        else:
            a, prefix = q, g
            raise AssertionError("unselected fusion prototype")
        kwargs = dict(
            scale=128**-0.5,
            num_raw_sequences=2,
            state_v_first=True,
            initial_v_first=True,
            output_final_state=True,
        )
        actual = forward(
            q,
            k,
            v,
            a,
            prefix,
            beta,
            plan.segments,
            initial,
            full_chunks=full,
            **kwargs,
        )
        generic = forward(
            q,
            k,
            v,
            a,
            prefix,
            beta,
            plan.segments,
            initial,
            full_chunks=False,
            **kwargs,
        )
        expected = _reference(q, k, v, g.exp(), beta, initial, lengths, 128**-0.5)
        for x, y, ref in zip(actual, generic, expected, strict=False):
            _assert_relative(x, ref)
            torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-4)
