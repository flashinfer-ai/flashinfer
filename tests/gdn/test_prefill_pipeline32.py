# SPDX-License-Identifier: Apache-2.0
"""SM120 p32 reference checks, including tails and public opt-in dispatch."""

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


@pytest.mark.parametrize("lengths", [(33,), (65,), (0, 31, 0, 33), (0,)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pipeline32_reference_and_dispatch(lengths, dtype, monkeypatch):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("pipeline32 requires SM120")
    from flashinfer import gdn_prefill

    torch.manual_seed(42)
    total, hq, hv = sum(lengths), 2, 6
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(total, hq, 128, device="cuda"), dim=-1
        ).to(dtype)
        for _ in range(2)
    ]
    v = torch.randn(total, hv, 128, device="cuda", dtype=dtype)
    alpha = (
        torch.nn.functional.logsigmoid(torch.randn(total, hv, device="cuda"))
        .div(16)
        .exp()
    )
    beta = torch.randn(total, hv, device="cuda").sigmoid()
    initial = torch.randn(len(lengths), hv, 128, 128, device="cuda")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    scale = 128**-0.5
    expected = _reference(q, k, v, alpha, beta, initial, lengths, scale)
    monkeypatch.setenv("FLASHINFER_GDN_SM120_PIPELINE32", "1")

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("pipeline32 opted in but official fuse was called")

    monkeypatch.setattr(
        gdn_prefill, "chunk_gated_delta_rule_sm120", unexpected_fallback
    )
    actual = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=alpha,
        beta=beta,
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
        backend="flashinfer",
    )
    for result, reference in zip(actual, expected, strict=False):
        _assert_relative(result, reference)
    # Replays exercise the same ping-pong slots without reusing updated state.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            scale=scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            use_cp=False,
            backend="flashinfer",
        )
    graph.replay()
    graph.replay()
    for result, reference in zip(captured, expected, strict=False):
        _assert_relative(result, reference)

    with pytest.raises(NotImplementedError, match="normalization"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            use_cp=False,
            backend="flashinfer",
            use_qk_l2norm_in_kernel=True,
        )

    with pytest.raises(NotImplementedError, match="state_indices"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            initial_state=initial,
            output_final_state=True,
            output_state=torch.empty_like(initial),
            cu_seqlens=cu,
            use_cp=False,
            backend="flashinfer",
            state_indices=torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        )
