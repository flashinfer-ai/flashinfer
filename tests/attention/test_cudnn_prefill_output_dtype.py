import math

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper


Q_TOKENS = 32
KV_TOKENS = 48
NUM_HEADS = 4
HEAD_DIM = 128


def _require_cudnn():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package is not available")


def _inputs(dtype):
    torch.manual_seed(20260725)
    q = torch.randn(Q_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(KV_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn(KV_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    return q, k, v


def _reference(q, k, v, causal=False):
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(HEAD_DIM)
    if causal:
        q_positions = torch.arange(Q_TOKENS, device=q.device)[:, None]
        kv_positions = torch.arange(KV_TOKENS, device=q.device)[None, :]
        mask = kv_positions > q_positions + KV_TOKENS - Q_TOKENS
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v.float())


def _direct_call(
    input_dtype,
    output_dtype=None,
    *,
    out=None,
    causal=False,
    return_lse=False,
    is_cuda_graph_compatible=False,
):
    q, k, v = _inputs(input_dtype)
    seq_q = torch.tensor([Q_TOKENS], device="cuda", dtype=torch.int32).view(1, 1, 1, 1)
    seq_kv = torch.tensor([KV_TOKENS], device="cuda", dtype=torch.int32).view(
        1, 1, 1, 1
    )
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.int8)
    result, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        1.0 / math.sqrt(HEAD_DIM),
        workspace,
        max_token_per_sequence=Q_TOKENS,
        max_sequence_kv=KV_TOKENS,
        actual_seq_lens_q=seq_q,
        actual_seq_lens_kv=seq_kv,
        causal=causal,
        return_lse=return_lse,
        out=out,
        o_data_type=output_dtype,
        is_cuda_graph_compatible=is_cuda_graph_compatible,
    )
    return result, lse, (q, k, v)


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_cudnn_prefill_output_dtype(input_dtype, output_dtype):
    _require_cudnn()
    output, _, inputs = _direct_call(input_dtype, output_dtype)

    assert output.dtype == output_dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(), _reference(*inputs), atol=1e-2, rtol=1e-2
    )


def test_cudnn_prefill_default_output_dtype_matches_input():
    _require_cudnn()
    implicit, _, _ = _direct_call(torch.bfloat16)
    explicit, _, _ = _direct_call(torch.bfloat16, torch.bfloat16)

    assert implicit.dtype == torch.bfloat16
    assert torch.equal(implicit, explicit)


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_cudnn_prefill_mixed_dtype_preallocated_output(input_dtype, output_dtype):
    _require_cudnn()
    out = torch.empty(Q_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=output_dtype)
    output, _, inputs = _direct_call(input_dtype, output_dtype, out=out)

    assert output is out
    torch.testing.assert_close(
        output.float(), _reference(*inputs), atol=1e-2, rtol=1e-2
    )


def test_cudnn_prefill_validates_preallocated_output():
    _require_cudnn()
    expected_shape = (Q_TOKENS, NUM_HEADS, HEAD_DIM)

    wrong_shape = torch.empty(
        Q_TOKENS - 1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16
    )
    with pytest.raises(ValueError, match="out must have shape"):
        _direct_call(torch.bfloat16, torch.float16, out=wrong_shape)

    wrong_dtype = torch.empty(expected_shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out must have dtype"):
        _direct_call(torch.bfloat16, torch.float16, out=wrong_dtype)

    wrong_device = torch.empty(expected_shape, device="cpu", dtype=torch.float16)
    with pytest.raises(ValueError, match="out must be on device"):
        _direct_call(torch.bfloat16, torch.float16, out=wrong_device)

    noncontiguous = torch.empty(
        Q_TOKENS, NUM_HEADS, HEAD_DIM * 2, device="cuda", dtype=torch.float16
    )[..., ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        _direct_call(torch.bfloat16, torch.float16, out=noncontiguous)


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_cudnn_ragged_prefill_honors_planned_output_dtype(input_dtype, output_dtype):
    _require_cudnn()
    q, k, v = _inputs(input_dtype)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.int8)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout="NHD", backend="cudnn"
    )
    q_indptr = torch.tensor([0, q.numel()], device="cuda", dtype=torch.int32)
    k_indptr = torch.tensor([0, k.numel()], device="cuda", dtype=torch.int32)
    v_indptr = torch.tensor([0, v.numel()], device="cuda", dtype=torch.int32)
    o_indptr = torch.tensor(
        [0, Q_TOKENS * NUM_HEADS * HEAD_DIM],
        device="cuda",
        dtype=torch.int32,
    )
    wrapper.plan(
        q_indptr,
        k_indptr,
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        causal=False,
        q_data_type=input_dtype,
        kv_data_type=input_dtype,
        o_data_type=output_dtype,
        seq_lens=torch.tensor([KV_TOKENS], device="cuda", dtype=torch.int32),
        seq_lens_q=torch.tensor([Q_TOKENS], device="cuda", dtype=torch.int32),
        max_token_per_sequence=Q_TOKENS,
        max_sequence_kv=KV_TOKENS,
        v_indptr=v_indptr,
        o_indptr=o_indptr,
    )

    output = wrapper.run(q, k, v)

    assert output.dtype == output_dtype
    torch.testing.assert_close(
        output.float(), _reference(q, k, v), atol=1e-2, rtol=1e-2
    )


def test_cudnn_prefill_mixed_dtype_preserves_causal_lse_path():
    _require_cudnn()
    output, lse, inputs = _direct_call(
        torch.bfloat16,
        torch.float16,
        causal=True,
        return_lse=True,
    )

    assert lse is not None
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(
        output.float(), _reference(*inputs, causal=True), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_cudnn_prefill_mixed_dtype_cuda_graph(input_dtype, output_dtype):
    _require_cudnn()
    q, k, v = _inputs(input_dtype)
    expected = _reference(q, k, v)
    seq_q = torch.tensor([Q_TOKENS], device="cuda", dtype=torch.int32).view(1, 1, 1, 1)
    seq_kv = torch.tensor([KV_TOKENS], device="cuda", dtype=torch.int32).view(
        1, 1, 1, 1
    )
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.int8)
    out = torch.empty(Q_TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=output_dtype)

    def run():
        cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            1.0 / math.sqrt(HEAD_DIM),
            workspace,
            max_token_per_sequence=Q_TOKENS,
            max_sequence_kv=KV_TOKENS,
            actual_seq_lens_q=seq_q,
            actual_seq_lens_kv=seq_kv,
            causal=False,
            return_lse=False,
            out=out,
            o_data_type=output_dtype,
            is_cuda_graph_compatible=True,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
