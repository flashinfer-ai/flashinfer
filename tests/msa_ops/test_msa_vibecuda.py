"""Correctness and dispatch tests for the VibeCUDA MSA backend (SM100/SM103).

Every internal kernel route is exercised against the same reference
implementation and input generator as the CAKE SM100 tests:

  1. warp-specialized UMMA/TMEM prefill (flat, GQA group 16, dense KV),
  2. block-bucketed UMMA/TMEM paged prefill (GQA group 4),
  3. the general per-token / packed-pair HMMA fallback (any group, flat or
     paged, fp8 K/V included).

The backend intentionally rejects LSE outputs, K/V scale arguments, ragged
prefill ``cu_seqlens_q``, and non-default softmax scales with loud errors.
Caller-owned workspaces are exercised through warm/capture/replay below.
"""

from typing import Any

import pytest
import torch

from tests.test_helpers.msa_attention_reference import (
    FP8,
    attention_case,
    make_attention_inputs,
    reference_attention,
    require_supported_msa_gpu,
)

HEAD_DIM = 128


def _case(operation: str, **kwargs: Any) -> dict[str, Any]:
    # The VibeCUDA backend emits no LSE outputs.
    return attention_case(
        operation,
        use_workspace=False,
        return_softmax_lse=False,
        **kwargs,
    )


CASES = [
    # Flat UMMA g16 route (group 16, uniform q lengths, causal).
    pytest.param(
        _case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            q_lens=[768, 768],
            kv_lens=[4096, 4096],
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            seed=101,
        ),
        id="g16-umma-flat-causal",
    ),
    # Flat UMMA g16 route, fp16 Q/K/V, non-causal.
    pytest.param(
        _case(
            "sparse_prefill",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="flat_varlen",
            q_lens=[528, 528],
            kv_lens=[2048, 2048],
            num_q_heads=64,
            num_kv_heads=4,
            topk=32,
            causal=False,
            seed=103,
        ),
        id="g16-umma-flat-fp16-noncausal",
    ),
    # Paged UMMA g4 route (group 4, topk <= 8).
    pytest.param(
        _case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            q_lens=[64, 64],
            kv_lens=[4096, 4096],
            num_q_heads=8,
            num_kv_heads=2,
            topk=4,
            causal=True,
            seed=107,
        ),
        id="g4-umma-paged",
    ),
    # Fallback: flat bf16 group-16 decode (seqlen_q too small for the tile).
    pytest.param(
        _case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="flat_varlen",
            batch_size=8,
            seqlen_q=4,
            seqlen_kv=2048,
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            force_fused=True,
            seed=109,
        ),
        id="fallback-flat-decode-pair",
    ),
    # Fallback: flat fp16 single-token decode.
    pytest.param(
        _case(
            "sparse_decode",
            q_dtype="float16",
            kv_dtype="float16",
            kv_layout="flat_varlen",
            batch_size=8,
            seqlen_q=1,
            seqlen_kv=2048,
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            force_fused=True,
            seed=113,
        ),
        id="fallback-flat-decode-fp16",
    ),
    # Fallback: fp8 K/V flat prefill (bf16 Q).
    pytest.param(
        _case(
            "sparse_prefill",
            q_dtype="bfloat16",
            kv_dtype=FP8,
            kv_layout="flat_varlen",
            q_lens=[256, 256],
            kv_lens=[2048, 2048],
            num_q_heads=32,
            num_kv_heads=2,
            topk=8,
            causal=True,
            seed=127,
        ),
        id="fallback-fp8-flat-prefill",
    ),
    # Fallback: fp8 K/V paged decode (bf16 Q).
    pytest.param(
        _case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype=FP8,
            kv_layout="paged",
            batch_size=4,
            seqlen_q=1,
            seqlen_kv=2048,
            num_q_heads=64,
            num_kv_heads=4,
            topk=16,
            causal=True,
            force_fused=True,
            seed=131,
        ),
        id="fallback-fp8-paged-decode",
    ),
    # Fallback: tiny paged decode, GQA group 1x8.
    pytest.param(
        _case(
            "sparse_decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            kv_layout="paged",
            batch_size=2,
            seqlen_q=1,
            seqlen_kv=257,
            num_q_heads=8,
            num_kv_heads=1,
            topk=4,
            causal=True,
            force_fused=True,
            seed=137,
        ),
        id="fallback-tiny-paged-g8",
    ),
]


def _run_vibecuda(inputs: dict[str, Any], workspace=None) -> torch.Tensor:
    from flashinfer.msa_ops import msa_sparse_attention, msa_sparse_decode_attention

    if inputs["operation"] == "sparse_prefill":
        return msa_sparse_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            causal=inputs["causal"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            workspace=workspace,
            backend="vibecuda",
        )
    return msa_sparse_decode_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        seqlen_q=inputs["seqlen_q"],
        causal=inputs["causal"],
        force_fused=inputs["force_fused"],
        workspace=workspace,
        backend="vibecuda",
    )


@pytest.mark.parametrize("case", CASES)
def test_vibecuda_msa_public_api_correctness(case: dict[str, Any]) -> None:
    device = require_supported_msa_gpu()
    inputs = make_attention_inputs(case, device)
    actual = _run_vibecuda(inputs)
    expected = reference_attention(inputs)[0]
    assert actual.shape == inputs["q"].shape
    assert actual.dtype == inputs["q"].dtype
    tolerance = 0.1 if inputs["kv_dtype"] == FP8 else 0.01
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_vibecuda_route_mirrors() -> None:
    """Python route mirrors must agree with the C++ dispatcher boundaries."""
    from flashinfer.msa_ops._vibecuda_sm100 import _g16_eligible, _g4_eligible

    # g16: flat dense group-16 with q tiles of 16 rows and topk in [12, 64].
    assert _g16_eligible(16, 16, 16, paged=False, kv_fp8=False)
    assert _g16_eligible(16, 4096, 64, paged=False, kv_fp8=False)
    assert not _g16_eligible(16, 15, 16, paged=False, kv_fp8=False)  # q tile
    assert not _g16_eligible(16, 16, 8, paged=False, kv_fp8=False)  # topk low
    assert not _g16_eligible(16, 16, 66, paged=False, kv_fp8=False)  # topk high
    assert not _g16_eligible(16, 16, 14, paged=False, kv_fp8=False)  # topk % 4
    assert not _g16_eligible(16, 16, 16, paged=True, kv_fp8=False)  # paged
    assert not _g16_eligible(16, 16, 16, paged=False, kv_fp8=True)  # fp8 KV
    assert not _g16_eligible(8, 16, 16, paged=False, kv_fp8=False)  # group

    # g4: paged dense group-4 with topk <= 8 and bounded bucket tables.
    assert _g4_eligible(4, True, False, 4, 3, 64, 2, 512)
    assert _g4_eligible(4, True, False, 8, 1, 1, 1, 32)
    assert not _g4_eligible(4, True, False, 9, 3, 64, 2, 512)  # topk high
    assert not _g4_eligible(4, False, False, 4, 3, 64, 2, 512)  # flat
    assert not _g4_eligible(4, True, True, 4, 3, 64, 2, 512)  # fp8 KV
    assert not _g4_eligible(16, True, False, 4, 3, 64, 2, 512)  # group
    assert not _g4_eligible(4, True, False, 4, 3, 64, 2, 1 << 24)  # slots


def _sample_prefill_args(device: torch.device, q_lens: list[int]):
    case = _case(
        "sparse_prefill",
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        q_lens=q_lens,
        kv_lens=[2048] * len(q_lens),
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
        causal=True,
        seed=139,
    )
    return make_attention_inputs(case, device)


def test_vibecuda_rejects_unsupported_options() -> None:
    device = require_supported_msa_gpu()
    from flashinfer.msa_ops import msa_sparse_attention

    inputs = _sample_prefill_args(device, [256, 256])
    positional = (
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        inputs["cu_seqlens_q"],
        inputs["cu_seqlens_k"],
    )
    with pytest.raises(NotImplementedError, match="softmax LSE"):
        msa_sparse_attention(
            *positional, causal=True, return_softmax_lse=True, backend="vibecuda"
        )
    with pytest.raises(TypeError, match="MSASparseAttentionWorkspace"):
        msa_sparse_attention(
            *positional, causal=True, workspace=object(), backend="vibecuda"
        )
    with pytest.raises(NotImplementedError, match="softmax scale"):
        msa_sparse_attention(
            *positional, causal=True, softmax_scale=0.5, backend="vibecuda"
        )
    ragged = _sample_prefill_args(device, [256, 128])
    with pytest.raises(NotImplementedError, match="uniform per-batch"):
        msa_sparse_attention(
            ragged["q"],
            ragged["k"],
            ragged["v"],
            ragged["q2k_indices"],
            ragged["cu_seqlens_q"],
            ragged["cu_seqlens_k"],
            causal=True,
            backend="vibecuda",
        )


def test_vibecuda_accepts_only_right_aligned_explicit_q_offsets() -> None:
    device = require_supported_msa_gpu()
    from flashinfer.msa_ops import msa_sparse_attention

    inputs = _sample_prefill_args(device, [256, 256])
    positional = (
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        inputs["cu_seqlens_q"],
        inputs["cu_seqlens_k"],
    )
    right_aligned = torch.full((2,), 2048 - 256, dtype=torch.int32, device=device)
    msa_sparse_attention(
        *positional,
        causal=True,
        q_offset=right_aligned,
        backend="vibecuda",
    )
    with pytest.raises(NotImplementedError, match="right-aligned queries"):
        msa_sparse_attention(
            *positional,
            causal=True,
            q_offset=right_aligned - 1,
            backend="vibecuda",
        )


@pytest.mark.parametrize(
    "case", [CASES[0].values[0], CASES[2].values[0], CASES[3].values[0]]
)
def test_vibecuda_workspace_cuda_graph_replays_new_values(case: dict[str, Any]) -> None:
    device = require_supported_msa_gpu()
    from flashinfer.msa_ops import MSASparseAttentionWorkspace

    inputs = make_attention_inputs(case, device)
    workspace = MSASparseAttentionWorkspace(device)

    def run():
        return _run_vibecuda(inputs, workspace)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        run()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured = run()

    inputs["q"].normal_(mean=0.0, std=0.2)
    expected = reference_attention(inputs)[0]
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(captured, expected, atol=0.01, rtol=0.01)


def test_vibecuda_general_route_rejects_topk_over_shared_list_bound() -> None:
    device = require_supported_msa_gpu()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    case = _case(
        "sparse_decode",
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        batch_size=1,
        seqlen_q=1,
        seqlen_kv=8192,
        num_q_heads=8,
        num_kv_heads=1,
        topk=40,
        causal=True,
        force_fused=True,
        seed=151,
    )
    inputs = make_attention_inputs(case, device)
    with pytest.raises(NotImplementedError, match="topk <= 36"):
        msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            cu_seqlens_k=inputs["cu_seqlens_k"],
            backend="vibecuda",
        )


def test_vibecuda_validates_group_size() -> None:
    device = require_supported_msa_gpu()
    from flashinfer.msa_ops import msa_sparse_attention

    case = _case(
        "sparse_prefill",
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        q_lens=[128, 128],
        kv_lens=[1024, 1024],
        num_q_heads=36,
        num_kv_heads=2,
        topk=8,
        causal=True,
        seed=149,
    )
    inputs = make_attention_inputs(case, device)
    with pytest.raises(ValueError, match="GQA group size"):
        msa_sparse_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            causal=True,
            backend="vibecuda",
        )
