"""Tests for batch_offsets_units="tokens" on the cuDNN prefill path.

Token-unit indptrs must produce the same results as the historical
element-unit offsets, both on the direct path (cuDNN consumes the indptrs
via cu_seq_len + ragged-offset multipliers, no conversion pre-pass) and on
the conversion path (FlashInfer scales them to element units internally).
"""

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [32, 87])
@pytest.mark.parametrize("s_kv", [87, 512])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr(
    monkeypatch,
    batch_size,
    s_qo,
    s_kv,
    num_kv_heads,
    num_qo_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    direct,
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    if not direct:
        # Force the conversion path even where the direct path is supported.
        monkeypatch.setattr(
            cudnn_prefill, "_cudnn_supports_direct_seqlens", lambda dtype: False
        )

    torch.manual_seed(0)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()

    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )

    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)

    common = dict(
        scale=float(1.0 / (head_dim_qk**0.5)),
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
        causal=causal,
        return_lse=True,
    )

    out_ref, lse_ref = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        batch_offsets_q=qo_indptr * (num_qo_heads * head_dim_qk),
        batch_offsets_o=qo_indptr * (num_qo_heads * head_dim_vo),
        batch_offsets_k=kv_indptr * (num_kv_heads * head_dim_qk),
        batch_offsets_v=kv_indptr * (num_kv_heads * head_dim_vo),
    )

    out, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )

    if direct:
        # There is no fallback: reaching here proves the direct
        # (unified-engine) path ran. Engines differ from the reference run,
        # so compare with tolerances.
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)
    else:
        # The conversion path scales the indptrs and builds the identical
        # legacy graph, so results are bitwise equal.
        assert torch.equal(out, out_ref)
        assert torch.equal(lse, lse_ref)


@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr_omit_actual_seq_lens(monkeypatch, direct):
    """actual_seq_lens_q/kv are optional with token-unit indptrs.

    The direct path never reads them, and the conversion path derives them
    from the indptrs, so omitting them must be bitwise-identical to passing
    them within either path.
    """
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")

    if not direct:
        monkeypatch.setattr(
            cudnn_prefill, "_cudnn_supports_direct_seqlens", lambda dtype: False
        )

    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_qo, s_kv = 4, 87, 512
    num_qo_heads, num_kv_heads, head_dim = 8, 4, 128

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()

    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    common = dict(
        scale=float(1.0 / (head_dim**0.5)),
        workspace_buffer=torch.empty(
            512 * 1024 * 1024, dtype=torch.int8, device=device
        ),
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        causal=True,
        return_lse=True,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )

    out_with, lse_with = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        **common,
        actual_seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
    )
    out_without, lse_without = cudnn_batch_prefill_with_kv_cache(
        q, k_cache, v_cache, **common
    )

    assert torch.equal(out_without, out_with)
    assert torch.equal(lse_without, lse_with)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_cudnn_prefill_lse_is_base2(num_kv_heads):
    """cuDNN returns base-2 LSE (log2 sum exp scores), matching every other
    FlashInfer backend. cuDNN's frontend emits natural-log stats, so the prefill
    path folds them to base-2; verify against a float reference."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    from flashinfer.utils import log2e

    torch.manual_seed(0)
    device = "cuda:0"
    s_q, s_kv, num_qo_heads, head_dim = 37, 64, 8, 128
    scale = float(head_dim**-0.5)

    q = torch.randn(s_q, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # batch_size == 1: packed q starts at token 0, so no ragged offsets needed.
    _, lse = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale=scale,
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_q,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=torch.tensor([s_q], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        actual_seq_lens_kv=torch.tensor([s_kv], dtype=torch.int32, device=device).view(
            1, 1, 1, 1
        ),
        causal=False,
        return_lse=True,
    )

    # Float reference LSE in base-2. GQA: broadcast each kv head to its q group.
    kf = k.float().repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale  # [h_qo, s_q, s_kv]
    lse_ref = torch.logsumexp(scores, dim=-1) * log2e  # base-2, [h_qo, s_q]

    lse_cudnn = lse[0, :s_q, :].transpose(0, 1)  # [h_qo, s_q]
    torch.testing.assert_close(lse_cudnn, lse_ref, atol=1e-2, rtol=1e-2)
