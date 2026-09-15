"""cuDNN paged prefill correctness at large KV page sizes (128/256/512).

Regression evidence for FlashInfer issue #4347: the SM100 TRTLLM-gen causal
paged-context prefill silently skips KV pages for page sizes > 128 (observed
at 256) on 4K-16K sequences (BF16, GQA num_qo_heads=32 / num_kv_heads=2).
The cuDNN backend is the proposed fallback, so this file pins down that cuDNN
attends *every* page at large page sizes by comparing against a dense torch
SDPA reference computed over the gathered pages.  An unattended-page error of
the #4347 class shifts the softmax-weighted output by an amount comparable to
the output itself across millions of elements, so the standard tolerances used
by the other cuDNN prefill tests suffice to catch it.

Scope: cuDNN's paged prefill serves head_dim_qk 128/192 only, so #4347's exact
D=256 shape CANNOT be covered here.  The evidence is therefore honestly scoped
to the same page-size / sequence-length / GQA / dtype class -- BF16, causal,
num_qo_heads=32 / num_kv_heads=2, page sizes 128/256/512, KV lengths around 4K
and 8K (both non-multiples and an exact multiple of every tested page size) --
at head_dim 128.

Page tables use a non-identity permutation of page ids, so a kernel that walks
pages in the wrong order (not just one that stops early) also mismatches.
"""

import pytest
import torch

import flashinfer
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.utils import get_compute_capability

# Per-request (q_lens, kv_lens) crossing page boundaries non-trivially:
# 4096/8192 are exact multiples of every tested page size; 4000, 3333, 2749
# and 7777 are multiples of none, so the last used page is partially filled
# and must be masked.  Q lengths mix full self-attention prefill (lq == lkv)
# with chunked-context requests (lq < lkv) whose every Q row must attend all
# KV pages under the bottom-right causal mask.
_SEQ_LEN_CASES = [
    ([4000, 4096, 512, 1024], [4000, 4096, 3333, 2749]),
    ([7777, 1024], [7777, 8192]),
]
_SEQ_LEN_IDS = ["4k", "8k"]

_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 2
_HEAD_DIM = 128

# Q-row chunk for the fp32 dense reference: bounds the materialized score
# matrix to (num_qo_heads, 1024, s_kv) per chunk.
_REF_Q_CHUNK = 1024


def _skip_if_unsupported():
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major not in (9, 10):
        pytest.skip(
            f"cuDNN large-page prefill evidence targets SM90/SM100, got sm{major}x"
        )


def _make_paged_inputs(q_lens, kv_lens, page_size, device):
    """Build packed Q and a page-permuted paged KV cache for the given lens."""
    torch.manual_seed(42)
    batch_size = len(q_lens)
    num_pages_per_seq = (max(kv_lens) + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    seq_q = torch.tensor(q_lens, dtype=torch.int32, device=device)
    seq_kv = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    q = torch.randn(
        int(seq_q.sum()), _NUM_QO_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    # Token-major backing buffer ([page][k/v][token][head][dim]); permute to
    # the (page, head, token, dim) view the cuDNN API takes -- the same NHD
    # in-page strides test_cudnn_prefill fabricates with as_strided.
    kv_data = torch.randn(
        total_num_pages,
        2,
        page_size,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = kv_data[:, 0].permute(0, 2, 1, 3)
    v_cache = kv_data[:, 1].permute(0, 2, 1, 3)

    # Non-identity page mapping (see module docstring).
    perm = torch.randperm(total_num_pages, device=device)
    block_tables = perm.reshape(batch_size, num_pages_per_seq).int().contiguous()

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr_tokens = torch.cat([zero, torch.cumsum(seq_q, 0)]).int()
    return q, k_cache, v_cache, block_tables, seq_q, seq_kv, qo_indptr_tokens


def _dense_reference(q, k_cache, v_cache, block_tables, q_lens, kv_lens, scale):
    """fp32 dense SDPA over the gathered pages (bottom-right causal, GQA)."""
    device = q.device
    group = _NUM_QO_HEADS // _NUM_KV_HEADS
    outs = []
    qo_off = 0
    for i, (lq, lkv) in enumerate(zip(q_lens, kv_lens, strict=True)):
        pages = block_tables[i].long()  # (num_pages_per_seq,)
        # (pages, h_kv, page, d) -> (h_kv, tokens, d), valid tokens only.
        kk = k_cache[pages].transpose(0, 1).reshape(_NUM_KV_HEADS, -1, _HEAD_DIM)
        vv = v_cache[pages].transpose(0, 1).reshape(_NUM_KV_HEADS, -1, _HEAD_DIM)
        kk = kk[:, :lkv].float().repeat_interleave(group, dim=0)
        vv = vv[:, :lkv].float().repeat_interleave(group, dim=0)
        q_i = q[qo_off : qo_off + lq].float().transpose(0, 1)  # (h_qo, lq, d)
        kpos = torch.arange(lkv, device=device).unsqueeze(0)
        chunks = []
        for r0 in range(0, lq, _REF_Q_CHUNK):
            r1 = min(r0 + _REF_Q_CHUNK, lq)
            scores = torch.einsum("hqd,hkd->hqk", q_i[:, r0:r1], kk) * scale
            qpos = torch.arange(r0, r1, device=device).unsqueeze(1)
            allowed = kpos <= (lkv - lq) + qpos  # bottom-right causal
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            chunks.append(torch.einsum("hqk,hkd->qhd", p, vv))
        outs.append(torch.cat(chunks))
        qo_off += lq
    return torch.cat(outs)


@pytest.mark.parametrize("page_size", [128, 256, 512])
@pytest.mark.parametrize("q_lens,kv_lens", _SEQ_LEN_CASES, ids=_SEQ_LEN_IDS)
def test_cudnn_prefill_large_page(page_size, q_lens, kv_lens):
    """Direct cudnn_batch_prefill_with_kv_cache paged path at large page sizes."""
    _skip_if_unsupported()
    device = "cuda:0"

    q, k_cache, v_cache, block_tables, seq_q, seq_kv, qo_indptr_tokens = (
        _make_paged_inputs(q_lens, kv_lens, page_size, device)
    )
    batch_size = len(q_lens)
    scale = float(_HEAD_DIM**-0.5)
    # Element-unit Q/O offsets, exactly as BatchPrefillWithPagedKVCacheWrapper
    # passes them (d_qk == d_vo, so Q and O offsets coincide).
    qo_indptr_elems = qo_indptr_tokens * _NUM_QO_HEADS * _HEAD_DIM
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_token_per_sequence=max(q_lens),
        max_sequence_kv=max(kv_lens),
        actual_seq_lens_q=seq_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=seq_kv.view(batch_size, 1, 1, 1),
        block_tables=block_tables,
        causal=True,
        return_lse=False,
        batch_offsets_q=qo_indptr_elems,
        batch_offsets_o=qo_indptr_elems,
    )

    ref = _dense_reference(q, k_cache, v_cache, block_tables, q_lens, kv_lens, scale)
    torch.testing.assert_close(output.float(), ref, atol=2e-2, rtol=2e-2)


def test_cudnn_prefill_large_page_wrapper():
    """BatchPrefillWithPagedKVCacheWrapper(backend="cudnn") at page_size=256."""
    _skip_if_unsupported()
    device = "cuda:0"
    page_size = 256
    q_lens, kv_lens = _SEQ_LEN_CASES[0]

    q, k_cache, v_cache, block_tables, seq_q, seq_kv, qo_indptr_tokens = (
        _make_paged_inputs(q_lens, kv_lens, page_size, device)
    )
    batch_size = len(q_lens)
    scale = float(_HEAD_DIM**-0.5)
    qo_indptr_elems = qo_indptr_tokens * _NUM_QO_HEADS * _HEAD_DIM

    num_pages_used = (seq_kv + page_size - 1) // page_size
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    kv_indptr = torch.cat([zero, torch.cumsum(num_pages_used, 0)]).int()
    kv_indices = torch.cat(
        [block_tables[i, : num_pages_used[i]] for i in range(batch_size)]
    ).int()
    kv_last_page_len = torch.where(
        seq_kv % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        seq_kv % page_size,
    ).int()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper.plan(
        qo_indptr_elems,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        page_size,
        pos_encoding_mode="NONE",
        causal=True,
        q_data_type=torch.bfloat16,
        seq_lens=seq_kv.view(batch_size, 1, 1, 1),
        seq_lens_q=seq_q.view(batch_size, 1, 1, 1),
        sm_scale=scale,
        max_token_per_sequence=max(q_lens),
        max_sequence_kv=max(kv_lens),
        block_tables=block_tables,
    )
    output = wrapper.run(q, (k_cache, v_cache))

    ref = _dense_reference(q, k_cache, v_cache, block_tables, q_lens, kv_lens, scale)
    torch.testing.assert_close(output.float(), ref, atol=2e-2, rtol=2e-2)
