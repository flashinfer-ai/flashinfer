import math

import pytest
import torch

import flashinfer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120",
)


def _fp8(shape):
    return torch.randint(-2, 3, shape, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )


def _reference(q, k, v, causal, scale):
    hq, hkv = q.shape[1], k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    vf = v.float().transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * scale
    if causal:
        q_pos = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
        k_pos = torch.arange(k.shape[0], device=q.device)
        scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -torch.inf)
    lse = torch.logsumexp(scores, -1).transpose(0, 1) * math.log2(math.e)
    out = torch.einsum("hqk,hkd->hqd", scores.softmax(-1), vf).transpose(0, 1)
    return out, lse


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_ragged_public_wrapper(causal, out_dtype):
    hq, hkv, d = 4, 2, 32
    q_lens, kv_lens = [3, 2], [5, 4]
    qo = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
    q, k, v = _fp8((5, hq, d)), _fp8((9, hkv, d)), _fp8((9, hkv, d))
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        kv,
        hq,
        hkv,
        d,
        causal=causal,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=out_dtype,
    )
    out = torch.empty_like(q, dtype=out_dtype)
    q_scale, k_scale, v_scale = 0.75, 1.25, 0.5
    actual, actual_lse = wrapper.run_return_lse(
        q,
        k,
        v,
        out=out,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert actual is out
    refs, lses = [], []
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens, strict=False)):
        ref, ref_lse = _reference(
            q[qo[i] : qo[i] + ql],
            k[kv[i] : kv[i] + kl],
            v[kv[i] : kv[i] + kl],
            causal,
            q_scale * k_scale / math.sqrt(d),
        )
        refs.append(ref * v_scale)
        lses.append(ref_lse)
    torch.testing.assert_close(actual.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(actual_lse, torch.cat(lses), atol=0.2, rtol=0.2)


@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_paged_public_wrapper(page_size):
    hq, hkv, d = 4, 2, 32
    qo = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    page_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    page_indices = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([5, 4], dtype=torch.int32, device="cuda")
    q = _fp8((5, hq, d))
    k_pool = _fp8((5, page_size, hkv, d))
    v_pool = _fp8((5, page_size, hkv, d))
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    assert wrapper.workspace_size(
        qo,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        q_data_type=q.dtype,
        kv_data_type=k_pool.dtype,
        o_data_type=torch.bfloat16,
    ) == (0, 0)
    wrapper.plan(
        qo,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k_pool.dtype,
        o_data_type=torch.bfloat16,
    )
    actual, actual_lse = wrapper.run_return_lse(q, (k_pool, v_pool))
    refs, lses = [], []
    for batch, (q_start, q_end) in enumerate(zip(qo[:-1], qo[1:], strict=False)):
        pages = page_indices[page_indptr[batch] : page_indptr[batch + 1]]
        k = torch.cat([k_pool[p] for p in pages])[
            : ((len(pages) - 1) * page_size + last_page_len[batch])
        ]
        v = torch.cat([v_pool[p] for p in pages])[: k.shape[0]]
        ref, ref_lse = _reference(q[q_start:q_end], k, v, True, 1 / math.sqrt(d))
        refs.append(ref)
        lses.append(ref_lse)
    torch.testing.assert_close(actual.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(actual_lse, torch.cat(lses), atol=0.2, rtol=0.2)


def test_prims_rejects_non_contiguous_combined_cache_and_pdl():
    workspace = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
    qo = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    last = torch.tensor([1], dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo,
        indptr,
        indices,
        last,
        2,
        1,
        32,
        16,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    q = _fp8((1, 2, 32))
    combined = _fp8((1, 2, 16, 1, 32))
    with pytest.raises(ValueError, match="tuple"):
        wrapper.run(q, combined)
    with pytest.raises(NotImplementedError, match="enable_pdl"):
        wrapper.run(
            q,
            (combined[:, 0].contiguous(), combined[:, 1].contiguous()),
            enable_pdl=True,
        )
