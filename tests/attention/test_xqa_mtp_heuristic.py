"""SM90 verification for the provisional automatic XQA MTP rule."""

import pytest
import torch

import flashinfer
import flashinfer.decode as decode


def _reference(q, kv, indices, batch, qlen, kvlen):
    cache = kv.index_select(0, indices.long()).reshape(
        batch, kvlen // 16, 2, 16, 8, 128
    )
    query = q.float().reshape(batch, qlen, 8, 4, 128)
    allowed = torch.arange(kvlen, device=q.device)[None, :] <= (
        kvlen - qlen + torch.arange(qlen, device=q.device)[:, None]
    )
    result = []
    for b in range(batch):
        k = cache[b, :, 0].reshape(kvlen, 8, 128).float()
        v = cache[b, :, 1].reshape(kvlen, 8, 128).float()
        scores = torch.einsum("qngd,knd->ngqk", query[b], k) / (128**0.5)
        scores.masked_fill_(~allowed, -torch.inf)
        result.append(torch.einsum("ngqk,knd->qngd", scores.softmax(-1), v))
    return torch.stack(result).reshape_as(q)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="The automatic XQA MTP policy is limited to SM90",
)
@pytest.mark.parametrize(
    "batch,qlen,kvlen",
    [(4, 8, 8192), (16, 8, 8192), (64, 8, 8192), (16, 4, 1024), (16, 8, 1024)],
)
@pytest.mark.parametrize("backend", ["auto", "fa2"])
def test_xqa_mtp_auto_graph_correctness_and_replan(
    batch, qlen, kvlen, backend, monkeypatch
):
    torch.manual_seed(3859)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    pages = kvlen // 16
    q = torch.randn(batch * qlen, 32, 128, dtype=torch.float16, device=device)
    kv = torch.randn(batch * pages, 2, 16, 8, 128, dtype=torch.float16, device=device)
    indices = torch.randperm(batch * pages, dtype=torch.int32, device=device)
    indptr = torch.arange(batch + 1, dtype=torch.int32) * pages
    last = torch.full((batch,), 16, dtype=torch.int32)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        use_cuda_graph=True,
        use_tensor_cores=True,
        backend=backend,
        paged_kv_indptr_buffer=indptr.to(device),
        paged_kv_indices_buffer=indices.clone(),
        paged_kv_last_page_len_buffer=last.to(device),
    )

    def plan():
        wrapper.plan(
            indptr,
            indices,
            last,
            32,
            8,
            128,
            16,
            q_len_per_req=qlen,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    plan()
    assert wrapper._use_xqa_mtp_decode == (backend == "auto")
    assert wrapper.resolved_backend == ("xqa" if backend == "auto" else "fa2")
    out = torch.empty_like(q)
    wrapper.run(q, kv, out=out)
    torch.testing.assert_close(
        out.float(),
        _reference(q, kv, indices, batch, qlen, kvlen),
        atol=2e-3,
        rtol=1e-2,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            wrapper.run(q, kv, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out)
    q.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    previous_reference = _reference(q, kv, indices, batch, qlen, kvlen)
    torch.testing.assert_close(out.float(), previous_reference, atol=2e-3, rtol=1e-2)

    fields = ("_block_tables", "_kv_lens_buffer", "_xqa_mtp_causal_mask")
    pointers = (
        {name: getattr(wrapper, name).data_ptr() for name in fields}
        if backend == "auto"
        else {}
    )
    indices.copy_(indices.flip(0))
    plan()
    assert all(
        getattr(wrapper, name).data_ptr() == pointer
        for name, pointer in pointers.items()
    )
    graph.replay()
    torch.cuda.synchronize()
    current_reference = _reference(q, kv, indices, batch, qlen, kvlen)
    assert not torch.allclose(
        previous_reference, current_reference, atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(out.float(), current_reference, atol=2e-3, rtol=1e-2)

    calls = []
    original = decode.xqa_batch_decode_with_kv_cache

    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(decode, "xqa_batch_decode_with_kv_cache", observe)
    lse_out, lse = wrapper.run(q, kv, return_lse=True)
    assert not calls
    assert lse.shape == (batch * qlen, 32)
    torch.testing.assert_close(lse_out.float(), current_reference, atol=2e-3, rtol=1e-2)
    # A fallback call must not invalidate the workspace used by captured XQA.
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), current_reference, atol=2e-3, rtol=1e-2)
