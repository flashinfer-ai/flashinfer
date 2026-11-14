# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import flashinfer
from flashinfer.cudnn import prefill as cudnn_prefill


@pytest.mark.skipif(
    not cudnn_prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support"
)
@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("dtype,backend", [(torch.bfloat16, "auto")])
def test_paged_gqa_auto_retained_capture(dtype, backend, return_lse, dim):
    """Physical token origins and explicit masks work on 9.24 without mixed forms."""
    from packaging.version import Version

    cudnn = cudnn_prefill.cudnn
    if (
        cudnn.backend_version() < 92400
        or Version(cudnn.__version__) < Version("1.33.0")
        or not cudnn_prefill._cudnn_supports_paged_gqa_auto(dim)
        or torch.cuda.get_device_capability() != (10, 0)
        or not cudnn_prefill.cudnn_frontend_frost_runtime_available()
    ):
        pytest.skip(
            "requires B200, cuDNN 9.24+, FE 1.33+ matching native and its DSL runtime"
        )
    torch.manual_seed(171)
    batch, capacity, max_kv, hq, hk, page = 2, 128, 16384, 32, 8, 16
    pages = max_kv // page
    # T3HD Q storage must not be confused with the contiguous output stride.
    q_storage = torch.randn(batch * capacity, 3, hq, dim, device="cuda", dtype=dtype)
    q = q_storage[:, 0]
    kv = torch.randn(batch * pages, 2, hk, page, dim, device="cuda", dtype=dtype)
    k, v = kv[:, 0], kv[:, 1]
    offsets = torch.arange(batch + 1, dtype=torch.int32) * capacity
    pageptr = torch.arange(batch + 1, dtype=torch.int32) * pages
    indices = torch.arange(batch * pages, device="cuda", dtype=torch.int32)
    table = indices.view(batch, pages).clone()
    q_lengths = torch.tensor([96, 128], dtype=torch.int32)
    kv_lengths = torch.tensor([16381, 8195], dtype=torch.int32)
    last = (kv_lengths - 1) % page + 1
    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "HND",
        backend=backend,
        use_cuda_graph=True,
        qo_indptr_buf=offsets.cuda(),
        paged_kv_indptr_buf=pageptr.cuda(),
        paged_kv_indices_buf=indices.clone(),
        paged_kv_last_page_len_buf=last.cuda(),
    )

    def plan():
        wrapper.plan(
            offsets,
            pageptr,
            indices,
            last,
            hq,
            hk,
            dim,
            page,
            causal=True,
            q_data_type=dtype,
            seq_lens=kv_lengths,
            seq_lens_q=q_lengths,
            block_tables=table,
            max_token_per_sequence=capacity,
            max_sequence_kv=max_kv,
        )
        assert wrapper._backend == "cudnn"
        assert wrapper._cudnn_plan is not None
        metadata = wrapper._cudnn_plan.metadata
        assert metadata._token_offsets
        assert metadata.cu_seq_lens_q is None
        assert metadata.actual_seq_lens_q is not None
        assert metadata.actual_seq_lens_kv is not None

    plan()
    out = torch.empty_like(q)
    lse = torch.empty(batch * capacity, hq, device="cuda", dtype=torch.float32)

    def run():
        return wrapper.run(
            q,
            (k, v),
            out=out,
            lse=lse if return_lse else None,
            return_lse=return_lse,
            lse_base="ln",
        )

    def check():
        for b, (nq, nk) in enumerate(
            zip(q_lengths.tolist(), kv_lengths.tolist(), strict=True)
        ):
            rows = slice(b * capacity, b * capacity + nq)
            keys = (
                k[table[b].long()]
                .transpose(1, 2)
                .reshape(max_kv, hk, dim)[:nk]
                .double()
            )
            values = (
                v[table[b].long()]
                .transpose(1, 2)
                .reshape(max_kv, hk, dim)[:nk]
                .double()
            )
            mask = torch.arange(nk, device="cuda")[None, :] > (
                torch.arange(nq, device="cuda")[:, None] + nk - nq
            )
            for head in range(hq):
                scores = (
                    q[rows, head].double() @ keys[:, head // (hq // hk)].T * dim**-0.5
                )
                scores.masked_fill_(mask, -torch.inf)
                ref = scores.softmax(-1) @ values[:, head // (hq // hk)]
                torch.testing.assert_close(
                    out[rows, head].float(), ref.float(), atol=0.02, rtol=0.02
                )
                if return_lse:
                    torch.testing.assert_close(
                        lse[rows, head],
                        scores.logsumexp(-1).float(),
                        atol=0.002,
                        rtol=0.002,
                    )

    run()
    check()
    builds = cudnn_prefill._prefill_graph_builds
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        run()
    q_lengths.sub_(16)
    kv_lengths.sub_(31)
    last.copy_((kv_lengths - 1) % page + 1)
    table.copy_(table.flip(1))
    q.mul_(0.75)
    kv.mul_(0.5)
    plan()
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    capture.replay()
    check()
    assert cudnn_prefill._prefill_graph_builds == builds
