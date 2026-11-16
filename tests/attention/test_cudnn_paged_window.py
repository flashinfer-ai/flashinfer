# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import flashinfer
from flashinfer.cudnn import prefill as cudnn_prefill


@pytest.mark.skipif(
    not cudnn_prefill.CUDNN_AVAILABLE, reason="requires cuDNN graph support"
)
@pytest.mark.parametrize(
    "dim,window,page,causal",
    [
        (64, 0, 16, True),
        (64, 127, 64, True),
        (128, 0, 64, True),
        (128, 127, 16, True),
        (256, 127, 64, True),
        (256, 1023, 16, True),
        (128, -1, 16, True),
        (128, 127, 16, False),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("dtype,backend", [(torch.bfloat16, "cudnn")])
def test_paged_window_retained_capture(
    dtype, backend, return_lse, dim, window, page, causal
):
    """Inclusive windows survive replanning, retained replay and scale fallback."""
    from packaging.version import Version

    cudnn = cudnn_prefill.cudnn
    if (
        cudnn.backend_version() < 92400
        or Version(cudnn.__version__) < Version("1.33.0")
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or not cudnn_prefill.cudnn_frontend_frost_runtime_available()
    ):
        pytest.skip("requires B200/B300, cuDNN 9.24+, FE 1.33+ and its DSL runtime")
    torch.manual_seed(171)
    batch, capacity, max_kv, hq, hk = 2, 128, 2048, 8, 2
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
    kv_lengths = torch.tensor([2045, 1027], dtype=torch.int32)
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
            causal=causal,
            window_left=window,
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
        assert metadata.window_left == (window if window >= 0 else None)
        assert metadata._token_offsets
        assert metadata.cu_seq_lens_q is None
        assert metadata.actual_seq_lens_q is not None
        assert metadata.actual_seq_lens_kv is not None

    plan()
    out = torch.empty_like(q)
    lse = torch.empty(batch * capacity, hq, device="cuda", dtype=torch.float32)

    def run(q_scale=None):
        return wrapper.run(
            q,
            (k, v),
            out=out,
            lse=lse if return_lse else None,
            return_lse=return_lse,
            lse_base="ln",
            q_scale=q_scale,
        )

    def check(q_scale=1.0, include_lse=True):
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
            positions = torch.arange(nq, device="cuda")[:, None] + nk - nq
            keys_idx = torch.arange(nk, device="cuda")[None, :]
            mask = torch.zeros((nq, nk), device="cuda", dtype=torch.bool)
            if causal:
                mask |= keys_idx > positions
            if window >= 0:
                mask |= keys_idx < positions - window
            for head in range(hq):
                scores = (
                    q[rows, head].double()
                    @ keys[:, head // (hq // hk)].T
                    * dim**-0.5
                    * q_scale
                )
                scores.masked_fill_(mask, -torch.inf)
                ref = scores.softmax(-1) @ values[:, head // (hq // hk)]
                torch.testing.assert_close(
                    out[rows, head].float(), ref.float(), atol=0.02, rtol=0.02
                )
                if return_lse and include_lse:
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

    # Non-unit scales use the non-prepared public path, which must carry the
    # same window; otherwise its dense attention produces a different result.
    run(q_scale=1.25)
    check(q_scale=1.25)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        capture.replay()
    torch.cuda.current_stream().wait_stream(stream)
    check()

    # The legacy forward API changes the window without calling plan(). It
    # must not execute the old window's prepared graph just because causal
    # stayed the same. Subsequent run() also uses the new window.
    window = 0 if window != 0 else 127
    out = wrapper.forward(q, (k, v), causal=causal, window_left=window)
    check(include_lse=False)
    run()
    check()
