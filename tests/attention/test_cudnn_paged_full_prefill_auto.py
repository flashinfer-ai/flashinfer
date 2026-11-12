# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Auto-selected full prefills preserve a capture across query partitions/pages."""

import pytest
import torch
import flashinfer
from flashinfer.cudnn import prefill as cp


@pytest.mark.parametrize("dim,group", [(128, 4), (256, 1)])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("return_lse", [False, True])
def test_full_prefill_auto_retained_capture(dim, group, page, return_lse):
    if torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ) or not cp._cudnn_supports_paged_full_prefill_auto(dim):
        pytest.skip(
            "requires SM100/SM103, cuDNN 9.24 and FE 1.33 matching native/runtime"
        )
    torch.manual_seed(1783)
    hq, hk, total, max_s = 16, 16 // group, 2048, 1280
    lengths = torch.tensor([1280, 768], dtype=torch.int32)
    ptr = torch.tensor([0, 1280, total], dtype=torch.int32)
    per_row = max_s // page
    pageptr = torch.tensor([0, per_row, 2 * per_row], dtype=torch.int32)
    indices = torch.randperm(2 * per_row, device="cuda", dtype=torch.int32)
    table = indices.view(2, per_row).clone()
    last = (lengths - 1) % page + 1
    q_storage = torch.randn(total, 3, hq, dim, device="cuda", dtype=torch.bfloat16)
    q = q_storage[:, 0]
    kv = torch.randn(2 * per_row, 2, hk, page, dim, device="cuda", dtype=q.dtype)
    k, v = kv[:, 0], kv[:, 1]
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8),
        "HND",
        backend="auto",
        use_cuda_graph=True,
        qo_indptr_buf=ptr.cuda(),
        paged_kv_indptr_buf=pageptr.cuda(),
        paged_kv_indices_buf=indices.clone(),
        paged_kv_last_page_len_buf=last.cuda(),
    )

    def plan():
        wrapper.plan(
            ptr,
            pageptr,
            indices,
            last,
            hq,
            hk,
            dim,
            page,
            causal=True,
            q_data_type=q.dtype,
            seq_lens=lengths,
            block_tables=table,
            max_token_per_sequence=max_s,
            max_sequence_kv=max_s,
        )
        assert wrapper._backend == "cudnn" and wrapper._cudnn_plan is not None

    out = torch.empty(total, hq, dim, device="cuda", dtype=q.dtype)
    lse = torch.empty(total, hq, device="cuda", dtype=torch.float32)

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
        for b, n in enumerate(lengths.tolist()):
            rows = slice(int(ptr[b]), int(ptr[b + 1]))
            keys = (
                k[table[b].long()].transpose(1, 2).reshape(max_s, hk, dim)[:n].double()
            )
            values = (
                v[table[b].long()].transpose(1, 2).reshape(max_s, hk, dim)[:n].double()
            )
            mask = (
                torch.arange(n, device="cuda")[None, :]
                > torch.arange(n, device="cuda")[:, None]
            )
            for h in range(hq):
                scores = q[rows, h].double() @ keys[:, h // group].T * dim**-0.5
                scores.masked_fill_(mask, -torch.inf)
                ref = scores.softmax(-1) @ values[:, h // group]
                torch.testing.assert_close(
                    out[rows, h].float(), ref.float(), atol=0.02, rtol=0.02
                )
                if return_lse:
                    torch.testing.assert_close(
                        lse[rows, h],
                        scores.logsumexp(-1).float(),
                        atol=0.002,
                        rtol=0.002,
                    )

    plan()
    run()
    check()
    builds = cp._prefill_graph_builds
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        run()
    lengths.copy_(lengths.flip(0))
    ptr[1] = lengths[0]
    last.copy_((lengths - 1) % page + 1)
    table.copy_(table.flip(1))
    q.mul_(0.75)
    kv.mul_(0.5)
    plan()
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    capture.replay()
    check()
    assert cp._prefill_graph_builds == builds
