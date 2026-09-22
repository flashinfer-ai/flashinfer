# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
import math
import torch
from flashinfer.testing.sparse_mla import sparse_mla_reference


def test_joint_softmax_duplicates_and_sink():
    q = torch.zeros(1, 1, 2, 512, dtype=torch.bfloat16)
    swa = torch.full((2, 1, 512), 2.0, dtype=torch.bfloat16)
    compressed = torch.full((1, 2, 512), 10.0, dtype=torch.bfloat16)
    out, lse = sparse_mla_reference(
        q,
        swa,
        compressed,
        torch.tensor([[[1, 1, -1]]], dtype=torch.int32),
        torch.tensor([[[0, 1]]], dtype=torch.int32),
        compressed_topk_lens=torch.tensor([[1]], dtype=torch.int32),
        sinks=torch.tensor([math.log(2), torch.inf]),
    )
    # Two duplicated SWA entries plus one compressed entry. Sink contributes
    # denominator mass only. This distinguishes a joint softmax from averaging.
    torch.testing.assert_close(
        out[0, 0, 0], torch.full((512,), 14 / 5, dtype=torch.float64)
    )
    assert torch.count_nonzero(out[0, 0, 1]) == 0
    torch.testing.assert_close(
        lse, torch.full((1, 1, 2), math.log(3), dtype=torch.float64)
    )


def test_source_scales_padded_pages_and_empty_queries():
    q = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
    backing = torch.zeros(2, 3, 512, dtype=torch.bfloat16)
    swa = backing[:, :1]
    swa[0].fill_(3)
    swa[1].fill_(7)
    compressed = torch.full((1, 1, 512), 5.0, dtype=torch.bfloat16)
    out, lse = sparse_mla_reference(
        q,
        swa,
        compressed,
        torch.tensor([[1, -1], [-1, -1]], dtype=torch.int32),
        torch.tensor([[0], [0]], dtype=torch.int32),
        compressed_topk_lens=torch.tensor([1, 0], dtype=torch.int32),
        swa_kv_scale=2,
        compressed_kv_scale=4,
    )
    torch.testing.assert_close(out[0], torch.full((1, 512), 17.0, dtype=torch.float64))
    assert torch.count_nonzero(out[1]) == 0
    assert torch.isneginf(lse[1]).all()


def test_nonuniform_logits_scales_and_chunk_tail():
    q = torch.zeros(3, 1, 512, dtype=torch.bfloat16)
    q[..., 0] = 1
    kv = torch.stack((torch.ones(1, 512), torch.full((1, 512), 3.0))).to(q.dtype)
    out, lse = sparse_mla_reference(
        q,
        kv,
        None,
        torch.tensor([[0, 1], [1, -1], [-1, -1]], dtype=torch.int32),
        swa_topk_lens=torch.tensor([2, 1, 0], dtype=torch.int32),
        softmax_scale=1,
        q_scale=2,
        swa_kv_scale=0.5,
        output_scale=2,
        sinks=torch.tensor([math.log(4)], dtype=torch.float64),
        chunk_rows=2,
    )
    # QK logits are 1 and 3; scaled V values are 1 and 3. The sink adds 4
    # only to the denominator. The final partial chunk is entirely empty.
    e1, e3 = math.exp(1), math.exp(3)
    expected = torch.tensor(
        [(e1 + 3 * e3) / (e1 + e3 + 4), 3 * e3 / (e3 + 4), 0], dtype=torch.float64
    )
    torch.testing.assert_close(
        out, expected[:, None, None].expand_as(out), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        lse[:, 0],
        torch.tensor([math.log(e1 + e3), 3, -math.inf], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
