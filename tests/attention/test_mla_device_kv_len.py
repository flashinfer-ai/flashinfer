"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import math

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.utils import is_sm90a_supported


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("head_dim_kpe", [0, 64])
@pytest.mark.parametrize(
    "kv_dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
)
def test_mla_device_kv_length_replay(page_size, head_dim_kpe, kv_dtype):
    device = torch.device("cuda:0")
    if not is_sm90a_supported(device):
        pytest.skip("FA3 requires SM90a")
    torch.manual_seed(42)
    dtype = torch.float16 if kv_dtype == torch.float16 else torch.bfloat16
    num_heads, head_dim_ckv = 16, 512
    q_offsets = [0, 0, 1, 4, 6]
    upper_bounds = [128, 1024, 65, 33]
    pages = [(length + page_size - 1) // page_size for length in upper_bounds]
    page_offsets = [0]
    for count in pages:
        page_offsets.append(page_offsets[-1] + count)
    qo_indptr = torch.tensor(q_offsets, dtype=torch.int32)
    kv_indptr = torch.tensor(page_offsets, dtype=torch.int32)
    kv_indices = torch.randperm(page_offsets[-1], device=device).to(torch.int32)
    bounds = torch.tensor(upper_bounds, dtype=torch.int32)
    q = torch.randn(
        6, num_heads, head_dim_ckv + head_dim_kpe, device=device, dtype=dtype
    )
    kv = (
        torch.randn(
            page_offsets[-1], page_size, head_dim_ckv + head_dim_kpe, device=device
        )
        * 0.25
    ).to(kv_dtype)
    actual_lengths = torch.tensor([0, 0, 64, 1], dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(
        workspace,
        backend="fa3",
        use_cuda_graph=True,
        qo_indptr=qo_indptr.to(device),
        kv_indptr=kv_indptr.to(device),
        kv_indices=kv_indices.clone(),
        kv_len_arr=bounds.to(device),
    )
    wrapper.plan(
        metadata=MLAPlanMetadata.csr(qo_indptr, kv_indptr, kv_indices, bounds),
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=1 / math.sqrt(head_dim_ckv + head_dim_kpe),
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        lse_mode="basee",
    )
    scales = (
        {"ckv_scale": 1.0, "kpe_scale": 1.0} if kv_dtype == torch.float8_e4m3fn else {}
    )
    out = torch.empty(6, num_heads, head_dim_ckv, device=device, dtype=dtype)
    lse = torch.empty(6, num_heads, device=device)
    run_args = dict(
        query=q,
        kv_cache=kv,
        kv_len=actual_lengths,
        out=out,
        lse=lse,
        return_lse=True,
        return_lse_base_on_e=True,
        **scales,
    )
    wrapper.run(**run_args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(**run_args)
    for lengths in (
        [0, 0, 64, 1],
        [128, 1024, 65, 33],
        [3, 7, 0, 17],
        [1, 513, 64, 32],
    ):
        actual_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        graph.replay()
        expected = torch.zeros_like(out)
        expected_lse = torch.full_like(lse, -torch.inf)
        for batch, length in enumerate(lengths):
            start, end = q_offsets[batch : batch + 2]
            if start == end or length == 0:
                continue
            indices = kv_indices[page_offsets[batch] : page_offsets[batch + 1]].long()
            keys = kv[indices].reshape(-1, head_dim_ckv + head_dim_kpe)[:length].float()
            logits = torch.einsum("qhd,kd->qhk", q[start:end].float(), keys)
            logits *= 1 / math.sqrt(head_dim_ckv + head_dim_kpe)
            expected[start:end] = torch.einsum(
                "qhk,kd->qhd", logits.softmax(-1), keys[:, :head_dim_ckv]
            ).to(dtype)
            expected_lse[start:end] = logits.logsumexp(-1)
        if not torch.allclose(out, expected, atol=3e-3, rtol=3e-3):
            raise RuntimeError(
                f"Attention output mismatch for device lengths {lengths}"
            )
        if not torch.allclose(lse, expected_lse, atol=3e-3, rtol=3e-3):
            raise RuntimeError(f"Attention LSE mismatch for device lengths {lengths}")
    with pytest.raises(ValueError, match="dtype"):
        wrapper.run(**(run_args | {"kv_len": actual_lengths.to(torch.int64)}))
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(
            **(
                run_args
                | {"kv_len": torch.zeros(8, device=device, dtype=torch.int32)[::2]}
            )
        )
    with pytest.raises(ValueError, match="shape"):
        wrapper.run(**(run_args | {"kv_len": actual_lengths[:2]}))
