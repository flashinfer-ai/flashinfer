# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Public D128/D256 paged-auto admission and captured execution on Blackwell."""

import pytest
import torch
from flashinfer.cudnn import prefill as cudnn_prefill
import flashinfer

# name, paired Q/KV lengths, Hq/Hkv, metadata location, causal admission.
_PAGED_D128_CASES = [
    ("prefix_floor", [64], [2048], 64, 8, "cpu", True),
    ("prefix_q_below", [63], [2048], 64, 8, "cpu", False),
    ("work_below", [64], [2048], 32, 8, "cpu", False),
    ("work_floor", [128], [8192], 8, 8, "cpu", True),
    ("prefix_upper", [1024], [4096], 8, 4, "cpu", True),
    ("extended_prefix", [1025], [4096], 8, 2, "cpu", True),
    ("mixed_q128", [128, 129, 129, 129, 129], [2048] * 5, 16, 1, "cpu", False),
    ("mixed_q129", [129] * 5, [2048] * 5, 16, 1, "cpu", True),
    ("square_floor", [1024], [1024], 8, 4, "cpu", True),
    ("square_below", [1023], [1023], 8, 4, "cpu", False),
    ("square_batch5", [1024] * 5, [1024] * 5, 8, 4, "cpu", True),
    ("square_upper", [16384], [16384], 8, 1, "cpu", True),
    ("device_prefix", [128], [8192], 8, 8, "cuda", True),
    ("device_square", [1024], [1024], 8, 4, "cuda", False),
    ("long_kv", [128, 257], [32768, 16385], 8, 2, "cpu", True),
]


_PAGED_D256_CASES = [
    ("work_floor", [128], [4096], 32, 8, "cpu", True),
    ("work_below", [128], [4095], 32, 8, "cpu", False),
    ("q_floor_gqa2", [64], [4096], 64, 32, "cpu", True),
    ("q_below", [63], [8192], 64, 32, "cpu", False),
    ("q_upper", [1024], [4096], 8, 1, "cpu", True),
    ("q_above", [1025], [8192], 8, 1, "cpu", False),
    ("long_kv", [64], [32768], 8, 4, "cpu", True),
    ("mixed_short", [64, 128], [8192, 4096], 16, 2, "cpu", True),
    ("mixed_long", [129, 257], [8192, 4097], 16, 1, "cpu", True),
    ("mixed_q_class", [128, 129], [8192, 8192], 16, 4, "cpu", False),
    ("batch4", [64, 64, 64, 64], [2048, 2048, 2048, 2048], 32, 8, "cpu", True),
    (
        "batch5",
        [64, 64, 64, 64, 64],
        [2048, 2048, 2048, 2048, 2048],
        32,
        8,
        "cpu",
        False,
    ),
    ("mha", [128], [16384], 8, 8, "cpu", False),
    ("device_lengths", [128], [16384], 8, 4, "cuda", False),
    ("kv_below", [128], [2047], 64, 8, "cpu", False),
    ("kv_above", [128], [32769], 8, 1, "cpu", False),
    ("kv_ratio_below", [1024], [4095], 8, 1, "cpu", False),
    ("gqa16", [128], [8192], 16, 1, "cpu", True),
]

_CASES = [
    (dim, case)
    for dim, cases in ((128, _PAGED_D128_CASES), (256, _PAGED_D256_CASES))
    for case in cases
]


@pytest.mark.parametrize(
    "dim,case", _CASES, ids=[f"d{dim}_{case[0]}" for dim, case in _CASES]
)
@pytest.mark.parametrize("causal", [True, False])
def test_paged_auto_boundary_replay(dim, case, causal):
    """Exercise both sides of real admission rules, then replay changed Q/K/V."""
    from flashinfer.prefill import (
        _cudnn_supports_paged_auto,
        _cudnn_supports_paged_gqa_auto,
    )

    if (
        not cudnn_prefill.CUDNN_AVAILABLE
        or torch.cuda.get_device_capability() != (10, 0)
        or not (
            _cudnn_supports_paged_auto()
            if dim == 128
            else _cudnn_supports_paged_gqa_auto(dim)
        )
    ):
        pytest.skip("requires SM100 and the cuDNN paged-auto runtime")
    name, qs, ks, hq, hk, metadata_device, admitted = case
    torch.manual_seed(1781)
    dtype, page = torch.bfloat16, 16
    qptr = torch.tensor(
        [0] + list(torch.tensor(qs).cumsum(0).tolist()), dtype=torch.int32
    )
    pages = [(n + page - 1) // page for n in ks]
    pptr = torch.tensor(
        [0] + list(torch.tensor(pages).cumsum(0).tolist()), dtype=torch.int32
    )
    indices = torch.randperm(sum(pages), device="cuda", dtype=torch.int32)
    last = torch.tensor([(n - 1) % page + 1 for n in ks], dtype=torch.int32)
    lengths = torch.tensor(ks, dtype=torch.int32, device=metadata_device)
    packed = torch.randn(sum(qs), 3, hq, dim, device="cuda", dtype=dtype)
    q = packed[:, 0]
    kv = torch.randn(sum(pages), 2, hk, page, dim, device="cuda", dtype=dtype)
    ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    auto = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="auto")
    kwargs = dict(
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        seq_lens=lengths,
        max_token_per_sequence=max(qs),
        max_sequence_kv=max(ks),
    )
    args = (qptr, pptr, indices, last, hq, hk, dim, page)
    auto.plan(*args, **kwargs)
    assert auto._backend == ("cudnn" if admitted and causal else "fa2"), (
        name,
        auto._backend,
    )
    ws_ref = torch.empty_like(ws)
    native = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws_ref, "HND", backend="fa2"
    )
    native.plan(*args, **kwargs)
    out = torch.empty_like(q)
    lse = torch.empty(sum(qs), hq, device="cuda", dtype=torch.float32)

    def run():
        return auto.run(q, kv, out=out, lse=lse, return_lse=True, lse_base="ln")

    def check():
        ref, stats = native.run(q, kv, return_lse=True, lse_base="ln")
        torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)
        torch.testing.assert_close(lse, stats, atol=0.002, rtol=0.002)
        # A separate FP64 oracle checks tile boundaries and both ends of every
        # request, without materializing a quadratic matrix at 16K square.
        for b, (nq, nk) in enumerate(zip(qs, ks, strict=True)):
            chosen = sorted(
                {
                    0,
                    nq - 1,
                    nq // 2,
                    min(63, nq - 1),
                    min(127, nq - 1),
                    min(128, nq - 1),
                }
            )
            rows = torch.tensor(chosen, device="cuda")
            ids = indices[int(pptr[b]) : int(pptr[b + 1])].long()
            k = kv[ids, 0].transpose(1, 2).reshape(-1, hk, dim)[:nk].double()
            v = kv[ids, 1].transpose(1, 2).reshape(-1, hk, dim)[:nk].double()
            for h in range(hq):
                scores = (
                    q[int(qptr[b]) + rows, h].double()
                    @ k[:, h // (hq // hk)].T
                    * dim**-0.5
                )
                if causal:
                    scores.masked_fill_(
                        torch.arange(nk, device="cuda")[None, :]
                        > rows[:, None] + nk - nq,
                        -torch.inf,
                    )
                value = scores.softmax(-1) @ v[:, h // (hq // hk)]
                torch.testing.assert_close(
                    out[int(qptr[b]) + rows, h].float(),
                    value.float(),
                    atol=0.02,
                    rtol=0.02,
                )
                torch.testing.assert_close(
                    lse[int(qptr[b]) + rows, h],
                    scores.logsumexp(-1).float(),
                    atol=0.002,
                    rtol=0.002,
                )

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.75)
    kv.mul_(0.5)
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    check()
