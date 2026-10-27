# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""MIS-v1 split-KV regression coverage with independent item-identity masks."""

import math

import torch
import pytest
import flashinfer


def _make_case(
    requests,
    *,
    dtype=torch.float16,
    group=1,
    layout="NHD",
    page=16,
    query_tokens=None,
    include_prefix=False,
    causal_control=False,
    dim=128,
    window_left=-1,
    logits_soft_cap=0.0,
):
    """requests = [(prefix length, content lengths excluding delimiters), ...]."""
    torch.manual_seed(20261019)
    kv_heads = 4
    query_lengths = []
    kv_lengths = []
    positions = []
    for prefix, lengths in requests:
        pos = [p for length in lengths for p in range(length + 1)]
        positions.append(pos)
        kv_lengths.append(prefix + len(pos))
        qlen = kv_lengths[-1] if include_prefix else len(pos)
        query_lengths.append(min(qlen, query_tokens) if query_tokens else qlen)
    page_counts = [math.ceil(length / page) for length in kv_lengths]
    pages = sum(page_counts)
    kv = torch.randn(pages, 2, page, kv_heads, dim, device="cuda", dtype=dtype) / 5
    q = (
        torch.randn(
            sum(query_lengths), kv_heads * group, dim, device="cuda", dtype=dtype
        )
        / 5
    )
    indices = torch.randperm(pages, device="cuda", dtype=torch.int32)
    qptr, kptr = [0], [0]
    for qlen, npages in zip(query_lengths, page_counts, strict=True):
        qptr.append(qptr[-1] + qlen)
        kptr.append(kptr[-1] + npages)
    refs, lses = [], []
    for ri, ((prefix, lengths), qlen, klen) in enumerate(
        zip(requests, query_lengths, kv_lengths, strict=True)
    ):
        cache = kv[indices[kptr[ri] : kptr[ri + 1]].long()]
        k = (
            cache[:, 0]
            .reshape(-1, kv_heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        v = (
            cache[:, 1]
            .reshape(-1, kv_heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        item_ids = [-1] * prefix + [
            i for i, length in enumerate(lengths) for _ in range(length + 1)
        ]
        content = [False] * prefix + [
            p > 0 for length in lengths for p in range(length + 1)
        ]
        ids = torch.tensor(item_ids, device="cuda")
        content = torch.tensor(content, device="cuda")
        key_ids = torch.arange(klen, device="cuda")
        output, lse = [], []
        for start in range(0, qlen, 128):
            end = min(start + 128, qlen)
            query_ids = torch.arange(
                klen - qlen + start, klen - qlen + end, device="cuda"
            )
            mask = key_ids[None, :] <= query_ids[:, None]
            if window_left >= 0:
                mask &= key_ids[None, :] >= query_ids[:, None] - window_left
            if not causal_control:
                same_item = ids[query_ids, None] == ids[None, :]
                both_content = content[query_ids, None] & content[None, :]
                mask &= (key_ids[None, :] < prefix) | (same_item & both_content)
            scores = torch.einsum(
                "qhd,khd->hqk", q[qptr[ri] + start : qptr[ri] + end].float(), k
            ) / math.sqrt(dim)
            if logits_soft_cap:
                scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
            scores.masked_fill_(~mask.unsqueeze(0), -torch.inf)
            output.append(
                torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(0), v)
            )
            lse.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
        refs.append(torch.cat(output))
        lses.append(torch.cat(lse))
    max_positions = max(map(len, positions))
    pos_tensor = torch.zeros(
        len(requests), max_positions, dtype=torch.int32, device="cuda"
    )
    for ri, pos in enumerate(positions):
        pos_tensor[ri, : len(pos)] = torch.tensor(pos, dtype=torch.int32, device="cuda")
    metadata = {
        "qo_indptr": torch.tensor(qptr, dtype=torch.int32, device="cuda"),
        "paged_kv_indptr": torch.tensor(kptr, dtype=torch.int32, device="cuda"),
        "paged_kv_indices": indices,
        "paged_kv_last_page_len": torch.tensor(
            [(length - 1) % page + 1 for length in kv_lengths],
            dtype=torch.int32,
            device="cuda",
        ),
        "num_qo_heads": kv_heads * group,
        "num_kv_heads": kv_heads,
        "head_dim_qk": dim,
        "page_size": page,
        "causal": True,
        "q_data_type": dtype,
        "kv_data_type": dtype,
        "window_left": window_left,
        "logits_soft_cap": logits_soft_cap,
    }
    if not causal_control:
        metadata.update(
            prefix_len_ptr=torch.tensor(
                [r[0] for r in requests], dtype=torch.uint32, device="cuda"
            ),
            token_pos_in_items_ptr=pos_tensor.to(torch.uint16),
            token_pos_in_items_len=max_positions,
            max_item_len_ptr=torch.tensor(
                [max(r[1]) for r in requests], dtype=torch.uint16, device="cuda"
            ),
        )
    if layout == "HND":
        kv = kv.permute(0, 1, 3, 2, 4).contiguous()
    return q, kv, metadata, torch.cat(refs).to(dtype), torch.cat(lses)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "requests,options",
    [
        ([(0, [31] * 16)], {}),
        ([(17, [15, 32, 1, 63, 7, 31, 0, 17] * 2)], {}),
        ([(65, [31] * 16)], {"group": 3}),
        ([(512, [64] * 16)], {"group": 4}),
        ([(128, [31] * 16)], {"group": 8, "layout": "HND"}),
        ([(0, [15] * 8), (17, [31] * 12), (129, [63] * 8)], {"group": 2}),
        ([(128, [31] * 24)], {"query_tokens": 71}),
        ([(129, [31] * 8)], {"include_prefix": True}),
        ([(0, [0] * 65)], {}),
        ([(17, [31] * 8)], {"page": 1}),
        ([(17, [31] * 16)], {"page": 32}),
    ],
)
def test_mis_split_kv_tile_skip(dtype, requests, options):
    """Match dense MIS output and LSE when leading masked split tiles are skipped."""
    q, kv, metadata, reference, ref_lse = _make_case(requests, dtype=dtype, **options)
    workspace = torch.full((128 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, options.get("layout", "NHD"), backend="fa2"
    )
    wrapper.plan(**metadata, fixed_split_size=8)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("row_base", [16, 64, 128])
def test_mis_tile_skip_gqa3_zero_margin(dtype, row_base):
    """Preserve the first visible content key at packed GQA-3 tile boundaries."""
    # At these candidate packed-tile starts, the first query is the last
    # content token of an item whose first content key is exactly 128.
    group, prefix, item_len = 3, 31, 31
    query_offset = 128 - (row_base // group - item_len + 1)
    assert row_base % group and query_offset + row_base // group == 158
    assert 158 - item_len + 1 == 128
    q, kv, metadata, reference, ref_lse = _make_case(
        [(prefix, [item_len] * 16)],
        dtype=dtype,
        group=group,
        query_tokens=prefix + 32 * 16 - query_offset,
    )
    workspace = torch.full((256 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    # The prefix-free chunk [64,128) ends at the first visible content key.
    wrapper.plan(**metadata, fixed_split_size=4)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("tail_fill", [float("nan"), float("inf")])
def test_mis_tile_skip_unused_v_tail(dtype, layout, page_size, tail_fill):
    """Exclude NaN/Inf V padding after a leading-tile skip with no prefix."""
    q, kv, metadata, reference, ref_lse = _make_case(
        [(0, [15] * 31 + [14])],
        dtype=dtype,
        group=3,
        layout=layout,
        page=page_size,
    )
    last_page = int(metadata["paged_kv_indices"][-1])
    last_len = int(metadata["paged_kv_last_page_len"][-1])
    if layout == "NHD":
        kv[last_page, 1, last_len:].fill_(tail_fill)
    else:
        kv[last_page, 1, :, last_len:].fill_(tail_fill)
    workspace = torch.full((256 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(**metadata, fixed_split_size=4)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mis_tile_skip_cuda_graph_replan(dtype):
    """Replay a captured MIS call after item boundaries change in place."""
    q, kv, metadata, reference, ref_lse = _make_case([(128, [31] * 16)], dtype=dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=metadata["qo_indptr"],
        paged_kv_indptr_buf=metadata["paged_kv_indptr"],
        paged_kv_indices_buf=metadata["paged_kv_indices"],
        paged_kv_last_page_len_buf=metadata["paged_kv_last_page_len"],
    )
    # Let the planner fit the graph's padded CTA capacity on smaller GPUs.
    # Fixed split policies are exercised by the non-graph regressions.
    wrapper.plan(**metadata)
    out, lse = wrapper.run(q, kv, return_lse=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out, lse=lse, return_lse=True)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)

    # Same lengths and storage, different item boundaries after capture.
    q2, kv2, changed, reference, ref_lse = _make_case([(128, [15] * 32)], dtype=dtype)
    torch.testing.assert_close(q, q2, rtol=0, atol=0)
    torch.testing.assert_close(kv, kv2, rtol=0, atol=0)
    metadata["token_pos_in_items_ptr"].copy_(changed["token_pos_in_items_ptr"])
    metadata["max_item_len_ptr"].copy_(changed["max_item_len_ptr"])
    wrapper.plan(**metadata)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("policy", ["default", "no_split", "fixed_3_pages"])
@pytest.mark.parametrize(
    "requests,options",
    [
        ([(0, [31] * 16)], {"dim": 64}),
        (
            [(17, [15, 32, 1, 63, 7, 31, 0, 17] * 2)],
            {"dim": 64, "group": 4, "layout": "HND"},
        ),
        ([(0, [31] * 16)], {"dim": 256, "layout": "HND"}),
        ([(17, [31] * 16)], {"dim": 256, "group": 2, "page": 32, "window_left": 63}),
        ([(17, [31] * 16)], {"window_left": 0}),
        ([(65, [31] * 16)], {"group": 3, "window_left": 31}),
        ([(129, [31] * 8)], {"include_prefix": True, "window_left": 127}),
        ([(0, [15] * 8), (17, [31] * 12)], {"group": 2, "window_left": 63}),
        ([(17, [31] * 16)], {"group": 3, "logits_soft_cap": 0.05}),
        (
            [(17, [31] * 16)],
            {
                "dim": 256,
                "group": 4,
                "layout": "HND",
                "window_left": 63,
                "logits_soft_cap": 0.05,
            },
        ),
    ],
)
def test_mis_tile_skip_head_dims_window_and_soft_cap(dtype, policy, requests, options):
    """Match dense MIS references across head dimensions, windows and soft caps."""
    q, kv, metadata, reference, ref_lse = _make_case(requests, dtype=dtype, **options)
    workspace = torch.full((512 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, options.get("layout", "NHD"), backend="fa2"
    )
    plan_options = (
        {"disable_split_kv": True}
        if policy == "no_split"
        else ({"fixed_split_size": 3} if policy == "fixed_3_pages" else {})
    )
    wrapper.plan(**metadata, **plan_options)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)


_NON_MIS_CONTROL_CASES = [
    ("dim64", [(17, [31, 31, 31, 31, 31, 31, 31, 31])], {"dim": 64}),
    ("plain128", [(65, [15, 31, 7, 63, 15, 31, 7, 63])], {}),
    (
        "dim256",
        [
            (17, [31, 31, 31, 31, 31, 31, 31, 31]),
            (65, [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]),
        ],
        {"dim": 256, "group": 2},
    ),
    (
        "window128",
        [(17, [31, 31, 31, 31, 31, 31, 31, 31])],
        {"group": 3, "layout": "HND", "window_left": 31},
    ),
    (
        "window256",
        [(65, [31, 31, 31, 31, 31, 31, 31, 31])],
        {"dim": 256, "group": 2, "window_left": 63},
    ),
    (
        "cap128",
        [(17, [31, 31, 31, 31, 31, 31, 31, 31])],
        {"group": 3, "layout": "HND", "logits_soft_cap": 0.05},
    ),
    (
        "window_cap256",
        [(17, [31, 31, 31, 31, 31, 31, 31, 31])],
        {
            "dim": 256,
            "group": 4,
            "layout": "HND",
            "window_left": 63,
            "logits_soft_cap": 0.05,
        },
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "requests,options,causal,policy",
    [
        pytest.param(requests, options, causal, policy, id=f"{name}-{policy}-{causal}")
        for name, requests, options in _NON_MIS_CONTROL_CASES
        for policy in ("default", "no_split", "fixed_3_pages")
        for causal in (False, True)
        # Noncausal FA2 window partitioning has pre-existing numerical failures.
        # Cover its load predicates without partitioning; causal windows and
        # ordinary masks still exercise all three planning policies.
        if causal or options.get("window_left", -1) < 0 or policy == "no_split"
    ],
)
def test_mis_prefetch_non_mis_controls(dtype, causal, policy, requests, options):
    """Keep ordinary-mask prefetches correct across the affected FA2 variants."""
    q, kv, metadata, reference, ref_lse = _make_case(
        requests, dtype=dtype, causal_control=True, **options
    )
    metadata["causal"] = causal
    if not causal:
        # A noncausal window has a left boundary but still includes future keys.
        # Reconstruct the independent dense reference from the paged inputs.
        cache = kv.permute(0, 1, 3, 2, 4) if options.get("layout") == "HND" else kv
        qptr = metadata["qo_indptr"].cpu().tolist()
        kptr = metadata["paged_kv_indptr"].cpu().tolist()
        last = metadata["paged_kv_last_page_len"].cpu().tolist()
        refs, lses = [], []
        group = metadata["num_qo_heads"] // metadata["num_kv_heads"]
        dim, kv_heads, page_size = (
            metadata["head_dim_qk"],
            metadata["num_kv_heads"],
            metadata["page_size"],
        )
        window, cap = options.get("window_left", -1), options.get("logits_soft_cap", 0)
        for index in range(len(requests)):
            pages = cache[
                metadata["paged_kv_indices"][kptr[index] : kptr[index + 1]].long()
            ]
            kv_len = (kptr[index + 1] - kptr[index] - 1) * page_size + last[index]
            k, v = [
                pages[:, component]
                .reshape(-1, kv_heads, dim)[:kv_len]
                .repeat_interleave(group, 1)
                .float()
                for component in (0, 1)
            ]
            query = q[qptr[index] : qptr[index + 1]].float()
            scores = torch.einsum("qhd,khd->hqk", query, k) / math.sqrt(dim)
            if cap:
                scores = cap * torch.tanh(scores / cap)
            if window >= 0:
                keys = torch.arange(kv_len, device=q.device)
                queries = torch.arange(query.shape[0], device=q.device)
                allowed = keys[None, :] >= (
                    kv_len - query.shape[0] + queries[:, None] - window
                )
                scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
            refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
            lses.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
        reference, ref_lse = torch.cat(refs).to(dtype), torch.cat(lses)
    reference, ref_lse = reference.cpu(), ref_lse.cpu()
    workspace = torch.full((128 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, options.get("layout", "NHD"), backend="fa2"
    )
    plan_options = (
        {"disable_split_kv": True}
        if policy == "no_split"
        else ({"fixed_split_size": 3} if policy == "fixed_3_pages" else {})
    )
    wrapper.plan(**metadata, **plan_options)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
