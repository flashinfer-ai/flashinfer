# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent-head sparse patterns and semantic-block/page-boundary coverage."""

import math

import pytest
import torch

from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    _build_q_token_kv_block_sparse_metadata,
    get_q_token_kv_block_sparse_workspace_size,
)

pytestmark = [
    pytest.mark.arch_blackwell,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_BLOCK_PAGE_PAIRS = (
    (4, 16),
    (8, 16),
    (16, 128),
    (32, 128),
    (64, 128),
    (128, 128),
    (32, 16),
    (128, 16),
    (128, 20),
)


@pytest.mark.parametrize("block,page", _BLOCK_PAGE_PAIRS)
@pytest.mark.parametrize("group", (1, 5, 8))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("shared", (False, True))
def test_sparse_pattern_metadata_graph(block, page, group, packed, shared):
    torch.manual_seed(78431)
    heads, topk, model_len = (1 if shared else 3), 17, 2048
    lengths = [group, max(1, group - 1)] if packed else [group, group]
    offsets = [0, lengths[0], sum(lengths)]
    rows = offsets[-1]
    ids = torch.full((rows, heads, topk), -1, device="cuda", dtype=torch.int32)
    input_ids = ids[:, 0] if shared else ids
    table_width = (model_len + page - 1) // page
    table_cpu = torch.randperm(2 * table_width, dtype=torch.int32).reshape(2, -1)
    table = table_cpu.cuda()
    requests = torch.tensor(
        [request for request, length in enumerate(lengths) for _ in range(length)],
        device="cuda",
        dtype=torch.int32,
    )
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)
    qo_indptr = (
        torch.tensor(offsets, device="cuda", dtype=torch.int32) if packed else None
    )
    fragment = math.gcd(block, page)
    inverse = {
        int(table_cpu[b, p]): (b, p) for b in range(2) for p in range(table_width)
    }

    def update(base, inert=False):
        ids_cpu = torch.full((rows, heads, topk), -1, dtype=torch.int32)
        expected = {}
        pos = []
        for request, length in enumerate(lengths):
            for qi in range(length):
                row = offsets[request] + qi
                visible = base + qi
                pos.append(visible - 1)
                for head in range(heads):
                    causal = visible // block
                    selected = torch.randperm(causal, dtype=torch.int32)[:topk]
                    ids_cpu[row, head, : selected.numel()] = selected
                    tokens = set(
                        (selected[:, None] * block + torch.arange(block))
                        .flatten()
                        .tolist()
                    )
                    tokens.update(range(causal * block, visible))
                    expected[row, head] = set() if inert and request == 1 else tokens
        ids.copy_(ids_cpu)
        positions.copy_(torch.tensor(pos, dtype=torch.int64, device="cuda"))
        requests[offsets[1] :].fill_(-1 if inert else 1)
        return expected

    def run(out=None):
        return _build_q_token_kv_block_sparse_metadata(
            input_ids,
            table,
            requests,
            positions,
            group_size=group,
            storage_page_size=page,
            max_seq_len_kv=model_len,
            sparse_block_size=block,
            qo_indptr=qo_indptr,
            out=out,
            share_pattern_across_kv_heads=shared,
        )

    expected = update(1024)
    outputs = run()

    def check():
        pages, memberships, seq_lens = outputs
        lens = seq_lens.cpu().tolist()
        pos = positions.cpu().tolist()
        for request, length in enumerate(lengths):
            for head in range(heads):
                route = request * heads + head
                live = lens[route]
                count = (live + fragment - 1) // fragment
                locators = pages[route, :count].cpu().tolist()
                words = (
                    memberships[route, : (count + 3) // 4].cpu().tolist()
                    if group > 1
                    else []
                )
                for qi in range(length):
                    row = offsets[request] + qi
                    actual = set()
                    for slot, locator in enumerate(locators):
                        member = (
                            group == 1
                            or (words[slot // 4] >> (8 * (slot % 4) + qi)) & 1
                        )
                        if locator < 0 or not member:
                            continue
                        physical, subpage = divmod(locator, page // fragment)
                        owner, logical_page = inverse[physical]
                        assert owner == request
                        origin = logical_page * page + subpage * fragment
                        assert origin <= pos[row], (
                            "future fragments must have no membership"
                        )
                        for lane in range(fragment):
                            if (
                                slot * fragment + lane < live
                                and origin + lane <= pos[row]
                            ):
                                actual.add(origin + lane)
                    assert actual == expected[row, head]
                    if not expected[row, head]:
                        assert locators[0] == -1 and live == 1

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(outputs)
    for base, inert in ((max(1, block - 3), False), (block + 1, True), (1024, False)):
        expected = update(base, inert)
        graph.replay()
        check()


@pytest.mark.parametrize("block,page", _BLOCK_PAGE_PAIRS)
@pytest.mark.parametrize("group", (1, 5))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("split_kv", (False, True))
def test_sparse_pattern_attention_graph(block, page, group, packed, shared, split_kv):
    _check_pattern_attention(block, page, group, packed, shared, split_kv)


def _check_pattern_attention(
    block,
    page,
    group,
    packed,
    shared,
    split_kv,
    *,
    dtype=torch.bfloat16,
    dim=128,
    ratio=12,
    context=8192,
    check_eager=False,
):
    torch.manual_seed(71845)
    hkv, topk = 2, 17
    long_base = max(1, min(4096, context - group + 1))
    lengths = [group, max(1, group - 1)] if packed else [group, group]
    offsets = [0, lengths[0], sum(lengths)]
    rows = offsets[-1]
    q = torch.randn(rows, hkv * ratio, dim, device="cuda").to(dtype)
    if not packed:
        q = q.view(2, 1, group, hkv * ratio, dim)
    out = torch.empty_like(
        q, dtype=torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    )
    width = (context + page - 1) // page
    k = torch.randn(2 * width, hkv, page, dim, device="cuda").to(dtype)
    v = torch.randn(2 * width, hkv, page, dim, device="cuda").to(dtype)
    table = torch.randperm(2 * width, device="cuda", dtype=torch.int32).view(2, width)
    pattern_heads = 1 if shared else hkv
    ids = torch.full((rows, pattern_heads, topk), -1, device="cuda", dtype=torch.int32)
    input_ids = ids[:, 0] if shared else ids
    requests = torch.tensor(
        [r for r, n in enumerate(lengths) for _ in range(n)],
        device="cuda",
        dtype=torch.int32,
    )
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)
    qo_indptr = (
        torch.tensor(offsets, device="cuda", dtype=torch.int32) if packed else None
    )
    workspace = torch.empty(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            k,
            table,
            block_topk=topk,
            max_seq_len_kv=context,
            qo_indptr=qo_indptr,
            seq_len_q=group if packed else None,
            kv_block_size=block,
            o_data_type=out.dtype,
            split_kv=split_kv,
            share_pattern_across_kv_heads=shared,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        2,
        group,
        hkv * ratio,
        hkv,
        dim,
        block,
        page,
        topk,
        context,
        device=q.device,
        workspace_buffer=workspace,
        use_packed_q=packed,
        split_kv=split_kv,
        share_pattern_across_kv_heads=shared,
        q_data_type=q.dtype,
        o_data_type=out.dtype,
    )

    def update(base):
        ids.fill_(-1)
        selected = {}
        for request, n in enumerate(lengths):
            for qi in range(n):
                row = offsets[request] + qi
                visible = base + qi
                positions[row] = visible - 1
                for ph in range(pattern_heads):
                    blocks = torch.randperm(
                        visible // block, device="cuda", dtype=torch.int32
                    )[:topk]
                    ids[row, ph, : blocks.numel()] = blocks
                    tokens = (
                        blocks[:, None].long() * block
                        + torch.arange(block, device="cuda")
                    ).flatten()
                    tail = torch.arange(
                        visible // block * block, visible, device="cuda"
                    )
                    selected[row, ph] = torch.cat((tokens, tail))
        return selected

    def run():
        wrapper.run(
            q,
            (k, v),
            table,
            input_ids,
            requests,
            positions,
            qo_indptr=qo_indptr,
            out=out,
        )

    def check(selected):
        q_flat = q.reshape(rows, hkv, ratio, dim).float()
        out_flat = out.reshape(rows, hkv, ratio, dim).float()
        if dtype == torch.float8_e4m3fn:
            from tests.attention.test_attention_ts_sparse_shapes import (
                _fp8_route_reference,
            )
            from flashinfer.attention.prims_ts._q_token_kv_block_sparse_policy import (
                select_sparse_launch,
            )

            launch = select_sparse_launch(
                group_size=group,
                heads_q_per_kv=ratio,
                head_dim=dim,
                q_dtype_key="float8_e4m3fn",
                num_routes=2,
                num_kv_heads=hkv,
                route_kv_tokens=min(
                    context,
                    topk * block + block - 1
                    if group == 1
                    else group * (topk + 1) * block,
                ),
                multi_processor_count=torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count,
                split_kv=split_kv,
            )
            for request, n in enumerate(lengths):
                begin = offsets[request]
                for head in range(hkv):
                    lists = [
                        selected[begin + qi, 0 if shared else head] for qi in range(n)
                    ]
                    if group == 1:
                        tokens = lists[0]
                    else:
                        logical_blocks = torch.unique(
                            torch.cat(lists) // block, sorted=True
                        )
                        tokens = (
                            logical_blocks[:, None] * block
                            + torch.arange(block, device="cuda")
                        ).flatten()
                        tokens = tokens[tokens <= positions[begin + n - 1]]
                    visible = torch.stack(
                        [torch.isin(tokens, chosen) for chosen in lists]
                    )
                    physical = table[request, tokens // page].long()
                    expected, uncertainty = _fp8_route_reference(
                        q_flat[begin : begin + n, head : head + 1],
                        k[physical, head, tokens % page][:, None],
                        v[physical, head, tokens % page][:, None],
                        visible,
                        instances=launch.num_insts_kv,
                        splits=launch.splits_kv,
                        out_dtype=out.dtype,
                    )
                    actual = out_flat[begin : begin + n, head : head + 1]
                    closest = torch.maximum(
                        torch.minimum(actual, expected + uncertainty),
                        expected - uncertainty,
                    )
                    torch.testing.assert_close(actual, closest, rtol=0.01, atol=0.002)
            return
        for request, n in enumerate(lengths):
            for qi in range(n):
                row = offsets[request] + qi
                for head in range(hkv):
                    tokens = selected[row, 0 if shared else head]
                    physical = table[request, tokens // page].long()
                    keys = k[physical, head, tokens % page].float()
                    values = v[physical, head, tokens % page].float()
                    expected = (q_flat[row, head] @ keys.T * dim**-0.5).softmax(
                        -1
                    ) @ values
                    torch.testing.assert_close(
                        out_flat[row, head], expected, rtol=0.03, atol=0.01
                    )

    selected = update(long_base)
    run()
    check(selected)
    if check_eager:
        from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
            q_token_kv_block_sparse_attention_with_paged_kv_cache,
        )

        q_token_kv_block_sparse_attention_with_paged_kv_cache(
            q,
            (k, v),
            table,
            input_ids,
            requests,
            positions,
            workspace,
            max_seq_len_kv=context,
            seq_len_q=group,
            kv_block_size=block,
            qo_indptr=qo_indptr,
            out=out,
            split_kv=split_kv,
            share_pattern_across_kv_heads=shared,
        )
        check(selected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for base in (
        max(1, min(block - 3, long_base)),
        min(block + 1, long_base),
        long_base,
    ):
        selected = update(base)
        out.fill_(torch.nan)
        graph.replay()
        check(selected)


@pytest.mark.parametrize("dim", (64, 128, 256))
@pytest.mark.parametrize("block", (4, 8, 16, 32, 64, 128))
def test_sparse_pattern_fp8_geometry(dim, block):
    group = 1 if block <= 16 else 8
    page = 16 if block == 32 else 128
    if block == 128 and dim != 256:
        page = 20 if dim == 64 else 16
    _check_pattern_attention(
        block,
        page,
        group,
        packed=dim != 128,
        shared=False,
        split_kv=group == 1,
        dtype=torch.float8_e4m3fn,
        dim=dim,
    )


@pytest.mark.parametrize(
    "block,page,group,dim,ratio,dtype",
    (
        (16, 128, 3, 64, 4, torch.float16),
        (128, 128, 8, 64, 3, torch.float8_e4m3fn),
        (64, 128, 3, 256, 17, torch.bfloat16),
        (128, 20, 8, 256, 12, torch.float16),
        (8, 16, 7, 128, 17, torch.bfloat16),
    ),
)
@pytest.mark.parametrize("packed,split_kv", ((False, False), (True, True)))
def test_sparse_pattern_other_query_geometry(
    block, page, group, dim, ratio, dtype, packed, split_kv
):
    _check_pattern_attention(
        block,
        page,
        group,
        packed,
        False,
        split_kv,
        dim=dim,
        ratio=ratio,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "block,page,group,context",
    ((4, 16, 1, 4), (32, 128, 5, 256), (128, 16, 1, 256), (128, 20, 8, 256)),
)
def test_sparse_pattern_model_bounded_workspace(block, page, group, context):
    _check_pattern_attention(
        block,
        page,
        group,
        True,
        False,
        True,
        context=context,
    )


@pytest.mark.parametrize(
    "storage,fmt", (("contiguous", "bsr"), ("contiguous", "bitmask"), ("paged", "bsr"))
)
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("block", (8, 64))
@pytest.mark.parametrize("mask", ("dense", "causal"))
def test_block_sparse_pattern_heads_graph(storage, fmt, shared, block, mask):
    from flashinfer.attention.prims_ts.block_sparse import (
        BlockSparseTSWrapper,
        BlockSparsePagedTSWrapper,
        block_sparse_attention,
        block_sparse_attention_with_paged_kv_cache,
    )
    from tests.attention.test_attention_ts_block_sparse import (
        _Case,
        _make_bsr,
        _make_exact_block_bits,
        _reference,
    )

    torch.manual_seed(45132)
    batch, sq, sk, hq, hkv, dim, qb, page = 2, 32, 256, 12, 3, 128, 16, 64
    q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, sk, hkv, dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    pattern_heads = 1 if shared else hkv
    case = _Case(
        "pattern-heads",
        batch,
        hq,
        sq,
        sk,
        qb,
        block,
        q.dtype,
        mask,
        "none",
        "static",
        num_kv_heads=hkv,
    )
    valid = tuple(frozenset(range(sk)) for _ in range(batch))

    def patterns(shift):
        raw = tuple(
            tuple(
                tuple(
                    (0, 1 + (b + h + row + shift) % (sk // block - 1))
                    for row in range(sq // qb)
                )
                for h in range(pattern_heads)
            )
            for b in range(batch)
        )
        full = (
            tuple(tuple(rows[0] for _ in range(hkv)) for rows in raw) if shared else raw
        )
        return raw, full

    raw, full = patterns(0)
    indptr, indices = _make_bsr(raw)
    bits = _make_exact_block_bits(raw, sk // block)
    static = dict(
        device=q.device,
        max_blocks_per_row=2,
        use_kv_valid_bits=False,
        mask_type=mask,
        q_data_type=q.dtype,
        share_pattern_across_kv_heads=shared,
    )
    if storage == "paged":
        pages = torch.randperm(batch * sk // page, device="cuda", dtype=torch.int32)
        cache_k = torch.empty(
            batch * sk // page, hkv, page, dim, device="cuda", dtype=q.dtype
        )
        cache_v = torch.empty_like(cache_k)
        for request in range(batch):
            for logical in range(sk // page):
                physical = pages[request * (sk // page) + logical]
                cache_k[physical] = k[
                    request, logical * page : (logical + 1) * page
                ].transpose(0, 1)
                cache_v[physical] = v[
                    request, logical * page : (logical + 1) * page
                ].transpose(0, 1)
        page_indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * (
            sk // page
        )
        lens = torch.full((batch,), sk, device="cuda", dtype=torch.int32)
        w = BlockSparsePagedTSWrapper()
        w.plan(batch, sq, sk, hq, hkv, dim, qb, block, page, **static)

        def run():
            return w.run(
                q,
                (cache_k, cache_v),
                page_indptr,
                pages,
                lens,
                indptr,
                indices,
                out=out,
            )

        eager = block_sparse_attention_with_paged_kv_cache(
            q,
            (cache_k, cache_v),
            page_indptr,
            pages,
            indptr,
            indices,
            qb,
            block,
            max_seq_len_kv=sk,
            seq_lens_kv=lens,
            mask_type=mask,
            share_pattern_across_kv_heads=shared,
        )
    else:
        w = BlockSparseTSWrapper()
        w.plan(batch, sq, sk, hq, hkv, dim, qb, block, sparse_format=fmt, **static)
        route_args = dict(
            block_indptr=indptr if fmt == "bsr" else None,
            block_indices=indices if fmt == "bsr" else None,
            exact_block_bits=bits if fmt == "bitmask" else None,
        )

        def run():
            return w.run(q, k, v, **route_args, out=out)

        eager = block_sparse_attention(
            q,
            k,
            v,
            q_block_size=qb,
            kv_block_size=block,
            sparse_format=fmt,
            mask_type=mask,
            share_pattern_across_kv_heads=shared,
            **route_args,
        )
    expected = _reference(case, q, k, v, full, valid, dim**-0.5)
    torch.testing.assert_close(eager, expected, rtol=0.03, atol=0.01)
    run()
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    raw, full = patterns(1)
    new_indptr, new_indices = _make_bsr(raw)
    indptr.copy_(new_indptr)
    indices.copy_(new_indices)
    bits.copy_(_make_exact_block_bits(raw, sk // block))
    graph.replay()
    expected = _reference(case, q, k, v, full, valid, dim**-0.5)
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.01)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_dense_encoded_eight_token_fragments(dtype):
    """The common paged loader also accepts eight-token encoded fragments."""
    from flashinfer.attention.prims_ts.decode import BatchDecodePagedTSWrapper

    torch.manual_seed(4173)
    batch, sq, hkv, ratio, dim, storage, fragment, context = (
        2,
        4,
        2,
        8,
        128,
        32,
        8,
        1024,
    )
    q = torch.randn(batch, sq, hkv * ratio, dim, device="cuda", dtype=dtype)
    k = torch.randn(
        batch * context // storage, hkv, storage, dim, device="cuda", dtype=dtype
    )
    v = torch.randn_like(k)
    physical = torch.randperm(k.shape[0], device="cuda", dtype=torch.int32).reshape(
        batch, -1
    )
    per_page = storage // fragment
    table = (
        physical[:, :, None] * per_page
        + torch.arange(per_page, device="cuda", dtype=torch.int32)
    ).reshape(batch, -1)
    lengths = torch.tensor((1024, 777), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        batch,
        hkv * ratio,
        hkv,
        dim,
        fragment,
        context,
        storage_page_size=storage,
        max_seq_len_q=sq,
        mask_type="causal",
        q_data_type=dtype,
        split_kv=False,
    )

    def run():
        wrapper.run(q, (k, v), lengths, table, out=out, validate=False)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(torch.nan)
    graph.replay()
    for b in range(batch):
        for qi in range(sq):
            tokens = torch.arange(int(lengths[b]) - sq + qi + 1, device="cuda")
            pages = physical[b, tokens // storage].long()
            for head in range(hkv):
                keys = k[pages, head, tokens % storage].float()
                values = v[pages, head, tokens % storage].float()
                qs = q[b, qi, head * ratio : (head + 1) * ratio].float()
                expected = (qs @ keys.T * dim**-0.5).softmax(-1) @ values
                torch.testing.assert_close(
                    out[b, qi, head * ratio : (head + 1) * ratio].float(),
                    expected,
                    rtol=0.03,
                    atol=0.01,
                )


@pytest.mark.parametrize("packed", (False, True))
def test_sparse_pattern_eager_interface(packed):
    _check_pattern_attention(128, 16, 3, packed, False, True, check_eager=True)
