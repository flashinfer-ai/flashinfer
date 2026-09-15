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

"""Sparse grouped-head geometry, causal routing and graph replay coverage."""

import math

import pytest
import torch

from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    _prepare_q_token_kv_block_sparse_attention,
    get_q_token_kv_block_sparse_workspace_size,
)

pytestmark = [
    pytest.mark.arch_blackwell,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _fp8_route_reference(
    queries, keys, values, visible, *, instances, splits, out_dtype
):
    """Independent dense-style FP8-P online softmax over a sparse route.

    All logical rows/heads are vectorized. Streams correspond to interleaved
    KV128 MMA instances within each active split. Membership comes from the
    input selections, never from kernel-produced metadata.
    """
    scores = torch.einsum("ghrd,thd->ghrt", queries.float(), keys.float())
    scale = queries.shape[-1] ** -0.5
    scale_log2 = scores.new_tensor(scale) * scores.new_tensor(math.log2(math.e))
    quant_log2 = scores.new_tensor(math.log2(448))
    scores.masked_fill_(~visible[:, None, None, :], -torch.inf)
    tiles = (keys.shape[0] + 127) // 128
    local_tiles = max(
        instances, (tiles + splits * instances - 1) // (splits * instances) * instances
    )

    def merge(streams):
        anchor = torch.stack([stream[0] for stream in streams]).max(0).values
        final_sum = torch.zeros_like(anchor)
        final_acc = torch.zeros_like(queries, dtype=torch.float32)
        final_error = torch.zeros_like(final_acc)
        for maximum, denominator, accumulator, error in streams:
            correction = torch.where(
                torch.isfinite(maximum), ((maximum - anchor) * scale).exp(), 0
            )
            final_sum += denominator * correction
            final_acc += accumulator * correction[..., None]
            final_error += error * correction[..., None]
        return anchor, final_sum, final_acc, final_error

    partials = []
    for split_begin in range(0, tiles, local_tiles):
        streams = []
        for instance in range(instances):
            maximum = torch.full_like(scores[..., 0], -torch.inf)
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros_like(queries, dtype=torch.float32)
            error = torch.zeros_like(accumulator)
            for tile in range(
                split_begin + instance, min(split_begin + local_tiles, tiles), instances
            ):
                begin, end = tile * 128, min((tile + 1) * 128, keys.shape[0])
                tile_scores = scores[..., begin:end]
                new_max = torch.maximum(maximum, tile_scores.max(-1).values)
                correction = torch.where(
                    torch.isfinite(maximum), ((maximum - new_max) * scale).exp(), 0
                )
                # Match the dense base-2 exponent convention, including its
                # FP32 max offset and fused score*scale+offset. Computing exp
                # then multiplying by 448 can choose another E4M3 bin near a
                # rounding boundary even when the exponent differs by one ulp.
                offset = -new_max * scale_log2 + quant_log2
                exponent = (
                    tile_scores.double() * scale_log2.double()
                    + offset[..., None].double()
                ).float()
                p = torch.where(torch.isfinite(tile_scores), exponent.exp2(), 0)
                quantized = p.to(torch.float8_e4m3fn).float()
                # Fast exp2 and host-reference FP32 arithmetic may straddle
                # an E4M3 midpoint. Propagate only that bin-edge uncertainty,
                # instead of relaxing every output's absolute tolerance.
                delta = p * (8 * torch.finfo(torch.float32).eps)
                lower = (p - delta).to(torch.float8_e4m3fn).float()
                upper = (p + delta).to(torch.float8_e4m3fn).float()
                uncertainty = torch.maximum(quantized - lower, upper - quantized)
                accumulator = accumulator * correction[..., None] + torch.einsum(
                    "ghrt,thd->ghrd", quantized, values[begin:end].float()
                )
                error = error * correction[..., None] + torch.einsum(
                    "ghrt,thd->ghrd", uncertainty, values[begin:end].float().abs()
                )
                denominator = denominator * correction + p.sum(-1)
                maximum = new_max
            streams.append((maximum, denominator, accumulator, error))
        anchor, denominator, accumulator, error = merge(streams)
        normalized = torch.where(
            denominator[..., None] > 0, accumulator / denominator[..., None], 0
        )
        normalized_error = torch.where(
            denominator[..., None] > 0, error / denominator[..., None], 0
        )
        if splits > 1:
            # Separate reduction publishes normalized 16-bit O plus FP32 LSE.
            # This rounding happens before the split outputs are combined.
            normalized = normalized.to(out_dtype).float()
        partials.append(
            (anchor * scale + denominator.log(), normalized, normalized_error)
        )
    weights = torch.stack([part[0] for part in partials]).softmax(0)[..., None]
    return (
        (weights * torch.stack([part[1] for part in partials])).sum(0),
        (weights * torch.stack([part[2] for part in partials])).sum(0),
    )


def _check_shape(
    dtype,
    dim,
    group,
    ratio,
    packed,
    *,
    out_dtype=None,
    topk=65,
    context=2048,
    heads_kv=2,
    split_kv=None,
    public_wrapper=False,
):
    # This fixture uses packed Q for prefill unless a test explicitly selects
    # packed decode. The production policy does not infer phase from layout.
    if split_kv is None:
        split_kv = not packed
    torch.manual_seed(28193)
    requests_count, groups_per_request = 2, 2
    page = 16
    rows = requests_count * groups_per_request * group - int(packed and group > 1)
    q = torch.randn(rows, ratio * heads_kv, dim, device="cuda").to(dtype)
    qo_indptr = (
        torch.tensor(
            [0, group, 2 * group, 3 * group, rows], device="cuda", dtype=torch.int32
        )
        if packed
        else None
    )
    if not packed:
        q = q.reshape(requests_count, groups_per_request, group, ratio * heads_kv, dim)
    pages = requests_count * context // page
    k = torch.randn(pages, heads_kv, page, dim, device="cuda").to(dtype)
    v = torch.randn(pages, heads_kv, page, dim, device="cuda").to(dtype)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(
        requests_count, -1
    )
    block_indices = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) // (
        groups_per_request * group
    )
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)
    if out_dtype is None:
        out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    out = torch.empty_like(q, dtype=out_dtype)
    workspace = torch.empty(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            k,
            table,
            block_topk=topk,
            max_seq_len_kv=context,
            o_data_type=out.dtype,
            qo_indptr=qo_indptr,
            seq_len_q=group if packed else None,
            split_kv=split_kv,
        ),
        dtype=torch.uint8,
        device="cuda",
    )

    def update(base):
        selections = []
        block_indices.fill_(-1)
        for row in range(rows):
            visible = base + row % (groups_per_request * group)
            positions[row] = visible - 1
            count = min(visible // 4, topk)
            blocks = torch.randperm(visible // 4, device="cuda", dtype=torch.int32)[
                :count
            ]
            block_indices[row, :count].copy_(blocks)
            tokens = (
                blocks.long()[:, None] * 4 + torch.arange(4, device="cuda")
            ).flatten()
            tail = torch.arange(visible // 4 * 4, visible, device="cuda")
            selections.append(torch.cat((tokens, tail)))
        return selections

    selected = update(context // 2)
    if public_wrapper:
        plan = QTokenKvBlockSparsePagedTSWrapper()
        plan.plan(
            4,
            group,
            ratio * heads_kv,
            heads_kv,
            dim,
            4,
            page,
            topk,
            context,
            device=q.device,
            workspace_buffer=workspace,
            use_packed_q=packed,
            split_kv=split_kv,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=out.dtype,
        )
    else:
        plan = _prepare_q_token_kv_block_sparse_attention(
            q,
            (k, v),
            block_indices,
            table,
            requests,
            positions,
            workspace,
            max_seq_len_kv=context,
            qo_indptr=qo_indptr,
            max_seq_len_q=group if packed else None,
            out=out,
            split_kv=split_kv,
        )
    from flashinfer.attention.prims_ts._q_token_kv_block_sparse_policy import (
        select_sparse_launch,
    )

    launch = select_sparse_launch(
        group_size=group,
        heads_q_per_kv=ratio,
        head_dim=dim,
        q_dtype_key=str(dtype).removeprefix("torch."),
        num_routes=4,
        num_kv_heads=heads_kv,
        route_kv_tokens=min(
            context, topk * 4 + 3 if group == 1 else group * (topk + 1) * 4
        ),
        multi_processor_count=torch.cuda.get_device_properties(
            q.device
        ).multi_processor_count,
        split_kv=split_kv,
    )

    def run():
        if public_wrapper:
            plan.run(
                q,
                (k, v),
                table,
                block_indices,
                requests,
                positions,
                qo_indptr=qo_indptr,
                out=out,
            )
        else:
            plan.run(q, block_indices, table, requests, positions, out=out)

    def check():
        if dtype == torch.float8_e4m3fn:
            for first in range(0, rows, group):
                live_rows = min(group, rows - first)
                selections = selected[first : first + live_rows]
                # G1 preserves indexer order; grouped metadata sorts the page
                # union and pads each causal tail page to four physical tokens.
                tokens = (
                    selections[0]
                    if group == 1
                    else (
                        torch.cat(selections)
                        .div(4, rounding_mode="floor")
                        .unique(sorted=True)[:, None]
                        * 4
                        + torch.arange(4, device="cuda")
                    ).flatten()
                )
                req = first // (groups_per_request * group)
                physical = table[req, tokens // page].long()
                keys = k[physical, :, tokens % page].float()
                values = v[physical, :, tokens % page].float()
                visible = torch.stack(
                    [torch.isin(tokens, selection) for selection in selections]
                )
                rounded, uncertainty = _fp8_route_reference(
                    q.reshape(rows, heads_kv, ratio, dim)[first : first + live_rows],
                    keys,
                    values,
                    visible,
                    instances=launch.num_insts_kv,
                    splits=launch.splits_kv,
                    out_dtype=out.dtype,
                )
                actual = out.reshape(rows, heads_kv, ratio, dim)[
                    first : first + live_rows
                ].float()
                expected = rounded.to(out.dtype).float()
                closest = actual.clamp(
                    min=expected - uncertainty, max=expected + uncertainty
                )
                torch.testing.assert_close(actual, closest, rtol=0.01, atol=0.002)
            # As in dense decode's FP8 oracle, correctness includes P's E4M3
            # rounding; FP32 softmax alone is not that kernel's arithmetic.
            return
        for row, tokens in enumerate(selected):
            req = row // (groups_per_request * group)
            physical = table[req, tokens // page].long()
            for head_kv in range(heads_kv):
                keys = k[physical, head_kv, tokens % page].float()
                vals = v[physical, head_kv, tokens % page].float()
                queries = q.reshape(rows, heads_kv, ratio, dim)[row, head_kv].float()
                logits = queries @ keys.T * dim**-0.5
                expected = logits.softmax(-1) @ vals
                actual = out.reshape(rows, heads_kv, ratio, dim)[row, head_kv].float()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.03,
                    atol=0.01,
                )

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Change the valid prefix and every query's membership without replanning.
    # The short replay crosses all causal tail sizes; packed Q has a short group.
    for base in (3, context // 2 + 1):
        selected = update(base)
        out.fill_(torch.nan)
        graph.replay()
        check()
    torch.cuda.synchronize()
    return plan._prepared_plan if public_wrapper else plan


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float8_e4m3fn),
    ids=("f16", "bf16", "fp8"),
)
@pytest.mark.parametrize("dim", (64, 128, 256), ids=("d64", "d128", "d256"))
@pytest.mark.parametrize(
    "group,ratio",
    ((1, 3), (3, 4), (5, 6), (5, 12), (8, 12)),
    ids=("rows3", "rows12", "rows30", "rows60", "rows96"),
)
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_sparse_head_dimensions_and_query_tiles(dtype, dim, group, ratio, packed):
    """Exercise TileQ8/16/32/64/128, except the D256 FP8 promoted recipes."""
    _check_shape(dtype, dim, group, ratio, packed)


@pytest.mark.parametrize("dim", (64, 128, 256), ids=("d64", "d128", "d256"))
@pytest.mark.parametrize("group,ratio", ((8, 1), (7, 17), (4, 31), (2, 64), (1, 128)))
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float8_e4m3fn),
    ids=("f16", "bf16", "fp8"),
)
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_sparse_non_power_of_two_and_wide_head_ratios(dim, group, ratio, dtype, packed):
    _check_shape(dtype, dim, group, ratio, packed)


@pytest.mark.parametrize("dim", (64, 128, 256), ids=("d64", "d128", "d256"))
@pytest.mark.parametrize("group,ratio", ((3, 4), (8, 12)))
def test_sparse_fp8_to_fp16(dim, group, ratio):
    _check_shape(
        torch.float8_e4m3fn, dim, group, ratio, packed=False, out_dtype=torch.float16
    )


@pytest.mark.parametrize("dim", (64, 128), ids=("d64", "d128"))
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float8_e4m3fn), ids=("bf16", "fp8")
)
@pytest.mark.parametrize("group", (4, 8))
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_sparse_two_instance_keeps_holds_one_locator_window(dim, dtype, group, packed):
    # 17 blocks/query keeps the complete grouped route within eight KV128
    # tiles. K0/K1/V0/V1 must share one read stage, including changed graph
    # prefixes and per-query membership bytes, without republishing that span.
    _check_shape(dtype, dim, group, 12, packed, topk=17)


@pytest.mark.parametrize("dim", (64, 128, 256), ids=("d64", "d128", "d256"))
@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float8_e4m3fn),
    ids=("f16", "bf16", "fp8"),
)
@pytest.mark.parametrize("group", (5, 8))
def test_sparse_keeps_split_reduction_and_short_graph_replay(dim, dtype, group):
    # Larger candidate lists force real split reduction. Graph replay changes
    # the active split prefix from long, independently selected lists to only
    # a few visible tokens and back, without touching the counters on the host.
    plan = _check_shape(dtype, dim, group, 12, packed=False, topk=512, context=8192)
    assert plan._attention_plan._compiled_reducer is not None


@pytest.mark.parametrize("group,ratio", ((1, 3), (3, 17), (8, 3)))
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_sparse_fp8_d64_odd_single_kv_head_stride(group, ratio, packed):
    # A single KV head and an odd GQA ratio give FP8 D64 queries a token
    # stride that is 64-byte, but not 128-byte, aligned.
    _check_shape(torch.float8_e4m3fn, 64, group, ratio, packed, heads_kv=1)


@pytest.mark.parametrize("split_kv", (False, True), ids=("no-split", "split"))
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
@pytest.mark.parametrize(
    "dim,dtype,group,ratio",
    (
        (64, torch.bfloat16, 1, 3),
        (128, torch.float16, 2, 12),
        (256, torch.bfloat16, 2, 12),
        (64, torch.float8_e4m3fn, 8, 12),
        (128, torch.bfloat16, 5, 12),
        (256, torch.float8_e4m3fn, 8, 12),
    ),
)
def test_sparse_split_control_is_independent_of_query_layout(
    dim, dtype, group, ratio, packed, split_kv
):
    plan = _check_shape(
        dtype,
        dim,
        group,
        ratio,
        packed,
        topk=512,
        context=8192,
        split_kv=split_kv,
        public_wrapper=True,
    )
    # This small route grid and large candidate count have useful split work
    # in either layout. False must also disable reduction for fixed Q.
    assert (plan._attention_plan._compiled_reducer is not None) is split_kv


def test_sparse_split_control_workspace_contract():
    """Sizing and planning agree on the caller's split permission."""
    q = torch.empty(1, 1, 4, 12, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(512, 1, 16, 128, device="cuda", dtype=q.dtype)
    table = torch.empty(1, 512, device="cuda", dtype=torch.int32)
    sizes = {
        allow: get_q_token_kv_block_sparse_workspace_size(
            q, k, table, block_topk=512, max_seq_len_kv=8192, split_kv=allow
        )
        for allow in (False, True)
    }
    assert sizes[False] < sizes[True]
    workspace = torch.empty(sizes[False], device="cuda", dtype=torch.uint8)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    params = dict(
        device=q.device,
        workspace_buffer=workspace,
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
    )
    wrapper.plan(1, 4, 12, 1, 128, 4, 16, 512, 8192, split_kv=False, **params)
    with pytest.raises(ValueError, match="workspace_buffer"):
        wrapper.plan(1, 4, 12, 1, 128, 4, 16, 512, 8192, split_kv=True, **params)
    with pytest.raises(TypeError, match="split_kv must be a bool"):
        get_q_token_kv_block_sparse_workspace_size(
            q, k, table, block_topk=512, max_seq_len_kv=8192, split_kv=1
        )
