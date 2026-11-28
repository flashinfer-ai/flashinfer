# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Reference-checked FA2 MIS benchmark; run on each checkout with identical flags.

Example: python benchmarks/bench_multi_item_scoring.py --output result.json
"""

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

import flashinfer


def make_inputs(batch, prefix, items, item_len, group, dtype):
    """Build paged MIS inputs and dense output/LSE references for one shape."""
    torch.manual_seed(20261020)
    page, heads, dim = 16, 4, 128
    qlen = items * (item_len + 1)
    klen = prefix + qlen
    pages = math.ceil(klen / page)
    q = torch.randn(batch * qlen, heads * group, dim, device="cuda", dtype=dtype) / 5
    kv = torch.randn(batch * pages, 2, page, heads, dim, device="cuda", dtype=dtype) / 5
    indices = torch.arange(batch * pages, device="cuda", dtype=torch.int32)
    metadata = dict(
        qo_indptr=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qlen,
        paged_kv_indptr=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        * pages,
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.full(
            (batch,), (klen - 1) % page + 1, device="cuda", dtype=torch.int32
        ),
        num_qo_heads=heads * group,
        num_kv_heads=heads,
        head_dim_qk=dim,
        page_size=page,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        prefix_len_ptr=torch.full((batch,), prefix, device="cuda", dtype=torch.uint32),
        token_pos_in_items_ptr=torch.arange(qlen, device="cuda", dtype=torch.int32)
        .remainder(item_len + 1)
        .to(torch.uint16)
        .repeat(batch, 1),
        token_pos_in_items_len=qlen,
        max_item_len_ptr=torch.full(
            (batch,), item_len, device="cuda", dtype=torch.uint16
        ),
    )
    outputs, lses = [], []
    keys = torch.arange(klen, device="cuda")
    key_items = (keys - prefix) // (item_len + 1)
    key_content = (keys - prefix).remainder(item_len + 1) > 0
    for b in range(batch):
        cache = kv[b * pages : (b + 1) * pages]
        k = (
            cache[:, 0]
            .reshape(-1, heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        v = (
            cache[:, 1]
            .reshape(-1, heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        for start in range(0, qlen, 128):
            end = min(start + 128, qlen)
            qi = torch.arange(start, end, device="cuda")
            mask = keys[None, :] <= prefix + qi[:, None]
            same_item = qi[:, None] // (item_len + 1) == key_items[None, :]
            content = (qi[:, None].remainder(item_len + 1) > 0) & key_content[None, :]
            mask &= (keys[None, :] < prefix) | (same_item & content)
            scores = torch.einsum(
                "qhd,khd->hqk", q[b * qlen + start : b * qlen + end].float(), k
            ) / math.sqrt(dim)
            scores.masked_fill_(~mask.unsqueeze(0), -torch.inf)
            outputs.append(
                torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(0), v).to(
                    dtype
                )
            )
            lses.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
    return q, kv, metadata, torch.cat(outputs), torch.cat(lses)


def plan_options(policy):
    """Translate the benchmark split policy into wrapper planning arguments."""
    if policy == "no_split":
        return {"disable_split_kv": True}
    if policy == "fixed_8_pages":
        return {"fixed_split_size": 8}
    return {}


def measure(function):
    """Return per-call CUDA-event times from warm graphs with 20 calls each."""
    for _ in range(5):
        function()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            function()
    for _ in range(40):
        graph.replay()
    samples = []
    for _ in range(25):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return samples


def main():
    """Validate MIS output and LSE, then save graph timing samples as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in [
        ("prefix", 8192),
        ("items", 256),
        ("item-length", 15),
        ("batch-size", 1),
        ("group", 1),
    ]:
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--policy", choices=["default", "no_split", "fixed_8_pages"], default="default"
    )
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Choose a fresh output path"
    assert (
        args.prefix >= 0
        and min(args.items, args.item_length, args.batch_size, args.group) > 0
    )
    q, kv, metadata, reference, ref_lse = make_inputs(
        args.batch_size,
        args.prefix,
        args.items,
        args.item_length,
        args.group,
        getattr(torch, args.dtype),
    )
    # Freeze independent truth before JIT loading or native execution. Keep
    # validation outside timing, including the final graph replay's outputs.
    reference, ref_lse = reference.cpu(), ref_lse.cpu()
    workspace = torch.empty(
        args.workspace_mib * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper.plan(**metadata, **plan_options(args.policy))
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
    samples = measure(lambda: wrapper.run(q, kv, out=out, lse=lse, return_lse=True))
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
    result = dict(
        config={k: v for k, v in vars(args).items() if k != "output"},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        revision=flashinfer.__git_commit__,
        plan_info=list(wrapper._plan_info),
        median_us=statistics.median(samples),
        samples_us=samples,
        independent_reference_passed=True,
        reference_snapshot="CPU copy before JIT loading and native execution",
        post_timing_reference_passed=True,
        timing="Preallocated output/LSE; warm-L2 CUDA graphs: 20 calls, 40 warm rounds, 25 samples; attention+merge, excludes planning and allocation",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
