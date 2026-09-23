"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the experimental Cake NVFP4 paged-KV MSA decode on SM100/SM103.

Times ``flashinfer.msa_ops.prepare_msa_nvfp4_sparse_decode`` (one persistent
generated kernel) against ``msa_sparse_decode_attention`` on the same planar
NVFP4 pages, top-k 16 selections and queries, with CUPTI and a cold L2 between
iterations.  The rows are the MiniMax-M3 decode matrix used to qualify the
program: tensor-parallel ranks 1/2/4/8 (64/32/16/8 query heads over 4/2/1/1 KV
heads), KV lengths from a 257-token tail to one million tokens, one to eight
query tokens per request.

Usage::

    python benchmarks/bench_cake_msa_nvfp4_decode.py [--rows tp1_b128_kv8k_q1 ...] [--json out.json]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.msa_nvfp4_decode.cake_backend import (  # noqa: E402
    msa_nvfp4_decode_workspace_size,
)
from flashinfer.msa_ops import (  # noqa: E402
    msa_sparse_decode_attention,
    prepare_msa_nvfp4_sparse_decode,
)
from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: E402
from tests.test_helpers.cake_msa_nvfp4_inputs import (  # noqa: E402
    build_decode_inputs,
    upstream_route_kwargs,
)

# name, batch, KV tokens per request, KV heads, query heads per KV head, seqlen_q, ragged
ROWS = [
    ("tp1_b128_kv8k_q1", 128, 8192, 4, 16, 1, False),
    ("tp1_b32_kv64k_q1", 32, 65536, 4, 16, 1, False),
    ("tp1_b32_kv64k_q1_ragged", 32, 65536, 4, 16, 1, True),
    ("tp1_b8_kv100k_q1", 8, 100_000, 4, 16, 1, False),
    ("tp1_b1_kv200k_q1", 1, 200_000, 4, 16, 1, False),
    ("tp2_b64_kv16k_q1", 64, 16384, 2, 16, 1, False),
    ("tp4_b128_kv8k_q1", 128, 8192, 1, 16, 1, False),
    ("tp4_b1_kv1m_q1", 1, 1_048_576, 1, 16, 1, False),
    ("tp8_b128_kv64k_q1", 128, 65536, 1, 8, 1, False),
    ("tp1_b2_kv257_q1_tail", 2, 257, 4, 16, 1, False),
    ("tp1_b32_kv8k_q2", 32, 8192, 4, 16, 2, False),
    ("tp1_b32_kv8k_q4", 32, 8192, 4, 16, 4, False),
    ("tp1_b32_kv8k_q8", 32, 8192, 4, 16, 8, False),
    ("tp4_b32_kv64k_q8", 32, 65536, 1, 16, 8, False),
    ("tp1_b8_kv100k_q8", 8, 100_000, 4, 16, 8, False),
]


def ragged_lengths(batch, nominal, seqlen_q, seed):
    """Symmetric +-(5..25)% spread around ``nominal`` (pairs), shuffled."""
    g = torch.Generator().manual_seed(seed + batch * 1009 + nominal)
    pairs = batch // 2
    lo, hi = max(1, nominal // 20), max(1, nominal // 4)
    deltas = torch.randint(lo, hi + 1, (pairs,), generator=g, dtype=torch.int64)
    parts = [nominal - deltas, nominal + deltas]
    if batch % 2:
        parts.append(torch.tensor([nominal], dtype=torch.int64))
    values = torch.cat(parts)[torch.randperm(batch, generator=g)]
    return values.clamp(min=seqlen_q).tolist()


def median_us(fn):
    times = bench_gpu_time_with_cupti(
        fn, dry_run_iters=10, repeat_iters=50, cold_l2_cache=True
    )
    return statistics.median(times) * 1e3


def bench_row(name, batch, kv, num_kv_heads, group, seqlen_q, ragged, *, device, seed):
    seq_lens = (
        ragged_lengths(batch, kv, seqlen_q, seed)
        if ragged and batch > 1
        else [kv] * batch
    )
    inputs = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group,
        seqlen_q=seqlen_q,
        device=device,
        seed=seed,
    )
    workspace = torch.empty(
        msa_nvfp4_decode_workspace_size(batch, num_kv_heads, device, seqlen_q=seqlen_q),
        dtype=torch.uint8,
        device=device,
    )
    out_cake = torch.empty_like(inputs["q"])
    runner = prepare_msa_nvfp4_sparse_decode(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        workspace_buffer=workspace,
        seqlen_q=seqlen_q,
        softmax_scale=inputs["softmax_scale"],
        out=out_cake,
    )
    out_route = torch.empty_like(inputs["q"])
    route_kwargs = upstream_route_kwargs(inputs)

    def route():
        msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            out=out_route,
            **route_kwargs,
        )

    runner()
    route()
    torch.cuda.synchronize()
    max_abs = float((out_cake.float() - out_route.float()).abs().max())
    cake_us = median_us(runner)
    route_us = median_us(route)
    items = batch * seqlen_q * num_kv_heads
    return dict(
        row=name,
        batch=batch,
        kv=kv,
        num_kv_heads=num_kv_heads,
        group=group,
        seqlen_q=seqlen_q,
        ragged=ragged,
        items=items,
        splits=runner.splits,
        cake_us=cake_us,
        route_us=route_us,
        max_abs_diff_vs_route=max_abs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = ROWS if not args.rows else [r for r in ROWS if r[0] in set(args.rows)]
    results = []
    print(
        f"{'row':28s} {'items':>6s} {'splits':>6s} {'cake_us':>9s} {'route_us':>9s} {'route/cake':>10s}"
    )
    for row in rows:
        record = bench_row(*row, device=device, seed=args.seed)
        print(
            f"{record['row']:28s} {record['items']:6d} {record['splits']:6d} "
            f"{record['cake_us']:9.2f} {record['route_us']:9.2f} "
            f"{record['route_us'] / record['cake_us']:10.3f}"
        )
        results.append(record)
        torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(device=torch.cuda.get_device_name(device), rows=results), indent=1
            )
        )


if __name__ == "__main__":
    main()
