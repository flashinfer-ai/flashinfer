#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Time the SDSA (Simplified DSA) indexer for each key format.

On SM10x it times, with BF16, FP8 and NVFP4 keys and queries:

* ``indexer_scores`` decode: ``--batch`` requests of 1 and 4 query rows, each
  scoring every key of its context;
* ``indexer_threshold_select``: ``--select-rows`` causal query rows of one
  request at the end of its context, with ``--top-k`` tie slots and the repair.

Off SM10x (Hopper), only decode scoring runs, so it times BF16 and FP8 decode
(the Triton kernel). FP8 calls pass a ``score_scale`` of 1, as a model does.

Keys and queries are random. The select's thresholds are each row's
``top_k * (1 + --slack)``-th largest score estimated from a key sample, so a
few rows miss and get repaired, as with sampled thresholds. Each timed select
call first restores the thresholds and zeroes the counters (two small copies).

Times are GPU kernel times from ``bench_gpu_time`` (CUPTI, or CUDA events when
CUPTI is missing; ``--no-cupti`` forces events), with a cold L2, after a
warm-up call that compiles the kernels. The bandwidth column counts the key
bytes (and NVFP4 scales) read once per request.

Usage::

    python benchmarks/bench_sdsa_indexer.py
    python benchmarks/bench_sdsa_indexer.py --contexts 32768 131072 --batch 16
"""

import argparse

import numpy as np
import torch

from flashinfer import nvfp4_kv_dequantize
from flashinfer.sdsa import indexer_scores, indexer_threshold_select
from flashinfer.testing.utils import bench_gpu_time

PAGE, DIM = 64, 128
# Bytes per key as stored: BF16, FP8 E4M3, NVFP4 (64 packed bytes + 8 scales).
KEY_BYTES = {"bf16": 2 * DIM, "fp8": DIM, "nvfp4": DIM // 2 + DIM // 16}


def make_rows(fmt: str, rows: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random queries or keys of a format, with NVFP4's scales."""
    if fmt == "nvfp4":
        values = torch.randint(256, (rows, DIM // 2), device="cuda").byte()
        scales = 2.0 ** torch.randint(-3, 2, (rows, DIM // 16), device="cuda")
        return values, scales.to(torch.float8_e4m3fn)
    values = torch.randn(rows, DIM, device="cuda")
    return values.to(torch.bfloat16 if fmt == "bf16" else torch.float8_e4m3fn), None


def as_float(values: torch.Tensor, scales: torch.Tensor | None) -> torch.Tensor:
    if scales is None:
        return values.float()
    one = torch.ones(1, device="cuda")
    return nvfp4_kv_dequantize(values, scales.view(torch.uint8), one).float()


def paged_keys(fmt: str, requests: int, context: int):
    """Each request's keys on its own shuffled pages."""
    blocks = -(-context // PAGE)
    pages = requests * blocks
    values, scales = make_rows(fmt, PAGE * pages)
    key_cache = values.view(pages, PAGE, -1)
    key_scales = None if scales is None else scales.view(pages, PAGE, -1)
    table = torch.randperm(pages, device="cuda").int().view(requests, blocks)
    return key_cache, key_scales, table


def time_us(fn, args) -> float:
    fn()  # compiles
    times = bench_gpu_time(fn, enable_cupti=not args.no_cupti, cold_l2_cache=True)
    return 1000 * float(np.median(times))


def bench_decode(fmt: str, rows: int, context: int, args) -> float:
    batch = args.batch
    key_cache, key_scales, table = paged_keys(fmt, batch, context)
    query, query_scales = make_rows(fmt, batch * rows)
    lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    requests = torch.arange(batch, dtype=torch.int32, device="cuda")
    scores = torch.empty(batch * rows, context, device="cuda")
    scale = torch.ones(1, device="cuda") if fmt == "fp8" else None

    def run() -> None:
        indexer_scores(
            query,
            key_cache,
            table,
            lengths,
            requests=requests,
            scores=scores,
            query_scales=query_scales,
            key_scales=key_scales,
            score_scale=scale,
        )

    return time_us(run, args)


def bench_select(fmt: str, context: int, args) -> float:
    rows, top_k = args.select_rows, args.top_k
    key_cache, key_scales, table = paged_keys(fmt, 1, context)
    query, query_scales = make_rows(fmt, rows)
    first = context - rows + 1
    ends = torch.arange(first, first + rows, dtype=torch.int32, device="cuda")
    tiles = rows // 128
    tile_requests = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    # Thresholds from a key sample: each row's estimated top_k * (1 + slack)-th
    # largest score.
    sample = torch.randint(first, (args.sample,), device="cuda")
    page, row = table[0, sample // PAGE].long(), sample % PAGE
    # Gathered as bytes: not every indexing op takes FP8.
    sampled = key_cache.view(torch.uint8)[page, row].view(key_cache.dtype)
    sampled = as_float(sampled, None if key_scales is None else key_scales[page, row])
    sample_scores = as_float(query, query_scales) @ sampled.T
    rank = max(1, round(top_k * (1 + args.slack) * args.sample / context))
    requested = sample_scores.topk(min(rank, args.sample), 1).values[:, -1]
    thresholds = requested.clone()
    counters = torch.zeros(rows, 2, dtype=torch.int32, device="cuda")
    candidates = torch.empty(rows, 2 * top_k, 2, dtype=torch.int32, device="cuda")
    arrivals = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    ranges = torch.empty(2, rows, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, device="cuda") if fmt == "fp8" else None

    def run() -> None:
        thresholds.copy_(requested)
        counters.zero_()
        indexer_threshold_select(
            query,
            key_cache,
            ends,
            tile_requests,
            table,
            thresholds=thresholds,
            counters=counters,
            candidates=candidates,
            max_context=context,
            top_k=top_k,
            query_scales=query_scales,
            key_scales=key_scales,
            arrivals=arrivals,
            ranges=ranges,
            score_scale=scale,
        )

    return time_us(run, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 131072, 1048576]
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--batch", type=int, default=8, help="decode requests")
    parser.add_argument("--formats", nargs="+", default=["bf16", "fp8", "nvfp4"])
    parser.add_argument("--select-rows", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--sample", type=int, default=16384, help="sampled keys")
    parser.add_argument("--slack", type=float, default=0.25)
    parser.add_argument("--no-cupti", action="store_true")
    args = parser.parse_args()
    if args.select_rows % 128:
        parser.error("--select-rows must be a multiple of 128")

    torch.manual_seed(0)
    major, minor = torch.cuda.get_device_capability()
    sm10x = major == 10
    formats = [f for f in args.formats if sm10x or f != "nvfp4"]
    print(f"{torch.cuda.get_device_name()} (SM{major}{minor})")
    print(f"{'op':<8} {'format':<6} {'rows':>5} {'context':>8} {'us':>10} {'GB/s':>8}")

    def report(op: str, fmt: str, rows: int, context: int, us: float, reads: int):
        gbps = reads * context * KEY_BYTES[fmt] / us / 1e3
        print(f"{op:<8} {fmt:<6} {rows:>5} {context:>8} {us:>10.1f} {gbps:>8.0f}")

    for context in args.contexts:
        for rows in args.rows:
            for fmt in formats:
                us = bench_decode(fmt, rows, context, args)
                report("decode", fmt, rows, context, us, args.batch)
            torch.cuda.empty_cache()
    if not sm10x:
        print("Threshold select needs SM10x: skipped")
        return
    for context in args.contexts:
        for fmt in formats:
            us = bench_select(fmt, context, args)
            report("select", fmt, args.select_rows, context, us, 1)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
