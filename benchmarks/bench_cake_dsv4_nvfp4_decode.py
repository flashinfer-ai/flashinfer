# Copyright (c) 2026 by FlashInfer team.
#
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

"""Benchmark the DeepSeek-V4 NVFP4 sparse-MLA decode through the public entry on SM100 / SM103.

``trtllm_batch_decode_sparse_mla_dsv4(kv_cache_format="nvfp4", backend="cake")`` over the
384-byte-per-token NVFP4 cache: representative decode rows (query tokens T, heads H, main
top-k K over a page-64 pool, optionally an extra top-k over a compressed pool), HBM-resident
pools sized well above L2, random selections without replacement, CUPTI timing with a cold L2
between iterations (whole call: gather/convert, attention and the split merge). Reports the
median latency, the gathered-cache bandwidth and the selected family member / split count.

``--with-trtllm`` adds the TRTLLM-GEN FP8 DSv4 sparse decode (``kv_cache_format="fp8"``,
584-byte rows, 128 SWA + compressed columns) on the same selections as an informational
column; it is a different storage format and masking contract, not a like-for-like baseline.

Usage::

    python benchmarks/bench_cake_dsv4_nvfp4_decode.py [--rows C-h16-k128-t1 ...] [--with-trtllm] [--json out.json]
"""

import argparse
import json
import statistics

import torch

from flashinfer.mla import (
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.mla.cake_dsv4 import _nvfp4_plan, get_cake_dsv4_workspace_bytes
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.utils import get_compute_capability

HEAD_DIM = 512
BYTES_PER_TOKEN = 384
PAGE_SIZE = 64
POOL_TOKENS = 1 << 20  # 1 Mi tokens * 384 B = 384 MiB per pool, well above L2
SM_SCALE = HEAD_DIM**-0.5

# (label, tokens, heads, main top-k, extra top-k, extra page size)
ROWS = [
    ("C-h16-k128-t1", 1, 16, 128, 0, 64),
    ("C-h16-k128-t32", 32, 16, 128, 0, 64),
    ("C-h16-k512-t32", 32, 16, 512, 0, 64),
    ("C-h64-k128-t32", 32, 64, 128, 0, 64),
    ("C-h128-k128-t1", 1, 128, 128, 0, 64),
    ("C-h128-k128-t32", 32, 128, 128, 0, 64),
    ("C-h128-k512-t32", 32, 128, 512, 0, 64),
    ("C-h128-k512-t128", 128, 128, 512, 0, 64),
    ("D-h16-m128-e512p64-t32", 32, 16, 128, 512, 64),
    ("D-h128-m128-e512p64-t32", 32, 128, 128, 512, 64),
    ("D-h128-m128-e132p2-t32", 32, 128, 128, 132, 2),
    ("M-h128-k128-t12", 12, 128, 128, 0, 64),
]


def median_ms(fn):
    return float(statistics.median(bench_gpu_time_with_cupti(fn, cold_l2_cache=True)))


def random_indices(num_tokens, topk, pool_tokens, generator, device):
    rows = [
        torch.randperm(pool_tokens, generator=generator, device=device)[:topk]
        for _ in range(num_tokens)
    ]
    return torch.stack(rows).to(torch.int32)


def pack_pool(pool_tokens, page_size, generator, device):
    """Quantize an HBM-resident latent pool page by page (bounded temporary memory)."""
    pages = pool_tokens // page_size
    cache = None
    chunk = max(1, (64 << 20) // (page_size * HEAD_DIM * 2))  # 64 MiB of BF16 latent per chunk
    for start in range(0, pages, chunk):
        count = min(chunk, pages - start)
        latent = (
            torch.randn(count, page_size, HEAD_DIM, generator=generator, device=device)
            .to(torch.bfloat16)
            * 0.1
        )
        packed = nvfp4_quantize_pack_sparse_mla_cache(latent)
        if cache is None:
            cache = torch.empty((pages,) + tuple(packed.shape[1:]), dtype=packed.dtype, device=device)
        cache[start : start + count].copy_(packed)
    return cache


def bench_row(label, num_tokens, num_heads, topk, extra_topk, extra_page_size, device, with_trtllm):
    gen = torch.Generator(device=device).manual_seed(17)
    main_cache = pack_pool(POOL_TOKENS, PAGE_SIZE, gen, device)
    main_idx = random_indices(num_tokens, topk, POOL_TOKENS, gen, device)
    extra_cache = extra_idx = None
    if extra_topk:
        extra_cache = pack_pool(POOL_TOKENS, extra_page_size, gen, device)
        extra_idx = random_indices(num_tokens, extra_topk, POOL_TOKENS, gen, device)
    query = torch.randn(num_tokens, num_heads, HEAD_DIM, generator=gen, device=device).to(torch.bfloat16)
    workspace = torch.empty(
        get_cake_dsv4_workspace_bytes(
            num_tokens, num_heads, topk, torch.bfloat16, kv_cache_format="nvfp4", extra_topk=extra_topk
        ),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.empty_like(query)

    def run():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=main_cache,
            workspace_buffer=workspace,
            sparse_indices=main_idx,
            compressed_kv_cache=extra_cache,
            extra_sparse_indices=extra_idx,
            out=out,
            bmm1_scale=SM_SCALE,
            kv_cache_format="nvfp4",
            backend="cake",
        )

    run()
    torch.cuda.synchronize()
    ms = median_ms(run)
    plan = _nvfp4_plan(
        num_query_tokens=num_tokens,
        num_heads=num_heads,
        sparse_topk=topk,
        extra_topk=extra_topk,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
    )
    gathered_bytes = num_tokens * (topk + extra_topk) * BYTES_PER_TOKEN
    result = dict(
        row=label,
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        extra_page_size=extra_page_size if extra_topk else None,
        member=plan.member,
        num_splits=plan.num_splits,
        cake_nvfp4_ms=ms,
        gathered_gb_per_s=gathered_bytes / (ms * 1e-3) / 1e9,
    )
    if with_trtllm:
        result["trtllm_fp8_ms"] = bench_trtllm_fp8(num_tokens, num_heads, topk + extra_topk, device, gen)
    del main_cache, extra_cache
    torch.cuda.empty_cache()
    return result


def bench_trtllm_fp8(num_tokens, num_heads, total_topk, device, gen):
    """Informational TRTLLM-GEN FP8 DSv4 sparse decode on the same selection count, or None.

    FP8 E4M3 pools of 512-wide rows (SWA and compressed), a per-tensor FP8 query, one combined
    selection table whose first 128 columns are SWA entries and the rest compressed entries.
    """
    total_topk = max(total_topk, 128)
    try:
        pages = POOL_TOKENS // PAGE_SIZE
        swa = torch.randn(pages, 1, PAGE_SIZE, HEAD_DIM, generator=gen, device=device).to(torch.float8_e4m3fn)
        comp = torch.randn(pages, 1, PAGE_SIZE, HEAD_DIM, generator=gen, device=device).to(torch.float8_e4m3fn)
        idx = random_indices(num_tokens, total_topk, POOL_TOKENS, gen, device)
        lens = torch.full((num_tokens,), total_topk, dtype=torch.int32, device=device)
        seq_lens = torch.full((num_tokens,), 4096, dtype=torch.int32, device=device)
        query = torch.randn(num_tokens, 1, num_heads, HEAD_DIM, generator=gen, device=device).to(torch.float8_e4m3fn)
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        def run():
            return trtllm_batch_decode_sparse_mla_dsv4(
                query=query,
                swa_kv_cache=swa,
                compressed_kv_cache=comp,
                workspace_buffer=workspace,
                sparse_indices=idx,
                sparse_topk_lens=lens,
                seq_lens=seq_lens,
                bmm1_scale=SM_SCALE,
                kv_cache_format="fp8",
                backend="trtllm-gen",
            )

        run()
        torch.cuda.synchronize()
        return median_ms(run)
    except Exception as exc:  # informational column only
        print(f"  trtllm-gen FP8 reference unavailable: {type(exc).__name__}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="*", default=None, help="row labels to run (default: all)")
    parser.add_argument("--with-trtllm", action="store_true")
    parser.add_argument("--json", default=None, help="write the results to this JSON file")
    args = parser.parse_args()
    device = torch.device("cuda")
    cc = tuple(get_compute_capability(device))
    if cc not in ((10, 0), (10, 3)):
        raise SystemExit(f"backend='cake' NVFP4 decode needs SM100/SM103, got SM{cc[0]}{cc[1]}")
    rows = [r for r in ROWS if args.rows is None or r[0] in args.rows]
    print(f"{torch.cuda.get_device_name(device)} (SM{cc[0]}{cc[1]}), pool {POOL_TOKENS} tokens x {BYTES_PER_TOKEN} B per cache")
    header = f"{'row':26s} {'member':10s} {'splits':>6s} {'cake nvfp4 ms':>14s} {'gathered GB/s':>14s}"
    if args.with_trtllm:
        header += f" {'trtllm fp8 ms':>14s}"
    print(header)
    results = []
    for label, t, h, k, ek, eps in rows:
        r = bench_row(label, t, h, k, ek, eps, device, args.with_trtllm)
        results.append(r)
        line = f"{label:26s} {r['member']:10s} {r['num_splits']:6d} {r['cake_nvfp4_ms'] * 1e3:11.2f} us {r['gathered_gb_per_s']:14.1f}"
        if args.with_trtllm:
            ref = r.get("trtllm_fp8_ms")
            line += f" {ref * 1e3:11.2f} us" if ref is not None else f" {'n/a':>14s}"
        print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
