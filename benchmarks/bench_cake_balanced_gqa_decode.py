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

"""Benchmark the experimental balanced BF16 paged GQA decode on SM100/SM103.

Ragged and uniform decode batches (eight query heads per KV head, head
dimension 128, 16-token pages) timed with CUPTI and a cold L2 between
iterations, against ``trtllm_batch_decode_with_kv_cache(backend="trtllm-gen")``
on the same tensors (``--with-trtllm``).  The AgentX rows reproduce the ragged
KV pattern of flashinfer-ai/flashinfer#4832.

Usage::

    python benchmarks/bench_cake_balanced_gqa_decode.py [--with-trtllm] [--rows agentx_b16 ...]
"""

import argparse
import json
import math

import torch

from flashinfer.decode import (
    prepare_balanced_batch_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
    GROUP_RATIO,
    HEAD_DIM,
    PAGE_SIZE,
    balanced_gqa_decode_workspace_size,
)
from flashinfer.testing import bench_gpu_time_with_cupti

AGENTX = [
    8193,
    57345,
    73729,
    81921,
    98305,
    106497,
    114689,
    131073,
    139265,
    147457,
    163841,
    180225,
    196609,
    212993,
    229377,
    237569,
]


def _agentx(batch):
    return [AGENTX[i % len(AGENTX)] for i in range(batch)]


def _random_lengths(batch, lo, hi, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi + 1, (batch,), generator=gen).tolist()


ROWS = {
    "agentx_b16_hkv1": (_agentx(16), 1, 1),
    "agentx_b32_hkv1": (_agentx(32), 1, 1),
    "agentx_b64_hkv1": (_agentx(64), 1, 1),
    "agentx_b128_hkv1": (_agentx(128), 1, 1),
    "agentx_b192_hkv1": (_agentx(192), 1, 1),
    "agentx_b256_hkv1": (_agentx(256), 1, 1),
    "random_128_128k_b256_hkv1": (_random_lengths(256, 128, 131072, 1), 1, 1),
    "random_128_65k_b64_hkv8": (_random_lengths(64, 128, 65536, 2), 8, 1),
    "uniform_b4_hkv8_s65536": ([65536] * 4, 8, 1),
    "uniform_b16_hkv8_s65536": ([65536] * 16, 8, 1),
    "uniform_b64_hkv8_s32768": ([32768] * 64, 8, 1),
    "uniform_b128_hkv8_s32768": ([32768] * 128, 8, 1),
    "uniform_b128_hkv8_s4096": ([4096] * 128, 8, 1),
    "agentx_b16_hkv1_mtp7": (_agentx(16), 1, 7),
}


def make_inputs(seq_lens, num_kv_heads, q_len, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    num_q_heads = GROUP_RATIO * num_kv_heads
    max_pages = (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = (max_pages + 7) // 8 * 8
    num_pages = batch * max_pages
    query = torch.randn(
        (batch * q_len, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache = torch.randn(
        (num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    v_cache = torch.randn(
        (num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    block_tables = (
        torch.randperm(num_pages, generator=gen, device=device)
        .to(torch.int32)
        .view(batch, max_pages)
    )
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return query, k_cache, v_cache, block_tables, seq_lens_dev


def _flops(seq_lens, num_q_heads, q_len):
    tokens = sum(s - (q_len - 1 - j) for s in seq_lens for j in range(q_len))
    return 4.0 * tokens * num_q_heads * HEAD_DIM


def _bytes(seq_lens, num_kv_heads, q_len):
    tokens = sum(seq_lens)
    kv = 2 * tokens * num_kv_heads * HEAD_DIM * 2
    q = len(seq_lens) * q_len * GROUP_RATIO * num_kv_heads * HEAD_DIM * 2
    return kv + 2 * q


def _median_ms(fn):
    times = bench_gpu_time_with_cupti(fn, cold_l2_cache=True)
    times = sorted(times)
    return float(times[len(times) // 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="*", default=list(ROWS))
    parser.add_argument("--with-trtllm", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    workspace = torch.empty(
        balanced_gqa_decode_workspace_size(device), dtype=torch.uint8, device=device
    )
    trtllm_workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    sm_scale = HEAD_DIM**-0.5
    results = []
    print(f"{torch.cuda.get_device_name(device)}, {len(args.rows)} rows")
    header = f"{'row':<30}{'balanced ms':>13}{'TFLOP/s':>9}{'GB/s':>8}"
    if args.with_trtllm:
        header += f"{'trtllm-gen ms':>15}{'speedup':>9}"
    print(header)
    for name in args.rows:
        seq_lens, num_kv_heads, q_len = ROWS[name]
        query, k_cache, v_cache, block_tables, seq_lens_dev = make_inputs(
            seq_lens, num_kv_heads, q_len, device
        )
        out = torch.empty_like(query)
        runner = prepare_balanced_batch_decode_with_kv_cache(
            query,
            (k_cache, v_cache),
            block_tables,
            seq_lens_dev,
            workspace,
            sm_scale=sm_scale,
            q_len_per_req=q_len,
            out=out,
        )
        ms = _median_ms(runner)
        flops = _flops(seq_lens, GROUP_RATIO * num_kv_heads, q_len)
        row = dict(
            row=name,
            batch=len(seq_lens),
            num_kv_heads=num_kv_heads,
            q_len=q_len,
            balanced_ms=ms,
            tflops=flops / ms / 1e9,
            gbps=_bytes(seq_lens, num_kv_heads, q_len) / ms / 1e6,
        )
        line = f"{name:<30}{ms:>13.4f}{row['tflops']:>9.1f}{row['gbps']:>8.0f}"
        if args.with_trtllm:
            kv_cache = torch.stack([k_cache, v_cache], dim=1).contiguous()
            trtllm_out = torch.empty_like(query)

            def _trtllm():
                trtllm_batch_decode_with_kv_cache(
                    query,
                    kv_cache,
                    trtllm_workspace,
                    block_tables,
                    seq_lens_dev,
                    max(seq_lens),
                    sm_scale,
                    1.0,
                    out=trtllm_out,
                    backend="trtllm-gen",
                    q_len_per_req=q_len,
                )

            _trtllm()
            torch.cuda.synchronize()
            max_diff = (trtllm_out.float() - out.float()).abs().max().item()
            trtllm_ms = _median_ms(_trtllm)
            row.update(trtllm_gen_ms=trtllm_ms, speedup=trtllm_ms / ms, max_abs_diff_vs_trtllm=max_diff)
            line += f"{trtllm_ms:>15.4f}{trtllm_ms / ms:>9.3f}"
            del kv_cache, trtllm_out
        print(line)
        results.append(row)
        del query, k_cache, v_cache, block_tables, seq_lens_dev, out, runner
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(device=torch.cuda.get_device_name(device), rows=results), handle, indent=2
            )
    if args.with_trtllm:
        speedups = [r["speedup"] for r in results]
        print(f"geomean speedup vs trtllm-gen: {math.exp(sum(map(math.log, speedups)) / len(speedups)):.3f}")


if __name__ == "__main__":
    main()
