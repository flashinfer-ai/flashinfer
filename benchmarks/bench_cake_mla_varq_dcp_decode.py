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

"""Benchmark the experimental Cake compact var-Q + DCP MLA decode on SM100/SM103.

The 24 performance rows compare the Cake prepared runner against the upstream
``cute_dsl_mla_decode(..., is_var_seq=True, enable_dcp=True)`` path on the
identical rank-0 inputs: page size 64, ``(H, W)`` in ``{(128, 8), (96, 8),
(48, 4), (24, 2), (64, 4), (32, 2), (16, 1)}``, batch 8..128, global KV
8K..128K (uniform random lengths, the last request at the maximum), query
patterns ``q1``, uniform ``q4`` and ragged 1..3 (``mtp3``), BF16 and FP8
query/KV.  Lengths and page permutations follow the upstream MLA benchmark RNG
rules (seed 42; rank-local pages permuted with seed ``42 + 7919 * rank``).
Both arms are timed with CUPTI and a cold L2 between iterations.

Usage::

    python benchmarks/bench_cake_mla_varq_dcp_decode.py [--dtype bf16 fp8] [--rows perf_bf16_h128_w8_b16_q1_s32k ...]
"""

import argparse
import math
import statistics

import torch

from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
    HEAD_DIM_QK,
    HEAD_DIM_V,
    cake_mla_varq_dcp_decode_workspace_size,
    prepare_cake_mla_varq_dcp_decode,
)
from flashinfer.testing import bench_gpu_time_with_cupti

LATENT_DIM = HEAD_DIM_V
ROPE_DIM = HEAD_DIM_QK - HEAD_DIM_V
PAGE_SIZE = 64
SEED = 42
PAGE_PERMUTATION_RANK_STRIDE = 7919
CUTE_WORKSPACE_BYTES = 128 * 1024 * 1024
SOFTMAX_SCALE = 1.0 / math.sqrt(LATENT_DIM)
DTYPES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}

# (label suffix, num_heads, cp_world, batch_size, q_pattern, q_len, max_global_len)
ROWS = (
    ("h128_w8_b16_q1_s32k", 128, 8, 16, "uniform", 1, 32768),
    ("h128_w8_b8_mtp3_s128k", 128, 8, 8, "ragged", 3, 131072),
    ("h128_w8_b64_q1_s8k", 128, 8, 64, "uniform", 1, 8192),
    ("h96_w8_b32_mtp3_s32k", 96, 8, 32, "ragged", 3, 32768),
    ("h96_w8_b16_q1_s128k", 96, 8, 16, "uniform", 1, 131072),
    ("h96_w8_b64_mtp3_s8k", 96, 8, 64, "ragged", 3, 8192),
    ("h48_w4_b32_mtp3_s32k", 48, 4, 32, "ragged", 3, 32768),
    ("h48_w4_b128_q1_s8k", 48, 4, 128, "uniform", 1, 8192),
    ("h24_w2_b64_mtp3_s16k", 24, 2, 64, "ragged", 3, 16384),
    ("h64_w4_b32_q4_s32k", 64, 4, 32, "uniform", 4, 32768),
    ("h32_w2_b128_q1_s16k", 32, 2, 128, "uniform", 1, 16384),
    ("h16_w1_b64_mtp3_s8k", 16, 1, 64, "ragged", 3, 8192),
)


def _ceil_div(a, b):
    return -(-a // b)


def _local_length(global_len, cp_world, cp_rank):
    return max(_ceil_div(global_len - cp_rank, cp_world), 0)


def draw_lengths(batch_size, q_pattern, q_len, max_global_len, generator):
    global_lens = [
        int(torch.randint(1, max_global_len + 1, (1,), generator=generator).item())
        for _ in range(batch_size)
    ]
    global_lens[-1] = max_global_len
    if q_pattern == "uniform":
        q_lens = [q_len] * batch_size
    else:
        q_lens = [
            int(torch.randint(1, q_len + 1, (1,), generator=generator).item())
            for _ in range(batch_size)
        ]
    return global_lens, q_lens


def make_rank_inputs(
    num_heads,
    cp_world,
    cp_rank,
    batch_size,
    q_pattern,
    q_len,
    max_global_len,
    dtype,
    device,
):
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    global_lens, q_lens = draw_lengths(
        batch_size, q_pattern, q_len, max_global_len, generator
    )
    total_q = sum(q_lens)
    storage = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    query = (
        torch.randn(total_q, num_heads, HEAD_DIM_QK, dtype=storage, device=device) * 0.1
    ).to(dtype)
    global_kv = (
        torch.randn(
            batch_size, max(global_lens), HEAD_DIM_QK, dtype=storage, device=device
        )
        * 0.1
    ).to(dtype)
    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(
        0
    )
    local_lens = [_local_length(g, cp_world, cp_rank) for g in global_lens]
    pages_per = [max(1, _ceil_div(n, PAGE_SIZE)) for n in local_lens]
    total_pages = sum(pages_per)
    page_gen = torch.Generator(device="cpu").manual_seed(
        SEED + PAGE_PERMUTATION_RANK_STRIDE * cp_rank
    )
    page_ids = torch.randperm(total_pages, generator=page_gen).to(device)
    cache = torch.zeros(total_pages, PAGE_SIZE, HEAD_DIM_QK, dtype=dtype, device=device)
    page_table = torch.zeros(
        batch_size, max(pages_per), dtype=torch.int32, device=device
    )
    offset = 0
    for b, (g, n, count) in enumerate(
        zip(global_lens, local_lens, pages_per, strict=True)
    ):
        ids = page_ids[offset : offset + count]
        page_table[b, :count] = ids.to(torch.int32)
        if n:
            padded = torch.zeros(
                count * PAGE_SIZE, HEAD_DIM_QK, dtype=dtype, device=device
            )
            padded[:n] = global_kv[b, cp_rank:g:cp_world]
            cache[ids] = padded.view(count, PAGE_SIZE, HEAD_DIM_QK)
        offset += count
    return dict(
        query=query,
        kv_cache=cache,
        page_table=page_table,
        seq_lens=torch.tensor(local_lens, dtype=torch.int32, device=device),
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=q_len,
        max_seq_len=max(1, max(local_lens)),
        causal=torch.tensor(global_lens, dtype=torch.int32, device=device),
        out=torch.empty(
            total_q, num_heads, HEAD_DIM_V, dtype=torch.bfloat16, device=device
        ),
        lse=torch.empty(total_q, num_heads, dtype=torch.float32, device=device),
        batch_size=batch_size,
        num_heads=num_heads,
        cp_world=cp_world,
        cp_rank=cp_rank,
        total_q=total_q,
        local_lens=local_lens,
    )


def median_ms(fn):
    return float(statistics.median(bench_gpu_time_with_cupti(fn, cold_l2_cache=True)))


def bench_cake(inputs, device):
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        cake_mla_varq_dcp_decode_workspace_size(
            batch_size=inputs["batch_size"],
            max_q_len=inputs["max_q_len"],
            num_heads=inputs["num_heads"],
            max_seq_len=inputs["max_seq_len"],
            num_sms=num_sms,
        ),
        dtype=torch.uint8,
        device=device,
    )
    runner = prepare_cake_mla_varq_dcp_decode(
        inputs["query"],
        inputs["kv_cache"],
        inputs["page_table"],
        inputs["seq_lens"],
        inputs["cum_seq_lens_q"],
        inputs["max_q_len"],
        max_seq_len=inputs["max_seq_len"],
        softmax_scale=SOFTMAX_SCALE,
        workspace_buffer=workspace,
        causal_seqlens_kv_global=inputs["causal"],
        cp_world=inputs["cp_world"],
        cp_rank=inputs["cp_rank"],
        out=inputs["out"],
        lse=inputs["lse"],
    )
    runner.launch()
    torch.cuda.synchronize()
    return median_ms(runner.launch), runner.route


def bench_cute_dsl(inputs, device):
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import cute_dsl_mla_decode

    workspace = torch.empty(CUTE_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    def call():
        return cute_dsl_mla_decode(
            query=inputs["query"],
            kv_cache=inputs["kv_cache"],
            workspace_buffer=workspace,
            kv_lora_rank=LATENT_DIM,
            qk_rope_head_dim=ROPE_DIM,
            block_tables=inputs["page_table"],
            seq_lens=inputs["seq_lens"],
            max_seq_len=inputs["max_seq_len"],
            softmax_scale=SOFTMAX_SCALE,
            is_var_seq=True,
            return_lse=True,
            out=inputs["out"],
            lse=inputs["lse"],
            cum_seq_lens_q=inputs["cum_seq_lens_q"],
            max_q_len=inputs["max_q_len"],
            enable_dcp=True,
            cp_world=inputs["cp_world"],
            cp_rank=inputs["cp_rank"],
            causal_seqlens_kv_global=inputs["causal"],
        )

    call()
    torch.cuda.synchronize()
    return median_ms(call)


def attention_gbps(inputs, ms):
    element = inputs["query"].element_size()
    nbytes = (
        inputs["query"].numel() * element
        + sum(inputs["local_lens"]) * HEAD_DIM_QK * element
    )
    nbytes += inputs["out"].numel() * inputs["out"].element_size()
    return nbytes / ms / 1e6


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dtype", nargs="+", choices=sorted(DTYPES), default=sorted(DTYPES)
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help="row labels to run (default: all 24)"
    )
    parser.add_argument(
        "--no-cute-dsl", action="store_true", help="skip the upstream CuTe-DSL column"
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    header = f"{'row':38s} {'route':28s} {'cake_us':>10s} {'cute_us':>10s} {'speedup':>8s} {'cake_GB/s':>10s}"
    print(header)
    ratios = []
    for dtype_name in args.dtype:
        for (
            suffix,
            num_heads,
            cp_world,
            batch_size,
            q_pattern,
            q_len,
            max_global_len,
        ) in ROWS:
            label = f"perf_{dtype_name}_{suffix}"
            if args.rows and label not in args.rows:
                continue
            inputs = make_rank_inputs(
                num_heads,
                cp_world,
                0,
                batch_size,
                q_pattern,
                q_len,
                max_global_len,
                DTYPES[dtype_name],
                device,
            )
            cake_ms, route = bench_cake(inputs, device)
            cute_ms = None if args.no_cute_dsl else bench_cute_dsl(inputs, device)
            speedup = "" if cute_ms is None else f"{cute_ms / cake_ms:8.3f}"
            cute = "" if cute_ms is None else f"{cute_ms * 1e3:10.2f}"
            if cute_ms is not None:
                ratios.append(cute_ms / cake_ms)
            print(
                f"{label:38s} {route:28s} {cake_ms * 1e3:10.2f} {cute:>10s} {speedup:>8s} {attention_gbps(inputs, cake_ms):10.1f}"
            )
    if ratios:
        print(
            f"geomean speedup over cute_dsl_mla_decode: {math.exp(sum(map(math.log, ratios)) / len(ratios)):.4f} "
            f"({sum(r > 1.0 for r in ratios)}/{len(ratios)} rows faster)"
        )


if __name__ == "__main__":
    main()
