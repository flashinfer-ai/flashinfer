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

"""Benchmark the experimental NVFP4 DeepSeek-V4 decode attention on SM100/SM103.

Five rows (batch 32, six query tokens per request, 64 heads, KV 8K..128K) timed
with CUPTI and a cold L2 between iterations. The optional trtllm-gen FP8 MLA
decode column (``--with-trtllm``) is informational only: it runs a different
numeric format (FP8 KV with 576-wide rows) and a different masking contract.

Usage::

    python benchmarks/bench_cake_nvfp4_mla_decode.py [--with-trtllm] [--kv 8192 16384]
"""

import argparse
import math
import statistics

import torch

from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
    HEAD_DIM,
    PAGE_SIZE,
    Q_LEN,
    ROW_BYTES,
    SF_ROW_BYTES,
    nvfp4_mla_decode_workspace_size,
    quantize_nvfp4,
)
from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla
from flashinfer.testing import bench_gpu_time_with_cupti

DEFAULT_KV = (8192, 16384, 32768, 65536, 131072)


def make_inputs(batch, kv_len, num_heads, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages = batch * (kv_len // PAGE_SIZE)
    q = torch.randn((batch * Q_LEN, num_heads, HEAD_DIM), generator=gen, device=device)
    kv = torch.randn((pages, PAGE_SIZE, HEAD_DIM), generator=gen, device=device)
    query, query_scale = quantize_nvfp4(q)
    kv_cache, kv_scale = quantize_nvfp4(kv)
    block_tables = (
        torch.randperm(pages, generator=gen, device=device)
        .to(torch.int32)
        .view(batch, -1)
    )
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    return q, kv, query, query_scale, kv_cache, kv_scale, block_tables, seq_lens


def median_ms(fn):
    times = bench_gpu_time_with_cupti(fn, cold_l2_cache=True)
    return float(statistics.median(times))


def bench_cake(batch, kv_len, num_heads, device):
    _, _, query, query_scale, kv_cache, kv_scale, block_tables, seq_lens = make_inputs(
        batch, kv_len, num_heads, device
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        nvfp4_mla_decode_workspace_size([kv_len] * batch, num_heads, num_sms=num_sms),
        dtype=torch.uint8,
        device=device,
    )
    decode = prepare_nvfp4_batch_decode_with_kv_cache_mla(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace,
        sm_scale=HEAD_DIM**-0.5,
        seq_lens_cpu=seq_lens.cpu(),
    )
    return median_ms(decode), decode.plan


def bench_trtllm_fp8(batch, kv_len, num_heads, device):
    """Informational trtllm-gen FP8 MLA decode (576-wide rows, q_len 6) or None."""
    try:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        return None
    try:
        gen = torch.Generator(device=device).manual_seed(0)
        pages = batch * (kv_len // PAGE_SIZE)
        query = torch.randn(
            (batch, Q_LEN, num_heads, 576), generator=gen, device=device
        ).to(torch.float8_e4m3fn)
        kv = torch.randn((pages, 1, PAGE_SIZE, 576), generator=gen, device=device).to(
            torch.float8_e4m3fn
        )
        block_tables = (
            torch.randperm(pages, generator=gen, device=device)
            .to(torch.int32)
            .view(batch, -1)
        )
        seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)

        def run():
            return trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv,
                workspace_buffer=workspace,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=kv_len,
                bmm1_scale=576**-0.5,
                bmm2_scale=1.0,
            )

        run()
        torch.cuda.synchronize()
        return median_ms(run)
    except Exception as exc:  # informational column only
        print(f"  trtllm FP8 MLA column unavailable: {type(exc).__name__}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv", type=int, nargs="+", default=list(DEFAULT_KV))
    parser.add_argument(
        "--with-trtllm",
        action="store_true",
        help="add the informational trtllm-gen FP8 MLA column",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    print(
        f"{name}: NVFP4 DeepSeek-V4 decode, batch {args.batch}, q_len {Q_LEN}, heads {args.heads}"
    )
    header = f"{'kv_len':>8} {'splits':>6} {'schedule':>9} {'ms':>9} {'TFLOPS':>8} {'GB/s':>8}"
    if args.with_trtllm:
        header += f" {'trtllm fp8 ms':>14}"
    print(header)
    for kv_len in args.kv:
        ms, plan = bench_cake(args.batch, kv_len, args.heads, device)
        rows = args.batch * Q_LEN * args.heads
        flops = 2.0 * rows * kv_len * (2 * HEAD_DIM)
        nbytes = args.batch * kv_len * (ROW_BYTES + SF_ROW_BYTES) + rows * (
            ROW_BYTES + SF_ROW_BYTES + 2 * HEAD_DIM
        )
        line = f"{kv_len:>8} {plan.max_splits:>6} {plan.schedule:>9} {ms:>9.4f} {flops / (ms * 1e9):>8.1f} {nbytes / (ms * 1e6):>8.1f}"
        if args.with_trtllm:
            ref = bench_trtllm_fp8(args.batch, kv_len, args.heads, device)
            line += f" {ref:>14.4f}" if ref is not None else f" {'n/a':>14}"
        print(line)
    assert math.isfinite(ms)


if __name__ == "__main__":
    main()
