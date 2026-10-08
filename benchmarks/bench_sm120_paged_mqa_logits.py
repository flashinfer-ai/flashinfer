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
"""Benchmark the prepared SM120a paged FP8 MQA lightning-indexer logits.

Sweeps decode points on the current SM120a device and reports the CUPTI kernel
time of one ``plan.run()`` (cold L2; both kernels of the sequence) with the
effective fused-KV stream rate and the MMA rate.

Examples:
    python benchmarks/bench_sm120_paged_mqa_logits.py
    python benchmarks/bench_sm120_paged_mqa_logits.py --heads 64 --page-kv 64 \
        --next-n 1 2 4 --batch 8 64 --context 4096 16384
"""

import argparse
import statistics

import torch
from flashinfer.experimental.deepgemm_sm120_paged_mqa_logits import sm120_paged_mqa
from flashinfer.sm120_paged_mqa_logits import prepare_sm120_paged_mqa_logits
from flashinfer.testing import bench_gpu_time_with_cupti

HEAD_DIM = 128
FUSED_ROW_BYTES = HEAD_DIM + 4


def make_inputs(heads, page_kv, next_n, batch, context, device):
    generator = torch.Generator(device=device).manual_seed(0)
    ctx_last = torch.full((batch,), context, device=device, dtype=torch.int32)
    offsets = (next_n - 1 - torch.arange(next_n, device=device, dtype=torch.int32))[
        None, :
    ]
    context_lens = (ctx_last[:, None] - offsets).clamp_min(1).contiguous()
    blocks = (context + page_kv - 1) // page_kv
    pages = blocks * batch + 1
    block_table = (
        torch.arange(batch * blocks, device=device, dtype=torch.int32)
        .reshape(batch, blocks)
        .contiguous()
    )
    kv = torch.randn(pages, page_kv, HEAD_DIM, device=device, generator=generator)
    scale = (kv.abs().amax(dim=-1).clamp_min(1e-4) / 448.0).contiguous()
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    fused = torch.zeros(
        pages, page_kv * FUSED_ROW_BYTES, device=device, dtype=torch.uint8
    )
    fused[:, : page_kv * HEAD_DIM] = kv_fp8.reshape(pages, -1).view(torch.uint8)
    fused[:, page_kv * HEAD_DIM :] = scale.view(torch.uint8)
    kv_cache = fused.view(pages, page_kv, 1, FUSED_ROW_BYTES)
    q = torch.randn(
        batch, next_n, heads, HEAD_DIM, device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    weights = torch.rand(batch * next_n, heads, device=device, generator=generator)
    return q, kv_cache, weights, context_lens, block_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--page-kv", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--next-n", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batch", nargs="+", type=int, default=[8, 64])
    parser.add_argument("--context", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--repeat-ms", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch, sms = sm120_paged_mqa.device_facts(index)
    print(f"{torch.cuda.get_device_name(device)} ({arch}, {sms} SMs)")
    header = (
        f"{'H':>4s} {'page':>5s} {'n':>3s} {'batch':>6s} {'ctx':>7s} "
        f"{'route':28s} {'us':>9s} {'KV GB/s':>9s} {'TFLOP/s':>8s}"
    )
    print(header)
    for heads in args.heads:
        for page_kv in args.page_kv:
            if not sm120_paged_mqa.route_available(heads, page_kv, args.next_n[0]):
                continue
            for next_n in args.next_n:
                if not sm120_paged_mqa.route_available(heads, page_kv, next_n):
                    continue
                for batch in args.batch:
                    for context in args.context:
                        operands = make_inputs(
                            heads, page_kv, next_n, batch, context, device
                        )
                        plan = prepare_sm120_paged_mqa_logits(
                            *operands, max_context_len=context
                        )
                        plan.run()
                        torch.cuda.synchronize()
                        times = bench_gpu_time_with_cupti(
                            plan.run,
                            dry_run_time_ms=25,
                            repeat_time_ms=args.repeat_ms,
                            cold_l2_cache=True,
                        )
                        ms = statistics.median(times)
                        kv_bytes = batch * context * FUSED_ROW_BYTES
                        flops = 2.0 * batch * next_n * heads * HEAD_DIM * context
                        print(
                            f"{heads:4d} {page_kv:5d} {next_n:3d} {batch:6d} "
                            f"{context:7d} {plan.route_name:28s} {ms * 1e3:9.1f} "
                            f"{kv_bytes / ms / 1e6:9.1f} {flops / ms / 1e9:8.1f}"
                        )
                        del plan, operands


if __name__ == "__main__":
    main()
