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
"""Benchmark the prepared dense FP4/FP8 MQA lightning-indexer logits.

Sweeps (precision, queries, KV length) rows on the current SM100a/SM103a device
and reports the CUPTI kernel time of one ``plan.run()`` (cold L2, all kernels of
the route) with the effective KV stream rate and the MMA rate.

Examples:
    python benchmarks/bench_dense_mqa_generated.py
    python benchmarks/bench_dense_mqa_generated.py --precision fp8 --queries 1 128 --keys 4096 1048576
"""

import argparse
import statistics

import torch

from flashinfer.dense_mqa import prepare_dense_mqa_logits
from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa
from flashinfer.testing import bench_gpu_time_with_cupti


def make_inputs(precision, queries, keys, device):
    generator = torch.Generator(device=device).manual_seed(0)
    if precision == "fp4":
        q = torch.randint(
            0,
            256,
            (queries, 32, 64),
            device=device,
            dtype=torch.uint8,
            generator=generator,
        )
        kv = torch.randint(
            0, 256, (keys, 64), device=device, dtype=torch.uint8, generator=generator
        )
        q_scales = torch.randint(
            124,
            131,
            (queries, 32, 4),
            device=device,
            dtype=torch.uint8,
            generator=generator,
        )
        kv_scales = torch.randint(
            124, 131, (keys, 4), device=device, dtype=torch.uint8, generator=generator
        )
        rows = queries
    else:
        rows = max(4, queries)
        q = torch.randn(rows, 32, 128, device=device, generator=generator).to(
            torch.float8_e4m3fn
        )
        kv = torch.randn(keys, 128, device=device, generator=generator).to(
            torch.float8_e4m3fn
        )
        q_scales = None
        kv_scales = torch.rand(keys, device=device, generator=generator) + 0.5
    weights = torch.randn(rows, 32, device=device, generator=generator)
    starts = torch.zeros(queries, device=device, dtype=torch.int32)
    ends = torch.full_like(starts, keys)
    return q, kv, q_scales, kv_scales, weights, starts, ends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision", nargs="+", default=["fp4", "fp8"], choices=["fp4", "fp8"]
    )
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--keys", nargs="+", type=int, default=[4096, 32768, 131072])
    parser.add_argument("--repeat-ms", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("cuda")
    arch, sms = dense_mqa.device_facts(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    print(f"{torch.cuda.get_device_name(device)} ({arch}, {sms} SMs)")
    print(
        f"{'precision':9s} {'Q':>5s} {'K':>8s} {'route':19s} {'launches':>8s} {'us':>9s} {'KV GB/s':>8s} {'TFLOP/s':>8s}"
    )
    for precision in args.precision:
        for queries in args.queries:
            for keys in args.keys:
                q, kv, q_scales, kv_scales, weights, starts, ends = make_inputs(
                    precision, queries, keys, device
                )
                plan = prepare_dense_mqa_logits(
                    precision,
                    q,
                    kv,
                    weights,
                    starts,
                    ends,
                    q_scales=q_scales,
                    kv_scales=kv_scales,
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
                kv_bytes = keys * (64 + 4 if precision == "fp4" else 128 + 4)
                flops = 2.0 * queries * 32 * 128 * keys
                print(
                    f"{precision:9s} {queries:5d} {keys:8d} {plan.route_name:19s} {plan.launch_count:8d} "
                    f"{ms * 1e3:9.1f} {kv_bytes / ms / 1e6:8.1f} {flops / ms / 1e9:8.1f}"
                )
                del plan, q, kv, q_scales, kv_scales, weights, starts, ends


if __name__ == "__main__":
    main()
