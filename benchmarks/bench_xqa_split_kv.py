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

"""Measure decode with a reserved block table and independently varying lengths.

Run unchanged against baseline and candidate installations. Timings are CUDA
Graph kernel times with a cold L2 cache, in microseconds.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from flashinfer import xqa
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(6694)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [
        (b, 8, 32, 128, length)
        for b in (20, 21, 22, 24, 32)
        for length in (1024, 4096, 32768)
    ]
    cases += [
        (1, 2, 16, dim, length) for dim in (128, 256) for length in (1024, 4096, 32768)
    ]
    if args.holdout:
        cases = [
            (b, 8, 32, 128, length)
            for b in (8, 16, 21, 22, 26, 27, 32)
            for length in (256, 2048, 8192, 65536)
        ]
        cases += [
            (b, 2, 16, 256, length)
            for b in (1, 2, 4)
            for length in (256, 2048, 8192, 65536)
        ]
    workspace = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    with args.output.open("w", buffering=1) as report:
        for batch, heads, qheads, dim, length in cases:
            page = 16
            shape = (batch * length // page, heads, page, dim)
            q = torch.randn(batch, 1, qheads, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            v = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            table = torch.zeros(batch, 65536 // page, device="cuda", dtype=torch.int32)
            table[:, : length // page] = torch.arange(
                batch * length // page, device="cuda"
            ).reshape(batch, -1)
            lens = torch.full((batch, 1), length, device="cuda", dtype=torch.uint32)
            out = torch.empty_like(q)
            sem = torch.zeros(2 * batch * heads + 8, device="cuda", dtype=torch.uint32)

            def run(q, k, v, table, lens):
                xqa(
                    q,
                    k,
                    v,
                    table,
                    lens,
                    out,
                    workspace,
                    sem,
                    heads,
                    page,
                    kv_layout="HND",
                    enable_pdl=False,
                )

            run(q, k, v, table, lens)
            max_error = 0.0
            for bi in range(batch):
                lo, hi = bi * length // page, (bi + 1) * length // page
                ki = k[lo:hi].permute(1, 0, 2, 3).reshape(heads, length, dim).float()
                vi = v[lo:hi].permute(1, 0, 2, 3).reshape(heads, length, dim).float()
                qi = q[bi, 0].reshape(heads, qheads // heads, dim).float()
                ref = (
                    torch.softmax(qi @ ki.transpose(-1, -2) / math.sqrt(dim), -1) @ vi
                ).reshape_as(out[bi])
                torch.testing.assert_close(out[bi].float(), ref, rtol=0.03, atol=0.004)
                max_error = max(max_error, (out[bi].float() - ref).abs().max().item())
            del ki, vi, qi, ref
            for repeat in range(args.rounds):
                times = bench_gpu_time(
                    run,
                    input_args=(q, k, v, table, lens),
                    dry_run_iters=5,
                    repeat_iters=30,
                    enable_cupti=True,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                )
                row = dict(
                    batch=batch,
                    heads=heads,
                    qheads=qheads,
                    dim=dim,
                    length=length,
                    capacity=65536,
                    round=repeat,
                    median_us=statistics.median(times) * 1000,
                    samples_ms=times,
                    max_abs=max_error,
                    gpu=torch.cuda.get_device_name(),
                    sms=torch.cuda.get_device_properties(0).multi_processor_count,
                )
                report.write(json.dumps(row) + "\n")
                print({k: v for k, v in row.items() if k != "samples_ms"}, flush=True)
            del q, k, v, table, lens
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
