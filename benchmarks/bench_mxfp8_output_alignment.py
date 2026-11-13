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

"""MXFP8 output-store regression benchmark (run per checkout)."""

import argparse
import json
import statistics
from pathlib import Path
import torch
from flashinfer import mm_mxfp8, mxfp8_quantize, SfLayout
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(5943)
    cases = [
        (m, n, k) for m in (1, 17, 128, 512) for n, k in ((128, 384), (4096, 4096))
    ]
    if args.tail:
        cases = [
            (m, n, k) for m in (1, 17, 129, 513) for n, k in ((129, 384), (8193, 128))
        ]
    with args.output.open("w", buffering=1) as output:
        for m, n, k in cases:
            a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
            b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16)
            aq, sa = mxfp8_quantize(a, sf_swizzle_layout=SfLayout.layout_128x4)
            bq, sb = mxfp8_quantize(b, sf_swizzle_layout=SfLayout.layout_128x4)
            aq.copy_(a.to(torch.float8_e4m3fn))
            bq.copy_(b.to(torch.float8_e4m3fn))
            sa.fill_(127)
            sb.fill_(127)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

            def run(a, b, sa, sb):
                mm_mxfp8(a, b.T, sa, sb, out=out, backend="b12x")

            run(aq, bq, sa, sb)
            torch.testing.assert_close(
                out, (a.float() @ b.float().T).to(out.dtype), rtol=0, atol=0
            )
            for repeat in range(3):
                times = bench_gpu_time(
                    run,
                    input_args=(aq, bq, sa, sb),
                    dry_run_iters=5,
                    repeat_iters=30,
                    enable_cupti=True,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                )
                row = dict(
                    m=m,
                    n=n,
                    k=k,
                    round=repeat,
                    median_us=statistics.median(times) * 1000,
                    samples_ms=times,
                )
                output.write(json.dumps(row) + "\n")
                print({k: v for k, v in row.items() if k != "samples_ms"}, flush=True)


if __name__ == "__main__":
    main()
