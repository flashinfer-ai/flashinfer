# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM12x MXFP8 native tail versus CUTLASS and pre-padded b12x.

python benchmarks/bench_mm_mxfp8_b12x_tail.py --output tail.json

Times are GEMM microbenchmarks: padding/quantization happens before timing.
By default, CUDA graphs rotate tensor copies to exceed L2 capacity. Use
--warm-l2 for a cache-resident comparison or --cupti for CUPTI timing.
--baseline-only also runs on the parent revision (copy this script there).
"""

import argparse
import importlib.metadata
import json
import statistics
import subprocess
from pathlib import Path

import torch

from flashinfer import SfLayout, mm_mxfp8, mxfp8_quantize
from flashinfer.testing import bench_gpu_time


def operand(rows, k):
    x, sf = mxfp8_quantize(
        torch.randn(rows, k, device="cuda", dtype=torch.bfloat16),
        sf_swizzle_layout=SfLayout.layout_128x4,
    )
    padded_k = (k + 127) // 128 * 128
    padded = torch.zeros(rows, padded_k, device="cuda", dtype=x.dtype)
    padded.view(torch.uint8)[:, :k].copy_(x.view(torch.uint8))
    return x, padded, sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 6, 128, 512, 2048])
    parser.add_argument("--k", type=int, nargs="+", default=[544, 576, 608, 640])
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--cupti", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("This benchmark requires an SM120/SM121 GPU")
    torch.manual_seed(5175)
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "diff", "--name-only"])),
        "timing": "cupti" if args.cupti else "cuda_graph",
        "cold_l2": not args.warm_l2,
        "rounds": args.rounds,
        "iterations_per_round": args.iters,
    }
    print(json.dumps(metadata), flush=True)
    print("M,N,K,CUTLASS_us,padded_b12x_us,native_b12x_us,CUTLASS/native,padded/native")
    results = []
    for m in args.m:
        for k in args.k:
            a, ap, sa = operand(m, k)
            b, bp, sb = operand(args.n, k)
            cases = {
                "cutlass": (a, b, "cutlass"),
                "padded_b12x": (ap, bp, "b12x"),
            }
            if not args.baseline_only or k % 128 == 0:
                cases["native_b12x"] = (a, b, "b12x")
            outputs = {}
            for name, (x, w, backend) in cases.items():
                outputs[name] = mm_mxfp8(x, w.T, sa, sb, backend=backend)
            if "native_b12x" in outputs:
                torch.testing.assert_close(
                    outputs["native_b12x"], outputs["padded_b12x"], rtol=0, atol=0
                )
            # Cross-backend FP32 reduction details can differ at output
            # rounding boundaries; exact parity is asserted within b12x above.
            torch.testing.assert_close(
                outputs["cutlass"], outputs["padded_b12x"], rtol=1e-2, atol=1e-2
            )
            samples = {name: [] for name in cases}
            names = list(cases)
            for round_idx in range(args.rounds):
                # Counterbalance backend order across rounds.
                order = (
                    names[round_idx % len(names) :] + names[: round_idx % len(names)]
                )
                if round_idx % 2:
                    order.reverse()
                for name in order:
                    x, w, backend = cases[name]

                    def run(x, w, sa, sb, out, backend=backend):
                        return mm_mxfp8(x, w.T, sa, sb, out=out, backend=backend)

                    times = bench_gpu_time(
                        run,
                        input_args=(x, w, sa, sb, outputs[name]),
                        dry_run_iters=10,
                        repeat_iters=args.iters,
                        use_cuda_graph=not args.cupti,
                        enable_cupti=args.cupti,
                        cold_l2_cache=not args.warm_l2,
                    )
                    samples[name].extend(t * 1000 for t in times)
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            row = {
                "m": m,
                "n": args.n,
                "k": k,
                "median_us": medians,
                "samples_us": samples,
            }
            results.append(row)
            native = medians.get("native_b12x")
            print(
                f"{m},{args.n},{k},{medians['cutlass']:.3f},{medians['padded_b12x']:.3f},"
                + (
                    f"{native:.3f},{medians['cutlass'] / native:.3f},{medians['padded_b12x'] / native:.3f}"
                    if native
                    else "N/A,N/A,N/A"
                ),
                flush=True,
            )
            if args.output:
                args.output.write_text(
                    json.dumps({"environment": metadata, "results": results}, indent=2)
                    + "\n"
                )


if __name__ == "__main__":
    main()
