"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the frozen radix top-k / top-p sampling pipeline against the default
``top_k_top_p_sampling_from_probs`` routes (top_k_first, joint)."""

import argparse

import torch

import flashinfer.sampling as fs
from flashinfer.cake_sampling import (
    cake_sampling_route,
    top_k_top_p_sampling_from_probs,
)
from flashinfer.testing import bench_gpu_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--vocabs", default="32768,128256,151936,262144")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument(
        "--cupti", action="store_true", help="CUPTI kernel time (cupti-python >= 13)"
    )
    args = ap.parse_args()
    print(
        f"{'V':>7} {'B':>4} {'route':>22} {'top_k_first':>12} {'joint':>10} {'cake':>10} {'speedup':>8}"
    )
    for vocab in (int(v) for v in args.vocabs.split(",")):
        for batch in (int(b) for b in args.batches.split(",")):
            g = torch.Generator(device="cuda").manual_seed(536)
            probs = torch.softmax(
                torch.randn(batch, vocab, device="cuda", generator=g), dim=-1
            )
            k, p = args.top_k, args.top_p
            kw = dict(use_cuda_graph=args.cuda_graph, enable_cupti=args.cupti)
            t_first = bench_gpu_time(
                lambda: fs.top_k_top_p_sampling_from_probs(
                    probs, k, p, filter_apply_order="top_k_first"
                ),
                **kw,
            )
            t_joint = bench_gpu_time(
                lambda: fs.top_k_top_p_sampling_from_probs(
                    probs, k, p, filter_apply_order="joint"
                ),
                **kw,
            )
            t_cake = bench_gpu_time(
                lambda: top_k_top_p_sampling_from_probs(probs, k, p), **kw
            )
            med = lambda t: 1000.0 * sorted(t)[len(t) // 2]
            route = cake_sampling_route(probs, k)
            print(
                f"{vocab:>7} {batch:>4} {route:>22} {med(t_first):>10.1f}us {med(t_joint):>8.1f}us {med(t_cake):>8.1f}us {med(t_first) / med(t_cake):>7.2f}x"
            )


if __name__ == "__main__":
    main()
