# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired SM120 DeepSeek-V4 NVFP4 sparse-MLA decode benchmark: ``backend="sparse"`` vs ``backend="cake"``.

Both arms run through ``SparseMLASm120Wrapper(kv_cache_format="nvfp4")`` with
caller-owned split scratch over the same packed NVFP4 pools, indices, lengths
and sink, so the comparison covers each route's planner plus its stage and
merge launches.  Rows are (tokens, heads, topk, page size[, extra topk @ extra
page size]); the pool holds ``--num-pages`` pages (default 128, L2 resident).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

_D = 512


def _median_us(fn, warmup_ms: int, measure_ms: int) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn, dry_run_time_ms=warmup_ms, repeat_time_ms=measure_ms
    )
    return float(np.median(measurements)) * 1e3


def _pool(num_pages: int, page_size: int) -> torch.Tensor:
    latent = (
        torch.randn(num_pages, page_size, _D, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
    return nvfp4_quantize_pack_sparse_mla_cache(latent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=(1, 8, 32))
    parser.add_argument("--num-heads", type=int, nargs="+", default=(8, 128))
    parser.add_argument("--topk", type=int, nargs="+", default=(128, 512))
    parser.add_argument("--page-size", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--extra-topk", type=int, default=0)
    parser.add_argument("--extra-page-size", type=int, default=64)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--with-lengths-sink", action="store_true")
    parser.add_argument("--backends", nargs="+", default=("sparse", "cake"))
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measure-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not is_sm12x_supported(torch.device("cuda")):
        raise SystemExit("SM120/SM121 required")
    torch.manual_seed(args.seed)
    sm_scale = _D**-0.5
    print("tokens,heads,topk,page,extra,backend,median_us,speedup_vs_first")
    for page_size in args.page_size:
        cache = _pool(args.num_pages, page_size)
        extra_cache = (
            _pool(
                max(args.num_pages, -(-args.extra_topk // args.extra_page_size)),
                args.extra_page_size,
            )
            if args.extra_topk
            else None
        )
        for num_tokens in args.num_tokens:
            for num_heads in args.num_heads:
                for topk in args.topk:
                    q = (
                        torch.randn(
                            num_tokens,
                            num_heads,
                            _D,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        / 10.0
                    ).clamp(-1, 1)
                    indices = torch.randint(
                        0,
                        args.num_pages * page_size,
                        (num_tokens, topk),
                        dtype=torch.int32,
                        device="cuda",
                    )
                    extra_indices = (
                        torch.randint(
                            0,
                            extra_cache.shape[0] * args.extra_page_size,
                            (num_tokens, args.extra_topk),
                            dtype=torch.int32,
                            device="cuda",
                        )
                        if extra_cache is not None
                        else None
                    )
                    lengths = (
                        torch.full(
                            (num_tokens,), topk, dtype=torch.int32, device="cuda"
                        )
                        if args.with_lengths_sink
                        else None
                    )
                    extra_lengths = (
                        torch.full(
                            (num_tokens,),
                            args.extra_topk,
                            dtype=torch.int32,
                            device="cuda",
                        )
                        if args.with_lengths_sink and extra_cache is not None
                        else None
                    )
                    sink = (
                        torch.zeros(num_heads, dtype=torch.float32, device="cuda")
                        if args.with_lengths_sink
                        else None
                    )
                    splits = (topk + 63) // 64 + (args.extra_topk + 63) // 64
                    mid_out = torch.empty(
                        num_tokens,
                        num_heads,
                        splits,
                        _D,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    mid_lse = torch.empty(
                        num_tokens,
                        num_heads,
                        splits,
                        dtype=torch.float32,
                        device="cuda",
                    )
                    output = torch.empty_like(q)
                    out_lse = torch.empty(
                        num_tokens, num_heads, dtype=torch.float32, device="cuda"
                    )
                    first = None
                    for backend in args.backends:
                        try:
                            runner = flashinfer.mla.SparseMLASm120Wrapper(
                                kv_cache_format="nvfp4",
                                backend=backend,
                                device=q.device,
                            )

                            def run() -> None:
                                runner.run(
                                    q,
                                    cache,
                                    indices,
                                    output,
                                    sm_scale,
                                    topk_length=lengths,
                                    attn_sink=sink,
                                    extra_kv_cache=extra_cache,
                                    extra_indices=extra_indices,
                                    extra_topk_length=extra_lengths,
                                    out_lse=out_lse,
                                    mid_out=mid_out,
                                    mid_lse=mid_lse,
                                )

                            median = _median_us(run, args.warmup_ms, args.measure_ms)
                        except (
                            ValueError
                        ) as error:  # e.g. a shape outside one backend's envelope
                            print(
                                f"{num_tokens},{num_heads},{topk},{page_size},{args.extra_topk}@{args.extra_page_size},{backend},skip,{error}"
                            )
                            continue
                        first = median if first is None else first
                        print(
                            f"{num_tokens},{num_heads},{topk},{page_size},{args.extra_topk}@{args.extra_page_size},{backend},{median:.2f},{first / median:.3f}"
                        )


if __name__ == "__main__":
    main()
