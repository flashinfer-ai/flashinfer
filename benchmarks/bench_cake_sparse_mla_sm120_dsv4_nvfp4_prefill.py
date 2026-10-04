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

"""Paired SM120 DeepSeek-V4 NVFP4 sparse-MLA **prefill** benchmark: ``backend="sparse"`` vs ``backend="cake"``.

Pool model (one prefill batch): ``requests = ceil(T / 2048)`` sequences, each
with its own 65536-token window of the packed NVFP4 pool; token ``t`` of
request ``r`` draws its ``topk`` candidates from request ``r``'s window
without replacement and unsorted (a random permutation prefix), so the
gathers are spread over a working set far beyond L2.  The pool holds
``max(requests * 65536, 2^20)`` tokens (>= 384 MB of 384-byte rows).

Rows: ``T in {128, 512, 2048, 8192} x H in {16, 32, 64, 128} x K in {128, 512}``,
page size 64 (override with the flags).  Arms:

* ``sparse`` -- ``SparseMLASm120Wrapper(kv_cache_format="nvfp4", backend="sparse")``;
* ``cake`` -- the same wrapper with ``backend="cake"`` (decode / prefill crossover);
* ``cake-prefill`` -- the low-level Cake prefill entry (``--head-tiles`` /
  ``--persistent`` override the planner);
* ``cake-decode`` -- the low-level Cake split decode with caller-owned scratch.

Each row prints the GPU median (CUPTI-timed, cold L2 between repetitions), the
gathered bytes per second (``T * (topk + extra_topk) * 384 B``) and the
speedup against the first arm.
"""

from __future__ import annotations

import argparse
from typing import Callable, Optional

import numpy as np
import torch

import flashinfer
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4 import (
    cake_sparse_mla_sm120_dsv4_nvfp4_decode,
    cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill,
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill,
    cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

_D = 512
_BYTES = 384
_TOKENS_PER_REQUEST = 2048
_WINDOW = 65536
_MIN_POOL_TOKENS = 1 << 20  # 384 MiB of packed rows
_POOL_SLAB_PAGES = 2048


def _median_us(fn: Callable[[], None], warmup_ms: int, measure_ms: int) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn, dry_run_time_ms=warmup_ms, repeat_time_ms=measure_ms
    )
    return float(np.median(measurements)) * 1e3


def _pool(num_tokens_in_pool: int, page_size: int, seed: int) -> torch.Tensor:
    """Packed NVFP4 pool of ``num_tokens_in_pool`` rows, quantized slab by slab."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = -(-num_tokens_in_pool // page_size)
    slabs = []
    for start in range(0, num_pages, _POOL_SLAB_PAGES):
        pages = min(_POOL_SLAB_PAGES, num_pages - start)
        latent = (
            torch.randn(
                pages,
                page_size,
                _D,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            / 10.0
        ).clamp(-1, 1)
        slabs.append(nvfp4_quantize_pack_sparse_mla_cache(latent))
        del latent
    return torch.cat(slabs, dim=0) if len(slabs) > 1 else slabs[0]


def _request_indices(
    num_tokens: int, topk: int, pool_tokens: int, seed: int, row_batch: int = 512
) -> torch.Tensor:
    """``[T, topk]`` int32: per token an unsorted sample without replacement of its request's window."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    requests = -(-num_tokens // _TOKENS_PER_REQUEST)
    assert requests * _WINDOW <= pool_tokens
    rows = []
    for start in range(0, num_tokens, row_batch):
        count = min(row_batch, num_tokens - start)
        keys = torch.rand(count, _WINDOW, device="cuda", generator=generator)
        local = torch.argsort(keys, dim=1)[:, :topk]
        request = (
            torch.arange(start, start + count, device="cuda") // _TOKENS_PER_REQUEST
        )
        rows.append((local + (request * _WINDOW).unsqueeze(1)).to(torch.int32))
        del keys
    return torch.cat(rows, dim=0).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=(128, 512, 2048, 8192)
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=(16, 32, 64, 128))
    parser.add_argument("--topk", type=int, nargs="+", default=(128, 512))
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--extra-topk", type=int, default=0, help="second (compressed) cache candidates"
    )
    parser.add_argument("--extra-page-size", type=int, default=64)
    parser.add_argument("--with-lengths-sink", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=("sparse", "cake"),
        choices=("sparse", "cake", "cake-prefill", "cake-decode"),
    )
    parser.add_argument(
        "--head-tiles",
        type=int,
        default=None,
        help="cake-prefill: head tiles per CTA (1, 2, 4); default = planner",
    )
    parser.add_argument(
        "--persistent",
        choices=("auto", "on", "off"),
        default="auto",
        help="cake-prefill: persistent CTAs; default = planner",
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measure-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not is_sm12x_supported(torch.device("cuda")):
        raise SystemExit("SM120/SM121 required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sm_scale = _D**-0.5
    max_requests = -(-max(args.num_tokens) // _TOKENS_PER_REQUEST)
    pool_tokens = max(max_requests * _WINDOW, _MIN_POOL_TOKENS)
    cache = _pool(pool_tokens, args.page_size, args.seed)
    extra_cache = (
        _pool(pool_tokens, args.extra_page_size, args.seed + 1)
        if args.extra_topk
        else None
    )
    print(
        f"# pool {pool_tokens} tokens ({pool_tokens * _BYTES / 2**20:.0f} MiB), page {args.page_size}, "
        f"window {_WINDOW} tokens per {_TOKENS_PER_REQUEST}-token request, {num_sms} SMs"
    )
    print(
        "tokens,heads,topk,extra,backend,plan,median_us,gathered_gbps,speedup_vs_first"
    )
    for num_tokens in args.num_tokens:
        for topk in args.topk:
            indices = _request_indices(num_tokens, topk, pool_tokens, args.seed + 7)
            extra_indices = (
                _request_indices(
                    num_tokens, args.extra_topk, pool_tokens, args.seed + 11
                )
                if extra_cache is not None
                else None
            )
            lengths = (
                torch.full((num_tokens,), topk, dtype=torch.int32, device="cuda")
                if args.with_lengths_sink
                else None
            )
            extra_lengths = (
                torch.full(
                    (num_tokens,), args.extra_topk, dtype=torch.int32, device="cuda"
                )
                if args.with_lengths_sink and extra_cache is not None
                else None
            )
            gathered_bytes = num_tokens * (topk + args.extra_topk) * _BYTES
            for num_heads in args.num_heads:
                q = (
                    torch.randn(
                        num_tokens, num_heads, _D, dtype=torch.bfloat16, device="cuda"
                    )
                    / 10.0
                ).clamp(-1, 1)
                sink = (
                    torch.zeros(num_heads, dtype=torch.float32, device="cuda")
                    if args.with_lengths_sink
                    else None
                )
                output = torch.empty_like(q)
                out_lse = torch.empty(
                    num_tokens, num_heads, dtype=torch.float32, device="cuda"
                )
                chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(
                    topk, args.extra_topk
                )
                mid_out = torch.empty(
                    num_tokens,
                    num_heads,
                    chunks,
                    _D,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                mid_lse = torch.empty(
                    num_tokens, num_heads, chunks, dtype=torch.float32, device="cuda"
                )
                first: Optional[float] = None
                for backend in args.backends:
                    plan = "-"
                    try:
                        if backend in ("sparse", "cake"):
                            runner = flashinfer.mla.SparseMLASm120Wrapper(
                                kv_cache_format="nvfp4", backend=backend, device=device
                            )
                            if backend == "cake":
                                plan = cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel(
                                    num_tokens=num_tokens,
                                    num_heads=num_heads,
                                    topk=topk,
                                    extra_topk=args.extra_topk,
                                    num_sms=num_sms,
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

                        elif backend == "cake-prefill":
                            head_tiles, persistent = (
                                cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill(
                                    num_tokens=num_tokens,
                                    num_heads=num_heads,
                                    topk=topk,
                                    extra_topk=args.extra_topk,
                                    num_sms=num_sms,
                                )
                            )
                            if args.head_tiles is not None:
                                head_tiles = args.head_tiles
                            if args.persistent != "auto":
                                persistent = args.persistent == "on"
                            plan = f"ht{head_tiles}{'-pers' if persistent else ''}"

                            def run() -> None:
                                cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
                                    q,
                                    cache,
                                    indices,
                                    output,
                                    out_lse,
                                    sm_scale,
                                    topk_length=lengths,
                                    attn_sink=sink,
                                    extra_kv_cache=extra_cache,
                                    extra_indices=extra_indices,
                                    extra_topk_length=extra_lengths,
                                    head_tiles=head_tiles,
                                    persistent=persistent,
                                )

                        else:
                            resolved = cake_sparse_mla_sm120_dsv4_nvfp4_decode(
                                q,
                                cache,
                                indices,
                                output,
                                out_lse,
                                sm_scale,
                                topk_length=lengths,
                                attn_sink=sink,
                                extra_kv_cache=extra_cache,
                                extra_indices=extra_indices,
                                extra_topk_length=extra_lengths,
                                mid_out=mid_out,
                                mid_lse=mid_lse,
                            )
                            plan = (
                                f"ht{resolved['head_tiles']}-s{resolved['num_splits']}"
                            )

                            def run() -> None:
                                cake_sparse_mla_sm120_dsv4_nvfp4_decode(
                                    q,
                                    cache,
                                    indices,
                                    output,
                                    out_lse,
                                    sm_scale,
                                    topk_length=lengths,
                                    attn_sink=sink,
                                    extra_kv_cache=extra_cache,
                                    extra_indices=extra_indices,
                                    extra_topk_length=extra_lengths,
                                    mid_out=mid_out,
                                    mid_lse=mid_lse,
                                )

                        median = _median_us(run, args.warmup_ms, args.measure_ms)
                    except ValueError as error:  # a shape outside one arm's envelope
                        print(
                            f"{num_tokens},{num_heads},{topk},{args.extra_topk}@{args.extra_page_size},"
                            f"{backend},{plan},skip,-,{error}"
                        )
                        continue
                    first = median if first is None else first
                    gbps = gathered_bytes / (median * 1e-6) / 1e9
                    print(
                        f"{num_tokens},{num_heads},{topk},{args.extra_topk}@{args.extra_page_size},"
                        f"{backend},{plan},{median:.2f},{gbps:.0f},{first / median:.3f}"
                    )


if __name__ == "__main__":
    main()
