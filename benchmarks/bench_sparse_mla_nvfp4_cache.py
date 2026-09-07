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

"""Benchmark DeepSeek-V4 NVFP4 sparse-MLA cache pack and append."""

import argparse

import numpy as np
import torch

from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache
from flashinfer.mla._sparse_mla_nvfp4_sm120 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm120a_supported


_D_LATENT = 512
_INPUT_BYTES_PER_TOKEN = _D_LATENT * 2
_CACHE_BYTES_PER_TOKEN = 384


def _median_us(fn, warmup_ms, measure_ms):
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn, dry_run_time_ms=warmup_ms, repeat_time_ms=measure_ms
    )
    return float(np.median(measurements)) * 1e3


def bench_full_page_pack(num_pages, page_size, warmup_ms=100, measure_ms=1000):
    latent_kv = torch.randn(
        num_pages,
        page_size,
        _D_LATENT,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache = torch.empty(
        num_pages,
        1,
        page_size,
        _CACHE_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device="cuda",
    )
    module = get_sparse_mla_nvfp4_sm120_module()

    def fn():
        module.sparse_mla_sm120_nvfp4_quantize_pack(latent_kv, cache)

    latency_us = _median_us(fn, warmup_ms, measure_ms)
    num_tokens = num_pages * page_size
    traffic_bytes = num_tokens * (_INPUT_BYTES_PER_TOKEN + _CACHE_BYTES_PER_TOKEN)
    bandwidth_gbps = traffic_bytes * 1e-3 / latency_us
    return latency_us, bandwidth_gbps


def bench_incremental_append(
    num_pages, page_size, num_tokens, warmup_ms=100, measure_ms=1000
):
    max_tokens = num_pages * page_size
    if num_tokens > max_tokens:
        raise ValueError(f"num_tokens={num_tokens} exceeds cache slots={max_tokens}")
    latent_kv = torch.randn(num_tokens, _D_LATENT, dtype=torch.bfloat16, device="cuda")
    slots = torch.randperm(max_tokens, dtype=torch.int64, device="cuda")[:num_tokens]
    cache = torch.empty(
        num_pages,
        1,
        page_size,
        _CACHE_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device="cuda",
    )

    def fn():
        nvfp4_quantize_append_sparse_mla_cache(latent_kv, slots, cache)

    latency_us = _median_us(fn, warmup_ms, measure_ms)
    traffic_bytes = num_tokens * (_INPUT_BYTES_PER_TOKEN + _CACHE_BYTES_PER_TOKEN)
    bandwidth_gbps = traffic_bytes * 1e-3 / latency_us
    return latency_us, bandwidth_gbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, choices=(2, 64), default=64)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument(
        "--append-tokens", type=int, nargs="+", default=(1, 32, 256, 8192)
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measure-ms", type=int, default=1000)
    args = parser.parse_args()

    if not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("NVFP4 sparse MLA requires SM120/SM121")

    print("operation,page_size,num_tokens,latency_us,effective_bandwidth_gbps")
    latency_us, bandwidth_gbps = bench_full_page_pack(
        args.num_pages, args.page_size, args.warmup_ms, args.measure_ms
    )
    print(
        f"full_pack,{args.page_size},{args.num_pages * args.page_size},"
        f"{latency_us:.3f},{bandwidth_gbps:.3f}"
    )

    for num_tokens in args.append_tokens:
        if num_tokens > args.num_pages * args.page_size:
            continue
        latency_us, bandwidth_gbps = bench_incremental_append(
            args.num_pages,
            args.page_size,
            num_tokens,
            args.warmup_ms,
            args.measure_ms,
        )
        print(
            f"append,{args.page_size},{num_tokens},{latency_us:.3f},"
            f"{bandwidth_gbps:.3f}"
        )


if __name__ == "__main__":
    main()
