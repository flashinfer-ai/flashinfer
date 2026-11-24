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
"""NVFP4 KV-cache append benchmark (``nvfp4_quantize_append_paged_kv_cache_with_slot_mapping``).

One decode-step write per row: ``nnz`` new tokens, each with ``num_kv_heads``
K and V heads, quantized to NVFP4 and scattered to random slots of an NHD
paged cache.  Timings are medians of CUDA-graph replays.  To compare two
builds, run this script once per build in separate processes and alternate
them a few times; each run prints one line per ``nnz``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from flashinfer.page import nvfp4_quantize_append_paged_kv_cache_with_slot_mapping
from flashinfer.testing.utils import bench_gpu_time

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _median_us(fn, warmup_ms: int, measure_ms: int, cupti: bool) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=measure_ms,
        enable_cupti=cupti,
        use_cuda_graph=True,
    )
    return float(np.median(measurements)) * 1e3


def _bench_one(args: argparse.Namespace, nnz: int) -> float:
    dtype = _DTYPES[args.dtype]
    shape = (nnz, args.num_kv_heads, args.head_dim)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    num_slots = args.num_pages * args.page_size
    slots = torch.randperm(num_slots, device="cuda", dtype=torch.int64)[:nnz]
    packed_shape = (
        args.num_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim // 2,
    )
    scale_shape = (
        args.num_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim // 16,
    )
    k_cache = torch.zeros(packed_shape, device="cuda", dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    k_scales = torch.zeros(scale_shape, device="cuda", dtype=torch.float8_e4m3fn)
    v_scales = torch.zeros_like(k_scales)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    def run() -> None:
        nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            k,
            v,
            slots,
            (k_cache, v_cache),
            (k_scales, v_scales),
            global_scale,
            global_scale,
            kv_layout="NHD",
        )

    return _median_us(run, args.warmup_ms, args.measure_ms, not args.no_cupti)


def main() -> None:
    parser = argparse.ArgumentParser(description="NVFP4 KV-cache append benchmark")
    parser.add_argument(
        "--nnz", type=int, nargs="+", default=[16, 64, 128, 160, 256, 512]
    )
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", type=int, default=4096)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measure-ms", type=int, default=1000)
    parser.add_argument("--no-cupti", action="store_true")
    parser.add_argument("--label", default="", help="free text echoed on every row")
    args = parser.parse_args()
    if args.head_dim % 16:
        raise ValueError(f"head_dim must be a multiple of 16, got {args.head_dim}")

    torch.manual_seed(0)
    print(
        f"# {torch.cuda.get_device_name()} {args.dtype} heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} page_size={args.page_size} NHD"
    )
    for nnz in args.nnz:
        us = _bench_one(args, nnz)
        print(f"label={args.label} nnz={nnz} median_us={us:.4f}", flush=True)


if __name__ == "__main__":
    main()
