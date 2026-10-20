#!/usr/bin/env python3
"""
Copyright (c) 2024 by FlashInfer team.

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

"""Benchmark of the FP8 per-(token, head) quantize-and-append paged KV cache
writer.

Measures ``flashinfer.fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping``:
quantize ``nnz`` tokens x ``num_kv_heads`` of K and V (fp16/bf16, one scale
per (token, head)) and write the FP8 data + float32 scale into a paged KV
cache.

Layout: K and V are separate paged tensors, last dim = ``head_dim + 16`` bytes
= [head_dim fp8 | 4B fp32 scale | 12B pad] (NHD/HND). The scale is a strided
float32 view of the same storage (``slot[..., hd:hd+4].view(float32)``).

Both sides move the same bytes per (token, head): read 2*head_dim bytes
(fp16/bf16), write head_dim bytes (fp8) + 4 bytes (scale).

The startup check validates the written bytes by dequantizing them and
comparing against the original K/V.

Timing: ``bench_gpu_time`` with CUDA events and a cold L2 (2x-L2 memset between
iterations). At small ``nnz`` the per-iteration time is dominated by kernel
launch overhead (~2-5 us), so treat small-nnz numbers as floor, not SOL.

Usage:
    python benchmarks/bench_fp8_token_head_quantize_append.py
    python benchmarks/bench_fp8_token_head_quantize_append.py \\
        --nnz 1 64 4096 --head-dims 64 128 256
    python benchmarks/bench_fp8_token_head_quantize_append.py \\
        --layouts NHD HND --dtypes fp16 bf16 --nnz 4096
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

FP8 = torch.float8_e4m3fn


def fi_quantize_append(key, value, slot_mapping, k_slot, v_slot, k_sf, v_sf, kv_layout):
    flashinfer.fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping(
        key,
        value,
        slot_mapping,
        (k_slot, v_slot),
        (k_sf, v_sf),
        kv_layout=kv_layout,
        kv_data_type=FP8,
    )


def _dequant(data_fp8, sf):
    # data [*, hd] fp8, sf [*, ] f32 -> fp32 [*, hd]
    return data_fp8.to(torch.float32) * sf.unsqueeze(-1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def main():
    ap = argparse.ArgumentParser(
        description="FP8 per-(token, head) quantize-and-append paged KV cache writer"
    )
    ap.add_argument(
        "--nnz",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256, 1024, 4096, 16384, 65536],
    )
    ap.add_argument("--head-dims", type=int, nargs="+", default=[128])
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dtypes", nargs="+", default=["fp16"])
    ap.add_argument("--layouts", nargs="+", default=["NHD"])
    ap.add_argument("--dry-run-ms", type=int, default=25)
    ap.add_argument("--repeat-ms", type=int, default=100)
    ap.add_argument(
        "--cupti", action="store_true", help="CUPTI timing instead of CUDA events"
    )
    ap.add_argument(
        "--no-check", action="store_true", help="skip the startup correctness check"
    )
    args = ap.parse_args()

    device = "cuda:0"
    prop = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {prop.name} (sm_{prop.major}{prop.minor}) | timing: "
        f"{'CUPTI' if args.cupti else 'cuda-event'} + cold L2"
    )

    torch.manual_seed(0)

    if not args.no_check:
        nnz, hd, nkv, page = 64, 128, 8, 16
        k = torch.randn(nnz, nkv, hd, device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        cap = (nnz + page - 1) // page * page
        # Identity slots: token i -> slot i, so slot-major == token order.
        slots = torch.arange(nnz, device=device).to(torch.int32)
        pages = cap // page
        k_slot = torch.zeros(
            pages, page, nkv, hd + 16, dtype=torch.uint8, device=device
        )
        v_slot = torch.zeros_like(k_slot)
        k_sf = k_slot[..., hd : hd + 4].view(torch.float32).squeeze(-1)
        v_sf = v_slot[..., hd : hd + 4].view(torch.float32).squeeze(-1)
        fi_quantize_append(k, v, slots, k_slot, v_slot, k_sf, v_sf, "NHD")
        fi_kd = _dequant(k_slot[..., :hd].view(FP8), k_sf)
        fi_vd = _dequant(v_slot[..., :hd].view(FP8), v_sf)
        ck, cv = _cos(fi_kd, k.float()), _cos(fi_vd, v.float())
        print(f"check: cos(fi_k,k)={ck:.5f} cos(fi_v,v)={cv:.5f}")
        assert min(ck, cv) > 0.999, "correctness check failed"

    nkv, page = args.num_kv_heads, args.page_size
    for kv_layout in args.layouts:
        for dtype_name in args.dtypes:
            in_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
            for hd in args.head_dims:
                print(
                    f"\nhead_dim={hd} nkv={nkv} page={page} {dtype_name} {kv_layout} "
                    f"e4m3 (per tok/head/side: rd {2 * hd}B wr {hd + 4}B)"
                )
                print(f"{'nnz':>8} | {'fi us':>10} | {'fi GB/s':>9}")
                for nnz in args.nnz:
                    cap = max(page, int(nnz * 1.25) + 1)
                    cap = (cap + page - 1) // page * page
                    pages = cap // page
                    slots = torch.randperm(cap, device=device)[:nnz].to(torch.int32)
                    k = torch.randn(nnz, nkv, hd, device=device, dtype=in_dtype)
                    v = torch.randn_like(k)

                    k_slot = torch.zeros(
                        pages,
                        page if kv_layout == "NHD" else nkv,
                        nkv if kv_layout == "NHD" else page,
                        hd + 16,
                        dtype=torch.uint8,
                        device=device,
                    )
                    v_slot = torch.zeros_like(k_slot)
                    k_sf = k_slot[..., hd : hd + 4].view(torch.float32).squeeze(-1)
                    v_sf = v_slot[..., hd : hd + 4].view(torch.float32).squeeze(-1)

                    fn_fi = lambda: fi_quantize_append(
                        k, v, slots, k_slot, v_slot, k_sf, v_sf, kv_layout
                    )
                    t_fi = bench_gpu_time(
                        fn_fi,
                        dry_run_time_ms=args.dry_run_ms,
                        repeat_time_ms=args.repeat_ms,
                        cold_l2_cache=True,
                        enable_cupti=args.cupti,
                    )
                    fi_us = float(np.median(t_fi)) * 1e3

                    bytes_moved = nnz * nkv * 2 * (3 * hd + 4)
                    fi_gbps = bytes_moved / (fi_us * 1e-6) / 1e9
                    print(f"{nnz:>8} | {fi_us:>10.2f} | {fi_gbps:>9.1f}")


if __name__ == "__main__":
    main()
