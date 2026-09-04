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

"""Paired FP8/NVFP4 DeepSeek-V4 sparse-MLA prefill benchmark for SM120.

Both paths consume the same BF16 Q/KV source tensors and sparse indices. The
NVFP4 headline includes selected-V transpose/requantization plus online Q/P
quantization. The FP8 path is the upstream 32-head MG prefill kernel rather
than the standalone split-K decode kernel.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from bench_sparse_mla_nvfp4_decode import _quantize_fp8_cache
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_nvfp4_sm120 import (
    get_sparse_mla_nvfp4_sm120_module,
)
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm120a_supported


_D_LATENT = 512
_PAGE_SIZE = 64


def _median_us(fn, warmup_ms: int, measure_ms: int) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn, dry_run_time_ms=warmup_ms, repeat_time_ms=measure_ms
    )
    return float(np.median(measurements)) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--topk", type=int, nargs="+", default=(128, 512))
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--extra-topk", type=int, default=0)
    parser.add_argument("--extra-page-size", type=int, choices=(2, 64), default=64)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measure-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help=(
            "warm up, bracket exactly one NVFP4 prefill call with the CUDA "
            "Profiler API, and exit"
        ),
    )
    args = parser.parse_args()

    if not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("NVFP4 sparse MLA requires SM120/SM121")
    if args.num_tokens <= 64:
        raise SystemExit("paired FP8 prefill requires --num-tokens > 64")
    if args.num_heads not in (16, 32, 64, 128):
        raise SystemExit("NVFP4 prefill supports 16/32/64/128 heads")
    if args.profile_once and len(args.topk) != 1:
        raise SystemExit("--profile-once requires exactly one --topk value")

    torch.manual_seed(args.seed)
    kv_bf16 = (
        torch.randn(
            args.num_pages,
            _PAGE_SIZE,
            1,
            _D_LATENT,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(
            args.num_tokens,
            args.num_heads,
            _D_LATENT,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    fp8_cache = _quantize_fp8_cache(kv_bf16)
    nvfp4_cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16.squeeze(2))
    fp8_extra_cache = None
    nvfp4_extra_cache = None
    extra_indices = None
    if args.extra_topk > 0:
        extra_num_pages = max(
            args.num_pages,
            (args.extra_topk + args.extra_page_size - 1) // args.extra_page_size,
        )
        extra_bf16 = (
            torch.randn(
                extra_num_pages,
                args.extra_page_size,
                1,
                _D_LATENT,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 10.0
        ).clamp(-1, 1)
        fp8_extra_cache = _quantize_fp8_cache(extra_bf16)
        nvfp4_extra_cache = nvfp4_quantize_pack_sparse_mla_cache(extra_bf16.squeeze(2))
        extra_indices = torch.randint(
            0,
            extra_num_pages * args.extra_page_size,
            (args.num_tokens, args.extra_topk),
            dtype=torch.int32,
            device="cuda",
        )
    nvfp4_module = get_sparse_mla_nvfp4_sm120_module()
    fp8_runner = _SparseMLAPagedAttentionRunner(
        max_num_tokens=args.num_tokens,
        max_num_heads=args.num_heads,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    sm_scale = _D_LATENT**-0.5

    print(
        "topk,extra_topk,extra_page_size,fp8_prefill_us,"
        "nvfp4_streaming_us,"
        "speedup_pct,mae,max_abs,cosine_mean,lse_max_abs"
    )
    for topk in args.topk:
        if topk not in (128, 512):
            raise ValueError("the initial NVFP4 prefill kernel supports topk 128/512")
        indices = torch.randint(
            0,
            args.num_pages * _PAGE_SIZE,
            (args.num_tokens, topk),
            dtype=torch.int32,
            device="cuda",
        )
        fp8_out = torch.empty_like(q)
        fp8_lse = torch.empty(
            args.num_tokens, args.num_heads, dtype=torch.float32, device="cuda"
        )
        nv_out = torch.empty_like(q)
        nv_lse = torch.empty_like(fp8_lse)

        def run_fp8() -> None:
            fp8_runner.run(
                q,
                fp8_cache,
                indices,
                fp8_out,
                sm_scale,
                extra_kv_cache=fp8_extra_cache,
                extra_indices=extra_indices,
                out_lse=fp8_lse,
            )

        def run_nv() -> None:
            nvfp4_module.sparse_mla_sm120_nvfp4_prefill(
                q,
                nvfp4_cache,
                indices,
                nv_out,
                nv_lse,
                sm_scale,
                None,
                None,
                nvfp4_extra_cache,
                extra_indices,
                None,
            )

        run_nv()
        torch.cuda.synchronize()
        if args.profile_once:
            torch.cuda.cudart().cudaProfilerStart()
            run_nv()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(
                "profiled_nvfp4_calls=1,"
                f"tokens={args.num_tokens},heads={args.num_heads},"
                f"topk={topk},extra_topk={args.extra_topk}"
            )
            return
        fp8_us = _median_us(run_fp8, args.warmup_ms, args.measure_ms)
        inclusive_us = _median_us(run_nv, args.warmup_ms, args.measure_ms)
        run_fp8()
        run_nv()
        torch.cuda.synchronize()
        delta = (nv_out.float() - fp8_out.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            nv_out.float(), fp8_out.float(), dim=-1
        )
        print(
            f"{topk},{args.extra_topk},{args.extra_page_size},"
            f"{fp8_us:.3f},{inclusive_us:.3f},"
            f"{(fp8_us / inclusive_us - 1.0) * 100.0:.2f},"
            f"{delta.mean().item():.6g},{delta.max().item():.6g},"
            f"{cosine.mean().item():.6g},"
            f"{(nv_lse - fp8_lse).abs().max().item():.6g}"
        )


if __name__ == "__main__":
    main()
