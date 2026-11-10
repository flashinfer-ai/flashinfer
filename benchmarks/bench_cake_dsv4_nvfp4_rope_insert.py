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

"""Paired SM120 DeepSeek-V4 NVFP4 cache-writer benchmark: fused RoPE + quantize + insert vs the unfused path.

Both arms write the same slots of the same packed NVFP4 pool from the same
inputs.  The fused arm is one public call
(``cake_dsv4_nvfp4_rope_quantize_insert`` for the sliding-window pool,
``cake_dsv4_nvfp4_kv_rope_quantize_insert`` for the compressed pool); the
unfused arm is the torch-level GPT-J RoPE (fp32 math, BF16 result), the
zero-padded query copy and ``nvfp4_quantize_append_sparse_mla_cache`` -- the
per-layer write path a serving stack runs without the fused writers.  Rows
are (entry, tokens, padded heads | compress ratio, page size); timings are
medians over the whole call, including the fused arm's ``q_out`` allocation.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from flashinfer.mla import (
    cake_dsv4_nvfp4_kv_rope_quantize_insert,
    cake_dsv4_nvfp4_rope_insert_format_info,
    cake_dsv4_nvfp4_rope_quantize_insert,
    nvfp4_quantize_append_sparse_mla_cache,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm12x_supported

_D = 512
_D_NOPE = 448
_D_ROPE = 64
_BYTES = 384
_MAX_POS = 65536


def _median_us(fn, warmup_ms: int, measure_ms: int, cupti: bool) -> float:
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(
        fn,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=measure_ms,
        enable_cupti=cupti,
    )
    return float(np.median(measurements)) * 1e3


def _cos_sin_cache(theta: float = 1e4) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, _D_ROPE, 2, dtype=torch.float32, device="cuda") / _D_ROPE)
    )
    freqs = torch.outer(
        torch.arange(_MAX_POS, dtype=torch.float32, device="cuda"), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _rope_bf16(x: torch.Tensor, cos_sin_rows: torch.Tensor) -> torch.Tensor:
    """Torch GPT-J RoPE of the last 64 dims (fp32 math), BF16 result -- the unfused arm's rotation."""

    xf = x.float()
    cos = cos_sin_rows[..., :32]
    sin = cos_sin_rows[..., 32:]
    rope = xf[..., _D_NOPE:].reshape(*xf.shape[:-1], 32, 2)
    x_even, x_odd = rope[..., 0], rope[..., 1]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    ).reshape(*xf.shape[:-1], _D_ROPE)
    return torch.cat((xf[..., :_D_NOPE], rotated), dim=-1).to(torch.bfloat16)


def _pool(num_tokens: int, page_size: int) -> torch.Tensor:
    num_pages = -(-num_tokens // page_size) + 1
    return torch.zeros(
        num_pages, 1, page_size, _BYTES, dtype=torch.uint8, device="cuda"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=(1, 8, 64, 384, 4096, 16384)
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--q-head-padded", type=int, nargs="+", default=(8, 16))
    parser.add_argument(
        "--unfused-filter-boundary",
        action="store_true",
        help="kv entry, ratio 2: the unfused baseline gathers the boundary rows before RoPE / quantize "
        "(host sync, not CUDA-graph capturable) instead of masking the slots like the pre-fix vLLM path",
    )
    parser.add_argument("--compress-ratio", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--swa-page-size", type=int, default=256)
    parser.add_argument("--compressed-page-size", type=int, default=128)
    parser.add_argument("--entries", nargs="+", default=("qkv", "kv"))
    parser.add_argument("--cupti", action="store_true", help="time with CUPTI")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measure-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not is_sm12x_supported(torch.device("cuda")):
        raise SystemExit("SM120/SM121 required")
    if not cake_dsv4_nvfp4_rope_insert_format_info()["kernels_available"]:
        raise SystemExit(
            "the Cake SM120 NVFP4 RoPE-insert kernels are not exported into this tree"
        )
    torch.manual_seed(args.seed)
    cos_sin = _cos_sin_cache()
    print("entry,tokens,heads_padded|ratio,page,fused_us,unfused_us,speedup")
    for num_tokens in args.num_tokens:
        kv = (torch.randn(num_tokens, _D, device="cuda") * 3.0).to(torch.bfloat16)
        positions = torch.randint(0, _MAX_POS, (num_tokens,), device="cuda")
        if "qkv" in args.entries:
            page_size = args.swa_page_size
            cache = _pool(num_tokens, page_size)
            slots = torch.randperm(cache.shape[0] * page_size, device="cuda")[
                :num_tokens
            ]
            q = (torch.randn(num_tokens, args.num_heads, _D, device="cuda") * 2.0).to(
                torch.bfloat16
            )
            for q_head_padded in args.q_head_padded:
                if q_head_padded < args.num_heads:
                    continue

                def fused() -> None:
                    cake_dsv4_nvfp4_rope_quantize_insert(
                        q,
                        kv,
                        cache,
                        slots,
                        positions,
                        cos_sin,
                        q_head_padded=q_head_padded,
                    )

                def unfused() -> None:
                    rows = cos_sin[positions]
                    q_out = torch.zeros(
                        num_tokens,
                        q_head_padded,
                        _D,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    q_out[:, : args.num_heads] = _rope_bf16(q, rows.unsqueeze(1))
                    nvfp4_quantize_append_sparse_mla_cache(
                        _rope_bf16(kv, rows), slots, cache
                    )

                fused_us = _median_us(
                    fused, args.warmup_ms, args.measure_ms, args.cupti
                )
                unfused_us = _median_us(
                    unfused, args.warmup_ms, args.measure_ms, args.cupti
                )
                print(
                    f"qkv,{num_tokens},{q_head_padded},{page_size},{fused_us:.2f},"
                    f"{unfused_us:.2f},{unfused_us / fused_us:.3f}"
                )
        if "kv" in args.entries:
            page_size = args.compressed_page_size
            cache = _pool(num_tokens, page_size)
            slots = torch.randperm(cache.shape[0] * page_size, device="cuda")[
                :num_tokens
            ]
            for ratio in args.compress_ratio:

                def fused() -> None:
                    cake_dsv4_nvfp4_kv_rope_quantize_insert(
                        kv, cache, slots, positions, cos_sin, compress_ratio=ratio
                    )

                def unfused() -> None:
                    boundary = (positions + 1) % ratio == 0
                    rows = cos_sin[positions // ratio * ratio]
                    if args.unfused_filter_boundary:
                        # Rope / quantize only the boundary rows.  The gather needs the row count on the host
                        # (``nonzero`` synchronises), so this form is not CUDA-graph capturable; the default
                        # reproduces the pre-fix vLLM path, which keeps every row and masks the slots instead.
                        idx = boundary.nonzero().squeeze(1)
                        nvfp4_quantize_append_sparse_mla_cache(
                            _rope_bf16(kv[idx], rows[idx]), slots[idx], cache
                        )
                    else:
                        nvfp4_quantize_append_sparse_mla_cache(
                            _rope_bf16(kv, rows),
                            torch.where(boundary, slots, torch.full_like(slots, -1)),
                            cache,
                        )

                fused_us = _median_us(
                    fused, args.warmup_ms, args.measure_ms, args.cupti
                )
                unfused_us = _median_us(
                    unfused, args.warmup_ms, args.measure_ms, args.cupti
                )
                print(
                    f"kv,{num_tokens},{ratio},{page_size},{fused_us:.2f},"
                    f"{unfused_us:.2f},{unfused_us / fused_us:.3f}"
                )


if __name__ == "__main__":
    main()
