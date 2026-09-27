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
"""Benchmark the SM120 FP8 MiniMax-H3 packed-varlen self-attention.

MiniMax-H3 (video DiT) runs one non-causal 56-head, head_dim-128 self-attention per block over a
packed token stream ``[T, 56, 128]`` with ``int32`` ``cu_seqlens`` segment bounds.  On GB202
(RTX 5090 / RTX PRO 6000 Blackwell) the tensor pipe is the bound, so
``flashinfer.diffusion_ops.minimax_h3_sm120_varlen_attention_nvfp4`` (experimental) follows the
SageAttention3 FP4 recipe: Q / K / V and the probabilities are NVFP4 (E2M1 + UE4M3 block scales)
and both the scores and the value product run block-scaled ``mma.sync kind::mxf4nvf4`` (FP32
softmax, per-segment K mean and per-128-row Q block mean removal), in three PDL-chained launches.  This
script times that operator against the shipped FP8 operator
(``minimax_h3_sm120_varlen_attention_fp8``) and the FlashInfer BF16 ragged routes (``fa2``,
``cudnn`` and ``auto``) with ``bench_gpu_time`` (CUPTI, cold L2), reports the error of every
backend against an FP32 exact-softmax oracle (the NVFP4 route is judged at the FP4 tolerance
``atol = 1.0, rtol = 0.1``, the FP8 operator at ``0.1``), and the peak extra device memory each
backend needs.

Usage::

    python benchmarks/bench_minimax_h3_sm120_nvfp4_varlen_attention.py --suite short
    python benchmarks/bench_minimax_h3_sm120_nvfp4_varlen_attention.py --shapes center_33472 seg4_33472
    python benchmarks/bench_minimax_h3_sm120_nvfp4_varlen_attention.py --suite all --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.diffusion_ops import (
    minimax_h3_sm120_varlen_attention_fp8,
    minimax_h3_sm120_varlen_attention_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    workspace_bytes_nvfp4,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    workspace_bytes,
)
from flashinfer.testing.utils import bench_gpu_time

FP4_ATOL, FP4_RTOL = 1.0, 0.1
FP8_ATOL, FP8_RTOL = 0.1, 0.1
BF16_ATOL, BF16_RTOL = 1e-2, 2e-2
CAKE_ROUTES = ("cake_nvfp4", "cake_fp8")

# label -> cu_seqlens. The representative MiniMax-H3 request sizes (single segment) plus
# multi-segment and ragged packed streams.
SHAPES = {
    "center_33472": (0, 33472),
    "center_38592": (0, 38592),
    "center_48768": (0, 48768),
    "center_58944": (0, 58944),
    "center_74240": (0, 74240),
    "center_109952": (0, 109952),
    "tail_33471": (0, 33471),
    "tail_33473": (0, 33473),
    "seg4_33472": (0, 8368, 16736, 25104, 33472),
    "ragged_4824": (0, 257, 4567, 4824),
    "empty_500": (0, 0, 129, 129, 500),
}
SUITES = {
    "short": ["center_33472", "seg4_33472", "ragged_4824"],
    "centers": [k for k in SHAPES if k.startswith("center_")],
    "all": list(SHAPES),
}
FI_ROUTES = ("fa2", "cudnn", "auto")
_FI_WORKSPACE = 512 * 1024 * 1024


def synthetic_inputs(tokens: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(
            tokens,
            MINIMAX_H3_NUM_HEADS,
            MINIMAX_H3_HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )


def fp32_oracle(q, k, v, cu_seqlens, query_chunk: int = 1024) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for a, b in zip(cu_seqlens, cu_seqlens[1:], strict=False):
            if b == a:
                continue
            for h in range(
                q.shape[1]
            ):  # per head: the FP32 score block stays < 0.5 GiB
                kf = k[a:b, h].float()
                vf = v[a:b, h].float()
                for start in range(a, b, query_chunk):
                    stop = min(start + query_chunk, b)
                    scores = (
                        torch.matmul(q[start:stop, h].float(), kf.transpose(0, 1))
                        * scale
                    )
                    out[start:stop, h] = torch.matmul(torch.softmax(scores, dim=-1), vf)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def attention_flops(cu_seqlens) -> int:
    return sum(
        4 * (b - a) ** 2 * MINIMAX_H3_HEAD_DIM * MINIMAX_H3_NUM_HEADS
        for a, b in zip(cu_seqlens, cu_seqlens[1:], strict=False)
    )


def error_stats(actual, expected, atol, rtol) -> dict:
    diff = (actual.float() - expected).abs()
    return {
        "max_abs_err": float(diff.max()) if diff.numel() else 0.0,
        "rel_l2_err": float(diff.norm() / expected.norm().clamp_min(1e-6))
        if diff.numel()
        else 0.0,
        "outside_tolerance": int((diff > atol + rtol * expected.abs()).sum()),
        "finite": bool(torch.isfinite(actual.float()).all()),
    }


class FlashInferRoute:
    def __init__(self, backend: str, q, k, v, cu: torch.Tensor):
        self.q, self.k, self.v = q, k, v
        self.out = torch.empty_like(q)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(_FI_WORKSPACE, dtype=torch.uint8, device=q.device),
            kv_layout="NHD",
            backend=backend,
        )
        self.wrapper.plan(
            cu,
            cu,
            MINIMAX_H3_NUM_HEADS,
            MINIMAX_H3_NUM_HEADS,
            MINIMAX_H3_HEAD_DIM,
            causal=False,
            sm_scale=1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM),
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=q.dtype,
        )
        self.resolved = getattr(self.wrapper, "_backend", backend)

    def run(self):
        result = self.wrapper.run(self.q, self.k, self.v, out=self.out)
        return self.out if result is None else result


def _measure(fn, bench_ms: float) -> float:
    times = bench_gpu_time(
        fn,
        dry_run_time_ms=50,
        repeat_time_ms=int(bench_ms),
        enable_cupti=True,
        cold_l2_cache=True,
    )
    return float(np.median(times))


def bench_shape(
    label: str, cu_seqlens, backends, bench_ms: float, device: torch.device
) -> dict:
    tokens = cu_seqlens[-1]
    q, k, v = synthetic_inputs(tokens, seed=tokens, device=device)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    expected = fp32_oracle(q, k, v, cu_seqlens)
    flops = attention_flops(cu_seqlens)
    row = {
        "label": label,
        "cu_seqlens": list(cu_seqlens),
        "tokens": tokens,
        "useful_flops": flops,
        "backends": {},
    }
    for backend in backends:
        entry: dict = {}
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            if backend == "cake_nvfp4":
                fn = lambda: minimax_h3_sm120_varlen_attention_nvfp4(
                    q, k, v, cu, cu_seqlens_host=cu_seqlens
                )  # noqa: E731
                out = fn()
                torch.cuda.synchronize()
                entry["error"] = error_stats(out, expected, FP4_ATOL, FP4_RTOL)
                entry["workspace_bytes"] = workspace_bytes_nvfp4(
                    tokens, MINIMAX_H3_NUM_HEADS, len(cu_seqlens) - 1
                )
            elif backend == "cake_fp8":
                fn = lambda: minimax_h3_sm120_varlen_attention_fp8(
                    q, k, v, cu, cu_seqlens_host=cu_seqlens
                )  # noqa: E731
                out = fn()
                torch.cuda.synchronize()
                entry["error"] = error_stats(out, expected, FP8_ATOL, FP8_RTOL)
                entry["workspace_bytes"] = workspace_bytes(
                    tokens, MINIMAX_H3_NUM_HEADS, len(cu_seqlens) - 1
                )
            else:
                route = FlashInferRoute(backend, q, k, v, cu)
                fn = route.run
                out = fn()
                torch.cuda.synchronize()
                entry["resolved_backend"] = route.resolved
                entry["error"] = error_stats(out, expected, BF16_ATOL, BF16_RTOL)
            entry["peak_extra_bytes"] = int(
                torch.cuda.max_memory_allocated(device) - before
            )
            if not entry["error"]["finite"]:
                entry["status"] = "non-finite output"
                continue
            entry["median_ms"] = _measure(fn, bench_ms)
            entry["tflops"] = flops / entry["median_ms"] / 1e9
            entry["status"] = "ok"
        except torch.OutOfMemoryError as exc:
            entry["status"] = f"memory_limited: {str(exc)[:120]}"
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - a missing route is recorded, not fatal
            entry["status"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        finally:
            row["backends"][backend] = entry
            torch.cuda.empty_cache()
    fi_ok = {
        b: e["median_ms"]
        for b, e in row["backends"].items()
        if b not in CAKE_ROUTES and e.get("status") == "ok"
    }
    cake = row["backends"].get("cake_nvfp4", {})
    fp8 = row["backends"].get("cake_fp8", {})
    if fi_ok and cake.get("status") == "ok":
        best = min(fi_ok, key=fi_ok.get)
        row["fastest_flashinfer_route"] = best
        row["speedup_vs_fastest_flashinfer"] = fi_ok[best] / cake["median_ms"]
    if cake.get("status") == "ok" and fp8.get("status") == "ok":
        row["speedup_vs_fp8"] = fp8["median_ms"] / cake["median_ms"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="short")
    parser.add_argument(
        "--shapes", nargs="*", default=None, help="shape labels (override --suite)"
    )
    parser.add_argument("--backends", nargs="*", default=[*CAKE_ROUTES, *FI_ROUTES])
    parser.add_argument("--bench-ms", type=float, default=300.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    labels = args.shapes or SUITES[args.suite]
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(device)}  backends={args.backends}")
    for label in labels:
        row = bench_shape(label, SHAPES[label], args.backends, args.bench_ms, device)
        rows.append(row)
        parts = []
        for backend, entry in row["backends"].items():
            if entry.get("status") == "ok":
                parts.append(
                    f"{backend}={entry['median_ms']:.3f}ms ({entry['tflops']:.0f} TF, relL2 {entry['error']['rel_l2_err']:.3f})"
                )
            else:
                parts.append(f"{backend}: {entry.get('status')}")
        speedup = row.get("speedup_vs_fastest_flashinfer")
        tail = (
            f"  speedup vs {row.get('fastest_flashinfer_route')} = {speedup:.3f}x"
            if speedup
            else ""
        )
        if row.get("speedup_vs_fp8"):
            tail += f"  vs cake_fp8 = {row['speedup_vs_fp8']:.3f}x"
        print(
            f"{label:16s} T={row['tokens']:7d}  " + "  ".join(parts) + tail, flush=True
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "timestamp": time.time(),
                    "bench_ms": args.bench_ms,
                    "rows": rows,
                },
                f,
                indent=1,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
