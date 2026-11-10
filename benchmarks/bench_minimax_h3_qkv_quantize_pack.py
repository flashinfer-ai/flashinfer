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
"""Benchmark the MiniMax-H3 one-pass QKV quantize-and-pack helper on SM100a / SM103a.

Compares the prepared fused operation (:class:`flashinfer.MiniMaxH3QkvQuantizePack`)
against the segmented FlashInfer fallback chain a caller runs today when the
projection epilogue cannot emit the communication layout: three strided torch
copies into a destination-major BF16 staging buffer, ``out_sf.zero_()`` for the
scale-tile padding, then ``fp4_quantize`` / ``mxfp8_quantize`` once per
destination.  Both arms write the same caller-owned outputs and are checked for
bitwise equality before timing.  The default suite measures the 5-second P8
centre; ``--suite final`` measures the 6 production T buckets x P in {1, 2, 4, 8}
for both formats.  JIT compilation, allocation and the static global scale are
outside the timing boundary.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.cake_minimax_h3 import MiniMaxH3QkvQuantizePack
from flashinfer.testing.utils import bench_gpu_time

NUM_HEADS = 56
HEAD_DIM = 128
QKV_KINDS = 3
FP4_BLOCK = 16
MXFP8_BLOCK = 32
FORMATS = ("nvfp4", "mxfp8")

# (duration_s, global tokens T); M = T / P per destination rank.
PRODUCTION_BUCKETS = [
    (4, 33472),
    (5, 38592),
    (6, 48768),
    (8, 58944),
    (10, 74240),
    (15, 109952),
]
PARTITIONS = (1, 2, 4, 8)
CENTER_SHAPES = [
    (duration, T // P, P) for duration, T in PRODUCTION_BUCKETS for P in PARTITIONS
]
ACTIVE_SHAPES = [(5, 4824, 8)]


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _scale_cols(fmt: str) -> int:
    return HEAD_DIM // (FP4_BLOCK if fmt == "nvfp4" else MXFP8_BLOCK)


def traffic_bytes(M: int, P: int, fmt: str) -> int:
    """Minimum HBM traffic of the fused operation: BF16 reads plus live writes."""
    rows = M * (NUM_HEADS // P) * QKV_KINDS
    read = M * NUM_HEADS * QKV_KINDS * HEAD_DIM * 2
    if fmt == "nvfp4":
        return read + rows * (HEAD_DIM // 2) + rows * _scale_cols(fmt)
    return read + rows * HEAD_DIM + rows * _scale_cols(fmt)


def _make_case(M: int, P: int, fmt: str, device: torch.device) -> dict:
    generator = torch.Generator(device=device)
    generator.manual_seed(4532 + M + P)
    sources = []
    for _ in range(QKV_KINDS):
        t = torch.empty((M, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
        t.normal_(0.0, 1.0, generator=generator)
        sources.append(t)
    hpd = NUM_HEADS // P
    rows = M * hpd * QKV_KINDS
    scale_stride = _round_up(rows, 128) * _scale_cols(fmt)
    if fmt == "nvfp4":
        q_shape, q_dtype = (P, M, hpd, QKV_KINDS, HEAD_DIM // 2), torch.uint8
        amax = torch.stack([s.float().abs().amax() for s in sources]).amax()
        global_scale = ((448.0 * 6.0) / amax).reshape(1).float()
    else:
        q_shape, q_dtype = (P, M, hpd, QKV_KINDS, HEAD_DIM), torch.float8_e4m3fn
        global_scale = None

    def outputs():
        return (
            torch.empty(q_shape, dtype=q_dtype, device=device),
            torch.empty((P, scale_stride), dtype=torch.uint8, device=device),
        )

    out_q, out_sf = outputs()
    baseline_q, baseline_sf = outputs()
    return {
        "M": M,
        "P": P,
        "format": fmt,
        "rows": rows,
        "q": sources[0],
        "k": sources[1],
        "v": sources[2],
        "out_global_scale": global_scale,
        "out_q": out_q,
        "out_sf": out_sf,
        "baseline_q": baseline_q,
        "baseline_sf": baseline_sf,
        "staging": torch.empty(
            (P, M, hpd, QKV_KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device
        ),
    }


def _segmented_baseline(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    M, P, rows = case["M"], case["P"], case["rows"]
    hpd = NUM_HEADS // P
    staging = case["staging"]
    for kind, source in enumerate((case["q"], case["k"], case["v"])):
        staging[:, :, :, kind, :].copy_(
            source.view(M, P, hpd, HEAD_DIM).permute(1, 0, 2, 3)
        )
    out_q, out_sf = case["baseline_q"], case["baseline_sf"]
    # The physical scale ABI defines the padding rows as zero bytes.
    out_sf.zero_()
    for destination in range(P):
        flat = staging[destination].view(rows, HEAD_DIM)
        if case["format"] == "nvfp4":
            x_q, sf = flashinfer.fp4_quantize(
                flat,
                case["out_global_scale"],
                sf_vec_size=FP4_BLOCK,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=True,
            )
            out_q[destination].view(torch.uint8).view(rows, HEAD_DIM // 2).copy_(
                x_q.view(torch.uint8).view(rows, HEAD_DIM // 2)
            )
        else:
            x_q, sf = flashinfer.mxfp8_quantize(flat, is_sf_swizzled_layout=True)
            out_q[destination].view(rows, HEAD_DIM).copy_(x_q.view(rows, HEAD_DIM))
        out_sf[destination].copy_(sf.reshape(-1).view(torch.uint8))
    return out_q, out_sf


def _bench_shape(duration: int, M: int, P: int, fmt: str, device: torch.device) -> dict:
    case = _make_case(M, P, fmt, device)
    operation = MiniMaxH3QkvQuantizePack(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        out_q=case["out_q"],
        out_sf=case["out_sf"],
        P=P,
        format=fmt,
        out_global_scale=case["out_global_scale"],
    )
    run_kwargs = {name: case[name] for name in ("q", "k", "v", "out_q", "out_sf")}
    run_kwargs["out_global_scale"] = case["out_global_scale"]

    expected_q, expected_sf = _segmented_baseline(case)
    actual_q, actual_sf = operation.run(**run_kwargs)
    torch.cuda.synchronize()
    if not torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8)):
        raise RuntimeError(
            f"{fmt} M={M} P={P}: packed values differ from the segmented chain"
        )
    if not torch.equal(actual_sf, expected_sf):
        raise RuntimeError(
            f"{fmt} M={M} P={P}: scale bytes differ from the segmented chain"
        )

    baseline_times = bench_gpu_time(
        lambda: _segmented_baseline(case),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    fused_times = bench_gpu_time(
        lambda: operation.run(**run_kwargs),
        enable_cupti=True,
        dry_run_iters=10,
        repeat_iters=100,
    )
    baseline_ms = float(np.median(baseline_times))
    fused_ms = float(np.median(fused_times))
    total_bytes = traffic_bytes(M, P, fmt)
    return {
        "duration": duration,
        "M": M,
        "P": P,
        "format": fmt,
        "baseline_us": baseline_ms * 1e3,
        "fused_us": fused_ms * 1e3,
        "fused_gbps": total_bytes / (fused_ms * 1e-3) / 1e9,
        "speedup": baseline_ms / fused_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the MiniMax-H3 one-pass QKV quantize-and-pack helper"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--suite", choices=("active", "final"), default="active")
    parser.add_argument("--format", choices=(*FORMATS, "both"), default="both")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise RuntimeError("This benchmark requires compute capability 10.0 or 10.3")

    shapes = ACTIVE_SHAPES if args.suite == "active" else CENTER_SHAPES
    formats = FORMATS if args.format == "both" else (args.format,)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"{'format':>6} {'duration':>8} {'M':>8} {'P':>3} {'segmented us':>13} "
        f"{'fused us':>10} {'fused GB/s':>11} {'speedup':>9}"
    )
    for fmt in formats:
        for duration, M, P in shapes:
            result = _bench_shape(duration, M, P, fmt, device)
            print(
                f"{fmt:>6} {duration:>7}s {M:>8} {P:>3} "
                f"{result['baseline_us']:>13.2f} {result['fused_us']:>10.2f} "
                f"{result['fused_gbps']:>11.1f} {result['speedup']:>8.3f}x"
            )


if __name__ == "__main__":
    main()
