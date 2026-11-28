#!/usr/bin/env python3
"""
Benchmark flashinfer.sampling.softcap_sample_with_stats (SM90 CuTe-DSL) against the two-kernel
Triton structure it replaces in vLLM's DiffusionGemma sampler: a softcap kernel that writes an
FP32 copy of the logits, then one 8-warp CTA per row for argmax / Gumbel-max sample / entropy and
a second pass for the BF16 probabilities. The baseline draws the same Philox stream (one full
Philox call per element, keeping one lane), so both compute the same outputs.

Rows are requests x 256 canvas positions; V = 262144 (Gemma vocabulary).

Usage: python benchmarks/bench_softcap_sample_with_stats.py [--rows 256 512 1024 1536 4096]
"""

import argparse

import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

V = 262144
CANVAS = 256
_BLOCK = 4096


@triton.jit(do_not_specialize=["logits_stride", "out_stride", "V"])
def _softcap_kernel(
    logits_ptr, logits_stride, out_ptr, out_stride, V, cap, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * out_stride + offs, tldevice.tanh(x / cap) * cap, mask=mask)


@triton.jit
def _uniform(seed, flat):
    q = flat >> 2
    c0 = (q & 0xFFFFFFFF).to(tl.uint32)
    c1 = (q >> 32).to(tl.uint32)
    zero = c0 * 0
    o0, o1, o2, o3 = tl.philox(seed, c0, c1, zero, zero)
    lane = flat & 3
    o = tl.where(lane == 0, o0, tl.where(lane == 1, o1, tl.where(lane == 2, o2, o3)))
    return tl.uint_to_uniform_float(o)


@triton.jit(do_not_specialize=["seed", "CL"])
def _row_stats_kernel(
    logits_ptr,
    logits_stride,
    temps_ptr,
    argmax_ptr,
    sample_ptr,
    entropy_ptr,
    probs_ptr,
    probs_stride,
    seed,
    V,
    CL,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row // CL).to(tl.float32)
    inv = 1.0 / tl.maximum(temp, 1e-10)
    noise = (temp > 0).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    rng_base = row.to(tl.int64) * V
    m = float("-inf")
    z = 0.0
    s = 0.0
    best = float("-inf")
    best_idx = 0
    best_noisy = float("-inf")
    best_noisy_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, 0)
        new_m = tl.maximum(m, block_max)
        safe_m = tl.where(new_m > float("-inf"), new_m, 0.0)
        alpha = tl.where(m > float("-inf"), tl.exp((m - safe_m) * inv), 0.0)
        shift = tl.where(z > 0, (m - safe_m) * inv * z, 0.0)
        d = (x - safe_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
        m = new_m
        block_idx = tl.argmax(x, 0)
        take = block_max > best
        best_idx = tl.where(take, start + block_idx, best_idx)
        best = tl.where(take, block_max, best)
        u = tl.maximum(_uniform(seed, rng_base + offs), 1e-20)
        gumbel = -tl.log(-tl.log(u))
        noisy = tl.where(mask, x * inv + gumbel * noise, float("-inf"))
        noisy_max = tl.max(noisy, 0)
        noisy_idx = tl.argmax(noisy, 0)
        take_n = noisy_max > best_noisy
        best_noisy_idx = tl.where(take_n, start + noisy_idx, best_noisy_idx)
        best_noisy = tl.where(take_n, noisy_max, best_noisy)
    tl.store(argmax_ptr + row, best_idx)
    tl.store(sample_ptr + row, best_noisy_idx)
    tl.store(entropy_ptr + row, tl.log(z) - s / z)
    out = probs_ptr + row.to(tl.int64) * probs_stride
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        p = tl.exp((x - m) * inv) / z
        tl.store(out + offs, p.to(probs_ptr.dtype.element_ty), mask=mask)


def triton_baseline(logits, temps, seed, cap, capped, argmax, sample, entropy, probs):
    R = logits.shape[0]
    _softcap_kernel[(R, triton.cdiv(V, _BLOCK))](
        logits, logits.stride(0), capped, capped.stride(0), V, cap, BLOCK=_BLOCK
    )
    _row_stats_kernel[(R,)](
        capped,
        capped.stride(0),
        temps,
        argmax,
        sample,
        entropy,
        probs,
        probs.stride(0),
        seed,
        V,
        R // temps.numel(),
        BLOCK=_BLOCK,
        num_warps=8,
    )


def make_inputs(rows):
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    sigma = 1.5 + 2.5 * torch.rand((rows, 1), generator=g, device="cuda")
    logits = torch.empty(rows, V, dtype=torch.bfloat16, device="cuda")
    for r0 in range(0, rows, 256):
        r1 = min(rows, r0 + 256)
        logits[r0:r1] = (
            torch.randn((r1 - r0, V), generator=g, device="cuda") * sigma[r0:r1]
        ).to(torch.bfloat16)
    temps = 0.4 + 0.4 * torch.rand(
        (max(1, rows // CANVAS),), generator=g, device="cuda"
    )
    return logits, temps


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[256, 512, 1024, 1536, 4096]
    )
    args = parser.parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}  V={V}  softcap=30")
    print(
        f"{'rows':>6} {'triton (ms)':>12} {'flashinfer (ms)':>16} {'speedup':>8} {'GB/s':>8}"
    )
    for rows in args.rows:
        logits, temps = make_inputs(rows)
        seed = 1234
        out = (
            torch.empty(rows, dtype=torch.int64, device="cuda"),
            torch.empty(rows, dtype=torch.int64, device="cuda"),
            torch.empty(rows, dtype=torch.float32, device="cuda"),
            torch.empty(rows, V, dtype=torch.bfloat16, device="cuda"),
        )
        capped = torch.empty(rows, V, dtype=torch.float32, device="cuda")
        t_base = np.median(
            bench_gpu_time(
                lambda: triton_baseline(logits, temps, seed, 30.0, capped, *out),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
        t_fi = np.median(
            bench_gpu_time(
                lambda: flashinfer.sampling.softcap_sample_with_stats(
                    logits, temps, seed, 30.0, out=out
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
        gbs = rows * V * 4 / (t_fi * 1e-3) / 1e9  # bf16 logits in + bf16 probs out
        print(
            f"{rows:>6} {t_base:>12.4f} {t_fi:>16.4f} {t_base / t_fi:>7.2f}x {gbs:>8.0f}"
        )


if __name__ == "__main__":
    main()
