#!/usr/bin/env python3
"""A/B the dashboard-derived MUSA Mamba/SSD fast paths.

This benchmark intentionally measures the provider wrappers used by the
standalone FlashInfer APIs. It does not claim serving performance: run the
compiled Nemotron benchmark separately with identical server flags.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torchada  # noqa: F401  # must precede torch ecosystem imports
import torch

from flashinfer.mamba.musa_ssd_chunk_state import _DT_MAX, _chunk_cumsum_fwd
from flashinfer.mamba.musa_ssd_cumsum_native import (
    musa_ssd_chunk_cumsum_native,
    preload_musa_ssd_chunk_cumsum,
)
from flashinfer.mamba.musa_ssd_scan_tce import (
    musa_ssd_chunk_scan_tce_native,
    preload_musa_ssd_chunk_scan_tce,
)
from flashinfer.mamba.musa_ssu_native import (
    musa_ssu_one_token_native,
    preload_musa_simple_stp,
)


def sync() -> None:
    torch.musa.synchronize()


def timed(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    values = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        values.append((time.perf_counter() - start) * 1e6)
    values.sort()
    return {
        "min_us": values[0],
        "median_us": values[len(values) // 2],
        "p90_us": values[int(len(values) * 0.9)],
        "mean_us": sum(values) / len(values),
    }


def make_ssu(device):
    slots = 2
    state = torch.randn((slots, 64, 64, 128), device=device, dtype=torch.float16)
    x = torch.randn((1, 64, 64), device=device, dtype=torch.bfloat16)
    dt0 = torch.randn((1, 64), device=device, dtype=torch.float32) * 0.1
    dt = dt0.unsqueeze(-1).expand(1, 64, 64)
    a0 = torch.randn(64, device=device, dtype=torch.float32) * 0.01
    a = a0[:, None, None].expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device=device, dtype=torch.bfloat16)
    c = torch.randn((1, 8, 128), device=device, dtype=torch.bfloat16)
    d0 = torch.randn(64, device=device, dtype=torch.float32)
    d = d0[:, None].expand(64, 64)
    bias0 = torch.randn(64, device=device, dtype=torch.float32) * 0.1
    bias = bias0[:, None].expand(64, 64)
    src = torch.tensor([0], device=device, dtype=torch.int32)
    dst = torch.tensor([1], device=device, dtype=torch.int32)
    seed = torch.tensor([1234567], device=device, dtype=torch.int64)

    def run(disable_fast: bool):
        os.environ["FLASHINFER_MUSA_SIMPLE_STP_DISABLE_FAST"] = (
            "1" if disable_fast else "0"
        )
        state_i = state.clone()
        out = torch.empty_like(x)

        def call():
            musa_ssu_one_token_native(
                state_i,
                x,
                dt,
                a,
                b,
                c,
                d,
                src,
                dst,
                bias,
                None,
                True,
                -1,
                out,
                seed,
                5,
            )

        return call, state_i, out

    return run


def make_cumsum(device):
    tokens = 4096
    dt = torch.randn(tokens, 64, device=device, dtype=torch.float32) * 0.1
    a = torch.randn(64, device=device, dtype=torch.float32) * 0.1
    bias = torch.randn(64, device=device, dtype=torch.float32) * 0.1
    chunks = torch.arange(0, tokens + 1, 128, device=device, dtype=torch.int32)

    def native():
        return musa_ssd_chunk_cumsum_native(dt, a, bias)

    def triton():
        return _chunk_cumsum_fwd(
            dt,
            a,
            128,
            chunks,
            dt_bias=bias,
            dt_softplus=True,
            dt_limit=(0.0, _DT_MAX),
            regular_full_chunks=False,
        )

    return native, triton


def make_scan(device):
    state = torch.randn(64, 64, 128, device=device, dtype=torch.float32) * 0.1
    x = torch.randn(128, 64, 64, device=device, dtype=torch.bfloat16) * 0.1
    dt = torch.rand(128, 64, device=device, dtype=torch.float32) * 0.1
    a = -torch.rand(64, device=device, dtype=torch.float32) * 0.05
    b = torch.randn(128, 8, 128, device=device, dtype=torch.bfloat16) * 0.1
    c = torch.randn(128, 8, 128, device=device, dtype=torch.bfloat16) * 0.1
    d = torch.randn(64, device=device, dtype=torch.float32) * 0.1

    def native():
        return musa_ssd_chunk_scan_tce_native(state, x, dt, a, b, c, d)

    def reference():
        cumsum = torch.cumsum(dt, dim=0)
        decay = torch.exp(a[None, :] * cumsum)
        weighted = x.float() * dt[:, :, None] / decay[:, :, None]
        cb = torch.einsum("tgn,sgn->gts", c.float(), b.float())
        cb = cb * torch.tril(torch.ones(128, 128, device=device, dtype=torch.bool))
        rep = 64 // 8
        grouped = weighted.reshape(128, 8, rep * 64).transpose(0, 1)
        y = torch.bmm(cb, grouped).reshape(8, 128, rep, 64)
        y = y.transpose(0, 1).reshape_as(x).float()
        c_expanded = c.float().repeat_interleave(rep, dim=1).transpose(0, 1)
        y = y + torch.bmm(c_expanded, state.transpose(1, 2)).transpose(0, 1)
        return (y * decay[:, :, None] + x.float() * d[None, :, None]).to(
            torch.bfloat16
        )

    return native, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="musa")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1234)
    os.environ.setdefault("FLASHINFER_MUSA_SIMPLE_STP_NATIVE", "1")
    preload_musa_simple_stp()
    preload_musa_ssd_chunk_cumsum()
    preload_musa_ssd_chunk_scan_tce()

    ssu = make_ssu(device)
    fast_call, _, _ = ssu(False)
    legacy_call, _, _ = ssu(True)
    fast_stats = timed(fast_call, args.warmup, args.iters)
    legacy_stats = timed(legacy_call, args.warmup, args.iters)
    fast_call, fast_state, fast_out = ssu(False)
    legacy_call, legacy_state, legacy_out = ssu(True)
    fast_call()
    legacy_call()
    sync()
    print(
        json.dumps(
            {
                "op": "simple_stp",
                "fast": fast_stats,
                "legacy": legacy_stats,
                "max_abs_output": float((fast_out.float() - legacy_out.float()).abs().max().cpu()),
                "max_abs_state": float((fast_state.float() - legacy_state.float()).abs().max().cpu()),
            }
        )
    )

    native_cumsum, triton_cumsum = make_cumsum(device)
    native_stats = timed(native_cumsum, args.warmup, args.iters)
    triton_stats = timed(triton_cumsum, args.warmup, args.iters)
    print(json.dumps({"op": "ssd_chunk_cumsum", "native": native_stats, "triton": triton_stats}))

    native_scan, reference_scan = make_scan(device)
    scan_stats = timed(native_scan, args.warmup, args.iters)
    y_native = native_scan()
    y_reference = reference_scan()
    sync()
    print(
        json.dumps(
            {
                "op": "ssd_chunk_scan_tce",
                "native": scan_stats,
                "reference": timed(reference_scan, 2, max(3, args.iters // 5)),
                "max_abs": float((y_native.float() - y_reference.float()).abs().max().cpu()),
            }
        )
    )


if __name__ == "__main__":
    main()
