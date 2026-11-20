# Copyright (c) 2026 by FlashInfer team.
"""KDA output-only (frozen-state) decode: the experimental tcgen05 / UMMA backend vs
the WY kernel and the grouped recurrence on main.

Modes:
  dense   ``kda_wy_output_only`` (precomputed log gate, outputs only; ``--emit``
          adds the bf16 corrections + kg cache)
  dropin  ``kda_recoverssm_verify`` (the vLLM RecoverSSM contract: Kimi K3
          lower-bound gate, beta logits, fp32 slot-indexed corrections, raw k|g
          cache, full-length sequences)

Kernel time: ``bench_gpu_time`` CUPTI median with a cold L2 (flushed per
iteration). Every point refchecks UMMA against WY before timing.

  python benchmarks/experimental/bench_kda_output_only_umma.py --mode dense \
      --tokens 1 4 8 16 --batches 1 8 64 256 1024 2048 --json out.json
"""

import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.kda_kernels.kda_decode_wy_output_only import (
    kda_recoverssm_verify,
    kda_wy_output_only,
)
from flashinfer.testing import bench_gpu_time

K = V = 128


def t_us(fn, iters):
    """CUPTI median and min kernel time (us), cold L2."""
    ms = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=10, repeat_iters=iters)
    return float(np.median(ms)) * 1e3, float(np.min(ms)) * 1e3


def dense_case(B, T, H, emit, dev="cuda", seed=0):
    g_ = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g_, device=dev).bfloat16()
    k = torch.randn(B, T, H, K, generator=g_, device=dev).bfloat16()
    v = torch.randn(B, T, H, V, generator=g_, device=dev).bfloat16()
    g = F.logsigmoid(torch.randn(B, T, H, K, generator=g_, device=dev)).bfloat16()
    beta = torch.rand(B, T, H, generator=g_, device=dev).bfloat16()
    pool = (torch.randn(B, H, V, K, generator=g_, device=dev) * 0.1).bfloat16()
    idx = torch.randperm(B, generator=g_, device=dev).int()

    def run(be):
        return kda_wy_output_only(
            q, k, v, g, beta, pool, idx, emit_corrections=emit, backend=be
        )

    return run


def dropin_case(B, T, H, dev="cuda", seed=0):
    g_ = torch.Generator(device=dev).manual_seed(seed)
    tot = B * T
    q = torch.randn(1, tot, H, K, generator=g_, device=dev).bfloat16()
    k = torch.randn(1, tot, H, K, generator=g_, device=dev).bfloat16()
    v = torch.randn(1, tot, H, V, generator=g_, device=dev).bfloat16()
    g = torch.randn(1, tot, H, K, generator=g_, device=dev).bfloat16()
    beta = torch.randn(1, tot, H, generator=g_, device=dev).bfloat16()
    A_log = torch.randn(H, generator=g_, device=dev) * 0.3
    dt = torch.randn(H * K, generator=g_, device=dev) * 0.1
    nblk = B + 1
    state = (torch.randn(nblk, H, V, K, generator=g_, device=dev) * 0.1).bfloat16()
    corr = torch.zeros(nblk, H, T, V, device=dev)
    kg = torch.zeros(nblk, H, T, 2 * K, dtype=torch.bfloat16, device=dev)
    qsl = torch.arange(0, tot + 1, T, dtype=torch.int32, device=dev)
    sidx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    out = torch.empty_like(v)

    def run(be, check=False):
        kda_recoverssm_verify(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt,
            -5.0,
            state,
            corr,
            kg,
            qsl,
            sidx,
            T,
            out=out,
            backend=be,
        )
        return (out.clone(), corr.clone()) if check else None

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "dropin"], default="dense")
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256, 1024])
    p.add_argument("--emit", action="store_true", help="dense: emit corrections + kg")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    arms = ["wy", "recurrent", "auto", "umma"] if a.mode == "dense" else ["wy", "umma"]
    points = []
    print(f"mode={a.mode} H={a.heads} emit={a.emit}  CUPTI cold-L2 median us")
    print("T     B  " + "  ".join(f"{x:>9}" for x in arms) + "  umma/wy")
    for T in a.tokens:
        for B in a.batches:
            run = (
                dense_case(B, T, a.heads, a.emit)
                if a.mode == "dense"
                else dropin_case(B, T, a.heads)
            )
            ref = run("wy", True) if a.mode == "dropin" else run("wy")
            got = run("umma", True) if a.mode == "dropin" else run("umma")
            ref = ref if isinstance(ref, tuple) else (ref,)
            got = got if isinstance(got, tuple) else (got,)
            for r, x in zip(ref, got, strict=True):
                err = (r.float() - x.float()).abs().max().item()
                tol = 3e-2 if r.dtype == torch.float32 or len(ref) > 1 else 4e-3
                assert err <= max(tol, 2e-2 * r.float().abs().max().item()), (T, B, err)
            row = {}
            for be in arms:
                med, mn = t_us(lambda be=be: run(be), a.iters)
                row[be] = med
                points.append(
                    dict(
                        mode=a.mode,
                        emit=a.emit,
                        H=a.heads,
                        T=T,
                        B=B,
                        backend=be,
                        cold_us=med,
                        cold_min_us=mn,
                    )
                )
            print(
                f"{T:<3} {B:>5}  "
                + "  ".join(f"{row[x]:9.2f}" for x in arms)
                + f"  {row['wy'] / row['umma']:.2f}x",
                flush=True,
            )
    if a.json:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        meta = dict(
            gpu=torch.cuda.get_device_name(),
            git_sha=sha,
            iters=a.iters,
            timing="bench_gpu_time CUPTI, cold L2, median of iters",
        )
        with open(a.json, "w") as f:
            json.dump(dict(meta=meta, points=points), f, indent=1)


if __name__ == "__main__":
    main()
