# Copyright (c) 2026 by FlashInfer team.
"""Benchmark the output-only KDA decode kernel vs the recurrent baseline.

Compares at Kimi K3 per-GPU shapes (H = HV = 12, K = V = 128):

  wy        - the WY-parallel tensor-core output-only kernel
  rec_oo    - the grouped register-recurrent output-only fork
  auto      - the dispatched public op (flashinfer.kda_output_only_decode)
  baseline  - flashinfer.recurrent_kda in fused spec-decode verify mode
              (per-token state checkpoints; the pre-existing way to obtain
              per-token outputs for T draft tokens). ``--baseline-backend
              cake`` selects the exported Cake CUDA kernels instead of the
              CuTe-DSL kernel (supported for T in {1, 2, 4, 5, 6} only).

Timing uses ``flashinfer.testing.bench_gpu_time`` with CUPTI and cold L2
(flushed before every iteration) — the same methodology as
``bench_gdn_decode.py`` — so numbers are directly comparable with the GDN
WY output-only tables. Before timing, every configuration refchecks that all
implementations produce matching outputs (guards against timing a
misconfigured baseline).

Usage:
    python benchmarks/bench_kda_output_only.py [--heads 12] [--tokens 4 8 16]
        [--batches 1 4 16 64 256] [--baseline-backend cute-dsl]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.kda_decode import kda_output_only_decode, recurrent_kda
from flashinfer.testing import bench_gpu_time

K = V = 128


def t_us(fn, warmup=10, iters=50):
    """Median CUPTI GPU kernel time (us), cold L2 flushed per iteration."""
    times_ms = bench_gpu_time(
        fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters
    )
    return float(np.median(times_ms)) * 1e3


def run(B, T, H, HV, baseline_backend, emit=False, dev="cuda"):
    """Refcheck all implementations for one (B, T) config, then time each."""
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    g = F.logsigmoid(torch.randn(B, T, HV, K, dtype=torch.float32, device=dev)).to(
        torch.bfloat16
    )
    beta = (
        torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
        .sigmoid()
        .to(torch.bfloat16)
    )
    h0 = torch.randn(max(B, 2), HV, V, K, dtype=torch.bfloat16, device=dev) * 0.1
    idx = torch.arange(B, dtype=torch.int32, device=dev)
    outb = torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    scale = K**-0.5

    corrb = torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    kgb = torch.empty(B, T, HV, 2 * K, dtype=torch.bfloat16, device=dev)

    def run_oo(be):
        """Run the output-only op with the selected backend."""
        # emit=True benchmarks the full RecoverSSM verify contract
        # (out + corrections U + kg cache).
        return kda_output_only_decode(
            q,
            k,
            v,
            g,
            beta,
            h0,
            idx,
            scale=scale,
            output=outb,
            backend=be,
            emit_corrections=emit,
            corrections_out=corrb if emit else None,
            kg_cache_out=kgb if emit else None,
        )

    # Baseline: fused spec-decode verify (per-token outputs + checkpoints).
    qp = q.reshape(1, B * T, H, K).contiguous()
    kp = k.reshape(1, B * T, H, K).contiguous()
    vp = v.reshape(1, B * T, HV, V).contiguous()
    gp = g.reshape(1, B * T, HV, K).contiguous()
    bp = beta.reshape(1, B * T, HV).contiguous()
    cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    ckpt = torch.zeros(B * T, HV, V, K, dtype=torch.bfloat16, device=dev)
    # T == 1 uses the standard-decode contract (flat [N] indices); the fused
    # spec-decode contract ([N, 1+S]) applies only when num_spec_tokens is set.
    ssm_idx = torch.arange(B * T, dtype=torch.int32, device=dev)
    if T > 1:
        ssm_idx = ssm_idx.reshape(B, T)
    rec_out = torch.empty(1, B * T, HV, V, dtype=torch.bfloat16, device=dev)

    def run_base():
        """Run the fused spec-decode verify baseline (recurrent_kda)."""
        kw = {"num_spec_tokens": T - 1} if T > 1 else {}
        return recurrent_kda(
            qp,
            kp,
            vp,
            gp,
            bp,
            scale=scale,
            initial_state=ckpt,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            ssm_state_indices=ssm_idx,
            initial_state_source=h0,
            initial_state_indices=idx,
            output=rec_out,
            backend=baseline_backend,
            **kw,
        )

    # ---- refcheck: all implementations must agree before timing ----
    out_base = run_base()[0].reshape(B, T, HV, V).clone()
    for be in ["wy", "recurrent", "auto"]:
        r = run_oo(be)
        o = r[0] if emit else r
        d = (o.float() - out_base.float()).abs().max().item()
        assert d < 2e-2, f"B={B} T={T} backend={be}: max|d|={d:.3e} vs baseline"

    times = {be: t_us(lambda be=be: run_oo(be)) for be in ["wy", "recurrent", "auto"]}
    times["baseline"] = t_us(run_base)
    return times


def main():
    """Parse args and print the benchmark table."""
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--tokens", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    p.add_argument(
        "--baseline-backend", choices=["cute-dsl", "cake"], default="cute-dsl"
    )
    p.add_argument(
        "--emit-corrections",
        action="store_true",
        help="benchmark the full verify contract (out + corrections + kg)",
    )
    args = p.parse_args()
    bad_tokens = [t for t in args.tokens if not 1 <= t <= 16]
    if bad_tokens:
        p.error(f"--tokens must be in [1, 16]; got {bad_tokens}")
    if min(args.batches) < 1:
        p.error("--batches must be positive")
    if args.baseline_backend == "cake":
        unsupported = [t for t in args.tokens if t not in (1, 2, 4, 5, 6)]
        if unsupported:
            p.error(
                f"--baseline-backend cake supports T in {{1, 2, 4, 5, 6}}; "
                f"got {unsupported}"
            )
    H = HV = args.heads

    print(
        f"KDA output-only decode benchmark (H=HV={H}, K=V=128, "
        f"baseline={args.baseline_backend}, CUPTI cold-L2)"
    )
    print(
        f"{'B':>4} {'T':>3} | {'wy(us)':>8} {'rec_oo':>8} {'auto':>8} "
        f"{'baseline':>8} {'auto_spdup':>10}"
    )
    for T in args.tokens:
        for B in args.batches:
            t = run(B, T, H, HV, args.baseline_backend, emit=args.emit_corrections)
            print(
                f"{B:>4} {T:>3} | {t['wy']:>8.2f} {t['recurrent']:>8.2f} "
                f"{t['auto']:>8.2f} {t['baseline']:>8.2f} "
                f"{t['baseline'] / t['auto']:>9.2f}x"
            )


if __name__ == "__main__":
    main()
