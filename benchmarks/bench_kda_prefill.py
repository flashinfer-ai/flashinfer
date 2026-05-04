"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

KDA (Kimi Delta Attention) chunk-prefill benchmark
==================================================

Measures kernel time of :func:`flashinfer.kda.chunk_kda_fwd` across a
representative set of equal-length and variable-length workloads on Blackwell.

Optional comparison columns are populated when the corresponding external
package is importable in the current environment — no flag needed:

  * ``fla.ops.kda.chunk_fwd.chunk_kda_fwd`` -> "FLA (us)" + "Speedup vs FLA"
  * ``flash_kda``                            -> "FlashKDA (us)" + "Speedup vs FlashKDA"

Note: ``flash_kda`` only implements the ``safe_gate`` mode (no softplus), so
it is skipped in softplus runs.

Usage::

    python benchmarks/bench_kda_prefill.py                   # default H=96 K=128
    python benchmarks/bench_kda_prefill.py --mode eqlen      # eqlen only
    python benchmarks/bench_kda_prefill.py --mode varlen     # varlen only
    python benchmarks/bench_kda_prefill.py --gate softplus   # softplus only
    python benchmarks/bench_kda_prefill.py --gate safe_gate  # safe_gate only
    python benchmarks/bench_kda_prefill.py -H 4 -K 128       # smaller config
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from flashinfer.kda import chunk_kda_fwd, prepare_chunk_indices
from flashinfer.testing import bench_gpu_time

# ---------------------------------------------------------------------------
# Optional baselines (graceful import — bench still runs without these)
# ---------------------------------------------------------------------------
try:
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd as fla_chunk_kda_fwd

    _has_fla = True
except ImportError:
    fla_chunk_kda_fwd = None  # type: ignore
    _has_fla = False

try:
    import flash_kda  # type: ignore

    _has_flash_kda = True
except ImportError:
    flash_kda = None  # type: ignore
    _has_flash_kda = False


def _l2_normalize_last(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2.0, dim=-1)


def _make_inputs(B, T, H, K, *, seed=0xC0FFEE, device="cuda"):
    torch.manual_seed(seed)
    q = _l2_normalize_last(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device))
    k = _l2_normalize_last(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device))
    v = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.5
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device) * 0.1
    h0 = torch.randn(B, H, K, K, dtype=torch.float32, device=device) * 0.01
    return q, k, v, g, beta, A_log, dt_bias, h0, K ** -0.5


def _t_padded(cu, BT=64):
    return max(b + math.ceil((e - b) / BT) * BT for b, e in zip(cu, cu[1:]))


def _make_varlen(seq_lens, H, K, *, device="cuda", seed=0xC0FFEE):
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    Tp = _t_padded(cu, 64)
    T_tot = cu[-1]
    N = len(seq_lens)
    q, k, v, g, beta, A_log, dt_bias, _, scale = _make_inputs(1, Tp, H, K, seed=seed, device=device)
    if Tp > T_tot:
        q[:, T_tot:] = 0
        k[:, T_tot:] = 0
        v[:, T_tot:] = 0
        g[:, T_tot:] = 0
        beta[:, T_tot:] = 0
    h0 = torch.zeros(N, H, K, K, dtype=torch.float32, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int64, device=device)
    ci = prepare_chunk_indices(cu_t, 64)
    return q, k, v, g, beta, A_log, dt_bias, h0, scale, cu_t, ci, T_tot, Tp


# ---------------------------------------------------------------------------
# flash_kda wrapper. flash_kda's API takes pre-allocated `out` and
# `final_state` tensors and supports only safe_gate mode (lower_bound).
# ---------------------------------------------------------------------------
def _flash_kda_call(q, k, v, g, beta, A_log, dt_bias, h0, scale,
                    *, lower_bound=-5.0, cu_seqlens=None):
    if not _has_flash_kda:
        raise RuntimeError("flash_kda is not installed.")
    H, K = q.shape[-2], q.shape[-1]
    out = torch.empty_like(v)
    dt_bias_2d = dt_bias.view(H, K) if dt_bias.dim() == 1 else dt_bias
    final_state = torch.empty_like(h0)
    extra = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
    flash_kda.fwd(q, k, v, g, beta, scale, out,
                  A_log=A_log, dt_bias=dt_bias_2d, lower_bound=lower_bound,
                  initial_state=h0, final_state=final_state, **extra)
    return out, final_state


# ---------------------------------------------------------------------------
# Workload definitions (mirror the upstream FLA bench cases for apples-to-
# apples comparison).
# ---------------------------------------------------------------------------
EQLEN_CASES = [
    ("eqlen_T8192", 1, 8192),
    ("eqlen_T4096", 1, 4096),
    ("eqlen_T2048", 1, 2048),
    ("eqlen_T1024", 1, 1024),
]

VARLEN_CASES = [
    ("varlen_1x8192",         [8192]),
    ("varlen_1x7999",         [7999]),
    ("varlen_2x4096",         [4096, 4096]),
    ("varlen_2x_skew",        [1024, 7168]),
    ("varlen_4x2048",         [2048] * 4),
    ("varlen_4x_mix",         [512, 2048, 4096, 1536]),
    # 10 medium seqs, 256-1280 range (typical inference batch)
    ("varlen_16x_div",        [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1280]),
    # mix of one long + several medium seqs
    ("varlen_extreme",        [4000, 256, 384, 512, 640, 2400]),
    # 8 medium-uniform seqs (typical batched-inference shape)
    ("varlen_8x1024",         [1024] * 8),
    ("varlen_chat",           [50, 200, 1500, 100, 300, 2000, 50, 150, 500, 3342]),
    ("varlen_prefill",        [8000, 192]),
    # 32 short uniform seqs — many-seq batched-decode stress test
    ("varlen_32x_short",      [256] * 32),
    # 9 unaligned seqs, 500-1200 range
    ("varlen_many_unaligned", [500, 600, 700, 800, 900, 1000, 1100, 1192, 1200]),
]


@dataclass
class CaseResult:
    name: str
    T: int
    safe_gate: bool
    fi_us: float
    fla_us: float = float("nan")
    flash_kda_us: float = float("nan")


def _bench_callable(fn, warmup: int, iters: int) -> float:
    times_ms = bench_gpu_time(fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters)
    return float(np.median(times_ms)) * 1000.0  # ms -> us


def _bench_eqlen(name, B, T, H, K, *, safe_gate, with_dt_bias, warmup, iters):
    q, k, v, g, beta, A_log, dt_bias, h0, scale = _make_inputs(B, T, H, K)
    bias = dt_bias if with_dt_bias else None
    lower_bound = -5.0 if safe_gate else None

    # FlashInfer
    fi_us = _bench_callable(
        lambda: chunk_kda_fwd(
            q, k, v, g, beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            A_log=A_log,
            dt_bias=bias,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        ),
        warmup, iters,
    )

    # FLA (Triton)
    fla_us = float("nan")
    if _has_fla:
        try:
            fla_us = _bench_callable(
                lambda: fla_chunk_kda_fwd(
                    q=q, k=k, v=v, g=g, beta=beta,
                    initial_state=h0,
                    dt_bias=bias,
                    scale=scale,
                    output_final_state=True,
                    chunk_size=64,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                    use_gate_in_kernel=True,
                    A_log=A_log,
                ),
                warmup, iters,
            )
        except Exception as ex:
            print(f"  [warn] FLA failed on {name}: {ex}")

    # flash_kda (safe_gate only)
    fk_us = float("nan")
    if _has_flash_kda and safe_gate and bias is not None:
        try:
            fk_us = _bench_callable(
                lambda: _flash_kda_call(
                    q, k, v, g, beta, A_log, dt_bias, h0, scale,
                    lower_bound=-5.0,
                ),
                warmup, iters,
            )
        except Exception as ex:
            print(f"  [warn] flash_kda failed on {name}: {ex}")

    return CaseResult(name, T, safe_gate, fi_us, fla_us, fk_us)


def _bench_varlen(name, seq_lens, H, K, *, safe_gate, with_dt_bias, warmup, iters):
    q, k, v, g, beta, A_log, dt_bias, h0, scale, cu_t, ci, T_tot, Tp = _make_varlen(seq_lens, H, K)
    bias = dt_bias if with_dt_bias else None
    lower_bound = -5.0 if safe_gate else None

    fi_us = _bench_callable(
        lambda: chunk_kda_fwd(
            q, k, v, g, beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_t,
            chunk_indices=ci,
            A_log=A_log,
            dt_bias=bias,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        ),
        warmup, iters,
    )

    fla_us = float("nan")
    if _has_fla:
        try:
            # FLA's varlen API accepts the trimmed [B, T_tot, ...] tensor.
            q_v = q[:, :T_tot]
            k_v = k[:, :T_tot]
            v_v = v[:, :T_tot]
            g_v = g[:, :T_tot]
            beta_v = beta[:, :T_tot]
            fla_us = _bench_callable(
                lambda: fla_chunk_kda_fwd(
                    q=q_v, k=k_v, v=v_v, g=g_v, beta=beta_v,
                    initial_state=h0,
                    dt_bias=bias,
                    scale=scale,
                    output_final_state=True,
                    chunk_size=64,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                    use_gate_in_kernel=True,
                    A_log=A_log,
                    cu_seqlens=cu_t,
                    chunk_indices=ci,
                ),
                warmup, iters,
            )
        except Exception as ex:
            print(f"  [warn] FLA failed on {name}: {ex}")

    fk_us = float("nan")
    if _has_flash_kda and safe_gate and bias is not None:
        try:
            q_v = q[:, :T_tot]; k_v = k[:, :T_tot]; v_v = v[:, :T_tot]
            g_v = g[:, :T_tot]; beta_v = beta[:, :T_tot]
            fk_us = _bench_callable(
                lambda: _flash_kda_call(
                    q_v, k_v, v_v, g_v, beta_v, A_log, dt_bias, h0, scale,
                    lower_bound=-5.0, cu_seqlens=cu_t,
                ),
                warmup, iters,
            )
        except Exception as ex:
            print(f"  [warn] flash_kda failed on {name}: {ex}")

    return CaseResult(name, T_tot, safe_gate, fi_us, fla_us, fk_us)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _fmt_us(v: float) -> str:
    if v != v:  # NaN
        return "    -    "
    return f"{v:>9.1f}"


def _fmt_speedup(base: float, other: float) -> str:
    if base != base or other != other or other <= 0:
        return "   -   "
    return f"{base / other:>5.2f}x"


def _print_header(title: str, show_fla: bool, show_flash_kda: bool) -> None:
    print(f"\n  ━━━ {title} ━━━\n")
    cols = ["Config", "T", "FlashInfer (us)"]
    if show_fla:
        cols += ["FLA (us)", "vs FLA"]
    if show_flash_kda:
        cols += ["FlashKDA (us)", "vs FlashKDA"]
    widths = [24, 6, 15] + ([10, 8] if show_fla else []) + ([14, 12] if show_flash_kda else [])
    line = "  " + "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths)))
    print(line)
    print("  " + "-" * (sum(widths) + 2 * (len(cols) - 1)))


def _print_row(r: CaseResult, show_fla: bool, show_flash_kda: bool) -> None:
    parts = [
        f"  {r.name:<24}",
        f"  {r.T:>6}",
        f"  {_fmt_us(r.fi_us):>15}",
    ]
    if show_fla:
        parts.append(f"  {_fmt_us(r.fla_us):>10}")
        parts.append(f"  {_fmt_speedup(r.fla_us, r.fi_us):>8}")
    if show_flash_kda:
        parts.append(f"  {_fmt_us(r.flash_kda_us):>14}")
        parts.append(f"  {_fmt_speedup(r.flash_kda_us, r.fi_us):>12}")
    print("".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="KDA chunk-prefill benchmark")
    parser.add_argument("--mode", choices=["all", "eqlen", "varlen"], default="all")
    parser.add_argument(
        "--gate",
        choices=["all", "softplus", "safe_gate"],
        default="all",
        help="Which gate activation(s) to benchmark.",
    )
    parser.add_argument("-H", "--num-heads", type=int, default=96, dest="H")
    parser.add_argument("-K", "--head-dim", type=int, default=128, dest="K")
    parser.add_argument("--no-dt-bias", action="store_true", help="Disable dt_bias.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    cc = torch.cuda.get_device_capability()
    if cc[0] < 10:
        raise SystemExit(f"KDA chunk prefill requires SM100+, got SM{cc[0]}{cc[1]}.")

    H, K = args.H, args.K
    with_dt_bias = not args.no_dt_bias

    print("=" * 72)
    print(f"KDA chunk-prefill benchmark on {torch.cuda.get_device_name()}  |  H={H} K={K}")
    print(f"FLA (Triton):    {'available' if _has_fla else 'NOT installed (skipped)'}")
    print(f"flash_kda (C++): {'available' if _has_flash_kda else 'NOT installed (skipped)'}")
    print("=" * 72)

    gates = []
    if args.gate in ("all", "softplus"):
        gates.append(("softplus", False))
    if args.gate in ("all", "safe_gate"):
        gates.append(("safe_gate", True))

    if args.mode in ("all", "eqlen"):
        for label, safe_gate in gates:
            show_flash_kda_here = _has_flash_kda and safe_gate and with_dt_bias
            _print_header(f"Equal-length benchmarks ({label})", _has_fla, show_flash_kda_here)
            for name, B, T in EQLEN_CASES:
                r = _bench_eqlen(
                    name, B, T, H, K,
                    safe_gate=safe_gate, with_dt_bias=with_dt_bias,
                    warmup=args.warmup, iters=args.iters,
                )
                _print_row(r, _has_fla, show_flash_kda_here)

    if args.mode in ("all", "varlen"):
        for label, safe_gate in gates:
            show_flash_kda_here = _has_flash_kda and safe_gate and with_dt_bias
            _print_header(f"Varlen benchmarks ({label})", _has_fla, show_flash_kda_here)
            for name, seq_lens in VARLEN_CASES:
                r = _bench_varlen(
                    name, seq_lens, H, K,
                    safe_gate=safe_gate, with_dt_bias=with_dt_bias,
                    warmup=args.warmup, iters=args.iters,
                )
                _print_row(r, _has_fla, show_flash_kda_here)


if __name__ == "__main__":
    main()
