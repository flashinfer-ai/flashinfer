"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
Baseline A/B: KDA ucache verify+flush kernel vs the PRODUCTION spec-decode
verify path (``recurrent_kda`` with num_spec_tokens, per-token checkpoint
scatter) — the kernel ReplaySSM replaces on the every-step verify.

Apples-to-apples config: hist_len = 0, 0%% fold — a bare-checkpoint verify of
T draft tokens, which is the ONLY workload recurrent_kda can express (it has
no history ring). Note the ucache kernel still does strictly MORE work here:
it also appends T (normed-k, u, per-channel-G) ring entries per (req, head).
A refcheck (max|Δy| < 2e-2) runs before any timing.

Methodology: flashinfer.testing.bench_gpu_time CUPTI cold-L2, median.
recurrent_kda runs eager (as in bench_kda_output_only.py); the ucache call
is replayed from a CUDA graph, its serving launch mode, which also keeps
the wrapper's eager-only input range checks (one host sync) out of timing.

Run:
  python benchmarks/experimental/bench_kda_ucache_vs_recurrent.py [--iters 100] [--t 4]
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import torch

from flashinfer.kda_decode import recurrent_kda
from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, K, V = 12, 128, 128  # Kimi K3 (96 q-heads) at TP=8; H == HV
SCALE = 1.0 / math.sqrt(K)
LOWER_BOUND = -5.0


def load_ucache():
    from flashinfer.api_logging import ExperimentalWarning
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    return kda_delta_rule_mtp_ucache_flush


def graphed(fn):
    """Capture fn in a CUDA graph (the serving launch mode). It also keeps the
    ucache wrapper's eager-only range checks (one host sync) out of the timed
    region; the kernel itself is unchanged."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def make_case(B, T, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=gen, device=DEV) * sc).bfloat16()

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, b = rn(B, T, H, V, sc=0.5), rn(B, T, H)
    g = rn(B, T, H, K, sc=0.5)
    A_log = (torch.rand(H, generator=gen, device=DEV) * 0.6 - 0.3).float()
    dt_bias = (torch.randn(H * K, generator=gen, device=DEV) * 0.5).float()
    pool = (torch.randn(B, H, V, K, generator=gen, device=DEV) * 0.5).bfloat16()
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, g, b, A_log, dt_bias, pool, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--t", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()
    T = args.t
    RING = 32

    uc = load_ucache()
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | KDA ucache (verify, P=0, ring "
        f"appends ON) vs recurrent_kda spec verify (per-token ckpt scatter) | "
        f"T={T} H=HV={H} K=V={K} lb={LOWER_BOUND} | ucache CUDA-graph, recurrent eager, CUPTI cold-L2 "
        f"median of {args.iters}",
        flush=True,
    )
    hdr = "   B | ucache(us) | recurrent(us) | speedup | max|dy|"
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        q, k, v, g, b, A_log, dt_bias, pool, idx = make_case(B, T, 1000 + B)
        kc = torch.zeros(B, H, RING, K, dtype=torch.bfloat16, device=DEV)
        ucache = torch.zeros(B, H, RING, V, dtype=torch.bfloat16, device=DEV)
        gc = torch.zeros(B, H, RING, K, dtype=torch.float32, device=DEV)
        hl = torch.zeros(B, dtype=torch.int32, device=DEV)
        cb = torch.zeros(B, dtype=torch.int32, device=DEV)
        out_uc = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=DEV)

        def run_ucache():
            uc(
                A_log,
                g,
                dt_bias,
                lower_bound=LOWER_BOUND,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=pool,
                initial_state_indices=idx,
                k_cache=kc,
                u_cache=ucache,
                g_cache=gc,
                hist_len=hl,
                cache_base=cb,
                scale=SCALE,
                flush_min=16 - T + 1,
                restart_hist_on_flush=False,
                output=out_uc,
                w_ring=16,
            )

        # production spec-decode verify: varlen-packed, one checkpoint slot
        # per speculative token (the rollback ledger recurrent_kda maintains)
        qs = q.reshape(1, B * T, H, K)
        ks = k.reshape(1, B * T, H, K)
        vs = v.reshape(1, B * T, H, V)
        gs = g.reshape(1, B * T, H, K)
        bs = torch.sigmoid(b.float()).bfloat16().reshape(1, B * T, H)
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=DEV)
        ckpt = torch.zeros(B * T, H, V, K, dtype=torch.bfloat16, device=DEV)
        ssm_idx = torch.arange(B * T, dtype=torch.int32, device=DEV).reshape(B, T)

        def run_recurrent():
            return recurrent_kda(
                qs,
                ks,
                vs,
                gs,
                bs,
                initial_state=ckpt,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=LOWER_BOUND,
                cu_seqlens=cu,
                ssm_state_indices=ssm_idx,
                num_spec_tokens=T - 1,
                initial_state_source=pool,
                initial_state_indices=idx,
                scale=SCALE,
            )

        # refcheck before timing
        run_ucache()
        y_ref = run_recurrent()
        y_ref = y_ref[0] if isinstance(y_ref, tuple) else y_ref
        dy = (out_uc.reshape(1, B * T, H, V).float() - y_ref.float()).abs().max().item()
        assert dy < 2e-2, f"B={B}: refcheck failed, max|dy|={dy:.3e}"

        t_uc = (
            float(
                np.median(
                    bench_gpu_time(
                        graphed(run_ucache),
                        enable_cupti=True,
                        cold_l2_cache=True,
                        dry_run_iters=10,
                        repeat_iters=args.iters,
                    )
                )
            )
            * 1000.0
        )
        t_rec = (
            float(
                np.median(
                    bench_gpu_time(
                        run_recurrent,
                        enable_cupti=True,
                        cold_l2_cache=True,
                        dry_run_iters=10,
                        repeat_iters=args.iters,
                    )
                )
            )
            * 1000.0
        )
        print(
            f"{B:4d} | {t_uc:10.2f} | {t_rec:13.2f} | {t_rec / t_uc:7.2f}x | {dy:.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
