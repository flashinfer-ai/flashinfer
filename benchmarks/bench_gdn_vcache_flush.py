"""Benchmark the GDN raw-v-cache fused verify+flush kernel (B200 / SM100).

Per-kernel latency of the FULL pipeline (main kernel + k-ring append
micro-kernel), CUDA-graph-captured and timed with CUPTI + cold-L2 rotating
buffers, swept over batch size at a fixed flush rate.

Run (from the repo root):
    python benchmarks/bench_gdn_vcache_flush.py
    python benchmarks/bench_gdn_vcache_flush.py --batches 8 64 256 --flush-pct 20

Optional: compare against the u-cache spec-decode kernel (PR #4081) by pointing
GDN_UCACHE_DIR at a checkout that contains
flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py:
    GDN_UCACHE_DIR=/path/to/ucache/checkout python benchmarks/bench_gdn_vcache_flush.py
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vcf_mod = _load(
    "vcf",
    os.path.join(REPO, "flashinfer/gdn_kernels/gdn_decode_bf16_wy_vcache_flush.py"),
)
vcf = _vcf_mod.gated_delta_rule_mtp_vcache_flush
ST_TORCH = _vcf_mod.ST_TORCH  # GDN_VCACHE_STATE_DTYPE: bf16 (default) / fp16

uc = None
_UC_DIR = os.environ.get("GDN_UCACHE_DIR")
if _UC_DIR:
    uc = _load(
        "uc",
        os.path.join(
            _UC_DIR, "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py"
        ),
    ).gated_delta_rule_mtp_ucache_flush

from flashinfer.testing import bench_gpu_time  # noqa: E402

H = HK = 16
HV = 64
K = V = 128
W = 16
T = 4
SCALE = 1.0 / math.sqrt(K)


def _us(fn, kw):
    return float(
        np.median(
            bench_gpu_time(
                fn,
                dry_run_iters=8,
                repeat_iters=200,
                enable_cupti=True,
                use_cuda_graph=True,
                cold_l2_cache=True,
                input_kwargs=kw,
            )
        )
        * 1000
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--flush-pct", type=int, default=33)
    ap.add_argument(
        "--base", type=int, default=0,
        help="ring window origin for all requests (e.g. 28 places the live "
        "window and appends across the 31->0 wrap)",
    )
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    print(
        f"GPU: {torch.cuda.get_device_name()}  HV={HV} H={H} K=V={K} T={T} "
        f"flush={args.flush_pct}%  (CUPTI, CUDA graph, cold L2)"
    )
    hdr = f"{'B':>5} {'vcache full (us)':>17}"
    if uc:
        hdr += f" {'#4081 ucache (us)':>18} {'ratio':>6}"
    print(hdr)
    for B in args.batches:
        nfl = max(1, B * args.flush_pct // 100)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(B, generator=g).to(dev)
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, T, HK, K, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
        a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev) * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
        A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
        S0 = torch.randn(B, HV, V, K, dtype=ST_TORCH, device=dev) * 0.1
        idx = torch.arange(B, dtype=torch.int32, device=dev)
        kc = torch.randn(B, HK, 32, K, dtype=torch.bfloat16, device=dev)
        vc_ = torch.randn(B, HV, 32, V, dtype=torch.bfloat16, device=dev)
        ac = torch.randn(B, HV, 32, dtype=torch.float32, device=dev) * 0.1
        bc = torch.randn(B, HV, 32, dtype=torch.float32, device=dev)
        hist = torch.full((B,), 8, dtype=torch.int32, device=dev)
        hist[perm[:nfl]] = 12
        cb = torch.full((B,), args.base, dtype=torch.int32, device=dev)
        kwA = dict(
            A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
            initial_state_source=S0.clone(), initial_state_indices=idx,
            k_cache=kc, v_cache=vc_, a_cache=ac, b_cache=bc, hist_len=hist,
            cache_base=cb,
            flush_min=12, restart_hist_on_flush=False, scale=SCALE,
        )
        tA = _us(lambda **kw: vcf(**kw), kwA)
        row = f"{B:>5} {tA:17.2f}"
        if uc:
            kcu = torch.zeros(B, HK, 32, K, dtype=torch.bfloat16, device=dev)
            ucu = torch.zeros(B, HV, 32, V, dtype=torch.bfloat16, device=dev)
            gcu = torch.zeros(B, HV, 32, dtype=torch.float32, device=dev)
            hu = torch.full((B,), 8, dtype=torch.int32, device=dev)
            hu[perm[:nfl]] = 13
            kwC = dict(
                A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
                initial_state_source=S0.clone(), initial_state_indices=idx,
                k_cache=kcu, u_cache=ucu, g_cache=gcu, hist_len=hu,
                cache_base=torch.full((B,), args.base, dtype=torch.int32, device=dev),
                flush_min=13, restart_hist_on_flush=False, scale=SCALE,
            )
            tC = _us(lambda **kw: uc(**kw), kwC)
            row += f" {tC:18.2f} {tA / tC:6.2f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
