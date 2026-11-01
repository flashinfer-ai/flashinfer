"""Bench fused MTP kernel vs baselines on the MTP flow: replay K accepted
tokens → compute outputs for T new tokens.

Baselines:
  * SEQ × 2: run the sequential bf16 kernel twice (first K, then T).
  * Triton v2 × 2: run the existing v2 Triton kernel twice.
  * Fused: the single-kernel path.

Timing: flashinfer.testing.bench_gpu_time(enable_cupti=True), cold L2 flush,
median over 40 iterations after 15 warmups.  No CPU/launch overhead counted.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as seq_kernel,
)
from flashinfer.gdn_kernels.gdn_decode_wy_triton import (
    gated_delta_rule_mtp_wy_triton as tri_v2,
)
from flashinfer.gdn_kernels.gdn_decode_wy_triton_mtp_fused import (
    gated_delta_rule_mtp_fused as fused_kernel,
)
from flashinfer.gdn_kernels.gdn_decode_wy_triton_mtp_split import (
    gated_delta_rule_mtp_split as split_kernel,
)
from flashinfer.testing import bench_gpu_time


def make_inputs(B, T, K_MAX, H, HV, D=128, seed=42):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    dev = "cuda"
    return dict(
        k_acc=torch.randn(B, K_MAX, H, D, dtype=dt, device=dev),
        v_acc=torch.randn(B, K_MAX, HV, D, dtype=dt, device=dev),
        a_acc=torch.randn(B, K_MAX, HV, dtype=dt, device=dev),
        b_acc=torch.randn(B, K_MAX, HV, dtype=dt, device=dev),
        q_new=torch.randn(B, T, H, D, dtype=dt, device=dev),
        k_new=torch.randn(B, T, H, D, dtype=dt, device=dev),
        v_new=torch.randn(B, T, HV, D, dtype=dt, device=dev),
        a_new=torch.randn(B, T, HV, dtype=dt, device=dev),
        b_new=torch.randn(B, T, HV, dtype=dt, device=dev),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        dt_bias=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        state=torch.randn(B, HV, D, D, dtype=dt, device=dev) * 0.01,
    )


def _call_two_kernels(kernel, ins, K_fixed, idx, scale):
    """Reference path: 2 calls.  Uses fixed K (same for every batch) because
    kernel signature doesn't support num_accepted[B]."""
    B, T = ins["q_new"].shape[:2]
    HV, D = ins["v_new"].shape[2], ins["v_new"].shape[3]
    dt = ins["q_new"].dtype
    dev = "cuda"

    # Phase A: run over first K_fixed accepted tokens (use dummy q since not needed)
    # SEQ kernel requires q, so we give zeros.
    q_dummy = torch.zeros(B, K_fixed, *ins["q_new"].shape[2:], dtype=dt, device=dev)
    k_phaseA_out = torch.empty(B, K_fixed, HV, D, dtype=dt, device=dev)
    kernel(
        A_log=ins["A_log"],
        a=ins["a_acc"][:, :K_fixed],
        dt_bias=ins["dt_bias"],
        q=q_dummy,
        k=ins["k_acc"][:, :K_fixed],
        v=ins["v_acc"][:, :K_fixed],
        b=ins["b_acc"][:, :K_fixed],
        initial_state_source=ins["state"],
        initial_state_indices=idx,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=k_phaseA_out,
    )
    # Phase B: run over T new tokens with h_K initial state
    out = torch.empty(B, T, HV, D, dtype=dt, device=dev)
    kernel(
        A_log=ins["A_log"],
        a=ins["a_new"],
        dt_bias=ins["dt_bias"],
        q=ins["q_new"],
        k=ins["k_new"],
        v=ins["v_new"],
        b=ins["b_new"],
        initial_state_source=ins["state"],
        initial_state_indices=idx,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=out,
    )


def _call_fused(ins, num_accepted, scale, output):
    fused_kernel(
        k_accepted=ins["k_acc"],
        v_accepted=ins["v_acc"],
        a_accepted=ins["a_acc"],
        b_accepted=ins["b_acc"],
        num_accepted=num_accepted,
        q_new=ins["q_new"],
        k_new=ins["k_new"],
        v_new=ins["v_new"],
        a_new=ins["a_new"],
        b_new=ins["b_new"],
        A_log=ins["A_log"],
        dt_bias=ins["dt_bias"],
        initial_state_source=ins["state"],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        output=output,
    )


def _call_split(ins, num_accepted, scale, output, launch_with_pdl):
    split_kernel(
        k_accepted=ins["k_acc"],
        v_accepted=ins["v_acc"],
        a_accepted=ins["a_acc"],
        b_accepted=ins["b_acc"],
        num_accepted=num_accepted,
        q_new=ins["q_new"],
        k_new=ins["k_new"],
        v_new=ins["v_new"],
        a_new=ins["a_new"],
        b_new=ins["b_new"],
        A_log=ins["A_log"],
        dt_bias=ins["dt_bias"],
        initial_state_source=ins["state"],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        output=output,
        launch_with_pdl=launch_with_pdl,
    )


def bench_fn(fn, warmup, iters):
    times = bench_gpu_time(
        fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters
    )
    return np.median(times) * 1000.0  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["qwen3-next", "qwen3.5"], default="qwen3.5")
    p.add_argument("--K-values", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--T", type=int, default=8)
    p.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256]
    )
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()

    if args.preset == "qwen3.5":
        H, HV, D = 16, 64, 128
    else:
        H, HV, D = 16, 32, 128

    print("=" * 120)
    print(
        f"MTP fused vs baselines — preset={args.preset}  T={args.T}  HV={HV}  K=V={D}"
    )
    print(f"CUPTI timing, cold L2 flush, warmup={args.warmup}, iters={args.iters}")
    print(
        "K accepted tokens uniform per batch (for apples-to-apples with 2-call baselines)"
    )
    print("=" * 120)

    hdr = (
        f"{'K':>2} {'BS':>4} | "
        f"{'SEQ×2':>10} {'TriV2×2':>10} {'Fused':>10} {'Split':>10} {'Split+PDL':>10} | "
        f"{'Split/SEQ×2':>11} {'Split/TriV2×2':>13} {'Split/Fused':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    for K in args.K_values:
        K_MAX = K  # for benchmark, use fixed K
        for BS in args.batch_sizes:
            ins = make_inputs(BS, args.T, K_MAX, H, HV, D)
            dev = "cuda"
            idx = torch.arange(BS, dtype=torch.int32, device=dev)
            scale = 1.0 / (D**0.5)
            num_accepted = torch.full((BS,), K, dtype=torch.int32, device=dev)
            out_buf = torch.empty(BS, args.T, HV, D, dtype=torch.bfloat16, device=dev)

            # SEQ × 2
            ins_seq = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in ins.items()
            }

            def run_seq_x2():
                _call_two_kernels(seq_kernel, ins_seq, K, idx, scale)

            t_seq = bench_fn(run_seq_x2, args.warmup, args.iters)

            # Tri v2 × 2
            ins_v2 = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in ins.items()
            }

            def run_v2_x2():
                _call_two_kernels(tri_v2, ins_v2, K, idx, scale)

            t_v2 = bench_fn(run_v2_x2, args.warmup, args.iters)

            # Fused
            ins_fused = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in ins.items()
            }

            def run_fused():
                _call_fused(ins_fused, num_accepted, scale, out_buf)

            t_fused = bench_fn(run_fused, args.warmup, args.iters)

            # Split (no PDL)
            ins_split = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in ins.items()
            }

            def run_split_nopdl():
                _call_split(
                    ins_split, num_accepted, scale, out_buf, launch_with_pdl=False
                )

            t_split_nopdl = bench_fn(run_split_nopdl, args.warmup, args.iters)

            # Split (with PDL)
            ins_split_pdl = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in ins.items()
            }

            def run_split_pdl():
                _call_split(
                    ins_split_pdl, num_accepted, scale, out_buf, launch_with_pdl=True
                )

            t_split_pdl = bench_fn(run_split_pdl, args.warmup, args.iters)

            t_split_best = min(t_split_nopdl, t_split_pdl)
            print(
                f"{K:>2} {BS:>4} | "
                f"{t_seq:>10.2f} {t_v2:>10.2f} {t_fused:>10.2f} {t_split_nopdl:>10.2f} {t_split_pdl:>10.2f} | "
                f"{t_seq / t_split_best:>10.2f}x {t_v2 / t_split_best:>12.2f}x {t_fused / t_split_best:>11.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
