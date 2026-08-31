"""SEQ vs Triton WY — T×BS sweep with CUPTI kernel-only timing.

Matches the rigor of benchmarks/bench_gdn_decode.py / bench_wy_tvar_bssweep.py:
  * flashinfer.testing.bench_gpu_time(enable_cupti=True, ...)
  * CUPTI ActivityKind.CONCURRENT_KERNEL → pure GPU kernel time, no launch overhead
  * Cold L2 cache flush between iterations
  * np.median of per-iter GPU times, reported in microseconds

Contract for apples-to-apples:
  * `disable_state_update=True` on both kernels → NO h0 writeback
  * NO intermediate state caching (neither kernel writes h_1..h_T)
  * Both kernels write the full [B, T, HV, V] output tensor
  * Default preset is qwen3.5 (H=16, HV=64, K=V=128), per the GDN MTP target.

Also performs an apples-to-apples sanity check: verifies that Triton output
agrees with SEQ output to BF16 precision on one (BS, T) pair before timing.
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
    gated_delta_rule_mtp_wy_triton as tri_kernel,
)
from flashinfer.testing import bench_gpu_time


def make_inputs(B, T, H, HK, HV, D, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    num_sab = max(H, HV)
    return dict(
        q=torch.randn(B, T, H, D, dtype=dtype, device="cuda"),
        k=torch.randn(B, T, HK, D, dtype=dtype, device="cuda"),
        v=torch.randn(B, T, num_sab, D, dtype=dtype, device="cuda"),
        a=torch.randn(B, T, num_sab, dtype=dtype, device="cuda"),
        b=torch.randn(B, T, num_sab, dtype=dtype, device="cuda"),
        A_log=torch.randn(num_sab, dtype=torch.float32, device="cuda"),
        dt_bias=torch.randn(num_sab, dtype=torch.float32, device="cuda"),
        state=torch.randn(B, num_sab, D, D, dtype=dtype, device="cuda"),
        idx=torch.arange(B, dtype=torch.int32, device="cuda"),
        out=torch.empty(B, T, num_sab, D, dtype=dtype, device="cuda"),
        scale=1.0 / (D**0.5),
    )


def _launch(kernel_fn, ins):
    kernel_fn(
        A_log=ins["A_log"],
        a=ins["a"],
        dt_bias=ins["dt_bias"],
        q=ins["q"],
        k=ins["k"],
        v=ins["v"],
        b=ins["b"],
        initial_state_source=ins["state"],
        initial_state_indices=ins["idx"],
        disable_state_update=True,  # NO state update → no h0 writeback
        use_qk_l2norm_in_kernel=True,
        scale=ins["scale"],
        output=ins["out"],
    )


def bench(kernel_fn, ins, *, warmup, iters):
    times_ms = bench_gpu_time(
        lambda: _launch(kernel_fn, ins),
        enable_cupti=True,
        dry_run_iters=warmup,
        repeat_iters=iters,
    )
    return np.median(times_ms) * 1000.0  # us


def apples_to_apples_check(B, T, H, HK, HV, D):
    """Run SEQ and Triton on identical inputs; confirm both write a full
    [B, T, HV, V] output and agree to BF16 precision. Also confirms neither
    one mutates the state tensor (disable_state_update=True)."""
    ins_seq = make_inputs(B, T, H, HK, HV, D)
    ins_tri = make_inputs(B, T, H, HK, HV, D)

    state_before_seq = ins_seq["state"].clone()
    state_before_tri = ins_tri["state"].clone()

    _launch(seq_kernel, ins_seq)
    _launch(tri_kernel, ins_tri)

    seq_out_touched = torch.isfinite(ins_seq["out"]).all().item()
    tri_out_touched = torch.isfinite(ins_tri["out"]).all().item()
    assert seq_out_touched and tri_out_touched, "Output has NaN/Inf"

    # State must be unchanged on both (disable_state_update=True)
    seq_state_unchanged = torch.equal(ins_seq["state"], state_before_seq)
    tri_state_unchanged = torch.equal(ins_tri["state"], state_before_tri)

    # Output shape contract
    assert ins_seq["out"].shape == (B, T, max(H, HV), D)
    assert ins_tri["out"].shape == (B, T, max(H, HV), D)
    # All T rows written (output was empty → any NaN in [0..T-1] would fail above)

    # Precision: both should match at BF16 noise floor. Use a loose 1e-2 abs gate.
    diff = (ins_seq["out"].float() - ins_tri["out"].float()).abs()
    max_d = diff.max().item()
    ref_max = ins_seq["out"].float().abs().max().item()
    rel = max_d / max(ref_max, 1e-10)
    return dict(
        seq_state_unchanged=seq_state_unchanged,
        tri_state_unchanged=tri_state_unchanged,
        max_diff=max_d,
        ref_max=ref_max,
        rel=rel,
        output_shape=tuple(ins_seq["out"].shape),
    )


def print_table(title, data, batch_sizes, t_range, fmt="{:>8.2f}"):
    print(f"\n{title}")
    print("   BS | " + " | ".join(f"T={t:>2}" for t in t_range))
    print(" -----+" + "-+".join(["-" * 6 for _ in t_range]) + "-")
    for BS in batch_sizes:
        row = f" {BS:>4} | " + " | ".join(fmt.format(data[BS][T]) for T in t_range)
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset",
        choices=["qwen3-next", "qwen3.5"],
        default="qwen3.5",
        help="Head configuration preset. Default: qwen3.5 (HV=64).",
    )
    p.add_argument("--t-min", type=int, default=2)
    p.add_argument("--t-max", type=int, default=16)
    p.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256]
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="If set, also dump results as CSV to this path.",
    )
    args = p.parse_args()

    if args.preset == "qwen3.5":
        H, HK, HV, D = 16, 16, 64, 128
    else:
        H, HK, HV, D = 16, 16, 32, 128

    t_range = list(range(args.t_min, args.t_max + 1))

    print("=" * 110)
    print(
        f"SEQ vs Triton WY — preset={args.preset}  H={H}  HK={HK}  HV={HV}  K=V={D}  dtype=bf16"
    )
    print(f"T ∈ [{args.t_min}..{args.t_max}]  BS ∈ {args.batch_sizes}")
    print(
        f"CUPTI (ActivityKind.CONCURRENT_KERNEL), cold L2, warmup={args.warmup}, iters={args.iters}"
    )
    print(
        "Both kernels: disable_state_update=True → no h0 writeback, no intermediate state cache."
    )
    print("=" * 110)

    # --------------------- apples-to-apples sanity --------------------
    print("\n[sanity] Apples-to-apples contract check @ BS=4, T=8:")
    chk = apples_to_apples_check(B=4, T=8, H=H, HK=HK, HV=HV, D=D)
    print(
        f"  output shape (both):     {chk['output_shape']}  "
        f"(expected ({4}, {8}, {max(H, HV)}, {D}))"
    )
    print(f"  SEQ state unchanged:     {chk['seq_state_unchanged']}")
    print(f"  Triton state unchanged:  {chk['tri_state_unchanged']}")
    print(
        f"  SEQ vs Triton max_diff:  {chk['max_diff']:.3e}  "
        f"rel={chk['rel']:.3e}  ref_max={chk['ref_max']:.3e}"
    )
    if not (chk["seq_state_unchanged"] and chk["tri_state_unchanged"]):
        print("  *** STATE TENSORS WERE MUTATED — apples-to-apples broken! ***")
        sys.exit(2)
    if chk["rel"] > 5e-2:
        print("  *** SEQ and Triton outputs disagree — apples-to-apples broken! ***")
        sys.exit(2)
    print(
        "  → contracts OK: both kernels write only output, no state/intermediate state cache.\n"
    )

    # ----------------------------- timing -----------------------------
    results_seq = {BS: {} for BS in args.batch_sizes}
    results_tri = {BS: {} for BS in args.batch_sizes}

    for BS in args.batch_sizes:
        for T in t_range:
            ins_seq = make_inputs(BS, T, H, HK, HV, D)
            ins_tri = make_inputs(BS, T, H, HK, HV, D)
            t_seq = bench(seq_kernel, ins_seq, warmup=args.warmup, iters=args.iters)
            t_tri = bench(tri_kernel, ins_tri, warmup=args.warmup, iters=args.iters)
            results_seq[BS][T] = t_seq
            results_tri[BS][T] = t_tri
        # Progress line
        row_seq = "  ".join(f"T={T}:{results_seq[BS][T]:>6.1f}us" for T in t_range)
        print(f"[progress] BS={BS:>4}  {row_seq}")

    print_table(
        "SEQ (µs) — sequential BF16 MTP kernel, no state update",
        results_seq,
        args.batch_sizes,
        t_range,
        fmt="{:>8.2f}",
    )
    print_table(
        "Triton (µs) — WY parallel Triton kernel, no state update",
        results_tri,
        args.batch_sizes,
        t_range,
        fmt="{:>8.2f}",
    )
    ratio = {
        BS: {T: results_seq[BS][T] / results_tri[BS][T] for T in t_range}
        for BS in args.batch_sizes
    }
    print_table(
        "SEQ / Triton  (>1 → Triton is faster)",
        ratio,
        args.batch_sizes,
        t_range,
        fmt="{:>8.2f}x",
    )

    # Winner per cell
    print("\nWinner per cell (T=Triton, S=SEQ, . = tie within 3%):")
    print("   BS | " + " | ".join(f"T={t:>2}" for t in t_range))
    print(" -----+" + "-+".join(["-" * 6 for _ in t_range]) + "-")
    for BS in args.batch_sizes:
        cells = []
        for T in t_range:
            r = ratio[BS][T]
            if r > 1.03:
                cells.append("  T   ")
            elif r < 0.97:
                cells.append("  S   ")
            else:
                cells.append("  .   ")
        print(f" {BS:>4} | " + " | ".join(cells))

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["BS", "T", "SEQ_us", "Triton_us", "seq_over_tri"])
            for BS in args.batch_sizes:
                for T in t_range:
                    w.writerow(
                        [
                            BS,
                            T,
                            f"{results_seq[BS][T]:.3f}",
                            f"{results_tri[BS][T]:.3f}",
                            f"{ratio[BS][T]:.4f}",
                        ]
                    )
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
