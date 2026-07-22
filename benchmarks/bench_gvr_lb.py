"""Benchmark GVR top-k load-balancing (``load_balance=True`` vs ``False``).

``flashinfer.top_k_decode`` with ``backend="gvr"`` has two kernel paths, selected
by the ``load_balance`` flag:

  * ``load_balance=True``  (default) — two-kernel LB path: a prepare kernel
    classifies requests into long/short buckets against ``long_threshold`` (64K),
    then the main kernel splits each *long* row across a ``cluster_size=4`` CTA
    cluster and packs *short* rows one-per-CTA.
  * ``load_balance=False`` — single-kernel path: one CTA per row, each CTA scans
    its whole row.

**Why LB helps.** In the non-LB path every row gets one CTA. When the batch fits
in a single hardware wave (``num_rows <= #SMs``) the wall-clock is bounded by the
*longest* row, so a few very-long rows create a tail: short-row CTAs finish and
idle. LB splits each long row across 4 CTAs (~4x less tail work) and packs the
short rows, cutting that tail. LB is *not* free — when there is no length variance
(all rows equally long, or all short) the extra prepare kernel + cluster sync is
pure overhead and LB *loses*. This benchmark sweeps a spectrum of workloads to
show both regimes and reports per-case speedups plus a geomean.

Example
-------
    python benchmarks/bench_gvr_lb.py
    python benchmarks/bench_gvr_lb.py --dtype bf16 --top-k 2048 --repeat-iters 100
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.topk_blackwell import _count_long_rows, _lb_decision_from_counts
from flashinfer.utils import get_compute_capability

# GVR prepare-kernel long/short cutoff (GvrTopKLBConfig.long_threshold). Rows with
# seq_len/compress_ratio above this take the multi-CTA cluster path.
LONG_THRESHOLD = 64 * 1024
SEED = 42


@dataclass(frozen=True)
class LBCase:
    """One workload: ``num_long`` rows of ``long_len`` + ``num_short`` of ``short_len``."""

    name: str
    num_long: int
    long_len: int
    num_short: int
    short_len: int

    @property
    def num_rows(self) -> int:
        return self.num_long + self.num_short

    @property
    def max_len(self) -> int:
        return max(self.long_len if self.num_long else 0, self.short_len)


def build_inputs(
    case: LBCase,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
    pre_idx_overlap: float = 1.0,
):
    """Build (logits, seq_lens, pre_idx) for a mixed long/short batch.

    ``pre_idx`` models the *previous* decode step's top-K, whose quality as a hint
    for the current step is controlled by ``pre_idx_overlap`` in ``[0, 1]``:

      * ``1.0`` — ``pre_idx`` IS the current step's true top-K (a perfect hint;
        GVR's Phase-1 threshold estimate lands right on the k-th value, so P2
        converges almost immediately).  Unrealistically optimistic.
      * ``f``   — a fraction ``f`` of the K hints are drawn from the true top-K
        and ``1 - f`` are random valid positions (mostly low-value), modelling a
        stale / drifted attention pattern.  GVR's seed threshold is pulled down,
        forcing more Phase-2 refine iterations.
      * ``0.0`` — no hint overlaps the true top-K (worst case).

    Column 0 is always kept as the argmax (the GVR convention / API requirement).
    All indices stay ``< seq_lens[i]`` for every row.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    num_rows = case.num_rows
    N = case.max_len

    logits = (torch.randn(num_rows, N, dtype=torch.float32, device=device) * 2.0).to(
        dtype
    )
    seq_len_list = [case.long_len] * case.num_long + [case.short_len] * case.num_short
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device=device)

    logits_f32 = logits.to(torch.float32)
    gen = torch.Generator(device=device).manual_seed(SEED)
    n_true = int(round(top_k * pre_idx_overlap))
    pre_idx = torch.zeros(num_rows, top_k, dtype=torch.int32, device=device)
    for r in range(num_rows):
        n_eff = int(seq_lens[r].item())
        row = logits_f32[r, :n_eff]
        true_topk = torch.topk(row, top_k).indices  # current-step ground truth
        # n_true hints from the true top-K (perfect part), rest random valid.
        perm = torch.randperm(top_k, generator=gen, device=device)
        hint = torch.empty(top_k, dtype=torch.int32, device=device)
        hint[:n_true] = true_topk[perm[:n_true]].int()
        if n_true < top_k:
            rand_pos = torch.randint(
                0,
                n_eff,
                (top_k - n_true,),
                generator=gen,
                device=device,
                dtype=torch.int32,
            )
            hint[n_true:] = rand_pos
        pre_idx[r] = hint
        # Column 0 must be the argmax (API requirement); overwrite regardless.
        pre_idx[r, 0] = int(row.argmax().item())
    return logits, seq_lens, pre_idx


def run_gvr(logits, seq_lens, top_k, pre_idx, load_balance, num_long_rows=None):
    return flashinfer.top_k_decode(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        backend="gvr",
        load_balance=load_balance,
        num_long_rows=num_long_rows,
    )


def run_radix(logits, seq_lens, top_k):
    # Radix masked fallback: no pre_idx, runs on any GPU. Masks logits to
    # seq_lens then calls the shared FlashInfer radix top-K kernel.
    return flashinfer.top_k_decode(logits, seq_lens, top_k, backend="radix")


def check_correct(indices, logits, seq_lens, top_k):
    """Tie-safe correctness check: every selected value >= the row's k-th largest."""
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    for row in range(indices.shape[0]):
        n_eff = int(seq_lens_host[row])
        if n_eff < top_k:
            continue
        row_logits = logits_f32[row, :n_eff]
        kth = torch.topk(row_logits, k=top_k).values[-1].item()
        sel = [int(i) for i in indices[row].cpu().tolist() if i >= 0]
        assert len(sel) == top_k, f"row={row}: got {len(sel)} indices, want {top_k}"
        assert len(set(sel)) == len(sel), f"row={row}: duplicate indices"
        assert all(0 <= i < n_eff for i in sel), f"row={row}: out-of-range index"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        assert (sel_vals < kth).sum() == 0, f"row={row}: value below k-th rank"


def bench_case(
    case: LBCase,
    top_k: int,
    dtype: torch.dtype,
    device,
    args,
    pre_idx_overlap: float = 1.0,
) -> dict:
    logits, seq_lens, pre_idx = build_inputs(
        case, top_k, dtype, device, pre_idx_overlap
    )

    # Host-side long-row count -> the graph-safe LB decision (pure host).
    n_long = _count_long_rows(seq_lens, 1, LONG_THRESHOLD)
    chose_lb = _lb_decision_from_counts(n_long, seq_lens.shape[0])

    # Warmup + JIT compile of every path, then verify correctness of each.
    idx_lb = run_gvr(logits, seq_lens, top_k, pre_idx, True)
    idx_no = run_gvr(logits, seq_lens, top_k, pre_idx, False)
    idx_auto = run_gvr(logits, seq_lens, top_k, pre_idx, "auto", num_long_rows=n_long)
    idx_radix = run_radix(logits, seq_lens, top_k)
    torch.cuda.synchronize()
    check_correct(idx_lb, logits, seq_lens, top_k)
    check_correct(idx_no, logits, seq_lens, top_k)
    check_correct(idx_auto, logits, seq_lens, top_k)
    check_correct(idx_radix, logits, seq_lens, top_k)

    # GVR LB uses dynamic counters => not CUDA-graph safe; use_cuda_graph=False.
    kw = dict(
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    lb_us = (
        float(
            np.median(
                bench_gpu_time(
                    fn=run_gvr,
                    input_args=(logits, seq_lens, top_k, pre_idx, True),
                    **kw,
                )
            )
        )
        * 1e3
    )
    no_us = (
        float(
            np.median(
                bench_gpu_time(
                    fn=run_gvr,
                    input_args=(logits, seq_lens, top_k, pre_idx, False),
                    **kw,
                )
            )
        )
        * 1e3
    )
    radix_us = (
        float(
            np.median(
                bench_gpu_time(fn=run_radix, input_args=(logits, seq_lens, top_k), **kw)
            )
        )
        * 1e3
    )
    # GVR LB heuristic picks between the two GVR paths (kernel cost of the chosen
    # path; the pure-host decision adds no device work).
    oracle_us = lb_us if chose_lb else no_us
    gvr_best = min(lb_us, no_us)  # best fixed GVR policy for this case
    return {
        "lb_us": lb_us,
        "no_us": no_us,
        "radix_us": radix_us,
        "oracle_us": oracle_us,
        "chose": "LB" if chose_lb else "noLB",
        # Speedups vs non-LB single-kernel GVR path (the LB heuristic's baseline).
        "lb_speedup": no_us / lb_us,
        "oracle_speedup": no_us / oracle_us,
        # Did the LB heuristic pick the genuinely faster fixed GVR path?
        "correct_pick": abs(oracle_us - gvr_best) < 1e-6,
        # Radix vs the heuristic-chosen GVR path: >1 means GVR (auto) is faster,
        # <1 means radix would have been the better backend for this case.
        "gvr_vs_radix": radix_us / oracle_us,
    }


# Varied workload spectrum. Weighted toward the ragged-decode regime LB targets
# (high length variance, a few rows above the 64K threshold, batch in a single
# wave), with two contrast cases at the end that deliberately defeat LB so the
# table shows the full tradeoff honestly.
def build_cases() -> list[LBCase]:
    return [
        # --- strong LB win: few very-long rows above threshold + many tiny ---
        LBCase("maxtail_1x128K+127x2K", 1, 131072, 127, 2048),
        LBCase("maxtail_8x128K+120x2K", 8, 131072, 120, 2048),
        LBCase("maxtail_16x128K+112x2K", 16, 131072, 112, 2048),
        LBCase("maxtail_8x256K+120x2K", 8, 262144, 120, 2048),
        # --- moderate variance, long rows comfortably above threshold ---
        LBCase("mid_8x96K+120x2K", 8, 98304, 120, 2048),
        LBCase("mid_24x128K+104x2K", 24, 131072, 104, 2048),
        LBCase("mid_32x128K+96x4K", 32, 131072, 96, 4096),
        # --- contrast: no length variance -> LB overhead not repaid ---
        LBCase("contrast_uniform_128x128K", 128, 131072, 0, 2048),
        LBCase("contrast_allshort_128x2K", 0, 131072, 128, 2048),
    ]


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def run_lb_comparison(dtype, device, args):
    """Table: GVR-LB vs GVR-noLB vs GVR-auto vs radix over the workload spectrum."""
    print("=" * 116)
    print(
        f"[LB comparison]  GVR-LB vs GVR-noLB vs GVR-auto vs radix   (top_k={args.top_k}, dtype={args.dtype})"
    )
    print(
        f"long_threshold={LONG_THRESHOLD} | LB speedups vs GVR-noLB baseline | gvr/radix = radix_us / gvr-auto_us"
    )
    print(
        f"pre_idx_overlap={args.pre_idx_overlap:.2f} (fraction of hints that are true top-K)"
    )
    print("=" * 116)
    header = (
        f"{'case':>26} {'rows':>5} {'#long':>6} "
        f"{'LB(us)':>8} {'noLB(us)':>8} {'auto(us)':>8} {'radix(us)':>9} "
        f"{'auto spdup':>11} {'ok':>3} {'gvr/radix':>10}"
    )
    print(header)
    print("-" * len(header))

    lb_speedups, auto_speedups, gvr_vs_radix = [], [], []
    n_correct = 0
    for case in build_cases():
        try:
            r = bench_case(case, args.top_k, dtype, device, args, args.pre_idx_overlap)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{case.name:>26} {case.num_rows:>5} {case.num_long:>6}  OOM")
                torch.cuda.empty_cache()
                continue
            raise
        lb_speedups.append(r["lb_speedup"])
        auto_speedups.append(r["oracle_speedup"])
        gvr_vs_radix.append(r["gvr_vs_radix"])
        n_correct += int(r["correct_pick"])
        print(
            f"{case.name:>26} {case.num_rows:>5} {case.num_long:>6} "
            f"{r['lb_us']:>8.2f} {r['no_us']:>8.2f} {r['oracle_us']:>8.2f} {r['radix_us']:>9.2f} "
            f"{r['oracle_speedup']:>10.2f}x {'Y' if r['correct_pick'] else 'N':>3} "
            f"{r['gvr_vs_radix']:>9.2f}x"
        )

    print("-" * len(header))
    if lb_speedups:
        total = len(lb_speedups)
        print(
            "  [1] GVR LB heuristic — geomean speedup vs GVR-noLB baseline (higher better):"
        )
        print(
            f"        always-LB : {geomean(lb_speedups):.3f}x   heuristic : {geomean(auto_speedups):.3f}x "
            f"  (picked faster path {n_correct}/{total})"
        )
        g_gr = geomean(gvr_vs_radix)
        worst = min(gvr_vs_radix)
        gvr_wins = sum(1 for x in gvr_vs_radix if x > 1.0)
        print(
            f"  [2] gvr-auto vs radix : geomean {g_gr:.2f}x, GVR faster in {gvr_wins}/{total} "
            f"(min {worst:.2f}x)"
        )
    print("=" * 116)


def _sweep_one(case, top_k, dtype, device, args):
    """Sweep overlap for one case; return (rows, radix_us). radix is hint-free."""
    overlaps = [1.0, 0.75, 0.5, 0.25, 0.0]
    kw = dict(
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    radix_us = None
    rows = []
    for ov in overlaps:
        logits, seq_lens, pre_idx = build_inputs(
            case, top_k, dtype, device, pre_idx_overlap=ov
        )
        i_lb = run_gvr(logits, seq_lens, top_k, pre_idx, True)
        i_no = run_gvr(logits, seq_lens, top_k, pre_idx, False)
        torch.cuda.synchronize()
        check_correct(i_lb, logits, seq_lens, top_k)
        check_correct(i_no, logits, seq_lens, top_k)
        lb_us = (
            float(
                np.median(
                    bench_gpu_time(
                        fn=run_gvr,
                        input_args=(logits, seq_lens, top_k, pre_idx, True),
                        **kw,
                    )
                )
            )
            * 1e3
        )
        no_us = (
            float(
                np.median(
                    bench_gpu_time(
                        fn=run_gvr,
                        input_args=(logits, seq_lens, top_k, pre_idx, False),
                        **kw,
                    )
                )
            )
            * 1e3
        )
        if radix_us is None:  # hint-independent; measure once
            i_rx = run_radix(logits, seq_lens, top_k)
            torch.cuda.synchronize()
            check_correct(i_rx, logits, seq_lens, top_k)
            radix_us = (
                float(
                    np.median(
                        bench_gpu_time(
                            fn=run_radix, input_args=(logits, seq_lens, top_k), **kw
                        )
                    )
                )
                * 1e3
            )
        rows.append((ov, no_us, lb_us, min(lb_us, no_us)))
    return rows, radix_us


def run_pre_idx_quality_sweep(dtype, device, args):
    """Sweep pre_idx quality (overlap with the true top-K) to test how much GVR's
    advantage over radix actually depends on the hint.

    GVR warm-starts its Phase-1 threshold from the logit VALUES at the pre_idx
    positions (their min/max/mean). ``overlap`` = fraction of the K hints that are
    the current step's true top-K: 1.0 = perfect (previous step == current step),
    0.0 = useless (random positions, modelling a fully drifted attention pattern).

    Two representative regimes are swept because they behave differently:
      * batched (128 rows): the fixed per-row scan dominates, so hint quality
        barely moves GVR time -- GVR's edge over radix is robust.
      * single long row: Phase-2 refinement is exposed, so hint quality (and
        even fp32 vs bf16) can swing GVR's time and, at the margin, let radix win.
    """
    regimes = [
        ("batched_8x128K+120x2K", LBCase("s", 8, 131072, 120, 2048)),
        ("single_1x128K", LBCase("s", 1, 131072, 0, 2048)),
    ]
    print("=" * 92)
    print(
        f"[pre_idx quality sweep]  GVR-vs-radix dependence on hint quality "
        f"(top_k={args.top_k}, dtype={args.dtype})"
    )
    print(
        "  overlap = fraction of the K hints that are the TRUE current-step top-K "
        "(1.0=perfect, 0.0=useless)"
    )
    print("=" * 92)

    all_ratios = []
    for name, case in regimes:
        rows, radix_us = _sweep_one(case, args.top_k, dtype, device, args)
        print(f"\n  {name}   (radix, hint-free = {radix_us:.2f} us)")
        print(
            f"    {'overlap':>8} {'GVR-noLB(us)':>13} {'GVR-LB(us)':>12} {'radix/GVR-best':>15}"
        )
        print("    " + "-" * 52)
        for ov, no_us, lb_us, best in rows:
            ratio = radix_us / best
            all_ratios.append(ratio)
            flag = "" if ratio >= 1.0 else "  <- radix wins"
            print(f"    {ov:>8.2f} {no_us:>13.2f} {lb_us:>12.2f} {ratio:>14.2f}x{flag}")

    print("\n" + "-" * 92)
    lo, hi = min(all_ratios), max(all_ratios)
    print(
        f"  radix/GVR-best spans {lo:.2f}x .. {hi:.2f}x across regimes and hint quality."
    )
    if lo >= 1.0:
        print(
            "  => GVR stays >= radix everywhere, even with a useless hint. No pre_idx-quality"
        )
        print("     backend heuristic is needed for these shapes.")
    else:
        print(
            f"  => radix overtakes GVR in the worst regime ({lo:.2f}x). GVR's runtime is NOT"
        )
        print(
            "     monotonic in hint quality (a perfect hint can seed a boundary-oscillating"
        )
        print(
            "     secant search); the crossover is narrow, so a static 'GVR when pre_idx'"
        )
        print("     policy is still reasonable, but this is the case to watch.")
    print("=" * 92)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Benchmark GVR top-k load-balancing")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--top-k", type=int, default=2048, choices=[512, 1024, 2048])
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument(
        "--op",
        choices=["all", "lb", "pre_idx_quality"],
        default="all",
        help="Which benchmark to run: lb comparison, pre_idx-quality sweep, or all.",
    )
    parser.add_argument(
        "--pre-idx-overlap",
        type=float,
        default=1.0,
        help="Fraction of pre_idx hints that are the true top-K, for the lb table "
        "(1.0=perfect hint; lower models a stale previous-step top-K).",
    )
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    cc = major * 10 + minor
    if not flashinfer.top_k_decode.is_backend_supported("gvr", cc):
        raise SystemExit(
            f"GVR backend not supported on sm_{cc} (needs Blackwell sm_100+ and CuTe DSL)."
        )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"# device sm_{cc}, {num_sms} SMs, pure GPU time via CUPTI\n")

    if args.op in ("all", "lb"):
        run_lb_comparison(dtype, device, args)
        print()
    if args.op in ("all", "pre_idx_quality"):
        run_pre_idx_quality_sweep(dtype, device, args)


if __name__ == "__main__":
    main()
