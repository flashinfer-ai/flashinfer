#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
AL-distribution-driven SSU checkpointing benchmark.

This is the in-process analogue of the original analytical tool: instead of
collapsing the stationary PNAT distribution into [0, T] buckets and running
homogeneous-PNAT batches one bucket at a time, we sample heterogeneous
per-slot ``prev_num_accepted_tokens`` vectors from the full [0, window]
chain and time the kernel on those mixed batches directly.

Pipeline
--------
1. Parse the AL (acceptance-length) histogram CSV.
2. Build the (window+1)x(window+1) Markov transition matrix from the AL
   distribution; solve for the stationary distribution pi over PNAT.
3. Cross-check via the "depletion sim" method (E[N] from PNAT=0).
4. Draw K independent (batch,)-shaped PNAT vectors from pi.
5. For each (batch, kernel, sample) tuple, call
   ``bench_checkpointing_ssu.time_kernel`` in-process and collect timings.
6. Aggregate per-sample medians into one steady-state row per kernel/batch.
7. Dump CSV and plot.

CSV layout
----------
One row per (kernel, batch, sample_idx).  ``sample_idx`` is an int in
[0, K) for per-sample rows; the aggregate row uses ``sample_idx="agg"``
(median across the K per-sample medians).

Usage
-----
  # Full run: collect data + plot
  python benchmarks/bench_ssu_checkpoint_mixed.py \\
      --al-csv benchmarks/histogram_T6.csv --batch-sizes 64

  # With conv1d (measures total span: conv1d + SSU, captures PDL overlap)
  python benchmarks/bench_ssu_checkpoint_mixed.py \\
      --al-csv benchmarks/histogram_T6.csv --batch-sizes 64 --with-conv1d

  # Plot only from existing CSV
  python benchmarks/bench_ssu_checkpoint_mixed.py \\
      --plot-only benchmarks/img/ssu_checkpoint_mixed_b64.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Import the in-process benchmarking helpers from bench_checkpointing_ssu.
SCRIPT_DIR = Path(__file__).parent
# Shared results dir with collect_checkpointing_ssu_runs.py:
# FLASHINFER_SSU_BENCH_OUTDIR if set, else cwd.
CSV_DIR = Path(os.environ.get("FLASHINFER_SSU_BENCH_OUTDIR", ".")).expanduser()
IMG_DIR = CSV_DIR / "img"
sys.path.insert(0, str(SCRIPT_DIR))

import bench_checkpointing_ssu as bench_ssu  # noqa: E402

DEFAULT_AL_CSV = SCRIPT_DIR / "histogram_T6.csv"
DEFAULT_BATCH_SIZES = [64]
DEFAULT_T = 6
DEFAULT_WINDOW = 16
DEFAULT_COLUMN = 1  # replay_count in histogram_T6.csv
DEFAULT_SEED = 42
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_NUM_PNAT_SAMPLES = 8
DEFAULT_KERNELS = "cuda-incr,cuda-incr-2k,triton-replay,triton-replay-pm"
# Kernels that honor Philox stochastic rounding on the gmem state writeback
# (the FlashInfer / baseline rows silently ignore rand_seed).  Includes the new
# 5D persistent paths (triton-replay = persistent_dynamic, -pm = persistent_main).
_PHILOX_KERNELS = (
    "cuda-incr",
    "cuda-incr-2k",
    "triton-replay",
    "triton-replay-pm",
)


# ---------------------------------------------------------------------------
# Analytical helpers — kept 1-to-1 with the upstream AL-distribution tool.
# Differences from the reference are limited to: (i) PEP 8 / type-hint
# cosmetics; (ii) the negative-AL check is hoisted out of `min()`.
# ---------------------------------------------------------------------------


def load_al_distribution(path: Path, T: int, column: int = 1) -> np.ndarray:
    """Return a length-(T+1) probability vector indexed by AL (0..T).

    Reads column 0 as AL, column ``column`` as count/probability.  Non-numeric
    rows (header, trailing summary rows like 'total'/'mean') are silently
    skipped.  The histogram is auto-normalized.
    """
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        sys.exit(f"empty CSV: {path}")
    al_to_count: dict[int, float] = {}
    for r in rows:
        if not r or not r[0].strip():
            continue
        try:
            al = int(float(r[0]))
            c = float(r[column])
        except (ValueError, IndexError):
            continue
        al_to_count[al] = al_to_count.get(al, 0.0) + c
    if not al_to_count:
        sys.exit(f"no numeric rows parsed from {path} (column index {column})")
    al_max = max(al_to_count)
    if al_max > T:
        sys.exit(
            f"CSV has AL={al_max} but --T={T} (a step processes only T tokens; "
            f"AL > T is impossible).  Mismatch likely indicates wrong --T or "
            f"wrong CSV."
        )
    if min(al_to_count) < 0:
        sys.exit(f"CSV has negative AL values (min={min(al_to_count)})")
    if al_to_count.get(0, 0) > 0:
        print(
            f"WARN: CSV has AL=0 mass ({al_to_count[0]:.4f} unnormalized).  "
            "AL=0 means no progress on a step; usually impossible in spec "
            "decoding (target token always accepted).  Treating as a real "
            "value (will produce a chain that doesn't converge if AL=0 has "
            "non-trivial mass).",
            file=sys.stderr,
        )
    dist = np.zeros(T + 1, dtype=np.float64)
    for al, c in al_to_count.items():
        dist[al] = c
    s = dist.sum()
    if s == 0:
        sys.exit(f"AL distribution sums to zero in {path}")
    return dist / s


def markov_stationary(al_dist: np.ndarray, T: int, window: int) -> np.ndarray:
    """Stationary distribution pi over PNAT in [0, window].

    Write decision: ``pnat + T > window``.  No-write: ``pnat_new = pnat + AL``.
    Write: ``pnat_new = AL`` (start fresh from staging buffer).
    """
    n = window + 1
    P = np.zeros((n, n))
    for p in range(n):
        is_write = p + T > window
        for al in range(1, T + 1):
            prob = al_dist[al]
            if prob == 0:
                continue
            p_new = al if is_write else p + al
            assert 0 <= p_new <= window, (
                f"unreachable transition: p={p} al={al} write={is_write} "
                f"-> p_new={p_new} outside [0, {window}]"
            )
            P[p, p_new] += prob
    # Solve pi P = pi via the left eigenvector for eigenvalue 1.
    # Equivalently: P^T pi = pi.
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[idx] - 1.0) > 1e-6:
        sys.exit(
            f"no stationary eigenvalue near 1 (closest is {eigvals[idx]}).  "
            "AL dist may not produce an ergodic chain."
        )
    pi = np.real(eigvecs[:, idx])
    # eig returns the Perron vector up to sign; flip an all-negative solve so
    # the clamp below doesn't zero it out into the uniform fallback.
    if pi.sum() < 0:
        pi = -pi
    # Iterative refinement: start near the eigenvector, then iterate
    # P^k to clean up any imaginary leakage from the eig solve.
    pi = np.maximum(pi, 0.0)
    if pi.sum() == 0:
        pi = np.ones(n) / n
    for _ in range(2000):
        new_pi = pi @ P
        if np.allclose(new_pi, pi, atol=1e-12, rtol=0):
            pi = new_pi
            break
        pi = new_pi
    return pi / pi.sum()


def sample_steady_state_pnat(
    al_dist: np.ndarray,
    T: int,
    window: int,
    batch: int,
    K: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw K independent (batch,)-shaped PNAT vectors from pi.

    Returns int64 array of shape (K, batch) with values in [0, window].
    PNAT=0 has weight ~0 in steady state so it is effectively never sampled
    (it's purely a boot-up state).
    """
    pi = markov_stationary(al_dist, T, window)
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()
    states = np.arange(window + 1)
    rng = np.random.default_rng(seed)
    return rng.choice(states, size=(K, batch), p=pi).astype(np.int64)


def depletion_sim(
    al_dist: np.ndarray, T: int, window: int, max_steps: int = 256
) -> np.ndarray:
    """Start mass 1 at PNAT=0; record checkpoint mass per step.

    At each step, mass at PNAT > WINDOW-T is removed (= it checkpoints this
    step).  Remaining mass advances by AL.  Returns array ``removed[step-1]``
    = mass that checkpointed at step.
    """
    n = window + 1
    mass = np.zeros(n)
    mass[0] = 1.0
    write_thresh = window - T  # PNAT > this => write
    removed = []
    for _ in range(max_steps):
        write_mass = mass[write_thresh + 1 :].sum()
        removed.append(float(write_mass))
        # Strip the mass that checkpointed (it's been "removed").
        mass = mass.copy()
        mass[write_thresh + 1 :] = 0.0
        # Propagate the rest by AL.
        new_mass = np.zeros(n)
        for p in range(write_thresh + 1):
            if mass[p] == 0:
                continue
            for al in range(1, T + 1):
                prob = al_dist[al]
                if prob == 0:
                    continue
                p_new = p + al
                assert p_new <= window, "unreachable"
                new_mass[p_new] += mass[p] * prob
        mass = new_mass
        if mass.sum() < 1e-15:
            break
    return np.array(removed)


def _format_pi(pi: np.ndarray, T: int, window: int) -> str:
    write_thresh = window - T
    lines = []
    for p, prob in enumerate(pi):
        if prob < 1e-9:
            continue
        marker = "  <- write state" if p > write_thresh else ""
        lines.append(f"  pi(PNAT={p:2d}) = {prob:.4f}{marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# In-process benchmark loop
# ---------------------------------------------------------------------------


# CSV "sample_idx" sentinel for the aggregate-over-K row.
AGG_SENTINEL = "agg"


# State-dtype parsing is delegated to bench_checkpointing_ssu so both
# scripts accept the same `<dtype>-philox-<N>` syntax.


def _aggregate_stats(
    stats: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Median-of-medians, median-of-p95, median-of-p99 across K samples.

    We don't keep individual per-iter latencies (the inner timing path only
    returns summary tuples), so the aggregate is computed elementwise.
    """
    medians = sorted(s[0] for s in stats)
    p95s = sorted(s[1] for s in stats)
    p99s = sorted(s[2] for s in stats)
    mid = len(stats) // 2
    return medians[mid], p95s[mid], p99s[mid]


def run_in_process_bench(
    *,
    al_csv: Path,
    column: int,
    T: int,
    window: int,
    batch_sizes: list[int],
    state_dtype_spec: str,
    kernels: list[str],
    K: int,
    seed: int,
    warmup: int,
    iters: int,
    cupti: bool,
    nheads: int,
    head_dim: int,
    d_state: int,
    ngroups: int,
    act_dtype: torch.dtype = torch.bfloat16,
    with_conv1d: bool = False,
    external_pdl: bool = True,
    cuda_graph: bool = True,
    varlen: bool = False,
) -> tuple[list[dict], dict]:
    """Collect rows for one full sweep.  Returns ``(rows, meta)``.

    ``state_dtype_spec`` follows the bench_checkpointing_ssu convention:
    plain dtype name (e.g. ``"e4m3"``) for philox_rounds=0, or
    ``"<dtype>-philox-<N>"`` to enable Philox-<N> stochastic rounding on
    gmem state writeback (the CUDA kernels + Triton replay;
    see _PHILOX_KERNELS).

    When *with_conv1d* is True, each kernel iteration prepends a
    causal_conv1d_update call and measures the total span (conv1d + SSU),
    capturing PDL overlap.  Uses a realistic reset: cold cache → L2 flush →
    hot xbc_input.

    ``meta`` holds the analytical fields (e_al, write_frac, pi, weights)
    so callers can include them in plot titles / printed summaries.
    """
    al = load_al_distribution(al_csv, T, column=column)
    pi = markov_stationary(al, T, window)
    write_frac = float(pi[window - T + 1 :].sum())
    e_al = float(sum(i * p for i, p in enumerate(al)))

    state_dtype, philox_rounds, state_label = bench_ssu.parse_state_spec(
        state_dtype_spec
    )
    # The two-kernel split ("cuda-incr-2k") takes 2-byte (fp16/bf16, LDSM) and
    # 4-byte (f32, LDS.64 + in-register narrow) state; only quantized
    # (fp8/int8) state has no split (matches bench_checkpointing_ssu.py).
    if "cuda-incr-2k" in kernels and state_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        print(
            f"NOTE: dropping cuda-incr-2k — two-kernel split needs 2- or 4-byte "
            f"state, got {state_label}.",
            file=sys.stderr,
        )
        kernels = [k for k in kernels if k != "cuda-incr-2k"]
    if varlen:
        dropped = [k for k in kernels if k not in ("cuda-incr", "cuda-incr-2k")]
        if dropped:
            print(
                f"NOTE: --varlen keeps only the CUDA kernels (the Triton "
                f"references take no cu_seqlens); dropping {dropped}.",
                file=sys.stderr,
            )
            kernels = [k for k in kernels if k in ("cuda-incr", "cuda-incr-2k")]
    rand_seed = (
        torch.tensor([0xDECAFBAD], device="cuda", dtype=torch.int64)
        if philox_rounds > 0
        else None
    )
    if philox_rounds > 0:
        # Philox is honored by the kernels in _PHILOX_KERNELS; the
        # FlashInfer / baseline kernels silently ignore it.  Warn so users
        # don't think their philox setting affected those rows.
        ignoring = [k for k in kernels if k not in _PHILOX_KERNELS]
        if ignoring:
            print(
                f"WARN: philox_rounds={philox_rounds} only applies to "
                f"the CUDA / Triton-replay kernels; kernels {ignoring} run without it.",
                file=sys.stderr,
            )

    # Cross-check write_frac via depletion sim.
    removed = depletion_sim(al, T, window)
    total = float(removed.sum())
    e_step = float(sum((i + 1) * r for i, r in enumerate(removed)) / max(total, 1e-12))
    cycle = e_step - 1.0
    write_frac_dep = (1.0 / cycle) if cycle > 0 else float("inf")

    print("=" * 72)
    print(
        f"state_dtype = {state_label} (torch={state_dtype}, philox_rounds={philox_rounds})"
    )
    print(f"AL distribution (T={T}, window={window}):")
    for i, p in enumerate(al):
        if p > 0:
            print(f"  AL={i}: {p:.4f}")
    print(f"  E[AL] = {e_al:.4f}")
    print()
    print("Stationary PNAT distribution (Markov chain, [0, window]):")
    print(_format_pi(pi, T, window))
    print(f"  write_frac (Markov)    = {write_frac:.6f}")
    print(f"  write_frac (depletion) = {write_frac_dep:.6f}")
    print("=" * 72)
    rel = abs(write_frac - write_frac_dep) / max(write_frac, 1e-12)
    if rel > 0.01:
        print(
            f"WARN: Markov vs depletion disagree by {rel:.2%}. "
            "Check AL distribution for AL=0 mass or non-ergodic chain.",
            file=sys.stderr,
        )
    if with_conv1d:
        print(
            f"conv1d: ENABLED (external_pdl={external_pdl}) — "
            "measuring total span (conv1d + SSU)"
        )
    else:
        print("conv1d: disabled — measuring SSU kernel only")
    if varlen:
        print(
            "varlen: ENABLED — packed (1, batch*T) layout + uniform cu_seqlens; "
            "same work as dense, only the VARLEN addressing path differs"
        )

    # Pre-init L2 flush buffer (bench_checkpointing_ssu's CUDA-graph timing
    # path assumes this is set up when TimingOptions.l2_flush is True).
    bench_ssu._init_l2_flush()
    timing = bench_ssu.TimingOptions(
        warmup=warmup, iters=iters, cupti=cupti, cuda_graph=cuda_graph, l2_flush=True
    )

    rows: list[dict] = []

    for batch in batch_sizes:
        print(f"\n[batch={batch}] sampling K={K} PNAT vectors...")
        pnats_np = sample_steady_state_pnat(al, T, window, batch, K, seed=seed)
        # Per-sample pre-built int32 tensors.  Built on the host once and copied
        # to the inputs.prev_tokens_* tensors at time_kernel() time.
        pnats_i32 = [
            torch.from_numpy(pnats_np[k].astype(np.int32)).cuda() for k in range(K)
        ]

        # Build kernel inputs once per (batch, dtype) — reused across K and kernels.
        inputs = bench_ssu.build_kernel_inputs(
            batch=batch,
            mtp_len=T,
            max_window=window,
            state_dtype=state_dtype,
            act_dtype=act_dtype,
            nheads=nheads,
            head_dim=head_dim,
            d_state=d_state,
            ngroups=ngroups,
            varlen=varlen,
        )

        for kernel in kernels:
            print(f"[batch={batch}] timing kernel={kernel}", flush=True)
            per_sample_stats: list[tuple[float, float, float]] = []
            for k in range(K):
                pt = pnats_i32[k]
                tag = f"mixed_b{batch}_T{T}_W{window}_{kernel}_s{k}"
                # philox applies only to the kernels in _PHILOX_KERNELS.
                kernel_philox = philox_rounds if kernel in _PHILOX_KERNELS else 0
                kernel_rand_seed = rand_seed if kernel_philox > 0 else None
                if with_conv1d:
                    median_us, p95_us, p99_us = bench_ssu.time_kernel_with_conv1d(
                        kernel=kernel,
                        inputs=inputs,
                        prev_tokens=pt,
                        timing=timing,
                        tag=tag,
                        philox_rounds=kernel_philox,
                        rand_seed=kernel_rand_seed,
                        external_pdl=external_pdl,
                    )
                else:
                    median_us, p95_us, p99_us = bench_ssu.time_kernel(
                        kernel=kernel,
                        inputs=inputs,
                        prev_tokens=pt,
                        timing=timing,
                        tag=tag,
                        philox_rounds=kernel_philox,
                        rand_seed=kernel_rand_seed,
                    )
                per_sample_stats.append((median_us, p95_us, p99_us))
                rows.append(
                    {
                        "kernel": kernel,
                        "batch": batch,
                        "mtp_len": T,
                        "window": window,
                        "sample_idx": k,
                        "state_dtype": state_label,
                        "act_dtype": str(act_dtype).split(".")[-1],
                        "varlen": varlen,
                        "median_us": median_us,
                        "p95_us": p95_us,
                        "p99_us": p99_us,
                        "write_frac": write_frac,
                        "e_al": e_al,
                    }
                )

            agg_median, agg_p95, agg_p99 = _aggregate_stats(per_sample_stats)
            rows.append(
                {
                    "kernel": kernel,
                    "batch": batch,
                    "mtp_len": T,
                    "window": window,
                    "sample_idx": AGG_SENTINEL,
                    "state_dtype": state_label,
                    "act_dtype": str(act_dtype).split(".")[-1],
                    "varlen": varlen,
                    "median_us": agg_median,
                    "p95_us": agg_p95,
                    "p99_us": agg_p99,
                    "write_frac": write_frac,
                    "e_al": e_al,
                }
            )
            print(
                f"  [agg over K={K}] median={agg_median:.2f}us "
                f"p95={agg_p95:.2f}us p99={agg_p99:.2f}us"
            )

    meta = {
        "al": al,
        "pi": pi,
        "write_frac": write_frac,
        "write_frac_dep": write_frac_dep,
        "e_al": e_al,
        "T": T,
        "window": window,
        "K": K,
    }
    # Finalize CUPTI ONCE, after the whole sweep — never per timing call (cuptiFinalize
    # is once-per-process; finalizing per call corrupted CUPTI into an illegal memory
    # access after ~40 calls).  No-op when --no-cupti / cupti not installed.
    if cupti:
        bench_ssu.finalize_cupti()
    return rows, meta


# ---------------------------------------------------------------------------
# CSV IO
# ---------------------------------------------------------------------------


_CSV_FIELDNAMES = [
    "kernel",
    "batch",
    "mtp_len",
    "window",
    "sample_idx",
    "state_dtype",
    "act_dtype",
    "varlen",
    "median_us",
    "p95_us",
    "p99_us",
    "write_frac",
    "e_al",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["batch"] = int(r["batch"])
            r["mtp_len"] = int(r["mtp_len"])
            if r.get("window"):
                r["window"] = int(r["window"])
            for k in ("median_us", "p95_us", "p99_us"):
                r[k] = float(r[k])
            if r.get("write_frac"):
                r["write_frac"] = float(r["write_frac"])
            if r.get("e_al"):
                r["e_al"] = float(r["e_al"])
            # sample_idx stays as a string ("agg") or int-as-string
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(rows: list[dict], png_path: Path) -> None:
    """Scatter per-sample latencies, overlay aggregate line per kernel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_sample = [r for r in rows if str(r["sample_idx"]) != AGG_SENTINEL]
    aggregate = [r for r in rows if str(r["sample_idx"]) == AGG_SENTINEL]
    if not per_sample:
        print("no per-sample rows to plot", file=sys.stderr)
        return

    colors = {
        "cuda-incr": "#d62728",
        "fi-dump": "#2ca02c",
        "baseline-triton": "#9467bd",
        "baseline-flashinfer": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-sample dots: light, small.
    by_kernel: dict[str, list[tuple[int, float]]] = {}
    for r in per_sample:
        by_kernel.setdefault(r["kernel"], []).append((r["batch"], r["median_us"]))
    for kernel, pts in by_kernel.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(
            xs,
            ys,
            color=colors.get(kernel, "gray"),
            marker="o",
            s=18,
            alpha=0.4,
            edgecolors="none",
            label=f"{kernel} (per-sample)",
        )

    # Aggregate line: heavy, marker with black edge.
    agg_by_kernel: dict[str, list[tuple[int, float]]] = {}
    for r in aggregate:
        agg_by_kernel.setdefault(r["kernel"], []).append((r["batch"], r["median_us"]))
    for kernel, pts in agg_by_kernel.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            color=colors.get(kernel, "gray"),
            marker="o",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=1.2,
            linewidth=2.5,
            label=f"{kernel} (steady-state median)",
            zorder=5,
        )

    wf = next((r.get("write_frac") for r in aggregate if r.get("write_frac")), None)
    e_al = next((r.get("e_al") for r in aggregate if r.get("e_al")), None)
    title = "SSU checkpointing-mixed: per-sample median latency vs batch"
    if wf is not None and e_al is not None:
        title += f"  (E[AL]={e_al:.2f}, write_frac={wf:.3f})"

    ax.set_xlabel("batch")
    ax.set_ylabel("Median latency (us)")
    ax.set_title(title)
    batch_vals = sorted({r["batch"] for r in per_sample})
    if len(batch_vals) > 1:
        ax.set_xscale("log", base=2)
        ax.set_xticks(batch_vals)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AL-distribution-driven SSU checkpointing bench (heterogeneous PNATs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--al-csv", type=Path, default=DEFAULT_AL_CSV)
    parser.add_argument("--column", type=int, default=DEFAULT_COLUMN)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--state-dtype",
        default="bf16",
        help=(
            "State dtype.  Plain name (e.g. 'e4m3', 'bf16') or "
            "'<dtype>-philox-<N>' (e.g. 'e4m3-philox-5') to enable Philox-<N> "
            "stochastic rounding on gmem state writeback.  "
            "Valid suffix dtypes: f16/fp16/int8/i8/fp8/e4m3."
        ),
    )
    parser.add_argument(
        "--kernels",
        default=DEFAULT_KERNELS,
        help=(
            "Comma-separated kernel names from bench_checkpointing_ssu.KernelName: "
            "cuda-incr, cuda-incr-2k, triton-replay, triton-replay-pm, fi-dump, "
            "baseline-triton, baseline-flashinfer"
        ),
    )
    parser.add_argument(
        "--num-pnat-samples",
        "-K",
        type=int,
        default=DEFAULT_NUM_PNAT_SAMPLES,
        help="Number of independent PNAT vectors to draw per (batch, kernel).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--cupti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUPTI hardware-level timing.  Pass --no-cupti when profiling "
        "under ncu (CUPTI and ncu both grab the GPU profiling API and conflict).",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture timed iterations in a CUDA graph.  Pass --no-cuda-graph "
        "when profiling under ncu for clean per-launch kernel capture.",
    )
    # Model dims (default = Nemotron @ TP=8).
    parser.add_argument(
        "--nheads", type=int, default=bench_ssu.NHEADS // bench_ssu.TP_SIZE
    )
    parser.add_argument("--head-dim", type=int, default=bench_ssu.HEAD_DIM)
    parser.add_argument("--d-state", type=int, default=bench_ssu.D_STATE)
    parser.add_argument(
        "--ngroups", type=int, default=bench_ssu.NGROUPS // bench_ssu.TP_SIZE
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help='Filename stem for the CSV/PNG.  Pass "-" to skip writing both '
        "(useful for throwaway optimization sweeps — keeps poll-watched dirs clean).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Version/label woven into the auto output filename, e.g. --tag v0.9 → "
        "ssu_checkpoint_mixed_v0.9_b..._conv1d.  Ignored if --output-prefix is set.",
    )
    parser.add_argument(
        "--plot-only", type=Path, default=None, help="Existing CSV to plot from."
    )
    # Conv1d combined timing (measures total span: conv1d + SSU, captures PDL overlap)
    parser.add_argument(
        "--with-conv1d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include conv1d kernel before SSU.  Measures total span (conv1d + SSU) "
            "to capture PDL overlap — important for production latency estimation.  "
            "Uses realistic reset: cold cache → L2 flush → hot in_proj output.  "
            "Pass --no-with-conv1d to time SSU only (default)."
        ),
    )
    parser.add_argument(
        "--external-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable external PDL: conv1d launches dependent kernels so SSU can "
            "start fetching state while conv1d is still finishing.  "
            "Only relevant with --with-conv1d.  --no-external-pdl disables."
        ),
    )
    parser.add_argument(
        "--varlen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pack inputs into the varlen (1, batch*T, ...) + cu_seqlens layout "
            "with uniform seq_len = T — an exact A/B against dense (same work, "
            "only the VARLEN addressing path differs).  CUDA kernels only.  "
            "Composes with --with-conv1d: uniform rows mean the conv1d's dense "
            "output IS the packed layout (the SSU consumes packed views of it)."
        ),
    )
    args = parser.parse_args()

    if args.window <= args.T:
        parser.error(f"--T ({args.T}) must be < --window ({args.window})")

    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]

    prefix = args.output_prefix or (
        f"ssu_checkpoint_mixed"
        f"{('_' + args.tag) if args.tag else ''}"
        f"_b{'_'.join(str(b) for b in args.batch_sizes)}"
        f"_{args.state_dtype}_K{args.num_pnat_samples}"
        f"{'_conv1d' if args.with_conv1d else ''}"
        f"{'_varlen' if args.varlen else ''}"
    )
    csv_path = CSV_DIR / f"{prefix}.csv"
    png_path = IMG_DIR / f"{prefix}.png"

    if args.plot_only is not None:
        csv_path = args.plot_only
        png_path = csv_path.with_suffix(".png")
        rows = read_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")
        plot_results(rows, png_path)
        return

    if args.with_conv1d and bench_ssu._causal_conv1d_update is None:
        parser.error(
            "--with-conv1d requires causal_conv1d_update but the import from "
            "tests/mamba/triton_reference/causal_conv1d_triton.py failed"
        )

    rows, _meta = run_in_process_bench(
        al_csv=args.al_csv,
        column=args.column,
        T=args.T,
        window=args.window,
        batch_sizes=args.batch_sizes,
        state_dtype_spec=args.state_dtype,
        kernels=args.kernels,
        K=args.num_pnat_samples,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        cupti=args.cupti,
        nheads=args.nheads,
        head_dim=args.head_dim,
        d_state=args.d_state,
        ngroups=args.ngroups,
        with_conv1d=args.with_conv1d,
        external_pdl=args.external_pdl,
        cuda_graph=args.cuda_graph,
        varlen=args.varlen,
    )

    if not rows:
        sys.exit("no rows collected")

    # Print aggregate summary.
    print("\nSteady-state summary (median across K samples):")
    print(
        f"| {'kernel':>20} | {'batch':>5} | {'median_us':>10} | "
        f"{'p95_us':>8} | {'p99_us':>8} |"
    )
    print(f"|{'-' * 22}|{'-' * 7}|{'-' * 12}|{'-' * 10}|{'-' * 10}|")
    for r in sorted(
        rows, key=lambda r: (r["kernel"], r["batch"], str(r["sample_idx"]))
    ):
        if str(r["sample_idx"]) != AGG_SENTINEL:
            continue
        print(
            f"| {r['kernel']:>20} | {r['batch']:>5} | {r['median_us']:>10.2f} | "
            f"{r['p95_us']:>8.2f} | {r['p99_us']:>8.2f} |"
        )

    if args.output_prefix == "-":
        print('\noutput-prefix is "-": skipping CSV/PNG write.')
    else:
        write_csv(rows, csv_path)
        plot_results(rows, png_path)


if __name__ == "__main__":
    main()
