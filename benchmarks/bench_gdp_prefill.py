# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time ``chunk_gated_delta_product`` prefill calls by CUDA graph replay, Cake against cuDNN.

Examples (run on an otherwise idle allocated GPU):
  python benchmarks/bench_gdp_prefill.py --output gdp.jsonl
  python benchmarks/bench_gdp_prefill.py --families A --heads 80 --lengths 8192 --short 0,1,15

The workloads (K = 128, V = 64, n = 3, scale 1/sqrt(128), BF16 q / k / v / o,
FP32 g / beta / state, q / k L2-normalized beforehand, final state stored to a
separate buffer):

* A: ``d`` one-token sequences with an initial state, then one sequence of
  ``min(L, 65536 - d)`` tokens with a zero initial state; H in {10, 20, 40, 80},
  L in {8192, 16384, 32768, 65536}, d in {0, 1, 4, 8, 15}.
* B: one sequence of 65536 tokens with an initial state.
* C: a sequence of 36864 tokens with an initial state, then one of 28672
  tokens with a zero initial state.

Each backend's call is captured once into a CUDA graph (outputs into the
caller's buffers; the initial state is read only, so replays are idempotent)
after eager warm-up calls.  A sample is one replay between two CUDA events,
replays back to back (warm L2); ``--cold-l2`` flushes L2 before each sample,
outside the events.  The d variants of one (H, L) are captured together and
timed paired: in each of ``--replays / --block`` rounds every variant (rotating,
backwards every other round, so no block always follows the same one) times one
block of ``--block`` replays per backend back to back (A B, then B A two rounds
later), so both backends of a pair and all variants see the same clock and
thermal state; the SM clock is read through NVML while each block runs.  A row's
speedup is the median over its pairs of cuDNN's block median / Cake's block
median, reported with p10, p90 and a distribution-free 95% confidence interval;
an interval containing 1.0 is "parity (noise)".  The report also gives the
median, p10 and p90 of each backend's samples, the geometric mean per family, and per
(H, L) of family A the cost of each extra one-token sequence,
``(t(d) - t(0)) / d`` per round (Cake's d block against its d = 0 block of the
same round, with a 95% interval), against ``max(1% c(0), 2 us)`` where ``c(0)``
is cuDNN's time at ``d = 0``.  Each row is also judged against the request's bar: Cake
within ``c(0) + d max(1% c(0), 2 us)`` (B and C: within cuDNN's time), paired
per round with that round's cuDNN ``d = 0`` block: within, over, or parity
(noise) when the interval of bar / Cake contains 1.0.
``--gates`` picks the gate distribution: ``gfi`` (FlashInfer's test generator,
``exp(-softplus(0.5 x - 2))``) or ``gsat`` (fast decay, ``exp(-softplus(x))``).
``--check`` compares the Cake output and final state with cuDNN's (Frobenius
relative error).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys

import torch
import torch.nn.functional as F

from flashinfer.gdp_prefill import chunk_gated_delta_product

K, V, N_HH = 128, 64, 3
REL_TOL = 1e-2  # FlashInfer's kernel tolerance (Frobenius relative error) against cuDNN
TOTAL = 65536
# the 95% confidence interval of a row's paired cuDNN / Cake ratio (or bar / Cake) against 1.0
SPEED_VERDICT = {
    "above": "faster",
    "below": "slower",
    "parity (noise)": "parity (noise)",
}
BAR_VERDICT = {"above": "within", "below": "over", "parity (noise)": "parity (noise)"}


def workloads(families, heads, lengths, short):
    for H in heads:
        if "A" in families:
            for L in lengths:
                for d in short:
                    yield {
                        "family": "A",
                        "name": f"A_H{H}_L{L}_d{d}",
                        "H": H,
                        "L": L,
                        "d": d,
                        "lengths": [1] * d + [min(L, TOTAL - d)],
                        "state": [1] * d + [0],
                    }
        if "B" in families:
            yield {
                "family": "B",
                "name": f"B_H{H}",
                "H": H,
                "lengths": [TOTAL],
                "state": [1],
            }
        if "C" in families:
            yield {
                "family": "C",
                "name": f"C_H{H}",
                "H": H,
                "lengths": [36864, 28672],
                "state": [1, 0],
            }


GATES = {
    "gfi": lambda x: torch.exp(-F.softplus(0.5 * x - 2.0)),
    "gsat": lambda x: torch.exp(-F.softplus(x)),
}


def make_inputs(lengths, state_mask, H, seed, gates="gfi"):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lengths)

    def unit(rows):
        x = torch.randn(rows, H, K, generator=gen, device="cuda")
        return F.normalize(x, dim=-1).to(torch.bfloat16)

    state = 0.1 * torch.randn(len(lengths), H, V, K, generator=gen, device="cuda")
    state *= torch.tensor(state_mask, dtype=torch.float32, device="cuda")[
        :, None, None, None
    ]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return {
        "q": unit(T),
        "k": unit(T * N_HH),
        "v": torch.randn(T * N_HH, H, V, generator=gen, device="cuda").to(
            torch.bfloat16
        ),
        "g": GATES[gates](torch.randn(T, H, generator=gen, device="cuda")),
        "beta": torch.rand(T * N_HH, H, generator=gen, device="cuda"),
        "initial_state": state.contiguous(),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cuda"),
        "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int32),
    }


def make_call(inputs, backend, out, final):
    kwargs = dict(
        num_householder=N_HH,
        scale=K**-0.5,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        output=out,
        output_state=final,
        backend=backend,
    )
    if backend == "cake":
        kwargs["cu_seqlens_cpu"] = inputs["cu_seqlens_cpu"]

    def call():
        return chunk_gated_delta_product(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], **kwargs
        )

    return call


def rel_err(actual, expected):
    a, e = actual.float(), expected.float()
    return ((a - e).norm() / e.norm().clamp_min(1e-6)).item()


def capture(call, args):
    """Eager warm-up calls, then one CUDA graph of ``call`` and ``args.warmup`` replays."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def sm_clock_reader():
    """A function returning the SM clock (MHz) of the current device through NVML, or ``None`` without NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
        index = torch.cuda._get_nvml_device_index(torch.cuda.current_device())
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except Exception:  # no NVML: the report carries no clocks
        return None


def time_block(graph, count, flush=None, clock=None):
    """One untimed replay (so the device is busy while the timed ones are queued), then ``count`` replays of
    ``graph``, each between two CUDA events (us).  Returns the samples and the median SM clock (MHz) read while
    the block ran, or ``None``."""
    start = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    graph.replay()
    for i in range(count):
        if flush is not None:
            flush.zero_()
        start[i].record()
        graph.replay()
        end[i].record()
    mhz = []
    if clock is not None:
        while not end[-1].query():
            mhz.append(clock())
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(start, end, strict=True)]
    return samples, (statistics.median(mhz) if mhz else None)


def quantile(values, f):
    values = sorted(values)
    return values[min(len(values) - 1, int(f * len(values)))]


def stats(samples):
    return {
        "median_us": statistics.median(samples),
        "p10_us": quantile(samples, 0.1),
        "p90_us": quantile(samples, 0.9),
        "replays": len(samples),
    }


def ratio_stats(ratios):
    """p10 / p50 / p90 of paired ratios and the distribution-free 95% confidence interval of their median, with
    the verdict against 1.0: ``above`` / ``below`` when the interval excludes 1.0, else ``parity (noise)``."""
    ratios = sorted(ratios)
    n = len(ratios)
    k = max(0, int((n - 1.96 * math.sqrt(n)) / 2))
    lo, hi = ratios[k], ratios[n - 1 - k]
    return {
        "pairs": n,
        "p10": quantile(ratios, 0.1),
        "p50": statistics.median(ratios),
        "p90": quantile(ratios, 0.9),
        "ci95": [lo, hi],
        "verdict": "above" if lo > 1.0 else "below" if hi < 1.0 else "parity (noise)",
    }


def run_group(group, args, flush, clock):
    """Capture every workload of ``group`` (the d variants of one (H, L), or one B / C workload) on every
    backend, then time them paired: in each round every variant (first variant rotating) times one block of
    ``args.block`` replays per backend back to back, the backend order alternating per round (A B, then B A), so
    both backends of a pair see the same clock and thermal state.  A pair's speedup is the cuDNN block median over
    the Cake block median; the request bar of a round compares the Cake block with the same round's cuDNN d = 0
    block."""
    # a CUDA graph holds no reference to the tensors it reads: every workload's inputs stay alive until the
    # group's replays are done (else the caching allocator hands their memory to the next workload)
    rows, graphs, results, alive = [], {}, {}, []
    for w in group:
        inputs = make_inputs(w["lengths"], w["state"], w["H"], args.seed, args.gates)
        alive.append(inputs)
        T, N, H = sum(w["lengths"]), len(w["lengths"]), w["H"]
        row = {k: w[k] for k in ("family", "name", "H", "lengths") if k in w}
        row.update({k: w[k] for k in ("L", "d") if k in w})
        for backend in args.backends:
            out = torch.empty(T, H, V, dtype=torch.bfloat16, device="cuda")
            final = torch.empty(N, H, V, K, dtype=torch.float32, device="cuda")
            try:
                graphs[(w["name"], backend)] = capture(
                    make_call(inputs, backend, out, final), args
                )
                results[(w["name"], backend)] = (out, final)
            except Exception as exc:  # report a declined backend and go on
                row[backend] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
    rounds = max(1, args.replays // args.block)
    samples = {key: [] for key in graphs}
    blocks = {key: [] for key in graphs}  # per round: (median us, SM MHz)
    names = [w["name"] for w in group]
    for r in range(rounds):
        # the variants rotate and every other round run backwards, so a block's predecessor is not always the same
        # variant (a power-capped device times a block by the heat of the one before it); the backend order flips
        # every two rounds, independent of the variant order
        order = args.backends if (r // 2) % 2 == 0 else args.backends[::-1]
        variants = names[r % len(names) :] + names[: r % len(names)]
        for name in variants[::-1] if r % 2 else variants:
            for backend in order:
                key = (name, backend)
                if key in graphs:
                    got, mhz = time_block(graphs[key], args.block, flush, clock)
                    samples[key].extend(got)
                    blocks[key].append((statistics.median(got), mhz))
    base = {
        w["family"] + str(w.get("L")): w["name"] for w in group if w.get("d", 0) == 0
    }
    for row in rows:
        name = row["name"]
        for backend in args.backends:
            if (name, backend) in samples:
                row[backend] = stats(samples[(name, backend)])
                mhz = [m for _, m in blocks[(name, backend)] if m is not None]
                if mhz:
                    row[backend]["sm_mhz"] = {
                        "p10": quantile(mhz, 0.1),
                        "p50": statistics.median(mhz),
                        "p90": quantile(mhz, 0.9),
                    }
        cake, cudnn = blocks.get((name, "cake")), blocks.get((name, "cudnn"))
        if cake and cudnn:
            row["paired"] = ratio_stats(
                [c[0] / k[0] for k, c in zip(cake, cudnn, strict=True)]
            )
            row["speedup"] = row["paired"]["p50"]
            row["speedup_of_medians"] = (
                row["cudnn"]["median_us"] / row["cake"]["median_us"]
            )
            c0 = blocks.get((base.get(row["family"] + str(row.get("L"))), "cudnn"))
            if c0:
                d = row.get("d", 0)
                row["bar_paired"] = ratio_stats(
                    [
                        (c[0] + d * max(args.budget_pct / 100.0 * c[0], args.budget_us))
                        / k[0]
                        for k, c in zip(cake, c0, strict=True)
                    ]
                )
            b0 = blocks.get((base.get(row["family"] + str(row.get("L"))), "cake"))
            if b0 and row.get("d", 0) > 0:
                d = row["d"]
                slope = sorted((k[0] - z[0]) / d for k, z in zip(cake, b0, strict=True))
                n = len(slope)
                q = max(0, int((n - 1.96 * math.sqrt(n)) / 2))
                row["slope_paired"] = {
                    "p10": quantile(slope, 0.1),
                    "p50": statistics.median(slope),
                    "p90": quantile(slope, 0.9),
                    "ci95": [slope[q], slope[n - 1 - q]],
                }
            if args.check:
                (o, s_), (ro, rs) = results[(name, "cake")], results[(name, "cudnn")]
                row["rel_err_out"] = rel_err(o, ro)
                row["rel_err_state"] = rel_err(s_, rs)
                row["finite"] = bool(
                    torch.isfinite(o).all() and torch.isfinite(s_).all()
                )
    del graphs, results, alive
    torch.cuda.empty_cache()
    return rows


def summarize(rows, budget_pct, budget_us):
    lines = []
    for family in ("A", "B", "C"):
        speedups = [
            r["speedup"] for r in rows if r["family"] == family and "speedup" in r
        ]
        if speedups:
            geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            verdicts = collections.Counter(
                SPEED_VERDICT[r["paired"]["verdict"]]
                for r in rows
                if r["family"] == family and "paired" in r
            )
            lines.append(
                f"family {family}: {len(speedups)} workloads, geomean speedup {geo:.3f}, "
                f"min {min(speedups):.3f}, max {max(speedups):.3f}"
                + (f", {dict(verdicts)}" if verdicts else "")
            )

    def median(r, backend):
        return r.get(backend, {}).get("median_us")

    def base(r):
        if r["family"] != "A":
            return r
        return next(
            (
                b
                for b in rows
                if b["family"] == "A"
                and b["H"] == r["H"]
                and b["L"] == r["L"]
                and b["d"] == 0
            ),
            None,
        )

    slopes = []
    for r in rows:
        b = base(r)
        c0 = None if b is None else median(b, "cudnn")
        if c0 is None or median(r, "cake") is None:
            continue
        d = r.get("d", 0)
        per_seq = max(budget_pct / 100.0 * c0, budget_us)
        r["bar_us"] = c0 + d * per_seq
        if "bar_paired" in r:
            r["bar_verdict"] = BAR_VERDICT[r["bar_paired"]["verdict"]]
            r["within_bar"] = r["bar_verdict"] != "over"
        else:
            r["within_bar"] = median(r, "cake") <= r["bar_us"]
        if r["family"] != "A" or d == 0 or median(b, "cake") is None:
            continue
        t0, td = median(b, "cake"), median(r, "cake")
        slope = (td - t0) / d
        entry = {
            "name": r["name"],
            "cudnn_t0_us": c0,
            "t0_us": t0,
            "td_us": td,
            "slope_us": slope,
            "budget_us": per_seq,
            "within": slope <= per_seq,
        }
        if (
            "slope_paired" in r
        ):  # per round, Cake d against Cake d = 0 of the same round
            sp = r["slope_paired"]
            lo, hi = sp["ci95"]
            entry["slope_us"] = sp["p50"]
            entry["slope_ci95"] = [lo, hi]
            entry["verdict"] = (
                "within"
                if hi <= per_seq
                else "over"
                if lo > per_seq
                else "parity (noise)"
            )
            entry["within"] = entry["verdict"] != "over"
        slopes.append(entry)
    judged = [r for r in rows if "within_bar" in r]
    bars = collections.Counter(
        r.get("bar_verdict") for r in judged if "bar_verdict" in r
    )
    lines.append(
        f"request bar (Cake <= cuDNN d=0 + d x max({budget_pct:g}% , {budget_us:g} us)): "
        f"{sum(r['within_bar'] for r in judged)} of {len(judged)} rows not over"
        + (f", {dict(bars)}" if bars else "")
    )
    for H in sorted({r["H"] for r in rows}):
        mine = [r for r in rows if r["H"] == H and "speedup" in r]
        if mine:
            geo = math.exp(sum(math.log(r["speedup"]) for r in mine) / len(mine))
            bar = [r for r in mine if "within_bar" in r]
            lines.append(
                f"H={H}: {len(mine)} workloads, geomean speedup {geo:.3f}, "
                f"min {min(r['speedup'] for r in mine):.3f}, "
                f"request bar {sum(r['within_bar'] for r in bar)} of {len(bar)}"
            )
    return lines, slopes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--families", default="A,B,C")
    p.add_argument("--heads", default="10,20,40,80")
    p.add_argument("--lengths", default="8192,16384,32768,65536")
    p.add_argument("--short", default="0,1,4,8,15")
    p.add_argument("--backends", default="cake,cudnn")
    p.add_argument("--replays", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--block",
        type=int,
        default=10,
        help="replays per timed block; the backends alternate block by block (--replays / --block rounds)",
    )
    p.add_argument("--cold-l2", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gates", choices=sorted(GATES), default="gfi")
    p.add_argument("--budget-pct", type=float, default=1.0)
    p.add_argument("--budget-us", type=float, default=2.0)
    p.add_argument("--output", default=None, help="JSON lines, one per workload")
    p.add_argument(
        "--summarize",
        default=None,
        help="summarize an existing --output file instead of measuring",
    )
    args = p.parse_args(argv)
    if args.summarize:
        with open(args.summarize) as f:
            rows = [json.loads(line) for line in f]
        return report([r for r in rows if "name" in r], args)
    if args.replays < 100:
        p.error("--replays must be at least 100")
    args.backends = args.backends.split(",")
    ints = lambda s: [int(x) for x in s.split(",") if x]  # noqa: E731
    props = torch.cuda.get_device_properties(0)
    clock = sm_clock_reader()
    header = {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "replays": args.replays,
        "warmup": args.warmup,
        "cold_l2": args.cold_l2,
        "gates": args.gates,
        "block": args.block,
        "rounds": max(1, args.replays // args.block),
        "pairing": "per round and variant one block per backend, order alternating per round",
        "nvml_sm_clock": clock is not None,
    }
    print(json.dumps(header), flush=True)
    sink = open(args.output, "w") if args.output else None  # noqa: SIM115
    if sink:
        sink.write(json.dumps({"header": header}) + "\n")
    groups: dict[tuple, list] = {}
    for w in workloads(
        args.families.split(","), ints(args.heads), ints(args.lengths), ints(args.short)
    ):
        groups.setdefault((w["family"], w["H"], w.get("L")), []).append(w)
    flush = (
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if args.cold_l2
        else None
    )
    rows = []
    for row in (
        r for group in groups.values() for r in run_group(group, args, flush, clock)
    ):
        rows.append(row)
        brief = {
            b: round(row[b]["median_us"], 2)
            if "median_us" in row.get(b, {})
            else row.get(b)
            for b in args.backends
        }
        extra = {
            k: row[k] for k in ("speedup", "rel_err_out", "rel_err_state") if k in row
        }
        if "paired" in row:
            extra["ci95"] = [round(x, 4) for x in row["paired"]["ci95"]]
        if "bar_paired" in row:
            extra["bar_p50"] = round(row["bar_paired"]["p50"], 4)
        print(row["name"], json.dumps(brief), json.dumps(extra), flush=True)
        if sink:
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return report(rows, args, sink)


def report(rows, args, sink=None):
    """Print the summary of ``rows`` (and append it to ``sink``); 1 when a backend failed, no row pairs Cake with
    cuDNN, or ``--check`` finds an error at or above the tolerance or a non-finite value, else 0."""

    lines, slopes = summarize(rows, args.budget_pct, args.budget_us)
    for line in lines:
        print(line)
    over = [s for s in slopes if not s["within"]]
    verdicts = collections.Counter(s["verdict"] for s in slopes if "verdict" in s)
    print(
        f"per-extra-sequence cost: {len(slopes) - len(over)} of {len(slopes)} not over budget"
        + (f", {dict(verdicts)}" if verdicts else "")
    )
    for s in over:
        print("  over budget:", json.dumps(s))
    for r in rows:
        if "bar_paired" in r and (
            r["bar_verdict"] != "within" or r["paired"]["verdict"] != "above"
        ):
            sp, bp = r["paired"], r["bar_paired"]
            mhz = {b: r[b].get("sm_mhz", {}).get("p50") for b in ("cake", "cudnn")}
            print(
                f"  {r['name']}: speedup p10/p50/p90 {sp['p10']:.3f}/{sp['p50']:.3f}/{sp['p90']:.3f} "
                f"ci95 [{sp['ci95'][0]:.3f}, {sp['ci95'][1]:.3f}] {SPEED_VERDICT[sp['verdict']]}; "
                f"bar/cake p10/p50/p90 {bp['p10']:.3f}/{bp['p50']:.3f}/{bp['p90']:.3f} "
                f"ci95 [{bp['ci95'][0]:.3f}, {bp['ci95'][1]:.3f}] {r['bar_verdict']}; SM MHz {mhz}"
            )
        elif r.get("within_bar") is False:
            print(
                f"  over the request bar: {r['name']} cake {r['cake']['median_us']:.1f} us "
                f"> {r['bar_us']:.1f} us"
            )
    if sink:
        sink.write(
            json.dumps(
                {
                    "summary": lines,
                    "slopes": slopes,
                    "bars": {
                        r["name"]: [r["bar_us"], r["within_bar"], r.get("bar_verdict")]
                        for r in rows
                        if "within_bar" in r
                    },
                }
            )
            + "\n"
        )
        sink.close()
    failed = [
        f"{r['name']} {b}: {r[b]['error']}"
        for r in rows
        for b in ("cake", "cudnn")
        if "error" in r.get(b, {})
    ]
    paired = sum(1 for r in rows if "speedup" in r)
    bad = [
        r["name"]
        for r in rows
        if r.get("finite") is False
        or max(r.get("rel_err_out", 0.0), r.get("rel_err_state", 0.0)) >= REL_TOL
    ]
    for f in failed[:10]:
        print("  backend failed:", f)
    if bad:
        print(f"  outside the tolerance {REL_TOL:g} or not finite: {bad}")
    if failed or bad or paired == 0:
        print(
            f"FAIL: {len(failed)} backend failures, {paired} paired rows, {len(bad)} rows outside tolerance"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
