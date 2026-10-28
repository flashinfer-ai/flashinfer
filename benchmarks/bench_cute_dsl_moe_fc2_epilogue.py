#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""FC2 (GEMM2 + finalize) tactic sweep for the CuTe-DSL NVFP4 MoE over token counts.

Times the FC2 kernel alone for every Blackwell autotuner tactic, grouped by the
GEMM2 epilogue, and reports the best tactic of each group per token count:

* ``old``: plain 3-element GEMM2 tactics (default epilogue, no swizzle)
* ``swizzle``: epilogue ``(1, 0, s)``: default epilogue, swizzled raster
* ``store_warps``: epilogue ``(c, w, s)`` with ``w > 0``: store-warp epilogue

Each GEMM2 tactic runs once, paired with the fastest GEMM1 of its tile size.
``--autotune`` additionally runs the autotuner per token count and times the
tactic it picks in the same rounds. Results are reported twice, as FC2 kernel
time and as the time of all MoE kernels; each ranks the groups by its own
metric, and speedups are over the best ``old`` tactic for that metric.

Routing is perfectly balanced: every expert receives tokens * top_k / experts
routed rows, give or take one.

Examples (defaults: 384 experts, H=4096, I=768, top-k=20, 4k-54k tokens):

    # Full sweep plus the MoE autotuner's pick, raw records to a JSONL file
    python benchmarks/bench_cute_dsl_moe_fc2_epilogue.py --autotune \\
        --jsonl fc2_sweep.jsonl

    # One token count, quick check (1 round)
    python benchmarks/bench_cute_dsl_moe_fc2_epilogue.py --tokens 15900 \\
        --rounds 1 --autotune

There is no FC2-only autotuner: the MoE autotuner picks (tile_size, gemm1,
gemm2) together by whole-MoE time. The ``best <group>`` rows are this script's
own sweep, ranked by FC2 time.

Reference results (B300, CUDA 13.3, defaults, ``--autotune``, 2026-10-28), in
us with the speedup over ``best old``. FC2 kernel time:

    tokens  best old  best swizzle   best store_warps  autotuned (epilogue)
    4000      218      223 (0.98x)     211 (1.03x)      211 (1.03x) (2,2,4)
    8000      424      429 (0.99x)     388 (1.09x)      388 (1.09x) (2,4,4)
    15900     881      799 (1.10x)     719 (1.23x)      719 (1.23x) (2,2,4)
    24000    1417     1137 (1.25x)    1013 (1.40x)     1013 (1.40x) (2,4,4)
    32000    1967     1592 (1.24x)    1471 (1.34x)     1524 (1.29x) (2,2,4)
    40000    2633     2021 (1.30x)    1942 (1.36x)     2021 (1.30x) (1,0,2)
    54000    3673     2754 (1.33x)    2728 (1.35x)     2754 (1.33x) (1,0,2)

Whole MoE (all kernels):

    tokens  best old  best swizzle   best store_warps  autotuned
    4000      528      533 (0.99x)     520 (1.02x)      520 (1.02x)
    8000      955      962 (0.99x)     920 (1.04x)      920 (1.04x)
    15900    1835     1755 (1.05x)    1678 (1.09x)     1678 (1.09x)
    24000    2760     2363 (1.17x)    2262 (1.22x)     2262 (1.22x)
    32000    3788     3301 (1.15x)    3207 (1.18x)     3332 (1.14x)
    40000    4902     4233 (1.16x)    4219 (1.16x)     4233 (1.16x)
    54000    6502     5540 (1.17x)    5587 (1.16x)     5540 (1.17x)

The autotuner's picks are within about 4% of the fastest whole-MoE tactic;
its top candidates are only a few percent apart, so picks can differ between
runs.

Timing uses the torch profiler, which attributes GPU time per kernel; the FC2
kernel is the one whose name contains ``FinalizeFusion``. Each tactic is timed
in ``--rounds`` interleaved rounds of ``--iters`` calls and the median round
is reported. Run-to-run noise is about 5%, so differences below that are not
significant.
"""

import argparse
import contextlib
import json
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from flashinfer import autotune, cute_dsl_fused_moe
from flashinfer.autotuner import AutoTuner
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl.tuner import ALL_BLACKWELL_MOE_TACTICS

SF_VEC_SIZE = 16
# The fastest GEMM1 tactic of each tile size on B300 for this workload.
GEMM1 = {128: ((128, 256), (1, 1), False), 256: ((256, 256), (2, 1), False)}


def quantize_weight(w):
    e, n, k = w.shape
    one = torch.ones(1, dtype=torch.float32, device=w.device)
    q, sf = fp4_quantize(
        w.reshape(e * n, k).contiguous(),
        global_scale=one,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    sf = convert_sf_to_mma_layout(sf, m=n, k=k, num_groups=e, sf_vec_size=SF_VEC_SIZE)
    return q.view(e, n, k // 2), sf


def make_weights(experts, hidden, inter):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device=dev)
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device=dev)
    w1_q, w1_sf = quantize_weight(w1 / 10)
    w2_q, w2_sf = quantize_weight(w2 / 10)
    ones = torch.ones(experts, dtype=torch.float32, device=dev)
    return dict(
        w1_weight=w1_q,
        w1_weight_sf=w1_sf,
        w1_alpha=ones,
        w2_weight=w2_q,
        w2_weight_sf=w2_sf,
        w2_alpha=ones,
        fc2_input_scale=torch.ones(1, dtype=torch.float32, device=dev),
    )


def make_activations(tokens, experts, hidden, top_k):
    torch.manual_seed(tokens)
    dev = torch.device("cuda")
    one = torch.ones(1, dtype=torch.float32, device=dev)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=dev) / 10
    x_fp4, x_sf = fp4_quantize(
        x, global_scale=one, sf_vec_size=SF_VEC_SIZE, is_sf_swizzled_layout=False
    )
    # Perfectly balanced routing: route slots go round-robin over the experts,
    # so expert loads differ by at most one token, and a token's top_k <= experts
    # consecutive slots pick distinct experts. Round-robin experts j and j+1
    # share almost all tokens; relabeling the experts with a random permutation
    # keeps those overlapping experts apart in GEMM2's expert-ordered M tiles,
    # whose concurrent output rows would otherwise stay in L2. Shuffling the
    # tokens keeps an expert's rows from being contiguous in x.
    slots = torch.arange(tokens * top_k, device=dev) % experts
    slots = torch.randperm(experts, device=dev)[slots]
    selected = slots.view(tokens, top_k)[torch.randperm(tokens, device=dev)]
    scales = torch.rand(tokens, top_k, device=dev) + 0.5
    return dict(
        x=x_fp4,
        x_sf=x_sf.unsqueeze(-1),
        token_selected_experts=selected.to(torch.int32),
        token_final_scales=(scales / scales.sum(dim=1, keepdim=True)).float(),
    )


GROUPS = ("old", "swizzle", "store_warps")


def gemm2_group(gemm2_tactic):
    if len(gemm2_tactic) == 3:
        return "old"
    return "swizzle" if gemm2_tactic[3][1] == 0 else "store_warps"


def fc2_tactics():
    """One tactic per distinct GEMM2 sub-tactic, with the tile size's GEMM1."""
    seen, tactics = set(), []
    for tile_size, _, gemm2 in ALL_BLACKWELL_MOE_TACTICS:
        if (tile_size, gemm2) not in seen:
            seen.add((tile_size, gemm2))
            tactics.append((tile_size, GEMM1[tile_size], gemm2))
    return tactics


def time_us(fn, iters):
    """(FC2 kernel, all kernels) GPU time per call in us."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    fc2 = total = 0.0
    for e in prof.key_averages():
        if e.device_type.name != "CUDA":
            continue
        total += e.device_time_total
        if "FinalizeFusion" in e.key:
            fc2 += e.device_time_total
    return fc2 / iters, total / iters


def as_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(as_tuple(v) for v in value)
    return value


def tuned_tactic():
    winners = [
        tactic
        for key, (tactic, _) in AutoTuner.get()._winner_cache().items()
        if "CuteDslFusedMoE" in key.runner_class_name
    ]
    return winners[-1] if winners else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", default="4000,8000,15900,24000,32000,40000,54000")
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--autotune", action="store_true", help="also report the autotuner's pick"
    )
    p.add_argument("--jsonl", help="append one record per (tokens, tactic)")
    args = p.parse_args()

    weights = make_weights(args.experts, args.hidden, args.inter)
    print(f"{torch.cuda.get_device_name()}: {len(fc2_tactics())} GEMM2 tactics")
    with open(args.jsonl, "a") if args.jsonl else contextlib.nullcontext() as out:
        for tokens in [int(t) for t in args.tokens.split(",")]:
            sweep(args, weights, tokens, out)
            torch.cuda.empty_cache()


def sweep(args, weights, tokens, out):
    """Time every GEMM2 tactic (and the autotuner's pick) at one token count."""
    tactics = fc2_tactics()
    inputs = {
        **weights,
        **make_activations(tokens, args.experts, args.hidden, args.top_k),
    }

    def run(tactic):
        return cute_dsl_fused_moe(
            **inputs,
            num_experts=args.experts,
            top_k=args.top_k,
            quant_mode="w4a4",
            tactic=tactic,
        )

    measured = list(tactics)
    tuned = None
    if args.autotune:
        AutoTuner.get().clear_cache()
        with autotune(True):
            run(None)
        tuned = as_tuple(tuned_tactic())
        if tuned not in measured:
            measured.append(tuned)

    ref = run(tactics[0]).float()
    rel = {}
    for t in measured:  # compile, and check against the first tactic
        rel[t] = ((run(t).float() - ref).norm() / ref.norm()).item()
    samples = {t: [] for t in measured}
    for _ in range(args.rounds):  # interleaved against power throttling
        for t in measured:
            samples[t].append(time_us(lambda: run(t), args.iters))
    fc2 = {t: statistics.median(s[0] for s in samples[t]) for t in measured}
    total = {t: statistics.median(s[1] for s in samples[t]) for t in measured}

    if out:
        for t in measured:
            rec = dict(
                tokens=tokens,
                group=gemm2_group(t[2]),
                autotuned=t == tuned,
                tactic=t,
                fc2_us=fc2[t],
                total_us=total[t],
                rel_l2=rel[t],
            )
            out.write(json.dumps(rec) + "\n")
        out.flush()

    print(f"\ntokens={tokens}  (max rel L2 vs first tactic {max(rel.values()):.1e})")
    # FC2 and the whole MoE are ranked separately: the tactic with the fastest
    # FC2 is not always the one with the fastest MoE.
    for metric, times in (("FC2", fc2), ("all kernels", total)):
        best = {}
        for t in tactics:
            g = gemm2_group(t[2])
            if g not in best or times[t] < times[best[g]]:
                best[g] = t
        rows = [(f"best {g}", best[g]) for g in GROUPS if g in best]
        if tuned is not None:
            rows.append(("autotuned", tuned))
        print(f"  {metric}:")
        for label, t in rows:
            print(
                f"    {label:17s} {times[t]:7.0f} us"
                f" {times[best['old']] / times[t]:5.2f}x  tactic={t}"
            )


if __name__ == "__main__":
    main()
