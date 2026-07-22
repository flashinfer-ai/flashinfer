"""Autotuner accuracy harness (skeleton) — quantify how well the tuner's
winner matches an oracle, per measurement policy.

Methodology (see FLASHINFER_AUTOTUNE_CACHE_PROPOSAL.md, "Quantifying tuner
accuracy"): for each shape of a real op (bmm_fp8, backend="auto"), enumerate
every (runner, tactic) candidate the autotuner would consider, measure each
exhaustively with a high-repetition oracle, then simulate tuner selections
under each measurement policy at production settings (repeat=10) across
several seeds and score:

- regret:   (t_oracle(chosen) - t_oracle(best)) / t_oracle(best)
- top-1:    chosen == oracle best
- flips:    number of distinct winners across seeds (measurement noise)
- oracle disagreement: do the events-oracle and cupti-oracle name the same
  best candidate?  (the two timers measure different things: steady-state
  throughput vs isolated-call span)

Decode focus: small-M shapes are where CUDA-event timing is noisiest.

CAVEAT: the events oracle at repeat=101 overflows the 5 ms delay-kernel
budget, so it measures the eager sustained rate — it is NOT a scaled-up v1
measurement (v1 at repeat=10 is blind to host-bound cost).  Deployment
oracles (eager wall-clock, CUDA-graph replay) are the next refinement.

Usage (pick an SM100 GPU via CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=1 CUDA_HOME=... python benchmarks/bench_autotuner_accuracy.py
"""

import argparse
import inspect
import json
import os
import statistics
import sys

os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "10.0a")

import torch  # noqa: E402

from flashinfer.autotune_cache import MeasurementPolicy  # noqa: E402
from flashinfer.autotuner import (  # noqa: E402
    AutoTuner,
    OptimizationProfile,
    StaticDim,
)


from tests.utils_fp8 import to_float8  # noqa: E402


def static_profile(inputs):
    return OptimizationProfile(
        [
            [StaticDim(x) for x in t.size()]
            if isinstance(t, torch.Tensor)
            else [StaticDim(0)]
            for t in inputs
        ],
        [None] * len(inputs),
    )


def capture_choose_one_call(make_call):
    """Run *make_call* once and capture the (runners, config, inputs, kwargs)
    the API hands to AutoTuner.choose_one."""
    captured = {}
    orig = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        captured[custom_op] = (runners, tuning_config, inputs, kwargs)
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = spy
    try:
        make_call()
    finally:
        AutoTuner.choose_one = orig
    assert len(captured) == 1, f"expected one tuned op, saw {list(captured)}"
    return next(iter(captured.items()))


def measure_candidate(tuner, runner, inputs, tactic, config, kwargs, policy, repeat):
    """One tuner-style measurement; returns ms or inf on failure."""
    saved_repeat = tuner.repeat
    stack = tuner._get_measure_stack()
    if policy is not None:
        stack.append(policy)
    try:
        tuner.repeat = repeat
        return tuner._profile_single_kernel(runner, inputs, tactic, config, **kwargs)
    except Exception as e:
        print(f"    [fail] {runner.__class__.__name__} tactic={tactic}: {e}")
        return float("inf")
    finally:
        tuner.repeat = saved_repeat
        if policy is not None:
            stack.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument(
        "--m-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 1024]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--oracle-repeat", type=int, default=101)
    parser.add_argument("--tuner-repeat", type=int, default=10)
    parser.add_argument("--out", type=str, default="autotuner_accuracy.json")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} sm{props.major}{props.minor}")

    import flashinfer

    tuner = AutoTuner.get()
    policies = {
        "v1_events": None,
        "v2_cupti": MeasurementPolicy(_timer="cupti"),
    }
    report = {"gpu": props.name, "n": args.n, "k": args.k, "shapes": []}

    for m in args.m_list:
        a = torch.randn([1, m, args.k], device="cuda", dtype=torch.bfloat16)
        b = torch.randn(
            [1, args.n, args.k], device="cuda", dtype=torch.bfloat16
        ).transpose(-2, -1)
        a8, a_scale = to_float8(a)
        b8, b_scale = to_float8(b)

        op, (runners, config, inputs, kwargs) = capture_choose_one_call(
            lambda: flashinfer.bmm_fp8(
                a8, b8, a_scale, b_scale, torch.bfloat16, backend="auto"
            )
        )
        profile = static_profile(inputs)

        # Enumerate candidates exactly as choose_one would.
        candidates = []
        for r_id, r in enumerate(runners):
            tactics = r.get_valid_tactics(inputs, profile)
            if (
                "do_preparation" in inspect.signature(r.forward).parameters
                and len(tactics) > 0
            ):
                r(inputs, tactic=-1, do_preparation=True, **kwargs)
            candidates.extend((r_id, r, t) for t in tactics)
        print(f"\nM={m}: op={op}, {len(runners)} runners, {len(candidates)} candidates")

        # Oracles: high-repetition, both timers.
        oracle = {}
        oracle_events = {}
        for r_id, r, t in candidates:
            oracle[(r_id, t)] = measure_candidate(
                tuner,
                r,
                inputs,
                t,
                config,
                kwargs,
                MeasurementPolicy(_timer="cupti"),
                args.oracle_repeat,
            )
            oracle_events[(r_id, t)] = measure_candidate(
                tuner, r, inputs, t, config, kwargs, None, args.oracle_repeat
            )
        best_key = min(oracle, key=oracle.get)
        best_ms = oracle[best_key]
        best_events_key = min(oracle_events, key=oracle_events.get)
        best_events_ms = oracle_events[best_events_key]
        valid = sum(1 for v in oracle.values() if v != float("inf"))

        def name(key):
            r_id, t = key
            return f"{runners[r_id].__class__.__name__}[{t}]"

        top3 = sorted(oracle, key=oracle.get)[:3]
        shape_row = {
            "m": m,
            "candidates": len(candidates),
            "valid": valid,
            "oracle_best": name(best_key),
            "oracle_best_ms": best_ms,
            "oracle_top3": [(name(k), oracle[k]) for k in top3],
            "oracle_events_best": name(best_events_key),
            "oracles_agree": best_key == best_events_key,
            "policies": {},
        }
        print(
            f"  oracle(cupti):  best={name(best_key)} {best_ms * 1e3:.1f}us  "
            f"top3={[f'{name(k)} {oracle[k] * 1e3:.1f}us' for k in top3]}"
        )
        print(
            f"  oracle(events): best={name(best_events_key)} "
            f"{best_events_ms * 1e3:.1f}us"
            f"{'' if best_key == best_events_key else '  << ORACLES DISAGREE'}"
        )

        # Tuner simulations at production settings.  Regret is scored
        # against BOTH oracles: each policy optimizes its own measurand,
        # so scoring only against one oracle would bias toward the policy
        # that shares it.
        for pname, policy in policies.items():
            chosen, regrets_c, regrets_e = [], [], []
            for _ in range(args.seeds):
                times = {
                    (r_id, t): measure_candidate(
                        tuner,
                        r,
                        inputs,
                        t,
                        config,
                        kwargs,
                        policy,
                        args.tuner_repeat,
                    )
                    for r_id, r, t in candidates
                }
                pick = min(times, key=times.get)
                chosen.append(pick)
                regrets_c.append((oracle[pick] - best_ms) / best_ms)
                regrets_e.append(
                    (oracle_events[pick] - best_events_ms) / best_events_ms
                )
            top1 = sum(1 for c in chosen if c == best_key)
            shape_row["policies"][pname] = {
                "top1_vs_cupti_oracle": f"{top1}/{args.seeds}",
                "regret_cupti_med": statistics.median(regrets_c),
                "regret_cupti_max": max(regrets_c),
                "regret_events_med": statistics.median(regrets_e),
                "regret_events_max": max(regrets_e),
                "distinct_winners": len(set(chosen)),
                "winners": [name(c) for c in chosen],
            }
            print(
                f"  {pname:10s} "
                f"regret(cupti) med={statistics.median(regrets_c) * 100:.2f}% "
                f"max={max(regrets_c) * 100:.2f}% | "
                f"regret(events) med={statistics.median(regrets_e) * 100:.2f}% "
                f"max={max(regrets_e) * 100:.2f}% | "
                f"flips={len(set(chosen))} winners={[name(c) for c in chosen]}"
            )
        report["shapes"].append(shape_row)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
