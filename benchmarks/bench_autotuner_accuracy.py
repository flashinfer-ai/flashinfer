"""Autotuner accuracy harness (skeleton) — quantify how well the tuner's
winner matches an oracle, per measurement policy.

Methodology (see RFC https://github.com/flashinfer-ai/flashinfer/issues/3920,
"Accuracy quantification"): for each shape of a real op (bmm_fp8,
backend="auto"), enumerate
every (runner, tactic) candidate the autotuner would consider, measure each
exhaustively with a high-repetition oracle, then simulate tuner selections
under each measurement policy at production settings (repeat=10) across
several seeds and score:

- regret:   (t_oracle(chosen) - t_oracle(best)) / t_oracle(best)
- top-1:    chosen == oracle best
- flips:    number of distinct winners across seeds (measurement noise)
- oracle disagreement: do the eager-oracle and cupti-oracle name the same
  best candidate?  (the two measure different deployments: host-included
  eager rate vs host-excluded kernel span)

Decode focus: small-M shapes are where CUDA-event timing is noisiest.

Drift protection: the oracle sweep is INTERLEAVED — every round measures
all candidates back-to-back (rotating the start offset per round), so
thermal/clock drift hits candidates evenly instead of manufacturing
phantom regret between phase-separated measurements.  The per-round SM
clock is recorded (pynvml) so drift is visible in the report.  Oracles:
``cupti`` (host-excluded span) and ``eager`` (execution_mode="eager":
host-included, no delay kernel) — both explicit policies, no reliance on
delay-budget overflow.

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


def sm_clock_mhz():
    """Current SM clock in MHz via pynvml, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        # CUDA_VISIBLE_DEVICES remapping: resolve via PCI bus id.
        bus = torch.cuda.get_device_properties(idx).pci_bus_id
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(
            f"00000000:{bus:02X}:00.0".encode()
        )
        return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def interleaved_oracle(tuner, candidates, inputs, config, kwargs, rounds, repeat):
    """Measure every candidate under BOTH oracle policies, interleaved.

    Each round visits all candidates back-to-back (start offset rotates),
    measuring the cupti and eager oracles adjacently so both see the same
    thermal state.  Returns (oracle_cupti, oracle_eager, clocks_mhz) where
    each oracle maps (r_id, tactic) -> median-over-rounds ms.
    """
    cupti_pol = MeasurementPolicy(_timer="cupti")
    eager_pol = MeasurementPolicy(execution_mode="eager")
    per_c = {(r_id, t): [] for r_id, _, t in candidates}
    per_e = {(r_id, t): [] for r_id, _, t in candidates}
    clocks = []
    for rnd in range(rounds):
        clocks.append(sm_clock_mhz())
        order = (
            candidates[rnd % len(candidates) :] + candidates[: rnd % len(candidates)]
        )
        for r_id, r, t in order:
            per_c[(r_id, t)].append(
                measure_candidate(
                    tuner, r, inputs, t, config, kwargs, cupti_pol, repeat
                )
            )
            per_e[(r_id, t)].append(
                measure_candidate(
                    tuner, r, inputs, t, config, kwargs, eager_pol, repeat
                )
            )
    oracle_c = {k: statistics.median(v) for k, v in per_c.items()}
    oracle_e = {k: statistics.median(v) for k, v in per_e.items()}
    return oracle_c, oracle_e, clocks


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
    parser.add_argument("--oracle-rounds", type=int, default=5)
    parser.add_argument("--oracle-repeat", type=int, default=21)
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
        "v2_eager": MeasurementPolicy(execution_mode="eager"),
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

        # Oracles: interleaved rounds, both deployment policies.
        oracle, oracle_eager, clocks = interleaved_oracle(
            tuner,
            candidates,
            inputs,
            config,
            kwargs,
            args.oracle_rounds,
            args.oracle_repeat,
        )
        best_key = min(oracle, key=oracle.get)
        best_ms = oracle[best_key]
        best_eager_key = min(oracle_eager, key=oracle_eager.get)
        best_eager_ms = oracle_eager[best_eager_key]
        valid = sum(1 for v in oracle.values() if v != float("inf"))

        def name(key):
            r_id, t = key
            return f"{runners[r_id].__class__.__name__}[{t}]"

        top3 = sorted(oracle, key=oracle.get)[:3]
        shape_row = {
            "m": m,
            "candidates": len(candidates),
            "valid": valid,
            "sm_clocks_mhz": clocks,
            "oracle_best": name(best_key),
            "oracle_best_ms": best_ms,
            "oracle_top3": [(name(k), oracle[k]) for k in top3],
            "oracle_eager_best": name(best_eager_key),
            "oracles_agree": best_key == best_eager_key,
            "policies": {},
        }
        print(
            f"  clocks(MHz)={clocks}\n"
            f"  oracle(cupti):  best={name(best_key)} {best_ms * 1e3:.1f}us  "
            f"top3={[f'{name(k)} {oracle[k] * 1e3:.1f}us' for k in top3]}"
        )
        print(
            f"  oracle(eager):  best={name(best_eager_key)} "
            f"{best_eager_ms * 1e3:.1f}us"
            f"{'' if best_key == best_eager_key else '  << ORACLES DISAGREE (expected when host-bound)'}"
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
                regrets_e.append((oracle_eager[pick] - best_eager_ms) / best_eager_ms)
            top1 = sum(1 for c in chosen if c == best_key)
            shape_row["policies"][pname] = {
                "top1_vs_cupti_oracle": f"{top1}/{args.seeds}",
                "regret_cupti_med": statistics.median(regrets_c),
                "regret_cupti_max": max(regrets_c),
                "regret_eager_med": statistics.median(regrets_e),
                "regret_eager_max": max(regrets_e),
                "distinct_winners": len(set(chosen)),
                "winners": [name(c) for c in chosen],
            }
            print(
                f"  {pname:10s} "
                f"regret(cupti) med={statistics.median(regrets_c) * 100:.2f}% "
                f"max={max(regrets_c) * 100:.2f}% | "
                f"regret(eager) med={statistics.median(regrets_e) * 100:.2f}% "
                f"max={max(regrets_e) * 100:.2f}% | "
                f"flips={len(set(chosen))} winners={[name(c) for c in chosen]}"
            )
        report["shapes"].append(shape_row)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
