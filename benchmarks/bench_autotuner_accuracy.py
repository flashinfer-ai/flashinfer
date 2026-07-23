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


def capture_choose_one_calls(make_call):
    """Run *make_call* once and capture EVERY choose_one invocation as
    {custom_op: (runners, config, inputs, kwargs)}.  Composite APIs (MoE)
    tune several internal ops per call; each is swept separately."""
    captured = {}
    orig = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        captured.setdefault(custom_op, (runners, tuning_config, inputs, kwargs))
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = spy
    try:
        make_call()
    finally:
        AutoTuner.choose_one = orig
    assert captured, "the call did not reach AutoTuner.choose_one"
    return captured


# ---------------------------------------------------------------------------
# Op registry: builder(grid_point) -> zero-arg callable that invokes the
# public API.  Input construction is lifted from the corresponding tests so
# probes are valid (quantization / scale layouts / routing).
# ---------------------------------------------------------------------------


def build_bmm_fp8(m, n, k):
    a = torch.randn([1, m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([1, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    a8, a_scale = to_float8(a)
    b8, b_scale = to_float8(b)

    def call():
        import flashinfer

        return flashinfer.bmm_fp8(
            a8, b8, a_scale, b_scale, torch.bfloat16, backend="auto"
        )

    return call


def build_mm_fp4(m, n, k):
    """NVFP4 GEMM, 128x4 SF layout, backend=auto (lifted from test_mm_fp4)."""
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    inp = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    gs_in = (448 * 6) / inp.float().abs().nan_to_num().max()
    gs_w = (448 * 6) / mat2.float().abs().nan_to_num().max()
    a4, a_sf = nvfp4_quantize(inp, gs_in, sfLayout=SfLayout.layout_128x4)
    b4, b_sf = nvfp4_quantize(mat2, gs_w, sfLayout=SfLayout.layout_128x4)
    alpha = 1.0 / (gs_in * gs_w)
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)

    def call():
        return mm_fp4(
            a4,
            b4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="auto",
            use_nvfp4=True,
        )

    return call


def build_cutlass_moe_fp8(num_tokens, hidden=4096, inter=2048, experts=64, top_k=8):
    """FP8 CUTLASS fused MoE (lifted from test_trtllm_cutlass_fused_moe).

    Routing: uniform random logits — the balanced-probe assumption; the
    known probe-realism caveat (#3622) applies and is exactly what the
    op-level track will improve.
    """
    from tests.moe.test_trtllm_cutlass_fused_moe import (
        cast_to_representable,
        compute_routing,
        dynamic_per_tensor_fp8_quant,
        gen_tensor,
    )
    import flashinfer.fused_moe as fused_moe

    otype, wtype = torch.bfloat16, torch.float8_e4m3fn
    x = cast_to_representable(gen_tensor((num_tokens, hidden), otype))
    router_logits = gen_tensor((num_tokens, experts), otype)
    w31_weight = gen_tensor((experts, 2 * inter, hidden), otype, wtype)
    w2_weight = gen_tensor((experts, hidden, inter), otype, wtype)
    w31_scales = torch.empty(experts, 2, dtype=otype).cuda()
    w2_scales = torch.empty(experts, 1, dtype=otype).cuda()
    for e in range(experts):
        w31q, s31 = dynamic_per_tensor_fp8_quant(
            cast_to_representable(gen_tensor((2 * inter, hidden), otype, scale=0.1))
        )
        w2q, s2 = dynamic_per_tensor_fp8_quant(
            cast_to_representable(gen_tensor((hidden, inter), otype, scale=0.09))
        )
        w31_weight.data[e].copy_(w31q)
        w2_weight.data[e].copy_(w2q)
        w31_scales.data[e].copy_(s31)
        w2_scales.data[e].copy_(s2)
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    # FP8 path expects the hidden state pre-quantized (mirrors test_moe_fp8).
    _, w1_scales = torch.chunk(w31_scales, 2, dim=-1)
    x_quant, hs_scale = dynamic_per_tensor_fp8_quant(x)
    hs_scale = torch.tensor(hs_scale[0]).cuda()
    quant_scales = [
        torch.squeeze(w1_scales * hs_scale).float(),
        torch.tensor(1.0).cuda(),
        torch.squeeze(1.0 * w2_scales).float(),
        hs_scale,
    ]
    out = torch.empty((num_tokens, hidden), device="cuda", dtype=otype)

    def call():
        return fused_moe.cutlass_fused_moe(
            x_quant,
            selected_experts.to(torch.int),
            routing_weights,
            w31_weight,
            w2_weight,
            otype,
            quant_scales=quant_scales,
            output=out,
        )

    return call


OPS = {
    "bmm_fp8": {
        "grid": [1, 2, 4, 8, 16, 32, 64, 1024],
        "build": lambda pt, args: build_bmm_fp8(pt, args.n, args.k),
        "axis": "m",
    },
    "mm_fp4": {
        "grid": [1, 2, 4, 8, 16, 32, 64, 1024],
        "build": lambda pt, args: build_mm_fp4(pt, args.n, args.k),
        "axis": "m",
    },
    "cutlass_moe_fp8": {
        "grid": [1, 8, 64, 256],
        "build": lambda pt, args: build_cutlass_moe_fp8(pt),
        "axis": "num_tokens",
    },
}


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


def sweep_captured_op(op, runners, config, inputs, kwargs, args, tuner, grid_label):
    """Oracle + policy simulations for one captured (op, shape) point.
    Returns the report row, or None if the op has no swappable candidates."""
    profile = static_profile(inputs)

    # Enumerate candidates exactly as choose_one would.
    candidates = []
    for r_id, r in enumerate(runners):
        tactics = list(r.get_valid_tactics(inputs, profile))
        if (
            "do_preparation" in inspect.signature(r.forward).parameters
            and len(tactics) > 0
        ):
            r(inputs, tactic=-1, do_preparation=True, **kwargs)
        candidates.extend((r_id, r, t) for t in tactics)
    total = len(candidates)
    if total <= 1:
        print(f"  [{op}] {total} candidate(s) — nothing to rank, skipped")
        return None
    if total > args.max_candidates:
        # No silent caps: subsample evenly and say so.
        step = total / args.max_candidates
        candidates = [candidates[int(i * step)] for i in range(args.max_candidates)]
        print(
            f"  [{op}] {total} candidates > cap {args.max_candidates}: "
            f"evenly subsampled (coverage is PARTIAL for this op)"
        )
    print(
        f"\n{grid_label}: op={op}, {len(runners)} runners, "
        f"{len(candidates)}/{total} candidates"
    )
    return _score_candidates(
        op, runners, candidates, total, config, inputs, kwargs, args, tuner, grid_label
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--grid", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--oracle-rounds", type=int, default=5)
    parser.add_argument("--oracle-repeat", type=int, default=21)
    parser.add_argument("--tuner-repeat", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--out", type=str, default="autotuner_accuracy.json")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} sm{props.major}{props.minor}")

    tuner = AutoTuner.get()
    report = {"gpu": props.name, "n": args.n, "k": args.k, "shapes": []}

    for api_name in args.ops:
        spec = OPS[api_name]
        grid = args.grid if args.grid is not None else spec["grid"]
        for pt in grid:
            grid_label = f"{api_name} {spec['axis']}={pt}"
            try:
                call = spec["build"](pt, args)
                captured = capture_choose_one_calls(call)
            except Exception as e:
                print(f"\n{grid_label}: BUILD/CALL FAILED: {type(e).__name__}: {e}")
                report["shapes"].append(
                    {"api": api_name, spec["axis"]: pt, "error": f"{e}"}
                )
                continue
            for op, (runners, config, inputs, kwargs) in captured.items():
                row = sweep_captured_op(
                    op, runners, config, inputs, kwargs, args, tuner, grid_label
                )
                if row is not None:
                    row["api"] = api_name
                    row[spec["axis"]] = pt
                    report["shapes"].append(row)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {args.out}")


def _score_candidates(
    op, runners, candidates, total, config, inputs, kwargs, args, tuner, grid_label
):
    policies = {
        "v1_events": None,
        "v2_cupti": MeasurementPolicy(_timer="cupti"),
        "v2_eager": MeasurementPolicy(execution_mode="eager"),
    }
    if True:  # keep the original indentation structure below
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
            "op": op,
            "candidates": len(candidates),
            "candidates_total": total,
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
        return shape_row


if __name__ == "__main__":
    sys.exit(main())
