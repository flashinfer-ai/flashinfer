"""Autotuner accuracy harness — quantify how well the tuner's
winner matches an oracle, per measurement policy.

For actual v2 searches followed by independent graph replay validation, use
--graph-selection-study. See README_autotuner_accuracy.md for its controls,
fixture interface and evidence limits. The methodology below describes the
original exhaustive-oracle mode.

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

Usage (pick a supported GPU via CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=1 CUDA_HOME=... python benchmarks/bench_autotuner_accuracy.py
"""

import argparse
import contextlib
import dataclasses
import importlib
import hashlib
from pathlib import Path
import subprocess
import math
import inspect
import json
import os
import statistics
import sys

import torch  # noqa: E402

from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2  # noqa: E402
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


def input_shapes(inputs):
    return tuple(
        tuple(t.shape) if isinstance(t, torch.Tensor) else (0,) for t in inputs
    )


def capture_choose_one_calls(make_call, *, require_single=False):
    """Run *make_call* once and capture EVERY choose_one invocation as
    {custom_op: (runners, config, inputs, kwargs)}.  Composite APIs (MoE)
    tune several internal ops per call; each is swept separately."""
    captured = {}
    orig = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if require_single and captured:
            raise ValueError(
                "graph study requires exactly one choose_one call per fixture invocation"
            )
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

    def reference():
        return torch.bmm(a8.float(), b8.float()) * a_scale * b_scale

    @torch.inference_mode()
    def prepare_validation(output, inputs):
        # bmm_fp8 allocates this output separately from its inputs and workspace.
        output.fill_(float("nan"))

    call.reference = reference
    call.prepare_validation = prepare_validation
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


class MeasurementUnavailable(RuntimeError):
    """Do not label fallback timings as the requested oracle."""


def measure_candidate(tuner, runner, inputs, tactic, config, kwargs, policy, repeat):
    """One explicit-policy measurement; returns ms or inf on tactic failure."""
    saved_repeat = tuner.repeat
    needs_cupti = policy is not None and policy.timer == "cupti"
    if needs_cupti:
        from flashinfer.autotuner.autotuner import _load_cupti

        if tuner._cupti_disabled or _load_cupti() is None:
            raise MeasurementUnavailable(
                "CUPTI oracle unavailable; refusing event-timer fallback"
            )
    try:
        tuner.repeat = repeat
        with (
            use_isolated_tuner(tuner),
            autotune_v2(persistent_cache=False, measurement_policy=policy),
        ):
            prepared_config, batches = tuner.prepare_tactic_profile(inputs, config)
            elapsed = tuner.profile_tactic(
                runner, inputs, tactic, prepared_config, batches, **kwargs
            )
            if needs_cupti and tuner._cupti_disabled:
                raise MeasurementUnavailable(
                    "CUPTI oracle fell back to events; result rejected"
                )
            return elapsed
    except MeasurementUnavailable:
        raise
    except Exception as e:
        print(f"    [fail] {runner.__class__.__name__} tactic={tactic}: {e}")
        return float("inf")
    finally:
        tuner.repeat = saved_repeat


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


@contextlib.contextmanager
def use_isolated_tuner(tuner):
    """The single-process harness owns its singleton and disables bundled winners."""
    previous = AutoTuner._instance
    bundled = os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE")
    AutoTuner._instance = tuner
    os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "0"
    try:
        yield
    finally:
        AutoTuner._instance = previous
        if bundled is None:
            os.environ.pop("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", None)
        else:
            os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = bundled


def retained_input_config(config, inputs):
    """One exact-shape experiment; actual serving tensors remain fixture-owned."""

    def retain(synthesized):
        if len(synthesized) != len(inputs):
            raise ValueError("retained input count changed")
        for generated, original in zip(synthesized, inputs, strict=True):
            if isinstance(original, torch.Tensor) and (
                not isinstance(generated, torch.Tensor)
                or generated.shape != original.shape
                or generated.dtype != original.dtype
                or generated.device != original.device
            ):
                raise ValueError("retained input metadata differs from the profile")
        return inputs

    return dataclasses.replace(
        config, dynamic_tensor_specs=(), constraint_specs=(), inputs_pre_hook=retain
    )


def capture_tactic_graph(runner, inputs, tactic, kwargs, warmup):
    # Exceptions abort this study. A bad capture must not be cleared and silently
    # reused for the next candidate in a potentially poisoned CUDA context.
    for _ in range(warmup):
        runner(inputs, tactic=tactic, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner(inputs, tactic=tactic, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def interleaved_graph_samples(graphs, rounds, replays, prequeue_cycles):
    """External events around replay; captured graphs contain only operator work.

    An unmeasured sleep before the start event lets the host enqueue the short
    replay window before the GPU reaches it. No delay/event/flush is captured.
    This measures hot-L2 component replay, not framework serving throughput.
    """
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for rnd in range(rounds):
        order = names[rnd % len(names) :] + names[: rnd % len(names)]
        for name in order:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            torch.cuda._sleep(prequeue_cycles)
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / replays
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("invalid graph validation measurement")
            samples[name].append(elapsed)
    return samples


def graph_selection_study(
    op, runners, config, inputs, kwargs, args, *, prepare_validation=None
):
    """Fresh cache searches with retained pre-/post-tuning default controls."""
    from torch.utils._pytree import tree_map

    policies = {
        "auto": None,
        "eager": MeasurementPolicy(execution_mode="eager", cold_l2=args.cold_l2),
        "cuda_graph": MeasurementPolicy(
            execution_mode="cuda_graph", cold_l2=args.cold_l2
        ),
        "individual": MeasurementPolicy(
            execution_mode="cuda_graph", cold_l2=args.cold_l2, graph_timing="individual"
        ),
    }
    if args.cold_l2 is not None:
        policies["auto"] = MeasurementPolicy(cold_l2=args.cold_l2)
    tuning_config = (
        retained_input_config(config, inputs)
        if args.profile_inputs == "retained"
        else config
    )
    rows = []
    names = args.graph_policies
    for trial in range(args.seeds):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            seed = args.seed + trial
            torch.manual_seed(seed)
            policy = policies[name]
            tuner = AutoTuner(warmup=args.warmup, repeat=args.tuner_repeat)
            measurements = []
            original_profile = tuner._profile_single_kernel
            observed_config = None

            def record(
                runner, profile_inputs, tactic, effective_config, **profile_kwargs
            ):
                nonlocal observed_config
                observed_config = effective_config
                measurement = {
                    "runner": type(runner).__name__,
                    "tactic": repr(tactic),
                    "input_shapes": input_shapes(profile_inputs),
                    "cache_profile": tuner._find_nearest_profile(
                        input_shapes(profile_inputs), effective_config
                    ),
                    "ms": None,
                    "use_cuda_graph": effective_config.use_cuda_graph,
                    "cold_l2": effective_config.use_cold_l2_cache,
                    "repeat": tuner._get_profiling_repeat(effective_config),
                    "warmup": tuner.warmup,
                    "profiling_policy": tuner._profiling_policy(effective_config),
                }
                measurements.append(measurement)
                try:
                    elapsed = original_profile(
                        runner,
                        profile_inputs,
                        tactic,
                        effective_config,
                        **profile_kwargs,
                    )
                except Exception as exc:
                    measurement["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                measurement["ms"] = elapsed if math.isfinite(elapsed) else None
                return elapsed

            tuner._profile_single_kernel = record
            with use_isolated_tuner(AutoTuner()):
                before, before_output = capture_tactic_graph(
                    runners[0], inputs, -1, kwargs, args.warmup
                )
                reference = tree_map(
                    lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x,
                    before_output,
                )
            with use_isolated_tuner(tuner):
                with autotune_v2(persistent_cache=False, measurement_policy=policy):
                    selected, tactic = tuner.choose_one(
                        op, runners, tuning_config, inputs, **kwargs
                    )
                    shapes = input_shapes(inputs)
                    hit, _, _, _ = tuner.search_cache(
                        op, runners, shapes, observed_config or tuning_config, inputs
                    )
                    if not hit:
                        raise RuntimeError(
                            "selection produced no cached winner; refusing to score fallback as tuning"
                        )
                # Keep the search policy active for any nested lookup in capture.
                with autotune_v2(
                    mode="replay", persistent_cache=False, measurement_policy=policy
                ):
                    chosen, chosen_output = capture_tactic_graph(
                        selected, inputs, tactic, kwargs, args.warmup
                    )
            with use_isolated_tuner(AutoTuner()):
                after, after_output = capture_tactic_graph(
                    runners[0], inputs, -1, kwargs, args.warmup
                )
            graphs = {
                "default_before": before,
                "selected": chosen,
                "default_after": after,
            }
            samples = interleaved_graph_samples(
                graphs,
                args.oracle_rounds,
                args.validation_replays,
                args.prequeue_cycles,
            )
            # Different tactics can bind the same caller-owned output buffer.
            # Replay and check each graph before another graph overwrites it.
            for graph, output in (
                (before, before_output),
                (chosen, chosen_output),
                (after, after_output),
            ):
                if prepare_validation is not None:
                    prepare_validation(output, inputs)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, reference, rtol=args.rtol, atol=args.atol
                )
            medians = {key: statistics.median(value) for key, value in samples.items()}
            rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "policy": name,
                    "runner": type(selected).__name__,
                    "tactic": repr(tactic),
                    "cache_hit": hit,
                    "input_shapes": shapes,
                    "cache_profile": tuner._find_nearest_profile(
                        shapes, observed_config or tuning_config
                    ),
                    "profile_count": len(measurements),
                    "measurements": measurements,
                    "validation_ms": samples,
                    "median_ms": medians,
                    "selected_over_default_after": medians["selected"]
                    / medians["default_after"],
                    "retained_over_fresh_default": medians["default_before"]
                    / medians["default_after"],
                    "numerics": "passed",
                    "sm_clock_mhz": sm_clock_mhz(),
                }
            )
    return {
        "op": op,
        "scope": "single-operation hot-L2 graph replay; not serving throughput",
        "profile_inputs": args.profile_inputs,
        "trials": rows,
        "trial_isolation": "fresh tuner/cache; same process and retained runner/input objects",
        "validation": {
            "output_refresh": "fixture_callback"
            if prepare_validation is not None
            else "unavailable",
            "events": "outside captured graph",
            "prequeue_cycles": args.prequeue_cycles,
            "replays": args.validation_replays,
            "rounds": args.oracle_rounds,
            "calls_per_graph": 1,
            "warmup": args.warmup,
            "atol": args.atol,
            "rtol": args.rtol,
        },
    }


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
    parser.add_argument("--graph-selection-study", action="store_true")
    parser.add_argument(
        "--fixture", help="module:function returning one public-API call from args"
    )
    parser.add_argument(
        "--graph-policies",
        nargs="+",
        choices=["auto", "eager", "cuda_graph", "individual"],
        default=["auto", "eager", "cuda_graph"],
    )
    parser.add_argument(
        "--profile-inputs", choices=["retained", "synthesized"], default="retained"
    )
    parser.add_argument(
        "--cold-l2", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--validation-replays", type=int, default=16)
    parser.add_argument("--prequeue-cycles", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.graph_selection_study:
        for field in (
            "seeds",
            "oracle_rounds",
            "tuner_repeat",
            "warmup",
            "validation_replays",
            "prequeue_cycles",
        ):
            if getattr(args, field) < 1:
                parser.error(f"{field} must be positive")
        for field in ("atol", "rtol"):
            if not math.isfinite(getattr(args, field)) or getattr(args, field) < 0:
                parser.error(f"{field} must be finite and nonnegative")
        if not args.fixture and (
            len(args.ops) != 1 or args.grid is None or len(args.grid) != 1
        ):
            parser.error(
                "graph study needs one --ops value and one --grid value, or --fixture"
            )
        torch.manual_seed(args.seed)
        if args.fixture:
            module, factory = args.fixture.split(":", 1)
            call = getattr(importlib.import_module(module), factory)(args)
        else:
            call = OPS[args.ops[0]]["build"](args.grid[0], args)
        reference = getattr(call, "reference", None)
        reference_kind = "untuned default only"
        reference_error = None
        if reference is not None:
            with use_isolated_tuner(AutoTuner()):
                actual = call()
                expected = reference()
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=args.rtol, atol=args.atol
                )
                error = actual.float() - expected.float()
                reference_error = {
                    "max_abs": error.abs().max().item(),
                    "relative_l2": (
                        error.norm() / expected.float().norm().clamp_min(1e-30)
                    ).item(),
                }
            reference_kind = "fixture reference and untuned default"
        with use_isolated_tuner(AutoTuner()):
            captured = capture_choose_one_calls(call, require_single=True)
        if len(captured) != 1:
            raise ValueError(
                "graph study requires a fixture with one tunable operation"
            )
        op, (runners, config, inputs, kwargs) = next(iter(captured.items()))
        try:
            result = graph_selection_study(
                op,
                runners,
                config,
                inputs,
                kwargs,
                args,
                prepare_validation=getattr(call, "prepare_validation", None),
            )
        except Exception as exc:
            with open(args.out, "w") as handle:
                json.dump(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "arguments": vars(args),
                    },
                    handle,
                    indent=2,
                )
            raise
        import flashinfer

        source_files = [Path(__file__), Path(inspect.getfile(AutoTuner))]
        try:
            source_commit = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[1]),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
        result.update(
            source_commit=source_commit,
            source_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files
            },
            flashinfer=flashinfer.__version__,
            reference=reference_kind,
            reference_error=reference_error,
            input_metadata=[
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "stride": list(t.stride()),
                }
                if isinstance(t, torch.Tensor)
                else {"type": type(t).__name__}
                for t in inputs
            ],
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            arguments=vars(args),
        )
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(f"graph study written to {args.out}")
        return

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
                try:
                    row = sweep_captured_op(
                        op, runners, config, inputs, kwargs, args, tuner, grid_label
                    )
                except MeasurementUnavailable as exc:
                    print(f"\n{grid_label}: MEASUREMENT UNAVAILABLE: {exc}")
                    report["shapes"].append(
                        {
                            "api": api_name,
                            "op": op,
                            spec["axis"]: pt,
                            "error": f"MeasurementUnavailable: {exc}",
                        }
                    )
                    continue
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
        if not any(math.isfinite(v) for v in oracle.values()) or not any(
            math.isfinite(v) for v in oracle_eager.values()
        ):
            raise MeasurementUnavailable("no finite oracle candidates")
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
                if not any(math.isfinite(v) for v in times.values()):
                    raise MeasurementUnavailable(f"no finite candidates for {pname}")
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
