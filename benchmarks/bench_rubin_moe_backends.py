# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare Rubin NVFP4 Prims-TS and CuTe DSL MoE on identical nonzero data.

Matches bench_rubin_moe's seed, shapes, balanced logits and quantization.
Weights are prepared separately in each backend's native layout. Preparation,
input quantization and compilation are outside timing. Both backends start with
supplied logits; the gate GEMM producing those logits is excluded. CuTe DSL is measured
both from logits (using trtllm_gen_routing) and from precomputed top-k inputs;
its own sorting and fused finalize are included in both measurements.

By default, sweep every shape-valid CuTe DSL tactic on the actual inputs,
then independently remeasure the default and fastest pair against Prims-TS.
This is an explicit sweep, not the framework tuner's synthetic-input search.
The current CuTe MoE menu enumerates routing tiles 128/256; its tile512
geometry is not reachable through that menu and is not measured here.
CUDA events cover ten complete blocks per graph, without L2 flushing.
Use --breakdown for per-stage events and --tactics-from REPORT to repeat
the winners of an earlier sweep. CuTe FC2 includes fused finalization;
its auxiliary-stream output zeroing overlaps FC1 and is not additive.
Native Prims FC2 also fuses finalize by default; its output zeroing overlaps
FC1 and is reported separately. The same run includes the matching Prims
unfused pair for comparison.
Use --cute-unfused-finalize to time CuTe's FC2 and deterministic finalizer
separately, and --disable-pdl to compare both backends without PDL.

CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ARCH=sm_107a python -m \
    benchmarks.bench_rubin_moe_backends --output runs/rubin_moe_backends.json
"""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch

from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import CuteDslConfig, PrimsTsConfig
from flashinfer.fused_moe import prims_ts_fp4_block_scale_moe
from flashinfer.fused_moe.api import QuantConfig, QuantFormat
from flashinfer.fused_moe.cute_dsl.fused_moe import CuteDslMoEWrapper
from flashinfer.fused_moe.cute_dsl.tuner import DEFAULT_RUBIN_MOE_TACTIC
from flashinfer.fused_moe.trtllm_gen_routing import trtllm_gen_routing
from flashinfer.testing import bench_gpu_time
from flashinfer.tllm_enums import RoutingMethodType, WeightLayout

from benchmarks.bench_rubin_moe import (
    capture_breakdown as capture_prims_breakdown,
    measure_breakdown,
)


def capture_cute_breakdown(run, route, *, from_logits, fused_finalize=True):
    """Measure stage boundaries, keeping fused FC2/finalize together.

    Output zeroing runs on the existing auxiliary stream and overlaps FC1.
    Record it separately; only its remaining wait is in fc2_finalize.
    External event nodes can inhibit PDL overlap, so always compare against
    a second graph without these markers.
    """
    import importlib

    core = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    fc2_stage = "fc2_finalize" if fused_finalize else "fc2"
    stages = {name: [] for name in ("sorting", "fc1", fc2_stage)}
    stages["output_zero_overlapped" if fused_finalize else "finalize"] = []
    if from_logits:
        stages["external_routing"] = []
    state = {}

    def event():
        marker = torch.cuda.Event(enable_timing=True, external=True)
        marker.record()
        return marker

    def instrument(fn, stage):
        def call(*args, **kwargs):
            result = fn(*args, **kwargs)
            end = event()
            stages[stage].append((state["boundary"], end))
            state["boundary"] = end
            return result

        return call

    memset = core.moe_output_memset_inplace

    def instrument_memset(*args, **kwargs):
        start = event()  # Record on the existing auxiliary stream.
        result = memset(*args, **kwargs)
        stages["output_zero_overlapped"].append((start, event()))
        return result

    wrapped_route = instrument(route, "external_routing") if from_logits else route
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with ExitStack() as stack:
        for name, stage in (
            ("moe_sort", "sorting"),
            ("blockscaled_contiguous_gather_grouped_gemm_act_fusion", "fc1"),
            ("blockscaled_contiguous_grouped_gemm_finalize_fusion", fc2_stage),
        ):
            stack.enter_context(
                patch.object(core, name, instrument(getattr(core, name), stage))
            )
        if fused_finalize:
            stack.enter_context(
                patch.object(core, "moe_output_memset_inplace", instrument_memset)
            )
        else:
            finalize = importlib.import_module(
                "flashinfer.fused_moe.cute_dsl.blackwell.moe_finalize"
            )
            stack.enter_context(
                patch.object(
                    finalize,
                    "moe_unpermute",
                    instrument(finalize.moe_unpermute, "finalize"),
                )
            )
        with torch.cuda.graph(graph):
            for _ in range(10):
                state["boundary"] = event()
                outputs.append(run(wrapped_route))
    assert all(len(events) == 10 for events in stages.values()), {
        name: len(events) for name, events in stages.items()
    }
    graph.replay()
    torch.cuda.synchronize()
    return graph, stages, outputs


def _tuple_tree(value):
    return (
        tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value
    )


def capture(run, blocks=10):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [run() for _ in range(blocks)]
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def measure(graph, *, blocks=10, samples=30, warm_ms=100):
    return [
        ms * 1000 / blocks
        for ms in bench_gpu_time(
            graph.replay,
            dry_run_time_ms=warm_ms,
            repeat_iters=samples,
            cold_l2_cache=False,
        )
    ]


def check(output, reference):
    output = output.float()
    error = float((output - reference).norm() / reference.norm())
    stats = dict(
        relative_l2=error,
        min=float(output.min()),
        max=float(output.max()),
        rms=float(output.square().mean().sqrt()),
        nonzero_fraction=float(torch.count_nonzero(output)) / output.numel(),
    )
    assert torch.isfinite(output).all() and error < 0.12, stats
    assert stats["nonzero_fraction"] > 0.99, stats
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prims-unfused-finalize", action="store_true")
    parser.add_argument("--default-only", action="store_true")
    parser.add_argument(
        "--cute-unfused-finalize",
        action="store_true",
        help="Use CuTe's separate deterministic finalizer for an unfused comparison.",
    )
    parser.add_argument(
        "--disable-pdl", action="store_true", help="Disable PDL in both backends."
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Measure stage boundaries in a separate instrumented graph.",
    )
    parser.add_argument(
        "--tactics-from",
        type=Path,
        help="Reuse winning CuTe tactics from a prior report instead of sweeping.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("Requires SM107.")
    if min(args.tokens) < 1 or args.rounds < 1:
        parser.error("Token counts and rounds must be positive.")
    if args.default_only and args.tactics_from:
        parser.error("--default-only and --tactics-from are mutually exclusive.")
    saved_tactics = {}
    if args.tactics_from:
        saved = json.loads(args.tactics_from.read_text())
        if saved.get("default_only", False):
            parser.error("A default-only report does not contain sweep winners.")
        if tuple(
            saved.get(key) for key in ("experts", "hidden", "intermediate", "top_k")
        ) != (32, 10240, 4096, 8):
            parser.error("The prior report must use E32, H10240, I4096, top-k 8.")
        saved_tactics = {
            row["tokens"]: _tuple_tree(row["tactic"])
            for row in saved["results"]
            if row["backend"] == "cute_best_from_logits"
            and row.get("is_sweep_winner", True)
        }
        if any(tokens not in saved_tactics for tokens in args.tokens):
            parser.error(
                "The prior report must contain a CuTe winner for every token count."
            )
    torch.manual_seed(17)
    e, h, i, k = 32, 10240, 4096, 8
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) / h**0.5
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) / i**0.5
    prep = dict(quant=quant, num_local_experts=e, hidden_size=h, intermediate_size=i)
    prims_weights = PrimsTsConfig.prepare_weights(
        w1, w2, **prep, nvfp4_weight_layout="BlockMajorK"
    )
    cute_weights = CuteDslConfig.prepare_weights(w1, w2, **prep)
    # Layout metadata belongs to the unified API, not the direct wrapper.
    gated_weight_group_size = cute_weights.pop("gated_weight_group_size")
    assert gated_weight_group_size == 64
    alpha2 = torch.linspace(0.7, 1.3, e, device="cuda")
    prims_weights["output2_scale_scalar"] = alpha2
    cute_weights["w2_alpha"] = alpha2
    del w1, w2
    moe = CuteDslMoEWrapper(
        num_experts=e,
        top_k=k,
        hidden_size=h,
        intermediate_size=i,
        use_cuda_graph=True,
        use_fused_finalize=not args.cute_unfused_finalize,
        enable_pdl=not args.disable_pdl,
    )
    report = dict(
        tokens=args.tokens,
        experts=e,
        hidden=h,
        intermediate=i,
        top_k=k,
        seed=17,
        rounds=args.rounds,
        gpu=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),
        timer="CUDA events; 10 blocks per graph; 30 samples per round; warm L2",
        gate_gemm_included=False,
        routing="Balanced Renormalize logits; CuTe from-logits uses trtllm_gen_routing plus FP32 router-weight cast",
        selection=(
            "CuTe default tactic only; no sweep winner"
            if args.default_only
            else "Explicit shape-valid CuTe tactic sweep using actual inputs and CUDA graphs"
        ),
        default_only=args.default_only,
        tactics_from=str(args.tactics_from) if args.tactics_from else None,
        breakdown=args.breakdown,
        cute_fused_finalize=not args.cute_unfused_finalize,
        prims_fused_finalize=not args.prims_unfused_finalize,
        requested_enable_pdl=not args.disable_pdl,
        breakdown_note=(
            "External CUDA events can inhibit PDL overlap. "
            + (
                "CuTe FC2 and finalize are separate launches."
                if args.cute_unfused_finalize
                else "CuTe FC2 includes fused finalize and any remaining output-zeroing wait; output_zero_overlapped is not additive."
            )
        ),
        results=[],
        sweeps=[],
    )
    if args.tactics_from:
        report["selection"] = (
            "Replay prior winning tactics, validated against the current candidate menu"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for tokens in args.tokens:
        x, sf = PrimsTsConfig.prepare_activations(
            torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16), quant=quant
        )
        logits = torch.randn(tokens, e, device="cuda", dtype=torch.float32)
        ids = (
            torch.arange(tokens, device="cuda")[:, None] * k
            + torch.arange(k, device="cuda")
        ) % e
        logits.fill_(-10).scatter_(1, ids, 10)

        def route():
            result = trtllm_gen_routing(
                logits, None, RoutingMethodType.Renormalize, k, enable_pdl=False
            )
            # SM107 CuTe finalize requires FP32 router weights. Include this
            # adapter in from-logits timing; balanced 1/8 weights are exact.
            return result._replace(topk_weights=result.topk_weights.float())

        routing = route()
        assert torch.equal(routing.topk_ids.sort(-1).values, ids.sort(-1).values)
        torch.testing.assert_close(
            routing.topk_weights.float(),
            torch.full_like(ids, 1 / k, dtype=torch.float32),
        )
        prims_kwargs = dict(
            routing_logits=logits,
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=sf,
            **prims_weights,
            gemm1_bias=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_bias=None,
            num_experts=e,
            top_k=k,
            n_group=None,
            topk_group=None,
            intermediate_size=i,
            local_expert_offset=0,
            local_num_experts=e,
            routed_scaling_factor=None,
            routing_method_type=int(RoutingMethodType.Renormalize),
            weight_layout=int(WeightLayout.BlockMajorK),
            enable_pdl=not args.disable_pdl,
        )
        selected = [-1, -1]

        def choose(custom_op, runners, tuning_config, inputs, **kw):
            return runners[0], selected

        prims_fused_finalize = False

        def run_prims():
            return prims_ts_fp4_block_scale_moe(
                **prims_kwargs, use_fused_finalize=prims_fused_finalize
            )[0]

        cute_kwargs = dict(x=x, x_sf=sf.view(torch.uint8).unsqueeze(-1), **cute_weights)

        def run_cute(tactic, from_logits=True, *, route_fn=None):
            routed = (route_fn or route)() if from_logits else routing
            return moe.run(
                **cute_kwargs,
                token_selected_experts=routed.topk_ids,
                token_final_scales=routed.topk_weights,
                tactic=tactic,
            )

        print(f"T={tokens}: warmup and reference", flush=True)
        with patch.object(AutoTuner.get(), "choose_one", choose):
            reference = run_prims().float()
            graphs = {}
            breakdown_graphs = {}
            prims_variants = [("prims_default", [-1, -1], False)]
            if not args.prims_unfused_finalize:
                prims_variants.append(("prims_1024_unfused", [1024, 0], False))
            prims_variants.append(
                ("prims_1024", [1024, 0], not args.prims_unfused_finalize)
            )
            for label, selected, fused in prims_variants:
                prims_fused_finalize = fused
                if selected[0] == 1024 and tokens * k // e < 512:
                    continue
                graph, outputs = capture(run_prims)
                graphs[label] = (
                    graph,
                    outputs,
                    check(outputs[-1], reference),
                    selected,
                )
                if args.breakdown:
                    breakdown_graphs[label] = capture_prims_breakdown(run_prims)
                    check(breakdown_graphs[label][2][-1], reference)

        inputs = [x, cute_kwargs["x_sf"], routing.topk_ids, routing.topk_weights]
        inputs.extend(
            cute_weights[name]
            for name in (
                "w1_weight",
                "w1_weight_sf",
                "w1_alpha",
                "fc2_input_scale",
                "w2_weight",
                "w2_weight_sf",
                "w2_alpha",
            )
        )
        available_tactics = moe._runner.get_valid_tactics(inputs, None)
        tactics = (
            [saved_tactics[tokens]]
            if args.tactics_from
            else (
                [DEFAULT_RUBIN_MOE_TACTIC] if args.default_only else available_tactics
            )
        )
        assert DEFAULT_RUBIN_MOE_TACTIC in available_tactics
        assert all(tactic in available_tactics for tactic in tactics)
        sweep = dict(tokens=tokens, candidates=[])
        report["sweeps"].append(sweep)
        best = (float("inf"), None)
        for idx, tactic in enumerate(tactics):
            print(
                f"T={tokens}: CuTe candidate {idx + 1}/{len(tactics)} {tactic}",
                flush=True,
            )
            graph, outputs = capture(lambda: run_cute(tactic), blocks=2)
            stats = check(outputs[-1], reference)
            times = measure(graph, blocks=2, samples=10, warm_ms=20)
            median = statistics.median(times)
            sweep["candidates"].append(
                dict(
                    tactic=tactic,
                    median_us=median,
                    samples_us=times,
                    output_stats=stats,
                )
            )
            print(
                f"  {median:.3f} us; relative L2={stats['relative_l2']:.6f}", flush=True
            )
            if median < best[0]:
                best = (median, tactic)
            del graph, outputs
            gc.collect()
            save()
        sweep["winner"] = None if args.default_only else best[1]
        for name, tactic in (("default", DEFAULT_RUBIN_MOE_TACTIC), ("best", best[1])):
            for from_logits in (True, False):
                label = f"cute_{name}_{'from_logits' if from_logits else 'prerouted'}"
                graph, outputs = capture(lambda: run_cute(tactic, from_logits))
                graphs[label] = (graph, outputs, check(outputs[-1], reference), tactic)
                if args.breakdown:
                    breakdown_graphs[label] = capture_cute_breakdown(
                        lambda route_fn: run_cute(
                            tactic, from_logits, route_fn=route_fn
                        ),
                        route,
                        from_logits=from_logits,
                        fused_finalize=not args.cute_unfused_finalize,
                    )
                    check(breakdown_graphs[label][2][-1], reference)
        samples = {label: [] for label in graphs}
        breakdown_samples = {
            label: {stage: [] for stage in (*stages, "instrumented_moe")}
            for label, (_, stages, _) in breakdown_graphs.items()
        }
        for round_id in range(args.rounds):
            order = list(graphs) if round_id % 2 == 0 else list(reversed(graphs))
            for label in order:
                samples[label].extend(measure(graphs[label][0]))
                if args.breakdown:
                    instrumented_graph, stages, _ = breakdown_graphs[label]
                    measure_breakdown(
                        instrumented_graph, stages, breakdown_samples[label]
                    )
        for label, times in samples.items():
            result = dict(
                tokens=tokens,
                backend=label,
                median_us=statistics.median(times),
                samples_us=times,
                output_stats=graphs[label][2],
                tactic=graphs[label][3],
                is_sweep_winner=label.startswith("cute_best_")
                and not args.default_only,
            )
            result["moe_tflops"] = 6 * tokens * k * h * i / (result["median_us"] * 1e6)
            if args.breakdown:
                components = breakdown_samples[label]
                if label.startswith("cute_"):
                    external = components.get(
                        "external_routing", [0.0] * len(components["sorting"])
                    )
                    components["routing"] = [
                        a + b
                        for a, b in zip(external, components["sorting"], strict=True)
                    ]
                result["breakdown_samples_us"] = components
                result["breakdown_median_us"] = {
                    stage: statistics.median(values)
                    for stage, values in components.items()
                }
            report["results"].append(result)
            print(f"T={tokens} {label}: {result['median_us']:.3f} us", flush=True)
            if args.breakdown:
                print(json.dumps(result["breakdown_median_us"]), flush=True)
        save()
        del graphs, graph, outputs, reference, inputs, breakdown_graphs
        if args.breakdown:
            del instrumented_graph, stages
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
