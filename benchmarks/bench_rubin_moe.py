# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare full Rubin MoE blocks using one prepared weight set.

The comparison forces the untuned swapAB default and the three native pairs;
it does not compare against an exhaustive autotune of the legacy menu.
Timing includes routing, FC1, FC2 and finalize in CUDA graphs, starting from
supplied routing logits. The gate GEMM that produces those logits, preparation
and compilation are outside timing. Expert counts are balanced by default.
Add --breakdown to measure routing, FC1, FC2 and finalize inside a second
graph with timing events. The ordinary total remains uninstrumented, so the
event overhead can be checked. Reported TFLOPS counts useful GEMM work;
input quantization is outside timing, while FC1's fused activation and
intermediate quantization are included in its elapsed time.

CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ARCH=sm_107a python -m benchmarks.bench_rubin_moe \
    --tokens 32 4096 8192 --layout BlockMajorK --output runs/rubin_moe.json
"""

import argparse
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch

from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import PrimsTsConfig, prims_ts_fp4_block_scale_moe
from flashinfer.fused_moe.api import QuantConfig, QuantFormat
from flashinfer.prims_ts.moe import runner as moe_runner
from flashinfer.testing import bench_gpu_time
from flashinfer.tllm_enums import RoutingMethodType, WeightLayout


def capture_breakdown(run):
    """Time routing, FC1, FC2 and finalize inside the complete pipeline."""
    from flashinfer.prims_ts.moe import finalize

    stages = {
        name: []
        for name in (
            "routing",
            "output_zero_overlapped",
            "fc1",
            "fc2",
            "fc2_finalize",
            "finalize",
        )
    }
    zero_output = finalize._zero_output

    def instrument_zero(output, stream):
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        start.record(stream)
        zero_output(output, stream)
        end.record(stream)
        # Auxiliary-stream time overlaps FC1; do not add it to main-stream stages.
        stages["output_zero_overlapped"].append((start, end))

    block_state = {}
    compile_gemm = moe_runner.get_compiled_gemm

    def instrument(key, label, io, stream):
        compiled = compile_gemm(key, label, io, stream)
        stage = label.removeprefix("nvfp4_")
        if stage == "fc1":
            start = torch.cuda.Event(enable_timing=True, external=True)
            stages["routing"].append((block_state["boundary"], start))
        else:
            assert stage == "fc2"
            block_state["fused"] = bool(io["cfg"].fuse_moe_finalize)
            if block_state["fused"]:
                stage = "fc2_finalize"
            start = block_state["boundary"]
        end = torch.cuda.Event(enable_timing=True, external=True)
        stages[stage].append((start, end))

        def launch(*args):
            if stage == "fc1":
                start.record()
            compiled(*args)
            end.record()
            block_state["boundary"] = end

        return launch

    graph = torch.cuda.CUDAGraph()
    outputs = []
    with (
        patch.object(moe_runner, "get_compiled_gemm", instrument),
        patch.object(finalize, "_zero_output", instrument_zero),
        torch.cuda.graph(graph),
    ):
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True, external=True)
            start.record()
            block_state["boundary"] = start
            outputs.append(run())
            if not block_state.get("fused", False):
                end = torch.cuda.Event(enable_timing=True, external=True)
                end.record()
                stages["finalize"].append((block_state["boundary"], end))
    stages = {name: events for name, events in stages.items() if events}
    assert all(len(events) == 10 for events in stages.values())
    graph.replay()
    torch.cuda.synchronize()
    return graph, stages, outputs


def measure_breakdown(graph, stages, samples):
    # Warm the complete pipeline. Event markers preserve its actual input
    # buffers; no FC1-only/FC2-only cache warming is used.
    bench_gpu_time(
        graph.replay, dry_run_time_ms=100, repeat_iters=1, cold_l2_cache=False
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(30):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples["instrumented_moe"].append(start.elapsed_time(end) * 1000 / 10)
        for stage, events in stages.items():
            samples[stage].append(sum(a.elapsed_time(b) for a, b in events) * 1000 / 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 4096, 8192])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=10240)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--layout", choices=("MajorK", "BlockMajorK"), default="BlockMajorK"
    )
    parser.add_argument("--routing", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prims-unfused-finalize", action="store_true")
    parser.add_argument(
        "--tactics",
        nargs="+",
        choices=("default", "512_0", "512_1", "1024_0"),
        default=["default", "512_0", "512_1", "1024_0"],
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Also time routing, FC1, FC2 and finalize inside a separate full-block graph.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("This benchmark requires SM107.")
    if not 0 < args.top_k <= args.experts or args.rounds < 1 or min(args.tokens) < 1:
        parser.error(
            "Require 0 < top-k <= experts, rounds >= 1 and positive token counts."
        )
    if args.hidden <= 0 or args.intermediate <= 0:
        parser.error("Hidden and intermediate dimensions must be positive.")
    if args.layout == "BlockMajorK" and (args.hidden % 256 or args.intermediate % 256):
        parser.error(
            "BlockMajorK hidden and intermediate dimensions must be divisible by 256."
        )
    tactics_by_tokens = {
        tokens: [
            [-1, -1] if name == "default" else list(map(int, name.split("_")))
            for name in args.tactics
            if name == "default" or tokens * args.top_k / args.experts >= 512
        ]
        for tokens in args.tokens
    }
    for tokens, tactics in tactics_by_tokens.items():
        if not tactics:
            parser.error(f"All requested tactics are excluded for tokens={tokens}.")
    torch.manual_seed(17)
    e, h, i, k = args.experts, args.hidden, args.intermediate, args.top_k
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    weights = PrimsTsConfig.prepare_weights(
        torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) / h**0.5,
        torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) / i**0.5,
        quant=quant,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        nvfp4_weight_layout=args.layout,
    )
    weights["output2_scale_scalar"] = torch.linspace(0.7, 1.3, e, device="cuda")
    report = dict(
        vars(args),
        output=str(args.output),
        gpu=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),
        timer="CUDA events, ten complete blocks per graph",
        gate_gemm_included=False,
        results=[],
    )
    for tokens in args.tokens:
        x, sf = PrimsTsConfig.prepare_activations(
            torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16), quant=quant
        )
        logits = torch.randn(tokens, e, device="cuda", dtype=torch.float32)
        if args.routing == "balanced":
            ids = (
                torch.arange(tokens, device="cuda")[:, None] * k
                + torch.arange(k, device="cuda")
            ) % e
            logits.fill_(-10).scatter_(1, ids, 10)
        kwargs = dict(
            routing_logits=logits,
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=sf,
            **weights,
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
            weight_layout=int(getattr(WeightLayout, args.layout)),
            enable_pdl=True,
            use_fused_finalize=not args.prims_unfused_finalize,
        )
        selected = [-1, -1]

        def choose(custom_op, runners, tuning_config, inputs, **kw):
            return runners[0], selected

        def run():
            return prims_ts_fp4_block_scale_moe(**kwargs)[0]

        graphs, outputs, samples, errors, configs, output_stats = {}, [], {}, {}, {}, {}
        breakdown_graphs, breakdown_samples = {}, {}
        tactics = tactics_by_tokens[tokens]
        with patch.object(AutoTuner.get(), "choose_one", choose):
            baseline = run().clone().float()
            for selected in tactics:
                label = str(selected)
                configs[label] = {}
                compile_gemm = moe_runner.get_compiled_gemm

                def record_config(key, stage, io, stream):
                    cfg = io["cfg"]
                    fc = stage.removeprefix("nvfp4_")
                    configs[label][fc] = dict(
                        swap_ab=cfg.is_swap_ab,
                        token_tile=cfg.tile_token,
                        cga=list(cfg.preferred_cluster_shape),
                        workid_stages=cfg.num_stages_workid,
                    )
                    return compile_gemm(key, stage, io, stream)

                with patch.object(moe_runner, "get_compiled_gemm", record_config):
                    output = run()
                error = float((output.float() - baseline).norm() / baseline.norm())
                assert torch.isfinite(output).all() and error < 0.12, (label, error)
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10):
                        outputs.append(run())
                graph.replay()
                torch.cuda.synchronize()
                replay_errors = []
                for output in outputs[-10:]:
                    error = float((output.float() - baseline).norm() / baseline.norm())
                    assert torch.isfinite(output).all() and error < 0.12, (label, error)
                    replay_errors.append(error)
                errors[label] = max(replay_errors)
                output_stats[label] = dict(
                    min=float(output.min()),
                    max=float(output.max()),
                    rms=float(output.float().square().mean().sqrt()),
                    nonzero_fraction=float(torch.count_nonzero(output))
                    / output.numel(),
                )
                graphs[label], samples[label] = graph, []
                if args.breakdown:
                    timed_graph, stages, timed_outputs = capture_breakdown(run)
                    outputs.extend(timed_outputs)
                    breakdown_graphs[label] = (timed_graph, stages)
                    breakdown_samples[label] = {
                        stage: [] for stage in (*stages, "instrumented_moe")
                    }
            for round_id in range(args.rounds):
                order = list(graphs) if round_id % 2 == 0 else list(reversed(graphs))
                for label in order:
                    # Events cover the complete graph's elapsed time, including
                    # gaps between stages; ten blocks amortize launch overhead.
                    times = bench_gpu_time(
                        graphs[label].replay,
                        dry_run_time_ms=100,
                        repeat_iters=30,
                        cold_l2_cache=False,
                    )
                    samples[label].extend(t * 1000 / 10 for t in times)
                    if args.breakdown:
                        measure_breakdown(
                            *breakdown_graphs[label], breakdown_samples[label]
                        )
        for label, values in samples.items():
            result = dict(
                tokens=tokens,
                tactic=json.loads(label),
                median_us=statistics.median(values),
                relative_l2=errors[label],
                configs=configs[label],
                output_stats=output_stats[label],
                samples_us=values,
            )
            # Dense-equivalent useful GEMM work: each FMA counts as two FLOPs.
            # FC1 contains both gate and up projections. FC2 contains down.
            fc1_flops, fc2_flops = 4 * tokens * k * h * i, 2 * tokens * k * h * i
            result["flops"] = dict(
                fc1=fc1_flops, fc2=fc2_flops, moe=fc1_flops + fc2_flops
            )
            result["moe_tflops"] = (fc1_flops + fc2_flops) / (result["median_us"] * 1e6)
            if args.breakdown:
                result["breakdown_samples_us"] = breakdown_samples[label]
                result["breakdown_median_us"] = {
                    stage: statistics.median(values)
                    for stage, values in breakdown_samples[label].items()
                }
                result["stage_tflops"] = {
                    stage: result["flops"][stage]
                    / (
                        result["breakdown_median_us"][
                            "fc2_finalize"
                            if stage == "fc2"
                            and "fc2_finalize" in result["breakdown_median_us"]
                            else stage
                        ]
                        * 1e6
                    )
                    for stage in ("fc1", "fc2")
                }
            report["results"].append(result)
            print(
                f"T={tokens} tactic={label}: {result['median_us']:.3f} us, relative L2={errors[label]:.5f}",
                flush=True,
            )
            if args.breakdown:
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("samples_us", "breakdown_samples_us")
                        }
                    ),
                    flush=True,
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del graphs, breakdown_graphs, outputs


if __name__ == "__main__":
    main()
