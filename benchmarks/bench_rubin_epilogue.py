# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Sweep validated FC1/FC2 epilogues on nonzero inputs of a real MoE block.

The isolated graph includes output zeroing before every FC2 launch. Separate
external event intervals report FC2 alone. Select the final winner in complete
MoE graphs from the best two candidates and the legacy default. No gate projection, preparation, JIT or host
time is included. Same-buffer warm-cache timing; no GPU clock modifications.
With --fc fc1, every candidate is measured inside complete MoE graphs, with
external events around FC1. FC2/finalization remains at its legacy setting.

CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_107a python -m \
    benchmarks.bench_rubin_epilogue --case e384 --max-candidates 8 --output result.json
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from cuda.bindings import driver as cuda

from benchmarks.bench_rubin_moe_backends import capture, check as _check
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import PrimsTsConfig, prims_ts_fp4_block_scale_moe
from flashinfer.fused_moe.api import QuantConfig, QuantFormat
from flashinfer.prims_ts.batched_gemm import batched_gemm_run as gemm
from flashinfer.prims_ts.batched_gemm.epilogue_search import iter_epilogue_configs
from flashinfer.prims_ts.moe import runner
from flashinfer.prims_ts.moe.compile_cache import (
    _disk_kernel_name,
    get_compiled_gemm,
    stable_config_hash,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.tllm_enums import RoutingMethodType, WeightLayout


CASES = {
    "small": (1031, 8, 1024, 512, 4),
    "e32": (4096, 32, 10240, 4096, 8),
    "e256": (16384, 256, 7168, 2048, 8),
    "e384": (24576, 384, 7168, 3072, 6),
}


def check(output, reference):
    result = _check(output, reference)
    assert result["relative_l2"] < 0.015, result
    return result


def timed(graph, *, samples, warm_ms=300):
    return [
        x * 100
        for x in bench_gpu_time(
            graph.replay,
            enable_cupti=True,
            dry_run_time_ms=warm_ms,
            repeat_iters=samples,
            cold_l2_cache=False,
        )
    ]  # ms -> us, divided by ten launches per graph


def benchmark_fc1(args, report, run, choose, reference, base, save):
    """Compare FC1 schedules with their real routing/FC2 consumers and caches."""
    configs = [base] + list(
        iter_epilogue_configs(
            base, exhaustive=args.exhaustive, max_candidates=args.max_candidates
        )
    )
    build_io = runner.build_nvfp4_launch_io
    graphs = []
    for index, cfg in enumerate(configs):
        row = dict(
            index=index,
            legacy=index == 0,
            warps=cfg.dense_epilogue_warps,
            c_stages=cfg.num_stages_c_smem,
            epilogue_regs=cfg.epilogue_regs,
            steps=[asdict(s) for s in cfg.epilogue_steps],
            fc1_samples_us=[],
            full_moe_samples_us=[],
        )
        report["candidates"].append(row)
        events = []

        def with_config(**kwargs):
            if kwargs["fc"] == "fc1":
                kwargs["cfg"] = cfg
            return build_io(**kwargs)

        def instrument(key, label, io, stream):
            fn = get_compiled_gemm(key, label, io, stream)
            if label != "nvfp4_fc1":
                return fn
            if "kernel_name" not in row:
                row["kernel_name"] = _disk_kernel_name(key, label, io)

            def launch(*launch_args):
                if not torch.cuda.is_current_stream_capturing():
                    return fn(*launch_args)
                a = torch.cuda.Event(enable_timing=True, external=True)
                b = torch.cuda.Event(enable_timing=True, external=True)
                a.record()
                fn(*launch_args)
                b.record()
                events.append((a, b))

            return launch

        print("Capturing FC1", index, row["warps"], row["c_stages"], flush=True)
        start = time.monotonic()
        with (
            patch.object(AutoTuner.get(), "choose_one", choose),
            patch.object(runner, "build_nvfp4_launch_io", with_config),
            patch.object(runner, "get_compiled_gemm", instrument),
        ):
            graph, outputs = capture(run)
        assert len(events) == 10
        row["compile_capture_seconds"] = time.monotonic() - start
        row["correctness"] = check(outputs[-1], reference)
        row["status"] = "passed"
        row["allocated_bytes"] = torch.cuda.memory_allocated()
        row["reserved_bytes"] = torch.cuda.memory_reserved()
        graphs.append((row, graph, outputs, events))
        save()
    # Compilation and allocation can leave the GPU at a different clock/thermal
    # state for the first candidate. Warm every graph before collecting the
    # interleaved rounds, keeping clock settings unchanged.
    report["initial_warmup"] = "two alternating passes, 1000 ms per candidate/pass"
    for warmup in range(2):
        for _, graph, _, _ in graphs if warmup == 0 else reversed(graphs):
            timed(graph, samples=3, warm_ms=1000)
    for round_index in range(args.rounds):
        order = graphs if round_index % 2 == 0 else reversed(graphs)
        for row, graph, outputs, events in order:
            row["full_moe_samples_us"].extend(timed(graph, samples=args.samples))
            for _ in range(args.samples):
                graph.replay()
                torch.cuda.synchronize()
                row["fc1_samples_us"].append(
                    sum(a.elapsed_time(b) for a, b in events) * 100
                )
            row["fc1_median_us"] = statistics.median(row["fc1_samples_us"])
            row["full_moe_median_us"] = statistics.median(row["full_moe_samples_us"])
            check(outputs[-1], reference)
            print(
                "Measured FC1",
                round_index,
                row["index"],
                row["fc1_median_us"],
                row["full_moe_median_us"],
                flush=True,
            )
            save()
    report["winner"] = min(report["candidates"], key=lambda r: r["full_moe_median_us"])[
        "index"
    ]
    report["fc1_winner"] = min(report["candidates"], key=lambda r: r["fc1_median_us"])[
        "index"
    ]
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="e384")
    parser.add_argument("--fc", choices=("fc1", "fc2"), default="fc2")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_candidates, args.rounds, args.samples) <= 0:
        parser.error("candidate, round and sample counts must be positive")
    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("this benchmark requires Rubin SM107")
    torch.manual_seed(17)
    tokens, experts, hidden, intermediate, top_k = CASES[args.case]
    report = dict(
        case=args.case,
        fc=args.fc,
        shape=CASES[args.case],
        seed=17,
        gpu=str(torch.cuda.get_device_properties(0)),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        compute_capability=torch.cuda.get_device_capability(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
        torch=torch.__version__,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        timing="CUPTI requested with CUDA-event fallback; FC2-only intervals use external CUDA events",
        cache="same-buffer warm cache, no L2 flush, unlocked clocks",
        boundaries="Isolated: output zero + FC2/finalize on saved FC1 data. Full: supplied logits through finalized output; excludes gate GEMM, prep, host and JIT.",
        candidates=[],
        max_candidates=args.max_candidates,
        exhaustive=args.exhaustive,
        rounds=args.rounds,
        samples=args.samples,
    )
    if args.fc == "fc1":
        report["timing"] = (
            "CUPTI requested with CUDA-event fallback; FC1 intervals use external CUDA events inside full MoE graphs"
        )
        report["boundaries"] = (
            "Supplied logits through finalized output, with zeroing overlapping FC1. Excludes gate GEMM, preparation, host and JIT. FC2 unchanged."
        )
    source_root = Path(__file__).resolve().parents[1]
    report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["compile_options"] = os.environ.get(
        "FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2"
    )
    report["source_sha256"] = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source_root / "flashinfer/prims_ts/batched_gemm").glob("*.py")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    print("Preparing real MoE inputs", args.case, flush=True)
    w1 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
        / hidden**0.5
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16)
        / intermediate**0.5
    )
    weights = PrimsTsConfig.prepare_weights(
        w1,
        w2,
        quant=quant,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        nvfp4_weight_layout="BlockMajorK",
    )
    weights["output2_scale_scalar"] = torch.linspace(0.7, 1.3, experts, device="cuda")
    del w1, w2
    gc.collect()
    torch.cuda.empty_cache()
    x, sf = PrimsTsConfig.prepare_activations(
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16), quant=quant
    )
    logits = torch.full((tokens, experts), -10, device="cuda", dtype=torch.float32)
    ids = (
        torch.arange(tokens, device="cuda")[:, None] * top_k
        + torch.arange(top_k, device="cuda")
    ) % experts
    logits.scatter_(1, ids, 10)
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
        num_experts=experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Renormalize),
        weight_layout=int(WeightLayout.BlockMajorK),
        enable_pdl=False,
    )
    saved = {}

    def record(key, label, io, stream):
        fn = get_compiled_gemm(key, label, io, stream)
        if label == "nvfp4_" + args.fc:
            saved.update(io=io, fn=fn)
        return fn

    def choose(name, runners, config, inputs, **kw):
        return runners[0], [512, 1]

    def run():
        return prims_ts_fp4_block_scale_moe(**kwargs, use_fused_finalize=True)[0]

    with (
        patch.object(AutoTuner.get(), "choose_one", choose),
        patch.object(runner, "get_compiled_gemm", record),
    ):
        output = run()
    torch.cuda.synchronize()
    reference = output.float().clone()
    base_io = saved["io"]
    base = base_io["cfg"]
    if args.fc == "fc1":
        benchmark_fc1(args, report, run, choose, reference, base, save)
        return
    cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    graphs = []
    configs = [base] + list(
        iter_epilogue_configs(
            base,
            exhaustive=args.exhaustive,
            max_candidates=args.max_candidates,
        )
    )
    for index, cfg in enumerate(configs):
        row = dict(
            index=index,
            legacy=index == 0,
            warps=cfg.dense_epilogue_warps,
            c_stages=cfg.num_stages_c_smem,
            steps=[asdict(s) for s in cfg.epilogue_steps],
        )
        report["candidates"].append(row)
        print("Compiling", index, row["warps"], row["c_stages"], flush=True)
        start = time.monotonic()
        io = dict(base_io, cfg=cfg)
        # A failed compilation is recorded and excluded. CUDA execution or
        # correctness failures terminate the run rather than contaminate timing.
        try:
            fn = get_compiled_gemm(
                stable_config_hash(cfg), "nvfp4_fc2", io, cuda_stream
            )
        except Exception as exc:
            row.update(status="compile_failed", error=str(exc))
            save()
            continue
        row["compile_seconds"] = time.monotonic() - start
        launch_args = gemm._launch_arg_tuple(io, cuda_stream)
        output.zero_()
        fn(*launch_args)
        torch.cuda.synchronize()
        row["correctness"] = check(output, reference)
        events = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capture_args = gemm._launch_arg_tuple(
                io, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            )
            for _ in range(10):
                output.zero_()
                a = torch.cuda.Event(enable_timing=True, external=True)
                b = torch.cuda.Event(enable_timing=True, external=True)
                a.record()
                fn(*capture_args)
                b.record()
                events.append((a, b))
        graph.replay()
        torch.cuda.synchronize()
        check(output, reference)
        row.update(
            status="passed",
            fc2_samples_us=[],
            zero_fc2_samples_us=[],
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
        graphs.append((row, graph, events, fn, io))
        save()
    for round_index in range(args.rounds):
        order = graphs if round_index % 2 == 0 else reversed(graphs)
        for row, graph, events, _, _ in order:
            row["zero_fc2_samples_us"].extend(timed(graph, samples=args.samples))
            for _ in range(args.samples):
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                row["fc2_samples_us"].append(
                    sum(a.elapsed_time(b) for a, b in events) * 100
                )
            row["fc2_median_us"] = statistics.median(row["fc2_samples_us"])
            row["zero_fc2_median_us"] = statistics.median(row["zero_fc2_samples_us"])
            print(
                "Measured", round_index, row["index"], row["fc2_median_us"], flush=True
            )
            save()
    best = min((r for r, *_ in graphs), key=lambda r: r["fc2_median_us"])
    report["fc2_winner"] = best["index"]
    # Graphs retain modules/buffers only within this bounded benchmark run.
    graphs.clear()
    gc.collect()
    full = {}
    with patch.object(AutoTuner.get(), "choose_one", choose):
        # Isolated FC2 ranking can change when FC1, routing and zeroing share
        # caches/streams with it. Choose the final winner from full-block
        # measurements of the best two generated candidates and the baseline.
        finalists = sorted(
            (
                row
                for row in report["candidates"]
                if row.get("status") == "passed" and not row["legacy"]
            ),
            key=lambda row: row["fc2_median_us"],
        )[:2]
        report["full_moe_indices"] = {
            "legacy": 0,
            **{f"candidate_{r['index']}": r["index"] for r in finalists},
        }
        for label, index in report["full_moe_indices"].items():
            cfg = configs[index]
            with patch.object(
                runner, "_fused_finalize_config", lambda old, selected=cfg: selected
            ):
                graph, outputs = capture(run)
            check(outputs[-1], reference)
            full[label] = (graph, outputs)
        report["full_moe"] = {name: [] for name in full}
        for round_index in range(args.rounds):
            order = tuple(full) if round_index % 2 == 0 else tuple(reversed(full))
            for name in order:
                report["full_moe"][name].extend(
                    timed(full[name][0], samples=args.samples)
                )
    report["full_moe_median_us"] = {
        name: statistics.median(values) for name, values in report["full_moe"].items()
    }
    winner = min(report["full_moe_median_us"], key=report["full_moe_median_us"].get)
    report["winner"] = report["full_moe_indices"][winner]
    save()
    print(
        "Winner",
        report["winner"],
        "FC2 winner",
        best["index"],
        "Full MoE",
        report["full_moe_median_us"],
        flush=True,
    )


if __name__ == "__main__":
    main()
