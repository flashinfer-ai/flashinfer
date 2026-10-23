# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired warm CPU enqueue and CUDA Graph GPU timing for Frost MoE.

Run from a candidate checkout with --baseline-root pointing to a baseline
checkout. The baseline Frost package is loaded under a separate module name;
shared API/layer files must match byte for byte. Both sides run the same fixed
tactic, tensor contents and weight layout. First-use compilation and selection
are excluded. The public API includes validation and per-call workspace packing.

CPU timing stops when Python returns, with device synchronization outside the
timer. GPU timing uses external CUDA event nodes around one or more calls and
reports time per call; it does not include CPU submission. Both timing graphs
use the same workspace addresses. This is a kernel/host microbenchmark, not a
model throughput measurement. Use --reverse in an independent process, and
--baseline-root pointing to the candidate checkout for an A/A noise control.
"""

import argparse
from functools import partial
import hashlib
import importlib
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import torch
import flashinfer
from flashinfer import fp4_quantize, mxfp8_quantize
from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from bench_frost_low_latency_bf16 import ACTIVATIONS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(root):
    current = Path(flashinfer.__file__).resolve().parent
    base = root.resolve() / "flashinfer"
    for relative in ("fused_moe/api.py", "fused_moe/layer.py", "fused_moe/runners.py"):
        if digest(base / relative) != digest(current / relative):
            raise ValueError(f"Shared public/runner wrapper differs: {relative}")
    path = base / "fused_moe/backends/frost_low_latency"
    name = "flashinfer.fused_moe.backends.frost_host_baseline"
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    runner = importlib.import_module(name + ".runner").FrostLowLatencyRunner
    manifests = {}
    for label, folder in (
        ("baseline", path),
        ("candidate", current / path.relative_to(base)),
    ):
        manifests[label] = {
            str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*.py"))
        }
    # Some device helpers use absolute package imports. This alias technique
    # is valid for host-only edits, not comparisons of different GPU kernels.
    host_files = {
        "runtime.py",
        "pipeline.py",
        "fma/runtime.py",
        "fma/bf16/runtime.py",
        "fma/nvfp4/runtime.py",
    }
    for relative in manifests["baseline"].keys() | manifests["candidate"].keys():
        if relative not in host_files and manifests["baseline"].get(
            relative
        ) != manifests["candidate"].get(relative):
            raise ValueError(
                f"Host-only comparison requires matching device helpers: {relative}"
            )
    return runner, manifests


def make_inputs(args, tokens, fmt, activation):
    h, i, e, k = args.hidden, args.intermediate, args.experts, args.top_k
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") / h**0.5
    ).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    quant = QuantConfig(
        weight=fmt,
        activation=fmt
        if fmt in (QuantFormat.BF16, QuantFormat.NVFP4)
        else QuantFormat.MXFP8,
        **({"nvfp4_4over6": None} if fmt == QuantFormat.NVFP4 else {}),
    )
    if fmt == QuantFormat.BF16:
        view = FrostLowLatencyConfig.prepare_bf16_weights(w1, w2, activation=activation)
        q, sf = x, None
    elif fmt == QuantFormat.NVFP4:
        view = FrostLowLatencyConfig.prepare_nvfp4_weights(
            w1, w2, activation=activation
        )
        q, sf = TrtllmFp4Config.prepare_activations(x, quant=quant)
    else:

        def weight(w):
            rows, cols = w.shape[-2:]
            if fmt == QuantFormat.MXFP4:
                packed, scales = fp4_quantize(
                    w.reshape(-1, cols),
                    torch.ones(1, device=w.device),
                    sf_vec_size=32,
                    sf_use_ue8m0=True,
                    is_sf_swizzled_layout=False,
                )
            else:
                packed, scales = mxfp8_quantize(
                    w.reshape(-1, cols), is_sf_swizzled_layout=False
                )
            return packed.reshape(e, rows, -1), scales.view(torch.uint8).reshape(
                e, rows, cols // 32
            )

        a, sa = weight(w1)
        b, sb = weight(w2)
        view = FrostLowLatencyConfig.prepare_weights(
            a, sa, b, sb, activation=activation
        )
        q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=False)
        sf = sf.view(torch.uint8).reshape(-1, h // 32)[:tokens].contiguous()
    ids = (
        torch.arange(tokens * k, device="cuda", dtype=torch.int32).reshape(tokens, k)
        % e
    )
    scores = torch.softmax(torch.randn(tokens, k, device="cuda"), -1)
    weights = MoEWeightPack({"frost_low_latency": view})
    cfg = MoEConfig(
        RoutingConfig(e, k),
        quant,
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        execution=ExecutionConfig(tune_max_num_tokens=tokens),
    )
    return cfg, MoEActivationPack(q, sf, ids, scores), weights


class FixedTacticLayer(MoELayer):
    """Keep normal public dispatch; fix selection outside the timed warm path."""

    def _select_winner(self, act_pack, weight_pack, runners):
        assert len(runners) == 1
        return runners[0], self.fixed_tactic


def statistics(values):
    values = sorted(values)
    return {
        "median_us": median(values),
        "p10_us": values[len(values) // 10],
        "p90_us": values[len(values) * 9 // 10],
        "samples_us": values,
    }


def host_times(calls, repeats, reverse):
    samples = {key: [] for key in calls}
    # Measure each operation in its own AB/BA loop. Interleaving different
    # operations can leave only one side cold after an unrelated public call.
    for metric in ("runner", "public", "pack", "graph_replay"):
        names = [f"{label}/{metric}" for label in ("baseline", "candidate")]
        for key in names:
            for _ in range(20):
                calls[key]()
        for repeat in range(repeats):
            for key in names[::-1] if bool(repeat % 2) != reverse else names:
                torch.cuda.synchronize()
                start = perf_counter_ns()
                output = calls[key]()
                elapsed = perf_counter_ns() - start
                samples[key].append(elapsed / 1000)
                del output
    torch.cuda.synchronize()
    return {key: statistics(values) for key, values in samples.items()}


def capture(fn, calls_per_graph, *, timed=True):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True) if timed else None
    end = torch.cuda.Event(enable_timing=True, external=True) if timed else None
    with torch.cuda.graph(graph):
        if timed:
            start.record()
        for _ in range(calls_per_graph):
            output = fn()
        if timed:
            end.record()
    return graph, start, end, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("bf16", "mxfp4", "mxfp8", "nvfp4"),
        default=["bf16", "mxfp4", "mxfp8", "nvfp4"],
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--activations", nargs="+", choices=tuple(ACTIVATIONS), default=["swiglu"]
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1408)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--calls-per-graph", type=int, default=1)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.calls_per_graph < 1:
        parser.error("--repeats and --calls-per-graph must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(8147)
    baseline, manifests = load_baseline(args.baseline_root)
    result = {
        "benchmark_sha256": digest(Path(__file__)),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "dsl": version("nvidia-cutlass-dsl"),
        "tvm_ffi": version("apache-tvm-ffi"),
        "python": sys.version.split()[0],
        "cuda_runtime": torch.version.cuda,
        "graph_cpu_includes_timing_events": False,
        "gpu_shared_workspace": True,
        "manifests": manifests,
        "rows": [],
    }
    formats = {
        "bf16": QuantFormat.BF16,
        "mxfp4": QuantFormat.MXFP4,
        "mxfp8": QuantFormat.MXFP8,
        "nvfp4": QuantFormat.NVFP4,
    }
    for activation_name in args.activations:
        for format_name in args.formats:
            for tokens in args.tokens:
                fmt = formats[format_name]
                cfg, act, weights = make_inputs(
                    args, tokens, fmt, ACTIVATIONS[activation_name]()
                )
                tactics = (
                    [("ffma", 1, 4)]
                    if fmt == QuantFormat.BF16
                    else (
                        [("nvfp4_ffma", 2), ("nvfp4_ffma", 4)]
                        if fmt == QuantFormat.NVFP4
                        else [128, 256, ("ffma", 1)]
                    )
                )
                for tactic in tactics:
                    print(
                        "Warm", format_name, activation_name, tokens, tactic, flush=True
                    )
                    calls, graphs, cpu_graphs = {}, {}, {}
                    gpu_inputs = None
                    for label, cls in (
                        ("baseline", baseline),
                        ("candidate", FrostLowLatencyRunner),
                    ):
                        runner = cls(cfg, act.hidden_states_q.device)
                        runner.check_support()
                        runner.build()
                        packed = runner.pack_inputs(act, weights)
                        assert tactic in runner.get_valid_tactics(packed, None)
                        fn = partial(runner.forward, packed, tactic=tactic)
                        layer = FixedTacticLayer(cfg, act.hidden_states_q.device)
                        layer.runners = [runner]
                        layer.fixed_tactic = tactic
                        public = partial(layer, act, weights)
                        public()
                        if gpu_inputs is None:
                            gpu_inputs = runner.pack_inputs(act, weights)
                        # Use identical addresses for GPU timing. Different
                        # workspace allocations can give sizable A/A deltas,
                        # even though neither implementation changed.
                        gpu_fn = partial(runner.forward, gpu_inputs, tactic=tactic)
                        graphs[label] = capture(gpu_fn, args.calls_per_graph)
                        # Event nodes themselves add replay submission work.
                        # Independent call storage also checks pointer binding.
                        cpu_graphs[label] = capture(
                            fn, args.calls_per_graph, timed=False
                        )
                        calls[label + "/runner"] = fn
                        calls[label + "/public"] = public
                        calls[label + "/pack"] = partial(
                            runner.pack_inputs, act, weights
                        )
                        calls[label + "/graph_replay"] = cpu_graphs[label][0].replay
                    # Changed content after capture detects stale inputs/route weights.
                    act.topk_ids.add_(1).remainder_(args.experts)
                    act.topk_weights.copy_(torch.rand_like(act.topk_weights))
                    for graph, _, _, _ in cpu_graphs.values():
                        graph.replay()
                    torch.testing.assert_close(
                        cpu_graphs["baseline"][3],
                        cpu_graphs["candidate"][3],
                        rtol=0,
                        atol=0,
                    )
                    for label in graphs:
                        torch.testing.assert_close(
                            cpu_graphs[label][3],
                            calls[label + "/public"](),
                            rtol=0,
                            atol=0,
                        )
                    host = host_times(calls, args.repeats, args.reverse)
                    gpu = {label: [] for label in graphs}
                    for repeat in range(args.repeats):
                        order = list(graphs)
                        if bool(repeat % 2) != args.reverse:
                            order.reverse()
                        for label in order:
                            graph, start, end, _ = graphs[label]
                            graph.replay()
                            end.synchronize()
                            gpu[label].append(
                                start.elapsed_time(end) * 1000 / args.calls_per_graph
                            )
                    # Timing graphs share storage, so snapshot each replay
                    # before the other implementation can overwrite its output.
                    snapshots = {}
                    for label, (graph, _, _, output) in graphs.items():
                        graph.replay()
                        snapshots[label] = output.clone()
                        torch.testing.assert_close(
                            snapshots[label], cpu_graphs[label][3], rtol=0, atol=0
                        )
                    torch.testing.assert_close(
                        snapshots["baseline"], snapshots["candidate"], rtol=0, atol=0
                    )
                    row = {
                        "format": format_name,
                        "activation": activation_name,
                        "tokens": tokens,
                        "tactic": tactic,
                        "host": host,
                        "gpu": {k: statistics(v) for k, v in gpu.items()},
                        "correctness": "bitwise baseline/candidate and public/replay after changed routing",
                    }
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        {k: round(v["median_us"], 3) for k, v in host.items()},
                        "gpu",
                        {k: round(median(v), 3) for k, v in gpu.items()},
                        flush=True,
                    )


if __name__ == "__main__":
    main()
