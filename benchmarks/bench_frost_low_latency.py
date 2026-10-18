# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired CUDA-Graph GPU time for the public routed-MoE contract.

Example: python benchmarks/bench_frost_low_latency.py --hidden 5120 \
  --intermediate 1152 --experts 384 --top-k 6 --tokens 1 --output result.json
Repeat in a fresh process with --reverse, then compare baseline A/A drift.
Synthetic tensors at model-relevant geometry; not a model throughput test.
"""

import argparse
import ast
import json
from importlib.metadata import version
from statistics import median
from pathlib import Path

import torch
import flashinfer
from flashinfer import fp4_quantize, mxfp8_quantize
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CuteDslConfig,
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
    SwiGLU,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1152)
    parser.add_argument("--logical-intermediate", type=int, default=None)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    parser.add_argument(
        "--timing-mode",
        choices=("latency", "amortized"),
        default="latency",
        help="Event-delimited individual calls, or warm batched replay time / 128",
    )
    parser.add_argument(
        "--retune-cache",
        action="store_true",
        help="Rank all runner-advertised tactics using this routing/cache condition",
    )
    parser.add_argument(
        "--route-pattern",
        choices=("disjoint", "shared"),
        default="disjoint",
        help="Distinct experts across tokens, or the same top-k experts per token",
    )
    parser.add_argument("--weight-format", choices=("mxfp4", "mxfp8"), default="mxfp4")
    parser.add_argument(
        "--activation-limit",
        type=float,
        default=10.0,
        help="SwiGLU clamp; pass 0 for ordinary unclamped SwiGLU",
    )
    parser.add_argument(
        "--routing-weights-dtype", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--include-cutedsl", action="store_true")
    parser.add_argument("--include-native-default", action="store_true")
    parser.add_argument(
        "--include-frost-tactics",
        action="store_true",
        help="Also measure both TC tiles and FFMA separately through MoELayer",
    )
    parser.add_argument(
        "--trtllm-tactic", help="Pin a previously measured tactic tuple"
    )
    parser.add_argument("--cute-tactic", help="Pin a previously measured tactic tuple")
    frost = parser.add_mutually_exclusive_group()
    frost.add_argument("--frost-tile-k", type=int, choices=(128, 256))
    frost.add_argument(
        "--frost-ffma", action="store_true", help="Pin the scalar FMA tactic"
    )
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.cache_mode == "cold" and a.timing_mode == "amortized":
        raise ValueError("Cold-cache measurement requires timing-mode latency")
    if (
        a.include_cutedsl
        and a.weight_format == "mxfp4"
        and a.routing_weights_dtype == "bf16"
    ):
        raise ValueError(
            "The native CuTe DSL W4A8 baseline requires FP32 routing weights. "
            "Omit --include-cutedsl for the BF16-routing control; this benchmark "
            "does not insert a routing dtype conversion for one backend."
        )
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(2107)
    t, h, i, e, k = a.tokens, a.hidden, a.intermediate, a.experts, a.top_k
    unique_experts = t * k if a.route_pattern == "disjoint" else k
    if unique_experts > e:
        raise ValueError("The requested routing pattern needs more experts than E")

    def route_ids(step):
        slots = torch.arange(unique_experts, device="cuda", dtype=torch.int32)
        values = (slots * max(1, e // unique_experts) + e - 1 + step * 7) % e
        return (
            values.reshape(t, k)
            if a.route_pattern == "disjoint"
            else values.expand(t, k)
        )

    if i % 128:
        raise ValueError("Original unified baselines require I divisible by 128")
    weight_format = (
        QuantFormat.MXFP4 if a.weight_format == "mxfp4" else QuantFormat.MXFP8
    )
    quant = QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8)
    native_config = (
        TrtllmFp4Config if weight_format == QuantFormat.MXFP4 else TrtllmFp8BlockConfig
    )
    activation = (
        SwiGLU()
        if a.activation_limit == 0
        else SwiGLU(alpha=1.0, beta=0.0, limit=a.activation_limit)
    )
    w1 = (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
    logical = a.logical_intermediate or i
    if not 0 < logical <= i or logical % 32:
        raise ValueError(
            "logical intermediate must fit physical I and be divisible by 32"
        )
    w1[:, logical:i].zero_()
    w1[:, i + logical : 2 * i].zero_()
    w2[:, :, logical:].zero_()
    one = torch.ones(1, device="cuda")

    def pack(w):
        rows, cols = w.shape[-2:]
        if weight_format == QuantFormat.MXFP8:
            q, s = mxfp8_quantize(w.reshape(-1, cols), is_sf_swizzled_layout=False)
            return q.reshape(e, rows, cols), s.reshape(e, rows, cols // 32).view(
                torch.uint8
            )
        q, s = fp4_quantize(
            w.reshape(-1, cols),
            global_scale=one,
            sf_vec_size=32,
            sf_use_ue8m0=True,
            is_sf_swizzled_layout=False,
        )
        return q.reshape(e, rows, cols // 2).view(torch.uint8), s.reshape(
            e, rows, cols // 32
        ).view(torch.uint8)

    q1, s1 = pack(w1)
    q2, s2 = pack(w2)
    weights = MoEWeightPack()
    weights.prepare_for(
        "frost_low_latency", FrostLowLatencyConfig.prepare_weights(q1, s1, q2, s2)
    )
    backends = [("trtllm", native_config())]
    if a.include_cutedsl and weight_format == QuantFormat.MXFP4:
        backends.append(("cute_dsl", CuteDslConfig()))
    for name, backend in backends:
        key = (
            (
                "trtllm_fp4_routed"
                if weight_format == QuantFormat.MXFP4
                else "trtllm_fp8_block"
            )
            if name == "trtllm"
            else name
        )
        weights.prepare_for(
            key,
            backend.prepare_weights(
                w1,
                w2,
                quant=quant,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size=i,
                activation=activation,
            ),
        )
    backends.append(("frost", FrostLowLatencyConfig(max_num_tokens=8)))
    x = torch.randn(t, h, device="cuda").bfloat16()
    q, sf = native_config.prepare_activations(x, quant=quant)
    ids = route_ids(0).contiguous()
    rw = torch.softmax(torch.randn(t, k, device="cuda"), -1)
    if a.routing_weights_dtype == "bf16":
        rw = rw.bfloat16()
    act = MoEActivationPack(q, sf, ids, rw)
    l2_bytes = 0
    flush = None
    if a.cache_mode == "cold":
        from cuda.bindings import driver

        err, l2_bytes = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0
        )
        if err != driver.CUresult.CUDA_SUCCESS or l2_bytes <= 0:
            raise RuntimeError("Cannot determine L2 capacity for the cold-cache test")
        flush = torch.zeros(
            max(512 * 1024**2, 2 * l2_bytes), device="cuda", dtype=torch.uint8
        )

    def capture_timing(layer):
        graph = torch.cuda.CUDAGraph()
        intervals = (
            [
                (
                    torch.cuda.Event(enable_timing=True, external=True),
                    torch.cuda.Event(enable_timing=True, external=True),
                )
                for _ in range(16)
            ]
            if a.timing_mode == "latency"
            else []
        )
        with torch.cuda.graph(graph):
            if not intervals:
                for _ in range(128):
                    output = layer(act, weights)
            else:
                # External event nodes exclude both eviction and host enqueue.
                for start, end in intervals:
                    if flush is not None:
                        flush.add_(1)
                    start.record()
                    output = layer(act, weights)
                    end.record()
        return graph, intervals, output

    def sample(graph, intervals):
        if intervals:
            graph.replay()
            intervals[-1][1].synchronize()
            return median(start.elapsed_time(end) * 1000 for start, end in intervals)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / 128

    layers = {}
    for name, backend in backends:
        cfg = MoEConfig(
            RoutingConfig(e, k),
            quant,
            ExpertConfig(logical if name == "frost" else i),
            activation=activation,
            backend=BackendOptions((backend,)),
            execution=ExecutionConfig(tune_max_num_tokens=t),
        )
        layer = MoELayer(cfg)
        with autotune(True):
            layer(act, weights)
        layers[name] = layer
    cache_tuning = {}
    if a.retune_cache:
        ref = layers["trtllm"](act, weights).double()
        assert ref.norm() > 1e-12
        for name, layer in layers.items():
            assert len(layer._winners) == 1, "Expected one exact input geometry"
            key, (runner, selected) = next(iter(layer._winners.items()))
            candidates = list(
                dict.fromkeys(
                    [
                        selected,
                        -1,
                        *runner.get_valid_tactics(
                            runner.pack_inputs(act, weights), None
                        ),
                    ]
                )
            )
            ranked = []
            for tactic in candidates:
                layer._winners[key] = (runner, tactic)
                layer(act, weights)
                graph, intervals, output = capture_timing(layer)
                graph.replay()
                values = [sample(graph, intervals) for _ in range(5)]
                diff = output.double() - ref
                l2 = float(diff.norm() / ref.norm())
                mx = float(diff.abs().max() / ref.abs().max())
                assert torch.isfinite(output).all() and l2 < 0.01 and mx < 0.03, (
                    name,
                    tactic,
                    l2,
                    mx,
                )
                ranked.append((median(values), tactic))
            winner = min(ranked, key=lambda item: item[0])[1]
            layer._winners[key] = (runner, winner)
            cache_tuning[name] = [
                {"tactic": repr(tactic), "median_us": us} for us, tactic in ranked
            ]
            print(
                f"{name}: ranked {len(ranked)} tactics in {a.cache_mode} cache; selected {winner}",
                flush=True,
            )
    if a.include_native_default:
        # A fresh layer may still hit the process-wide autotune cache. Pin -1
        # explicitly to test the native heuristic through the same public call.
        default_layer = MoELayer(layers["trtllm"].config)
        default_layer(act, weights)
        for key, (runner, _) in list(default_layer._winners.items()):
            default_layer._winners[key] = (runner, -1)
        default_layer(act, weights)
        layers["trtllm_default"] = default_layer
    if a.include_frost_tactics:
        for name, tactic in (
            ("frost_tc128", 128),
            ("frost_tc256", 256),
            ("frost_ffma", ("ffma", 1)),
        ):
            layer = MoELayer(layers["frost"].config)
            layer(act, weights)
            for key, (runner, _) in list(layer._winners.items()):
                packed = runner.pack_inputs(act, weights)
                if tactic not in runner.get_valid_tactics(packed, None):
                    raise ValueError(f"{name} does not support this geometry")
                layer._winners[key] = (runner, tactic)
            layer(act, weights)
            layers[name] = layer
    for name, requested in (
        ("trtllm", a.trtllm_tactic),
        ("cute_dsl", a.cute_tactic),
        ("frost", ("ffma", 1) if a.frost_ffma else a.frost_tile_k),
    ):
        if requested is None:
            continue
        if name not in layers:
            raise ValueError(f"{name} tactic requested without that backend")
        tactic = ast.literal_eval(str(requested))
        layer = layers[name]
        for key, (runner, _) in list(layer._winners.items()):
            layer._winners[key] = (runner, tactic)
        layer(act, weights)
        assert all(winner[1] == tactic for winner in layer._winners.values())
    # Capture the whole public call, including any per-call scale/weight packing
    # and output reduction. Compilation and one-time weight prep are excluded.
    graphs, outputs = {}, {}
    for name, layer in layers.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = layer(act, weights)
        graphs[name] = graph
    assert len({out.data_ptr() for out in outputs.values()}) == len(outputs)
    checks, reference_norms = [], []
    previous_reference = None
    for step in range(6):
        x = (torch.randn(t, h, device="cuda") * (0.01, 1.0, 3.0)[step % 3]).bfloat16()
        qn, sn = native_config.prepare_activations(x, quant=quant)
        q.copy_(qn)
        sf.copy_(sn)
        ids.copy_(route_ids(step))
        rw.copy_(torch.softmax(torch.randn_like(rw), -1))
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        ref = outputs["trtllm"].double()
        norm = float(ref.norm())
        assert norm > 1e-12, "Degenerate all-zero baseline is not a numerical check"
        if previous_reference is not None:
            assert not torch.equal(ref, previous_reference), (
                "Replay did not change output"
            )
        previous_reference = ref.clone()
        reference_norms.append(norm)
        metrics = {}
        for name, out in outputs.items():
            diff = out.double() - ref
            l2 = float(diff.norm() / ref.norm().clamp_min(1e-30))
            mx = float(diff.abs().max() / ref.abs().max().clamp_min(1e-30))
            assert torch.isfinite(out).all() and l2 < 0.01 and mx < 0.03, (
                name,
                step,
                l2,
                mx,
            )
            metrics[name] = dict(relative_l2=l2, normalized_max=mx)
        checks.append(metrics)
    names = list(graphs)
    if a.reverse:
        names.reverse()
    timing_graphs, timing_events, timing_outputs = {}, {}, {}
    for name, layer in layers.items():
        graph, intervals, output = capture_timing(layer)
        timing_graphs[name] = graph
        timing_events[name] = intervals
        timing_outputs[name] = output
    samples = {name: [] for name in names}
    controls = []
    for _sweep in range(2):
        for name in ["trtllm", *names, "trtllm"]:
            graph = timing_graphs[name]
            for _ in range(4):
                graph.replay()
            row = []
            for _ in range(9):
                row.append(sample(graph, timing_events[name]))
            if name == "trtllm":
                controls.append(median(row))
            samples[name].append(row)
    post_timing_checks = {}
    for name, out in timing_outputs.items():
        diff = out.double() - ref
        l2 = float(diff.norm() / ref.norm().clamp_min(1e-30))
        mx = float(diff.abs().max() / ref.abs().max().clamp_min(1e-30))
        assert torch.isfinite(out).all() and l2 < 0.01 and mx < 0.03, (name, l2, mx)
        post_timing_checks[name] = dict(relative_l2=l2, normalized_max=mx)
    medians = {
        name: median([v for row in rows for v in row]) for name, rows in samples.items()
    }
    best_original = min(
        (name for name in medians if not name.startswith("frost")), key=medians.get
    )
    result = dict(
        args={**vars(a), "output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cutlass_dsl=version("nvidia-cutlass-dsl"),
        flashinfer=flashinfer.__version__,
        flashinfer_base_commit=flashinfer.__git_commit__,
        cuda=torch.version.cuda,
        scope=f"{a.cache_mode} CUDA-Graph public routed-MoE {a.timing_mode} GPU time. Includes adapters and combine; excludes cache eviction, input quantization, routing selection, shared experts, communication, attention and serving.",
        cache_eviction_bytes=flush.numel() if flush is not None else 0,
        l2_bytes=l2_bytes,
        distinct_experts=unique_experts,
        expert_reuse=t * k / unique_experts,
        cache_tuning=cache_tuning,
        quant=f"{weight_format.name} weights / MXFP8 activations / BF16 output",
        checks=checks,
        post_timing_checks=post_timing_checks,
        reference_norms=reference_norms,
        baseline_controls_us=controls,
        baseline_aa=(max(controls) - min(controls)) / min(controls),
        median_us=medians,
        samples_us=samples,
        best_original_backend=best_original,
        speedup=medians[best_original] / medians["frost"],
        speedup_vs_trtllm=medians["trtllm"] / medians["frost"],
        winner_backend={n: l.winner_backend for n, l in layers.items()},
        selected_tactics={
            n: [
                {"bucket": repr(key), "tactic": repr(winner[1])}
                for key, winner in l._winners.items()
            ]
            for n, l in layers.items()
        },
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "median_us",
                    "baseline_aa",
                    "best_original_backend",
                    "speedup",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
