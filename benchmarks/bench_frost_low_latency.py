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
    CudnnGroupedGemmMxfp8Config,
    CuteDslConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    PrimsTsConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
)


def _decode_mx(payload, scales):
    if payload.dtype == torch.uint8:
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=payload.device,
            dtype=torch.float64,
        )
        values = torch.stack(
            (lut[(payload & 15).long()], lut[(payload >> 4).long()]), dim=-1
        ).reshape(*payload.shape[:-1], -1)
    else:
        values = payload.double()
    return (
        values.reshape(*values.shape[:-1], -1, 32)
        * torch.exp2(scales.view(torch.uint8).double() - 127)[..., None]
    ).reshape(values.shape)


def _mx_intermediate_reference(
    q, sf, w1, s1, w2, s2, ids, scores, activation, *, prims_scale=False
):
    """Model native MX SwiGLU numerical boundaries outside timed regions.

    CUTLASS/cuDNN write BF16 GEMM1 and activation before MXFP8 quantization.
    Prims keeps FP32 activation and uses an exponent-only MX scale instead.
    FP64 dots and ordered FP32 combine provide a tolerance-based reference;
    they do not model the native reduction order bit for bit.
    """
    x = _decode_mx(q, sf)
    result = torch.zeros_like(x, dtype=torch.float32)
    for token in range(ids.shape[0]):
        for slot, expert in enumerate(ids[token].tolist()):
            gu = (x[token] @ _decode_mx(w1[expert], s1[expert]).T).float()
            if not prims_scale:
                gu = gu.bfloat16().float()
            up, gate = gu.chunk(2)
            gate = gate.clamp_max(activation.limit)
            up = up.clamp(-activation.limit, activation.limit)
            mid = (
                gate
                / (1 + torch.exp(-gate * activation.alpha))
                * (up + activation.beta)
            )
            if not prims_scale:
                mid = mid.bfloat16().float()
            mid = mid.reshape(-1, 32)
            amax = mid.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
            # Prims gmem_c_resources truncates amax to a power of two, then
            # divides by 256; it can saturate values that the /448 rule retains.
            exponent = (
                torch.floor(torch.log2(amax)) - 8
                if prims_scale
                else torch.ceil(torch.log2(amax / 448))
            ).clamp(-127, 127)
            scale = torch.exp2(exponent)[..., None]
            dq = (
                (mid / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            ).flatten()
            down = (
                (dq.double() @ _decode_mx(w2[expert], s2[expert]).T).bfloat16().float()
            )
            result[token] = result[token] + down * scores[token, slot].float()
    return result.bfloat16().double()


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
    parser.add_argument("--include-cutlass", action="store_true")
    parser.add_argument("--include-prims-ts", action="store_true")
    parser.add_argument(
        "--include-cudnn-grouped",
        action="store_true",
        help="Include cuDNN grouped GEMM (MXFP8 weights, default SwiGLU only)",
    )
    parser.add_argument("--include-native-default", action="store_true")
    parser.add_argument(
        "--backend-references",
        action="store_true",
        help="Use CUTLASS/cuDNN BF16-intermediate and Prims MX-scale references; record disagreement with TRT separately",
    )
    parser.add_argument(
        "--include-frost-tactics",
        action="store_true",
        help="Also measure both TC tiles and FFMA separately through MoELayer",
    )
    parser.add_argument(
        "--trtllm-tactic", help="Pin a previously measured tactic tuple"
    )
    parser.add_argument("--cute-tactic", help="Pin a previously measured tactic tuple")
    parser.add_argument("--cutlass-tactic", help="Pin a measured CUTLASS tactic")
    parser.add_argument("--prims-ts-tactic", help="Pin a measured Prims-TS tactic")
    parser.add_argument(
        "--cudnn-grouped-tactic", help="Pin a measured cuDNN grouped-GEMM tactic"
    )
    frost = parser.add_mutually_exclusive_group()
    frost.add_argument("--frost-tile-k", type=int, choices=(128, 256))
    frost.add_argument(
        "--frost-ffma", action="store_true", help="Pin the scalar FMA tactic"
    )
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    requested_tactics = {
        "trtllm": a.trtllm_tactic,
        "cute_dsl": a.cute_tactic,
        "cutlass": a.cutlass_tactic,
        "prims_ts": a.prims_ts_tactic,
        "cudnn_grouped": a.cudnn_grouped_tactic,
        "frost": ("ffma", 1) if a.frost_ffma else a.frost_tile_k,
    }
    if a.cache_mode == "cold" and a.timing_mode == "amortized":
        raise ValueError("Cold-cache measurement requires timing-mode latency")
    if a.include_cutedsl and a.weight_format != "mxfp4":
        raise ValueError("--include-cutedsl requires --weight-format mxfp4")
    if a.include_cudnn_grouped and (
        a.weight_format != "mxfp8" or a.activation_limit != 0
    ):
        raise ValueError(
            "--include-cudnn-grouped requires --weight-format mxfp8 "
            "and --activation-limit 0; clamped SwiGLU is not the same operation."
        )
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
    if a.include_cutlass:
        cutlass_config = (
            CutlassMxfp8Mxfp4Config
            if weight_format == QuantFormat.MXFP4
            else CutlassMxfp8Config
        )
        backends.append(("cutlass", cutlass_config()))
    if a.include_prims_ts:
        backends.append(("prims_ts", PrimsTsConfig()))
    if a.include_cudnn_grouped:
        backends.append(("cudnn_grouped", CudnnGroupedGemmMxfp8Config()))
    weight_keys = {
        "trtllm": (
            "trtllm_fp4_routed"
            if weight_format == QuantFormat.MXFP4
            else "trtllm_fp8_block"
        ),
        "cute_dsl": "cute_dsl",
        "cutlass": (
            "cutlass_mxfp8_mxfp4"
            if weight_format == QuantFormat.MXFP4
            else "cutlass_mxfp8"
        ),
        "prims_ts": "prims_ts",
        "cudnn_grouped": "cudnn_grouped_gemm_mxfp8",
    }
    for name, backend in backends:
        # CUTLASS and cuDNN config classes each describe one quant pair; their
        # preparation methods therefore do not accept a separate quant config.
        quant_kwargs = (
            {"quant": quant} if name in ("trtllm", "cute_dsl", "prims_ts") else {}
        )
        weights.prepare_for(
            weight_keys[name],
            backend.prepare_weights(
                w1,
                w2,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size=i,
                activation=activation,
                **quant_kwargs,
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
        # A frozen-tactic validation must not recompile/tune the entire space.
        # The initial default call still establishes the normal runner binding;
        # the requested tactic is applied before capture below.
        initial_autotune = a.retune_cache or requested_tactics.get(name) is None
        # Prims advertises its full tactic list without a preliminary search.
        # The cache-specific sweep below already ranks every candidate once.
        if a.retune_cache and name == "prims_ts":
            initial_autotune = False
        with autotune(initial_autotune):
            layer(act, weights)
        layers[name] = layer

    def backend_references():
        references = {}
        if a.backend_references:
            for name in ("cutlass", "cudnn_grouped", "prims_ts"):
                if name not in layers:
                    continue
                # This benchmark constructs a PackedPrecomputed activation
                # pack. These adapters narrow routing scores to BF16.
                scores = rw.bfloat16()
                references[name] = _mx_intermediate_reference(
                    q,
                    sf,
                    q1,
                    s1,
                    q2,
                    s2,
                    ids,
                    scores,
                    activation,
                    prims_scale=name == "prims_ts",
                )
        return references

    references = backend_references()
    cache_tuning = {}
    if a.retune_cache:
        ref = layers["trtllm"](act, weights).double()
        assert ref.norm() > 1e-12
        for name, layer in layers.items():
            assert len(layer._winners) == 1, "Expected one exact input geometry"
            key, (runner, selected) = next(iter(layer._winners.items()))
            # Factorized backends rank each GEMM inside get_valid_tactics.
            # Outside tuning mode they return only a cached/default pair.
            with autotune(True):
                valid_tactics = runner.get_valid_tactics(
                    runner.pack_inputs(act, weights), None
                )
            # Prims-TS tactics contain lists; preserve their original types.
            candidates = []
            for tactic in [selected, -1, *valid_tactics]:
                if tactic not in candidates:
                    candidates.append(tactic)
            check_ref = references.get(name, ref)
            ranked = []
            print(f"{name}: ranking {len(candidates)} tactics", flush=True)
            for tactic in candidates:
                layer._winners[key] = (runner, tactic)
                layer(act, weights)
                graph, intervals, output = capture_timing(layer)
                graph.replay()
                values = [sample(graph, intervals) for _ in range(5)]
                diff = output.double() - check_ref
                l2 = float(diff.norm() / check_ref.norm())
                mx = float(diff.abs().max() / check_ref.abs().max())
                assert torch.isfinite(output).all() and l2 < 0.01 and mx < 0.03, (
                    name,
                    tactic,
                    l2,
                    mx,
                )
                ranked.append((median(values), tactic))
                if len(ranked) % 64 == 0:
                    print(f"{name}: ranked {len(ranked)}/{len(candidates)}", flush=True)
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
        for name in tuple(layers):
            if name == "frost":
                continue
            default_layer = MoELayer(layers[name].config)
            default_layer(act, weights)
            for key, (runner, _) in list(default_layer._winners.items()):
                default_layer._winners[key] = (runner, -1)
            default_layer(act, weights)
            layers[name + "_default"] = default_layer
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
    for name, requested in requested_tactics.items():
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
    checks, reference_norms, cross_reference_checks = [], [], []
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
        references = backend_references()
        cross_reference_checks.append(
            {
                name: dict(
                    relative_l2=float(
                        (outputs[name].double() - ref).norm() / ref.norm()
                    ),
                    normalized_max=float(
                        (outputs[name].double() - ref).abs().max() / ref.abs().max()
                    ),
                )
                for name in references
            }
        )
        metrics = {}
        for name, out in outputs.items():
            check_ref = references.get(name.removesuffix("_default"), ref)
            diff = out.double() - check_ref
            l2 = float(diff.norm() / check_ref.norm().clamp_min(1e-30))
            mx = float(diff.abs().max() / check_ref.abs().max().clamp_min(1e-30))
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
        check_ref = references.get(name.removesuffix("_default"), ref)
        diff = out.double() - check_ref
        l2 = float(diff.norm() / check_ref.norm().clamp_min(1e-30))
        mx = float(diff.abs().max() / check_ref.abs().max().clamp_min(1e-30))
        assert torch.isfinite(out).all() and l2 < 0.01 and mx < 0.03, (name, l2, mx)
        post_timing_checks[name] = dict(relative_l2=l2, normalized_max=mx)
    medians = {
        name: median([v for row in rows for v in row]) for name, rows in samples.items()
    }
    best_original = min(
        (name for name in medians if not name.startswith("frost")), key=medians.get
    )
    cudnn_backend_version = None
    if a.include_cudnn_grouped:
        import cudnn

        cudnn_backend_version = cudnn.backend_version()
    result = dict(
        args={**vars(a), "output": str(a.output)},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cutlass_dsl=version("nvidia-cutlass-dsl"),
        flashinfer=flashinfer.__version__,
        flashinfer_base_commit=flashinfer.__git_commit__,
        cuda=torch.version.cuda,
        cudnn_backend_version=cudnn_backend_version,
        scope=f"{a.cache_mode} CUDA-Graph public routed-MoE {a.timing_mode} GPU time. Includes adapters and combine; excludes cache eviction, input quantization, routing selection, shared experts, communication, attention and serving.",
        cache_eviction_bytes=flush.numel() if flush is not None else 0,
        l2_bytes=l2_bytes,
        distinct_experts=unique_experts,
        expert_reuse=t * k / unique_experts,
        cache_tuning=cache_tuning,
        numerical_references={
            name: (
                "FP32 activation; exponent-only MX scale with saturating E4M3; BF16 expert output; ordered FP32 combine reference"
                if name.removesuffix("_default") == "prims_ts"
                and name.removesuffix("_default") in references
                else "BF16 GEMM1 and activation before MXFP8; BF16 expert output; ordered FP32 combine reference"
                if name.removesuffix("_default") in references
                else "TRT-LLM output"
            )
            for name in layers
        },
        comparison_scope=(
            "Different intermediate rounding: absolute kernel latency comparison, not interchangeable numerical contracts."
            if references
            else "Cross-backend outputs checked against TRT-LLM with unchanged tolerances."
        ),
        cross_reference_checks=cross_reference_checks,
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
