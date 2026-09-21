# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Benchmark frozen Frost MXFP8 stages and complete routed MoELayer pools.

Use ``benchmark`` for full MoELayer comparisons: the complete applicable original
backend pool versus the same pool plus automatic Frost candidates, plus Frost's
independently autotuned four-plan result. The original backend tactic pools are
preserved. For example::

    python benchmarks/bench_cudnn_frost_moe_mxfp8.py benchmark \
        --activation all --experts 8 --hidden 4096 --intermediate 14336 \
        --top-k 2 --output results.jsonl

The default mode (also available as ``sweep``) is a stage sweep over already-grouped
synthetic inputs. FC1 and FC2 use independent inputs. Routing, scale packing,
reference calculation and compilation happen before timing. CUDA graph timing
includes the prepared launcher's workspace reset and uses repeated, hot inputs.

Example (select an idle GPU via CUDA_VISIBLE_DEVICES):
    python benchmarks/bench_cudnn_frost_moe_mxfp8.py \
        --activation all --stage both --tokens 128,1024 \
        --experts 8 --hidden 1024 --intermediate 512 --top-k 2 \
        --output /tmp/frost_mxfp8.jsonl

All matching artifacts are checked and timed. Results rank candidates separately
for each stage, activation and shape; timings must not be summed into a full MoE
latency or compared directly with a full routed backend. The cuDNN Frost compiler
is not required.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
    runtime,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.activations import (
    ACTIVATIONS,
    is_gated,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import (
    runtime as mxfp8,
)

if __package__:
    from .bench_cudnn_frost_common import (
        activation_reference,
        capture,
        check_idle_gpu,
        emit,
        error,
        measure,
    )
else:
    from bench_cudnn_frost_common import (
        activation_reference,
        capture,
        check_idle_gpu,
        emit,
        error,
        measure,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--artifacts", type=Path, default=runtime.artifact_root("mxfp8")
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="JSONL results (new file)"
    )
    parser.add_argument("--activation", choices=(*ACTIVATIONS, "all"), default="swiglu")
    parser.add_argument("--stage", choices=("fc1", "fc2", "both"), default="both")
    parser.add_argument(
        "--tokens",
        default="128,1024,4096",
        help="Comma-separated pre-routing token counts",
    )
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--routing", choices=("uniform", "skew"), default="uniform")
    parser.add_argument(
        "--kernel-id",
        action="append",
        help="Restrict the sweep to these artifact IDs (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph-batch", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        args.tokens = [int(value) for value in args.tokens.split(",")]
    except ValueError:
        parser.error("--tokens must be comma-separated integers")
    if (
        min(
            *args.tokens,
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.rounds,
            args.iterations,
            args.graph_batch,
        )
        <= 0
    ):
        parser.error("shapes, top-k, and timing counts must be positive")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("--hidden and --intermediate must be divisible by 128")
    if args.top_k > args.experts:
        parser.error("--top-k must not exceed --experts")
    if (
        max(args.tokens) * args.top_k >= 2**31
        or max(args.hidden, args.intermediate, args.experts) >= 2**31
    ):
        parser.error("grouped tensor dimensions must fit int32")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def routing_offsets(tokens, experts, top_k, routing):
    """Generate the same ragged expert counts for all candidates of a shape."""
    logits = torch.rand(tokens, experts, device="cuda")
    if routing == "skew":
        logits[: tokens // 2, 0] += 2
    ids = logits.topk(top_k, dim=-1).indices
    counts = torch.bincount(ids.flatten(), minlength=experts)
    offsets = (counts.cumsum(0) - counts).to(torch.int32)
    return offsets, counts.tolist()


def random_operand(shape):
    data = (torch.randn(shape, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    scales = torch.randint(
        125, 129, (*shape[:-1], shape[-1] // 32), dtype=torch.uint8, device="cuda"
    )
    return data, scales


def dequantize(data, scales):
    return data.double() * scales.view(torch.float8_e8m0fnu).double().repeat_interleave(
        32, -1
    )


def stage_reference(x, weights, x_scales, weight_scales, counts, stage, activation):
    """FP64 GEMMs followed by the fused FP32 activation and BF16 output."""
    n = weights[0].shape[1]
    output = torch.empty(x.shape[0], n, device=x.device, dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            values = dequantize(x[start:end], x_scales[start:end])
            products = [
                values @ dequantize(w[expert], sf[expert]).T
                for w, sf in zip(weights, weight_scales, strict=True)
            ]
            result = products[0].float()
            if stage == "fc1":
                if is_gated(activation):
                    # Runtime weights are [gate, up]; reference expects [up, gate].
                    result = torch.cat((products[1].float(), result), dim=-1)
                result = activation_reference(result, activation)
            output[start:end] = result
        start = end
    return output


def matching_kernels(kernels, stage, activation, rows, n, k, experts, arch):
    geometry = dict(s=rows, n=n, k=k, experts=experts, groups=experts)
    return tuple(
        kernel
        for kernel in kernels
        if kernel.arch == arch
        and kernel.fc1 == (stage == "fc1")
        and (stage == "fc2" or kernel.activation == activation)
        and all(
            runtime._dimension_matches(value, kernel.contract.get(key))
            for key, value in geometry.items()
        )
    )


def benchmark_stage(
    args, file, kernels, offsets, counts, tokens, stage, activation, arch, gpu_uuid
):
    rows = tokens * args.top_k
    n, k = (
        (args.intermediate, args.hidden)
        if stage == "fc1"
        else (args.hidden, args.intermediate)
    )
    candidates = matching_kernels(
        kernels, stage, activation, rows, n, k, args.experts, arch
    )
    if not candidates:
        raise ValueError(
            f"no matching MXFP8 artifacts for {stage}/{activation}: S={rows}, N={n}, K={k}, E={args.experts}"
        )
    x, x_sf = random_operand((rows, k))
    operands = [
        random_operand((args.experts, n, k))
        for _ in range(2 if stage == "fc1" and is_gated(activation) else 1)
    ]
    weights, weight_sf = zip(*operands, strict=True)
    reference = stage_reference(x, weights, x_sf, weight_sf, counts, stage, activation)
    packed_x_sf = mxfp8.pack_token_scales(x_sf, offsets)
    packed_w_sf = tuple(mxfp8.pack_weight_scales(sf) for sf in weight_sf)
    out = torch.empty_like(reference)
    workspace = torch.empty(
        max(kernel.workspace_bytes for kernel in candidates),
        dtype=torch.uint8,
        device="cuda",
    )
    # Plans/graphs share buffers and execute sequentially on one stream.
    plans, graphs, errors = [], [], []
    for kernel in candidates:
        check_idle_gpu(gpu_uuid)
        plan = mxfp8.PreparedMxfp8GroupedGemm(
            kernel,
            x,
            weights,
            offsets,
            packed_x_sf,
            packed_w_sf,
            out,
            workspace=workspace,
        )
        out.fill_(float("nan"))
        plan()
        relative_l2 = error(out, reference)
        graph = capture(plan, args.graph_batch)
        out.fill_(float("nan"))
        graph.replay()
        relative_l2 = max(relative_l2, error(out, reference))
        plans.append(plan)  # Keep bound tensors and launchables alive through replay.
        graphs.append(graph)
        errors.append(relative_l2)
    samples = [[] for _ in candidates]
    order = list(range(len(candidates)))
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        rng.shuffle(order)
        for index in order:
            check_idle_gpu(gpu_uuid)
            samples[index].append(
                measure(graphs[index], args.iterations, args.graph_batch)
            )
    check_idle_gpu(gpu_uuid)
    context = dict(
        scope="grouped_stage",
        tokens=tokens,
        grouped_rows=rows,
        experts=args.experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        top_k=args.top_k,
        routing=args.routing,
        expert_rows=counts,
        stage=stage,
        activation=activation,
    )
    results = []
    for kernel, timings, relative_l2 in zip(candidates, samples, errors, strict=True):
        result = dict(
            kind="candidate",
            **context,
            artifact_id=kernel.artifact_id,
            source_sha256=kernel.source_sha256,
            tactic=kernel.tactic_metadata,
            workspace_bytes=kernel.workspace_bytes,
            milliseconds=statistics.median(timings),
            samples=timings,
            relative_l2=relative_l2,
        )
        emit(file, result)
        results.append(result)
    ranked = sorted(results, key=lambda record: record["milliseconds"])
    emit(
        file,
        dict(
            kind="best",
            **context,
            artifact_id=ranked[0]["artifact_id"],
            milliseconds=ranked[0]["milliseconds"],
            candidate_count=len(candidates),
            ranked=[
                dict(
                    artifact_id=record["artifact_id"],
                    milliseconds=record["milliseconds"],
                )
                for record in ranked
            ],
        ),
    )


def stage_main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "a working CUDA GPU is required; select it with CUDA_VISIBLE_DEVICES"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    arch = runtime._arch_for(device)
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    check_idle_gpu(gpu_uuid)
    kernels = mxfp8.discover(args.artifacts)
    if args.kernel_id:
        unknown = set(args.kernel_id) - {kernel.artifact_id for kernel in kernels}
        if unknown:
            raise ValueError(f"unknown kernel IDs: {sorted(unknown)}")
        kernels = tuple(
            kernel for kernel in kernels if kernel.artifact_id in args.kernel_id
        )
    activations = tuple(ACTIVATIONS) if args.activation == "all" else (args.activation,)
    stages = (
        [("fc1", activation) for activation in activations]
        if args.stage != "fc2"
        else []
    )
    if args.stage != "fc1":
        stages.append(
            ("fc2", "identity")
        )  # One shared FC2, independent of FC1 activation.
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        emit(
            file,
            dict(
                kind="metadata",
                schema_version=1,
                scope="grouped_stage",
                gpu=torch.cuda.get_device_name(device),
                gpu_uuid=gpu_uuid,
                arch=arch,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                cutlass_dsl=version("nvidia-cutlass-dsl"),
                artifacts=str(args.artifacts.resolve()),
                seed=args.seed,
                rounds=args.rounds,
                iterations=args.iterations,
                graph_batch=args.graph_batch,
                timing_mode="cuda_graph_hot_replay",
                includes=["workspace_reset", "grouped_gemm"]
                + (["fc1_activation"] if args.stage != "fc2" else []),
                excludes=[
                    "routing",
                    "scale_packing",
                    "intermediate_requantization",
                    "finalization",
                    "compilation",
                ],
                stage_inputs="independent_synthetic",
                kernel_ids=args.kernel_id,
            ),
        )
        for tokens in args.tokens:
            offsets, counts = routing_offsets(
                tokens, args.experts, args.top_k, args.routing
            )
            for stage, activation in stages:
                benchmark_stage(
                    args,
                    file,
                    kernels,
                    offsets,
                    counts,
                    tokens,
                    stage,
                    activation,
                    arch,
                    gpu_uuid,
                )
        emit(
            file,
            dict(
                kind="complete",
                shapes=len(args.tokens),
                stage_cases=len(args.tokens) * len(stages),
            ),
        )


_FROST_BACKEND = "cudnn_frost_mxfp8"


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    try:
        return list(value)
    except TypeError:
        return str(value)


def emit_e2e(record, file=None):
    line = json.dumps(record, default=serializable)
    print(line, flush=True)
    if file is not None:
        file.write(line + "\n")
        file.flush()


def gpu_telemetry():
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    fields = "index,uuid,name,clocks.current.sm,power.limit,power.max_limit,temperature.gpu,utilization.gpu"
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=" + fields,
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [v.strip() for v in text.split(",")]
    result = dict(zip(fields.split(","), values, strict=True))
    result["sample_time"] = time.time()
    return result


def validate_clocks(args):
    # Query after sustained graph replay; an idle clock is not validation.
    telemetry = gpu_telemetry()
    if (
        args.verify_locked_clocks
        and abs(float(telemetry["clocks.current.sm"]) - args.clock_mhz) > 25
    ):
        raise RuntimeError(
            f"SM clock not locked near {args.clock_mhz} MHz: {telemetry}"
        )
    if args.verify_locked_clocks and float(telemetry["power.limit"]) != float(
        telemetry["power.max_limit"]
    ):
        raise RuntimeError(f"Power limit differs from hardware maximum: {telemetry}")
    return telemetry


def reference(act, view, activation, rows):
    """Independent FP32 dequantized math, without intermediate rounding noise.

    Evaluate the requested prefix; round FC1 activation and FC2 result to BF16.
    Omission of the intermediate E4M3 requantization deliberately leaves a
    quantization-noise reference, independent of either native MoE pipeline.
    """
    rows = min(rows, act.hidden_states_q.shape[0])
    x = act.hidden_states_q[:rows].float().reshape(rows, -1, 32)
    x = (
        x
        * act.hidden_states_scale[:rows]
        .contiguous()
        .view(torch.float8_e8m0fnu)
        .float()[..., None]
    ).flatten(1)
    ids, scores = act.topk_ids[:rows], act.topk_weights[:rows]
    result = torch.zeros(rows, ids.shape[1], x.shape[1], device=x.device)
    for expert in torch.unique(ids).tolist():
        token, slot = torch.where(ids == expert)
        weights = []
        for stage in (1, 2):
            q = view[f"fc{stage}_expert_weights"][expert]
            n, k = q.shape
            sf = view[f"fc{stage}_expert_scales"][expert].view(torch.uint8)
            sf = sf.reshape(n // 128, k // 128, 32, 4, 4)
            sf = sf.permute(0, 3, 2, 1, 4).reshape(n, k // 32)
            weights.append(
                (
                    q.float().reshape(n, k // 32, 32)
                    * sf.contiguous().view(torch.float8_e8m0fnu).float()[..., None]
                ).reshape(n, k)
            )
        middle = (
            activation_reference(x[token] @ weights[0].T, activation).bfloat16().float()
        )
        result[token, slot] = (middle @ weights[1].T).bfloat16().float() * scores[
            token, slot, None
        ]
    return result.sum(1).bfloat16()


def relative_l2(actual, expected):
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("Nonfinite MoE output")
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-20)
    ).item()


def discover_original(arch, quant, activation):
    from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

    eligible, audit = {}, []
    for cfg_cls, runner_cls in _BACKEND_RUNNERS.items():
        supported_acts = (
            runner_cls.supported_activation_classes_by_quant.get(quant.pair, ())
            if runner_cls.supported_activation_classes_by_quant
            else runner_cls.supported_activation_classes
        )
        checks = dict(
            arch=cfg_cls.supported(arch),
            quant=runner_cls.supports_quant(quant),
            activation=isinstance(activation, supported_acts),
        )
        audit.append(
            dict(
                backend=runner_cls.backend_key,
                config=cfg_cls.__name__,
                support_checks=checks,
                activation_classes=[cls.__name__ for cls in supported_acts],
            )
        )
        if all(checks.values()):
            eligible[cfg_cls] = runner_cls
    return eligible, audit


def prepare_problem(args, activation_name, geometry):
    from flashinfer.fused_moe import (
        BackendOptions,
        CutlassMxfp8Config,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        TrtllmFp8BlockConfig,
    )

    e, h, i, top_k = geometry
    activation = ACTIVATIONS[activation_name]()
    quant = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    major, minor = torch.cuda.get_device_capability()
    eligible, audit = discover_original(major * 10 + minor, quant, activation)
    unsupported = set(eligible) - {CutlassMxfp8Config, TrtllmFp8BlockConfig}
    if unsupported:
        raise RuntimeError(
            "Missing preparation for applicable backends: "
            + ", ".join(sorted(cls.__name__ for cls in unsupported))
        )
    if not eligible:
        raise RuntimeError("No original MXFP8 backend applies")
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=top_k),
        quant=quant,
        experts=ExpertConfig(intermediate_size=i),
        backend=BackendOptions(tuple(cls() for cls in eligible)),
        activation=activation,
        execution=ExecutionConfig(tune_max_num_tokens=max(args.tokens)),
    )
    torch.manual_seed(args.seed)
    w1 = (
        torch.randn(
            e,
            i * (2 if activation.is_gated else 1),
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * args.weight_std
    )
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * args.weight_std
    kwargs = dict(
        num_local_experts=e, hidden_size=h, intermediate_size=i, activation=activation
    )
    # The Frost runner shares this canonical CUTLASS weight view, byte for byte.
    views = {"cutlass_mxfp8": CutlassMxfp8Config.prepare_weights(w1, w2, **kwargs)}
    if TrtllmFp8BlockConfig in eligible:
        views["trtllm_fp8_block"] = TrtllmFp8BlockConfig.prepare_weights(
            w1, w2, quant=quant, **kwargs
        )
    del w1, w2
    return (
        config,
        MoEWeightPack(views),
        audit,
        {cls.backend_key for cls in eligible.values()},
    )


def benchmark_case(args, config, weights, activation, geometry, routing, tokens, file):
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import CutlassMxfp8Config, MoEActivationPack, MoELayer

    e, h, i, top_k = geometry
    uuid = torch.cuda.get_device_properties(0).uuid
    check_idle_gpu(uuid)
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * args.input_std
    xq, xsf = CutlassMxfp8Config.prepare_activations(x, quant=config.quant)
    logits = torch.rand(tokens, e, device="cuda")
    if routing == "skew":
        logits[: tokens // 2, 0] += 2
    ids = logits.topk(top_k, dim=1).indices.int()
    scores = torch.rand(tokens, top_k, device="cuda").softmax(-1)
    act = MoEActivationPack(xq, xsf, ids, scores)
    layers = {"original": MoELayer(config)}
    layers["original"]._additional_candidates = lambda *unused: []
    if not args.baseline_only:
        layers["with_cudnn_frost"] = MoELayer(config)
    expected = set(
        discover_original(
            torch.cuda.get_device_capability()[0] * 10
            + torch.cuda.get_device_capability()[1],
            config.quant,
            config.activation,
        )[0].values()
    )
    expected_keys = {runner.backend_key for runner in expected}
    outputs, graphs, winners, tactic_counts, packed_keepalive = {}, {}, {}, {}, []
    native_tactic_counts = {}
    for label, layer in layers.items():
        actual = {runner.backend_key for runner in layer.runners}
        if actual != expected_keys:
            raise RuntimeError(f"Original pool mismatch: {actual} != {expected_keys}")
        with autotune():
            layer(act, weights)
        outputs[label] = layer(act, weights).clone()
        winners[label] = [
            (runner.backend_key, tactic) for runner, tactic in layer._winners.values()
        ]
        tactic_counts[label] = {}
        native_tactic_counts[label] = {}
        for runner in layer.runners:
            packed = runner.pack_inputs(act, weights)
            packed_keepalive.append(packed)
            tactic_counts[label][runner.backend_key] = len(
                runner.get_valid_tactics(packed, None)
            )
            if runner.backend_key == "cutlass_mxfp8":
                native = runner._inner.fused_moe_runner
                native_tactic_counts[label][runner.backend_key] = dict(
                    fc1=native.get_gemm1_tactic_count(),
                    fc2=native.get_gemm2_tactic_count(),
                    retained_per_stage=runner._num_top_tactics_per_stage,
                )
        graphs[label] = capture(
            lambda layer=layer: layer(act, weights), args.graph_batch, warmup=5
        )
    candidate_count = None
    frost_tactic = None
    plan_checks = []
    if not args.baseline_only:
        layer = layers["with_cudnn_frost"]
        runner = layer._automatic_runners.get(_FROST_BACKEND)
        if runner is None or not runner.accepts(act, weights):
            raise RuntimeError(
                f"Normal automatic admission failed: {activation} {geometry} {tokens}"
            )
        packed = runner.pack_inputs(act, weights)
        packed_keepalive.append(packed)
        frost_tactics = runner.get_valid_tactics(packed, None)
        candidate_count = len(frost_tactics)
        if candidate_count != 4:
            raise RuntimeError(
                f"Expected top-2 x top-2 = 4 plans, got {candidate_count}"
            )
        for tactic in frost_tactics:
            out = runner.forward(packed, tactic)
            eager_error = relative_l2(out, outputs["original"])
            graph = capture(lambda tactic=tactic: runner.forward(packed, tactic), 1)
            out.fill_(float("nan"))
            graph.replay()
            graph_error = relative_l2(out, outputs["original"])
            if max(eager_error, graph_error) >= args.cross_l2:
                raise RuntimeError(
                    f"Frost plan failed validation: {tactic}: "
                    f"eager={eager_error}, graph={graph_error}"
                )
            plan_checks.append(
                dict(
                    tactic=tactic,
                    eager_relative_l2=eager_error,
                    graph_relative_l2=graph_error,
                )
            )
            del graph
        with autotune():
            _, frost_tactic = layer.tuner.choose_one(
                custom_op=f"moe_{_FROST_BACKEND}",
                runners=[runner],
                inputs=packed,
                tuning_config=runner.tuning_config_for(packed),
                **runner.launch_kwargs_for(packed),
            )
        outputs["cudnn_frost_only"] = runner.forward(packed, frost_tactic).clone()
        graphs["cudnn_frost_only"] = capture(
            lambda: runner.forward(packed, frost_tactic), args.graph_batch, warmup=5
        )
        winners["cudnn_frost_only"] = [(_FROST_BACKEND, frost_tactic)]
    errors = {
        label: relative_l2(out, outputs["original"]) for label, out in outputs.items()
    }
    if any(err >= args.cross_l2 for err in errors.values()):
        raise RuntimeError(
            f"Cross-backend relative L2 exceeds {args.cross_l2}: {errors}"
        )
    reference_errors = {}
    if args.reference_tokens:
        ref = reference(
            act, weights.get_view("cutlass_mxfp8"), activation, args.reference_tokens
        )
        reference_errors = {
            label: relative_l2(out[: len(ref)], ref) for label, out in outputs.items()
        }
        if any(err >= args.reference_l2 for err in reference_errors.values()):
            raise RuntimeError(
                f"Reference relative L2 exceeds {args.reference_l2}: {reference_errors}"
            )
    samples = {label: [] for label in graphs}
    telemetry = []
    for round_idx in range(args.rounds):
        check_idle_gpu(uuid)
        labels = list(graphs)
        offset = round_idx % len(labels)
        labels = labels[offset:] + labels[:offset]
        if round_idx % 2:
            labels.reverse()
        for label in labels:
            samples[label].append(
                measure(graphs[label], args.iterations, args.graph_batch)
            )
        telemetry.append(validate_clocks(args))
    check_idle_gpu(uuid)
    medians = {label: statistics.median(values) for label, values in samples.items()}
    record = dict(
        kind="e2e",
        activation=activation,
        geometry=geometry,
        tokens=tokens,
        routing=routing,
        routed_rows=tokens * top_k,
        baseline_backends=sorted(expected_keys),
        actual_backend_pool={
            label: sorted(
                expected_keys
                | ({_FROST_BACKEND} if label == "with_cudnn_frost" else set())
            )
            for label in layers
        },
        winners=winners,
        tactic_counts=tactic_counts,
        native_tactic_counts=native_tactic_counts,
        frost_candidate_count=candidate_count,
        frost_plan_checks=plan_checks,
        milliseconds=medians,
        relative_l2=errors,
        reference_relative_l2=reference_errors,
        reference_tokens=min(tokens, args.reference_tokens),
        samples_ms=samples,
        telemetry=telemetry,
    )
    if not args.baseline_only:
        record.update(
            speedup=medians["original"] / medians["with_cudnn_frost"],
            frost_speedup=medians["original"] / medians["cudnn_frost_only"],
        )
    emit_e2e(record, file)
    del graphs, layers, packed_keepalive
    gc.collect()
    torch.cuda.empty_cache()


def e2e_environment(args):
    return dict(
        kind="environment",
        scope="routed_moe_e2e",
        args=vars(args),
        gpu=gpu_telemetry(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        arch=f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        includes=[
            "routing_permutation",
            "input_scale_repacking",
            "fc1_activation",
            "intermediate_requantization",
            "fc2",
            "finalization",
        ],
        excludes=[
            "input_quantization",
            "weight_preparation",
            "compilation",
            "autotuning",
        ],
        timing_mode="alternating_cuda_graph_hot_replay",
    )


def parse_e2e_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark full routed MXFP8 MoELayer pools"
    )
    parser.add_argument("--activation", choices=(*ACTIVATIONS, "all"), default="swiglu")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens", default="1,16,128,512,2048,4096,8192,12288")
    parser.add_argument(
        "--routing", choices=("uniform", "skew", "both"), default="uniform"
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--input-std", type=float, default=0.02)
    parser.add_argument("--cross-l2", type=float, default=0.08)
    parser.add_argument("--reference-l2", type=float, default=0.12)
    parser.add_argument("--reference-tokens", type=int, default=8)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Selected artifact root, including moe_shortlists.json (default: packaged)",
    )
    parser.add_argument("--source-jit", action="store_true")
    parser.add_argument("--verify-locked-clocks", action="store_true")
    parser.add_argument("--clock-mhz", type=int, default=1100)
    parser.add_argument("--output", type=Path, help="Optional JSONL results (new file)")
    args = parser.parse_args(argv)
    try:
        args.tokens = [int(value) for value in args.tokens.split(",")]
    except ValueError:
        parser.error("--tokens must be comma-separated integers")
    if (
        not args.tokens
        or min(
            *args.tokens,
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.rounds,
            args.iterations,
            args.graph_batch,
        )
        < 1
    ):
        parser.error("shapes, top-k, and timing counts must be positive")
    if args.hidden % 128 or args.intermediate % 128 or args.top_k > args.experts:
        parser.error(
            "hidden/intermediate must be multiples of 128 and top-k <= experts"
        )
    if args.reference_tokens < 0 or min(args.cross_l2, args.reference_l2) <= 0:
        parser.error("reference-tokens must be nonnegative and L2 bounds positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def benchmark(args):
    if args.source_jit:
        from flashinfer.jit import env

        root = Path(__file__).resolve().parents[1]
        env.FLASHINFER_CSRC_DIR = root / "csrc"
        env.FLASHINFER_INCLUDE_DIR = root / "include"
        cccl = root / "3rdparty/cccl"
        env.CCCL_INCLUDE_DIRS = [
            cccl / "cub",
            cccl / "libcudacxx/include",
            cccl / "thrust",
        ]
    torch.backends.cuda.matmul.allow_tf32 = False
    output = None
    original_artifact_roots = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("x")
    try:
        if args.artifacts is not None:
            from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import (
                moe,
            )

            # Keep normal MoELayer admission and tuning, using this campaign's pool.
            original_artifact_roots = moe._artifact_roots
            moe._artifact_roots = lambda: (args.artifacts.resolve(),)
        emit_e2e(e2e_environment(args), output)
        activations = (
            list(ACTIVATIONS) if args.activation == "all" else [args.activation]
        )
        geometry = (args.experts, args.hidden, args.intermediate, args.top_k)
        routings = ("uniform", "skew") if args.routing == "both" else [args.routing]
        for activation in activations:
            config, weights, audit, expected = prepare_problem(
                args, activation, geometry
            )
            emit_e2e(
                dict(
                    kind="backend_support",
                    activation=activation,
                    geometry=geometry,
                    baseline_backends=sorted(expected),
                    candidates=audit,
                ),
                output,
            )
            for routing in routings:
                for tokens in args.tokens:
                    benchmark_case(
                        args,
                        config,
                        weights,
                        activation,
                        geometry,
                        routing,
                        tokens,
                        output,
                    )
            del weights
            gc.collect()
            torch.cuda.empty_cache()
        emit_e2e(dict(kind="completed"), output)
    finally:
        if original_artifact_roots is not None:
            moe._artifact_roots = original_artifact_roots
        if output is not None:
            output.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "benchmark":
        return benchmark(parse_e2e_args(argv[1:]))
    if argv and argv[0] == "sweep":
        argv = argv[1:]
    return stage_main(argv)


if __name__ == "__main__":
    main()
