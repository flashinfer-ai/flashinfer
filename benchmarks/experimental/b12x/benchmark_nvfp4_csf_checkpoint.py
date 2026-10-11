#!/usr/bin/env python3
"""Compare NVFP4 CSF and expanded scales using a checkpoint's routed experts.

Each arm uses the checkpoint's packed weights and calibration, caller-owned
scratch, normal autotuning, and captured CUDA graphs. Activations and routing
are seeded synthetic inputs to this MoE component benchmark. Timings exclude
checkpoint loading and preparation; lower graph replay microseconds is better.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

import torch

from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder, make_nvfp4_csf_batch
from b12x._lib.quant.csf_blocks import (
    Nvfp4BlockPlane,
    Nvfp4CompactBlockPlane,
    upload_csf_blocks,
    upload_csf_block_payload,
)
from b12x.loader._csf_blocks import export_csf_blocks
from b12x._lib.quant.csf_blocks_tuning import plan_csf_block_expansion
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation._measurement import (
    _prepare_race,
    capture_benchmark_call,
    measure_race_steps,
)
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from checkpoint_loader import IndexedSafetensorLoader
from checkpoint_csf import prepare_reference_weights


def benchmark_cache_dir(root, execution):
    if root is None:
        from b12x._lib.compiler import _cute_compile_cache_dir

        root = _cute_compile_cache_dir() / "preparation"
    return root / f"csf-{execution}"


def check_reference(metrics, *, min_cosine=0.9999):
    """Apply the SiLU NVFP4 oracle bounds used by benchmark_moe."""
    assert metrics.cos >= min_cosine, metrics
    assert metrics.max_abs <= 0.05 and metrics.rmse <= 0.0075, metrics
    assert metrics.mean_abs <= 0.005, metrics


def checkpoint_weights(args, device):
    from checkpoint_csf import (
        CsfMatrix,
        TensorView,
        prepare_nvfp4_csf_weights,
    )

    config = json.loads((args.checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    args.model_type = config.get("model_type")
    args.swiglu_limit = config.get("swiglu_limit") or None
    e = config.get("num_experts", config.get("n_routed_experts"))
    h, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    if intermediate % args.tp or not 0 <= args.rank < args.tp:
        raise ValueError("The checkpoint must have a uniform TP partition")
    n = intermediate // args.tp
    scratch = tuple(
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        for shape in (
            (e, (2 * n + 127) // 128 * 128, h // 16),
            (e, h, (n // 16 + 3) // 4 * 4),
        )
    )
    reader = IndexedSafetensorLoader(args.checkpoint)

    def experts():
        for expert in range(e):
            projections = []
            for role in ("up", "gate", "down"):
                prefix = args.expert_prefix.format(layer=args.layer)
                prefix = f"{prefix}.{expert}.{role}_proj"
                projections.append(
                    CsfMatrix(
                        weight=TensorView(reader.get_tensor(f"{prefix}.weight")),
                        fixed=reader.get_tensor(
                            f"{prefix}.weight_scale.nvfp4_csf_fixed"
                        ),
                        exceptions=reader.get_tensor(
                            f"{prefix}.weight_scale.nvfp4_csf_exceptions"
                        ),
                        global_scale=reader.get_tensor(f"{prefix}.weight_scale_2"),
                        input_scale=reader.get_tensor(f"{prefix}.input_scale"),
                    )
                )
            yield tuple(projections)

    weights = prepare_nvfp4_csf_weights(
        experts(),
        num_experts=e,
        hidden_size=h,
        intermediate_size=intermediate,
        tp_rank=args.rank,
        tp_size=args.tp,
        device=device,
        w13_scale_scratch=scratch[0],
        w2_scale_scratch=scratch[1],
    )
    if args.layer_max_input_scale:
        weights = replace(
            weights,
            packed=replace(
                weights.packed,
                input_scale=weights.packed.input_scale.amin(),
                intermediate_scale=weights.packed.intermediate_scale.amin(),
            ),
        )
    return weights, e, h, n, config["num_experts_per_tok"]


def block_checkpoint_weights(args, device, experts, hidden, intermediate, buffers):
    from vllm.model_executor.model_loader.csf_utils import CsfTensorReader
    from vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils import (
        prepare_nvfp4_csf_weights,
    )

    root = args.block_checkpoint
    names = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with CsfTensorReader(root, names, "nvfp4") as reader:

        def matrices():
            for expert in range(experts):
                prefix = args.expert_prefix.format(layer=args.layer) + f".{expert}"
                yield tuple(
                    reader.matrix(
                        f"{prefix}.{role}_proj.weight",
                        f"{prefix}.{role}_proj.weight_scale",
                        global_scale=f"{prefix}.{role}_proj.weight_scale_2",
                        input_scale=f"{prefix}.{role}_proj.input_scale",
                    )
                    for role in ("up", "gate", "down")
                )

        weights = prepare_nvfp4_csf_weights(
            matrices(),
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate * args.tp,
            tp_rank=args.rank,
            tp_size=args.tp,
            device=device,
            w13_scale_scratch=buffers[0],
            w2_scale_scratch=buffers[1],
        )
    if args.layer_max_input_scale:
        weights = replace(
            weights,
            packed=replace(
                weights.packed,
                input_scale=weights.packed.input_scale.amin(),
                intermediate_scale=weights.packed.intermediate_scale.amin(),
            ),
        )
    return weights


def storage_bytes(decoder):
    tensors = [
        tensor
        for plane in (decoder.first, decoder.second)
        for tensor in (plane.fixed, plane.exceptions, plane.task_offsets)
    ]
    if decoder.indexed_scales is not None:
        tensors.extend(plane.storage for plane in decoder.indexed_scales)
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
        for tensor in tensors
    }
    return sum(storage.nbytes() for storage in storages.values())


def race_graphs(graphs, device, args, produce):
    warm_until = time.monotonic() + args.warmup_seconds
    while time.monotonic() < warm_until:
        for graph in graphs.values():
            produce()
            graph.replay()
        torch.cuda.synchronize()
    calls = tuple(
        PreparedCall(run=g.replay, produce=produce, owners=(g,))
        for g in graphs.values()
    )
    race = _prepare_race(
        calls, device_ordinal=device.index, samples=args.samples, primed=True
    )
    rounds, seen = [], 0
    try:
        for _ in measure_race_steps(
            race, device_ordinal=device.index, rounds=args.rounds
        ):
            if race.completed_rounds != seen:
                seen = race.completed_rounds
                rounds.append(tuple(race.latest_round_us))
        if race.completed_rounds != seen:
            rounds.append(tuple(race.latest_round_us))
    finally:
        race.close()
    return {name: [row[i] for row in rounds] for i, name in enumerate(graphs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--block-checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--expert-prefix", default="model.language_model.layers.{layer}.mlp.experts"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", default="1,8,32,128,512,4096")
    parser.add_argument("--layer-max-input-scale", action="store_true")
    parser.add_argument("--block-scales", action="store_true")
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--compact-records", action="store_true")
    parser.add_argument("--resident-blocks", action="store_true")
    parser.add_argument("--native-codes", action="store_true")
    parser.add_argument("--group-records", action="store_true")
    parser.add_argument("--sweep-expansion", action="store_true")
    parser.add_argument("--compact-groups", default="8,16,32")
    parser.add_argument("--compact-exceptions", default="shared,global")
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument(
        "--tuning-execution", choices=("direct", "graph"), default="direct"
    )
    parser.add_argument("--compile-workers", type=int, default=0)
    parser.add_argument("--profile-graphs", action="store_true")
    parser.add_argument("--trace-graphs", type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    args = parser.parse_args()
    args.tuning_cache_dir = benchmark_cache_dir(
        args.tuning_cache_dir, args.tuning_execution
    )
    if args.block_checkpoint:
        args.resident_blocks = True
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "block_checkpoint": str(args.block_checkpoint)
                if args.block_checkpoint
                else None,
                "layer": args.layer,
                "tp": args.tp,
                "rank": args.rank,
                "device": torch.cuda.get_device_name(device),
                "layer_max_input_scale": args.layer_max_input_scale,
                "tuning_cache_dir": str(args.tuning_cache_dir)
                if args.tuning_cache_dir
                else None,
                "compile_workers": args.compile_workers,
                "tuning_execution": args.tuning_execution,
                "native_codes": args.native_codes,
                "group_records": args.group_records,
                "warmup_seconds": args.warmup_seconds,
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    weights, e, h, n, topk = checkpoint_weights(args, device)
    block_planes = None
    compact_planes = None
    exported = None
    if args.block_scales or args.compact_records or args.resident_blocks:
        exported = tuple(
            tuple(
                export_csf_blocks(
                    fixed,
                    exceptions,
                    rows=rows,
                    columns=columns,
                    quantization="nvfp4",
                )
                for fixed, exceptions in zip(plane.fixed, plane.exceptions, strict=True)
            )
            for plane, rows, columns in (
                (weights.w13_scales, 2 * n, h // 16),
                (weights.w2_scales, h, n // 16),
            )
        )
        if args.block_scales:
            block_planes = tuple(
                Nvfp4BlockPlane(upload_csf_blocks(p, device)) for p in exported
            )
        if args.compact_records:
            compact_planes = {
                label: tuple(
                    Nvfp4CompactBlockPlane(
                        upload_csf_block_payload(
                            p,
                            device,
                            native_codes=native_codes,
                            group_record_bytes=record_bytes,
                        )
                    )
                    for p in exported
                )
                for label, native_codes, record_bytes in (
                    ("compact", False, 0),
                    ("compact-native", True, 0),
                    ("compact-group3", True, 3),
                    ("compact-group4", True, 4),
                )
                if (not native_codes or args.native_codes)
                and (not record_bytes or args.group_records)
            }
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a4",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=args.swiglu_limit,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    planes = tuple(
        make_nvfp4_csf_batch(
            p.fixed,
            p.exceptions,
            rows=r,
            columns=c,
            device=device,
        )
        for p, r, c in (
            (weights.w13_scales, 2 * n, h // 16),
            (weights.w2_scales, h, n // 16),
        )
    )
    s13, s2 = weights.packed.w13_block_scales, weights.packed.w2_block_scales
    decoder = Nvfp4CsfDecoder.prepare(*planes, s13, s2)
    decoder.decode_all(s13, s2)
    native = replace(
        weights.packed,
        w13=weights.packed.w13.clone(),
        w2=weights.packed.w2.clone(),
        w13_block_scales=s13.clone(),
        w2_block_scales=s2.clone(),
    )
    arms = {
        "expanded": prepare_reference_weights(plan=plan, weights=native),
        "csf": prepare_reference_weights(
            plan=plan,
            weights=replace(weights, w13_scales=planes[0], w2_scales=planes[1]),
        ),
    }
    if args.resident_blocks:
        block_weights = (
            block_checkpoint_weights(args, device, e, h, n, (s13, s2))
            if args.block_checkpoint
            else replace(
                weights,
                w13_scales=moe.CsfBlockPlanes(exported[0]),
                w2_scales=moe.CsfBlockPlanes(exported[1]),
            )
        )
        for field in (
            "w13",
            "w2",
            "w13_global_scales",
            "w2_global_scales",
            "input_scale",
            "intermediate_scale",
        ):
            torch.testing.assert_close(
                getattr(block_weights.packed, field),
                getattr(weights.packed, field),
                rtol=0,
                atol=0,
            )
        arms["compact-resident"] = prepare_reference_weights(
            plan=plan,
            weights=block_weights,
        )
        block_owner = arms["compact-resident"]._impl
        block_owner.nvfp4_csf.decode_all(
            block_owner.w1_blockscale, block_owner.w2_blockscale
        )
        for actual, expected, rows, columns in (
            (block_owner.w1_blockscale, native.w13_block_scales, 2 * n, h // 16),
            (block_owner.w2_blockscale, native.w2_block_scales, h, n // 16),
        ):

            def logical(tensor):
                batch, r, c = tensor.shape
                return (
                    tensor.view(torch.uint8)
                    .reshape(batch, r // 128, c // 4, 32, 4, 4)
                    .permute(0, 1, 4, 3, 2, 5)
                    .reshape(batch, r, c)[:, :rows, :columns]
                )

            torch.testing.assert_close(
                logical(actual), logical(expected), rtol=0, atol=0
            )
        del block_weights, block_owner
    del exported
    if block_planes is not None:
        block_decoder = Nvfp4CsfDecoder.prepare(*block_planes, s13, s2)
        block_decoder.decode_all(s13, s2)
        torch.testing.assert_close(
            s13.view(torch.uint8),
            native.w13_block_scales.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            s2.view(torch.uint8),
            native.w2_block_scales.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        arms["blocks"] = prepare_reference_weights(
            plan=plan,
            weights=replace(
                weights, w13_scales=block_planes[0], w2_scales=block_planes[1]
            ),
        )
        del block_decoder, block_planes
    if compact_planes is not None:
        for label, pair in compact_planes.items():
            owner = prepare_reference_weights(
                plan=plan,
                weights=replace(weights, w13_scales=pair[0], w2_scales=pair[1]),
            )
            compact_plan = owner.plan
            sweep_expansion = args.sweep_expansion or args.decoder_only
            placements = (
                args.compact_exceptions.split(",") if sweep_expansion else ("shared",)
            )
            group_counts = (
                tuple(map(int, args.compact_groups.split(",")))
                if sweep_expansion
                else (8,)
            )
            for placement in placements:
                if placement not in ("shared", "global"):
                    raise ValueError(
                        "Compact exceptions require shared or global placement"
                    )
                for groups in group_counts:
                    variant = tuple(
                        replace(
                            p,
                            groups_per_task=groups,
                            shared_exceptions=placement == "shared",
                        )
                        for p in pair
                    )
                    compact_decoder = Nvfp4CsfDecoder.prepare(*variant, s13, s2)
                    compact_decoder.decode_all(s13, s2)
                    for result, expected in (
                        (s13, native.w13_block_scales),
                        (s2, native.w2_block_scales),
                    ):
                        torch.testing.assert_close(
                            result.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                    name = f"{label}-{placement}-{groups}" if sweep_expansion else label
                    arms[name] = replace(
                        owner,
                        plan=compact_plan,
                        _impl=replace(
                            owner._impl,
                            plan=compact_plan._impl,
                            nvfp4_csf=compact_decoder,
                        ),
                    )
        del compact_decoder, compact_planes
    del decoder, planes, native, weights
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "preparation_seconds": time.monotonic() - started,
                "experts": e,
                "hidden": h,
                "intermediate": n,
                "topk": topk,
                "compressed_scale_bytes": storage_bytes(arms["csf"]._impl.nvfp4_csf),
                "expansion_scratch_bytes": s13.numel() + s2.numel(),
                "block_scale_bytes": storage_bytes(arms["blocks"]._impl.nvfp4_csf)
                if "blocks" in arms
                else None,
                "compact_scale_bytes": {
                    name: storage_bytes(owner._impl.nvfp4_csf)
                    for name, owner in arms.items()
                    if name.startswith("compact")
                },
            }
        ),
        flush=True,
    )

    for tokens in map(int, args.tokens.split(",")):
        torch.manual_seed(907 + tokens)
        x = torch.randn(tokens, h, device=device, dtype=torch.bfloat16) * 0.5
        source_x = x.clone()
        ids = (
            torch.rand(tokens, e, device=device)
            .topk(topk, dim=1)
            .indices.to(torch.int32)
        )
        probabilities = torch.randn(tokens, topk, device=device).softmax(dim=-1)
        if args.decoder_only:
            graphs = {}
            active_ids = (
                ids.unique() if ids.numel() < 16 * e else torch.arange(e, device=device)
            )
            for name, owner in arms.items():
                decoder = owner._impl.nvfp4_csf
                if decoder is None:
                    continue
                s13.view(torch.uint8).fill_(0x7F)
                s2.view(torch.uint8).fill_(0x7F)
                decoder.decode(ids, s13, s2)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    decoder.decode(ids, s13, s2)
                graph.replay()
                native_owner = arms["expanded"]._impl
                for result, expected in (
                    (s13, native_owner.w1_blockscale),
                    (s2, native_owner.w2_blockscale),
                ):
                    torch.testing.assert_close(
                        result.view(torch.uint8)[active_ids],
                        expected.view(torch.uint8)[active_ids],
                        rtol=0,
                        atol=0,
                    )
                graphs[name] = graph
            samples = race_graphs(graphs, device, args, lambda: None)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "active_experts": active_ids.numel(),
                        "decoder_median_us": {
                            name: statistics.median(values)
                            for name, values in samples.items()
                        },
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )
            for graph in graphs.values():
                graph.reset()
            continue

        def prepare(state):
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch.scratch_specs()
            )
            output = torch.empty_like(x)
            binding = state.bind(
                a=x,
                topk_ids=ids,
                topk_weights=probabilities,
                output=output,
                scratch=scratch,
                input_scales_static=True,
            )
            return PreparedCall(
                run=lambda: state.run(binding),
                produce=lambda: x.copy_(source_x),
                output=output,
                owners=(scratch, binding, source_x),
            )

        def benchmark(state):
            call = prepare(state)
            return (
                capture_benchmark_call(call)
                if args.tuning_execution == "graph"
                else call
            )

        with PreparationSession(
            device=device,
            autotune=True,
            compile_workers=args.compile_workers,
            cache_dir=args.tuning_cache_dir,
        ) as session:
            for arm_name in (
                "compact",
                "compact-native",
                "compact-group3",
                "compact-group4",
                "compact-resident",
            ):
                if arm_name not in arms:
                    continue
                owner = arms[arm_name]
                decoder = owner._impl.nvfp4_csf
                expansion = plan_csf_block_expansion(
                    decoder.first,
                    decoder.second,
                    routes=ids.numel(),
                    ids_dtype=ids.dtype,
                )

                def prepare_expansion(state):
                    active = torch.empty(e, dtype=torch.int32, device=device)
                    binding = state.bind(ids=ids, out13=s13, out2=s2, scratch=active)
                    return PreparedCall(
                        run=binding.run, produce=lambda: None, owners=(active, binding)
                    )

                def benchmark_expansion(state):
                    call = prepare_expansion(state)
                    return (
                        capture_benchmark_call(call)
                        if args.tuning_execution == "graph"
                        else call
                    )

                session.prepare(
                    (
                        expansion.request(
                            name=f"scale-expansion-{arm_name}-{tokens}",
                            prepare_call=prepare_expansion,
                            benchmark_call=benchmark_expansion,
                        ),
                    )
                )
                active = torch.empty(e, dtype=torch.int32, device=device)
                decoder = expansion.prepared.state.decoder(active)
                arms[arm_name] = replace(
                    owner, _impl=replace(owner._impl, nvfp4_csf=decoder)
                )
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "arm": arm_name,
                            "expansion_selection": asdict(expansion.selection.config),
                            "selection_source": expansion.selection.source,
                        }
                    ),
                    flush=True,
                )
            plans = {
                name: moe.plan_execution(
                    experts=owner,
                    capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
                )
                for name, owner in arms.items()
            }
            print(f"Preparing tokens={tokens}", flush=True)
            session.prepare(
                tuple(
                    p.request(
                        name=f"checkpoint-{name}-{tokens}",
                        prepare_call=prepare,
                        benchmark_call=benchmark,
                    )
                    for name, p in plans.items()
                )
            )
            bindings, outputs, graphs, owners = {}, {}, {}, []
            for name, p in plans.items():
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in p.scratch_specs()
                )
                outputs[name] = torch.empty_like(x)
                bindings[name] = moe.bind(
                    p,
                    a=x,
                    topk_ids=ids,
                    topk_weights=probabilities,
                    output=outputs[name],
                    scratch=scratch,
                    input_scales_static=True,
                )
                owners.append(scratch)
            session.freeze()
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "configs": {
                            name: asdict(binding.execution_plan.decode_config)
                            for name, binding in bindings.items()
                        },
                    }
                ),
                flush=True,
            )
            for name, binding in bindings.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    moe.run(binding=binding)
                graphs[name] = graph
            previous, replay_variation = {}, {name: [] for name in graphs}
            reference_replays = {name: [] for name in graphs}
            for _ in range(3):
                s13.view(torch.uint8).fill_(0x7F)
                s2.view(torch.uint8).fill_(0x7F)
                for output in outputs.values():
                    output.fill_(float("nan"))
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocated
                )
                assert all(
                    torch.isfinite(o).all() and torch.count_nonzero(o)
                    for o in outputs.values()
                )
                for name, output in outputs.items():
                    if name in previous:
                        replay_variation[name].append(
                            asdict(compare_to_reference(output, previous[name]))
                        )
                    previous[name] = output.clone()
                    if tokens <= 8:
                        reference_replays[name].append(previous[name])
            metrics = compare_to_reference(outputs["csf"], outputs["expanded"])
            if tokens <= 8:
                reference_weights = arms["expanded"]._impl
                from b12x.moe.fused_moe._impl import _FAST_MATH_DEFAULT

                modes, reference_metrics, references, cosine_bounds = {}, {}, {}, {}
                for name, binding in bindings.items():
                    fast = (
                        _FAST_MATH_DEFAULT
                        if binding.fast_math is None
                        else binding.fast_math
                    )
                    mode = (
                        "micro"
                        if binding.implementation == "micro"
                        else "dynamic_fast"
                        if fast
                        else "dynamic_precise"
                    )
                    modes[name] = mode
                    cosine_bounds[name] = (
                        0.99989
                        if args.model_type == "glm5_next_text"
                        and mode == "dynamic_fast"
                        else 0.9999
                    )
                    if mode not in references:
                        references[mode] = moe_reference_nvfp4(
                            x,
                            reference_weights.w1_fp4,
                            reference_weights.w1_blockscale,
                            reference_weights.w1_alphas,
                            reference_weights.w2_fp4,
                            reference_weights.w2_blockscale,
                            reference_weights.w2_alphas,
                            reference_weights.a1_gscale,
                            reference_weights.a2_gscale,
                            ids,
                            probabilities,
                            e,
                            h,
                            n,
                            quant_scale_math=mode,
                            swiglu_limit=args.swiglu_limit,
                        )
                    reference_metrics[name] = compare_to_reference(
                        outputs[name], references[mode]
                    )
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "oracle_quant_scale_math": modes,
                            "arithmetic_min_cosine": cosine_bounds,
                            "arithmetic_reference": {
                                name: asdict(value)
                                for name, value in reference_metrics.items()
                            },
                            "arithmetic_replays": {
                                name: [
                                    asdict(
                                        compare_to_reference(
                                            value, references[modes[name]]
                                        )
                                    )
                                    for value in values
                                ]
                                for name, values in reference_replays.items()
                            },
                            "replay_variation": replay_variation,
                        }
                    ),
                    flush=True,
                )
                for name, value in reference_metrics.items():
                    check_reference(value, min_cosine=cosine_bounds[name])
                    for replay in reference_replays[name]:
                        check_reference(
                            compare_to_reference(replay, references[modes[name]]),
                            min_cosine=cosine_bounds[name],
                        )
            relative_l2 = (
                (outputs["csf"].float() - outputs["expanded"].float()).norm()
                / outputs["expanded"].float().norm()
            ).item()
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "correctness": asdict(metrics),
                        "relative_l2": relative_l2,
                        "replay_variation": replay_variation,
                    }
                ),
                flush=True,
            )
            check_reference(metrics)
            if "blocks" in outputs:
                block_metrics = compare_to_reference(outputs["blocks"], outputs["csf"])
                print(
                    json.dumps(
                        {"tokens": tokens, "block_correctness": asdict(block_metrics)}
                    ),
                    flush=True,
                )
                check_reference(block_metrics)
            for name, output in outputs.items():
                if name.startswith("compact"):
                    compact_metrics = compare_to_reference(output, outputs["expanded"])
                    print(
                        json.dumps(
                            {
                                "tokens": tokens,
                                "arm": name,
                                "compact_correctness": asdict(compact_metrics),
                            }
                        ),
                        flush=True,
                    )
                    check_reference(compact_metrics)
            samples = race_graphs(graphs, device, args, lambda: x.copy_(source_x))
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "configs": {
                            name: asdict(b.execution_plan.decode_config)
                            for name, b in bindings.items()
                        },
                        "median_us": {
                            name: statistics.median(v) for name, v in samples.items()
                        },
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )
            if args.trace_graphs is not None:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    for name, graph in graphs.items():
                        with torch.profiler.record_function(f"csf-{name}-{tokens}"):
                            graph.replay()
                            torch.cuda.synchronize()
                profiler.export_chrome_trace(
                    str(args.trace_graphs.with_suffix(f".{tokens}.json"))
                )
            if args.profile_graphs:
                torch.cuda.profiler.start()
                try:
                    for name, graph in graphs.items():
                        with torch.cuda.nvtx.range(f"csf-{name}-{tokens}"):
                            graph.replay()
                            torch.cuda.synchronize()
                finally:
                    torch.cuda.profiler.stop()
            for graph in graphs.values():
                graph.reset()


if __name__ == "__main__":
    main()
