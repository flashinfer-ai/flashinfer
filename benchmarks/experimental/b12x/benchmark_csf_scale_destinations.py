#!/usr/bin/env python3
"""Compare scale expansion into native NVFP4 and MXFP4 destinations.

NVFP4 uses a checkpoint's exact scales. MXFP4 uses seeded synthetic scales with
configurable exception density. Both arms write identical native bytes into the
same caller-owned buffers. The compact expander is selected by the preparation
autotuner. Measurements are balanced cold-L2 CUDA graph races; lower is better.
These timings include route marking and expansion, and exclude expert compute.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from b12x._lib.quant.csf_blocks_tuning import plan_csf_block_expansion
from b12x._lib.quant.csf_routing import compile_csf_active_experts, mark_active_experts
from b12x._lib.quant.x4t_packed_scales import decode_x4t_packed_scale_pair
from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x.loader._csf_blocks import export_csf_blocks
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation._measurement import capture_benchmark_call
from benchmark_nvfp4_csf_checkpoint import (
    benchmark_cache_dir,
    checkpoint_weights,
    race_graphs,
    storage_bytes,
)
from checkpoint_csf import SlabScalePlanes, prepare_reference_weights, synthetic_planes


def cpu_planes(batch):
    if isinstance(batch, SlabScalePlanes):
        return batch
    boundaries = batch.exception_offsets.cpu().tolist()
    records = batch.exceptions.cpu()
    return SlabScalePlanes(
        tuple(t.cpu() for t in batch.fixed),
        tuple(
            records[a:b] for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        ),
    )


def mxfp4_storage_bytes(decoder, sources):
    tensors = (
        tuple(
            t
            for p in (decoder.first, decoder.second)
            for t in (p.bases, p.selectors, p.exceptions, p.task_offsets)
        )
        if decoder is not None
        else tuple(
            t
            for p in sources
            for t in (
                p.fixed,
                p.exceptions,
                p.exception_offsets,
                p.task_exception_offsets,
            )
            if t is not None
        )
    )
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage() for t in tensors}
    return sum(s.nbytes() for s in storages.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--expert-prefix", default="model.language_model.layers.{layer}.mlp.experts"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--activation", choices=("a4", "a8", "a16"), required=True)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=576)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--exceptions", type=float, default=0.0003)
    parser.add_argument("--tokens", default="1,8,32,128,512,4096")
    parser.add_argument("--compile-workers", type=int, default=8)
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument(
        "--tuning-execution", choices=("direct", "graph"), default="direct"
    )
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    args = parser.parse_args()
    args.tuning_cache_dir = benchmark_cache_dir(
        args.tuning_cache_dir, args.tuning_execution
    )
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    nv = args.checkpoint is not None
    if (nv and args.activation == "a8") or (not nv and args.activation == "a4"):
        parser.error(
            "Checkpoint NVFP4 requires A4/A16; synthetic MXFP4 requires A8/A16"
        )
    args.layer_max_input_scale = False
    if nv:
        weights, e, h, n, topk = checkpoint_weights(args, device)
        buffers = (weights.packed.w13_block_scales, weights.packed.w2_block_scales)
        swiglu_limit = json.loads((args.checkpoint / "config.json").read_text())
        swiglu_limit = swiglu_limit.get("text_config", swiglu_limit).get(
            "swiglu_limit", 0.0
        )
    else:
        e, h, n, topk = args.experts, args.hidden, args.intermediate, args.topk
        rng = np.random.default_rng(41)
        sources = tuple(
            synthetic_planes(rng, e, r, c, args.exceptions, 2, device)[0]
            for r, c in ((2 * n, h // 32), (h, n // 32))
        )
        logical = tuple(cpu_planes(s) for s in sources)
        sources = tuple(
            make_x4t_scale_batch(
                s.fixed,
                s.exceptions,
                rows=r,
                columns=c,
                device=device,
                exception_task_rows=64,
            )
            for s, r, c in zip(logical, (2 * n, h), (h // 32, n // 32), strict=True)
        )
        destination = 1 if args.activation == "a16" else 3 if n % 128 == 64 else 2
        shapes = tuple(
            (e, r, c)
            if destination in (1, 3)
            else (e, (r + 255) // 256 * 256, (c + 3) // 4 * 4)
            for r, c in ((2 * n, h // 32), (h, n // 32))
        )
        buffers = tuple(
            torch.empty(shape, dtype=torch.uint8, device=device) for shape in shapes
        )
        weights = moe.Mxfp4CsfWeights(
            w13=torch.zeros((e, 2 * n, h // 2), dtype=torch.uint8, device=device),
            w2=torch.zeros((e, h, n // 2), dtype=torch.uint8, device=device),
            w13_scales=sources[0],
            w2_scales=sources[1],
            w13_scale_scratch=buffers[0],
            w2_scale_scratch=buffers[1],
        )
        swiglu_limit = 0.0
    source = moe.PackedSource(
        format="modelopt_nvfp4" if nv else "fp4_e8m0_k32",
        w13_layout="w13" if nv else "w31",
    )
    plan = moe.plan_weights(
        source=source,
        activation=moe.ActivationSpec(
            mode=args.activation,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=swiglu_limit,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    legacy_weights = (
        replace(
            weights,
            packed=replace(
                weights.packed,
                w13=weights.packed.w13.clone(),
                w2=weights.packed.w2.clone(),
            ),
        )
        if nv
        else None
    )
    legacy = (
        prepare_reference_weights(plan=plan, weights=legacy_weights) if nv else None
    )
    block_planes = []
    for batch, r, c in zip(
        (weights.w13_scales, weights.w2_scales),
        (2 * n, h),
        (h // (16 if nv else 32), n // (16 if nv else 32)),
        strict=True,
    ):
        batch = cpu_planes(batch)
        block_planes.append(
            moe.CsfBlockPlanes(
                tuple(
                    export_csf_blocks(
                        f, x, rows=r, columns=c, quantization="nvfp4" if nv else "mxfp4"
                    )
                    for f, x in zip(batch.fixed, batch.exceptions, strict=True)
                )
            )
        )
    compact = (prepare_reference_weights if nv else moe.prepare_weights)(
        plan=plan,
        weights=replace(weights, w13_scales=block_planes[0], w2_scales=block_planes[1]),
    )
    outputs = (compact._impl.w1_blockscale, compact._impl.w2_blockscale)
    decoder = compact._impl.nvfp4_csf
    native_bytes = sum(t.numel() * t.element_size() for t in outputs)
    old_decoder = legacy._impl.nvfp4_csf if nv else None
    if not nv and args.activation == "a8":
        from b12x._lib.quant.mxfp4_csf import Mxfp4CsfDecoder, repack_mxfp4_csf_batch

        native_planes = tuple(
            repack_mxfp4_csf_batch(
                p, compact=n % 128 == 64, group_rows=g, row_rotation=rot
            )
            for p, g, rot in zip(sources, (n, h), (n, 0), strict=True)
        )
        old_decoder = Mxfp4CsfDecoder.prepare(*native_planes, *outputs)
    all_ids = torch.arange(e, dtype=torch.int32, device=device)
    presence = torch.empty(e, dtype=torch.int32, device=device)
    marking = compile_csf_active_experts(False)

    def baseline(ids):
        if old_decoder is not None:
            old_decoder.decode(ids, *outputs)
        else:
            selected, counts, unique = ids.reshape(-1), False, False
            if ids.numel() >= 16 * e:
                selected, unique = all_ids, True
            elif ids.numel() >= 64:
                mark_active_experts(ids, presence, marking)
                selected, counts = presence, True
            decode_x4t_packed_scale_pair(
                sources[0],
                sources[1],
                selected,
                *(t.view(torch.uint8) for t in outputs),
                expert_counts=counts,
                expert_ids_unique=unique,
            )

    expected = tuple(t.view(torch.uint8).clone() for t in outputs)
    decoder.decode_all(*outputs)
    for out, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(out.view(torch.uint8), reference, rtol=0, atol=0)
    baseline(all_ids)
    for out, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(out.view(torch.uint8), reference, rtol=0, atol=0)
    print(
        json.dumps(
            dict(
                checkpoint=str(args.checkpoint) if nv else None,
                scale_source="checkpoint" if nv else "seeded synthetic MXFP4",
                tuning_execution=args.tuning_execution,
                activation=args.activation,
                experts=e,
                hidden=h,
                intermediate=n,
                topk=topk,
                gpu=torch.cuda.get_device_name(device),
                native_bytes=native_bytes,
                compact_bytes=storage_bytes(decoder),
                destination=decoder.first.scales.destination,
                baseline_bytes=storage_bytes(old_decoder)
                if nv
                else mxfp4_storage_bytes(old_decoder, sources),
                exception_probability=args.exceptions if not nv else None,
            )
        ),
        flush=True,
    )
    for tokens in map(int, args.tokens.split(",")):
        torch.manual_seed(907 + tokens)
        ids = (
            torch.rand(tokens, e, device=device)
            .topk(topk, dim=1)
            .indices.to(torch.int32)
        )
        active = ids.unique() if ids.numel() < 16 * e else all_ids
        expansion = plan_csf_block_expansion(
            decoder.first, decoder.second, routes=ids.numel()
        )

        def prepare(state):
            scratch = torch.empty(e, dtype=torch.int32, device=device)
            binding = state.bind(
                ids=ids, out13=outputs[0], out2=outputs[1], scratch=scratch
            )
            return PreparedCall(
                run=binding.run, produce=lambda: None, owners=(scratch, binding)
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
            session.prepare(
                (
                    expansion.request(
                        name=f"expand-{tokens}",
                        prepare_call=prepare,
                        benchmark_call=benchmark,
                    ),
                )
            )
            call = prepare(expansion.prepared.state)
            baseline(ids)
            graphs = {}
            for label, run in (
                ("baseline", lambda: baseline(ids)),
                ("compact", call.run),
            ):
                for out in outputs:
                    out.view(torch.uint8).fill_(0xD3)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                for out, reference in zip(outputs, expected, strict=True):
                    torch.testing.assert_close(
                        out.view(torch.uint8)[active], reference[active], rtol=0, atol=0
                    )
                graphs[label] = graph
            samples = race_graphs(graphs, device, args, lambda: None)
            print(
                json.dumps(
                    dict(
                        tokens=tokens,
                        active_experts=active.numel(),
                        config=asdict(expansion.selection.config),
                        median_us={k: statistics.median(v) for k, v in samples.items()},
                        samples_us=samples,
                    )
                ),
                flush=True,
            )
            for graph in graphs.values():
                graph.reset()


if __name__ == "__main__":
    main()
