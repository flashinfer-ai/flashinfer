#!/usr/bin/env python3
"""W4A8 MoE with MXFP4-CSF scales: native, per-call expansion and inline reads.

DeepSeek-V4.1-Flash TP4 geometry by default: 384 experts, hidden 5120,
intermediate 576 per rank (compact N64 W4A8), top-6, MXFP8 activations. The
three arms share FP4 weights, scales and routed inputs, are planned, prepared
and graph captured outside timing, replay interleaved, and must produce
identical output.

- native: uncompressed checkpoint scales (no CSF).
- expand: CSF scales expanded for the routed experts into the shared native
          scale scratch before every call (B12X_W4A8_CSF_INLINE=0).
- inline: the compact W4A8 kernels read CSF scales inline (the default).

Scales are synthetic by default: one base per row with a 0/1 selector and
``--exceptions`` of the values one below or two above the base, plus
``--dense-rows`` rows of out-of-window values (the clustered exceptions of
the last DS4.1 layers). ``--checkpoint`` reads one real layer and TP rank
through the vLLM MXFP4-CSF reader instead (weights and scales).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x._lib.quant.x4t_scales import make_x4t_scale_batch  # noqa: E402
from b12x.moe import fused_moe as moe  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from b12x.preparation._measurement import _prepare_race, measure_race_steps  # noqa: E402
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_w4a8_mx  # noqa: E402


def synthetic_planes(rng, experts, rows, columns, exceptions, dense_rows, device):
    """CSF scale planes and their logical E8M0 grids."""
    fixed, records, grids = [], [], []
    for _ in range(experts):
        bases = rng.integers(112, 124, rows).astype(np.uint8)
        grid = bases[:, None] + rng.integers(0, 2, (rows, columns)).astype(np.uint8)
        hot = rng.random((rows, columns)) < exceptions
        grid[hot] = (bases[:, None] + rng.choice([-1, 2], (rows, columns)))[hot]
        for row in rng.choice(rows, dense_rows, replace=False):
            grid[row] = bases[row] + rng.integers(-1, 4, columns)
        selected = grid == bases[:, None] + 1
        outside = (grid != bases[:, None]) & ~selected
        selectors = np.packbits(selected, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate((bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1)
            )
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        records.append(
            torch.from_numpy(position | (grid.reshape(-1)[position].astype(np.uint32) << 24))
        )
        grids.append(grid)
    batch = make_x4t_scale_batch(fixed, records, rows=rows, columns=columns, device=device)
    return batch, torch.from_numpy(np.stack(grids)).to(device)


def checkpoint_layer(args, device):
    """Weights and CSF planes of one real layer and TP rank, plus native grids."""
    from b12x._lib.quant.x4t_scales import decode_x4t_scales
    from vllm.model_executor.model_loader.mxfp4_csf_loader import read_mxfp4_csf_layer

    e, h = args.experts, args.hidden
    full = args.intermediate * args.tp
    n = args.intermediate
    scratch = (
        torch.empty((e, h // 32, 2 * n), dtype=torch.uint8, device=device),
        torch.empty((e, n // 32, h), dtype=torch.uint8, device=device),
    )
    weights = read_mxfp4_csf_layer(
        args.checkpoint, args.layer, num_experts=e, hidden_size=h,
        intermediate_size=full, tp_rank=args.rank, tp_size=args.tp,
        device=device, w13_scale_scratch=scratch[0], w2_scale_scratch=scratch[1],
    )
    planes, grids = [], []
    ids = torch.arange(e, dtype=torch.int32, device=device)
    for source, rows, columns in (
        (weights.w13_scales, 2 * n, h // 32), (weights.w2_scales, h, n // 32)
    ):
        batch = make_x4t_scale_batch(
            source.fixed, source.exceptions, rows=rows, columns=columns, device=device
        )
        grid = torch.empty((e, rows, columns), dtype=torch.uint8, device=device)
        decode_x4t_scales(batch, ids, grid)
        planes.append(batch)
        grids.append(grid)
    return weights.w13, weights.w2, planes, grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=576, help="per TP rank")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", default="8,64,128")
    parser.add_argument("--exceptions", type=float, default=3e-4)
    parser.add_argument("--dense-rows", type=int, default=2)
    parser.add_argument("--w13-layout", choices=("w13", "w31"), default="w31")
    parser.add_argument("--checkpoint", type=Path, help="MXFP4-CSF checkpoint root")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--replays", type=int, default=20, help="samples per balanced timing round")
    parser.add_argument("--rounds", type=int, default=15, help="balanced cold-L2 timing rounds")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(41)
    e, h, n = args.experts, args.hidden, args.intermediate
    if args.checkpoint is not None:
        w13, w2, planes, grids = checkpoint_layer(args, device)
    else:
        w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device=device)
        w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device=device)
        planes, grids = zip(
            *(
                synthetic_planes(rng, e, rows, columns, args.exceptions, args.dense_rows, device)
                for rows, columns in ((2 * n, h // 32), (h, n // 32))
            ),
            strict=True,
        )
    one = torch.ones(e, device=device)
    plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout=args.w13_layout),
        activation=moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    arms = {
        "native": moe.prepare_weights(
            plan=plan,
            weights=moe.PackedWeights(
                w13=w13.clone(), w2=w2.clone(),
                w13_block_scales=grids[0].clone(), w2_block_scales=grids[1].clone(),
                w13_global_scales=one, w2_global_scales=one,
            ),
        )
    }
    for name, inline in (("expand", "0"), ("inline", "1")):
        os.environ["B12X_W4A8_CSF_INLINE"] = inline
        arms[name] = moe.prepare_weights(
            plan=plan,
            weights=moe.Mxfp4CsfWeights(
                w13=w13.clone(), w2=w2.clone(),
                w13_scales=planes[0], w2_scales=planes[1],
                w13_scale_scratch=torch.empty_like(grids[0]),
                w2_scale_scratch=torch.empty_like(grids[1]),
            ),
        )
    os.environ.pop("B12X_W4A8_CSF_INLINE")
    stored = arms["inline"]._impl.mxfp4_csf_inline
    assert stored is not None and arms["expand"]._impl.mxfp4_csf is not None
    native_bytes = sum(g.numel() for g in grids)
    inline_bytes = sum(p.storage.numel() for p in stored)
    print(
        f"scale bytes: native {native_bytes / 2**20:.1f} MiB, inline "
        f"{inline_bytes / 2**20:.1f} MiB ({inline_bytes / native_bytes:.1%}), raw tiles "
        f"{[p.heavy_tiles for p in stored]}",
        flush=True,
    )

    results = []
    for tokens in (int(t) for t in args.tokens.split(",")):
        torch.manual_seed(tokens)
        x = (torch.randn(tokens, h, device=device) * 0.5).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(e, device=device)[: args.topk] for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.softmax(torch.randn(tokens, args.topk, device=device), dim=-1).float()
        reference = moe_reference_w4a8_mx(
            x, w13, grids[0], None, one, w2, grids[1], None, one,
            ids, weights, e, h, n, w13_layout=args.w13_layout,
        )
        plans = {
            name: moe.plan_execution(
                experts=experts,
                capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=args.topk),
                invocation={"fast_math": True},
                routing=moe.RoutingSpec(deterministic_output=True),
            )
            for name, experts in arms.items()
        }

        def prepare(state):
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch.scratch_specs()
            )
            output = torch.empty_like(x)
            binding = state.bind(a=x, topk_ids=ids, topk_weights=weights, output=output,
                                 scratch=scratch, input_scales_static=True)
            return PreparedCall(run=lambda: state.run(binding), output=output,
                                owners=(scratch, binding))

        graphs, outputs, owners = {}, {}, []
        with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
            session.prepare(tuple(
                p.request(name=f"w4a8-csf-{name}-{tokens}", prepare_call=prepare)
                for name, p in plans.items()
            ))
            bindings = {}
            for name, p in plans.items():
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device) for s in p.scratch_specs()
                )
                outputs[name] = torch.empty_like(x)
                bindings[name] = moe.bind(
                    p, a=x, topk_ids=ids, topk_weights=weights, output=outputs[name],
                    scratch=scratch, input_scales_static=True,
                )
                owners.append(scratch)
            session.freeze()
            backends = {name: p.decode_config.backend for name, p in
                        ((name, b.execution_plan) for name, b in bindings.items())}
            for name, binding in bindings.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    moe.run(binding=binding)
                graphs[name] = graph
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            for name in ("expand", "inline"):
                if not torch.equal(outputs[name].view(torch.int16), outputs["native"].view(torch.int16)):
                    raise AssertionError(f"{name} output differs from native")
            oracle_metrics = {}
            for output in outputs.values():
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                metrics = compare_to_reference(output, reference)
                assert metrics.cos >= 0.9975, (name, metrics)
                oracle_metrics[name] = vars(metrics)
            for _ in range(150):
                for graph in graphs.values():
                    graph.replay()
            for output in outputs.values():
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            for output in outputs.values():
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                assert torch.equal(output.view(torch.int16), outputs["native"].view(torch.int16))
            calls = tuple(PreparedCall(run=graph.replay, produce=lambda: None,
                                      owners=(graph, bindings, owners)) for graph in graphs.values())
            race = _prepare_race(calls, device_ordinal=torch.cuda.current_device(),
                                 samples=args.replays, primed=True)
            raw_rounds, seen = [], 0
            try:
                for _ in measure_race_steps(race, device_ordinal=torch.cuda.current_device(), rounds=args.rounds):
                    if race.completed_rounds != seen:
                        seen = race.completed_rounds
                        raw_rounds.append(tuple(race.latest_round_us))
                if race.completed_rounds != seen:
                    raw_rounds.append(tuple(race.latest_round_us))
            finally:
                race.close()
            samples = {name: [row[i] for row in raw_rounds] for i, name in enumerate(graphs)}
            for graph in graphs.values():
                graph.reset()
        row = {
            "tokens": tokens,
            "backend": backends,
            **{name: statistics.median(v) for name, v in samples.items()},
            "samples_us": samples, "oracle_metrics": oracle_metrics,
        }
        results.append(row)
        print(
            f"T={tokens:4d} [{backends['inline']}]  native {row['native']:8.1f} us  "
            f"expand {row['expand']:8.1f} us ({row['expand'] / row['native'] - 1:+6.1%})  "
            f"inline {row['inline']:8.1f} us ({row['inline'] / row['native'] - 1:+6.1%}, "
            f"{row['inline'] / row['expand'] - 1:+6.1%} vs expand)  bit-exact",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()},
             "device": torch.cuda.get_device_name(), "results": results,
             "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "diff": subprocess.check_output(["git", "diff", "--binary"], text=True),
             "method": "balanced cold-L2 graph replay using preparation timing; lower microseconds is better"},
            indent=1,
        ))


if __name__ == "__main__":
    main()
