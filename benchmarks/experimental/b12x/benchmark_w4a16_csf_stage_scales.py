#!/usr/bin/env python3
"""W4A16 MoE with NVFP4-CSF scales: native, expansion pass and stage-read scales.

GLM-5.3-Flash TP2 geometry by default: 288 experts, hidden 4096, intermediate
1024 per rank, top-8. FP4 weights are random; block scales follow a narrow
per-row range with a configurable share of out-of-window values, close to
the QAD checkpoint (3.6% of four-byte words carry an exception there). The
three arms share weights and routed inputs, are planned, prepared and graph
captured outside timing and replayed in balanced order. Every arm must pass the
W4A16 reference gate, with at most one differing element per 10,000 versus native scales.

- native: uncompressed MMA-packed scales.
- pass:   compressed scales expanded into scratch before every call
          (B12X_W4A16_CSF_INLINE=0 at preparation).
- stage:  stage-readable compressed scales. Calls planned up to
          B12X_W4A16_CSF_STAGE_MAX_TOKENS rebuild each pipeline stage's scales
          in shared memory; larger calls expand the routed experts from the
          same storage first.
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

from b12x._lib.quant.nvfp4_csf import make_nvfp4_csf_batch  # noqa: E402
from b12x.moe import fused_moe as moe  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from b12x.preparation._measurement import _prepare_race, measure_race_steps  # noqa: E402
from b12x.testing.reference.w4a16_reference import (
    compare_to_reference,
    moe_reference_w4a16,
)  # noqa: E402


def logical_scales(rng, experts, rows, columns, outliers):
    base = rng.integers(0x28, 0x48, size=(experts, rows, 1))
    scales = base + rng.integers(0, 14, size=(experts, rows, columns))
    hot = rng.random((experts, rows, columns)) < outliers
    scales[hot] = rng.integers(0x40, 0x7E, size=int(hot.sum()))
    return scales.astype(np.uint8)


def swizzled(logical, device):
    """F8_128x4 memory order of logical [E, rows, columns] scales."""
    e, rows, columns = logical.shape
    order = logical.reshape(e, rows // 128, 4, 32, columns // 4, 4).transpose(
        0, 1, 4, 3, 2, 5
    )
    return (
        torch.from_numpy(np.ascontiguousarray(order).reshape(e, rows, columns))
        .to(device)
        .view(torch.float8_e4m3fn)
    )


def compressed(logical, device):
    experts, rows, columns = logical.shape
    fixed, exceptions = [], []
    for plane in logical:
        base = np.minimum(plane.min(axis=1), 240).astype(np.uint8)
        offsets = plane.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(position | (plane.ravel()[position].astype(np.uint32) << 24))
    return make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, device=device
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", default="4,32,64,3072")
    parser.add_argument("--outliers", type=float, default=0.03)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument(
        "--block-size-m", type=int, help="pin the route block size of every arm"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(53)
    e, h, n = args.experts, args.hidden, args.intermediate
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device=device)
    l13 = logical_scales(rng, e, 2 * n, h // 16, args.outliers)
    l2 = logical_scales(rng, e, h, n // 16, args.outliers)
    one = torch.ones(e, device=device)
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )

    def packed_weights(w13_scales, w2_scales):
        return moe.PackedWeights(
            w13=w13.clone(),
            w2=w2.clone(),
            w13_block_scales=w13_scales,
            w2_block_scales=w2_scales,
            w13_global_scales=one,
            w2_global_scales=one,
            input_scale=one,
            intermediate_scale=one,
            immutable_input_scales=True,
        )

    native_scales = (swizzled(l13, device), swizzled(l2, device))
    arms = {
        "native": moe.prepare_weights(
            plan=plan, weights=packed_weights(*(s.clone() for s in native_scales))
        )
    }
    batches = (compressed(l13, device), compressed(l2, device))
    for name, inline in (("pass", "0"), ("stage", "1")):
        os.environ["B12X_W4A16_CSF_INLINE"] = inline
        scratch = (swizzled(l13, device).clone(), swizzled(l2, device).clone())
        arms[name] = moe.prepare_weights(
            plan=plan,
            weights=moe.Nvfp4CsfWeights(
                packed=packed_weights(*scratch),
                w13_scales=batches[0],
                w2_scales=batches[1],
            ),
        )
    assert (
        arms["pass"]._impl.w4a16_expanded is None
        and arms["stage"]._impl.w4a16_expanded is not None
    )
    native_bytes = l13.nbytes + l2.nbytes
    stored = (
        arms["stage"]._impl.w1_blockscale.numel()
        + arms["stage"]._impl.w2_blockscale.numel()
    )
    print(
        f"scale bytes: native {native_bytes / 2**20:.1f} MiB, stage-readable storage "
        f"{stored / 2**20:.1f} MiB ({stored / native_bytes:.1%})",
        flush=True,
    )

    results = []
    for tokens in (int(t) for t in args.tokens.split(",")):
        torch.manual_seed(tokens)
        x = (torch.randn(tokens, h, device=device) * 0.5).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(e, device=device)[: args.topk] for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.softmax(
            torch.randn(tokens, args.topk, device=device), dim=-1
        ).float()
        reference = moe_reference_w4a16(
            x,
            w13,
            native_scales[0],
            one,
            w2,
            native_scales[1],
            one,
            ids,
            weights,
            e,
            h,
            n,
            activation="silu",
        )
        plans = {
            name: moe.plan_execution(
                experts=experts,
                capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=args.topk),
                invocation={"fast_math": True}
                | (
                    {"w4a16_block_size_m": args.block_size_m}
                    if args.block_size_m
                    else {}
                ),
            )
            for name, experts in arms.items()
        }

        def prepare(state):
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch.scratch_specs()
            )
            output = torch.empty_like(x)
            binding = state.bind(
                a=x,
                topk_ids=ids,
                topk_weights=weights,
                output=output,
                scratch=scratch,
                input_scales_static=True,
            )
            return PreparedCall(
                run=lambda: state.run(binding), output=output, owners=(scratch, binding)
            )

        graphs, outputs, owners = {}, {}, []
        with PreparationSession(
            device=device, autotune=False, compile_workers=0
        ) as session:
            session.prepare(
                tuple(
                    p.request(name=f"csf-{name}-{tokens}", prepare_call=prepare)
                    for name, p in plans.items()
                )
            )
            bindings = {}
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
                    topk_weights=weights,
                    output=outputs[name],
                    scratch=scratch,
                    input_scales_static=True,
                )
                owners.append(scratch)
            session.freeze()
            for name, binding in bindings.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    moe.run(binding=binding)
                graphs[name] = graph
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            # Large calls may reduce top-k routes in a nondeterministic order;
            # the GPU tests check exact equality under deterministic reduction.
            oracle_metrics = {}
            for name, output in outputs.items():
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                metrics = compare_to_reference(output, reference)
                assert metrics.cos >= 0.9975, (name, metrics)
                oracle_metrics[name] = vars(metrics)
            mismatches = {}
            for name in ("pass", "stage"):
                mismatches[name] = int((outputs[name] != outputs["native"]).sum())
                if mismatches[name] > outputs[name].numel() // 10000:
                    raise AssertionError(
                        f"{name} differs at {mismatches[name]} elements"
                    )
            for _ in range(150):
                for graph in graphs.values():
                    graph.replay()
            calls = tuple(
                PreparedCall(
                    run=graph.replay,
                    produce=lambda: None,
                    owners=(graph, bindings, owners),
                )
                for graph in graphs.values()
            )
            race = _prepare_race(
                calls,
                device_ordinal=torch.cuda.current_device(),
                samples=args.replays,
                primed=True,
            )
            raw_rounds, seen = [], 0
            try:
                for _ in measure_race_steps(
                    race, device_ordinal=torch.cuda.current_device(), rounds=args.rounds
                ):
                    if race.completed_rounds != seen:
                        seen = race.completed_rounds
                        raw_rounds.append(tuple(race.latest_round_us))
                if race.completed_rounds != seen:
                    raw_rounds.append(tuple(race.latest_round_us))
            finally:
                race.close()
            samples = {
                name: [row[i] for row in raw_rounds] for i, name in enumerate(graphs)
            }
            for graph in graphs.values():
                graph.reset()
        row = {
            "tokens": tokens,
            **{name: statistics.median(v) for name, v in samples.items()},
            "mismatches": mismatches,
            "oracle_metrics": oracle_metrics,
            "samples_us": samples,
        }
        results.append(row)
        print(
            f"T={tokens:5d}  native {row['native']:8.1f} us  pass {row['pass']:8.1f} us "
            f"({row['pass'] / row['native'] - 1:+6.1%})  stage {row['stage']:8.1f} us "
            f"({row['stage'] / row['native'] - 1:+6.1%})  differing outputs {mismatches}",
            flush=True,
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "args": vars(args) | {"json": str(args.json)},
        "results": results,
        "gpu": props.name,
        "uuid": str(props.uuid),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "diff": subprocess.check_output(["git", "diff", "--binary"], text=True),
        "method": "balanced cold-L2 frozen graph replay using preparation timing; lower microseconds is better",
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
