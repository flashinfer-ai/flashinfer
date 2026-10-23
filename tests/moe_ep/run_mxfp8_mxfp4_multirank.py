# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded NVLink correctness gate, run with torchrun --nproc-per-node=2.

Use an external process timeout: a broken persistent synchronization protocol
cannot be safely interrupted by a Python exception on the submitting thread.
This is correctness validation, not a performance benchmark or an IB test.
Use --allow-single-rank for EP1 relaunch diagnostics without peer coverage.

All-output model-width regression (four tokens per rank, not sampled):
    timeout 300s torchrun --standalone --nproc-per-node=2 \
      tests/moe_ep/run_mxfp8_mxfp4_multirank.py --hidden 6400 \
      --intermediate 4096 --experts 4 --top-k 2 --tokens-per-rank 4 \
      --bounded-reference
"""

import argparse
import json
import os
import faulthandler
import traceback
import time

import torch
import torch.distributed as dist


def _knobs(args):
    """Merge --knobs JSON and --num-stages into one tactic override."""
    knobs = {}
    if args.knobs:
        parsed = json.loads(args.knobs)
        if not isinstance(parsed, dict):
            raise SystemExit("--knobs must be a JSON object")
        knobs = {k: tuple(v) if isinstance(v, list) else v for k, v in parsed.items()}
    if args.num_stages is not None:
        if "num_stages" in knobs:
            raise SystemExit("pass num_stages via --num-stages or --knobs, not both")
        knobs["num_stages"] = args.num_stages
    return knobs or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens-per-rank", type=int, default=7)
    parser.add_argument("--bounded-reference", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Tune two workspace layouts through the general autotuner before each scenario.",
    )
    parser.add_argument(
        "--allow-single-rank",
        action="store_true",
        help="Allow EP1 relaunch diagnostics; does not validate peer traffic or rank skew.",
    )
    parser.add_argument(
        "--rank-skew-us",
        type=int,
        default=0,
        help="Rotate a host-side delay across ranks before each eager/replay launch.",
    )
    parser.add_argument(
        "--random-routing",
        action="store_true",
        help="Use unsorted unique random experts instead of the remote-owner stress pattern.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        choices=range(2, 9),
        default=None,
        help="Override the operand-pipeline depth (tactic num_stages).",
    )
    parser.add_argument(
        "--knobs",
        type=str,
        default=None,
        help="JSON object merged into the tactic (e.g. group_hint, mma_tiler_mnk).",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Save each scenario's eager output to <path>-rank<r>.pt, to compare "
        "two runs bitwise (e.g. in-kernel vs external top-k reduction).",
    )
    args = parser.parse_args()
    for name in ("hidden", "intermediate"):
        if getattr(args, name) <= 0 or getattr(args, name) % 128:
            parser.error(f"{name} must be a positive multiple of 128")
    if args.tokens_per_rank <= 0 or args.top_k <= 0:
        parser.error("tokens-per-rank and top-k must be positive")
    if args.repeat <= 0 or args.rank_skew_us < 0:
        parser.error("repeat must be positive and rank-skew-us non-negative")
    faulthandler.dump_traceback_later(120)
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.comm import (
        bootstrap_dist,
        finalize_dist,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
        get_symm_buffer_for_mxfp8_mxfp4_mega_moe,
        mxfp8_mxfp4_mega_moe,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
        compute_reference,
        interleave_gate_up_8,
        quantize_mxfp4,
        quantize_mxfp8,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        to_atom_swizzled,
    )

    _, rank, world, _ = bootstrap_dist()
    if world < 2 and not args.allow_single_rank:
        raise ValueError("this gate requires at least two ranks")
    tokens, hidden, intermediate, topk = (
        args.tokens_per_rank,
        args.hidden,
        args.intermediate,
        args.top_k,
    )
    experts = args.experts if args.experts is not None else world * 2
    if experts <= 0 or experts % world or topk > experts:
        raise ValueError(
            "experts must be positive/divisible by world; top-k <= experts"
        )
    local_experts = experts // world
    gen = torch.Generator(device="cuda").manual_seed(20260922)

    def random(shape):
        return torch.randn(shape, generator=gen, device="cuda") * 0.125

    def quantize_weights(value):
        if not args.bounded_reference:
            return quantize_mxfp4(value)
        # Bound the FP4 candidate-distance temporary independently of model
        # dimensions. Quantization is row-local, so chunking changes no bytes.
        quantized, scales = [], []
        for expert in value:
            chunks = [quantize_mxfp4(part) for part in expert.split(128)]
            quantized.append(torch.cat([chunk[0] for chunk in chunks]))
            scales.append(torch.cat([chunk[1] for chunk in chunks]))
        return torch.stack(quantized), torch.stack(scales)

    xq, xs = quantize_mxfp8(random((world, tokens, hidden)))
    w13q, w13s = quantize_weights(
        interleave_gate_up_8(random((experts, 2 * intermediate, hidden)))
    )
    w2q, w2s = quantize_weights(random((experts, hidden, intermediate)))
    # Both routes visit a remote owner. Adjacent slots go to distinct experts;
    # alternating local-expert selections leave skewed and empty expert work.
    ids = torch.empty((world, tokens, topk), dtype=torch.int64, device="cuda")
    for source in range(world):
        for slot in range(topk):
            # Start with remote experts. If top-k exceeds one peer's experts,
            # continue around the global expert ring without duplicate IDs.
            ids[source, :, slot] = (
                ((source + 1) % world) * local_experts + slot
            ) % experts
    if args.random_routing:
        if topk > experts:
            raise ValueError("random routing requires top-k <= experts")
        ids = (
            torch.rand((world, tokens, experts), device="cuda", generator=gen)
            .argsort(dim=-1)[..., :topk]
            .contiguous()
        )
    ids[:, -1, -1] = -1
    scores = random((world, tokens, topk))
    local = slice(rank * local_experts, (rank + 1) * local_experts)
    w13 = (
        w13q[local].clone().transpose(1, 2),
        torch.stack([to_atom_swizzled(s) for s in w13s[local]]),
    )
    w2 = (
        w2q[local].clone().transpose(1, 2),
        torch.stack([to_atom_swizzled(s) for s in w2s[local]]),
    )
    tuning_candidates = None
    if args.autotune:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
            resolve_tactic,
        )

        base = dict(
            resolve_tactic(
                tokens, _knobs(args), hidden=hidden, intermediate=intermediate
            )
        )
        m, _, k = base["mma_tiler_mnk"]
        tuning_candidates = [
            dict(base, mma_tiler_mnk=(m, 64, k), in_kernel_reduce=False),
            dict(base, mma_tiler_mnk=(m, 128, k), in_kernel_reduce=True),
        ]
    workspace = get_symm_buffer_for_mxfp8_mxfp4_mega_moe(
        experts,
        tokens,
        topk,
        hidden,
        intermediate,
        rank,
        world,
        knobs=_knobs(args),
        capacity_tactics=tuning_candidates,
    )
    workspace.x.copy_(xq[rank])
    workspace.x_sf.view(torch.uint8).copy_(xs[rank])
    workspace.topk_weights.copy_(scores[rank])

    def launch():
        return mxfp8_mxfp4_mega_moe(None, w13, w2, workspace, num_tokens=active_tokens)

    def oracle(routes):
        kwargs = dict(
            input_q=xq,
            input_scale=xs,
            topk_ids=routes,
            topk_weights=scores,
            w13_packed=w13q,
            w13_scale=w13s,
            w2_packed=w2q,
            w2_scale=w2s,
        )
        if not args.bounded_reference:
            ref = compute_reference(**kwargs)
            return ref.route_terms, ref.output
        # Only one expert's weights are dequantized at a time. Keep unweighted
        # BF16 route terms, NOT the oracle's already rounded weighted outputs.
        terms = torch.zeros(
            (world, tokens, topk, hidden), dtype=torch.bfloat16, device="cuda"
        )
        for expert in range(experts):
            selected = routes == expert
            if not selected.any().item():
                continue
            kwargs.update(
                topk_ids=torch.where(selected, 0, -1),
                w13_packed=w13q[expert : expert + 1],
                w13_scale=w13s[expert : expert + 1],
                w2_packed=w2q[expert : expert + 1],
                w2_scale=w2s[expert : expert + 1],
            )
            # Expert route sets are disjoint; adding zero changes no BF16 term.
            terms.add_(compute_reference(**kwargs).route_terms)
        result = torch.zeros(
            (world, tokens, hidden), dtype=torch.float32, device="cuda"
        )
        for slot in range(topk):
            product = terms[:, :, slot].float() * scores[:, :, slot, None].float()
            result.add_(
                torch.where((routes[:, :, slot] >= 0).unsqueeze(-1), product, 0.0)
            )
        return terms, result.to(torch.bfloat16)

    saved = {}
    try:
        first_scenario = "local_only" if world == 1 else "cross_peer"
        for scenario in (first_scenario, "empty_source", "restored_routes"):
            active_tokens = 0 if scenario == "empty_source" and rank == 0 else tokens
            routes = ids.clone()
            if scenario == "empty_source":
                routes[0].fill_(-1)
            workspace.topk_idx.copy_(routes[rank])
            _, expected_output = (
                value[rank, :active_tokens] for value in oracle(routes)
            )
            expected = expected_output
            if args.autotune:
                from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
                    autotune_mxfp8_mxfp4_mega_moe,
                )

                winner = autotune_mxfp8_mxfp4_mega_moe(
                    None,
                    w13,
                    w2,
                    workspace,
                    num_tokens=active_tokens,
                    candidates=tuning_candidates,
                    warmup_iters=0,
                    timed_iters=3,
                )
                if world > 1:
                    winners = [None] * world
                    dist.all_gather_object(winners, winner)
                    assert all(item == winner for item in winners)
                print(
                    f"rank={rank} scenario={scenario} autotune winner={winner}",
                    flush=True,
                )
            print(f"rank={rank} scenario={scenario} eager begin", flush=True)
            baseline = None
            for iteration in range(args.repeat):
                if args.rank_skew_us and rank == iteration % world:
                    time.sleep(args.rank_skew_us / 1e6)
                actual = launch()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
                if baseline is None:
                    baseline = actual.clone()
                else:
                    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            saved[scenario] = baseline.cpu()
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                launch()
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                actual = launch()
            for iteration in range(args.repeat):
                if args.rank_skew_us and rank == iteration % world:
                    time.sleep(args.rank_skew_us / 1e6)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "world_size": world,
                        "scenario": scenario,
                        "num_tokens": active_tokens,
                        "hidden": hidden,
                        "intermediate": intermediate,
                        "experts": experts,
                        "top_k": topk,
                        "oracle": "all-output",
                        "status": "passed",
                        "max_error": (
                            (actual.float() - expected.float()).abs().max().item()
                            if active_tokens
                            else 0.0
                        ),
                    }
                ),
                flush=True,
            )
            del graph
        if args.save_output:
            torch.save(saved, f"{args.save_output}-rank{rank}.pt")
    except BaseException:
        # A failed rank must not enter collective free while peers are still
        # compiling/launching: that hides the original error behind a hang.
        traceback.print_exc()
        os._exit(1)
    workspace.destroy()
    finalize_dist()


if __name__ == "__main__":
    main()
