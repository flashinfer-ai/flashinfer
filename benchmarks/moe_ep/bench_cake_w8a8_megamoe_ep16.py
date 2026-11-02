"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Same-process EP16 latency comparison of the experimental Cake W8A8 MXFP8
# MegaMoE backend (``sm103_mxfp8_mxfp8_bf16_cake``) against the
# ``sm100_mxfp8_mxfp8_bf16_cutedsl`` backend (legacy inference pipeline,
# GenPhase heuristic, GenPhase ``knobs="auto"``) through ``MoEEpMegaLayer``.
#
# Six contract rows: 16 / 32 / 64 tokens per rank (256 / 512 / 1024 global)
# x balanced / hotset50 routing. The timed call is one ``forward`` (bf16
# staging + in-kernel quantization + push dispatch + FC1/FC2 + combine), or with
# ``--cuda-graph`` one replay of a graph that captured it (the deployment
# path; removes host launch gaps); CUDA events around it, cold L2 (a 256 MiB
# flush before every iteration), per-rank median over iterations, maximum over
# ranks, median over counterbalanced rounds.
# Prints a markdown table with absolute microseconds and baseline / Cake ratios.
#
#   torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint $HOST:29500 \
#       benchmarks/moe_ep/bench_cake_w8a8_megamoe_ep16.py --arms cake fi_inference fi_gp_heuristic fi_gp_auto

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

WORLD = 16
LOCAL_EXPERTS = 32
EXPERTS = WORLD * LOCAL_EXPERTS
HIDDEN = 3072
INTERMEDIATE = 5120
TOP_K = 8
L2_FLUSH_BYTES = 256 << 20
ROWS = tuple(
    (tokens, routing) for tokens in (16, 32, 64) for routing in ("balanced", "hotset50")
)
ARMS = {
    "cake": dict(kind="cake"),
    "fi_inference": dict(kind="fi", kernel_variant="inference"),
    "fi_gp_heuristic": dict(kind="fi", kernel_variant="genphase"),
    "fi_gp_auto": dict(kind="fi", kernel_variant="genphase", knobs="auto"),
}


def _routing(
    tokens: int, routing: str, rank: int, device: torch.device
) -> torch.Tensor:
    """The contract's routing generators (same global load pattern per row)."""

    token = torch.arange(tokens, dtype=torch.int64, device=device).unsqueeze(1)
    slot = torch.arange(TOP_K, dtype=torch.int64, device=device).unsqueeze(0)
    if routing == "balanced":
        return (token * TOP_K + slot + rank * TOP_K).remainder(EXPERTS)
    hot_slots, hot_experts = TOP_K // 2, EXPERTS // 4
    hot = (token * hot_slots + slot + rank * hot_slots).remainder(hot_experts)
    cold = hot_experts + (
        token * (TOP_K - hot_slots) + (slot - hot_slots) + rank * (TOP_K - hot_slots)
    ).remainder(EXPERTS - hot_experts)
    return torch.where(slot < hot_slots, hot, cold)


def _inputs(tokens: int, routing: str, rank: int, device: torch.device, seed: int):
    from flashinfer.moe_ep import MoEEpTensors

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    hidden = torch.randn(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
    )
    logits = torch.randn(
        (tokens, TOP_K), dtype=torch.float32, device=device, generator=generator
    )
    return MoEEpTensors(
        hidden_states=hidden.contiguous(),
        topk_ids=_routing(tokens, routing, rank, device).contiguous(),
        topk_weights=torch.softmax(logits, dim=-1).contiguous(),
    )


def _weights(rank: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    w13 = torch.randn(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w13.mul_(HIDDEN**-0.5)
    w2 = torch.randn(
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(INTERMEDIATE**-0.5)
    return w13.contiguous(), w2.contiguous()


def _make_layer(arm: str, rank: int, device: torch.device, w13, w2, tokens: int):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    spec = ARMS[arm]
    if spec["kind"] == "cake":
        config: Any = Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig()
    else:
        config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            kind="mxfp8_e4m3",
            kernel_variant=spec["kernel_variant"],
            knobs=spec.get("knobs"),
        )
    return MoEEpMegaLayer(
        bootstrap=BootstrapConfig(world_size=WORLD, rank=rank, device=device.index),
        fleet_params=FleetParams(
            num_experts=EXPERTS, max_tokens_per_rank=tokens, token_hidden_size=HIDDEN
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=config, quantize_input=True, preprocess_weights=True
        ),
    )


def _max_rank(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _capture(layer, inputs, tokens: int, device: torch.device) -> torch.cuda.CUDAGraph:
    """Capture one bounded forward (``stage_inputs`` + ``compute_staged``) into a CUDA graph.

    ``compile_tokens_per_rank=tokens`` is the rank-uniform bound the capture needs;
    one eager pass through the same staged path precedes the capture.
    """

    layer.stage_inputs(inputs, compile_tokens_per_rank=tokens)
    layer.compute_staged(output=None)
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.stage_inputs(inputs, compile_tokens_per_rank=tokens)
        layer.compute_staged(output=None)
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph


def _time(run, iters: int, flush: torch.Tensor | None) -> float:
    spans = []
    for _ in range(iters):
        dist.barrier()
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        stop.synchronize()
        spans.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(spans)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Same-process EP16 latency comparison of the experimental Cake W8A8 MXFP8 "
            "MegaMoE backend against sm100_mxfp8_mxfp8_bf16_cutedsl "
            "(torchrun, 16 ranks)."
        )
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=sorted(ARMS))
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--routing",
        nargs="+",
        default=["balanced", "hotset50"],
        choices=("balanced", "hotset50"),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-l2-flush", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture stage_inputs + compute_staged of every arm into a CUDA graph "
        "and time replays instead of eager forward calls",
    )
    parser.add_argument("--seed", type=int, default=13007)
    parser.add_argument(
        "--jsonl", default=None, help="append one JSON record per row (rank 0)"
    )
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD:
        raise RuntimeError("this benchmark requires torchrun with WORLD_SIZE=16")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    flush = (
        None
        if args.no_l2_flush
        else torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    )
    props = torch.cuda.get_device_properties(device)
    caption = (
        f"EP16, {props.name}, sm_{props.major}{props.minor}, CUDA {torch.version.cuda}, "
        f"torch {torch.__version__}, cold L2={flush is not None}, "
        f"{'CUDA-graph replay' if args.cuda_graph else 'eager forward'}, "
        f"{args.iters} iters x {args.rounds} rounds, max over ranks of per-rank medians"
    )
    rows: list[dict[str, Any]] = []
    try:
        w13, w2 = _weights(rank, device, args.seed)
        for tokens in args.tokens:
            for routing in args.routing:
                inputs = _inputs(tokens, routing, rank, device, args.seed + 7)
                layers = {}
                runs = {}
                graphs = {}
                for arm in args.arms:
                    layer = _make_layer(arm, rank, device, w13, w2, tokens)
                    layer.warmup()
                    for _ in range(args.warmup):
                        layer.forward(inputs)
                    torch.cuda.synchronize(device)
                    layers[arm] = layer
                    if args.cuda_graph:
                        graphs[arm] = _capture(layer, inputs, tokens, device)
                        runs[arm] = graphs[arm].replay
                    else:
                        runs[arm] = lambda layer=layer: layer.forward(inputs)
                dist.barrier()
                per_arm: dict[str, list[float]] = {arm: [] for arm in args.arms}
                for rnd in range(args.rounds):
                    order = (
                        list(args.arms) if rnd % 2 == 0 else list(reversed(args.arms))
                    )
                    for arm in order:
                        local_us = _time(runs[arm], args.iters, flush)
                        per_arm[arm].append(_max_rank(local_us, device))
                record = {
                    "tokens_per_rank": tokens,
                    "global_tokens": WORLD * tokens,
                    "routing": routing,
                    **{
                        f"{arm}_us": statistics.median(per_arm[arm])
                        for arm in args.arms
                    },
                }
                if "cake" in args.arms:
                    from flashinfer.experimental.cake_w8a8_megamoe_ep16 import (
                        routing_overflow,
                    )

                    cake_overflow = int(routing_overflow(layers["cake"]).item())
                    record["cake_overflow_flag"] = int(
                        _max_rank(float(cake_overflow), device)
                    )
                    for arm in args.arms:
                        if arm != "cake":
                            record[f"{arm}_over_cake"] = (
                                record[f"{arm}_us"] / record["cake_us"]
                            )
                rows.append(record)
                torch.cuda.synchronize(device)
                graphs.clear()  # retire the graphs before their workspace is released
                runs.clear()
                for layer in reversed(list(layers.values())):
                    layer.destroy()
                layers.clear()
                torch.cuda.empty_cache()
                dist.barrier()
    finally:
        if rank == 0 and rows:
            arms = args.arms
            header = [
                "global tokens",
                "tokens/rank",
                "routing",
                *[f"{arm} (us)" for arm in arms],
            ]
            header += [
                f"{arm} / cake" for arm in arms if arm != "cake" and "cake" in arms
            ]
            lines = [
                f"Latency per forward ({caption})",
                "",
                "| " + " | ".join(header) + " |",
                "|" + "---|" * len(header),
            ]
            for record in rows:
                cells = [
                    str(record["global_tokens"]),
                    str(record["tokens_per_rank"]),
                    record["routing"],
                ]
                cells += [f"{record[f'{arm}_us']:.1f}" for arm in arms]
                cells += [
                    f"{record[f'{arm}_over_cake']:.2f}x"
                    for arm in arms
                    if arm != "cake" and "cake" in arms
                ]
                lines.append("| " + " | ".join(cells) + " |")
            print("\n".join(lines), flush=True)
            if args.jsonl:
                with open(args.jsonl, "a") as handle:
                    for record in rows:
                        handle.write(json.dumps({"caption": caption, **record}) + "\n")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
