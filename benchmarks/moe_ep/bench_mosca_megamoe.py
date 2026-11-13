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

# Kernel-level timing of the experimental MOSCA MegaMoE backend
# (``sm100_fp8_fp4_bf16_mosca``) against the stable ``sm100_fp8_fp4_bf16_deepgemm``
# backend, both through ``MoEEpMegaLayer``, on the DeepSeek-V4-Pro MoE layer
# (384 experts, hidden 7168, intermediate 3072, top-6, SwiGLU clamp 10,
# FP8 x FP4). Weights are random E2M1 bytes with random UE8M0 scales.
#
# --layout replicated  (TP attention + EP MoE: every rank holds the same G tokens)
#   deepgemm+nccl_ag   stable DeepGEMM on this rank's tensor_split slice, then an
#                      NCCL all_gather of the slices (the pattern MOSCA replaces)
#   mosca_replicated   token_layout="replicated": in-kernel all-gather; forward
#                      returns all G rows (the copy into the output included)
#   deepgemm_only, nccl_ag_only   the two parts of the baseline, for reference
# --layout partitioned (DP attention + EP MoE: every rank holds its own T tokens)
#   deepgemm           stable DeepGEMM
#   mosca              small_t + sparse_count (add --tile-rule RULE to time a
#                      tile rule; DSV4_PRO_DEP8_TILE_RULE was tuned in SGLang
#                      with the shared expert in kernel, not on this backend)
#
# Timing: per arm, one CUDA graph captures 2 x --layers forwards that rotate
# over --layers independent weight sets (each larger than L2) and is replayed
# --reps times; the per-rank median over --rounds, then the maximum over ranks,
# in microseconds per forward. Every arm includes input staging (BF16 -> FP8).
#
#   torchrun --nproc-per-node 8 benchmarks/moe_ep/bench_mosca_megamoe.py --layout replicated
#   torchrun --nproc-per-node 8 benchmarks/moe_ep/bench_mosca_megamoe.py --layout partitioned
#
# The DeepGEMM arms need the ``deep_gemm`` package (sgl-deep-gemm 0.2.0 is the
# upstream revision MOSCA builds on); without it only the MOSCA arms run.

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import statistics

import torch
import torch.distributed as dist

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K, CLAMP = 7168, 3072, 384, 6, 10.0
REPLICATED_BATCHES = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
PARTITIONED_TOKENS = (1, 8, 16, 32, 48, 64, 128, 256, 384, 512)


def _log(*args) -> None:
    if dist.get_rank() == 0:
        print(*args, flush=True)


def _random_pack(seed: int, local_experts: int, device: torch.device):
    """Random packed-FP4 weights + FP32 UE8M0 scales (checkpoint layout)."""
    from flashinfer.moe_ep import MoEWeightPack

    g = torch.Generator(device=device).manual_seed(seed)

    def fp4(*shape):
        return torch.randint(
            -128, 128, shape, generator=g, device=device, dtype=torch.int16
        ).to(torch.int8)

    def scale(*shape):
        return torch.exp2(
            torch.randint(-9, -5, shape, generator=g, device=device).float()
        )

    return MoEWeightPack(
        w13=fp4(local_experts, 2 * INTERMEDIATE, HIDDEN // 2),
        w2=fp4(local_experts, HIDDEN, INTERMEDIATE // 2),
        w13_scale=scale(local_experts, 2 * INTERMEDIATE, HIDDEN // 32),
        w2_scale=scale(local_experts, HIDDEN, INTERMEDIATE // 32),
    )


def _inputs(tokens: int, seed: int, device: torch.device):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=g, device=device).to(torch.bfloat16)
    scores = torch.randn(tokens, EXPERTS, generator=g, device=device)
    w, ids = torch.topk(scores, TOP_K, dim=-1, sorted=False)
    return x, ids.to(torch.int64), (torch.softmax(w, dim=-1) * 2.5).float()


def _forward(layer, x, ids, w, out):
    from flashinfer.moe_ep import MoEEpTensors

    return layer.forward(
        MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=w, output=out)
    )


def _graph_time(fn, n_calls: int, reps: int, rounds: int, group) -> float:
    """Median over rounds of microseconds per call, maximum over ranks."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm: compiles and loads every module
        for i in range(n_calls):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(n_calls):
            fn(i)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        dist.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (reps * n_calls))
    del graph
    dist.barrier(group=group)
    value = torch.tensor([statistics.median(samples)], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return round(float(value.item()), 2)


def _time_saved(baseline_us: float, candidate_us: float) -> str:
    """Share of the baseline's time the candidate saves ("+" = less time)."""
    return f"{100.0 * (baseline_us - candidate_us) / baseline_us:+.1f} %"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOSCA MegaMoE vs stable DeepGEMM MegaMoE (torchrun, EP over all ranks)"
    )
    parser.add_argument(
        "--layout", choices=("replicated", "partitioned"), required=True
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        help="replicated: global batch G; partitioned: tokens per rank T",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens-per-rank", type=int, default=512)
    parser.add_argument(
        "--tile-rule",
        default=None,
        help="partitioned: tile_rule of the MOSCA arm (default: DeepGEMM's heuristic)",
    )
    parser.add_argument("--json", help="write the results to this file (rank 0)")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=20))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.new_group(list(range(world)), backend="nccl")
    if EXPERTS % world:
        raise SystemExit(f"{EXPERTS} experts do not split over {world} ranks")

    from flashinfer.experimental.mosca.megamoe import (
        Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
        preprocess_mega_weights as mosca_preprocess,
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
        preprocess_mega_weights as deepgemm_preprocess,
    )

    has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None
    replicated = args.layout == "replicated"
    cap = args.max_tokens_per_rank
    tokens = args.tokens or (REPLICATED_BATCHES if replicated else PARTITIONED_TOKENS)
    bootstrap = BootstrapConfig(world_size=world, rank=rank, process_group=group)
    fleet = FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=cap, token_hidden_size=HIDDEN
    )
    fleet_rep = FleetParams(
        num_experts=EXPERTS, max_tokens_per_rank=cap * world, token_hidden_size=HIDDEN
    )
    packs = [
        _random_pack(7919 * layer + 104729 * rank + 1, EXPERTS // world, device)
        for layer in range(args.layers)
    ]

    def build(config, fleet_params, preprocess):
        layers = []
        for pack in packs:
            transformed = preprocess(
                pack, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
            )
            layers.append(
                MoEEpLayer(
                    bootstrap,
                    fleet_params,
                    None,
                    backend=MegaConfig(
                        megakernel=config,
                        preprocess_weights=False,
                        transformed_weights=transformed,
                    ),
                )
            )
        return layers

    common = dict(intermediate_size=INTERMEDIATE, top_k=TOP_K, activation_clamp=CLAMP)
    deepgemm = (
        build(
            Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(**common),
            fleet,
            deepgemm_preprocess,
        )
        if has_deep_gemm
        else []
    )
    if replicated:
        mosca = build(
            Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(**common, token_layout="replicated"),
            fleet_rep,
            mosca_preprocess,
        )
    else:
        mosca = build(
            Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(
                **common, sparse_count=True, tile_rule=args.tile_rule
            ),
            fleet,
            mosca_preprocess,
        )
    packs.clear()

    n_calls, L = 2 * args.layers, args.layers
    results = {}
    _log(
        f"MOSCA MegaMoE vs stable DeepGEMM, {args.layout} layout, EP{world}, "
        f"{torch.cuda.get_device_name()}, us per forward (max over ranks)"
        + ("" if has_deep_gemm else " -- deep_gemm not installed: MOSCA arms only")
    )
    for count in tokens:
        if replicated:
            slices = [len(c) for c in torch.arange(count).tensor_split(world)]
            if max(slices) > cap:
                continue
            start, own = sum(slices[:rank]), slices[rank]
            x, ids, w = _inputs(count, 7000 + count, device)
            xs, ids_s, ws = (t[start : start + own].contiguous() for t in (x, ids, w))
            part = torch.empty(own, HIDDEN, device=device, dtype=torch.bfloat16)
            gathered = [
                torch.empty(c, HIDDEN, device=device, dtype=torch.bfloat16)
                for c in slices
            ]
            full = torch.empty(count, HIDDEN, device=device, dtype=torch.bfloat16)
            ref = torch.empty_like(full)

            def all_gather_rows(ref=ref, gathered=gathered, part=part, slices=slices):
                # The post-MoE all-gather of a framework: every rank ends with
                # the contiguous [count, hidden] batch, as the MOSCA arm's output.
                if len(set(slices)) == 1:
                    dist.all_gather_into_tensor(ref, part, group=group)
                else:
                    dist.all_gather(gathered, part, group=group)
                    torch.cat(gathered, out=ref)

            arms = {
                "mosca_replicated": lambda i: _forward(mosca[i % L], x, ids, w, full),
                "nccl_ag_only": lambda i: all_gather_rows(),
            }
            if deepgemm:
                arms["deepgemm+nccl_ag"] = lambda i: (
                    _forward(deepgemm[i % L], xs, ids_s, ws, part),
                    all_gather_rows(),
                )
                arms["deepgemm_only"] = lambda i: _forward(
                    deepgemm[i % L], xs, ids_s, ws, part
                )
            baseline, candidate = "deepgemm+nccl_ag", "mosca_replicated"
        else:
            if count > cap:
                continue
            x, ids, w = _inputs(count, 7000 + 31 * count + rank, device)
            out = torch.empty(count, HIDDEN, device=device, dtype=torch.bfloat16)
            arms = {"mosca": lambda i: _forward(mosca[i % L], x, ids, w, out)}
            if deepgemm:
                arms["deepgemm"] = lambda i: _forward(deepgemm[i % L], x, ids, w, out)
            baseline, candidate = "deepgemm", "mosca"
        row = {
            arm: _graph_time(fn, n_calls, args.reps, args.rounds, group)
            for arm, fn in arms.items()
        }
        results[count] = row
        ratio = (
            f"  time saved {_time_saved(row[baseline], row[candidate])}"
            if baseline in row and row.get(candidate)
            else ""
        )
        _log(
            f"[{'G' if replicated else 'T'}={count:5d}] "
            + " ".join(f"{k}={v}" for k, v in row.items())
            + ratio
        )

    if rank == 0:
        key = "G (global batch)" if replicated else "T (tokens per rank)"
        cols = (
            ["deepgemm+nccl_ag", "mosca_replicated", "deepgemm_only", "nccl_ag_only"]
            if replicated
            else ["deepgemm", "mosca"]
        )
        print(f"\n| {key} | " + " | ".join(cols) + " | time saved |")
        print("|---" * (len(cols) + 2) + "|")
        for count, row in results.items():
            sp = (
                _time_saved(row[baseline], row[candidate])
                if baseline in row and row.get(candidate)
                else "-"
            )
            print(
                f"| {count} | "
                + " | ".join(str(row.get(c, "-")) for c in cols)
                + f" | {sp} |"
            )
        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {
                        "layout": args.layout,
                        "tile_rule": args.tile_rule,
                        "world": world,
                        "device": torch.cuda.get_device_name(),
                        "us": results,
                    },
                    f,
                    indent=1,
                )
    for layer in mosca + deepgemm:
        layer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
