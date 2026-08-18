# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the GLM5 fused MoE on rank-specific TP8 tensor dumps.

Example::

    torchrun --nproc_per_node=8 benchmarks/bench_glm5_fused_moe.py \
      --dump-dir ~/dev/debug_output --warmup 20 --iterations 100

The reported CUDA time covers fused routing, expert-up, SwiGLU, expert-down,
and the local routed/shared reduction. The router GEMM and TP all-reduce are
outside this operator and therefore outside the timed region.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics

import torch

from flashinfer.fused_moe import (
    alloc_glm5_fused_moe_workspace,
    glm5_fused_moe,
    prepare_glm5_fused_moe_weights,
)


def _one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one tensor matching {pattern!r} under {path}, got {matches}"
        )
    return matches[0]


def _load(path: Path, rank: int, layer: int, name: str, device) -> torch.Tensor:
    return torch.load(path / f"r{rank}_l{layer}_{name}.pt", map_location="cpu").to(
        device
    )


def _profile(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--packed-weight-stages", type=int, default=2, choices=(1, 2))
    parser.add_argument("--no-tma", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dump_dir = args.dump_dir.expanduser()

    router_path = _one(dump_dir, f"r{rank}_l*_router_weight.pt")
    hidden_path = _one(dump_dir, f"r{rank}_l*_hidden_states.pt")
    weight_layer = int(router_path.name.split("_", 2)[1][1:])
    activation_layer = int(hidden_path.name.split("_", 2)[1][1:])
    hidden_states = _load(dump_dir, rank, activation_layer, "hidden_states", device)[
        : args.tokens
    ].contiguous()
    router_weight = _load(dump_dir, rank, weight_layer, "router_weight", device)
    routing_bias = _load(dump_dir, rank, weight_layer, "routing_bias", device)
    router_logits = torch.matmul(
        hidden_states.float(), router_weight.float().transpose(0, 1)
    ).contiguous()

    weights = prepare_glm5_fused_moe_weights(
        _load(dump_dir, rank, weight_layer, "shared_gate_up_weight_org", device),
        _load(
            dump_dir,
            rank,
            weight_layer,
            "shared_gate_up_weight_scale_org",
            device,
        ),
        _load(dump_dir, rank, weight_layer, "routed_w3_w1_weight", device),
        _load(
            dump_dir,
            rank,
            weight_layer,
            "routed_w3_w1_weight_scaling_factor",
            device,
        ),
        _load(dump_dir, rank, weight_layer, "routed_w2_weight", device),
        _load(
            dump_dir,
            rank,
            weight_layer,
            "routed_w2_weight_scaling_factor",
            device,
        ),
        _load(dump_dir, rank, weight_layer, "shared_down_weight_org", device),
        _load(
            dump_dir,
            rank,
            weight_layer,
            "shared_down_weight_scale_org",
            device,
        ),
    )
    workspace = alloc_glm5_fused_moe_workspace(
        args.tokens, weights.shared_down_weight.shape[1], device
    )
    output = torch.empty_like(hidden_states)

    def run() -> torch.Tensor:
        return glm5_fused_moe(
            hidden_states,
            router_logits,
            routing_bias,
            **weights.as_kwargs(),
            out=output,
            workspace=workspace,
            packed_weight_stages=args.packed_weight_stages,
            use_tma=not args.no_tma,
        )

    times_us = _profile(run, args.warmup, args.iterations)
    print(
        f"rank={rank} tokens={args.tokens} mean_us={statistics.mean(times_us):.3f} "
        f"median_us={statistics.median(times_us):.3f} min_us={min(times_us):.3f} "
        f"stages={args.packed_weight_stages} tma={int(not args.no_tma)}"
    )


if __name__ == "__main__":
    main()
