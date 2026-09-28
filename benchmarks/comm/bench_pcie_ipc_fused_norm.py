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

"""Fused AllReduce+residual+RMSNorm against NCCL plus a separate RMSNorm.

The baseline is what the caller runs today for the same result: an all-reduce
followed by ``fused_add_rmsnorm``. Comparing a fused kernel against a bare
all-reduce would measure the norm's absence, not the fusion.

    sudo nvidia-smi -lgc 2520,2520   # nothing below pins clocks

    for n in 2 4 8; do
        torchrun --standalone --nproc_per_node=$n \\
            benchmarks/comm/bench_pcie_ipc_fused_norm.py --json fused_tp$n.json
    done

Options:
    --hidden N        Hidden size (default 6144)
    --batches a,b,c   Batch sizes to sweep
    --tune            Measure the launch configuration first, then benchmark the
                      winner. Without it the seed is used, which is what an
                      untuned deployment runs.
    --variant V       Force a transport (STAGED, STAGED_RING)
    --blocks N        Force a block count. With --variant, one point of the
                      grid, which is how the seed's block cap was placed.
    --json FILE       Write results to JSON
"""

import argparse
import json
import os
from dataclasses import replace
from typing import List

import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_policy import IpcVariant
from flashinfer.norm import fused_add_rmsnorm
from flashinfer.testing.utils import bench_gpu_time

_EPS = 1e-6
_DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

# Pinned for the same reason as the plain benchmark: left to auto-tune, each
# rank derives its own iteration count and a collective whose ranks disagree on
# how many times to call it deadlocks.
_BENCH_KWARGS = dict(
    use_cuda_graph=True,
    num_iters_within_graph=20,
    dry_run_iters=5,
    repeat_iters=20,
    cold_l2_cache=False,
)


def _group_median_us(samples, device, group) -> float:
    """Median over iterations of the group maximum at each iteration.

    The max has to be taken per sample, before the median: a collective costs
    what its straggler costs, and taking each rank's median first would report a
    number no iteration actually had.
    """
    t = torch.as_tensor(list(samples), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.median()) * 1e3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--batches", type=str, default=None)
    p.add_argument(
        "--tune",
        action="store_true",
        help="tune both ops first and benchmark the winners",
    )
    p.add_argument(
        "--variant",
        choices=[v.name for v in IpcVariant],
        default=None,
        help="force a transport instead of the one the policy picks",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="force a block count; with --variant, sweeps one point of the grid",
    )
    p.add_argument("--json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    batches = (
        [int(b) for b in args.batches.split(",")] if args.batches else _DEFAULT_BATCHES
    )
    hidden = args.hidden

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    device = torch.device(f"cuda:{rank}")

    workspace = comm.PcieIpcAllReduceWorkspace(
        group=group,
        max_numel=hidden * max(batches),
        dtype=torch.bfloat16,
        tune_batches=tuple(batches),
    )
    shared = torch.Generator(device=device).manual_seed(0)
    gamma = torch.randn(hidden, dtype=torch.bfloat16, device=device, generator=shared)

    if args.tune:
        # Collective, and slow: it launches every candidate for every bucket.
        workspace.tune([hidden], dtype=torch.bfloat16)

    if rank == 0:
        print(f"world_size={world_size} hidden={hidden} dtype=bfloat16")
        print(f"profile={workspace.profile} ({workspace.profile_reason})")
        print(f"config source: {'tuned' if args.tune else 'seed'}")
        print(f"{'batch':>7} {'fused':>10} {'nccl+norm':>10} {'speedup':>9}  config")

    results: List[dict] = []
    for batch in batches:
        shape = (batch, hidden)
        inp = torch.randn(shape, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            shape, dtype=torch.bfloat16, device=device, generator=shared
        )
        # Resolved once, outside the timed region, and passed explicitly: the
        # lookup takes the autotuner's global lock, which is real overhead at
        # this operator's scale. Collective on every rank, not just rank 0.
        config = (
            workspace.tuned_fused_launch_config(inp)
            if args.tune
            else workspace.fused_launch_config(inp)
        )
        if config is None:
            continue
        # Overrides last, so an unsupported combination is rejected by the
        # launcher rather than silently rounded to something the policy likes.
        if args.variant is not None:
            config = replace(config, variant=IpcVariant[args.variant])
        if args.blocks is not None:
            config = replace(config, blocks=args.blocks)
        norm_out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp)

        def fused():
            workspace.all_reduce_fused_add_rms_norm(
                inp,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=_EPS,
                residual_out=residual_out,
                norm_out=norm_out,
                config=config,
            )

        # The baseline mutates its inputs, so it gets its own buffers; both
        # arrangements read and write the same number of bytes.
        base_x = inp.clone()
        base_residual = residual.clone()

        def unfused():
            dist.all_reduce(base_x, group=group)
            fused_add_rmsnorm(base_x, base_residual, gamma, _EPS)

        # bench_gpu_time warms up on a side stream before capturing a graph.
        # That is sequential, not concurrent, so tell the workspace it may move
        # its binding -- after making sure the previous stream is drained.
        torch.cuda.synchronize()
        workspace.rebind_stream()
        fused_us = _group_median_us(
            bench_gpu_time(fused, **_BENCH_KWARGS), device, group
        )
        unfused_us = _group_median_us(
            bench_gpu_time(unfused, **_BENCH_KWARGS), device, group
        )
        results.append(
            {
                "batch": batch,
                "hidden": hidden,
                "world_size": world_size,
                "fused_us": fused_us,
                "nccl_plus_norm_us": unfused_us,
                "speedup": unfused_us / fused_us,
                "blocks": config.blocks,
                "threads": config.threads,
                "transport": config.variant.name,
                "tuned": bool(args.tune),
            }
        )
        if rank == 0:
            print(
                f"{batch:>7} {fused_us:>10.2f} {unfused_us:>10.2f} "
                f"{unfused_us / fused_us:>8.2f}x  "
                f"{config.variant.name} blocks={config.blocks} "
                f"threads={config.threads}"
            )

    if args.json and rank == 0:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")

    workspace.destroy()
    dist.destroy_process_group(group)


if __name__ == "__main__":
    main()
