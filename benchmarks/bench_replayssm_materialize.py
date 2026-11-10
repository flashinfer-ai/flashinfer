# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure ReplaySSM prefix materialization with a persistent launch grid.

The materializer uses a fixed CTA pool which grid-strides over the logical
``(request, layer, head)`` grid. The caller supplies a compact active-request
map, so only mapped requests execute replay while preserving their physical
batch indices. This benchmark constructs that map directly.

The default shape is the proposed Nemotron-v4 full-model shape:
``batch=256, layers=48, heads=256, dim=64, dstate=128``.  It is 3,145,728
logical work items. Cache storage is deliberately allocated only for active requests:
inactive requests return before dereferencing a cache pointer, so this is
equivalent to the full-grid no-op path without requiring a production-sized
state pool.

Examples:

  # Pure device kernel time; CUPTI is preferred and falls back if unavailable.
  python benchmarks/bench_replayssm_materialize.py --active-requests 0 1

  # A smaller exploratory sweep.
  python benchmarks/bench_replayssm_materialize.py --batch 32 --layers 48 \
      --heads 256 --active-requests 0 1 2

  # Override the distributed default positions for the first two active requests.
  python benchmarks/bench_replayssm_materialize.py --active-requests 0 1 2 \
      --active-request-indices 17 191

  # Full active batch without allocating per-layer state. Outputs intentionally
  # race and are invalid; this is a launch-cost experiment only.
  python benchmarks/bench_replayssm_materialize.py --active-requests 256 \
      --alias-layers

  # CUDA-graph replay timing after the materializer has been JIT compiled.
  python benchmarks/bench_replayssm_materialize.py --timing graph \
      --active-requests 0 1
"""

from __future__ import annotations

import argparse
import statistics

import torch

from flashinfer.mamba.replayssm_materialize import replayssm_materialize
from flashinfer.testing import bench_gpu_time


def _pointer_table(tensor: torch.Tensor, logical_layers: int) -> torch.Tensor:
    """Return a CUDA int64 table of one base pointer per layer."""
    if tensor.size(0) == 1:
        return torch.full(
            (logical_layers,), tensor.data_ptr(), device="cuda", dtype=torch.int64
        )
    return torch.tensor(
        [tensor[layer].data_ptr() for layer in range(tensor.size(0))],
        device="cuda",
        dtype=torch.int64,
    )


def _stride_table(tensor: torch.Tensor, logical_layers: int) -> torch.Tensor:
    return torch.full(
        (logical_layers,), tensor.stride(1), device="cuda", dtype=torch.int64
    )


def _active_indices(
    batch: int, active_requests: int, requested_indices: list[int] | None
) -> list[int]:
    if active_requests == 0:
        return []
    if requested_indices is None:
        # Spread flushes across the decode batch. This deliberately avoids
        # rewarding an implementation that happens to front- or back-load work.
        return [
            (2 * i + 1) * batch // (2 * active_requests) for i in range(active_requests)
        ]
    if len(requested_indices) < active_requests:
        raise ValueError("active-request-indices needs one entry per active request")
    indices = requested_indices[:active_requests]
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= batch for index in indices
    ):
        raise ValueError("active-request-indices must be unique indices in [0, batch)")
    return indices


def _build_call(
    *,
    batch: int,
    layers: int,
    heads: int,
    dim: int,
    dstate: int,
    heads_per_group: int,
    max_window: int,
    replay_prefix_len: int,
    active_requests: int,
    requested_indices: list[int] | None,
    alias_layers: bool,
):
    if not 0 <= active_requests <= batch:
        raise ValueError("active_requests must be in [0, batch]")
    if not 0 <= replay_prefix_len <= max_window:
        raise ValueError("replay_prefix_len must be in [0, max_window]")
    if heads % heads_per_group:
        raise ValueError("heads must be divisible by heads_per_group")
    active_indices = _active_indices(batch, active_requests, requested_indices)

    # One source slot plus one unique destination per active request. Requests
    # outside the active-map prefix return before any cache or state access.
    state_slots = max(1, active_requests + 1)
    ring_buffer_len = max_window + 1
    groups = heads // heads_per_group
    storage_layers = 1 if alias_layers else layers

    state = torch.empty(
        storage_layers,
        state_slots,
        heads,
        dim,
        dstate,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_cache = torch.empty(
        storage_layers,
        1,
        heads,
        ring_buffer_len,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b_cache = torch.empty(
        storage_layers,
        1,
        groups,
        ring_buffer_len,
        dstate,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dt_cache = torch.empty(
        storage_layers, 1, heads, ring_buffer_len, device="cuda", dtype=torch.float32
    )
    A = -torch.ones(storage_layers, heads, device="cuda", dtype=torch.float32)

    src_slots = torch.zeros((layers, batch), device="cuda", dtype=torch.int32)
    dst_slots = torch.zeros_like(src_slots)
    if active_requests:
        active_indices_tensor = torch.tensor(
            active_indices, device="cuda", dtype=torch.int64
        )
        destinations = torch.arange(
            1, active_requests + 1, device="cuda", dtype=torch.int32
        )
        dst_slots[:, active_indices_tensor] = destinations.unsqueeze(0)
    ring_start = torch.zeros(batch, device="cuda", dtype=torch.int32)
    counts = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
    if active_requests:
        counts[active_indices_tensor] = replay_prefix_len
    active_request_indices = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
    if active_requests:
        active_request_indices[:active_requests] = torch.tensor(
            active_indices, device="cuda", dtype=torch.int32
        )

    state_ptrs = _pointer_table(state, layers)
    state_slot_strides = _stride_table(state, layers)
    x_cache_ptrs = _pointer_table(x_cache, layers)
    x_cache_slot_strides = _stride_table(x_cache, layers)
    b_cache_ptrs = _pointer_table(b_cache, layers)
    b_cache_slot_strides = _stride_table(b_cache, layers)
    dt_cache_ptrs = _pointer_table(dt_cache, layers)
    dt_cache_slot_strides = _stride_table(dt_cache, layers)
    A_ptrs = _pointer_table(A, layers)
    zero_table = torch.zeros(layers, device="cuda", dtype=torch.int64)

    # The device pointer table holds raw addresses, not Tensor ownership.  Keep
    # every backing allocation alive for CUDA-graph replay; eager execution can
    # otherwise appear to work until the caching allocator reuses an address.
    def run(_keepalive=(state, x_cache, b_cache, dt_cache, A)) -> None:
        replayssm_materialize(
            state_ptrs,
            state_slot_strides,
            x_cache_ptrs,
            x_cache_slot_strides,
            b_cache_ptrs,
            b_cache_slot_strides,
            dt_cache_ptrs,
            dt_cache_slot_strides,
            A_ptrs,
            zero_table,
            zero_table,
            src_slots,
            dst_slots,
            ring_start,
            counts,
            active_request_indices,
            state_dtype=torch.bfloat16,
            input_dtype=torch.bfloat16,
            matrixA_dtype=torch.float32,
            dim=dim,
            dstate=dstate,
            num_heads=heads,
            heads_per_group=heads_per_group,
            max_window=max_window,
            ring_buffer_len=ring_buffer_len,
        )

    # Retain all backing tensors and GPU-resident launch tables in the callable.
    return run, active_indices


def _timing_kwargs(args: argparse.Namespace) -> dict[str, object]:
    common: dict[str, object] = {
        "dry_run_iters": args.warmup,
        "repeat_iters": args.iters,
        # The empty-map path reads only 1 KiB of active-map entries. Cold-L2 flushing
        # measures an artificial cache-flush kernel, not the steady-state
        # graph/launch behavior under study.
        "cold_l2_cache": False,
    }
    if args.timing == "cupti":
        common["enable_cupti"] = True
    elif args.timing == "graph":
        common["use_cuda_graph"] = True
        common["num_iters_within_graph"] = args.graph_iters
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--heads", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--heads-per-group", type=int, default=1)
    parser.add_argument("--max-window", type=int, default=16)
    parser.add_argument("--replay-prefix-len", type=int, default=1)
    parser.add_argument(
        "--alias-layers",
        action="store_true",
        help="reuse one physical layer for all layer pointers; output is invalid",
    )
    parser.add_argument(
        "--active-requests",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="number of requests with non-negative replay prefix length (default: 0 1 2)",
    )
    parser.add_argument(
        "--active-request-indices",
        type=int,
        nargs="+",
        help=(
            "request positions for active flushes; each sweep entry takes the first N "
            "positions, otherwise they are spread through the batch"
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-iters", type=int, default=10)
    parser.add_argument(
        "--timing",
        choices=("cupti", "event", "graph"),
        default="cupti",
        help="CUPTI kernel timing (with event fallback), CUDA events, or graph replay",
    )
    args = parser.parse_args()

    work_items = args.batch * args.layers * args.heads
    print(
        f"ReplaySSM materialize: B={args.batch}, L={args.layers}, H={args.heads}, "
        f"logical_grid={work_items:,} items, timing={args.timing}, "
        f"alias_layers={args.alias_layers}"
    )
    for active_requests in args.active_requests:
        run, active_indices = _build_call(
            batch=args.batch,
            layers=args.layers,
            heads=args.heads,
            dim=args.dim,
            dstate=args.dstate,
            heads_per_group=args.heads_per_group,
            max_window=args.max_window,
            replay_prefix_len=args.replay_prefix_len,
            active_requests=active_requests,
            requested_indices=args.active_request_indices,
            alias_layers=args.alias_layers,
        )
        measurements = bench_gpu_time(run, **_timing_kwargs(args))
        median_us = statistics.median(measurements) * 1000
        stdev_us = statistics.pstdev(measurements) * 1000
        print(
            f"active_requests={active_requests:3d}, "
            f"indices={active_indices}, "
            f"replay_prefix_len={args.replay_prefix_len if active_requests else -1:2d}: "
            f"median {median_us:9.2f} us, std {stdev_us:7.2f} us"
        )


if __name__ == "__main__":
    main()
