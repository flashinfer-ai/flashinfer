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

"""Offline transport-table tuner of the Cake SM120 MegaMoE backend.

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim::

    torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tune --dtype sm120_mxfp4_mxfp8 \\
        --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 \\
        --max-tokens 512 2048 8192 [--shared-expert]

For every row capacity it times each transport table the launched topology can serve
(``all_gin``; ``lsa_where_reachable``; LSA only inside rank pairs ``(2i, 2i+1)``;
``all_lsa``) on the same synthetic problem (random MXFP4 weights, Gaussian activations,
softmax top-k routing, ``--live-tokens`` rows per rank), takes the maximum over ranks of
the per-rank median layer-forward time, and records the fastest table in the moe_ep knob
cache (:mod:`.cake_knob_cache`).  ``transport_table="tuned"`` then resolves it at workspace
creation; the per-call path never times or probes anything.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

import torch


def _init_dist() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl", device_id=torch.cuda.current_device())
    return dist.get_rank(), dist.get_world_size()


def candidate_tables(
    world: int, pair_ok: Any, gin_available: bool
) -> list[tuple[str, Any]]:
    """Distinct feasible ``(name, transport_table spec)`` candidates of a topology."""

    from .cake_transport import (
        TRANSPORT_GIN,
        TRANSPORT_LSA,
        lsa_where_reachable,
        normalize_pair_ok,
        table_from_pairs,
        uniform_table,
        validate_table,
    )

    pairs = normalize_pair_ok(world, pair_ok)

    raw = [
        uniform_table(world, TRANSPORT_GIN),
        lsa_where_reachable(world, pair_ok),
        table_from_pairs(
            world,
            lambda a, b: TRANSPORT_LSA
            if a // 2 == b // 2 and pairs[(a, b)]
            else TRANSPORT_GIN,
            "lsa_pairs",
        ),
        uniform_table(world, TRANSPORT_LSA),
    ]
    out: list[tuple[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for table in raw:
        try:
            validate_table(table, pair_ok, gin_available)
        except ValueError:
            continue
        if table.cells in seen:
            continue
        seen.add(table.cells)
        spec = table.name if table.name in ("all_gin", "all_lsa") else table.cells
        out.append((table.name, spec))
    return out


def _time_layer(layer: Any, inputs: Any, warmup: int, iters: int) -> float:
    """Per-rank median microseconds of ``layer(inputs)`` (CUDA events)."""

    for _ in range(warmup):
        layer(inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        layer(inputs)
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1e3)
    return statistics.median(samples)


def tune_one(args: Any, rank: int, world: int, max_tokens: int) -> dict[str, Any]:
    import torch.distributed as dist

    from .cake_geometry import kernel_max_rows
    from .cake_knob_cache import record_knobs, topology_digest
    from .cake_synthetic import Geometry, make_layer, random_inputs, random_weights
    from .cake_transport import gather_lsa_pair_matrix

    shared = bool(getattr(args, "shared_expert", False))
    geometry = Geometry(args.hidden, args.intermediate, args.num_experts, args.topk)
    live = args.live_tokens if args.live_tokens is not None else max_tokens
    if live > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    weights = random_weights(rank, world, geometry, shared=shared, seed=args.seed)
    inputs = random_inputs(live, geometry, seed=args.seed + 1 + rank)

    def build(spec: Any) -> Any:
        return make_layer(
            rank,
            world,
            weights,
            geometry,
            max_tokens_per_rank=max_tokens,
            shared=shared,
            transport_table=spec,
        )

    layer = build("all_gin")
    layer.warmup(inputs)  # the workspace (and its transport session) is created lazily
    session = layer._workspace.session
    pair_ok = gather_lsa_pair_matrix(session.lsa_reachable, dist.group.WORLD)
    gin_available = bool(session.gin_available)
    layer.destroy()
    dist.barrier()
    candidates = candidate_tables(world, pair_ok, gin_available)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    timed: dict[str, tuple[float, tuple[int, ...]]] = {}
    for name, spec in candidates:
        layer = build(spec)
        local = _time_layer(layer, inputs, args.warmup_iters, args.timed_iters)
        worst = torch.tensor([local], dtype=torch.float64, device="cuda")
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        timed[name] = (float(worst.item()), layer._workspace.transport_table.cells)
        layer.destroy()
        dist.barrier()
        if rank == 0:
            print(
                f"[moe_ep-tune] sm120 cake max_tokens={max_tokens} live={live} "
                f"{name}: {timed[name][0]:.1f} us",
                flush=True,
            )
    best = min(timed, key=lambda n: timed[n][0])
    knobs = {"transport_table": list(timed[best][1]), "table_name": best}
    if rank == 0:
        path = record_knobs(
            knobs,
            max_tokens=kernel_max_rows(max_tokens),
            p50_us=timed[best][0],
            candidates_us={n: round(t, 1) for n, (t, _) in timed.items()},
            device=None,
            variant="fp8shared" if shared else "noshared",
            world_size=world,
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_experts=args.num_experts,
            topk=args.topk,
            topology=topology_digest(pair_ok, gin_available),
        )
        print(
            f"[moe_ep-tune] recorded {best} for max_tokens={max_tokens} -> {path or 'DISABLED'}",
            flush=True,
        )
    return knobs


def run_tuning(args: Any) -> int:
    import torch.distributed as dist

    rank, world = _init_dist()
    try:
        for max_tokens in args.max_tokens:
            tune_one(args, rank, world, max_tokens)
        torch.cuda.synchronize()
    finally:
        dist.barrier()
        dist.destroy_process_group()
    return 0


__all__ = ["candidate_tables", "run_tuning", "tune_one"]
