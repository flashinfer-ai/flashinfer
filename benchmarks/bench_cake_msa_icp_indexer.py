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

Benchmark of the experimental Cake ICP indexer chain (SM100 / SM103).

Run under ``torchrun`` with one process per rank of the group::

    torchrun --nproc_per_node 2 benchmarks/bench_cake_msa_icp_indexer.py --rows tp2_q4_b8_ctx8k tp2_q4_b64_ctx150k
    torchrun --nproc_per_node 4 benchmarks/bench_cake_msa_icp_indexer.py --world 4

Every rank prepares one decode step of its own fragment, checks the merged
output of the first launch against the CPU oracle and times the chain with
CUPTI (cold L2, fixed iteration counts so the ranks stay in step).  Rank 0
prints the per-rank medians and their maximum per row.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from flashinfer.msa_ops import (
    create_msa_icp_indexer_workspace,
    destroy_msa_icp_indexer_workspace,
    prepare_msa_icp_indexer,
)
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.test_helpers.cake_msa_icp_inputs import (
    build_case,
    build_rank_inputs,
    case_params,
    expected_out,
    local_topk_lists,
)

# name -> (world, query_len, batch, context tokens, block-table columns, page size)
ROWS = {
    "tp2_q4_b1_ctx8k": (2, 4, 1, 8192, 8192, 128),
    "tp2_q4_b8_ctx8k": (2, 4, 8, 8192, 8192, 128),
    "tp2_q4_b8_ctx150k": (2, 4, 8, 153600, 8192, 128),
    "tp2_q4_b16_ctx150k": (2, 4, 16, 153600, 8192, 128),
    "tp2_q4_b64_ctx8k": (2, 4, 64, 8192, 8192, 128),
    "tp2_q4_b64_ctx150k": (2, 4, 64, 153600, 8192, 128),
    "tp2_q1_b8_ctx150k": (2, 1, 8, 153600, 8192, 128),
    "tp2_q1_b64_ctx8k": (2, 1, 64, 8192, 8192, 128),
    "tp4_q4_b8_ctx150k": (4, 4, 8, 153600, 8192, 128),
    "tp4_q4_b64_ctx8k": (4, 4, 64, 8192, 8192, 128),
    "tp2_p256_q4_b8_ctx150k": (2, 4, 8, 153600, 8192, 256),
}


def median_us(fn, *, dry_run_iters: int, repeat_iters: int) -> float:
    times = bench_gpu_time_with_cupti(
        fn, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters, cold_l2_cache=True
    )
    return statistics.median(times) * 1e3


def bench_row(
    name,
    world,
    query_len,
    batch,
    ctx,
    columns,
    page,
    *,
    rank,
    device,
    seed,
    check,
    iters,
):
    params = case_params(
        world_size=world,
        query_len=query_len,
        batch=batch,
        context_tokens=ctx,
        score_columns=columns,
        page_size=page,
        seed=seed,
    )
    case = build_case(name, params)
    inp = build_rank_inputs(case, rank, device)
    workspace = create_msa_icp_indexer_workspace(
        world_size=world, rank=rank, tokens_cap=case.tokens, device=device
    )
    runner = prepare_msa_icp_indexer(
        inp.index_q,
        inp.index_k,
        inp.block_table,
        inp.query_start_loc,
        inp.seq_lens,
        inp.positions,
        inp.active_rows_u8,
        inp.forced_column,
        inp.forced_block,
        inp.n_ordinary,
        workspace=workspace,
        query_len=case.query_len,
    )
    runner()
    torch.cuda.synchronize(device)
    status = workspace.status_value()
    exact = None
    if check:
        lists = local_topk_lists(case, inp)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, lists)
        exact = bool(
            torch.equal(runner.out.cpu(), expected_out(case, gathered, inp, rank))
        )
    dist.barrier()
    us = median_us(runner, dry_run_iters=iters[0], repeat_iters=iters[1])
    torch.cuda.synchronize(device)
    status_after = workspace.status_value()
    per_rank: list = [None] * world
    dist.all_gather_object(
        per_rank,
        dict(rank=rank, us=us, status=status, status_after=status_after, exact=exact),
    )
    dist.barrier()
    destroy_msa_icp_indexer_workspace(workspace)
    plan = runner.plan.as_dict()
    return dict(
        row=name,
        world=world,
        query_len=query_len,
        batch=batch,
        context_tokens=ctx,
        columns=columns,
        page_size=page,
        split_count=plan["split_count"],
        merge_warps=plan["merge_warps"],
        merge_entries_per_thread=plan["merge_entries_per_thread"],
        pool_stride=plan["pool_stride"],
        per_rank=per_rank,
        max_rank_us=max(r["us"] for r in per_rank),
        all_exact=all(r["exact"] in (None, True) for r in per_rank),
        all_status_zero=all(
            r["status"] == 0 and r["status_after"] == 0 for r in per_rank
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument(
        "--world",
        type=int,
        default=None,
        help="only rows of this world size (default: the launch's)",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=1132)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the oracle check of the first launch",
    )
    parser.add_argument(
        "--iters", type=int, nargs=2, default=(10, 50), metavar=("WARMUP", "REPEAT")
    )
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    want_world = args.world or world
    if want_world != world:
        raise SystemExit(f"--world {want_world} but the launch has {world} ranks")
    rows = [
        (n, *spec)
        for n, spec in ROWS.items()
        if spec[0] == world and (not args.rows or n in set(args.rows))
    ]
    results = []
    if rank == 0:
        print(
            f"{'row':26s} {'split':>5s} {'w/epl':>6s} "
            + " ".join(f"{'r' + str(r) + '_us':>9s}" for r in range(world))
            + f" {'max_us':>9s} exact status"
        )
    for row in rows:
        record = bench_row(
            *row,
            rank=rank,
            device=device,
            seed=args.seed,
            check=not args.no_check,
            iters=tuple(args.iters),
        )
        if rank == 0:
            per = " ".join(
                f"{r['us']:9.2f}"
                for r in sorted(record["per_rank"], key=lambda r: r["rank"])
            )
            print(
                f"{record['row']:26s} {record['split_count']:5d} "
                f"{record['merge_warps']:>2d}/{record['merge_entries_per_thread']:<3d} {per} "
                f"{record['max_rank_us']:9.2f} {str(record['all_exact']):5s} {'ok' if record['all_status_zero'] else 'BAD'}"
            )
        results.append(record)
        torch.cuda.empty_cache()
    if args.json and rank == 0:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    device=torch.cuda.get_device_name(device), world=world, rows=results
                ),
                indent=1,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
