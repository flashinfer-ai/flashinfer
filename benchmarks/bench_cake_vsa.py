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

# Benchmark the Cake SM100/SM103 block-sparse attention backend.
#
# Every row plans a ``BlockSparseAttentionWrapper(backend="cake")`` once and
# times ``run()`` with CUPTI-correlated GPU time (cold L2 per iteration):
#
#     python benchmarks/bench_cake_vsa.py [--rows all|tests|deployment] [--json out.json]

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time


@dataclass(frozen=True)
class Row:
    block_size: int
    dtype: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    M: int
    N: int
    selected: int
    return_lse: bool = False

    @property
    def label(self) -> str:
        return (
            f"blk{self.block_size}_{self.dtype}_h{self.num_qo_heads}x{self.num_kv_heads}"
            f"_d{self.head_dim}_m{self.M}_n{self.N}_k{self.selected}"
            + ("_lse" if self.return_lse else "")
        )


TEST_ROWS = (
    Row(128, "bfloat16", 8, 8, 128, 256, 512, 2, True),
    Row(64, "bfloat16", 8, 8, 128, 128, 256, 2, True),
    Row(128, "float16", 8, 1, 128, 256, 512, 2),
    Row(128, "float16", 8, 8, 128, 256, 512, 2, True),
    Row(128, "bfloat16", 8, 2, 128, 256, 512, 2),
    Row(128, "bfloat16", 8, 8, 64, 256, 512, 2),
    Row(128, "bfloat16", 8, 8, 96, 256, 512, 2),
    Row(128, "bfloat16", 8, 8, 128, 128, 16384, 8),
    Row(128, "float16", 8, 2, 128, 512, 1024, 3, True),
)

DEPLOYMENT_ROWS = (
    Row(128, "bfloat16", 8, 8, 128, 4096, 4096, 8),
    Row(128, "bfloat16", 8, 8, 128, 8192, 32768, 16),
    Row(128, "bfloat16", 8, 8, 128, 80000, 80000, 6),
    Row(64, "bfloat16", 8, 8, 128, 2048, 4096, 16),
    Row(64, "bfloat16", 8, 8, 128, 2048, 4096, 28),
    Row(128, "float16", 8, 2, 128, 4096, 4096, 8),
    Row(128, "float16", 8, 8, 128, 4096, 4096, 8),
    Row(128, "bfloat16", 8, 8, 64, 4096, 4096, 8),
    Row(128, "bfloat16", 8, 8, 96, 4096, 4096, 8),
    Row(128, "bfloat16", 8, 2, 128, 4096, 4096, 8),
)


def strided_mask(row: Row, device: torch.device) -> torch.Tensor:
    mb, nb = row.M // row.block_size, row.N // row.block_size
    mask = torch.zeros((row.num_qo_heads, mb, nb), dtype=torch.bool, device=device)
    for block_row in range(mb):
        columns = (torch.arange(row.selected, device=device) * 7 + block_row) % nb
        mask[:, block_row, columns] = True
    return mask


def make_inputs(row: Row, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    dtype = getattr(torch, row.dtype)
    q = torch.randn((row.M, row.num_qo_heads, row.head_dim), dtype=dtype, device=device)
    k = torch.randn((row.N, row.num_kv_heads, row.head_dim), dtype=dtype, device=device)
    v = torch.randn((row.N, row.num_kv_heads, row.head_dim), dtype=dtype, device=device)
    return q, k, v, strided_mask(row, device)


def plan_wrapper(row: Row, mask: torch.Tensor, workspace: torch.Tensor):
    dtype = getattr(torch, row.dtype)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    wrapper.plan(
        None,
        None,
        row.M,
        row.N,
        row.block_size,
        row.block_size,
        row.num_qo_heads,
        row.num_kv_heads,
        row.head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
        block_mask=mask,
    )
    return wrapper


def bench_row(row: Row, device: torch.device, workspace: torch.Tensor) -> float:
    q, k, v, mask = make_inputs(row, device)
    wrapper = plan_wrapper(row, mask, workspace)
    wrapper.run(q, k, v, return_lse=row.return_lse)
    torch.cuda.synchronize()
    times = bench_gpu_time(
        lambda: wrapper.run(q, k, v, return_lse=row.return_lse),
        enable_cupti=True,
        cold_l2_cache=True,
    )
    return float(statistics.median(list(times)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", choices=("all", "tests", "deployment"), default="all")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    rows = {
        "tests": TEST_ROWS,
        "deployment": DEPLOYMENT_ROWS,
        "all": TEST_ROWS + DEPLOYMENT_ROWS,
    }[args.rows]
    device = torch.device("cuda")
    workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=device)
    results = []
    print("| row | median_ms |\n|---|---|")
    for row in rows:
        median_ms = bench_row(row, device, workspace)
        results.append({**asdict(row), "label": row.label, "median_ms": median_ms})
        print(f"| {row.label} | {median_ms:.4f} |", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
