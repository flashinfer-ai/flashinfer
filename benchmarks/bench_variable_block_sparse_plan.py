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

"""Measure host planning and plan+run latency; run on both comparison revisions."""

import argparse
import json
import statistics
import time

import torch

from flashinfer.sparse import VariableBlockSparseAttentionWrapper


def make_case(heads, rows, cols, seq_len, density, group_size=4, head_dim=64):
    torch.manual_seed(42)
    row_sizes = torch.full((heads, rows), seq_len // rows, dtype=torch.int32)
    row_sizes[:, -1] += seq_len % rows
    col_sizes = torch.full((heads, cols), seq_len // cols, dtype=torch.int32)
    col_sizes[:, -1] += seq_len % cols
    mask = torch.rand(heads, rows, cols, device="cuda") < density
    mask[:, :, 0] = True
    plan_args = (
        mask,
        row_sizes.cuda(),
        col_sizes.cuda(),
        heads * group_size,
        heads,
        head_dim,
    )
    q = torch.randn(
        heads * group_size, seq_len, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    return plan_args, (q, k, torch.randn_like(k))


def measure(wrapper, plan_args, run_args, repeat):
    stream = torch.cuda.current_stream()
    for _ in range(3):
        wrapper.plan(*plan_args)
        wrapper.run(*run_args)
    stream.synchronize()
    plan_times, total_times = [], []
    for _ in range(repeat):
        start = time.perf_counter()
        wrapper.plan(*plan_args)
        stream.synchronize()
        plan_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        wrapper.plan(*plan_args)
        wrapper.run(*run_args)
        stream.synchronize()
        total_times.append((time.perf_counter() - start) * 1000)
    return {
        "plan_ms": statistics.median(plan_times),
        "plan_run_ms": statistics.median(total_times),
        "nnz": wrapper._paged_kv_indices_buf.numel(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    plan_args, run_args = make_case(
        args.heads, args.rows, args.cols, args.seq_len, args.density
    )
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="fa2",
    )
    result = measure(wrapper, plan_args, run_args, args.repeat)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                **vars(args),
                **result,
            },
            indent=2,
        )
    )
