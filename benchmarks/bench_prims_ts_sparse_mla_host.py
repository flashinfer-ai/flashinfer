# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure sparse MLA CPU launch and completed eager wall time.

Run this same script with PYTHONPATH selecting the baseline and candidate
checkouts. GPU work is included in completed_wall_us, but excluded from the
post-loop synchronization in cpu_submit_us. Neither metric is pure kernel time.
"""

import argparse
import json
import statistics
import time

import torch

from flashinfer.attention.prims_ts import mla_decode as mla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    query = torch.randn(
        args.rows, 1, args.heads, 512, device=device, dtype=torch.bfloat16
    )
    pool = torch.randn(args.topk * 2, 1, 512, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        args.topk * 2, (args.rows, args.topk), device=device, dtype=torch.int32
    )
    lengths = torch.full((args.rows,), args.topk, device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    wrapper = mla.BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(device, args.rows, args.heads, max_topk=args.topk)

    def run():
        wrapper.run(query, pool, indices, lengths, out=output, validate=False)

    for _ in range(20):
        run()
    submit, wall, plan = [], [], []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            run()
        submitted = time.perf_counter()
        torch.cuda.synchronize()
        done = time.perf_counter()
        submit.append((submitted - start) * 1e6 / args.iterations)
        wall.append((done - start) * 1e6 / args.iterations)
        start = time.perf_counter()
        for _ in range(args.iterations):
            mla.get_prims_ts_sparse_mla_decode_workspace_size(
                device, args.rows, args.heads, max_topk=args.topk
            )
        plan.append((time.perf_counter() - start) * 1e6 / args.iterations)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                module=mla.__file__,
                rows=args.rows,
                heads=args.heads,
                topk=args.topk,
                cpu_submit_us=statistics.median(submit),
                completed_wall_us=statistics.median(wall),
                workspace_query_us=statistics.median(plan),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
