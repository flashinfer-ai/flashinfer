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

"""Warm paged cuDNN submission and whole-graph replay, including metadata work.

Run in separate parent/candidate processes with the same installed dependencies:
    python benchmarks/bench_cudnn_paged_prepare.py --kv-length 512
    python benchmarks/bench_cudnn_paged_prepare.py --kv-length 8192

Compilation and warmup are excluded. Neither metric is isolated kernel time or
serving latency. The backend version is checked after torch imports because an
already-loaded cuDNN library can override the intended library search path.
"""

import argparse
import json
import statistics
import time

import torch

import flashinfer
from flashinfer.cudnn import prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-length", type=int, choices=[512, 8192], default=512)
    parser.add_argument("--expect-cudnn", type=int, default=92400)
    args = parser.parse_args()
    cudnn = prefill.cudnn
    actual_version = None if cudnn is None else cudnn.backend_version()
    if actual_version != args.expect_cudnn:
        raise RuntimeError(f"expected cuDNN {args.expect_cudnn}, got {actual_version}")
    torch.manual_seed(171)
    nq, nk, hq, hk, dim, page = 128, args.kv_length, 16, 2, 128, 16
    q = torch.randn(nq, hq, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(nk // page, 2, hk, page, dim, device="cuda", dtype=q.dtype)
    k, v = kv[:, 0], kv[:, 1]
    indices = torch.arange(nk // page, device="cuda", dtype=torch.int32)
    workspace = torch.empty(256 << 20, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cudnn"
    )
    wrapper.plan(
        torch.tensor([0, nq], dtype=torch.int32),
        torch.tensor([0, nk // page], dtype=torch.int32),
        indices,
        torch.tensor([page], dtype=torch.int32),
        hq,
        hk,
        dim,
        page,
        causal=True,
        q_data_type=q.dtype,
        seq_lens=torch.tensor([nk], dtype=torch.int32),
        block_tables=indices.view(1, -1),
        max_token_per_sequence=nq,
        max_sequence_kv=nk,
    )
    out = torch.empty_like(q)
    lse = torch.empty(nq, hq, device="cuda", dtype=torch.float32)

    def run():
        return wrapper.run(q, (k, v), out=out, lse=lse, return_lse=True, lse_base="ln")

    keys = k.transpose(1, 2).reshape(nk, hk, dim).double()
    values = v.transpose(1, 2).reshape(nk, hk, dim).double()
    ref = torch.empty_like(q, dtype=torch.float64)
    ref_lse = torch.empty_like(lse, dtype=torch.float64)
    mask = torch.arange(nk, device="cuda")[None, :] > (
        torch.arange(nq, device="cuda")[:, None] + nk - nq
    )
    for head in range(hq):
        scores = q[:, head].double() @ keys[:, head // (hq // hk)].T * dim**-0.5
        scores.masked_fill_(mask, -torch.inf)
        ref[:, head] = scores.softmax(-1) @ values[:, head // (hq // hk)]
        ref_lse[:, head] = scores.logsumexp(-1)

    def check():
        torch.testing.assert_close(out.float(), ref.float(), atol=0.02, rtol=0.02)
        torch.testing.assert_close(lse, ref_lse.float(), atol=0.002, rtol=0.002)

    run()
    check()
    for _ in range(100):
        run()
    builds = prefill._prefill_graph_builds
    host_us = []
    for _ in range(11):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(32):
            run()
        host_us.append((time.perf_counter_ns() - start) / 32_000)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    check()
    for _ in range(100):
        graph.replay()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    replay_us = []
    for _ in range(11):
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(100):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        replay_us.append(start_event.elapsed_time(end_event) * 10)
    assert prefill._prefill_graph_builds == builds
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "frontend": cudnn.__version__,
                "backend_version": cudnn.backend_version(),
                "flashinfer_source": flashinfer.__file__,
                "cudnn_source": cudnn.__file__,
                "q_length": nq,
                "kv_length": nk,
                "query_heads": hq,
                "kv_heads": hk,
                "head_dim": dim,
                "prepared": wrapper._cudnn_plan is not None,
                "graph_builds": builds,
                "host_submit_median_us": statistics.median(host_us),
                "graph_replay_median_us": statistics.median(replay_us),
                "host_submit_samples_us": host_us,
                "graph_replay_samples_us": replay_us,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
