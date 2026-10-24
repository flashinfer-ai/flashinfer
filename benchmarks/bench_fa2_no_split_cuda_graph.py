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

"""Run unchanged in separate, exact-revision FlashInfer environments.

Example: python benchmarks/bench_fa2_no_split_cuda_graph.py --batch 2 --kv-len 8192 --dtype bfloat16
Repeat with --fast-plan and --cold-l2; use distinct FLASHINFER_WORKSPACE_BASE
directories for main and candidate. Prints measurements, never assumed gains.
"""
import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
import flashinfer
from flashinfer.testing import bench_gpu_time_with_cudagraph
from flashinfer.testing.utils import calculate_rotation_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--qo-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--fast-plan", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    args = parser.parse_args()
    assert args.kv_len >= args.query_len >= 1
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    b, length, t = args.batch, args.kv_len, args.query_len
    hq, hk, d = args.qo_heads, args.kv_heads, args.head_dim
    page = args.page_size
    pages = (length + page - 1) // page
    indptr = torch.arange(b + 1, dtype=torch.int32, device="cuda") * pages
    indices = torch.arange(b * pages, dtype=torch.int32, device="cuda")
    last = torch.full((b,), (length - 1) % page + 1, dtype=torch.int32, device="cuda")
    q = torch.randn(b * t, hq, d, dtype=dtype, device="cuda")
    kv = torch.randn(b * pages, 2, page, hk, d, dtype=dtype, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        use_tensor_cores=True,
        backend="fa2",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr.clone(),
        paged_kv_indices_buffer=indices.clone(),
        paged_kv_last_page_len_buffer=last.clone(),
    )
    plan_args = (indptr, indices, last, hq, hk, d, page)
    plan_kwargs = dict(
        q_data_type=dtype, kv_data_type=dtype, disable_split_kv=True, q_len_per_req=t
    )
    float_bytes, int_bytes = wrapper.workspace_size(*plan_args, **plan_kwargs)
    wrapper.reset_workspace_buffer(
        torch.empty(float_bytes, dtype=torch.uint8, device="cuda"),
        torch.empty(int_bytes, dtype=torch.uint8, device="cuda"),
    )
    wrapper._pin_memory_int_workspace_buffer.fill_(0xA5)
    wrapper.plan(*plan_args, **plan_kwargs)
    if args.fast_plan:
        torch.cuda.current_stream().synchronize()
        wrapper._pin_memory_int_workspace_buffer.fill_(0xA5)
        flashinfer.fast_decode_plan(
            wrapper, *plan_args, **plan_kwargs, global_override_indptr_cpu=indptr.cpu()
        )
    out = wrapper.run(q, kv)
    k, v = kv.reshape(b, pages, 2, page, hk, d).unbind(2)
    k = k.reshape(b, pages * page, hk, d)[:, :length]
    v = v.reshape(b, pages * page, hk, d)[:, :length]
    k = k.permute(0, 2, 1, 3).repeat_interleave(hq // hk, dim=1)
    v = v.permute(0, 2, 1, 3).repeat_interleave(hq // hk, dim=1)
    query = q.reshape(b, t, hq, d).permute(0, 2, 1, 3)
    mask = torch.arange(length, device="cuda")[None, :] <= (
        length - t + torch.arange(t, device="cuda")[:, None]
    )
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            query.float(),
            k.float(),
            v.float(),
            attn_mask=mask,
        )
        .permute(0, 2, 1, 3)
        .reshape_as(out)
    )
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    # A graph must actually visit every rotated buffer. Capturing only 64
    # calls can otherwise leave tiny inputs resident in L2 on every replay.
    rotations = calculate_rotation_count([q, kv, out], q.device) if args.cold_l2 else 1
    graph_iters = math.ceil(64 / rotations) * rotations
    repeat_iters = max(5, math.ceil(6400 / graph_iters))
    trials = []
    for _ in range(5):
        timings = bench_gpu_time_with_cudagraph(
            wrapper.run,
            input_args=(q, kv),
            input_kwargs={"out": out},
            cold_l2_cache=args.cold_l2,
            dry_run_iters=2,
            repeat_iters=repeat_iters,
            num_iters_within_graph=graph_iters,
        )
        trials.append(statistics.median(timings) * 1000)
    root = Path(flashinfer.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        json.dumps(
            dict(
                revision=revision,
                source=str(root),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                config=vars(args),
                grid_x=wrapper._plan_info[0],
                float_bytes=float_bytes,
                int_bytes=int_bytes,
                rotations=rotations,
                graph_iters=graph_iters,
                repeat_iters=repeat_iters,
                max_abs_error=(out.float() - ref).abs().max().item(),
                trial_medians_us=trials,
                median_us=statistics.median(trials),
            )
        )
    )


if __name__ == "__main__":
    main()
