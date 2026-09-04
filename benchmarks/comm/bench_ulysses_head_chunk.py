"""Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference and benchmark for head-chunked Ulysses primitives.

This is intentionally a benchmark-local scheduler, not a FlashInfer runtime
API. It demonstrates how a framework can use two process groups,
communicators, side streams, and the head-chunk transport primitives.

Examples (MiniMax H3 T2V physical attention shape):

    torchrun --standalone --nproc_per_node=4 \
      benchmarks/comm/bench_ulysses_head_chunk.py \
      --global-seq 37760 --heads 56 --head-dim 128 --schedule 2,10,2

    torchrun --standalone --nproc_per_node=2 \
      benchmarks/comm/bench_ulysses_head_chunk.py \
      --global-seq 37760 --heads 56 --head-dim 128 --schedule 7,14,7

The reported time is the maximum rank latency for each iteration. ``ordinary``
uses three input scatter collectives and one output gather. ``whole_fused``
uses one fused-QKV input collective and one output collective. The two chunked
variants use the same schedule either serially or with the reference three-
stream pipeline. Attention is PyTorch SDPA and treats the full physical
sequence as dense; model integrations remain responsible for varlen metadata.
"""

import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flashinfer.comm import UlyssesCommunicator


def _parse_schedule(text: str, local_heads: int):
    if text:
        schedule = tuple(int(item) for item in text.split(","))
    elif local_heads == 14:
        schedule = (2, 10, 2)
    elif local_heads == 28:
        schedule = (7, 14, 7)
    else:
        left = max(1, local_heads // 4)
        middle = local_heads - 2 * left
        schedule = (left, middle, left) if middle > 0 else (local_heads,)
    if not schedule or any(item <= 0 for item in schedule):
        raise ValueError(f"schedule must contain positive head counts: {schedule}")
    if sum(schedule) != local_heads:
        raise ValueError(
            f"schedule {schedule} sums to {sum(schedule)}, expected "
            f"local_heads={local_heads}"
        )
    return schedule


def _attention(q, k, v):
    # BSHD -> BHSD -> BSHD. The explicit contiguous output gives every
    # transport variant the same attention-output layout.
    result = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    )
    return result.permute(0, 2, 1, 3).contiguous()


def _attention_from_fused(payload, head_dim):
    q, k, v = payload.split(head_dim, dim=-1)
    return _attention(q, k, v)


class ReferenceHeadChunkPipeline:
    """Benchmark-only input-A2A/attention/output-A2A stream pipeline."""

    def __init__(
        self,
        *,
        input_comm,
        output_comm,
        batch,
        local_seq,
        heads,
        head_dim,
        schedule,
        dtype,
        device,
    ):
        self.input_comm = input_comm
        self.output_comm = output_comm
        self.batch = batch
        self.local_seq = local_seq
        self.heads = heads
        self.local_heads = heads // input_comm.world_size
        self.head_dim = head_dim
        self.schedule = schedule
        self.device = device
        self.input_stream = torch.cuda.Stream(device=device)
        self.output_stream = torch.cuda.Stream(device=device)
        self.input_ready = [torch.cuda.Event() for _ in schedule]
        self.compute_ready = [torch.cuda.Event() for _ in schedule]

        max_chunk = max(schedule)
        input_capacity = (
            3 * batch * local_seq * input_comm.world_size * max_chunk * head_dim
        )
        output_capacity = (
            batch * local_seq * output_comm.world_size * max_chunk * head_dim
        )
        self.input_workspace = input_comm.create_workspace(max_elems=input_capacity)
        self.output_workspace = output_comm.create_workspace(max_elems=output_capacity)
        global_seq = local_seq * input_comm.world_size
        self.qkv_chunks = [
            torch.empty(
                batch,
                global_seq,
                head_count,
                3 * head_dim,
                dtype=dtype,
                device=device,
            )
            for head_count in schedule
        ]
        self.output = torch.empty(
            batch, local_seq, heads, head_dim, dtype=dtype, device=device
        )

    def sequential(self, q, k, v):
        offset = 0
        for index, head_count in enumerate(self.schedule):
            payload = self.input_comm.scatter_qkv_head_chunk(
                q,
                k,
                v,
                head_offset=offset,
                head_count=head_count,
                out=self.qkv_chunks[index],
                workspace=self.input_workspace,
            )
            attention_out = _attention_from_fused(payload, self.head_dim)
            self.output_comm.gather_output_head_chunk(
                attention_out,
                local_heads=self.local_heads,
                head_offset=offset,
                out=self.output,
                workspace=self.output_workspace,
            )
            offset += head_count
        return self.output

    def overlap(self, q, k, v):
        compute_stream = torch.cuda.current_stream(self.device)
        self.input_stream.wait_stream(compute_stream)
        outputs = []
        offset = 0
        for index, head_count in enumerate(self.schedule):
            with torch.cuda.stream(self.input_stream):
                self.input_comm.scatter_qkv_head_chunk(
                    q,
                    k,
                    v,
                    head_offset=offset,
                    head_count=head_count,
                    out=self.qkv_chunks[index],
                    workspace=self.input_workspace,
                )
                self.input_ready[index].record(self.input_stream)

            compute_stream.wait_event(self.input_ready[index])
            attention_out = _attention_from_fused(self.qkv_chunks[index], self.head_dim)
            self.compute_ready[index].record(compute_stream)
            outputs.append(attention_out)

            with torch.cuda.stream(self.output_stream):
                self.output_stream.wait_event(self.compute_ready[index])
                self.output_comm.gather_output_head_chunk(
                    attention_out,
                    local_heads=self.local_heads,
                    head_offset=offset,
                    out=self.output,
                    workspace=self.output_workspace,
                )
                attention_out.record_stream(self.output_stream)
            offset += head_count

        compute_stream.wait_stream(self.output_stream)
        # Keep SDPA outputs alive until their cross-stream consumers have been
        # enqueued and joined to the caller stream.
        del outputs
        return self.output


def _rank_max_elapsed_ms(function, warmup, iterations, device):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            [start.elapsed_time(end)], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))
    return samples


def _summarize(samples):
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[p90_index],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, default=37760)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--schedule", type=str, default="")
    parser.add_argument("--backend", choices=("auto", "nccl", "nvlink"), default="nccl")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.global_seq % world_size != 0:
        raise ValueError("global sequence length must be divisible by world size")
    if args.heads % world_size != 0:
        raise ValueError("head count must be divisible by world size")
    local_seq = args.global_seq // world_size
    local_heads = args.heads // world_size
    schedule = _parse_schedule(args.schedule, local_heads)

    # Both groups contain the same ranks but own independent NCCL
    # communicators, allowing input and output directions to overlap.
    ranks = list(range(world_size))
    input_group = dist.new_group(ranks=ranks, backend="nccl")
    output_group = dist.new_group(ranks=ranks, backend="nccl")
    dtype = torch.bfloat16
    whole_input_elems = 3 * args.batch * local_seq * args.heads * args.head_dim
    whole_output_elems = args.batch * local_seq * args.heads * args.head_dim
    input_comm = UlyssesCommunicator(
        input_group,
        max_elems=whole_input_elems,
        dtype=dtype,
        backend=args.backend,
        device=device,
    )
    output_comm = UlyssesCommunicator(
        output_group,
        max_elems=whole_output_elems,
        dtype=dtype,
        backend=args.backend,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    q = torch.randn(
        args.batch,
        local_seq,
        args.heads,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k = torch.randn(q.shape, dtype=dtype, device=device, generator=generator)
    v = torch.randn(q.shape, dtype=dtype, device=device, generator=generator)

    input_workspace = input_comm.create_workspace()
    output_workspace = output_comm.create_workspace()
    q_global = torch.empty(
        args.batch,
        args.global_seq,
        local_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    k_global = torch.empty_like(q_global)
    v_global = torch.empty_like(q_global)
    ordinary_output = torch.empty_like(q)
    whole_payload = torch.empty(
        args.batch,
        args.global_seq,
        local_heads,
        3 * args.head_dim,
        dtype=dtype,
        device=device,
    )
    whole_output = torch.empty_like(q)

    def ordinary():
        input_comm.scatter_heads(q, out=q_global, workspace=input_workspace)
        input_comm.scatter_heads(k, out=k_global, workspace=input_workspace)
        input_comm.scatter_heads(v, out=v_global, workspace=input_workspace)
        attention_out = _attention(q_global, k_global, v_global)
        output_comm.gather_heads(
            attention_out, out=ordinary_output, workspace=output_workspace
        )
        return ordinary_output

    def whole_fused():
        input_comm.scatter_qkv_head_chunk(
            q,
            k,
            v,
            head_offset=0,
            head_count=local_heads,
            out=whole_payload,
            workspace=input_workspace,
        )
        attention_out = _attention_from_fused(whole_payload, args.head_dim)
        output_comm.gather_output_head_chunk(
            attention_out,
            local_heads=local_heads,
            head_offset=0,
            out=whole_output,
            workspace=output_workspace,
        )
        return whole_output

    pipeline = ReferenceHeadChunkPipeline(
        input_comm=input_comm,
        output_comm=output_comm,
        batch=args.batch,
        local_seq=local_seq,
        heads=args.heads,
        head_dim=args.head_dim,
        schedule=schedule,
        dtype=dtype,
        device=device,
    )

    # Compile kernels and verify the complete mapping before timing.
    reference = ordinary()
    fused_result = whole_fused()
    sequential_result = pipeline.sequential(q, k, v)
    overlap_result = pipeline.overlap(q, k, v)
    torch.cuda.synchronize(device)
    for name, result in (
        ("whole_fused", fused_result),
        ("chunk_sequential", sequential_result),
        ("chunk_overlap", overlap_result),
    ):
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
        if rank == 0:
            max_abs = float((result - reference).abs().max().item())
            print(f"correctness {name}: max_abs={max_abs:.6g}")

    variants = (
        ("ordinary", ordinary),
        ("whole_fused", whole_fused),
        ("chunk_sequential", lambda: pipeline.sequential(q, k, v)),
        ("chunk_overlap", lambda: pipeline.overlap(q, k, v)),
    )
    results = {}
    for name, function in variants:
        dist.barrier()
        samples = _rank_max_elapsed_ms(function, args.warmup, args.iters, device)
        results[name] = _summarize(samples)
        if rank == 0:
            print(
                f"{name:>18}: median={results[name]['median_ms']:.3f} ms "
                f"p90={results[name]['p90_ms']:.3f} ms"
            )

    baseline = results["ordinary"]["median_ms"]
    for value in results.values():
        value["speedup_vs_ordinary"] = baseline / value["median_ms"]
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "backend": input_comm.backend,
        "shape": {
            "batch": args.batch,
            "global_seq": args.global_seq,
            "local_seq": local_seq,
            "heads": args.heads,
            "local_heads": local_heads,
            "head_dim": args.head_dim,
        },
        "schedule": schedule,
        "warmup": args.warmup,
        "iterations": args.iters,
        "results": results,
    }
    if rank == 0:
        print(json.dumps(record, indent=2))
        if args.json:
            with open(args.json, "w") as output_file:
                json.dump(record, output_file, indent=2)

    input_comm.close()
    output_comm.close()
    dist.barrier()
    dist.destroy_process_group(input_group)
    dist.destroy_process_group(output_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
