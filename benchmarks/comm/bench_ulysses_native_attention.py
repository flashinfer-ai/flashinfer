# SPDX-License-Identifier: Apache-2.0
"""Complete QKV -> Ulysses -> native FlashInfer attention -> local O example.

PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=2 \
  benchmarks/comm/bench_ulysses_native_attention.py --attention sm120-sage \
  --sequence 512 --used 497 --heads 8 --schedule 2,2

BF16 communication for all modes. Quantization, casts, layout conversions and
output copies in run() are timed. Planning/compilation
is excluded. No external FA4 patch or external Sage package is required.
"""

import argparse
from datetime import timedelta
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.comm.ulysses_native_attention import (
    BACKENDS,
    NativeAttention,
    validate_backend,
)
from flashinfer.comm import UlyssesCommunicator


class NativePipeline:
    """Example-local, serial-request pipeline with separate chunk outputs."""

    def __init__(self, args, cin, cout):
        self.cin, self.cout, self.args = cin, cout, args
        self.world = cin.world_size
        self.local_heads = args.heads // self.world
        rows = args.sequence // self.world
        self.schedule = tuple(int(x) for x in args.schedule.split(","))
        self.wi, self.wo = cin.create_workspace(), cout.create_workspace()
        self.input_stream, self.output_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self.input_ready = [torch.cuda.Event() for _ in self.schedule]
        self.compute_ready = [torch.cuda.Event() for _ in self.schedule]
        self.output = torch.empty(
            1, rows, args.heads, 128, device="cuda", dtype=torch.bfloat16
        )
        self.global_qkv = [
            torch.empty(
                1,
                args.sequence,
                self.local_heads,
                128,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        ]
        self.whole_payload = torch.empty(
            1, args.sequence, self.local_heads, 384, device="cuda", dtype=torch.bfloat16
        )
        self.payloads = [
            torch.empty(
                1, args.sequence, count, 384, device="cuda", dtype=torch.bfloat16
            )
            for count in self.schedule
        ]

        def attention(count):
            return NativeAttention(
                args.attention,
                sequence=args.sequence,
                used=args.used,
                heads=count,
                fp8_scale=args.fp8_scale,
            )

        self.whole = attention(self.local_heads)
        self.chunks = [attention(count) for count in self.schedule]
        # JIT/setup must finish on every rank before entering timed collectives.
        for op in (self.whole, *self.chunks):
            sample = torch.zeros(
                1, args.sequence, op.heads, 128, device="cuda", dtype=torch.bfloat16
            )
            op(sample, sample, sample)
        torch.cuda.synchronize()

    def run(self, mode, q, k, v):
        if mode == "ordinary":
            for src, dst in zip((q, k, v), self.global_qkv, strict=True):
                self.cin.scatter_heads(src, out=dst, workspace=self.wi)
            self.cout.gather_heads(
                self.whole(*self.global_qkv), out=self.output, workspace=self.wo
            )
        elif mode == "whole_fused":
            self.cin.scatter_qkv_head_chunk(
                q,
                k,
                v,
                head_offset=0,
                head_count=self.local_heads,
                out=self.whole_payload,
                workspace=self.wi,
            )
            result = self.whole(*self.whole_payload.split(128, -1))
            self.cout.gather_output_head_chunk(
                result,
                local_heads=self.local_heads,
                head_offset=0,
                out=self.output,
                workspace=self.wo,
            )
        elif mode in ("chunk_serial", "chunk_overlap"):
            caller = torch.cuda.current_stream()
            overlap = mode == "chunk_overlap"
            self.input_stream.wait_stream(caller)
            self.output_stream.wait_stream(caller)
            offset = 0
            for i, count in enumerate(self.schedule):
                ins = self.input_stream if overlap else caller
                outs = self.output_stream if overlap else caller
                with torch.cuda.stream(ins):
                    self.cin.scatter_qkv_head_chunk(
                        q,
                        k,
                        v,
                        head_offset=offset,
                        head_count=count,
                        out=self.payloads[i],
                        workspace=self.wi,
                    )
                    self.input_ready[i].record()
                caller.wait_event(self.input_ready[i])
                result = self.chunks[i](*self.payloads[i].split(128, -1))
                self.compute_ready[i].record()
                with torch.cuda.stream(outs):
                    outs.wait_event(self.compute_ready[i])
                    self.cout.gather_output_head_chunk(
                        result,
                        local_heads=self.local_heads,
                        head_offset=offset,
                        out=self.output,
                        workspace=self.wo,
                    )
                offset += count
            caller.wait_stream(self.output_stream)
        else:
            raise ValueError(f"unknown mode {mode}")
        # All persistent chunk outputs remain alive and cannot be overwritten
        # by a subsequent request until this caller/output-stream join finishes.
        return self.output


def agree_error(error, stage):
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, error)
    if any(reports):
        raise RuntimeError(f"{stage} failed across ranks: {reports}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", choices=BACKENDS, required=True)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument(
        "--used", type=int, default=0, help="0 means the full physical sequence"
    )
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--schedule", default="", help="positive bands summing to H/U")
    parser.add_argument(
        "--fp8-scale",
        type=float,
        default=1 / 32,
        help="fixed Q/K/V dequantization scale; not per-head calibration",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    world, rank = dist.get_world_size(), dist.get_rank()
    error = None
    try:
        args.used = args.used or args.sequence
        if (
            args.sequence % world
            or args.heads % world
            or args.iterations <= 0
            or args.warmup < 0
        ):
            raise ValueError(
                "S/H must divide U; positive iterations, nonnegative warmup required"
            )
        validate_backend(
            args.attention,
            torch.cuda.get_device_capability(),
            args.sequence,
            args.used,
            args.heads,
            128,
            args.fp8_scale,
        )
        hlocal = args.heads // world
        if not args.schedule:
            args.schedule = (
                f"{hlocal // 2},{hlocal - hlocal // 2}" if hlocal > 1 else "1"
            )
        schedule = tuple(int(x) for x in args.schedule.split(","))
        if min(schedule) <= 0 or sum(schedule) != hlocal:
            raise ValueError("schedule must partition H/U")
    except Exception as exc:
        error = str(exc)
    agree_error(error, "configuration")
    configs = [None] * world
    dist.all_gather_object(configs, vars(args))
    if any(c != configs[0] for c in configs):
        raise ValueError("all ranks must use identical arguments")
    gin, gout = dist.new_group(), dist.new_group()
    capacity = args.sequence * (args.heads // world) * 128
    cin = UlyssesCommunicator(
        group=gin, backend="nccl", max_elems=3 * capacity, dtype=torch.bfloat16
    )
    cout = UlyssesCommunicator(
        group=gout, backend="nccl", max_elems=capacity, dtype=torch.bfloat16
    )
    pipeline, error = None, None
    try:
        pipeline = NativePipeline(args, cin, cout)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree_error(error, "local native kernel preparation")
    torch.manual_seed(321 + rank)
    qkv = [
        torch.randn(
            1,
            args.sequence // world,
            args.heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    modes = ("ordinary", "whole_fused", "chunk_serial", "chunk_overlap")
    expected = pipeline.run("ordinary", *qkv).clone()
    # Independent reference, outside timing, on the same head-owned prefix.
    ref = torch.zeros_like(pipeline.global_qkv[0])
    ref[:, : args.used] = F.scaled_dot_product_attention(
        *(t[:, : args.used].transpose(1, 2) for t in pipeline.global_qkv)
    ).transpose(1, 2)
    ref = cout.gather_heads(ref)
    quality = (
        expected.float() - ref.float()
    ).square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-12)
    dist.all_reduce(quality, op=dist.ReduceOp.MAX)
    error = None
    try:
        tol = 0.01 if args.attention.endswith("bf16") else 0.06
        torch.testing.assert_close(expected, ref, atol=tol, rtol=tol)
    except Exception as exc:
        error = str(exc)
    agree_error(error, "native kernel vs BF16 SDPA reference")
    for mode in modes:
        actual = pipeline.run(mode, *qkv)
        error = None
        try:
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
        except Exception as exc:
            error = str(exc)
        agree_error(error, f"{mode} vs native ordinary reference")
    for _ in range(args.warmup):
        for mode in modes:
            pipeline.run(mode, *qkv)
    torch.cuda.synchronize()
    samples = {mode: [] for mode in modes}
    for i in range(args.iterations):
        order = modes[i % len(modes) :] + modes[: i % len(modes)]
        for mode in order:
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            pipeline.run(mode, *qkv)
            end.record()
            end.synchronize()
            latency = torch.tensor(start.elapsed_time(end), device="cuda")
            dist.all_reduce(latency, op=dist.ReduceOp.MAX)
            samples[mode].append(latency.item())
    if rank == 0:
        medians = {k: statistics.median(v) for k, v in samples.items()}
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "world_size": world,
                    "distributed_transport_exercised": world > 1,
                    "config": vars(args),
                    "wire_dtype": "bfloat16",
                    "scope": "QKV -> input A2A -> native quantization/layout/attention -> output A2A -> local O",
                    "correctness_passed": True,
                    "max_rank_relative_rmse_vs_bf16": quality.item(),
                    "median_ms": medians,
                    "speedup_vs_ordinary": {
                        k: medians["ordinary"] / v for k, v in medians.items()
                    },
                    "samples_ms": samples,
                },
                indent=2,
            )
        )
    cin.close()
    cout.close()
    dist.destroy_process_group(gin)
    dist.destroy_process_group(gout)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
