#!/usr/bin/env python3
import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


TRTLLM_GEN_MLA_BATCH_SPLIT_LUT_ENV = "FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TRTLLM-GEN MLA decode batch-split LUT by benchmarking "
            "multi-launch splits."
        )
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSON LUT path"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Largest batch size to benchmark; defaults to the device SM count",
    )
    parser.add_argument(
        "--seq-len", type=int, default=20480, help="Maximum KV sequence length"
    )
    parser.add_argument(
        "--fixed-seq-len",
        action="store_true",
        default=True,
        help="Use the same KV sequence length for every request",
    )
    parser.add_argument(
        "--var-seq-len",
        action="store_false",
        dest="fixed_seq_len",
        help="Use random per-request KV sequence lengths up to seq-len",
    )
    parser.add_argument(
        "--q-len-per-request", type=int, default=2, help="Query length per request"
    )
    parser.add_argument("--page-size", type=int, default=64, choices=(32, 64))
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=("bfloat16", "float8_e4m3fn")
    )
    parser.add_argument("--num-q-heads", type=int, default=128)
    parser.add_argument("--qk-nope-head-dim", type=int, default=128)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.02,
        help="Only keep splits at least this much faster than baseline",
    )
    parser.add_argument(
        "--max-num-splits",
        type=int,
        default=3,
        help="Maximum number of sub-batches to benchmark per batch size",
    )
    return parser.parse_args()


def torch_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def make_benchmark_case(batch_size, args, device, dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + batch_size)

    head_dim = args.kv_lora_rank + args.qk_rope_head_dim
    query = torch.randn(
        batch_size,
        args.q_len_per_request,
        args.num_q_heads,
        head_dim,
        device=device,
        generator=generator,
    ).to(dtype)

    if args.fixed_seq_len:
        seq_lens = torch.full(
            (batch_size,), args.seq_len, dtype=torch.int32, device=device
        )
    else:
        seq_lens = torch.randint(
            1,
            args.seq_len + 1,
            (batch_size,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        seq_lens[-1] = args.seq_len
    max_seq_len = int(seq_lens.max().item())

    blocks_per_seq = (seq_lens + args.page_size - 1) // args.page_size
    max_num_blocks_per_seq = int(blocks_per_seq.max().item())
    total_blocks_needed = int(blocks_per_seq.sum().item())
    block_ids = torch.randperm(total_blocks_needed, device=device, generator=generator)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device
    )

    block_offset = 0
    for batch_idx in range(batch_size):
        num_blocks = int(blocks_per_seq[batch_idx].item())
        block_tables[batch_idx, :num_blocks] = block_ids[
            block_offset : block_offset + num_blocks
        ]
        block_offset += num_blocks

    kv_cache = torch.randn(
        total_blocks_needed,
        1,
        args.page_size,
        head_dim,
        device=device,
        generator=generator,
    ).to(dtype)
    workspace_buffer = torch.zeros(
        args.workspace_mib * 1024 * 1024, dtype=torch.int8, device=device
    )
    out = torch.empty(
        batch_size,
        args.q_len_per_request,
        args.num_q_heads,
        args.kv_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    return query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, out


def run_decode_tensors(
    query,
    kv_cache,
    workspace_buffer,
    block_tables,
    seq_lens,
    max_seq_len,
    out,
    flashinfer_module,
    args,
    start_batch,
    end_batch,
):
    flashinfer_module.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query[start_batch:end_batch],
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=args.qk_nope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        block_tables=block_tables[start_batch:end_batch],
        seq_lens=seq_lens[start_batch:end_batch],
        max_seq_len=max_seq_len,
        out=out[start_batch:end_batch],
        bmm1_scale=1.0 / ((args.qk_nope_head_dim + args.qk_rope_head_dim) ** 0.5),
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )


def run_decode(case, args, start_batch, end_batch):
    import flashinfer as flashinfer_module

    run_decode_tensors(*case, flashinfer_module, args, start_batch, end_batch)


def iter_splits(batch_size, max_num_splits):
    max_num_splits = min(max_num_splits, batch_size)
    for num_splits in range(2, max_num_splits + 1):
        for split_points in itertools.combinations(
            range(1, batch_size), num_splits - 1
        ):
            previous_split_point = 0
            split = []
            for split_point in split_points:
                split.append(split_point - previous_split_point)
                previous_split_point = split_point
            split.append(batch_size - previous_split_point)
            yield tuple(split)


def count_splits(batch_size, max_num_splits):
    max_num_splits = min(max_num_splits, batch_size)
    return sum(
        math.comb(batch_size - 1, num_splits - 1)
        for num_splits in range(2, max_num_splits + 1)
    )


def split_count_breakdown(batch_size, max_num_splits):
    max_num_splits = min(max_num_splits, batch_size)
    return ", ".join(
        f"{num_splits} parts: {math.comb(batch_size - 1, num_splits - 1)}"
        for num_splits in range(2, max_num_splits + 1)
    )


def median_ms(fn, bench_gpu_time_fn, args, input_args):
    measurements = bench_gpu_time_fn(
        fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        enable_cupti=False,
        use_cuda_graph=True,
        input_args=input_args,
        cold_l2_cache=True,
    )
    return float(np.median(measurements))


def calculate_decode_metrics(case, args, execution_time_ms):
    query, _, _, _, seq_lens, _, _ = case
    elem_size = query.element_size()
    actual_kv_tokens = int(seq_lens.sum().item())
    query_bytes = query.numel() * elem_size
    kv_bytes = (
        actual_kv_tokens * (args.kv_lora_rank + args.qk_rope_head_dim) * elem_size
    )
    output_bytes = (
        query.size(0)
        * args.q_len_per_request
        * args.num_q_heads
        * args.kv_lora_rank
        * elem_size
    )
    total_bytes = query_bytes + kv_bytes + output_bytes
    flops = (
        2
        * args.num_q_heads
        * (2 * args.kv_lora_rank + args.qk_rope_head_dim)
        * actual_kv_tokens
        * args.q_len_per_request
    )
    return {
        "actual_kv_tokens": actual_kv_tokens,
        "memory_bandwidth_gbps": total_bytes / execution_time_ms / 1e6,
        "tflops": flops / execution_time_ms / 1e9,
    }


def benchmark_single_batch(
    batch_size,
    max_batch_size,
    flashinfer_module,
    bench_gpu_time_fn,
    args,
    device,
    dtype,
):
    case = make_benchmark_case(batch_size, args, device, dtype)
    print(
        f"[{batch_size}/1..{max_batch_size}] measuring single-launch latency",
        flush=True,
    )
    single_launch_ms = median_ms(
        run_decode_tensors,
        bench_gpu_time_fn,
        args,
        (*case, flashinfer_module, args, 0, batch_size),
    )
    metrics = calculate_decode_metrics(case, args, single_launch_ms)
    print(
        f"[{batch_size}/1..{max_batch_size}] execution time: {single_launch_ms:.4f} ms, "
        f"memory bandwidth: {metrics['memory_bandwidth_gbps']:.2f} GB/s, "
        f"FLOPs: {metrics['tflops']:.2f} TFLOPs/s",
        flush=True,
    )
    return {"batch_size": batch_size, "single_launch_ms": single_launch_ms, **metrics}


def find_best_split_from_table(
    batch_size, max_batch_size, args, single_launch_ms_by_batch
):
    baseline_ms = single_launch_ms_by_batch[batch_size]
    best_split = None
    best_split_ms = baseline_ms
    split_count = count_splits(batch_size, args.max_num_splits)
    print(
        f"[{batch_size}/1..{max_batch_size}] searching {split_count} "
        f"split candidates up to {args.max_num_splits} parts "
        f"({split_count_breakdown(batch_size, args.max_num_splits)}, "
        f"baseline={baseline_ms:.4f}ms) from measured single-launch table",
        flush=True,
    )
    for split_idx, split in enumerate(
        iter_splits(batch_size, args.max_num_splits), start=1
    ):
        split_ms = sum(
            single_launch_ms_by_batch[split_batch_size] for split_batch_size in split
        )
        if split_ms < best_split_ms:
            best_split_ms = split_ms
            best_split = split
            split_label = "+".join(str(value) for value in split)
            speedup = baseline_ms / split_ms
            print(
                f"[{batch_size}/1..{max_batch_size}] found new best "
                f"split {split_label} at candidate {split_idx}/{split_count}: "
                f"{split_ms:.4f}ms speedup={speedup:.4f}",
                flush=True,
            )

    speedup = baseline_ms / best_split_ms if best_split is not None else 1.0
    if best_split is None or speedup < args.min_speedup:
        best_split = None

    return {
        "batch_size": batch_size,
        "baseline_ms": baseline_ms,
        "best_split": list(best_split) if best_split is not None else None,
        "best_split_ms": best_split_ms,
        "speedup": speedup,
    }


def print_run_config(args, device, sm_count, max_batch_size, dtype):
    config = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "sm_count": sm_count,
        "max_batch_size": max_batch_size,
        "seq_len": args.seq_len,
        "fixed_seq_len": args.fixed_seq_len,
        "q_len_per_request": args.q_len_per_request,
        "page_size": args.page_size,
        "dtype": str(dtype).removeprefix("torch."),
        "num_q_heads": args.num_q_heads,
        "qk_nope_head_dim": args.qk_nope_head_dim,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "kv_lora_rank": args.kv_lora_rank,
        "workspace_mib": args.workspace_mib,
        "seed": args.seed,
        "dry_run_iters": args.dry_run_iters,
        "repeat_iters": args.repeat_iters,
        "min_speedup": args.min_speedup,
        "max_num_splits": args.max_num_splits,
        "use_cuda_graph": True,
    }
    print("TRTLLM-GEN MLA batch-split LUT generation config:", flush=True)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)


def main():
    args = parse_args()
    os.environ.pop(TRTLLM_GEN_MLA_BATCH_SPLIT_LUT_ENV, None)
    if args.max_num_splits < 2:
        raise ValueError("--max-num-splits must be at least 2")

    import flashinfer as flashinfer_module
    from flashinfer.testing.utils import bench_gpu_time as bench_gpu_time_fn

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to generate a TRTLLM-GEN MLA batch-split LUT"
        )

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    max_batch_size = args.max_batch_size or sm_count
    dtype = torch_dtype(args.dtype)
    print_run_config(args, device, sm_count, max_batch_size, dtype)

    output_path = args.output or Path(
        f"trtllm_gen_mla_batch_split_lut_sm{sm_count}.json"
    )
    measurements = []
    single_launch_measurements = []
    single_launch_ms_by_batch = {}
    splits = {}

    print(
        f"Precomputing single-launch latencies for batch sizes 1..{max_batch_size}",
        flush=True,
    )
    for batch_size in range(1, max_batch_size + 1):
        single_launch_result = benchmark_single_batch(
            batch_size,
            max_batch_size,
            flashinfer_module,
            bench_gpu_time_fn,
            args,
            device,
            dtype,
        )
        single_launch_ms = single_launch_result["single_launch_ms"]
        single_launch_ms_by_batch[batch_size] = single_launch_ms
        single_launch_measurements.append(single_launch_result)

    print(
        f"Searching LUT splits for batch sizes 1..{max_batch_size}",
        flush=True,
    )
    for batch_size in range(1, max_batch_size + 1):
        result = find_best_split_from_table(
            batch_size, max_batch_size, args, single_launch_ms_by_batch
        )
        measurements.append(result)
        if result["best_split"] is not None:
            splits[str(batch_size)] = result["best_split"]
            split_label = "+".join(str(value) for value in result["best_split"])
        else:
            split_label = "none"
        print(
            f"batch_size={batch_size:4d} baseline={result['baseline_ms']:.4f}ms "
            f"best={result['best_split_ms']:.4f}ms speedup={result['speedup']:.4f} "
            f"split={split_label}",
            flush=True,
        )

    lut = {
        "format_version": 1,
        "backend": "trtllm-gen",
        "kernel": "mla_decode",
        "sm_count": sm_count,
        "device_name": torch.cuda.get_device_name(device),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "max_batch_size": max_batch_size,
            "seq_len": args.seq_len,
            "fixed_seq_len": args.fixed_seq_len,
            "q_len_per_request": args.q_len_per_request,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "num_q_heads": args.num_q_heads,
            "qk_nope_head_dim": args.qk_nope_head_dim,
            "qk_rope_head_dim": args.qk_rope_head_dim,
            "kv_lora_rank": args.kv_lora_rank,
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
            "min_speedup": args.min_speedup,
            "max_num_splits": args.max_num_splits,
            "use_cuda_graph": True,
        },
        "splits": splits,
    }
    lut["single_launch_measurements"] = single_launch_measurements
    lut["measurements"] = measurements

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(lut, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(splits)} splits to {output_path}")


if __name__ == "__main__":
    main()
