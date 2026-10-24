#!/usr/bin/env python3
"""Real GPU scheduling comparison for FlashInfer CUDA-core paged decode.

Requires Python 3.10+, both libraries installed in the SAME environment, and an
SM90 GPU for the default dual-library mode. No simulated scheduling is used.
--ported-hpc instead compares native FlashInfer vs the original parallel planner
port INSIDE FlashInfer. It runs on B200/SM100a without installing original HPC.

Source layouts checked:
  HPC-Ops 1cd332980ed46bd0172091c1c35d55338fcae47a
  FlashInfer 388e7a311e9f79941134dcc298cbc942b5c607a1
The diagnostic decoders use private layouts; they fail on incompatible versions.

Examples:
  python compare_dynamic_decode.py --self-test
  python compare_dynamic_decode.py --timing events --json-out result.json
  python compare_dynamic_decode.py --flashinfer-only --cases small --iters 5
  python compare_dynamic_decode.py --ported-hpc --iters 30 --json-out result.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

PAGE = 64
DIM = 128
KV_HEADS = 8
Q_HEADS = 32  # GQA=4; HPC-Ops supports GQA=4/8 in this path.
TASK_STRIDE = 12  # int32, sizeof(TaskScheduleInfo)=48 bytes


def make_cases(batch: int, short: int, long: int) -> dict[str, list[int]]:
    # Preserve B and total KV tokens between uniform and skewed.
    skewed = [short] * (batch - 1) + [long]
    pages, extra = divmod(sum(skewed) // PAGE, batch)
    uniform = [(pages + (i < extra)) * PAGE for i in range(batch)]
    assert sum(uniform) == sum(skewed)
    return {
        "small": [long],
        "uniform": uniform,
        "skewed": skewed,
        "short_uniform": [short] * batch,
        "medium_uniform": [4096] * batch,
        "mixed": [short] * (batch - max(1, batch // 16)) + [long] * max(1, batch // 16),
    }


def describe_loads(loads: list[int]) -> dict:
    mean = statistics.mean(loads)
    return {
        "count": len(loads),
        "active": sum(x > 0 for x in loads),
        "min_tokens": min(loads),
        "max_tokens": max(loads),
        "mean_tokens": mean,
        "cv": statistics.pstdev(loads) / mean if mean else 0.0,
        "max_over_mean": max(loads) / mean if mean else 0.0,
    }


def parse_hpc_map(raw: list[int], lengths: list[int], heads: int, tile=64) -> dict:
    """Decode original task-map bytes, not a reimplementation of scheduling."""
    stride_rows, ctas, map_heads, max_batch = raw[:4]
    if stride_rows < 2 or ctas < 1 or map_heads != heads or max_batch != len(lengths):
        raise RuntimeError("Unsupported HPC-Ops task-map header/layout")
    chunk_start = TASK_STRIDE * (1 + ctas * stride_rows)
    chunk_counts = raw[chunk_start : chunk_start + heads * max_batch]
    if len(chunk_counts) != heads * max_batch:
        raise RuntimeError("Truncated HPC-Ops task map")
    intervals = {(h, b): [] for h in range(heads) for b in range(len(lengths))}
    loads, tasks_per_cta, tasks = [], [], []
    for cta in range(ctas):
        load = count = 0
        for slot in range(stride_rows):
            start = TASK_STRIDE * (1 + cta * stride_rows + slot)
            row = raw[start : start + TASK_STRIDE]
            if len(row) != TASK_STRIDE:
                raise RuntimeError("Truncated HPC-Ops task record")
            h, b, chunk, offset, size = row[:5]
            if h < 0 or b < 0:
                break
            if not (0 <= h < heads and 0 <= b < len(lengths) and size > 0):
                raise RuntimeError("Unsupported HPC-Ops task record/layout")
            intervals[h, b].append((offset, offset + size))
            tasks.append(
                {
                    "cta": cta,
                    "head": h,
                    "request": b,
                    "chunk": chunk,
                    "start": offset,
                    "tokens": size,
                }
            )
            load += size
            count += 1
        loads.append(load)
        tasks_per_cta.append(count)
    # Check exact coverage and the per-request chunk table, including all heads.
    for (h, b), ranges in intervals.items():
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor:
                raise RuntimeError(f"HPC map gap/overlap at head={h}, request={b}")
            cursor = end
        if cursor != lengths[b] or len(ranges) != chunk_counts[h * max_batch + b]:
            raise RuntimeError("HPC map coverage/chunk-count check failed")
    if sum(loads) != sum(lengths) * heads:
        raise RuntimeError("HPC map total work check failed")
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    return {
        "tile_tokens": tile,
        "capacity_tiles_per_cta": stride_rows - 1,
        "cta_load": describe_loads(loads),
        "tokens_per_cta": loads,
        "tasks_per_cta": tasks_per_cta,
        "multi_request_ctas": sum(
            len({t["request"] for t in tasks if t["cta"] == c}) > 1 for c in range(ctas)
        ),
        "longest_request": longest,
        "longest_chunks_per_kv_head": [
            chunk_counts[h * max_batch + longest] for h in range(heads)
        ],
        "chunks_head_major": chunk_counts,
        "tasks": tasks,
    }


def parse_fi_plan(plan, read_i32, lengths: list[int], heads: int) -> dict:
    """read_i32 reads byte offsets in the real wrapper int workspace."""
    if len(plan) != 10:
        raise RuntimeError("FlashInfer DecodePlanInfo changed; expected 10 fields")
    padded_batch, _, _, req_off, tile_off, ptr_off, _, size_off, graph, split = map(
        int, plan
    )
    if graph:
        raise RuntimeError("This diagnostic expects use_cuda_graph=False")
    if not split:
        counts = [1] * len(lengths)
        loads = [length for length in lengths for _ in range(heads)]
        chunk_size = None  # Not used by the unsplit kernel; do not read stale arrays.
        tasks = [
            {"request": b, "chunk": 0, "start": 0, "tokens": n}
            for b, n in enumerate(lengths)
        ]
    else:
        ptr = read_i32(ptr_off, len(lengths) + 1)
        if ptr[0] != 0 or any(b < a for a, b in zip(ptr, ptr[1:], strict=False)):
            raise RuntimeError("Invalid FlashInfer output indptr")
        n_tasks = ptr[-1]
        if not 0 < n_tasks <= padded_batch:
            raise RuntimeError("Invalid FlashInfer expanded batch size")
        requests = read_i32(req_off, n_tasks)
        tiles = read_i32(tile_off, n_tasks)
        chunk_size = read_i32(size_off, 1)[0]
        if chunk_size <= 0:
            raise RuntimeError("Invalid FlashInfer chunk size")
        counts = [ptr[b + 1] - ptr[b] for b in range(len(lengths))]
        tasks, loads = [], []
        for b in range(len(lengths)):
            cursor = 0
            for j in range(ptr[b], ptr[b + 1]):
                start = tiles[j] * chunk_size
                n = min(chunk_size, lengths[b] - start)
                if requests[j] != b or start != cursor or n <= 0:
                    raise RuntimeError("FlashInfer task coverage/layout check failed")
                tasks.append(
                    {"request": b, "chunk": tiles[j], "start": start, "tokens": n}
                )
                loads.extend([n] * heads)
                cursor += n
            if cursor != lengths[b]:
                raise RuntimeError("FlashInfer task coverage check failed")
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    if sum(loads) != sum(lengths) * heads:
        raise RuntimeError("FlashInfer total work check failed")
    return {
        "split_kv": bool(split),
        "chunk_tokens": chunk_size,
        "expanded_requests": sum(counts),
        "attention_ctas": len(loads),
        "cta_load": describe_loads(loads),
        "tokens_per_attention_cta": loads,
        "longest_request_chunks": counts[longest],
        "chunks_per_request": counts,
        "request_chunk_tasks": tasks,  # Each is replicated across KV heads.
    }


def parse_port_map(raw, lengths, heads):
    ptr, state_ptr, tasks = raw["cta_indptr"], raw["state_indptr"], raw["tasks"]
    intervals = {(h, b): [] for h in range(heads) for b in range(len(lengths))}
    loads, counts = [], []
    for c in range(len(ptr) - 1):
        load = 0
        for b, h, begin, end, state in tasks[ptr[c] : ptr[c + 1]]:
            assert 0 <= b < len(lengths) and 0 <= h < heads
            assert 0 <= begin < end <= lengths[b]
            assert state_ptr[b * heads + h] <= state < state_ptr[b * heads + h + 1]
            intervals[h, b].append((begin, end))
            load += end - begin
        loads.append(load)
        counts.append(ptr[c + 1] - ptr[c])
    assert sorted(t[4] for t in tasks) == list(range(len(tasks)))
    for (_h, b), ranges in intervals.items():
        cursor = 0
        for begin, end in sorted(ranges):
            assert begin == cursor, "Ported task map has a gap or overlap"
            cursor = end
        assert cursor == lengths[b]
    assert sum(loads) == sum(lengths) * heads
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    return {
        "policy": "head-major sequential fixed-capacity flat-bin, GPU-planned",
        "tile_tokens": raw.get("tile_size", 64),
        "capacity_tiles": raw["capacity_tiles"],
        "cta_load": describe_loads(loads),
        "tokens_per_cta": loads,
        "tasks_per_cta": counts,
        "task_count": len(tasks),
        "multi_request_ctas": sum(
            len({t[0] for t in tasks[ptr[c] : ptr[c + 1]]}) > 1
            for c in range(len(ptr) - 1)
        ),
        "longest_chunks_per_kv_head": [
            len(intervals[h, longest]) for h in range(heads)
        ],
        "tasks": tasks,
        "state_indptr": state_ptr,
    }


def wall_us(fn, torch, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1000)
    return statistics.median(samples)


def gpu_us(fn, bench_gpu_time, args) -> float:
    # bench_gpu_time returns a list of milliseconds, NOT (median, stdev).
    samples = bench_gpu_time(
        fn,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        enable_cupti=args.timing == "cupti",
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    return statistics.median(samples) * 1000


def build_inputs(torch, lengths: list[int], seed: int) -> dict:
    torch.manual_seed(seed)
    n_pages = [(n + PAGE - 1) // PAGE for n in lengths]
    ptr = [0]
    for n in n_pages:
        ptr.append(ptr[-1] + n)
    # Same physical page pool and same Q/K/V for BOTH libraries.
    indices_cpu = torch.randperm(ptr[-1], dtype=torch.int64).to(torch.int32)
    block_ids_cpu = torch.zeros((len(lengths), max(n_pages)), dtype=torch.int32)
    for b, n in enumerate(n_pages):
        block_ids_cpu[b, :n] = indices_cpu[ptr[b] : ptr[b + 1]]
    last_page_len = [(n - 1) % PAGE + 1 for n in lengths]
    q = torch.randn((len(lengths), Q_HEADS, DIM), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((ptr[-1], PAGE, KV_HEADS, DIM), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # HPC requires unused last-page values to be zero. Also exercise ragged pages.
    for b, tail in enumerate(last_page_len):
        if tail != PAGE:
            page = int(block_ids_cpu[b, n_pages[b] - 1])
            k[page, tail:].zero_()
            v[page, tail:].zero_()
    return {
        "q": q,
        "k": k,
        "v": v,
        "block_ids": block_ids_cpu.cuda(),
        "block_ids_cpu": block_ids_cpu,
        "lengths": torch.tensor(lengths, device="cuda", dtype=torch.int32),
        # Pass CPU indptr to plan so host planning does not include a redundant D2H.
        "indptr": torch.tensor(ptr, dtype=torch.int32),
        "indices": indices_cpu.cuda(),
        "last_page_len": torch.tensor(last_page_len, dtype=torch.int32).cuda(),
    }


def reference_check(
    torch, data, lengths, outputs: dict, atol: float, rtol: float
) -> dict:
    # FP32 reference for representative short/long rows, not just library-vs-library.
    selected = sorted(
        {0, len(lengths) // 2, max(range(len(lengths)), key=lengths.__getitem__)}
    )
    diagnostics = {}
    group = Q_HEADS // KV_HEADS
    for name, out in outputs.items():
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError(f"{name} returned non-finite output")
    for b in selected:
        pages = (lengths[b] + PAGE - 1) // PAGE
        ids = data["block_ids_cpu"][b, :pages].to(device="cuda", dtype=torch.int64)
        k = (
            data["k"]
            .index_select(0, ids)
            .reshape(-1, KV_HEADS, DIM)[: lengths[b]]
            .float()
        )
        v = (
            data["v"]
            .index_select(0, ids)
            .reshape(-1, KV_HEADS, DIM)[: lengths[b]]
            .float()
        )
        q = data["q"][b].float().reshape(KV_HEADS, group, DIM)
        scores = torch.einsum("hgd,nhd->hgn", q, k) / math.sqrt(DIM)
        ref = torch.einsum("hgn,nhd->hgd", scores.softmax(dim=-1), v).reshape(
            Q_HEADS, DIM
        )
        for name, out in outputs.items():
            torch.testing.assert_close(out[b].float(), ref, atol=atol, rtol=rtol)
            diagnostics[f"{name}_row{b}_max_abs_error"] = float(
                (out[b].float() - ref).abs().max()
            )
    return diagnostics


def run_case(torch, flashinfer, hpc, bench_gpu_time, lengths, args) -> dict:
    data = build_inputs(torch, lengths, args.seed)
    fi_workspace = torch.empty(
        args.workspace_mb * 1024**2, device="cuda", dtype=torch.uint8
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        fi_workspace,
        "NHD",
        use_cuda_graph=False,
        use_tensor_cores=False,
        backend="fa2",
    )
    fi_out = torch.empty_like(data["q"])

    def fi_plan():
        wrapper.plan(
            data["indptr"],
            data["indices"],
            data["last_page_len"],
            Q_HEADS,
            KV_HEADS,
            DIM,
            PAGE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            pos_encoding_mode="NONE",
            sm_scale=1 / math.sqrt(DIM),
        )

    def fi_run():
        return wrapper.run(data["q"], (data["k"], data["v"]), out=fi_out)

    fi_plan()  # JIT compilation is outside ALL timing regions.
    fi_run()
    torch.cuda.synchronize()
    int_buffer_cpu = wrapper._int_workspace_buffer.cpu()

    def read_i32(offset, count):
        if offset < 0 or offset % 4 or offset + count * 4 > int_buffer_cpu.numel():
            raise RuntimeError("FlashInfer int-workspace byte offsets changed")
        return int_buffer_cpu[offset : offset + count * 4].view(torch.int32).tolist()

    fi_schedule = parse_fi_plan(wrapper._plan_info, read_i32, lengths, KV_HEADS)
    record = {
        "batch": len(lengths),
        "total_kv_tokens": sum(lengths),
        "lengths": lengths,
        "flashinfer_schedule": fi_schedule,
    }
    outputs = {"flashinfer": fi_out}
    if args.ported_hpc:
        from flashinfer.hpc_decode import create_hpc_flat_bin_decode_wrapper

        port = create_hpc_flat_bin_decode_wrapper(
            device=data["q"].device,
            cta_per_sm=args.cta_per_sm,
            min_process_len=args.min_process_len,
        )
        hpc_out = torch.empty_like(data["q"])

        def hpc_assign():
            port.plan(
                data["indptr"],
                data["indices"],
                data["last_page_len"],
                Q_HEADS,
                KV_HEADS,
                DIM,
                PAGE,
                sm_scale=1 / math.sqrt(DIM),
            )

        def hpc_run():
            return port.run(data["q"], (data["k"], data["v"]), out=hpc_out)

        hpc_assign()
        hpc_run()
        torch.cuda.synchronize()
        record["hpc_schedule"] = parse_port_map(port.schedule(), lengths, KV_HEADS)
        record["hpc_original_schedule"] = parse_hpc_map(
            port.original_schedule(), lengths, KV_HEADS, tile=128
        )
        outputs["hpc_port"] = hpc_out
        torch.testing.assert_close(
            hpc_out.float(), fi_out.float(), atol=args.atol, rtol=args.rtol
        )
        record["cross_library_max_abs_error"] = float(
            (hpc_out.float() - fi_out.float()).abs().max()
        )
    elif hpc is not None:
        task_map = hpc.get_attention_decode_task_workspace(
            len(lengths), max(lengths), KV_HEADS, min_process_len=args.min_process_len
        )
        hpc_out = torch.empty_like(data["q"])

        def hpc_assign():
            # Assigner wants num_seq_q=1 (called mtp in this Python API).
            hpc.assign_attention_decode_task(
                data["lengths"],
                task_map,
                KV_HEADS,
                mtp=1,
                new_kv_included=True,
                min_process_len=args.min_process_len,
            )

        def hpc_run():
            # Decode API wants number of EXTRA draft tokens: mtp=0 for one Q.
            return hpc.attention_decode_bf16(
                data["q"],
                data["k"],
                data["v"],
                data["block_ids"],
                data["lengths"],
                mtp=0,
                new_kv_included=True,
                splitk=True,
                task_map=task_map,
                output=hpc_out,
            )

        hpc_assign()
        hpc_run()
        torch.cuda.synchronize()
        # The workspace is sized for B * max_length, much larger than the active
        # schedule for a skewed batch. Copy only the active bins + chunk table.
        map_i32 = task_map.view(torch.int32)
        header = map_i32[:4].cpu().tolist()
        used_ints = TASK_STRIDE * (1 + header[0] * header[1]) + KV_HEADS * len(lengths)
        if not 0 < used_ints <= map_i32.numel():
            raise RuntimeError("Unsupported HPC-Ops task-map workspace layout")
        record["hpc_schedule"] = parse_hpc_map(
            map_i32[:used_ints].cpu().tolist(), lengths, KV_HEADS
        )
        outputs["hpc"] = hpc_out
        # Compare the FULL outputs, then also check selected rows against FP32.
        torch.testing.assert_close(
            hpc_out.float(), fi_out.float(), atol=args.atol, rtol=args.rtol
        )
        record["cross_library_max_abs_error"] = float(
            (hpc_out.float() - fi_out.float()).abs().max()
        )
    record["reference_check"] = reference_check(
        torch, data, lengths, outputs, args.atol, args.rtol
    )

    def fi_step():
        fi_plan()
        fi_run()

    times = {
        "flashinfer_gpu_run_us": gpu_us(fi_run, bench_gpu_time, args),
        "flashinfer_plan_wall_us": wall_us(fi_plan, torch, args.iters, args.warmup),
        "flashinfer_step_wall_us": wall_us(fi_step, torch, args.iters, args.warmup),
    }
    if hpc is not None or args.ported_hpc:

        def hpc_step():
            hpc_assign()
            hpc_run()

        times.update(
            {
                "hpc_gpu_run_us": gpu_us(hpc_run, bench_gpu_time, args),
                "hpc_assign_wall_us": wall_us(
                    hpc_assign, torch, args.iters, args.warmup
                ),
                "hpc_step_wall_us": wall_us(hpc_step, torch, args.iters, args.warmup),
            }
        )
        if args.ported_hpc:
            times["hpc_prepare_gpu_us"] = gpu_us(
                lambda: port.assign_gpu(1), bench_gpu_time, args
            )
            times["hpc_original_assign_gpu_us"] = gpu_us(
                lambda: port.assign_gpu(2), bench_gpu_time, args
            )
            times["hpc_adapt_gpu_us"] = gpu_us(
                lambda: port.assign_gpu(3), bench_gpu_time, args
            )
            times["hpc_full_assign_gpu_us"] = gpu_us(
                port.assign_gpu, bench_gpu_time, args
            )
    record["timing"] = times
    return record


def self_test():
    cases = make_cases(32, 512, 65536)
    assert sum(cases["uniform"]) == sum(cases["skewed"])
    lengths = [512, 512, 512, 6656]
    stride, ctas = 33, 4
    raw = [0] * (TASK_STRIDE * (1 + ctas * stride) + 4)
    raw[:4] = [stride, ctas, 1, 4]
    buckets = [
        [(0, 0, 512), (1, 0, 512), (2, 0, 512), (3, 0, 512)],
        [(3, 512, 2048)],
        [(3, 2560, 2048)],
        [(3, 4608, 2048)],
    ]
    chunks = [0] * 4
    for c, bucket in enumerate(buckets):
        for j, (b, offset, size) in enumerate(bucket):
            pos = TASK_STRIDE * (1 + c * stride + j)
            raw[pos : pos + 5] = [0, b, chunks[b], offset, size]
            chunks[b] += 1
        pos = TASK_STRIDE * (1 + c * stride + len(bucket))
        raw[pos : pos + 2] = [-1, -1]
    raw[TASK_STRIDE * (1 + ctas * stride) :] = chunks
    assert parse_hpc_map(raw, lengths, 1)["tokens_per_cta"] == [2048] * 4
    fi = parse_fi_plan([4, 0, 0, 0, 0, 0, 0, 0, 0, 0], None, lengths, 1)
    assert fi["tokens_per_attention_cta"] == lengths
    arrays = {0: [0] * 4, 16: [0, 1, 2, 3], 32: [0, 4], 40: [1664]}
    fi = parse_fi_plan(
        [4, 0, 0, 0, 16, 32, 0, 40, 0, 1],
        lambda offset, count: arrays[offset][:count],
        [6656],
        1,
    )
    assert fi["tokens_per_attention_cta"] == [1664] * 4
    damaged = raw.copy()
    damaged[TASK_STRIDE + 4] = 511
    try:
        parse_hpc_map(damaged, lengths, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Corrupt HPC task coverage was not rejected")
    print(
        "PASS: matched-total cases, real-layout parsers, split/no-split, coverage guard"
    )
    print(
        "Self-test uses synthetic records to test diagnostics only; it is NOT a GPU benchmark."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--batch", type=int, default=0, help="0: choose 2 * actual SM count"
    )
    parser.add_argument("--short", type=int, default=512)
    parser.add_argument("--long", type=int, default=65536)
    parser.add_argument("--cases", default="small,uniform,skewed")
    parser.add_argument("--min-process-len", type=int, default=512)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing", choices=("cupti", "events"), default="cupti")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--hpc-python-path", type=Path, help="installed/build directory containing hpc/"
    )
    parser.add_argument(
        "--flashinfer-python-path", type=Path, help="directory containing flashinfer/"
    )
    parser.add_argument(
        "--flashinfer-only",
        action="store_true",
        help="smoke test; NOT a dual-library comparison",
    )
    parser.add_argument(
        "--ported-hpc",
        action="store_true",
        help="Compare two real schedulers INSIDE patched FlashInfer, not two original libraries",
    )
    parser.add_argument(
        "--cta-per-sm",
        type=int,
        default=None,
        help="Ported CTA pool multiplier; default SM100=1; overrides are experiments",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="CPU-only diagnostic tests; no library imports",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.iters < 1 or args.warmup < 0 or args.workspace_mb < 1:
        parser.error("iters/workspace-mb must be positive and warmup nonnegative")
    if not 0 < args.short <= args.long or args.short % PAGE or args.long % PAGE:
        parser.error("short/long must be positive multiples of 64 with short <= long")
    if args.min_process_len < PAGE or args.min_process_len % PAGE:
        parser.error("min-process-len must be a positive multiple of 64")
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the real benchmark; --self-test is CPU-only")
    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability()
    if args.flashinfer_only and args.ported_hpc:
        parser.error("--flashinfer-only and --ported-hpc are mutually exclusive")
    if capability != (9, 0) and not args.flashinfer_only and not args.ported_hpc:
        parser.error(
            f"Dual-library mode requires SM90; found SM{capability[0]}{capability[1]}. "
            "HPC-Ops SM90 dynamic kernel is unavailable on B200/SM100. "
            "--flashinfer-only can validate only the FlashInfer side."
        )
    for path in (args.flashinfer_python_path, args.hpc_python_path):
        if path is not None:
            sys.path.insert(0, str(path.resolve()))
    try:
        flashinfer = importlib.import_module("flashinfer")
        hpc = (
            None
            if args.flashinfer_only or args.ported_hpc
            else importlib.import_module("hpc")
        )
    except (ImportError, AssertionError, OSError) as exc:
        parser.error(
            f"Library import failed: {exc}. Install both libraries in this Python environment."
        )
    from flashinfer.testing import bench_gpu_time

    if "repeat_iters" not in inspect.signature(bench_gpu_time).parameters:
        parser.error(
            "Unsupported FlashInfer bench_gpu_time API; this script expects repeat_iters"
        )
    props = torch.cuda.get_device_properties(args.device)
    batch = args.batch or 2 * props.multi_processor_count
    if not 2 <= batch <= 2048:
        parser.error("batch must be in [2, 2048] for these HPC-Ops workloads")
    cases = make_cases(batch, args.short, args.long)
    selected = args.cases.split(",")
    if len(selected) != len(set(selected)) or any(
        name not in cases for name in selected
    ):
        parser.error(
            "cases must be a unique comma-separated subset of " + ",".join(cases)
        )
    report = {
        "gpu": props.name,
        "compute_capability": capability,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "hpc": getattr(hpc, "__version__", None),
        "mode": "ported-scheduler"
        if args.ported_hpc
        else ("flashinfer-only" if hpc is None else "dual-library"),
        "cta_per_sm": args.cta_per_sm,
        "port_scheduler_tile": 128 if args.ported_hpc else None,
        "port_planner": "original_parallel" if args.ported_hpc else None,
        "seed": args.seed,
        "repeat_iters": args.iters,
        "warmup_iters": args.warmup,
        "gpu_timer_requested": args.timing,
        "kv_heads": KV_HEADS,
        "q_heads": Q_HEADS,
        "head_dim": DIM,
        "page_size": PAGE,
        "cache_dtype": "bfloat16",
        "use_tensor_cores": False,
        "use_cuda_graph": False,
        "timing_notes": {
            "gpu_run": "median, cold L2; attention + merge only; CUPTI falls back to CUDA events if unavailable",
            "prepare_wall": "median synchronized host+device time; JIT and input allocation excluded",
            "step_wall": "median synchronized prepare+run; prepared page metadata; no cache flushing",
        },
        "cases": {},
    }
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    for name in selected:
        print(f"\n--- {name} ---", flush=True)
        result = run_case(torch, flashinfer, hpc, bench_gpu_time, cases[name], args)
        report["cases"][name] = result
        for key in ("hpc_schedule", "flashinfer_schedule"):
            if key in result:
                summary = {
                    k: v
                    for k, v in result[key].items()
                    if k
                    not in (
                        "tasks",
                        "request_chunk_tasks",
                        "tokens_per_cta",
                        "tokens_per_attention_cta",
                        "tasks_per_cta",
                        "chunks_head_major",
                        "chunks_per_request",
                        "state_indptr",
                    )
                }
                print(key, json.dumps(summary))
        print("timing_us", json.dumps(result["timing"]))
        print("correctness PASS", json.dumps(result["reference_check"]))
        if name == "skewed" and not result["flashinfer_schedule"]["split_kv"]:
            print("OBSERVED: FlashInfer did not split the saturated skewed batch.")
            if hpc is not None or args.ported_hpc:
                print(
                    "HPC longest-request chunks/head:",
                    result["hpc_schedule"]["longest_chunks_per_kv_head"],
                )
        torch.cuda.synchronize()
    if args.json_out:
        with args.json_out.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print("Wrote", args.json_out)
    print(
        "\nCTA loads are scheduling diagnostics, not measured CTA durations. "
        "Ported mode shares FlashInfer math but changes task consumption/merge topology; "
        "it is not an original HPC-Ops-vs-FlashInfer whole-library benchmark."
    )


if __name__ == "__main__":
    main()
