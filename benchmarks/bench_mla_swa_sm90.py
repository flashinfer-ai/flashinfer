# SPDX-License-Identifier: Apache-2.0
"""Standalone SM90 MLA+SWA complete-op CUDA Graph benchmark.

Example: python benchmarks/bench_mla_swa_sm90.py --refcheck

The PyTorch baseline implements the same causal window and BF16 score rounding,
but uses PyTorch gather/mask operations, NOT SGLang's Triton fallback kernels.
Its latency must not be labeled as the original SGLang fallback latency.
No SGLang installation, model weights, or previously built binary is required.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

import flashinfer
from flashinfer.mla import (
    batch_decode_with_kv_cache_mla_swa,
    get_mla_swa_workspace_size_bytes,
)


def torch_reference(q, kv, table, lengths, scale, dtype=torch.bfloat16, with_abs=False):
    b, sq, heads, dim = q.shape
    width = math.ceil((513 + sq - 1) / 64) * 64
    start = (lengths - 512 - sq).clamp_min(0)
    positions = start[:, None] + torch.arange(width, device=q.device)
    pages = table.gather(1, (positions // 64).clamp_max(table.shape[1] - 1).long())
    keys = kv.reshape(-1, dim)[pages.long() * 64 + positions % 64].to(dtype)
    scores = torch.bmm(q.to(dtype).reshape(b, sq * heads, dim), keys.transpose(1, 2))
    if dtype == torch.bfloat16:
        scores = scores.float()
    scores = scores.view(b, sq, heads, width) * scale
    q_pos = lengths[:, None] - sq + torch.arange(sq, device=q.device)
    valid = (positions[:, None, :] <= q_pos[:, :, None]) & (
        positions[:, None, :] >= (q_pos - 512).clamp_min(0)[:, :, None]
    )
    probabilities = scores.masked_fill(~valid[:, :, None, :], -torch.inf).softmax(-1)
    probabilities = probabilities.to(dtype).reshape(b, sq * heads, width)
    values = keys[..., :1024]
    output = torch.bmm(probabilities, values).view(b, sq, heads, 1024)
    if with_abs:
        return output, torch.bmm(probabilities, values.abs()).view_as(output)
    return output


def check_accuracy(actual, inputs, scale):
    gold, weighted_abs = torch_reference(
        *inputs, scale, dtype=torch.float64, with_abs=True
    )
    actual = actual.double()
    diff = actual - gold
    floor = 1e-5 * math.sqrt(1024)
    relative_l2 = (diff.norm(dim=-1) / gold.norm(dim=-1).clamp_min(floor)).max().item()
    budget_ratio = (diff.abs() / (4 * 2**-8 * weighted_abs + 1e-5)).max().item()
    assert torch.isfinite(actual).all(), "Nonfinite output"
    assert relative_l2 <= 2 * 2**-8, relative_l2
    assert budget_ratio <= 1, budget_ratio
    return dict(max_row_relative_l2=relative_l2, max_error_budget_ratio=budget_ratio)


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def benchmark(args, b, sq):
    device = torch.device("cuda")
    pages = math.ceil(args.kv_len / 64)
    q = torch.randn(b, sq, 64, 1088, dtype=torch.bfloat16, device=device)
    kv = torch.randn(b * pages, 64, 1088, dtype=torch.bfloat16, device=device)
    table = torch.arange(b * pages, dtype=torch.int32, device=device).view(b, pages)
    lengths = torch.full((b,), args.kv_len, dtype=torch.int32, device=device)
    workspace = torch.empty(
        get_mla_swa_workspace_size_bytes(b, sq), dtype=torch.uint8, device=device
    )
    out = torch.empty(b, sq, 64, 1024, dtype=torch.bfloat16, device=device)

    def native():
        return batch_decode_with_kv_cache_mla_swa(
            q,
            kv,
            workspace,
            table,
            lengths,
            window_left=512,
            sm_scale=args.scale,
            out=out,
        )

    if args.autotune:
        with flashinfer.autotune(True, cache=args.autotune_cache):
            native()
    functions = {
        "flashinfer": native,
        "pytorch_reference": lambda: torch_reference(q, kv, table, lengths, args.scale),
    }
    graphs = {name: capture(fn) for name, fn in functions.items()}
    accuracy = (
        check_accuracy(graphs["flashinfer"][1], (q, kv, table, lengths), args.scale)
        if args.refcheck
        else None
    )
    rounds = {name: [] for name in functions}
    names = list(functions)
    for index in range(args.rounds):
        for name in names if index % 2 == 0 else reversed(names):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graphs[name][0].replay()
            end.record()
            end.synchronize()
            rounds[name].append(start.elapsed_time(end) * 1000 / args.replays)
    native_us, reference_us = (statistics.median(rounds[name]) for name in names)
    return dict(
        local_bs=b,
        s_q=sq,
        kv_len=args.kv_len,
        window_left=512,
        sm_scale=args.scale,
        autotune=args.autotune,
        flashinfer_us=native_us,
        pytorch_reference_us=reference_us,
        pytorch_reference_over_flashinfer=reference_us / native_us,
        accuracy=accuracy,
        rounds_us=rounds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--q-lens", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--kv-len", type=int, default=4127)
    parser.add_argument("--scale", type=float, default=0.0625)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--refcheck", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--autotune-cache", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(b not in (1, 2, 4, 8, 16) for b in args.batch_sizes):
        parser.error("batch sizes must be selected from 1, 2, 4, 8, 16")
    if any(sq not in (1, 4) for sq in args.q_lens) or args.kv_len < max(args.q_lens):
        parser.error("q-lens must be 1 or 4; kv-len must be >= every q-len")
    if args.rounds < 1 or args.replays < 1 or not 0 < args.scale < float("inf"):
        parser.error("rounds/replays and finite scale must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("an SM90 CUDA device is required")
    torch.manual_seed(123)
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        flashinfer_version=flashinfer.__version__,
        timing="interleaved CUDA Graph, complete op, warm cache; microseconds",
        baseline="PyTorch reference; NOT the SGLang Triton fallback",
    )
    print(json.dumps(metadata), flush=True)
    rows = []
    for sq in args.q_lens:
        for b in args.batch_sizes:
            row = benchmark(args, b, sq)
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2))


if __name__ == "__main__":
    main()
