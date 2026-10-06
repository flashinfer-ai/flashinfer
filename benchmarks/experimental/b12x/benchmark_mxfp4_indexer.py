"""Benchmark V4.1's public MXFP4 paged-score/selection path and exact oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from b12x.preparation import PreparationSession
from benchmarks.experimental.b12x.attention_preparation import prepare_mxfp4
from b12x.attention import dsa_indexer as indexer
from benchmarks.experimental.b12x.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from benchmarks.experimental.b12x.benchmark_v41_serving import repository_state


def quantized_reference(x):
    """Independent RN-even E2M1 encoding and BF16 dequantization oracle."""
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax / 6).contiguous().view(torch.int32)
    scales = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    factors = (scales.int() << 23).view(torch.float32)
    normalized = groups / factors[..., None]
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    choice = (normalized.abs()[..., None] - lut[order]).abs().argmin(-1)
    codes = order[choice].to(torch.uint8) | (
        torch.signbit(normalized).to(torch.uint8) << 3
    )
    decoded = (
        (
            lut[(codes & 7).long()]
            * torch.where((codes & 8) != 0, -1, 1)
            * factors[..., None]
        )
        .bfloat16()
        .reshape_as(x)
    )
    codes = codes.reshape(*x.shape[:-1], 128)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scales, decoded


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def check_selection(scores, indices, values, candidates=None):
    for row in range(scores.shape[0]):
        valid = torch.isfinite(scores[row])
        count = min(512, int(valid.sum()))
        selected = indices[row, :count].long()
        assert bool((indices[row, count:] == -1).all())
        assert bool(torch.isneginf(values[row, count:]).all())
        assert bool((selected[1:] > selected[:-1]).all())
        expected_values = scores[row, valid].float().topk(count).values.sort().values
        torch.testing.assert_close(
            values[row, :count].sort().values, expected_values, rtol=0, atol=0
        )
        if candidates is None:
            torch.testing.assert_close(
                values[row, :count], scores[row, selected].float(), rtol=0, atol=0
            )
        else:
            assert bool(torch.isin(selected, candidates[row, valid].long()).all())


@torch.inference_mode()
def run_mode(args, candidates_mode):
    device = torch.device("cuda", args.device)
    rows_capacity, context = max(args.rows), max(args.widths)
    generator = torch.Generator(device=device).manual_seed(410831)
    q = (
        torch.randn(
            (rows_capacity, args.heads, 128), generator=generator, device=device
        )
        / 4
    ).bfloat16()
    keys = (
        torch.randn((context, 128), generator=generator, device=device) / 4
    ).bfloat16()
    weights = (
        torch.randn((rows_capacity, args.heads), generator=generator, device=device)
        / 32
    ).bfloat16()
    q_reference, sf_reference, decoded_q = quantized_reference(q)
    _, _, decoded_k = quantized_reference(keys)
    packed = torch.empty_like(q_reference)
    scales = torch.empty_like(sf_reference)
    page_bytes = indexer.index_mxfp4_page_bytes(args.page_size)
    if args.page_stride < page_bytes or args.page_stride % 16:
        raise ValueError(
            "page stride must be aligned and cover the native record planes"
        )
    page_count = (context + args.page_size - 1) // args.page_size
    base_pid = 2**31 // args.page_stride + 17 if args.high_pid else 7
    backing = torch.empty(
        (base_pid + page_count, args.page_stride), dtype=torch.uint8, device=device
    )
    pool = backing.as_strided(
        (base_pid + page_count, page_bytes), (args.page_stride, 1)
    )
    physical_pages = (
        torch.randperm(page_count, generator=generator, device=device).int() + base_pid
    )
    positions = torch.arange(context, device=device)
    slots = (
        physical_pages[positions // args.page_size].long() * args.page_size
        + positions % args.page_size
    )
    planned_pages = (args.capacity + args.page_size - 1) // args.page_size
    table = torch.full((1, planned_pages), -1, dtype=torch.int32, device=device)
    table[0, :page_count].copy_(physical_pages)
    lengths = torch.full((rows_capacity,), context, dtype=torch.int32, device=device)
    active = torch.tensor([context], dtype=torch.int32, device=device)
    candidate_capacity = args.candidate_capacity if candidates_mode else 0
    candidate_ids = None
    candidate_lengths = None
    if candidates_mode:
        if candidate_capacity > context:
            raise ValueError("candidate capacity must not exceed the populated context")
        candidate_ids = torch.stack(
            [
                torch.randperm(context, generator=generator, device=device)[
                    :candidate_capacity
                ].int()
                for _ in range(rows_capacity)
            ]
        )
        candidate_lengths = torch.full(
            (rows_capacity,), candidate_capacity, dtype=torch.int32, device=device
        )
    plan = indexer.plan(
        indexer.Caps(
            device=device,
            num_q_heads=args.heads,
            max_q_rows=rows_capacity,
            max_page_table_width=planned_pages,
            page_size=args.page_size,
            cache_format="mxfp4",
            topk=512,
            max_candidates=candidate_capacity,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    output_indices = torch.empty((rows_capacity, 512), dtype=torch.int32, device=device)
    output_scores = torch.empty(
        (rows_capacity, 512), dtype=torch.float32, device=device
    )

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    prepare_mxfp4(session, plan, q=q, keys=keys, slots=slots, arguments=dict(
        q_mxfp4=packed, q_scales=scales, query_weights=weights,
        index_k_cache=pool, page_table=table, cache_lengths=lengths, active_width=active,
        output_indices=output_indices, output_scores=output_scores,
        candidate_indices=candidate_ids, candidate_lengths=candidate_lengths,
    ))
    torch.testing.assert_close(packed, q_reference, rtol=0, atol=0)
    torch.testing.assert_close(scales, sf_reference, rtol=0, atol=0)

    def bind(rows, visible):
        score_width = candidate_capacity if candidates_mode else visible
        if args.full_score_width and not candidates_mode:
            score_width = args.capacity
        return indexer.bind(
            plan,
            scratch=scratch,
            q_mxfp4=packed[:rows],
            q_scales=scales[:rows],
            query_weights=weights[:rows],
            index_k_cache=pool,
            page_table=table,
            cache_lengths=lengths[:rows],
            active_width=active,
            output_indices=output_indices[:rows],
            output_scores=output_scores[:rows],
            candidate_indices=None if candidate_ids is None else candidate_ids[:rows],
            candidate_lengths=None
            if candidate_lengths is None
            else candidate_lengths[:rows],
            score_width=score_width,
        )

    def oracle(rows, visible):
        dot = torch.einsum("rhd,kd->rhk", decoded_q[:rows], decoded_k)
        expected = (dot.relu() * weights[:rows, :, None]).sum(1)
        logical = positions[None]
        expected.masked_fill_(
            (logical >= lengths[:rows, None]) | (logical >= active[0]), -torch.inf
        )
        if candidates_mode:
            expected = expected.gather(1, candidate_ids[:rows].long())
            expected.masked_fill_(
                torch.arange(candidate_capacity, device=device)[None]
                >= candidate_lengths[:rows, None],
                -torch.inf,
            )
        elif args.full_score_width:
            full = torch.full(
                (rows, args.capacity), -torch.inf, dtype=expected.dtype, device=device
            )
            full[:, :context] = expected
            expected = full
        else:
            expected = expected[:, :visible]
        return expected

    # Resolve at the largest populated shape, then exercise smaller live row and
    # visibility counts under freeze. Planned capacity and allocation never shrink.
    warm_binding = bind(rows_capacity, context)
    indexer.score(warm_binding)
    indexer.select(warm_binding)
    torch.cuda.synchronize(device)
    records = []
    session.freeze()
    graphs = {}
    try:
        for visible in args.widths:
            active.fill_(visible)
            lengths.fill_(visible)
            for rows in args.rows:
                binding = bind(rows, visible)
                expected = oracle(rows, visible)
                actual = indexer.score(binding)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert bool(torch.count_nonzero(actual[torch.isfinite(actual)]))
                indexer.select(binding)
                selected_candidates = (
                    None if candidate_ids is None else candidate_ids[:rows]
                )
                check_selection(
                    expected,
                    output_indices[:rows],
                    output_scores[:rows],
                    selected_candidates,
                )
                score_graph, full_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(score_graph):
                    indexer.score(binding)
                with torch.cuda.graph(full_graph):
                    indexer.score(binding)
                    indexer.select(binding)
                # Runtime visibility/empty-row mutation must affect replay, not
                # cached host decisions or a separately allocated output tensor.
                lengths[0] = 0
                if rows > 1:
                    lengths[1] = min(31, visible)
                full_graph.replay()
                torch.cuda.synchronize(device)
                mutated = oracle(rows, visible)
                torch.testing.assert_close(actual, mutated, rtol=0, atol=0)
                check_selection(
                    mutated,
                    output_indices[:rows],
                    output_scores[:rows],
                    selected_candidates,
                )
                lengths.fill_(visible)
                full_graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                output_hash = tensor_hash(actual)
                allocation_before_warmup = torch.cuda.memory_allocated(device)
                addresses = (actual.data_ptr(), scratch.data_ptr(), pool.data_ptr())
                graphs = {"score": score_graph, "score_select": full_graph}
                for graph in graphs.values():
                    for _ in range(args.warmup):
                        graph.replay()
                torch.cuda.synchronize(device)
                flush = make_l2_flush_fn(args.cold_l2)
                allocated = torch.cuda.memory_allocated(device)
                samples = {name: [] for name in graphs}
                for sample in range(args.samples):
                    for name in (
                        list(graphs) if sample % 2 == 0 else list(reversed(graphs))
                    ):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        if flush is not None:
                            flush()
                        begin.record()
                        for _ in range(args.replays):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(
                            begin.elapsed_time(end) * 1000 / args.replays
                        )
                allocation_after_timing = torch.cuda.memory_allocated(device)
                assert allocation_after_timing == allocated, (
                    allocation_before_warmup,
                    allocated,
                    allocation_after_timing,
                )
                assert addresses == (
                    actual.data_ptr(),
                    scratch.data_ptr(),
                    pool.data_ptr(),
                )
                record = {
                    "mode": "candidates" if candidates_mode else "dense",
                    "rows": rows,
                    "heads": args.heads,
                    "visible": visible,
                    "score_width": actual.shape[1],
                    "allocation_bytes": {
                        "before_warmup": allocation_before_warmup,
                        "after_warmup": allocated,
                        "after_timing": allocation_after_timing,
                    },
                    "planned_capacity": args.capacity,
                    "scratch_bytes": scratch.numel(),
                    "high_pid_byte_offset": base_pid * args.page_stride,
                    "samples_us": samples,
                    "median_us": {
                        name: median(values) for name, values in samples.items()
                    },
                    "score_sha256": output_hash,
                    "correctness": "Exact independent BF16-stage score and top-k values; frozen live-count reuse, mutated-visibility graph replay, stable addresses, no replay allocation and high physical page offsets.",
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ("mode", "rows", "visible", "median_us")
                        }
                    ),
                    flush=True,
                )
                for graph in graphs.values():
                    graph.reset()
    finally:
        for graph in graphs.values():
            graph.reset()
        session.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", default="6,64")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--widths", default="2048,8192,16384")
    parser.add_argument("--capacity", type=int, default=1048576)
    parser.add_argument(
        "--full-score-width",
        action="store_true",
        help="Replay dense scoring at its full reserved width with only a live prefix populated",
    )
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--page-stride", type=int, default=700416)
    parser.add_argument("--candidate-capacity", type=int, default=1024)
    parser.add_argument("--modes", default="dense,candidates")
    parser.add_argument(
        "--high-pid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--cold-l2", action="store_true", help="Read-evict L2 before each timed graph; requires --replays 1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rows = [int(value) for value in args.rows.split(",")]
    args.widths = [int(value) for value in args.widths.split(",")]
    args.modes = args.modes.split(",")
    if args.cold_l2 and args.replays != 1:
        parser.error("--cold-l2 requires --replays 1 so every measured replay is cold")
    if (
        min(*args.rows, *args.widths, args.samples, args.replays, args.warmup) <= 0
        or max(args.widths) > args.capacity
    ):
        parser.error(
            "counts must be positive and visible widths must fit planned capacity"
        )
    if any(mode not in ("dense", "candidates") for mode in args.modes):
        parser.error("modes must be dense and/or candidates")
    if args.full_score_width and "candidates" in args.modes:
        parser.error("--full-score-width applies only to dense source scoring")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {
        "command": [sys.executable, *sys.argv],
        "repository": repository_state(ROOT),
        "arguments": {**vars(args), "output": str(args.output)},
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl")
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "CUTE_", "B12X_", "OMP_"))
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "flashinfer/experimental/b12x/attention/dsa_indexer/mxfp4.py")
        },
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "scope": "Real public MXFP4 score/selection plans, no TP collective. Latencies are not full-model throughput. Source-version comparisons use identical score hashes and explicit baseline/candidate latency ratios.",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        result["results"].extend(run_mode(args, mode == "candidates"))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
