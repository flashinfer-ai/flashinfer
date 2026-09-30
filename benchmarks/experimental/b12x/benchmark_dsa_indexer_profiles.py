#!/usr/bin/env python3
"""Benchmark model-derived DeepSeek V4/V4.1 DSA indexer recipes.

Examples:
  python benchmarks/benchmark_dsa_indexer_profiles.py --model-profile deepseek-v4.1-flash --forms c2-dense --output /tmp/c2.json
  python benchmarks/benchmark_dsa_indexer_profiles.py --model-profile deepseek-v4.1-flash --forms c1-source,c1-reindex --output /tmp/c1.json
  python benchmarks/benchmark_dsa_indexer_profiles.py --model-profile deepseek-v4-flash --forms c4-dense --output /tmp/v4.json

This measures prepared synthetic index Q/K only: it is neither learned-projection
nor whole-vLLM-forward timing.  V4.1 timings separate native BF16-stage score
from score+select.  The c1-source score+select boundary includes candidate
publication; c1-reindex consumes that publisher's output from an independent
source query, never generated candidate ids.
"""

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
from b12x.attention.dsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
import torch

from b12x.preparation import PreparationSession, PreparedCall
from benchmarks.experimental.b12x.attention_preparation import prepare_mxfp4
from b12x.attention import dsa_indexer as api
from benchmarks.experimental.b12x.benchmark_mxfp4_indexer import (
    check_selection,
    quantized_reference,
    tensor_hash,
)
from benchmarks.experimental.b12x.benchmark_v41_serving import repository_state
from benchmarks.experimental.b12x.common import nvidia_smi_gpu_mode_snapshot
from benchmarks.experimental.b12x.deepseek_attention_profiles import (
    integration_provenance,
    load_attention_profile,
)


def _csv(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _timed(
    graphs: dict[str, torch.cuda.CUDAGraph], args, device: torch.device
) -> dict[str, list[float]]:
    for graph in graphs.values():
        for _ in range(args.warmup):
            graph.replay()
    torch.cuda.synchronize(device)
    samples = {name: [] for name in graphs}
    for sample in range(args.samples):
        names = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
        for name in names:
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / args.replays)
    return samples


def _pool_and_table(
    *, keys, page_size, page_bytes, page_stride, capacity, high_pid, generator
):
    """Build the native physical-page contract, including a >2**31 byte offset."""
    device = keys.device
    live_pages = (keys.shape[0] + page_size - 1) // page_size
    planned_pages = (capacity + page_size - 1) // page_size
    if page_stride < page_bytes or page_stride % 16:
        raise ValueError("page stride must be >= native page bytes and 16-byte aligned")
    base = 2**31 // page_stride + 17 if high_pid else 7
    backing = torch.empty(
        (base + live_pages, page_stride), dtype=torch.uint8, device=device
    )
    pool = backing.as_strided((base + live_pages, page_bytes), (page_stride, 1))
    physical = (
        torch.randperm(live_pages, generator=generator, device=device).int() + base
    )
    table = torch.full((1, planned_pages), -1, dtype=torch.int32, device=device)
    table[0, :live_pages] = physical
    return backing, pool, physical, table, base


def _mxfp4_oracle(
    decoded_q,
    decoded_k,
    weights,
    rows,
    visible,
    active,
    candidates=None,
    candidate_lengths=None,
    lengths=None,
):
    scores = (
        torch.einsum("rhd,kd->rhk", decoded_q[:rows], decoded_k[:visible]).relu()
        * weights[:rows, :, None]
    ).sum(1)
    if lengths is not None:
        scores.masked_fill_(
            torch.arange(visible, device=scores.device)[None] >= lengths[:rows, None],
            -torch.inf,
        )
    if candidates is not None:
        scores = scores.gather(1, candidates[:rows].long().clamp_min(0))
        positions = torch.arange(scores.shape[1], device=scores.device)[None]
        scores.masked_fill_(positions >= candidate_lengths[:rows, None], -torch.inf)
        return scores
    # Captured V4.1 calls retain the planned score columns.  The live cache
    # length masks their suffix; it must not cause a compact active-width score.
    planned = torch.full(
        (rows, active), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    planned[:, :visible] = scores
    return planned


def _assert_published_blocks(scores, published, lengths, visible):
    """Check exact block-score winners without prescribing boundary-tie IDs."""
    for row in range(scores.shape[0]):
        live = int(visible[row]) if isinstance(visible, torch.Tensor) else visible
        count = int(lengths[row])
        values = published[row, :count].long()
        assert bool(published[row, count:].eq(-1).all())
        if not live:
            assert count == 0
            continue
        blocks = (live + 7) // 8
        assert count == min(2047, blocks - 1) * 8 + live - (blocks - 1) * 8
        assert bool(((values >= 0) & (values < live)).all())
        assert bool((values[1:] > values[:-1]).all())
        winners = (values // 8).unique_consecutive()
        assert winners.numel() == min(2048, blocks)
        assert winners[-1].item() == blocks - 1
        expanded = (
            winners[:, None] * 8 + torch.arange(8, device=scores.device)
        ).reshape(-1)
        torch.testing.assert_close(values, expanded[expanded < live], rtol=0, atol=0)
        padded = torch.full(
            (blocks * 8,), -torch.inf, dtype=scores.dtype, device=scores.device
        )
        padded[:live] = scores[row, :live]
        block_scores = padded.view(blocks, 8).amax(1)
        torch.testing.assert_close(
            block_scores[winners[:-1]].sort(descending=True).values,
            block_scores[:-1].topk(min(2047, blocks - 1)).values,
            rtol=0,
            atol=0,
        )


def _run_mxfp4_form(args, profile, form):
    device = torch.device("cuda", args.device)
    rows_capacity = 256 if args.mode in ("prefill", "extend") else 64
    if max(args.rows) > rows_capacity:
        raise ValueError(f"{args.mode} profile capacity is {rows_capacity} rows")
    planned_pages = (args.max_model_len + profile.block_size - 1) // profile.block_size
    capacity = planned_pages * form.page_size
    live_contexts = [context // form.ratio for context in args.contexts]
    max_live = max(live_contexts)
    if max_live > capacity or min(live_contexts) < 1:
        raise ValueError("compressed live contexts must fit the planned index capacity")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = (
        torch.randn(
            (rows_capacity, form.heads, 128), generator=generator, device=device
        )
        / 4
    ).bfloat16()
    source_q = (
        torch.randn(
            (rows_capacity, form.heads, 128), generator=generator, device=device
        )
        / 4
    ).bfloat16()
    keys = (
        torch.randn((max_live, 128), generator=generator, device=device) / 4
    ).bfloat16()
    weights = (
        torch.randn((rows_capacity, form.heads), generator=generator, device=device)
        / 32
    ).bfloat16()
    source_weights = (
        torch.randn((rows_capacity, form.heads), generator=generator, device=device)
        / 32
    ).bfloat16()
    packed, scales, decoded_q = quantized_reference(q)
    source_packed, source_scales, decoded_source = quantized_reference(source_q)
    _, _, decoded_k = quantized_reference(keys)
    native_q, native_scales = torch.empty_like(packed), torch.empty_like(scales)
    native_source, native_source_scales = (
        torch.empty_like(source_packed),
        torch.empty_like(source_scales),
    )
    page_bytes = api.index_mxfp4_page_bytes(form.page_size)
    page_stride = args.page_stride or ((page_bytes + 255) // 256 * 256)
    backing, pool, physical, table, base = _pool_and_table(
        keys=keys,
        page_size=form.page_size,
        page_bytes=page_bytes,
        page_stride=page_stride,
        capacity=capacity,
        high_pid=args.high_pid,
        generator=generator,
    )
    table = table.expand(rows_capacity, -1).clone()
    slots = (
        physical[torch.arange(max_live, device=device) // form.page_size].long()
        * form.page_size
        + torch.arange(max_live, device=device) % form.page_size
    )
    active = torch.full((1,), capacity, dtype=torch.int32, device=device)
    lengths = torch.full((rows_capacity,), max_live, dtype=torch.int32, device=device)
    output = torch.empty((rows_capacity, form.topk), dtype=torch.int32, device=device)
    output_scores = torch.empty(
        (rows_capacity, form.topk), dtype=torch.float32, device=device
    )
    source_output = torch.empty_like(output)
    source_scores = torch.empty_like(output_scores)
    candidate_output = torch.empty(
        (rows_capacity, 16384), dtype=torch.int32, device=device
    )
    candidate_lengths = torch.empty((rows_capacity,), dtype=torch.int32, device=device)
    plan_kwargs = dict(
        device=device,
        num_q_heads=form.heads,
        max_q_rows=rows_capacity,
        max_page_table_width=planned_pages,
        topk=form.topk,
        mode="prefill" if args.mode == "extend" else args.mode,
        cache_format="mxfp4",
        page_size=form.page_size,
    )
    source_form = form.name == "c1-source"
    reindex_form = form.name == "c1-reindex"
    if source_form:
        plan_kwargs["candidate_topk_blocks"] = form.candidate_topk_blocks
    elif reindex_form:
        plan_kwargs["max_candidates"] = form.max_candidates
    plan = api.plan(api.Caps(**plan_kwargs))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    source_plan = source_scratch = None
    if reindex_form:
        source_caps = {
            **plan_kwargs,
            "max_candidates": 0,
            "candidate_topk_blocks": 2048,
        }
        source_plan = api.plan(api.Caps(**source_caps))
        (source_spec,) = source_plan.scratch_specs()
        source_scratch = torch.empty(
            source_spec.shape, dtype=source_spec.dtype, device=device
        )

    def publisher_args(rows):
        assert source_plan is not None and source_scratch is not None
        return dict(
            scratch=source_scratch,
            q_mxfp4=native_source[:rows],
            q_scales=native_source_scales[:rows],
            query_weights=source_weights[:rows],
            index_k_cache=pool,
            page_table=table[:rows],
            cache_lengths=lengths[:rows],
            active_width=active,
            output_indices=source_output[:rows],
            output_scores=source_scores[:rows],
            candidate_output=candidate_output[:rows],
            candidate_output_lengths=candidate_lengths[:rows],
        )

    def bind_args(rows, *, source=False):
        kwargs = dict(
            scratch=scratch,
            q_mxfp4=(native_source if source else native_q)[:rows],
            q_scales=(native_source_scales if source else native_scales)[:rows],
            query_weights=(source_weights if source else weights)[:rows],
            index_k_cache=pool,
            page_table=table[:rows],
            cache_lengths=lengths[:rows],
            active_width=active,
            output_indices=(source_output if source else output)[:rows],
            output_scores=(source_scores if source else output_scores)[:rows],
        )
        if source_form:
            kwargs.update(
                candidate_output=candidate_output[:rows],
                candidate_output_lengths=candidate_lengths[:rows],
            )
        elif reindex_form:
            kwargs.update(
                candidate_indices=candidate_output[:rows],
                candidate_lengths=candidate_lengths[:rows],
            )
        return kwargs

    def bind_publisher(rows):
        return api.bind(source_plan, **publisher_args(rows))

    def bind(rows, *, source=False):
        return api.bind(plan, **bind_args(rows, source=source))

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    if reindex_form:
        prepare_mxfp4(session, source_plan, q=source_q, keys=keys, slots=slots,
                      arguments=publisher_args(rows_capacity))
        api.run(bind_publisher(rows_capacity))
    prepare_mxfp4(session, plan, q=source_q if source_form else q, keys=keys, slots=slots,
                  arguments=bind_args(rows_capacity, source=source_form))
    api.quantize_q_mxfp4(plan, q, q_mxfp4=native_q, q_scales=native_scales)
    api.quantize_q_mxfp4(plan, source_q, q_mxfp4=native_source, q_scales=native_source_scales)
    torch.testing.assert_close(native_q, packed, rtol=0, atol=0)
    torch.testing.assert_close(native_scales, scales, rtol=0, atol=0)
    records = []
    # Resolve once at the fixed capacity.  Each later live-row binding is a view
    # over the same storage and reuses the frozen selected kernels.
    if reindex_form:
        publisher_warm = bind_publisher(rows_capacity)
        api.score(publisher_warm)
        api.select(publisher_warm)
    warm = bind(rows_capacity, source=source_form)
    api.score(warm)
    api.select(warm)
    torch.cuda.synchronize(device)
    session.freeze()
    graphs = {}
    try:
        for live_context, visible in zip(args.contexts, live_contexts, strict=True):
            lengths.fill_(visible)
            for rows in args.rows:
                source_binding = bind(rows, source=source_form)
                binding = bind(rows)
                if source_form:
                    source_expected = _mxfp4_oracle(
                        decoded_source,
                        decoded_k,
                        source_weights,
                        rows,
                        visible,
                        capacity,
                    )
                    actual = api.score(source_binding)
                    torch.testing.assert_close(actual, source_expected, rtol=0, atol=0)
                    api.select(source_binding)
                    check_selection(
                        source_expected, source_output[:rows], source_scores[:rows]
                    )
                    _assert_published_blocks(
                        source_expected[:, :visible],
                        candidate_output[:rows],
                        candidate_lengths[:rows],
                        visible,
                    )
                    score_graph, full_graph = (
                        torch.cuda.CUDAGraph(),
                        torch.cuda.CUDAGraph(),
                    )
                    with torch.cuda.graph(score_graph):
                        api.score(source_binding)
                    with torch.cuda.graph(full_graph):
                        api.score(source_binding)
                        api.select(source_binding)
                    graphs = {
                        "prepared_qk_score": score_graph,
                        "prepared_qk_score_select_publish": full_graph,
                    }
                elif reindex_form:
                    # Publication comes from an independent source query through
                    # the real source plan, before the consumer graph is captured.
                    source_binding = bind_publisher(rows)
                    source_expected = _mxfp4_oracle(
                        decoded_source,
                        decoded_k,
                        source_weights,
                        rows,
                        visible,
                        capacity,
                    )
                    source_actual = api.score(source_binding)
                    torch.testing.assert_close(
                        source_actual, source_expected, rtol=0, atol=0
                    )
                    api.select(source_binding)
                    check_selection(
                        source_expected, source_output[:rows], source_scores[:rows]
                    )
                    _assert_published_blocks(
                        source_expected[:, :visible],
                        candidate_output[:rows],
                        candidate_lengths[:rows],
                        visible,
                    )
                    expected = _mxfp4_oracle(
                        decoded_q,
                        decoded_k,
                        weights,
                        rows,
                        visible,
                        capacity,
                        candidate_output,
                        candidate_lengths,
                    )
                    actual = api.score(binding)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    api.select(binding)
                    check_selection(
                        expected,
                        output[:rows],
                        output_scores[:rows],
                        candidate_output[:rows],
                    )
                    source_graph, source_full, score_graph, full_graph = (
                        torch.cuda.CUDAGraph() for _ in range(4)
                    )
                    with torch.cuda.graph(source_graph):
                        api.score(source_binding)
                    with torch.cuda.graph(source_full):
                        api.score(source_binding)
                        api.select(source_binding)
                    with torch.cuda.graph(score_graph):
                        api.score(binding)
                    with torch.cuda.graph(full_graph):
                        api.score(binding)
                        api.select(binding)
                    graphs = {
                        "source_score": source_graph,
                        "source_score_select_publish": source_full,
                        "reindex_score": score_graph,
                        "reindex_score_select": full_graph,
                    }
                else:
                    expected = _mxfp4_oracle(
                        decoded_q, decoded_k, weights, rows, visible, capacity
                    )
                    actual = api.score(binding)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    api.select(binding)
                    check_selection(expected, output[:rows], output_scores[:rows])
                    score_graph, full_graph = (
                        torch.cuda.CUDAGraph(),
                        torch.cuda.CUDAGraph(),
                    )
                    with torch.cuda.graph(score_graph):
                        api.score(binding)
                    with torch.cuda.graph(full_graph):
                        api.score(binding)
                        api.select(binding)
                    graphs = {
                        "prepared_qk_score": score_graph,
                        "prepared_qk_score_select": full_graph,
                    }
                # Mutate runtime metadata and inputs under the captured path.
                # Inspect the tensors written by replay, never an eager rerun.
                lengths[0] = 0
                if rows > 1:
                    lengths[1] = min(31, visible)
                weights[:rows].mul_(-0.5)
                source_weights[:rows].mul_(-0.5)
                if source_form or reindex_form:
                    publisher_graph = graphs[
                        "prepared_qk_score_select_publish"
                        if source_form
                        else "source_score_select_publish"
                    ]
                    publisher_graph.replay()
                    torch.cuda.synchronize(device)
                    source_expected = _mxfp4_oracle(
                        decoded_source,
                        decoded_k,
                        source_weights,
                        rows,
                        visible,
                        capacity,
                        lengths=lengths,
                    )
                    torch.testing.assert_close(
                        actual if source_form else source_actual,
                        source_expected,
                        rtol=0,
                        atol=0,
                    )
                    check_selection(
                        source_expected, source_output[:rows], source_scores[:rows]
                    )
                    _assert_published_blocks(
                        source_expected,
                        candidate_output[:rows],
                        candidate_lengths[:rows],
                        lengths,
                    )
                if not source_form:
                    graphs[
                        "reindex_score_select"
                        if reindex_form
                        else "prepared_qk_score_select"
                    ].replay()
                    torch.cuda.synchronize(device)
                    expected = _mxfp4_oracle(
                        decoded_q,
                        decoded_k,
                        weights,
                        rows,
                        visible,
                        capacity,
                        candidate_output if reindex_form else None,
                        candidate_lengths if reindex_form else None,
                        lengths=lengths,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    check_selection(
                        expected,
                        output[:rows],
                        output_scores[:rows],
                        candidate_output[:rows] if reindex_form else None,
                    )
                lengths.fill_(visible)
                weights[:rows].mul_(-2)
                source_weights[:rows].mul_(-2)
                if source_form or reindex_form:
                    publisher_graph.replay()
                if not source_form:
                    graphs[
                        "reindex_score_select"
                        if reindex_form
                        else "prepared_qk_score_select"
                    ].replay()
                torch.cuda.synchronize(device)
                allocation_count = torch.cuda.memory_stats(device)[
                    "allocation.all.allocated"
                ]
                gpu_before_timing = nvidia_smi_gpu_mode_snapshot()
                before = torch.cuda.memory_allocated(device)
                addresses = (pool.data_ptr(), scratch.data_ptr(), output.data_ptr())
                samples = _timed(graphs, args, device)
                after = torch.cuda.memory_allocated(device)
                assert before == after and addresses == (
                    pool.data_ptr(),
                    scratch.data_ptr(),
                    output.data_ptr(),
                )
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocation_count
                )
                records.append(
                    {
                        "profile": profile.name,
                        "form": form.name,
                        "layers": form.layers,
                        "mode": args.mode,
                        "rows": rows,
                        "live_context": live_context,
                        "visible_index_states": visible,
                        "planned_max_model_len": args.max_model_len,
                        "planned_index_capacity": capacity,
                        "planned_row_capacity": rows_capacity,
                        "planning_contract": "fixed decode/prefill capacity",
                        "planned_page_table_width": planned_pages,
                        "active_width": int(active.item()),
                        "score_width": form.max_candidates
                        or planned_pages * form.page_size,
                        "cache_format": form.cache_format,
                        "index_heads_replicated": form.heads,
                        "page_size": form.page_size,
                        "native_page_bytes": page_bytes,
                        "physical_page_stride_bytes": page_stride,
                        "high_pid_byte_offset": base * page_stride,
                        "raw_samples_us": samples,
                        "median_us": {k: median(v) for k, v in samples.items()},
                        "score_sha256": tensor_hash(actual),
                        "allocation_bytes": before,
                        "correctness": "Exact independent BF16-stage MXFP4 score and top-k; source candidate block oracle; fixed-capacity frozen live-count reuse; physical page offset checks.",
                    }
                )
                records[-1]["gpu_before_timing"] = gpu_before_timing
                records[-1]["gpu_after_timing"] = nvidia_smi_gpu_mode_snapshot()
                records[-1]["replay_mutation"] = (
                    "Input weights, empty row and partial visible prefix changed under captured execution and independently checked."
                )
                records[-1]["high_pid_qualified"] = args.high_pid
                for graph in graphs.values():
                    graph.reset()
    finally:
        for graph in graphs.values():
            graph.reset()
        session.close()
    return records


def _check_v4_selection(expected, indices):
    """V4 returns native selection order, not V4.1's position-sorted order."""
    for row in range(expected.shape[0]):
        count = min(indices.shape[1], int(torch.isfinite(expected[row]).sum()))
        selected = indices[row, :count].long()
        assert selected.unique().numel() == count
        assert bool(((selected >= 0) & (selected < expected.shape[1])).all())
        values = expected[row, selected]
        torch.testing.assert_close(
            values.sort(descending=True).values,
            expected[row].topk(count).values,
            rtol=1e-5,
            atol=1e-5,
        )
        assert bool(indices[row, count:].eq(-1).all())


def _run_v4_c4(args, profile, form):
    """V4 C4 FP8 comparison using its native fused paged score/select contract."""
    device = torch.device("cuda", args.device)
    capacity = (
        (args.max_model_len + profile.block_size - 1) // profile.block_size
    ) * form.page_size
    rows_capacity = 256 if args.mode in ("prefill", "extend") else 64
    if max(args.rows) > rows_capacity:
        raise ValueError(f"{args.mode} profile capacity is {rows_capacity} rows")
    live_contexts = [context // form.ratio for context in args.contexts]
    max_live = max(live_contexts)
    g = torch.Generator(device=device).manual_seed(args.seed)
    q = (
        torch.randn((rows_capacity, form.heads, 128), generator=g, device=device) / 4
    ).to(torch.float8_e4m3fn)
    weights = (
        torch.randn((rows_capacity, form.heads), generator=g, device=device) / 32
    ).float()
    keys = (torch.randn((max_live, 128), generator=g, device=device) / 4).bfloat16()
    page_bytes = 64 * 132
    page_stride = args.page_stride or ((page_bytes + 255) // 256 * 256)
    backing, pool, physical, table, base = _pool_and_table(
        keys=keys,
        page_size=64,
        page_bytes=page_bytes,
        page_stride=page_stride,
        capacity=capacity,
        high_pid=args.high_pid,
        generator=g,
    )
    if args.mode == "decode":
        table = table.expand(rows_capacity, -1).clone()
    packed = pack_index_k_cache_reference(keys)
    pool[physical.long()] = packed
    active = torch.full((1,), capacity, dtype=torch.int32, device=device)
    lengths = torch.full((rows_capacity,), max_live, dtype=torch.int32, device=device)
    out = torch.empty((rows_capacity, form.topk), dtype=torch.int32, device=device)
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    plans, workspaces = {}, {}
    for rows in args.rows:
        caps = api.Caps(
                device=device,
                num_q_heads=form.heads,
                max_q_rows=rows,
                max_page_table_width=(capacity + 63) // 64,
                topk=form.topk,
                mode="prefill" if args.mode == "extend" else args.mode,
                cache_format="fp8",
        )
        tensors = dict(q_fp8=q[:rows], query_weights=weights[:rows], index_k_cache=pool,
                       page_table=table[:rows] if args.mode == "decode" else table,
                       cache_lengths=lengths[:rows], active_width=active, output_indices=out[:rows])
        plan = api.plan(caps, invocation=api.invocation_from_tensors(caps, **tensors))
        def prepare(state):
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                            for spec in state.layout.scratch_specs())
            trial_output = torch.empty_like(out[:rows])
            binding = state.bind(scratch=scratch, real_page_table=tensors["page_table"],
                                 cache_seqlens_int32=tensors["cache_lengths"], active_width=active,
                                 expected_num_q_heads=form.heads,
                                 shared_page_table=plan.query.shared_page_table,
                                 output_physical_slots=False)
            return PreparedCall(run=lambda: state.run(binding, q_fp8=tensors["q_fp8"],
                                query_weights=tensors["query_weights"], index_k_cache=pool,
                                output_indices=trial_output), owners=(scratch, binding, trial_output))
        session.prepare((plan.request(name="fp8-indexer", prepare_call=prepare),))
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        plans[rows], workspaces[rows] = plan, scratch
        warm = api.bind(
            plan,
            scratch=scratch,
            q_fp8=q[:rows],
            query_weights=weights[:rows],
            index_k_cache=pool,
            page_table=table[:rows] if args.mode == "decode" else table,
            cache_lengths=lengths[:rows],
            active_width=active,
            output_indices=out[:rows],
        )
        api.run(warm)
    records = []
    torch.cuda.synchronize(device)
    session.freeze()
    graph = torch.cuda.CUDAGraph()
    try:
        for live_context, visible in zip(args.contexts, live_contexts, strict=True):
            lengths.fill_(visible)
            active.fill_(visible)
            for rows in args.rows:
                plan, scratch = plans[rows], workspaces[rows]
                binding = api.bind(
                    plan,
                    scratch=scratch,
                    q_fp8=q[:rows],
                    query_weights=weights[:rows],
                    index_k_cache=pool,
                    page_table=table[:rows] if args.mode == "decode" else table,
                    cache_lengths=lengths[:rows],
                    active_width=active,
                    output_indices=out[:rows],
                    output_scores=None,
                )
                api.run(binding)
                expected = paged_decode_logits_reference(
                    q_fp8=q[:rows],
                    weights=weights[:rows],
                    index_k_cache=packed,
                    real_page_table=torch.arange(
                        packed.shape[0], dtype=torch.int32, device=device
                    )[None],
                    query_row_to_batch=torch.zeros(
                        rows, dtype=torch.int64, device=device
                    ),
                    seqlens_per_query=lengths[:rows],
                )
                _check_v4_selection(expected, out[:rows])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    api.run(binding)
                lengths[0] = 0
                if rows > 1:
                    lengths[1] = min(31, visible)
                weights[:rows].mul_(-0.5)
                graph.replay()
                torch.cuda.synchronize(device)
                mutated = paged_decode_logits_reference(
                    q_fp8=q[:rows],
                    weights=weights[:rows],
                    index_k_cache=packed,
                    real_page_table=torch.arange(
                        packed.shape[0], dtype=torch.int32, device=device
                    )[None],
                    query_row_to_batch=torch.zeros(
                        rows, dtype=torch.int64, device=device
                    ),
                    seqlens_per_query=lengths[:rows],
                )
                _check_v4_selection(mutated, out[:rows])
                lengths.fill_(visible)
                weights[:rows].mul_(-2)
                graph.replay()
                torch.cuda.synchronize(device)
                _check_v4_selection(expected, out[:rows])
                allocation_count = torch.cuda.memory_stats(device)[
                    "allocation.all.allocated"
                ]
                gpu_before_timing = nvidia_smi_gpu_mode_snapshot()
                before = torch.cuda.memory_allocated(device)
                samples = _timed({"prepared_qk_score_select": graph}, args, device)
                after = torch.cuda.memory_allocated(device)
                assert before == after
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocation_count
                )
                records.append(
                    {
                        "profile": profile.name,
                        "form": form.name,
                        "layers": form.layers,
                        "mode": args.mode,
                        "rows": rows,
                        "live_context": live_context,
                        "visible_index_states": visible,
                        "planned_max_model_len": args.max_model_len,
                        "planned_index_capacity": capacity,
                        "planned_page_table_width": table.shape[1],
                        "active_width": int(active.item()),
                        "cache_format": form.cache_format,
                        "index_heads_replicated": form.heads,
                        "page_size": 64,
                        "native_page_bytes": page_bytes,
                        "physical_page_stride_bytes": page_stride,
                        "high_pid_byte_offset": base * page_stride,
                        "raw_samples_us": samples,
                        "median_us": {k: median(v) for k, v in samples.items()},
                        "score_sha256": tensor_hash(expected),
                        "allocation_bytes": before,
                        "correctness": "Independent quantization-aware FP8 "
                        "paged oracle and native V4 fused score/select. FP8 has no "
                        "exposed staged score tensor.",
                    }
                )
                records[-1]["gpu_before_timing"] = gpu_before_timing
                records[-1]["gpu_after_timing"] = nvidia_smi_gpu_mode_snapshot()
                records[-1]["replay_mutation"] = (
                    "Weights, empty row and partial visible prefix checked against the FP8 oracle after graph replay."
                )
                records[-1]["planned_row_capacity"] = rows
                records[-1]["planning_contract"] = (
                    "one prepared plan per declared capture shape"
                )
                records[-1]["high_pid_qualified"] = args.high_pid
                graph.reset()
    finally:
        graph.reset()
        session.close()
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-profile",
        dest="profile",
        choices=("deepseek-v4-flash", "deepseek-v4.1-flash"),
        required=True,
    )
    parser.add_argument(
        "--forms", default="all", help="comma-separated profile indexer forms, or all"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--vllm-path", help="integration checkout for source provenance"
    )
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill", "extend"),
        default="decode",
        help="extend maps to V4.1's native prefill indexer plan",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="planned/captured capacity; not the live context",
    )
    parser.add_argument(
        "--contexts",
        default="16384",
        help="comma-separated live contexts within planned capacity",
    )
    parser.add_argument(
        "--rows",
        default="1,2,4,8",
        help="comma-separated live rows reusing the same frozen plan",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--page-stride",
        type=int,
        default=0,
        help="allocator page stride; default native payload aligned to 256 bytes",
    )
    parser.add_argument(
        "--high-pid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=410831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.contexts = _csv(args.contexts)
    args.rows = _csv(args.rows)
    if (
        min(
            *args.contexts,
            *args.rows,
            args.max_model_len,
            args.samples,
            args.replays,
            args.warmup,
        )
        <= 0
        or max(args.contexts) > args.max_model_len
    ):
        parser.error(
            "counts must be positive and live contexts must fit --max-model-len"
        )
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    profile = load_attention_profile(
        args.profile, args.model_path, tp_size=args.tp_size, block_size=args.block_size
    )
    requested = (
        {f.name for f in profile.indexer_forms}
        if args.forms == "all"
        else set(args.forms.split(","))
    )
    forms = [f for f in profile.indexer_forms if f.name in requested]
    if not forms or requested - {f.name for f in forms}:
        parser.error("--forms must name one or more forms in the selected profile")
    result = {
        "command": [sys.executable, *sys.argv],
        "repository": repository_state(ROOT),
        "arguments": {
            **vars(args),
            "output": str(args.output),
            "model_path": str(args.model_path) if args.model_path else None,
        },
        "profile_config": profile.config,
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl")
        },
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA_", "CUTE_", "B12X_", "OMP_"))
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "benchmarks/experimental/b12x/deepseek_attention_profiles.py",
                ROOT / "flashinfer/experimental/b12x/attention/dsa_indexer/mxfp4.py",
            )
        },
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "scope": "Native DSA plans over synthetic prepared Q/K. Timings exclude learned projections, cache writes, TP collective and whole-model forward.",
        "results": [],
    }
    result["integration"] = integration_provenance(args.vllm_path)
    result["cache_sharing"] = (
        "All query rows share one physical prefix; queries and source/reindex weights differ."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for form in forms:
        runner = _run_v4_c4 if form.cache_format == "fp8" else _run_mxfp4_form
        result["results"].extend(runner(args, profile, form))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
