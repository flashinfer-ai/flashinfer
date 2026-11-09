#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare full/public-GVR2 and selective QK-to-TopK graph replay.

Examples (original-token contexts; ratio4 consumes synthetic precompressed KV)::

    python benchmarks/bench_fp8_mqa_selective_logits.py --q 8192 --kv 8192 \
        --num-heads 64 --top-k 512 --compression-ratio 4
    python benchmarks/bench_fp8_mqa_selective_logits.py --summarize results/*.json

Per-step setup runs once per forward pass and is shared by all indexer layers.
Full builds row/group metadata and request IDs, gathers the page table, and
computes the scorer schedule. Selective binds continuous K/scales and calls
plan(): prefix copy, validation,
row metadata, scorer schedules, and sampling-metadata refresh.

Layer is one indexer TopK operation for one layer. Full calls
fp8_paged_mqa_logits followed by top_k_varlen(backend="gvr_2"); selective calls
run(), including sampling, candidate publication, selection, and exact repair.
For L indexer layers, model cost per forward pass is per-step setup + L * layer;
a standalone single call costs per-step setup + layer.

This benchmark generates seeded inputs and measures separate per-step setup
and layer CUDA graph pairs in an AxN-5s-BxN-5s sequence. Immediately before
each timed block, its own route runs continuously for --settle-seconds (2s
by default), so both orders measure sustained load after the idle pauses.
Successful runs also print a validated Markdown report; --output
remains raw JSON. The --summarize mode regenerates that report without importing
Torch or using CUDA. Inputs must share one environment, GPU and schema;
partial successful prefixes are labeled and duplicate cells are rejected.
Use --max-context-length to reserve a fixed original-token capacity while
--kv sets the live context; omission sizes each route to its live context.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
import tempfile
from pathlib import Path

# Imported helpers retain their Torch dependency; the reporting CLI is stdlib-only.
if __name__ != "__main__":
    import torch

ROWS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(samples),
        "p10_us": _percentile(samples, 0.10),
        "p90_us": _percentile(samples, 0.90),
    }


def _validated_summary(values, repeats):
    if (
        len(values) != repeats
        or not values
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in values
        )
    ):
        raise ValueError("each route requires the declared positive finite samples")
    summary = _summarize(values)
    return tuple(summary[field] for field in ("median_us", "p10_us", "p90_us"))


def _validate_pauses(pauses, repeats, reverse=False, settle_seconds=None):
    if len(pauses) != 2 or any(
        pause["after"] != label
        or pause["samples"] != repeats
        or not math.isfinite(pause["elapsed_s"])
        or pause["elapsed_s"] < 5.0
        for label, pause in zip(
            ("b", "a") if reverse else ("a", "b"), pauses, strict=True
        )
    ):
        raise ValueError("expected two synchronized arm pauses")
    if settle_seconds is not None:
        if not math.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError("settling warmup must be finite and nonnegative")
        for pause in pauses:
            warmup = pause.get("pre_block_warmup", {})
            if (
                warmup.get("route") != pause["after"]
                or warmup.get("requested_s") != settle_seconds
                or not math.isfinite(warmup.get("elapsed_s", float("nan")))
                or warmup["elapsed_s"] < settle_seconds
                or type(warmup.get("replays")) is not int
                or warmup["replays"] < (1 if settle_seconds > 0 else 0)
                or warmup.get("start_utc_ns", -1) <= 0
                or warmup.get("end_utc_ns", -1) < warmup["start_utc_ns"]
                or warmup["end_utc_ns"] > pause["block_start_utc_ns"]
            ):
                raise ValueError("expected same-route warmup before each timed block")


def _validated_routes(component, repeats):
    summaries = []
    for route in ("full_gvr2", "selective"):
        summary = _validated_summary(component[route]["raw_us"], repeats)
        for field, value in zip(
            ("median_us", "p10_us", "p90_us"), summary, strict=True
        ):
            if not math.isclose(
                component[route][field], value, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ValueError(f"raw/summary mismatch: {route}, {field}")
        summaries.append(summary)
    return summaries


def render_summary(paths):
    """Render one environment/GPU's block-paired JSON without CUDA work."""
    return _render_results((str(path), json.loads(path.read_text())) for path in paths)


def _render_results(results):
    rows, identities, seen, partial = [], set(), set(), []
    for path, result in results:
        schema = result["schema_version"]
        if schema not in (21, 22):
            raise ValueError(f"expected block-paired schema 21 or 22: {path}")
        identities.add(
            (
                result["device"]["name"],
                json.dumps(result["environment"], sort_keys=True),
                schema,
                result["device"].get("uuid", result["device"]["name"]),
            )
        )
        if result["status"] != "complete":
            partial.append(str(path))
        for row in result["rows"]:
            key = tuple(
                row[field]
                for field in (
                    "bs",
                    "q",
                    "num_heads",
                    "top_k",
                    "compression_ratio",
                    "kv",
                )
            )
            if key in seen:
                raise ValueError(f"duplicate comparison: {key}")
            seen.add(key)
            protocol = row["protocol"]
            if (
                protocol["order"] not in ("AxN-5s-BxN-5s", "BxN-5s-AxN-5s")
                or type(protocol["samples_per_route"]) is not int
                or protocol["samples_per_route"] <= 0
                or protocol["samples_per_block"] != protocol["samples_per_route"]
                or protocol["sleep_between_samples"] is not False
                or protocol["a"] != "full_gvr2"
                or protocol["b"] != "selective"
            ):
                raise ValueError(f"unexpected timing protocol: {key}")
            repeats = protocol["samples_per_route"]
            reverse = protocol["order"] == "BxN-5s-AxN-5s"
            settling = (
                protocol["steady_state_warmup_requested_s"] if schema >= 17 else None
            )
            _validate_pauses(
                protocol["post_component_pauses"], repeats, reverse, settling
            )
            if row["correctness"]["exact_error"] != 0:
                raise ValueError(f"exact repair failed: {key}")
            summaries = _validated_routes(row, repeats)
            step_summaries = None
            if schema == 21:
                if not row.get("step"):
                    raise ValueError(
                        "setup/layer schema requires per-step setup samples"
                    )
                _validate_pauses(
                    row["step"]["post_component_pauses"], repeats, reverse, settling
                )
                step_summaries = _validated_routes(row["step"], repeats)
            elif row.get("step") is not None:
                raise ValueError(
                    "layer-only schema must omit per-step setup measurements"
                )
            rows.append((row, summaries, step_summaries))
    if len(identities) != 1 or not rows:
        raise ValueError("expected nonempty results from one environment/GPU")
    identity = next(iter(identities))
    settling_modes = {
        row["protocol"].get("steady_state_warmup_requested_s", 0.0)
        for row, _, _ in rows
    }
    if len(settling_modes) != 1:
        raise ValueError("cannot combine different pre-block warmup durations")
    settling_seconds = next(iter(settling_modes))
    sample_counts = {row["protocol"]["samples_per_route"] for row, _, _ in rows}
    median_scope = (
        f"median of {next(iter(sample_counts))} consecutive replays"
        if len(sample_counts) == 1
        else "medians of the configured consecutive replay blocks"
    )
    lines = [
        f"GPU `{identity[0]}`. {len(rows)} comparisons.",
        "",
        f"Latency in ms: {median_scope}.",
        f"Same-route settling warmup immediately before each timed block: {settling_seconds:g} seconds; five-second pauses remain after each block.",
        "",
        "Per-step setup runs once per forward pass and is shared across indexer layers: full builds row/group metadata and request IDs, gathers the page table and computes the scorer schedule; selective calls plan() for "
        + "prefix copy"
        + ", validation, metadata, schedules and sampling-metadata refresh.",
        "Layer is one indexer TopK operation: FP8 full calls fp8_paged_mqa_logits + top_k_varlen(gvr_2); block-scaled full uses a PyTorch FP32 dense fallback; selective calls run().",
        "For L indexer layers, model cost per forward pass is per-step setup + L * layer; a standalone single call costs per-step setup + layer.",
    ]
    rows.sort(
        key=lambda item: (
            -item[0]["q"],
            item[0]["bs"],
            item[0]["num_heads"],
            item[0]["top_k"],
            item[0]["compression_ratio"],
            item[0]["kv"],
        )
    )
    groups = {}
    for item in rows:
        row = item[0]
        groups.setdefault((row["bs"], row["q"]), []).append(item)
    for (batch, query), group in groups.items():
        mode = "Decode" if query == 1 else "Prefill"
        dimension = "MTP (Q/request)" if query == 1 else "Chunk Size"
        profiles = sorted(
            {
                (row["num_heads"], row["top_k"], row["compression_ratio"])
                for row, _, _ in group
            }
        )
        show_top_k = len({(heads, ratio) for heads, _, ratio in profiles}) != len(
            profiles
        )
        heads_label = "Heads / TopK" if show_top_k else "Heads"
        profile_text = "; ".join(
            f"H{heads}/K{top_k}/R{ratio}" for heads, top_k, ratio in profiles
        )
        lines.extend(
            [
                "",
                f"#### {mode} (B{batch}/Q{query})",
                "",
                "Profiles: " + profile_text + ".",
                "",
                f"| {heads_label} | Compression Ratio | {dimension} | Original context | Full indexer + GVRv2 layer (ms) | Indexer-throwaway + TopK layer (ms) | Layer speedup |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row, summaries, _ in group:
            context = str(row["kv"])
            if row["kv"] % 2**20 == 0:
                context = f"{row['kv'] // 2**20}M"
            elif row["kv"] % 1024 == 0:
                context = f"{row['kv'] // 1024}K"
            heads = (
                f"{row['num_heads']} / {row['top_k']}"
                if show_top_k
                else str(row["num_heads"])
            )
            size = query if query == 1 else batch * query
            lines.append(
                f"| {heads} | {row['compression_ratio']} | {size} | {context} | "
                f"{summaries[0][0] / 1000:.6f} | {summaries[1][0] / 1000:.6f} | "
                f"{summaries[0][0] / summaries[1][0]:.4f}x |"
            )
    lines.extend(
        [
            "",
            "Per-row AxN-5s-BxN-5s or reverse order, as recorded in protocol.order; one event-timed replay per sample, no deliberate intra-arm sleep. Host event/enqueue gaps remain. Device snapshots do not establish power isolation.",
        ]
    )
    if identity[2] == 21:
        lines.extend(
            [
                "",
                "### Per-step setup (ms)",
                "",
                "Selective includes device validation and table copy; full does not provide equivalent validation.",
                "",
                "| B / Q / H / K / ratio / KV | Full indexer + GVRv2 per-step setup | Indexer-throwaway + TopK per-step setup |",
                "| --- | --- | --- |",
            ]
        )
        for row, _, steps in rows:
            label = " / ".join(
                str(row[k])
                for k in ("bs", "q", "num_heads", "top_k", "compression_ratio", "kv")
            )
            lines.append(
                f"| {label} | {steps[0][0] / 1000:.6f} | {steps[1][0] / 1000:.6f} |"
            )
    if partial:
        lines.extend(["", "Partial successful prefixes only: " + ", ".join(partial)])
    return "\n".join(lines)


def _record_event(call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def _paired_events(
    call_a, call_b, warmups: int, repeats: int, *, reverse=False, settle_seconds=2.0
):
    """After warmup, run A×N, sleep 5s, B×N, sleep 5s on identical inputs.

    Before each block, replay that block's route for the requested settling
    duration and synchronize it. Each sample times one graph replay with its
    own completed CUDA end event.
    There is no deliberate sleep between samples in an arm. Synchronize before
    each arm's pause, including the final B pause, and return both durations
    separately from GPU latency. Host event/enqueue overhead remains between replays.
    """
    if repeats <= 0:
        raise ValueError("paired measurement requires a positive repeat count")
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise ValueError("settle_seconds must be finite and nonnegative")
    for repeat in range(warmups):
        first, second = (call_a, call_b) if repeat % 2 == 0 else (call_b, call_a)
        first()
        second()
    torch.cuda.synchronize()

    samples_a: list[float] = []
    samples_b: list[float] = []
    pauses: list[dict] = []
    order = (("a", call_a), ("b", call_b))
    if reverse:
        order = tuple(reversed(order))
    for label, call in order:
        warmup_start_ns = time.time_ns()
        warmup_start = time.perf_counter()
        warmup_replays = 0
        while time.perf_counter() - warmup_start < settle_seconds:
            call()
            torch.cuda.synchronize()
            warmup_replays += 1
        warmup_elapsed = time.perf_counter() - warmup_start
        warmup_end_ns = time.time_ns()
        block_start_ns = time.time_ns()
        for _ in range(repeats):
            sample = _record_event(call)
            (samples_a if label == "a" else samples_b).append(sample)
        torch.cuda.synchronize()
        block_end_ns = time.time_ns()
        sleep_start = time.perf_counter()
        time.sleep(5.0)
        pauses.append(
            {
                "after": label,
                "samples": repeats,
                "elapsed_s": time.perf_counter() - sleep_start,
                "block_start_utc_ns": block_start_ns,
                "block_end_utc_ns": block_end_ns,
                "pre_block_warmup": {
                    "route": label,
                    "requested_s": settle_seconds,
                    "elapsed_s": warmup_elapsed,
                    "replays": warmup_replays,
                    "start_utc_ns": warmup_start_ns,
                    "end_utc_ns": warmup_end_ns,
                },
            }
        )
    return samples_a, samples_b, pauses


def _save_result(path, result):
    """Atomically replace this invocation's already-reserved output file."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(result, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = temporary.name
    os.replace(temporary_path, path)


def _quantize(rows: int, columns: int, *, generator: torch.Generator):
    source = (
        torch.randn(
            (rows, columns), device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.1
    )
    amax = source.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return (source / scale[:, None]).to(torch.float8_e4m3fn), scale


def _fuse_paged_kv(
    kv: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pack contiguous KV and scales for the paged-logits comparison."""
    head_dim = kv.shape[1]
    valid_rows = kv.shape[0]
    num_blocks = math.ceil(valid_rows / block_size)
    padded_rows = num_blocks * block_size
    if padded_rows != valid_rows:
        padded_kv = torch.zeros(
            (padded_rows, head_dim), device=kv.device, dtype=kv.dtype
        )
        padded_scale = torch.full(
            (padded_rows, *scale.shape[1:]),
            127 if scale.dtype == torch.uint8 else 1,
            device=scale.device,
            dtype=scale.dtype,
        )
        padded_kv[:valid_rows] = kv
        padded_scale[:valid_rows] = scale
        kv, scale = padded_kv, padded_scale
    scale_offset = block_size * head_dim
    fused = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=kv.device,
        dtype=torch.uint8,
    )
    fused[:, :scale_offset] = (
        kv.view(num_blocks, block_size, head_dim)
        .view(torch.uint8)
        .reshape(num_blocks, scale_offset)
    )
    fused[:, scale_offset:] = (
        scale.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def _decode_mxfp4(packed, scales=None):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
        dtype=torch.float32,
    )
    decoded = torch.stack(
        (values[(packed & 15).long()], values[(packed >> 4).long()]), -1
    ).flatten(-2)
    if scales is not None:
        decoded.mul_(torch.pow(2.0, scales.float() - 127).repeat_interleave(32, -1))
    return decoded


def _model_quantize(rows, columns, *, generator, fp4):
    """Standard-normal model inputs; linear MXFP4 scales use the public encoder."""
    source = torch.randn(
        (rows, columns), device="cuda", dtype=torch.float32, generator=generator
    )
    if not fp4:
        return source.to(torch.float8_e4m3fn), None
    from flashinfer.quantization.fp4_quantization import mxfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    packed, scales = mxfp4_quantize(
        source.to(torch.bfloat16), backend="cuda", sfLayout=SfLayout.layout_linear
    )
    return packed.view(rows, columns // 2), scales.view(rows, columns // 32)


@dataclass(frozen=True)
class _ReferenceGroups:
    """Immutable host metadata for request-local full-score query grouping.

    Use :meth:`build` to validate prefixes and construct the fields. Query
    prefixes count real rows; token prefixes count original, uncompressed
    tokens. All stored KV ends and ``physical_prefix`` use physical KV slots.
    No device tensors, query payloads or page tables are allocated here.

    ``rows`` contains ``(output_base, valid_queries, *physical_ends)`` for each
    group, with zero ends in padded slots. ``group_ends`` and ``request_ids``
    give each group's maximum physical end and originating request.
    ``gather_rows`` maps flattened group slots to real input query rows;
    padded slots repeat a valid row in that group and are identified by
    ``invalid_queries`` so callers can zero their gathered Q/weights.
    ``real_ends`` contains one physical end per real output row.
    ``physical_prefix`` accumulates each request's floored physical length.
    Empty-query requests produce no groups. Grouping never crosses a request.
    """

    query_tile: int
    q_prefix: tuple
    token_prefix: tuple
    ratio: int
    rows: tuple
    group_ends: tuple
    request_ids: tuple
    gather_rows: tuple
    invalid_queries: tuple
    real_ends: tuple
    physical_prefix: tuple

    @classmethod
    def build(cls, q_prefix, token_prefix, ratio, query_tile):
        """Build grouping metadata from cumulative host integer prefixes.

        ``q_prefix`` and ``token_prefix`` must have the same batch-plus-one
        length, start at zero and be nondecreasing. A request with Q queries
        and L original KV tokens requires ``L >= Q``. For ratio 1 or 4, query
        i's exclusive physical end is ``(L - Q + i + 1) // ratio``; physical
        request lengths are floored before forming their cumulative prefix.

        ``query_tile`` is a positive integer number of queries per group.
        The last group of a request may contain padded slots, but output
        bases and valid counts cover only real rows. Return an _ReferenceGroups
        containing tuples independent of the input sequences; raise ValueError
        for invalid prefix, ratio, grouping or row-encoding contracts.
        """
        if type(ratio) is not int or ratio not in (1, 4):
            raise ValueError("ratio must be 1 or 4")
        if type(query_tile) is not int or query_tile <= 0:
            raise ValueError("positive integer query tile required")
        q_prefix, token_prefix = tuple(q_prefix), tuple(token_prefix)
        if len(q_prefix) < 2 or len(q_prefix) != len(token_prefix):
            raise ValueError("matching request prefixes required")
        for prefix in (q_prefix, token_prefix):
            if (
                any(type(x) is not int for x in prefix)
                or prefix[0] != 0
                or any(a > b for a, b in zip(prefix, prefix[1:], strict=False))
            ):
                raise ValueError(
                    "nondecreasing integer prefixes starting at zero required"
                )
        rows, maxima, requests, gathers, invalid, real_ends = [], [], [], [], [], []
        physical = [0]
        for req, (qb, qe, kb, ke) in enumerate(
            zip(q_prefix, q_prefix[1:], token_prefix, token_prefix[1:], strict=False)
        ):
            count, tokens = qe - qb, ke - kb
            if tokens < count:
                raise ValueError("original KV length must be at least Q length")
            physical.append(physical[-1] + tokens // ratio)
            ends = tuple((tokens - count + i + 1) // ratio for i in range(count))
            real_ends.extend(ends)
            for first in range(0, count, query_tile):
                valid = min(query_tile, count - first)
                slot_ends = ends[first : first + valid] + (0,) * (query_tile - valid)
                rows.append((qb + first, valid, *slot_ends))
                maxima.append(max(slot_ends))
                requests.append(req)
                # Dummy slots stay within this request/group, then are zeroed.
                gathers.extend(
                    qb + first + t if t < valid else qb + first
                    for t in range(query_tile)
                )
                invalid.extend(t >= valid for t in range(query_tile))
        return cls(
            query_tile,
            q_prefix,
            token_prefix,
            ratio,
            tuple(rows),
            tuple(maxima),
            tuple(requests),
            tuple(gathers),
            tuple(invalid),
            tuple(real_ends),
            tuple(physical),
        )


class _FullGVR:
    """Public scorer/GVR composition with separate step and layer callables.

    Uniform requests must divide evenly into the configured query tile. Q and
    weights then use address-stable views, without cross-request grouping.
    Query topology is fixed; every step derives physical row/group bounds and
    request IDs from device prefixes, gathers the live map and refreshes the
    public scorer schedule. Layer replay consumes those address-stable owners.

    FP8 uses the public paged scorer and GVR. Block-scaled encodings use
    a PyTorch FP32 dense scorer followed by GVR.
    """

    def __init__(
        self,
        q,
        kv_fused,
        weights,
        block_table,
        groups,
        top_k,
        *,
        qo_indptr,
        kv_indptr,
        q_scale=None,
        physical_capacity=None,
    ):
        from flashinfer import top_k_varlen
        from flashinfer.attn_scores import (
            compute_paged_mqa_logits_schedule,
            fp8_paged_mqa_logits,
        )
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import workspace_bytes

        device = q.device
        width = max(
            b - a
            for a, b in zip(
                groups.physical_prefix, groups.physical_prefix[1:], strict=False
            )
        )
        if physical_capacity is not None:
            if physical_capacity < width:
                raise ValueError("Full capacity must cover the live physical context")
            width = physical_capacity
        rows, heads, dimension = q.shape
        tile = groups.query_tile
        causal_envelope = groups.ratio != 1 and tile > 1
        self.scorer_mode = (
            "ordinary-causal-envelope" if causal_envelope else "ordinary-causal"
        )
        self.scorer_width = width + tile - 1 if width and causal_envelope else width
        self.output = torch.empty((rows, top_k), device=device, dtype=torch.int32)
        self.workspace_owners = ()
        output = self.output
        self.scores = None
        if not width:
            self.step = lambda: None
            self.layer = lambda: output.fill_(-1)
            return
        if rows == 1 and width % 4:
            raise ValueError(
                "public GVR2 requires singleton score width divisible by four"
            )
        columns = max(8192, top_k, ((self.scorer_width + 255) // 256 * 256))
        self.scores = torch.empty((rows, columns), device=device, dtype=torch.float32)
        self.scorer_chunks = []
        scores = self.scores
        logical_scores = scores[:, :width]
        row_ids = torch.arange(rows, device=device, dtype=torch.int32)
        row_ends = torch.empty((rows,), device=device, dtype=torch.int32)
        group_ends = torch.empty((len(groups.rows),), device=device, dtype=torch.int32)
        request_ids = torch.empty((len(groups.rows),), device=device, dtype=torch.int64)
        table_rows = torch.empty(
            (len(groups.rows), block_table.shape[1] + int(causal_envelope)),
            device=device,
            dtype=torch.int32,
        )
        gathered_table = table_rows[:, : block_table.shape[1]]
        if causal_envelope:
            # Page zero is a real pool page. Padding scores are outside GVR's
            # logical prefixes, so its contents never affect the selection.
            table_rows[:, -1].zero_()
        schedule = torch.empty(
            (torch.cuda.get_device_properties(device).multi_processor_count + 1, 2),
            device=device,
            dtype=torch.int32,
        )
        workspace = {
            "gvr2_workspace": torch.zeros(
                workspace_bytes(), device=device, dtype=torch.uint8
            ).view(torch.int32)
        }
        padded_heads = (
            max(8, 1 << (heads - 1).bit_length()) if heads * tile % 8 else heads
        )
        padded_q = torch.zeros(
            (rows, padded_heads, dimension), device=device, dtype=q.dtype
        )
        padded_weights = torch.zeros(
            (rows, padded_heads), device=device, dtype=weights.dtype
        )
        grouped_q = padded_q.view(len(groups.rows), tile, padded_heads, dimension)
        scorer_width = self.scorer_width
        self.workspace_owners = (
            scores,
            logical_scores,
            row_ends,
            group_ends,
            request_ids,
            row_ids,
            table_rows,
            gathered_table,
            schedule,
            workspace,
            self.scorer_chunks,
        )

        def step():
            # These vectorized operations execute on every replay; none of the
            # length-dependent metadata is a host-precomputed graph constant.
            requests = torch.searchsorted(qo_indptr[1:], row_ids, right=True)
            query_starts = qo_indptr[requests]
            query_lengths = qo_indptr[requests + 1] - query_starts
            token_lengths = kv_indptr[requests + 1] - kv_indptr[requests]
            row_ends.copy_(
                torch.div(
                    token_lengths - query_lengths + row_ids - query_starts + 1,
                    groups.ratio,
                    rounding_mode="floor",
                )
            )
            torch.amax(row_ends.view(-1, tile), dim=1, out=group_ends)
            if causal_envelope:
                # Ordinary row t promises original_group_max + t, covering
                # every required physical end <= original_group_max.
                group_ends.add_(tile - 1)
            request_ids.copy_(requests.view(-1, tile)[:, 0])
            torch.index_select(block_table, 0, request_ids, out=gathered_table)
            compute_paged_mqa_logits_schedule(
                group_ends, device=device, next_n=tile, variant="fp8", out=schedule
            )

        def layer():
            padded_q[:, :heads].copy_(q)
            padded_weights[:, :heads].copy_(weights)
            fp8_paged_mqa_logits(
                grouped_q,
                kv_fused,
                padded_weights,
                table_rows,
                group_ends,
                scorer_width,
                output_dtype=torch.float32,
                epi_dtype=torch.float32,
                schedule_meta=schedule,
                out=scores,
            )
            top_k_varlen(
                logical_scores,
                row_ends,
                top_k,
                next_n=1,
                compress_ratio=1,
                backend="gvr_2",
                return_values=False,
                out_indices=output,
                workspace=workspace,
            )
            return output

        self.step, self.layer = step, layer


class _DenseGVR:
    """PyTorch FP32 dense fallback for block-scaled encodings."""

    def __init__(self, q, keys, scales, weights, groups, top_k):
        from flashinfer import top_k_varlen
        from flashinfer.topk_varlen.kernels.gvr2_topk_host import workspace_bytes

        if len(keys) != 1:
            raise ValueError("dense fallback supports one request")
        rows, width = q.shape[0], keys[0].shape[0]
        self.output = torch.empty((rows, top_k), device=q.device, dtype=torch.int32)
        self.scores = torch.empty((rows, max(width, top_k, 8192)), device=q.device)
        self.scorer_chunks = []
        self.scorer_width = width
        self.scorer_mode = "pytorch-fp32-dense"
        ends = torch.tensor(groups.real_ends, device=q.device, dtype=torch.int32)
        workspace = {
            "gvr2_workspace": torch.zeros(
                workspace_bytes(), device=q.device, dtype=torch.uint8
            ).view(torch.int32)
        }
        query = q.float()
        key = keys[0].float()
        self.workspace_owners = (ends, workspace, query, key)
        self.step = lambda: None

        def layer():
            scores = self.scores[:, :width]
            scores.zero_()
            for head in range(query.shape[1]):
                dots = query[:, head, :] @ key.T
                scores.add_(dots.relu() * weights[:, head, None])
            scores.mul_(scales[0][None, :])
            top_k_varlen(
                self.scores[:, :width],
                ends,
                top_k,
                next_n=1,
                compress_ratio=1,
                backend="gvr_2",
                return_values=False,
                out_indices=self.output,
                workspace=workspace,
            )
            return self.output

        self.layer = layer


def _check_reference(
    outputs,
    q,
    kv,
    scales,
    weights,
    groups,
    top_k,
    *,
    atol,
    rtol,
    logits_dtype=None,
    native_scores=None,
    math_mode=None,
    encoding="fp8",
):
    """Check independent FP32 math and exact native-rounded selection per row.

    Independent FP32 reductions can straddle a lower-precision midpoint.
    Validate the dense producer against independent math before using its
    rounded scores as the exact lower-precision selection reference.
    """
    maximum_error = {name: 0.0 for name in outputs}
    rounded_native = logits_dtype not in (None, torch.float32)
    if rounded_native and native_scores is not None:
        if native_scores.dtype != torch.float32:
            raise ValueError("native-rounded checks require FP32 producer scores")
        maximum_error["native_producer_fp32"] = 0.0
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for request, (qb, qe) in enumerate(
            zip(groups.q_prefix, groups.q_prefix[1:], strict=False)
        ):
            key = kv[request].float()
            scale = scales[request]
            for first in range(qb, qe, 32):
                stop = min(first + 32, qe)
                query = q[first:stop].float()
                weight = weights[first:stop]
                ends = groups.real_ends[first:stop]
                reference = torch.empty(
                    (stop - first, key.shape[0]), device=q.device, dtype=torch.float32
                )
                native_reference = (
                    torch.empty_like(reference)
                    if math_mode in ("native_bf16", "native_fp16")
                    else None
                )
                for kb in range(0, key.shape[0], 4096):
                    ke = min(kb + 4096, key.shape[0])
                    dots = torch.einsum("qhd,kd->qhk", query, key[kb:ke])
                    dots.relu_()
                    if native_reference is not None:
                        factor = (
                            512
                            if math_mode == "native_fp16" and encoding == "fp8"
                            else 1
                        )
                        rounded = (dots / factor).to(logits_dtype).double()
                        w = (weight.float() * factor).to(logits_dtype).double()
                        chains = [
                            torch.zeros(
                                (stop - first, ke - kb),
                                device=q.device,
                                dtype=torch.float64,
                            )
                            for _ in range(4)
                        ]
                        for h in range(weight.shape[1]):
                            i = h % 4
                            chains[i] = (
                                (chains[i] + rounded[:, h, :] * w[:, h, None])
                                .to(logits_dtype)
                                .double()
                            )
                        left = (chains[0] + chains[2]).to(logits_dtype).double()
                        right = (chains[1] + chains[3]).to(logits_dtype).double()
                        native_reference[:, kb:ke] = (
                            (
                                (left + right).to(logits_dtype).float()
                                * scale[None, kb:ke]
                            ).to(logits_dtype)
                        ).float()
                    dots.mul_(weight.float()[:, :, None])
                    reference[:, kb:ke] = dots.sum(dim=1) * scale[None, kb:ke]
                if rounded_native and native_scores is not None:
                    native = native_scores[first:stop, : key.shape[0]]
                    valid = (
                        torch.arange(key.shape[0], device=q.device)[None, :]
                        < torch.tensor(ends, device=q.device)[:, None]
                    )
                    native_valid, independent_valid = native[valid], reference[valid]
                    torch.testing.assert_close(
                        native_valid,
                        independent_valid,
                        atol=atol,
                        rtol=rtol,
                        msg=lambda message: f"native producer FP32, rows {first}:{stop}: {message}",
                    )
                    if native_valid.numel():
                        maximum_error["native_producer_fp32"] = max(
                            maximum_error["native_producer_fp32"],
                            float((native_valid - independent_valid).abs().max()),
                        )
                for local, end in enumerate(ends):
                    count = min(top_k, end)
                    expected_fp32 = (
                        reference[local, :end].topk(count).values.sort().values
                    )
                    for name, output in outputs.items():
                        expected = expected_fp32
                        indices = output[first + local, :count].long()
                        if not bool(((indices >= 0) & (indices < end)).all()):
                            raise AssertionError(
                                f"{name}: out-of-bounds row {first + local}"
                            )
                        if indices.unique().numel() != count:
                            raise AssertionError(
                                f"{name}: duplicate indices in row {first + local}"
                            )
                        if not bool((output[first + local, count:] == -1).all()):
                            raise AssertionError(
                                f"{name}: invalid suffix in row {first + local}"
                            )
                        actual = reference[local, indices].sort().values
                        if name != "full_gvr2" and native_reference is not None:
                            values = native_reference[local, :end]
                            expected = values.topk(count).values.sort().values
                            actual = values[indices].sort().values
                        elif name != "full_gvr2" and rounded_native:
                            values = (
                                native_scores[first + local, :end]
                                if native_scores is not None
                                else reference[local, :end]
                            ).to(logits_dtype)
                            expected = values.topk(count).values.sort().values.float()
                            actual = values[indices].sort().values.float()
                        exact_native = (
                            name != "full_gvr2"
                            and rounded_native
                            and native_scores is not None
                            and native_reference is None
                        )
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=0
                            if exact_native
                            else (
                                max(atol, 2e-3)
                                if native_reference is not None and name != "full_gvr2"
                                else atol
                            ),
                            rtol=0
                            if exact_native
                            else (
                                max(rtol, 2e-3)
                                if native_reference is not None and name != "full_gvr2"
                                else rtol
                            ),
                            msg=lambda message: f"{name}, row {first + local}: {message}",
                        )
                        if count:
                            maximum_error[name] = max(
                                maximum_error[name],
                                float(
                                    torch.where(
                                        actual == expected, 0, (actual - expected).abs()
                                    ).max()
                                ),
                            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return maximum_error


def _context_capacity(kv_len, max_context_length, compression_ratio):
    """Resolve original-token capacity independently of the live context."""
    capacity = kv_len if max_context_length is None else max_context_length
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("max-context-length must be a positive integer")
    if capacity < kv_len:
        raise ValueError("max-context-length must cover every live KV length")
    if capacity // compression_ratio == 0:
        raise ValueError("max-context-length must include at least one physical key")
    return capacity, capacity // compression_ratio


def run_row(kv_len: int, args) -> dict:
    """Benchmark equivalent full/GVR and selective routes on original-token bounds."""
    from flashinfer.attn_scores import FP8MQATopKWrapper

    max_context, physical_capacity = _context_capacity(
        kv_len, getattr(args, "max_context_length", None), args.compression_ratio
    )
    if kv_len < args.q:
        raise ValueError("original KV length must be at least Q per request")
    total_rows = args.batch_size * args.q
    physical_length = kv_len // args.compression_ratio
    encoding = getattr(args, "qkv_dtype", "fp8")
    model = getattr(args, "model_config", None)
    input_model = getattr(args, "model_profile", None) or model
    distribution = getattr(args, "input_distribution", "signed")
    model_inputs = distribution == "model"
    logits_name = getattr(args, "logits_dtype", "float32")
    logits_dtype = getattr(torch, logits_name)
    fp4 = encoding == "mxfp4"
    mxfp8 = encoding == "mxfp8"
    tile = args.full_query_tile
    if tile is None:
        family_tile = 2 if fp4 else 16 if args.num_heads == 8 else 4
        tile = family_tile if args.q % family_tile == 0 else 1
    if args.q % tile:
        raise ValueError("uniform request Q must be divisible by full-query-tile")
    groups = _ReferenceGroups.build(
        [i * args.q for i in range(args.batch_size + 1)],
        [i * kv_len for i in range(args.batch_size + 1)],
        args.compression_ratio,
        tile,
    )
    if groups.token_prefix[-1] >= 2**31:
        raise ValueError("cumulative original-token prefixes must fit int32")
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    if model_inputs:
        q, query_scales = _model_quantize(
            total_rows * args.num_heads, args.head_dim, generator=generator, fp4=fp4
        )
        q = q.reshape(total_rows, args.num_heads, args.head_dim // (2 if fp4 else 1))
        if fp4:
            query_scales = query_scales.reshape(total_rows, args.num_heads, 4)
            q_scale = query_scales.contiguous().view(torch.int32).squeeze(-1)
            reference_q = _decode_mxfp4(q, query_scales)
        else:
            q_scale, reference_q = None, q
    elif fp4:
        q = torch.randint(
            0,
            256,
            (total_rows, args.num_heads, args.head_dim // 2),
            device="cuda",
            dtype=torch.uint8,
            generator=generator,
        )
        q_scale = torch.full(
            (total_rows, args.num_heads), 0x7F7F7F7F, device="cuda", dtype=torch.int32
        )
        reference_q = _decode_mxfp4(q)
    else:
        q, _ = _quantize(
            total_rows * args.num_heads, args.head_dim, generator=generator
        )
        q = q.reshape(total_rows, args.num_heads, args.head_dim)
        q_scale, reference_q = None, q
    if mxfp8:
        q_scale = torch.full(
            (total_rows, args.num_heads), 0x7F7F7F7F, device="cuda", dtype=torch.int32
        )
    keys, scales, pages = [], [], []
    continuous_keys, continuous_scales = [], []
    # Generate one independent request at a time to bound the FP32 temporary.
    for _ in range(args.batch_size):
        if mxfp8:
            key = torch.randn(
                (physical_length, args.head_dim), device="cuda", generator=generator
            ).to(torch.float8_e4m3fn)
            scale = torch.full(
                (physical_length, 4), 127, device="cuda", dtype=torch.uint8
            )
            keys.append(key)
            scales.append(torch.ones(physical_length, device="cuda"))
        elif model_inputs:
            key, scale = _model_quantize(
                physical_length, args.head_dim, generator=generator, fp4=fp4
            )
            if fp4:
                keys.append(_decode_mxfp4(key, scale))
                scales.append(torch.ones(physical_length, device="cuda"))
            else:
                scale = (
                    torch.rand(physical_length, device="cuda", generator=generator)
                    * 0.1
                    + 0.01
                )
                if input_model == "Qwen3.8-Flash-Next":
                    scale.fill_(1)
                keys.append(key)
                scales.append(scale)
        elif fp4:
            key = torch.randint(
                0,
                256,
                (physical_length, args.head_dim // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            scale = torch.full(
                (physical_length, 4), 127, device="cuda", dtype=torch.uint8
            )
            keys.append(_decode_mxfp4(key))
            scales.append(torch.ones(physical_length, device="cuda"))
        else:
            key, scale = _quantize(physical_length, args.head_dim, generator=generator)
            if input_model == "Qwen3.8-Flash-Next":
                scale.fill_(1)
            keys.append(key)
            scales.append(scale)
        continuous_keys.append(key)
        continuous_scales.append(scale)
        # A dummy page keeps the public paged ABI valid for all-empty ratio4.
        live_pages = (
            _fuse_paged_kv(key, scales[-1] if mxfp8 else scale, args.page_size)
            if physical_length
            else torch.zeros(
                (1, args.page_size, 1, args.head_dim // (2 if fp4 else 1) + 4),
                device="cuda",
                dtype=torch.uint8,
            )
        )
        capacity_pages = max(1, math.ceil(physical_capacity / args.page_size))
        if capacity_pages > live_pages.shape[0]:
            padded_pages = torch.zeros(
                (capacity_pages, *live_pages.shape[1:]),
                device=live_pages.device,
                dtype=live_pages.dtype,
            )
            padded_pages[: live_pages.shape[0]].copy_(live_pages)
            live_pages = padded_pages
        pages.append(live_pages)
    dense_keys = torch.cat(continuous_keys)
    dense_scales = torch.cat(continuous_scales)
    storage_keys = max(1, args.batch_size * physical_capacity)
    if fp4 or mxfp8:
        storage_keys = (storage_keys + 255) // 256 * 256
    if dense_keys.shape[0] < storage_keys:
        padded_keys = torch.zeros(
            (storage_keys, *dense_keys.shape[1:]),
            device=dense_keys.device,
            dtype=dense_keys.dtype,
        )
        padded_scales = torch.full(
            (storage_keys, *dense_scales.shape[1:]),
            127 if fp4 or mxfp8 else 1,
            device=dense_scales.device,
            dtype=dense_scales.dtype,
        )
        padded_keys[: dense_keys.shape[0]].copy_(dense_keys)
        padded_scales[: dense_scales.shape[0]].copy_(dense_scales)
        dense_keys, dense_scales = padded_keys, padded_scales
    kv_fused = torch.cat(pages).flip(0).contiguous()
    pages_per_request = pages[0].shape[0]
    block_table = (
        kv_fused.shape[0]
        - 1
        - torch.arange(
            args.batch_size * pages_per_request, device="cuda", dtype=torch.int32
        )
    ).view(args.batch_size, pages_per_request)
    del pages
    weights = (torch.rand if model_inputs else torch.randn)(
        (total_rows, args.num_heads),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    if model_inputs:
        weights.mul_(0.01)
    if input_model == "Qwen3.8-Flash-Next":
        weights.fill_(128**-0.5)
    weights = weights.to(getattr(torch, getattr(args, "weights_dtype", "float32")))
    cu_q = torch.tensor(groups.q_prefix, device="cuda", dtype=torch.int32)
    cu_tokens = torch.tensor(groups.token_prefix, device="cuda", dtype=torch.int32)
    selective = FP8MQATopKWrapper(
        backend="cutedsl",
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        top_k=args.top_k,
        math_mode=getattr(args, "math_mode", None),
        weights_dtype=getattr(torch, getattr(args, "weights_dtype", "float32")),
        compression_ratio=args.compression_ratio,
        max_context_length=max_context,
        max_chunk_size=q.shape[0],
        max_batch_size=args.batch_size,
        device=q.device,
        num_keys=dense_keys.shape[0],
        qkv_dtype=encoding,
        logits_dtype=logits_dtype,
        model_config=model,
    )
    staged = selective.input_buffers()
    staged["q"].copy_(q)
    staged["weights"].copy_(weights)
    if fp4 or mxfp8:
        staged["q_scale"].copy_(q_scale)
    staged["qo_indptr"].copy_(cu_q)
    staged["kv_indptr"].copy_(cu_tokens)
    selective.prepare_pool(dense_keys, k_scale=dense_scales)
    if fp4 or mxfp8:
        full = _DenseGVR(reference_q, keys, scales, weights, groups, args.top_k)
    else:
        full = _FullGVR(
            q,
            kv_fused,
            weights.float(),
            block_table,
            groups,
            args.top_k,
            qo_indptr=staged["qo_indptr"],
            kv_indptr=staged["kv_indptr"],
            physical_capacity=physical_capacity,
        )
    # Compile both step and layer paths before capturing their independent pairs.
    full.step()
    full.layer()
    selective.plan(staged["qo_indptr"], staged["kv_indptr"])
    run_options = dict(q_scale=staged["q_scale"]) if fp4 or mxfp8 else {}
    selective.run(
        staged["q"], staged["weights"], dense_keys, k_scale=dense_scales, **run_options
    )
    torch.cuda.synchronize()
    full_graph, selective_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_graph):
        full.layer()
    with torch.cuda.graph(selective_graph):
        selective_output = selective.run(
            staged["q"],
            staged["weights"],
            dense_keys,
            k_scale=dense_scales,
            **run_options,
        )
    full_step_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_step_graph):
        full.step()
    step_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(step_graph):
        selective.plan(staged["qo_indptr"], staged["kv_indptr"])
    full_graph.replay()
    selective_graph.replay()
    torch.cuda.synchronize()
    atol, rtol = (5e-5, 1e-5) if args.q == 1 else (1e-4, 1e-4)
    if getattr(args, "exact_oracle", False):
        atol, rtol = 0.0, 0.0
    errors = _check_reference(
        {"full_gvr2": full.output, "selective": selective_output},
        reference_q,
        keys,
        scales,
        weights,
        groups,
        args.top_k,
        atol=atol,
        rtol=rtol,
        logits_dtype=logits_dtype,
        native_scores=full.scores,
        math_mode=getattr(args, "math_mode", None),
        encoding=encoding,
    )
    if int(selective.error.item()) or int(selective.repair_error.item()):
        raise AssertionError("selective exact completion failed before timing")
    layer_only = getattr(args, "layer_only", False)
    step = None
    if not layer_only:
        step_full_us, step_selective_us, step_pauses = _paired_events(
            full_step_graph.replay,
            step_graph.replay,
            args.warmups,
            args.repeats,
            reverse=getattr(args, "reverse_order", False),
            settle_seconds=args.settle_seconds,
        )
        step = {
            "full_gvr2": {**_summarize(step_full_us), "raw_us": step_full_us},
            "selective": {**_summarize(step_selective_us), "raw_us": step_selective_us},
            "post_component_pauses": step_pauses,
        }
    if int(selective.error.item()) or int(selective.repair_error.item()):
        raise AssertionError("selective step validation failed")
    full_us, selective_us, pauses = _paired_events(
        full_graph.replay,
        selective_graph.replay,
        args.warmups,
        args.repeats,
        reverse=getattr(args, "reverse_order", False),
        settle_seconds=args.settle_seconds,
    )
    # Check outputs from the measured replay, not only the pre-timing warmup.
    errors_after = _check_reference(
        {"full_gvr2": full.output, "selective": selective_output},
        reference_q,
        keys,
        scales,
        weights,
        groups,
        args.top_k,
        atol=atol,
        rtol=rtol,
        logits_dtype=logits_dtype,
        native_scores=full.scores,
        math_mode=getattr(args, "math_mode", None),
        encoding=encoding,
    )
    exact_error = int(selective.repair_error.item())
    if int(selective.error.item()):
        raise AssertionError("selective device capacity validation failed")
    if exact_error:
        raise AssertionError(f"selective exact completion failed: {exact_error}")
    return {
        "bs": args.batch_size,
        "backend": "cutedsl",
        "qkv_dtype": encoding,
        "model_config": model,
        "model_profile": input_model,
        "input_distribution": distribution,
        "math_mode": selective.math_mode,
        "weights_dtype": getattr(args, "weights_dtype", "float32"),
        "logits_dtype": logits_name,
        "q": args.q,
        "kv": kv_len,
        "physical_kv": physical_length,
        "max_context_length": max_context,
        "physical_capacity": physical_capacity,
        "capacity_pages_per_request": pages_per_request,
        "compression_ratio": args.compression_ratio,
        "top_k": args.top_k,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "route_label": "public-"
        + encoding
        + "-scorer-gvr2-vs-public-selective-topk-"
        + "cutedsl",
        "execution_mode": "cuda-graph-replay",
        "full_config": {
            "scorer_encoding": "fp32" if fp4 or mxfp8 else "fp8",
            "logits_dtype": "float32",
            "query_chunks": [(first, stop) for first, stop, _ in full.scorer_chunks],
            "scorer_mode": full.scorer_mode,
            "config_scope": "PyTorch FP32 dense fallback"
            if fp4 or mxfp8
            else "public FP8 paged scorer",
            "logical_score_width": physical_capacity,
            "scorer_max_seq_len": full.scorer_width,
            "query_materialization": "head padding when required"
            if not (fp4 or mxfp8)
            else "decoded FP32",
            "step_metadata": "device prefix-derived row/group ends and request IDs; "
            "live table gather and capturable public schedule builder",
            "gvr_mode": "per-row physical ends; next_n=1, compress_ratio=1",
        },
        "selective_policy": {
            "backend": "cutedsl",
            "needs_sampling": True,
            "sample_threshold_policy": "stratified",
        },
        "full_gvr2": {**_summarize(full_us), "raw_us": full_us},
        "selective": {**_summarize(selective_us), "raw_us": selective_us},
        "full_over_selective": statistics.median(full_us)
        / statistics.median(selective_us),
        "step": step,
        "correctness": {
            "reference": "all-row independent FP32 math; lower-precision selective "
            "uses native-rounded FMA reference when requested; Full remains FP32; "
            "TF32 disabled; outside timing",
            "oracle_dtypes": {"full_gvr2": "float32", "selective": logits_name},
            "exact_score_multiset": atol == 0 and rtol == 0,
            "oracle_tolerances": {
                "full_gvr2": {"atol": atol, "rtol": rtol},
                "selective": (
                    {"atol": max(atol, 2e-3), "rtol": max(rtol, 2e-3)}
                    if getattr(args, "math_mode", None)
                    in ("native_bf16", "native_fp16")
                    else {"atol": 0.0, "rtol": 0.0}
                    if logits_dtype != torch.float32
                    else {"atol": atol, "rtol": rtol}
                ),
                "native_producer_fp32": (
                    {"atol": atol, "rtol": rtol}
                    if logits_dtype != torch.float32
                    else None
                ),
            },
            "atol": atol,
            "rtol": rtol,
            "max_topk_value_error_before": errors,
            "max_topk_value_error_after": errors_after,
            "exact_error": exact_error,
            "rows": groups.q_prefix[-1],
            "full_physical_page_mapping": "global reverse permutation; independent requests",
            "selective_kv_layout": "continuous request-concatenated K",
        },
        "protocol": {
            "warmups_per_route": args.warmups,
            "steady_state_warmup_requested_s": args.settle_seconds,
            "steady_state_warmup_route": "same route as the following timed block",
            "samples_per_route": args.repeats,
            "order": (
                "BxN-5s-AxN-5s"
                if getattr(args, "reverse_order", False)
                else "AxN-5s-BxN-5s"
            ),
            "samples_per_block": args.repeats,
            "sleep_between_samples": False,
            "a": "full_gvr2",
            "b": "selective",
            "component": (
                "one layer graph replay; per-step setup pair omitted"
                if layer_only
                else "one layer graph replay; separate per-step setup graph pair"
            ),
            "selective_inputs": "Q/weights/indptr staging outside timing; "
            "continuous K/scales bound directly",
            "layer_work": "full logits+GVR2; selective run including resets, "
            "publication, selection, repair and output",
            "step_work": "per-step setup: full row/group metadata and request IDs, "
            "table gather and scorer schedule; selective plan with prefix copies, "
            "validation, metadata, schedules and sampling-metadata refresh",
            "step_order": "prepared once outside timing; no per-step setup measurements"
            if layer_only
            else "per-step setup pair before layer pair; each pair AxN-5s-BxN-5s",
            "capacity_scope": "fixed original-token context capacity; live prefixes "
            "and oracle remain request-local; no multi-layer amortization"
            if max_context != kv_len
            else "exact-fit uniform query-topology bucket; "
            "no measured multi-layer amortization or oversized-bucket claim",
            "post_component_sleep_requested_s": 5.0,
            "post_component_pauses": pauses,
            "inner": 1,
            "seed": args.seed,
            "cache_flush": False,
            "percentiles": "inclusive linear interpolation",
        },
    }


def main() -> None:
    global torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Validate existing result JSON and print Markdown without Torch/CUDA.",
    )
    parser.add_argument("--q", type=int, default=4096)
    parser.add_argument(
        "--math-mode",
        choices=("relu_strict", "relu_fast", "native_bf16", "native_fp16"),
        default=None,
    )
    parser.add_argument(
        "--weights-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--qkv-dtype", choices=("fp8", "mxfp4", "mxfp8"), default="fp8")
    parser.add_argument(
        "--logits-dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--exact-oracle", action="store_true")
    parser.add_argument(
        "--input-distribution",
        choices=("signed", "model"),
        default="signed",
        help="signed/normalized controls or standard-normal model inputs with positive weights; MXFP4 model inputs use BF16 public quantization",
    )
    parser.add_argument(
        "--model-profile",
        choices=(
            "DS-V3.2",
            "DS-V4-Flash",
            "DS-V4-Pro",
            "GLM-5.3-Flash",
            "GLM-5.3",
            "Qwen3.8-Flash-Next",
        ),
        help="Synthetic model input contract, independent of the optional model configuration.",
    )
    parser.add_argument(
        "--model-config",
        choices=(
            "DS-V3.2",
            "DS-V4-Flash",
            "DS-V4-Pro",
            "GLM-5.3-Flash",
            "GLM-5.3",
            "Qwen3.8-Flash-Next",
        ),
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Measure selective before the Full anchor for a drift-control rerun.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--compression-ratio", type=int, choices=(1, 4), default=1)
    parser.add_argument("--full-query-tile", type=int)
    parser.add_argument("--kv", type=int, nargs="*", default=ROWS)
    parser.add_argument(
        "--max-context-length",
        type=int,
        help="Original-token capacity for every route; defaults to each live --kv length.",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Continuous same-route warmup before each timed block; 0 reproduces cold-start diagnostics.",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--layer-only",
        action="store_true",
        help="Measure only the per-layer pair; omit per-step setup measurements.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.summarize is not None:
        print(render_summary(args.summarize))
        return
    if args.q <= 0 or args.batch_size <= 0 or args.warmups < 0:
        raise ValueError("q and batch-size must be positive; warmups nonnegative")
    if args.qkv_dtype == "mxfp4" and args.num_heads != 64:
        raise ValueError("MXFP4 Full comparison requires H64")
    if args.batch_size != 1 or args.num_heads not in (4, 32, 64):
        raise ValueError("selective prefill requires batch size 1 and H4/H32/H64")
    if args.full_query_tile is not None and args.full_query_tile <= 0:
        raise ValueError("full-query-tile must be positive")
    if not args.kv or any(length < args.q for length in args.kv):
        raise ValueError("each original KV length must be at least Q per request")
    for length in args.kv:
        _context_capacity(length, args.max_context_length, args.compression_ratio)
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        raise ValueError("settle-seconds must be finite and nonnegative")
    if args.num_heads <= 0 or args.head_dim <= 0:
        raise ValueError("num-heads and head-dim must be positive")
    if len(set(args.kv)) != len(args.kv):
        raise ValueError("duplicate KV lengths would repeat a comparison")
    torch = importlib.import_module("torch")
    if args.output is not None:
        # Reserve before CUDA work; never overwrite a prior invocation's data.
        with args.output.open("x") as output:
            output.write('{"status": "starting", "rows": []}\n')
    if args.kv != ROWS:
        print("warning: production KV subset requested")
    importlib.import_module("flashinfer.attn_scores.kernels.fp8_paged_mqa_logits")
    result = {
        "schema_version": 22 if args.layer_only else 21,
        "status": "running",
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "uuid": str(
                getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        },
        "rows": [],
    }
    try:
        for kv_len in args.kv:
            row = run_row(kv_len, args)
            result["rows"].append(row)
            if args.output is not None:
                _save_result(args.output, result)
            print(
                f"PASS H{args.num_heads} Q{args.q} R{args.compression_ratio} KV{kv_len}",
                flush=True,
            )
            # Release graph owners before the next cell; do not flush L2/cache.
            gc.collect()
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if args.output is not None:
            _save_result(args.output, result)
        raise
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        _save_result(args.output, result)
    print(_render_results([("benchmark", result)]))


if __name__ == "__main__":
    main()
