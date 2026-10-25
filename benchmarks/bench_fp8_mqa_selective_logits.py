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
        --num-heads 8 --top-k 512 --compression-ratio 4
    python benchmarks/bench_fp8_mqa_selective_logits.py --batch-size 148 --q 1 \
        --kv 8192 --num-heads 64 --top-k 2048
    python benchmarks/bench_fp8_mqa_selective_logits.py --summarize results/*.json

Per-step setup runs once per forward pass and is shared by all indexer layers.
Full builds row/group metadata and request IDs, gathers the page table, and
computes the scorer schedule. Selective calls plan(): table copy, validation,
row metadata, scorer schedules, and sampling-metadata refresh.

Layer is one indexer TopK operation for one layer. Full calls
fp8_paged_mqa_logits followed by top_k_varlen(backend="gvr_2"); selective calls
run(), including sampling, candidate publication, selection, and exact repair.
For L indexer layers, model cost per forward pass is per-step setup + L * layer;
a standalone single call costs per-step setup + layer.

This benchmark generates seeded inputs and measures separate per-step setup
and layer CUDA graph pairs in an AxN-5s-BxN-5s sequence.
Successful clean-source runs also print a validated Markdown report; --output
remains raw JSON. The --summarize mode regenerates that report without importing
Torch or using CUDA. Inputs must share one source, environment, GPU and schema;
partial successful prefixes are labeled and duplicate cells are rejected.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
import tempfile
from pathlib import Path

# Imported helpers retain their Torch dependency; the reporting CLI is stdlib-only.
if __name__ != "__main__":
    import torch

ROWS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
EXECUTION_SOURCE_PATHS = (
    "benchmarks/bench_fp8_mqa_selective_logits.py",
    "flashinfer/attn_scores/attn_scores.py",
    "flashinfer/attn_scores/fp8_kernel_config.py",
    "flashinfer/attn_scores/query_group_metadata.py",
    "flashinfer/attn_scores/kernels/schedule_kernel.py",
    "flashinfer/attn_scores/kernels/fp8_paged_mqa_logits.py",
    "flashinfer/attn_scores/kernels/_fp8_mqa_selective.py",
    "flashinfer/topk_varlen/kernels/gvr2_topk_decode.py",
    "flashinfer/topk_varlen/kernels/gvr_topk_decode.py",
    "flashinfer/topk_varlen/kernels/gvr_topk_decode_lb.py",
    "flashinfer/topk_varlen/kernels/gvr2_topk_host.py",
    "flashinfer/topk_varlen/kernels/block_scan.py",
    "flashinfer/attn_scores/kernels/selective_logits_metadata.py",
    "flashinfer/attn_scores/selective_logits.py",
    "flashinfer/attn_scores/selective_capacity.py",
    "flashinfer/attn_scores/_math_policy.py",
    "flashinfer/attn_scores/_selective_contract.py",
    "flashinfer/jit/selective_sample_ladder.py",
    "flashinfer/jit/selective_repair_if.py",
    "csrc/selective_sample_ladder.cu",
    "csrc/selective_repair_if.cu",
    "csrc/tvm_ffi_utils.h",
    "include/flashinfer/attention/selective_sample_ladder.cuh",
    "include/flashinfer/attention/selective_repair_if.cuh",
    "flashinfer/topk_varlen/kernels/radix_topk.py",
    "flashinfer/topk_varlen/topk_varlen.py",
)


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


def _validate_pauses(pauses, repeats):
    if len(pauses) != 2 or any(
        pause["after"] != label
        or pause["samples"] != repeats
        or not math.isfinite(pause["elapsed_s"])
        or pause["elapsed_s"] < 5.0
        for label, pause in zip(("a", "b"), pauses, strict=True)
    ):
        raise ValueError("expected two synchronized arm pauses")


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
    """Render one source/environment/GPU's block-paired JSON without CUDA work."""
    return _render_results((str(path), json.loads(path.read_text())) for path in paths)


def _render_results(results):
    rows, identities, seen, partial = [], set(), set(), []
    for path, result in results:
        schema = result["schema_version"]
        if schema not in (11, 12, 13, 14):
            raise ValueError(f"expected block-paired schema 11, 12, 13 or 14: {path}")
        source = result["source"]
        if not source["tracked_worktree_clean"]:
            raise ValueError(f"dirty execution source: {path}")
        identities.add(
            (
                source["git_commit"],
                json.dumps(source["source_sha256"], sort_keys=True),
                result["device"]["uuid"],
                json.dumps(result["environment"], sort_keys=True),
                schema,
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
                protocol["order"] != "AxN-5s-BxN-5s"
                or type(protocol["samples_per_route"]) is not int
                or protocol["samples_per_route"] <= 0
                or protocol["samples_per_block"] != protocol["samples_per_route"]
                or protocol["sleep_between_samples"] is not False
                or protocol["a"] != "full_gvr2"
                or protocol["b"] != "selective"
            ):
                raise ValueError(f"unexpected timing protocol: {key}")
            repeats = protocol["samples_per_route"]
            _validate_pauses(protocol["post_component_pauses"], repeats)
            if row["correctness"]["exact_error"] != 0:
                raise ValueError(f"exact repair failed: {key}")
            summaries = _validated_routes(row, repeats)
            step_summaries = None
            if schema in (11, 13):
                if not row.get("step"):
                    raise ValueError(
                        "setup/layer schema requires per-step setup samples"
                    )
                _validate_pauses(row["step"]["post_component_pauses"], repeats)
                step_summaries = _validated_routes(row["step"], repeats)
            elif row.get("step") is not None:
                raise ValueError(
                    "layer-only schema must omit per-step setup measurements"
                )
            rows.append((row, summaries, step_summaries))
    if len(identities) != 1 or not rows:
        raise ValueError("expected nonempty results from one source/environment/GPU")
    identity = next(iter(identities))
    sample_counts = {row["protocol"]["samples_per_route"] for row, _, _ in rows}
    median_scope = (
        f"median of {next(iter(sample_counts))} consecutive replays"
        if len(sample_counts) == 1
        else "medians of the configured consecutive replay blocks"
    )
    lines = [
        f"Source `{identity[0]}`; GPU `{identity[2]}`. {len(rows)} comparisons.",
        "",
        f"Latency in ms: {median_scope}.",
        "Workspace in MiB: reachable PyTorch tensors owned by each route, excluding inputs/outputs; not a device peak.",
        "",
        "Per-step setup runs once per forward pass and is shared across indexer layers: full builds row/group metadata and request IDs, gathers the page table and computes the scorer schedule; selective calls plan() for table copy, validation, metadata, schedules and sampling-metadata refresh.",
        "Layer is one indexer TopK operation: full calls fp8_paged_mqa_logits + top_k_varlen(gvr_2); selective calls run().",
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
                f"| {heads_label} | Compression Ratio | {dimension} | Original context | Full indexer + GVRv2 layer (ms) | Indexer-throwaway + TopK layer (ms) | Layer speedup | Full indexer + GVRv2 workspace (MiB) | Indexer-throwaway + TopK workspace (MiB) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row, summaries, _ in group:
            memory = [
                row["memory"][route]["owned_reachable_tensor_bytes"] / 2**20
                for route in ("full_gvr2", "selective")
            ]
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
                f"{summaries[0][0] / summaries[1][0]:.4f}x | {memory[0]:.1f} | {memory[1]:.1f} |"
            )
    lines.extend(
        [
            "",
            "Fixed full-first AxN-5s-BxN-5s; one event-timed replay per sample, no deliberate intra-arm sleep. Host event/enqueue gaps remain. Device snapshots do not establish power isolation.",
        ]
    )
    if identity[4] in (11, 13):
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


def _source_provenance(repository: Path) -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.splitlines()
    source_sha256 = {
        relative_path: hashlib.sha256(
            (repository / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in EXECUTION_SOURCE_PATHS
    }
    return {
        "git_status_porcelain": status,
        "tracked_worktree_clean": not status,
        "source_sha256": source_sha256,
    }


def _paired_events(call_a, call_b, warmups: int, repeats: int):
    """After warmup, run A×N, sleep 5s, B×N, sleep 5s on identical inputs.

    Each sample times one graph replay with its own completed CUDA end event.
    There is no deliberate sleep between samples in an arm. Synchronize before
    each arm's pause, including the final B pause, and return both durations
    separately from GPU latency. Host event/enqueue overhead remains between replays.
    """
    if repeats <= 0:
        raise ValueError("paired measurement requires a positive repeat count")
    for repeat in range(warmups):
        first, second = (call_a, call_b) if repeat % 2 == 0 else (call_b, call_a)
        first()
        second()
    torch.cuda.synchronize()

    samples_a: list[float] = []
    samples_b: list[float] = []
    pauses: list[dict] = []
    order = (("a", call_a), ("b", call_b))
    for label, call in order:
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
            }
        )
    return samples_a, samples_b, pauses


def _tensor_storage(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _workspace_inventory(owner, excluded):
    """Count reachable PyTorch storage once, excluding shared inputs/outputs.

    Opaque native allocations and graph-private pools are not traversed. This
    is neither an allocator delta nor a device-memory peak.
    """
    excluded_storage = {_tensor_storage(tensor) for tensor in excluded}
    visited, allocations = set(), {}

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            key = _tensor_storage(value)
            if key not in excluded_storage and key[2]:
                allocations.setdefault(key, []).append(path)
        elif isinstance(value, dict):
            for name, item in value.items():
                visit(item, f"{path}.{name}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif type(value).__module__.startswith("flashinfer.attn_scores") and hasattr(
            value, "__dict__"
        ):
            for name, item in vars(value).items():
                visit(item, f"{path}.{name}")

    visit(owner, "workspace")
    return {
        "owned_reachable_tensor_bytes": sum(key[2] for key in allocations),
        "storage_count": len(allocations),
        "allocations": [
            {"device": key[0], "bytes": key[2], "paths": paths}
            for key, paths in allocations.items()
        ],
        "scope": "deduplicated reachable PyTorch storage; input/output excluded; "
        "opaque native and graph-only storage unknown; not a device peak",
    }


def _device_observation():
    """Untimed device snapshot; not route-specific telemetry or power isolation."""
    fields = (
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
    )
    try:
        observation = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "returncode": observation.returncode,
            "gpus": [
                dict(zip(fields, (value.strip() for value in row), strict=True))
                for row in csv.reader(observation.stdout.splitlines())
                if len(row) == len(fields)
            ]
            if observation.returncode == 0
            else [],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


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
        padded_scale = torch.ones(padded_rows, device=scale.device, dtype=scale.dtype)
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
        scale.view(num_blocks, block_size)
        .contiguous()
        .view(torch.uint8)
        .reshape(num_blocks, block_size * 4)
    )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


class _FullGVR:
    """Public scorer/GVR composition with separate step and layer callables.

    Uniform requests must divide evenly into the configured query tile. Q and
    weights then use address-stable views, without cross-request grouping.
    Query topology is fixed; every step derives physical row/group bounds and
    request IDs from device prefixes, gathers the live map and refreshes the
    public scorer schedule. Layer replay consumes those address-stable owners.
    """

    def __init__(
        self,
        q,
        kv_fused,
        weights,
        block_table,
        groups,
        config,
        top_k,
        *,
        qo_indptr,
        kv_indptr,
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
        rows, heads, dimension = q.shape
        tile = groups.query_tile
        config.validate_geometry(
            heads=heads, dimension=dimension, page=kv_fused.shape[1], next_n=tile
        )
        config.require_dispatchable()
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
        columns = max(8192, top_k, config.padded_columns(width))
        self.scores = torch.empty((rows, columns), device=device, dtype=torch.float32)
        scores = self.scores
        logical_scores = scores[:, :width]
        row_ids = torch.arange(rows, device=device, dtype=torch.int32)
        row_ends = torch.empty((rows,), device=device, dtype=torch.int32)
        metadata = torch.empty(
            (len(groups.rows), tile + 2), device=device, dtype=torch.int32
        )
        group_ends = torch.empty((len(groups.rows),), device=device, dtype=torch.int32)
        request_ids = torch.empty((len(groups.rows),), device=device, dtype=torch.int64)
        table_rows = torch.empty(
            (len(groups.rows), block_table.shape[1]), device=device, dtype=torch.int32
        )
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
        grouped_q = q.view(len(groups.rows), tile, heads, dimension)
        self.workspace_owners = (
            scores,
            logical_scores,
            row_ends,
            metadata,
            group_ends,
            request_ids,
            row_ids,
            table_rows,
            schedule,
            workspace,
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
            metadata[:, 0].copy_(row_ids.view(-1, tile)[:, 0])
            metadata[:, 1].fill_(tile)
            metadata[:, 2:].copy_(row_ends.view(-1, tile))
            torch.amax(row_ends.view(-1, tile), dim=1, out=group_ends)
            request_ids.copy_(requests.view(-1, tile)[:, 0])
            torch.index_select(block_table, 0, request_ids, out=table_rows)
            compute_paged_mqa_logits_schedule(
                group_ends, device=device, next_n=tile, variant="fp8", out=schedule
            )

        def layer():
            fp8_paged_mqa_logits(
                grouped_q,
                kv_fused,
                weights,
                table_rows,
                group_ends,
                width,
                output_dtype=torch.float32,
                epi_dtype=torch.float32,
                acc_dtype=torch.float32,
                math_mode="relu_strict",
                schedule_meta=schedule,
                out=scores,
                kernel_config=config,
                query_metadata=metadata,
                output_rows=rows,
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


def _check_reference(outputs, q, kv, scales, weights, groups, top_k, *, atol, rtol):
    """Check every request/row against independent, chunked FP32 score math."""
    maximum_error = {name: 0.0 for name in outputs}
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
                for kb in range(0, key.shape[0], 4096):
                    ke = min(kb + 4096, key.shape[0])
                    dots = torch.einsum("qhd,kd->qhk", query, key[kb:ke])
                    dots.relu_().mul_(weight[:, :, None])
                    reference[:, kb:ke] = dots.sum(dim=1) * scale[None, kb:ke]
                for local, end in enumerate(ends):
                    count = min(top_k, end)
                    expected = reference[local, :end].topk(count).values.sort().values
                    for name, output in outputs.items():
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
                        torch.testing.assert_close(
                            actual, expected, atol=atol, rtol=rtol
                        )
                        if count:
                            maximum_error[name] = max(
                                maximum_error[name],
                                float((actual - expected).abs().max()),
                            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return maximum_error


def run_row(kv_len: int, args) -> dict:
    """Benchmark equivalent full/GVR and selective routes on original-token bounds."""
    from flashinfer import FP8PagedMQATopKWrapper
    from flashinfer.attn_scores.fp8_kernel_config import FP8MQAKernelConfig
    from flashinfer.attn_scores.query_group_metadata import FP8QueryGroups

    observation_before = _device_observation()
    torch.cuda.reset_peak_memory_stats()
    if kv_len < args.q:
        raise ValueError("original KV length must be at least Q per request")
    total_rows = args.batch_size * args.q
    physical_length = kv_len // args.compression_ratio
    tile = args.full_query_tile
    if tile is None:
        family_tile = 16 if args.num_heads == 8 else 4
        tile = family_tile if args.q % family_tile == 0 else 1
    if args.q % tile:
        raise ValueError("uniform request Q must be divisible by full-query-tile")
    groups = FP8QueryGroups.build(
        [i * args.q for i in range(args.batch_size + 1)],
        [i * kv_len for i in range(args.batch_size + 1)],
        args.compression_ratio,
        tile,
    )
    if groups.token_prefix[-1] >= 2**31:
        raise ValueError("cumulative original-token prefixes must fit int32")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q, _ = _quantize(total_rows * args.num_heads, args.head_dim, generator=generator)
    q = q.reshape(total_rows, args.num_heads, args.head_dim)
    keys, scales, pages = [], [], []
    # Generate one independent request at a time to bound the FP32 temporary.
    for _ in range(args.batch_size):
        key, scale = _quantize(physical_length, args.head_dim, generator=generator)
        keys.append(key)
        scales.append(scale)
        # A dummy page keeps the public paged ABI valid for all-empty ratio4.
        pages.append(
            _fuse_paged_kv(key, scale, args.page_size)
            if physical_length
            else torch.zeros(
                (1, args.page_size, 1, args.head_dim + 4),
                device="cuda",
                dtype=torch.uint8,
            )
        )
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
    weights = torch.randn(
        (total_rows, args.num_heads),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cu_q = torch.tensor(groups.q_prefix, device="cuda", dtype=torch.int32)
    cu_tokens = torch.tensor(groups.token_prefix, device="cuda", dtype=torch.int32)
    config = FP8MQAKernelConfig(
        128,
        args.full_epi_subtiles,
        args.full_w_cache_regs,
        tile,
        num_q_stages=3,
        num_kv_stages=3,
    )
    selective = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        top_k=args.top_k,
        math_mode="relu_strict",
        compression_ratio=args.compression_ratio,
        max_context_length=kv_len,
        max_chunk_size=q.shape[0],
        max_batch_size=args.batch_size,
        kv_layout=kv_fused,
    )
    staged = selective.input_buffers()
    staged["q"].copy_(q)
    staged["weights"].copy_(weights)
    staged["qo_indptr"].copy_(cu_q)
    staged["kv_indptr"].copy_(cu_tokens)
    staged["block_table"].copy_(block_table)
    selective.prepare_pool(kv_fused)
    full = _FullGVR(
        q,
        kv_fused,
        weights,
        block_table,
        groups,
        config,
        args.top_k,
        qo_indptr=staged["qo_indptr"],
        kv_indptr=staged["kv_indptr"],
    )
    engine = selective._capacity_plan._engine
    # Compile both step and layer paths before capturing their independent pairs.
    full.step()
    full.layer()
    selective.plan(staged["qo_indptr"], staged["kv_indptr"], block_table)
    selective.run(staged["q"], staged["weights"], kv_fused)
    torch.cuda.synchronize()
    full_graph, selective_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_graph):
        full.layer()
    with torch.cuda.graph(selective_graph):
        selective_output = selective.run(staged["q"], staged["weights"], kv_fused)
    full_step_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(full_step_graph):
        full.step()
    step_graph = torch.cuda.CUDAGraph()
    copy_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(step_graph):
        selective.plan(staged["qo_indptr"], staged["kv_indptr"], block_table)
    with torch.cuda.graph(copy_graph):
        staged["block_table"].copy_(block_table)
    full_graph.replay()
    selective_graph.replay()
    torch.cuda.synchronize()
    atol, rtol = (5e-5, 1e-5) if args.q == 1 else (1e-4, 1e-4)
    errors = _check_reference(
        {"full_gvr2": full.output, "selective": selective_output},
        q,
        keys,
        scales,
        weights,
        groups,
        args.top_k,
        atol=atol,
        rtol=rtol,
    )
    if int(selective.error.item()) or int(selective.repair_error.item()):
        raise AssertionError("selective exact completion failed before timing")
    layer_only = getattr(args, "layer_only", False)
    step = None
    if not layer_only:
        step_full_us, step_selective_us, step_pauses = _paired_events(
            full_step_graph.replay, step_graph.replay, args.warmups, args.repeats
        )
        step = {
            "full_gvr2": {**_summarize(step_full_us), "raw_us": step_full_us},
            "selective": {**_summarize(step_selective_us), "raw_us": step_selective_us},
            "post_component_pauses": step_pauses,
        }
    if int(selective.error.item()) or int(selective.repair_error.item()):
        raise AssertionError("selective step validation failed")
    full_us, selective_us, pauses = _paired_events(
        full_graph.replay, selective_graph.replay, args.warmups, args.repeats
    )
    # Check outputs from the measured replay, not only the pre-timing warmup.
    errors_after = _check_reference(
        {"full_gvr2": full.output, "selective": selective_output},
        q,
        keys,
        scales,
        weights,
        groups,
        args.top_k,
        atol=atol,
        rtol=rtol,
    )
    exact_error = int(selective.repair_error.item())
    if int(selective.error.item()):
        raise AssertionError("selective device capacity validation failed")
    if exact_error:
        raise AssertionError(f"selective exact completion failed: {exact_error}")
    # These warm component diagnostics follow both measured-output checks.
    # They are independently timed replays, not an additive decomposition of
    # the block-paired A/B medians, and never run between measured components.
    component_diagnostics = {}
    for name, graph in (("block_table_copy", copy_graph),):
        for _ in range(args.warmups):
            graph.replay()
        torch.cuda.synchronize()
        samples = [_record_event(graph.replay) for _ in range(args.repeats)]
        component_diagnostics[name] = {**_summarize(samples), "raw_us": samples}
    if int(selective.error.item()) or int(selective.repair_error.item()):
        raise AssertionError("selective component diagnostic failed")
    excluded = (q, kv_fused, weights, block_table, cu_q, cu_tokens)
    memory = {
        "full_gvr2": _workspace_inventory(
            full.workspace_owners, (*excluded, full.output)
        ),
        "selective": _workspace_inventory(selective, (*excluded, selective_output)),
        "whole_cell_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "whole_cell_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_high_water_rss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024,
        "peak_scope": "whole cell includes inputs, both routes, graphs and oracle; "
        "RSS is cumulative process high-water, not incremental workspace",
    }
    return {
        "bs": args.batch_size,
        "q": args.q,
        "kv": kv_len,
        "physical_kv": physical_length,
        "compression_ratio": args.compression_ratio,
        "top_k": args.top_k,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "route_label": "public-fp8-scorer-gvr2-vs-public-selective-topk",
        "execution_mode": "cuda-graph-replay",
        "full_config": {
            "launch_identity": config.launch_identity,
            "logical_score_width": physical_length,
            "score_pitch": 0 if full.scores is None else full.scores.stride(0),
            "score_arena_bytes": 0 if full.scores is None else full.scores.numel() * 4,
            "page_table_gathers_per_layer": 0,
            "page_table_gathers_per_step": int(physical_length > 0),
            "query_materialization": "uniform request-local views",
            "step_metadata": "device prefix-derived row/group ends and request IDs; "
            "live table gather and capturable public schedule builder",
            "gvr_mode": "per-row physical ends; next_n=1, compress_ratio=1",
        },
        "candidate_split_kv": engine._candidate_prepared.split_kv
        if engine is not None
        else None,
        "selective_policy": {
            "needs_sampling": engine._needs_sampling if engine is not None else False,
            "candidate_schedule": vars(engine._candidate_schedule).copy()
            if engine is not None
            else None,
            "sample_threshold_policy": "native_ladder",
            "candidate_num_umma_stages": engine._candidate_prepared.num_umma_stages
            if engine is not None
            else None,
        },
        "full_gvr2": {**_summarize(full_us), "raw_us": full_us},
        "selective": {**_summarize(selective_us), "raw_us": selective_us},
        "full_over_selective": statistics.median(full_us)
        / statistics.median(selective_us),
        "step": step,
        "selective_components": {
            **component_diagnostics,
            "block_table_copy_bytes": block_table.numel() * block_table.element_size(),
            "scope": "warm graph diagnostics after paired timing and both row checks; "
            "copy diagnostic is not an additive decomposition of paired step latency",
        },
        "memory": memory,
        "device_observations": {
            "before": observation_before,
            "after": _device_observation(),
            "scope": "untimed device snapshots, not per-route activity or power isolation",
        },
        "correctness": {
            "reference": "all-row independent FP32; TF32 disabled; outside timing",
            "atol": atol,
            "rtol": rtol,
            "max_topk_value_error_before": errors,
            "max_topk_value_error_after": errors_after,
            "exact_error": exact_error,
            "first_repair_rows": int(torch.count_nonzero(engine._repair_flags))
            if engine is not None
            else 0,
            "terminal_repair_rows": int(
                torch.count_nonzero(engine._second_repair_flags)
            )
            if engine is not None
            else 0,
            "physical_page_mapping": "global reverse permutation; independent requests",
        },
        "protocol": {
            "warmups_per_route": args.warmups,
            "samples_per_route": args.repeats,
            "order": "AxN-5s-BxN-5s",
            "samples_per_block": args.repeats,
            "sleep_between_samples": False,
            "a": "full_gvr2",
            "b": "selective",
            "component": "one layer graph replay; per-step setup pair omitted"
            if layer_only
            else "one layer graph replay; separate per-step setup graph pair",
            "selective_inputs": "Q/weights/indptr staging outside timing; "
            "caller block-table copy outside layer timing",
            "layer_work": "full logits+GVR2; selective run including resets, "
            "publication, selection, repair and output",
            "step_work": "per-step setup: full row/group metadata and request IDs, "
            "table gather and scorer schedule; selective plan with table copy, "
            "validation, metadata, schedules and sampling-metadata refresh",
            "step_order": "prepared once outside timing; no per-step setup measurements"
            if layer_only
            else "per-step setup pair before layer pair; each pair AxN-5s-BxN-5s",
            "capacity_scope": "exact-fit uniform query-topology bucket; "
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--compression-ratio", type=int, choices=(1, 4), default=1)
    parser.add_argument("--full-query-tile", type=int)
    parser.add_argument("--full-epi-subtiles", type=int, default=1)
    parser.add_argument("--full-w-cache-regs", type=int, default=160)
    parser.add_argument("--kv", type=int, nargs="*", default=ROWS)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--layer-only",
        action="store_true",
        help="Measure only the per-layer pair; omit per-step setup measurements.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.summarize is not None:
        print(render_summary(args.summarize))
        return
    if args.q <= 0 or args.batch_size <= 0 or args.warmups < 0:
        raise ValueError("q and batch-size must be positive; warmups nonnegative")
    if args.full_query_tile is not None and args.full_query_tile <= 0:
        raise ValueError("full-query-tile must be positive")
    if not args.kv or any(length < args.q for length in args.kv):
        raise ValueError("each original KV length must be at least Q per request")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
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
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    result = {
        "schema_version": 14 if args.layer_only else 13,
        "status": "running",
        "source": {
            "git_commit": commit or None,
            "command": sys.argv,
            **_source_provenance(repository),
        },
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
            current_source = _source_provenance(repository)
            if current_source["source_sha256"] != result["source"]["source_sha256"]:
                raise RuntimeError("execution source changed during comparison")
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
    if result["source"]["tracked_worktree_clean"]:
        print(_render_results([("benchmark", result)]))
    else:
        print("Markdown summary omitted: execution source has tracked changes.")


if __name__ == "__main__":
    main()
