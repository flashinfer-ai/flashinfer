#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Benchmark captured continuous inputs with Full/GVRv2 and selective v1/v2.

FlashInfer-only arms need --data-root, not an external kernel checkout. The
capture directory contains Torch tensor dictionaries or lossless .igemm archives
with their portable indexer_inputs loader. Required tensors are q, k, weights,
starts, ends and optional q_scale/k_scale; metadata identifies model/encoding.
API scope crops oldest keys and translates causal bounds. Optional reference
arms retain their original loader, arithmetic and timing when --reference-root
is supplied. Standalone runs use FP32 logits and the independent all-row oracle.

Example::

    python benchmarks/bench_real_input_data.py --data-root /path/to/captures \
        --model DS-V3.2 --scope api --length-axis original --lengths 8192 \
        --qkv fp8 --logits fp32 --arms full_gvr2 v1 v2_generic \
        --samples-per-config 1 --output results/captures
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

MODELS = {
    "DS-V3.2": (64, 2048, 1),
    "DS-V4-Flash": (64, 512, 4),
    "DS-V4-Pro": (64, 1024, 4),
    "GLM-5.3-Flash": (32, 512, 4),
    "GLM-5.3": (32, 2048, 1),
    "Qwen3.8-Flash-Next": (4, 512, 4),
}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference-root", type=Path, help="optional comparison suite checkout"
    )
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="DS-V3.2")
    p.add_argument("--scope", choices=("api", "kernel"), default="api")
    p.add_argument(
        "--length-axis",
        choices=("historical", "original", "history"),
        default="historical",
        help="historical: captured history/keys axis; original: total tokens before KV compression",
    )
    p.add_argument(
        "--lengths", type=int, nargs="+", default=[8192 * 2**i for i in range(8)]
    )
    p.add_argument(
        "--qkv", nargs="+", choices=("fp8", "mxfp4"), default=["fp8", "mxfp4"]
    )
    p.add_argument(
        "--logits",
        nargs="+",
        choices=("fp16", "bf16", "fp32"),
        default=None,
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline", "full_gvr2", "v1", "v2", "v2_generic", "source"),
        default=None,
    )
    p.add_argument(
        "--math-mode", choices=("relu_strict", "relu_fast", "native_bf16", "native_fp16"), default="relu_fast"
    )
    p.add_argument("--prefill-q", type=int, choices=(4096, 8192), default=8192)
    p.add_argument("--samples-per-config", type=int, default=120)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--max-jobs", type=int)
    p.add_argument(
        "--source-short-rows",
        action="store_true",
        help="Qualify optional source on rows shorter than TopK (requires its empty-tile fixes)",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="CUDA profiler range around one replay per arm",
    )
    p.add_argument(
        "--staged-inputs",
        action="store_true",
        help="Use public input_buffers() for producer-owned Q/weights/scales; stage once outside timing",
    )
    p.add_argument("--fp8-quantization", choices=("captured", "deepgemm"), default="captured", help="deepgemm: quantize once to E4M3+UE8M0/32 and share Q/K/scales between v2 and DG")
    p.add_argument("--gvr-root", type=Path, help="Unmodified BF16-capable TRT-LLM GVR prefill module directory for aligned DG comparisons")
    a = p.parse_args()
    a.arms = list(dict.fromkeys("v2" if arm == "v2_generic" else arm for arm in a.arms))
    if a.fp8_quantization == "deepgemm":
        if a.qkv != ["fp8"] or a.scope != "api" or not set(a.arms) <= {"v2", "deepgemm", "deepgemm_prepared"}:
            p.error("aligned FP8 requires --qkv fp8 --scope api and --arms v2 deepgemm [deepgemm_prepared]")
        if a.gvr_root is None:
            p.error("aligned FP8 requires --gvr-root")
    elif any(arm.startswith("deepgemm") for arm in a.arms):
        p.error("DeepGEMM arms require --fp8-quantization deepgemm")
    if not 0 <= a.shard < a.shards or min(a.samples_per_config, a.repetitions) <= 0:
        p.error("positive counts and 0 <= shard < shards required")
    if a.length_axis == "original" and (
        a.scope != "api"
        or any(n < a.prefill_q or n % MODELS[a.model][2] for n in a.lengths)
    ):
        p.error(
            "original lengths require API scope, length >= Q, and divisibility by compression ratio"
        )
    a.reference_root = a.reference_root.resolve() if a.reference_root else None
    if a.data_root is None and a.reference_root is None:
        p.error("--data-root is required without --reference-root")
    a.data_root = (a.data_root or a.reference_root.parent / "real_input_data").resolve()
    a.arms = a.arms or (
        ["full_gvr2", "v1", "v2", "source"]
        if a.reference_root
        else ["full_gvr2", "v1", "v2"]
    )
    a.logits = a.logits or (["fp16", "bf16", "fp32"] if a.reference_root else ["fp32"])
    if a.reference_root is None and (
        a.scope != "api" or a.logits != ["fp32"] or set(a.arms) & {"baseline", "source"}
    ):
        p.error(
            "standalone arms require API scope and FP32 logits; optional reference/kernel arms require --reference-root"
        )
    return a


def fingerprint(root, reference=None, data_root=None):
    digest = hashlib.sha256()
    files = [
        Path(__file__),
        root / "benchmarks/attn_scores_capture.py",
        root / "benchmarks/bench_fp8_mqa_selective_logits.py",
    ]
    for folder in (
        root / "flashinfer/experimental/attn_scores",
        root / "flashinfer/attn_scores",
        root / "flashinfer/topk_varlen",
    ):
        files.extend(sorted(folder.rglob("*.py")))
    if reference:
        files.extend(sorted(reference.rglob("*.py")))
    if data_root:
        files.extend(sorted((data_root / "indexer_inputs").glob("*.py")))
    for path in files:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configure_reference(root):
    """Discover the optional suite without assuming its checkout location."""
    suites = [
        p.parent
        for p in root.rglob("baseline.py")
        if (p.parent / "data.py").is_file() and (p.parent / "timing.py").is_file()
    ]
    if len(suites) != 1:
        raise ValueError(
            "reference checkout must contain one data/baseline/timing suite"
        )
    sys.path.insert(0, str(suites[0]))
    apis = [
        p
        for p in root.rglob("api.py")
        if (p.parent / "ordering.py").is_file() and (p.parent / "packed.py").is_file()
    ]
    if len(apis) != 1:
        raise ValueError(
            "reference checkout must contain one source API with ordering/packed helpers"
        )
    api = apis[0]
    package_root = next(p.parent for p in api.parents if p.name == "blackwell")
    sys.path.insert(0, str(package_root))
    return ".".join(api.relative_to(package_root).with_suffix("").parts[:-1])


def load_inputs(a, case, n, loader):
    import torch

    if a.scope == "kernel":
        return loader.load(a.data_root, case, n, a.prefill_q)
    # Load original bounds first, then translate a tail crop. This preserves
    # actual causal history, including compressed-key coordinates.
    inputs, metadata = loader.load(
        a.data_root, case, case["source_key_tokens"], a.prefill_q
    )
    crop = inputs["k"].shape[0] - n
    if crop < 0:
        raise ValueError(
            f"capture has {inputs['k'].shape[0]} keys, fewer than requested {n}"
        )
    inputs["k"] = inputs["k"][crop:].contiguous()
    if case["qkv_dtype"] == "mxfp4":
        # Captures can use signed packed bytes; the public API uses uint8 storage.
        inputs["q"] = inputs["q"].view(torch.uint8)
        inputs["k"] = inputs["k"].view(torch.uint8)
    if inputs["k_scale"] is not None:
        inputs["k_scale"] = inputs["k_scale"][crop:].contiguous()
    inputs["ends"] = (inputs["ends"] - crop).contiguous()
    ratio = MODELS[a.model][2]
    expected = (
        n * ratio
        - a.prefill_q
        + torch.arange(1, a.prefill_q + 1, device="cuda", dtype=torch.int32)
    ) // ratio
    if not torch.equal(inputs["ends"], expected) or bool(inputs["starts"].any()):
        raise ValueError("capture cannot represent public API causal prefixes")
    return inputs, metadata


def paged_inputs(inputs):
    import torch

    k = inputs["k"]
    page = 128
    n, width = k.shape
    scales = inputs["k_scale"]
    if scales is None:
        scales = torch.ones(n, device="cuda")
    pages = (n + page - 1) // page
    pool = torch.zeros((pages, page * (width + 4)), dtype=torch.uint8, device="cuda")
    key_bytes = torch.zeros((pages * page, width), dtype=torch.uint8, device="cuda")
    scale_bytes = torch.zeros((pages * page, 4), dtype=torch.uint8, device="cuda")
    key_bytes[:n].copy_(k.view(torch.uint8))
    scale_bytes[:n].copy_(scales.contiguous().view(torch.uint8).reshape(n, 4))
    pool[:, : page * width].copy_(key_bytes.view(pages, -1))
    pool[:, page * width :].copy_(scale_bytes.view(pages, -1))
    pool = pool.flip(0).contiguous().view(pages, page, 1, width + 4)
    table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32)[None]
    return pool, table


def continuous(a, inputs, dtype, backend, *, pool_inputs=None, generic=False):
    import torch
    from flashinfer.attn_scores import FP8MQATopKWrapper

    q, k = inputs["q"], inputs["k"]
    heads, topk, ratio = MODELS[a.model]
    n, _ = k.shape
    k_scale = inputs["k_scale"]
    if k_scale is None:
        k_scale = torch.ones(n, device=k.device, dtype=torch.float32)
    elif a.current_qkv == "mxfp4":
        k_scale = k_scale.view(torch.uint8).reshape(n, 4)
    weights_dtype = inputs["weights"].dtype
    wrapper = FP8MQATopKWrapper(
        backend=backend,
        num_heads=heads,
        head_dim=128,
        top_k=topk,
        compression_ratio=ratio,
        max_context_length=n * ratio,
        max_chunk_size=q.shape[0],
        max_batch_size=1,
        device=q.device,
        num_keys=n,
        qkv_dtype=a.current_qkv,
        logits_dtype=dtype,
        weights_dtype=weights_dtype,
        model_config=a.model if backend == "v2" and not generic else None,
        math_mode=a.math_mode,
    )
    qo = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    kv = torch.tensor([0, n * ratio], device="cuda", dtype=torch.int32)
    wrapper.plan(qo, kv)
    options = {}
    if inputs["q_scale"] is not None:
        options["q_scale"] = (
            inputs["q_scale"].view(torch.uint8).reshape(*q.shape[:2], 4)
        )
    weights = inputs["weights"]
    if a.staged_inputs:
        staged = wrapper.input_buffers()
        staged["q"].copy_(q)
        staged["weights"].copy_(weights)
        q, weights = staged["q"], staged["weights"]
        if "q_scale" in options:
            staged["q_scale"].copy_(
                options["q_scale"].view(torch.int32).reshape(q.shape[:2])
            )
            options["q_scale"] = staged["q_scale"]
    output = wrapper.run(q, weights, k, k_scale=k_scale, **options)

    def run():
        return wrapper.run(q, weights, k, k_scale=k_scale, **options)

    return wrapper, run, output


def full_gvr2(a, inputs, pool_inputs):
    """Use the FlashInfer Full/GVR2 baseline with FP32 strict scoring."""
    import torch
    from bench_fp8_mqa_selective_logits import _FullGVR
    from flashinfer.attn_scores.fp8_kernel_config import _FP8MQAKernelConfig
    from flashinfer.attn_scores.query_group_metadata import _FP8QueryGroups

    heads, topk, ratio = MODELS[a.model]
    q, k = inputs["q"], inputs["k"]
    pool, table = pool_inputs
    tile = 2 if a.current_qkv == "mxfp4" else 4
    groups = _FP8QueryGroups.build(
        [0, q.shape[0]], [0, k.shape[0] * ratio], ratio, tile
    )
    config = _FP8MQAKernelConfig(128, 1, 160, tile, num_q_stages=3, num_kv_stages=3)
    qo = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    kv = torch.tensor([0, k.shape[0] * ratio], device=q.device, dtype=torch.int32)
    qs = inputs["q_scale"]
    if qs is not None:
        qs = qs.contiguous().view(torch.int32).reshape(q.shape[:2])
    op = _FullGVR(
        q,
        pool,
        inputs["weights"],
        table,
        groups,
        config,
        topk,
        qo_indptr=qo,
        kv_indptr=kv,
        q_scale=qs,
    )
    op.step()
    op.layer()
    return op


def validate_full_native(op, inputs):
    """Tie-aware exact selection against the unchanged Full producer's scores."""
    import torch

    topk = op.output.shape[1]
    n = inputs["k"].shape[0]
    columns = torch.arange(n, device=op.output.device)[None]
    for first in range(0, op.output.shape[0], 64):
        last = min(first + 64, op.output.shape[0])
        scores = op.scores[first:last, :n]
        ends = inputs["ends"][first:last, None]
        masked = scores.masked_fill(columns >= ends, -float("inf"))
        expected = masked.topk(min(topk, n), dim=1).values.sort(dim=1).values
        actual = masked.gather(
            1, op.output[first:last, : expected.shape[1]].long().clamp(min=0)
        )
        actual = actual.masked_fill(
            torch.arange(actual.shape[1], device=actual.device)[None] >= ends,
            -float("inf"),
        )
        if not torch.equal(expected, actual.sort(dim=1).values):
            raise AssertionError(
                "Full GVR2 differs from its native FP32 score multiset"
            )
    return {"exact_native_fp32_score_multiset": True}


def validate(base, output, inputs, *, strict=True):
    import torch

    idx = output.long()
    required = (inputs["ends"] - inputs["starts"]).clamp(max=idx.shape[1])
    valid = torch.arange(idx.shape[1], device=idx.device)[None] < required[:, None]
    if bool(((idx < inputs["starts"][:, None]) & valid).any()) or bool(
        ((idx >= inputs["ends"][:, None]) & valid).any()
    ):
        raise AssertionError("index outside visible range")
    if bool((idx[~valid] != -1).any()):
        raise AssertionError("invalid suffix padding")
    ids = idx.masked_fill(~valid, inputs["k"].shape[0]).sort(-1).values
    duplicates = (ids[:, 1:] == ids[:, :-1]) & (ids[:, 1:] < inputs["k"].shape[0])
    if bool(duplicates.any()):
        raise AssertionError("duplicate index")
    reference = base.output.long() + inputs["starts"][:, None]
    expected = (
        base.gather_scores(reference.clamp(min=0))
        .float()
        .masked_fill(~valid, -float("inf"))
        .sort(-1)
        .values
    )
    actual = (
        base.gather_scores(idx.clamp(min=0))
        .float()
        .masked_fill(~valid, -float("inf"))
        .sort(-1)
        .values
    )
    exact = torch.equal(expected, actual)
    mismatches = int((expected != actual).sum())
    if strict and not exact:
        raise AssertionError(f"score multiset differs in {mismatches} entries")
    if not strict:
        # V1 and the optional reference can reduce FP32 scores in different orders. This is a separate
        # numerical check, never an exactness claim for v2 or the source arm.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    return dict(
        exact_score_multiset=exact,
        differing_scores=mismatches,
        max_abs_score_difference=(
            float((actual[finite] - expected[finite]).abs().max())
            if bool(finite.any())
            else 0
        ),
        rtol=0 if strict else 1e-6,
    )


def measure(calls, warmup, repetitions, profile):
    import torch
    from attn_scores_capture import capture

    graphs = {name: capture(call) for name, call in calls.items()}
    flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
    raw = {name: [] for name in calls}
    names = list(calls)
    for rep in range(warmup + repetitions):
        order = names[rep % len(names) :] + names[: rep % len(names)]
        if rep % 2:
            order.reverse()
        for name in order:
            flush.zero_()
            graph, begin, end = graphs[name]
            graph.replay()
            end.synchronize()
            if rep >= warmup:
                raw[name].append(begin.elapsed_time(end) * 1000)
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
        for name, (graph, _, _) in graphs.items():
            torch.cuda.nvtx.range_push(name)
            graph.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    if any(not math.isfinite(t) or t <= 0 for values in raw.values() for t in values):
        raise RuntimeError("invalid timing sample")
    return dict(
        raw_us=raw,
        mean_us={k: statistics.mean(v) for k, v in raw.items()},
        median_us={k: statistics.median(v) for k, v in raw.items()},
        warmup=warmup,
        repetitions=repetitions,
        cold_l2_bytes=flush.numel(),
        timer="CUDA graph external events",
        order="rotating and alternating",
    )


def execute_public(a, job, loader):
    import torch
    from types import SimpleNamespace
    from bench_fp8_mqa_selective_logits import (
        _check_reference,
        _decode_mxfp4,
        _workspace_inventory,
    )

    inputs, metadata = load_inputs(a, job["case"], job["n"], loader)
    a.current_qkv = job["case"]["qkv_dtype"]
    owners, calls, outputs, skipped, workspace = [], {}, {}, {}, {}
    pool = paged_inputs(inputs) if "full_gvr2" in a.arms else None
    for name in a.arms:
        if name == "v1" and a.current_qkv != "fp8":
            skipped[name] = "v1 requires FP8 inputs"
            continue
        if name == "full_gvr2":
            if a.current_qkv == "mxfp4" and MODELS[a.model][0] != 64:
                skipped[name] = "packed Full comparison requires H64"
                continue
            op = full_gvr2(a, inputs, pool)
            run, output = op.layer, op.output
        else:
            op, run, output = continuous(
                a,
                inputs,
                torch.float32,
                "v1" if name == "v1" else "v2",
                generic=name == "v2_generic",
            )
        owners.append(op)
        calls[name], outputs[name] = run, output
        workspace[name] = _workspace_inventory(
            op.workspace_owners if name == "full_gvr2" else op,
            tuple(v for v in (*inputs.values(), output) if isinstance(v, torch.Tensor)),
        )
    if not calls:
        raise ValueError("no supported FlashInfer arm")
    q, key = inputs["q"], inputs["k"]
    scale = inputs["k_scale"]
    if a.current_qkv == "mxfp4":
        q = _decode_mxfp4(q, inputs["q_scale"].reshape(*q.shape[:2], 4))
        key = _decode_mxfp4(key, scale)
        scale = None
    if scale is None:
        scale = torch.ones(key.shape[0], device=key.device)
    groups = SimpleNamespace(
        q_prefix=[0, q.shape[0]], real_ends=inputs["ends"].cpu().tolist()
    )

    def check():
        return _check_reference(
            outputs,
            q,
            [key],
            [scale],
            inputs["weights"],
            groups,
            MODELS[a.model][1],
            atol=5e-5,
            rtol=1e-5,
        )

    before = check()
    timing = measure(calls, a.warmup, a.repetitions, a.profile)
    after = check()
    for op in owners:
        if hasattr(op, "repair_error") and (
            bool(op.error.any()) or bool(op.repair_error.any())
        ):
            raise AssertionError("public API device error flags are nonzero")
    return dict(
        job=job,
        timing=timing,
        skipped=skipped,
        workspace=workspace,
        validation=dict(
            independent_fp32_before=before,
            independent_fp32_after=after,
            rows=q.shape[0],
        ),
        original_context=key.shape[0] * MODELS[a.model][2],
        source_metadata=metadata,
        baseline_chunk_rows=[],
        baseline_chunk_policy="independent all-row FP32 oracle",
    )


def execute(a, job, loader):
    if getattr(a, "fp8_quantization", "captured") == "deepgemm":
        import bench_mxfp8_alignment
        return bench_mxfp8_alignment.execute(sys.modules[__name__], a, job, loader)
    import torch
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline

    if a.reference_root is None:
        return execute_public(a, job, loader)
    inputs, metadata = load_inputs(a, job["case"], job["n"], loader)
    a.current_qkv = job["case"]["qkv_dtype"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        job["logits"]
    ]
    # The optional source installs compatibility methods. Compile its baseline
    # and source under those methods, then restore shared methods before FI JIT.
    saved = (
        pipeline.PipelineTmaAsync.consumer_release,
        cute.arch.fma_packed_f32x2,
        cute.arch.add_packed_f32x2,
    )
    from baseline import Baseline

    ThrowawayTopK = importlib.import_module(
        a.reference_namespace + ".api"
    ).ThrowawayTopK
    install = importlib.import_module(a.reference_namespace + ".ordering").install
    packed = importlib.import_module(a.reference_namespace + ".packed")

    install()
    cute.arch.fma_packed_f32x2 = packed.fma_packed_f32x2
    cute.arch.add_packed_f32x2 = packed.add_packed_f32x2
    args = dict(inputs, top_k=MODELS[a.model][1], logits_dtype=dtype)
    owners, calls, outputs, skipped = [], {}, {}, {}
    references, workspace = {}, {}
    try:
        base = Baseline(**args)
        base.run()
        full_reference = base
        if "full_gvr2" in a.arms and dtype != torch.float32:
            full_reference = Baseline(**dict(args, logits_dtype=torch.float32))
            full_reference.run()
        if "baseline" in a.arms:
            calls["baseline"] = base.run
        source_short = bool(
            ((inputs["ends"] - inputs["starts"]) < MODELS[a.model][1]).any()
        )
        if "source" in a.arms and source_short and not a.source_short_rows:
            skipped["source"] = (
                "original source is not qualified for underfilled API rows; use kernel scope for historical parity"
            )
        if "source" in a.arms and (not source_short or a.source_short_rows):
            source = ThrowawayTopK(**args, model_config=a.model)
            source.run()
            owners.append(source)
            calls["source"] = source.run
            outputs["source"] = source.output
    finally:
        (
            pipeline.PipelineTmaAsync.consumer_release,
            cute.arch.fma_packed_f32x2,
            cute.arch.add_packed_f32x2,
        ) = saved
    pool_inputs = (
        paged_inputs(inputs) if a.scope == "api" and "full_gvr2" in a.arms else None
    )
    if "full_gvr2" in a.arms:
        if a.scope != "api" or (a.current_qkv == "mxfp4" and MODELS[a.model][0] != 64):
            skipped["full_gvr2"] = (
                "MR Full baseline requires API scope; MXFP4 is qualified for H64 only"
            )
        else:
            from bench_fp8_mqa_selective_logits import _workspace_inventory

            full = full_gvr2(a, inputs, pool_inputs)
            owners.append(full)
            calls["full_gvr2"], outputs["full_gvr2"] = full.layer, full.output
            references["full_gvr2"] = full_reference
            workspace["full_gvr2"] = _workspace_inventory(
                full.workspace_owners,
                (*pool_inputs, inputs["q"], inputs["weights"], full.output),
            )
    for arm in ("v1", "v2"):
        if arm not in a.arms:
            continue
        if arm == "v1" and (
            a.scope == "kernel" or a.current_qkv != "fp8" or dtype != torch.float32
        ):
            skipped[arm] = (
                "v1 public API supports FP8 QK / FP32 logits; no captured-bounds kernel adapter"
            )
            continue
        if a.scope == "kernel":
            from flashinfer.experimental.attn_scores.kernels.indexer_topk_v2.api import (
                ThrowawayTopK as SelectiveTopK,
            )

            op = SelectiveTopK(
                **args,
                compression_ratio=MODELS[a.model][2],
                math_mode=a.math_mode,
            )
            run, output = op.run, op.output
        else:
            backend = "v1" if arm == "v1" else "v2"
            op, run, output = continuous(
                a,
                inputs,
                dtype,
                backend,
                pool_inputs=pool_inputs,
                generic=True,
            )
            from bench_fp8_mqa_selective_logits import _workspace_inventory

            workspace[arm] = _workspace_inventory(
                op,
                tuple(
                    t for t in (*inputs.values(), output) if isinstance(t, torch.Tensor)
                ),
            )
        run()
        owners.append(op)
        calls[arm], outputs[arm] = run, output
    torch.cuda.synchronize()
    validation = {}
    for name, output in outputs.items():
        try:
            reference = references.get(name, base)
            strict = name == "source" or (
                name.startswith("v2") and a.math_mode == "relu_fast"
            )
            validate(reference, output, inputs, strict=strict)
            calls[name]()
            validation[name] = validate(reference, output, inputs, strict=strict)
            if name == "full_gvr2":
                validation[name].update(validate_full_native(full, inputs))
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error
    timing = measure(calls, a.warmup, a.repetitions, a.profile)
    for name, output in outputs.items():
        try:
            strict = name == "source" or (
                name.startswith("v2") and a.math_mode == "relu_fast"
            )
            validation[name]["timed_graph"] = validate(
                references.get(name, base), output, inputs, strict=strict
            )
            if name == "full_gvr2":
                validation[name]["timed_graph"].update(
                    validate_full_native(full, inputs)
                )
        except AssertionError as error:
            raise AssertionError(f"{name} timed graph: {error}") from error
    for op in owners:
        if hasattr(op, "repair_error"):
            if bool(op.error.any()) or bool(op.repair_error.any()):
                raise AssertionError("public API device error flags are nonzero")
    return dict(
        job=job,
        timing=timing,
        skipped=skipped,
        validation=validation,
        workspace=workspace,
        original_context=inputs["k"].shape[0] * MODELS[a.model][2],
        full_logits_dtype="fp32" if "full_gvr2" in calls else None,
        full_math_mode="relu_strict" if "full_gvr2" in calls else None,
        source_metadata=metadata,
        baseline_chunk_rows=[r.stop - r.start for r, _ in base.chunks] or [a.prefill_q],
        baseline_chunk_policy=base.chunk_policy,
    )


def main():
    a = parse()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "benchmarks"))
    if a.reference_root:
        a.reference_namespace = configure_reference(a.reference_root)
        loader = importlib.import_module("data")
    else:
        from attn_scores_capture import CaptureLoader

        loader = CaptureLoader(a.prefill_q)
    code = fingerprint(root, a.reference_root, a.data_root)
    jobs = []
    ratio = MODELS[a.model][2]
    for encoding in a.qkv:
        for length in a.lengths:
            requested = requested_key_length(length, a.length_axis, ratio, a.prefill_q)
            cases = loader.choose(
                a.data_root,
                requested,
                encoding,
                a.samples_per_config,
                model=a.model,
                length_axis=(
                    "history"
                    if ratio == 4 and a.length_axis == "historical"
                    else "keys"
                ),
            )
            for logits in a.logits:
                for case in cases:
                    stat = (a.data_root / case["file"]).stat()
                    job = dict(
                        case=case,
                        n=case.get("gemm_key_tokens", requested),
                        length=length,
                        length_axis=a.length_axis,
                        physical_key_tokens=case.get("gemm_key_tokens", requested),
                        total_tokens=case.get("gemm_key_tokens", requested) * ratio,
                        history_tokens=(
                            case.get("gemm_key_tokens", requested) * ratio - a.prefill_q
                        )
                        if a.scope == "api"
                        else None,
                        logits=logits,
                        scope=a.scope,
                        input_stat=[stat.st_size, stat.st_mtime_ns],
                    )
                    job["id"] = hashlib.sha256(
                        json.dumps(job, sort_keys=True).encode()
                    ).hexdigest()[:24]
                    jobs.append(job)
    if not jobs:
        raise ValueError("no matching captures in --data-root")
    selected = jobs[a.shard :: a.shards]
    if a.max_jobs:
        selected = selected[: a.max_jobs]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    plan = dict(code_sha256=code, config=config, jobs=selected, total_jobs=len(jobs))
    path = a.output / f"plan_{a.shard}.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("source/config/input changed; use a new output directory")
    atomic(path, plan)
    print("PLAN", len(selected), flush=True)
    if a.plan_only:
        return
    import torch

    atomic(
        a.output / f"environment_{a.shard}.json",
        dict(
            host=platform.node(),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        ),
    )
    failures = 0
    for job in selected:
        path = a.output / "records" / f"{job['id']}.json"
        if path.exists() and json.loads(path.read_text()).get("code_sha256") == code:
            continue
        atomic(a.output / f"active_{a.shard}.json", job)
        try:
            record = execute(a, job, loader)
            record.update(code_sha256=code, completed_at=time.time())
            atomic(path, record)
            print("DONE", job["id"], record["timing"]["mean_us"], flush=True)
        except Exception:
            failures += 1
            atomic(
                a.output / "errors" / f"{job['id']}.json",
                dict(job=job, traceback=traceback.format_exc()),
            )
            traceback.print_exc()
        torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"{failures} cases failed")


if __name__ == "__main__":
    main()
