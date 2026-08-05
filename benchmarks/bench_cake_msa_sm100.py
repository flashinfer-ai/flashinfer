# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark FlashInfer's public MSA API against pinned MiniMax MSA.

Each measured backend/shape pair runs in a fresh process because
``flashinfer.testing.bench_gpu_time`` finalizes CUPTI after collecting its
samples.  Timings are cold-L2 spans from the first to the last correlated GPU
activity (kernel, memcpy, or memset) launched by exactly one public API call.
Each comparable row is first checked in another isolated process that invokes
both public APIs on the same tensor objects.

The six cases reproduce the canonical SM100/SM103 performance matrix.  The
pinned MiniMax public sparse-forward API supports BF16 and FP8 E4M3 storage,
but not FP16 input, so the FP16 row is reported explicitly as unsupported and
never cast to another dtype.

Example
-------
Clone the baseline at the pinned revision, then run from a clean FlashInfer
checkout::

    python benchmarks/bench_cake_msa_sm100.py \
      --expected-source-root "$PWD" \
      --expected-source-sha "$(git rev-parse HEAD)" \
      --baseline-root /path/to/MSA \
      --json /tmp/msa-sm100.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable


BASELINE_REPOSITORY = "https://github.com/MiniMax-AI/MSA.git"
BASELINE_SHA = "80434d7f67877c6570ca19cac444b84bc9855dac"
SOURCE_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer.git"
BLOCK_SIZE = 128
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
TOPK = 16
SUPPORTED_ARCHITECTURES = {(10, 0): "sm100a", (10, 3): "sm103a"}
ACTIVITY_SCOPE = "first_to_last_correlated_gpu_activity_for_one_public_api_call"
CORRECTNESS_TOLERANCES = {
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
    "float8_e4m3fn": {"atol": 0.1, "rtol": 0.1},
}

SHAPES: tuple[dict[str, Any], ...] = (
    {
        "label": "prefill_bf16_b1_q4096_kv4096_h64",
        "operation": "sparse_prefill",
        "batch_size": 1,
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "q_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "kv_layout": "flat_varlen",
        "causal": True,
        "force_fused": None,
        "seed": 43,
    },
    {
        "label": "decode_bf16_b128_q1_kv4096_h64",
        "operation": "sparse_decode",
        "batch_size": 128,
        "seqlen_q": 1,
        "seqlen_kv": 4096,
        "q_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "kv_layout": "flat_varlen",
        "causal": True,
        "force_fused": True,
        "seed": 47,
    },
    {
        "label": "speculative_bf16_b128_q4_kv4096_h64",
        "operation": "sparse_decode",
        "batch_size": 128,
        "seqlen_q": 4,
        "seqlen_kv": 4096,
        "q_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "kv_layout": "flat_varlen",
        "causal": True,
        "force_fused": True,
        "seed": 48,
    },
    {
        "label": "mtp_bf16_b128_q16_kv4096_h64",
        "operation": "sparse_decode",
        "batch_size": 128,
        "seqlen_q": 16,
        "seqlen_kv": 4096,
        "q_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "kv_layout": "flat_varlen",
        "causal": True,
        "force_fused": True,
        "seed": 50,
    },
    {
        "label": "decode_fp16_b128_q1_kv4096_h64",
        "operation": "sparse_decode",
        "batch_size": 128,
        "seqlen_q": 1,
        "seqlen_kv": 4096,
        "q_dtype": "float16",
        "kv_dtype": "float16",
        "kv_layout": "flat_varlen",
        "causal": True,
        "force_fused": True,
        "seed": 49,
    },
    {
        "label": "decode_fp8_b128_q1_kv4096_h64",
        "operation": "sparse_decode",
        "batch_size": 128,
        "seqlen_q": 1,
        "seqlen_kv": 4096,
        "q_dtype": "bfloat16",
        "kv_dtype": "float8_e4m3fn",
        "kv_layout": "paged",
        "causal": True,
        "force_fused": True,
        "seed": 53,
    },
)
SHAPES_BY_LABEL = {shape["label"]: shape for shape in SHAPES}


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checkout(root: Path, expected_sha: str, name: str) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"{name} checkout does not exist: {root}")
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise RuntimeError(f"{name} root must be {top_level}, got {root}")
    actual_sha = _git_output(root, "rev-parse", "HEAD")
    if actual_sha != expected_sha:
        raise RuntimeError(f"{name} must be at {expected_sha}, got {actual_sha}")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"{name} checkout must be clean:\n{status}")
    return actual_sha


def _validate_script_root(source_root: Path) -> None:
    script_root = Path(__file__).resolve().parents[1]
    if script_root != source_root:
        raise RuntimeError(
            f"benchmark script must come from {source_root}, got {script_root}"
        )


def _require_cupti() -> str:
    try:
        from cupti import cupti

        cupti_python_version = version("cupti-python")
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("reportable timings require cupti-python >= 13") from error
    del cupti
    try:
        major = int(cupti_python_version.split(".", 1)[0])
    except ValueError as error:
        raise RuntimeError(
            f"could not parse cupti-python version {cupti_python_version!r}"
        ) from error
    if major < 13:
        raise RuntimeError(
            f"reportable timings require cupti-python >= 13, got {cupti_python_version}"
        )
    return cupti_python_version


def _torch_dtype(torch, name: str):
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}") from error


def _make_q2k(torch, shape: dict[str, Any], device) -> Any:
    """Reproduce the canonical random-valid, bottom-right-causal selection."""

    batch_size = int(shape["batch_size"])
    seqlen_q = int(shape["seqlen_q"])
    seqlen_kv = int(shape["seqlen_kv"])
    total_q = batch_size * seqlen_q
    output = torch.full(
        (NUM_KV_HEADS, total_q, TOPK),
        -1,
        dtype=torch.int32,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(shape["seed"]) + 101)
    all_blocks = (seqlen_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    q_start = 0
    for _ in range(batch_size):
        offset = seqlen_kv - seqlen_q
        for local_q in range(seqlen_q):
            visible_tokens = offset + local_q + 1
            visible_blocks = max(
                0,
                min(
                    all_blocks,
                    (visible_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
                ),
            )
            for kv_head in range(NUM_KV_HEADS):
                candidates = torch.randperm(visible_blocks, generator=generator)
                selected = candidates[: min(TOPK, visible_blocks)].sort().values
                output[kv_head, q_start + local_q, : selected.numel()] = selected.to(
                    torch.int32
                )
        q_start += seqlen_q
    return output.to(device=device).contiguous()


def _make_paged_cache(torch, logical, *, batch_size: int, seqlen_kv: int):
    pages_per_sequence = seqlen_kv // BLOCK_SIZE
    total_pages = batch_size * pages_per_sequence
    logical_pages = (
        logical.view(
            batch_size,
            pages_per_sequence,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    )
    paged = logical_pages.flip(0).contiguous()
    page_table = torch.arange(
        total_pages - 1,
        -1,
        -1,
        dtype=torch.int32,
        device=logical.device,
    ).view(batch_size, pages_per_sequence)
    return paged, page_table.contiguous()


def _make_inputs(torch, shape: dict[str, Any], device) -> dict[str, Any]:
    batch_size = int(shape["batch_size"])
    seqlen_q = int(shape["seqlen_q"])
    seqlen_kv = int(shape["seqlen_kv"])
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q_dtype = _torch_dtype(torch, str(shape["q_dtype"]))
    kv_dtype = _torch_dtype(torch, str(shape["kv_dtype"]))
    generator = torch.Generator(device=device).manual_seed(int(shape["seed"]))

    q = (
        torch.randn(
            (total_q, NUM_Q_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 3.0
    ).to(q_dtype)
    logical_k = (
        torch.randn(
            (total_k, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 3.0
    ).to(kv_dtype)
    logical_v = (
        torch.randn(
            (total_k, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 3.0
    ).to(kv_dtype)

    cu_q = torch.arange(
        0,
        (batch_size + 1) * seqlen_q,
        seqlen_q,
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.arange(
        0,
        (batch_size + 1) * seqlen_kv,
        seqlen_kv,
        dtype=torch.int32,
        device=device,
    )
    q2k = _make_q2k(torch, shape, device)

    page_table = None
    seqused_k = None
    if shape["kv_layout"] == "flat_varlen":
        k = logical_k.contiguous()
        v = logical_v.contiguous()
    elif shape["kv_layout"] == "paged":
        if seqlen_kv % BLOCK_SIZE:
            raise ValueError("paged canonical shapes require a full final page")
        k, page_table = _make_paged_cache(
            torch,
            logical_k,
            batch_size=batch_size,
            seqlen_kv=seqlen_kv,
        )
        v, v_page_table = _make_paged_cache(
            torch,
            logical_v,
            batch_size=batch_size,
            seqlen_kv=seqlen_kv,
        )
        if not torch.equal(page_table, v_page_table):
            raise RuntimeError("K/V page tables differ")
        seqused_k = torch.full(
            (batch_size,),
            seqlen_kv,
            dtype=torch.int32,
            device=device,
        )
    else:
        raise ValueError(f"unsupported KV layout {shape['kv_layout']!r}")

    return {
        "q": q.contiguous(),
        "k": k,
        "v": v,
        "q2k": q2k,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
    }


def _candidate_call(
    shape: dict[str, Any], inputs: dict[str, Any]
) -> tuple[Callable[[], Any], str, dict[str, Any]]:
    msa_ops = importlib.import_module("flashinfer.msa_ops")
    if shape["operation"] == "sparse_prefill":
        public_api = "flashinfer.msa_ops.msa_sparse_attention"

        def call():
            return msa_ops.msa_sparse_attention(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["q2k"],
                inputs["cu_q"],
                inputs["cu_k"],
                causal=bool(shape["causal"]),
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                return_softmax_lse=False,
            )

    else:
        public_api = "flashinfer.msa_ops.msa_sparse_decode_attention"

        def call():
            return msa_ops.msa_sparse_decode_attention(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["q2k"],
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                cu_seqlens_k=inputs["cu_k"],
                seqlen_q=int(shape["seqlen_q"]),
                causal=bool(shape["causal"]),
                return_softmax_lse=False,
                force_fused=shape["force_fused"],
            )

    return call, public_api, {"excluded_setup": ["deterministic_input_construction"]}


def _baseline_call(
    torch, shape: dict[str, Any], inputs: dict[str, Any]
) -> tuple[Callable[[], Any], str, dict[str, Any]]:
    if shape["q_dtype"] == "float16":
        raise RuntimeError("the pinned public baseline does not accept FP16 input")
    baseline = importlib.import_module("fmha_sm100")
    k2q_row_ptr, k2q_q_indices, schedule = baseline.build_k2q_csr(
        inputs["q2k"],
        inputs["cu_q"],
        inputs["cu_k"],
        BLOCK_SIZE,
        total_k=int(shape["batch_size"]) * int(shape["seqlen_kv"]),
        max_seqlen_k=int(shape["seqlen_kv"]),
        max_seqlen_q=int(shape["seqlen_q"]),
        total_rows=(int(shape["batch_size"]) * int(shape["seqlen_kv"]) // BLOCK_SIZE),
        qhead_per_kv=NUM_Q_HEADS // NUM_KV_HEADS,
        return_schedule=True,
    )
    torch.cuda.synchronize()
    public_api = "fmha_sm100.sparse_atten_func"

    def call():
        return baseline.sparse_atten_func(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            k2q_row_ptr,
            k2q_q_indices,
            TOPK,
            cu_seqlens_q=inputs["cu_q"],
            cu_seqlens_k=inputs["cu_k"],
            max_seqlen_q=int(shape["seqlen_q"]),
            max_seqlen_k=int(shape["seqlen_kv"]),
            blk_kv=BLOCK_SIZE,
            causal=bool(shape["causal"]),
            return_softmax_lse=False,
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            schedule=schedule,
        )

    setup = {
        "excluded_setup": [
            "deterministic_input_construction",
            "fmha_sm100.build_k2q_csr",
            "sparse_forward_schedule_construction",
        ]
    }
    return call, public_api, setup


def _primary_output(value):
    if isinstance(value, dict):
        for name in ("out", "output"):
            if name in value:
                return value[name]
        raise RuntimeError("public API result dictionary has no output tensor")
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("public API returned an empty result")
        return value[0]
    return value


def _verify_public_outputs(
    torch,
    shape: dict[str, Any],
    candidate_call: Callable[[], Any],
    baseline_call: Callable[[], Any],
    *,
    candidate_api: str,
    baseline_api: str,
) -> dict[str, Any]:
    candidate_output = _primary_output(candidate_call())
    baseline_output = _primary_output(baseline_call())
    torch.cuda.synchronize()
    expected_shape = (
        int(shape["batch_size"]) * int(shape["seqlen_q"]),
        NUM_Q_HEADS,
        HEAD_DIM,
    )
    shape_matches = (
        tuple(candidate_output.shape) == expected_shape
        and tuple(baseline_output.shape) == expected_shape
    )
    dtype_matches = candidate_output.dtype == baseline_output.dtype
    tolerance = CORRECTNESS_TOLERANCES[str(shape["kv_dtype"])]
    if not shape_matches or not dtype_matches:
        return {
            "status": "failed",
            "passed": False,
            "candidate_public_api": candidate_api,
            "baseline_public_api": baseline_api,
            "same_q_k_v_tensor_objects": True,
            "same_sequence_metadata_tensor_objects": True,
            "same_page_table_argument": True,
            "baseline_csr_built_from_same_q2k_tensor": True,
            "expected_shape": list(expected_shape),
            "candidate_shape": list(candidate_output.shape),
            "baseline_shape": list(baseline_output.shape),
            "candidate_dtype": str(candidate_output.dtype),
            "baseline_dtype": str(baseline_output.dtype),
            **tolerance,
            "max_abs_error": None,
            "mismatch_count": None,
        }

    candidate_float = candidate_output.float()
    baseline_float = baseline_output.float()
    close = torch.isclose(
        candidate_float,
        baseline_float,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
        equal_nan=False,
    )
    candidate_nonfinite_count = int((~torch.isfinite(candidate_float)).sum().item())
    baseline_nonfinite_count = int((~torch.isfinite(baseline_float)).sum().item())
    passed = (
        bool(close.all().item())
        and candidate_nonfinite_count == 0
        and baseline_nonfinite_count == 0
    )
    mismatch_count = int((~close).sum().item())
    finite = torch.isfinite(candidate_float) & torch.isfinite(baseline_float)
    finite_count = int(finite.sum().item())
    max_abs_error = None
    if finite_count:
        max_abs_error = float(
            (candidate_float[finite] - baseline_float[finite]).abs().max().item()
        )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reference": "pinned_public_fmha_sm100_sparse_atten_func",
        "candidate_public_api": candidate_api,
        "baseline_public_api": baseline_api,
        "same_q_k_v_tensor_objects": True,
        "same_sequence_metadata_tensor_objects": True,
        "same_page_table_argument": True,
        "baseline_csr_built_from_same_q2k_tensor": True,
        "expected_shape": list(expected_shape),
        "candidate_dtype": str(candidate_output.dtype),
        "baseline_dtype": str(baseline_output.dtype),
        **tolerance,
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
        "candidate_nonfinite_count": candidate_nonfinite_count,
        "baseline_nonfinite_count": baseline_nonfinite_count,
    }


def _hardware(torch) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "this benchmark requires exact CC 10.0 (SM100a) or CC 10.3 "
            f"(SM103a), got CC {compute_capability[0]}.{compute_capability[1]}"
        )
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": properties.name,
        "compute_capability": list(compute_capability),
        "cuda_arch": SUPPORTED_ARCHITECTURES[compute_capability],
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    }


def _measure_strict_cupti(
    timing_utils,
    call: Callable[[], Any],
    *,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    def reject_fallback(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUPTI fallback is forbidden for reportable timings")

    original_event = timing_utils.bench_gpu_time_with_cuda_event
    original_graph = timing_utils.bench_gpu_time_with_cudagraph
    timing_utils.bench_gpu_time_with_cuda_event = reject_fallback
    timing_utils.bench_gpu_time_with_cudagraph = reject_fallback
    try:
        measured = timing_utils.bench_gpu_time(
            call,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=False,
            dry_run_iters=warmup,
            repeat_iters=samples,
        )
    finally:
        timing_utils.bench_gpu_time_with_cudagraph = original_graph
        timing_utils.bench_gpu_time_with_cuda_event = original_event

    samples_ms = [float(value) for value in measured]
    if len(samples_ms) != samples:
        raise RuntimeError(f"expected {samples} CUPTI samples, got {len(samples_ms)}")
    if any(not math.isfinite(value) or value <= 0.0 for value in samples_ms):
        raise RuntimeError(f"invalid CUPTI samples: {samples_ms}")
    median_ms = float(statistics.median(samples_ms))
    return {
        "timing_backend": "CUPTI",
        "cold_l2": True,
        "cuda_graph": False,
        "activity_scope": ACTIVITY_SCOPE,
        "included_gpu_activities": ["concurrent_kernel", "memcpy", "memset"],
        "single_public_api_call_per_sample": True,
        "samples_ms": samples_ms,
        "sample_count": len(samples_ms),
        "median_ms": median_ms,
        "sampling_protocol": {
            "initial_untimed_calls": 6,
            "additional_warmup_calls": warmup,
            "timed_calls": samples,
        },
    }


def _configure_imports(source_root: Path, baseline_root: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(baseline_root / "python"))
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    imported_source = Path(flashinfer.__file__).resolve().parents[1]
    if imported_source != source_root:
        raise RuntimeError(
            f"expected flashinfer from {source_root}, imported {imported_source}"
        )
    return torch, flashinfer


def _run_worker(args: argparse.Namespace) -> None:
    source_root = args.expected_source_root.resolve()
    baseline_root = args.baseline_root.resolve()
    _validate_script_root(source_root)
    source_sha = _validate_checkout(
        source_root, args.expected_source_sha, "FlashInfer source"
    )
    baseline_sha = _validate_checkout(baseline_root, BASELINE_SHA, "MiniMax baseline")
    if not (baseline_root / "python" / "fmha_sm100").is_dir():
        raise RuntimeError("baseline checkout does not contain python/fmha_sm100")
    cupti_python_version = _require_cupti()
    torch, flashinfer = _configure_imports(source_root, baseline_root)
    hardware = _hardware(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    shape = SHAPES_BY_LABEL[args.worker_shape]
    inputs = _make_inputs(torch, shape, device)
    software = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "cupti_python_version": cupti_python_version,
    }
    if args.worker_backend == "verify":
        imported_baseline = importlib.import_module("fmha_sm100")
        imported_baseline_root = Path(imported_baseline.__file__).resolve().parents[2]
        if imported_baseline_root != baseline_root:
            raise RuntimeError(
                f"expected fmha_sm100 from {baseline_root}, "
                f"imported {imported_baseline_root}"
            )
        candidate_call, candidate_api, _ = _candidate_call(shape, inputs)
        baseline_call, baseline_api, _ = _baseline_call(torch, shape, inputs)
        correctness = _verify_public_outputs(
            torch,
            shape,
            candidate_call,
            baseline_call,
            candidate_api=candidate_api,
            baseline_api=baseline_api,
        )
        result = {
            "status": "verified" if correctness["passed"] else "failed",
            "backend": "verify",
            "shape": shape["label"],
            "correctness": correctness,
            "source_sha": source_sha,
            "baseline_sha": baseline_sha,
            "hardware": hardware,
            "software": software,
        }
        args.worker_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return
    if args.worker_backend == "flashinfer":
        call, public_api, setup = _candidate_call(shape, inputs)
    else:
        imported_baseline = importlib.import_module("fmha_sm100")
        imported_baseline_root = Path(imported_baseline.__file__).resolve().parents[2]
        if imported_baseline_root != baseline_root:
            raise RuntimeError(
                f"expected fmha_sm100 from {baseline_root}, "
                f"imported {imported_baseline_root}"
            )
        call, public_api, setup = _baseline_call(torch, shape, inputs)

    torch.cuda.synchronize()
    timing_utils = importlib.import_module("flashinfer.testing.utils")
    timing = _measure_strict_cupti(
        timing_utils,
        call,
        samples=args.samples,
        warmup=args.warmup,
    )
    result = {
        "status": "measured",
        "backend": args.worker_backend,
        "public_api": public_api,
        **setup,
        **timing,
        "shape": shape["label"],
        "source_sha": source_sha,
        "baseline_sha": baseline_sha,
        "hardware": hardware,
        "software": software,
    }
    args.worker_json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_isolated(
    args: argparse.Namespace,
    *,
    backend: str,
    shape: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--expected-source-root",
        str(args.expected_source_root.resolve()),
        "--expected-source-sha",
        args.expected_source_sha,
        "--baseline-root",
        str(args.baseline_root.resolve()),
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--worker-backend",
        backend,
        "--worker-shape",
        shape["label"],
        "--worker-json",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"isolated {backend}/{shape['label']} worker failed "
            f"with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output.is_file():
        raise RuntimeError(f"worker did not write {output}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("backend") != backend or result.get("shape") != shape["label"]:
        raise RuntimeError(f"worker returned mismatched result: {result}")
    return result


def _public_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        **shape,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "topk": TOPK,
        "block_size": BLOCK_SIZE,
    }


def _validate_common_metadata(results: list[dict[str, Any]]) -> None:
    if not results:
        raise RuntimeError("no measurements were collected")
    expected_hardware = results[0]["hardware"]
    expected_software = results[0]["software"]
    for result in results:
        if result["hardware"] != expected_hardware:
            raise RuntimeError("workers ran on different hardware")
        if result["software"] != expected_software:
            raise RuntimeError("workers used different software environments")
        if result["activity_scope"] != ACTIVITY_SCOPE:
            raise RuntimeError("worker reported an unexpected activity scope")
        if result["timing_backend"] != "CUPTI":
            raise RuntimeError("worker did not use CUPTI")
        if result["source_sha"] != results[0]["source_sha"]:
            raise RuntimeError("workers used different FlashInfer revisions")
        if result["baseline_sha"] != BASELINE_SHA:
            raise RuntimeError("worker used the wrong baseline revision")


def _validate_correctness_metadata(
    results: list[dict[str, Any]], measured_reference: dict[str, Any]
) -> None:
    if len(results) != 5:
        raise RuntimeError(f"expected five correctness results, got {len(results)}")
    for result in results:
        if result["hardware"] != measured_reference["hardware"]:
            raise RuntimeError("correctness and timing workers used different hardware")
        if result["software"] != measured_reference["software"]:
            raise RuntimeError(
                "correctness and timing workers used different software environments"
            )
        if result["source_sha"] != measured_reference["source_sha"]:
            raise RuntimeError(
                "correctness worker used a different FlashInfer revision"
            )
        if result["baseline_sha"] != BASELINE_SHA:
            raise RuntimeError("correctness worker used the wrong baseline revision")
        if result["status"] != "verified" or not result["correctness"]["passed"]:
            raise RuntimeError(
                f"public output parity failed for {result['shape']}: "
                f"{result['correctness']}"
            )


def _unsupported_baseline() -> dict[str, Any]:
    return {
        "status": "unsupported",
        "public_api": "fmha_sm100.sparse_atten_func",
        "reason": (
            "The pinned public sparse-forward API accepts BF16 or FP8 E4M3 "
            "Q/K/V storage, not FP16 input."
        ),
        "evidence": {
            "source_path": "python/fmha_sm100/cute/interface.py",
            "symbol": "_SUPPORTED_FWD_DTYPES",
            "baseline_sha": BASELINE_SHA,
        },
    }


def _run_parent(args: argparse.Namespace) -> None:
    source_root = args.expected_source_root.resolve()
    baseline_root = args.baseline_root.resolve()
    _validate_script_root(source_root)
    source_sha = _validate_checkout(
        source_root, args.expected_source_sha, "FlashInfer source"
    )
    baseline_sha = _validate_checkout(baseline_root, BASELINE_SHA, "MiniMax baseline")
    if not (baseline_root / "python" / "fmha_sm100").is_dir():
        raise RuntimeError("baseline checkout does not contain python/fmha_sm100")

    rows = []
    measured_results: list[dict[str, Any]] = []
    correctness_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="flashinfer-msa-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, shape in enumerate(SHAPES):
            comparable = shape["q_dtype"] != "float16"
            if comparable:
                print(
                    f"Verifying public output parity for {shape['label']}", flush=True
                )
                correctness_worker = _run_isolated(
                    args,
                    backend="verify",
                    shape=shape,
                    output=temp_root / f"{index}-verify.json",
                )
                correctness_results.append(correctness_worker)
                if not correctness_worker["correctness"]["passed"]:
                    raise RuntimeError(
                        f"public output parity failed for {shape['label']}: "
                        f"{correctness_worker['correctness']}"
                    )
                correctness = correctness_worker["correctness"]
            else:
                correctness = {
                    "status": "not_run",
                    "passed": None,
                    "reason": "official_baseline_unsupported",
                }
            if comparable and index % 2 == 0:
                process_order = ("minimax", "flashinfer")
            elif comparable:
                process_order = ("flashinfer", "minimax")
            else:
                process_order = ("flashinfer",)
            print(
                f"Measuring {shape['label']} ({', '.join(process_order)})",
                flush=True,
            )
            by_backend = {}
            for backend in process_order:
                worker_output = temp_root / f"{index}-{backend}.json"
                result = _run_isolated(
                    args,
                    backend=backend,
                    shape=shape,
                    output=worker_output,
                )
                by_backend[backend] = result
                measured_results.append(result)

            candidate = by_backend["flashinfer"]
            if comparable:
                baseline = by_backend["minimax"]
                speedup = baseline["median_ms"] / candidate["median_ms"]
                if not math.isfinite(speedup) or speedup <= 0.0:
                    raise RuntimeError(
                        f"invalid speedup for {shape['label']}: {speedup}"
                    )
                comparison_status = "measured"
            else:
                baseline = _unsupported_baseline()
                speedup = None
                comparison_status = "official_baseline_unsupported"
            row = {
                "shape": _public_shape(shape),
                "comparison_status": comparison_status,
                "correctness": correctness,
                "correctness_process": (
                    "separate_untimed_public_api_parity_worker" if comparable else None
                ),
                "process_order": list(process_order),
                "baseline": baseline,
                "candidate": candidate,
                "speedup_baseline_over_candidate": speedup,
                "source_sha": source_sha,
                "baseline_sha": baseline_sha,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    _validate_checkout(source_root, source_sha, "FlashInfer source")
    _validate_checkout(baseline_root, baseline_sha, "MiniMax baseline")
    _validate_common_metadata(measured_results)
    _validate_correctness_metadata(correctness_results, measured_results[0])
    comparable_speedups = [
        row["speedup_baseline_over_candidate"]
        for row in rows
        if row["speedup_baseline_over_candidate"] is not None
    ]
    if len(comparable_speedups) != 5:
        raise RuntimeError(
            f"expected five comparable rows, got {len(comparable_speedups)}"
        )
    geometric_mean = math.exp(
        sum(math.log(value) for value in comparable_speedups) / len(comparable_speedups)
    )
    first = measured_results[0]
    result = {
        "schema_version": 1,
        "repositories": {
            "candidate": {
                "repository": SOURCE_REPOSITORY,
                "source_sha": source_sha,
            },
            "baseline": {
                "repository": BASELINE_REPOSITORY,
                "baseline_sha": baseline_sha,
            },
        },
        "hardware": first["hardware"],
        "software": first["software"],
        "protocol": {
            "timing_backend": "CUPTI",
            "cold_l2": True,
            "cuda_graph": False,
            "activity_scope": ACTIVITY_SCOPE,
            "included_gpu_activities": [
                "concurrent_kernel",
                "memcpy",
                "memset",
            ],
            "one_public_api_call_per_sample": True,
            "worker_isolation": "one_process_per_measured_backend_shape_pair",
            "correctness_worker_isolation": (
                "one_separate_untimed_process_per_comparable_shape"
            ),
            "correctness_reference": "pinned_public_fmha_sm100_sparse_atten_func",
            "baseline_api_selection": (
                "sparse_atten_func supports the required flat BF16 and mixed "
                "BF16-query/paged-FP8-KV inputs; the pinned decode API requires "
                "paged FP8 Q/K/V."
            ),
            "fallback_policy": "reject",
            "samples_per_pair": args.samples,
            "additional_warmup_calls_per_pair": args.warmup,
            "speedup_formula": "baseline_median_ms / candidate_median_ms",
            "input_identity": (
                "Both backends reconstruct identical tensors, sparse block "
                "selections, sequence metadata, and page tables from each "
                "row's recorded seed and shape."
            ),
        },
        "matrix": {
            "shape_count": len(SHAPES),
            "comparable_shape_count": 5,
            "official_baseline_unsupported_shape_count": 1,
            "output_parity_checked_shape_count": len(correctness_results),
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "topk": TOPK,
            "block_size": BLOCK_SIZE,
        },
        "rows": rows,
        "summary": {
            "all_required_measurements_valid": True,
            "all_comparable_outputs_match": True,
            "measured_comparisons": len(comparable_speedups),
            "geometric_mean_speedup": geometric_mean,
            "minimum_speedup": min(comparable_speedups),
            "maximum_speedup": max(comparable_speedups),
        },
    }
    args.json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.json}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker-backend",
        choices=("flashinfer", "minimax", "verify"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-shape",
        choices=tuple(SHAPES_BY_LABEL),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup <= 0:
        parser.error("--samples and --warmup must be positive")
    worker_values = (args.worker_backend, args.worker_shape, args.worker_json)
    if any(value is not None for value in worker_values):
        if not all(value is not None for value in worker_values):
            parser.error("all internal worker options must be supplied together")
        if args.json is not None:
            parser.error("--json is not valid in worker mode")
    elif args.json is None:
        parser.error("--json is required")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker_backend is not None:
        _run_worker(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
