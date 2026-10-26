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
both public APIs on the same tensor objects. Correctness and performance use a
stable serving portfolio below. Complete source-dispatch coverage is
tracked separately by ``csrc/blackwell_msa/route_manifest.json``.

The pinned MiniMax public sparse-forward API supports BF16 and FP8 E4M3
storage, but not FP16 input, so FP16 rows are reported explicitly as
candidate-only and are never cast to another dtype.

Example
-------
Clone the baseline at the pinned revision, then run from a clean FlashInfer
checkout::

    python benchmarks/bench_blackwell_msa_sm100.py \
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
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable


BASELINE_REPOSITORY = "https://github.com/MiniMax-AI/MSA.git"
BASELINE_SHA = "80434d7f67877c6570ca19cac444b84bc9855dac"
SOURCE_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer.git"
BLOCK_SIZE = 128
HEAD_DIM = 128
SUPPORTED_ARCHITECTURES = {(10, 0): "sm100a", (10, 3): "sm103a"}
ACTIVITY_SCOPE = "first_to_last_correlated_gpu_activity_for_one_public_api_call"
SEMANTIC_ENTRYPOINTS = (
    "flashinfer.msa_ops.msa_sparse_attention",
    "flashinfer.msa_ops.msa_sparse_decode_attention",
)
CORRECTNESS_TOLERANCES = {
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "float8_e4m3fn": {"atol": 0.1, "rtol": 0.1},
}

MANIFEST_VERSION = "msa-sm100-sm103-exact-routes-v4"
MINIMAX_BENCHMARK_PROVENANCE = (
    "MiniMax-AI/MSA@80434d7f benchmarks/bench_sparse_attention_ops.py:421-528"
)
MINIMAX_CORRECTNESS_PROVENANCE = (
    "MiniMax-AI/MSA@80434d7f "
    "python/fmha_sm100/cute/test_sparse_atten.py:1474-1711,1914-2035"
)


@dataclass(frozen=True, slots=True)
class MSAShape:
    """One immutable correctness/performance row in the public-API harness."""

    stable_id: str
    tier: str
    source: str
    provenance: str
    selection_rationale: str
    operation: str
    batch_size: int
    seqlen_q: int
    seqlen_kv: int
    q_dtype: str
    kv_dtype: str
    kv_layout: str
    num_q_heads: int
    num_kv_heads: int
    topk: int
    causal: bool
    force_fused: bool | None
    seed: int
    baseline_mode: str
    head_dim: int = HEAD_DIM
    block_size: int = BLOCK_SIZE
    selection_mode: str = "random_valid_bottom_right_causal"

    @property
    def baseline_comparable(self) -> bool:
        return self.baseline_mode == "minimax_public"

    def as_public_dict(self) -> dict[str, Any]:
        # Keep ``label`` for schema-v1 readers while making the stable key
        # explicit in schema v2.
        return {"label": self.stable_id, **asdict(self)}


FROZEN_SHAPE_IDS = (
    "prefill_bf16_b1_q4096_kv4096_h64",
    "decode_bf16_b128_q1_kv4096_h64",
    "speculative_bf16_b128_q4_kv4096_h64",
    "mtp_bf16_b128_q16_kv4096_h64",
    "decode_fp16_b128_q1_kv4096_h64",
    "decode_fp8_b128_q1_kv4096_h64",
    "official_decode_bf16_b32_q8_kv8192_h64_hkv4_k16_paged",
    "official_decode_bf16_b64_q8_kv65536_h64_hkv4_k32_paged",
    "official_prefill_mixed_fp8_b3_q1024_kv8192_h32_hkv2_k8_flat",
    "official_prefill_bf16_b3_q4096_kv8192_h8_hkv2_k4_paged",
    "coverage_decode_fp16_b32_q4_kv8192_h64_hkv4_k16_paged",
    "coverage_decode_mixed_fp8_b32_q1_kv8192_h64_hkv4_k16_flat",
    "boundary_decode_bf16_b2_q1_kv257_h8_hkv1_k4_paged",
)

_PRODUCTION = {
    "tier": "production",
    "source": "flashinfer_pr_4355_original_matrix",
    "provenance": "https://github.com/flashinfer-ai/flashinfer/pull/4355",
    "selection_rationale": (
        "Preserved verbatim from the original six-row production matrix; "
        "random-valid sparse blocks use a recorded seed."
    ),
    "num_q_heads": 64,
    "num_kv_heads": 4,
    "topk": 16,
    "causal": True,
}

SHAPE_MANIFEST: tuple[MSAShape, ...] = (
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[0],
        operation="sparse_prefill",
        batch_size=1,
        seqlen_q=4096,
        seqlen_kv=4096,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        force_fused=None,
        seed=43,
        baseline_mode="minimax_public",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[1],
        operation="sparse_decode",
        batch_size=128,
        seqlen_q=1,
        seqlen_kv=4096,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        force_fused=True,
        seed=47,
        baseline_mode="minimax_public",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[2],
        operation="sparse_decode",
        batch_size=128,
        seqlen_q=4,
        seqlen_kv=4096,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        force_fused=True,
        seed=48,
        baseline_mode="minimax_public",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[3],
        operation="sparse_decode",
        batch_size=128,
        seqlen_q=16,
        seqlen_kv=4096,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="flat_varlen",
        force_fused=True,
        seed=50,
        baseline_mode="minimax_public",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[4],
        operation="sparse_decode",
        batch_size=128,
        seqlen_q=1,
        seqlen_kv=4096,
        q_dtype="float16",
        kv_dtype="float16",
        kv_layout="flat_varlen",
        force_fused=True,
        seed=49,
        baseline_mode="candidate_only_fp16",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[5],
        operation="sparse_decode",
        batch_size=128,
        seqlen_q=1,
        seqlen_kv=4096,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        kv_layout="paged",
        force_fused=True,
        seed=53,
        baseline_mode="minimax_public",
        **_PRODUCTION,
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[6],
        tier="official_coverage",
        source="minimax_official_sparse_decode_benchmark",
        provenance=MINIMAX_BENCHMARK_PROVENANCE,
        selection_rationale=(
            "Smallest official sparse-decode benchmark coordinate; adds B32, "
            "Q8, KV8192, and the BF16 paged decode route."
        ),
        operation="sparse_decode",
        batch_size=32,
        seqlen_q=8,
        seqlen_kv=8192,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="paged",
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
        causal=True,
        force_fused=True,
        seed=61,
        baseline_mode="minimax_public",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[7],
        tier="official_coverage",
        source="minimax_official_sparse_decode_benchmark",
        provenance=MINIMAX_BENCHMARK_PROVENANCE,
        selection_rationale=(
            "Long-KV TopK32 decode coordinate covering B64, Q8, KV65536, "
            "and the exact runtime-TopK direct-M16 route."
        ),
        operation="sparse_decode",
        batch_size=64,
        seqlen_q=8,
        seqlen_kv=65536,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="paged",
        num_q_heads=64,
        num_kv_heads=4,
        topk=32,
        causal=True,
        force_fused=True,
        seed=67,
        baseline_mode="minimax_public",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[8],
        tier="official_coverage",
        source="minimax_official_mixed_fp8_correctness",
        provenance=MINIMAX_CORRECTNESS_PROVENANCE,
        selection_rationale=(
            "BF16-query/FP8-KV asymmetric prefill coordinate covering TP2 "
            "heads, TopK8, and flat FP8 storage."
        ),
        operation="sparse_prefill",
        batch_size=3,
        seqlen_q=1024,
        seqlen_kv=8192,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        kv_layout="flat_varlen",
        num_q_heads=32,
        num_kv_heads=2,
        topk=8,
        causal=True,
        force_fused=None,
        seed=71,
        baseline_mode="minimax_public",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[9],
        tier="official_coverage",
        source="minimax_official_paged_correctness",
        provenance=MINIMAX_CORRECTNESS_PROVENANCE,
        selection_rationale=(
            "Paged correctness coordinate covering GQA4, TopK4, batched "
            "asymmetric prefill, and the exact reverse-prefill route."
        ),
        operation="sparse_prefill",
        batch_size=3,
        seqlen_q=4096,
        seqlen_kv=8192,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="paged",
        num_q_heads=8,
        num_kv_heads=2,
        topk=4,
        causal=True,
        force_fused=None,
        seed=73,
        baseline_mode="minimax_public",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[10],
        tier="flashinfer_coverage",
        source="flashinfer_paged_fp16_correctness_route",
        provenance=("tests/msa_ops/test_blackwell_msa_sm100.py::decode-paged-fp16-q2"),
        selection_rationale=(
            "Completes the production FP16 row's missing paged-layout axis at "
            "an official decode batch/KV coordinate; MiniMax remains unsupported."
        ),
        operation="sparse_decode",
        batch_size=32,
        seqlen_q=4,
        seqlen_kv=8192,
        q_dtype="float16",
        kv_dtype="float16",
        kv_layout="paged",
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
        causal=True,
        force_fused=True,
        seed=79,
        baseline_mode="candidate_only_fp16",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[11],
        tier="flashinfer_coverage",
        source="production_fp8_layout_complement",
        provenance=(
            "csrc/blackwell_msa/route_manifest.json::"
            "direct_m16_decode.bf16_query_fp8_kv_flat"
        ),
        selection_rationale=(
            "Complements the production paged FP8 decode row so the distinct "
            "flat FP8 decode kernel is correctness-checked and timed."
        ),
        operation="sparse_decode",
        batch_size=32,
        seqlen_q=1,
        seqlen_kv=8192,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        kv_layout="flat_varlen",
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
        causal=True,
        force_fused=True,
        seed=83,
        baseline_mode="minimax_public",
    ),
    MSAShape(
        stable_id=FROZEN_SHAPE_IDS[12],
        tier="boundary",
        source="flashinfer_partial_page_regression",
        provenance=(
            "tests/msa_ops/test_blackwell_msa_sm100.py::decode-paged-bf16-m16-ragged"
        ),
        selection_rationale=(
            "Freezes a partial-final-page boundary with four selected blocks; "
            "exercises the exact 512-thread direct-M16 tail path."
        ),
        operation="sparse_decode",
        batch_size=2,
        seqlen_q=1,
        seqlen_kv=257,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        kv_layout="paged",
        num_q_heads=8,
        num_kv_heads=1,
        topk=4,
        causal=True,
        force_fused=True,
        seed=89,
        baseline_mode="minimax_public",
    ),
)


def _validate_shape_manifest(shapes: tuple[MSAShape, ...]) -> None:
    stable_ids = tuple(shape.stable_id for shape in shapes)
    if stable_ids != FROZEN_SHAPE_IDS:
        raise ValueError("the stable serving portfolio must not change")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("MSA shape stable IDs must be unique")

    for shape in shapes:
        if not all(
            value.strip()
            for value in (
                shape.stable_id,
                shape.tier,
                shape.source,
                shape.provenance,
                shape.selection_rationale,
            )
        ):
            raise ValueError(f"shape metadata must be non-empty: {shape.stable_id!r}")
        if shape.operation not in {"sparse_prefill", "sparse_decode"}:
            raise ValueError(f"unsupported operation in {shape.stable_id}")
        if shape.kv_layout not in {"flat_varlen", "paged"}:
            raise ValueError(f"unsupported KV layout in {shape.stable_id}")
        if shape.q_dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"unsupported Q dtype in {shape.stable_id}")
        if shape.kv_dtype not in {"bfloat16", "float16", "float8_e4m3fn"}:
            raise ValueError(f"unsupported KV dtype in {shape.stable_id}")
        if shape.kv_dtype == "float8_e4m3fn" and shape.q_dtype != "bfloat16":
            raise ValueError(f"FP8 KV requires BF16 Q in {shape.stable_id}")
        if min(shape.batch_size, shape.seqlen_q, shape.seqlen_kv) <= 0:
            raise ValueError(f"non-positive extent in {shape.stable_id}")
        if shape.seqlen_q > shape.seqlen_kv:
            raise ValueError(f"Q cannot exceed KV in {shape.stable_id}")
        if shape.num_q_heads % shape.num_kv_heads:
            raise ValueError(f"Q heads must divide by KV heads in {shape.stable_id}")
        group_size = shape.num_q_heads // shape.num_kv_heads
        if not 1 <= group_size <= 16:
            raise ValueError(f"GQA group size out of range in {shape.stable_id}")
        if shape.topk not in {4, 8, 16, 32}:
            raise ValueError(f"unsupported TopK in {shape.stable_id}")
        if shape.head_dim != HEAD_DIM or shape.block_size != BLOCK_SIZE:
            raise ValueError(f"MSA requires D128/block128 in {shape.stable_id}")
        if (
            not shape.causal
            or shape.selection_mode != "random_valid_bottom_right_causal"
        ):
            raise ValueError(f"manifest protocol changed in {shape.stable_id}")
        if shape.operation == "sparse_prefill" and shape.force_fused is not None:
            raise ValueError(f"prefill cannot set force_fused in {shape.stable_id}")
        if shape.baseline_mode not in {"minimax_public", "candidate_only_fp16"}:
            raise ValueError(f"unsupported baseline mode in {shape.stable_id}")
        if (shape.q_dtype == "float16") != (
            shape.baseline_mode == "candidate_only_fp16"
        ):
            raise ValueError(f"baseline support metadata is stale in {shape.stable_id}")


_validate_shape_manifest(SHAPE_MANIFEST)
SHAPES_BY_ID = {shape.stable_id: shape for shape in SHAPE_MANIFEST}


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


def _make_q2k(torch, shape: MSAShape, device) -> Any:
    """Reproduce the canonical random-valid, bottom-right-causal selection."""

    batch_size = shape.batch_size
    seqlen_q = shape.seqlen_q
    seqlen_kv = shape.seqlen_kv
    total_q = batch_size * seqlen_q
    output = torch.full(
        (shape.num_kv_heads, total_q, shape.topk),
        -1,
        dtype=torch.int32,
    )
    generator = torch.Generator(device="cpu").manual_seed(shape.seed + 101)
    all_blocks = (seqlen_kv + shape.block_size - 1) // shape.block_size
    q_start = 0
    for _ in range(batch_size):
        offset = seqlen_kv - seqlen_q
        for local_q in range(seqlen_q):
            visible_tokens = offset + local_q + 1
            visible_blocks = max(
                0,
                min(
                    all_blocks,
                    (visible_tokens + shape.block_size - 1) // shape.block_size,
                ),
            )
            for kv_head in range(shape.num_kv_heads):
                candidates = torch.randperm(visible_blocks, generator=generator)
                selected = candidates[: min(shape.topk, visible_blocks)].sort().values
                output[kv_head, q_start + local_q, : selected.numel()] = selected.to(
                    torch.int32
                )
        q_start += seqlen_q
    return output.to(device=device).contiguous()


def _make_paged_cache(
    torch,
    logical,
    *,
    batch_size: int,
    seqlen_kv: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
):
    pages_per_sequence = (seqlen_kv + block_size - 1) // block_size
    total_pages = batch_size * pages_per_sequence
    padded_tokens_per_sequence = pages_per_sequence * block_size
    if padded_tokens_per_sequence == seqlen_kv:
        padded = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    else:
        padded = torch.zeros(
            (batch_size, padded_tokens_per_sequence, num_kv_heads, head_dim),
            dtype=logical.dtype,
            device=logical.device,
        )
        padded[:, :seqlen_kv] = logical.view(
            batch_size, seqlen_kv, num_kv_heads, head_dim
        )
    logical_pages = (
        padded.view(
            batch_size,
            pages_per_sequence,
            block_size,
            num_kv_heads,
            head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, num_kv_heads, block_size, head_dim)
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


def _make_inputs(torch, shape: MSAShape, device) -> dict[str, Any]:
    batch_size = shape.batch_size
    seqlen_q = shape.seqlen_q
    seqlen_kv = shape.seqlen_kv
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q_dtype = _torch_dtype(torch, shape.q_dtype)
    kv_dtype = _torch_dtype(torch, shape.kv_dtype)
    generator = torch.Generator(device=device).manual_seed(shape.seed)

    q = (
        torch.randn(
            (total_q, shape.num_q_heads, shape.head_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 3.0
    ).to(q_dtype)
    logical_k = (
        torch.randn(
            (total_k, shape.num_kv_heads, shape.head_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 3.0
    ).to(kv_dtype)
    logical_v = (
        torch.randn(
            (total_k, shape.num_kv_heads, shape.head_dim),
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
    if shape.kv_layout == "flat_varlen":
        k = logical_k.contiguous()
        v = logical_v.contiguous()
    elif shape.kv_layout == "paged":
        k, page_table = _make_paged_cache(
            torch,
            logical_k,
            batch_size=batch_size,
            seqlen_kv=seqlen_kv,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            block_size=shape.block_size,
        )
        v, v_page_table = _make_paged_cache(
            torch,
            logical_v,
            batch_size=batch_size,
            seqlen_kv=seqlen_kv,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            block_size=shape.block_size,
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
        raise ValueError(f"unsupported KV layout {shape.kv_layout!r}")

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
    shape: MSAShape, inputs: dict[str, Any]
) -> tuple[Callable[[], Any], str, dict[str, Any]]:
    msa_ops = importlib.import_module("flashinfer.msa_ops")
    if shape.operation == "sparse_prefill":
        public_api = "flashinfer.msa_ops.msa_sparse_attention"

        def call():
            return msa_ops.msa_sparse_attention(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["q2k"],
                inputs["cu_q"],
                inputs["cu_k"],
                causal=shape.causal,
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                return_softmax_lse=True,
                return_temperature_lse=True,
                lse_temperature_scale=1.0,
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
                seqlen_q=shape.seqlen_q,
                causal=shape.causal,
                return_softmax_lse=False,
                force_fused=shape.force_fused,
            )

    return call, public_api, {"excluded_setup": ["deterministic_input_construction"]}


def _baseline_call(
    torch, shape: MSAShape, inputs: dict[str, Any]
) -> tuple[Callable[[], Any], str, dict[str, Any]]:
    if not shape.baseline_comparable:
        raise RuntimeError("the pinned public baseline does not accept FP16 input")
    baseline = importlib.import_module("fmha_sm100")
    k2q_row_ptr, k2q_q_indices, schedule = baseline.build_k2q_csr(
        inputs["q2k"],
        inputs["cu_q"],
        inputs["cu_k"],
        shape.block_size,
        total_k=shape.batch_size * shape.seqlen_kv,
        max_seqlen_k=shape.seqlen_kv,
        max_seqlen_q=shape.seqlen_q,
        total_rows=(
            shape.batch_size
            * ((shape.seqlen_kv + shape.block_size - 1) // shape.block_size)
        ),
        qhead_per_kv=shape.num_q_heads // shape.num_kv_heads,
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
            shape.topk,
            cu_seqlens_q=inputs["cu_q"],
            cu_seqlens_k=inputs["cu_k"],
            max_seqlen_q=shape.seqlen_q,
            max_seqlen_k=shape.seqlen_kv,
            blk_kv=shape.block_size,
            causal=shape.causal,
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
    shape: MSAShape,
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
        shape.batch_size * shape.seqlen_q,
        shape.num_q_heads,
        shape.head_dim,
    )
    shape_matches = (
        tuple(candidate_output.shape) == expected_shape
        and tuple(baseline_output.shape) == expected_shape
    )
    dtype_matches = candidate_output.dtype == baseline_output.dtype
    tolerance = CORRECTNESS_TOLERANCES[shape.kv_dtype]
    if not shape_matches or not dtype_matches:
        return {
            "status": "failed",
            "passed": False,
            "reference": "pinned_public_fmha_sm100_sparse_atten_func",
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
            "candidate_nonfinite_count": None,
            "baseline_nonfinite_count": None,
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


def _logical_dense_kv(torch, shape: MSAShape, inputs: dict[str, Any]):
    """Recover logical per-sequence K/V without relying on the candidate."""

    if shape.kv_layout == "flat_varlen":
        logical_shape = (
            shape.batch_size,
            shape.seqlen_kv,
            shape.num_kv_heads,
            shape.head_dim,
        )
        return inputs["k"].view(logical_shape), inputs["v"].view(logical_shape)

    page_ids = inputs["page_table"].long()

    def unpack(cache):
        # Physical pages are deliberately shuffled by _make_paged_cache.
        # Resolve the page table before flattening page-local token rows.
        pages = cache[page_ids]
        dense = pages.permute(0, 1, 3, 2, 4).reshape(
            shape.batch_size,
            -1,
            shape.num_kv_heads,
            shape.head_dim,
        )
        return dense[:, : shape.seqlen_kv]

    return unpack(inputs["k"]), unpack(inputs["v"])


def _candidate_reference_output(torch, shape: MSAShape, inputs: dict[str, Any]):
    """Independent FP32 sparse-attention reference for baseline-unsupported rows."""

    q = (
        inputs["q"]
        .view(
            shape.batch_size,
            shape.seqlen_q,
            shape.num_q_heads,
            shape.head_dim,
        )
        .float()
    )
    k, v = _logical_dense_kv(torch, shape, inputs)
    k = k.float()
    v = v.float()
    selections = (
        inputs["q2k"]
        .view(shape.num_kv_heads, shape.batch_size, shape.seqlen_q, shape.topk)
        .permute(1, 2, 0, 3)
    )

    token_ids = torch.arange(shape.seqlen_kv, device=q.device)
    block_ids = token_ids // shape.block_size
    allowed = (
        block_ids.view(1, 1, 1, shape.seqlen_kv, 1) == selections.unsqueeze(-2)
    ).any(-1)
    if shape.causal:
        q_positions = (
            shape.seqlen_kv
            - shape.seqlen_q
            + torch.arange(shape.seqlen_q, device=q.device)
        )
        allowed &= token_ids.view(1, 1, 1, shape.seqlen_kv) <= q_positions.view(
            1, shape.seqlen_q, 1, 1
        )

    group_size = shape.num_q_heads // shape.num_kv_heads
    output = torch.zeros_like(q)
    scale = shape.head_dim**-0.5
    for kv_head in range(shape.num_kv_heads):
        head_start = kv_head * group_size
        head_end = head_start + group_size
        logits = (
            torch.einsum(
                "bqgd,bkd->bqgk",
                q[:, :, head_start:head_end],
                k[:, :, kv_head],
            )
            * scale
        )
        mask = allowed[:, :, kv_head].unsqueeze(2)
        probabilities = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        probabilities = torch.where(
            mask.any(-1).unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
        )
        output[:, :, head_start:head_end] = torch.einsum(
            "bqgk,bkd->bqgd", probabilities, v[:, :, kv_head]
        )
    return output.reshape(-1, shape.num_q_heads, shape.head_dim).to(inputs["q"].dtype)


def _verify_candidate_reference(
    torch,
    shape: MSAShape,
    inputs: dict[str, Any],
    candidate_call: Callable[[], Any],
    *,
    candidate_api: str,
) -> dict[str, Any]:
    candidate_output = _primary_output(candidate_call())
    reference_output = _candidate_reference_output(torch, shape, inputs)
    torch.cuda.synchronize()
    expected_shape = (
        shape.batch_size * shape.seqlen_q,
        shape.num_q_heads,
        shape.head_dim,
    )
    tolerance = CORRECTNESS_TOLERANCES[shape.q_dtype]
    shape_matches = (
        tuple(candidate_output.shape) == expected_shape
        and tuple(reference_output.shape) == expected_shape
    )
    dtype_matches = candidate_output.dtype == reference_output.dtype
    if not shape_matches or not dtype_matches:
        return {
            "status": "failed",
            "passed": False,
            "reference": "independent_torch_fp32_masked_attention",
            "candidate_public_api": candidate_api,
            "same_q_k_v_tensor_objects": True,
            "same_sequence_metadata_tensor_objects": True,
            "same_page_table_argument": True,
            "expected_shape": list(expected_shape),
            "candidate_shape": list(candidate_output.shape),
            "reference_shape": list(reference_output.shape),
            "candidate_dtype": str(candidate_output.dtype),
            "reference_dtype": str(reference_output.dtype),
            **tolerance,
            "max_abs_error": None,
            "mismatch_count": None,
            "candidate_nonfinite_count": None,
            "reference_nonfinite_count": None,
        }

    candidate_float = candidate_output.float()
    reference_float = reference_output.float()
    close = torch.isclose(
        candidate_float,
        reference_float,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
        equal_nan=False,
    )
    candidate_nonfinite_count = int((~torch.isfinite(candidate_float)).sum().item())
    reference_nonfinite_count = int((~torch.isfinite(reference_float)).sum().item())
    passed = (
        bool(close.all().item())
        and candidate_nonfinite_count == 0
        and reference_nonfinite_count == 0
    )
    finite = torch.isfinite(candidate_float) & torch.isfinite(reference_float)
    max_abs_error = None
    if bool(finite.any().item()):
        max_abs_error = float(
            (candidate_float[finite] - reference_float[finite]).abs().max().item()
        )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reference": "independent_torch_fp32_masked_attention",
        "candidate_public_api": candidate_api,
        "same_q_k_v_tensor_objects": True,
        "same_sequence_metadata_tensor_objects": True,
        "same_page_table_argument": True,
        "expected_shape": list(expected_shape),
        "candidate_dtype": str(candidate_output.dtype),
        "reference_dtype": str(reference_output.dtype),
        **tolerance,
        "max_abs_error": max_abs_error,
        "mismatch_count": int((~close).sum().item()),
        "candidate_nonfinite_count": candidate_nonfinite_count,
        "reference_nonfinite_count": reference_nonfinite_count,
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
    shape = SHAPES_BY_ID[args.worker_shape]
    inputs = _make_inputs(torch, shape, device)
    software = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "cupti_python_version": cupti_python_version,
    }
    if args.worker_backend == "verify":
        candidate_call, candidate_api, _ = _candidate_call(shape, inputs)
        if shape.baseline_comparable:
            imported_baseline = importlib.import_module("fmha_sm100")
            imported_baseline_root = (
                Path(imported_baseline.__file__).resolve().parents[2]
            )
            if imported_baseline_root != baseline_root:
                raise RuntimeError(
                    f"expected fmha_sm100 from {baseline_root}, "
                    f"imported {imported_baseline_root}"
                )
            baseline_call, baseline_api, _ = _baseline_call(torch, shape, inputs)
            correctness = _verify_public_outputs(
                torch,
                shape,
                candidate_call,
                baseline_call,
                candidate_api=candidate_api,
                baseline_api=baseline_api,
            )
        else:
            correctness = _verify_candidate_reference(
                torch,
                shape,
                inputs,
                candidate_call,
                candidate_api=candidate_api,
            )
        result = {
            "status": "verified" if correctness["passed"] else "failed",
            "backend": "verify",
            "shape": shape.stable_id,
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
        "shape": shape.stable_id,
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
    shape: MSAShape,
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
        shape.stable_id,
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
            f"isolated {backend}/{shape.stable_id} worker failed "
            f"with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output.is_file():
        raise RuntimeError(f"worker did not write {output}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("backend") != backend or result.get("shape") != shape.stable_id:
        raise RuntimeError(f"worker returned mismatched result: {result}")
    return result


def _public_shape(shape: MSAShape) -> dict[str, Any]:
    return shape.as_public_dict()


def _validate_common_metadata(
    results: list[dict[str, Any]],
    expected_measurements: list[tuple[str, str]],
) -> None:
    if not results:
        raise RuntimeError("no measurements were collected")
    actual_measurements = [(result["shape"], result["backend"]) for result in results]
    if sorted(actual_measurements) != sorted(expected_measurements):
        raise RuntimeError(
            "timing workers did not traverse the selected manifest exactly: "
            f"expected {expected_measurements}, got {actual_measurements}"
        )
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
    results: list[dict[str, Any]],
    measured_reference: dict[str, Any],
    expected_shape_ids: list[str],
) -> None:
    actual_shape_ids = [result["shape"] for result in results]
    if actual_shape_ids != expected_shape_ids:
        raise RuntimeError(
            "correctness workers did not traverse the comparable manifest rows "
            f"exactly: expected {expected_shape_ids}, got {actual_shape_ids}"
        )
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


def _unsupported_baseline(shape: MSAShape) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "baseline_mode": shape.baseline_mode,
        "public_api": "fmha_sm100.sparse_atten_func",
        "reason": (
            "The pinned public sparse-forward API accepts BF16 or FP8 E4M3 "
            "Q/K/V storage, not FP16 input. No cross-dtype timing proxy is used."
        ),
        "evidence": {
            "source_path": "python/fmha_sm100/cute/interface.py",
            "symbol": "_SUPPORTED_FWD_DTYPES",
            "baseline_sha": BASELINE_SHA,
        },
    }


def _selected_shapes(args: argparse.Namespace) -> tuple[MSAShape, ...]:
    if args.shapes is None:
        return SHAPE_MANIFEST
    if len(set(args.shapes)) != len(args.shapes):
        raise ValueError("--shapes must not contain duplicate stable IDs")
    requested = set(args.shapes)
    # Preserve manifest order even when the CLI list is reordered, keeping
    # backend alternation and output JSON deterministic.
    return tuple(shape for shape in SHAPE_MANIFEST if shape.stable_id in requested)


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

    selected_shapes = _selected_shapes(args)
    rows = []
    measured_results: list[dict[str, Any]] = []
    correctness_results: list[dict[str, Any]] = []
    expected_measurements: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="flashinfer-msa-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        comparable_index = 0
        for index, shape in enumerate(selected_shapes):
            comparable = shape.baseline_comparable
            reference_kind = (
                "pinned MiniMax public output"
                if comparable
                else "independent FP32 masked-attention reference"
            )
            print(f"Verifying {shape.stable_id} against {reference_kind}", flush=True)
            correctness_worker = _run_isolated(
                args,
                backend="verify",
                shape=shape,
                output=temp_root / f"{index}-verify.json",
            )
            correctness_results.append(correctness_worker)
            if not correctness_worker["correctness"]["passed"]:
                raise RuntimeError(
                    f"correctness failed for {shape.stable_id}: "
                    f"{correctness_worker['correctness']}"
                )
            correctness = correctness_worker["correctness"]
            if comparable and comparable_index % 2 == 0:
                process_order = ("minimax", "flashinfer")
            elif comparable:
                process_order = ("flashinfer", "minimax")
            else:
                process_order = ("flashinfer",)
            if comparable:
                comparable_index += 1
            print(
                f"Measuring {shape.stable_id} ({', '.join(process_order)})",
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
                expected_measurements.append((shape.stable_id, backend))

            candidate = by_backend["flashinfer"]
            if comparable:
                baseline = by_backend["minimax"]
                speedup = baseline["median_ms"] / candidate["median_ms"]
                if not math.isfinite(speedup) or speedup <= 0.0:
                    raise RuntimeError(
                        f"invalid speedup for {shape.stable_id}: {speedup}"
                    )
                comparison_status = "measured"
            else:
                baseline = _unsupported_baseline(shape)
                speedup = None
                comparison_status = "official_baseline_unsupported"
            row = {
                "shape": _public_shape(shape),
                "comparison_status": comparison_status,
                "correctness": correctness,
                "correctness_process": "separate_untimed_correctness_worker",
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
    _validate_common_metadata(measured_results, expected_measurements)
    expected_comparison_ids = [
        shape.stable_id for shape in selected_shapes if shape.baseline_comparable
    ]
    expected_correctness_ids = [shape.stable_id for shape in selected_shapes]
    _validate_correctness_metadata(
        correctness_results,
        measured_results[0],
        expected_correctness_ids,
    )
    comparable_speedups = [
        row["speedup_baseline_over_candidate"]
        for row in rows
        if row["speedup_baseline_over_candidate"] is not None
    ]
    if len(comparable_speedups) != len(expected_comparison_ids):
        raise RuntimeError(
            f"expected {len(expected_comparison_ids)} comparable rows, "
            f"got {len(comparable_speedups)}"
        )
    geometric_mean = (
        math.exp(
            sum(math.log(value) for value in comparable_speedups)
            / len(comparable_speedups)
        )
        if comparable_speedups
        else None
    )
    first = measured_results[0]
    result = {
        "schema_version": 2,
        "manifest_version": MANIFEST_VERSION,
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
                "one_separate_untimed_process_per_selected_shape"
            ),
            "correctness_reference": (
                "pinned_public_fmha_sm100_sparse_atten_func for MiniMax-supported "
                "rows; independent torch FP32 masked attention for FP16 rows"
            ),
            "baseline_api_selection": (
                "sparse_atten_func supports the required flat/paged BF16 and "
                "mixed BF16-query/FP8-KV inputs. FP16 rows are candidate-only; "
                "no cross-dtype proxy is reported."
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
            "manifest_shape_count": len(SHAPE_MANIFEST),
            "selected_shape_count": len(selected_shapes),
            "selected_shape_ids": [shape.stable_id for shape in selected_shapes],
            "comparable_shape_count": len(expected_comparison_ids),
            "official_baseline_unsupported_shape_count": sum(
                not shape.baseline_comparable for shape in selected_shapes
            ),
            "correctness_checked_shape_count": len(correctness_results),
            "minimax_output_parity_checked_shape_count": len(expected_comparison_ids),
            "independent_reference_checked_shape_count": len(expected_correctness_ids)
            - len(expected_comparison_ids),
            "num_q_heads_values": sorted(
                {shape.num_q_heads for shape in selected_shapes}
            ),
            "num_kv_heads_values": sorted(
                {shape.num_kv_heads for shape in selected_shapes}
            ),
            "head_dim_values": sorted({shape.head_dim for shape in selected_shapes}),
            "topk_values": sorted({shape.topk for shape in selected_shapes}),
            "block_size_values": sorted(
                {shape.block_size for shape in selected_shapes}
            ),
        },
        "validation": {
            "expected_shape_count": len(FROZEN_SHAPE_IDS),
            "semantic_entrypoints": list(SEMANTIC_ENTRYPOINTS),
            "route_manifest": [shape.as_public_dict() for shape in SHAPE_MANIFEST],
        },
        "rows": rows,
        "summary": {
            "all_required_measurements_valid": True,
            "all_comparable_outputs_match": True,
            "all_frozen_shape_correctness_passed": True,
            "measured_comparisons": len(comparable_speedups),
            "geometric_mean_speedup": geometric_mean,
            "minimum_speedup": min(comparable_speedups)
            if comparable_speedups
            else None,
            "maximum_speedup": max(comparable_speedups)
            if comparable_speedups
            else None,
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
        "--shapes",
        nargs="+",
        choices=tuple(SHAPES_BY_ID),
        metavar="STABLE_ID",
        help=(
            "run only these frozen manifest rows (default: all rows); output "
            "records both the full manifest count and the selected stable IDs"
        ),
    )
    parser.add_argument(
        "--worker-backend",
        choices=("flashinfer", "minimax", "verify"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-shape",
        choices=tuple(SHAPES_BY_ID),
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
        if args.shapes is not None:
            parser.error("--shapes is not valid in worker mode")
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
