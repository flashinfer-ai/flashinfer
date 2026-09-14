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

"""Benchmark contract-F DSV4 CSA/HCA FMHA on TRT-LLM-gen geometry.

The default is the geometry from ``smoke_dsv4/csa_ctx_1_8k.sh``: B=1,
Q=KV=8192, Hq=128, Dqk=Dv=512, E4M3 input, BF16 output, R=4, and Kmax=1152.
Unlike that historical smoke command, the production benchmark explicitly
defaults to TRTLLM-gen's recommended E4M3 skip-correction threshold of 8.
It constructs the same dynamic sparse length rule as
``fillDynamicSparseMlaIndices`` before timing: tile 0 has a causal SWA prefix
of at most 128 tokens and the remaining slots have at most 1024 compressed
tokens. The compressed rows use a deterministic permutation rather than the
runner's ``std::mt19937`` shuffle; selection order does not change FMHA work.

Cache append/compression, HCA/CSA score selection, and page-table lowering are
deliberately outside the timed region. This makes the timing comparable to the
generated FMHA kernel itself, not to an end-to-end cache manager.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import statistics
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashinfer.attention.prims_ts import (
    prims_ts_dsv4_csa,
    prims_ts_dsv4_sparse_mla_rope_quant,
)
from flashinfer.testing import bench_gpu_time


HEADS = 128
HEAD_DIM = 512
SWA_TOPK = 128
TILE_K = 128
SELECTOR_VECTOR_WIDTH = 4


def _l2_cache_size_bytes(device: torch.device) -> int:
    """Read ``CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`` without a CUDA struct ABI.

    Some PyTorch builds omit ``l2_cache_size`` from ``DeviceProperties``.  The
    driver attribute is stable, avoids re-declaring ``cudaDeviceProp``, and is
    the same quantity Fmha.cpp uses to size its one-time eviction buffer.
    """

    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    if driver.cuInit(0) != 0:
        raise RuntimeError("cuInit failed while reading L2 cache size")
    cuda_device = ctypes.c_int()
    ordinal = torch.cuda.current_device() if device.index is None else device.index
    if driver.cuDeviceGet(ctypes.byref(cuda_device), ordinal) != 0:
        raise RuntimeError(f"cuDeviceGet failed for CUDA device {ordinal}")
    l2_cache_size = ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE / cudaDevAttrL2CacheSize.
    if driver.cuDeviceGetAttribute(ctypes.byref(l2_cache_size), 38, cuda_device) != 0:
        raise RuntimeError("cuDeviceGetAttribute(L2_CACHE_SIZE) failed")
    return l2_cache_size.value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=8192)
    parser.add_argument(
        "--kv-len",
        type=int,
        default=None,
        help="raw post-append KV length per request (default: --q-len)",
    )
    parser.add_argument("--topk", type=int, default=1152)
    parser.add_argument(
        "--compress-ratio",
        type=int,
        default=4,
        help="DSV4 sparseMlaCompressRatio (CSA=4, HCA=128)",
    )
    parser.add_argument(
        "--pool-rows",
        type=int,
        default=8192,
        help="physical rows in each pool; must cover B * --kv-len",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--skip-corr-threshold",
        type=float,
        default=8.0,
        help=(
            "TRTLLM-gen-compatible E4M3 skip-correction threshold; positive "
            "values enable the specialization, 0 disables it (default: 8)"
        ),
    )
    parser.add_argument(
        "--paired-env",
        type=str,
        default=None,
        help=(
            "compile two variants selected by one FLASHINFER_DSV4_CSA_* environment "
            "variable and time them in same-process ABBA blocks"
        ),
    )
    parser.add_argument("--paired-baseline", type=str, default="0")
    parser.add_argument("--paired-candidate", type=str, default="1")
    parser.add_argument(
        "--paired-rounds",
        type=int,
        default=5,
        help="number of ABBA rounds for --paired-env",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--workload-dir",
        type=Path,
        default=None,
        help=(
            "load q.e4m3.bin, kv_pool_p1.e4m3.bin, swa_pool_p1.e4m3.bin, "
            "sparse_indices.i32.bin, and topk_lens.i32.bin from a shared "
            "source/TS workload directory instead of constructing tensors"
        ),
    )
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="flush L2 before each measured iteration (default: warm L2)",
    )
    parser.add_argument(
        "--source-timing",
        action="store_true",
        help=(
            "match the source runner's timing shape: clear 2x L2 once, run the requested "
            "warmups, then report one CUDA-event mean over --iters launches; this cannot "
            "emulate the source runner's distinct generated code instances"
        ),
    )
    parser.add_argument(
        "--profile-launches",
        type=int,
        default=0,
        help=(
            "after JIT warmup, bracket this many kernel launches with CUDA profiler "
            "start/stop; intended for ncu --profile-from-start off"
        ),
    )
    parser.add_argument(
        "--check-samples",
        type=int,
        default=0,
        help=(
            "after all timing, validate this many B=1 --workload-dir query rows against "
            "an eager FP32 sparse reference; zero disables the non-timed correctness gate"
        ),
    )
    parser.add_argument(
        "--output-o",
        type=Path,
        default=None,
        help="write the final output tensor as raw contiguous bytes after timing",
    )
    parser.add_argument(
        "--rope-quant",
        action="store_true",
        help=(
            "select TRTLLM-gen's fused inverse-RoPE + grouped E4M3 output "
            "epilogue instead of the BF16 output kernel"
        ),
    )
    parser.add_argument(
        "--output-scale",
        type=Path,
        default=None,
        help="write the fused FP32 D128 scale buffer as raw contiguous bytes",
    )
    return parser.parse_args()


def _runner_range_fp8(
    shape: tuple[int, ...],
    *,
    value_range: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Match the smoke runner's non-degenerate E4M3 value ranges."""

    return (
        torch.empty(shape, device=device, dtype=torch.bfloat16)
        .uniform_(-value_range, value_range, generator=generator)
        .to(torch.float8_e4m3fn)
    )


def _build_runner_geometry_metadata(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    topk: int,
    compress_ratio: int,
    pool_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lower the smoke runner's dynamic sparse fixture to contract-F tensors."""

    del pool_rows  # Validation in main establishes every generated route is valid.
    total_q = batch_size * q_len
    q_pos = torch.arange(q_len, dtype=torch.int32, device=device).repeat(batch_size)
    batch_ids = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(q_len)
    # The Fmha runner uses: kvPosition = seqLenKv - seqLenQ + tokenIdxQ.
    raw_visible = kv_len - q_len + q_pos + 1
    swa_valid = raw_visible.clamp(min=0, max=SWA_TOPK)
    primary_capacity = topk - SWA_TOPK
    compressed_visible = torch.div(raw_visible, compress_ratio, rounding_mode="floor")
    compressed_count = compressed_visible.clamp(min=0, max=primary_capacity)
    sparse_lens = swa_valid.new_full((total_q,), SWA_TOPK) + compressed_count

    # Match `fillDynamicSparseMlaIndices`: inactive slots in the fixed SWA
    # tile are -1.  The generated Gather4 path reaches those coordinates
    # before QK masking; replacing them with row 0 changes the address stream
    # and is not a source-parity benchmark.  The unselected compressed tail
    # stays zero because the runner zero-initializes the selector allocation.
    routes = torch.zeros((total_q, topk), dtype=torch.int32, device=device)
    swa_slot = torch.arange(SWA_TOPK, dtype=torch.int32, device=device)
    swa_rows = batch_ids[:, None] * kv_len + raw_visible[:, None] - swa_valid[:, None]
    swa_rows = swa_rows + swa_slot
    routes[:, :SWA_TOPK] = torch.where(
        swa_slot[None, :] < swa_valid[:, None], swa_rows, torch.full_like(swa_rows, -1)
    )

    # ``std::shuffle`` in the runner creates a unique selection from all visible
    # compressed tokens. This stride is coprime to the 2K smoke capacity, so it
    # is likewise a per-query permutation while avoiding router work in timing.
    # Keep arithmetic defined for short diagnostic workloads; then all primary
    # slots remain inactive because ``compressed_count`` is zero.
    compressed_capacity = max(kv_len // compress_ratio, 1)
    compressed_slot = torch.arange(primary_capacity, dtype=torch.int32, device=device)
    compressed_base = torch.remainder(q_pos * 313, compressed_capacity)
    compressed_rows = torch.remainder(
        compressed_base[:, None] + compressed_slot[None, :] * 37,
        compressed_capacity,
    )
    compressed_rows = compressed_rows + batch_ids[:, None] * compressed_capacity
    routes[:, SWA_TOPK:] = torch.where(
        compressed_slot[None, :] < compressed_count[:, None],
        compressed_rows,
        torch.zeros_like(compressed_rows),
    )
    raw_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device).mul_(
        q_len
    )
    return routes, sparse_lens, raw_seq_lens, cu_seqlens_q


def _from_file_exact(
    path: Path, *, dtype: torch.dtype, numel: int, name: str
) -> torch.Tensor:
    """Map one exact-size raw tensor, refusing implicit truncation/repetition."""

    if not path.is_file():
        raise FileNotFoundError(f"shared workload is missing {name}: {path}")
    expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{name} has {actual_bytes} bytes; expected {expected_bytes} bytes"
        )
    return torch.from_file(str(path), shared=False, size=numel, dtype=dtype)


def _load_shared_workload(
    workload_dir: Path,
    *,
    q_len: int,
    kv_len: int,
    topk: int,
    pool_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the B=1 page-1 source runner ABI without changing byte layout.

    ``kv_pool_p1`` and ``swa_pool_p1`` contain K then V halves for the
    standalone source runner.  This MLA-generation kernel reads the first
    (K/latent) half for both BMMs, so TS intentionally views exactly that half
    as its flat physical pool.
    """

    q_cpu = (
        _from_file_exact(
            workload_dir / "q.e4m3.bin",
            dtype=torch.uint8,
            numel=q_len * HEADS * HEAD_DIM,
            name="q.e4m3.bin",
        )
        .view(torch.float8_e4m3fn)
        .reshape(q_len, HEADS, HEAD_DIM)
    )
    pool_numel = pool_rows * HEAD_DIM

    def load_pool(name: str) -> torch.Tensor:
        pool_cpu = _from_file_exact(
            workload_dir / name,
            dtype=torch.uint8,
            numel=2 * pool_numel,
            name=name,
        ).view(torch.float8_e4m3fn)
        return pool_cpu[:pool_numel].reshape(pool_rows, HEAD_DIM).to(device)

    routes = (
        _from_file_exact(
            workload_dir / "sparse_indices.i32.bin",
            dtype=torch.int32,
            numel=q_len * topk,
            name="sparse_indices.i32.bin",
        )
        .reshape(q_len, topk)
        .to(device)
    )
    sparse_lens = _from_file_exact(
        workload_dir / "topk_lens.i32.bin",
        dtype=torch.int32,
        numel=q_len,
        name="topk_lens.i32.bin",
    ).to(device)
    if int(sparse_lens.min().item()) < SWA_TOPK or int(sparse_lens.max().item()) > topk:
        raise ValueError(
            "shared topk_lens violates the DSV4 [128, Kmax] scan-width ABI"
        )
    if int(routes[:, :SWA_TOPK].min().item()) < -1:
        raise ValueError("shared SWA route has an index below the legal -1 sentinel")
    if int(routes[:, SWA_TOPK:].min().item()) < 0:
        raise ValueError("shared compressed route has a negative physical row")
    return (
        q_cpu.to(device),
        load_pool("kv_pool_p1.e4m3.bin"),
        load_pool("swa_pool_p1.e4m3.bin"),
        routes,
        sparse_lens,
    )


def _shared_fp32_sample_reference(
    *,
    out: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    compressed_pool: torch.Tensor,
    swa_pool: torch.Tensor,
    routes: torch.Tensor,
    sparse_lens: torch.Tensor,
    kv_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    sample_count: int,
) -> tuple[int, float, float, float, float]:
    """Check representative shared-file rows outside the timed interval.

    The selected rows include each SWA/last-K-tile transition before adding
    evenly distributed positions.  This is deliberately a sampled FP32 oracle:
    a full 8K eager attention materialization is not a benchmark harness.
    """

    total_q = query.shape[0]
    if sample_count <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    anchors = (0, 1, 2, 126, 127, 128, 129, 255, 256, 4094, 4095, total_q - 1)
    positions: list[int] = []
    for position in anchors:
        if 0 <= position < total_q and position not in positions:
            positions.append(position)
        if len(positions) >= sample_count:
            break
    if len(positions) < sample_count:
        for ordinal in range(sample_count):
            position = ordinal * (total_q - 1) // max(sample_count - 1, 1)
            if position not in positions:
                positions.append(position)
            if len(positions) >= sample_count:
                break

    max_out_abs = 0.0
    max_out_rel = 0.0
    max_lse_abs = 0.0
    max_lse_rel = 0.0
    for query_idx in positions:
        active_width = int(sparse_lens[query_idx].item())
        raw_visible = kv_len - total_q + query_idx + 1
        swa_valid = min(raw_visible, SWA_TOPK, active_width)
        kv_tiles = [swa_pool[routes[query_idx, :swa_valid].long()].float()]
        compressed_routes = routes[query_idx, SWA_TOPK:active_width]
        for tile_begin in range(0, compressed_routes.numel(), TILE_K):
            tile_routes = compressed_routes[tile_begin : tile_begin + TILE_K]
            kv_tiles.append(compressed_pool[tile_routes.long()].float())

        # Reproduce the generated online-softmax numerical protocol rather
        # than comparing threshold=8 against exact torch.softmax.  In
        # particular, compressed rows start at logical tile 1 even when the
        # causal SWA prefix has fewer than 128 valid rows.
        scale_log2 = bmm1_scale / math.log(2.0)
        adjusted_threshold = (
            skip_corr_threshold / scale_log2 if skip_corr_threshold > 0.0 else 0.0
        )
        p_scale = 1.75 if skip_corr_threshold > 0.0 else 448.0
        row_max = torch.full(
            (HEADS,), -torch.finfo(torch.float32).max, device=query.device
        )
        row_sum = torch.zeros((HEADS,), dtype=torch.float32, device=query.device)
        expected_out = torch.zeros(
            (HEADS, HEAD_DIM), dtype=torch.float32, device=query.device
        )
        q = query[query_idx].float()
        for kv_tile in kv_tiles:
            tile_scores = q @ kv_tile.T
            candidate_max = torch.maximum(row_max, tile_scores.max(dim=-1).values)
            if skip_corr_threshold > 0.0:
                candidate_max = torch.where(
                    candidate_max - row_max <= adjusted_threshold,
                    row_max,
                    candidate_max,
                )
            # ``torch.exp2`` is JIT-backed in some internal PyTorch builds and
            # can fail when their NVRTC builtins are not on LD_LIBRARY_PATH.
            # The ordinary exp/log kernels are prebuilt and are mathematically
            # identical for this out-of-band reference.
            correction = torch.exp(
                (row_max - candidate_max) * scale_log2 * math.log(2.0)
            )
            p = (
                torch.exp(
                    (tile_scores - candidate_max[:, None]) * scale_log2 * math.log(2.0)
                )
                * p_scale
            )
            p_fp8 = p.to(torch.float8_e4m3fn).float()
            expected_out = expected_out * correction[:, None] + p_fp8 @ kv_tile
            row_sum = row_sum * correction + p.sum(dim=-1)
            row_max = candidate_max
        expected_out = expected_out * (bmm2_scale / row_sum[:, None])
        expected_lse = (
            torch.log(row_sum / p_scale) / math.log(2.0) + row_max * scale_log2
        )
        actual_out = out[query_idx].float()
        actual_lse = lse[query_idx]
        out_abs = (actual_out - expected_out).abs()
        lse_abs = (actual_lse - expected_lse).abs()
        max_out_abs = max(max_out_abs, float(out_abs.max().item()))
        max_out_rel = max(
            max_out_rel,
            float((out_abs / expected_out.abs().clamp_min(1e-4)).max().item()),
        )
        max_lse_abs = max(max_lse_abs, float(lse_abs.max().item()))
        max_lse_rel = max(
            max_lse_rel,
            float((lse_abs / expected_lse.abs().clamp_min(1e-4)).max().item()),
        )
        torch.testing.assert_close(actual_out, expected_out, atol=1e-2, rtol=5e-2)
        torch.testing.assert_close(actual_lse, expected_lse, atol=1e-3, rtol=1e-3)
    return len(positions), max_out_abs, max_out_rel, max_lse_abs, max_lse_rel


def main() -> None:
    args = _parse_args()
    kv_len = args.q_len if args.kv_len is None else args.kv_len
    if args.batch_size <= 0 or args.q_len <= 0 or kv_len <= 0:
        raise ValueError("batch-size, q-len, and kv-len must be positive")
    if kv_len < args.q_len:
        raise ValueError("kv-len must be at least q-len for the causal DSV4 contract")
    if args.topk < SWA_TOPK or args.topk % SELECTOR_VECTOR_WIDTH:
        raise ValueError(
            f"topk must be at least {SWA_TOPK} and divisible by {SELECTOR_VECTOR_WIDTH}"
        )
    if args.compress_ratio <= 0:
        raise ValueError("compress-ratio must be positive")
    if args.pool_rows < args.batch_size * kv_len:
        raise ValueError("pool-rows must cover the physical raw pool for every request")
    if not math.isfinite(args.skip_corr_threshold) or not (
        0.0 <= args.skip_corr_threshold <= 8.0
    ):
        raise ValueError("skip-corr-threshold must be finite and in [0, 8]")
    if (
        args.warmup < 0
        or args.iters <= 0
        or args.paired_rounds <= 0
        or args.profile_launches < 0
        or args.check_samples < 0
    ):
        raise ValueError(
            "warmup/profile/check samples must be non-negative; iters/paired-rounds "
            "must be positive"
        )
    if args.paired_env is not None:
        if not args.paired_env.startswith("FLASHINFER_DSV4_CSA_"):
            raise ValueError("--paired-env must name a FLASHINFER_DSV4_CSA_* variable")
        if args.paired_baseline == args.paired_candidate:
            raise ValueError("paired baseline and candidate values must differ")
        if (
            args.cold_l2
            or args.source_timing
            or args.profile_launches
            or args.check_samples
        ):
            raise ValueError(
                "--paired-env has its own timing protocol and cannot be combined with "
                "cold/source timing, profiling, or sample checking"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("the DSV4 CSA benchmark requires a CUDA GPU")

    device = torch.device("cuda")
    total_q = args.batch_size * args.q_len
    if args.workload_dir is not None:
        if args.batch_size != 1 or total_q != args.q_len:
            raise ValueError(
                "--workload-dir currently implements the source B=1 packed-Q ABI"
            )
        query, compressed_pool, swa_pool, routes, sparse_lens = _load_shared_workload(
            args.workload_dir,
            q_len=args.q_len,
            kv_len=kv_len,
            topk=args.topk,
            pool_rows=args.pool_rows,
            device=device,
        )
        raw_seq_lens = torch.full((1,), kv_len, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, args.q_len], dtype=torch.int32, device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        query = _runner_range_fp8(
            (total_q, HEADS, HEAD_DIM),
            value_range=7.0,
            generator=generator,
            device=device,
        )
        compressed_pool = _runner_range_fp8(
            (args.pool_rows, HEAD_DIM),
            value_range=5.0,
            generator=generator,
            device=device,
        )
        swa_pool = _runner_range_fp8(
            (args.pool_rows, HEAD_DIM),
            value_range=5.0,
            generator=generator,
            device=device,
        )
        routes, sparse_lens, raw_seq_lens, cu_seqlens_q = (
            _build_runner_geometry_metadata(
                batch_size=args.batch_size,
                q_len=args.q_len,
                kv_len=kv_len,
                topk=args.topk,
                compress_ratio=args.compress_ratio,
                pool_rows=args.pool_rows,
                device=device,
            )
        )
    if args.rope_quant:
        out = torch.empty(
            (HEADS // 8, total_q, 8, HEAD_DIM),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        scale_buf_m = (total_q + 3) // 4 * 4
        out_scale = torch.empty(
            (HEADS // 8, 8 * 4, scale_buf_m),
            device=device,
            dtype=torch.float32,
        )
        positions = torch.arange(kv_len, device=device, dtype=torch.float32) + 1
        dims = torch.arange(32, device=device, dtype=torch.float32) + 1
        angles = positions[:, None] * dims[None, :] * 0.001
        inv_rope_cos_sin_cache = torch.cat(
            (torch.cos(angles), torch.sin(angles)), dim=1
        ).contiguous()
    else:
        out = torch.empty(
            (total_q, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
        )
        out_scale = None
        inv_rope_cos_sin_cache = None
    # The source benchmark disables softmax-stat stores.  Keep the default
    # output-only timing on that contract; requesting the FP32 sample oracle
    # explicitly asks the DSV4 wrapper to materialize LSE as well.
    lse = (
        torch.empty((total_q, HEADS), device=device, dtype=torch.float32)
        if args.check_samples
        else None
    )
    bmm1_scale = 1.0 / math.sqrt(HEAD_DIM)

    def run() -> None:
        if args.rope_quant:
            prims_ts_dsv4_sparse_mla_rope_quant(
                query,
                compressed_pool,
                swa_pool,
                routes,
                sparse_lens,
                raw_seq_lens,
                cu_seqlens_q,
                inv_rope_cos_sin_cache,
                max_seq_len_q=args.q_len,
                bmm1_scale=bmm1_scale,
                skip_corr_threshold=args.skip_corr_threshold,
                out=out,
                out_scale=out_scale,
                lse=lse,
            )
        else:
            prims_ts_dsv4_csa(
                query,
                compressed_pool,
                swa_pool,
                routes,
                sparse_lens,
                raw_seq_lens,
                cu_seqlens_q,
                max_seq_len_q=args.q_len,
                bmm1_scale=bmm1_scale,
                skip_corr_threshold=args.skip_corr_threshold,
                out=out,
                lse=lse,
            )

    if args.paired_env is not None:
        variant_env = args.paired_env

        def select_variant(value: str) -> None:
            os.environ[variant_env] = value

        def time_block(value: str) -> float:
            select_variant(value)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                run()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / args.iters

        # Materialize both JIT specializations before warmup/timing.  Changing
        # the environment only selects an already cached compiled callable in
        # the measured ABBA blocks; compilation is outside every CUDA event.
        for value in (args.paired_baseline, args.paired_candidate):
            select_variant(value)
            run()
            torch.cuda.synchronize()
        for value in (args.paired_baseline, args.paired_candidate):
            select_variant(value)
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()

        baseline_rounds: list[float] = []
        candidate_rounds: list[float] = []
        for _ in range(args.paired_rounds):
            baseline_first = time_block(args.paired_baseline)
            candidate_first = time_block(args.paired_candidate)
            candidate_second = time_block(args.paired_candidate)
            baseline_second = time_block(args.paired_baseline)
            baseline_rounds.append((baseline_first + baseline_second) / 2.0)
            candidate_rounds.append((candidate_first + candidate_second) / 2.0)

        deltas = [
            (candidate - baseline) / baseline * 100.0
            for baseline, candidate in zip(
                baseline_rounds, candidate_rounds, strict=True
            )
        ]
        capability = torch.cuda.get_device_capability(device)
        print(
            f"device={torch.cuda.get_device_name(device)} "
            f"SM{capability[0]}{capability[1]}"
        )
        print(
            f"paired_env={variant_env} baseline={args.paired_baseline} "
            f"candidate={args.paired_candidate} protocol=ABBA "
            f"rounds={args.paired_rounds} launches_per_block={args.iters}"
        )
        print(
            "baseline_round_ms=" + ",".join(f"{value:.6f}" for value in baseline_rounds)
        )
        print(
            "candidate_round_ms="
            + ",".join(f"{value:.6f}" for value in candidate_rounds)
        )
        print("candidate_delta_pct=" + ",".join(f"{value:+.3f}" for value in deltas))
        print(
            f"paired_summary baseline_mean={statistics.fmean(baseline_rounds):.6f} "
            f"candidate_mean={statistics.fmean(candidate_rounds):.6f} "
            f"delta_mean_pct={statistics.fmean(deltas):+.3f} "
            f"wins={sum(delta < 0.0 for delta in deltas)}/{len(deltas)}"
        )
        return

    # Compile and populate all device allocations before measurements.
    run()
    torch.cuda.synchronize()
    if args.profile_launches:
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        for _ in range(args.profile_launches):
            run()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        print(f"profiled_launches={args.profile_launches}")
        return
    if args.check_samples and args.workload_dir is None:
        raise ValueError(
            "--check-samples requires --workload-dir to fix the B=1 source ABI"
        )

    attended_tokens = int(sparse_lens.sum().item())
    sparse_lens_min = int(sparse_lens.min().item())
    sparse_lens_max = int(sparse_lens.max().item())
    if args.source_timing and args.cold_l2:
        raise ValueError("--source-timing and --cold-l2 are mutually exclusive")

    if args.source_timing:
        # Fmha.cpp clears 2x L2 once before source warmup, then brackets all
        # benchmark launches in one event. Keep the eviction buffer alive until
        # after the stop event because it is an asynchronous stream operation.
        l2_bytes = _l2_cache_size_bytes(device)
        if l2_bytes <= 0:
            raise RuntimeError(
                "CUDA device does not expose l2_cache_size for --source-timing"
            )
        l2_clear = torch.empty(2 * l2_bytes, dtype=torch.uint8, device=device)
        l2_clear.zero_()
        for _ in range(args.warmup):
            run()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run()
        stop.record()
        stop.synchronize()
        source_mean_ms = start.elapsed_time(stop) / args.iters
        timing_label = f"source_mean={source_mean_ms:.6f}"
        primary_ms = source_mean_ms
    else:
        measurements = bench_gpu_time(
            run,
            dry_run_iters=args.warmup,
            repeat_iters=args.iters,
            cold_l2_cache=args.cold_l2,
        )
        median_ms = statistics.median(measurements)
        sorted_ms = sorted(measurements)
        p90_ms = sorted_ms[math.ceil(0.9 * len(sorted_ms)) - 1]
        timing_label = (
            f"median={median_ms:.6f} mean={statistics.fmean(measurements):.6f} "
            f"min={min(sorted_ms):.6f} p90={p90_ms:.6f}"
        )
        primary_ms = median_ms
    # QK and PV each comprise D multiply-adds per attended KV token.
    algorithmic_flops = 4 * attended_tokens * HEADS * HEAD_DIM
    tflops = algorithmic_flops / (primary_ms * 1e9)

    capability = torch.cuda.get_device_capability(device)
    print(
        f"device={torch.cuda.get_device_name(device)} SM{capability[0]}{capability[1]}"
    )
    print(
        f"B={args.batch_size} Q={args.q_len} KV={kv_len} H={HEADS} D={HEAD_DIM} "
        f"Kmax={args.topk} R={args.compress_ratio} pool_rows={args.pool_rows}"
    )
    l2_mode = (
        "source-clear-once"
        if args.source_timing
        else ("cold" if args.cold_l2 else "warm")
    )
    print(
        f"contract_f_only=true l2={l2_mode} "
        f"rope_quant={str(args.rope_quant).lower()} "
        f"skip_corr_threshold={args.skip_corr_threshold:g} "
        f"Lq_min={sparse_lens_min} Lq_max={sparse_lens_max} "
        f"attended_tokens={attended_tokens}"
    )
    if args.workload_dir is not None:
        print(f"shared_workload={args.workload_dir.resolve()}")
    print(f"latency_ms: {timing_label}")
    if args.source_timing:
        print(
            "timing_protocol=source-style(one 2xL2 clear, requested warmups, one event / iters); "
            "TS reuses one compiled code instance"
        )
    print(f"algorithmic_throughput={tflops:.3f} TFLOPS/s")
    if args.check_samples:
        if args.rope_quant:
            raise NotImplementedError(
                "--check-samples for --rope-quant is pending the grouped-output "
                "dequant oracle; use the dedicated correctness regression"
            )
        checked, out_abs, out_rel, lse_abs, lse_rel = _shared_fp32_sample_reference(
            out=out,
            lse=lse,
            query=query,
            compressed_pool=compressed_pool,
            swa_pool=swa_pool,
            routes=routes,
            sparse_lens=sparse_lens,
            kv_len=kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            skip_corr_threshold=args.skip_corr_threshold,
            sample_count=args.check_samples,
        )
        print(
            f"fp32_sample_check=PASS rows={checked} max_out_abs={out_abs:.6g} "
            f"max_out_rel={out_rel:.6g} max_lse_abs={lse_abs:.6g} max_lse_rel={lse_rel:.6g}"
        )
    if args.output_o is not None:
        args.output_o.parent.mkdir(parents=True, exist_ok=True)
        out.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(args.output_o)
        print(
            f"output_o={args.output_o.resolve()} bytes={out.numel() * out.element_size()}"
        )
    if args.output_scale is not None:
        if out_scale is None:
            raise ValueError("--output-scale requires --rope-quant")
        args.output_scale.parent.mkdir(parents=True, exist_ok=True)
        out_scale.detach().contiguous().cpu().numpy().tofile(args.output_scale)
        print(
            f"output_scale={args.output_scale.resolve()} "
            f"bytes={out_scale.numel() * out_scale.element_size()}"
        )


if __name__ == "__main__":
    main()
