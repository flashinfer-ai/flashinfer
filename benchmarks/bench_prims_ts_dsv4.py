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

"""Benchmark the PrimTS DeepSeek V4 sparse MLA kernels.

The default is the CSA context shape: B=1, Q=KV=8192, Hq=128, Dqk=Dv=512,
E4M3 input, BF16 output, compression ratio 4, and Kmax=1152, with the
recommended E4M3 skip-correction threshold of 8.  Routing follows the DSV4
dynamic sparse length rule: the first K tile holds a causal sliding-window
prefix of at most 128 tokens and the remaining slots hold at most
``Kmax - 128`` compressed tokens.  Compressed rows are a deterministic
per-query permutation; which rows are selected does not change the FMHA work.

Cache append/compression, candidate selection, and page-table lowering are
outside the timed region, so the numbers measure the attention kernel alone.

``--refcheck`` compares sampled query rows against an FP32 reference that
reproduces the kernel's online-softmax protocol (tile structure, skip
correction, and E4M3 P quantization).  ``--scheduler`` pins the static or
CLC-persistent 2CTA variant for A/B timing; ``auto`` applies the resident-wave
policy the public API uses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flashinfer.attention.prims_ts.dsv4 as dsv4_module
from flashinfer.attention.prims_ts import (
    prims_ts_dsv4_sparse_mla,
    prims_ts_dsv4_sparse_mla_rope_quant,
    prims_ts_dsv4_sparse_mla_rope_quant_ue8m0,
)
from flashinfer.testing import bench_gpu_time


HEADS = 128
HEAD_DIM = 512
SWA_TOPK = 128
TILE_K = 128
SELECTOR_VECTOR_WIDTH = 4


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--skip-corr-threshold",
        type=float,
        default=8.0,
        help=(
            "E4M3 skip-correction threshold in log2 units, in [0, 8]; positive "
            "values enable the specialization, 0 disables it (default: 8)"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--scheduler",
        choices=("auto", "static", "persistent"),
        default="auto",
        help=(
            "benchmark-only override of the 2CTA scheduler: 'auto' applies the "
            "resident-wave policy (static grid within one wave, CLC persistent "
            "beyond), 'static'/'persistent' pin one variant for A/B timing"
        ),
    )
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="flush L2 before each measured iteration (default: warm L2)",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help=(
            "after JIT warmup, bracket this many launches with cudaProfilerStart/Stop "
            "and exit; intended for ncu --profile-from-start off"
        ),
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        help=(
            "after timing, compare sampled query rows against an FP32 reference of "
            "the kernel's online-softmax protocol (BF16 output kernel only)"
        ),
    )
    parser.add_argument(
        "--refcheck-samples",
        type=int,
        default=12,
        help="number of query rows --refcheck compares (default: 12)",
    )
    parser.add_argument(
        "--rope-quant",
        action="store_true",
        help=(
            "select the fused inverse-RoPE + grouped E4M3 output epilogue "
            "instead of the BF16 output kernel"
        ),
    )
    parser.add_argument(
        "--scale-format",
        choices=("fp32", "ue8m0"),
        default="fp32",
        help=(
            "--rope-quant dequant-scale format: one FP32 value per D128 block "
            "(default) or packed UE8M0 exponents, four D128-block bytes per INT32"
        ),
    )
    return parser.parse_args()


def _uniform_fp8(
    shape: tuple[int, ...],
    *,
    value_range: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw non-degenerate E4M3 values in ``[-value_range, value_range]``."""

    return (
        torch.empty(shape, device=device, dtype=torch.bfloat16)
        .uniform_(-value_range, value_range, generator=generator)
        .to(torch.float8_e4m3fn)
    )


def _build_geometry_metadata(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    topk: int,
    compress_ratio: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DSV4 physical routing metadata for a uniform packed batch."""

    total_q = batch_size * q_len
    q_pos = torch.arange(q_len, dtype=torch.int32, device=device).repeat(batch_size)
    batch_ids = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(q_len)
    # Causal position of each query in its request: kv_len - q_len + q_pos.
    raw_visible = kv_len - q_len + q_pos + 1
    swa_valid = raw_visible.clamp(min=0, max=SWA_TOPK)
    primary_capacity = topk - SWA_TOPK
    compressed_visible = torch.div(raw_visible, compress_ratio, rounding_mode="floor")
    compressed_count = compressed_visible.clamp(min=0, max=primary_capacity)
    sparse_lens = swa_valid.new_full((total_q,), SWA_TOPK) + compressed_count

    # Inactive slots in the fixed SWA tile are -1, exactly as the routing
    # contract specifies.  The Gather4 path reaches those coordinates before QK
    # masking, so replacing them with row 0 would change the address stream
    # and hence the measured kernel.  The unselected compressed tail stays 0.
    page_idx_kv = torch.zeros((total_q, topk), dtype=torch.int32, device=device)
    swa_slot = torch.arange(SWA_TOPK, dtype=torch.int32, device=device)
    swa_rows = batch_ids[:, None] * kv_len + raw_visible[:, None] - swa_valid[:, None]
    swa_rows = swa_rows + swa_slot
    page_idx_kv[:, :SWA_TOPK] = torch.where(
        swa_slot[None, :] < swa_valid[:, None], swa_rows, torch.full_like(swa_rows, -1)
    )

    # A deterministic per-query permutation of the visible compressed tokens.
    # Stride 37 is coprime to power-of-two capacities (2048 for CSA R=4 and 64
    # for HCA R=128 at 8K); other geometries may repeat rows, which changes
    # neither the work nor the addresses' validity.  ``compressed_count`` is
    # zero for very short workloads, leaving every primary slot inactive.
    compressed_capacity = max(kv_len // compress_ratio, 1)
    compressed_slot = torch.arange(primary_capacity, dtype=torch.int32, device=device)
    compressed_base = torch.remainder(q_pos * 313, compressed_capacity)
    compressed_rows = torch.remainder(
        compressed_base[:, None] + compressed_slot[None, :] * 37,
        compressed_capacity,
    )
    compressed_rows = compressed_rows + batch_ids[:, None] * compressed_capacity
    page_idx_kv[:, SWA_TOPK:] = torch.where(
        compressed_slot[None, :] < compressed_count[:, None],
        compressed_rows,
        torch.zeros_like(compressed_rows),
    )
    raw_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device).mul_(
        q_len
    )
    return page_idx_kv, sparse_lens, raw_seq_lens, cu_seqlens_q


def _fp32_sample_reference(
    *,
    out: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    compressed_pool: torch.Tensor,
    swa_pool: torch.Tensor,
    page_idx_kv: torch.Tensor,
    sparse_lens: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    sample_count: int,
) -> tuple[int, float, float, float, float]:
    """Check sampled query rows against the kernel's numerical protocol.

    The selected rows include the SWA-prefix and first-K-tile transitions
    before adding evenly distributed positions.  This is deliberately a
    sampled FP32 oracle: a full eager attention over 8K queries is not a
    benchmark harness.  Active SWA slots are read from the routing tensor, so
    the check is independent of batch layout.
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
        swa_slots = page_idx_kv[query_idx, :SWA_TOPK]
        # -1 marks an inactive SWA slot; the active prefix is contiguous.
        swa_valid = int((swa_slots >= 0).sum().item())
        swa_page_idx = swa_slots[:swa_valid]
        if bool((swa_page_idx < 0).any()):
            raise ValueError(
                f"query {query_idx}: inactive SWA slot inside the active prefix"
            )
        kv_tiles = [swa_pool[swa_page_idx.long()].float()]
        compressed_page_idx = page_idx_kv[query_idx, SWA_TOPK:active_width]
        for tile_begin in range(0, compressed_page_idx.numel(), TILE_K):
            tile_page_idx = compressed_page_idx[tile_begin : tile_begin + TILE_K]
            kv_tiles.append(compressed_pool[tile_page_idx.long()].float())

        # Reproduce the kernel's online-softmax protocol rather than comparing
        # against exact torch.softmax: compressed rows start at logical tile 1
        # even when the causal SWA prefix has fewer than 128 valid rows, the
        # row max is frozen under the skip-correction threshold, and P is
        # quantized to E4M3 before PV.
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
    if not math.isfinite(args.skip_corr_threshold) or not (
        0.0 <= args.skip_corr_threshold <= 8.0
    ):
        raise ValueError("skip-corr-threshold must be finite and in [0, 8]")
    if args.warmup < 0 or args.iters <= 0 or args.profile_iters < 0:
        raise ValueError("warmup/profile-iters must be non-negative; iters positive")
    if args.refcheck and args.refcheck_samples <= 0:
        raise ValueError("--refcheck-samples must be positive")
    if args.scale_format != "fp32" and not args.rope_quant:
        raise ValueError("--scale-format applies only with --rope-quant")
    if args.refcheck and args.rope_quant:
        raise NotImplementedError(
            "--refcheck covers the BF16 output kernel; the grouped E4M3 output is "
            "validated by tests/attention/test_attention_ts_dsv4.py"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the DSV4 sparse MLA benchmark requires a CUDA GPU")

    device = torch.device("cuda")
    total_q = args.batch_size * args.q_len
    # Pool geometry follows the routing rule: page size 1 with uniform request
    # lengths needs B * KV sliding-window rows and B * floor(KV / R) compressed
    # rows, exactly the rows _build_geometry_metadata can address.
    swa_pool_rows = args.batch_size * kv_len
    compressed_pool_rows = args.batch_size * max(kv_len // args.compress_ratio, 1)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    query = _uniform_fp8(
        (total_q, HEADS, HEAD_DIM),
        value_range=7.0,
        generator=generator,
        device=device,
    )
    compressed_pool = _uniform_fp8(
        (compressed_pool_rows, HEAD_DIM),
        value_range=5.0,
        generator=generator,
        device=device,
    )
    swa_pool = _uniform_fp8(
        (swa_pool_rows, HEAD_DIM),
        value_range=5.0,
        generator=generator,
        device=device,
    )
    page_idx_kv, sparse_lens, raw_seq_lens, cu_seqlens_q = _build_geometry_metadata(
        batch_size=args.batch_size,
        q_len=args.q_len,
        kv_len=kv_len,
        topk=args.topk,
        compress_ratio=args.compress_ratio,
        device=device,
    )
    if args.rope_quant:
        out = torch.empty(
            (HEADS // 8, total_q, 8, HEAD_DIM),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        scale_buf_m = (total_q + 3) // 4 * 4
        if args.scale_format == "ue8m0":
            # One INT32 word per (group, head, padded token) packing four
            # D128-block exponent bytes.
            out_scale = torch.empty(
                (HEADS // 8, 8, scale_buf_m), device=device, dtype=torch.int32
            )
        else:
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
    # Serving does not store softmax stats, so time the output-only variant by
    # default; --refcheck needs LSE and therefore selects the storing variant.
    lse = (
        torch.empty((total_q, HEADS), device=device, dtype=torch.float32)
        if args.refcheck
        else None
    )
    bmm1_scale = 1.0 / math.sqrt(HEAD_DIM)

    rope_quant_entry = (
        prims_ts_dsv4_sparse_mla_rope_quant_ue8m0
        if args.scale_format == "ue8m0"
        else prims_ts_dsv4_sparse_mla_rope_quant
    )

    def run() -> None:
        if args.rope_quant:
            rope_quant_entry(
                query,
                compressed_pool,
                swa_pool,
                page_idx_kv,
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
            prims_ts_dsv4_sparse_mla(
                query,
                compressed_pool,
                swa_pool,
                page_idx_kv,
                sparse_lens,
                raw_seq_lens,
                cu_seqlens_q,
                max_seq_len_q=args.q_len,
                bmm1_scale=bmm1_scale,
                skip_corr_threshold=args.skip_corr_threshold,
                out=out,
                lse=lse,
            )

    if args.scheduler == "auto":
        selected_scheduler = (
            "persistent"
            if dsv4_module._dsv4_uses_persistent_scheduler(
                int(cu_seqlens_q.numel()) - 1,
                args.q_len,
                dsv4_module._dsv4_max_active_clusters(torch.cuda.current_device()),
            )
            else "static"
        )
    else:
        selected_scheduler = args.scheduler
        forced_persistent = args.scheduler == "persistent"
        dsv4_module._dsv4_uses_persistent_scheduler = (
            lambda *unused_args, **unused_kwargs: forced_persistent
        )
    # Compile and populate all device allocations before measurements.
    run()
    torch.cuda.synchronize()
    if args.profile_iters:
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        for _ in range(args.profile_iters):
            run()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        print(f"profiled_iters={args.profile_iters}")
        return

    # Count only routed tokens: active SWA slots plus the compressed scan width.
    attended_tokens = int((page_idx_kv[:, :SWA_TOPK] >= 0).sum().item()) + int(
        (sparse_lens - SWA_TOPK).sum().item()
    )
    sparse_lens_min = int(sparse_lens.min().item())
    sparse_lens_max = int(sparse_lens.max().item())

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
        f"Kmax={args.topk} R={args.compress_ratio} "
        f"swa_pool_rows={swa_pool.shape[0]} compressed_pool_rows={compressed_pool.shape[0]}"
    )
    l2_mode = "cold" if args.cold_l2 else "warm"
    print(
        f"l2={l2_mode} "
        f"rope_quant={str(args.rope_quant).lower()} "
        f"scale_format={args.scale_format if args.rope_quant else 'n/a'} "
        f"skip_corr_threshold={args.skip_corr_threshold:g} "
        f"Lq_min={sparse_lens_min} Lq_max={sparse_lens_max} "
        f"attended_tokens={attended_tokens}"
    )
    print(f"latency_ms: {timing_label}")
    print(f"scheduler={selected_scheduler} ({args.scheduler})")
    print(f"algorithmic_throughput={tflops:.3f} TFLOPS/s")
    if args.refcheck:
        checked, out_abs, out_rel, lse_abs, lse_rel = _fp32_sample_reference(
            out=out,
            lse=lse,
            query=query,
            compressed_pool=compressed_pool,
            swa_pool=swa_pool,
            page_idx_kv=page_idx_kv,
            sparse_lens=sparse_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            skip_corr_threshold=args.skip_corr_threshold,
            sample_count=args.refcheck_samples,
        )
        print(
            f"refcheck=PASS rows={checked} max_out_abs={out_abs:.6g} "
            f"max_out_rel={out_rel:.6g} max_lse_abs={lse_abs:.6g} max_lse_rel={lse_rel:.6g}"
        )


if __name__ == "__main__":
    main()
