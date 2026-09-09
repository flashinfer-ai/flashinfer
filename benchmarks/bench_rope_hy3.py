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

"""Benchmark the HY3 fused RoPE/KV-store path on B200.

Two comparisons are reported because existing FlashInfer has no operator that
matches HY3's dynamic per-token/per-Q-head scale:

* ``static_norm2_flashinfer`` compares the static-Q HY3 fusion with the closest
  existing FlashInfer pipeline: two RMSNorm calls followed by
  ``rope_quantize_fp8_append_paged_kv_cache``. Q/K/V views, metadata, and all
  outputs are prepared outside timing, following the HPC-Ops benchmark. The
  FlashInfer path does not clear the unused cache tail or write split flags, so
  this comparison is conservative in its favor.
* ``dynamic_norm2_b200`` compares the source-faithful HY3 dynamic-Q kernel with
  its B200 one-token-decode specialization. These two paths have identical
  outputs and side effects.

Usage:
$ python -m benchmarks.bench_rope_hy3
$ python -m benchmarks.bench_rope_hy3 --batches 256 512 --hot-l2
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 64
Q_SCALE_INVERSE = 2.0
KV_SCALE = 0.5


def _measure(call, args):
    values = bench_gpu_time(
        call,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        enable_cupti=not args.cuda_events,
        cold_l2_cache=not args.hot_l2,
    )
    values = np.asarray(values, dtype=np.float64) * 1e3
    return (
        float(np.median(values)),
        float(np.percentile(values, 20)),
        float(np.percentile(values, 80)),
    )


def _make_inputs(batch_size, device):
    packed_width = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    generator = torch.Generator(device=device).manual_seed(1000 + batch_size)
    packed_qkv = torch.randn(
        (batch_size, packed_width),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_width = NUM_Q_HEADS * HEAD_DIM
    k_width = NUM_KV_HEADS * HEAD_DIM
    q = packed_qkv[:, :q_width].reshape(batch_size, NUM_Q_HEADS, HEAD_DIM)
    k = packed_qkv[:, q_width : q_width + k_width].reshape(
        batch_size, NUM_KV_HEADS, HEAD_DIM
    )
    v = packed_qkv[:, q_width + k_width :].reshape(batch_size, NUM_KV_HEADS, HEAD_DIM)

    sequence_lengths = (
        torch.arange(batch_size, dtype=torch.int32, device=device) % PAGE_SIZE + 1
    )
    q_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(batch_size, dtype=torch.int32, device=device).reshape(
        batch_size, 1
    )
    positions = sequence_lengths - 1
    batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    kv_indices = batch_indices.clone()
    kv_indptr = q_indptr.clone()
    kv_last_page_len = sequence_lengths.clone()

    rotary_positions = torch.arange(PAGE_SIZE + 1, dtype=torch.float32, device=device)
    inverse_frequency = 1.0 / (
        10000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    frequencies = torch.outer(rotary_positions, inverse_frequency)
    cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

    weight_bf16 = torch.linspace(
        0.75, 1.25, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    return {
        "packed_qkv": packed_qkv,
        # Existing APIs consume separated contiguous Q/K/V. Preparing these
        # outside timing favors the FlashInfer baseline and matches HPC-Ops.
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "sequence_lengths": sequence_lengths,
        "q_indptr": q_indptr,
        "block_table": block_table,
        "positions": positions,
        "batch_indices": batch_indices,
        "kv_indices": kv_indices,
        "kv_indptr": kv_indptr,
        "kv_last_page_len": kv_last_page_len,
        "cos_sin_cache": cos_sin_cache,
        "weight_bf16": weight_bf16,
        "weight_fp32": weight_bf16.float(),
    }


def _make_cache(batch_size, device):
    shape = (batch_size, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    key = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    value = torch.zeros_like(key)
    return key, value


@torch.inference_mode()
def _benchmark_batch(batch_size, args):
    device = torch.device(f"cuda:{args.device}")
    inputs = _make_inputs(batch_size, device)
    k_scale = torch.tensor([KV_SCALE], dtype=torch.float32, device=device)
    v_scale = torch.tensor([KV_SCALE], dtype=torch.float32, device=device)
    q_scale_inverse = torch.tensor(
        [Q_SCALE_INVERSE], dtype=torch.float32, device=device
    )

    static_hy3_cache = _make_cache(batch_size, device)
    static_fi_cache = _make_cache(batch_size, device)
    static_hy3_q = torch.empty(
        batch_size,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    static_fi_q = torch.empty_like(static_hy3_q)
    static_flags = torch.full(
        (batch_size, NUM_KV_HEADS), -1, dtype=torch.int32, device=device
    )
    q_norm = torch.empty_like(inputs["q"])
    k_norm = torch.empty_like(inputs["k"])
    q_nope = torch.empty(
        batch_size, NUM_Q_HEADS, 0, dtype=torch.bfloat16, device=device
    )
    k_nope = torch.empty(
        batch_size, NUM_KV_HEADS, 0, dtype=torch.bfloat16, device=device
    )
    q_nope_out = torch.empty_like(q_nope, dtype=torch.float8_e4m3fn)

    def hy3_static():
        return flashinfer.qk_rmsnorm_rope_append_paged_kv_cache_hy3(
            inputs["packed_qkv"],
            inputs["cos_sin_cache"],
            inputs["sequence_lengths"],
            inputs["q_indptr"],
            inputs["block_table"],
            static_hy3_cache,
            False,
            q_norm_weight=inputs["weight_fp32"],
            k_norm_weight=inputs["weight_fp32"],
            norm_policy=2,
            quant_policy=2,
            k_scale=k_scale,
            v_scale=v_scale,
            q_scale_inverse=q_scale_inverse,
            out_q=static_hy3_q,
            split_k_flag=static_flags,
        )

    def flashinfer_static():
        flashinfer.rmsnorm(
            inputs["q"],
            inputs["weight_bf16"],
            out=q_norm,
            enable_pdl=False,
        )
        flashinfer.rmsnorm(
            inputs["k"],
            inputs["weight_bf16"],
            out=k_norm,
            enable_pdl=False,
        )
        return flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_norm,
            k_norm,
            q_nope,
            k_nope,
            inputs["v"],
            inputs["cos_sin_cache"],
            inputs["positions"],
            static_fi_cache,
            inputs["kv_indices"],
            inputs["kv_indptr"],
            inputs["batch_indices"],
            inputs["positions"],
            is_neox=True,
            quantize_dtype=torch.float8_e4m3fn,
            quant_scale_q=Q_SCALE_INVERSE,
            quant_scale_kv=1.0 / KV_SCALE,
            page_size=PAGE_SIZE,
            kv_layout="NHD",
            q_rope_out=static_fi_q,
            q_nope_out=q_nope_out,
            enable_pdl=False,
        )

    dynamic_source_cache = _make_cache(batch_size, device)
    dynamic_fast_cache = _make_cache(batch_size, device)
    dynamic_source_q = torch.empty_like(static_hy3_q)
    dynamic_fast_q = torch.empty_like(static_hy3_q)
    dynamic_source_scale = torch.empty(
        batch_size, NUM_Q_HEADS, dtype=torch.float32, device=device
    )
    dynamic_fast_scale = torch.empty_like(dynamic_source_scale)
    dynamic_source_flags = torch.full_like(static_flags, -1)
    dynamic_fast_flags = torch.full_like(static_flags, -1)

    def hy3_dynamic_source():
        return flashinfer.qk_rmsnorm_rope_append_paged_kv_cache_hy3(
            inputs["packed_qkv"],
            inputs["cos_sin_cache"],
            inputs["sequence_lengths"],
            inputs["q_indptr"],
            inputs["block_table"],
            dynamic_source_cache,
            False,
            q_norm_weight=inputs["weight_fp32"],
            k_norm_weight=inputs["weight_fp32"],
            norm_policy=2,
            quant_policy=1,
            k_scale=k_scale,
            v_scale=v_scale,
            out_q=dynamic_source_q,
            out_q_scale=dynamic_source_scale,
            split_k_flag=dynamic_source_flags,
            uniform_one_token_decode=False,
        )

    def hy3_dynamic_fast():
        return flashinfer.qk_rmsnorm_rope_append_paged_kv_cache_hy3(
            inputs["packed_qkv"],
            inputs["cos_sin_cache"],
            inputs["sequence_lengths"],
            inputs["q_indptr"],
            inputs["block_table"],
            dynamic_fast_cache,
            False,
            q_norm_weight=inputs["weight_fp32"],
            k_norm_weight=inputs["weight_fp32"],
            norm_policy=2,
            quant_policy=1,
            k_scale=k_scale,
            v_scale=v_scale,
            out_q=dynamic_fast_q,
            out_q_scale=dynamic_fast_scale,
            split_k_flag=dynamic_fast_flags,
            uniform_one_token_decode=True,
        )

    # Compile every provider and establish the common-output correctness
    # boundary before measuring. Tail clearing and flags are HY3-only work.
    hy3_static()
    flashinfer_static()
    torch.cuda.synchronize(device)
    written = inputs["positions"].long()
    pages = torch.arange(batch_size, dtype=torch.long, device=device)
    torch.testing.assert_close(
        static_hy3_q.float(), static_fi_q.float(), rtol=0.2, atol=0.5
    )
    torch.testing.assert_close(
        static_hy3_cache[0][pages, written].float(),
        static_fi_cache[0][pages, written].float(),
        rtol=0.2,
        atol=0.5,
    )
    torch.testing.assert_close(
        static_hy3_cache[1][pages, written].float(),
        static_fi_cache[1][pages, written].float(),
        rtol=0.0,
        atol=0.0,
    )
    if torch.any(static_flags == -1):
        raise AssertionError("HY3 static path did not write every split_k_flag")

    hy3_dynamic_source()
    hy3_dynamic_fast()
    torch.cuda.synchronize(device)
    for source, optimized in (
        (dynamic_source_q, dynamic_fast_q),
        (dynamic_source_scale, dynamic_fast_scale),
        (dynamic_source_flags, dynamic_fast_flags),
        (dynamic_source_cache[0], dynamic_fast_cache[0]),
        (dynamic_source_cache[1], dynamic_fast_cache[1]),
    ):
        if not torch.equal(source, optimized):
            raise AssertionError(
                "B200 dynamic fast path differs from source-faithful HY3"
            )

    results = {
        "hy3_fused_static": _measure(hy3_static, args),
        "flashinfer_composed_static": _measure(flashinfer_static, args),
        "hy3_source_faithful_dynamic": _measure(hy3_dynamic_source, args),
        "hy3_b200_tail_fused_dynamic": _measure(hy3_dynamic_fast, args),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--hot-l2",
        action="store_true",
        help="reuse hot inputs instead of flushing L2 between measurements",
    )
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="use CUDA events instead of the preferred CUPTI timer",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("HY3 RoPE/KV-store benchmark skipped: CUDA is unavailable")
        return
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device) != (10, 0):
        print("HY3 RoPE/KV-store benchmark skipped: requires SM100/B200")
        return

    cache_mode = "hot" if args.hot_l2 else "cold"
    timer = "cuda_event" if args.cuda_events else "cupti_or_event_fallback"
    print(f"# cache={cache_mode}, timer={timer}, heads=64/8, dim=128, page=64")
    print("case,batch,provider,median_us,p20_us,p80_us,speedup")
    for batch_size in args.batches:
        results = _benchmark_batch(batch_size, args)
        static_speedup = (
            results["flashinfer_composed_static"][0] / results["hy3_fused_static"][0]
        )
        dynamic_speedup = (
            results["hy3_source_faithful_dynamic"][0]
            / results["hy3_b200_tail_fused_dynamic"][0]
        )
        for case, providers, optimized, speedup in (
            (
                "static_norm2_flashinfer",
                ("flashinfer_composed_static", "hy3_fused_static"),
                "hy3_fused_static",
                static_speedup,
            ),
            (
                "dynamic_norm2_b200",
                (
                    "hy3_source_faithful_dynamic",
                    "hy3_b200_tail_fused_dynamic",
                ),
                "hy3_b200_tail_fused_dynamic",
                dynamic_speedup,
            ),
        ):
            for provider in providers:
                median, p20, p80 = results[provider]
                row_speedup = speedup if provider == optimized else 1.0
                print(
                    f"{case},{batch_size},{provider},{median:.3f},{p20:.3f},"
                    f"{p80:.3f},{row_speedup:.3f}"
                )


if __name__ == "__main__":
    main()
