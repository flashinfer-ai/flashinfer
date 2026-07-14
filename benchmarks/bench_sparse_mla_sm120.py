# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Microbenchmark for sparse-MLA paged attention on SM120.

Sweeps representative shapes:

* DSv4   (d_qk=512, page_block_size=64, 584 B/token)
* DSv4 dual cache (fixed main cache + secondary cache)
* DSv3.2 (d_qk=576, page_block_size=64, 656 B/token)

The KV pool is sized ≫ L2 so the analytical KV bandwidth reflects DRAM-bound
large-cache prefill. Reports median latency, KV bandwidth, and analytical
attention TFLOPs. Requires sm120a / sm121a.
"""

import numpy as np
import torch

import flashinfer
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import is_sm120a_supported


_BPT_DSV4 = 584
_PAGE_BLOCK_SIZE_DSV4 = 64
_D_QK_DSV4 = 512
_D_V = 512
_WORKSPACE_BYTES = 128 * 1024 * 1024


# ── FP8 FOOTER pack (DSV4) ───────────────────────────────────────────────────


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_model1(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into the FP8 FOOTER format the SM120 sparse-MLA kernel reads."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    result_flat = torch.zeros(nb, bs * bpt, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)
        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]
    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]
    return result_flat.view(nb, bs, 1, bpt)


# ── DSv3.2 INLINE pack (656 B/token) ─────────────────────────────────────────


def quantize_kv_dsv3_2(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV (nb, pbs, 1, 576) into DSv3.2 FP8 INLINE format (656 B/token).

    Per-token layout: [0:512) FP8 nope (4×128 tile), [512:528) 4×FP32
    power-of-2 scale, [528:656) BF16 rope (64 elems).
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    bpt = d_nope + scale_bytes + d_rope * 2  # 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    kv = kv_bf16.squeeze(2)  # (nb, bs, 576)

    result = torch.zeros(nb, bs, bpt, dtype=torch.uint8, device=kv.device)
    for ti in range(num_tiles):
        tile = kv[:, :, ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, :, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, :, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nb, bs, 4)
        )
    rope = kv[:, :, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, :, d_nope + scale_bytes :] = rope.view(nb, bs, d_rope * 2)
    return result.view(nb, bs, 1, bpt)


# ── Benchmark one config ─────────────────────────────────────────────────────


def bench_sparse_mla_sm120(num_heads, topk, num_tokens, with_sink=False, seed=0):
    """Returns (median_us, kv_bw_gbps, attn_tflops)."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    d_qk, d_v = _D_QK_DSV4, _D_V
    page_block_size = _PAGE_BLOCK_SIZE_DSV4
    # Pool ≫ L2 (~96 MB on SM120) so random topk indices land in DRAM.
    # 16384 × 64 × 584 B ≈ 612 MB.
    num_blocks = 16384
    s_kv = num_blocks * page_block_size  # = 1 M slots

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    sm_scale = d_qk**-0.5

    def fn():
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q.unsqueeze(1),
            swa_kv_cache=kv_packed,
            workspace_buffer=workspace,
            sparse_indices=indices,
            swa_topk_lens=indices.new_full((num_tokens,), topk),
            out=output.unsqueeze(1),
            bmm1_scale=sm_scale,
            sinks=attn_sink,
        )

    # Warm + measure.
    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=1000)
    ms = float(np.median(measurements))

    # KV bandwidth: read num_tokens * topk slots × 584 bytes/slot from gmem.
    bpt = _BPT_DSV4  # DSV4 FOOTER per-token bytes
    kv_bytes = num_tokens * topk * bpt
    kv_bw_gbps = kv_bytes * 1e-6 / ms  # GB/s
    # Analytical attention FLOPs: 2 * num_heads * topk * (d_qk + d_v) per token.
    flops = 2 * num_tokens * num_heads * topk * (d_qk + d_v)
    tflops = flops * 1e-9 / ms

    return ms * 1e3, kv_bw_gbps, tflops  # us, GB/s, TFLOPs


def _build_dsv4_pool(num_blocks, page_block_size, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    kv_bf16 = (
        torch.randn(
            num_blocks,
            page_block_size,
            1,
            _D_QK_DSV4,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        / 10.0
    ).clamp(-1, 1)
    return quantize_kv_model1(kv_bf16)


def _actual_extra_topk(topk_extra, extra_topk_length):
    if extra_topk_length is None:
        return topk_extra
    return min(max(extra_topk_length, 0), topk_extra)


def bench_sparse_mla_sm120_dsv4_dual(
    num_heads,
    topk_extra,
    num_tokens,
    extra_page_block_size,
    extra_topk_length=None,
    with_sink=False,
    seed=0,
):
    """Returns (median_us, kv_bw_gbps, attn_tflops) for DSv4 dual-cache prefill.

    The main cache is fixed at ``topk=128, page_block_size=64``. The secondary
    cache covers both supported page sizes and runtime ``topk_extra`` values.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    d_qk, d_v = _D_QK_DSV4, _D_V
    topk_main = 128
    main_page_block_size = _PAGE_BLOCK_SIZE_DSV4

    main_blocks = 16384  # 16384 × 64 × 584 B ≈ 612 MB.
    s_kv_main = main_blocks * main_page_block_size
    target_extra_bytes = 256 * 1024 * 1024
    extra_blocks = max(1024, target_extra_bytes // (extra_page_block_size * _BPT_DSV4))
    s_kv_extra = extra_blocks * extra_page_block_size

    kv_main = _build_dsv4_pool(main_blocks, main_page_block_size, device, seed)
    kv_extra = _build_dsv4_pool(extra_blocks, extra_page_block_size, device, seed + 1)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv_main, (num_tokens, topk_main), device=device, dtype=torch.int32
    )
    indices_extra = torch.randint(
        0, s_kv_extra, (num_tokens, topk_extra), device=device, dtype=torch.int32
    )
    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    sm_scale = d_qk**-0.5

    def fn():
        extra_lens = indices.new_full(
            (num_tokens,), _actual_extra_topk(topk_extra, extra_topk_length)
        )
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q.unsqueeze(1),
            swa_kv_cache=kv_main,
            workspace_buffer=workspace,
            sparse_indices=indices,
            compressed_kv_cache=kv_extra,
            swa_topk_lens=indices.new_full((num_tokens,), topk_main),
            extra_sparse_indices=indices_extra,
            extra_sparse_topk_lens=extra_lens,
            out=output.unsqueeze(1),
            bmm1_scale=sm_scale,
            sinks=attn_sink,
        )

    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=1000)
    ms = float(np.median(measurements))

    actual_extra_topk = _actual_extra_topk(topk_extra, extra_topk_length)
    kv_bytes = num_tokens * (topk_main + actual_extra_topk) * _BPT_DSV4
    kv_bw_gbps = kv_bytes * 1e-6 / ms
    flops = 2 * num_tokens * num_heads * (topk_main + actual_extra_topk) * (d_qk + d_v)
    tflops = flops * 1e-9 / ms

    return ms * 1e3, kv_bw_gbps, tflops


def bench_sparse_mla_sm120_dsv3_2(num_heads, num_tokens, with_sink=False, seed=0):
    """Returns (median_us, kv_bw_gbps, attn_tflops) for DSv3.2.

    Fixed: topk=2048, page_block_size=64 (= _DECODE_DSV3_2_PAGE_BLOCK_SIZE),
    d_qk=576, d_v=512.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048
    page_block_size = 64
    # Pool ≫ L2 (~96 MB on SM120) so random topk indices land in DRAM.
    # 16384 × 64 × 656 B ≈ 688 MB.
    num_blocks = 16384
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    sm_scale = d_qk**-0.5
    runner = _SparseMLAPagedAttentionRunner(
        max_num_tokens=num_tokens,
        max_num_heads=num_heads,
        kv_scale_format="arbitrary_fp32",
        device=device,
    )

    def fn():
        runner.run(
            q,
            kv_packed,
            indices,
            output,
            sm_scale,
            attn_sink=attn_sink,
        )

    fn()
    torch.cuda.synchronize()
    measurements = bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=1000)
    ms = float(np.median(measurements))

    bpt = 656  # DSv3.2 INLINE per-token bytes
    kv_bytes = num_tokens * topk * bpt
    kv_bw_gbps = kv_bytes * 1e-6 / ms  # GB/s
    flops = 2 * num_tokens * num_heads * topk * (d_qk + d_v)
    tflops = flops * 1e-9 / ms

    return ms * 1e3, kv_bw_gbps, tflops


if __name__ == "__main__":
    if not is_sm120a_supported(torch.device("cuda")):
        raise SystemExit("Sparse-MLA SM120 requires sm120a.")

    # (num_heads, topk, num_tokens). T <= 64 hits decode; T > 64 hits prefill.
    decode_configs = [
        (16, 128, 1),
        (16, 128, 16),
        (16, 512, 16),
        (32, 128, 1),
        (32, 128, 16),
        (32, 512, 16),
        (64, 128, 1),
        (64, 128, 16),
        (64, 512, 16),
        (64, 1024, 16),
        (128, 128, 1),
        (128, 128, 16),
        (128, 512, 16),
        (128, 1024, 16),
        (128, 128, 32),
        (128, 512, 32),
    ]
    prefill_configs = [
        (64, 128, 128),
        (64, 512, 128),
        (128, 128, 128),
        (128, 512, 128),
        (64, 128, 512),
        (64, 512, 512),
        (128, 128, 512),
        (128, 512, 512),
        (128, 128, 1024),
        (128, 512, 1024),
    ]
    dual_prefill_configs = [
        # (num_heads, extra_topk, num_tokens, extra_page_block_size, extra_topk_length)
        (64, 512, 256, 64, None),
        (128, 512, 512, 64, None),
        (128, 512, 512, 2, None),
        (128, 768, 512, 2, None),
        (128, 1536, 512, 2, None),
        (128, 1664, 512, 2, None),
    ]

    header = (
        f"{'num_heads':>10}  {'topk':>6}  {'num_tokens':>11}  "
        f"{'lat (us)':>10}  {'kv BW (GB/s)':>13}  {'attn TFLOPs':>12}"
    )
    dual_header = (
        f"{'num_heads':>10}  {'extra_topk':>10}  {'num_tokens':>11}  "
        f"{'extra_pbs':>9}  {'extra_len':>9}  {'lat (us)':>10}  "
        f"{'kv BW (GB/s)':>13}  {'attn TFLOPs':>12}"
    )

    print("DSv4 decode-dsv4 path (num_tokens ≤ 64):")
    print(header)
    print("-" * len(header))
    for h, k, t in decode_configs:
        lat_us, kvbw, tfl = bench_sparse_mla_sm120(h, k, t)
        print(f"{h:>10}  {k:>6}  {t:>11}  {lat_us:>10.1f}  {kvbw:>13.1f}  {tfl:>12.2f}")

    print()
    print("DSv4 prefill path (num_tokens > 64):")
    print(header)
    print("-" * len(header))
    for h, k, t in prefill_configs:
        lat_us, kvbw, tfl = bench_sparse_mla_sm120(h, k, t)
        print(f"{h:>10}  {k:>6}  {t:>11}  {lat_us:>10.1f}  {kvbw:>13.1f}  {tfl:>12.2f}")

    print()
    print("DSv4 dual-cache prefill path (main topk=128 + secondary cache):")
    print(dual_header)
    print("-" * len(dual_header))
    for h, k_extra, t, extra_pbs, extra_len in dual_prefill_configs:
        lat_us, kvbw, tfl = bench_sparse_mla_sm120_dsv4_dual(
            h, k_extra, t, extra_pbs, extra_topk_length=extra_len
        )
        extra_len_label = "full" if extra_len is None else str(extra_len)
        print(
            f"{h:>10}  {k_extra:>10}  {t:>11}  {extra_pbs:>9}  "
            f"{extra_len_label:>9}  {lat_us:>10.1f}  {kvbw:>13.1f}  {tfl:>12.2f}"
        )

    # DSv3.2 decode-dsv3_2: topk fixed at 2048, page_block_size=1, 656 B/token.
    # Sweep is num_heads × num_tokens (with_sink off; sink is a per-head
    # epilogue scale, negligible on the bandwidth-bound critical path).
    dsv3_2_configs = [(h, t) for h in (8, 16, 32, 64, 128) for t in (1, 16, 32, 64)]

    print()
    print("DSv3.2 decode-dsv3_2 path (topk=2048, page_block_size=1):")
    print(header)
    print("-" * len(header))
    for h, t in dsv3_2_configs:
        lat_us, kvbw, tfl = bench_sparse_mla_sm120_dsv3_2(h, t)
        print(
            f"{h:>10}  {2048:>6}  {t:>11}  {lat_us:>10.1f}  {kvbw:>13.1f}  {tfl:>12.2f}"
        )
