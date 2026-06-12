"""Isolated FlashInfer kernel benchmark at HunyuanImage-3.0 shapes.

Strips the Python wrappers (FlashInferLinear weight prep, online activation
quantization, MoE Python loop) out of the loop — measures only the kernel
plus the minimal scale/cast that the kernel API requires.

Reference baseline for GEMM is ``torch.matmul(bf16, bf16) -> bf16``, the
same kernel cuBLAS picks for ``nn.Linear``. Reference for RMSNorm is the
upstream FP32-reduction PyTorch implementation. Reference for SwiGLU is
``silu(gate) * up`` in eager mode. Reference for MoE is the upstream
per-expert loop (HunyuanMoE's ``moe_impl='eager'`` path). Reference for
attention is ``torch.nn.functional.scaled_dot_product_attention``.

Shapes are HunyuanImage-3 canonical (config.json):
  hidden_size = 4096, intermediate_size (per expert) = 3072
  num_attention_heads = 32, num_key_value_heads = 8, head_dim = 128
  num_experts = 64, moe_topk = 8

Run on a single CUDA device. Output: per-kernel ms + speedup vs baseline.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import torch

import flashinfer

# ---- FlashInfer symbol resolution before any local-name shadowing -----------
_mm_bf16 = flashinfer.mm_bf16
_mm_fp8 = flashinfer.mm_fp8
_mm_fp4 = flashinfer.mm_fp4
_mm_mxfp8 = flashinfer.mm_mxfp8
_bmm_fp8 = flashinfer.bmm_fp8
_bmm_mxfp8 = flashinfer.bmm_mxfp8
_nvfp4_quantize = flashinfer.nvfp4_quantize
_mxfp8_quantize = flashinfer.mxfp8_quantize
_prepare_low_latency_gemm_weights = flashinfer.prepare_low_latency_gemm_weights
_SfLayout = flashinfer.SfLayout

_rmsnorm = flashinfer.rmsnorm
_silu_and_mul = flashinfer.silu_and_mul

from flashinfer.gemm import (  # noqa: E402
    gemm_fp8_nt_groupwise,
    gemm_fp8_nt_blockscaled,
)
from flashinfer.prefill import single_prefill_with_kv_cache  # noqa: E402
from flashinfer.decode import single_decode_with_kv_cache  # noqa: E402
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache  # noqa: E402
from flashinfer.fused_moe import cutlass_fused_moe  # noqa: E402


# HunyuanImage-3 architectural constants (from the published config.json).
HIDDEN = 4096
INTERMEDIATE = 3072          # per-expert (SwiGLU branch width)
NUM_HEADS = 32               # query heads
NUM_KV_HEADS = 8             # GQA
HEAD_DIM = 128
NUM_EXPERTS = 64
TOPK = 8
RMS_EPS = 1e-5

WARMUP = 3
ITERS = 20


# ----------------------------------------------------------------------------
# Timing helper
# ----------------------------------------------------------------------------


def bench(fn: Callable[[], None], label: str, ms_baseline: float | None = None) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    speedup = f"  ({ms_baseline / ms:.2f}x vs baseline)" if ms_baseline else ""
    print(f"  {label:42s}: {ms:8.4f} ms{speedup}")
    return ms


def _maybe_warn_unsupported(name: str, exc: BaseException) -> None:
    print(f"  {name:42s}: SKIPPED ({type(exc).__name__}: {str(exc)[:90]})")


# ----------------------------------------------------------------------------
# Section A. GEMM at HunyuanImage-3 Linear shapes
# ----------------------------------------------------------------------------

# (label, M, K, N) for A=(M,K), B=(N,K). Output (M,N).
# Each backend JIT-compiles per (M, K, N) on first call. To keep the wall
# time tractable on B200 (where CUTLASS-DSL kernels JIT for several minutes
# each), test only one representative GEMM shape and one sequence length.
# This still covers all backends; per-shape variation can be inferred from
# the wan benchmark which has the same kernels at different (M, K, N).
GEMM_SHAPES = [
    # MoE / MLP expert shapes
    ("mlp gate_and_up  (4096->6144)", None, HIDDEN, 2 * INTERMEDIATE),
    ("mlp down         (3072->4096)", None, INTERMEDIATE, HIDDEN),
    # Attention projections (qkv fused = num_q*hd + 2*num_kv*hd = 4096+2*1024 = 6144).
    ("attn qkv_proj    (4096->6144)", None, HIDDEN, HIDDEN + 2 * NUM_KV_HEADS * HEAD_DIM),
    ("attn o_proj      (4096->4096)", None, HIDDEN, HIDDEN),
]

# Wider M sweep used for the cheap kernel sections (RMSNorm/SwiGLU/MoE/attn).
M_VALUES = [1024, 4096]
# Narrower M sweep used for GEMM (where each (M,K,N) triggers a separate JIT).
GEMM_M_VALUES = [4096]


def make_finfo_fp8():
    return torch.finfo(torch.float8_e4m3fn)


def bench_gemm(args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print("Section A. GEMM at HunyuanImage-3 Linear shapes")
    print("=" * 72)

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max

    for tag, _, K, N in GEMM_SHAPES:
        for M in GEMM_M_VALUES:
            print(f"\n--- {tag}  M={M} K={K} N={N} ---")
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

            # 0. baseline
            ms_base = bench(lambda: torch.matmul(a, b.T), "torch.matmul bf16 (baseline)")

            # 1. mm_bf16 — same precision, FlashInfer kernel
            try:
                b_T = b.T.contiguous()
                bench(lambda: _mm_bf16(a, b_T), "mm_bf16", ms_base)
            except Exception as e:
                _maybe_warn_unsupported("mm_bf16", e)

            # 2. mm_fp8 (per-tensor, TRT-LLM low-latency)
            try:
                b_amax = b.abs().max().clamp(min=1e-12)
                b_scale = fp8_max / b_amax
                b_fp8 = (b * b_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
                b_fp8_prepared = _prepare_low_latency_gemm_weights(b_fp8, {})
                b_inv_scale = (1.0 / b_scale).to(torch.float32)

                # offline activation scale = 1.0 (no per-call amax)
                a_scale_offline = torch.tensor(1.0, device="cuda", dtype=torch.float32)

                def _fp8_offline():
                    a_fp8 = (a * a_scale_offline).clamp(finfo.min, finfo.max).to(
                        torch.float8_e4m3fn
                    )
                    alpha = (1.0 / a_scale_offline) * b_inv_scale
                    _mm_fp8(a_fp8, b_fp8_prepared, alpha=alpha)
                bench(_fp8_offline, "mm_fp8 offline-actquant", ms_base)

                # online activation scale = fp8_max / amax(a)
                def _fp8_online():
                    a_amax = a.abs().max().clamp(min=1e-12)
                    a_scale = fp8_max / a_amax
                    a_fp8 = (a * a_scale).clamp(finfo.min, finfo.max).to(
                        torch.float8_e4m3fn
                    )
                    alpha = (1.0 / a_scale) * b_inv_scale
                    _mm_fp8(a_fp8, b_fp8_prepared, alpha=alpha)
                bench(_fp8_online, "mm_fp8 online-actquant", ms_base)
            except Exception as e:
                _maybe_warn_unsupported("mm_fp8", e)

            # 3. fp8_blockscale_gemm_sm90 (SM90). Accepts BF16 input directly
            # with internal quantization when ``input_scale=None``.
            try:
                from flashinfer.gemm import fp8_blockscale_gemm_sm90

                # Pad weight to multiples of 128 for 128x128 block quantization.
                n_pad = ((N + 127) // 128) * 128
                k_pad = ((K + 127) // 128) * 128
                b_padded = torch.zeros(n_pad, k_pad, dtype=b.dtype, device=b.device)
                b_padded[:N, :K] = b
                bv = b_padded.view(n_pad // 128, 128, k_pad // 128, 128)
                bsf = (bv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
                       / fp8_max)
                w_fp8 = (bv / bsf).to(torch.float8_e4m3fn).view(n_pad, k_pad)[:N, :K].contiguous()
                w_scale = bsf.squeeze(1).squeeze(-1)[
                    : (N + 127) // 128, : (K + 127) // 128
                ].contiguous().to(torch.float32)

                # The kernel does its own activation quantization when
                # input_scale=None; this is the "online" mode of the wrapper.
                def _fp8_sm90():
                    fp8_blockscale_gemm_sm90(
                        a, w_fp8,
                        input_scale=None,
                        weight_scale=w_scale,
                        out_dtype=torch.bfloat16,
                    )
                bench(_fp8_sm90, "fp8_blockscale_gemm_sm90 online", ms_base)
                # offline mode: pretend amax = fp8_max -> scale = 1/fp8_max
                # for every (M_block, K_block). Note for sm90 the activation
                # scale layout is (m_blocks, k_blocks).
                m_pad = ((M + 127) // 128) * 128
                a_scale_offline_blocks = torch.full(
                    ((M + 127) // 128, (K + 127) // 128),
                    1.0 / fp8_max,
                    device=a.device, dtype=torch.float32,
                )
                a_padded = torch.zeros(m_pad, k_pad, dtype=a.dtype, device=a.device)
                a_padded[:M, :K] = a
                a_fp8_offline = (a_padded * fp8_max).clamp(-fp8_max, fp8_max).to(
                    torch.float8_e4m3fn
                )[:M, :K].contiguous()

                def _fp8_sm90_offline():
                    fp8_blockscale_gemm_sm90(
                        a_fp8_offline, w_fp8,
                        input_scale=a_scale_offline_blocks,
                        weight_scale=w_scale,
                        out_dtype=torch.bfloat16,
                    )
                bench(_fp8_sm90_offline, "fp8_blockscale_gemm_sm90 offline", ms_base)
            except Exception as e:
                _maybe_warn_unsupported("fp8_sm90", e)

            # 4. gemm_fp8_nt_groupwise (Blackwell SM100+, uses CUTLASS Python DSL).
            # Pre-quantize with constant scales (== offline mode) outside the
            # timed loop, wan-bench-style.
            if args.skip_cutlass_dsl:
                print(f"  {'gemm_fp8_nt_groupwise (kernel only)':42s}: "
                      "SKIPPED (--skip-cutlass-dsl)")
            else:
                try:
                    block_size = 128
                    if K % block_size != 0 or N % block_size != 0 or M % block_size != 0:
                        raise ValueError(
                            f"fp8_groupwise needs K/N/M divisible by {block_size}; "
                            f"got M={M}, K={K}, N={N}"
                        )
                    a_fp8_gw = (a * fp8_max).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                    b_fp8_gw = (b * fp8_max).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                    a_scale_gw = torch.full(
                        (M, K // block_size), 1.0 / fp8_max,
                        device=a.device, dtype=torch.float32,
                    ).t().contiguous()
                    b_scale_gw = torch.full(
                        (K // block_size, N // block_size), 1.0 / fp8_max,
                        device=a.device, dtype=torch.float32,
                    )

                    def _fp8_gw():
                        gemm_fp8_nt_groupwise(
                            a_fp8_gw, b_fp8_gw, a_scale_gw, b_scale_gw,
                            scale_major_mode="MN",
                            out_dtype=torch.bfloat16,
                        )
                    bench(_fp8_gw, "gemm_fp8_nt_groupwise (kernel only)", ms_base)
                except Exception as e:
                    _maybe_warn_unsupported("fp8_groupwise", e)

            # 5. mm_fp4 (NVFP4, SM100+). Uses CUTLASS DSL.
            if args.skip_cutlass_dsl:
                print(f"  {'mm_fp4 / NVFP4 (kernel only)':42s}: "
                      "SKIPPED (--skip-cutlass-dsl)")
            else:
                try:
                    a_sf = ((448.0 * 6.0) / a.float().abs().nan_to_num().max().clamp(min=1e-12)).to(torch.float32)
                    w_sf = ((448.0 * 6.0) / b.float().abs().nan_to_num().max().clamp(min=1e-12)).to(torch.float32)
                    a_fp4, a_descale = _nvfp4_quantize(a, a_sf)
                    w_fp4, w_descale = _nvfp4_quantize(b, w_sf)
                    alpha_fp4 = (1.0 / (a_sf * w_sf)).to(torch.float32)
                    bench(
                        lambda: _mm_fp4(
                            a_fp4, w_fp4.T, a_descale, w_descale.T,
                            alpha=alpha_fp4, out_dtype=torch.bfloat16,
                        ),
                        "mm_fp4 / NVFP4 (kernel only)", ms_base,
                    )
                except Exception as e:
                    _maybe_warn_unsupported("mm_fp4", e)

            # 6. mm_mxfp8 (SM100+). Uses CUTLASS DSL.
            if args.skip_cutlass_dsl:
                print(f"  {'mm_mxfp8 (kernel only)':42s}: SKIPPED (--skip-cutlass-dsl)")
            else:
                try:
                    a_mx, a_sf_mx = _mxfp8_quantize(a.contiguous(), is_sf_swizzled_layout=True)
                    w_mx, w_sf_mx = _mxfp8_quantize(b.contiguous(), is_sf_swizzled_layout=True)
                    bench(
                        lambda: _mm_mxfp8(a_mx, w_mx.T, a_sf_mx, w_sf_mx, out_dtype=torch.bfloat16),
                        "mm_mxfp8 (kernel only)", ms_base,
                    )
                except Exception as e:
                    _maybe_warn_unsupported("mm_mxfp8", e)


# ----------------------------------------------------------------------------
# Section B. RMSNorm at hidden_size=4096
# ----------------------------------------------------------------------------


def bench_rmsnorm() -> None:
    print("\n" + "=" * 72)
    print(f"Section B. RMSNorm at hidden_size={HIDDEN}")
    print("=" * 72)

    for M in M_VALUES:
        print(f"\n--- M={M} N={HIDDEN} ---")
        x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)

        # Baseline: upstream HunyuanRMSNorm (FP32 reduction).
        def _torch_rmsnorm():
            x32 = x.to(torch.float32)
            var = x32.pow(2).mean(-1, keepdim=True)
            y = x32 * torch.rsqrt(var + RMS_EPS)
            return (w * y.to(x.dtype))
        ms_base = bench(_torch_rmsnorm, "torch fp32-reduction rmsnorm (baseline)")

        def _flashinfer_rmsnorm():
            _rmsnorm(x.contiguous(), w, RMS_EPS)
        bench(_flashinfer_rmsnorm, "flashinfer.rmsnorm", ms_base)


# ----------------------------------------------------------------------------
# Section C. SwiGLU activation (silu + mul)
# ----------------------------------------------------------------------------


def bench_swiglu() -> None:
    print("\n" + "=" * 72)
    print(f"Section C. SwiGLU activation at intermediate_size={INTERMEDIATE}")
    print("=" * 72)

    for M in M_VALUES:
        print(f"\n--- M={M} 2*N={2 * INTERMEDIATE} ---")
        x = torch.randn(M, 2 * INTERMEDIATE, device="cuda", dtype=torch.bfloat16)

        def _torch_swiglu():
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up
        ms_base = bench(_torch_swiglu, "torch silu(gate)*up (baseline)")

        def _flashinfer_swiglu():
            _silu_and_mul(x.contiguous())
        bench(_flashinfer_swiglu, "flashinfer.silu_and_mul", ms_base)


# ----------------------------------------------------------------------------
# Section D. Fused MoE (top-8 of 64, shared expert excluded)
# ----------------------------------------------------------------------------


def bench_moe() -> None:
    print("\n" + "=" * 72)
    print(f"Section D. Fused MoE  (num_experts={NUM_EXPERTS}, topk={TOPK}, "
          f"intermediate={INTERMEDIATE})")
    print("=" * 72)

    # Active tokens per MoE forward: M sequence positions * 1 routed pass.
    for M in [1024, 4096]:
        print(f"\n--- M={M} routed tokens ---")
        x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)

        # Per-expert weights stacked (num_experts, ...) for the cutlass path.
        w_gate_up = torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN,
            device="cuda", dtype=torch.bfloat16,
        )
        w_down = torch.randn(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE,
            device="cuda", dtype=torch.bfloat16,
        )

        # Synthetic routing weights for benchmarking only.
        gate_logits = torch.randn(M, NUM_EXPERTS, device="cuda", dtype=torch.float32)
        gates = torch.nn.functional.softmax(gate_logits, dim=-1)
        topk_w_raw, topk_idx = torch.topk(gates, TOPK, dim=-1)
        topk_w = (topk_w_raw / topk_w_raw.sum(dim=-1, keepdim=True)).to(torch.float32).contiguous()
        topk_idx_i32 = topk_idx.to(torch.int32).contiguous()

        # Baseline: per-expert eager loop (matches HunyuanMoE moe_impl='eager').
        def _torch_moe():
            x_rep = x.repeat_interleave(TOPK, dim=0)
            flat_idx = topk_idx.view(-1)
            out = torch.zeros_like(x_rep)
            for e in range(NUM_EXPERTS):
                mask = (flat_idx == e)
                if not mask.any():
                    continue
                xe = x_rep[mask]
                up_gate = torch.nn.functional.linear(xe, w_gate_up[e])
                g, u = up_gate.chunk(2, dim=-1)
                h = torch.nn.functional.silu(g) * u
                out[mask] = torch.nn.functional.linear(h, w_down[e])
            combined = (out.view(M, TOPK, HIDDEN) * topk_w_raw.unsqueeze(-1)).sum(dim=1)
            return combined

        ms_base = bench(_torch_moe, "torch eager top-8/64 MoE (baseline)")

        # FlashInfer fused MoE.
        try:
            out_buf = torch.empty_like(x)

            def _fi_moe():
                cutlass_fused_moe(
                    x.contiguous(), topk_idx_i32, topk_w,
                    w_gate_up, w_down,
                    torch.bfloat16, output=out_buf, quant_scales=None,
                )
            bench(_fi_moe, "flashinfer cutlass_fused_moe bf16", ms_base)
        except Exception as e:
            _maybe_warn_unsupported("cutlass_fused_moe bf16", e)


# ----------------------------------------------------------------------------
# Section E. GQA attention (32 Q heads, 8 KV heads, head_dim=128)
# ----------------------------------------------------------------------------


def bench_attention() -> None:
    print("\n" + "=" * 72)
    print(f"Section E. GQA attention "
          f"(num_q={NUM_HEADS}, num_kv={NUM_KV_HEADS}, head_dim={HEAD_DIM})")
    print("=" * 72)

    # Prefill shapes: q_len == kv_len.
    for seq in [1024, 4096]:
        print(f"\n--- prefill: seq_q = seq_kv = {seq} ---")
        q = torch.randn(seq, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        # GQA: pre-expand KV to num_heads for SDPA baseline; pass the
        # un-expanded form to FlashInfer (which doesn't repeat internally for
        # single_prefill — caller must repeat).
        k_kv = torch.randn(seq, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v_kv = torch.randn(seq, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k_expanded = k_kv.unsqueeze(2).expand(seq, NUM_KV_HEADS, n_rep, HEAD_DIM).reshape(seq, NUM_HEADS, HEAD_DIM).contiguous()
        v_expanded = v_kv.unsqueeze(2).expand(seq, NUM_KV_HEADS, n_rep, HEAD_DIM).reshape(seq, NUM_HEADS, HEAD_DIM).contiguous()

        # Baseline: SDPA in BHSD layout.
        q_bhsd = q.transpose(0, 1).unsqueeze(0)            # (1, H, S, D)
        k_bhsd = k_expanded.transpose(0, 1).unsqueeze(0)
        v_bhsd = v_expanded.transpose(0, 1).unsqueeze(0)

        def _sdpa_causal():
            return torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=True,
            )
        ms_base = bench(_sdpa_causal, "torch SDPA causal (baseline)")

        try:
            def _fi_single_prefill_causal():
                single_prefill_with_kv_cache(q, k_expanded, v_expanded, causal=True)
            bench(_fi_single_prefill_causal, "single_prefill_with_kv_cache causal", ms_base)
        except Exception as e:
            _maybe_warn_unsupported("single_prefill_with_kv_cache", e)

        try:
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            actual_q = torch.full((1, 1, 1, 1), seq, dtype=torch.int32, device="cuda")
            actual_kv = torch.full((1, 1, 1, 1), seq, dtype=torch.int32, device="cuda")
            offsets_q = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
            offsets_kv = torch.tensor([0, seq], dtype=torch.int32, device="cuda")

            def _fi_cudnn():
                cudnn_batch_prefill_with_kv_cache(
                    q=q, k_cache=k_expanded, v_cache=v_expanded,
                    scale=1.0 / math.sqrt(HEAD_DIM),
                    workspace_buffer=workspace,
                    max_token_per_sequence=seq, max_sequence_kv=seq,
                    actual_seq_lens_q=actual_q, actual_seq_lens_kv=actual_kv,
                    causal=True, return_lse=False,
                    batch_offsets_q=offsets_q, batch_offsets_o=offsets_q,
                    batch_offsets_k=offsets_kv, batch_offsets_v=offsets_kv,
                )
            bench(_fi_cudnn, "cudnn_batch_prefill_with_kv_cache causal", ms_base)
        except Exception as e:
            _maybe_warn_unsupported("cudnn_batch_prefill_with_kv_cache", e)

    # Decode: q_len=1 against a context of kv_len.
    for kv_len in [1024, 4096]:
        print(f"\n--- decode: q_len=1, kv_len={kv_len} ---")
        q1 = torch.randn(NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        vv = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k_exp = kv.unsqueeze(2).expand(kv_len, NUM_KV_HEADS, n_rep, HEAD_DIM).reshape(kv_len, NUM_HEADS, HEAD_DIM).contiguous()
        v_exp = vv.unsqueeze(2).expand(kv_len, NUM_KV_HEADS, n_rep, HEAD_DIM).reshape(kv_len, NUM_HEADS, HEAD_DIM).contiguous()

        q_bhsd = q1.unsqueeze(0).unsqueeze(2)              # (1, H, 1, D)
        k_bhsd = k_exp.transpose(0, 1).unsqueeze(0)
        v_bhsd = v_exp.transpose(0, 1).unsqueeze(0)

        def _sdpa_decode():
            return torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=False,
            )
        ms_base = bench(_sdpa_decode, "torch SDPA decode (baseline)")

        try:
            def _fi_single_decode():
                single_decode_with_kv_cache(q1, k_exp, v_exp)
            bench(_fi_single_decode, "single_decode_with_kv_cache", ms_base)
        except Exception as e:
            _maybe_warn_unsupported("single_decode_with_kv_cache", e)


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", default="all",
                   help="Comma-separated subset of {gemm,rmsnorm,swiglu,moe,attn}; default 'all'.")
    p.add_argument(
        "--skip-cutlass-dsl", action="store_true",
        help=(
            "Skip backends that need the CUTLASS Python DSL "
            "(gemm_fp8_nt_groupwise, mm_fp4, mm_mxfp8). On the "
            "nvcr.io pytorch:26.03 + flashinfer 0.6.7 stack we hit a JIT-compile "
            "hang inside the DSL on B200 even after the recommended "
            "nvidia-cutlass-dsl force-reinstall. Use this flag to get the rest "
            "of the bench numbers; the skipped paths are known good in the wan "
            "bench (different shapes)."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[bench] python={sys.version.split()[0]} torch={torch.__version__} "
          f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    print(f"[bench] device={torch.cuda.get_device_name(dev)} cc={cc[0]}.{cc[1]}")
    print(f"[bench] warmup={WARMUP} iters={ITERS}")

    sections = {s.strip() for s in args.sections.split(",")}
    if "all" in sections or "gemm" in sections:
        bench_gemm(args)
    if "all" in sections or "rmsnorm" in sections:
        bench_rmsnorm()
    if "all" in sections or "swiglu" in sections:
        bench_swiglu()
    if "all" in sections or "moe" in sections:
        bench_moe()
    if "all" in sections or "attn" in sections:
        bench_attention()


if __name__ == "__main__":
    main()
