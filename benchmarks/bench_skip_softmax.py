"""
Microbenchmark for trtllm_fmha_v2_prefill with skip-softmax.

Usage:
    python benchmarks/bench_skip_softmax.py
"""

import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from flashinfer.prefill import trtllm_fmha_v2_prefill
from flashinfer.testing import bench_gpu_time

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 1
SEQ_LEN = 20_000
NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
MASK_MODE = "CAUSAL"
WORKSPACE_BYTES = 256 * 1024 * 1024  # 256 MiB
NUM_ITERS = 50
# ─────────────────────────────────────────────────────────────────────────────


def compute_causal_tflops(seq_len, num_qo_heads, head_dim, time_ms):
    tri = seq_len * (seq_len + 1) / 2
    flops = 2 * tri * head_dim * 2 * num_qo_heads  # QK^T + PV
    return flops / (time_ms * 1e-3) / 1e12


PAGE_SIZE = 32


def build_inputs(layout, device):
    seq_lens = torch.full((BATCH_SIZE,), SEQ_LEN, dtype=torch.int32, device=device)
    cum_seq_lens = torch.zeros(BATCH_SIZE + 1, dtype=torch.int32, device=device)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    total_tokens = int(cum_seq_lens[-1].item())
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    block_tables = None

    if layout == "PACKED_QKV":
        # requires num_kv_heads == num_qo_heads
        packed = torch.randn(
            total_tokens, 3, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        qkv_arg = packed
    elif layout == "CONTIGUOUS_Q_KV":
        q = torch.randn(
            total_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        kv = torch.randn(
            total_tokens, 2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        qkv_arg = (q, kv)
    elif layout == "SEPARATE_Q_K_V":
        q = torch.randn(
            total_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        k = torch.randn(
            total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        v = torch.randn(
            total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        qkv_arg = (q, k, v)
    elif layout in ("Q_PAGED_KV_NHD", "Q_PAGED_KV_HND"):
        max_num_blocks = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages = BATCH_SIZE * max_num_blocks
        is_nhd = layout == "Q_PAGED_KV_NHD"
        paged_shape = (
            (num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
            if is_nhd
            else (num_pages, 2, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
        )
        q = torch.randn(
            total_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        paged_kv = torch.randn(*paged_shape, dtype=DTYPE, device=device)
        block_tables = torch.zeros(
            BATCH_SIZE, max_num_blocks, dtype=torch.int32, device=device
        )
        for i in range(BATCH_SIZE):
            n = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
            block_tables[i, :n] = torch.arange(
                i * max_num_blocks, i * max_num_blocks + n, device=device
            )
        qkv_arg = (q, paged_kv)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return qkv_arg, seq_lens, cum_seq_lens, workspace, sm_scale, block_tables


def run_bench(layout, skip_softmax_threshold_scale_factor, cold_cache, use_cupti):
    device = torch.device("cuda")
    qkv_arg, seq_lens, cum_seq_lens, workspace, sm_scale, block_tables = build_inputs(
        layout, device
    )

    def fn():
        trtllm_fmha_v2_prefill(
            qkv=qkv_arg,
            input_layout=layout,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=SEQ_LEN,
            max_kv_len=SEQ_LEN,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            batch_size=BATCH_SIZE,
            cum_seq_lens_q=cum_seq_lens,
            cum_seq_lens_kv=cum_seq_lens,
            block_tables=block_tables,
            mask_mode=MASK_MODE,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

    times = bench_gpu_time(
        fn,
        enable_cupti=use_cupti,
        repeat_iters=NUM_ITERS,
        cold_l2_cache=cold_cache,
    )
    median_ms = float(np.median(times))
    std_ms = float(np.std(times))
    tflops = compute_causal_tflops(SEQ_LEN, NUM_QO_HEADS, HEAD_DIM, median_ms)
    return median_ms, std_ms, tflops


def main():
    from flashinfer.utils import (
        is_sm90a_supported,
        is_sm100a_supported,
        is_sm120a_supported,
    )

    dev = torch.device("cuda")
    if (
        not is_sm90a_supported(dev)
        and not is_sm100a_supported(dev)
        and not is_sm120a_supported(dev)
    ):
        raise RuntimeError(
            "trtllm_fmha_v2_prefill requires SM90+, SM100+, or SM120+ GPU."
        )

    print(
        f"Config: batch={BATCH_SIZE}, seqlen={SEQ_LEN}, "
        f"qo_heads={NUM_QO_HEADS}, kv_heads={NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, dtype={DTYPE}\n"
    )

    tsf = 0.0  # baseline: no skip-softmax

    all_layouts = [
        "CONTIGUOUS_Q_KV",
        "SEPARATE_Q_K_V",
        "Q_PAGED_KV_NHD",
        "Q_PAGED_KV_HND",
    ]
    if NUM_KV_HEADS == NUM_QO_HEADS:
        all_layouts.insert(0, "PACKED_QKV")

    # Use cold_l2=True (conservative) and CUPTI (most accurate) for the sweep
    print(f"{'layout':>20}  {'median_ms':>10}  {'std_ms':>8}  {'TFLOPs':>8}")
    print("-" * 52)
    for layout in all_layouts:
        try:
            ms, std, tflops = run_bench(layout, tsf, cold_cache=True, use_cupti=True)
            print(f"{layout:>20}  {ms:>10.4f}  {std:>8.4f}  {tflops:>8.3f}")
        except Exception as e:
            print(f"{layout:>20}  ERROR: {e}")


if __name__ == "__main__":
    main()
