"""
Microbenchmark comparing:
  - flash_attn_varlen_func  (FA2/FA3 from the flash_attn package)
  - FlashInfer BatchPrefillWithRaggedKVCacheWrapper, backend='fa3'

Usage:
    python benchmarks/bench_fa3.py
"""

import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
import flashinfer
from flashinfer.testing import bench_gpu_time

FA3_HOPPER_PATH = "/scratch/projects/flash-attention/hopper"
FA3_LIB_PATH = "/usr/local/lib/python3.12/dist-packages/torch/lib"

# ── Config (matches bench_skip_softmax.py) ────────────────────────────────────
BATCH_SIZE = 1
SEQ_LEN = 20_000
NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
NUM_ITERS = 50
WORKSPACE_BYTES = 128 * 1024 * 1024
# ─────────────────────────────────────────────────────────────────────────────


def compute_causal_tflops(seq_len, num_qo_heads, head_dim, time_ms):
    tri = seq_len * (seq_len + 1) / 2
    flops = 2 * tri * head_dim * 2 * num_qo_heads  # QK^T + PV
    return flops / (time_ms * 1e-3) / 1e12


def run(label, fn):
    times = bench_gpu_time(
        fn, enable_cupti=True, repeat_iters=NUM_ITERS, cold_l2_cache=True
    )
    ms = float(np.median(times))
    std = float(np.std(times))
    tflops = compute_causal_tflops(SEQ_LEN, NUM_QO_HEADS, HEAD_DIM, ms)
    print(f"{label:>40}  {ms:>10.4f}  {std:>8.4f}  {tflops:>8.3f}")


def main():
    device = torch.device("cuda")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    total_tokens = BATCH_SIZE * SEQ_LEN

    print(
        f"Config: batch={BATCH_SIZE}, seqlen={SEQ_LEN}, "
        f"qo_heads={NUM_QO_HEADS}, kv_heads={NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, dtype={DTYPE}\n"
    )
    # Load FA3 from Tri Dao's hopper directory
    import ctypes
    import importlib.util

    ctypes.CDLL(f"{FA3_LIB_PATH}/libc10.so")
    ctypes.CDLL(f"{FA3_LIB_PATH}/libtorch_cpu.so")
    ctypes.CDLL(f"{FA3_LIB_PATH}/libtorch_cuda.so")
    spec = importlib.util.spec_from_file_location(
        "fa3_interface", f"{FA3_HOPPER_PATH}/flash_attn_interface.py"
    )
    fa3_mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, FA3_HOPPER_PATH)
    spec.loader.exec_module(fa3_mod)
    fa3_varlen_func = fa3_mod.flash_attn_varlen_func

    print(f"{'kernel':>40}  {'median_ms':>10}  {'std_ms':>8}  {'TFLOPs':>8}")
    print("-" * 72)

    # ── flash_attn_varlen_func ────────────────────────────────────────────────
    q_fa = torch.randn(total_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k_fa = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v_fa = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    cu_fa = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN, SEQ_LEN, dtype=torch.int32, device=device
    )

    # ── Tri Dao FA3 (hopper) ─────────────────────────────────────────────────
    run(
        "Tri Dao FA3 (flash_attn_3)",
        lambda: fa3_varlen_func(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=cu_fa,
            cu_seqlens_k=cu_fa,
            max_seqlen_q=SEQ_LEN,
            max_seqlen_k=SEQ_LEN,
            softmax_scale=sm_scale,
            causal=True,
        ),
    )

    run(
        "flash_attn_varlen_func (FA2 pkg)",
        lambda: flash_attn_varlen_func(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=cu_fa,
            cu_seqlens_k=cu_fa,
            max_seqlen_q=SEQ_LEN,
            max_seqlen_k=SEQ_LEN,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=True,
        ),
    )

    # ── FlashInfer FA3 backend ────────────────────────────────────────────────
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    q_fi = torch.randn(total_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k_fi = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v_fi = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    qo_indptr = torch.arange(
        0, (BATCH_SIZE + 1) * SEQ_LEN, SEQ_LEN, dtype=torch.int32, device=device
    )
    kv_indptr = qo_indptr.clone()

    for backend in ["fa3", "fa2"]:
        try:
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, backend=backend
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads=NUM_QO_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim_qk=HEAD_DIM,
                head_dim_vo=HEAD_DIM,
                causal=True,
                q_data_type=DTYPE,
            )
            run(
                f"flashinfer BatchPrefill ({backend})",
                lambda: wrapper.run(q_fi, k_fi, v_fi),
            )
        except Exception as e:
            print(f"{'flashinfer BatchPrefill (' + backend + ')':>40}  ERROR: {e}")


if __name__ == "__main__":
    main()
