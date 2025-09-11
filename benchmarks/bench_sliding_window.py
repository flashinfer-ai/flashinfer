#!/usr/bin/env python3
import itertools
from dataclasses import dataclass

import torch
from triton.testing import do_bench

# Optional: pin to a device via env CUDA_VISIBLE_DEVICES
DEVICE = torch.device("cuda:0")

import flashinfer


@dataclass
class Case:
    batch_size: int
    kv_len: int
    qo_len: int
    window_left: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    page_size: int


# Same grids as your pytest params
BATCH_SIZES = [16]
KV_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
QO_LENS = [1, 16]
WINDOW_LEFTS = [128]
NUM_KV_HEADS = [8]
NUM_QO_HEADS = [32]
HEAD_DIMS = [64]
PAGE_SIZES = [16]

DTYPE = torch.float16
INDEX_DTYPE = torch.int32


# --- FLOPs & bytes estimation helpers ----------------------------------------
def total_attn_pairs(kv_len: int, qo_len: int, window_left: int) -> int:
    total = 0
    for i in range(1, qo_len + 1):
        total += min(kv_len - i + 1, window_left)
    return total


def estimate_flops(
    batch: int,
    num_qo_heads: int,
    head_dim: int,
    kv_len: int,
    qo_len: int,
    window_left: int,
) -> float:
    """
    Matmul-only FLOPs (ignoring softmax) ~ QK^T + softmax@V ≈ 4 * head_dim * total_pairs
    """
    pairs = total_attn_pairs(kv_len, qo_len, window_left)
    return 4.0 * head_dim * pairs * batch * num_qo_heads


def estimate_min_io_bytes(
    batch: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    kv_len: int,
    qo_len: int,
    window_left: int,
    dtype_size: int = 2,
) -> int:
    """
    A *rough* lower-bound I/O estimate (no re-reads, no intermediate buffers):
      - Read Q: B * qo * Hq * D
      - Read unique K, V from KV cache: B * min(kv_len, window_left) * Hk * D each
      - Write O: B * qo * Hq * D
    NOTE: Real kernels may read more/less due to paging, softmax stats, and reuse.
    """
    uniq_kv = min(kv_len, window_left)
    q_bytes = batch * qo_len * num_qo_heads * head_dim * dtype_size
    k_bytes = batch * uniq_kv * num_kv_heads * head_dim * dtype_size
    v_bytes = batch * uniq_kv * num_kv_heads * head_dim * dtype_size
    o_bytes = batch * qo_len * num_qo_heads * head_dim * dtype_size
    return q_bytes + k_bytes + v_bytes + o_bytes


# --- Benchmark runner ---------------------------------------------------------
def run_one(case: Case, warmup=25, rep=100):
    torch.cuda.synchronize()
    # Inputs (mirror your test)
    q = torch.randn(
        case.batch_size * case.qo_len,
        case.num_qo_heads,
        case.head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )
    q_indptr = (
        torch.arange(0, case.batch_size + 1, device=DEVICE, dtype=INDEX_DTYPE)
        * case.qo_len
    )

    num_pages_per_seq = (case.kv_len + case.page_size - 1) // case.page_size
    total_num_pages = num_pages_per_seq * case.batch_size

    k_data = torch.randn(
        total_num_pages,
        case.page_size,
        case.num_kv_heads,
        case.head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )
    v_data = torch.randn(
        total_num_pages,
        case.page_size,
        case.num_kv_heads,
        case.head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )

    kv_indptr = (
        torch.arange(0, case.batch_size + 1, device=DEVICE, dtype=INDEX_DTYPE)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=DEVICE, dtype=INDEX_DTYPE)
    kv_last_page_len = torch.full(
        (case.batch_size,),
        (case.kv_len - 1) % case.page_size + 1,
        dtype=INDEX_DTYPE,
        device=DEVICE,
    )

    # Workspace & wrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )

    # Plan
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        case.num_qo_heads,
        case.num_kv_heads,
        case.head_dim,
        case.page_size,
        window_left=case.window_left,
        causal=False,
    )

    # Warmup + timing
    def _run():
        o = wrapper.run(q, (k_data, v_data))
        return o

    # Warmup (helps stabilize clocks/caches)
    for _ in range(warmup):
        _ = _run()
    torch.cuda.synchronize()

    ms = do_bench(_run, rep=rep)  # returns milliseconds per run
    torch.cuda.synchronize()

    # Metrics
    flops = estimate_flops(
        case.batch_size,
        case.num_qo_heads,
        case.head_dim,
        case.kv_len,
        case.qo_len,
        case.window_left,
    )
    bytes_min = estimate_min_io_bytes(
        case.batch_size,
        case.num_kv_heads,
        case.num_qo_heads,
        case.head_dim,
        case.kv_len,
        case.qo_len,
        case.window_left,
        dtype_size=2,
    )

    s = ms / 1e3
    tflops = (flops / s) / 1e12
    gbps = (bytes_min / s) / 1e9

    return ms, tflops, gbps


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    configs = [
        Case(b, kv, qo, w, hk, hq, d, ps)
        for b, qo, w, hk, hq, d, ps, kv in itertools.product(
            BATCH_SIZES,
            QO_LENS,
            WINDOW_LEFTS,
            NUM_KV_HEADS,
            NUM_QO_HEADS,
            HEAD_DIMS,
            PAGE_SIZES,
            KV_LENS,
        )
    ]

    # Header
    print(
        "batch kv_len qo_len win_left kv_heads qo_heads head_dim page_size | "
        "latency_ms  est_TFLOPs  est_GB/s(min-IO)"
    )
    print("-" * 110)

    for c in configs:
        try:
            ms, tflops, gbps = run_one(c)
            print(
                f"{c.batch_size:5d} {c.kv_len:6d} {c.qo_len:6d} {c.window_left:8d} "
                f"{c.num_kv_heads:8d} {c.num_qo_heads:8d} {c.head_dim:8d} {c.page_size:9d} | "
                f"{ms:10.3f} {tflops:10.3f} {gbps:15.3f}"
            )
        except Exception as e:
            # Keep going if some configs are unsupported by the current build
            print(
                f"{c.batch_size:5d} {c.kv_len:6d} {c.qo_len:6d} {c.window_left:8d} "
                f"{c.num_kv_heads:8d} {c.num_qo_heads:8d} {c.head_dim:8d} {c.page_size:9d} | "
                f"ERROR: {type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()
