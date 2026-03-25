"""
End-to-end benchmark for the DeepSeek v3 sparse-attention indexer.

Compares:
  - mqa_topk_indexer        : fused logits + histogram top-K (FlashInfer, SM100a)
  - mqa_topk_indexer_non_fused: separate logits + radix top-K (FlashInfer, SM100a)
  - deep_gemm reference     : fp8_paged_mqa_logits + flashinfer.top_k
                              (optional, requires deep_gemm)

Usage:
  python benchmarks/bench_dsv3_sparse_indexer.py
  python benchmarks/bench_dsv3_sparse_indexer.py --pdl
  python benchmarks/bench_dsv3_sparse_indexer.py --batch-sizes 1 4 64 --seq-lens 4096 32768
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.dsv3_ops import (
    get_mqa_metadata,
    mqa_topk_indexer,
)
from flashinfer.testing.utils import bench_gpu_time

# ---------------------------------------------------------------------------
# Optional deep_gemm import
# ---------------------------------------------------------------------------

try:
    import deep_gemm

    HAS_DEEP_GEMM = True
except ImportError:
    HAS_DEEP_GEMM = False


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------


def _make_kv_cache(num_pages: int) -> torch.Tensor:
    """Build a random FP8 KV cache [num_pages, 64, 1, 132] uint8."""
    cache = torch.empty(num_pages, 64, 1, 132, dtype=torch.uint8, device="cuda")
    flat = cache.view(num_pages, -1)
    flat[:, : 64 * 128].view(torch.float8_e4m3fn).copy_(
        torch.randn(num_pages, 64 * 128, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
    )
    flat[:, 64 * 128 :].view(torch.float32).copy_(
        torch.randn(num_pages, 64, dtype=torch.float32, device="cuda").abs()
    )
    return flat.view(num_pages, 64, 1, 132)


def _make_test_data(batch_size: int, seq_len: int):
    """Build fixed-length test data for the sparse indexer.

    Returns:
        (q, k_cache, weights, seq_lens, block_table)
        q:           [batch, 64, 128] float8_e4m3fn
        k_cache:     [num_pages, 64, 1, 132] uint8
        weights:     [batch, 64] float32
        seq_lens:    [batch] int32  (all equal to seq_len)
        block_table: [batch, max_num_pages] int32
    """
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages_per_seq = (seq_len + 64 - 1) // 64
    total_pages = batch_size * num_pages_per_seq
    # ensure max_num_pages is divisible by 2 (kernel requirement)
    max_num_pages = (num_pages_per_seq + 1) // 2 * 2

    q = torch.randn(batch_size, 64, 128, dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )
    k_cache = _make_kv_cache(total_pages)
    weights = torch.randn(batch_size, 64, dtype=torch.float32, device="cuda")

    block_table = torch.zeros(
        batch_size, max_num_pages, dtype=torch.int32, device="cuda"
    )
    for b in range(batch_size):
        start = b * num_pages_per_seq
        block_table[b, :num_pages_per_seq] = torch.arange(
            start, start + num_pages_per_seq, dtype=torch.int32, device="cuda"
        )

    return q, k_cache, weights, seq_lens, block_table


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def bench_dsv3_sparse_indexer(
    batch_size: int,
    seq_len: int,
    compare_deepgemm: bool = False,
) -> dict:
    """Benchmark the DSv3 sparse-attention indexer for one (batch, seq_len) point.

    Args:
        batch_size:       Number of sequences in the batch.
        seq_len:          Sequence length (uniform across the batch).
        pdl_enabled:      Enable PDL grid synchronisation in the fused kernel.
        compare_deepgemm: Also time deep_gemm fp8_paged_mqa_logits + flashinfer top_k.

    Returns:
        dict with keys:
          batch_size, seq_len,
          mqa_fused_us, mqa_non_fused_us,
          [deepgemm_ref_us, speedup_vs_deepgemm]
    """
    q, k_cache, weights, seq_lens, block_table = _make_test_data(batch_size, seq_len)
    max_model_len = block_table.shape[1] * 64
    sm_map = get_mqa_metadata(seq_lens)

    # for multiple kernels best use cuda graph with or without cupti
    enable_cupti = True
    use_cuda_graph = True

    # -- fused: logits + histogram top-K in one pass --------------------------
    measurements = bench_gpu_time(
        lambda: mqa_topk_indexer(
            q,
            k_cache,
            weights,
            seq_lens,
            block_table,
            sm_map=sm_map,
            max_model_len=max_model_len,
        ),
        enable_cupti=enable_cupti,
        use_cuda_graph=use_cuda_graph,
        dry_run_iters=10,
        repeat_iters=100,
    )
    fused_ms = np.median(measurements)

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mqa_fused_us": fused_ms * 1e3,
    }

    # -- deep_gemm reference: fp8_paged_mqa_logits + flashinfer top_k --------
    if compare_deepgemm and HAS_DEEP_GEMM:
        # deep_gemm expects q as [batch, num_next, 64, 128]; num_next=1 here
        q_dg = q.view(batch_size, 1, 64, 128).contiguous()
        # weights shape matches: [batch, 64] (num_next=1 so no repeat needed)
        meta_dg = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens, 64, deep_gemm.get_num_sms()
        )

        def _deepgemm_ref():
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_dg,
                k_cache,
                weights,
                seq_lens,
                block_table,
                meta_dg,
                seq_len,
                clean_logits=False,
            )
            flashinfer.top_k(logits, k=2048)

        measurements = bench_gpu_time(
            _deepgemm_ref,
            enable_cupti=enable_cupti,
            use_cuda_graph=use_cuda_graph,
            dry_run_iters=10,
            repeat_iters=100,
        )
        dg_ms = np.median(measurements)
        result["deepgemm_ref_us"] = dg_ms * 1e3
        result["speedup_vs_deepgemm"] = dg_ms / fused_ms

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek v3 sparse-attention indexer"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 32, 64],
        metavar="B",
        help="Batch sizes to benchmark (default: 1 2 4 8 32 64)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 32768, 40960],
        metavar="L",
        help="Sequence lengths to benchmark (default: 1024 4096 8192 32768 40960)",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("dsv3_sparse_indexer: DeepSeek v3 sparse-attention end-to-end (k=2048, fp8)")
    print("  mqa_fused:     FlashInfer fused logits + histogram top-K")
    print("  mqa_non_fused: FlashInfer separate logits + radix top-K")
    if HAS_DEEP_GEMM:
        print("  deepgemm_ref:  deep_gemm fp8_paged_mqa_logits + flashinfer top_k")
    print("=" * 100)

    if HAS_DEEP_GEMM:
        header = (
            f"{'batch':>6} {'seq_len':>10} |"
            f" {'mqa_fused':>12} {'mqa_non_fused':>15}"
            f" {'deepgemm_ref':>14} {'speedup':>10}"
        )
    else:
        header = (
            f"{'batch':>6} {'seq_len':>10} | {'mqa_fused':>12} {'mqa_non_fused':>15}"
        )
    print(header)
    print("-" * (60 if not HAS_DEEP_GEMM else 82))

    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            try:
                result = bench_dsv3_sparse_indexer(
                    batch_size,
                    seq_len,
                    compare_deepgemm=HAS_DEEP_GEMM,
                )
                line = (
                    f"{result['batch_size']:>6} {result['seq_len']:>10} |"
                    f" {result['mqa_fused_us']:>10.2f}us"
                    f" {result['mqa_non_fused_us']:>13.2f}us"
                )
                if "deepgemm_ref_us" in result:
                    line += (
                        f" {result['deepgemm_ref_us']:>12.2f}us"
                        f" {result['speedup_vs_deepgemm']:>9.2f}x"
                    )
                print(line)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{batch_size:>6} {seq_len:>10} | OOM")
                    torch.cuda.empty_cache()
                else:
                    raise


if __name__ == "__main__":
    main()
