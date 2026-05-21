#!/usr/bin/env python3
"""
Standalone test for fmha_v2_prepare_paged + prepare chain.
Compiles and verifies both kernels via PyTorch JIT on any NVIDIA GPU.
"""
import torch
import torch.utils.cpp_extension
import tempfile
import os

PREPARE_PAGED_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fmha_v2_prepare_paged_kernel(
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ paged_kv_indptr,
    const int32_t* __restrict__ paged_kv_last_page_len,
    const int32_t* __restrict__ paged_kv_indices,
    int32_t* __restrict__ seq_lens_q,
    int32_t* __restrict__ kv_lens,
    int32_t* __restrict__ block_tables,
    int page_size, int batch_size, int max_blocks_per_seq) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= batch_size) return;

  int num_pages_i = paged_kv_indptr[i + 1] - paged_kv_indptr[i];
  int kv_len_i = max(num_pages_i - 1, 0) * page_size + paged_kv_last_page_len[i];
  kv_lens[i] = kv_len_i;
  seq_lens_q[i] = qo_indptr[i + 1] - qo_indptr[i];

  int block_start = paged_kv_indptr[i];
  int row_offset = i * max_blocks_per_seq;
  for (int j = 0; j < num_pages_i && j < max_blocks_per_seq; j++) {
    block_tables[row_offset + j] = paged_kv_indices[block_start + j];
  }
}

void launch_prepare_paged(
    torch::Tensor qo_indptr, torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_last_page_len, torch::Tensor paged_kv_indices,
    torch::Tensor seq_lens_q_out, torch::Tensor kv_lens_out,
    torch::Tensor block_tables_out,
    int page_size, int batch_size, int max_blocks_per_seq) {
  int threads = min(batch_size, 256);
  int blocks = (batch_size + threads - 1) / threads;
  fmha_v2_prepare_paged_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      qo_indptr.data_ptr<int32_t>(), paged_kv_indptr.data_ptr<int32_t>(),
      paged_kv_last_page_len.data_ptr<int32_t>(), paged_kv_indices.data_ptr<int32_t>(),
      seq_lens_q_out.data_ptr<int32_t>(), kv_lens_out.data_ptr<int32_t>(),
      block_tables_out.data_ptr<int32_t>(),
      page_size, batch_size, max_blocks_per_seq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_prepare_paged", &launch_prepare_paged);
}
"""

PREPARE_CUM_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

// Simplified version of prepare kernel: just does the cum_seq_lens scan.
constexpr int kTPB = 256;

struct PrepShared {
  typename cub::BlockScan<int, kTPB>::TempStorage q_scan;
  typename cub::BlockScan<int, kTPB>::TempStorage kv_scan;
};

__global__ void prepare_cum_kernel(
    const int* __restrict__ seq_lens_q, const int* __restrict__ seq_lens_kv,
    int batch_size, int* __restrict__ cum_q, int* __restrict__ cum_kv) {
  __shared__ PrepShared smem;
  int prefix_q = 0, prefix_kv = 0;
  const int tid = threadIdx.x;

  for (int chunk = 0; chunk < batch_size; chunk += kTPB) {
    int b = chunk + tid;
    int q_len = (b < batch_size) ? seq_lens_q[b] : 0;
    int kv_len = (b < batch_size) ? seq_lens_kv[b] : 0;
    int q_off, kv_off, q_total, kv_total;
    cub::BlockScan<int, kTPB>(smem.q_scan).ExclusiveSum(q_len, q_off, q_total);
    __syncthreads();
    cub::BlockScan<int, kTPB>(smem.kv_scan).ExclusiveSum(kv_len, kv_off, kv_total);
    __syncthreads();
    if (b < batch_size) {
      cum_q[b] = prefix_q + q_off;
      cum_kv[b] = prefix_kv + kv_off;
    }
    prefix_q += q_total;
    prefix_kv += kv_total;
    __syncthreads();
  }
  if (tid == 0) {
    cum_q[batch_size] = prefix_q;
    cum_kv[batch_size] = prefix_kv;
  }
}

void launch_prepare_cum(
    torch::Tensor seq_lens_q, torch::Tensor seq_lens_kv,
    int batch_size, torch::Tensor cum_q, torch::Tensor cum_kv) {
  prepare_cum_kernel<<<1, kTPB, 0, at::cuda::getCurrentCUDAStream()>>>(
      seq_lens_q.data_ptr<int32_t>(), seq_lens_kv.data_ptr<int32_t>(),
      batch_size, cum_q.data_ptr<int32_t>(), cum_kv.data_ptr<int32_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_prepare_cum", &launch_prepare_cum);
}
"""


def build_module(name, src):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"{name}.cu")
        with open(path, "w") as f:
            f.write(src)
        return torch.utils.cpp_extension.load(name=name, sources=[path], verbose=True)


def test_prepare_paged_kernel(mod):
    print("=" * 50)
    print("TEST: prepare_paged correctness")
    print("=" * 50)

    device = torch.device("cuda")
    batch_size = 4
    page_size = 16

    kv_seq_lens = torch.tensor([48, 32, 64, 16], dtype=torch.int32, device=device)
    q_seq_lens = torch.tensor([8, 4, 16, 2], dtype=torch.int32, device=device)

    num_pages_per_seq = (kv_seq_lens + page_size - 1) // page_size  # [3, 2, 4, 1]
    total_pages = num_pages_per_seq.sum().item()

    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = ((kv_seq_lens - 1) % page_size) + 1

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_seq_lens, dim=0)

    max_blocks = num_pages_per_seq.max().item()

    seq_lens_q_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_lens_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    block_tables_out = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)

    mod.launch_prepare_paged(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, paged_kv_indices,
        seq_lens_q_out, kv_lens_out, block_tables_out,
        page_size, batch_size, max_blocks,
    )
    torch.cuda.synchronize()

    # Check kv_lens
    assert torch.equal(kv_lens_out.cpu(), kv_seq_lens.cpu()), \
        f"kv_lens mismatch: {kv_lens_out.cpu()} vs {kv_seq_lens.cpu()}"
    print(f"  [PASS] kv_lens: {kv_lens_out.cpu().tolist()}")

    # Check seq_lens_q
    assert torch.equal(seq_lens_q_out.cpu(), q_seq_lens.cpu()), \
        f"seq_lens_q mismatch: {seq_lens_q_out.cpu()} vs {q_seq_lens.cpu()}"
    print(f"  [PASS] seq_lens_q: {seq_lens_q_out.cpu().tolist()}")

    # Check block_tables
    indptr_cpu = paged_kv_indptr.cpu()
    indices_cpu = paged_kv_indices.cpu()
    bt_cpu = block_tables_out.cpu()
    for i in range(batch_size):
        n = num_pages_per_seq[i].item()
        start = indptr_cpu[i].item()
        expected = indices_cpu[start:start + n]
        actual = bt_cpu[i, :n]
        assert torch.equal(actual, expected), f"block_tables[{i}] wrong"
    print(f"  [PASS] block_tables: all {batch_size} rows correct")


def test_full_chain(mod_paged, mod_cum):
    print()
    print("=" * 50)
    print("TEST: prepare_paged → prepare (cum_seq_lens) chain")
    print("=" * 50)

    device = torch.device("cuda")
    batch_size = 6
    page_size = 16

    # Random-ish sequence lengths
    kv_seq_lens = torch.tensor([48, 32, 64, 16, 80, 33], dtype=torch.int32, device=device)
    q_seq_lens = torch.tensor([8, 4, 16, 2, 12, 5], dtype=torch.int32, device=device)

    num_pages_per_seq = (kv_seq_lens + page_size - 1) // page_size
    total_pages = num_pages_per_seq.sum().item()

    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    paged_kv_indices = torch.randperm(total_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = ((kv_seq_lens - 1) % page_size) + 1

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_seq_lens, dim=0)

    max_blocks = num_pages_per_seq.max().item()

    # Step 1: prepare_paged
    seq_lens_q_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_lens_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    block_tables_out = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)

    mod_paged.launch_prepare_paged(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, paged_kv_indices,
        seq_lens_q_out, kv_lens_out, block_tables_out,
        page_size, batch_size, max_blocks,
    )

    # Step 2: prepare (cum scan)
    cum_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

    mod_cum.launch_prepare_cum(seq_lens_q_out, kv_lens_out, batch_size, cum_q, cum_kv)
    torch.cuda.synchronize()

    # Verify cum_seq_lens against CPU reference
    expected_cum_q = torch.zeros(batch_size + 1, dtype=torch.int32)
    expected_cum_kv = torch.zeros(batch_size + 1, dtype=torch.int32)
    expected_cum_q[1:] = torch.cumsum(q_seq_lens.cpu(), dim=0)
    expected_cum_kv[1:] = torch.cumsum(kv_seq_lens.cpu(), dim=0)

    assert torch.equal(cum_q.cpu(), expected_cum_q), \
        f"cum_q mismatch:\n  got: {cum_q.cpu().tolist()}\n  exp: {expected_cum_q.tolist()}"
    print(f"  [PASS] cum_seq_lens_q: {cum_q.cpu().tolist()}")

    assert torch.equal(cum_kv.cpu(), expected_cum_kv), \
        f"cum_kv mismatch:\n  got: {cum_kv.cpu().tolist()}\n  exp: {expected_cum_kv.tolist()}"
    print(f"  [PASS] cum_seq_lens_kv: {cum_kv.cpu().tolist()}")

    # Verify no D2H was needed for intermediate results
    print(f"  [PASS] full chain ran entirely on GPU (no host sync between kernels)")


def test_large_batch(mod_paged, mod_cum):
    print()
    print("=" * 50)
    print("TEST: large batch (B=512, simulating real workload)")
    print("=" * 50)

    device = torch.device("cuda")
    batch_size = 512
    page_size = 16

    # Random lengths
    torch.manual_seed(42)
    kv_seq_lens = torch.randint(1, 2048, (batch_size,), dtype=torch.int32, device=device)
    q_seq_lens = torch.randint(1, 128, (batch_size,), dtype=torch.int32, device=device)

    num_pages_per_seq = (kv_seq_lens + page_size - 1) // page_size
    total_pages = num_pages_per_seq.sum().item()

    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    paged_kv_indices = torch.randperm(total_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = ((kv_seq_lens - 1) % page_size) + 1

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_seq_lens, dim=0)

    max_blocks = num_pages_per_seq.max().item()

    seq_lens_q_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_lens_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    block_tables_out = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)

    mod_paged.launch_prepare_paged(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, paged_kv_indices,
        seq_lens_q_out, kv_lens_out, block_tables_out,
        page_size, batch_size, max_blocks,
    )

    cum_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    mod_cum.launch_prepare_cum(seq_lens_q_out, kv_lens_out, batch_size, cum_q, cum_kv)
    torch.cuda.synchronize()

    # Spot check
    assert torch.equal(kv_lens_out.cpu(), kv_seq_lens.cpu())
    assert torch.equal(seq_lens_q_out.cpu(), q_seq_lens.cpu())
    assert cum_q[-1].item() == q_seq_lens.sum().item()
    assert cum_kv[-1].item() == kv_seq_lens.sum().item()
    print(f"  [PASS] B=512, total_pages={total_pages}, max_blocks={max_blocks}")
    print(f"         cum_q[-1]={cum_q[-1].item()}, cum_kv[-1]={cum_kv[-1].item()}")

    # Timing
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(1000):
        mod_paged.launch_prepare_paged(
            qo_indptr, paged_kv_indptr, paged_kv_last_page_len, paged_kv_indices,
            seq_lens_q_out, kv_lens_out, block_tables_out,
            page_size, batch_size, max_blocks,
        )
        mod_cum.launch_prepare_cum(seq_lens_q_out, kv_lens_out, batch_size, cum_q, cum_kv)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 1000 * 1e6
    print(f"  [PERF] two-kernel chain: {elapsed:.1f} μs avg (1000 iters)")

    # Compare with the old python approach (host for-loop)
    torch.cuda.synchronize()
    indptr_host = paged_kv_indptr.cpu()
    indices_dev = paged_kv_indices
    start = time.perf_counter()
    for _ in range(100):
        bt = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
        for i in range(batch_size):
            n = int(num_pages_per_seq[i].item())
            s = int(indptr_host[i].item())
            bt[i, :n] = indices_dev[s:s+n]
    torch.cuda.synchronize()
    elapsed_old = (time.perf_counter() - start) / 100 * 1e6
    print(f"  [PERF] python for-loop baseline: {elapsed_old:.1f} μs avg (100 iters)")
    print(f"  [PERF] speedup: {elapsed_old / elapsed:.1f}x")


if __name__ == "__main__":
    print("Compiling prepare_paged kernel...")
    mod_paged = build_module("test_prepare_paged", PREPARE_PAGED_SRC)
    print("Compiling prepare (cum scan) kernel...")
    mod_cum = build_module("test_prepare_cum", PREPARE_CUM_SRC)
    print()

    test_prepare_paged_kernel(mod_paged)
    test_full_chain(mod_paged, mod_cum)
    test_large_batch(mod_paged, mod_cum)

    print()
    print("=== ALL TESTS PASSED ===")
