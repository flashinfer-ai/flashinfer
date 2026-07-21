#!/usr/bin/env python3
"""
Standalone verification for fmha_v2_prepare_kernel.
Compiles and runs the kernel directly via PyTorch's CUDA extension,
bypassing FlashInfer's import chain.

Requirements: torch (with CUDA), Python 3.8+
Works on any NVIDIA GPU (SM86 included).
"""
import torch
import torch.utils.cpp_extension
import tempfile
import os

KERNEL_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fmha_v2_prepare_kernel(
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ paged_kv_indptr,
    const int32_t* __restrict__ paged_kv_last_page_len,
    const int32_t* __restrict__ paged_kv_indices,
    int32_t* __restrict__ kv_lens,
    int32_t* __restrict__ block_tables,
    int32_t* __restrict__ metadata,
    int page_size, int batch_size, int max_blocks_per_seq) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= batch_size) return;

  int num_pages_i = paged_kv_indptr[i + 1] - paged_kv_indptr[i];
  int kv_len_i = max(num_pages_i - 1, 0) * page_size + paged_kv_last_page_len[i];
  kv_lens[i] = kv_len_i;

  int q_len_i = qo_indptr[i + 1] - qo_indptr[i];

  int block_start = paged_kv_indptr[i];
  for (int j = 0; j < num_pages_i && j < max_blocks_per_seq; j++) {
    block_tables[i * max_blocks_per_seq + j] = paged_kv_indices[block_start + j];
  }

  atomicMax(&metadata[0], q_len_i);
  atomicMax(&metadata[1], kv_len_i);

  if (i == 0) {
    metadata[2] = qo_indptr[batch_size];
    metadata[3] = max_blocks_per_seq;
  }
}

void launch_prepare(
    torch::Tensor qo_indptr,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_last_page_len,
    torch::Tensor paged_kv_indices,
    torch::Tensor kv_lens_out,
    torch::Tensor block_tables_out,
    torch::Tensor metadata_out,
    int page_size, int batch_size, int max_blocks_per_seq) {

  cudaMemsetAsync(metadata_out.data_ptr(), 0, 4 * sizeof(int32_t),
                  at::cuda::getCurrentCUDAStream());

  int threads = min(batch_size, 1024);
  int blocks = (batch_size + threads - 1) / threads;

  fmha_v2_prepare_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      qo_indptr.data_ptr<int32_t>(),
      paged_kv_indptr.data_ptr<int32_t>(),
      paged_kv_last_page_len.data_ptr<int32_t>(),
      paged_kv_indices.data_ptr<int32_t>(),
      kv_lens_out.data_ptr<int32_t>(),
      block_tables_out.data_ptr<int32_t>(),
      metadata_out.data_ptr<int32_t>(),
      page_size, batch_size, max_blocks_per_seq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_prepare", &launch_prepare, "FMHAv2 GPU prepare kernel");
}
"""


def test_prepare_kernel():
    # JIT compile
    print("Compiling prepare kernel...")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "fmha_v2_prepare_test.cu")
        with open(src_path, "w") as f:
            f.write(KERNEL_SOURCE)

        mod = torch.utils.cpp_extension.load(
            name="fmha_v2_prepare_test",
            sources=[src_path],
            verbose=True,
        )

    print("Compilation OK! Running test...")

    device = torch.device("cuda")
    batch_size = 4
    page_size = 16

    # Variable-length sequences
    kv_seq_lens = torch.tensor([48, 32, 64, 16], dtype=torch.int32, device=device)
    q_seq_lens = torch.tensor([8, 4, 16, 2], dtype=torch.int32, device=device)

    # Build paged KV structures
    num_pages_per_seq = (kv_seq_lens + page_size - 1) // page_size  # [3, 2, 4, 1]
    total_pages = num_pages_per_seq.sum().item()

    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)

    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = ((kv_seq_lens - 1) % page_size) + 1

    # Build qo_indptr
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_seq_lens, dim=0)

    max_blocks_per_seq = num_pages_per_seq.max().item()

    # Outputs
    kv_lens_out = torch.zeros(batch_size, dtype=torch.int32, device=device)
    block_tables_out = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device=device)
    metadata_out = torch.zeros(4, dtype=torch.int32, device=device)

    # Run GPU kernel
    mod.launch_prepare(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, paged_kv_indices,
        kv_lens_out, block_tables_out, metadata_out,
        page_size, batch_size, max_blocks_per_seq,
    )
    torch.cuda.synchronize()

    # ===== Verify =====
    print("\n--- Results ---")

    # 1. kv_lens
    expected_kv_lens = kv_seq_lens.cpu()
    actual_kv_lens = kv_lens_out.cpu()
    assert torch.equal(actual_kv_lens, expected_kv_lens), \
        f"kv_lens mismatch:\n  expected: {expected_kv_lens}\n  got: {actual_kv_lens}"
    print(f"[PASS] kv_lens: {actual_kv_lens.tolist()}")

    # 2. metadata
    meta = metadata_out.cpu()
    assert meta[0].item() == q_seq_lens.max().item(), f"max_q_len: expected {q_seq_lens.max().item()}, got {meta[0].item()}"
    assert meta[1].item() == kv_seq_lens.max().item(), f"max_kv_len: expected {kv_seq_lens.max().item()}, got {meta[1].item()}"
    assert meta[2].item() == qo_indptr[-1].item(), f"total_rows: expected {qo_indptr[-1].item()}, got {meta[2].item()}"
    assert meta[3].item() == max_blocks_per_seq, f"max_blocks: expected {max_blocks_per_seq}, got {meta[3].item()}"
    print(f"[PASS] metadata: max_q_len={meta[0]}, max_kv_len={meta[1]}, total_rows={meta[2]}, max_blocks={meta[3]}")

    # 3. block_tables
    indptr_cpu = paged_kv_indptr.cpu()
    indices_cpu = paged_kv_indices.cpu()
    bt_cpu = block_tables_out.cpu()
    for i in range(batch_size):
        n_pages = num_pages_per_seq[i].item()
        start = indptr_cpu[i].item()
        expected_row = indices_cpu[start: start + n_pages]
        actual_row = bt_cpu[i, :n_pages]
        assert torch.equal(actual_row, expected_row), \
            f"block_tables[{i}] mismatch:\n  expected: {expected_row}\n  got: {actual_row}"
    print(f"[PASS] block_tables: all {batch_size} rows correct")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_prepare_kernel()
