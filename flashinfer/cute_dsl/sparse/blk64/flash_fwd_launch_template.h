/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Host launch for BSA fused attention forward kernel (blk=64, C++ AOT)
// BSA tensor convention: (batch, seq, heads, dim)
#pragma once
#define CUDA_CTA_RECONFIG_ACTIVATED 1

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/extension.h>

#include <cutlass/cluster_launch.hpp>
#include <limits>
#include <sstream>
#include <vector>

#include "flash_fwd_kernel_sm100.h"

namespace flash {

// Templated launch: HasVarBlockNums/HasBlockSizes select kernel variant at compile time.
template <bool HasVarBlockNums, bool HasBlockSizes>
std::vector<torch::Tensor> bsa_fused_fwd_blk64_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor q2k_block_index,
    int block_sparse_num, torch::Tensor block_sizes, float softmax_scale,
    torch::Tensor q2k_block_nums) {
  using namespace cute;

  using Kernel = FusedAttnKernel<HasVarBlockNums, HasBlockSizes>;
  using bf16 = typename Kernel::ElementA;
  using ML = typename Kernel::CollectiveMainloop;
  constexpr int kRows = Kernel::kRows;
  constexpr int kDualCols = Kernel::kDualCols;
  constexpr int kQkK = Kernel::kQkK;
  constexpr int kOutputCols = Kernel::kOutputCols;
  constexpr int kSparseBlockSize = ML::kSparseBlockSize;
  constexpr int kDimHalf = ML::kDimHalf;
  constexpr int kSparseBlocksPerKV = ML::kSparseBlocksPerKV;

  const int batch = static_cast<int>(q.size(0));
  const int seq_q = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int seq_k = static_cast<int>(k.size(1));

  // Phantom block padding: round up to multiple of 8 (kSparseBlocksPerKV*2) for even kv_iters.
  constexpr int kAlign = kSparseBlocksPerKV * 2;  // 8
  int raw_block_sparse_num;
  int num_kv_iters;
  if constexpr (!HasVarBlockNums) {
    TORCH_CHECK(block_sparse_num >= 1, "block_sparse_num must be >= 1");
    raw_block_sparse_num = block_sparse_num;
    int padded = ((block_sparse_num + kAlign - 1) / kAlign) * kAlign;
    num_kv_iters = padded / kSparseBlocksPerKV;
  } else {
    // Variable: use max possible for TMA/grid sizing (actual per-tile count read in kernel)
    (void)block_sparse_num;
    int num_kv_blocks_max = (seq_k + kSparseBlockSize - 1) / kSparseBlockSize;
    raw_block_sparse_num = num_kv_blocks_max;
    int padded = ((num_kv_blocks_max + kAlign - 1) / kAlign) * kAlign;
    num_kv_iters = padded / kSparseBlocksPerKV;
  }

  const int rows_padded = ((seq_q + kRows - 1) / kRows) * kRows;
  const int num_row_tiles = rows_padded / kRows;
  const int seq_padded = num_kv_iters * kDualCols;

  if constexpr (!HasVarBlockNums) {
    TORCH_CHECK(q2k_block_index.size(0) == batch, "q2k_block_index batch mismatch");
    TORCH_CHECK(q2k_block_index.size(1) == heads, "q2k_block_index heads mismatch");
    TORCH_CHECK(q2k_block_index.size(2) == num_row_tiles, "q2k_block_index q_blocks mismatch");
    TORCH_CHECK(q2k_block_index.size(3) == block_sparse_num, "q2k_block_index kv_blocks mismatch");
  }

  // ======== Prepare Q ========
  auto q_bhsd = q.permute({0, 2, 1, 3}).contiguous();
  auto q_padded = torch::zeros({batch, heads, rows_padded, kQkK}, q.options());
  q_padded.narrow(2, 0, seq_q).copy_(q_bhsd);
  auto q_tiled = q_padded.view({batch * heads * num_row_tiles, kRows, kQkK}).contiguous();

  // ======== Prepare K ========
  auto k_bhsd = k.permute({0, 2, 1, 3}).contiguous();
  int const total_k_padded = ((seq_k + kSparseBlockSize - 1) / kSparseBlockSize) * kSparseBlockSize;
  int const total_sparse_blocks = total_k_padded / kSparseBlockSize;
  auto k_padded = torch::zeros({batch, heads, total_k_padded, kQkK}, k.options());
  k_padded.narrow(2, 0, seq_k).copy_(k_bhsd);
  auto k_blocks =
      k_padded.view({batch, heads, total_sparse_blocks, kSparseBlockSize, 2, kDimHalf})
          .permute({0, 1, 2, 4, 3, 5})
          .reshape({batch * heads * total_sparse_blocks * 2, kSparseBlockSize, kDimHalf})
          .contiguous();

  // ======== Prepare V ========
  auto v_bhds = v.permute({0, 2, 3, 1}).contiguous();
  auto v_padded = torch::zeros({batch, heads, kOutputCols, total_k_padded}, v.options());
  v_padded.narrow(3, 0, seq_k).copy_(v_bhds);
  auto v_blocks =
      v_padded.view({batch, heads, 2, kDimHalf, total_sparse_blocks, kSparseBlockSize})
          .permute({0, 1, 4, 2, 3, 5})
          .reshape({batch * heads * total_sparse_blocks * 2, kDimHalf, kSparseBlockSize})
          .contiguous();

  auto out_flat = torch::zeros({batch * heads * num_row_tiles, kRows, kOutputCols}, q.options());
  auto lse_flat =
      torch::full({batch * heads * num_row_tiles * kRows}, -std::numeric_limits<float>::infinity(),
                  torch::dtype(torch::kFloat32).device(q.device()));

  // block_indices: flatten to (B*H, Q, max_KV)
  auto bi_flat = q2k_block_index.reshape({batch * heads, num_row_tiles, -1}).contiguous();
  int block_indices_stride = static_cast<int>(bi_flat.size(2));

  // q2k_block_nums: flatten to (B*H*Q,) for kernel indexing
  int const* ptr_q2k_block_nums = nullptr;
  torch::Tensor bn_flat;
  if constexpr (HasVarBlockNums) {
    bn_flat = q2k_block_nums.reshape({batch * heads * num_row_tiles}).contiguous();
    ptr_q2k_block_nums = bn_flat.data_ptr<int>();
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  typename Kernel::Arguments args{
      // mainloop
      {
          reinterpret_cast<bf16 const*>(q_tiled.data_ptr<at::BFloat16>()),
          reinterpret_cast<bf16 const*>(k_blocks.data_ptr<at::BFloat16>()),
          reinterpret_cast<bf16 const*>(v_blocks.data_ptr<at::BFloat16>()),
          softmax_scale,
          bi_flat.data_ptr<int>(),
          block_indices_stride,
          block_sizes.defined() && block_sizes.numel() > 0 ? block_sizes.data_ptr<int>() : nullptr,
          ptr_q2k_block_nums,
          raw_block_sparse_num,
      },
      // epilogue
      {reinterpret_cast<bf16*>(out_flat.data_ptr<at::BFloat16>()), lse_flat.data_ptr<float>()},
      // dimensions
      rows_padded,
      seq_padded,
      heads,
      batch,
      total_k_padded,
  };

  auto kernel_params = Kernel::to_underlying_arguments(args);

  dim3 dim_grid_full = Kernel::get_grid_shape(kernel_params);
  dim3 dim_block = Kernel::get_block_shape();
  int smem_bytes = Kernel::SharedStorageSize;
  int total_tiles = dim_grid_full.x * dim_grid_full.y * dim_grid_full.z;
  dim3 dim_grid = dim3(total_tiles, 1, 1);

  auto* kernel_ptr = &fused_attn_device<Kernel>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();

  nvtxRangePushA("bsa_attn_fwd_kernel");
  fused_attn_device<Kernel><<<dim_grid, dim_block, smem_bytes, stream>>>(kernel_params);
  nvtxRangePop();
  C10_CUDA_CHECK(cudaGetLastError());
#else
  TORCH_CHECK(false, "requires SM100");
#endif

  auto out = out_flat.view({batch, heads, rows_padded, kOutputCols})
                 .narrow(2, 0, seq_q)
                 .permute({0, 2, 1, 3})
                 .contiguous();
  auto lse = lse_flat.view({batch, heads, rows_padded}).narrow(2, 0, seq_q).contiguous();
  return {out, lse};
}

}  // namespace flash
