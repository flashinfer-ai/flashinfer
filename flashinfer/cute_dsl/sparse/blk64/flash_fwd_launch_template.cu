/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Dispatch + Python binding for BSA fused attention forward kernel (blk=64)
// Kernel instantiations are in instantiations/*.cu (separate TUs).

#include <torch/extension.h>

#include <vector>

#include "static_switch.h"

namespace flash {

// Declarations of explicit instantiations (defined in instantiations/*.cu)
template <bool HasVarBlockNums, bool HasBlockSizes>
std::vector<torch::Tensor> bsa_fused_fwd_blk64_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor q2k_block_index,
    int block_sparse_num, torch::Tensor block_sizes, float softmax_scale,
    torch::Tensor q2k_block_nums);

// Entry point: BOOL_SWITCH dispatches to the correct template instantiation
inline std::vector<torch::Tensor> bsa_fused_fwd_blk64_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor q2k_block_index,
    int block_sparse_num, torch::Tensor block_sizes, float softmax_scale,
    torch::Tensor q2k_block_nums) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
  TORCH_CHECK(q.size(3) == 128, "requires D=128");

  bool has_variable_block_nums = q2k_block_nums.defined() && q2k_block_nums.numel() > 0;
  bool has_block_sizes = block_sizes.defined() && block_sizes.numel() > 0;
  return BOOL_SWITCH(has_variable_block_nums, HasVarBlockNums, [&] {
    return BOOL_SWITCH(has_block_sizes, HasBlockSizes, [&] {
      return bsa_fused_fwd_blk64_launch<HasVarBlockNums, HasBlockSizes>(
          q, k, v, q2k_block_index, block_sparse_num, block_sizes, softmax_scale, q2k_block_nums);
    });
  });
}

}  // namespace flash

// Python binding
std::vector<torch::Tensor> bsa_fused_fwd_blk64(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                               torch::Tensor q2k_block_index, int block_sparse_num,
                                               torch::Tensor block_sizes, float softmax_scale,
                                               torch::Tensor q2k_block_nums) {
  return flash::bsa_fused_fwd_blk64_impl(q, k, v, q2k_block_index, block_sparse_num, block_sizes,
                                         softmax_scale, q2k_block_nums);
}
