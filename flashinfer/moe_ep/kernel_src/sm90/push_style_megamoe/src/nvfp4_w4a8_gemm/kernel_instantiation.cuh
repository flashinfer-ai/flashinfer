// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel.cuh"
#include "kernel_launchers.cuh"

namespace flashinfer {
namespace sm90_w4a8 {
namespace detail {

#if W4A8_RESIDUAL_TMA
#define FLASHINFER_SM90_W4A8_RESIDUAL_LAUNCH_ARGUMENT params.residual_map
#else
#define FLASHINFER_SM90_W4A8_RESIDUAL_LAUNCH_ARGUMENT \
  static_cast<const typename ResidualDecoder<Scheme>::Storage*>(params.residual)
#endif

#if W4A8_GROUP_SCALE_TMA
#define FLASHINFER_SM90_W4A8_GROUP_SCALE_LAUNCH_ARGUMENT params.group_scale_map
#else
#define FLASHINFER_SM90_W4A8_GROUP_SCALE_LAUNCH_ARGUMENT params.group_scales
#endif

template <typename Kernel>
cudaError_t query_w4a8_kernel_resources(Kernel kernel, int threads, size_t dynamic_smem_bytes,
                                        W4A8KernelResources* resources) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  resources->num_regs = attributes.numRegs;
  resources->local_memory_bytes = attributes.localSizeBytes;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resources->blocks_per_sm, kernel, threads,
                                                       dynamic_smem_bytes);
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
cudaError_t configure_w4a8_kernel_variant(int opt_in_smem, W4A8KernelResources* bf16_resources,
                                          W4A8KernelResources* fp32_resources) {
  (void)opt_in_smem;
  constexpr size_t kSmemBytes = w4a8_smem_bytes<BlockM, BlockN, GroupSize, Scheme>();
  constexpr int kThreads = W4A8LaunchTraits<BlockM, BlockN>::kThreads;
  auto bf16_kernel = grouped_w4a8_bf16_kernel<BlockM, BlockN, GroupSize, Scheme>;
  auto fp32_kernel = grouped_w4a8_fp32_debug_kernel<BlockM, BlockN, GroupSize, Scheme>;
  cudaError_t status = cudaFuncSetAttribute(
      bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaFuncSetAttribute(fp32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(kSmemBytes));
  if (status != cudaSuccess) {
    return status;
  }
  status = query_w4a8_kernel_resources(bf16_kernel, kThreads, kSmemBytes, bf16_resources);
  if (status != cudaSuccess) {
    return status;
  }
  return query_w4a8_kernel_resources(fp32_kernel, kThreads, kSmemBytes, fp32_resources);
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
cudaError_t launch_w4a8_kernel_variant(bool debug_fp32, int32_t blocks, cudaStream_t stream,
                                       const W4A8KernelLaunchParams& params) {
  constexpr size_t kSmemBytes = w4a8_smem_bytes<BlockM, BlockN, GroupSize, Scheme>();
  constexpr int kThreads = W4A8LaunchTraits<BlockM, BlockN>::kThreads;
  if (debug_fp32) {
    grouped_w4a8_fp32_debug_kernel<BlockM, BlockN, GroupSize, Scheme>
        <<<blocks, kThreads, kSmemBytes, stream>>>(
            static_cast<float*>(params.output), params.activation_scales, params.alpha,
            params.expert_mapping, params.source_offsets, params.tile_prefix, params.task_counter,
            params.row_capacity, params.logical_n, params.padded_n, params.padded_k,
            params.launch_n_tiles, params.n_tile_begin, params.bucket_experts,
            params.activation_scale_stride, params.alpha_per_expert, params.activation_map,
            params.payload_map, FLASHINFER_SM90_W4A8_RESIDUAL_LAUNCH_ARGUMENT,
            FLASHINFER_SM90_W4A8_GROUP_SCALE_LAUNCH_ARGUMENT);
  } else {
    grouped_w4a8_bf16_kernel<BlockM, BlockN, GroupSize, Scheme>
        <<<blocks, kThreads, kSmemBytes, stream>>>(
            static_cast<__nv_bfloat16*>(params.output), params.activation_scales, params.alpha,
            params.expert_mapping, params.source_offsets, params.tile_prefix, params.task_counter,
            params.row_capacity, params.logical_n, params.padded_n, params.padded_k,
            params.launch_n_tiles, params.n_tile_begin, params.bucket_experts,
            params.activation_scale_stride, params.alpha_per_expert, params.activation_map,
            params.payload_map, FLASHINFER_SM90_W4A8_RESIDUAL_LAUNCH_ARGUMENT,
            FLASHINFER_SM90_W4A8_GROUP_SCALE_LAUNCH_ARGUMENT);
  }
  return cudaGetLastError();
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
W4A8KernelVariant make_w4a8_kernel_variant() {
  using Traits = W4A8LaunchTraits<BlockM, BlockN>;
  return W4A8KernelVariant{
      BlockM,
      BlockN,
      GroupSize,
      Scheme,
      Traits::kThreads,
      Traits::kPipelineStages,
      Traits::kMinBlocksPerSm,
      Traits::kRegisterFootprintTarget,
      w4a8_smem_bytes<BlockM, BlockN, GroupSize, Scheme>(),
      &configure_w4a8_kernel_variant<BlockM, BlockN, GroupSize, Scheme>,
      &launch_w4a8_kernel_variant<BlockM, BlockN, GroupSize, Scheme>,
  };
}

}  // namespace detail
}  // namespace sm90_w4a8
}  // namespace flashinfer

#undef FLASHINFER_SM90_W4A8_GROUP_SCALE_LAUNCH_ARGUMENT
#undef FLASHINFER_SM90_W4A8_RESIDUAL_LAUNCH_ARGUMENT

#define FLASHINFER_SM90_W4A8_ACCESSOR_NAME_IMPL(BlockM, BlockN) \
  get_w4a8_m##BlockM##_n##BlockN##_variants
#define FLASHINFER_SM90_W4A8_ACCESSOR_NAME(BlockM, BlockN) \
  FLASHINFER_SM90_W4A8_ACCESSOR_NAME_IMPL(BlockM, BlockN)

#define FLASHINFER_SM90_W4A8_DEFINE_MN_VARIANTS(BlockM, BlockN)                            \
  namespace flashinfer {                                                                   \
  namespace sm90_w4a8 {                                                                    \
  const W4A8KernelVariant* FLASHINFER_SM90_W4A8_ACCESSOR_NAME(BlockM, BlockN)() {          \
    static const W4A8KernelVariant variants[kW4A8VariantsPerMN] = {                        \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 32, ResidualScheme::kGeneric>(),  \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 64, ResidualScheme::kGeneric>(),  \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 128, ResidualScheme::kGeneric>(), \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 32, ResidualScheme::kPow2>(),     \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 64, ResidualScheme::kPow2>(),     \
        detail::make_w4a8_kernel_variant<BlockM, BlockN, 128, ResidualScheme::kPow2>(),    \
    };                                                                                     \
    return variants;                                                                       \
  }                                                                                        \
  }                                                                                        \
  }
