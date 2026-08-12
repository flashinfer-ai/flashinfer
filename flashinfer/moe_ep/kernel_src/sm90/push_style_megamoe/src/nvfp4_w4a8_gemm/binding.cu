// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include <cudaTypedefs.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernel.cuh"
#include "kernel_launchers.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm90_w4a8 {

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

inline PFN_cuTensorMapEncodeTiled_v12000 get_tma_encoder() {
  cudaDriverEntryPointQueryResult driver_status;
  void* encoder = nullptr;
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
  const cudaError_t status = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &encoder, 12000, cudaEnableDefault, &driver_status);
#else
  const cudaError_t status = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &encoder,
                                                     cudaEnableDefault, &driver_status);
#endif
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "get_tma_encoder: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK_EQ(driver_status, cudaDriverEntryPointSuccess)
      << "get_tma_encoder: cuTensorMapEncodeTiled is unavailable";
  TVM_FFI_ICHECK(encoder != nullptr) << "get_tma_encoder: encoder is null";
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encoder);
}

inline CUtensorMap make_activation_tma_map(void* address, uint64_t rows, int32_t padded_k,
                                           int32_t block_m,
                                           PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  CUtensorMap tensor_map{};
  const uint64_t global_dims[2] = {static_cast<uint64_t>(padded_k), rows};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(padded_k)};
  const uint32_t box_dims[2] = {static_cast<uint32_t>(kBlockK), static_cast<uint32_t>(block_m)};
  const uint32_t element_strides[2] = {1, 1};
  const CUresult result =
      encoder(&tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, address,
              global_dims, global_strides, box_dims, element_strides,
              CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "make_activation_tma_map: cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

inline CUtensorMap make_payload_tma_map(void* address, int32_t bucket_experts, int32_t k_tiles,
                                        int32_t n_tiles,
                                        PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  CUtensorMap tensor_map{};
  const uint64_t tile_bytes = static_cast<uint64_t>(kV3PackedBytesPerRow) * kV3PayloadTileN;
  const uint64_t cells = static_cast<uint64_t>(bucket_experts) * k_tiles * n_tiles;
  const uint64_t global_dims[3] = {static_cast<uint64_t>(kV3PackedBytesPerRow),
                                   static_cast<uint64_t>(kV3PayloadTileN), cells};
  const uint64_t global_strides[2] = {static_cast<uint64_t>(kV3PackedBytesPerRow), tile_bytes};
  const uint32_t box_dims[3] = {static_cast<uint32_t>(kV3PackedBytesPerRow),
                                static_cast<uint32_t>(kV3PayloadTileN), 1};
  const uint32_t element_strides[3] = {1, 1, 1};
  const CUresult result =
      encoder(&tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, address,
              global_dims, global_strides, box_dims, element_strides,
              CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "make_payload_tma_map: cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

inline CUtensorMap make_residual_tma_map(void* address, int32_t bucket_experts, int32_t k_tiles,
                                         int32_t n_tiles, ResidualScheme scheme,
                                         PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  CUtensorMap tensor_map{};
  const uint64_t element_bytes = scheme == ResidualScheme::kGeneric ? 2 : 1;
  const uint64_t cells = static_cast<uint64_t>(bucket_experts) * k_tiles * n_tiles;
  constexpr uint64_t kElementsPerCell = kV3PayloadTileN * kV3ResidualsPerPayloadTile;
  const uint64_t global_dims[2] = {kElementsPerCell, cells};
  const uint64_t global_strides[1] = {kElementsPerCell * element_bytes};
  const uint32_t box_dims[2] = {static_cast<uint32_t>(kElementsPerCell), 1};
  const uint32_t element_strides[2] = {1, 1};
  const auto data_type = scheme == ResidualScheme::kGeneric
                             ? CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                             : CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  const CUresult result =
      encoder(&tensor_map, data_type, 2, address, global_dims, global_strides, box_dims,
              element_strides, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "make_residual_tma_map: cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

inline CUtensorMap make_group_scale_tma_map(void* address, int32_t bucket_experts, int32_t k_groups,
                                            int32_t n_tiles,
                                            PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  CUtensorMap tensor_map{};
  const uint64_t cells = static_cast<uint64_t>(bucket_experts) * k_groups * n_tiles;
  const uint64_t global_dims[2] = {static_cast<uint64_t>(kV3PayloadTileN), cells};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(kV3PayloadTileN) * sizeof(float)};
  const uint32_t box_dims[2] = {static_cast<uint32_t>(kV3PayloadTileN), 1};
  const uint32_t element_strides[2] = {1, 1};
  const CUresult result =
      encoder(&tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, address,
              global_dims, global_strides, box_dims, element_strides,
              CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "make_group_scale_tma_map: cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

// One thread validates the inputs and builds exact prefixes for the disjoint
// M64-tail and M128 queues.
__global__ void prepare_grouped_schedule_kernel(
    const int64_t* source_offsets, const int32_t* expert_mapping, int32_t bucket_experts,
    int32_t total_experts, int64_t row_capacity, unsigned long long* task_counters,
    int64_t* tile_prefix_m64, int64_t* tile_prefix_m128, bool trusted_offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

#pragma unroll
  for (int counter = 0; counter < kNumTaskCounters; ++counter) {
    task_counters[counter] = 0;
  }
  tile_prefix_m64[0] = -1;
  tile_prefix_m128[0] = -1;

  if (bucket_experts < 0 || total_experts < 0 || row_capacity < 0) {
    trap_invalid_schedule(1);
    return;
  }
  if (!trusted_offsets) {
    int64_t previous = source_offsets[0];
    if (previous != 0 || previous > row_capacity) {
      trap_invalid_schedule(2);
      return;
    }
    for (int32_t expert = 0; expert < total_experts; ++expert) {
      const int64_t current = source_offsets[expert + 1];
      if (current < previous || current > row_capacity) {
        trap_invalid_schedule(3);
        return;
      }
      previous = current;
    }
  }

  int64_t prefix_m64 = 0;
  int64_t prefix_m128 = 0;
  tile_prefix_m64[0] = 0;
  tile_prefix_m128[0] = 0;
  for (int32_t bucket_expert = 0; bucket_expert < bucket_experts; ++bucket_expert) {
    const int32_t source_expert = expert_mapping[bucket_expert];
    if (source_expert < 0 || source_expert >= total_experts) {
      tile_prefix_m64[0] = -1;
      tile_prefix_m128[0] = -1;
      trap_invalid_schedule(4);
      return;
    }
    if (!trusted_offsets) {
      for (int32_t previous = 0; previous < bucket_expert; ++previous) {
        if (expert_mapping[previous] == source_expert) {
          tile_prefix_m64[0] = -1;
          tile_prefix_m128[0] = -1;
          trap_invalid_schedule(5);
          return;
        }
      }
    }

    const int64_t begin = source_offsets[source_expert];
    const int64_t end = source_offsets[source_expert + 1];
    if (begin < 0 || end < begin || end > row_capacity) {
      tile_prefix_m64[0] = -1;
      tile_prefix_m128[0] = -1;
      trap_invalid_schedule(6);
      return;
    }
    const int64_t rows = end - begin;
#if W4A8_SPLIT_M64_TAIL
    const int64_t tiles_m64 = m64_tile_count(rows);
    const int64_t tiles_m128 = m128_tile_count(rows);
#else
    constexpr int64_t tiles_m64 = 0;
    const int64_t tiles_m128 = ceil_div_nonnegative(rows, 128);
#endif
    if (tiles_m64 > std::numeric_limits<int64_t>::max() - prefix_m64 ||
        tiles_m128 > std::numeric_limits<int64_t>::max() - prefix_m128) {
      tile_prefix_m64[0] = -1;
      tile_prefix_m128[0] = -1;
      trap_invalid_schedule(7);
      return;
    }
    prefix_m64 += tiles_m64;
    prefix_m128 += tiles_m128;
    tile_prefix_m64[bucket_expert + 1] = prefix_m64;
    tile_prefix_m128[bucket_expert + 1] = prefix_m128;
  }
}

class Sm90W4A8GroupedGemmRunner final : public tvm::ffi::ModuleObj {
 public:
  const char* type_key() const { return "flashinfer.Sm90W4A8GroupedGemmRunner"; }

  const char* kind() const final { return "sm90_w4a8_grouped_gemm_runner"; }

  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "get_workspace_size") {
      return Function::FromTyped([this](int64_t max_m, int64_t logical_n, int64_t padded_n,
                                        int64_t padded_k, int64_t bucket_experts,
                                        int64_t total_experts, int64_t group_size,
                                        tvm::ffi::String residual_scheme) -> int64_t {
        return get_workspace_size(max_m, logical_n, padded_n, padded_k, bucket_experts,
                                  total_experts, group_size, residual_scheme);
      });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configure_workspace(workspace); });
    }
    if (name == "configure_workspace_bank") {
      return Function::FromTyped([this](TensorView workspace, int64_t counter_bank) {
        configure_workspace_bank(workspace, counter_bank);
      });
    }
    if (name == "kernel_resource_usage") {
      return Function::FromTyped([this](int64_t block_m, int64_t block_n, bool debug_fp32) {
        return kernel_resource_usage(block_m, block_n, debug_fp32);
      });
    }
    if (name == "grouped_run") {
      return Function::FromTyped(
          [this](TensorView output, TensorView activation, TensorView activation_scales,
                 TensorView payload, TensorView residual, TensorView group_scales, TensorView alpha,
                 TensorView expert_mapping, TensorView offsets, bool trusted_offsets) {
            grouped_run(output, activation, activation_scales, payload, residual, group_scales,
                        alpha, expert_mapping, offsets, trusted_offsets);
          });
    }
    if (name == "grouped_run_prepared") {
      return Function::FromTyped(
          [this](TensorView output, TensorView activation, TensorView activation_scales,
                 TensorView payload, TensorView residual, TensorView group_scales, TensorView alpha,
                 TensorView expert_mapping, TensorView offsets, bool trusted_offsets) {
            grouped_run_prepared(output, activation, activation_scales, payload, residual,
                                 group_scales, alpha, expert_mapping, offsets, trusted_offsets);
          });
    }
    // Internal correctness-test entry points share the configured workspace.
    if (name == "debug_decode") {
      return Function::FromTyped(
          [this](TensorView output, TensorView payload, TensorView residual) {
            debug_decode(output, payload, residual);
          });
    }
    if (name == "debug_run_fp32") {
      return Function::FromTyped(
          [this](TensorView output, TensorView activation, TensorView activation_scales,
                 TensorView payload, TensorView residual, TensorView group_scales, TensorView alpha,
                 TensorView expert_mapping, TensorView offsets, bool trusted_offsets) {
            debug_run_fp32(output, activation, activation_scales, payload, residual, group_scales,
                           alpha, expert_mapping, offsets, trusted_offsets);
          });
    }
    return Function(nullptr);
  }

 private:
  int64_t get_workspace_size(int64_t max_m, int64_t logical_n, int64_t padded_n, int64_t padded_k,
                             int64_t bucket_experts, int64_t total_experts, int64_t group_size,
                             const tvm::ffi::String& residual_scheme) {
    TVM_FFI_ICHECK_GE(max_m, 0) << "get_workspace_size: max_m must be nonnegative";
    TVM_FFI_ICHECK_GT(logical_n, 0) << "get_workspace_size: logical_n must be positive";
    TVM_FFI_ICHECK_GT(padded_n, 0) << "get_workspace_size: padded_n must be positive";
    TVM_FFI_ICHECK_EQ(padded_n % kV3PayloadTileN, 0)
        << "get_workspace_size: padded_n must be N64 aligned";
    TVM_FFI_ICHECK_EQ(padded_n, ceil_div_nonnegative(logical_n, kV3PayloadTileN) *
                                    static_cast<int64_t>(kV3PayloadTileN))
        << "get_workspace_size: padded_n must be minimal N64 padding";
    TVM_FFI_ICHECK_GT(padded_k, 0) << "get_workspace_size: padded_k must be positive";
    TVM_FFI_ICHECK_EQ(padded_k % kBlockK, 0) << "get_workspace_size: padded_k must be K128 aligned";
    TVM_FFI_ICHECK_GT(bucket_experts, 0) << "get_workspace_size: bucket_experts must be positive";
    TVM_FFI_ICHECK_GT(total_experts, 0) << "get_workspace_size: total_experts must be positive";
    TVM_FFI_ICHECK_LE(bucket_experts, total_experts)
        << "get_workspace_size: bucket_experts exceeds total_experts";
    TVM_FFI_ICHECK(group_size == 32 || group_size == 64 || group_size == 128)
        << "get_workspace_size: group_size must be 32, 64, or 128";
    TVM_FFI_ICHECK_EQ(padded_k % group_size, 0)
        << "get_workspace_size: padded_k must be divisible by group_size";
    TVM_FFI_ICHECK(residual_scheme == "generic" || residual_scheme == "pow2")
        << "get_workspace_size: residual_scheme must be generic or pow2";
    for (const auto value :
         {max_m, logical_n, padded_n, padded_k, bucket_experts, total_experts, group_size}) {
      TVM_FFI_ICHECK_LE(value, std::numeric_limits<int32_t>::max())
          << "get_workspace_size: dimension exceeds int32";
    }
    uint64_t payload_cells = static_cast<uint64_t>(bucket_experts);
    constexpr uint64_t kMaxPayloadCells = std::numeric_limits<int32_t>::max();
    for (const uint64_t factor : {static_cast<uint64_t>(padded_k / kV3PayloadTileK),
                                  static_cast<uint64_t>(padded_n / kV3PayloadTileN)}) {
      TVM_FFI_ICHECK_LE(payload_cells, kMaxPayloadCells / factor)
          << "get_workspace_size: flattened payload TMA dimension exceeds int32";
      payload_cells *= factor;
    }
    uint64_t group_scale_cells = static_cast<uint64_t>(bucket_experts);
    for (const uint64_t factor : {static_cast<uint64_t>(padded_k / group_size),
                                  static_cast<uint64_t>(padded_n / kV3PayloadTileN)}) {
      TVM_FFI_ICHECK_LE(group_scale_cells, kMaxPayloadCells / factor)
          << "get_workspace_size: flattened group-scale TMA dimension exceeds int32";
      group_scale_cells *= factor;
    }
    TVM_FFI_ICHECK_LE(total_experts, (std::numeric_limits<int64_t>::max() - max_m) / 31)
        << "get_workspace_size: activation scale stride overflows";

    max_m_ = max_m;
    logical_n_ = static_cast<int32_t>(logical_n);
    padded_n_ = static_cast<int32_t>(padded_n);
    padded_k_ = static_cast<int32_t>(padded_k);
    bucket_experts_ = static_cast<int32_t>(bucket_experts);
    total_experts_ = static_cast<int32_t>(total_experts);
    group_size_ = static_cast<int32_t>(group_size);
    residual_scheme_ =
        residual_scheme == "generic" ? ResidualScheme::kGeneric : ResidualScheme::kPow2;
    padded_scale_stride_ = std::max<int64_t>(padded_offset(max_m, total_experts), 1);
    required_workspace_bytes_ = static_cast<int64_t>(schedule_workspace_size(bucket_experts_));
    workspace_ = nullptr;
    workspace_bytes_ = 0;
    workspace_queried_ = true;
    workspace_configured_ = false;
    counter_bank_ = 0;
    return required_workspace_bytes_;
  }

  template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
  void configure_kernel_variant(int opt_in_smem, W4A8KernelResources* bf16_resources,
                                W4A8KernelResources* fp32_resources) {
    const auto& variant = get_w4a8_kernel_variant<BlockM, BlockN, GroupSize, Scheme>();
    TVM_FFI_ICHECK_LE(variant.dynamic_smem_bytes, static_cast<size_t>(std::max(opt_in_smem, 0)))
        << "configure_workspace: W4A8 M" << BlockM << "N" << BlockN << " kernel requires "
        << variant.dynamic_smem_bytes << " shared-memory bytes, device permits " << opt_in_smem;
    const cudaError_t status = variant.configure(opt_in_smem, bf16_resources, fp32_resources);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: W4A8 M" << BlockM << "N" << BlockN
        << " kernel configuration failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GE(bf16_resources->blocks_per_sm, variant.min_blocks_per_sm)
        << "configure_workspace: BF16 W4A8 M" << BlockM << "N" << BlockN << " requires "
        << variant.min_blocks_per_sm << " blocks per SM, got " << bf16_resources->blocks_per_sm;
    if (variant.register_footprint_target > 0) {
      TVM_FFI_ICHECK_LE(bf16_resources->num_regs, variant.register_footprint_target)
          << "configure_workspace: BF16 W4A8 M" << BlockM << "N" << BlockN << " uses "
          << bf16_resources->num_regs << " registers per thread, target is "
          << variant.register_footprint_target;
    }
    // local_memory_bytes is recorded (kernel_resource_usage, bench provenance)
    // but not gated: cudaFuncGetAttributes reports it inconsistently across
    // toolchain/driver versions; the precise spill gate is the ptxas -v log.
    TVM_FFI_ICHECK_GE(fp32_resources->blocks_per_sm, 1)
        << "configure_workspace: FP32 debug W4A8 M" << BlockM << "N" << BlockN
        << " has zero occupancy";
  }

  template <int GroupSize, ResidualScheme Scheme>
  void configure_group_scheme(int opt_in_smem) {
    configure_kernel_variant<64, 64, GroupSize, Scheme>(opt_in_smem, &bf16_resources_[0][0],
                                                        &fp32_resources_[0][0]);
    configure_kernel_variant<64, 128, GroupSize, Scheme>(opt_in_smem, &bf16_resources_[0][1],
                                                         &fp32_resources_[0][1]);
    configure_kernel_variant<128, 64, GroupSize, Scheme>(opt_in_smem, &bf16_resources_[1][0],
                                                         &fp32_resources_[1][0]);
    configure_kernel_variant<128, 128, GroupSize, Scheme>(opt_in_smem, &bf16_resources_[1][1],
                                                          &fp32_resources_[1][1]);
  }

  void configure_static_variants(int opt_in_smem) {
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      if (group_size_ == 32)
        return configure_group_scheme<32, ResidualScheme::kGeneric>(opt_in_smem);
      if (group_size_ == 64)
        return configure_group_scheme<64, ResidualScheme::kGeneric>(opt_in_smem);
      return configure_group_scheme<128, ResidualScheme::kGeneric>(opt_in_smem);
    }
    if (group_size_ == 32) return configure_group_scheme<32, ResidualScheme::kPow2>(opt_in_smem);
    if (group_size_ == 64) return configure_group_scheme<64, ResidualScheme::kPow2>(opt_in_smem);
    return configure_group_scheme<128, ResidualScheme::kPow2>(opt_in_smem);
  }

  template <ResidualScheme Scheme>
  void configure_debug_decode_kernel(int opt_in_smem) {
    constexpr size_t kSmemBytes = debug_decode_smem_bytes();
    TVM_FFI_ICHECK_LE(kSmemBytes, static_cast<size_t>(std::max(opt_in_smem, 0)))
        << "configure_workspace: operand-byte debug kernel requires " << kSmemBytes
        << " shared-memory bytes, device permits " << opt_in_smem;
    auto kernel = debug_decode_v3_kernel<Scheme>;
    cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(kSmemBytes));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: operand-byte debug shared-memory opt-in failed: "
        << cudaGetErrorString(status);
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&debug_decode_blocks_per_sm_, kernel,
                                                           kProducerThreads, kSmemBytes);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: operand-byte debug occupancy query failed: "
        << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GT(debug_decode_blocks_per_sm_, 0)
        << "configure_workspace: operand-byte debug kernel has zero occupancy";
  }

  void configure_debug_decode_variant(int opt_in_smem) {
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      configure_debug_decode_kernel<ResidualScheme::kGeneric>(opt_in_smem);
    } else {
      configure_debug_decode_kernel<ResidualScheme::kPow2>(opt_in_smem);
    }
  }

  void configure_workspace(const TensorView& workspace) { configure_workspace_bank(workspace, 0); }

  void configure_workspace_bank(const TensorView& workspace, int64_t counter_bank) {
    TVM_FFI_ICHECK(workspace_queried_) << "configure_workspace: query workspace first";
    TVM_FFI_ICHECK_GE(counter_bank, 0) << "configure_workspace: counter_bank must be 0 or 1";
    TVM_FFI_ICHECK_LT(counter_bank, kNumCounterBanks)
        << "configure_workspace: counter_bank must be 0 or 1";
    CHECK_INPUT(workspace);
    CHECK_INPUT_TYPE(workspace, dl_uint8);
    TVM_FFI_ICHECK_GE(workspace.numel(), required_workspace_bytes_)
        << "configure_workspace: workspace is too small";
    TVM_FFI_ICHECK_EQ(
        reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(unsigned long long), 0)
        << "configure_workspace: workspace must be 8-byte aligned";

    ffi::CUDADeviceGuard device_guard(workspace.device().device_id);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, workspace.device().device_id);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: device query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0)
        << "configure_workspace: SM90 is required";
    int opt_in_smem = 0;
    status = cudaDeviceGetAttribute(&opt_in_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    workspace.device().device_id);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: opt-in shared-memory query failed: " << cudaGetErrorString(status);
    configure_static_variants(opt_in_smem);
    configure_debug_decode_variant(opt_in_smem);

    workspace_ = workspace.data_ptr();
    workspace_bytes_ = workspace.numel();
    workspace_device_ = workspace.device();
    sm_count_ = properties.multiProcessorCount;
    tma_encoder_ = get_tma_encoder();
    tma_cache_valid_ = false;
    counter_bank_ = static_cast<int32_t>(counter_bank);
    workspace_configured_ = true;
  }

  Array<int64_t> kernel_resource_usage(int64_t block_m, int64_t block_n, bool debug_fp32) const {
    TVM_FFI_ICHECK(workspace_configured_) << "kernel_resource_usage: configure workspace first";
    TVM_FFI_ICHECK(block_m == 64 || block_m == 128)
        << "kernel_resource_usage: block_m must be 64 or 128";
    TVM_FFI_ICHECK(block_n == 64 || block_n == 128)
        << "kernel_resource_usage: block_n must be 64 or 128";
    const int m_index = block_m == 64 ? 0 : 1;
    const int n_index = block_n == 64 ? 0 : 1;
    const auto& resource =
        debug_fp32 ? fp32_resources_[m_index][n_index] : bf16_resources_[m_index][n_index];
    std::vector<int64_t> values = {
        resource.blocks_per_sm,
        resource.num_regs,
        static_cast<int64_t>(resource.local_memory_bytes),
    };
    return Array(values);
  }

  void check_common_inputs(const TensorView& output, const TensorView& activation,
                           const TensorView& activation_scales, const TensorView& payload,
                           const TensorView& residual, const TensorView& group_scales,
                           const TensorView& alpha, const TensorView& expert_mapping,
                           const TensorView& offsets, bool fp32_output) const {
    CHECK_INPUT(output);
    CHECK_INPUT(activation);
    CHECK_INPUT(activation_scales);
    CHECK_INPUT(payload);
    CHECK_INPUT(residual);
    CHECK_INPUT(group_scales);
    CHECK_INPUT(alpha);
    CHECK_INPUT(expert_mapping);
    CHECK_INPUT(offsets);
    CHECK_DEVICE(output, activation);
    CHECK_DEVICE(activation_scales, activation);
    CHECK_DEVICE(payload, activation);
    CHECK_DEVICE(residual, activation);
    CHECK_DEVICE(group_scales, activation);
    CHECK_DEVICE(alpha, activation);
    CHECK_DEVICE(expert_mapping, activation);
    CHECK_DEVICE(offsets, activation);
    TVM_FFI_ICHECK_EQ(workspace_device_.device_type, activation.device().device_type)
        << "grouped_run: workspace device type mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, activation.device().device_id)
        << "grouped_run: workspace device mismatch";

    CHECK_INPUT_TYPE(activation, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(activation_scales, dl_float32);
    CHECK_INPUT_TYPE(payload, dl_uint8);
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      CHECK_INPUT_TYPE(residual, dl_bfloat16);
    } else {
      CHECK_INPUT_TYPE(residual, dl_int8);
    }
    CHECK_INPUT_TYPE(group_scales, dl_float32);
    CHECK_INPUT_TYPE(alpha, dl_float32);
    CHECK_INPUT_TYPE(expert_mapping, dl_int32);
    CHECK_INPUT_TYPE(offsets, dl_int64);
    if (fp32_output) {
      CHECK_INPUT_TYPE(output, dl_float32);
    } else {
      CHECK_INPUT_TYPE(output, dl_bfloat16);
    }

    CHECK_DIM(2, output);
    CHECK_DIM(2, activation);
    CHECK_DIM(2, activation_scales);
    CHECK_DIM(5, payload);
    CHECK_DIM(5, residual);
    CHECK_DIM(4, group_scales);
    CHECK_DIM(1, expert_mapping);
    CHECK_DIM(1, offsets);
    TVM_FFI_ICHECK(alpha.ndim() == 0 || alpha.ndim() == 1)
        << "grouped_run: alpha must be a scalar or 1D tensor";

    const int64_t rows = activation.size(0);
    TVM_FFI_ICHECK_EQ(output.size(0), rows) << "grouped_run: output row mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), logical_n_) << "grouped_run: output N mismatch";
    TVM_FFI_ICHECK_EQ(activation.size(1), padded_k_) << "grouped_run: activation K mismatch";
    TVM_FFI_ICHECK_LE(rows, max_m_) << "grouped_run: rows exceed configured max_m";
    TVM_FFI_ICHECK_EQ(activation_scales.size(0), padded_k_ / kBlockK)
        << "grouped_run: activation scale K-stage mismatch";
    TVM_FFI_ICHECK_EQ(activation_scales.size(1), padded_scale_stride_)
        << "grouped_run: activation scale padded stride mismatch";

    const int32_t k_tiles = padded_k_ / kV3PayloadTileK;
    const int32_t n_tiles = padded_n_ / kV3PayloadTileN;
    TVM_FFI_ICHECK_EQ(payload.size(0), bucket_experts_) << "grouped_run: payload expert mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(1), k_tiles) << "grouped_run: payload K-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(2), n_tiles) << "grouped_run: payload N-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(3), kV3PayloadTileN) << "grouped_run: payload tile N mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(4), kV3PackedBytesPerRow)
        << "grouped_run: payload packed K mismatch";
    for (int axis = 0; axis < 4; ++axis) {
      TVM_FFI_ICHECK_EQ(residual.size(axis), payload.size(axis))
          << "grouped_run: residual layout mismatch";
    }
    TVM_FFI_ICHECK_EQ(residual.size(4), kV3ResidualsPerPayloadTile)
        << "grouped_run: residual K-block mismatch";
    TVM_FFI_ICHECK_EQ(group_scales.size(0), bucket_experts_)
        << "grouped_run: group-scale expert mismatch";
    TVM_FFI_ICHECK_EQ(group_scales.size(1), padded_k_ / group_size_)
        << "grouped_run: group-scale K-group mismatch";
    TVM_FFI_ICHECK_EQ(group_scales.size(2), n_tiles) << "grouped_run: group-scale N-tile mismatch";
    TVM_FFI_ICHECK_EQ(group_scales.size(3), kV3PayloadTileN)
        << "grouped_run: group-scale tile N mismatch";
    if (alpha.ndim() == 1) {
      TVM_FFI_ICHECK_EQ(alpha.size(0), bucket_experts_)
          << "grouped_run: per-expert alpha shape mismatch";
    }
    TVM_FFI_ICHECK_EQ(expert_mapping.size(0), bucket_experts_)
        << "grouped_run: expert mapping shape mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(total_experts_) + 1)
        << "grouped_run: source offsets shape mismatch";
    TVM_FFI_ICHECK_GE(workspace_bytes_, required_workspace_bytes_)
        << "grouped_run: configured workspace is too small";

    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0)
        << "grouped_run: output must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(activation.data_ptr()) % 16, 0)
        << "grouped_run: activation must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16, 0)
        << "grouped_run: payload must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16, 0)
        << "grouped_run: residual must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(group_scales.data_ptr()) % 16, 0)
        << "grouped_run: group scales must be 16-byte aligned";
  }

  void prepare_schedule(const TensorView& activation, const TensorView& expert_mapping,
                        const TensorView& offsets, bool trusted_offsets, cudaStream_t stream) {
    auto* task_counters = schedule_task_counters(workspace_);
    auto* tile_prefix_m64 = schedule_tile_prefix(workspace_, MTileFamily::kM64, bucket_experts_);
    auto* tile_prefix_m128 = schedule_tile_prefix(workspace_, MTileFamily::kM128, bucket_experts_);
    prepare_grouped_schedule_kernel<<<1, 1, 0, stream>>>(
        static_cast<const int64_t*>(offsets.data_ptr()),
        static_cast<const int32_t*>(expert_mapping.data_ptr()), bucket_experts_, total_experts_,
        activation.size(0), task_counters, tile_prefix_m64, tile_prefix_m128, trusted_offsets);
    const cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "grouped_run: schedule preparation failed: " << cudaGetErrorString(status);
  }

  template <int BlockM>
  int32_t grid_blocks(int32_t n_tiles, int32_t active_blocks_per_sm, int64_t rows) const {
    static_assert(BlockM == 64 || BlockM == 128);
    if (n_tiles <= 0 || rows <= 0) return 0;
    const uint64_t row_tiles_upper = BlockM == 64
                                         ? static_cast<uint64_t>(bucket_experts_)
                                         : static_cast<uint64_t>(ceil_div_nonnegative(rows, 128)) +
                                               static_cast<uint64_t>(bucket_experts_);
    TVM_FFI_ICHECK_LE(static_cast<uint64_t>(n_tiles),
                      std::numeric_limits<uint64_t>::max() / row_tiles_upper)
        << "grouped_run: task count overflows uint64";
    const uint64_t task_upper = static_cast<uint64_t>(n_tiles) * row_tiles_upper;
    const uint64_t resident_blocks =
        static_cast<uint64_t>(std::max(sm_count_, 1)) * static_cast<uint64_t>(active_blocks_per_sm);
    return static_cast<int32_t>(std::max<uint64_t>(1, std::min(task_upper, resident_blocks)));
  }

  void update_tma_maps(const TensorView& activation, const TensorView& payload,
                       const TensorView& residual, const TensorView& group_scales) {
    TVM_FFI_ICHECK(tma_encoder_ != nullptr) << "grouped_run: TMA encoder is unavailable";
#if !W4A8_RESIDUAL_TMA
    residual_ptr_ = residual.data_ptr();
#endif
#if !W4A8_GROUP_SCALE_TMA
    group_scales_ptr_ = static_cast<const float*>(group_scales.data_ptr());
#endif
    const uintptr_t activation_address = reinterpret_cast<uintptr_t>(activation.data_ptr());
    const uintptr_t payload_address = reinterpret_cast<uintptr_t>(payload.data_ptr());
#if W4A8_RESIDUAL_TMA
    const uintptr_t residual_address = reinterpret_cast<uintptr_t>(residual.data_ptr());
#endif
#if W4A8_GROUP_SCALE_TMA
    const uintptr_t group_scale_address = reinterpret_cast<uintptr_t>(group_scales.data_ptr());
#endif
    const int64_t rows = activation.size(0);
    if (!tma_cache_valid_ || activation_address != cached_activation_address_ ||
        payload_address != cached_payload_address_ ||
#if W4A8_RESIDUAL_TMA
        residual_address != cached_residual_address_ ||
#endif
#if W4A8_GROUP_SCALE_TMA
        group_scale_address != cached_group_scale_address_ ||
#endif
        rows != cached_activation_rows_ || group_size_ != cached_group_size_ ||
        residual_scheme_ != cached_residual_scheme_ || padded_n_ != cached_padded_n_ ||
        padded_k_ != cached_padded_k_ || bucket_experts_ != cached_bucket_experts_) {
      activation_map_m64_ = make_activation_tma_map(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(activation.data_ptr())),
          static_cast<uint64_t>(rows), padded_k_, 64, tma_encoder_);
      activation_map_m128_ = make_activation_tma_map(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(activation.data_ptr())),
          static_cast<uint64_t>(rows), padded_k_, 128, tma_encoder_);
      payload_map_ = make_payload_tma_map(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(payload.data_ptr())), bucket_experts_,
          padded_k_ / kV3PayloadTileK, padded_n_ / kV3PayloadTileN, tma_encoder_);
#if W4A8_RESIDUAL_TMA
      residual_map_ = make_residual_tma_map(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(residual.data_ptr())), bucket_experts_,
          padded_k_ / kV3PayloadTileK, padded_n_ / kV3PayloadTileN, residual_scheme_, tma_encoder_);
#endif
#if W4A8_GROUP_SCALE_TMA
      group_scale_map_ = make_group_scale_tma_map(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(group_scales.data_ptr())),
          bucket_experts_, padded_k_ / group_size_, padded_n_ / kV3PayloadTileN, tma_encoder_);
#endif
      cached_activation_address_ = activation_address;
      cached_payload_address_ = payload_address;
#if W4A8_RESIDUAL_TMA
      cached_residual_address_ = residual_address;
#endif
#if W4A8_GROUP_SCALE_TMA
      cached_group_scale_address_ = group_scale_address;
#endif
      cached_activation_rows_ = rows;
      cached_group_size_ = group_size_;
      cached_residual_scheme_ = residual_scheme_;
      cached_padded_n_ = padded_n_;
      cached_padded_k_ = padded_k_;
      cached_bucket_experts_ = bucket_experts_;
      tma_cache_valid_ = true;
    }
  }

  template <bool DebugFp32, int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
  void launch_tile_family(const TensorView& output, const TensorView& activation_scales,
                          const TensorView& alpha, const TensorView& expert_mapping,
                          const TensorView& offsets, int32_t n_tiles, int32_t n_tile_begin,
                          int32_t active_blocks_per_sm, cudaStream_t stream) {
    const int32_t blocks = grid_blocks<BlockM>(n_tiles, active_blocks_per_sm, output.size(0));
    if (blocks == 0) return;
    constexpr MTileFamily kMFamily = m_tile_family<BlockM>();
    constexpr NTileFamily kNFamily = n_tile_family<BlockN>();
    const auto& variant = get_w4a8_kernel_variant<BlockM, BlockN, GroupSize, Scheme>();
    W4A8KernelLaunchParams params{
        output.data_ptr(),
        static_cast<const float*>(activation_scales.data_ptr()),
        static_cast<const float*>(alpha.data_ptr()),
        static_cast<const int32_t*>(expert_mapping.data_ptr()),
        static_cast<const int64_t*>(offsets.data_ptr()),
        schedule_tile_prefix(workspace_, kMFamily, bucket_experts_),
        schedule_task_counter(workspace_, counter_bank_, kMFamily, kNFamily),
        output.size(0),
        logical_n_,
        padded_n_,
        padded_k_,
        n_tiles,
        n_tile_begin,
        bucket_experts_,
        padded_scale_stride_,
        alpha.ndim() == 1,
#if !W4A8_RESIDUAL_TMA
        residual_ptr_,
#endif
#if !W4A8_GROUP_SCALE_TMA
        group_scales_ptr_,
#endif
        BlockM == 64 ? activation_map_m64_ : activation_map_m128_,
        payload_map_,
#if W4A8_RESIDUAL_TMA
        residual_map_,
#endif
#if W4A8_GROUP_SCALE_TMA
        group_scale_map_,
#endif
    };
    // An empty M-family prefix makes every persistent block exit at its first task mapping.
    const cudaError_t status = variant.launch(DebugFp32, blocks, stream, params);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "grouped_run: W4A8 M" << BlockM << "N" << BlockN
        << " kernel launch failed: " << cudaGetErrorString(status);
  }

  template <bool DebugFp32, int GroupSize, ResidualScheme Scheme>
  void launch_group_scheme(const TensorView& output, const TensorView& activation_scales,
                           const TensorView& alpha, const TensorView& expert_mapping,
                           const TensorView& offsets, cudaStream_t stream) {
    const int32_t n128_tiles = padded_n_ / 128;
    const bool has_n64_tail = padded_n_ % 128 != 0;
    const auto& resources = DebugFp32 ? fp32_resources_ : bf16_resources_;
    launch_tile_family<DebugFp32, 128, 128, GroupSize, Scheme>(
        output, activation_scales, alpha, expert_mapping, offsets, n128_tiles, 0,
        resources[1][1].blocks_per_sm, stream);
#if W4A8_SPLIT_M64_TAIL
    launch_tile_family<DebugFp32, 64, 128, GroupSize, Scheme>(
        output, activation_scales, alpha, expert_mapping, offsets, n128_tiles, 0,
        resources[0][1].blocks_per_sm, stream);
#endif
    if (has_n64_tail) {
      launch_tile_family<DebugFp32, 128, 64, GroupSize, Scheme>(
          output, activation_scales, alpha, expert_mapping, offsets, 1, n128_tiles * 2,
          resources[1][0].blocks_per_sm, stream);
#if W4A8_SPLIT_M64_TAIL
      launch_tile_family<DebugFp32, 64, 64, GroupSize, Scheme>(
          output, activation_scales, alpha, expert_mapping, offsets, 1, n128_tiles * 2,
          resources[0][0].blocks_per_sm, stream);
#endif
    }
  }

  template <bool DebugFp32>
  void dispatch_launch(const TensorView& output, const TensorView& activation_scales,
                       const TensorView& alpha, const TensorView& expert_mapping,
                       const TensorView& offsets, cudaStream_t stream) {
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      if (group_size_ == 32)
        return launch_group_scheme<DebugFp32, 32, ResidualScheme::kGeneric>(
            output, activation_scales, alpha, expert_mapping, offsets, stream);
      if (group_size_ == 64)
        return launch_group_scheme<DebugFp32, 64, ResidualScheme::kGeneric>(
            output, activation_scales, alpha, expert_mapping, offsets, stream);
      return launch_group_scheme<DebugFp32, 128, ResidualScheme::kGeneric>(
          output, activation_scales, alpha, expert_mapping, offsets, stream);
    }
    if (group_size_ == 32)
      return launch_group_scheme<DebugFp32, 32, ResidualScheme::kPow2>(
          output, activation_scales, alpha, expert_mapping, offsets, stream);
    if (group_size_ == 64)
      return launch_group_scheme<DebugFp32, 64, ResidualScheme::kPow2>(
          output, activation_scales, alpha, expert_mapping, offsets, stream);
    return launch_group_scheme<DebugFp32, 128, ResidualScheme::kPow2>(
        output, activation_scales, alpha, expert_mapping, offsets, stream);
  }

  template <bool DebugFp32>
  void run_impl(const TensorView& output, const TensorView& activation,
                const TensorView& activation_scales, const TensorView& payload,
                const TensorView& residual, const TensorView& group_scales, const TensorView& alpha,
                const TensorView& expert_mapping, const TensorView& offsets, bool trusted_offsets,
                bool prepare) {
    TVM_FFI_ICHECK(workspace_queried_) << "grouped_run: query workspace first";
    TVM_FFI_ICHECK(workspace_configured_) << "grouped_run: configure workspace first";
    check_common_inputs(output, activation, activation_scales, payload, residual, group_scales,
                        alpha, expert_mapping, offsets, DebugFp32);
    ffi::CUDADeviceGuard device_guard(activation.device().device_id);
    const cudaStream_t stream = get_stream(activation.device());
    if (prepare) {
      prepare_schedule(activation, expert_mapping, offsets, trusted_offsets, stream);
    }
    if (activation.size(0) == 0) return;
    update_tma_maps(activation, payload, residual, group_scales);
    dispatch_launch<DebugFp32>(output, activation_scales, alpha, expert_mapping, offsets, stream);
  }

  void grouped_run(const TensorView& output, const TensorView& activation,
                   const TensorView& activation_scales, const TensorView& payload,
                   const TensorView& residual, const TensorView& group_scales,
                   const TensorView& alpha, const TensorView& expert_mapping,
                   const TensorView& offsets, bool trusted_offsets) {
    TVM_FFI_ICHECK_EQ(counter_bank_, 0) << "grouped_run requires counter bank 0";
    run_impl<false>(output, activation, activation_scales, payload, residual, group_scales, alpha,
                    expert_mapping, offsets, trusted_offsets, true);
  }

  void grouped_run_prepared(const TensorView& output, const TensorView& activation,
                            const TensorView& activation_scales, const TensorView& payload,
                            const TensorView& residual, const TensorView& group_scales,
                            const TensorView& alpha, const TensorView& expert_mapping,
                            const TensorView& offsets, bool trusted_offsets) {
    // Counter bank 1 consumes one prepared schedule and requires preparation before reuse.
    TVM_FFI_ICHECK_EQ(counter_bank_, 1) << "grouped_run_prepared requires counter bank 1";
    TVM_FFI_ICHECK(trusted_offsets) << "grouped_run_prepared requires trusted offsets";
    run_impl<false>(output, activation, activation_scales, payload, residual, group_scales, alpha,
                    expert_mapping, offsets, trusted_offsets, false);
  }

  void debug_run_fp32(const TensorView& output, const TensorView& activation,
                      const TensorView& activation_scales, const TensorView& payload,
                      const TensorView& residual, const TensorView& group_scales,
                      const TensorView& alpha, const TensorView& expert_mapping,
                      const TensorView& offsets, bool trusted_offsets) {
    TVM_FFI_ICHECK_EQ(counter_bank_, 0) << "debug_run_fp32 requires counter bank 0";
    run_impl<true>(output, activation, activation_scales, payload, residual, group_scales, alpha,
                   expert_mapping, offsets, trusted_offsets, true);
  }

  template <ResidualScheme Scheme>
  void launch_debug_decode(const TensorView& output, const TensorView& payload,
                           const TensorView& residual, cudaStream_t stream) {
    using ResidualStorage = typename ResidualDecoder<Scheme>::Storage;
    const int64_t tasks = static_cast<int64_t>(bucket_experts_) * (padded_k_ / kBlockK) *
                          (padded_n_ / kV3PayloadTileN);
    if (tasks == 0) return;
    const int64_t resident_blocks =
        static_cast<int64_t>(std::max(sm_count_, 1)) * debug_decode_blocks_per_sm_;
    const int blocks = static_cast<int>(std::min(tasks, resident_blocks));
    debug_decode_v3_kernel<Scheme><<<blocks, kProducerThreads, debug_decode_smem_bytes(), stream>>>(
        static_cast<uint8_t*>(output.data_ptr()), static_cast<const uint8_t*>(payload.data_ptr()),
        static_cast<const ResidualStorage*>(residual.data_ptr()), bucket_experts_,
        padded_k_ / kV3PayloadTileK, padded_n_ / kV3PayloadTileN);
    const cudaError_t status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "debug_decode: kernel launch failed: " << cudaGetErrorString(status);
  }

  void debug_decode(const TensorView& output, const TensorView& payload,
                    const TensorView& residual) {
    TVM_FFI_ICHECK(workspace_queried_) << "debug_decode: query workspace first";
    TVM_FFI_ICHECK(workspace_configured_) << "debug_decode: configure workspace first";
    CHECK_INPUT(output);
    CHECK_INPUT(payload);
    CHECK_INPUT(residual);
    CHECK_INPUT_TYPE(output, dl_uint8);
    CHECK_INPUT_TYPE(payload, dl_uint8);
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      CHECK_INPUT_TYPE(residual, dl_bfloat16);
    } else {
      CHECK_INPUT_TYPE(residual, dl_int8);
    }
    CHECK_DEVICE(output, payload);
    CHECK_DEVICE(residual, payload);
    TVM_FFI_ICHECK_EQ(workspace_device_.device_type, payload.device().device_type)
        << "debug_decode: workspace device type mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, payload.device().device_id)
        << "debug_decode: workspace device mismatch";
    CHECK_DIM(5, output);
    CHECK_DIM(5, payload);
    CHECK_DIM(5, residual);
    const int32_t k_tiles = padded_k_ / kV3PayloadTileK;
    const int32_t n_tiles = padded_n_ / kV3PayloadTileN;
    TVM_FFI_ICHECK_EQ(output.size(0), bucket_experts_) << "debug_decode: output expert mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), k_tiles) << "debug_decode: output K-tile mismatch";
    TVM_FFI_ICHECK_EQ(output.size(2), n_tiles) << "debug_decode: output N-tile mismatch";
    TVM_FFI_ICHECK_EQ(output.size(3), kV3PayloadTileN) << "debug_decode: output tile N mismatch";
    TVM_FFI_ICHECK_EQ(output.size(4), kV3PayloadTileK) << "debug_decode: output tile K mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(0), bucket_experts_) << "debug_decode: payload expert mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(1), k_tiles) << "debug_decode: payload K-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(2), n_tiles) << "debug_decode: payload N-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(3), kV3PayloadTileN) << "debug_decode: payload tile N mismatch";
    TVM_FFI_ICHECK_EQ(payload.size(4), kV3PackedBytesPerRow)
        << "debug_decode: payload packed K mismatch";
    for (int axis = 0; axis < 4; ++axis) {
      TVM_FFI_ICHECK_EQ(residual.size(axis), payload.size(axis))
          << "debug_decode: residual layout mismatch";
    }
    TVM_FFI_ICHECK_EQ(residual.size(4), kV3ResidualsPerPayloadTile)
        << "debug_decode: residual K-block mismatch";
    ffi::CUDADeviceGuard device_guard(payload.device().device_id);
    const cudaStream_t stream = get_stream(payload.device());
    if (residual_scheme_ == ResidualScheme::kGeneric) {
      launch_debug_decode<ResidualScheme::kGeneric>(output, payload, residual, stream);
    } else {
      launch_debug_decode<ResidualScheme::kPow2>(output, payload, residual, stream);
    }
  }

  int64_t max_m_ = 0;
  int32_t logical_n_ = 0;
  int32_t padded_n_ = 0;
  int32_t padded_k_ = 0;
  int32_t bucket_experts_ = 0;
  int32_t total_experts_ = 0;
  int32_t group_size_ = 0;
  int32_t counter_bank_ = 0;
  int64_t padded_scale_stride_ = 0;
  ResidualScheme residual_scheme_ = ResidualScheme::kGeneric;
  int64_t required_workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  int64_t workspace_bytes_ = 0;
  DLDevice workspace_device_{kDLCPU, 0};
  int32_t sm_count_ = 0;
  std::array<std::array<W4A8KernelResources, 2>, 2> bf16_resources_{};
  std::array<std::array<W4A8KernelResources, 2>, 2> fp32_resources_{};
  int32_t debug_decode_blocks_per_sm_ = 0;
  PFN_cuTensorMapEncodeTiled_v12000 tma_encoder_ = nullptr;
  CUtensorMap activation_map_m64_{};
  CUtensorMap activation_map_m128_{};
  CUtensorMap payload_map_{};
#if W4A8_RESIDUAL_TMA
  CUtensorMap residual_map_{};
#endif
#if W4A8_GROUP_SCALE_TMA
  CUtensorMap group_scale_map_{};
#endif
#if !W4A8_RESIDUAL_TMA
  const void* residual_ptr_ = nullptr;
#endif
#if !W4A8_GROUP_SCALE_TMA
  const float* group_scales_ptr_ = nullptr;
#endif
  uintptr_t cached_activation_address_ = 0;
  uintptr_t cached_payload_address_ = 0;
#if W4A8_RESIDUAL_TMA
  uintptr_t cached_residual_address_ = 0;
#endif
#if W4A8_GROUP_SCALE_TMA
  uintptr_t cached_group_scale_address_ = 0;
#endif
  int64_t cached_activation_rows_ = 0;
  int32_t cached_group_size_ = 0;
  int32_t cached_padded_n_ = 0;
  int32_t cached_padded_k_ = 0;
  int32_t cached_bucket_experts_ = 0;
  ResidualScheme cached_residual_scheme_ = ResidualScheme::kGeneric;
  bool tma_cache_valid_ = false;
  bool workspace_queried_ = false;
  bool workspace_configured_ = false;
};

tvm::ffi::Module init() {
  auto runner = tvm::ffi::make_object<Sm90W4A8GroupedGemmRunner>();
  return tvm::ffi::Module(runner);
}

}  // namespace sm90_w4a8
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, flashinfer::sm90_w4a8::init);
