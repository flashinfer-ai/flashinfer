// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#include <cudaTypedefs.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "sm90_nvfp4_rs_kernel.cuh"
#include "tvm_ffi_utils.h"

#ifndef SM90_NVFP4_RS_USE_WGMMA
#define SM90_NVFP4_RS_USE_WGMMA 1
#endif

namespace flashinfer {
namespace sm90_nvfp4_rs {

using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int kCompiledNTactic = SM90_NVFP4_RS_N_TACTIC;
[[maybe_unused]] constexpr int kCompiledStages = SM90_NVFP4_RS_STAGES;
static_assert(kCompiledStages == 3);

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

template <typename T>
constexpr CUtensorMapDataType tma_data_type();

template <>
constexpr CUtensorMapDataType tma_data_type<uint8_t>() {
  return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

template <>
constexpr CUtensorMapDataType tma_data_type<__nv_bfloat16>() {
  return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}

template <typename T, uint32_t BoxDim0, uint32_t BoxDim1, CUtensorMapSwizzle Swizzle>
inline CUtensorMap make_tma_map(T* address, uint64_t global_dim_0, uint64_t global_dim_1,
                                uint64_t global_stride_1,
                                PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, __nv_bfloat16>);
  CUtensorMap tensor_map{};
  const uint64_t global_dims[2] = {global_dim_0, global_dim_1};
  const uint64_t global_strides[1] = {global_stride_1};
  const uint32_t box_dims[2] = {BoxDim0, BoxDim1};
  const uint32_t element_strides[2] = {1, 1};
  const CUresult result =
      encoder(&tensor_map, tma_data_type<T>(), 2, address, global_dims, global_strides, box_dims,
              element_strides, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, Swizzle,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "make_tma_map: cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

class Sm90Nvfp4RsGroupedGemmRunner final : public tvm::ffi::ModuleObj {
 public:
  const char* type_key() const { return "flashinfer.Sm90Nvfp4RsGroupedGemmRunner"; }

  const char* kind() const final { return "sm90_nvfp4_rs_grouped_gemm_runner"; }

  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "get_workspace_size") {
      return Function::FromTyped(
          [this](int64_t max_rows, int64_t num_experts, int64_t n, int64_t k) -> int64_t {
            return get_workspace_size(max_rows, num_experts, n, k);
          });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configure_workspace(workspace); });
    }
    if (name == "grouped_run") {
      return Function::FromTyped([this](TensorView output, TensorView activations,
                                        TensorView payload_rs, TensorView scales_rs,
                                        TensorView alpha, TensorView offsets,
                                        bool trusted_offsets) {
        grouped_run(output, activations, payload_rs, scales_rs, alpha, offsets, trusted_offsets);
      });
    }
    if (name == "grouped_run_padded") {
      return Function::FromTyped([this](TensorView output, TensorView activations,
                                        TensorView payload_rs, TensorView scales_rs,
                                        TensorView alpha, TensorView offsets,
                                        TensorView tile_prefix, bool trusted_offsets) {
        grouped_run_padded(output, activations, payload_rs, scales_rs, alpha, offsets, tile_prefix,
                           trusted_offsets);
      });
    }
#if SM90_NVFP4_RS_USE_WGMMA
    if (name == "oracle_run") {
      return Function::FromTyped([this](TensorView output, TensorView activations,
                                        TensorView canonical_weights, TensorView alpha,
                                        TensorView offsets, bool trusted_offsets) {
        oracle_run(output, activations, canonical_weights, alpha, offsets, trusted_offsets);
      });
    }
#endif
    return Function(nullptr);
  }

 private:
  static constexpr int64_t kCounterOffset = 0;
  static constexpr int64_t kWorkspaceBytes = 8;

  int64_t get_workspace_size(int64_t max_rows, int64_t num_experts, int64_t n, int64_t k) {
    TVM_FFI_ICHECK_GE(max_rows, 0) << "get_workspace_size: max_rows must be nonnegative";
    TVM_FFI_ICHECK_GT(num_experts, 0) << "get_workspace_size: num_experts must be positive";
    TVM_FFI_ICHECK_GT(n, 0) << "get_workspace_size: N must be positive";
    TVM_FFI_ICHECK_GT(k, 0) << "get_workspace_size: K must be positive";
#if SM90_NVFP4_RS_USE_WGMMA
    TVM_FFI_ICHECK_EQ(k % kStageK, 0) << "get_workspace_size: K must be divisible by " << kStageK;
#else
    TVM_FFI_ICHECK_EQ(k % kBlockK, 0) << "get_workspace_size: K must be divisible by " << kBlockK;
#endif
    TVM_FFI_ICHECK_EQ(n % kBlockM, 0) << "get_workspace_size: N must be divisible by 64";
    TVM_FFI_ICHECK_LE(max_rows, std::numeric_limits<int32_t>::max())
        << "get_workspace_size: max_rows exceeds TMA coordinates";
    TVM_FFI_ICHECK_LE(num_experts, std::numeric_limits<int32_t>::max())
        << "get_workspace_size: num_experts exceeds int32";
    TVM_FFI_ICHECK_LE(n, std::numeric_limits<int32_t>::max())
        << "get_workspace_size: N exceeds int32";
    TVM_FFI_ICHECK_LE(k, std::numeric_limits<int32_t>::max())
        << "get_workspace_size: K exceeds int32";

    const int64_t output_tiles = n / kBlockM;
    const int64_t k_tiles = k / kBlockK;
    TVM_FFI_ICHECK_LE(output_tiles, std::numeric_limits<int64_t>::max() / num_experts)
        << "get_workspace_size: payload tile count overflows";
    const int64_t expert_output_tiles = num_experts * output_tiles;
    TVM_FFI_ICHECK_LE(k_tiles, std::numeric_limits<int64_t>::max() / expert_output_tiles)
        << "get_workspace_size: payload tile count overflows";
    const int64_t payload_tiles = expert_output_tiles * k_tiles;
    TVM_FFI_ICHECK_LE(payload_tiles, std::numeric_limits<int32_t>::max() / 2)
        << "get_workspace_size: payload TMA coordinate exceeds int32";

    max_rows_ = max_rows;
    num_experts_ = static_cast<int32_t>(num_experts);
    n_ = static_cast<int32_t>(n);
    k_ = static_cast<int32_t>(k);
    payload_tiles_ = payload_tiles;
    required_workspace_bytes_ = kWorkspaceBytes;
    workspace_ = nullptr;
    workspace_bytes_ = 0;
    workspace_queried_ = true;
    workspace_configured_ = false;
    return required_workspace_bytes_;
  }

  void configure_workspace(TensorView workspace) {
    TVM_FFI_ICHECK(workspace_queried_) << "configure_workspace: query workspace first";
    CHECK_INPUT(workspace);
    CHECK_INPUT_TYPE(workspace, dl_uint8);
    TVM_FFI_ICHECK_GE(workspace.numel(), required_workspace_bytes_)
        << "configure_workspace: workspace is too small";
    TVM_FFI_ICHECK_EQ(
        reinterpret_cast<std::uintptr_t>(workspace.data_ptr()) % alignof(unsigned long long), 0)
        << "configure_workspace: workspace must be 8-byte aligned";

    ffi::CUDADeviceGuard device_guard(workspace.device().device_id);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, workspace.device().device_id);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: device query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0)
        << "configure_workspace: SM90 is required";

    int blocks_per_sm = 0;
#if SM90_NVFP4_RS_USE_WGMMA
    constexpr size_t kSmemBytes = wgmma_smem_bytes<kCompiledNTactic, kCompiledStages>();
    int max_optin_smem = 0;
    status = cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    workspace.device().device_id);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: opt-in shared-memory query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_LE(kSmemBytes, static_cast<size_t>(max_optin_smem))
        << "configure_workspace: kernel requires " << kSmemBytes
        << " shared-memory bytes, device permits " << max_optin_smem;
    auto kernel = grouped_rs_wgmma_kernel<kCompiledNTactic, kCompiledStages, false>;
    auto oracle_kernel = grouped_rs_wgmma_kernel<kCompiledNTactic, kCompiledStages, true>;
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kSmemBytes));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: dynamic shared-memory opt-in failed: "
        << cudaGetErrorString(status);
    status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, kSmemBytes);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: occupancy query failed: " << cudaGetErrorString(status);
    status = cudaFuncSetAttribute(oracle_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kSmemBytes));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: oracle dynamic shared-memory opt-in failed: "
        << cudaGetErrorString(status);
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oracle_blocks_per_sm_, oracle_kernel,
                                                           kThreads, kSmemBytes);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: oracle occupancy query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GT(oracle_blocks_per_sm_, 0)
        << "configure_workspace: oracle kernel has zero occupancy";
    tma_encoder_ = get_tma_encoder();
#else
    auto kernel = grouped_rs_scalar_kernel<kCompiledNTactic>;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, 0);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "configure_workspace: occupancy query failed: " << cudaGetErrorString(status);
#endif
    TVM_FFI_ICHECK_GT(blocks_per_sm, 0)
        << "configure_workspace: selected kernel has zero occupancy";

    workspace_ = workspace.data_ptr();
    workspace_bytes_ = workspace.numel();
    workspace_device_ = workspace.device();
    sm_count_ = properties.multiProcessorCount;
    blocks_per_sm_ = blocks_per_sm;
    tma_cache_valid_ = false;
    workspace_configured_ = true;
  }

  void check_devices(const TensorView& output, const TensorView& activations,
                     const TensorView& payload_rs, const TensorView& scales_rs,
                     const TensorView& alpha, const TensorView& offsets) const {
    CHECK_DEVICE(output, activations);
    CHECK_DEVICE(payload_rs, activations);
    CHECK_DEVICE(scales_rs, activations);
    CHECK_DEVICE(alpha, activations);
    CHECK_DEVICE(offsets, activations);
    TVM_FFI_ICHECK_EQ(workspace_device_.device_type, activations.device().device_type)
        << "grouped_run: workspace device type mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, activations.device().device_id)
        << "grouped_run: workspace device mismatch";
  }

  void check_shapes(const TensorView& output, const TensorView& activations,
                    const TensorView& payload_rs, const TensorView& scales_rs,
                    const TensorView& alpha, const TensorView& offsets) const {
    CHECK_DIM(2, output);
    CHECK_DIM(2, activations);
    CHECK_DIM(5, payload_rs);
    CHECK_DIM(4, scales_rs);
    CHECK_DIM(1, alpha);
    CHECK_DIM(1, offsets);
    const int64_t rows = activations.size(0);
    TVM_FFI_ICHECK_EQ(rows, output.size(0)) << "grouped_run: row mismatch";
    TVM_FFI_ICHECK_EQ(activations.size(1), k_) << "grouped_run: K mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), n_) << "grouped_run: N mismatch";
    TVM_FFI_ICHECK_EQ(alpha.size(0), num_experts_) << "grouped_run: expert count mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "grouped_run: offsets shape mismatch";
    TVM_FFI_ICHECK_LE(rows, max_rows_) << "grouped_run: rows exceed configured max";

    const int32_t output_tiles = n_ / kBlockM;
    const int32_t k_tiles = k_ / kBlockK;
    TVM_FFI_ICHECK_EQ(payload_rs.size(0), num_experts_) << "grouped_run: payload expert mismatch";
    TVM_FFI_ICHECK_EQ(payload_rs.size(1), output_tiles) << "grouped_run: payload N-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload_rs.size(2), k_tiles) << "grouped_run: payload K-tile mismatch";
    TVM_FFI_ICHECK_EQ(payload_rs.size(3), kRsThreads) << "grouped_run: payload thread mismatch";
    TVM_FFI_ICHECK_EQ(payload_rs.size(4), kRsBytesPerThread)
        << "grouped_run: payload bytes/thread mismatch";
    TVM_FFI_ICHECK_EQ(scales_rs.size(0), num_experts_) << "grouped_run: scale expert mismatch";
    TVM_FFI_ICHECK_EQ(scales_rs.size(1), output_tiles) << "grouped_run: scale N-tile mismatch";
    TVM_FFI_ICHECK_EQ(scales_rs.size(2), k_tiles) << "grouped_run: scale K-tile mismatch";
    TVM_FFI_ICHECK_EQ(scales_rs.size(3), kTileN) << "grouped_run: scale tile width mismatch";
  }

  void check_types(const TensorView& output, const TensorView& activations,
                   const TensorView& payload_rs, const TensorView& scales_rs,
                   const TensorView& alpha, const TensorView& offsets) const {
    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(activations, dl_bfloat16);
    CHECK_INPUT_TYPE(payload_rs, dl_uint8);
    CHECK_INPUT_TYPE(scales_rs, dl_float8_e4m3fn);
    CHECK_INPUT_TYPE(alpha, dl_float32);
    CHECK_INPUT_TYPE(offsets, dl_int64);
  }

  void grouped_run(TensorView output, TensorView activations, TensorView payload_rs,
                   TensorView scales_rs, TensorView alpha, TensorView offsets,
                   bool trusted_offsets) {
    grouped_run_impl(output, activations, payload_rs, scales_rs, alpha, offsets, nullptr,
                     trusted_offsets);
  }

  void grouped_run_padded(TensorView output, TensorView activations, TensorView payload_rs,
                          TensorView scales_rs, TensorView alpha, TensorView offsets,
                          TensorView tile_prefix, bool trusted_offsets) {
    grouped_run_impl(output, activations, payload_rs, scales_rs, alpha, offsets, &tile_prefix,
                     trusted_offsets);
  }

  void grouped_run_impl(TensorView output, TensorView activations, TensorView payload_rs,
                        TensorView scales_rs, TensorView alpha, TensorView offsets,
                        const TensorView* tile_prefix, bool trusted_offsets) {
    TVM_FFI_ICHECK(workspace_queried_) << "grouped_run: query workspace first";
    TVM_FFI_ICHECK(workspace_configured_) << "grouped_run: configure workspace first";
    CHECK_INPUT(output);
    CHECK_INPUT(activations);
    CHECK_INPUT(payload_rs);
    CHECK_INPUT(scales_rs);
    CHECK_INPUT(alpha);
    CHECK_INPUT(offsets);
    check_devices(output, activations, payload_rs, scales_rs, alpha, offsets);
    check_shapes(output, activations, payload_rs, scales_rs, alpha, offsets);
    check_types(output, activations, payload_rs, scales_rs, alpha, offsets);
    if (tile_prefix != nullptr) {
      const TensorView& prefix = *tile_prefix;
      CHECK_INPUT(prefix);
      CHECK_INPUT_TYPE(prefix, dl_int64);
      CHECK_DIM(1, prefix);
      CHECK_DEVICE(prefix, activations);
      TVM_FFI_ICHECK_EQ(prefix.size(0), static_cast<int64_t>(num_experts_) + 1)
          << "grouped_run_padded: prefix shape mismatch";
#if !SM90_NVFP4_RS_USE_WGMMA
      TVM_FFI_THROW(NotImplementedError) << "grouped_run_padded requires rs_wgmma";
#endif
    }
    TVM_FFI_ICHECK_GE(workspace_bytes_, required_workspace_bytes_)
        << "grouped_run: configured workspace is too small";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(int4), 0)
        << "grouped_run: output must be 16-byte aligned";

    ffi::CUDADeviceGuard device_guard(activations.device().device_id);
    const cudaStream_t stream = get_stream(activations.device());
    const int64_t rows = activations.size(0);
    auto* workspace_bytes = static_cast<uint8_t*>(workspace_);
    auto* task_counter = reinterpret_cast<unsigned long long*>(workspace_bytes + kCounterOffset);
    cudaError_t launch_status = cudaSuccess;

    if (!trusted_offsets) {
      if (tile_prefix == nullptr) {
        validate_offsets_kernel<<<1, 1, 0, stream>>>(
            static_cast<const int64_t*>(offsets.data_ptr()), num_experts_, rows);
      } else {
        validate_padded_schedule_kernel<kCompiledNTactic><<<1, 1, 0, stream>>>(
            static_cast<const int64_t*>(offsets.data_ptr()),
            static_cast<const int64_t*>(tile_prefix->data_ptr()), num_experts_, rows);
      }
      launch_status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
          << "grouped_run: offset validation launch failed: " << cudaGetErrorString(launch_status);
    }

    launch_status = cudaMemsetAsync(task_counter, 0, sizeof(unsigned long long), stream);
    TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
        << "grouped_run: counter reset failed: " << cudaGetErrorString(launch_status);
    if (rows == 0) {
      return;
    }

    const uint64_t output_tiles = static_cast<uint64_t>(n_ / kBlockM);
    const uint64_t row_tiles_upper =
        static_cast<uint64_t>(ceil_div_nonnegative(rows, static_cast<int64_t>(kCompiledNTactic))) +
        static_cast<uint64_t>(num_experts_);
    TVM_FFI_ICHECK_LE(output_tiles, std::numeric_limits<uint64_t>::max() / row_tiles_upper)
        << "grouped_run: task count overflows uint64";
    const uint64_t task_upper = output_tiles * row_tiles_upper;
    const uint64_t resident_blocks =
        static_cast<uint64_t>(std::max(sm_count_, 1)) * static_cast<uint64_t>(blocks_per_sm_);
    const int grid_blocks =
        static_cast<int>(std::max<uint64_t>(1, std::min(task_upper, resident_blocks)));

#if SM90_NVFP4_RS_USE_WGMMA
    TVM_FFI_ICHECK(tma_encoder_ != nullptr) << "grouped_run: TMA encoder is unavailable";
    const uintptr_t activation_address = reinterpret_cast<uintptr_t>(activations.data_ptr());
    const uintptr_t payload_address = reinterpret_cast<uintptr_t>(payload_rs.data_ptr());
    const uintptr_t scale_address = reinterpret_cast<uintptr_t>(scales_rs.data_ptr());
    TVM_FFI_ICHECK_EQ(activation_address % 16, 0)
        << "grouped_run: activations must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(payload_address % 16, 0) << "grouped_run: payload must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(scale_address % 16, 0) << "grouped_run: scales must be 16-byte aligned";
    if (!tma_cache_valid_ || activation_address != cached_activation_address_ ||
        payload_address != cached_payload_address_ || scale_address != cached_scale_address_ ||
        rows != cached_activation_rows_) {
      activation_map_ = make_tma_map<__nv_bfloat16, kBlockK, kCompiledNTactic,
                                     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B>(
          const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(activations.data_ptr())),
          static_cast<uint64_t>(k_), static_cast<uint64_t>(rows),
          static_cast<uint64_t>(k_) * sizeof(__nv_bfloat16), tma_encoder_);
      payload_map_ = make_tma_map<uint8_t, 256, 2 * kStageSubtiles,
                                  CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE>(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(payload_rs.data_ptr())), 256,
          static_cast<uint64_t>(payload_tiles_) * 2, 256, tma_encoder_);
      scale_map_ = make_tma_map<uint8_t, kTileN, kStageSubtiles,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE>(
          const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(scales_rs.data_ptr())), kTileN,
          static_cast<uint64_t>(payload_tiles_), kTileN, tma_encoder_);
      cached_activation_address_ = activation_address;
      cached_payload_address_ = payload_address;
      cached_scale_address_ = scale_address;
      cached_activation_rows_ = rows;
      tma_cache_valid_ = true;
    }
    constexpr size_t kSmemBytes = wgmma_smem_bytes<kCompiledNTactic, kCompiledStages>();
    grouped_rs_wgmma_kernel<kCompiledNTactic, kCompiledStages, false>
        <<<grid_blocks, kThreads, kSmemBytes, stream>>>(
            static_cast<__nv_bfloat16*>(output.data_ptr()), nullptr,
            static_cast<const float*>(alpha.data_ptr()),
            static_cast<const int64_t*>(offsets.data_ptr()),
            tile_prefix == nullptr ? nullptr : static_cast<const int64_t*>(tile_prefix->data_ptr()),
            task_counter, rows, n_, k_, num_experts_, activation_map_, payload_map_, scale_map_);
#else
    grouped_rs_scalar_kernel<kCompiledNTactic><<<grid_blocks, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<const __nv_bfloat16*>(activations.data_ptr()),
        static_cast<const uint8_t*>(payload_rs.data_ptr()),
        reinterpret_cast<const uint8_t*>(scales_rs.data_ptr()),
        static_cast<const float*>(alpha.data_ptr()),
        static_cast<const int64_t*>(offsets.data_ptr()), task_counter, rows, n_, k_, num_experts_);
#endif
    launch_status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
        << "grouped_run: kernel launch failed: " << cudaGetErrorString(launch_status);
  }

#if SM90_NVFP4_RS_USE_WGMMA
  void oracle_run(TensorView output, TensorView activations, TensorView canonical_weights,
                  TensorView alpha, TensorView offsets, bool trusted_offsets) {
    TVM_FFI_ICHECK(workspace_queried_) << "oracle_run: query workspace first";
    TVM_FFI_ICHECK(workspace_configured_) << "oracle_run: configure workspace first";
    CHECK_INPUT(output);
    CHECK_INPUT(activations);
    CHECK_INPUT(canonical_weights);
    CHECK_INPUT(alpha);
    CHECK_INPUT(offsets);
    CHECK_DEVICE(output, activations);
    CHECK_DEVICE(canonical_weights, activations);
    CHECK_DEVICE(alpha, activations);
    CHECK_DEVICE(offsets, activations);
    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(activations, dl_bfloat16);
    CHECK_INPUT_TYPE(canonical_weights, dl_bfloat16);
    CHECK_INPUT_TYPE(alpha, dl_float32);
    CHECK_INPUT_TYPE(offsets, dl_int64);
    CHECK_DIM(2, output);
    CHECK_DIM(2, activations);
    CHECK_DIM(3, canonical_weights);
    CHECK_DIM(1, alpha);
    CHECK_DIM(1, offsets);
    TVM_FFI_ICHECK_EQ(workspace_device_.device_type, activations.device().device_type)
        << "oracle_run: workspace device type mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, activations.device().device_id)
        << "oracle_run: workspace device mismatch";
    const int64_t rows = activations.size(0);
    TVM_FFI_ICHECK_EQ(rows, output.size(0)) << "oracle_run: row mismatch";
    TVM_FFI_ICHECK_EQ(activations.size(1), k_) << "oracle_run: K mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), n_) << "oracle_run: N mismatch";
    TVM_FFI_ICHECK_EQ(canonical_weights.size(0), num_experts_)
        << "oracle_run: weight expert mismatch";
    TVM_FFI_ICHECK_EQ(canonical_weights.size(1), n_) << "oracle_run: weight N mismatch";
    TVM_FFI_ICHECK_EQ(canonical_weights.size(2), k_) << "oracle_run: weight K mismatch";
    TVM_FFI_ICHECK_EQ(alpha.size(0), num_experts_) << "oracle_run: alpha shape mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "oracle_run: offsets shape mismatch";
    TVM_FFI_ICHECK_LE(rows, max_rows_) << "oracle_run: rows exceed configured max";
    TVM_FFI_ICHECK_GE(workspace_bytes_, required_workspace_bytes_)
        << "oracle_run: configured workspace is too small";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(int4), 0)
        << "oracle_run: output must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(activations.data_ptr()) % 16, 0)
        << "oracle_run: activations must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(canonical_weights.data_ptr()) % alignof(uint32_t),
                      0)
        << "oracle_run: canonical weights must be 4-byte aligned";

    ffi::CUDADeviceGuard device_guard(activations.device().device_id);
    const cudaStream_t stream = get_stream(activations.device());
    auto* workspace_bytes = static_cast<uint8_t*>(workspace_);
    auto* task_counter = reinterpret_cast<unsigned long long*>(workspace_bytes + kCounterOffset);
    cudaError_t launch_status = cudaSuccess;
    if (!trusted_offsets) {
      validate_offsets_kernel<<<1, 1, 0, stream>>>(static_cast<const int64_t*>(offsets.data_ptr()),
                                                   num_experts_, rows);
      launch_status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
          << "oracle_run: offset validation launch failed: " << cudaGetErrorString(launch_status);
    }
    launch_status = cudaMemsetAsync(task_counter, 0, sizeof(unsigned long long), stream);
    TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
        << "oracle_run: counter reset failed: " << cudaGetErrorString(launch_status);
    if (rows == 0) {
      return;
    }

    const uint64_t output_tiles = static_cast<uint64_t>(n_ / kBlockM);
    const uint64_t row_tiles_upper =
        static_cast<uint64_t>(ceil_div_nonnegative(rows, static_cast<int64_t>(kCompiledNTactic))) +
        static_cast<uint64_t>(num_experts_);
    TVM_FFI_ICHECK_LE(output_tiles, std::numeric_limits<uint64_t>::max() / row_tiles_upper)
        << "oracle_run: task count overflows uint64";
    const uint64_t task_upper = output_tiles * row_tiles_upper;
    const uint64_t resident_blocks = static_cast<uint64_t>(std::max(sm_count_, 1)) *
                                     static_cast<uint64_t>(oracle_blocks_per_sm_);
    const int grid_blocks =
        static_cast<int>(std::max<uint64_t>(1, std::min(task_upper, resident_blocks)));
    TVM_FFI_ICHECK(tma_encoder_ != nullptr) << "oracle_run: TMA encoder is unavailable";
    const CUtensorMap activation_map = make_tma_map<__nv_bfloat16, kBlockK, kCompiledNTactic,
                                                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B>(
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(activations.data_ptr())),
        static_cast<uint64_t>(k_), static_cast<uint64_t>(rows),
        static_cast<uint64_t>(k_) * sizeof(__nv_bfloat16), tma_encoder_);
    const CUtensorMap unused_payload_map{};
    const CUtensorMap unused_scale_map{};
    constexpr size_t kSmemBytes = wgmma_smem_bytes<kCompiledNTactic, kCompiledStages>();
    grouped_rs_wgmma_kernel<kCompiledNTactic, kCompiledStages, true>
        <<<grid_blocks, kThreads, kSmemBytes, stream>>>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(canonical_weights.data_ptr()),
            static_cast<const float*>(alpha.data_ptr()),
            static_cast<const int64_t*>(offsets.data_ptr()), nullptr, task_counter, rows, n_, k_,
            num_experts_, activation_map, unused_payload_map, unused_scale_map);
    launch_status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(launch_status, cudaSuccess)
        << "oracle_run: kernel launch failed: " << cudaGetErrorString(launch_status);
  }
#endif

  int64_t max_rows_ = 0;
  int64_t payload_tiles_ = 0;
  int32_t num_experts_ = 0;
  int32_t n_ = 0;
  int32_t k_ = 0;
  int64_t required_workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  int64_t workspace_bytes_ = 0;
  DLDevice workspace_device_{kDLCPU, 0};
  int sm_count_ = 0;
  int blocks_per_sm_ = 0;
  int oracle_blocks_per_sm_ = 0;
  PFN_cuTensorMapEncodeTiled_v12000 tma_encoder_ = nullptr;
  CUtensorMap activation_map_{};
  CUtensorMap payload_map_{};
  CUtensorMap scale_map_{};
  uintptr_t cached_activation_address_ = 0;
  uintptr_t cached_payload_address_ = 0;
  uintptr_t cached_scale_address_ = 0;
  int64_t cached_activation_rows_ = 0;
  bool tma_cache_valid_ = false;
  bool workspace_queried_ = false;
  bool workspace_configured_ = false;
};

tvm::ffi::Module init() {
  auto runner = tvm::ffi::make_object<Sm90Nvfp4RsGroupedGemmRunner>();
  return tvm::ffi::Module(runner);
}

}  // namespace sm90_nvfp4_rs
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, flashinfer::sm90_nvfp4_rs::init);
