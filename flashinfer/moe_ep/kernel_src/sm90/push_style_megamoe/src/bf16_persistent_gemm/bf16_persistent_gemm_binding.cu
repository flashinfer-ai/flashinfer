// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include <cudaTypedefs.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "bf16_persistent_gemm.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::sm90_push_bf16_persistent {

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

inline bool is_bfloat16(DLDataType dtype) { return encode_dlpack_dtype(dtype) == bfloat16_code; }
inline bool is_int64(DLDataType dtype) { return encode_dlpack_dtype(dtype) == int64_code; }
inline bool is_uint8(DLDataType dtype) { return encode_dlpack_dtype(dtype) == uint8_code; }

inline PFN_cuTensorMapEncodeTiled_v12000 get_tma_encoder() {
  cudaDriverEntryPointQueryResult driver_status;
  void* encoder = nullptr;
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
  cudaError_t const status = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &encoder, 12000, cudaEnableDefault, &driver_status);
#else
  cudaError_t const status = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &encoder,
                                                     cudaEnableDefault, &driver_status);
#endif
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  TVM_FFI_ICHECK_EQ(driver_status, cudaDriverEntryPointSuccess)
      << "cuTensorMapEncodeTiled is unavailable";
  TVM_FFI_ICHECK(encoder != nullptr) << "cuTensorMapEncodeTiled is null";
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encoder);
}

template <uint32_t BoxRows>
inline CUtensorMap make_bf16_tma_map(__nv_bfloat16* address, uint64_t rows,
                                     PFN_cuTensorMapEncodeTiled_v12000 encoder) {
  CUtensorMap tensor_map{};
  uint64_t const global_dims[2] = {static_cast<uint64_t>(kShapeK), rows};
  uint64_t const global_strides[1] = {static_cast<uint64_t>(kShapeK) * sizeof(__nv_bfloat16)};
  uint32_t const box_dims[2] = {kMmaK, BoxRows};
  uint32_t const element_strides[2] = {1, 1};
  CUresult const result =
      encoder(&tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, address,
              global_dims, global_strides, box_dims, element_strides,
              CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed: " << static_cast<int>(result);
  return tensor_map;
}

struct FamilyMaps {
  CUtensorMap activation{};
  CUtensorMap weight{};
  uintptr_t activation_address = 0;
  uintptr_t weight_address = 0;
  int64_t rows = -1;
  bool valid = false;
};

class Sm90PushBf16PersistentRunner final : public tvm::ffi::ModuleObj {
 public:
  const char* type_key() const { return "flashinfer.Sm90PushBf16PersistentRunner"; }
  const char* kind() const final { return "sm90_push_bf16_persistent_offsets_gemm_runner"; }

  Optional<Function> GetFunction(tvm::ffi::String const& name) final {
    if (name == "get_workspace_size") {
      return Function::FromTyped([this](int64_t max_rows, int64_t num_experts, int64_t shape_n,
                                        int64_t shape_k, int64_t device_id) {
        return get_workspace_size(max_rows, num_experts, shape_n, shape_k, device_id);
      });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configure_workspace(workspace); });
    }
    if (name == "kernel_resource_usage") {
      return Function::FromTyped(
          [this](int64_t block_m) { return kernel_resource_usage(block_m); });
    }
    if (name == "grouped_run") {
      return Function::FromTyped([this](TensorView output, TensorView activation,
                                        TensorView weights, TensorView offsets,
                                        bool trusted_offsets) {
        grouped_run(output, activation, weights, offsets, trusted_offsets);
      });
    }
    return Function(nullptr);
  }

 private:
  static constexpr int64_t kCounterCount =
      static_cast<int64_t>(kEnableM64) + static_cast<int64_t>(kEnableM128);
  static constexpr int64_t kWorkspaceBytes =
      std::max<int64_t>(kCounterCount * sizeof(unsigned long long), sizeof(unsigned long long));

  int64_t get_workspace_size(int64_t max_rows, int64_t num_experts, int64_t shape_n,
                             int64_t shape_k, int64_t device_id) {
    TVM_FFI_ICHECK_GE(max_rows, 0) << "max_rows must be nonnegative";
    TVM_FFI_ICHECK_GT(num_experts, 0) << "num_experts must be positive";
    TVM_FFI_ICHECK_EQ(shape_n, kShapeN) << "shape_n differs from the compiled specialization";
    TVM_FFI_ICHECK_EQ(shape_k, kShapeK) << "shape_k differs from the compiled specialization";
    TVM_FFI_ICHECK_GE(device_id, 0) << "device_id must be nonnegative";
    TVM_FFI_ICHECK_LE(max_rows, std::numeric_limits<int32_t>::max()) << "max_rows exceeds int32";
    TVM_FFI_ICHECK_LE(num_experts, std::numeric_limits<int32_t>::max())
        << "num_experts exceeds int32";
#if SM90_PUSH_BF16_FAMILY_MASK & 1
    TVM_FFI_ICHECK_EQ(kShapeN % M64Traits::kBlockN, 0)
        << "compiled N is not divisible by the M64 BlockN";
    TVM_FFI_ICHECK_EQ(kShapeK % M64Traits::kBlockK, 0)
        << "compiled K is not divisible by the M64 BlockK";
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
    TVM_FFI_ICHECK_EQ(kShapeN % M128Traits::kBlockN, 0)
        << "compiled N is not divisible by the M128 BlockN";
    TVM_FFI_ICHECK_EQ(kShapeK % M128Traits::kBlockK, 0)
        << "compiled K is not divisible by the M128 BlockK";
#endif

    ffi::CUDADeviceGuard device_guard(static_cast<int32_t>(device_id));
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, static_cast<int32_t>(device_id));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0) << "SM90 is required";
    int32_t opt_in_smem_capacity = 0;
    status = cudaDeviceGetAttribute(&opt_in_smem_capacity, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    static_cast<int32_t>(device_id));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);

#if SM90_PUSH_BF16_FAMILY_MASK & 1
    status = configure_and_query_kernel<M64Traits>(&m64_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "M64 persistent kernel query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_LE(m64_resources_.dynamic_smem_bytes, opt_in_smem_capacity)
        << "M64 persistent tactic exceeds shared-memory capacity";
    TVM_FFI_ICHECK_GT(m64_resources_.blocks_per_sm, 0)
        << "M64 persistent tactic has zero resident blocks";
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
    status = configure_and_query_kernel<M128Traits>(&m128_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "M128 persistent kernel query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_LE(m128_resources_.dynamic_smem_bytes, opt_in_smem_capacity)
        << "M128 persistent tactic exceeds shared-memory capacity";
    TVM_FFI_ICHECK_GT(m128_resources_.blocks_per_sm, 0)
        << "M128 persistent tactic has zero resident blocks";
#endif

    max_rows_ = max_rows;
    num_experts_ = static_cast<int32_t>(num_experts);
    device_id_ = static_cast<int32_t>(device_id);
    sm_count_ = properties.multiProcessorCount;
    tma_encoder_ = get_tma_encoder();
    queried_ = true;
    configured_ = false;
    workspace_ = nullptr;
    m64_maps_.valid = false;
    m128_maps_.valid = false;
    return kWorkspaceBytes;
  }

  void configure_workspace(TensorView const& workspace) {
    TVM_FFI_ICHECK(queried_) << "call get_workspace_size first";
    CHECK_INPUT(workspace);
    TVM_FFI_ICHECK(is_uint8(workspace.dtype())) << "workspace must be uint8";
    TVM_FFI_ICHECK_GE(workspace.numel(), kWorkspaceBytes) << "workspace is too small";
    TVM_FFI_ICHECK_EQ(workspace.device().device_id, device_id_) << "workspace device mismatch";
    TVM_FFI_ICHECK_EQ(
        reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(unsigned long long), 0)
        << "workspace must be eight-byte aligned";
    workspace_ = reinterpret_cast<unsigned long long*>(workspace.data_ptr());
    workspace_device_ = workspace.device();
    configured_ = true;
  }

  Array<int64_t> kernel_resource_usage(int64_t block_m) const {
    TVM_FFI_ICHECK(queried_) << "call get_workspace_size first";
    KernelResources const* resources = nullptr;
    int64_t cluster_m = 0;
    int64_t threads = 0;
    if (block_m == 64 && kEnableM64) {
      resources = &m64_resources_;
      cluster_m = M64Traits::kClusterM;
      threads = M64Traits::kThreads;
    } else if (block_m == 128 && kEnableM128) {
      resources = &m128_resources_;
      cluster_m = M128Traits::kClusterM;
      threads = M128Traits::kThreads;
    }
    TVM_FFI_ICHECK(resources != nullptr) << "requested M-tile family is not compiled";
    std::vector<int64_t> values = {
        resources->blocks_per_sm,      resources->num_regs, resources->local_memory_bytes,
        resources->dynamic_smem_bytes, cluster_m,           threads,
    };
    return Array(values);
  }

  template <typename Traits>
  void ensure_maps(TensorView const& activation, TensorView const& weights, int64_t rows,
                   FamilyMaps& maps) {
    uintptr_t const activation_address = reinterpret_cast<uintptr_t>(activation.data_ptr());
    uintptr_t const weight_address = reinterpret_cast<uintptr_t>(weights.data_ptr());
    if (maps.valid && maps.activation_address == activation_address &&
        maps.weight_address == weight_address && maps.rows == rows)
      return;
    TVM_FFI_ICHECK(tma_encoder_ != nullptr) << "TMA encoder is unavailable";
    if constexpr (Traits::kSwapAB) {
      maps.activation = make_bf16_tma_map<Traits::kBlockM>(
          const_cast<__nv_bfloat16*>(static_cast<__nv_bfloat16 const*>(activation.data_ptr())),
          static_cast<uint64_t>(rows), tma_encoder_);
      maps.weight = make_bf16_tma_map<Traits::kBlockN>(
          const_cast<__nv_bfloat16*>(static_cast<__nv_bfloat16 const*>(weights.data_ptr())),
          static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kShapeN), tma_encoder_);
    } else {
      maps.activation = make_bf16_tma_map<Traits::kBlockM>(
          const_cast<__nv_bfloat16*>(static_cast<__nv_bfloat16 const*>(activation.data_ptr())),
          static_cast<uint64_t>(rows), tma_encoder_);
      maps.weight = make_bf16_tma_map<Traits::kBlockN>(
          const_cast<__nv_bfloat16*>(static_cast<__nv_bfloat16 const*>(weights.data_ptr())),
          static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kShapeN), tma_encoder_);
    }
    maps.activation_address = activation_address;
    maps.weight_address = weight_address;
    maps.rows = rows;
    maps.valid = true;
  }

  void grouped_run(TensorView const& output, TensorView const& activation,
                   TensorView const& weights, TensorView const& offsets, bool trusted_offsets) {
    TVM_FFI_ICHECK(queried_ && configured_)
        << "query and configure this runner before launching GEMM";
    CHECK_INPUT(output);
    CHECK_INPUT(activation);
    CHECK_INPUT(weights);
    CHECK_INPUT(offsets);
    CHECK_DEVICE(output, activation);
    CHECK_DEVICE(weights, activation);
    CHECK_DEVICE(offsets, activation);
    CHECK_DIM(2, output);
    CHECK_DIM(2, activation);
    CHECK_DIM(3, weights);
    CHECK_DIM(1, offsets);
    TVM_FFI_ICHECK_EQ(activation.device().device_id, device_id_) << "activation device mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, device_id_) << "workspace device mismatch";
    TVM_FFI_ICHECK(is_bfloat16(output.dtype())) << "output must be bfloat16";
    TVM_FFI_ICHECK(is_bfloat16(activation.dtype())) << "activation must be bfloat16";
    TVM_FFI_ICHECK(is_bfloat16(weights.dtype())) << "weights must be bfloat16";
    TVM_FFI_ICHECK(is_int64(offsets.dtype())) << "offsets must be int64";
    int64_t const rows = activation.size(0);
    TVM_FFI_ICHECK_LE(rows, max_rows_) << "activation exceeds configured max_rows";
    TVM_FFI_ICHECK_EQ(activation.size(1), kShapeK) << "activation K mismatch";
    TVM_FFI_ICHECK_EQ(output.size(0), rows) << "output row mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), kShapeN) << "output N mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(0), num_experts_) << "weight expert mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(1), kShapeN) << "weight N mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(2), kShapeK) << "weight K mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "offsets must contain num_experts plus one entries";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(int4), 0)
        << "output must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(activation.data_ptr()) % 16, 0)
        << "activation must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16, 0)
        << "weights must be 16-byte aligned";

    ffi::CUDADeviceGuard device_guard(device_id_);
    cudaStream_t const stream = get_stream(activation.device());
    if (!trusted_offsets) {
      offsets_preflight_kernel<<<1, 32, 0, stream>>>(
          static_cast<int64_t const*>(offsets.data_ptr()), num_experts_, rows);
      cudaError_t const status = cudaGetLastError();
      TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }
    cudaError_t status = cudaMemsetAsync(workspace_, 0, kWorkspaceBytes, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (rows == 0) return;

    auto* output_ptr = static_cast<__nv_bfloat16*>(output.data_ptr());
    auto const* activation_ptr = static_cast<__nv_bfloat16 const*>(activation.data_ptr());
    auto const* weight_ptr = static_cast<__nv_bfloat16 const*>(weights.data_ptr());
    auto const* offsets_ptr = static_cast<int64_t const*>(offsets.data_ptr());
    int counter_slot = 0;
#if SM90_PUSH_BF16_FAMILY_MASK & 2
    ensure_maps<M128Traits>(activation, weights, rows, m128_maps_);
    status = launch_persistent_family<M128Traits>(
        output_ptr, activation_ptr, weight_ptr, offsets_ptr, workspace_ + counter_slot, rows,
        num_experts_, sm_count_, m128_resources_, m128_maps_.activation, m128_maps_.weight, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "M128 persistent launch failed: " << cudaGetErrorString(status);
    ++counter_slot;
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 1
    ensure_maps<M64Traits>(activation, weights, rows, m64_maps_);
    status = launch_persistent_family<M64Traits>(
        output_ptr, activation_ptr, weight_ptr, offsets_ptr, workspace_ + counter_slot, rows,
        num_experts_, sm_count_, m64_resources_, m64_maps_.activation, m64_maps_.weight, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "M64 persistent launch failed: " << cudaGetErrorString(status);
#endif
  }

  bool queried_ = false;
  bool configured_ = false;
  int64_t max_rows_ = 0;
  int32_t num_experts_ = 0;
  int32_t device_id_ = -1;
  int32_t sm_count_ = 0;
  unsigned long long* workspace_ = nullptr;
  DLDevice workspace_device_{kDLCPU, 0};
  PFN_cuTensorMapEncodeTiled_v12000 tma_encoder_ = nullptr;
  KernelResources m64_resources_{};
  KernelResources m128_resources_{};
  FamilyMaps m64_maps_{};
  FamilyMaps m128_maps_{};
};

tvm::ffi::Module init() {
  auto runner = tvm::ffi::make_object<Sm90PushBf16PersistentRunner>();
  return tvm::ffi::Module(runner);
}

}  // namespace flashinfer::sm90_push_bf16_persistent

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, flashinfer::sm90_push_bf16_persistent::init);
