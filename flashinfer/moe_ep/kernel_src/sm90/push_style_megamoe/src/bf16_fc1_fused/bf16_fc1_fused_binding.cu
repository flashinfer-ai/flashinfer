// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "bf16_fc1_fused.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::sm90_push_bf16::fc1_fused {

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

struct KernelResources {
  int blocks_per_sm;
  int num_regs;
  int local_memory_bytes;
  int shared_memory_bytes;
};

template <typename Kernel>
cudaError_t query_kernel_resources(Kernel kernel, int threads, KernelResources* resources) {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  int blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0);
  if (status != cudaSuccess) return status;
  *resources = {blocks_per_sm, attributes.numRegs, static_cast<int>(attributes.localSizeBytes),
                static_cast<int>(attributes.sharedSizeBytes)};
  return cudaSuccess;
}

class Bf16Fc1FusedRunner final : public tvm::ffi::ModuleObj {
 public:
  char const* type_key() const { return "flashinfer.Sm90PushBf16Fc1FusedRunner"; }
  char const* kind() const final { return "sm90_push_bf16_fc1_fused_runner"; }

  Optional<Function> GetFunction(tvm::ffi::String const& name) final {
    if (name == "get_workspace_size") {
      return Function::FromTyped([this](int64_t max_rows, int64_t num_experts,
                                        int64_t intermediate_size, int64_t k, int64_t device_id) {
        return get_workspace_size(max_rows, num_experts, intermediate_size, k, device_id);
      });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configure_workspace(workspace); });
    }
    if (name == "kernel_resource_usage") {
      return Function::FromTyped([this](int64_t fused) { return kernel_resource_usage(fused); });
    }
    if (name == "run") {
      return Function::FromTyped(
          [this](TensorView output, TensorView activation, TensorView weights, TensorView offsets) {
            run(output, activation, weights, offsets);
          });
    }
    if (name == "run_unfused_epilogue") {
      return Function::FromTyped(
          [this](TensorView output, TensorView projected, TensorView offsets) {
            run_unfused_epilogue(output, projected, offsets);
          });
    }
    return Function(nullptr);
  }

 private:
  int64_t get_workspace_size(int64_t max_rows, int64_t num_experts, int64_t intermediate_size,
                             int64_t k, int64_t device_id) {
    TVM_FFI_ICHECK_GE(max_rows, 0) << "get_workspace_size: max_rows must be nonnegative";
    TVM_FFI_ICHECK_GT(num_experts, 0) << "get_workspace_size: num_experts must be positive";
    TVM_FFI_ICHECK_GT(intermediate_size, 0)
        << "get_workspace_size: intermediate_size must be positive";
    TVM_FFI_ICHECK_GT(k, 0) << "get_workspace_size: K must be positive";
    TVM_FFI_ICHECK_EQ(intermediate_size % kTileN, 0)
        << "get_workspace_size: intermediate_size must be divisible by " << kTileN;
    TVM_FFI_ICHECK_EQ(k % kTileK, 0) << "get_workspace_size: K must be divisible by " << kTileK;
    TVM_FFI_ICHECK_GE(device_id, 0) << "get_workspace_size: device_id must be nonnegative";
    for (int64_t const value : {max_rows, num_experts, intermediate_size, k, device_id}) {
      TVM_FFI_ICHECK_LE(value, std::numeric_limits<int>::max())
          << "get_workspace_size: dimension exceeds int32";
    }
    TVM_FFI_ICHECK_LE(intermediate_size, std::numeric_limits<int>::max() / 2)
        << "get_workspace_size: paired FC1 dimension exceeds int32";
    TVM_FFI_ICHECK_LE(num_experts, std::numeric_limits<int64_t>::max() / 2)
        << "get_workspace_size: weight shape overflows int64";
    TVM_FFI_ICHECK_LE(num_experts * 2, std::numeric_limits<int64_t>::max() / intermediate_size)
        << "get_workspace_size: weight shape overflows int64";
    TVM_FFI_ICHECK_LE(num_experts * 2 * intermediate_size, std::numeric_limits<int64_t>::max() / k)
        << "get_workspace_size: weight shape overflows int64";

    max_rows_ = static_cast<int>(max_rows);
    num_experts_ = static_cast<int>(num_experts);
    intermediate_size_ = static_cast<int>(intermediate_size);
    k_ = static_cast<int>(k);
    device_id_ = static_cast<int>(device_id);

    ffi::CUDADeviceGuard device_guard(device_id_);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device_id_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: device query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0)
        << "get_workspace_size: SM90 is required";
    status = query_kernel_resources(bf16_fc1_fused_kernel, 32, &fused_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: fused kernel resource query failed: " << cudaGetErrorString(status);
    status =
        query_kernel_resources(bf16_fc1_unfused_epilogue_kernel, 256, &unfused_epilogue_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: unfused epilogue resource query failed: "
        << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GT(fused_resources_.blocks_per_sm, 0)
        << "get_workspace_size: fused FC1 tactic has zero occupancy";
    TVM_FFI_ICHECK_GT(unfused_epilogue_resources_.blocks_per_sm, 0)
        << "get_workspace_size: unfused epilogue has zero occupancy";

    workspace_bytes_ = workspace_size(num_experts_);
    TVM_FFI_ICHECK_LE(workspace_bytes_, static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "get_workspace_size: workspace exceeds int64";
    queried_ = true;
    configured_ = false;
    return static_cast<int64_t>(workspace_bytes_);
  }

  Array<int64_t> kernel_resource_usage(int64_t fused) const {
    TVM_FFI_ICHECK(queried_) << "kernel_resource_usage: query workspace first";
    TVM_FFI_ICHECK(fused == 0 || fused == 1) << "kernel_resource_usage: fused must be zero or one";
    KernelResources const& resources = fused ? fused_resources_ : unfused_epilogue_resources_;
    return {resources.blocks_per_sm, resources.num_regs, resources.local_memory_bytes,
            resources.shared_memory_bytes};
  }

  void configure_workspace(TensorView workspace) {
    TVM_FFI_ICHECK(queried_) << "configure_workspace: query workspace first";
    CHECK_INPUT(workspace);
    CHECK_INPUT_TYPE(workspace, dl_uint8);
    CHECK_DIM(1, workspace);
    TVM_FFI_ICHECK_EQ(workspace.device().device_id, device_id_)
        << "configure_workspace: workspace device differs from workspace query";
    TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.numel()), workspace_bytes_)
        << "configure_workspace: workspace is too small";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16, 0)
        << "configure_workspace: workspace must be 16-byte aligned";
    workspace_ = workspace.data_ptr();
    configured_ = true;
  }

  void check_common(TensorView output, TensorView activation) const {
    TVM_FFI_ICHECK(queried_ && configured_)
        << "run: query and configure this runner before launching FC1";
    CHECK_INPUT(output);
    CHECK_INPUT(activation);
    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(activation, dl_bfloat16);
    CHECK_DIM(2, output);
    CHECK_DIM(2, activation);
    CHECK_DEVICE(output, activation);
    TVM_FFI_ICHECK_EQ(activation.device().device_id, device_id_)
        << "run: tensor device differs from workspace query";
    TVM_FFI_ICHECK_EQ(output.size(0), activation.size(0)) << "run: output row mismatch";
    TVM_FFI_ICHECK_LE(activation.size(0), max_rows_) << "run: row capacity exceeds query";
    TVM_FFI_ICHECK_EQ(activation.size(1), k_) << "run: activation K mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), intermediate_size_)
        << "run: output intermediate dimension mismatch";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(activation.data_ptr()) % 16, 0)
        << "run: activation must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0)
        << "run: output must be 16-byte aligned";
  }

  void run(TensorView output, TensorView activation, TensorView weights, TensorView offsets) {
    check_common(output, activation);
    CHECK_INPUT(weights);
    CHECK_INPUT(offsets);
    CHECK_INPUT_TYPE(weights, dl_bfloat16);
    CHECK_INPUT_TYPE(offsets, dl_int64);
    CHECK_DIM(3, weights);
    CHECK_DIM(1, offsets);
    CHECK_DEVICE(weights, activation);
    CHECK_DEVICE(offsets, activation);
    TVM_FFI_ICHECK_EQ(weights.size(0), num_experts_) << "run: weight expert mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(1), static_cast<int64_t>(intermediate_size_) * 2)
        << "run: weight paired-output dimension mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(2), k_) << "run: weight K mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "run: offsets shape mismatch";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 32, 0)
        << "run: weights must be 32-byte aligned";

    ffi::CUDADeviceGuard device_guard(device_id_);
    cudaStream_t const stream = get_stream(activation.device());
    cudaError_t const status = launch_fc1_fused(
        workspace_, workspace_bytes_, static_cast<int>(activation.size(0)), num_experts_,
        intermediate_size_, k_, static_cast<__nv_bfloat16 const*>(activation.data_ptr()),
        static_cast<__nv_bfloat16 const*>(weights.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int64_t const*>(offsets.data_ptr()), stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "run: BF16 fused FC1 failed: " << cudaGetErrorString(status);
  }

  void run_unfused_epilogue(TensorView output, TensorView projected, TensorView offsets) {
    TVM_FFI_ICHECK(queried_ && configured_)
        << "run_unfused_epilogue: query and configure this runner first";
    CHECK_INPUT(output);
    CHECK_INPUT(projected);
    CHECK_INPUT(offsets);
    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(projected, dl_bfloat16);
    CHECK_INPUT_TYPE(offsets, dl_int64);
    CHECK_DIM(2, output);
    CHECK_DIM(2, projected);
    CHECK_DIM(1, offsets);
    CHECK_DEVICE(output, projected);
    CHECK_DEVICE(offsets, projected);
    TVM_FFI_ICHECK_EQ(projected.device().device_id, device_id_)
        << "run_unfused_epilogue: tensor device differs from workspace query";
    TVM_FFI_ICHECK_EQ(output.size(0), projected.size(0))
        << "run_unfused_epilogue: output row mismatch";
    TVM_FFI_ICHECK_LE(projected.size(0), max_rows_)
        << "run_unfused_epilogue: row capacity exceeds query";
    TVM_FFI_ICHECK_EQ(projected.size(1), static_cast<int64_t>(intermediate_size_) * 2)
        << "run_unfused_epilogue: projected width mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), intermediate_size_)
        << "run_unfused_epilogue: output width mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "run_unfused_epilogue: offsets shape mismatch";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(projected.data_ptr()) % 16, 0)
        << "run_unfused_epilogue: projected must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0)
        << "run_unfused_epilogue: output must be 16-byte aligned";
    ffi::CUDADeviceGuard device_guard(device_id_);
    cudaStream_t const stream = get_stream(projected.device());
    cudaError_t const status =
        launch_unfused_epilogue(static_cast<__nv_bfloat16*>(output.data_ptr()),
                                static_cast<__nv_bfloat16 const*>(projected.data_ptr()),
                                static_cast<int64_t const*>(offsets.data_ptr()), num_experts_,
                                static_cast<int>(projected.size(0)), intermediate_size_, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "run_unfused_epilogue: BF16 gated activation failed: " << cudaGetErrorString(status);
  }

  bool queried_ = false;
  bool configured_ = false;
  int max_rows_ = 0;
  int num_experts_ = 0;
  int intermediate_size_ = 0;
  int k_ = 0;
  int device_id_ = -1;
  size_t workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  KernelResources fused_resources_{};
  KernelResources unfused_epilogue_resources_{};
};

tvm::ffi::Module init() { return tvm::ffi::Module(tvm::ffi::make_object<Bf16Fc1FusedRunner>()); }

}  // namespace flashinfer::sm90_push_bf16::fc1_fused

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, flashinfer::sm90_push_bf16::fc1_fused::init);
