// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "bf16_grouped_gemm.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::sm90_push_bf16 {

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class Bf16GroupedGemmRunner final : public tvm::ffi::ModuleObj {
 public:
  char const* type_key() const { return "flashinfer.Sm90PushBf16GroupedGemmRunner"; }
  char const* kind() const final { return "sm90_push_bf16_grouped_gemm_runner"; }

  Optional<Function> GetFunction(tvm::ffi::String const& name) final {
    if (name == "get_workspace_size") {
      return Function::FromTyped([this](int64_t max_rows, int64_t num_experts, int64_t n, int64_t k,
                                        int64_t device_id) -> int64_t {
        return get_workspace_size(max_rows, num_experts, n, k, device_id);
      });
    }
    if (name == "get_schedule_workspace_size") {
      return Function::FromTyped([this]() -> int64_t { return get_schedule_workspace_size(); });
    }
    if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace, TensorView schedule_workspace) {
        configure_workspace(workspace, schedule_workspace);
      });
    }
    if (name == "kernel_resource_usage") {
      return Function::FromTyped(
          [this](int64_t block_m) { return kernel_resource_usage(block_m); });
    }
    if (name == "grouped_run") {
      return Function::FromTyped(
          [this](TensorView output, TensorView activation, TensorView weights, TensorView offsets,
                 int64_t epoch) { grouped_run(output, activation, weights, offsets, epoch); });
    }
    if (name == "grouped_run_prepared") {
      return Function::FromTyped([this](TensorView output, TensorView activation,
                                        TensorView weights, TensorView offsets, int64_t epoch) {
        grouped_run_prepared(output, activation, weights, offsets, epoch);
      });
    }
    return Function(nullptr);
  }

 private:
  template <typename Traits>
  size_t query_cutlass_workspace_size() const {
    using Gemm = typename Traits::Gemm;
    WorkspaceView const workspace{};
    auto arguments = make_arguments<Traits>(workspace, num_experts_, device_id_, sm_count_);
    return Gemm::get_workspace_size(arguments);
  }

  int64_t get_workspace_size(int64_t max_rows, int64_t num_experts, int64_t n, int64_t k,
                             int64_t device_id) {
    TVM_FFI_ICHECK_GE(max_rows, 0) << "get_workspace_size: max_rows must be nonnegative";
    TVM_FFI_ICHECK_GT(num_experts, 0) << "get_workspace_size: num_experts must be positive";
    TVM_FFI_ICHECK_GT(n, 0) << "get_workspace_size: N must be positive";
    TVM_FFI_ICHECK_GT(k, 0) << "get_workspace_size: K must be positive";
    TVM_FFI_ICHECK_EQ(n % kAlignment, 0)
        << "get_workspace_size: N must be divisible by " << kAlignment;
    TVM_FFI_ICHECK_EQ(k % kAlignment, 0)
        << "get_workspace_size: K must be divisible by " << kAlignment;
    for (int64_t const value : {max_rows, num_experts, n, k, device_id}) {
      TVM_FFI_ICHECK_LE(value, std::numeric_limits<int>::max())
          << "get_workspace_size: dimension exceeds int32";
    }
    TVM_FFI_ICHECK_GE(device_id, 0) << "get_workspace_size: device_id must be nonnegative";
    TVM_FFI_ICHECK_LE(num_experts, std::numeric_limits<int64_t>::max() / n)
        << "get_workspace_size: weight shape overflows int64";
    TVM_FFI_ICHECK_LE(num_experts * n, std::numeric_limits<int64_t>::max() / k)
        << "get_workspace_size: weight shape overflows int64";
    if (max_rows != 0) {
      TVM_FFI_ICHECK_LE(k, std::numeric_limits<int64_t>::max() / max_rows)
          << "get_workspace_size: activation shape overflows int64";
      TVM_FFI_ICHECK_LE(n, std::numeric_limits<int64_t>::max() / max_rows)
          << "get_workspace_size: output shape overflows int64";
    }

    max_rows_ = static_cast<int>(max_rows);
    num_experts_ = static_cast<int>(num_experts);
    n_ = static_cast<int>(n);
    k_ = static_cast<int>(k);
    device_id_ = static_cast<int>(device_id);

    ffi::CUDADeviceGuard device_guard(device_id_);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device_id_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: device query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0)
        << "get_workspace_size: SM90 is required";
    sm_count_ = properties.multiProcessorCount;

    cutlass_workspace_bytes_ = 0;
#if SM90_PUSH_BF16_FAMILY_MASK & 1
    cutlass_workspace_bytes_ = query_cutlass_workspace_size<M64Traits>();
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
    cutlass_workspace_bytes_ =
        std::max(cutlass_workspace_bytes_, query_cutlass_workspace_size<M128Traits>());
#endif
    TVM_FFI_ICHECK_LE(cutlass_workspace_bytes_,
                      std::numeric_limits<size_t>::max() / kNumMTileFamilies)
        << "get_workspace_size: CUTLASS workspace size overflows";
    required_workspace_bytes_ =
        make_workspace_layout(num_experts_, cutlass_workspace_bytes_).total_bytes;
    required_schedule_workspace_bytes_ = make_schedule_workspace_layout(num_experts_).total_bytes;
    TVM_FFI_ICHECK_LE(required_workspace_bytes_,
                      static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "get_workspace_size: workspace exceeds int64";
    TVM_FFI_ICHECK_LE(required_schedule_workspace_bytes_,
                      static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "get_workspace_size: schedule workspace exceeds int64";

#if SM90_PUSH_BF16_FAMILY_MASK & 1
    status = query_kernel_resources<M64Traits>(&m64_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: M64 resource query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GT(m64_resources_.blocks_per_sm, 0)
        << "get_workspace_size: M64 tactic has zero occupancy";
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
    status = query_kernel_resources<M128Traits>(&m128_resources_);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "get_workspace_size: M128 resource query failed: " << cudaGetErrorString(status);
    TVM_FFI_ICHECK_GT(m128_resources_.blocks_per_sm, 0)
        << "get_workspace_size: M128 tactic has zero occupancy";
#endif

    queried_ = true;
    configured_ = false;
    return static_cast<int64_t>(required_workspace_bytes_);
  }

  int64_t get_schedule_workspace_size() const {
    TVM_FFI_ICHECK(queried_) << "get_schedule_workspace_size: query GEMM workspace first";
    return static_cast<int64_t>(required_schedule_workspace_bytes_);
  }

  void configure_workspace(TensorView workspace, TensorView schedule_workspace) {
    TVM_FFI_ICHECK(queried_) << "configure_workspace: query workspace first";
    CHECK_INPUT(workspace);
    CHECK_INPUT(schedule_workspace);
    CHECK_INPUT_TYPE(workspace, dl_uint8);
    CHECK_INPUT_TYPE(schedule_workspace, dl_uint8);
    CHECK_DIM(1, workspace);
    CHECK_DIM(1, schedule_workspace);
    CHECK_DEVICE(workspace, schedule_workspace);
    TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.numel()), required_workspace_bytes_)
        << "configure_workspace: execution workspace is too small";
    TVM_FFI_ICHECK_GE(static_cast<size_t>(schedule_workspace.numel()),
                      required_schedule_workspace_bytes_)
        << "configure_workspace: schedule workspace is too small";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 64, 0)
        << "configure_workspace: execution workspace must be 64-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(schedule_workspace.data_ptr()) % 16, 0)
        << "configure_workspace: schedule workspace must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(workspace.device().device_id, device_id_)
        << "configure_workspace: workspace device differs from workspace query";

    workspace_ = workspace.data_ptr();
    workspace_bytes_ = static_cast<size_t>(workspace.numel());
    schedule_workspace_ = schedule_workspace.data_ptr();
    schedule_workspace_bytes_ = static_cast<size_t>(schedule_workspace.numel());
    workspace_device_ = workspace.device();
    configured_ = true;
  }

  Array<int64_t> kernel_resource_usage(int64_t block_m) const {
    TVM_FFI_ICHECK(queried_) << "kernel_resource_usage: query workspace first";
    TVM_FFI_ICHECK((block_m == 64 && kEnableM64) || (block_m == 128 && kEnableM128))
        << "kernel_resource_usage: requested M-tile family is not compiled";
    KernelResources const& resources = block_m == 64 ? m64_resources_ : m128_resources_;
    return {resources.blocks_per_sm, resources.num_regs, resources.local_memory_bytes,
            resources.dynamic_smem_bytes};
  }

  void check_inputs(TensorView output, TensorView activation, TensorView weights,
                    TensorView offsets, int64_t epoch) const {
    TVM_FFI_ICHECK(queried_ && configured_)
        << "run: query and configure this runner before launching GEMM";
    TVM_FFI_ICHECK_GT(epoch, 0) << "run: schedule epoch must be positive";
    CHECK_INPUT(output);
    CHECK_INPUT(activation);
    CHECK_INPUT(weights);
    CHECK_INPUT(offsets);
    CHECK_INPUT_TYPE(output, dl_bfloat16);
    CHECK_INPUT_TYPE(activation, dl_bfloat16);
    CHECK_INPUT_TYPE(weights, dl_bfloat16);
    CHECK_INPUT_TYPE(offsets, dl_int64);
    CHECK_DIM(2, output);
    CHECK_DIM(2, activation);
    CHECK_DIM(3, weights);
    CHECK_DIM(1, offsets);
    CHECK_DEVICE(output, activation);
    CHECK_DEVICE(weights, activation);
    CHECK_DEVICE(offsets, activation);
    TVM_FFI_ICHECK_EQ(workspace_device_.device_type, activation.device().device_type)
        << "run: workspace device type mismatch";
    TVM_FFI_ICHECK_EQ(workspace_device_.device_id, activation.device().device_id)
        << "run: workspace device mismatch";
    TVM_FFI_ICHECK_EQ(activation.size(0), output.size(0)) << "run: output row mismatch";
    TVM_FFI_ICHECK_LE(activation.size(0), max_rows_) << "run: row capacity exceeds query";
    TVM_FFI_ICHECK_EQ(activation.size(1), k_) << "run: activation K mismatch";
    TVM_FFI_ICHECK_EQ(output.size(1), n_) << "run: output N mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(0), num_experts_) << "run: weight expert mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(1), n_) << "run: weight N mismatch";
    TVM_FFI_ICHECK_EQ(weights.size(2), k_) << "run: weight K mismatch";
    TVM_FFI_ICHECK_EQ(offsets.size(0), static_cast<int64_t>(num_experts_) + 1)
        << "run: offsets shape mismatch";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(activation.data_ptr()) % 16, 0)
        << "run: activation must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16, 0)
        << "run: weights must be 16-byte aligned";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16, 0)
        << "run: output must be 16-byte aligned";
  }

  void launch(TensorView output, TensorView activation, TensorView weights, TensorView offsets,
              int64_t epoch, bool prepare_schedule) {
    check_inputs(output, activation, weights, offsets, epoch);
    ffi::CUDADeviceGuard device_guard(activation.device().device_id);
    cudaStream_t const stream = get_stream(activation.device());
    cudaError_t const status = launch_grouped_gemm(
        workspace_, workspace_bytes_, schedule_workspace_, schedule_workspace_bytes_, num_experts_,
        static_cast<int>(activation.size(0)), n_, k_,
        static_cast<__nv_bfloat16 const*>(activation.data_ptr()),
        static_cast<__nv_bfloat16 const*>(weights.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int64_t const*>(offsets.data_ptr()), static_cast<uint64_t>(epoch),
        prepare_schedule, cutlass_workspace_bytes_, activation.device().device_id, sm_count_,
        stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "run: BF16 grouped GEMM failed: " << cudaGetErrorString(status);
  }

  void grouped_run(TensorView output, TensorView activation, TensorView weights, TensorView offsets,
                   int64_t epoch) {
    launch(output, activation, weights, offsets, epoch, true);
  }

  void grouped_run_prepared(TensorView output, TensorView activation, TensorView weights,
                            TensorView offsets, int64_t epoch) {
    launch(output, activation, weights, offsets, epoch, false);
  }

  bool queried_ = false;
  bool configured_ = false;
  int max_rows_ = 0;
  int num_experts_ = 0;
  int n_ = 0;
  int k_ = 0;
  int device_id_ = -1;
  int sm_count_ = 0;
  size_t cutlass_workspace_bytes_ = 0;
  size_t required_workspace_bytes_ = 0;
  size_t required_schedule_workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
  void* schedule_workspace_ = nullptr;
  size_t schedule_workspace_bytes_ = 0;
  DLDevice workspace_device_{kDLCPU, 0};
  KernelResources m64_resources_{};
  KernelResources m128_resources_{};
};

tvm::ffi::Module init() { return tvm::ffi::Module(tvm::ffi::make_object<Bf16GroupedGemmRunner>()); }

}  // namespace flashinfer::sm90_push_bf16

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, flashinfer::sm90_push_bf16::init);
