/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_W8A16_HOST_HPP_
#define FLASHINFER_GEMM_W8A16_HOST_HPP_

// Host-side lifecycle shared by the Hopper and Blackwell W8A16 kernel variants: the CUTLASS
// problem arguments common to both, the workspace size, and the run sequence.  A variant
// type V (W8A16Sm90Gemm / W8A16Sm100Gemm) provides
//   V::Gemm, V::StrideA/B/D, V::ElementW/ElementAct/ElementD/ElementBias,
//   V::IsSplitK, V::IsStreamK, V::TileK,
//   V::MakeArguments(act, weight, scale, bias, out, M, N, K, device_id, num_sms, splits)
//       = W8A16BaseArguments<V>(...) plus the architecture's scheduler / mainloop fields,
//   V::Launch(gemm, stream): the adapter's run call with the architecture's launch flags.
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// Operand mapping: the narrow operand must be the builder's A, so the weight [N, K] is A
// with M' = N, the activation [M, K] is B with N' = M, and D is [M', N'] column-major, which
// is out[M, N] row-major.  The per-tensor scale is the epilogue alpha (fp32, read from the
// device pointer), the bias a per-row (M' = N) broadcast in the same fused epilogue.  A
// split-K variant writes fp32 partials instead and applies scale and bias in its reduce.

#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "flashinfer/gemm/w8a16_common.hpp"

namespace flashinfer::gemm {

// Carries a variant type through the per-architecture algo -> kernel table
// (W8A16Sm90Visit / W8A16Sm100Visit) into a generic lambda.
template <typename V>
struct W8A16VariantTag {
  using type = V;
};

// The problem arguments every variant shares.  `splits` > 1 only with split-K: problem
// {N, M, K / splits, splits}; A and B batch stride = K / splits elements (the next K-slice;
// row strides stay the full K, so TMA walks the sliced operands in place), D batch stride =
// M * N (the next fp32 partial).  Stream-K variants run CUTLASS's heuristic decomposition
// (data-parallel full waves, stream-K tail) with the deterministic, K-ordered peer
// reduction.
template <typename V, typename T>
typename V::Gemm::Arguments W8A16BaseArguments(const T* act, const __nv_fp8_e4m3* weight,
                                               const float* scale, const T* bias, void* out, int M,
                                               int N, int K, int device_id, int num_sms,
                                               int splits) {
  using StrideA = typename V::StrideA;
  using StrideB = typename V::StrideB;
  using StrideD = typename V::StrideD;
  const int kc = K / splits;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, K, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));
  if (splits > 1) {
    cute::get<2>(stride_a) = kc;
    cute::get<2>(stride_b) = kc;
    cute::get<2>(stride_d) = static_cast<int64_t>(M) * N;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = num_sms;

  typename V::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, kc, splits},
      {reinterpret_cast<const typename V::ElementW*>(weight), stride_a,
       reinterpret_cast<const typename V::ElementAct*>(act), stride_b},
      {{}, nullptr, stride_d, reinterpret_cast<typename V::ElementD*>(out), stride_d},
      hw_info};
  auto& fusion = args.epilogue.thread;
  fusion.alpha = 1.0f;
  fusion.beta = 0.0f;
  if constexpr (!V::IsSplitK) {
    fusion.alpha_ptr = scale;  // per-tensor weight scale, read on device
    fusion.beta_ptr = nullptr;
    fusion.bias_ptr = reinterpret_cast<const typename V::ElementBias*>(bias);  // nullptr -> 0
    fusion.dBias = {cute::_1{}, cute::_0{}, 0};
  } else {
    (void)scale;
    (void)bias;
  }
  if constexpr (V::IsStreamK) {
    args.scheduler.decomposition_mode = cutlass::gemm::kernel::detail::DecompositionMode::Heuristic;
    args.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
  }
  return args;
}

// Workspace of one variant for one problem: CUTLASS's own (persistent scheduler: usually 0;
// stream-K: the peers' fp32 partials plus per-tile barriers) rounded up to 256 B, followed
// by the split-K fp32 partials (`splits` x M x N floats).
struct W8A16WorkspaceLayout {
  size_t cutlass_bytes = 0;  // where the split-K partials start
  size_t total_bytes = 0;
};

// cudaErrorInvalidValue when the total leaves the signed 64-bit range the workspace query
// reports in (checked arithmetic: the admitted dimensions reach 2^31 - 512 each); `*layout`
// is left untouched then.
template <typename V>
cudaError_t W8A16WorkspaceOf(int M, int N, int K, int device_id, int num_sms, int splits,
                             W8A16WorkspaceLayout* layout) {
  using T = typename V::ActType;
  auto args = V::MakeArguments(static_cast<const T*>(nullptr), nullptr, nullptr,
                               static_cast<const T*>(nullptr), nullptr, M, N, K, device_id, num_sms,
                               splits);
  const uint64_t cutlass_bytes = W8A16Align256(V::Gemm::get_workspace_size(args));
  uint64_t total = cutlass_bytes;
  if (V::IsSplitK && splits > 1) {
    uint64_t partials = static_cast<uint64_t>(splits) * sizeof(float);
    if (!W8A16CheckedMul(partials, static_cast<uint64_t>(M), &partials) ||
        !W8A16CheckedMul(partials, static_cast<uint64_t>(N), &partials) ||
        !W8A16CheckedAdd(total, partials, &total)) {
      return cudaErrorInvalidValue;
    }
  }
  layout->cutlass_bytes = static_cast<size_t>(cutlass_bytes);
  layout->total_bytes = static_cast<size_t>(total);
  return cudaSuccess;
}

// Workspace bytes of one variant for one problem (see W8A16WorkspaceOf for the error).
template <typename V>
cudaError_t W8A16WorkspaceBytesOf(int M, int N, int K, int device_id, int num_sms, int splits,
                                  size_t* bytes) {
  W8A16WorkspaceLayout layout;
  const cudaError_t status = W8A16WorkspaceOf<V>(M, N, K, device_id, num_sms, splits, &layout);
  if (status == cudaSuccess) *bytes = layout.total_bytes;
  return status;
}

// Runs one variant: validates the split, places the split-K partials after CUTLASS's
// workspace, checks the capacity, initializes and launches the GEMM, and for split-K runs
// the reduce.  cudaErrorInvalidValue: an unsliceable K for split-K or a short workspace;
// cudaErrorNotSupported: CUTLASS cannot implement the problem on this variant.
template <typename V, typename T>
cudaError_t W8A16RunVariant(const T* act, const __nv_fp8_e4m3* weight, const float* scale,
                            const T* bias, T* out, int M, int N, int K, void* workspace,
                            size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream,
                            int splits) {
  if (V::IsSplitK && (splits < 2 || K % (splits * V::TileK) != 0)) return cudaErrorInvalidValue;
  void* d_out = out;
  if constexpr (V::IsSplitK) {
    W8A16WorkspaceLayout layout;
    if (W8A16WorkspaceOf<V>(M, N, K, device_id, num_sms, splits, &layout) != cudaSuccess ||
        layout.total_bytes > workspace_bytes) {
      return cudaErrorInvalidValue;
    }
    d_out = static_cast<char*>(workspace) + layout.cutlass_bytes;
  }
  auto args =
      V::MakeArguments(act, weight, scale, bias, d_out, M, N, K, device_id, num_sms, splits);
  typename V::Gemm gemm;
  if (V::Gemm::get_workspace_size(args) > workspace_bytes) return cudaErrorInvalidValue;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorNotSupported;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  if (V::Launch(gemm, stream) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  if constexpr (V::IsSplitK) {
    return W8A16SplitKReduce<T>(static_cast<const float*>(d_out), splits, scale, bias, out, M, N,
                                stream);
  }
  return cudaSuccess;
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_HOST_HPP_
