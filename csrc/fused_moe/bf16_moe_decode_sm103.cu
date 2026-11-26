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

// TVM-FFI entry point of the fused BF16 MoE decode kernels for SM103 (the
// bf16_decode_sm103_glm52 kernel of the specialized MoELayer backend). The kernels, their
// layouts and device helpers are in include/flashinfer/fused_moe/bf16_moe_decode_sm103.cuh;
// this file encodes the tensor maps and launches them.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

#include "flashinfer/fused_moe/bf16_moe_decode_sm103.cuh"

#define BF16_MOE_DECODE_CHECK_CU(call)                                                           \
  do {                                                                                           \
    CUresult status_ = (call);                                                                   \
    TVM_FFI_ICHECK(status_ == CUDA_SUCCESS) << "bf16 MoE decode: " #call " failed: " << status_; \
  } while (0)
#define BF16_MOE_DECODE_CHECK_CUDA(call)                                         \
  do {                                                                           \
    cudaError_t status_ = (call);                                                \
    TVM_FFI_ICHECK(status_ == cudaSuccess)                                       \
        << "bf16 MoE decode: " #call " failed: " << cudaGetErrorString(status_); \
  } while (0)

using namespace flashinfer::fused_moe::bf16_decode_sm103;

void Run(tvm::ffi::TensorView routing_logits, tvm::ffi::TensorView routing_bias,
         tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView gemm1_weights,
         tvm::ffi::TensorView gemm2_weights, tvm::ffi::TensorView output,
         double routed_scaling_factor) {
  const float route_scale = static_cast<float>(routed_scaling_factor);
  const int dev = hidden_states.device().device_id;
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, dev));
  static int num_sms[64] = {0};
  if (num_sms[dev] == 0) {
    int n = 0;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    BF16_MOE_DECODE_CHECK_CUDA(cudaFuncSetAttribute(
        kA::moe_t1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kA::DSMEM));
    BF16_MOE_DECODE_CHECK_CUDA(cudaFuncSetAttribute(
        kA::moe_fused_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kA::DSMEM));
    BF16_MOE_DECODE_CHECK_CUDA(cudaFuncSetAttribute(
        kB::moe_fused_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kB::DSMEM));
    BF16_MOE_DECODE_CHECK_CUDA(cudaFuncSetAttribute(
        kB::moe_fused_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, kB::DSMEM));
    BF16_MOE_DECODE_CHECK_CUDA(cudaFuncSetAttribute(
        kB::moe_fused_kernel<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, kB::DSMEM));
    TVM_FFI_ICHECK(n >= 150 && n <= 152)
        << "bf16 MoE decode: the work split needs 150..152 SMs, device has " << n;
    num_sms[dev] = n;
  }
  const int T = (int)hidden_states.size(0);
  TVM_FFI_ICHECK(T >= 1 && T <= kB::TCH)
      << "bf16 MoE decode: one launch serves 1.." << kB::TCH << " tokens, got " << T;
  const float* lg = static_cast<const float*>(routing_logits.data_ptr());
  const float* bs = static_cast<const float*>(routing_bias.data_ptr());
  const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
  const __nv_bfloat16* w1 = static_cast<const __nv_bfloat16*>(gemm1_weights.data_ptr());
  const __nv_bfloat16* w2 = static_cast<const __nv_bfloat16*>(gemm2_weights.data_ptr());
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(output.data_ptr());
  CUtensorMap w1_map{};
  CUtensorMap x_map{};
  CUtensorMap x_map1{};
  CUtensorMap x_gather4{};
  CUtensorMap x_map6{};
  CUtensorMap w2_map32{};
  CUtensorMap w2_map9{};
  CUtensorMap w2_map_t1{};
  {
    constexpr uint32_t rank = 4;
    const uint64_t dims[rank] = {64, kA::M1, kA::KB1, kA::E};
    const uint64_t strides[rank - 1] = {64 * sizeof(__nv_bfloat16),
                                        kA::M1 * 64 * sizeof(__nv_bfloat16),
                                        kA::KB1 * kA::M1 * 64 * sizeof(__nv_bfloat16)};
    const uint32_t box[rank] = {64, 128, 1, 1};
    const uint32_t elem_stride[rank] = {1, 1, 1, 1};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &w1_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(w1), dims,
        strides, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  {
    constexpr uint32_t rank = 3;
    const uint64_t dims[rank] = {64, (uint64_t)hidden_states.size(0), kA::KB1};
    const uint64_t strides[rank - 1] = {kA::H * sizeof(__nv_bfloat16), 64 * sizeof(__nv_bfloat16)};
    const uint32_t box[rank] = {64, 2, 1};
    const uint32_t elem_stride[rank] = {1, 1, 1};
    if (T != 1)
      BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
          &x_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(x), dims,
          strides, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    const uint32_t box1[rank] = {64, 1, 1};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &x_map1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(x), dims,
        strides, box1, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  if (T != 1) {
    constexpr uint32_t rank = 2;
    const uint64_t dims[rank] = {kA::H, (uint64_t)hidden_states.size(0)};
    const uint64_t strides[rank - 1] = {kA::H * sizeof(__nv_bfloat16)};
    const uint32_t box[rank] = {64, 1};
    const uint32_t elem_stride[rank] = {1, 1};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &x_gather4, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(x), dims,
        strides, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    const uint32_t box6[rank] = {64, 6};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &x_map6, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(x), dims,
        strides, box6, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  {
    constexpr uint32_t rank = 4;
    const uint64_t dims[rank] = {64, kA::H, kA::KB2, kA::E};
    const uint64_t strides[rank - 1] = {64 * sizeof(__nv_bfloat16),
                                        kA::H * 64 * sizeof(__nv_bfloat16),
                                        kA::KB2 * kA::H * 64 * sizeof(__nv_bfloat16)};
    const uint32_t box32[rank] = {64, (uint32_t)(T == 1 || T == 6 || T == 10 ? 40 : 32),
                                  (uint32_t)(T == 1 || T == 6 || T == 10 ? kA::KB2 : 1), 1};
    const uint32_t box9[rank] = {64, 9, 1, 1};
    const uint32_t elem_stride[rank] = {1, 1, 1, 1};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &w2_map32, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(w2), dims,
        strides, box32, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    if (T != 1) {
      BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
          &w2_map9, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(w2), dims,
          strides, box9, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    const uint32_t box_t1[rank] = {64, kA::W2_TX_ROWS, kA::KB2, 1};
    BF16_MOE_DECODE_CHECK_CU(cuTensorMapEncodeTiled(
        &w2_map_t1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<__nv_bfloat16*>(w2), dims,
        strides, box_t1, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  constexpr int TCH = kB::TCH;
  for (int t0 = 0; t0 < T; t0 += TCH) {
    int tc = T - t0 < TCH ? T - t0 : TCH;
    const float* a0 = lg + (size_t)t0 * kA::E;
    const __nv_bfloat16* a2 = x + (size_t)t0 * kA::H;
    __nv_bfloat16* a5 = out + (size_t)t0 * kA::H;
    if (tc == 1) {
      void* args[] = {(void*)&a0,        (void*)&bs,     (void*)&a5,
                      (void*)&w1_map,    (void*)&x_map1, (void*)&w2_map32,
                      (void*)&w2_map_t1, (void*)&t0,     (void*)&route_scale};
      BF16_MOE_DECODE_CHECK_CUDA(cudaLaunchCooperativeKernel((const void*)kA::moe_t1_kernel,
                                                             dim3(num_sms[dev]), dim3(kA::NTHREADS),
                                                             args, kA::DSMEM, stream));
    } else if (tc == 2) {
      void* args[] = {(void*)&a0,       (void*)&bs,      (void*)&a2,     (void*)&w1,
                      (void*)&w2,       (void*)&a5,      (void*)&w1_map, (void*)&x_map,
                      (void*)&w2_map32, (void*)&w2_map9, (void*)&t0,     (void*)&route_scale};
      BF16_MOE_DECODE_CHECK_CUDA(cudaLaunchCooperativeKernel((const void*)kA::moe_fused_kernel<2>,
                                                             dim3(num_sms[dev]), dim3(kA::NTHREADS),
                                                             args, kA::DSMEM, stream));
    } else if (tc == 6) {
      void* args[] = {(void*)&a0,         (void*)&bs,        (void*)&a2,     (void*)&w1,
                      (void*)&w2,         (void*)&a5,        (void*)&w1_map, (void*)&x_map6,
                      (void*)&w2_map32,   (void*)&w2_map_t1, (void*)&tc,     (void*)&t0,
                      (void*)&route_scale};
      BF16_MOE_DECODE_CHECK_CUDA(cudaLaunchCooperativeKernel((const void*)kB::moe_fused_kernel<6>,
                                                             dim3(num_sms[dev]), dim3(kB::NTHREADS),
                                                             args, kB::DSMEM, stream));
    } else if (tc == 10) {
      void* args[] = {(void*)&a0,         (void*)&bs,        (void*)&a2,     (void*)&w1,
                      (void*)&w2,         (void*)&a5,        (void*)&w1_map, (void*)&x_map1,
                      (void*)&w2_map32,   (void*)&w2_map_t1, (void*)&tc,     (void*)&t0,
                      (void*)&route_scale};
      BF16_MOE_DECODE_CHECK_CUDA(cudaLaunchCooperativeKernel((const void*)kB::moe_fused_kernel<10>,
                                                             dim3(num_sms[dev]), dim3(kB::NTHREADS),
                                                             args, kB::DSMEM, stream));
    } else {
      void* args[] = {(void*)&a0,         (void*)&bs,      (void*)&a2,     (void*)&w1,
                      (void*)&w2,         (void*)&a5,      (void*)&w1_map, (void*)&x_map1,
                      (void*)&w2_map32,   (void*)&w2_map9, (void*)&tc,     (void*)&t0,
                      (void*)&route_scale};
      BF16_MOE_DECODE_CHECK_CUDA(cudaLaunchCooperativeKernel((const void*)kB::moe_fused_kernel<0>,
                                                             dim3(num_sms[dev]), dim3(kB::NTHREADS),
                                                             args, kB::DSMEM, stream));
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
