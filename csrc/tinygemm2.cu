/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from TensorRT-LLM tinygemm2 kernel for FlashInfer integration.
 * Original: TensorRT-LLM/cpp/tensorrt_llm/kernels/tinygemm2/
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda/barrier>
#include <cuda/std/utility>

#include "cuda_bf16.h"
#include "cuda_pipeline.h"
#include "tvm_ffi_utils.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace ptx = cuda::ptx;

// ============================================================================
// Device helpers (from tinygemm2_kernel.cuh)
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ void ldmatrix(__nv_bfloat16 rv[2], uint32_t smem_ptr) {
  int dst;
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(dst)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = dst;
}

__device__ void ldmatrix2(__nv_bfloat16 rv[4], uint32_t smem_ptr) {
  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(x), "=r"(y)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
}

__device__ void ldmatrix4(__nv_bfloat16 rv[8], uint32_t smem_ptr) {
  int x, y, z, w;
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
  rvi[2] = z;
  rvi[3] = w;
}

__device__ void HMMA_1688(float d[4], __nv_bfloat16 a[4], __nv_bfloat16 b[2], float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ void HMMA_16816(float d[4], __nv_bfloat16 a[8], __nv_bfloat16 b[4], float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
        "f"(C[2]), "f"(C[3]));
}

__device__ void bar_wait(uint32_t bar_ptr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar_ptr),
      "r"(phase));
}

__device__ bool bar_try_wait(uint32_t bar_ptr, int phase) {
  uint32_t success;
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_ptr), "r"(phase));
  return success;
}

__device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}
#endif  // __CUDA_ARCH__ >= 900

// ============================================================================
// Kernel template (from tinygemm2_kernel.cuh)
// ============================================================================

template <int WARP_TILE_M, int TILE_M, int TILE_N, int TILE_K, int STAGES, int STAGE_UNROLL,
          bool USE_PDL = false>
__global__ __launch_bounds__(384, 1) void tinygemm_kernel(
    __nv_bfloat16* output, __nv_bfloat16* weights, __nv_bfloat16* activations, __nv_bfloat16* bias,
    int M, int N, int K, const __grid_constant__ CUtensorMap weight_map,
    const __grid_constant__ CUtensorMap activation_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

  extern __shared__ __align__(128) char smem[];

  __nv_bfloat16* sh_weights = (__nv_bfloat16*)&smem[0];
  __nv_bfloat16* sh_activations =
      (__nv_bfloat16*)&smem[STAGES * STAGE_UNROLL * TILE_M * TILE_K * sizeof(__nv_bfloat16)];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar_wt_ready[STAGES];
  __shared__ barrier bar_act_ready[STAGES];
  __shared__ barrier bar_data_consumed[STAGES];

  __shared__ float4 reduction_buffer[128];

  __shared__ nv_bfloat16 sh_bias[TILE_M];

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init(&bar_wt_ready[i], 1);
      init(&bar_act_ready[i], 1);
      init(&bar_data_consumed[i], 32);
    }
    ptx::fence_proxy_async(ptx::space_shared);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&weight_map))
                 : "memory");
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&activation_map))
                 : "memory");
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int phase = 0;

  int mib = blockIdx.x * TILE_M;
  int ni = blockIdx.y * TILE_N;

  float accum[4];
  for (int i = 0; i < 4; i++) accum[i] = 0.f;

  int const K_LOOPS_DMA = (K + 4 * TILE_K * STAGE_UNROLL - 1) / (4 * (TILE_K * STAGE_UNROLL));
  int const K_LOOPS_COMPUTE = K_LOOPS_DMA;

  // Data loading thread
  if (warp_id >= 4 && elect_one_sync()) {
    int stage = warp_id % 4;

    bool weight_warp = warp_id < 8;
    if constexpr (USE_PDL) {
      if (!weight_warp) {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
      }
    }

    for (int ki = 0; ki < K_LOOPS_DMA; ki++) {
      int k = (ki * 4 + (warp_id % 4)) * TILE_K * STAGE_UNROLL;

      uint64_t desc_ptr_wt = reinterpret_cast<uint64_t>(&weight_map);
      uint64_t desc_ptr_act = reinterpret_cast<uint64_t>(&activation_map);

      uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
      uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);
      int bytes_wt = TILE_M * TILE_K * sizeof(__nv_bfloat16);
      int bytes_act = TILE_N * TILE_K * sizeof(__nv_bfloat16);

      bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);

      if (weight_warp)
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :
                     : "r"(bar_ptr_wt), "r"(STAGE_UNROLL * bytes_wt));
      if (!weight_warp)
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :
                     : "r"(bar_ptr_act), "r"(STAGE_UNROLL * bytes_act));

      for (int i = 0; i < STAGE_UNROLL; i++) {
        uint32_t smem_ptr_wt =
            __cvta_generic_to_shared(&sh_weights[(stage * STAGE_UNROLL + i) * TILE_M * TILE_K]);
        uint32_t crd0 = k + i * TILE_K;
        uint32_t crd1 = mib;
        if (weight_warp)
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, "
              "{%3,%4}], [%2];"
              :
              : "r"(smem_ptr_wt), "l"(desc_ptr_wt), "r"(bar_ptr_wt), "r"(crd0), "r"(crd1)
              : "memory");

        uint32_t smem_ptr_act =
            __cvta_generic_to_shared(&sh_activations[(stage * STAGE_UNROLL + i) * TILE_N * TILE_K]);
        crd0 = k + i * TILE_K;
        crd1 = ni;
        if (!weight_warp)
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, "
              "{%3,%4}], [%2];"
              :
              : "r"(smem_ptr_act), "l"(desc_ptr_act), "r"(bar_ptr_act), "r"(crd0), "r"(crd1)
              : "memory");
      }

      stage += 4;
      if (stage >= STAGES) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
    // Wait for pending loads to be consumed before exiting, to avoid race
    for (int i = 0; i < (STAGES / 4) - 1; i++) {
      bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);
      stage += 4;
      if (stage >= STAGES) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
  }
  // Compute threads
  else if (warp_id < 4) {
    // Sneak the bias load into the compute warps since they're just waiting
    if (threadIdx.x < TILE_M) sh_bias[threadIdx.x] = bias[mib + threadIdx.x];

    int stage = warp_id;

    int phase = 0;
    int lane_id_div8 = lane_id / 8;
    int lane_id_mod8 = lane_id % 8;

    int lane_row_offset_wt = (lane_id_div8 % 2) ? 8 : 0;
    int lane_col_offset_wt = (lane_id_div8 / 2) ? 1 : 0;

    int row_wt = lane_id_mod8 + lane_row_offset_wt;
    int row_act = lane_id_mod8;

    int row_offset_wt = (reinterpret_cast<uintptr_t>(sh_weights) / 128) % 8;
    int row_offset_act = row_offset_wt;

    uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
    uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);

    bool weight_ready = bar_try_wait(bar_ptr_wt, phase);
    bool act_ready = bar_try_wait(bar_ptr_act, phase);

#pragma unroll 2
    for (int ki = 0; ki < K_LOOPS_COMPUTE; ki++) {
      int next_stage = stage + 4;
      int next_phase = phase;
      if (next_stage >= STAGES) {
        next_stage = warp_id;
        next_phase ^= 1;
      }

      while (!weight_ready || !act_ready) {
        weight_ready = bar_try_wait(bar_ptr_wt, phase);
        act_ready = bar_try_wait(bar_ptr_act, phase);
      }

      if (ki + 1 < K_LOOPS_COMPUTE) {
        weight_ready =
            bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[next_stage]), next_phase);
        act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[next_stage]), next_phase);
      }

#pragma unroll
      for (int su = 0; su < STAGE_UNROLL; su++) {
        __nv_bfloat16* ptr_weights = &sh_weights[(stage * STAGE_UNROLL + su) * TILE_M * TILE_K];
        __nv_bfloat16* ptr_act = &sh_activations[(stage * STAGE_UNROLL + su) * TILE_N * TILE_K];

#pragma unroll
        for (int kii = 0; kii < TILE_K / 16; kii++) {
          __nv_bfloat16 a[8];
          __nv_bfloat16 b[4];

          int col = 2 * kii + lane_col_offset_wt;
          int col_sw = ((row_wt + row_offset_wt) % 8) ^ col;

          ldmatrix4(a, __cvta_generic_to_shared(&ptr_weights[row_wt * TILE_K + col_sw * 8]));

          col = 2 * kii + lane_id_div8;
          col_sw = ((row_act + row_offset_act) % 8) ^ col;

          ldmatrix2(b, __cvta_generic_to_shared(&ptr_act[row_act * TILE_K + 8 * col_sw]));

          HMMA_16816(accum, a, b, accum);
        }
      }

      uint32_t bar_c = __cvta_generic_to_shared(&bar_data_consumed[stage]);
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(bar_c));

      stage = next_stage;
      phase = next_phase;
    }

    float4 accum4;
    accum4.x = accum[0];
    accum4.y = accum[1];
    accum4.z = accum[2];
    accum4.w = accum[3];
    reduction_buffer[threadIdx.x] = accum4;

    __syncthreads();

    if (warp_id == 0) {
      int mi = mib + warp_id * WARP_TILE_M;
      int tm = mi + lane_id / 4;
      int tn = ni + 2 * (lane_id % 4);

      float4 accum1 = reduction_buffer[32 + threadIdx.x];
      float4 accum2 = reduction_buffer[64 + threadIdx.x];
      float4 accum3 = reduction_buffer[96 + threadIdx.x];

      accum[0] = accum[0] + accum1.x + accum2.x + accum3.x;
      accum[1] = accum[1] + accum1.y + accum2.y + accum3.y;
      accum[2] = accum[2] + accum1.z + accum2.z + accum3.z;
      accum[3] = accum[3] + accum1.w + accum2.w + accum3.w;

      float bias_lo = __bfloat162float(sh_bias[tm - mib]);
      float bias_hi = __bfloat162float(sh_bias[tm + 8 - mib]);

      if (tn < N && tm < M) output[tn * M + tm] = __float2bfloat16(accum[0] + bias_lo);
      if (tn + 1 < N && tm < M) output[(tn + 1) * M + tm] = __float2bfloat16(accum[1] + bias_lo);
      if (tn < N && tm + 8 < M) output[tn * M + tm + 8] = __float2bfloat16(accum[2] + bias_hi);
      if (tn + 1 < N && tm + 8 < M)
        output[(tn + 1) * M + tm + 8] = __float2bfloat16(accum[3] + bias_hi);
    }
  }
#endif  // __CUDA_ARCH__ >= 900
}

// ============================================================================
// Host launcher (adapted from tinygemm2_cuda.cu)
// ============================================================================

namespace flashinfer {
namespace tinygemm2 {

// Kernel mapping for the linear operation output = input @ weight.T + bias:
//   input (batch_size, input_features) -> activations (gA), indexed as (N, K)
//   weight (output_features, input_features) -> weights (gB), indexed as (M, K)
//   output (batch_size, output_features) -> gC, stored column-major as (M, N)
//   bias (output_features,) -> bias, indexed as (M,)
//
// The kernel internally computes C[n,m] = sum_k A[n,k] * B[m,k] + bias[m]
// and writes output in column-major: output[n * M + m].
// The Python wrapper transposes this back to row-major (batch_size, output_features).

void launch_tinygemm2(__nv_bfloat16* gA, __nv_bfloat16* gB, __nv_bfloat16* gC, __nv_bfloat16* bias,
                      int batch_size, int output_features, int input_features, cudaStream_t stream,
                      bool use_pdl) {
  static int const WARP_TILE_M = 16;
  static int const TILE_M = WARP_TILE_M;
  static int const TILE_N = 8;
  static int const TILE_K = 64;
  static int const STAGES = 16;
  static int const STAGE_UNROLL = 4;

  CUtensorMap weight_map{};
  CUtensorMap activation_map{};

  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {(uint64_t)input_features, (uint64_t)output_features};
  uint64_t stride[rank - 1] = {input_features * sizeof(__nv_bfloat16)};
  uint32_t box_size[rank] = {TILE_K, TILE_M};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &weight_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gB, size, stride,
      box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for weight_map with error code " << res;

  size[1] = batch_size;
  box_size[1] = TILE_N;

  res = cuTensorMapEncodeTiled(
      &activation_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gA, size,
      stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for activation_map with error code " << res;

  int smem_size =
      STAGES * STAGE_UNROLL *
      (TILE_M * TILE_K * sizeof(__nv_bfloat16) + TILE_N * TILE_K * sizeof(__nv_bfloat16));

  int tiles_m = (output_features + TILE_M - 1) / TILE_M;
  int tiles_n = (batch_size + TILE_N - 1) / TILE_N;

  dim3 grid(tiles_m, tiles_n);
  dim3 block(384);

  if (use_pdl) {
    // PDL path: use cudaLaunchKernelEx with programmatic stream serialization.
    auto status = cudaFuncSetAttribute(
        tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    status = cudaLaunchKernelEx(
        &config, &tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, true>,
        gC, gA, gB, bias, output_features, batch_size, input_features, weight_map, activation_map);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaLaunchKernelEx failed: " << cudaGetErrorString(status);
  } else {
    // Standard path: no PDL intrinsics compiled into the kernel, plain <<<>>> launch.
    auto status = cudaFuncSetAttribute(
        tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, false>
        <<<grid, block, smem_size, stream>>>(gC, gA, gB, bias, output_features, batch_size,
                                             input_features, weight_map, activation_map);
    status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "tinygemm_kernel launch failed: " << cudaGetErrorString(status);
  }
}

void tinygemm2_op(TensorView input, TensorView weight, TensorView bias, TensorView output,
                  bool use_pdl) {
  auto stream = get_stream(input.device());

  int batch_size = input.shape()[0];
  int input_features = input.shape()[1];
  int output_features = weight.shape()[0];

  launch_tinygemm2(reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
                   reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
                   reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                   reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()), batch_size, output_features,
                   input_features, stream, use_pdl);
}

}  // namespace tinygemm2
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tinygemm2_op, flashinfer::tinygemm2::tinygemm2_op);
