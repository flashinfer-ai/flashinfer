/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Shared tvm-ffi launcher of the generated Cake kernels of this package:
// exporter output, hand-folded here pending exporter support.  The export
// writes one host binding per program; this file is that binding with the
// per-program values made compile definitions, so every program's library is
// its kernel source plus this launcher.  cake_jit.launch_definitions() composes
// the definitions from the registry rows (`_PROGRAMS`, `_LAUNCH`):
//
//   CAKE_PROGRAM_KERNEL   the kernel symbol (kernel_<program>)
//   CAKE_PROGRAM_SHIM     the host-shim namespace of the program
//   CAKE_PLAN_<PLAN>      exactly one of GEMM, ATTENTION, ATTENTION_MERGE,
//                         MERGE, RMSNORM_APPLY: the argument plan, i.e. the
//                         kernel signature, the checks and the TMA descriptors
//   CAKE_DYNAMIC_SMEM     dynamic shared memory in bytes (0: no opt-in call)
//   CAKE_BLOCK_X          block extent
//   CAKE_CLUSTER_X        cluster extent, 0 for a launch without cluster
//                         attributes (2 or 4 add the cluster dimension, the
//                         spread scheduling policy and the grid check)
//   CAKE_B_BOX_N          GEMM only: second extent of the B TMA box
//   CAKE_A_TMA_RANK       GEMM only: 3, or 2 for the position-table GEMMs,
//                         whose A / A2 descriptors map pairs of rows
//
// The TMA descriptor encoders come from cake_kimi_k3_vision_tower_tma_encode.cuh,
// included once per descriptor with its name, rank and box as parameters.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

#include "cake_kimi_k3_vision_tower_host_common.cuh"
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#if !defined(CAKE_PROGRAM_KERNEL) || !defined(CAKE_PROGRAM_SHIM)
#error "CAKE_PROGRAM_KERNEL and CAKE_PROGRAM_SHIM select the program this launcher builds"
#endif
#if defined(CAKE_PLAN_GEMM) + defined(CAKE_PLAN_ATTENTION) + defined(CAKE_PLAN_ATTENTION_MERGE) + defined(CAKE_PLAN_MERGE) + defined(CAKE_PLAN_RMSNORM_APPLY) != 1
#error "exactly one CAKE_PLAN_<PLAN> must be defined"
#endif
#if !defined(CAKE_DYNAMIC_SMEM) || !defined(CAKE_BLOCK_X) || !defined(CAKE_CLUSTER_X)
#error "CAKE_DYNAMIC_SMEM, CAKE_BLOCK_X and CAKE_CLUSTER_X must be defined"
#endif
#if defined(CAKE_PLAN_GEMM) && (!defined(CAKE_B_BOX_N) || !defined(CAKE_A_TMA_RANK))
#error "a GEMM program defines CAKE_B_BOX_N and CAKE_A_TMA_RANK"
#endif

#if defined(CAKE_PLAN_GEMM)
extern "C" __global__ void CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ C2, __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ R, float* __restrict__ COS, float* __restrict__ SIN, __nv_bfloat16* __restrict__ CS, float* __restrict__ SQ, __nv_bfloat16* __restrict__ XW, __nv_bfloat16* __restrict__ WN, float* __restrict__ WS, unsigned int* __restrict__ FLAGS, const __grid_constant__ CUtensorMap RT, const __grid_constant__ CUtensorMap CT, const __grid_constant__ CUtensorMap XWT, int M, int m_tiles, int full_tiles, int tail_split, int pf_l2, float eps);
#elif defined(CAKE_PLAN_ATTENTION)
extern "C" __global__ void CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap Q, __nv_bfloat16* __restrict__ Q_raw, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, const __grid_constant__ CUtensorMap O_tma, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ unit_table, unsigned long long* __restrict__ probe, unsigned int total_tiles, int num_heads, float softmax_scale_log2, float* __restrict__ partial_O, float* __restrict__ partial_ML);
#elif defined(CAKE_PLAN_ATTENTION_MERGE)
extern "C" __global__ void CAKE_PROGRAM_KERNEL(float* __restrict__ partial_O, float* __restrict__ partial_ML, int* __restrict__ merge_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, __nv_bfloat16* __restrict__ O, int num_heads, int part_rows, float softmax_scale_log2);
#elif defined(CAKE_PLAN_MERGE)
extern "C" __global__ void CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ norm_weight, int* __restrict__ merge_table, __nv_bfloat16* __restrict__ m_out, float eps);
#elif defined(CAKE_PLAN_RMSNORM_APPLY)
extern "C" __global__ void CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ weight, float* __restrict__ rowsumsq, int M, float eps);
#endif


namespace CAKE_PROGRAM_SHIM {

using namespace cake_host_shim_common;
using tvm::ffi::TensorView;

#if defined(CAKE_PLAN_GEMM)

// A and A2: 3-D (64, 128, 1) boxes, or the 2-D pair maps of the position-table GEMMs.
#if CAKE_A_TMA_RANK == 2
#define CAKE_TMA_NAME A
#define CAKE_TMA_RANK 2
#define CAKE_TMA_PAIR_FORM 1
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME A2
#define CAKE_TMA_RANK 2
#define CAKE_TMA_PAIR_FORM 2
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"
#elif CAKE_A_TMA_RANK == 3
#define CAKE_TMA_NAME A
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_CHECK_BOX_1 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME A2
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_CHECK_BOX_1 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"
#else
#error "CAKE_A_TMA_RANK must be 3 or 2"
#endif

// B: (64, CAKE_B_BOX_N, 1) box, its second extent checked against the operand.
#define CAKE_TMA_NAME B
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 CAKE_B_BOX_N
#define CAKE_TMA_CHECK_BOX_1 1
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

// RT, CT, XWT: 3-D (64, 128, 1) boxes.
#define CAKE_TMA_NAME RT
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_CHECK_BOX_1 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME CT
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_CHECK_BOX_1 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME XWT
#define CAKE_TMA_RANK 3
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_CHECK_BOX_1 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

void Run(TensorView arg_A, TensorView arg_A2, TensorView arg_B, TensorView arg_C, TensorView arg_C2, TensorView arg_C3, TensorView arg_R, TensorView arg_COS, TensorView arg_SIN, TensorView arg_CS, TensorView arg_SQ, TensorView arg_XW, TensorView arg_WN, TensorView arg_WS, TensorView arg_FLAGS, TensorView arg_RT, TensorView arg_CT, TensorView arg_XWT, int64_t arg_M, int64_t arg_m_tiles, int64_t arg_full_tiles, int64_t arg_tail_split, int64_t arg_pf_l2, double arg_eps, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_A.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));  // binds the context for the encoders
  check_cuda_tensor(arg_A, "A");
  check_dtype(arg_A, DLDataType{kDLBfloat, 16, 1}, "A");
  check_cuda_tensor(arg_A2, "A2");
  check_dtype(arg_A2, DLDataType{kDLBfloat, 16, 1}, "A2");
  check_cuda_tensor(arg_B, "B");
  check_dtype(arg_B, DLDataType{kDLBfloat, 16, 1}, "B");
  check_cuda_tensor(arg_C, "C");
  check_dtype(arg_C, DLDataType{kDLBfloat, 16, 1}, "C");
  check_contiguous(arg_C, "C");
  check_cuda_tensor(arg_C2, "C2");
  check_dtype(arg_C2, DLDataType{kDLBfloat, 16, 1}, "C2");
  check_contiguous(arg_C2, "C2");
  check_cuda_tensor(arg_C3, "C3");
  check_dtype(arg_C3, DLDataType{kDLBfloat, 16, 1}, "C3");
  check_contiguous(arg_C3, "C3");
  check_cuda_tensor(arg_R, "R");
  check_dtype(arg_R, DLDataType{kDLBfloat, 16, 1}, "R");
  check_contiguous(arg_R, "R");
  check_cuda_tensor(arg_COS, "COS");
  check_dtype(arg_COS, DLDataType{kDLFloat, 32, 1}, "COS");
  check_contiguous(arg_COS, "COS");
  check_cuda_tensor(arg_SIN, "SIN");
  check_dtype(arg_SIN, DLDataType{kDLFloat, 32, 1}, "SIN");
  check_contiguous(arg_SIN, "SIN");
  check_cuda_tensor(arg_CS, "CS");
  check_dtype(arg_CS, DLDataType{kDLBfloat, 16, 1}, "CS");
  check_contiguous(arg_CS, "CS");
  check_cuda_tensor(arg_SQ, "SQ");
  check_dtype(arg_SQ, DLDataType{kDLFloat, 32, 1}, "SQ");
  check_contiguous(arg_SQ, "SQ");
  check_cuda_tensor(arg_XW, "XW");
  check_dtype(arg_XW, DLDataType{kDLBfloat, 16, 1}, "XW");
  check_contiguous(arg_XW, "XW");
  check_cuda_tensor(arg_WN, "WN");
  check_dtype(arg_WN, DLDataType{kDLBfloat, 16, 1}, "WN");
  check_contiguous(arg_WN, "WN");
  check_cuda_tensor(arg_WS, "WS");
  check_dtype(arg_WS, DLDataType{kDLFloat, 32, 1}, "WS");
  check_contiguous(arg_WS, "WS");
  check_cuda_tensor(arg_FLAGS, "FLAGS");
  check_dtype(arg_FLAGS, DLDataType{kDLUInt, 32, 1}, "FLAGS");
  check_contiguous(arg_FLAGS, "FLAGS");
  check_cuda_tensor(arg_RT, "RT");
  check_dtype(arg_RT, DLDataType{kDLBfloat, 16, 1}, "RT");
  check_cuda_tensor(arg_CT, "CT");
  check_dtype(arg_CT, DLDataType{kDLBfloat, 16, 1}, "CT");
  check_cuda_tensor(arg_XWT, "XWT");
  check_dtype(arg_XWT, DLDataType{kDLBfloat, 16, 1}, "XWT");
  TVM_FFI_CHECK(arg_M >= -2147483648LL && arg_M <= 2147483647LL, ValueError)
      << "scalar 'M' value " << arg_M
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_m_tiles >= -2147483648LL && arg_m_tiles <= 2147483647LL, ValueError)
      << "scalar 'm_tiles' value " << arg_m_tiles
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_full_tiles >= -2147483648LL && arg_full_tiles <= 2147483647LL, ValueError)
      << "scalar 'full_tiles' value " << arg_full_tiles
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_tail_split >= -2147483648LL && arg_tail_split <= 2147483647LL, ValueError)
      << "scalar 'tail_split' value " << arg_tail_split
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_pf_l2 >= -2147483648LL && arg_pf_l2 <= 2147483647LL, ValueError)
      << "scalar 'pf_l2' value " << arg_pf_l2
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_A2, arg_A, "A2", "A");
  check_same_device(arg_B, arg_A, "B", "A");
  check_same_device(arg_C, arg_A, "C", "A");
  check_same_device(arg_C2, arg_A, "C2", "A");
  check_same_device(arg_C3, arg_A, "C3", "A");
  check_same_device(arg_R, arg_A, "R", "A");
  check_same_device(arg_COS, arg_A, "COS", "A");
  check_same_device(arg_SIN, arg_A, "SIN", "A");
  check_same_device(arg_CS, arg_A, "CS", "A");
  check_same_device(arg_SQ, arg_A, "SQ", "A");
  check_same_device(arg_XW, arg_A, "XW", "A");
  check_same_device(arg_WN, arg_A, "WN", "A");
  check_same_device(arg_WS, arg_A, "WS", "A");
  check_same_device(arg_FLAGS, arg_A, "FLAGS", "A");
  check_same_device(arg_RT, arg_A, "RT", "A");
  check_same_device(arg_CT, arg_A, "CT", "A");
  check_same_device(arg_XWT, arg_A, "XWT", "A");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
#if CAKE_CLUSTER_X == 2
  TVM_FFI_CHECK(grid_x % 2 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";
#elif CAKE_CLUSTER_X == 4
  TVM_FFI_CHECK(grid_x % 4 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (4, 1, 1)";
#elif CAKE_CLUSTER_X != 0
#error "CAKE_CLUSTER_X must be 0 (no cluster), 2 or 4"
#endif

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_A = EncodeTma_A(arg_A);
  CUtensorMap p_A2 = EncodeTma_A2(arg_A2);
  CUtensorMap p_B = EncodeTma_B(arg_B);
  __nv_bfloat16* p_C = static_cast<__nv_bfloat16*>(arg_C.data_ptr());
  __nv_bfloat16* p_C2 = static_cast<__nv_bfloat16*>(arg_C2.data_ptr());
  __nv_bfloat16* p_C3 = static_cast<__nv_bfloat16*>(arg_C3.data_ptr());
  __nv_bfloat16* p_R = static_cast<__nv_bfloat16*>(arg_R.data_ptr());
  float* p_COS = static_cast<float*>(arg_COS.data_ptr());
  float* p_SIN = static_cast<float*>(arg_SIN.data_ptr());
  __nv_bfloat16* p_CS = static_cast<__nv_bfloat16*>(arg_CS.data_ptr());
  float* p_SQ = static_cast<float*>(arg_SQ.data_ptr());
  __nv_bfloat16* p_XW = static_cast<__nv_bfloat16*>(arg_XW.data_ptr());
  __nv_bfloat16* p_WN = static_cast<__nv_bfloat16*>(arg_WN.data_ptr());
  float* p_WS = static_cast<float*>(arg_WS.data_ptr());
  unsigned int* p_FLAGS = static_cast<unsigned int*>(arg_FLAGS.data_ptr());
  CUtensorMap p_RT = EncodeTma_RT(arg_RT);
  CUtensorMap p_CT = EncodeTma_CT(arg_CT);
  CUtensorMap p_XWT = EncodeTma_XWT(arg_XWT);
  int32_t v_M = (int32_t)arg_M;
  int32_t v_m_tiles = (int32_t)arg_m_tiles;
  int32_t v_full_tiles = (int32_t)arg_full_tiles;
  int32_t v_tail_split = (int32_t)arg_tail_split;
  int32_t v_pf_l2 = (int32_t)arg_pf_l2;
  float v_eps = (float)arg_eps;
  void* kargs[] = {&p_A, &p_A2, &p_B, &p_C, &p_C2, &p_C3, &p_R, &p_COS, &p_SIN, &p_CS, &p_SQ, &p_XW, &p_WN, &p_WS, &p_FLAGS, &p_RT, &p_CT, &p_XWT, &v_M, &v_m_tiles, &v_full_tiles, &v_tail_split, &v_pf_l2, &v_eps};

  static const bool smem_ready = CakeSetMaxDynamicSmem(
      reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), CAKE_DYNAMIC_SMEM);
  (void)smem_ready;
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(CAKE_BLOCK_X), 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
#if CAKE_CLUSTER_X
  cudaLaunchAttribute attrs[3]{};
#else
  cudaLaunchAttribute attrs[1]{};
#endif
  int n = 0;
#if CAKE_CLUSTER_X
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = CAKE_UNSIGNED(CAKE_CLUSTER_X);
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
#endif
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(CAKE_DYNAMIC_SMEM);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

#elif defined(CAKE_PLAN_ATTENTION)

// Q, K, O_tma: boxes (64, 128 | 64, 1, 2) over the axes (64, d3, d2, d1 / 64);
// V: box (64, 128, 1, 1) over the axes (64, d3, d1 / 64, d2).
#define CAKE_TMA_NAME Q
#define CAKE_TMA_RANK 4
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_BOX_3 2
#define CAKE_TMA_AXES 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME K
#define CAKE_TMA_RANK 4
#define CAKE_TMA_BOX_1 64
#define CAKE_TMA_BOX_3 2
#define CAKE_TMA_AXES 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME V
#define CAKE_TMA_RANK 4
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_BOX_3 1
#define CAKE_TMA_AXES 1
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

#define CAKE_TMA_NAME O_tma
#define CAKE_TMA_RANK 4
#define CAKE_TMA_BOX_1 128
#define CAKE_TMA_BOX_3 2
#define CAKE_TMA_AXES 0
#include "cake_kimi_k3_vision_tower_tma_encode.cuh"

void Run(TensorView arg_Q, TensorView arg_Q_raw, TensorView arg_K, TensorView arg_V, TensorView arg_O, TensorView arg_O_tma, TensorView arg_seg_begin, TensorView arg_seg_len, TensorView arg_unit_table, TensorView arg_probe, int64_t arg_total_tiles, int64_t arg_num_heads, double arg_softmax_scale_log2, TensorView arg_partial_O, TensorView arg_partial_ML, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_Q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));  // binds the context for the encoders
  check_cuda_tensor(arg_Q, "Q");
  check_dtype(arg_Q, DLDataType{kDLBfloat, 16, 1}, "Q");
  check_contiguous(arg_Q, "Q");
  check_cuda_tensor(arg_Q_raw, "Q_raw");
  check_dtype(arg_Q_raw, DLDataType{kDLBfloat, 16, 1}, "Q_raw");
  check_contiguous(arg_Q_raw, "Q_raw");
  check_cuda_tensor(arg_K, "K");
  check_dtype(arg_K, DLDataType{kDLBfloat, 16, 1}, "K");
  check_contiguous(arg_K, "K");
  check_cuda_tensor(arg_V, "V");
  check_dtype(arg_V, DLDataType{kDLBfloat, 16, 1}, "V");
  check_contiguous(arg_V, "V");
  check_cuda_tensor(arg_O, "O");
  check_dtype(arg_O, DLDataType{kDLBfloat, 16, 1}, "O");
  check_contiguous(arg_O, "O");
  check_cuda_tensor(arg_O_tma, "O_tma");
  check_dtype(arg_O_tma, DLDataType{kDLBfloat, 16, 1}, "O_tma");
  check_contiguous(arg_O_tma, "O_tma");
  check_cuda_tensor(arg_seg_begin, "seg_begin");
  check_dtype(arg_seg_begin, DLDataType{kDLInt, 32, 1}, "seg_begin");
  check_contiguous(arg_seg_begin, "seg_begin");
  check_cuda_tensor(arg_seg_len, "seg_len");
  check_dtype(arg_seg_len, DLDataType{kDLInt, 32, 1}, "seg_len");
  check_contiguous(arg_seg_len, "seg_len");
  check_cuda_tensor(arg_unit_table, "unit_table");
  check_dtype(arg_unit_table, DLDataType{kDLInt, 32, 1}, "unit_table");
  check_contiguous(arg_unit_table, "unit_table");
  check_cuda_tensor(arg_probe, "probe");
  check_dtype(arg_probe, DLDataType{kDLUInt, 64, 1}, "probe");
  check_contiguous(arg_probe, "probe");
  TVM_FFI_CHECK(arg_total_tiles >= 0LL && arg_total_tiles <= 4294967295LL, ValueError)
      << "scalar 'total_tiles' value " << arg_total_tiles
      << " is outside u32 range [0, 4294967295]";
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  check_cuda_tensor(arg_partial_O, "partial_O");
  check_dtype(arg_partial_O, DLDataType{kDLFloat, 32, 1}, "partial_O");
  check_contiguous(arg_partial_O, "partial_O");
  check_cuda_tensor(arg_partial_ML, "partial_ML");
  check_dtype(arg_partial_ML, DLDataType{kDLFloat, 32, 1}, "partial_ML");
  check_contiguous(arg_partial_ML, "partial_ML");
  check_same_device(arg_Q_raw, arg_Q, "Q_raw", "Q");
  check_same_device(arg_K, arg_Q, "K", "Q");
  check_same_device(arg_V, arg_Q, "V", "Q");
  check_same_device(arg_O, arg_Q, "O", "Q");
  check_same_device(arg_O_tma, arg_Q, "O_tma", "Q");
  check_same_device(arg_seg_begin, arg_Q, "seg_begin", "Q");
  check_same_device(arg_seg_len, arg_Q, "seg_len", "Q");
  check_same_device(arg_unit_table, arg_Q, "unit_table", "Q");
  check_same_device(arg_probe, arg_Q, "probe", "Q");
  check_same_device(arg_partial_O, arg_Q, "partial_O", "Q");
  check_same_device(arg_partial_ML, arg_Q, "partial_ML", "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
#if CAKE_CLUSTER_X == 2
  TVM_FFI_CHECK(grid_x % 2 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";
#elif CAKE_CLUSTER_X == 4
  TVM_FFI_CHECK(grid_x % 4 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (4, 1, 1)";
#elif CAKE_CLUSTER_X != 0
#error "CAKE_CLUSTER_X must be 0 (no cluster), 2 or 4"
#endif

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  __nv_bfloat16* p_Q_raw = static_cast<__nv_bfloat16*>(arg_Q_raw.data_ptr());
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_V = EncodeTma_V(arg_V);
  __nv_bfloat16* p_O = static_cast<__nv_bfloat16*>(arg_O.data_ptr());
  CUtensorMap p_O_tma = EncodeTma_O_tma(arg_O_tma);
  int* p_seg_begin = static_cast<int*>(arg_seg_begin.data_ptr());
  int* p_seg_len = static_cast<int*>(arg_seg_len.data_ptr());
  int* p_unit_table = static_cast<int*>(arg_unit_table.data_ptr());
  unsigned long long* p_probe = static_cast<unsigned long long*>(arg_probe.data_ptr());
  uint32_t v_total_tiles = (uint32_t)arg_total_tiles;
  int32_t v_num_heads = (int32_t)arg_num_heads;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  float* p_partial_O = static_cast<float*>(arg_partial_O.data_ptr());
  float* p_partial_ML = static_cast<float*>(arg_partial_ML.data_ptr());
  void* kargs[] = {&p_Q, &p_Q_raw, &p_K, &p_V, &p_O, &p_O_tma, &p_seg_begin, &p_seg_len, &p_unit_table, &p_probe, &v_total_tiles, &v_num_heads, &v_softmax_scale_log2, &p_partial_O, &p_partial_ML};

  static const bool smem_ready = CakeSetMaxDynamicSmem(
      reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), CAKE_DYNAMIC_SMEM);
  (void)smem_ready;
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(CAKE_BLOCK_X), 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
#if CAKE_CLUSTER_X
  cudaLaunchAttribute attrs[3]{};
#else
  cudaLaunchAttribute attrs[1]{};
#endif
  int n = 0;
#if CAKE_CLUSTER_X
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = CAKE_UNSIGNED(CAKE_CLUSTER_X);
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
#endif
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(CAKE_DYNAMIC_SMEM);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

#elif defined(CAKE_PLAN_ATTENTION_MERGE)

#if CAKE_CLUSTER_X != 0
#error "this plan launches without cluster attributes"
#endif

void Run(TensorView arg_partial_O, TensorView arg_partial_ML, TensorView arg_merge_table, TensorView arg_seg_begin, TensorView arg_seg_len, TensorView arg_O, int64_t arg_num_heads, int64_t arg_part_rows, double arg_softmax_scale_log2, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_partial_O.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(arg_partial_O, "partial_O");
  check_dtype(arg_partial_O, DLDataType{kDLFloat, 32, 1}, "partial_O");
  check_contiguous(arg_partial_O, "partial_O");
  check_cuda_tensor(arg_partial_ML, "partial_ML");
  check_dtype(arg_partial_ML, DLDataType{kDLFloat, 32, 1}, "partial_ML");
  check_contiguous(arg_partial_ML, "partial_ML");
  check_cuda_tensor(arg_merge_table, "merge_table");
  check_dtype(arg_merge_table, DLDataType{kDLInt, 32, 1}, "merge_table");
  check_contiguous(arg_merge_table, "merge_table");
  check_cuda_tensor(arg_seg_begin, "seg_begin");
  check_dtype(arg_seg_begin, DLDataType{kDLInt, 32, 1}, "seg_begin");
  check_contiguous(arg_seg_begin, "seg_begin");
  check_cuda_tensor(arg_seg_len, "seg_len");
  check_dtype(arg_seg_len, DLDataType{kDLInt, 32, 1}, "seg_len");
  check_contiguous(arg_seg_len, "seg_len");
  check_cuda_tensor(arg_O, "O");
  check_dtype(arg_O, DLDataType{kDLBfloat, 16, 1}, "O");
  check_contiguous(arg_O, "O");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_part_rows >= -2147483648LL && arg_part_rows <= 2147483647LL, ValueError)
      << "scalar 'part_rows' value " << arg_part_rows
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_partial_ML, arg_partial_O, "partial_ML", "partial_O");
  check_same_device(arg_merge_table, arg_partial_O, "merge_table", "partial_O");
  check_same_device(arg_seg_begin, arg_partial_O, "seg_begin", "partial_O");
  check_same_device(arg_seg_len, arg_partial_O, "seg_len", "partial_O");
  check_same_device(arg_O, arg_partial_O, "O", "partial_O");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  float* p_partial_O = static_cast<float*>(arg_partial_O.data_ptr());
  float* p_partial_ML = static_cast<float*>(arg_partial_ML.data_ptr());
  int* p_merge_table = static_cast<int*>(arg_merge_table.data_ptr());
  int* p_seg_begin = static_cast<int*>(arg_seg_begin.data_ptr());
  int* p_seg_len = static_cast<int*>(arg_seg_len.data_ptr());
  __nv_bfloat16* p_O = static_cast<__nv_bfloat16*>(arg_O.data_ptr());
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_part_rows = (int32_t)arg_part_rows;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  void* kargs[] = {&p_partial_O, &p_partial_ML, &p_merge_table, &p_seg_begin, &p_seg_len, &p_O, &v_num_heads, &v_part_rows, &v_softmax_scale_log2};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(CAKE_BLOCK_X), 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
  cudaLaunchAttribute attrs[1]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(CAKE_DYNAMIC_SMEM);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

#elif defined(CAKE_PLAN_MERGE)

#if CAKE_CLUSTER_X != 0
#error "this plan launches without cluster attributes"
#endif

void Run(TensorView arg_x, TensorView arg_norm_weight, TensorView arg_merge_table, TensorView arg_m_out, double arg_eps, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_x.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(arg_x, "x");
  check_dtype(arg_x, DLDataType{kDLBfloat, 16, 1}, "x");
  check_contiguous(arg_x, "x");
  check_cuda_tensor(arg_norm_weight, "norm_weight");
  check_dtype(arg_norm_weight, DLDataType{kDLBfloat, 16, 1}, "norm_weight");
  check_contiguous(arg_norm_weight, "norm_weight");
  check_cuda_tensor(arg_merge_table, "merge_table");
  check_dtype(arg_merge_table, DLDataType{kDLInt, 32, 1}, "merge_table");
  check_contiguous(arg_merge_table, "merge_table");
  check_cuda_tensor(arg_m_out, "m_out");
  check_dtype(arg_m_out, DLDataType{kDLBfloat, 16, 1}, "m_out");
  check_contiguous(arg_m_out, "m_out");
  check_same_device(arg_norm_weight, arg_x, "norm_weight", "x");
  check_same_device(arg_merge_table, arg_x, "merge_table", "x");
  check_same_device(arg_m_out, arg_x, "m_out", "x");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __nv_bfloat16* p_x = static_cast<__nv_bfloat16*>(arg_x.data_ptr());
  __nv_bfloat16* p_norm_weight = static_cast<__nv_bfloat16*>(arg_norm_weight.data_ptr());
  int* p_merge_table = static_cast<int*>(arg_merge_table.data_ptr());
  __nv_bfloat16* p_m_out = static_cast<__nv_bfloat16*>(arg_m_out.data_ptr());
  float v_eps = (float)arg_eps;
  void* kargs[] = {&p_x, &p_norm_weight, &p_merge_table, &p_m_out, &v_eps};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(CAKE_BLOCK_X), 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
  cudaLaunchAttribute attrs[1]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(CAKE_DYNAMIC_SMEM);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

#elif defined(CAKE_PLAN_RMSNORM_APPLY)

#if CAKE_CLUSTER_X != 0
#error "this plan launches without cluster attributes"
#endif

void Run(TensorView arg_y, TensorView arg_weight, TensorView arg_rowsumsq, int64_t arg_M, double arg_eps, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_y.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(arg_y, "y");
  check_dtype(arg_y, DLDataType{kDLBfloat, 16, 1}, "y");
  check_contiguous(arg_y, "y");
  check_cuda_tensor(arg_weight, "weight");
  check_dtype(arg_weight, DLDataType{kDLBfloat, 16, 1}, "weight");
  check_contiguous(arg_weight, "weight");
  check_cuda_tensor(arg_rowsumsq, "rowsumsq");
  check_dtype(arg_rowsumsq, DLDataType{kDLFloat, 32, 1}, "rowsumsq");
  check_contiguous(arg_rowsumsq, "rowsumsq");
  TVM_FFI_CHECK(arg_M >= -2147483648LL && arg_M <= 2147483647LL, ValueError)
      << "scalar 'M' value " << arg_M
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_weight, arg_y, "weight", "y");
  check_same_device(arg_rowsumsq, arg_y, "rowsumsq", "y");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __nv_bfloat16* p_y = static_cast<__nv_bfloat16*>(arg_y.data_ptr());
  __nv_bfloat16* p_weight = static_cast<__nv_bfloat16*>(arg_weight.data_ptr());
  float* p_rowsumsq = static_cast<float*>(arg_rowsumsq.data_ptr());
  int32_t v_M = (int32_t)arg_M;
  float v_eps = (float)arg_eps;
  void* kargs[] = {&p_y, &p_weight, &p_rowsumsq, &v_M, &v_eps};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(CAKE_BLOCK_X), 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
  cudaLaunchAttribute attrs[1]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(CAKE_DYNAMIC_SMEM);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

#endif

}  // namespace CAKE_PROGRAM_SHIM

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_PROGRAM_SHIM::Run);
