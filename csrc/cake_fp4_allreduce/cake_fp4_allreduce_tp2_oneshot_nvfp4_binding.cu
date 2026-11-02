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

// Host launcher shared by the Cake FP4-transport all-reduce world-size-2 one-shot all-reduce
// programs with NVFP4 input: tensor-metadata checks, the one-time dynamic shared-memory opt-in and
// one launch.  The loader selects the program with -DCAKE_FP4_ALLREDUCE_KERNEL=<kernel symbol>,
// -DCAKE_FP4_ALLREDUCE_THREADS=<block> and -DCAKE_FP4_ALLREDUCE_DYNAMIC_SMEM=<dynamic_smem_bytes>
// (the registry record's block and dynamic_smem_bytes).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#ifndef CAKE_FP4_ALLREDUCE_KERNEL
#error "CAKE_FP4_ALLREDUCE_KERNEL: the loader passes the program's kernel symbol with -D"
#endif
#ifndef CAKE_FP4_ALLREDUCE_THREADS
#error "CAKE_FP4_ALLREDUCE_THREADS: the loader passes the registry block with -D"
#endif
#ifndef CAKE_FP4_ALLREDUCE_DYNAMIC_SMEM
#error "CAKE_FP4_ALLREDUCE_DYNAMIC_SMEM: the loader passes the registry dynamic_smem_bytes with -D"
#endif
#ifndef CAKE_STRINGIZE
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#endif

using tvm::ffi::TensorView;

// clang-format off
extern "C" __global__ void CAKE_FP4_ALLREDUCE_KERNEL(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ q_wire, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap);
// clang-format on

namespace {

constexpr int64_t kWorldSize = 2;
constexpr int64_t kCapMax = 103563;
constexpr int64_t kGridMax = 4096;
constexpr int64_t kRecordBytes = 3456;
constexpr int64_t kHidden = 6144;
constexpr int64_t kTableEntries = 2;
constexpr int64_t kCtrlWords = 16384;
constexpr unsigned kThreads = CAKE_FP4_ALLREDUCE_THREADS;
constexpr unsigned kDynamicSmemBytes = CAKE_FP4_ALLREDUCE_DYNAMIC_SMEM;

int64_t byte_size(const TensorView& t) { return t.numel() * get_element_size(t); }

// [rows, cols]: the dense row-major matrix the kernel indexes.
bool is_matrix(const TensorView& t, int64_t rows, int64_t cols) {
  return t.ndim() == 2 && t.size(0) == rows && t.size(1) == cols;
}

}  // namespace

void run(TensorView xq, TensorView gs, TensorView out, TensorView q_wire, TensorView peer_ptrs,
         TensorView ctrl, int64_t rank, int64_t rows, int64_t cap, int64_t grid_x,
         bool launch_with_pdl) {
  TVM_FFI_CHECK(rank >= 0 && rank < kWorldSize, ValueError)
      << "rank must be in [0, " << kWorldSize << "), got " << rank;
  TVM_FFI_CHECK(cap >= 1 && cap <= kCapMax, ValueError)
      << "cap must be in [1, " << kCapMax << "], got " << cap;
  TVM_FFI_CHECK(rows >= 1 && rows <= cap, ValueError)
      << "rows must be in [1, cap = " << cap << "], got " << rows;
  TVM_FFI_CHECK(grid_x >= 1 && grid_x <= kGridMax, ValueError)
      << "grid_x must be in [1, " << kGridMax << "], got " << grid_x;
  const int64_t wire_bytes = 20736 * cap;
  check_cuda_tensor(xq, "xq");
  check_dtype(xq, dl_uint8, "xq");
  check_contiguous(xq, "xq");
  TVM_FFI_CHECK(is_matrix(xq, rows, kRecordBytes), ValueError)
      << "xq must be [rows, " << kRecordBytes << "] NVFP4 records with rows = " << rows;
  check_cuda_tensor(gs, "gs");
  check_dtype(gs, dl_float32, "gs");
  check_contiguous(gs, "gs");
  check_same_device(gs, xq, "gs", "xq");
  TVM_FFI_CHECK(gs.numel() >= kWorldSize, ValueError)
      << "gs must hold one global scale per rank (" << kWorldSize << "), got " << gs.numel();
  check_cuda_tensor(out, "out");
  check_dtype(out, dl_bfloat16, "out");
  check_contiguous(out, "out");
  check_same_device(out, xq, "out", "xq");
  TVM_FFI_CHECK(is_matrix(out, rows, kHidden), ValueError)
      << "out must be [rows, " << kHidden << "] with rows = " << rows;
  check_cuda_tensor(q_wire, "q_wire");
  check_contiguous(q_wire, "q_wire");
  check_same_device(q_wire, xq, "q_wire", "xq");
  TVM_FFI_CHECK(byte_size(q_wire) >= wire_bytes, ValueError)
      << "q_wire must hold at least " << wire_bytes << " bytes at cap = " << cap << ", got "
      << byte_size(q_wire);
  check_cuda_tensor(peer_ptrs, "peer_ptrs");
  check_dtype(peer_ptrs, dl_int64, "peer_ptrs");
  check_contiguous(peer_ptrs, "peer_ptrs");
  check_same_device(peer_ptrs, xq, "peer_ptrs", "xq");
  TVM_FFI_CHECK(peer_ptrs.numel() == kTableEntries, ValueError)
      << "peer_ptrs must hold exactly " << kTableEntries << " window addresses, got "
      << peer_ptrs.numel();
  check_cuda_tensor(ctrl, "ctrl");
  TVM_FFI_CHECK(ctrl.dtype() == dl_uint32 || ctrl.dtype() == dl_int32, TypeError)
      << "ctrl must be uint32 or int32";
  check_contiguous(ctrl, "ctrl");
  check_same_device(ctrl, xq, "ctrl", "xq");
  TVM_FFI_CHECK(ctrl.numel() == kCtrlWords, ValueError)
      << "ctrl must hold exactly " << kCtrlWords << " control words, got " << ctrl.numel();
  const DLDevice device = xq.device();
  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const void* kernel = reinterpret_cast<const void*>(&CAKE_FP4_ALLREDUCE_KERNEL);
  static const cudaError_t smem_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicSmemBytes));
  TVM_FFI_CHECK(smem_status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: "
      << cudaGetErrorString(smem_status);
  void* p_xq = xq.data_ptr();
  void* p_gs = gs.data_ptr();
  void* p_out = out.data_ptr();
  void* p_q_wire = q_wire.data_ptr();
  void* p_peer_ptrs = peer_ptrs.data_ptr();
  void* p_ctrl = ctrl.data_ptr();
  int32_t v_rank = static_cast<int32_t>(rank);
  int32_t v_rows = static_cast<int32_t>(rows);
  int32_t v_cap = static_cast<int32_t>(cap);
  void* args[9];
  args[0] = &p_xq;
  args[1] = &p_gs;
  args[2] = &p_out;
  args[3] = &p_q_wire;
  args[4] = &p_peer_ptrs;
  args[5] = &p_ctrl;
  args[6] = &v_rank;
  args[7] = &v_rows;
  args[8] = &v_cap;
  cudaLaunchAttribute attrs[1]{};
  unsigned num_attrs = 0;
  if (launch_with_pdl) {
    attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;
  const cudaError_t status = cudaLaunchKernelExC(&config, kernel, args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << CAKE_STRINGIZE(CAKE_FP4_ALLREDUCE_KERNEL) " launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
