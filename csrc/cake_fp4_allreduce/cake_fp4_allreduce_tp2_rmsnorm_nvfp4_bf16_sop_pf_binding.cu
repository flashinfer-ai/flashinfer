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

// Host launcher of the Cake FP4-transport all-reduce family program tp2_rmsnorm_nvfp4_bf16_sop_pf:
// tensor-metadata checks, the one-time dynamic shared-memory opt-in and one launch.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;

// clang-format off
extern "C" __global__ void cake_fp4_allreduce_tp2_rmsnorm_nvfp4_bf16_sop_pf(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias);
// clang-format on

namespace {

constexpr int64_t kWorldSize = 2;
constexpr int64_t kCapMax = 103563;
constexpr int64_t kGridMax = 2147483647;
constexpr int64_t kRecordBytes = 3456;
constexpr int64_t kHidden = 6144;
constexpr int64_t kTableEntries = 4;
constexpr int64_t kCtrlWords = 4;
constexpr unsigned kThreads = 288;
constexpr unsigned kDynamicSmemBytes = 195072;

int64_t byte_size(const TensorView& t) { return t.numel() * get_element_size(t); }

// [rows, cols]: the dense row-major matrix the kernel indexes.
bool is_matrix(const TensorView& t, int64_t rows, int64_t cols) {
  return t.ndim() == 2 && t.size(0) == rows && t.size(1) == cols;
}

}  // namespace

void run(TensorView xq, TensorView gs, TensorView res_in, TensorView w, TensorView out,
         TensorView res_out, TensorView os_wire, TensorView os_rs, TensorView peer_ptrs,
         TensorView ctrl, int64_t rank, int64_t rows, int64_t cap, double eps, double wbias,
         int64_t grid_x, bool launch_with_pdl) {
  TVM_FFI_CHECK(rank >= 0 && rank < kWorldSize, ValueError)
      << "rank must be in [0, " << kWorldSize << "), got " << rank;
  TVM_FFI_CHECK(cap >= 1 && cap <= kCapMax, ValueError)
      << "cap must be in [1, " << kCapMax << "], got " << cap;
  TVM_FFI_CHECK(rows >= 1 && rows <= cap, ValueError)
      << "rows must be in [1, cap = " << cap << "], got " << rows;
  TVM_FFI_CHECK(grid_x >= 1 && grid_x <= kGridMax, ValueError)
      << "grid_x must be in [1, " << kGridMax << "], got " << grid_x;
  const int64_t wire_bytes = 20736 * cap;
  const int64_t rsplane_bytes = 96 * cap;
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
  check_cuda_tensor(res_in, "res_in");
  check_dtype(res_in, dl_bfloat16, "res_in");
  check_contiguous(res_in, "res_in");
  check_same_device(res_in, xq, "res_in", "xq");
  TVM_FFI_CHECK(is_matrix(res_in, rows, kHidden), ValueError)
      << "res_in must be [rows, " << kHidden << "] with rows = " << rows;
  check_cuda_tensor(w, "w");
  check_dtype(w, dl_bfloat16, "w");
  check_contiguous(w, "w");
  check_same_device(w, xq, "w", "xq");
  TVM_FFI_CHECK(w.numel() == kHidden, ValueError)
      << "w must hold " << kHidden << " weights, got " << w.numel();
  check_cuda_tensor(out, "out");
  check_dtype(out, dl_bfloat16, "out");
  check_contiguous(out, "out");
  check_same_device(out, xq, "out", "xq");
  TVM_FFI_CHECK(is_matrix(out, rows, kHidden), ValueError)
      << "out must be [rows, " << kHidden << "] with rows = " << rows;
  check_cuda_tensor(res_out, "res_out");
  check_dtype(res_out, dl_bfloat16, "res_out");
  check_contiguous(res_out, "res_out");
  check_same_device(res_out, xq, "res_out", "xq");
  TVM_FFI_CHECK(is_matrix(res_out, rows, kHidden), ValueError)
      << "res_out must be [rows, " << kHidden << "] with rows = " << rows;
  check_cuda_tensor(os_wire, "os_wire");
  check_contiguous(os_wire, "os_wire");
  check_same_device(os_wire, xq, "os_wire", "xq");
  TVM_FFI_CHECK(byte_size(os_wire) >= wire_bytes, ValueError)
      << "os_wire must hold at least " << wire_bytes << " bytes at cap = " << cap << ", got "
      << byte_size(os_wire);
  check_cuda_tensor(os_rs, "os_rs");
  check_contiguous(os_rs, "os_rs");
  check_same_device(os_rs, xq, "os_rs", "xq");
  TVM_FFI_CHECK(byte_size(os_rs) >= rsplane_bytes, ValueError)
      << "os_rs must hold at least " << rsplane_bytes << " bytes at cap = " << cap << ", got "
      << byte_size(os_rs);
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
  const void* kernel =
      reinterpret_cast<const void*>(&cake_fp4_allreduce_tp2_rmsnorm_nvfp4_bf16_sop_pf);
  static const cudaError_t smem_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicSmemBytes));
  TVM_FFI_CHECK(smem_status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: "
      << cudaGetErrorString(smem_status);
  void* p_xq = xq.data_ptr();
  void* p_gs = gs.data_ptr();
  void* p_res_in = res_in.data_ptr();
  void* p_w = w.data_ptr();
  void* p_out = out.data_ptr();
  void* p_res_out = res_out.data_ptr();
  void* p_os_wire = os_wire.data_ptr();
  void* p_os_rs = os_rs.data_ptr();
  void* p_peer_ptrs = peer_ptrs.data_ptr();
  void* p_ctrl = ctrl.data_ptr();
  int32_t v_rank = static_cast<int32_t>(rank);
  int32_t v_rows = static_cast<int32_t>(rows);
  int32_t v_cap = static_cast<int32_t>(cap);
  float v_eps = static_cast<float>(eps);
  float v_wbias = static_cast<float>(wbias);
  void* args[15];
  args[0] = &p_xq;
  args[1] = &p_gs;
  args[2] = &p_res_in;
  args[3] = &p_w;
  args[4] = &p_out;
  args[5] = &p_res_out;
  args[6] = &p_os_wire;
  args[7] = &p_os_rs;
  args[8] = &p_peer_ptrs;
  args[9] = &p_ctrl;
  args[10] = &v_rank;
  args[11] = &v_rows;
  args[12] = &v_cap;
  args[13] = &v_eps;
  args[14] = &v_wbias;
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
      << "cake_fp4_allreduce_tp2_rmsnorm_nvfp4_bf16_sop_pf launch failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
