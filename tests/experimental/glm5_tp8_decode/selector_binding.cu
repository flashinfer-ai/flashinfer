/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
// Test-only launchers. Both modes call the selector in long_topk.cuh directly.
#include "long_topk.cuh"
#include "tvm_ffi_utils.h"

namespace selector = flashinfer::glm5_tp8::indexer_q4::long_topk;

namespace {
__global__ void summarize(const float* logits, int stride, const int32_t* position,
                          uint64_t* workspace) {
  const int q = blockIdx.y;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const float value = id < stride ? selector::load_cg(logits + size_t(q) * stride + id) : 0;
  selector::publish_group(value, id, *position + q + 1, stride,
                          workspace + size_t(q) * selector::summary_stride(stride));
}

__global__ void select_rows(const float* logits, int stride, const int32_t* position, int32_t* out,
                            const uint64_t* workspace, bool grouped) {
  extern __shared__ __align__(16) unsigned char scratch[];
  const int q = blockIdx.x;
  const int rows = *position + q + 1;
  if (rows < 1 || rows > stride || rows > selector::kLogicalMax) __trap();
  selector::select(logits + size_t(q) * stride, rows,
                   workspace + size_t(q) * selector::summary_stride(stride),
                   out + size_t(q) * selector::kTopK, scratch, grouped);
}

void initialize() {
  cudaFuncAttributes attributes{};
  auto status = cudaFuncGetAttributes(&attributes, summarize);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  status = cudaFuncSetAttribute(select_rows, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(selector::Scratch));
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView position, tvm::ffi::TensorView out,
         tvm::ffi::TensorView workspace, bool grouped) {
  CHECK_INPUT_AND_TYPE(logits, dl_float32);
  CHECK_INPUT_AND_TYPE(position, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_int32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, logits);
  CHECK_DIM(2, out);
  TVM_FFI_ICHECK_EQ(logits.size(0), 4);
  TVM_FFI_ICHECK_EQ(out.size(0), 4);
  TVM_FFI_ICHECK_EQ(out.size(1), selector::kTopK);
  TVM_FFI_ICHECK_EQ(position.numel(), 1);
  const int64_t stride = logits.size(1);
  TVM_FFI_ICHECK(stride >= 4 && stride <= selector::kLogicalMax + 8);
  // The operator allocates int32[4, C] for this scratch; C includes the padding rows.
  TVM_FFI_ICHECK_EQ(workspace.numel(), 16 * stride);
  for (auto value : {position, out, workspace}) {
    TVM_FFI_ICHECK_EQ(value.device().device_id, logits.device().device_id);
  }
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16, 0);
  tvm::ffi::CUDADeviceGuard guard(logits.device().device_id);
  auto stream = get_stream(logits.device());
  const auto* scores = static_cast<const float*>(logits.data_ptr());
  const auto* pos = static_cast<const int32_t*>(position.data_ptr());
  auto* ids = static_cast<int32_t*>(out.data_ptr());
  auto* temp = static_cast<uint64_t*>(workspace.data_ptr());
  if (grouped) {
    summarize<<<dim3((stride + 255) / 256, 4), 256, 0, stream>>>(scores, stride, pos, temp);
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
  }
  select_rows<<<4, selector::kThreads, sizeof(selector::Scratch), stream>>>(scores, stride, pos,
                                                                            ids, temp, grouped);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
