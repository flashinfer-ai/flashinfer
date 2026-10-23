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

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "tvm_ffi_utils.h"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE 1
#else
#define FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE 0
#endif

namespace flashinfer {
namespace gemm {
namespace cublaslt_bias {

inline constexpr int kMaxAlgorithms = 100;
inline constexpr std::size_t kAlgorithmBytes = sizeof(cublasLtMatmulAlgo_t);
static_assert(kAlgorithmBytes == 64,
              "the serialized cuBLASLt algorithm buffer assumes 64-byte algorithms");

class ThreadLocalHandleCache {
 public:
  ThreadLocalHandleCache() = default;
  ThreadLocalHandleCache(ThreadLocalHandleCache const&) = delete;
  ThreadLocalHandleCache& operator=(ThreadLocalHandleCache const&) = delete;

  ~ThreadLocalHandleCache() noexcept {
    for (auto const& entry : handles_) {
      if (entry.second != nullptr) (void)cublasLtDestroy(entry.second);
    }
  }

  cublasStatus_t Get(int device, cublasLtHandle_t* handle) {
    auto const found = handles_.find(device);
    if (found != handles_.end()) {
      *handle = found->second;
      return CUBLAS_STATUS_SUCCESS;
    }
    cublasLtHandle_t created = nullptr;
    cublasStatus_t const status = cublasLtCreate(&created);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    handles_.emplace(device, created);
    *handle = created;
    return CUBLAS_STATUS_SUCCESS;
  }

 private:
  std::unordered_map<int, cublasLtHandle_t> handles_;
};

inline cublasStatus_t GetHandle(int device, cublasLtHandle_t* handle) {
  static thread_local ThreadLocalHandleCache cache;
  return cache.Get(device, handle);
}

#if FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE

// Row-major Y[M,N] = alpha * X[M,K] * W[N,K]^T + bias[N] is represented as
// column-major Y^T[N,M] = W[N,K] * X^T[K,M]. cuBLASLt therefore consumes W as
// operand A, X as operand B, and a length-N bias over the rows of Y^T.
class Descriptors {
 public:
  Descriptors() = default;
  Descriptors(Descriptors const&) = delete;
  Descriptors& operator=(Descriptors const&) = delete;

  ~Descriptors() noexcept {
    if (output_layout_ != nullptr) (void)cublasLtMatrixLayoutDestroy(output_layout_);
    if (b_layout_ != nullptr) (void)cublasLtMatrixLayoutDestroy(b_layout_);
    if (a_layout_ != nullptr) (void)cublasLtMatrixLayoutDestroy(a_layout_);
    if (matmul_desc_ != nullptr) (void)cublasLtMatmulDescDestroy(matmul_desc_);
  }

  cublasStatus_t Initialize(int m, int n, int k, void const* x_scale, void const* w_scale,
                            void const* bias) {
    cublasStatus_t status = cublasLtMatmulDescCreate(&matmul_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    cublasOperation_t const trans_a = CUBLAS_OP_T;
    cublasOperation_t const trans_b = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a,
                                            sizeof(trans_a));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b,
                                            sizeof(trans_b));
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    cublasLtPointerMode_t const pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                            &pointer_mode, sizeof(pointer_mode));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    int8_t const fast_accum = 0;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                            &fast_accum, sizeof(fast_accum));
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    status = cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_4F_E2M1, k, n, k);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_4F_E2M1, k, m, k);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasLtMatrixLayoutCreate(&output_layout_, CUDA_R_16BF, n, m, n);
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    void const* const lt_a_scale = w_scale;
    void const* const lt_b_scale = x_scale;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &lt_a_scale, sizeof(lt_a_scale));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &lt_b_scale, sizeof(lt_b_scale));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    cublasLtMatmulMatrixScale_t const scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                            &scale_mode, sizeof(scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                            &scale_mode, sizeof(scale_mode));
    if (status != CUBLAS_STATUS_SUCCESS) return status;

    cudaDataType_t const bias_type = CUDA_R_16BF;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                            &bias_type, sizeof(bias_type));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    void const* const bias_pointer = bias;
    status = cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias_pointer, sizeof(bias_pointer));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    cublasLtEpilogue_t const epilogue = CUBLASLT_EPILOGUE_BIAS;
    return cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                          sizeof(epilogue));
  }

  cublasLtMatmulDesc_t matmul_desc() const { return matmul_desc_; }
  cublasLtMatrixLayout_t a_layout() const { return a_layout_; }
  cublasLtMatrixLayout_t b_layout() const { return b_layout_; }
  cublasLtMatrixLayout_t output_layout() const { return output_layout_; }

 private:
  cublasLtMatmulDesc_t matmul_desc_ = nullptr;
  cublasLtMatrixLayout_t a_layout_ = nullptr;
  cublasLtMatrixLayout_t b_layout_ = nullptr;
  cublasLtMatrixLayout_t output_layout_ = nullptr;
};

cublasStatus_t GetAlgorithms(int m, int n, int k, void const* x_scale, void const* w_scale,
                             void const* bias, std::size_t workspace_bytes, cublasLtHandle_t handle,
                             void* algorithm_buffer, int algorithm_capacity, int* algorithm_count) {
  *algorithm_count = 0;
  Descriptors desc;
  cublasStatus_t status = desc.Initialize(m, n, k, x_scale, w_scale, bias);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status =
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    (void)cublasLtMatmulPreferenceDestroy(preference);
    return status;
  }

  int const requested = std::max(0, std::min(algorithm_capacity, kMaxAlgorithms));
  std::array<cublasLtMatmulHeuristicResult_t, kMaxAlgorithms> results{};
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      handle, desc.matmul_desc(), desc.a_layout(), desc.b_layout(), desc.output_layout(),
      desc.output_layout(), preference, requested, results.data(), &returned);
  (void)cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  auto* serialized = static_cast<std::uint8_t*>(algorithm_buffer);
  int accepted = 0;
  for (int i = 0; i < returned && accepted < requested; ++i) {
    if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > workspace_bytes) {
      continue;
    }
    std::memcpy(serialized + static_cast<std::size_t>(accepted) * kAlgorithmBytes, &results[i].algo,
                kAlgorithmBytes);
    ++accepted;
  }
  *algorithm_count = accepted;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t RunWithAlgorithm(void const* x, void const* w, void const* x_scale,
                                void const* w_scale, float const* alpha, float const* beta,
                                void const* bias, void* output, int m, int n, int k,
                                void* workspace, std::size_t workspace_bytes,
                                cublasLtHandle_t handle, cudaStream_t stream,
                                void const* algorithm_buffer, int algorithm_index) {
  Descriptors desc;
  cublasStatus_t status = desc.Initialize(m, n, k, x_scale, w_scale, bias);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  cublasLtMatmulAlgo_t algorithm{};
  std::memcpy(&algorithm,
              static_cast<std::uint8_t const*>(algorithm_buffer) +
                  static_cast<std::size_t>(algorithm_index) * kAlgorithmBytes,
              kAlgorithmBytes);
  return cublasLtMatmul(handle, desc.matmul_desc(), alpha, w, desc.a_layout(), x, desc.b_layout(),
                        beta, output, desc.output_layout(), output, desc.output_layout(),
                        &algorithm, workspace, workspace_bytes, stream);
}

#endif  // FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE

}  // namespace cublaslt_bias
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {
namespace {

using flashinfer::gemm::cublaslt_bias::kAlgorithmBytes;

struct ProblemShape {
  int m;
  int n;
  int k;
};

int64_t PadUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

int64_t SwizzledScaleSize(int64_t rows, int64_t k) { return PadUp(rows, 128) * PadUp(k / 16, 4); }

void CheckAlignment(TensorView tensor, std::uintptr_t alignment, char const* name) {
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % alignment, 0)
      << name << " data pointer must be " << alignment << "-byte aligned";
}

ProblemShape CheckProblem(TensorView x, TensorView w, TensorView x_scale, TensorView w_scale,
                          TensorView bias, TensorView output, TensorView workspace) {
  CHECK_INPUT_AND_TYPE(x, dl_uint8);
  CHECK_INPUT_AND_TYPE(w, dl_uint8);
  CHECK_INPUT_AND_TYPE(x_scale, dl_uint8);
  CHECK_INPUT_AND_TYPE(w_scale, dl_uint8);
  CHECK_INPUT_AND_TYPE(bias, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DEVICE(w, x);
  CHECK_DEVICE(x_scale, x);
  CHECK_DEVICE(w_scale, x);
  CHECK_DEVICE(bias, x);
  CHECK_DEVICE(output, x);
  CHECK_DEVICE(workspace, x);
  CHECK_DIM(2, x);
  CHECK_DIM(2, w);
  CHECK_DIM(1, bias);
  CHECK_DIM(2, output);

  int64_t const m = x.size(0);
  int64_t const n = w.size(0);
  int64_t const k_packed = x.size(1);
  TVM_FFI_ICHECK_GT(m, 0) << "M must be positive";
  TVM_FFI_ICHECK_GT(n, 0) << "N must be positive";
  TVM_FFI_ICHECK_GT(k_packed, 0) << "K must be positive";
  TVM_FFI_ICHECK_EQ(w.size(1), k_packed) << "packed K dimensions must match";
  TVM_FFI_ICHECK_LE(m, std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_LE(n, std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_LE(k_packed, std::numeric_limits<int>::max() / 2);
  int64_t const k = k_packed * 2;
  TVM_FFI_ICHECK_EQ(k % 32, 0) << "K must be divisible by 32";
  TVM_FFI_ICHECK_EQ(n % 8, 0) << "N must be divisible by 8";
  TVM_FFI_ICHECK_EQ(output.size(0), m) << "output M dimension mismatch";
  TVM_FFI_ICHECK_EQ(output.size(1), n) << "output N dimension mismatch";
  TVM_FFI_ICHECK_EQ(bias.size(0), n) << "bias must have shape [N]";
  TVM_FFI_ICHECK_GE(x_scale.numel(), SwizzledScaleSize(m, k));
  TVM_FFI_ICHECK_GE(w_scale.numel(), SwizzledScaleSize(n, k));
  CheckAlignment(x, 16, "x");
  CheckAlignment(w, 16, "w");
  CheckAlignment(x_scale, 16, "x_scale");
  CheckAlignment(w_scale, 16, "w_scale");
  CheckAlignment(bias, 16, "bias");
  CheckAlignment(output, 16, "output");
  if (workspace.numel() != 0) CheckAlignment(workspace, 256, "workspace");

  ffi::CUDADeviceGuard guard(x.device().device_id);
  cudaDeviceProp properties{};
  cudaError_t const cuda_status = cudaGetDeviceProperties(&properties, x.device().device_id);
  TVM_FFI_ICHECK_EQ(cuda_status, cudaSuccess)
      << "cudaGetDeviceProperties failed: " << cudaGetErrorString(cuda_status);
  TVM_FFI_ICHECK_GE(properties.major, 10) << "cuBLASLt NVFP4 GEMM requires a Blackwell-class GPU";
  return {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
}

void CheckAlgorithmBuffer(TensorView algorithm_buffer) {
  CHECK_CPU(algorithm_buffer);
  CHECK_CONTIGUOUS(algorithm_buffer);
  CHECK_INPUT_TYPE(algorithm_buffer, dl_uint8);
}

std::size_t WorkspaceBytes(TensorView workspace) {
  return static_cast<std::size_t>(workspace.numel());
}

}  // namespace

int64_t nvfp4_cublaslt_bias_get_algorithms(TensorView x, TensorView w, TensorView x_scale,
                                           TensorView w_scale, TensorView bias, TensorView output,
                                           TensorView workspace, TensorView algorithm_buffer) {
  ProblemShape const shape = CheckProblem(x, w, x_scale, w_scale, bias, output, workspace);
  CheckAlgorithmBuffer(algorithm_buffer);

#if FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE
  int64_t const capacity = algorithm_buffer.numel() / static_cast<int64_t>(kAlgorithmBytes);
  TVM_FFI_ICHECK(capacity > 0 && capacity <= std::numeric_limits<int>::max());
  ffi::CUDADeviceGuard guard(x.device().device_id);
  cublasLtHandle_t handle = nullptr;
  cublasStatus_t status = flashinfer::gemm::cublaslt_bias::GetHandle(x.device().device_id, &handle);
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "cublasLtCreate failed with status " << static_cast<int>(status);
  int count = 0;
  status = flashinfer::gemm::cublaslt_bias::GetAlgorithms(
      shape.m, shape.n, shape.k, x_scale.data_ptr(), w_scale.data_ptr(), bias.data_ptr(),
      WorkspaceBytes(workspace), handle, algorithm_buffer.data_ptr(), static_cast<int>(capacity),
      &count);
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "bias-enabled cublasLtMatmulAlgoGetHeuristic failed with status "
      << static_cast<int>(status);
  return count;
#else
  TVM_FFI_LOG_AND_THROW(NotImplementedError)
      << "cuBLASLt NVFP4 GEMM requires CUDA Toolkit 12.8 or newer";
  return 0;
#endif
}

void nvfp4_cublaslt_bias_run_with_algorithm(TensorView x, TensorView w, TensorView x_scale,
                                            TensorView w_scale, TensorView alpha, TensorView beta,
                                            TensorView bias, TensorView output,
                                            TensorView workspace, TensorView algorithm_buffer,
                                            int64_t algorithm_count, int64_t algorithm_index) {
  ProblemShape const shape = CheckProblem(x, w, x_scale, w_scale, bias, output, workspace);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(beta, dl_float32);
  CHECK_DEVICE(alpha, x);
  CHECK_DEVICE(beta, x);
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1);
  TVM_FFI_ICHECK_EQ(beta.numel(), 1);
  CheckAlignment(alpha, alignof(float), "alpha");
  CheckAlignment(beta, alignof(float), "beta");
  CheckAlgorithmBuffer(algorithm_buffer);
  int64_t const capacity = algorithm_buffer.numel() / static_cast<int64_t>(kAlgorithmBytes);
  TVM_FFI_ICHECK(algorithm_count > 0 && algorithm_count <= capacity);
  TVM_FFI_ICHECK(algorithm_index >= 0 && algorithm_index < algorithm_count);

#if FLASHINFER_NVFP4_CUBLASLT_BIAS_AVAILABLE
  ffi::CUDADeviceGuard guard(x.device().device_id);
  cublasLtHandle_t handle = nullptr;
  cublasStatus_t status = flashinfer::gemm::cublaslt_bias::GetHandle(x.device().device_id, &handle);
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "cublasLtCreate failed with status " << static_cast<int>(status);
  status = flashinfer::gemm::cublaslt_bias::RunWithAlgorithm(
      x.data_ptr(), w.data_ptr(), x_scale.data_ptr(), w_scale.data_ptr(),
      static_cast<float const*>(alpha.data_ptr()), static_cast<float const*>(beta.data_ptr()),
      bias.data_ptr(), output.data_ptr(), shape.m, shape.n, shape.k, workspace.data_ptr(),
      WorkspaceBytes(workspace), handle, get_stream(x.device()), algorithm_buffer.data_ptr(),
      static_cast<int>(algorithm_index));
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "bias-enabled cublasLtMatmul failed with status " << static_cast<int>(status);
#else
  TVM_FFI_LOG_AND_THROW(NotImplementedError)
      << "cuBLASLt NVFP4 GEMM requires CUDA Toolkit 12.8 or newer";
#endif
}

int64_t nvfp4_cublaslt_bias_runtime_version() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  cublasStatus_t status = cublasLtGetProperty(MAJOR_VERSION, &major);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtGetProperty(MINOR_VERSION, &minor);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtGetProperty(PATCH_LEVEL, &patch);
  TVM_FFI_ICHECK_EQ(status, CUBLAS_STATUS_SUCCESS)
      << "cublasLtGetProperty failed with status " << static_cast<int>(status);
  return static_cast<int64_t>(major) * 10000 + static_cast<int64_t>(minor) * 100 + patch;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cublaslt_bias_get_algorithms,
                              torch_ext::nvfp4_cublaslt_bias_get_algorithms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cublaslt_bias_run_with_algorithm,
                              torch_ext::nvfp4_cublaslt_bias_run_with_algorithm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cublaslt_bias_runtime_version,
                              torch_ext::nvfp4_cublaslt_bias_runtime_version);
