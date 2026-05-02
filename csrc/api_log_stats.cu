/*
 * Copyright (c) 2025 by FlashInfer team.
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
//
// CUDA-graph-friendly tensor statistics printer used by @flashinfer_api at
// FLASHINFER_LOGLEVEL=5. Reductions and printing both happen on the device
// so the launch can be captured into a graph; on graph replay the kernel
// runs and emits a single line via printf.
//
// The host correlates the printed `id=N` back to the API call/argument
// before launching, so users can match the GPU-side line to the eager log.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "tvm_ffi_utils.h"

namespace {

// Convert any supported tensor dtype to double for reduction. Using double
// (not float) preserves precision for int32_t/int64_t values past the 24-bit
// float mantissa (~16.7M); the kernel emits the result as %.6f anyway.
template <typename T>
__device__ inline double to_double_impl(T x) {
  return static_cast<double>(x);
}

__device__ inline double to_double_impl(nv_half x) { return static_cast<double>(__half2float(x)); }
__device__ inline double to_double_impl(nv_bfloat16 x) {
  return static_cast<double>(__bfloat162float(x));
}

template <typename T>
struct IsFloatLike {
  static constexpr bool value = std::is_floating_point<T>::value ||
                                std::is_same<T, nv_half>::value ||
                                std::is_same<T, nv_bfloat16>::value;
};

constexpr int kBlockSize = 256;

// Single-block reduction: each thread strides over the input, then we do a
// shared-memory tree reduction. All operations are computable inside a
// captured graph; the final printf is a device-side intrinsic that emits
// during kernel execution (i.e. during graph replay).
template <typename T>
__global__ void PrintTensorStatsKernel(const T* __restrict__ data, int64_t numel,
                                       int64_t tensor_id) {
  const int tid = threadIdx.x;

  double thread_min = CUDART_INF;
  double thread_max = -CUDART_INF;
  double thread_sum = 0.0;
  int64_t thread_nan = 0;
  int64_t thread_inf = 0;

  for (int64_t i = tid; i < numel; i += kBlockSize) {
    double v = to_double_impl(data[i]);
    bool is_nan = false;
    bool is_inf = false;
    if constexpr (IsFloatLike<T>::value) {
      is_nan = isnan(v);
      is_inf = isinf(v);
    }
    if (is_nan) {
      thread_nan += 1;
      continue;
    }
    if (is_inf) {
      thread_inf += 1;
      // Match eager (torch.min/torch.max include +/-Inf in the result):
      // fall through and update min/max but skip the sum so mean stays
      // finite. Counted separately in `inf=N` below.
    } else {
      thread_sum += v;
    }
    thread_min = fmin(thread_min, v);
    thread_max = fmax(thread_max, v);
  }

  __shared__ double s_min[kBlockSize];
  __shared__ double s_max[kBlockSize];
  __shared__ double s_sum[kBlockSize];
  __shared__ int64_t s_nan[kBlockSize];
  __shared__ int64_t s_inf[kBlockSize];

  s_min[tid] = thread_min;
  s_max[tid] = thread_max;
  s_sum[tid] = thread_sum;
  s_nan[tid] = thread_nan;
  s_inf[tid] = thread_inf;
  __syncthreads();

#pragma unroll
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_min[tid] = fmin(s_min[tid], s_min[tid + stride]);
      s_max[tid] = fmax(s_max[tid], s_max[tid + stride]);
      s_sum[tid] += s_sum[tid + stride];
      s_nan[tid] += s_nan[tid + stride];
      s_inf[tid] += s_inf[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int64_t valid = numel - s_nan[0] - s_inf[0];
    double mean = valid > 0 ? s_sum[0] / static_cast<double>(valid) : 0.0;
    if (numel == 0) {
      printf("[flashinfer stats] id=%lld numel=0 (empty tensor)\n",
             static_cast<long long>(tensor_id));
    } else if (valid == 0) {
      // All non-finite. Don't print sentinel `min=inf max=-inf mean=0` —
      // it looks broken; report the non-finite counts instead.
      printf(
          "[flashinfer stats] id=%lld numel=%lld (all non-finite) "
          "nan=%lld inf=%lld\n",
          static_cast<long long>(tensor_id), static_cast<long long>(numel),
          static_cast<long long>(s_nan[0]), static_cast<long long>(s_inf[0]));
    } else {
      printf(
          "[flashinfer stats] id=%lld numel=%lld min=%.6f max=%.6f mean=%.6f "
          "nan=%lld inf=%lld\n",
          static_cast<long long>(tensor_id), static_cast<long long>(numel), s_min[0], s_max[0],
          mean, static_cast<long long>(s_nan[0]), static_cast<long long>(s_inf[0]));
    }
  }
}

template <typename T>
void launch_print_tensor_stats(const void* data_ptr, int64_t numel, int64_t tensor_id,
                               cudaStream_t stream) {
  PrintTensorStatsKernel<T>
      <<<1, kBlockSize, 0, stream>>>(static_cast<const T*>(data_ptr), numel, tensor_id);
}

}  // namespace

// Launch a single-block kernel that computes basic statistics (min/max/mean/
// nan/inf) over `tensor` and prints them via device-side printf. Safe to call
// inside torch.cuda.graph(...) capture: the launch is captured and runs on
// graph replay. Only the dtypes in the switch below are supported; for
// unsupported dtypes this is a no-op (the host caller decides whether to
// fall back to its plain "[statistics skipped]" message).
void api_log_print_tensor_stats(TensorView tensor, int64_t tensor_id) {
  int64_t numel = 1;
  for (int i = 0; i < tensor.ndim(); ++i) {
    numel *= tensor.size(i);
  }

  ffi::CUDADeviceGuard device_guard(tensor.device().device_id);
  const cudaStream_t stream = get_stream(tensor.device());

  const void* data = tensor.data_ptr();
  switch (encode_dlpack_dtype(tensor.dtype())) {
    case float32_code:
      launch_print_tensor_stats<float>(data, numel, tensor_id, stream);
      break;
    case float16_code:
      launch_print_tensor_stats<nv_half>(data, numel, tensor_id, stream);
      break;
    case bfloat16_code:
      launch_print_tensor_stats<nv_bfloat16>(data, numel, tensor_id, stream);
      break;
    case int32_code:
      launch_print_tensor_stats<int32_t>(data, numel, tensor_id, stream);
      break;
    case int64_code:
      launch_print_tensor_stats<int64_t>(data, numel, tensor_id, stream);
      break;
    case uint8_code:
      launch_print_tensor_stats<uint8_t>(data, numel, tensor_id, stream);
      break;
    default:
      // Silent no-op for unsupported dtypes (e.g. fp8, fp4). The host-side
      // logger already records dtype in the eager pre-execution metadata.
      break;
  }
}
