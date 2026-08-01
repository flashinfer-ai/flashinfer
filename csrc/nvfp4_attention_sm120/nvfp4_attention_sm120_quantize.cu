/*
 * Copyright (c) 2025 by SageAttention team.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <flashinfer/math.cuh>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "tvm_ffi_utils.h"

#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16_LOCAL(dlpack_dtype, c_type, ...)         \
  if ((dlpack_dtype) == dl_float16) {                                                \
    using c_type = half;                                                             \
    __VA_ARGS__                                                                      \
  } else if ((dlpack_dtype) == dl_bfloat16) {                                        \
    using c_type = nv_bfloat16;                                                      \
    __VA_ARGS__                                                                      \
  } else {                                                                           \
    TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type"; \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                      \
  if (head_dim == 64) {                                                 \
    constexpr int HEAD_DIM = 64;                                        \
    __VA_ARGS__                                                         \
  } else if (head_dim == 128) {                                         \
    constexpr int HEAD_DIM = 128;                                       \
    __VA_ARGS__                                                         \
  } else {                                                              \
    TVM_FFI_ICHECK(false) << "Unsupported head dim: " << int(head_dim); \
  }

#define CHECK_QUANT_CUDA(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA)
#define CHECK_QUANT_DTYPE(x, true_dtype) \
  TVM_FFI_ICHECK((x).dtype() == (true_dtype)) << #x " dtype mismatch"
#define CHECK_QUANT_DIMS(x, true_dim) TVM_FFI_ICHECK_EQ((x).ndim(), true_dim) << #x " rank mismatch"
#define CHECK_QUANT_LASTDIM_CONTIGUOUS(x) \
  TVM_FFI_ICHECK_EQ((x).stride(-1), 1) << #x " must be contiguous at the last dimension"

namespace flashinfer {
namespace nvfp4_attention_sm120 {

void check_shape(TensorView x, std::initializer_list<int64_t> shape, const char* name) {
  TVM_FFI_ICHECK_EQ(x.ndim(), static_cast<int>(shape.size())) << name << " rank mismatch";
  int i = 0;
  for (int64_t expected : shape) {
    TVM_FFI_ICHECK_EQ(x.size(i), expected) << name << " shape mismatch at dim " << i;
    ++i;
  }
}

#define CHECK_QUANT_SHAPE(x, ...) check_shape((x), {__VA_ARGS__}, #x)

constexpr int CVT_FP4_ELTS_PER_THREAD = 16;

template <typename T>
struct TypeConverter {
  using Type = half2;
};

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[8];
};

template <uint32_t head_dim, uint32_t BLOCK_SIZE, bool permute, typename T>
__global__ void scaled_fp4_quant_kernel(const T* input, uint8_t* output, uint8_t* output_sf,
                                        int batch_size, int num_heads, int num_tokens,
                                        int stride_bz_input, int stride_h_input,
                                        int stride_seq_input, int stride_bz_output,
                                        int stride_h_output, int stride_seq_output,
                                        int stride_bz_output_sf, int stride_h_output_sf,
                                        int stride_seq_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;

  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  int load_token_id;
  if constexpr (!permute) {
    load_token_id = token_id;
  } else {
    int local_token_id = threadIdx.x / NUM_THREADS_PER_TOKEN;
    int local_token_id_residue = local_token_id % 32;

    load_token_id = token_block_id * BLOCK_SIZE + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 + ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PackedVec in_vec;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }

  if (load_token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        load_token_id * stride_seq_input +
        (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0];
  }

  auto localMax = __habs2(in_vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = flashinfer::math::fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + batch_id * stride_bz_output + head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD /
                                    2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + batch_id * stride_bz_output + head_id * stride_h_output +
                                token_id * stride_seq_output +
                                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD /
                                    2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf +
                                 head_id * stride_h_output_sf +
                                 (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local = (col_id_local / 4) * 256 + (token_id_local % 16) * 16 +
                            (token_id_local / 16) * 4 + (col_id_local % 4);
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (token_id_local % 16) * 16 +
                              (token_id_local / 16) * 4 + (col_id_local % 4);
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, typename T>
__global__ void scaled_fp4_quant_trans_kernel(const T* input, uint8_t* output, uint8_t* output_sf,
                                              int batch_size, int num_heads, int num_tokens,
                                              int stride_bz_input, int stride_h_input,
                                              int stride_seq_input, int stride_bz_output,
                                              int stride_h_output, int stride_d_output,
                                              int stride_bz_output_sf, int stride_h_output_sf,
                                              int stride_d_output_sf) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                "Only half and bfloat16 input are supported");
  using PackedVec = PackedVec<T>;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;

  static_assert(CVT_FP4_ELTS_PER_THREAD == 8 || CVT_FP4_ELTS_PER_THREAD == 16,
                "CVT_FP4_ELTS_PER_THREAD must be 8 or 16");
  static_assert(sizeof(PackedVec) == sizeof(T) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  constexpr uint32_t NUM_THREADS_PER_TOKEN = head_dim / CVT_FP4_ELTS_PER_THREAD;
  constexpr uint32_t NUM_THREADS_PER_SEQ = BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD;

  const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

  PackedVec in_vec;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }

  if (token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        token_id * stride_seq_input +
        (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD)[0];
  }

  struct alignas(16) SharedInputStorage {
    T data[BLOCK_SIZE * head_dim];
  };
  __shared__ SharedInputStorage shared_input_storage;
  T* shared_input = shared_input_storage.data;
  reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    in_vec.elts[i].x =
        shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) +
                     ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i) *
                         head_dim];
    in_vec.elts[i].y =
        shared_input[(threadIdx.x / NUM_THREADS_PER_SEQ) +
                     ((threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD + 2 * i + 1) *
                         head_dim];
  }

  auto localMax = __habs2(in_vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(in_vec.elts[i]));
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1, 32), localMax);
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = vecMax / 6.0f;
  uint8_t SFValueFP8;
  reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
  SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));

  float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
    fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
    fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
  }

  uint32_t e2m1Vals[CVT_FP4_ELTS_PER_THREAD / 8];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 8; i++) {
    e2m1Vals[i] = flashinfer::math::fp32_vec_to_e2m1(fp2Vals + i * 4);
  }

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
    reinterpret_cast<uint32_t*>(output + batch_id * stride_bz_output + head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE +
                                 (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) /
                                    2)[0] = e2m1Vals[0];
  } else {
    reinterpret_cast<uint64_t*>(output + batch_id * stride_bz_output + head_id * stride_h_output +
                                (threadIdx.x / NUM_THREADS_PER_SEQ) * stride_d_output +
                                (token_block_id * BLOCK_SIZE +
                                 (threadIdx.x % NUM_THREADS_PER_SEQ) * CVT_FP4_ELTS_PER_THREAD) /
                                    2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
  }

  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf +
                                 head_id * stride_h_output_sf +
                                 (threadIdx.x / NUM_THREADS_PER_SEQ / 64) * 64 * stride_d_output_sf;
  uint32_t row_id_local = (threadIdx.x / NUM_THREADS_PER_SEQ) % 64;

  if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
    uint32_t col_id_local =
        token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD + threadIdx.x % NUM_THREADS_PER_SEQ;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                            (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
  } else {
    if (threadIdx.x % 2 == 0) {
      uint32_t col_id_local = token_block_id * BLOCK_SIZE / CVT_FP4_ELTS_PER_THREAD +
                              (threadIdx.x % NUM_THREADS_PER_SEQ) / 2;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                              (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] = SFValueFP8;
    }
  }
}

void scaled_fp4_quant(TensorView input, TensorView output, TensorView output_sf,
                      int64_t tensor_layout) {
  constexpr int BLOCK_SIZE = 128;

  CHECK_QUANT_CUDA(input);
  CHECK_QUANT_CUDA(output);
  CHECK_QUANT_CUDA(output_sf);

  CHECK_QUANT_LASTDIM_CONTIGUOUS(input);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_QUANT_DTYPE(output, dl_uint8);
  CHECK_QUANT_DTYPE(output_sf, dl_float8_e4m3fn);

  CHECK_QUANT_DIMS(input, 4);
  CHECK_QUANT_DIMS(output, 4);
  CHECK_QUANT_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_QUANT_SHAPE(output, batch_size, num_tokens, num_heads, head_dim / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size, num_tokens, num_heads, head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_QUANT_SHAPE(output, batch_size, num_heads, num_tokens, head_dim / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size, num_heads, num_tokens, head_dim / 16);
  }

  auto input_dtype = input.dtype();
  cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16_LOCAL(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, c_type><<<grid, block, 0, stream>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          reinterpret_cast<uint8_t*>(output_sf.data_ptr()), batch_size, num_heads, num_tokens,
          stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output, stride_h_output,
          stride_seq_output, stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_permute(TensorView input, TensorView output, TensorView output_sf,
                              int64_t tensor_layout) {
  constexpr int BLOCK_SIZE = 128;

  CHECK_QUANT_CUDA(input);
  CHECK_QUANT_CUDA(output);
  CHECK_QUANT_CUDA(output_sf);

  CHECK_QUANT_LASTDIM_CONTIGUOUS(input);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_QUANT_DTYPE(output, dl_uint8);
  CHECK_QUANT_DTYPE(output_sf, dl_float8_e4m3fn);

  CHECK_QUANT_DIMS(input, 4);
  CHECK_QUANT_DIMS(output, 4);
  CHECK_QUANT_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_seq_output, stride_seq_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_seq_output = output.stride(1);
    stride_seq_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_QUANT_SHAPE(output, batch_size, ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                      num_heads, head_dim / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, num_heads,
                      head_dim / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_seq_output = output.stride(2);
    stride_seq_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_QUANT_SHAPE(output, batch_size, num_heads,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size, num_heads,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, head_dim / 16);
  }

  auto input_dtype = input.dtype();
  cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16_LOCAL(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr int BLOCK_SIZE = 128;
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, true, c_type><<<grid, block, 0, stream>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          reinterpret_cast<uint8_t*>(output_sf.data_ptr()), batch_size, num_heads, num_tokens,
          stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output, stride_h_output,
          stride_seq_output, stride_bz_output_sf, stride_h_output_sf, stride_seq_output_sf);
    });
  });
}

void scaled_fp4_quant_trans(TensorView input, TensorView output, TensorView output_sf,
                            int64_t tensor_layout) {
  constexpr int BLOCK_SIZE = 128;

  CHECK_QUANT_CUDA(input);
  CHECK_QUANT_CUDA(output);
  CHECK_QUANT_CUDA(output_sf);

  CHECK_QUANT_LASTDIM_CONTIGUOUS(input);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output);
  CHECK_QUANT_LASTDIM_CONTIGUOUS(output_sf);

  CHECK_QUANT_DTYPE(output, dl_uint8);
  CHECK_QUANT_DTYPE(output_sf, dl_float8_e4m3fn);

  CHECK_QUANT_DIMS(input, 4);
  CHECK_QUANT_DIMS(output, 4);
  CHECK_QUANT_DIMS(output_sf, 4);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  const int stride_bz_input = input.stride(0);
  const int stride_bz_output = output.stride(0);
  const int stride_bz_output_sf = output_sf.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input;
  int stride_d_output, stride_d_output_sf;
  int stride_h_input, stride_h_output, stride_h_output_sf;
  if (tensor_layout == 0) {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_d_output = output.stride(1);
    stride_d_output_sf = output_sf.stride(1);
    stride_h_input = input.stride(2);
    stride_h_output = output.stride(2);
    stride_h_output_sf = output_sf.stride(2);

    CHECK_QUANT_SHAPE(output, batch_size, head_dim, num_heads,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size, head_dim, num_heads,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  } else {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_d_output = output.stride(2);
    stride_d_output_sf = output_sf.stride(2);
    stride_h_input = input.stride(1);
    stride_h_output = output.stride(1);
    stride_h_output_sf = output_sf.stride(1);

    CHECK_QUANT_SHAPE(output, batch_size, num_heads, head_dim,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2);
    CHECK_QUANT_SHAPE(output_sf, batch_size, num_heads, head_dim,
                      ((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 16);
  }

  auto input_dtype = input.dtype();
  cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16_LOCAL(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
      dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);

      scaled_fp4_quant_trans_kernel<HEAD_DIM, BLOCK_SIZE, c_type><<<grid, block, 0, stream>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          reinterpret_cast<uint8_t*>(output.data_ptr()),
          reinterpret_cast<uint8_t*>(output_sf.data_ptr()), batch_size, num_heads, num_tokens,
          stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output, stride_h_output,
          stride_d_output, stride_bz_output_sf, stride_h_output_sf, stride_d_output_sf);
    });
  });
}

}  // namespace nvfp4_attention_sm120
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scaled_fp4_quant,
                              flashinfer::nvfp4_attention_sm120::scaled_fp4_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scaled_fp4_quant_permute,
                              flashinfer::nvfp4_attention_sm120::scaled_fp4_quant_permute);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scaled_fp4_quant_trans,
                              flashinfer::nvfp4_attention_sm120::scaled_fp4_quant_trans);
