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

// Fused epilogue of the split-sequence affine KDA prefill composite.
//
// One launch replaces the torch sequence that followed the main / map / scan /
// correction part launches:
//   checkpoint rows : out_rows[cu_starts[seq_ids[r]] + offsets[r]] =
//                       main_rows[r] (+ corr_rows[r - first_rows] for r >= first_rows)
//   output tail     : out_tail += corr_out                      (bf16, fp32 accumulate)
//   final state     : final_compact[s] = main_final[last_parts[s]]
//                       (+ corr_final[last_corr_parts[s]] unless s == 0 && zero_first)
//                     final_pool[state_indices[s]] = final_compact[s]  (fp32 or bf16 pool)
// Every value is produced by exactly the operation torch used (one fp32 add,
// round-to-nearest-even to bf16), so the result is bitwise identical to the
// unfused path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace cake_kda_affine_epilogue {

using tvm::ffi::TensorView;

constexpr int kThreads = 256;

struct Params {
  const void* main_rows;
  const void* corr_rows;
  void* out_rows;
  const int64_t* offsets;
  const int64_t* seq_ids;
  const int64_t* cu_starts;
  int64_t first_rows;
  int64_t row_elems;  // heads * 128 * 128
  int64_t rows_vec;   // vectors (of 4 fp32 or 8 bf16) over all checkpoint rows
  int rows_bf16;
  __nv_bfloat16* out_tail;
  const __nv_bfloat16* corr_out;
  int64_t tail_vec;  // vectors of 8 bf16
  const float* main_final;
  const float* corr_final;
  const int64_t* last_parts;
  const int64_t* last_corr_parts;
  int zero_first_correction;
  float* final_compact;
  void* final_pool;
  int64_t pool_slot_stride;  // elements between pool slots
  int pool_bf16;
  const int64_t* state_indices;
  int64_t final_vec;  // vectors of 4 fp32 over num_sequences * row_elems
  int64_t blocks_rows;
  int64_t blocks_tail;
};

__device__ __forceinline__ __nv_bfloat16 add_bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
}

__global__ void __launch_bounds__(kThreads) affine_epilogue_kernel(Params p) {
  const int64_t block = blockIdx.x;
  if (block < p.blocks_rows) {
    const int64_t vec = block * kThreads + threadIdx.x;
    if (vec >= p.rows_vec) return;
    if (p.rows_bf16) {
      const int64_t e = vec * 8;
      const int64_t r = e / p.row_elems;
      const int64_t within = e - r * p.row_elems;
      const int64_t dst = p.cu_starts[p.seq_ids[r]] + p.offsets[r];
      const uint4 mv =
          *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.main_rows) + e);
      uint4 ov = mv;
      if (r >= p.first_rows) {
        const uint4 cv =
            *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.corr_rows) +
                                            (r - p.first_rows) * p.row_elems + within);
        const __nv_bfloat16* m = reinterpret_cast<const __nv_bfloat16*>(&mv);
        const __nv_bfloat16* c = reinterpret_cast<const __nv_bfloat16*>(&cv);
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = add_bf16(m[i], c[i]);
      }
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.out_rows) + dst * p.row_elems +
                                within) = ov;
    } else {
      const int64_t e = vec * 4;
      const int64_t r = e / p.row_elems;
      const int64_t within = e - r * p.row_elems;
      const int64_t dst = p.cu_starts[p.seq_ids[r]] + p.offsets[r];
      float4 mv = *reinterpret_cast<const float4*>(static_cast<const float*>(p.main_rows) + e);
      if (r >= p.first_rows) {
        const float4 cv = *reinterpret_cast<const float4*>(
            static_cast<const float*>(p.corr_rows) + (r - p.first_rows) * p.row_elems + within);
        mv.x += cv.x;
        mv.y += cv.y;
        mv.z += cv.z;
        mv.w += cv.w;
      }
      *reinterpret_cast<float4*>(static_cast<float*>(p.out_rows) + dst * p.row_elems + within) = mv;
    }
    return;
  }
  if (block < p.blocks_rows + p.blocks_tail) {
    const int64_t vec = (block - p.blocks_rows) * kThreads + threadIdx.x;
    if (vec >= p.tail_vec) return;
    const int64_t e = vec * 8;
    uint4 ov = *reinterpret_cast<const uint4*>(p.out_tail + e);
    const uint4 cv = *reinterpret_cast<const uint4*>(p.corr_out + e);
    __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&ov);
    const __nv_bfloat16* c = reinterpret_cast<const __nv_bfloat16*>(&cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = add_bf16(o[i], c[i]);
    *reinterpret_cast<uint4*>(p.out_tail + e) = ov;
    return;
  }
  const int64_t vec = (block - p.blocks_rows - p.blocks_tail) * kThreads + threadIdx.x;
  if (vec >= p.final_vec) return;
  const int64_t e = vec * 4;
  const int64_t s = e / p.row_elems;
  const int64_t within = e - s * p.row_elems;
  float4 v =
      *reinterpret_cast<const float4*>(p.main_final + p.last_parts[s] * p.row_elems + within);
  // torch adds an explicitly zeroed correction for a first sequence with a
  // single part; adding 0.0f reproduces it (including the -0.0 -> +0.0 case).
  float4 c = make_float4(0.f, 0.f, 0.f, 0.f);
  if (!(p.zero_first_correction && s == 0)) {
    c = *reinterpret_cast<const float4*>(p.corr_final + p.last_corr_parts[s] * p.row_elems +
                                         within);
  }
  v.x += c.x;
  v.y += c.y;
  v.z += c.z;
  v.w += c.w;
  *reinterpret_cast<float4*>(p.final_compact + e) = v;
  const int64_t slot = p.state_indices[s];
  if (p.pool_bf16) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(
        static_cast<__nv_bfloat16*>(p.final_pool) + slot * p.pool_slot_stride + within);
    dst[0] = lo;
    dst[1] = hi;
  } else {
    *reinterpret_cast<float4*>(static_cast<float*>(p.final_pool) + slot * p.pool_slot_stride +
                               within) = v;
  }
}

inline bool is_bf16(const TensorView& t) {
  return t.dtype().code == kDLBfloat && t.dtype().bits == 16;
}
inline bool is_f32(const TensorView& t) {
  return t.dtype().code == kDLFloat && t.dtype().bits == 32;
}
inline bool is_i64(const TensorView& t) { return t.dtype().code == kDLInt && t.dtype().bits == 64; }

void Run(TensorView main_rows, TensorView corr_rows, TensorView out_rows, TensorView offsets,
         TensorView seq_ids, TensorView cu_starts, int64_t first_rows, int64_t num_rows,
         TensorView out_tail, TensorView corr_out, TensorView main_final, TensorView corr_final,
         TensorView last_parts, TensorView last_corr_parts, int64_t zero_first_correction,
         TensorView final_compact, TensorView final_pool, int64_t pool_slot_stride,
         TensorView state_indices, int64_t num_sequences, int64_t heads, int64_t tail_elems,
         int64_t has_rows) {
  TVM_FFI_CHECK(is_f32(main_final) && is_f32(corr_final) && is_f32(final_compact), ValueError)
      << "affine epilogue: final states must be fp32";
  TVM_FFI_CHECK(is_bf16(out_tail) && is_bf16(corr_out), ValueError)
      << "affine epilogue: output tail must be bf16";
  TVM_FFI_CHECK(is_i64(last_parts) && is_i64(last_corr_parts) && is_i64(state_indices), ValueError)
      << "affine epilogue: index tensors must be int64";
  TVM_FFI_CHECK(is_f32(final_pool) || is_bf16(final_pool), ValueError)
      << "affine epilogue: state pool must be fp32 or bf16";
  TVM_FFI_CHECK(tail_elems % 8 == 0 && heads > 0 && num_sequences > 0, ValueError)
      << "affine epilogue: tail must be a multiple of 8 elements";
  Params p{};
  p.row_elems = heads * 128 * 128;
  p.rows_vec = 0;
  p.blocks_rows = 0;
  if (has_rows) {
    TVM_FFI_CHECK((is_f32(main_rows) && is_f32(corr_rows) && is_f32(out_rows)) ||
                      (is_bf16(main_rows) && is_bf16(corr_rows) && is_bf16(out_rows)),
                  ValueError)
        << "affine epilogue: checkpoint rows must share one dtype (fp32 or bf16)";
    TVM_FFI_CHECK(is_i64(offsets) && is_i64(seq_ids) && is_i64(cu_starts), ValueError)
        << "affine epilogue: checkpoint index tensors must be int64";
    p.main_rows = main_rows.data_ptr();
    p.corr_rows = corr_rows.data_ptr();
    p.out_rows = out_rows.data_ptr();
    p.offsets = static_cast<const int64_t*>(offsets.data_ptr());
    p.seq_ids = static_cast<const int64_t*>(seq_ids.data_ptr());
    p.cu_starts = static_cast<const int64_t*>(cu_starts.data_ptr());
    p.first_rows = first_rows;
    p.rows_bf16 = is_bf16(main_rows) ? 1 : 0;
    const int64_t per_vec = p.rows_bf16 ? 8 : 4;
    p.rows_vec = num_rows * p.row_elems / per_vec;
    p.blocks_rows = (p.rows_vec + kThreads - 1) / kThreads;
  }
  p.out_tail = static_cast<__nv_bfloat16*>(out_tail.data_ptr());
  p.corr_out = static_cast<const __nv_bfloat16*>(corr_out.data_ptr());
  p.tail_vec = tail_elems / 8;
  p.blocks_tail = (p.tail_vec + kThreads - 1) / kThreads;
  p.main_final = static_cast<const float*>(main_final.data_ptr());
  p.corr_final = static_cast<const float*>(corr_final.data_ptr());
  p.last_parts = static_cast<const int64_t*>(last_parts.data_ptr());
  p.last_corr_parts = static_cast<const int64_t*>(last_corr_parts.data_ptr());
  p.zero_first_correction = zero_first_correction ? 1 : 0;
  p.final_compact = static_cast<float*>(final_compact.data_ptr());
  p.final_pool = final_pool.data_ptr();
  p.pool_slot_stride = pool_slot_stride;
  p.pool_bf16 = is_bf16(final_pool) ? 1 : 0;
  p.state_indices = static_cast<const int64_t*>(state_indices.data_ptr());
  p.final_vec = num_sequences * p.row_elems / 4;
  const int64_t blocks_final = (p.final_vec + kThreads - 1) / kThreads;
  const int64_t blocks = p.blocks_rows + p.blocks_tail + blocks_final;
  TVM_FFI_CHECK(blocks > 0 && blocks < (int64_t{1} << 31), ValueError)
      << "affine epilogue: grid out of range";
  cudaStream_t stream = get_stream(main_final.device());
  affine_epilogue_kernel<<<dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0, stream>>>(p);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "affine epilogue launch failed: " << cudaGetErrorString(status);
}

}  // namespace cake_kda_affine_epilogue

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_kda_affine_epilogue::Run);
