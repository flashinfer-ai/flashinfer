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
 *
 * SM90 serve-shape RMSNorm specialist (bf16, 2D, H in {512, 2048, 6144}).
 *
 * Optional alternate backend for the launch-bound decode-serve envelope
 * (small token counts at hidden sizes 512/2048/6144 on H100/H200). Selected
 * only by the dispatch guard in flashinfer/norm/kernels/rmsnorm_serve_sm90.py
 * when FLASHINFER_SM90_SERVE_RMSNORM=cuda; the stock rmsnorm path is
 * unaffected otherwise. The launch heuristic is from the measured artifact;
 * the epilogue normalizes before weighting to avoid intermediate overflow.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tvm_ffi_utils.h"

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta != 0; delta >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, delta);
  }
  return value;
}

__device__ __forceinline__ float2 unpack_bf16x2(uint32_t v) {
  return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__global__ __launch_bounds__(384) void rmsnorm_6144_osplit8(const uint4* __restrict__ x,
                                                            const uint4* __restrict__ weight,
                                                            uint4* __restrict__ output,
                                                            float inverse_h, float epsilon) {
  constexpr int BLOCK = 384;
  constexpr int CHUNK = 96;
  const int tid = threadIdx.x;
  const int part = blockIdx.x;
  const int slot = part >> 2;
  const int sub = part & 3;
  const size_t base = static_cast<size_t>(blockIdx.y) * 768;
  const uint4 x0 = x[base + tid];
  const uint4 x1 = x[base + tid + BLOCK];
  const int out_idx = tid + slot * BLOCK;
  const bool mine = tid >= sub * CHUNK && tid < (sub + 1) * CHUNK;
  uint4 wv;
  if (mine) wv = weight[out_idx];

  float s = 0.f;
  const uint4 xs[2] = {x0, x1};
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    float2 a = unpack_bf16x2(xs[k].x), b = unpack_bf16x2(xs[k].y);
    float2 c = unpack_bf16x2(xs[k].z), d = unpack_bf16x2(xs[k].w);
    s = fmaf(a.x, a.x, s);
    s = fmaf(a.y, a.y, s);
    s = fmaf(b.x, b.x, s);
    s = fmaf(b.y, b.y, s);
    s = fmaf(c.x, c.x, s);
    s = fmaf(c.y, c.y, s);
    s = fmaf(d.x, d.x, s);
    s = fmaf(d.y, d.y, s);
  }
  s = warp_sum(s);
  __shared__ __align__(16) float sm[12];
  if ((tid & 31) == 0) sm[tid >> 5] = s;
  __syncthreads();
  const float4* sm4 = reinterpret_cast<const float4*>(sm);
  float4 q = sm4[0];
  s = (q.x + q.y) + (q.z + q.w);
#pragma unroll
  for (int i = 1; i < 3; ++i) {
    q = sm4[i];
    s += (q.x + q.y) + (q.z + q.w);
  }
  const float scale = rsqrtf(s * inverse_h + epsilon);
  if (mine) {
    const uint4 xv = slot == 0 ? x0 : x1;
    float2 a = unpack_bf16x2(xv.x), b = unpack_bf16x2(xv.y);
    float2 c = unpack_bf16x2(xv.z), d = unpack_bf16x2(xv.w);
    float2 wa = unpack_bf16x2(wv.x), wb = unpack_bf16x2(wv.y);
    float2 wc = unpack_bf16x2(wv.z), wd = unpack_bf16x2(wv.w);
    uint4 out;
    out.x = pack_bf16x2((a.x * scale) * wa.x, (a.y * scale) * wa.y);
    out.y = pack_bf16x2((b.x * scale) * wb.x, (b.y * scale) * wb.y);
    out.z = pack_bf16x2((c.x * scale) * wc.x, (c.y * scale) * wc.y);
    out.w = pack_bf16x2((d.x * scale) * wd.x, (d.y * scale) * wd.y);
    output[base + out_idx] = out;
  }
}

template <int THREADS, int VECS, int ROW_VECS = THREADS * VECS>
__global__ __launch_bounds__(THREADS) void rmsnorm_exact(const uint4* __restrict__ x,
                                                         const uint4* __restrict__ weight,
                                                         uint4* __restrict__ output,
                                                         float inverse_h, float epsilon) {
  const int lane = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * ROW_VECS;
  uint4 x_registers[VECS];
  uint4 w_registers[VECS];

#pragma unroll
  for (int v = 0; v < VECS; ++v) {
    const int column = lane + v * THREADS;
    if constexpr (ROW_VECS == THREADS * VECS) {
      x_registers[v] = x[row + column];
      w_registers[v] = weight[column];
    } else if (column < ROW_VECS) {
      x_registers[v] = x[row + column];
      w_registers[v] = weight[column];
    }
  }

  float square_sum = 0.0f;
#pragma unroll
  for (int v = 0; v < VECS; ++v) {
    const int column = lane + v * THREADS;
    if constexpr (ROW_VECS != THREADS * VECS) {
      if (column >= ROW_VECS) continue;
    }
    const bf16x2* pairs = reinterpret_cast<const bf16x2*>(&x_registers[v]);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 values = __bfloat1622float2(pairs[pair]);
      square_sum = fmaf(values.x, values.x, square_sum);
      square_sum = fmaf(values.y, values.y, square_sum);
    }
  }

  square_sum = warp_sum(square_sum);
  if constexpr (THREADS > 32) {
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[WARPS];
    if ((lane & 31) == 0) warp_sums[lane >> 5] = square_sum;
    __syncthreads();
    square_sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < WARPS; ++warp) square_sum += warp_sums[warp];
  }

  const float inverse_rms = rsqrtf(square_sum * inverse_h + epsilon);
#pragma unroll
  for (int v = 0; v < VECS; ++v) {
    const int column = lane + v * THREADS;
    if constexpr (ROW_VECS != THREADS * VECS) {
      if (column >= ROW_VECS) continue;
    }
    const bf16x2* x_pairs = reinterpret_cast<const bf16x2*>(&x_registers[v]);
    const bf16x2* w_pairs = reinterpret_cast<const bf16x2*>(&w_registers[v]);
    uint4 result;
    bf16x2* out_pairs = reinterpret_cast<bf16x2*>(&result);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      float2 xv = __bfloat1622float2(x_pairs[pair]);
      const float2 wv = __bfloat1622float2(w_pairs[pair]);
      xv.x = (xv.x * inverse_rms) * wv.x;
      xv.y = (xv.y * inverse_rms) * wv.y;
      out_pairs[pair] = __float22bfloat162_rn(xv);
    }
    output[row + column] = result;
  }
}

__global__ __launch_bounds__(256) void rmsnorm_2048_osplit2(const uint4* __restrict__ x,
                                                            const uint4* __restrict__ weight,
                                                            uint4* __restrict__ output,
                                                            float inverse_h, float epsilon) {
  const int tid = threadIdx.x;
  const int part = blockIdx.x;
  const size_t base = size_t(blockIdx.y) * 256;
  const uint4 xv = x[base + tid];
  const bool mine = tid >= part * 128 && tid < (part + 1) * 128;
  uint4 wv;
  if (mine) wv = weight[tid];
  const float2 a = unpack_bf16x2(xv.x), b = unpack_bf16x2(xv.y);
  const float2 c = unpack_bf16x2(xv.z), d = unpack_bf16x2(xv.w);
  float s = 0.0f;
  s = fmaf(a.x, a.x, s);
  s = fmaf(a.y, a.y, s);
  s = fmaf(b.x, b.x, s);
  s = fmaf(b.y, b.y, s);
  s = fmaf(c.x, c.x, s);
  s = fmaf(c.y, c.y, s);
  s = fmaf(d.x, d.x, s);
  s = fmaf(d.y, d.y, s);
  s = warp_sum(s);
  __shared__ __align__(16) float sm[8];
  if ((tid & 31) == 0) sm[tid >> 5] = s;
  __syncthreads();
  const float4* sm4 = reinterpret_cast<const float4*>(sm);
  float4 q = sm4[0];
  s = (q.x + q.y) + (q.z + q.w);
  q = sm4[1];
  s += (q.x + q.y) + (q.z + q.w);
  const float scale = rsqrtf(s * inverse_h + epsilon);
  if (mine) {
    const float2 wa = unpack_bf16x2(wv.x), wb = unpack_bf16x2(wv.y);
    const float2 wc = unpack_bf16x2(wv.z), wd = unpack_bf16x2(wv.w);
    uint4 out;
    out.x = pack_bf16x2((a.x * scale) * wa.x, (a.y * scale) * wa.y);
    out.y = pack_bf16x2((b.x * scale) * wb.x, (b.y * scale) * wb.y);
    out.z = pack_bf16x2((c.x * scale) * wc.x, (c.y * scale) * wc.y);
    out.w = pack_bf16x2((d.x * scale) * wd.x, (d.y * scale) * wd.y);
    output[base + tid] = out;
  }
}

__global__ __launch_bounds__(256) void rmsnorm_fallback(const bf16* __restrict__ x,
                                                        const bf16* __restrict__ weight,
                                                        bf16* __restrict__ output, int hidden,
                                                        float inverse_h, float epsilon) {
  const int thread = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.x) * hidden;
  float square_sum = 0.0f;
  for (int column = thread; column < hidden; column += 256) {
    const float value = __bfloat162float(x[row + column]);
    square_sum = fmaf(value, value, square_sum);
  }
  square_sum = warp_sum(square_sum);
  __shared__ float warp_sums[8];
  if ((thread & 31) == 0) warp_sums[thread >> 5] = square_sum;
  __syncthreads();
  square_sum = 0.0f;
#pragma unroll
  for (int warp = 0; warp < 8; ++warp) square_sum += warp_sums[warp];
  const float inverse_rms = rsqrtf(square_sum * inverse_h + epsilon);
  for (int column = thread; column < hidden; column += 256) {
    const float result =
        __bfloat162float(x[row + column]) * inverse_rms * __bfloat162float(weight[column]);
    output[row + column] = __float2bfloat16_rn(result);
  }
}

template <int THREADS, int VECS, int ROW_VECS = THREADS * VECS>
void launch_exact(const uint4* x, const uint4* weight, uint4* output, int tokens, float inverse_h,
                  float epsilon, cudaStream_t stream) {
  rmsnorm_exact<THREADS, VECS, ROW_VECS>
      <<<tokens, THREADS, 0, stream>>>(x, weight, output, inverse_h, epsilon);
}

}  // namespace

void rmsnorm_serve_sm90(TensorView output, TensorView input, TensorView weight, double epsilon) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DEVICE(input, output);
  CHECK_DIM(2, input);
  CHECK_DIM(2, output);
  CHECK_DIM(1, weight);
  TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
  TVM_FFI_ICHECK_EQ(output.size(0), input.size(0));
  TVM_FFI_ICHECK_EQ(output.size(1), input.size(1));
  if (input.size(0) != 1) {
    TVM_FFI_ICHECK_EQ(input.stride(0), input.size(1)) << "input must be contiguous";
    TVM_FFI_ICHECK_EQ(output.stride(0), output.size(1)) << "output must be contiguous";
  }
  TVM_FFI_ICHECK(input.dtype() == dl_bfloat16) << "rmsnorm_serve_sm90 is bf16-only";

  const int tokens = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const float inverse_h = 1.0f / static_cast<float>(hidden);
  const float eps = static_cast<float>(epsilon);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  // Re-queried per call so the launch follows the caller's current stream,
  // including a CUDA-graph capture stream.
  const cudaStream_t stream = get_stream(input.device());

  // H-dispatch launch heuristic, verbatim from the measured artifact.
  if (hidden == 512) {
    launch_exact<32, 2>(static_cast<const uint4*>(input.data_ptr()),
                        static_cast<const uint4*>(weight.data_ptr()),
                        static_cast<uint4*>(output.data_ptr()), tokens, inverse_h, eps, stream);
  } else if (hidden == 2048) {
    if (tokens == 2) {
      rmsnorm_2048_osplit2<<<dim3(2, tokens), 256, 0, stream>>>(
          static_cast<const uint4*>(input.data_ptr()), static_cast<const uint4*>(weight.data_ptr()),
          static_cast<uint4*>(output.data_ptr()), inverse_h, eps);
    } else {
      launch_exact<256, 1>(static_cast<const uint4*>(input.data_ptr()),
                           static_cast<const uint4*>(weight.data_ptr()),
                           static_cast<uint4*>(output.data_ptr()), tokens, inverse_h, eps, stream);
    }
  } else if (hidden == 6144) {
    if (tokens <= 64) {
      rmsnorm_6144_osplit8<<<dim3(8, tokens), 384, 0, stream>>>(
          static_cast<const uint4*>(input.data_ptr()), static_cast<const uint4*>(weight.data_ptr()),
          static_cast<uint4*>(output.data_ptr()), inverse_h, eps);
    } else {
      launch_exact<384, 2>(static_cast<const uint4*>(input.data_ptr()),
                           static_cast<const uint4*>(weight.data_ptr()),
                           static_cast<uint4*>(output.data_ptr()), tokens, inverse_h, eps, stream);
    }
  } else {
    // Correct generic fallback; the Python-side guard does not route here
    // (off the exact-H allowlist this specialist is slower than the stock
    // path), it exists so the kernel remains total over its inputs.
    rmsnorm_fallback<<<tokens, 256, 0, stream>>>(
        static_cast<const bf16*>(input.data_ptr()), static_cast<const bf16*>(weight.data_ptr()),
        static_cast<bf16*>(output.data_ptr()), hidden, inverse_h, eps);
  }
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "rmsnorm_serve_sm90 launch failed: " << cudaGetErrorString(status);
}
