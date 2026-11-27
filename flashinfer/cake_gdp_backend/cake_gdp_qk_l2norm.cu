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

// The q / k L2 normalization of the Cake GDP prefill (``use_qk_l2norm_in_kernel=True``): one launch
// reads every 128-element row of q ``[T, H, 128]`` and k ``[T n, H, 128]`` once and writes
// ``x * rsqrt(sum(x^2) + 1e-6)``, the sum and the product in FP32, rounded once to nearest even
// into dense BF16 rows, the operands the prefill kernels read.  The inputs are BF16 or FP16 with a
// dense innermost dimension and 16-byte aligned outer strides; 16 threads own a row (8 threads with
// two 16-byte accesses each when the rows stream through L2), every thread holds four rows in
// flight.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include <cstdint>

namespace cake_gdp_qk_l2norm {

using tvm::ffi::TensorView;

constexpr int kHead = 128;     // elements per row
constexpr int kVector = 8;     // elements per 16-byte access
constexpr int kThreads = 256;  // threads per CTA
constexpr int kUnroll = 4;     // rows in flight per thread
constexpr float kEps = 1e-6f;
// Above this many input bytes (q + k) the rows stream through L2 (they cannot stay there for the
// prefill kernel): 8 threads own a row with two 16-byte accesses each, the loads prefetch 256-byte
// L2 lines and the stores are evict-first.  Below it, 16 threads own a row (more CTAs for a short
// batch) and the stores keep their L2 lines.
constexpr int64_t kStreamBytes = int64_t{128} << 20;

// One operand: ``rows`` = tokens x heads rows of 128 elements, row ``r`` of token ``r / heads``
// and head ``r % heads`` at element offset ``t * stride_t + h * stride_h`` of ``src``; ``dst``
// dense ``[rows, 128]``.
struct Operand {
  const void* src;
  __nv_bfloat16* dst;
  int64_t stride_t;
  int64_t stride_h;
  uint32_t rows;
  uint32_t heads;
  uint32_t tiles;
};

template <typename T>
__device__ __forceinline__ float2 widen(uint32_t word);

template <>
__device__ __forceinline__ float2 widen<__nv_bfloat16>(uint32_t word) {
  __nv_bfloat162 pair;
  memcpy(&pair, &word, sizeof(word));
  return __bfloat1622float2(pair);
}

template <>
__device__ __forceinline__ float2 widen<__half>(uint32_t word) {
  __half2 pair;
  memcpy(&pair, &word, sizeof(word));
  return __half22float2(pair);
}

__device__ __forceinline__ uint32_t narrow(float lo, float hi) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  uint32_t word;
  memcpy(&word, &pair, sizeof(word));
  return word;
}

template <bool kStream>
__device__ __forceinline__ uint4 load(const uint4* p) {
  if constexpr (kStream) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
  } else {
    return __ldcs(p);
  }
}

template <bool kStream>
__device__ __forceinline__ void store(uint4* p, uint4 v) {
  if constexpr (kStream) {
    __stcs(p, v);
  } else {
    *p = v;
  }
}

// kLanes threads per row, kPer 16-byte accesses per thread and row, kUnroll rows in flight per
// thread.
template <bool kStream>
struct Shape {
  static constexpr int kLanes = kStream ? 8 : 16;
  static constexpr int kPer = kHead / kVector / kLanes;
  static constexpr int kRowsPerPass = kThreads / kLanes;
  static constexpr int kRowsPerTile = kRowsPerPass * kUnroll;
};

template <typename T, bool kDense, bool kStream>
__global__ void __launch_bounds__(kThreads) qk_l2norm_kernel(Operand q, Operand k) {
  using S = Shape<kStream>;
  const bool is_q = blockIdx.x < q.tiles;  // uniform per CTA: the q tiles first, then the k tiles
  const Operand op = is_q ? q : k;
  const uint32_t tile = is_q ? blockIdx.x : blockIdx.x - q.tiles;
  const int lane = threadIdx.x % S::kLanes;
  const uint32_t first = tile * S::kRowsPerTile + threadIdx.x / S::kLanes;

  uint4 raw[kUnroll][S::kPer];
#pragma unroll
  for (int u = 0; u < kUnroll; ++u) {
    const uint32_t row = first + u * S::kRowsPerPass;
    int64_t offset = 0;
    if constexpr (kDense) {
      offset = static_cast<int64_t>(row) * kHead;
    } else {
      const uint32_t t = row / op.heads;
      const uint32_t h = row - t * op.heads;
      offset = static_cast<int64_t>(t) * op.stride_t + static_cast<int64_t>(h) * op.stride_h;
    }
    const uint4* src =
        reinterpret_cast<const uint4*>(static_cast<const T*>(op.src) + offset) + lane * S::kPer;
#pragma unroll
    for (int v = 0; v < S::kPer; ++v) {
      raw[u][v] = row < op.rows ? load<kStream>(src + v) : make_uint4(0u, 0u, 0u, 0u);
    }
  }
#pragma unroll
  for (int u = 0; u < kUnroll; ++u) {
    const uint32_t row = first + u * S::kRowsPerPass;
    float2 x[S::kPer][4];
    float part[S::kPer];
#pragma unroll
    for (int v = 0; v < S::kPer; ++v) {
      const uint32_t words[4] = {raw[u][v].x, raw[u][v].y, raw[u][v].z, raw[u][v].w};
      part[v] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[v][i] = widen<T>(words[i]);
        part[v] = fmaf(x[v][i].x, x[v][i].x, part[v]);
        part[v] = fmaf(x[v][i].y, x[v][i].y, part[v]);
      }
    }
    // The sum of squares in one order for both shapes (the same bits for a row in any batch):
    // 8-element partial sums, added pairwise across the 16 eighths of the row as a 16-lane
    // butterfly (xor 8, 4, 2, then 1) would.  The lanes of a row are an aligned group of the warp,
    // so the xor offsets stay inside it; every lane takes part (a lane past the last row holds
    // zeros), so the full mask is exact.
#pragma unroll
    for (int offset = S::kLanes / 2; offset > 0; offset >>= 1) {
#pragma unroll
      for (int v = 0; v < S::kPer; ++v) {
        part[v] += __shfl_xor_sync(0xffffffffu, part[v], offset);
      }
    }
    float ss = part[0];
#pragma unroll
    for (int v = 1; v < S::kPer; ++v) {
      ss += part[v];
    }
    const float inv = rsqrtf(ss + kEps);
    if (row < op.rows) {
      uint4* dst =
          reinterpret_cast<uint4*>(op.dst + static_cast<int64_t>(row) * kHead) + lane * S::kPer;
#pragma unroll
      for (int v = 0; v < S::kPer; ++v) {
        uint4 out;
        out.x = narrow(x[v][0].x * inv, x[v][0].y * inv);
        out.y = narrow(x[v][1].x * inv, x[v][1].y * inv);
        out.z = narrow(x[v][2].x * inv, x[v][2].y * inv);
        out.w = narrow(x[v][3].x * inv, x[v][3].y * inv);
        store<kStream>(dst + v, out);
      }
    }
  }
}

inline Operand MakeOperand(const TensorView& src, const TensorView& dst, const char* name) {
  TVM_FFI_ICHECK(src.ndim() == 3 && src.size(2) == kHead)
      << name << " must be [tokens, heads, 128]";
  TVM_FFI_ICHECK(src.stride(2) == 1) << name << " must have a dense innermost dimension";
  TVM_FFI_ICHECK(src.stride(0) % kVector == 0 && src.stride(1) % kVector == 0 &&
                 reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0)
      << name << " must have 16-byte aligned rows";
  TVM_FFI_ICHECK(dst.ndim() == 3 && dst.size(0) == src.size(0) && dst.size(1) == src.size(1) &&
                 dst.size(2) == kHead && dst.IsContiguous())
      << name << " output must be a dense tensor of the input's shape";
  TVM_FFI_ICHECK(dst.dtype().code == kDLBfloat && dst.dtype().bits == 16)
      << name << " output must be bfloat16";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0)
      << name << " output must be 16-byte aligned";
  const int64_t rows = src.size(0) * src.size(1);
  TVM_FFI_ICHECK(rows < (int64_t{1} << 31)) << name << " has too many rows";
  Operand op;
  op.src = src.data_ptr();
  op.dst = static_cast<__nv_bfloat16*>(dst.data_ptr());
  op.stride_t = src.stride(0);
  op.stride_h = src.stride(1);
  op.rows = static_cast<uint32_t>(rows);
  op.heads = static_cast<uint32_t>(src.size(1) > 0 ? src.size(1) : 1);
  op.tiles = 0;
  return op;
}

inline bool Dense(const Operand& op) {
  return op.stride_h == kHead && op.stride_t == int64_t{op.heads} * kHead;
}

template <typename T, bool kStream>
void Launch(Operand q, Operand k, cudaStream_t stream) {
  constexpr uint32_t per_tile = Shape<kStream>::kRowsPerTile;
  q.tiles = (q.rows + per_tile - 1) / per_tile;
  k.tiles = (k.rows + per_tile - 1) / per_tile;
  const uint32_t grid = q.tiles + k.tiles;
  if (grid == 0) {
    return;
  }
  if (Dense(q) && Dense(k)) {
    qk_l2norm_kernel<T, true, kStream><<<grid, kThreads, 0, stream>>>(q, k);
  } else {
    qk_l2norm_kernel<T, false, kStream><<<grid, kThreads, 0, stream>>>(q, k);
  }
}

template <typename T>
void Launch(const Operand& q, const Operand& k, cudaStream_t stream) {
  const int64_t bytes = (int64_t{q.rows} + k.rows) * kHead * static_cast<int64_t>(sizeof(T));
  if (bytes > kStreamBytes) {
    Launch<T, true>(q, k, stream);
  } else {
    Launch<T, false>(q, k, stream);
  }
}

// q ``[T, H, 128]`` and k ``[T n, H, 128]`` (BF16 or FP16, one dtype) into the dense BF16 ``q_out``
// / ``k_out``.
void Run(TensorView q, TensorView k, TensorView q_out, TensorView k_out) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  tvm::ffi::CUDADeviceGuard guard(q.device().device_id);
  TVM_FFI_ICHECK(q.dtype() == k.dtype()) << "q and k must share one dtype";
  const bool bf16 = q.dtype().code == kDLBfloat && q.dtype().bits == 16;
  const bool f16 = q.dtype().code == kDLFloat && q.dtype().bits == 16;
  TVM_FFI_ICHECK(bf16 || f16) << "q and k must be bfloat16 or float16";
  const Operand oq = MakeOperand(q, q_out, "q");
  const Operand ok = MakeOperand(k, k_out, "k");
  TVM_FFI_ICHECK(int64_t{oq.rows} + ok.rows < (int64_t{1} << 31)) << "too many rows";
  const auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(q.device().device_type, q.device().device_id));
  if (bf16) {
    Launch<__nv_bfloat16>(oq, ok, stream);
  } else {
    Launch<__half>(oq, ok, stream);
  }
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "qk_l2norm launch failed: " << cudaGetErrorString(status);
}

}  // namespace cake_gdp_qk_l2norm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_gdp_qk_l2norm::Run);
