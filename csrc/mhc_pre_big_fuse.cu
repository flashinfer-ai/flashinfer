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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace flashinfer::mhc {
namespace {

static constexpr int kHc4 = 4;
static constexpr int kMixHc4 = kHc4 * (2 + kHc4);
static constexpr int kWarpThreads = 32;
static constexpr int kVecWidth = 8;

struct Bf16x8 {
  uint4 raw;
};

__device__ inline Bf16x8 load_bf16x8(const __nv_bfloat16* ptr) {
  Bf16x8 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.raw.x), "=r"(v.raw.y), "=r"(v.raw.z), "=r"(v.raw.w)
               : "l"(ptr));
  return v;
}

__device__ inline void store_bf16x8(__nv_bfloat16* ptr, const Bf16x8& v) {
  asm volatile("st.global.v4.u32 [%4], {%0, %1, %2, %3};\n"
               :
               : "r"(v.raw.x), "r"(v.raw.y), "r"(v.raw.z), "r"(v.raw.w), "l"(ptr));
}

__device__ inline __nv_bfloat162 bf16x8_get_pair(const Bf16x8& v, int i) {
  const uint32_t bits = i == 0 ? v.raw.x : (i == 1 ? v.raw.y : (i == 2 ? v.raw.z : v.raw.w));
  __nv_bfloat162 out;
  *reinterpret_cast<uint32_t*>(&out) = bits;
  return out;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_sum(float v, float* shared) {
  constexpr int kWarps = BLOCK_SIZE / kWarpThreads;
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const int warp = threadIdx.x / kWarpThreads;

  v = warp_sum(v);
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();

  v = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    v = warp_sum(v);
  }
  if (threadIdx.x == 0) {
    shared[0] = v;
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float residual_square_sum_vec8(const __nv_bfloat16* residual, int K) {
  float sum = 0.0f;
  for (int idx = threadIdx.x * kVecWidth; idx < K; idx += BLOCK_SIZE * kVecWidth) {
    const Bf16x8 raw = load_bf16x8(residual + idx);
#pragma unroll
    for (int i = 0; i < kVecWidth / 2; ++i) {
      const float2 f = __bfloat1622float2(bf16x8_get_pair(raw, i));
      sum = fmaf(f.x, f.x, sum);
      sum = fmaf(f.y, f.y, sum);
    }
  }
  return sum;
}

__device__ __forceinline__ void write_token_metadata(
    const float* y_local, float rstd, const float* mhc_scale, const float* mhc_base,
    float* post_mix, float* comb_mix, float* pre_mix, float mhc_pre_eps, float mhc_sinkhorn_eps,
    float mhc_post_mult_value, int sinkhorn_repeat) {
  const int lane = threadIdx.x & (kWarpThreads - 1);
  float cm[kHc4];

  const float scale_pre = mhc_scale[0];
  const float scale_post = mhc_scale[1];
  const float scale_comb = mhc_scale[2];

  float v = y_local[lane] * rstd * scale_pre + mhc_base[lane];
  pre_mix[lane] = 1.0f / (1.0f + expf(-v)) + mhc_pre_eps;

  v = y_local[kHc4 + lane] * rstd * scale_post + mhc_base[kHc4 + lane];
  post_mix[lane] = 1.0f / (1.0f + expf(-v)) * mhc_post_mult_value;

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = y_local[2 * kHc4 + lane * kHc4 + k] * rstd * scale_comb +
            mhc_base[2 * kHc4 + lane * kHc4 + k];
  }

  constexpr unsigned kLaneMask = (1u << kHc4) - 1;
  float row_max = fmaxf(fmaxf(cm[0], cm[1]), fmaxf(cm[2], cm[3]));
#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = expf(cm[k] - row_max);
  }
  float row_sum = cm[0] + cm[1] + cm[2] + cm[3];
#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = cm[k] / row_sum + mhc_sinkhorn_eps;
  }

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    float col_sum = cm[k];
    col_sum += __shfl_xor_sync(kLaneMask, col_sum, 1);
    col_sum += __shfl_xor_sync(kLaneMask, col_sum, 2);
    cm[k] /= (col_sum + mhc_sinkhorn_eps);
  }

  for (int it = 1; it < sinkhorn_repeat; ++it) {
    row_sum = cm[0] + cm[1] + cm[2] + cm[3] + mhc_sinkhorn_eps;
#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      cm[k] /= row_sum;
    }

#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      float col_sum = cm[k];
      col_sum += __shfl_xor_sync(kLaneMask, col_sum, 1);
      col_sum += __shfl_xor_sync(kLaneMask, col_sum, 2);
      cm[k] /= (col_sum + mhc_sinkhorn_eps);
    }
  }

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    comb_mix[lane * kHc4 + k] = cm[k];
  }
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void write_layer_input(const __nv_bfloat16* residual,
                                                  const float* pre_mix, __nv_bfloat16* layer_input,
                                                  int H) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpThreads;
  if (warp_id == 0) {
    return;
  }

  const int p2_tid = tid - kWarpThreads;
  constexpr int p2_threads = BLOCK_SIZE - kWarpThreads;
  for (int h = p2_tid * kVecWidth; h < H; h += p2_threads * kVecWidth) {
    float acc[kVecWidth] = {};

#pragma unroll
    for (int j = 0; j < kHc4; ++j) {
      const Bf16x8 raw = load_bf16x8(residual + j * H + h);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const float2 f = __bfloat1622float2(bf16x8_get_pair(raw, v));
        acc[2 * v + 0] += pre_mix[j] * f.x;
        acc[2 * v + 1] += pre_mix[j] * f.y;
      }
    }

    Bf16x8 out;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&out.raw);
#pragma unroll
    for (int v = 0; v < kVecWidth / 2; ++v) {
      pairs[v] = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
    }
    store_bf16x8(layer_input + h, out);
  }
}

template <int NUM_SPLITS, int BLOCK_SIZE, bool COMPUTE_SQRSUM>
__launch_bounds__(BLOCK_SIZE) __global__ void mhc_pre_big_fuse_kernel(
    const float* __restrict__ dot_mix, const float* __restrict__ sqrsum,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ layer_input, int total_tokens, int K, int H, float rms_eps,
    float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value, int sinkhorn_repeat) {
  static_assert(NUM_SPLITS == 1 || !COMPUTE_SQRSUM,
                "internal sqrsum path only supports NUM_SPLITS=1");

  const int token = blockIdx.x;
  if (token >= total_tokens) {
    return;
  }

  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const __nv_bfloat16* residual_token = residual + static_cast<long long>(token) * kHc4 * H;

  __shared__ float pre_mix[kHc4];
  __shared__ float sq_partials[BLOCK_SIZE / kWarpThreads];

  float sq_total = 0.0f;
  if constexpr (COMPUTE_SQRSUM) {
    const float local_sq = residual_square_sum_vec8<BLOCK_SIZE>(residual_token, kHc4 * H);
    sq_total = block_sum<BLOCK_SIZE>(local_sq, sq_partials);
  }

  if (warp_id == 0 && lane < kHc4) {
    float y_local[kMixHc4];
    if constexpr (NUM_SPLITS == 1) {
      if constexpr (!COMPUTE_SQRSUM) {
        sq_total = sqrsum[token];
      }
      const float* y_row = dot_mix + static_cast<long long>(token) * kMixHc4;
#pragma unroll
      for (int c = 0; c < kMixHc4; ++c) {
        y_local[c] = y_row[c];
      }
    } else {
      sq_total = 0.0f;
#pragma unroll
      for (int c = 0; c < kMixHc4; ++c) {
        y_local[c] = 0.0f;
      }
#pragma unroll
      for (int s = 0; s < NUM_SPLITS; ++s) {
        sq_total += sqrsum[s * total_tokens + token];
        const float* y_row = dot_mix + (static_cast<long long>(s) * total_tokens + token) * kMixHc4;
#pragma unroll
        for (int c = 0; c < kMixHc4; ++c) {
          y_local[c] += y_row[c];
        }
      }
    }

    const float rstd = rsqrtf(sq_total / static_cast<float>(K) + rms_eps);
    write_token_metadata(y_local, rstd, mhc_scale, mhc_base, post_mix + token * kHc4,
                         comb_mix + token * kHc4 * kHc4, pre_mix, mhc_pre_eps, mhc_sinkhorn_eps,
                         mhc_post_mult_value, sinkhorn_repeat);
  }

  __syncthreads();
  write_layer_input<BLOCK_SIZE>(residual_token, pre_mix,
                                layer_input + static_cast<long long>(token) * H, H);
}

static int select_pre_big_fuse_block_size(int /* total_tokens */) { return 256; }

static int select_pre_big_fuse_with_prenorm_block_size(int /* total_tokens */) { return 128; }

template <int NUM_SPLITS, bool COMPUTE_SQRSUM>
void dispatch_pre_big_fuse(const float* dot_mix, const float* sqrsum, const __nv_bfloat16* residual,
                           const float* mhc_scale, const float* mhc_base, float* post_mix,
                           float* comb_mix, __nv_bfloat16* layer_input, int total_tokens, int K,
                           int H, float rms_eps, float mhc_pre_eps, float mhc_sinkhorn_eps,
                           float mhc_post_mult_value, int sinkhorn_repeat, int block_size,
                           cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(total_tokens));

#define LAUNCH_PRE_BIG_FUSE(BS)                                                                  \
  mhc_pre_big_fuse_kernel<NUM_SPLITS, BS, COMPUTE_SQRSUM>                                        \
      <<<grid, BS, 0, stream>>>(dot_mix, sqrsum, residual, mhc_scale, mhc_base, post_mix,        \
                                comb_mix, layer_input, total_tokens, K, H, rms_eps, mhc_pre_eps, \
                                mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat)

  if (block_size >= 512) {
    LAUNCH_PRE_BIG_FUSE(512);
  } else if (block_size >= 256) {
    LAUNCH_PRE_BIG_FUSE(256);
  } else {
    LAUNCH_PRE_BIG_FUSE(128);
  }

#undef LAUNCH_PRE_BIG_FUSE
}

void launch_pre_big_fuse(TensorView dot_mix, TensorView sqrsum, TensorView residual,
                         TensorView mhc_scale, TensorView mhc_base, TensorView post_mix,
                         TensorView comb_mix, TensorView layer_input, int64_t total_tokens,
                         int64_t K, int64_t H, double rms_eps, double mhc_pre_eps,
                         double mhc_sinkhorn_eps, double mhc_post_mult_value,
                         int64_t sinkhorn_repeat, int64_t num_splits, int64_t block_size,
                         cudaStream_t stream) {
  const int bs = block_size > 0 ? static_cast<int>(block_size)
                                : select_pre_big_fuse_block_size(static_cast<int>(total_tokens));
  const auto* dot_ptr = static_cast<const float*>(dot_mix.data_ptr());
  const auto* sq_ptr = static_cast<const float*>(sqrsum.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(mhc_scale.data_ptr());
  const auto* base_ptr = static_cast<const float*>(mhc_base.data_ptr());
  auto* post_ptr = static_cast<float*>(post_mix.data_ptr());
  auto* comb_ptr = static_cast<float*>(comb_mix.data_ptr());
  auto* layer_ptr = reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr());

#define DISPATCH_SPLITS(NS)                                                              \
  dispatch_pre_big_fuse<NS, false>(                                                      \
      dot_ptr, sq_ptr, residual_ptr, scale_ptr, base_ptr, post_ptr, comb_ptr, layer_ptr, \
      static_cast<int>(total_tokens), static_cast<int>(K), static_cast<int>(H),          \
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),                      \
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),     \
      static_cast<int>(sinkhorn_repeat), bs, stream)

  switch (num_splits) {
    case 1:
      DISPATCH_SPLITS(1);
      break;
    case 2:
      DISPATCH_SPLITS(2);
      break;
    case 4:
      DISPATCH_SPLITS(4);
      break;
    case 8:
      DISPATCH_SPLITS(8);
      break;
    case 16:
      DISPATCH_SPLITS(16);
      break;
  }
#undef DISPATCH_SPLITS
}

void launch_pre_big_fuse_with_prenorm(TensorView dot_mix, TensorView residual, TensorView mhc_scale,
                                      TensorView mhc_base, TensorView post_mix, TensorView comb_mix,
                                      TensorView layer_input, int64_t total_tokens, int64_t H,
                                      double rms_eps, double mhc_pre_eps, double mhc_sinkhorn_eps,
                                      double mhc_post_mult_value, int64_t sinkhorn_repeat,
                                      int64_t block_size, cudaStream_t stream) {
  const int bs = block_size > 0
                     ? static_cast<int>(block_size)
                     : select_pre_big_fuse_with_prenorm_block_size(static_cast<int>(total_tokens));
  const auto* dot_ptr = static_cast<const float*>(dot_mix.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(mhc_scale.data_ptr());
  const auto* base_ptr = static_cast<const float*>(mhc_base.data_ptr());
  auto* post_ptr = static_cast<float*>(post_mix.data_ptr());
  auto* comb_ptr = static_cast<float*>(comb_mix.data_ptr());
  auto* layer_ptr = reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr());

  dispatch_pre_big_fuse<1, true>(
      dot_ptr, nullptr, residual_ptr, scale_ptr, base_ptr, post_ptr, comb_ptr, layer_ptr,
      static_cast<int>(total_tokens), static_cast<int>(kHc4 * H), static_cast<int>(H),
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),
      static_cast<int>(sinkhorn_repeat), bs, stream);
}

void check_common_shapes(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                         TensorView residual, TensorView mhc_scale, TensorView mhc_base,
                         int64_t* total_tokens, int64_t* H) {
  CHECK_INPUT_AND_TYPE(post_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(comb_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(layer_input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(mhc_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(mhc_base, dl_float32);
  CHECK_DEVICE(residual, post_mix);
  CHECK_DEVICE(residual, comb_mix);
  CHECK_DEVICE(residual, layer_input);
  CHECK_DEVICE(residual, mhc_scale);
  CHECK_DEVICE(residual, mhc_base);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, post_mix);
  CHECK_DIM(3, comb_mix);
  CHECK_DIM(2, layer_input);
  CHECK_DIM(1, mhc_scale);
  CHECK_DIM(1, mhc_base);

  *total_tokens = residual.size(0);
  const int64_t HC = residual.size(1);
  *H = residual.size(2);
  TVM_FFI_ICHECK_EQ(HC, kHc4) << "residual.shape[1] / HC must be 4";
  TVM_FFI_ICHECK_GT(*H, 0) << "hidden size must be positive";
  TVM_FFI_ICHECK_EQ(*H % kVecWidth, 0) << "hidden size must be divisible by 8";
  TVM_FFI_ICHECK_EQ(mhc_scale.size(0), 3) << "mhc_scale must have shape [3]";
  TVM_FFI_ICHECK_EQ(mhc_base.size(0), kMixHc4) << "mhc_base must have shape [24]";
  TVM_FFI_ICHECK_EQ(post_mix.size(0), *total_tokens)
      << "post_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(post_mix.size(1), kHc4) << "post_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_mix.size(0), *total_tokens)
      << "comb_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(comb_mix.size(1), kHc4) << "comb_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_mix.size(2), kHc4) << "comb_mix.shape[2] must be 4";
  TVM_FFI_ICHECK_EQ(layer_input.size(0), *total_tokens)
      << "layer_input.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(layer_input.size(1), *H) << "layer_input.shape[1] must match hidden size";
}

}  // namespace

void mhc_pre_big_fuse(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                      TensorView dot_mix, TensorView sqrsum, TensorView residual,
                      TensorView mhc_scale, TensorView mhc_base, int64_t K, double rms_eps,
                      double mhc_pre_eps, double mhc_sinkhorn_eps, double mhc_post_mult_value,
                      int64_t sinkhorn_repeat, int64_t num_splits, int64_t block_size) {
  CHECK_INPUT_AND_TYPE(dot_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(sqrsum, dl_float32);
  int64_t total_tokens = 0;
  int64_t H = 0;
  check_common_shapes(post_mix, comb_mix, layer_input, residual, mhc_scale, mhc_base, &total_tokens,
                      &H);
  CHECK_DEVICE(residual, dot_mix);
  CHECK_DEVICE(residual, sqrsum);

  TVM_FFI_ICHECK_GT(K, 0) << "K must be positive";
  TVM_FFI_ICHECK_GE(sinkhorn_repeat, 1) << "sinkhorn_repeat must be >= 1";
  TVM_FFI_ICHECK(num_splits == 1 || num_splits == 2 || num_splits == 4 || num_splits == 8 ||
                 num_splits == 16)
      << "num_splits must be one of {1,2,4,8,16}";

  if (num_splits == 1) {
    CHECK_DIM(2, dot_mix);
    CHECK_DIM(1, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), total_tokens)
        << "dot_mix.shape[0] must match residual.shape[0]";
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), kMixHc4) << "dot_mix.shape[1] must be 24";
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), total_tokens)
        << "sqrsum.shape[0] must match residual.shape[0]";
  } else {
    CHECK_DIM(3, dot_mix);
    CHECK_DIM(2, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), num_splits) << "dot_mix.shape[0] must match num_splits";
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), total_tokens)
        << "dot_mix.shape[1] must match residual.shape[0]";
    TVM_FFI_ICHECK_EQ(dot_mix.size(2), kMixHc4) << "dot_mix.shape[2] must be 24";
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), num_splits) << "sqrsum.shape[0] must match num_splits";
    TVM_FFI_ICHECK_EQ(sqrsum.size(1), total_tokens)
        << "sqrsum.shape[1] must match residual.shape[0]";
  }

  if (total_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());
  launch_pre_big_fuse(dot_mix, sqrsum, residual, mhc_scale, mhc_base, post_mix, comb_mix,
                      layer_input, total_tokens, K, H, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                      mhc_post_mult_value, sinkhorn_repeat, num_splits, block_size, stream);
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse kernel launch failed with error code " << cudaGetErrorString(status);
}

void mhc_pre_big_fuse_with_prenorm(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                                   TensorView dot_mix, TensorView residual, TensorView mhc_scale,
                                   TensorView mhc_base, double rms_eps, double mhc_pre_eps,
                                   double mhc_sinkhorn_eps, double mhc_post_mult_value,
                                   int64_t sinkhorn_repeat, int64_t block_size) {
  CHECK_INPUT_AND_TYPE(dot_mix, dl_float32);
  int64_t total_tokens = 0;
  int64_t H = 0;
  check_common_shapes(post_mix, comb_mix, layer_input, residual, mhc_scale, mhc_base, &total_tokens,
                      &H);
  CHECK_DEVICE(residual, dot_mix);
  CHECK_DIM(2, dot_mix);
  TVM_FFI_ICHECK_EQ(dot_mix.size(0), total_tokens)
      << "dot_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(dot_mix.size(1), kMixHc4) << "dot_mix.shape[1] must be 24";
  TVM_FFI_ICHECK_GE(sinkhorn_repeat, 1) << "sinkhorn_repeat must be >= 1";

  if (total_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());
  launch_pre_big_fuse_with_prenorm(dot_mix, residual, mhc_scale, mhc_base, post_mix, comb_mix,
                                   layer_input, total_tokens, H, rms_eps, mhc_pre_eps,
                                   mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat,
                                   block_size, stream);
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse_with_prenorm kernel launch failed with error code "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_big_fuse, flashinfer::mhc::mhc_pre_big_fuse);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_big_fuse_with_prenorm,
                              flashinfer::mhc::mhc_pre_big_fuse_with_prenorm);

}  // namespace flashinfer::mhc
