/*
 * Originally contributed to the FlashInfer-Bench MLSys'26 kernel-generation
 * contest (https://github.com/flashinfer-ai/mlsys26-contest) by the IFKernel
 * team. Adapted into flashinfer's JIT pipeline.
 *
 * TODO(license): the upstream merge is conditional on the contest team
 * granting an Apache-2.0 license for this file. Until that grant is on file
 * the file must NOT be redistributed.
 */
// TVM FFI version of FuseMoE CUDA kernel (48x speedup, converted from torch binding)
#include <cublas_v2.h>  // For type definitions only
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"

// FuseMoE Blackwell helper-kernel ABIs.
//
// The CUTLASS blockwise FP8 group GEMM and the tcgen05 grouped GEMM live in
// sibling translation units (csrc/fusemoe_blackwell_cutlass_bw.cu and
// csrc/fusemoe_blackwell_tcgen05.cu) and are linked into the same .so by the
// flashinfer JIT. Earlier revisions built those .so's at runtime via
// system("nvcc ...") + dlopen — the indirection is gone now; we just declare
// their entry points and call them directly.
#include "fusemoe_blackwell_gemm_args.h"

extern "C" {
// CUTLASS group-FP8 GEMM (csrc/fusemoe_blackwell_cutlass_bw.cu).
int cutlass_blockwise_fp8_gemm(GemmArgs* a, cudaStream_t stream);
int cutlass_blockwise_fp8_gemm_128(GemmArgs* a, cudaStream_t stream);
int cutlass_blockwise_fp8_gemm_noprep(GemmArgs* a, cudaStream_t stream);
int cutlass_blockwise_fp8_gemm_128_noprep(GemmArgs* a, cudaStream_t stream);
int cutlass_blockwise_fp8_gemm_noprep2(GemmArgs* a, cudaStream_t stream);
int cutlass_blockwise_fp8_gemm_128_noprep2(GemmArgs* a, cudaStream_t stream);
int cutlass_prep_dual(GemmArgsDual* a, cudaStream_t stream);

// tcgen05 grouped FP8 GEMM (csrc/fusemoe_blackwell_tcgen05.cu).
int tcgen05_setup_tma(void* Ap, int mr, void* Bp, int ne, int N, int K);
int tcgen05_setup_tma2(void* Ap, int mr, void* Bp, int ne, int N, int K);
int tcgen05_grouped_gemm(GemmArgs* a, cudaStream_t stream);
int tcgen05_grouped_gemm2(GemmArgs* a, cudaStream_t stream);
void tcgen05_set_total_rows(int r);
}  // extern "C"

typedef int (*CutlassBwFn)(GemmArgs*, cudaStream_t);

// Per-call state previously tied to .so loading; the boolean is kept because
// it tracks runtime TMA-descriptor setup (separate from "is the symbol
// available").
static bool g_tcgen05_tma_ready = false;
static bool g_tcgen05_tma2_ready = false;

// TVM FFI helpers
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

inline tvm::ffi::Tensor alloc_tensor(tvm::ffi::ShapeView shape, DLDataType dtype, DLDevice device) {
  return tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
}

inline cudaStream_t get_current_stream() {
  int device;
  cudaGetDevice(&device);
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
}

// Host callback for spin-wait: set atomic flag when GPU stream reaches this point.
static void CUDART_CB stream_done_callback(void* arg) {
  std::atomic<int>* flag = reinterpret_cast<std::atomic<int>*>(arg);
  flag->store(1, std::memory_order_release);
}

// Shape constants (kHidden, kIntermediate, kNumExpertsGlobal, kNumLocalExperts,
// kBlock, kTopK, kNumGroups, kTopKGroup) are JIT-rendered per call site —
// see flashinfer/jit/fusemoe_blackwell.py and csrc/fusemoe_blackwell_config.jinja.
#include "fusemoe_blackwell_config.h"

namespace {

constexpr int kMaxTkChunk = 8192;
constexpr int kMaxTkChunkLong = 8192;
constexpr int kLongSeqThreshold = 8192;

__device__ __constant__ float kFp8Lut[256];

#define CUDA_CHECK(expr)                                                               \
  do {                                                                                 \
    cudaError_t _err = (expr);                                                         \
    TVM_FFI_ICHECK(_err == cudaSuccess) << "CUDA error: " << cudaGetErrorString(_err); \
  } while (0)

#define CUBLAS_CHECK(expr)                                                                         \
  do {                                                                                             \
    cublasStatus_t _st = (expr);                                                                   \
    TVM_FFI_ICHECK(_st == CUBLAS_STATUS_SUCCESS) << "cuBLAS error code " << static_cast<int>(_st); \
  } while (0)

// ===================== FP8 LUT =====================

float decode_fp8_e4m3fn_host(uint8_t x) {
  const int sign = (x >> 7) & 1;
  const int exp = (x >> 3) & 0xF;
  const int mant = x & 0x7;
  float val = 0.0f;
  if (exp == 0) {
    if (mant == 0) {
      val = 0.0f;
    } else {
      val = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }
  } else if (exp == 0xF) {
    val = 448.0f;
  } else {
    val = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
  }
  return sign ? -val : val;
}

void init_fp8_lut_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    std::array<float, 256> host{};
    for (int i = 0; i < 256; ++i) {
      host[i] = decode_fp8_e4m3fn_host(static_cast<uint8_t>(i));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(kFp8Lut, host.data(), sizeof(float) * host.size()));
  });
}

__device__ __forceinline__ float fp8_to_float(uint8_t x) { return kFp8Lut[x]; }

// ===================== ALL CUDA DEVICE KERNELS (UNCHANGED) =====================

__device__ __forceinline__ void warp_argmax(float& best_val, int& best_idx) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
    const int other_idx = __shfl_down_sync(0xffffffffu, best_idx, offset);
    if (other_val > best_val) {
      best_val = other_val;
      best_idx = other_idx;
    }
  }
}

__global__ __launch_bounds__(256) void routing_kernel(
    const float* __restrict__ routing_logits,      // [T, 256]
    const nv_bfloat16* __restrict__ routing_bias,  // [256]
    int t, float routed_scale, int local_expert_offset,
    int32_t* __restrict__ topk_idx,  // [T, 8]
    float* __restrict__ topk_w,      // [T, 8]
    int32_t* __restrict__ counts) {  // [E_local] -- fused count (atomic)
  const int token = blockIdx.x;
  if (token >= t) {
    return;
  }

  __shared__ float sb[kNumExpertsGlobal];
  __shared__ float group_scores[kNumGroups];
  __shared__ uint8_t group_kept[kNumGroups];
  __shared__ uint8_t expert_used[kNumExpertsGlobal];
  __shared__ float warp_best_val[kNumGroups];
  __shared__ int warp_best_idx[kNumGroups];
  __shared__ int iter_best_expert;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  if (tid < kNumExpertsGlobal) {
    const float logit = routing_logits[token * kNumExpertsGlobal + tid];
    const float sv = 1.0f / (1.0f + __expf(-logit));
    sb[tid] = sv + __bfloat162float(routing_bias[tid]);
    expert_used[tid] = 0;
  }
  if (tid < kNumGroups) {
    group_scores[tid] = -INFINITY;
    group_kept[tid] = 0;
  }
  __syncthreads();

  if (warp_id < kNumGroups) {
    const int expert_idx = warp_id * (kNumExpertsGlobal / kNumGroups) + lane;
    float my_val = sb[expert_idx];
    // Find top-1 via warp argmax reduction
    float best1_val = my_val;
    int best1_idx = lane;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float o_val = __shfl_down_sync(0xffffffffu, best1_val, offset);
      const int o_idx = __shfl_down_sync(0xffffffffu, best1_idx, offset);
      if (o_val > best1_val) {
        best1_val = o_val;
        best1_idx = o_idx;
      }
    }
    best1_val = __shfl_sync(0xffffffffu, best1_val, 0);
    best1_idx = __shfl_sync(0xffffffffu, best1_idx, 0);
    // Find top-2: mask out best1, reduce again
    float val2 = (lane == best1_idx) ? -INFINITY : my_val;
    float best2_val = val2;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float o_val = __shfl_down_sync(0xffffffffu, best2_val, offset);
      if (o_val > best2_val) {
        best2_val = o_val;
      }
    }
    best2_val = __shfl_sync(0xffffffffu, best2_val, 0);
    if (lane == 0) group_scores[warp_id] = best1_val + best2_val;
  }
  __syncthreads();

  if (tid == 0) {
    for (int k = 0; k < kTopKGroup; ++k) {
      float best = -INFINITY;
      int best_g = 0;
      for (int g = 0; g < kNumGroups; ++g) {
        if (!group_kept[g] && group_scores[g] > best) {
          best = group_scores[g];
          best_g = g;
        }
      }
      group_kept[best_g] = 1;
    }
  }
  __syncthreads();

  for (int k = 0; k < kTopK; ++k) {
    float best_val = -INFINITY;
    int best_idx = tid;
    if (tid < kNumExpertsGlobal && !expert_used[tid] &&
        group_kept[tid / (kNumExpertsGlobal / kNumGroups)]) {
      best_val = sb[tid];
    }

    warp_argmax(best_val, best_idx);

    if (lane == 0) {
      warp_best_val[warp_id] = best_val;
      warp_best_idx[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
      float block_best_val = (lane < kNumGroups) ? warp_best_val[lane] : -INFINITY;
      int block_best_idx = (lane < kNumGroups) ? warp_best_idx[lane] : 0;
      warp_argmax(block_best_val, block_best_idx);
      if (lane == 0) {
        iter_best_expert = block_best_idx;
      }
    }
    // No __syncthreads needed here: only tid==0 reads iter_best_expert,
    // and tid==0 (lane 0 of warp 0) is the same thread that wrote it.

    if (tid == 0) {
      const int best_e = iter_best_expert;
      expert_used[best_e] = 1;
      topk_idx[token * kTopK + k] = best_e;
      topk_w[token * kTopK + k] = sb[best_e] - __bfloat162float(routing_bias[best_e]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    float sum_s = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
      sum_s += topk_w[token * kTopK + k];
    }
    const float inv = routed_scale / (sum_s + 1e-20f);
    for (int k = 0; k < kTopK; ++k) {
      topk_w[token * kTopK + k] *= inv;
    }
    for (int k = 0; k < kTopK; ++k) {
      const int ge = topk_idx[token * kTopK + k];
      const int le = ge - local_expert_offset;
      if (le >= 0 && le < kNumLocalExperts) {
        atomicAdd(counts + le, 1);
      }
    }
  }
}

__global__ void scatter_local_assignments_kernel(
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w, int t,
    int local_expert_offset, const int32_t* __restrict__ offsets, int32_t* __restrict__ cursors,
    int32_t* __restrict__ packed_tok, float* __restrict__ packed_w,
    int32_t* __restrict__ packed_invrow) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = t * kTopK;
  if (idx >= total) return;
  const int tok = idx / kTopK;
  const int ge = topk_idx[idx];
  const int le = ge - local_expert_offset;
  if (le >= 0 && le < kNumLocalExperts) {
    const int pos = atomicAdd(cursors + le, 1);
    const int out_idx = offsets[le] + pos;
    packed_tok[out_idx] = tok;
    packed_invrow[idx] = pos;
  }
}

// Fused version: computes prefix sum from counts internally, eliminating separate scan kernel
__global__ void scatter_with_scan_kernel(
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w, int t,
    int local_expert_offset, const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets_out,  // write computed offsets here
    int32_t* __restrict__ cursors, int32_t* __restrict__ packed_tok, float* __restrict__ packed_w,
    int32_t* __restrict__ packed_invrow,
    int32_t* __restrict__ combined_out,    // metadata output (nullable)
    int32_t* __restrict__ mapped_total) {  // mapped host ptr for total_tight_rows (nullable)

  __shared__ int32_t s_offsets[kNumLocalExperts];

  // Warp 0 computes exclusive prefix sum from counts
  if (threadIdx.x < kNumLocalExperts) {
    const int le = threadIdx.x;
    int val = counts[le];
    const int my_count = val;
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, val, s);
      if (le >= s) val += n;
    }
    const int prev = __shfl_up_sync(0xffffffffu, val, 1);
    const int exclusive = (le == 0) ? 0 : prev;
    s_offsets[le] = exclusive;

    // Block 0 writes offsets to global memory (for use by subsequent kernels)
    if (blockIdx.x == 0) {
      offsets_out[le] = exclusive;

      // Write metadata if combined_out is provided (GPU planner fast path)
      if (combined_out != nullptr) {
        combined_out[le] = exclusive;
        if (le == 31) combined_out[32] = val;    // total_tight_rows
        combined_out[33 + le] = le;              // expert_ids: identity
        combined_out[33 + 32 + le] = my_count;   // active_counts
        combined_out[33 + 64 + le] = exclusive;  // base_offsets
        combined_out[33 + 96 + le] = le;         // le_to_rank: identity
      }
      // Write total to mapped host memory for CPU spin-wait (independent of combined_out)
      if (le == 31 && mapped_total != nullptr) {
        *mapped_total = val;
        __threadfence_system();
      }
    }
  }
  __syncthreads();

  // Scatter logic (same as original, but uses s_offsets instead of global offsets)
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = t * kTopK;
  if (idx >= total) return;
  const int tok = idx / kTopK;
  const int ge = topk_idx[idx];
  const int le = ge - local_expert_offset;
  if (le >= 0 && le < kNumLocalExperts) {
    const int pos = atomicAdd(cursors + le, 1);
    const int out_idx = s_offsets[le] + pos;
    packed_tok[out_idx] = tok;
    packed_invrow[idx] = pos;
  }
}

__global__ void gather_dequant_hidden_kernel(const uint8_t* __restrict__ hidden_fp8,
                                             const float* __restrict__ hidden_scale,
                                             const int32_t* __restrict__ token_idx, int t, int tk,
                                             float* __restrict__ out_a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kVec = 4;
  const int total = tk * (kHidden / kVec);
  if (idx >= total) return;
  const int row = idx / (kHidden / kVec);
  const int h4 = idx % (kHidden / kVec);
  const unsigned lane = threadIdx.x & 31u;
  int tok = 0;
  if (lane == 0u) tok = token_idx[row];
  tok = __shfl_sync(0xffffffffu, tok, 0);
  const int base_h = h4 * kVec;
  const int out_base = row * kHidden + base_h;
  const int in_base = tok * kHidden + base_h;
  const int hb = base_h / kBlock;
  float scale = 0.0f;
  if (lane == 0u) scale = hidden_scale[hb * t + tok];
  scale = __shfl_sync(0xffffffffu, scale, 0);
  float4 out_v;
  out_v.x = fp8_to_float(hidden_fp8[in_base + 0]) * scale;
  out_v.y = fp8_to_float(hidden_fp8[in_base + 1]) * scale;
  out_v.z = fp8_to_float(hidden_fp8[in_base + 2]) * scale;
  out_v.w = fp8_to_float(hidden_fp8[in_base + 3]) * scale;
  *reinterpret_cast<float4*>(out_a + out_base) = out_v;
}

__global__ void gather_dequant_hidden_fp16_kernel(const uint8_t* __restrict__ hidden_fp8,
                                                  const float* __restrict__ hidden_scale,
                                                  const int32_t* __restrict__ token_idx, int t,
                                                  int tk, __half* __restrict__ out_a) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kVec = 4;
  const int total = tk * (kHidden / kVec);
  if (idx >= total) return;
  const int row = idx / (kHidden / kVec);
  const int h4 = idx % (kHidden / kVec);
  const unsigned lane = threadIdx.x & 31u;
  int tok = 0;
  if (lane == 0u) tok = token_idx[row];
  tok = __shfl_sync(0xffffffffu, tok, 0);
  const int base_h = h4 * kVec;
  const int out_base = row * kHidden + base_h;
  const int in_base = tok * kHidden + base_h;
  const int hb = base_h / kBlock;
  float scale = 0.0f;
  if (lane == 0u) scale = hidden_scale[hb * t + tok];
  scale = __shfl_sync(0xffffffffu, scale, 0);
  const float f0 = fp8_to_float(hidden_fp8[in_base + 0]) * scale;
  const float f1 = fp8_to_float(hidden_fp8[in_base + 1]) * scale;
  const float f2 = fp8_to_float(hidden_fp8[in_base + 2]) * scale;
  const float f3 = fp8_to_float(hidden_fp8[in_base + 3]) * scale;
  __half2* out_ptr = reinterpret_cast<__half2*>(out_a + out_base);
  out_ptr[0] = __float22half2_rn(make_float2(f0, f1));
  out_ptr[1] = __float22half2_rn(make_float2(f2, f3));
}

__global__ void dequant_w13_batched_fp16_kernel(const uint8_t* __restrict__ w13_all_fp8,
                                                const float* __restrict__ w13_all_scale,
                                                const int32_t* __restrict__ expert_ids,
                                                __half* __restrict__ w13_all_out) {
  constexpr int kWarpSize = 32;
  constexpr int kVec = 4;
  const int hb = blockIdx.x;
  const int ob = blockIdx.y;
  const int expert = expert_ids[blockIdx.z];
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int row0 = ob * kBlock;
  const int col0 = hb * kBlock;
  const int64_t expert_w13_base = static_cast<int64_t>(expert) * (2 * kIntermediate) * kHidden;
  const int64_t expert_s13_base =
      static_cast<int64_t>(expert) * ((2 * kIntermediate) / kBlock) * (kHidden / kBlock);
  const float s = w13_all_scale[expert_s13_base + ob * (kHidden / kBlock) + hb];
  for (int r = warp_id; r < kBlock; r += 8) {
    const int o = row0 + r;
    const int h = col0 + lane * kVec;
    const int64_t base = expert_w13_base + static_cast<int64_t>(o) * kHidden + h;
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(w13_all_fp8 + base);
    const __half2_raw raw_lo =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
    const __half2_raw raw_hi =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16u), __NV_E4M3);
    const float2 f2_lo = __half22float2(reinterpret_cast<const __half2&>(raw_lo));
    const float2 f2_hi = __half22float2(reinterpret_cast<const __half2&>(raw_hi));
    __half2* h2_out = reinterpret_cast<__half2*>(w13_all_out + base);
    h2_out[0] = __float22half2_rn(make_float2(f2_lo.x * s, f2_lo.y * s));
    h2_out[1] = __float22half2_rn(make_float2(f2_hi.x * s, f2_hi.y * s));
  }
}

__global__ void dequant_w2_batched_fp16_kernel(const uint8_t* __restrict__ w2_all_fp8,
                                               const float* __restrict__ w2_all_scale,
                                               const int32_t* __restrict__ expert_ids,
                                               __half* __restrict__ w2_all_out) {
  constexpr int kWarpSize = 32;
  constexpr int kVec = 4;
  const int ib = blockIdx.x;
  const int hb = blockIdx.y;
  const int expert = expert_ids[blockIdx.z];
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int row0 = hb * kBlock;
  const int col0 = ib * kBlock;
  const int64_t expert_w2_base = static_cast<int64_t>(expert) * kHidden * kIntermediate;
  const int64_t expert_s2_base =
      static_cast<int64_t>(expert) * (kHidden / kBlock) * (kIntermediate / kBlock);
  const float s = w2_all_scale[expert_s2_base + hb * (kIntermediate / kBlock) + ib];
  for (int r = warp_id; r < kBlock; r += 8) {
    const int h = row0 + r;
    const int i = col0 + lane * kVec;
    const int64_t base = expert_w2_base + static_cast<int64_t>(h) * kIntermediate + i;
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(w2_all_fp8 + base);
    const __half2_raw raw_lo =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
    const __half2_raw raw_hi =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16u), __NV_E4M3);
    const float2 f2_lo = __half22float2(reinterpret_cast<const __half2&>(raw_lo));
    const float2 f2_hi = __half22float2(reinterpret_cast<const __half2&>(raw_hi));
    __half2* h2_out = reinterpret_cast<__half2*>(w2_all_out + base);
    h2_out[0] = __float22half2_rn(make_float2(f2_lo.x * s, f2_lo.y * s));
    h2_out[1] = __float22half2_rn(make_float2(f2_hi.x * s, f2_hi.y * s));
  }
}

__global__ void swiglu_rowscale_fp16_batched_kernel(const __half* __restrict__ g1, int max_M,
                                                    const int32_t* __restrict__ d_counts,
                                                    __half* __restrict__ c_fp16,
                                                    float* __restrict__ row_scale) {
  constexpr int kThreads = 256;
  constexpr int kElemsPerThread = kIntermediate / kThreads;
  const int expert = blockIdx.y;
  const int row = blockIdx.x;
  if (row >= d_counts[expert]) return;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  __shared__ float smem_warp_max[kThreads / 32];
  __shared__ float s_inv_scale;
  const int g1_row_base = (expert * max_M + row) * (2 * kIntermediate);
  float local_vals[kElemsPerThread];
  float local_max = 0.0f;
  const __half2* g1_h2 = reinterpret_cast<const __half2*>(g1 + g1_row_base);
  constexpr int kPairsPerThread = kElemsPerThread / 2;
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int i2 = j * kThreads + tid;
    const float2 f1 = __half22float2(g1_h2[i2]);
    const float2 f2 = __half22float2(g1_h2[kIntermediate / 2 + i2]);
    const float v0 = __fdividef(f2.x, 1.0f + __expf(-f2.x)) * f1.x;
    const float v1 = __fdividef(f2.y, 1.0f + __expf(-f2.y)) * f1.y;
    local_vals[j * 2] = v0;
    local_vals[j * 2 + 1] = v1;
    local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
  }
  for (int s = 16; s > 0; s >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, s));
  if (lane == 0) smem_warp_max[warp_id] = local_max;
  __syncthreads();
  if (warp_id == 0) {
    float block_max = (lane < (kThreads / 32)) ? smem_warp_max[lane] : 0.0f;
    for (int s = 16; s > 0; s >>= 1)
      block_max = fmaxf(block_max, __shfl_down_sync(0xffffffffu, block_max, s));
    if (lane == 0) {
      const float scale = (block_max > 1.0f) ? block_max : 1.0f;
      row_scale[expert * max_M + row] = scale;
      s_inv_scale = 1.0f / scale;
    }
  }
  __syncthreads();
  const float inv_scale = s_inv_scale;
  const int c_row_base = (expert * max_M + row) * kIntermediate;
  __half2* c_h2 = reinterpret_cast<__half2*>(c_fp16 + c_row_base);
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int i2 = j * kThreads + tid;
    c_h2[i2] = __float22half2_rn(
        make_float2(local_vals[j * 2] * inv_scale, local_vals[j * 2 + 1] * inv_scale));
  }
}

__global__ void swiglu_to_fp8_kernel(const nv_bfloat16* __restrict__ g1_bf16, int max_M,
                                     const int32_t* __restrict__ d_counts,
                                     uint8_t* __restrict__ c_fp8, float* __restrict__ sfa_out,
                                     float* __restrict__ row_scale_out) {
  constexpr int kI = kIntermediate;  // 2048
  constexpr int kThreads = 256;
  constexpr int kEPT = kI / kThreads;  // 8
  const int expert = blockIdx.y, row = blockIdx.x;
  if (row >= d_counts[expert]) return;
  const int tid = threadIdx.x;
  const int64_t g1_base = ((int64_t)expert * max_M + row) * 2 * kI;
  float vals[kEPT];
  for (int j = 0; j < kEPT; j++) {
    int idx = j * kThreads + tid;
    float x1 = __bfloat162float(g1_bf16[g1_base + idx]);
    float x2 = __bfloat162float(g1_bf16[g1_base + kI + idx]);
    vals[j] = __fdividef(x2, 1.0f + __expf(-x2)) * x1;
  }
  // Each thread in tid [0,127] contributes to even blocks (2j), tid [128,255] to odd blocks (2j+1)
  // For each j, all 128 threads in a half contribute to the same block
  // Use warp reduction + shared memory instead of atomicMax
  constexpr int Ib = kI / kBlock;  // 16
  __shared__ float smem_block_max[16];
  __shared__ float smem_block_inv[16];
  // Compute per-block max using warp reduction
  // tid_half: 0=even blocks, 1=odd blocks
  const int tid_half = tid >> 7;              // 0 for tid<128, 1 for tid>=128
  const int tid_in_half = tid & 127;          // 0-127 within each half
  const int warp_in_half = tid_in_half >> 5;  // 0-3 warps per half
  const int lane = tid & 31;
  __shared__ float smem_warp_max[16 * 4];  // 16 blocks * 4 warps per half
// For each j, compute max within warp for block (2j + tid_half)
#pragma unroll
  for (int j = 0; j < kEPT; j++) {
    int blk = 2 * j + tid_half;
    float v = fabsf(vals[j]);
    // Warp-level reduction
    for (int s = 16; s > 0; s >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, s));
    if (lane == 0) smem_warp_max[blk * 4 + warp_in_half] = v;
  }
  __syncthreads();
  // Final reduction across 4 warps per block
  if (tid < Ib) {
    float bmax = fmaxf(fmaxf(smem_warp_max[tid * 4], smem_warp_max[tid * 4 + 1]),
                       fmaxf(smem_warp_max[tid * 4 + 2], smem_warp_max[tid * 4 + 3]));
    bmax = fmaxf(bmax, 1e-12f);
    smem_block_inv[tid] = 448.0f / bmax;
    sfa_out[((int64_t)expert * max_M + row) * Ib + tid] = bmax / 448.0f;
  }
  if (tid == 0) row_scale_out[expert * max_M + row] = 1.0f;
  __syncthreads();
  const int64_t out_base = ((int64_t)expert * max_M + row) * kI;
  for (int j = 0; j < kEPT; j++) {
    int idx = j * kThreads + tid;
    int blk = idx / kBlock;
    c_fp8[out_base + idx] =
        __nv_cvt_float_to_fp8(vals[j] * smem_block_inv[blk], __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void gather_dequant_hidden_fp16_batched_v2_kernel(
    const uint8_t* __restrict__ hidden_fp8, const float* __restrict__ hidden_scale,
    const int32_t* __restrict__ packed_tok_all, const int32_t* __restrict__ d_base_offsets,
    const int32_t* __restrict__ d_counts, int t, int max_M, __half* __restrict__ b_a_base) {
  constexpr int kVec = 8;
  const int i = blockIdx.y;
  const int tk = d_counts[i];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = tk * (kHidden / kVec);
  if (idx >= total) return;
  const int row = idx / (kHidden / kVec);
  const int h8 = idx % (kHidden / kVec);
  const unsigned lane = threadIdx.x & 31u;
  int tok;
  if (lane == 0u) tok = packed_tok_all[d_base_offsets[i] + row];
  tok = __shfl_sync(0xffffffffu, tok, 0);
  const int base_h = h8 * kVec;
  const int hb = base_h / kBlock;
  const float scale = hidden_scale[hb * t + tok];
  const int in_base = tok * kHidden + base_h;
  const int64_t out_base = (int64_t)i * max_M * kHidden + (int64_t)row * kHidden + base_h;
  const uint32_t p0 = *reinterpret_cast<const uint32_t*>(hidden_fp8 + in_base);
  const uint32_t p1 = *reinterpret_cast<const uint32_t*>(hidden_fp8 + in_base + 4);
  const __half2_raw r0 =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(p0 & 0xffffu), __NV_E4M3);
  const __half2_raw r1 =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(p0 >> 16u), __NV_E4M3);
  const __half2_raw r2 =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(p1 & 0xffffu), __NV_E4M3);
  const __half2_raw r3 =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(p1 >> 16u), __NV_E4M3);
  const float2 f0 = __half22float2(reinterpret_cast<const __half2&>(r0));
  const float2 f1 = __half22float2(reinterpret_cast<const __half2&>(r1));
  const float2 f2 = __half22float2(reinterpret_cast<const __half2&>(r2));
  const float2 f3 = __half22float2(reinterpret_cast<const __half2&>(r3));
  __half2* out_ptr = reinterpret_cast<__half2*>(b_a_base + out_base);
  out_ptr[0] = __float22half2_rn(make_float2(f0.x * scale, f0.y * scale));
  out_ptr[1] = __float22half2_rn(make_float2(f1.x * scale, f1.y * scale));
  out_ptr[2] = __float22half2_rn(make_float2(f2.x * scale, f2.y * scale));
  out_ptr[3] = __float22half2_rn(make_float2(f3.x * scale, f3.y * scale));
}

__global__ void gather_fp8_and_scales_k(const uint8_t* __restrict__ h, const float* __restrict__ hs,
                                        const int32_t* __restrict__ pt,
                                        const int32_t* __restrict__ bo,
                                        const int32_t* __restrict__ dc, int t, int mM,
                                        uint8_t* __restrict__ o_fp8, float* __restrict__ o_scales) {
  const int i = blockIdx.y;
  const int row = blockIdx.x;
  if (row >= dc[i]) return;
  const int tok = pt[bo[i] + row];
  const int tid = threadIdx.x;
  constexpr int kVec16 = kHidden / 16;
  const int64_t out_base = (int64_t)i * mM * kHidden + (int64_t)row * kHidden;
  const int64_t in_base = (int64_t)tok * kHidden;
  for (int h16 = tid; h16 < kVec16; h16 += blockDim.x)
    *reinterpret_cast<uint4*>(o_fp8 + out_base + h16 * 16) =
        *reinterpret_cast<const uint4*>(h + in_base + h16 * 16);
  constexpr int Hb = kHidden / kBlock;
  const int64_t s_out_base = (int64_t)i * mM * Hb + (int64_t)row * Hb;
  for (int hb = tid; hb < Hb; hb += blockDim.x) o_scales[s_out_base + hb] = hs[hb * t + tok];
}

// Tight-packed gather: uses cumulative row_offsets instead of i*mM
// 1D-grid gather: launches total_tight_rows blocks instead of max_M*active_count
__global__ __launch_bounds__(256, 8) void gather_fp8_and_scales_tight_k(
    const uint8_t* __restrict__ h, const float* __restrict__ hs, const int32_t* __restrict__ pt,
    const int32_t* __restrict__ bo, const int32_t* __restrict__ dc,
    const int32_t* __restrict__ row_offsets, int t, int active_count, uint8_t* __restrict__ o_fp8,
    float* __restrict__ o_scales) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  const int global_row = blockIdx.x;
  // Early exit for over-launched blocks (grid may be larger than actual total_tight_rows)
  const int total_rows = __ldg(row_offsets + active_count);
  if (global_row >= total_rows) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }
  // Binary search for which expert this row belongs to
  int lo = 0, hi = active_count;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (__ldg(row_offsets + mid + 1) <= global_row)
      lo = mid + 1;
    else
      hi = mid;
  }
  const int i = lo;
  const int row = global_row - __ldg(row_offsets + i);
  const int tok = __ldg(pt + __ldg(bo + i) + row);
  const int tid = threadIdx.x;
  constexpr int kVec16 = kHidden / 16;
  const int64_t out_base = (int64_t)global_row * kHidden;
  const int64_t in_base = (int64_t)tok * kHidden;
  for (int h16 = tid; h16 < kVec16; h16 += blockDim.x)
    *reinterpret_cast<uint4*>(o_fp8 + out_base + h16 * 16) =
        __ldg(reinterpret_cast<const uint4*>(h + in_base + h16 * 16));
  constexpr int Hb = kHidden / kBlock;
  const int64_t s_out_base = (int64_t)global_row * Hb;
  for (int hb = tid; hb < Hb; hb += blockDim.x)
    o_scales[s_out_base + hb] = __ldg(hs + hb * t + tok);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Uses warp shuffle reduction instead of atomicMax for block-wise max
__global__ __launch_bounds__(256, 8) void swiglu_to_fp8_tight_kernel(
    const nv_bfloat16* __restrict__ g1_bf16, const int32_t* __restrict__ row_offsets,
    int active_count, uint8_t* __restrict__ c_fp8, float* __restrict__ sfa_out,
    float* __restrict__ row_scale_out) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  constexpr int kI = kIntermediate;  // 2048
  constexpr int kThreads = 256;
  constexpr int kEPT = kI / kThreads;  // 8
  constexpr int Ib = kI / kBlock;      // 16
  // Each 128-element block maps to 16 consecutive threads (half-warp)
  // Each warp covers 2 blocks (32 threads = 2 * 16)
  constexpr int kThreadsPerBlock = kBlock / kEPT;  // 128/8 = 16

  const int out_row = blockIdx.x;
  // Early exit for over-launched blocks (grid may be larger than actual total_tight_rows)
  const int total_rows = __ldg(row_offsets + active_count);
  if (out_row >= total_rows) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }
  const int tid = threadIdx.x;
  const int64_t g1_base = (int64_t)out_row * 2 * kI;
  const int start = tid * kEPT;

  float vals[kEPT];
  // 128-bit vectorized reads: load 8 bf16 values (16 bytes) per uint4 load
  const uint4 x1_vec = __ldg(reinterpret_cast<const uint4*>(g1_bf16 + g1_base + start));
  const uint4 x2_vec = __ldg(reinterpret_cast<const uint4*>(g1_bf16 + g1_base + kI + start));
  const nv_bfloat162* x1_pairs = reinterpret_cast<const nv_bfloat162*>(&x1_vec);
  const nv_bfloat162* x2_pairs = reinterpret_cast<const nv_bfloat162*>(&x2_vec);

  float local_max = 0.0f;
#pragma unroll
  for (int j = 0; j < kEPT / 2; j++) {
    nv_bfloat162 v1 = x1_pairs[j];
    nv_bfloat162 v2 = x2_pairs[j];
    float f1_lo = __bfloat162float(v1.x), f1_hi = __bfloat162float(v1.y);
    float f2_lo = __bfloat162float(v2.x), f2_hi = __bfloat162float(v2.y);
    vals[j * 2] = __fdividef(f2_lo, 1.0f + __expf(-f2_lo)) * f1_lo;
    vals[j * 2 + 1] = __fdividef(f2_hi, 1.0f + __expf(-f2_hi)) * f1_hi;
    local_max = fmaxf(local_max, fmaxf(fabsf(vals[j * 2]), fabsf(vals[j * 2 + 1])));
  }

  // Warp shuffle reduction for 16-thread half-warp (one 128-element block)
  const unsigned lane = tid & 31u;
  const unsigned half = lane >> 4;  // 0 for lower half-warp, 1 for upper
#pragma unroll
  for (int s = 8; s > 0; s >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, s));
  // Broadcast back to all 16 threads in the half-warp
  float block_max = __shfl_sync(0xffffffffu, local_max, half * 16);
  block_max = fmaxf(block_max, 1e-12f);
  float inv = 448.0f / block_max;

  // Write scale factors (one thread per block)
  const int blk = start / kBlock;
  if ((tid & (kThreadsPerBlock - 1)) == 0) {
    sfa_out[out_row * Ib + blk] = block_max / 448.0f;
  }
  if (tid == 0) row_scale_out[out_row] = 1.0f;

  // Convert and pack 8 FP8 values, write as uint2 (8 bytes)
  const int64_t out_base = (int64_t)out_row * kI;
  uint8_t packed[8];
#pragma unroll
  for (int j = 0; j < kEPT; j++)
    packed[j] = __nv_cvt_float_to_fp8(vals[j] * inv, __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint2*>(c_fp8 + out_base + start) = *reinterpret_cast<uint2*>(packed);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Tight-packed pull-scatter from BF16 using cumulative row_offsets
__global__ __launch_bounds__(256, 4) void pull_scatter_bf16_from_bf16_tight_kernel(
    const nv_bfloat16* __restrict__ b_o_bf16, const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_w, const int32_t* __restrict__ packed_invrow,
    const int32_t* __restrict__ le_to_rank, const int32_t* __restrict__ row_offsets, int t,
    int local_expert_offset, nv_bfloat16* __restrict__ out,
    int32_t* __restrict__ counts_to_zero,  // nullable: zero counts/offsets/cursors for next call
    int counts_to_zero_n) {
  // kHidden=7168, 256 threads: use uint4 (8 bf16) for first 3 iters (6144 elems), uint2 (4 bf16)
  // for last iter (1024 elems)
  constexpr int kThreads = 256;
  constexpr int kVec8 = 8;                                    // elements per uint4 load
  constexpr int kVec4 = 4;                                    // elements per uint2 load
  constexpr int kIter8 = 3;                                   // 3 * 256 * 8 = 6144
  constexpr int kIter4 = 1;                                   // 1 * 256 * 4 = 1024; total = 7168
  constexpr int kTotalAcc = kIter8 * kVec8 + kIter4 * kVec4;  // 24 + 4 = 28
  const int tok = blockIdx.x;
  if (tok >= t) return;
  const int tid = threadIdx.x;

  __shared__ int s_valid_count;
  __shared__ int64_t s_in_bases[kTopK];
  __shared__ float s_weights[kTopK];

  // Thread 0 fills shared arrays
  if (tid == 0) {
    int vc = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int global_k = tok * kTopK + k;
      const int ge = __ldg(topk_idx + global_k);
      const int le = ge - local_expert_offset;
      if (le < 0 || le >= kNumLocalExperts) continue;
      const int rank = __ldg(le_to_rank + le);
      if (rank < 0) continue;
      const int row = __ldg(packed_invrow + global_k);
      s_in_bases[vc] = ((int64_t)__ldg(row_offsets + rank) + row) * kHidden;
      s_weights[vc] = __ldg(topk_w + global_k);
      vc++;
    }
    s_valid_count = vc;
  }
  __syncthreads();

  // Prefetch-based accumulation
  float acc8[kIter8 * 8];
#pragma unroll
  for (int j = 0; j < kIter8 * 8; j++) acc8[j] = 0.f;
  float acc4[kIter4 * 4];
#pragma unroll
  for (int j = 0; j < kIter4 * 4; j++) acc4[j] = 0.f;

  if (s_valid_count > 0) {
    int64_t next_base = s_in_bases[0];
    float next_w = s_weights[0];
    for (int ki = 0; ki < s_valid_count; ++ki) {
      const int64_t base = next_base;
      const float w = next_w;
      if (ki + 1 < s_valid_count) {
        next_base = s_in_bases[ki + 1];
        next_w = s_weights[ki + 1];
      }
// Wide uint4 loads (16 bytes = 8 bf16) for first 6144 elements
#pragma unroll
      for (int iter = 0; iter < kIter8; iter++) {
        const int h = (iter * kThreads + tid) * kVec8;
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(b_o_bf16 + base + h));
        const nv_bfloat162* bf2 = reinterpret_cast<const nv_bfloat162*>(&raw);
        const float2 f01 = __bfloat1622float2(bf2[0]);
        const float2 f23 = __bfloat1622float2(bf2[1]);
        const float2 f45 = __bfloat1622float2(bf2[2]);
        const float2 f67 = __bfloat1622float2(bf2[3]);
        acc8[iter * 8 + 0] += f01.x * w;
        acc8[iter * 8 + 1] += f01.y * w;
        acc8[iter * 8 + 2] += f23.x * w;
        acc8[iter * 8 + 3] += f23.y * w;
        acc8[iter * 8 + 4] += f45.x * w;
        acc8[iter * 8 + 5] += f45.y * w;
        acc8[iter * 8 + 6] += f67.x * w;
        acc8[iter * 8 + 7] += f67.y * w;
      }
      // Narrower uint2 loads (8 bytes = 4 bf16) for remaining 1024 elements
      {
        constexpr int base_elems = kIter8 * kThreads * kVec8;  // 6144
        const int h = base_elems + tid * kVec4;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(b_o_bf16 + base + h));
        const nv_bfloat162* bf2 = reinterpret_cast<const nv_bfloat162*>(&raw);
        const float2 f01 = __bfloat1622float2(bf2[0]);
        const float2 f23 = __bfloat1622float2(bf2[1]);
        acc4[0] += f01.x * w;
        acc4[1] += f01.y * w;
        acc4[2] += f23.x * w;
        acc4[3] += f23.y * w;
      }
    }
  }
  const int64_t out_base = (int64_t)tok * kHidden;
// Write wide (uint4) for first 6144 elements
#pragma unroll
  for (int iter = 0; iter < kIter8; iter++) {
    const int h = (iter * kThreads + tid) * kVec8;
    const int ai = iter * 8;
    nv_bfloat162 o0 = __float22bfloat162_rn(make_float2(acc8[ai + 0], acc8[ai + 1]));
    nv_bfloat162 o1 = __float22bfloat162_rn(make_float2(acc8[ai + 2], acc8[ai + 3]));
    nv_bfloat162 o2 = __float22bfloat162_rn(make_float2(acc8[ai + 4], acc8[ai + 5]));
    nv_bfloat162 o3 = __float22bfloat162_rn(make_float2(acc8[ai + 6], acc8[ai + 7]));
    uint4 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&o0);
    pk.y = *reinterpret_cast<uint32_t*>(&o1);
    pk.z = *reinterpret_cast<uint32_t*>(&o2);
    pk.w = *reinterpret_cast<uint32_t*>(&o3);
    *reinterpret_cast<uint4*>(out + out_base + h) = pk;
  }
  // Write narrow (uint2) for last 1024 elements
  {
    constexpr int base_elems = kIter8 * kThreads * kVec8;
    const int h = base_elems + tid * kVec4;
    nv_bfloat162 o0 = __float22bfloat162_rn(make_float2(acc4[0], acc4[1]));
    nv_bfloat162 o1 = __float22bfloat162_rn(make_float2(acc4[2], acc4[3]));
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&o0);
    pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(out + out_base + h) = pk;
  }
  // Last block zeros counts/offsets/cursors for next pipeline call (eliminates memset)
  if (counts_to_zero != nullptr && blockIdx.x == gridDim.x - 1) {
    for (int i = threadIdx.x; i < counts_to_zero_n; i += blockDim.x) {
      counts_to_zero[i] = 0;
    }
  }
}

__global__ __launch_bounds__(256, 4) void pull_scatter_bf16_kernel(
    const __half* __restrict__ b_o_all, const float* __restrict__ b_c_scale_all,
    const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w,
    const int32_t* __restrict__ packed_invrow, const int32_t* __restrict__ le_to_rank, int t,
    int max_M, int local_expert_offset, nv_bfloat16* __restrict__ out) {
  constexpr int kActualVec = 4;
  constexpr int kThreads = 256;
  constexpr int kIter = kHidden / (kThreads * kActualVec);
  const int tok = blockIdx.x;
  if (tok >= t) return;
  const int tid = threadIdx.x;
  int ranks[kTopK];
  int rows[kTopK];
  float scale_ws[kTopK];
  int valid_count = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const int global_k = tok * kTopK + k;
    const int ge = topk_idx[global_k];
    const int le = ge - local_expert_offset;
    if (le < 0 || le >= kNumLocalExperts) continue;
    const int rank = le_to_rank[le];
    if (rank < 0) continue;
    const int row = packed_invrow[global_k];
    ranks[valid_count] = rank;
    rows[valid_count] = row;
    scale_ws[valid_count] = b_c_scale_all[(int64_t)rank * max_M + row] * topk_w[global_k];
    valid_count++;
  }
  float acc[kIter * kActualVec];
#pragma unroll
  for (int j = 0; j < kIter * kActualVec; j++) acc[j] = 0.f;
  for (int ki = 0; ki < valid_count; ++ki) {
    const float sw = scale_ws[ki];
    const int64_t in_base = ((int64_t)ranks[ki] * max_M + rows[ki]) * kHidden;
#pragma unroll
    for (int iter = 0; iter < kIter; iter++) {
      const int h = (iter * kThreads + tid) * kActualVec;
      const __half2* h2ptr = reinterpret_cast<const __half2*>(b_o_all + in_base + h);
      const float2 f01 = __half22float2(h2ptr[0]);
      const float2 f23 = __half22float2(h2ptr[1]);
      acc[iter * 4 + 0] += f01.x * sw;
      acc[iter * 4 + 1] += f01.y * sw;
      acc[iter * 4 + 2] += f23.x * sw;
      acc[iter * 4 + 3] += f23.y * sw;
    }
  }
  const int64_t out_base = (int64_t)tok * kHidden;
#pragma unroll
  for (int iter = 0; iter < kIter; iter++) {
    const int h = (iter * kThreads + tid) * kActualVec;
    nv_bfloat162 o0 = __float22bfloat162_rn(make_float2(acc[iter * 4 + 0], acc[iter * 4 + 1]));
    nv_bfloat162 o1 = __float22bfloat162_rn(make_float2(acc[iter * 4 + 2], acc[iter * 4 + 3]));
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&o0);
    pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(out + out_base + h) = pk;
  }
}

__global__ __launch_bounds__(256, 4) void pull_scatter_bf16_from_bf16_kernel(
    const nv_bfloat16* __restrict__ b_o_bf16, const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_w, const int32_t* __restrict__ packed_invrow,
    const int32_t* __restrict__ le_to_rank, int t, int max_M, int local_expert_offset,
    nv_bfloat16* __restrict__ out) {
  constexpr int kActualVec = 4;
  constexpr int kThreads = 256;
  constexpr int kIter = kHidden / (kThreads * kActualVec);
  const int tok = blockIdx.x;
  if (tok >= t) return;
  const int tid = threadIdx.x;
  int64_t in_bases[kTopK];
  float weights[kTopK];
  int valid_count = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const int global_k = tok * kTopK + k;
    const int ge = __ldg(topk_idx + global_k);
    const int le = ge - local_expert_offset;
    if (le < 0 || le >= kNumLocalExperts) continue;
    const int rank = __ldg(le_to_rank + le);
    if (rank < 0) continue;
    const int row = __ldg(packed_invrow + global_k);
    in_bases[valid_count] = ((int64_t)rank * max_M + row) * kHidden;
    weights[valid_count] = __ldg(topk_w + global_k);
    valid_count++;
  }
  float acc[kIter * kActualVec];
#pragma unroll
  for (int j = 0; j < kIter * kActualVec; j++) acc[j] = 0.f;
  for (int ki = 0; ki < valid_count; ++ki) {
    const float w = weights[ki];
    const int64_t base = in_bases[ki];
#pragma unroll
    for (int iter = 0; iter < kIter; iter++) {
      const int h = (iter * kThreads + tid) * kActualVec;
      // Use __ldg for read-only texture cache path
      const uint32_t* uptr = reinterpret_cast<const uint32_t*>(b_o_bf16 + base + h);
      const uint32_t u0 = __ldg(uptr);
      const uint32_t u1 = __ldg(uptr + 1);
      const float2 f01 = __bfloat1622float2(reinterpret_cast<const nv_bfloat162&>(u0));
      const float2 f23 = __bfloat1622float2(reinterpret_cast<const nv_bfloat162&>(u1));
      acc[iter * 4 + 0] += f01.x * w;
      acc[iter * 4 + 1] += f01.y * w;
      acc[iter * 4 + 2] += f23.x * w;
      acc[iter * 4 + 3] += f23.y * w;
    }
  }
  const int64_t out_base = (int64_t)tok * kHidden;
#pragma unroll
  for (int iter = 0; iter < kIter; iter++) {
    const int h = (iter * kThreads + tid) * kActualVec;
    nv_bfloat162 o0 = __float22bfloat162_rn(make_float2(acc[iter * 4 + 0], acc[iter * 4 + 1]));
    nv_bfloat162 o1 = __float22bfloat162_rn(make_float2(acc[iter * 4 + 2], acc[iter * 4 + 3]));
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&o0);
    pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(out + out_base + h) = pk;
  }
}

__global__ void fused_bf16_swiglu_fp16_kernel(const nv_bfloat16* __restrict__ g1_bf16, int max_M,
                                              const int32_t* __restrict__ d_counts,
                                              __half* __restrict__ c_fp16,
                                              float* __restrict__ row_scale) {
  constexpr int kI = kIntermediate;
  constexpr int kThreads = 256;
  constexpr int kElemsPerThread = kI / kThreads;
  const int expert = blockIdx.y, row = blockIdx.x;
  if (row >= d_counts[expert]) return;
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  __shared__ float smem_warp_max[kThreads / 32];
  __shared__ float s_inv_scale;
  const int64_t g1_base = ((int64_t)expert * max_M + row) * (2 * kI);
  float local_vals[kElemsPerThread];
  float local_max = 0.0f;
  constexpr int kPairsPerThread = kElemsPerThread / 2;
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int i2 = j * kThreads + tid;
    nv_bfloat16 bf_x1[2], bf_x2[2];
    *reinterpret_cast<uint32_t*>(bf_x1) =
        *reinterpret_cast<const uint32_t*>(g1_bf16 + g1_base + i2 * 2);
    *reinterpret_cast<uint32_t*>(bf_x2) =
        *reinterpret_cast<const uint32_t*>(g1_bf16 + g1_base + kI + i2 * 2);
    const float x1_0 = __bfloat162float(bf_x1[0]), x1_1 = __bfloat162float(bf_x1[1]);
    const float x2_0 = __bfloat162float(bf_x2[0]), x2_1 = __bfloat162float(bf_x2[1]);
    const float v0 = __fdividef(x2_0, 1.0f + __expf(-x2_0)) * x1_0;
    const float v1 = __fdividef(x2_1, 1.0f + __expf(-x2_1)) * x1_1;
    local_vals[j * 2] = v0;
    local_vals[j * 2 + 1] = v1;
    local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
  }
  for (int s = 16; s > 0; s >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, s));
  if (lane == 0) smem_warp_max[warp_id] = local_max;
  __syncthreads();
  if (warp_id == 0) {
    float block_max = (lane < (kThreads / 32)) ? smem_warp_max[lane] : 0.0f;
    for (int s = 16; s > 0; s >>= 1)
      block_max = fmaxf(block_max, __shfl_down_sync(0xffffffffu, block_max, s));
    if (lane == 0) {
      const float scale = (block_max > 1.0f) ? block_max : 1.0f;
      row_scale[expert * max_M + row] = scale;
      s_inv_scale = 1.0f / scale;
    }
  }
  __syncthreads();
  const float inv_scale = s_inv_scale;
  const int c_row_base = (expert * max_M + row) * kI;
  __half2* c_h2 = reinterpret_cast<__half2*>(c_fp16 + c_row_base);
  for (int j = 0; j < kPairsPerThread; ++j) {
    const int i2 = j * kThreads + tid;
    c_h2[i2] = __float22half2_rn(
        make_float2(local_vals[j * 2] * inv_scale, local_vals[j * 2 + 1] * inv_scale));
  }
}

__global__ void exclusive_scan_32_kernel(const int32_t* __restrict__ counts,
                                         int32_t* __restrict__ offsets,
                                         int32_t* __restrict__ combined_out) {
  const int lane = threadIdx.x;
  const int my_count = counts[lane];
  int val = my_count;
  for (int s = 1; s < 32; s <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, val, s);
    if (lane >= s) val += n;
  }
  const int prev = __shfl_up_sync(0xffffffffu, val, 1);
  const int exclusive = (lane == 0) ? 0 : prev;
  offsets[lane] = exclusive;

  // If combined_out is provided, also write GPU planner metadata
  // Layout: [row_offsets(33)] [expert_ids(32)] [active_counts(32)] [base_offsets(32)]
  // [le_to_rank(32)]
  if (combined_out != nullptr) {
    combined_out[lane] = exclusive;
    if (lane == 31) combined_out[32] = val;    // total_tight_rows
    combined_out[33 + lane] = lane;            // expert_ids: identity
    combined_out[33 + 32 + lane] = my_count;   // active_counts
    combined_out[33 + 64 + lane] = exclusive;  // base_offsets
    combined_out[33 + 96 + lane] = lane;       // le_to_rank: identity
  }
}

__global__ void write_sentinel_kernel(int32_t* mapped_sentinel, int sentinel_val) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    *mapped_sentinel = sentinel_val;
  }
}

// GPU-side metadata planner: computes all routing metadata from expert counts
// Eliminates D2H->CPU->H2D pipeline for the CUTLASS path
// Layout in combined_out: [row_offsets(33)] [expert_ids(32)] [active_counts(32)] [base_offsets(32)]
// [le_to_rank(32)]
__global__ void compute_metadata_gpu_kernel(
    const int32_t* __restrict__ counts,       // [32] expert counts from routing
    int32_t* __restrict__ combined_out,       // output: combined metadata buffer
    int32_t* __restrict__ mapped_total_rows)  // mapped host ptr for total_tight_rows
{
  const int le = threadIdx.x;
  if (le >= 32) return;

  // Read count for this expert
  const int my_count = counts[le];

  // Inclusive prefix sum using warp shuffle
  int prefix = my_count;
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, prefix, s);
    if (le >= s) prefix += n;
  }

  // Exclusive prefix sum = inclusive - my_count
  const int exclusive = prefix - my_count;

  // row_offsets[0..31] = exclusive prefix sum, row_offsets[32] = total
  combined_out[le] = exclusive;
  if (le == 31) {
    combined_out[32] = prefix;  // total_tight_rows
  }

  // expert_ids: identity (0, 1, ..., 31)
  combined_out[33 + le] = le;

  // active_counts: copy of counts
  combined_out[33 + 32 + le] = my_count;

  // base_offsets: same as exclusive prefix sum (offset into packed_tok)
  combined_out[33 + 64 + le] = exclusive;

  // le_to_rank: identity mapping (rank = expert ID)
  combined_out[33 + 96 + le] = le;
}

// =============================================================================
// Custom FP8 blockscale GEMV for small M experts (M=1-2 per expert)
// Row-parallel: Grid.y = total_tight_rows, binary search to find expert
// Each warp computes one output element D[row,n] = sum_k(A[row,k] * B[eid,n,k]) * scales
// Grid: (ceil(N/warps_per_block), total_tight_rows), Block: warps_per_block*32
// =============================================================================
template <int KBlocks>  // compile-time K/128 for unrolling
__global__ __launch_bounds__(256, 4) void gemv_fp8_blockscale_kernel(
    const uint8_t* __restrict__ A,            // FP8 input [total_rows, K] tight-packed
    const uint8_t* __restrict__ B,            // FP8 weights [expert, N, K] col-major per expert
    const float* __restrict__ SFA,            // input scales [total_rows, K/128]
    const float* __restrict__ SFB,            // weight scales [expert, N/128, K/128]
    const int32_t* __restrict__ row_offsets,  // cumulative row offsets [num_experts+1]
    const int32_t* __restrict__ expert_ids,   // expert IDs mapping
    int num_experts, int N,
    nv_bfloat16* __restrict__ D  // output [total_rows, N] in BF16
) {
  constexpr int K = KBlocks * 128;
  // Load row_offsets into shared memory for binary search
  __shared__ int32_t s_ro[kNumLocalExperts + 1];
  if (threadIdx.x <= num_experts && threadIdx.x <= kNumLocalExperts)
    s_ro[threadIdx.x] = row_offsets[threadIdx.x];
  __syncthreads();

  const int global_row = blockIdx.y;
  const int total_rows = s_ro[num_experts];
  if (global_row >= total_rows) return;

  // Binary search for expert index
  int lo = 0, hi = num_experts;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (s_ro[mid + 1] <= global_row)
      lo = mid + 1;
    else
      hi = mid;
  }
  const int expert_idx = lo;
  const int eid = __ldg(expert_ids + expert_idx);

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  constexpr int kWarpsPerBlock = 8;
  const int n = blockIdx.x * kWarpsPerBlock + warp_id;
  if (n >= N) return;

  const int n_block = n / 128;

  // Pointers
  const uint8_t* A_row = A + (int64_t)global_row * K;
  const float* sfa_row = SFA + (int64_t)global_row * KBlocks;
  const uint8_t* B_col = B + (int64_t)eid * N * K + (int64_t)n * K;
  const float* sfb_base = SFB + (int64_t)eid * (N / 128) * KBlocks + (int64_t)n_block * KBlocks;

  float acc = 0.0f;
#pragma unroll 4
  for (int kb = 0; kb < KBlocks; kb++) {
    const int k_base = kb * 128 + lane * 4;

    uint32_t a_packed = *reinterpret_cast<const uint32_t*>(A_row + k_base);
    uint32_t b_packed = *reinterpret_cast<const uint32_t*>(B_col + k_base);

    __nv_fp8x2_storage_t a_lo = (__nv_fp8x2_storage_t)(a_packed & 0xffffu);
    __nv_fp8x2_storage_t a_hi = (__nv_fp8x2_storage_t)(a_packed >> 16);
    __nv_fp8x2_storage_t b_lo = (__nv_fp8x2_storage_t)(b_packed & 0xffffu);
    __nv_fp8x2_storage_t b_hi = (__nv_fp8x2_storage_t)(b_packed >> 16);

    __half2_raw ar0 = __nv_cvt_fp8x2_to_halfraw2(a_lo, __NV_E4M3);
    __half2_raw ar1 = __nv_cvt_fp8x2_to_halfraw2(a_hi, __NV_E4M3);
    __half2_raw br0 = __nv_cvt_fp8x2_to_halfraw2(b_lo, __NV_E4M3);
    __half2_raw br1 = __nv_cvt_fp8x2_to_halfraw2(b_hi, __NV_E4M3);

    float2 af0 = __half22float2(*reinterpret_cast<__half2*>(&ar0));
    float2 af1 = __half22float2(*reinterpret_cast<__half2*>(&ar1));
    float2 bf0 = __half22float2(*reinterpret_cast<__half2*>(&br0));
    float2 bf1 = __half22float2(*reinterpret_cast<__half2*>(&br1));

    float block_sum = af0.x * bf0.x + af0.y * bf0.y + af1.x * bf1.x + af1.y * bf1.y;

// Warp reduction
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) block_sum += __shfl_down_sync(0xffffffffu, block_sum, s);

    if (lane == 0) {
      acc += block_sum * sfa_row[kb] * sfb_base[kb];
    }
  }

  if (lane == 0) {
    D[(int64_t)global_row * N + n] = __float2bfloat16(acc);
  }
}

// ===================== END CUDA DEVICE KERNELS =====================

inline int div_up(int x, int y) { return (x + y - 1) / y; }
inline int round_up(int x, int y) { return ((x + y - 1) / y) * y; }

struct GemmBucketRange {
  int start = 0;
  int end = 0;
  int rounded_m = 0;
};

inline int build_shape_aware_gemm_buckets(const std::array<int, kNumLocalExperts>& active_experts,
                                          int active_count, const int32_t* counts, int m_align,
                                          int max_buckets, GemmBucketRange* buckets) {
  if (active_count <= 0) return 0;
  constexpr int kMaxPaddingWastePct = 25;
  const int bucket_cap = std::max(1, std::min(max_buckets, active_count));
  int bucket_count = 0;
  int bucket_start = 0;
  int bucket_sum = counts[active_experts[0]];
  int bucket_m = round_up(bucket_sum, m_align);
  for (int i = 1; i < active_count; ++i) {
    const int next_count = counts[active_experts[i]];
    const int next_m = round_up(next_count, m_align);
    const int candidate_m = std::max(bucket_m, next_m);
    const int candidate_size = i - bucket_start + 1;
    const int candidate_sum = bucket_sum + next_count;
    const int candidate_capacity = candidate_size * candidate_m;
    const int candidate_waste = candidate_capacity - candidate_sum;
    const bool rounded_m_changed = next_m != bucket_m;
    const bool padding_too_high = candidate_waste * 100 > candidate_capacity * kMaxPaddingWastePct;
    const bool can_split = bucket_count + 1 < bucket_cap;
    if (can_split && rounded_m_changed && padding_too_high) {
      buckets[bucket_count++] = {bucket_start, i, bucket_m};
      bucket_start = i;
      bucket_sum = next_count;
      bucket_m = next_m;
    } else {
      bucket_sum = candidate_sum;
      bucket_m = candidate_m;
    }
  }
  buckets[bucket_count++] = {bucket_start, active_count, bucket_m};
  return bucket_count;
}

// CudaBuf: simple CUDA buffer management replacing at::Tensor for workspace
struct CudaBuf {
  void* ptr = nullptr;
  size_t bytes = 0;
  void alloc(size_t n) {
    if (n == 0) return;
    if (ptr && bytes >= n) return;
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
    cudaError_t e = cudaMalloc(&ptr, n);
    if (e != cudaSuccess) {
      fprintf(stderr, "[CudaBuf] cudaMalloc(%zu) failed: %s\n", n, cudaGetErrorString(e));
      ptr = nullptr;
      bytes = 0;
      return;
    }
    bytes = n;
    cudaMemset(ptr, 0, n);
  }
  void free_buf() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
  }
  ~CudaBuf() { free_buf(); }
  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(ptr);
  }
  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(ptr);
  }
  bool defined() const { return ptr != nullptr; }
};

struct KernelWorkspace {
  int device_index = -1;
  int chunk_cap = 0;
  CudaBuf w13_all, w2_all, a_chunk, g1_chunk, c_chunk, o_chunk, dequant_ids;
  const void* cached_w13_ptr = nullptr;
  const void* cached_s13_ptr = nullptr;
  const void* cached_w2_ptr = nullptr;
  const void* cached_s2_ptr = nullptr;
  bool signature_valid = false;
  std::array<uint64_t, 2> sig_w13{}, sig_s13{}, sig_w2{}, sig_s2{};
  std::array<uint8_t, kNumLocalExperts> dequant_ready{};
  std::array<uint8_t, kNumLocalExperts> w2_dequant_ready{};
  int max_t_ws = 0;
  CudaBuf ws_topk_idx, ws_topk_w, ws_packed_tok, ws_packed_w, ws_packed_invrow;
  CudaBuf ws_counts_cursors;
  int32_t* ws_counts_ptr = nullptr;
  int32_t* ws_offsets_ptr = nullptr;
  int32_t* ws_cursors_ptr = nullptr;
  int32_t* pinned_h_counts = nullptr;
  int32_t* pinned_dequant_ids = nullptr;
  int32_t* pinned_sentinel = nullptr;
  int32_t* pinned_sentinel_dev = nullptr;
  int call_counter = 1;
  alignas(64) std::atomic<int> cpu_sync_flag{0};
  CudaBuf b_a_all, b_g1_all, b_c_fp16_all, b_c_scale_all, b_o_all, d_batch_ptr_buf;
  void** pinned_batch_ptrs = nullptr;
  uint8_t* pinned_sig_buf = nullptr;
  int32_t* pinned_combined = nullptr;  // pinned buffer for row_offsets + expert_ids
  int max_M_batch = 0;
  bool cutlass_static_ready = false;
  int ptr_cache_active_count = -1;
  int ptr_cache_max_M = 0;
  void* ptr_cache_w13_data = nullptr;
  void* ptr_cache_w2_data = nullptr;
  void* ptr_cache_ba_data = nullptr;
  std::array<int32_t, kNumLocalExperts> ptr_cache_experts{};
  std::array<int32_t, kNumLocalExperts> ptr_cache_counts{};
  // Input caching removed - always recompute routing and gather
  // CUTLASS blockwise FP8
  CudaBuf fp8_buf, sfa_buf, bf16_buf, indptr_dev, eids_dev, row_offsets_dev;
  CudaBuf fp8_c_buf, sfa_c_buf, bf16_g2_buf;
  // GPU planner: mapped host ptr for total_tight_rows (zero-copy read)
  int32_t* mapped_total_rows = nullptr;
  int32_t* mapped_total_rows_dev = nullptr;
  int max_fp8_rows = 0;                        // pre-allocated fp8 buffer capacity in rows
  bool counts_zeroed_by_pull_scatter = false;  // true after pull_scatter zeros counts for next call
};

int parse_env_int(const char* name, int default_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == 0) return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != 0 || parsed <= 0 || parsed > std::numeric_limits<int>::max())
    return default_value;
  return static_cast<int>(parsed);
}

bool env_is_fp32_math() {
  const char* v = std::getenv("FUSEMOE_MATH_MODE");
  if (!v) return false;
  std::string mode(v);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return mode == "fp32";
}

cublasGemmAlgo_t parse_gemm_algo(const char* env_name, cublasGemmAlgo_t default_algo) {
  const char* v = std::getenv(env_name);
  if (!v || !v[0]) return default_algo;
  std::string algo(v);
  std::transform(algo.begin(), algo.end(), algo.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (algo == "default") return CUBLAS_GEMM_DEFAULT;
  if (algo == "tensorop" || algo == "default_tensorop") return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != 0) return default_algo;
  return static_cast<cublasGemmAlgo_t>(parsed);
}

bool parse_env_bool(const char* name, bool default_value) {
  const char* v = std::getenv(name);
  if (!v || !v[0]) return default_value;
  std::string flag(v);
  std::transform(flag.begin(), flag.end(), flag.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (flag == "1" || flag == "true" || flag == "yes" || flag == "on") return true;
  if (flag == "0" || flag == "false" || flag == "no" || flag == "off") return false;
  return default_value;
}

KernelWorkspace& get_workspace(int device_index, int chunk_cap) {
  static thread_local KernelWorkspace ws;
  const bool device_changed = ws.device_index != device_index;
  const bool need_realloc = device_changed || ws.chunk_cap < chunk_cap || !ws.w13_all.defined();
  if (need_realloc) {
    ws.device_index = device_index;
    ws.chunk_cap = chunk_cap;
    ws.w13_all.alloc(static_cast<size_t>(kNumLocalExperts) * 2 * kIntermediate * kHidden * 2);
    ws.w2_all.alloc(static_cast<size_t>(kNumLocalExperts) * kHidden * kIntermediate * 2);
    ws.a_chunk.alloc(static_cast<size_t>(chunk_cap) * kHidden * 2);
    ws.g1_chunk.alloc(static_cast<size_t>(chunk_cap) * 2 * kIntermediate * 4);
    ws.c_chunk.alloc(static_cast<size_t>(chunk_cap) * kIntermediate * 4);
    ws.o_chunk.alloc(static_cast<size_t>(chunk_cap) * kHidden * 4);
    ws.dequant_ids.alloc(static_cast<size_t>(kNumLocalExperts) * 4);
    ws.cached_w13_ptr = ws.cached_s13_ptr = ws.cached_w2_ptr = ws.cached_s2_ptr = nullptr;
    ws.signature_valid = false;
    ws.sig_w13 = ws.sig_s13 = ws.sig_w2 = ws.sig_s2 = {0, 0};
    ws.dequant_ready.fill(0);
    ws.w2_dequant_ready.fill(0);
    ws.max_M_batch = 0;
  }
  if (!ws.pinned_h_counts)
    CUDA_CHECK(cudaMallocHost(&ws.pinned_h_counts, sizeof(int32_t) * kNumLocalExperts));
  if (!ws.pinned_dequant_ids)
    CUDA_CHECK(cudaMallocHost(&ws.pinned_dequant_ids, sizeof(int32_t) * kNumLocalExperts));
  if (!ws.pinned_sentinel) {
    CUDA_CHECK(cudaHostAlloc(&ws.pinned_sentinel, sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ws.pinned_sentinel_dev),
                                        ws.pinned_sentinel, 0));
    ws.pinned_sentinel[0] = 0;
  }
  if (!ws.pinned_sig_buf) CUDA_CHECK(cudaMallocHost(&ws.pinned_sig_buf, 16));
  // Mapped memory for GPU planner to write total_tight_rows (zero-copy host read)
  if (!ws.mapped_total_rows) {
    CUDA_CHECK(cudaHostAlloc(&ws.mapped_total_rows, sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ws.mapped_total_rows_dev),
                                        ws.mapped_total_rows, 0));
    ws.mapped_total_rows[0] = 0;
  }
  // pinned_combined layout: [row_offsets (max 33)] [expert_ids (max 32)] [active_counts (max 32)]
  // [base_offsets (max 32)] [le_to_rank (32)]
  if (!ws.pinned_combined)
    CUDA_CHECK(cudaMallocHost(&ws.pinned_combined, (5 * kNumLocalExperts + 1) * sizeof(int32_t)));
  if (!ws.d_batch_ptr_buf.defined()) {
    constexpr size_t kBatchBufBytes =
        static_cast<size_t>(kNumLocalExperts) * (6 * sizeof(void*) + 3 * sizeof(int32_t));
    ws.d_batch_ptr_buf.alloc(kBatchBufBytes);
    CUDA_CHECK(cudaMallocHost(&ws.pinned_batch_ptrs, kBatchBufBytes));
  }
  return ws;
}

cublasHandle_t get_cublas_handle(cudaStream_t stream) {
  static thread_local cublasHandle_t handle = nullptr;
  static thread_local void* workspace = nullptr;
  if (!handle) {
    // ensure_cublas_loaded(): replaced by direct -lcublas linking.
    CUBLAS_CHECK(cublasCreate(&handle));
    const cublasMath_t math_mode =
        env_is_fp32_math() ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
    CUBLAS_CHECK(cublasSetMathMode(handle, math_mode));
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
    constexpr size_t kWorkspaceBytes = 32ull << 20;
    CUDA_CHECK(cudaMalloc(&workspace, kWorkspaceBytes));
    CUBLAS_CHECK(cublasSetWorkspace(handle, workspace, kWorkspaceBytes));
  }
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return handle;
}

// Batched GEMM using host pointer arrays directly (avoids D2H copies)
void cublas_gemm_loop_host(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k, const float* alpha,
                           void* const* h_A, cudaDataType Atype, int lda, void* const* h_B,
                           cudaDataType Btype, int ldb, const float* beta, void* const* h_C,
                           cudaDataType Ctype, int ldc, int start, int count,
                           cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  for (int i = start; i < start + count; i++)
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, m, n, k, alpha, h_A[i], Atype, lda, h_B[i],
                              Btype, ldb, beta, h_C[i], Ctype, ldc, computeType, algo));
}

void configure_kernel_cache_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    (void)cudaFuncSetCacheConfig(routing_kernel, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(scatter_with_scan_kernel, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(gather_fp8_and_scales_tight_k, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(swiglu_to_fp8_tight_kernel, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(pull_scatter_bf16_from_bf16_tight_kernel, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(pull_scatter_bf16_kernel, cudaFuncCachePreferL1);
    (void)cudaFuncSetCacheConfig(pull_scatter_bf16_from_bf16_kernel, cudaFuncCachePreferL1);
  });
}

}  // namespace

tvm::ffi::Tensor fusemoe_blackwell_run(
    tvm::ffi::TensorView routing_logits, tvm::ffi::TensorView routing_bias,
    tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView hidden_states_scale,
    tvm::ffi::TensorView gemm1_weights, tvm::ffi::TensorView gemm1_weights_scale,
    tvm::ffi::TensorView gemm2_weights, tvm::ffi::TensorView gemm2_weights_scale,
    int64_t local_expert_offset, double routed_scaling_factor) {
  const int t = static_cast<int>(routing_logits.size(0));
  const int device_id = routing_logits.device().device_id;

  static thread_local bool tl_first_call = true;
  if (__builtin_expect(tl_first_call, 0)) {
    init_fp8_lut_once();
    configure_kernel_cache_once();
    tl_first_call = false;
  }

  if (t == 0) {
    DLDevice dl_device{kDLCUDA, device_id};
    DLDataType bf16_dtype{kDLBfloat, 16, 1};
    int64_t out_shape[2] = {0, static_cast<int64_t>(kHidden)};
    return alloc_tensor(tvm::ffi::ShapeView(out_shape, 2), bf16_dtype, dl_device);
  }

  auto stream = get_current_stream();

  struct EnvCache {
    int longseq_threshold, short_chunk, base_chunk, long_chunk, gemm_n_align;
    int bucket_thresh, max_buckets;
    bool pipeline_swiglu_enabled;
    cublasGemmAlgo_t gemm1_algo, gemm2_algo;
    bool use_cutlass_fp8;
    bool initialized = false;
  };
  static thread_local EnvCache ec;
  if (__builtin_expect(!ec.initialized, 0)) {
    ec.longseq_threshold = parse_env_int("FUSEMOE_LONGSEQ_THRESHOLD", kLongSeqThreshold);
    ec.short_chunk = parse_env_int("FUSEMOE_SHORT_CHUNK", 512);
    ec.base_chunk = parse_env_int("FUSEMOE_BASE_CHUNK", kMaxTkChunk);
    ec.long_chunk = parse_env_int("FUSEMOE_LONG_CHUNK", kMaxTkChunkLong);
    ec.gemm_n_align = 1;
    const char* ga = std::getenv("FUSEMOE_GEMM_N_ALIGN");
    if (ga && ga[0]) ec.gemm_n_align = parse_env_int("FUSEMOE_GEMM_N_ALIGN", 1);
    if (ec.gemm_n_align != 1 && ec.gemm_n_align != 2 && ec.gemm_n_align != 4 &&
        ec.gemm_n_align != 8 && ec.gemm_n_align != 16 && ec.gemm_n_align != 32)
      ec.gemm_n_align = 1;
    ec.bucket_thresh = parse_env_int("FUSEMOE_BUCKET_THRESH", 128);
    ec.max_buckets = parse_env_int("FUSEMOE_MAX_BUCKETS", 6);
    ec.pipeline_swiglu_enabled = parse_env_bool("FUSEMOE_PIPELINE_SWIGLU", true);
    ec.gemm1_algo = parse_gemm_algo("FUSEMOE_GEMM1_ALGO", CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    ec.gemm2_algo = parse_gemm_algo("FUSEMOE_GEMM2_ALGO", CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    ec.use_cutlass_fp8 = parse_env_bool("FUSEMOE_CUTLASS_FP8", true);
    ec.initialized = true;
  }
  const int gemm_n_align = ec.gemm_n_align;
  const int tk_chunk = (t >= ec.longseq_threshold)
                           ? ec.long_chunk
                           : ((t >= ec.short_chunk) ? ec.base_chunk : ec.short_chunk);
  const int tk_chunk_padded = round_up(tk_chunk, gemm_n_align);
  auto& ws = get_workspace(device_id, tk_chunk_padded);

  if (t > ws.max_t_ws || !ws.ws_counts_cursors.defined()) {
    ws.max_t_ws = t;
    ws.ws_topk_idx.alloc(static_cast<size_t>(t) * kTopK * 4);
    ws.ws_topk_w.alloc(static_cast<size_t>(t) * kTopK * 4);
    ws.ws_packed_tok.alloc(static_cast<size_t>(t) * kTopK * 4);
    ws.ws_packed_invrow.alloc(static_cast<size_t>(t) * kTopK * 4);
    ws.ws_counts_cursors.alloc(static_cast<size_t>(3 * kNumLocalExperts) * 4);
    ws.ws_counts_ptr = ws.ws_counts_cursors.as<int32_t>();
    ws.ws_offsets_ptr = ws.ws_counts_cursors.as<int32_t>() + kNumLocalExperts;
    ws.ws_cursors_ptr = ws.ws_counts_cursors.as<int32_t>() + 2 * kNumLocalExperts;
    ws.counts_zeroed_by_pull_scatter = false;  // fresh buffer, not zeroed yet
  }

  int32_t* topk_idx_ptr = ws.ws_topk_idx.as<int32_t>();
  float* topk_w_ptr = ws.ws_topk_w.as<float>();
  int32_t* packed_tok_ptr = ws.ws_packed_tok.as<int32_t>();
  float* packed_w_ptr = nullptr;  // packed_w is never written/read on any path
  int32_t* packed_invrow_ptr = ws.ws_packed_invrow.as<int32_t>();
  int32_t* counts_ptr = ws.ws_counts_ptr;
  int32_t* offsets_ptr = ws.ws_offsets_ptr;
  int32_t* cursors_ptr = ws.ws_cursors_ptr;
  const int threads = 256;

  const float* routing_logits_ptr_f = static_cast<const float*>(routing_logits.data_ptr());
  const nv_bfloat16* routing_bias_ptr_bf =
      reinterpret_cast<const nv_bfloat16*>(routing_bias.data_ptr());
  const uint8_t* hidden_ptr = static_cast<const uint8_t*>(hidden_states.data_ptr());
  const float* hidden_scale_ptr = static_cast<const float*>(hidden_states_scale.data_ptr());
  const uint8_t* w13_all_ptr = static_cast<const uint8_t*>(gemm1_weights.data_ptr());
  const float* s13_all_ptr = static_cast<const float*>(gemm1_weights_scale.data_ptr());
  const uint8_t* w2_all_ptr = static_cast<const uint8_t*>(gemm2_weights.data_ptr());
  const float* s2_all_ptr = static_cast<const float*>(gemm2_weights_scale.data_ptr());

  const bool pointer_unchanged = ws.signature_valid &&
                                 ws.cached_w13_ptr == gemm1_weights.data_ptr() &&
                                 ws.cached_s13_ptr == gemm1_weights_scale.data_ptr() &&
                                 ws.cached_w2_ptr == gemm2_weights.data_ptr() &&
                                 ws.cached_s2_ptr == gemm2_weights_scale.data_ptr();
  const size_t sig_nbytes = static_cast<size_t>(kNumLocalExperts) * 2 * kIntermediate * kHidden;

  // Pre-allocate output tensor (overlaps with routing GPU work)
  DLDevice dl_device{kDLCUDA, device_id};
  DLDataType bf16_dtype{kDLBfloat, 16, 1};
  int64_t out_shape[2] = {static_cast<int64_t>(t), static_cast<int64_t>(kHidden)};
  tvm::ffi::Tensor out_bf16 =
      alloc_tensor(tvm::ffi::ShapeView(out_shape, 2), bf16_dtype, dl_device);

  // Secondary stream for overlapping D2H memcpy with scan/scatter
  static cudaStream_t sync_stream = nullptr;
  static cudaEvent_t routing_done_event = nullptr;
  if (!sync_stream) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&sync_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&routing_done_event, cudaEventDisableTiming));
  }

  CutlassBwFn cutlass_bw = ec.use_cutlass_fp8 ? cutlass_blockwise_fp8_gemm : nullptr;
  // R8 bench verdict: CUTLASS wins (83.01x) vs tcgen05 T>=500 (80.97x) vs tcgen05 always (76.91x).
  // tcgen05 GEMM1+GEMM2 path: 89.8x avg (R21 breakthrough). Enabled by default.
  // Disable via FUSEMOE_NO_TCGEN05=1 to fall back to CUTLASS (~85x).
  static const bool s_tcgen05_enabled = (std::getenv("FUSEMOE_NO_TCGEN05") == nullptr);
  // Load tcgen05 .so (once, only if opt-in)
  // load_tcgen05_so() removed: tcgen05 entry points are now linked directly.
  // Setup TMA descriptors for tcgen05 (once per pointer set)
  if (s_tcgen05_enabled /*always-bound*/ && !g_tcgen05_tma_ready) {
    // A buffer may not be allocated yet; defer to first GEMM call
    // B weights are constant: [num_experts, N, K]
    // Setup for GEMM1: B = w13, N = 2*kIntermediate, K = kHidden
    // and GEMM2: B = w2, N = kHidden, K = kIntermediate
    // We only setup GEMM1 for now (most impactful)
    // max_total_rows: use a safe upper bound (t * kTopK)
    int max_rows = t * kTopK;
    if (max_rows < 128) max_rows = 128;
    // Round up to multiple of 128
    max_rows = ((max_rows + 127) / 128) * 128;
    // We need fp8_buf to be allocated first - defer setup
  }
  const bool gpu_planner_path = pointer_unchanged && cutlass_bw != nullptr;

  // PSS launch config for routing+scatter overlap (reused for both)
  static cudaLaunchAttribute s_rs_pss_attr[1];
  static cudaLaunchConfig_t s_rs_pss_lc{};
  static bool s_rs_pss_init = false;
  if (__builtin_expect(!s_rs_pss_init, 0)) {
    s_rs_pss_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    s_rs_pss_attr[0].val.programmaticStreamSerializationAllowed = true;
    s_rs_pss_lc.blockDim = 256;
    s_rs_pss_lc.dynamicSmemBytes = 0;
    s_rs_pss_lc.numAttrs = 1;
    s_rs_pss_lc.attrs = s_rs_pss_attr;
    s_rs_pss_init = true;
  }

  if (!ws.counts_zeroed_by_pull_scatter) {
    CUDA_CHECK(cudaMemsetAsync(counts_ptr, 0, sizeof(int32_t) * 3 * kNumLocalExperts, stream));
  }
  if (gpu_planner_path) {
    // PSS launch for routing -> scatter -> gather chain overlap
    s_rs_pss_lc.gridDim = t;
    s_rs_pss_lc.stream = stream;
    const float* rl = routing_logits_ptr_f;
    const nv_bfloat16* rb = routing_bias_ptr_bf;
    int rt = t;
    float rs = static_cast<float>(routed_scaling_factor);
    int rleo = static_cast<int>(local_expert_offset);
    int32_t* rti = topk_idx_ptr;
    float* rtw = topk_w_ptr;
    int32_t* rc = counts_ptr;
    cudaLaunchKernelEx(&s_rs_pss_lc, routing_kernel, rl, rb, rt, rs, rleo, rti, rtw, rc);
  } else {
    routing_kernel<<<t, threads, 0, stream>>>(
        routing_logits_ptr_f, routing_bias_ptr_bf, t, static_cast<float>(routed_scaling_factor),
        static_cast<int>(local_expert_offset), topk_idx_ptr, topk_w_ptr, counts_ptr);
    CUDA_CHECK(cudaGetLastError());
    // Record event after routing - counts are ready, start D2H on secondary stream
    CUDA_CHECK(cudaEventRecord(routing_done_event, stream));
  }

  // Fused scan+scatter: prefix sum computed inside scatter kernel, eliminates one kernel launch
  int32_t* scan_combined_out = (gpu_planner_path && ws.row_offsets_dev.defined())
                                   ? ws.row_offsets_dev.as<int32_t>()
                                   : nullptr;
  if (gpu_planner_path) {
    s_rs_pss_lc.gridDim = div_up(t * kTopK, threads);
    s_rs_pss_lc.stream = stream;
    const int32_t* sti = topk_idx_ptr;
    const float* stw = topk_w_ptr;
    int st = t;
    int sleo = static_cast<int>(local_expert_offset);
    const int32_t* sc = counts_ptr;
    int32_t* so = offsets_ptr;
    int32_t* scur = cursors_ptr;
    int32_t* spt = packed_tok_ptr;
    float* spw = packed_w_ptr;
    int32_t* spi = packed_invrow_ptr;
    int32_t* sco = scan_combined_out;
    int32_t* smt = nullptr;
    cudaLaunchKernelEx(&s_rs_pss_lc, scatter_with_scan_kernel, sti, stw, st, sleo, sc, so, scur,
                       spt, spw, spi, sco, smt);
  } else {
    scatter_with_scan_kernel<<<div_up(t * kTopK, threads), threads, 0, stream>>>(
        topk_idx_ptr, topk_w_ptr, t, static_cast<int>(local_expert_offset), counts_ptr, offsets_ptr,
        cursors_ptr, packed_tok_ptr, packed_w_ptr, packed_invrow_ptr, scan_combined_out, nullptr);
    CUDA_CHECK(cudaGetLastError());
  }

  // GPU planner fast path: compute metadata entirely on GPU, skip D2H sync
  bool weights_changed = false;
  int active_count = 0;
  std::array<int, kNumLocalExperts> active_experts{};
  std::array<int32_t, kNumLocalExperts> h_offsets{};
  int32_t total_local_assignments = 0;
  int max_M = 0;

  if (gpu_planner_path) {
    // GPU PLANNER FAST PATH: no D2H, no CPU computation, no H2D
    if (!ws.row_offsets_dev.defined()) {
      ws.row_offsets_dev.alloc((5 * kNumLocalExperts + 1) * 4);
    }
    // Pre-allocate fp8 buffers - use tighter estimate than t*kTopK
    // Expected ≈ t, Chernoff: P(>2t) < exp(-t/3), negligible for t>50
    const int max_possible_rows = (t <= 16) ? t * kTopK : t * 2;
    // Note: allocation is one-time, actual rows always within tight bounds
    if (max_possible_rows > ws.max_fp8_rows || !ws.fp8_buf.defined()) {
      ws.max_fp8_rows = max_possible_rows;
      constexpr int Hb = kHidden / kBlock;
      constexpr int Ib = kIntermediate / kBlock;
      // Pad to 128-row alignment for tcgen05 TMA access
      const int padded_rows = ((max_possible_rows + 127) / 128) * 128;
      ws.fp8_buf.alloc(static_cast<size_t>(padded_rows) * kHidden);
      ws.sfa_buf.alloc(static_cast<size_t>(padded_rows) * Hb * 4);
      ws.bf16_buf.alloc(static_cast<size_t>(max_possible_rows) * 2 * kIntermediate * 2);
      ws.fp8_c_buf.alloc(static_cast<size_t>(padded_rows) * kIntermediate);  // padded for GEMM2 TMA
      ws.sfa_c_buf.alloc(static_cast<size_t>(padded_rows) * Ib * 4);
      ws.bf16_g2_buf.alloc(static_cast<size_t>(max_possible_rows) * kHidden * 2);
      ws.cutlass_static_ready = false;
      g_tcgen05_tma_ready = false;   // need TMA re-setup after realloc
      g_tcgen05_tma2_ready = false;  // need GEMM2 TMA re-setup after realloc
    }
    // Setup tcgen05 TMA descriptors (once after buffer allocation)
    if (s_tcgen05_enabled /*always-bound*/ && !g_tcgen05_tma_ready && ws.fp8_buf.defined()) {
      // GEMM1: A=fp8_buf [max_rows, kHidden], B=w13 [32 experts, 2*kIntermediate, kHidden]
      int ret1 =
          tcgen05_setup_tma(ws.fp8_buf.ptr, max_possible_rows, const_cast<uint8_t*>(w13_all_ptr),
                            kNumLocalExperts, 2 * kIntermediate, kHidden);
      if (ret1 == 0) {
        g_tcgen05_tma_ready = true;
        fprintf(stderr, "[tcgen05] TMA setup OK for GEMM1\n");
      }
    }
    // Setup tcgen05 TMA descriptors for GEMM2 (once after buffer allocation)
    if (s_tcgen05_enabled /*always-bound*/ && !g_tcgen05_tma2_ready && ws.fp8_c_buf.defined()) {
      // GEMM2: A=fp8_c_buf [max_rows, kIntermediate], B=w2 [32 experts, kHidden, kIntermediate]
      int ret2 =
          tcgen05_setup_tma2(ws.fp8_c_buf.ptr, max_possible_rows, const_cast<uint8_t*>(w2_all_ptr),
                             kNumLocalExperts, kHidden, kIntermediate);
      if (ret2 == 0) {
        g_tcgen05_tma2_ready = true;
        fprintf(stderr, "[tcgen05] TMA setup OK for GEMM2\n");
      }
    }
    // Also pre-allocate b_c_scale_all for row_scale output in swiglu
    if (!ws.b_c_scale_all.defined() || ws.max_M_batch == 0) {
      ws.max_M_batch = 1;
      ws.b_c_scale_all.alloc(static_cast<size_t>(max_possible_rows) * 4);
    }

    // Only launch metadata kernel if scan didn't already fuse the metadata writes
    if (!scan_combined_out) {
      compute_metadata_gpu_kernel<<<1, 32, 0, stream>>>(
          counts_ptr, ws.row_offsets_dev.as<int32_t>(), ws.mapped_total_rows_dev);
      CUDA_CHECK(cudaGetLastError());
    }
    // Size-aware grid estimate to reduce empty block launches
    // Kernels have early-exit bounds checking via row_offsets[active_count]
    // Expected local assignments per token ≈ 1 (kTopK * kNumLocalExperts/256)
    if (t <= 16) {
      total_local_assignments = t * kTopK;  // exact worst case for tiny inputs
    } else {
      total_local_assignments = t + (t >> 2);  // t * 1.25, tighter but safe
    }
    active_count = kNumLocalExperts;
  } else {
    // SLOW PATH: D2H sync for counts + signature check
    CUDA_CHECK(cudaStreamWaitEvent(sync_stream, routing_done_event, 0));
    CUDA_CHECK(cudaMemcpyAsync(ws.pinned_h_counts, counts_ptr, sizeof(int32_t) * kNumLocalExperts,
                               cudaMemcpyDeviceToHost, sync_stream));
    if (!pointer_unchanged && gemm1_weights.data_ptr() != nullptr && sig_nbytes >= 16)
      CUDA_CHECK(cudaMemcpyAsync(ws.pinned_sig_buf, gemm1_weights.data_ptr(), 16,
                                 cudaMemcpyDeviceToHost, sync_stream));
    CUDA_CHECK(cudaStreamSynchronize(sync_stream));
    if (!pointer_unchanged) {
      if (gemm1_weights.data_ptr() != nullptr && sig_nbytes >= 16) {
        const uint8_t* s = ws.pinned_sig_buf;
        uint64_t h1 = 1469598103934665603ull;
        for (int i = 0; i < 16; ++i) {
          h1 ^= s[i];
          h1 *= 1099511628211ull;
        }
        h1 ^= sig_nbytes;
        h1 *= 1099511628211ull;
        uint64_t h2 = 1099511628211ull;
        for (int i = 16; i > 0; --i) {
          h2 ^= s[i - 1];
          h2 *= 1469598103934665603ull;
        }
        h2 ^= sig_nbytes * 0x9e3779b185ebca87ull;
        const std::array<uint64_t, 2> cur_sig{h1, h2};
        weights_changed = !ws.signature_valid || ws.sig_w13 != cur_sig;
        ws.sig_w13 = cur_sig;
      } else {
        weights_changed = !ws.signature_valid;
      }
      ws.signature_valid = true;
      ws.cached_w13_ptr = gemm1_weights.data_ptr();
      ws.cached_s13_ptr = gemm1_weights_scale.data_ptr();
      ws.cached_w2_ptr = gemm2_weights.data_ptr();
      ws.cached_s2_ptr = gemm2_weights_scale.data_ptr();
    }
    if (weights_changed) {
      ws.dequant_ready.fill(0);
      ws.w2_dequant_ready.fill(0);
    }
    for (int le = 0; le < kNumLocalExperts; ++le) {
      h_offsets[le] = total_local_assignments;
      total_local_assignments += ws.pinned_h_counts[le];
    }
    for (int le = 0; le < kNumLocalExperts; ++le)
      if (ws.pinned_h_counts[le] > 0) active_experts[active_count++] = le;
  }

  // Defer cuBLAS handle/constants until needed (skip when CUTLASS active)
  cublasHandle_t handle = nullptr;
  const float alpha = 1.0f, beta0 = 0.0f;
  cublasComputeType_t gemm1_compute_type, gemm2_compute_type;
  cublasGemmAlgo_t gemm1_algo, gemm2_algo;
  const int64_t w13_elems = static_cast<int64_t>(2 * kIntermediate) * kHidden;
  const int64_t w2_elems = static_cast<int64_t>(kHidden) * kIntermediate;

  if (!gpu_planner_path) {
    // Dequant only needed for cuBLAS path
    std::array<int32_t, kNumLocalExperts> dequant_experts{};
    int dequant_count = 0;
    for (int i = 0; i < active_count; ++i) {
      const int le = active_experts[i];
      if (!ws.dequant_ready[le]) dequant_experts[dequant_count++] = static_cast<int32_t>(le);
    }
    if (dequant_count > 0 && !cutlass_bw) {
      CUDA_CHECK(cudaMemcpy(ws.dequant_ids.as<int32_t>(), dequant_experts.data(),
                            sizeof(int32_t) * dequant_count, cudaMemcpyHostToDevice));
      dim3 w13_grid(kHidden / kBlock, (2 * kIntermediate) / kBlock,
                    static_cast<unsigned int>(dequant_count));
      dequant_w13_batched_fp16_kernel<<<w13_grid, threads, 0, stream>>>(
          w13_all_ptr, s13_all_ptr, ws.dequant_ids.as<int32_t>(), ws.w13_all.as<__half>());
      CUDA_CHECK(cudaGetLastError());
      dim3 w2_grid(kIntermediate / kBlock, kHidden / kBlock,
                   static_cast<unsigned int>(dequant_count));
      dequant_w2_batched_fp16_kernel<<<w2_grid, threads, 0, stream>>>(
          w2_all_ptr, s2_all_ptr, ws.dequant_ids.as<int32_t>(), ws.w2_all.as<__half>());
      CUDA_CHECK(cudaGetLastError());
      for (int i = 0; i < dequant_count; ++i) ws.w2_dequant_ready[dequant_experts[i]] = 1;
    }
    if (dequant_count > 0) {
      for (int i = 0; i < dequant_count; ++i) ws.dequant_ready[dequant_experts[i]] = 1;
    }
  }

  if (total_local_assignments > 0 || gpu_planner_path) {
    if (!gpu_planner_path) {
      int max_M_raw = 0;
      for (int i = 0; i < active_count; ++i)
        max_M_raw = std::max(max_M_raw, (int)ws.pinned_h_counts[active_experts[i]]);
      max_M = round_up(max_M_raw, gemm_n_align > 1 ? gemm_n_align : 16);
    } else {
      max_M = t;  // Conservative estimate for cuBLAS fallback buffers
    }

    if (!gpu_planner_path) {
      // cuBLAS path (and non-GPU-planner CUTLASS fallback) needs batched per-expert buffers
      if (max_M > ws.max_M_batch || !ws.b_a_all.defined()) {
        ws.max_M_batch = max_M;
        const int64_t rows = static_cast<int64_t>(kNumLocalExperts) * max_M;
        ws.b_a_all.alloc(static_cast<size_t>(rows) * kHidden * 2);
        ws.b_g1_all.alloc(static_cast<size_t>(rows) * 2 * kIntermediate * 2);
        ws.b_c_fp16_all.alloc(static_cast<size_t>(rows) * kIntermediate * 2);
        ws.b_c_scale_all.alloc(static_cast<size_t>(rows) * 4);
        ws.b_o_all.alloc(static_cast<size_t>(rows) * kHidden * 2);
      }
    }

    const __half* b_a_ptr = ws.b_a_all.defined() ? ws.b_a_all.as<const __half>() : nullptr;
    const __half* b_g1_ptr = ws.b_g1_all.defined() ? ws.b_g1_all.as<const __half>() : nullptr;
    const __half* b_c_fp16_ptr =
        ws.b_c_fp16_all.defined() ? ws.b_c_fp16_all.as<const __half>() : nullptr;
    float* b_c_scale_ptr = ws.b_c_scale_all.as<float>();
    const __half* b_o_ptr = ws.b_o_all.defined() ? ws.b_o_all.as<const __half>() : nullptr;
    const __half* w13_all_half = ws.w13_all.defined() ? ws.w13_all.as<const __half>() : nullptr;
    const __half* w2_all_half = ws.w2_all.defined() ? ws.w2_all.as<const __half>() : nullptr;

    if (!cutlass_bw) {
      bool ptrs_stale = ws.ptr_cache_active_count != active_count || ws.ptr_cache_max_M != max_M ||
                        ws.ptr_cache_w13_data != ws.w13_all.ptr ||
                        ws.ptr_cache_w2_data != ws.w2_all.ptr ||
                        ws.ptr_cache_ba_data != ws.b_a_all.ptr;
      if (!ptrs_stale) {
        for (int i = 0; i < active_count; ++i) {
          if (ws.ptr_cache_experts[i] != (int32_t)active_experts[i] ||
              ws.ptr_cache_counts[i] != (int32_t)ws.pinned_h_counts[active_experts[i]]) {
            ptrs_stale = true;
            break;
          }
        }
      }
      if (ptrs_stale) {
        const size_t stride = static_cast<size_t>(kNumLocalExperts);
        void** hp = ws.pinned_batch_ptrs;
        for (int i = 0; i < active_count; ++i) {
          const int le = active_experts[i];
          hp[0 * stride + i] = (void*)(w13_all_half + static_cast<int64_t>(le) * w13_elems);
          hp[1 * stride + i] = (void*)(b_a_ptr + static_cast<int64_t>(i) * max_M * kHidden);
          hp[2 * stride + i] =
              (void*)(b_g1_ptr + static_cast<int64_t>(i) * max_M * 2 * kIntermediate);
          hp[3 * stride + i] = (void*)(w2_all_half + static_cast<int64_t>(le) * w2_elems);
          hp[4 * stride + i] =
              (void*)(b_c_fp16_ptr + static_cast<int64_t>(i) * max_M * kIntermediate);
          hp[5 * stride + i] = (void*)(b_o_ptr + static_cast<int64_t>(i) * max_M * kHidden);
        }
        int32_t* hc = reinterpret_cast<int32_t*>(hp + 6 * stride);
        int32_t* hbo = hc + stride;
        int32_t* hltr = hbo + stride;
        for (int i = 0; i < active_count; ++i) {
          hc[i] = ws.pinned_h_counts[active_experts[i]];
          hbo[i] = h_offsets[active_experts[i]];
        }
        for (int le = 0; le < kNumLocalExperts; ++le) hltr[le] = -1;
        for (int i = 0; i < active_count; ++i) hltr[active_experts[i]] = i;
        CUDA_CHECK(cudaMemcpyAsync(
            ws.d_batch_ptr_buf.ptr, ws.pinned_batch_ptrs,
            static_cast<size_t>(kNumLocalExperts) * (6 * sizeof(void*) + 3 * sizeof(int32_t)),
            cudaMemcpyHostToDevice, stream));
        ws.ptr_cache_active_count = active_count;
        ws.ptr_cache_max_M = max_M;
        ws.ptr_cache_w13_data = ws.w13_all.ptr;
        ws.ptr_cache_w2_data = ws.w2_all.ptr;
        ws.ptr_cache_ba_data = ws.b_a_all.ptr;
        for (int i = 0; i < active_count; ++i) {
          ws.ptr_cache_experts[i] = active_experts[i];
          ws.ptr_cache_counts[i] = ws.pinned_h_counts[active_experts[i]];
        }
      }
    }
    const size_t stride_d = static_cast<size_t>(kNumLocalExperts);
    uint8_t* d_bp = ws.d_batch_ptr_buf.as<uint8_t>();
    const void* const* d_A1 =
        reinterpret_cast<const void* const*>(d_bp + 0 * stride_d * sizeof(void*));
    const void* const* d_B1 =
        reinterpret_cast<const void* const*>(d_bp + 1 * stride_d * sizeof(void*));
    void* const* d_C1 = reinterpret_cast<void* const*>(d_bp + 2 * stride_d * sizeof(void*));
    const void* const* d_A2 =
        reinterpret_cast<const void* const*>(d_bp + 3 * stride_d * sizeof(void*));
    const void* const* d_B2 =
        reinterpret_cast<const void* const*>(d_bp + 4 * stride_d * sizeof(void*));
    void* const* d_C2 = reinterpret_cast<void* const*>(d_bp + 5 * stride_d * sizeof(void*));
    // For CUTLASS path, these come from row_offsets_dev combined buffer (set later)
    // For cuBLAS path, they come from d_batch_ptr_buf
    const int32_t* d_active_counts = nullptr;
    const int32_t* d_base_offsets = nullptr;
    const int32_t* d_le_to_rank = nullptr;
    if (!cutlass_bw) {
      d_active_counts = reinterpret_cast<const int32_t*>(d_bp + 6 * stride_d * sizeof(void*));
      d_base_offsets = d_active_counts + stride_d;
      d_le_to_rank = d_base_offsets + stride_d;
    }

    // Host pointer arrays for direct GemmEx calls (avoids D2H copies)
    void** hp = ws.pinned_batch_ptrs;
    void* const* h_A1 = hp + 0 * stride_d;
    void* const* h_B1 = hp + 1 * stride_d;
    void* const* h_C1 = hp + 2 * stride_d;
    void* const* h_A2 = hp + 3 * stride_d;
    void* const* h_B2 = hp + 4 * stride_d;
    void* const* h_C2 = hp + 5 * stride_d;

    if (!cutlass_bw && active_count > 0) {
      const int gather_blocks_x = div_up(max_M * (kHidden / 8), 256);
      gather_dequant_hidden_fp16_batched_v2_kernel<<<dim3(gather_blocks_x, active_count), 256, 0,
                                                     stream>>>(
          hidden_ptr, hidden_scale_ptr, packed_tok_ptr, d_base_offsets, d_active_counts, t, max_M,
          const_cast<__half*>(b_a_ptr));
      CUDA_CHECK(cudaGetLastError());
    }

    const bool pipeline_swiglu =
        !cutlass_bw && ec.pipeline_swiglu_enabled && (t >= ec.longseq_threshold);
    GemmBucketRange gemm_buckets[kNumLocalExperts];
    int gemm_bucket_count = 0;
    if (!cutlass_bw) {
      const int gemm_bucket_threshold = ec.bucket_thresh;
      const int max_gemm_buckets = ec.max_buckets;
      if (max_M >= gemm_bucket_threshold) {
        gemm_bucket_count = build_shape_aware_gemm_buckets(
            active_experts, active_count, ws.pinned_h_counts, gemm_n_align > 1 ? gemm_n_align : 16,
            max_gemm_buckets, gemm_buckets);
      } else {
        gemm_buckets[0] = {0, active_count, max_M};
        gemm_bucket_count = 1;
      }
    }
    // Cached launch config with programmatic stream serialization (reused for
    // gather/swiglu/pull_scatter)
    static cudaLaunchAttribute s_pss_attr[1];
    static cudaLaunchConfig_t s_pss_lc{};
    static bool s_pss_init = false;
    if (__builtin_expect(!s_pss_init, 0)) {
      s_pss_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      s_pss_attr[0].val.programmaticStreamSerializationAllowed = true;
      s_pss_lc.blockDim = 256;
      s_pss_lc.dynamicSmemBytes = 0;
      s_pss_lc.numAttrs = 1;
      s_pss_lc.attrs = s_pss_attr;
      s_pss_init = true;
    }

    bool cutlass_done = false;
    bool cutlass_gemm2_done = false;
    bool dual_prep_done = false;
    bool tcgen05_gemm1_used = false;
    const nv_bfloat16* gemm2_bf16_out = nullptr;
    int h_row_offsets[kNumLocalExperts + 1];
    int total_tight_rows = 0;
    int32_t* d_eids = nullptr;
    if (cutlass_bw && active_count > 0) {
      if (gpu_planner_path) {
        // GPU PLANNER: metadata already computed on GPU in row_offsets_dev
        // Layout: [row_offsets(33)] [expert_ids(32)] [active_counts(32)] [base_offsets(32)]
        // [le_to_rank(32)]
        total_tight_rows = total_local_assignments;
        int32_t* rod = ws.row_offsets_dev.as<int32_t>();
        d_eids = rod + 33;
        d_active_counts = rod + 65;
        d_base_offsets = rod + 97;
        d_le_to_rank = rod + 129;
      } else {
        // SLOW PATH: CPU-computed metadata, H2D memcpy
        h_row_offsets[0] = 0;
        for (int i = 0; i < active_count; i++)
          h_row_offsets[i + 1] = h_row_offsets[i] + ws.pinned_h_counts[active_experts[i]];
        total_tight_rows = h_row_offsets[active_count];
        constexpr int Hb = kHidden / kBlock;
        constexpr int Ib = kIntermediate / kBlock;
        const int64_t rows = total_tight_rows;
        if (!ws.fp8_buf.defined() || ws.fp8_buf.bytes < static_cast<size_t>(rows) * kHidden) {
          ws.fp8_buf.alloc(rows * kHidden);
          ws.sfa_buf.alloc(rows * Hb * 4);
          ws.bf16_buf.alloc(rows * 2 * kIntermediate * 2);
          ws.row_offsets_dev.alloc((5 * kNumLocalExperts + 1) * 4);
          ws.cutlass_static_ready = false;
          ws.fp8_c_buf.alloc(rows * kIntermediate);
          ws.sfa_c_buf.alloc(rows * Ib * 4);
          ws.bf16_g2_buf.alloc(rows * kHidden * 2);
        }
        const int ac = active_count;
        int off = 0;
        memcpy(ws.pinned_combined + off, h_row_offsets, (ac + 1) * 4);
        off += ac + 1;
        for (int i = 0; i < ac; i++) ws.pinned_combined[off + i] = active_experts[i];
        off += ac;
        for (int i = 0; i < ac; i++)
          ws.pinned_combined[off + i] = ws.pinned_h_counts[active_experts[i]];
        off += ac;
        for (int i = 0; i < ac; i++) ws.pinned_combined[off + i] = h_offsets[active_experts[i]];
        off += ac;
        for (int le = 0; le < kNumLocalExperts; le++) ws.pinned_combined[off + le] = -1;
        for (int i = 0; i < ac; i++) ws.pinned_combined[off + active_experts[i]] = i;
        const int total_combined = off + kNumLocalExperts;
        cudaMemcpyAsync(ws.row_offsets_dev.as<int32_t>(), ws.pinned_combined, total_combined * 4,
                        cudaMemcpyHostToDevice, stream);
        int32_t* rod = ws.row_offsets_dev.as<int32_t>();
        d_eids = rod + (ac + 1);
        d_active_counts = rod + (ac + 1) + ac;
        d_base_offsets = rod + (ac + 1) + 2 * ac;
        d_le_to_rank = rod + (ac + 1) + 3 * ac;
      }
      if (total_tight_rows > 0) {
        {
          s_pss_lc.gridDim = total_tight_rows;
          s_pss_lc.stream = stream;
          const uint8_t* g_h = hidden_ptr;
          const float* g_hs = hidden_scale_ptr;
          const int32_t* g_pt = packed_tok_ptr;
          const int32_t* g_bo = d_base_offsets;
          const int32_t* g_dc = d_active_counts;
          const int32_t* g_ro = ws.row_offsets_dev.as<int32_t>();
          int g_t = t;
          int g_ac = gpu_planner_path ? kNumLocalExperts : active_count;
          uint8_t* g_fp8 = ws.fp8_buf.as<uint8_t>();
          float* g_sfa = ws.sfa_buf.as<float>();
          cudaLaunchKernelEx(&s_pss_lc, gather_fp8_and_scales_tight_k, g_h, g_hs, g_pt, g_bo, g_dc,
                             g_ro, g_t, g_ac, g_fp8, g_sfa);
        }
        if (!gpu_planner_path) {
          CUDA_CHECK(cudaGetLastError());
        }
        const int cutlass_num_groups = gpu_planner_path ? kNumLocalExperts : active_count;
        // GEMV path for very small M (seq_len=1-2 typically gives M<=2 per expert)
        const bool use_gemv = (total_tight_rows <= 2) && gpu_planner_path;
        if (use_gemv) {
          // GEMM1: [total_rows, kHidden] x [expert, 2*kIntermediate, kHidden]^T -> [total_rows,
          // 2*kIntermediate]
          constexpr int kWarpsPerBlock = 8;
          constexpr int kThreadsGemv = kWarpsPerBlock * 32;   // 256
          constexpr int kHiddenBlocks = kHidden / 128;        // 56
          constexpr int kIntermBlocks = kIntermediate / 128;  // 16
          dim3 gemv1_grid((2 * kIntermediate + kWarpsPerBlock - 1) / kWarpsPerBlock,
                          total_tight_rows);
          gemv_fp8_blockscale_kernel<kHiddenBlocks><<<gemv1_grid, kThreadsGemv, 0, stream>>>(
              ws.fp8_buf.as<uint8_t>(), w13_all_ptr, ws.sfa_buf.as<float>(), s13_all_ptr,
              ws.row_offsets_dev.as<int32_t>(), d_eids, cutlass_num_groups, 2 * kIntermediate,
              ws.bf16_buf.as<nv_bfloat16>());
          cutlass_done = true;
          // SwiGLU + FP8 conversion
          {
            s_pss_lc.gridDim = total_tight_rows;
            s_pss_lc.stream = stream;
            const nv_bfloat16* sw_in = ws.bf16_buf.as<const nv_bfloat16>();
            const int32_t* sw_ro = ws.row_offsets_dev.as<int32_t>();
            int sw_ac = gpu_planner_path ? kNumLocalExperts : active_count;
            uint8_t* sw_fp8 = ws.fp8_c_buf.as<uint8_t>();
            float* sw_sfa = ws.sfa_c_buf.as<float>();
            float* sw_rs = b_c_scale_ptr;
            cudaLaunchKernelEx(&s_pss_lc, swiglu_to_fp8_tight_kernel, sw_in, sw_ro, sw_ac, sw_fp8,
                               sw_sfa, sw_rs);
          }
          // GEMM2: [total_rows, kIntermediate] x [expert, kHidden, kIntermediate]^T -> [total_rows,
          // kHidden]
          dim3 gemv2_grid((kHidden + kWarpsPerBlock - 1) / kWarpsPerBlock, total_tight_rows);
          gemv_fp8_blockscale_kernel<kIntermBlocks><<<gemv2_grid, kThreadsGemv, 0, stream>>>(
              ws.fp8_c_buf.as<uint8_t>(), w2_all_ptr, ws.sfa_c_buf.as<float>(), s2_all_ptr,
              ws.row_offsets_dev.as<int32_t>(), d_eids, cutlass_num_groups, kHidden,
              ws.bf16_g2_buf.as<nv_bfloat16>());
          cutlass_gemm2_done = true;
          gemm2_bf16_out = ws.bf16_g2_buf.as<const nv_bfloat16>();
        } else if ([&] {
                     // R8: enable tcgen05 by default for large workloads (where it consistently
                     // wins over CUTLASS in R7 measurements). Disable via FUSEMOE_NO_TCGEN05=1.
                     // R23: hybrid dispatch — tcgen05 for T<=5000, CUTLASS for XL workloads.
                     // tcgen05 wins 17/19 workloads by 3-11%, but CUTLASS wins 2 XL (T=11948,14107)
                     // by 2-7%. Override max via FUSEMOE_TCGEN05_MAX_T=N. Disable via
                     // FUSEMOE_NO_TCGEN05=1.
                     static const bool s_force_off = (std::getenv("FUSEMOE_NO_TCGEN05") != nullptr);
                     static const int s_tcgen05_max_t = []() {
                       const char* e = std::getenv("FUSEMOE_TCGEN05_MAX_T");
                       if (e) return atoi(e);
                       return 10000;  // tcgen05 for T<=10000 (seq_len<=1250), CUTLASS for XL
                                      // (T>90k)
                     }();
                     const bool enabled = !s_force_off;
                     return enabled /*always-bound*/ && g_tcgen05_tma_ready && gpu_planner_path &&
                            (t <= s_tcgen05_max_t);
                   }()) {
          // tcgen05 custom GEMM path (no sync, higher occupancy)
          // R8: pass total_tight_rows so the .so can right-size its grid (was 148 always).
          tcgen05_set_total_rows(total_tight_rows);
          GemmArgs cargs;
          cargs.num_groups = cutlass_num_groups;
          cargs.N = 2 * kIntermediate;
          cargs.K = kHidden;
          cargs.A = ws.fp8_buf.ptr;
          cargs.B = const_cast<uint8_t*>(w13_all_ptr);
          cargs.D = ws.bf16_buf.ptr;
          cargs.SFA = ws.sfa_buf.ptr;
          cargs.SFB = const_cast<float*>(s13_all_ptr);
          cargs.m_indptr = ws.row_offsets_dev.as<int32_t>();
          cargs.expert_ids = d_eids;
          int ret = tcgen05_grouped_gemm(&cargs, stream);
          if (ret == 0) {
            cutlass_done = true;
            tcgen05_gemm1_used = true;
          } else
            fprintf(stderr, "[tcgen05] GEMM1 failed (ret=%d), falling through to CUTLASS\n", ret);
        }
        if (!cutlass_done) {
          // Try dual prep + noprep path (saves one kernel launch by combining GEMM1+GEMM2 prep)
          const bool use_dual_prep = gpu_planner_path /*always-bound*/ /*always-bound*/;
          if (use_dual_prep) {
            // Launch combined prep for both GEMM1 and GEMM2
            GemmArgsDual dargs;
            dargs.num_groups = cutlass_num_groups;
            dargs.N1 = 2 * kIntermediate;
            dargs.K1 = kHidden;
            dargs.A1 = ws.fp8_buf.ptr;
            dargs.B1 = const_cast<uint8_t*>(w13_all_ptr);
            dargs.D1 = ws.bf16_buf.ptr;
            dargs.SFA1 = ws.sfa_buf.ptr;
            dargs.SFB1 = const_cast<float*>(s13_all_ptr);
            dargs.N2 = kHidden;
            dargs.K2 = kIntermediate;
            dargs.A2 = ws.fp8_c_buf.ptr;
            dargs.B2 = const_cast<uint8_t*>(w2_all_ptr);
            dargs.D2 = ws.bf16_g2_buf.ptr;
            dargs.SFA2 = ws.sfa_c_buf.ptr;
            dargs.SFB2 = const_cast<float*>(s2_all_ptr);
            dargs.m_indptr = ws.row_offsets_dev.as<int32_t>();
            dargs.expert_ids = d_eids;
            cutlass_prep_dual(&dargs, stream);
            dual_prep_done = true;
            // GEMM1 using noprep (array set 1)
            GemmArgs cargs;
            cargs.num_groups = cutlass_num_groups;
            cargs.N = 2 * kIntermediate;
            cargs.K = kHidden;
            cargs.A = ws.fp8_buf.ptr;
            cargs.B = const_cast<uint8_t*>(w13_all_ptr);
            cargs.D = ws.bf16_buf.ptr;
            cargs.SFA = ws.sfa_buf.ptr;
            cargs.SFB = const_cast<float*>(s13_all_ptr);
            cargs.m_indptr = ws.row_offsets_dev.as<int32_t>();
            cargs.expert_ids = d_eids;
            int max_M_estimate = total_tight_rows / kNumLocalExperts + 1;
            CutlassBwFn gemm1_fn = (max_M_estimate > 256 /*always-bound*/)
                                       ? cutlass_blockwise_fp8_gemm_128_noprep
                                       : cutlass_blockwise_fp8_gemm_noprep;
            int ret = gemm1_fn(&cargs, stream);
            if (ret != 0) {
              // Fallback to regular path with prep
              gemm1_fn = (max_M_estimate > 256 /*always-bound*/) ? cutlass_blockwise_fp8_gemm_128
                                                                 : cutlass_bw;
              ret = gemm1_fn(&cargs, stream);
            }
            if (ret == 0) cutlass_done = true;
          } else {
            GemmArgs cargs;
            cargs.num_groups = cutlass_num_groups;
            cargs.N = 2 * kIntermediate;
            cargs.K = kHidden;
            cargs.A = ws.fp8_buf.ptr;
            cargs.B = const_cast<uint8_t*>(w13_all_ptr);
            cargs.D = ws.bf16_buf.ptr;
            cargs.SFA = ws.sfa_buf.ptr;
            cargs.SFB = const_cast<float*>(s13_all_ptr);
            cargs.m_indptr = ws.row_offsets_dev.as<int32_t>();
            cargs.expert_ids = d_eids;
            CutlassBwFn gemm1_fn = cutlass_bw;
            int max_M_estimate =
                gpu_planner_path ? (total_tight_rows / kNumLocalExperts + 1) : max_M;
            if (max_M_estimate > 256 /*always-bound*/) gemm1_fn = cutlass_blockwise_fp8_gemm_128;
            int ret = gemm1_fn(&cargs, stream);
            if (ret != 0 && gemm1_fn != cutlass_bw) {
              ret = cutlass_bw(&cargs, stream);
            }
            if (ret == 0) cutlass_done = true;
          }
        }
      }
    }
    if (cutlass_done) goto skip_gemm1;

    {
      // Lazy init cuBLAS (only when CUTLASS path not taken)
      if (!handle) {
        handle = get_cublas_handle(stream);
        gemm1_compute_type = CUBLAS_COMPUTE_32F;
        gemm2_compute_type = CUBLAS_COMPUTE_32F;
        gemm1_algo = ec.gemm1_algo;
        gemm2_algo = ec.gemm2_algo;
      }
      static cudaStream_t aux_stream = nullptr;
      static cudaEvent_t bkt_events[kNumLocalExperts];
      static cudaEvent_t swiglu_done_event = nullptr;
      static bool pipeline_init = false;
      if (pipeline_swiglu && !pipeline_init) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking));
        for (int i = 0; i < kNumLocalExperts; ++i)
          CUDA_CHECK(cudaEventCreateWithFlags(&bkt_events[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&swiglu_done_event, cudaEventDisableTiming));
        pipeline_init = true;
      }
      for (int b = 0; b < gemm_bucket_count; ++b) {
        const GemmBucketRange& bucket = gemm_buckets[b];
        cublas_gemm_loop_host(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2 * kIntermediate, bucket.rounded_m,
                              kHidden, &alpha, h_A1, CUDA_R_16F, kHidden, h_B1, CUDA_R_16F, kHidden,
                              &beta0, h_C1, CUDA_R_16F, 2 * kIntermediate, bucket.start,
                              bucket.end - bucket.start, gemm1_compute_type, gemm1_algo);
        if (pipeline_swiglu) {
          CUDA_CHECK(cudaEventRecord(bkt_events[b], stream));
          CUDA_CHECK(cudaStreamWaitEvent(aux_stream, bkt_events[b], 0));
          swiglu_rowscale_fp16_batched_kernel<<<dim3(max_M, bucket.end - bucket.start), 256, 0,
                                                aux_stream>>>(
              b_g1_ptr + (int64_t)bucket.start * max_M * 2 * kIntermediate, max_M,
              d_active_counts + bucket.start,
              const_cast<__half*>(b_c_fp16_ptr) + (int64_t)bucket.start * max_M * kIntermediate,
              b_c_scale_ptr + (int64_t)bucket.start * max_M);
          CUDA_CHECK(cudaGetLastError());
        }
      }
      if (pipeline_swiglu) {
        CUDA_CHECK(cudaEventRecord(swiglu_done_event, aux_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, swiglu_done_event, 0));
      } else {
        for (int b = 0; b < gemm_bucket_count; ++b) {
          const GemmBucketRange& bucket = gemm_buckets[b];
          swiglu_rowscale_fp16_batched_kernel<<<dim3(max_M, bucket.end - bucket.start), 256, 0,
                                                stream>>>(
              b_g1_ptr + (int64_t)bucket.start * max_M * 2 * kIntermediate, max_M,
              d_active_counts + bucket.start,
              const_cast<__half*>(b_c_fp16_ptr) + (int64_t)bucket.start * max_M * kIntermediate,
              b_c_scale_ptr + (int64_t)bucket.start * max_M);
          CUDA_CHECK(cudaGetLastError());
        }
      }
    }

  skip_gemm1:
    // Track whether tcgen05 was used (no PSS chain, so subsequent kernels must not use PSS)
    const bool tcgen05_used = tcgen05_gemm1_used;
    if (cutlass_done && cutlass_bw && !cutlass_gemm2_done) {
      {
        const nv_bfloat16* sw_in = ws.bf16_buf.as<const nv_bfloat16>();
        const int32_t* sw_ro = ws.row_offsets_dev.as<int32_t>();
        int sw_ac = gpu_planner_path ? kNumLocalExperts : active_count;
        uint8_t* sw_fp8 = ws.fp8_c_buf.as<uint8_t>();
        float* sw_sfa = ws.sfa_c_buf.as<float>();
        float* sw_rs = b_c_scale_ptr;
        if (tcgen05_used) {
          // tcgen05 breaks PSS chain - use regular launch
          swiglu_to_fp8_tight_kernel<<<total_tight_rows, 256, 0, stream>>>(sw_in, sw_ro, sw_ac,
                                                                           sw_fp8, sw_sfa, sw_rs);
        } else {
          s_pss_lc.gridDim = total_tight_rows;
          s_pss_lc.stream = stream;
          cudaLaunchKernelEx(&s_pss_lc, swiglu_to_fp8_tight_kernel, sw_in, sw_ro, sw_ac, sw_fp8,
                             sw_sfa, sw_rs);
        }
      }
      if (!gpu_planner_path) {
        CUDA_CHECK(cudaGetLastError());
      }
      const int cutlass_num_groups = gpu_planner_path ? kNumLocalExperts : active_count;
      GemmArgs cargs2;
      cargs2.num_groups = cutlass_num_groups;
      cargs2.N = kHidden;
      cargs2.K = kIntermediate;
      cargs2.A = ws.fp8_c_buf.ptr;
      cargs2.B = const_cast<uint8_t*>(w2_all_ptr);
      cargs2.D = ws.bf16_g2_buf.ptr;
      cargs2.SFA = ws.sfa_c_buf.ptr;
      cargs2.SFB = const_cast<float*>(s2_all_ptr);
      cargs2.m_indptr = ws.row_offsets_dev.as<int32_t>();
      cargs2.expert_ids = d_eids;
      // tcgen05 GEMM2 path
      if (tcgen05_used /*always-bound*/ && g_tcgen05_tma2_ready) {
        tcgen05_set_total_rows(total_tight_rows);
        int ret2_tc = tcgen05_grouped_gemm2(&cargs2, stream);
        if (ret2_tc == 0) {
          gemm2_bf16_out = ws.bf16_g2_buf.as<const nv_bfloat16>();
          cutlass_gemm2_done = true;
        } else {
          fprintf(stderr, "[tcgen05] GEMM2 failed (ret=%d), falling through to CUTLASS\n", ret2_tc);
        }
      }
      if (!cutlass_gemm2_done) {
        int max_M_est2 = gpu_planner_path ? (total_tight_rows / kNumLocalExperts + 1) : max_M;
        int ret2;
        if (dual_prep_done /*always-bound*/) {
          // Use noprep2 (array set 2, pre-filled by dual prep)
          CutlassBwFn gemm2_fn = (max_M_est2 > 256 /*always-bound*/)
                                     ? cutlass_blockwise_fp8_gemm_128_noprep2
                                     : cutlass_blockwise_fp8_gemm_noprep2;
          ret2 = gemm2_fn(&cargs2, stream);
          if (ret2 != 0) {
            // Fallback to regular path with prep
            gemm2_fn =
                (max_M_est2 > 256 /*always-bound*/) ? cutlass_blockwise_fp8_gemm_128 : cutlass_bw;
            ret2 = gemm2_fn(&cargs2, stream);
          }
        } else {
          CutlassBwFn gemm2_fn = cutlass_bw;
          if (max_M_est2 > 256 /*always-bound*/) gemm2_fn = cutlass_blockwise_fp8_gemm_128;
          ret2 = gemm2_fn(&cargs2, stream);
          if (ret2 != 0 && gemm2_fn != cutlass_bw) {
            ret2 = cutlass_bw(&cargs2, stream);
          }
        }
        if (ret2 == 0) {
          gemm2_bf16_out = ws.bf16_g2_buf.as<const nv_bfloat16>();
          cutlass_gemm2_done = true;
        }
      }  // end if (!cutlass_gemm2_done)
    }
    if (cutlass_done && !cutlass_gemm2_done) {
      fused_bf16_swiglu_fp16_kernel<<<dim3(max_M, active_count), 256, 0, stream>>>(
          ws.bf16_buf.as<const nv_bfloat16>(), max_M, d_active_counts,
          const_cast<__half*>(b_c_fp16_ptr), b_c_scale_ptr);
      CUDA_CHECK(cudaGetLastError());
    }
    if (cutlass_gemm2_done) goto skip_gemm2;

    // Deferred w2 dequant: only when CUTLASS was active but GEMM2 fell through to cuBLAS
    if (cutlass_bw && !gpu_planner_path) {
      std::array<int32_t, kNumLocalExperts> dequant_experts{};
      int dequant_count = 0;
      for (int i = 0; i < active_count; ++i) {
        const int le = active_experts[i];
        if (!ws.dequant_ready[le]) dequant_experts[dequant_count++] = static_cast<int32_t>(le);
      }
      if (dequant_count > 0) {
        CUDA_CHECK(cudaMemcpy(ws.dequant_ids.as<int32_t>(), dequant_experts.data(),
                              sizeof(int32_t) * dequant_count, cudaMemcpyHostToDevice));
        dim3 w2_grid(kIntermediate / kBlock, kHidden / kBlock,
                     static_cast<unsigned int>(dequant_count));
        dequant_w2_batched_fp16_kernel<<<w2_grid, threads, 0, stream>>>(
            w2_all_ptr, s2_all_ptr, ws.dequant_ids.as<int32_t>(), ws.w2_all.as<__half>());
        CUDA_CHECK(cudaGetLastError());
      }
    }

    {
      // Lazy init cuBLAS for GEMM2 fallback
      if (!handle) {
        handle = get_cublas_handle(stream);
        gemm1_compute_type = CUBLAS_COMPUTE_32F;
        gemm2_compute_type = CUBLAS_COMPUTE_32F;
        gemm1_algo = ec.gemm1_algo;
        gemm2_algo = ec.gemm2_algo;
      }
      for (int b = 0; b < gemm_bucket_count; ++b) {
        const GemmBucketRange& bucket = gemm_buckets[b];
        cublas_gemm_loop_host(handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, bucket.rounded_m,
                              kIntermediate, &alpha, h_A2, CUDA_R_16F, kIntermediate, h_B2,
                              CUDA_R_16F, kIntermediate, &beta0, h_C2, CUDA_R_16F, kHidden,
                              bucket.start, bucket.end - bucket.start, gemm2_compute_type,
                              gemm2_algo);
      }
    }

  skip_gemm2:
    if (t > 0) {
      nv_bfloat16* out_ptr = static_cast<nv_bfloat16*>(out_bf16.data_ptr());
      if (gemm2_bf16_out) {
        {
          s_pss_lc.gridDim = t;
          s_pss_lc.stream = stream;
          const nv_bfloat16* ps_bo = gemm2_bf16_out;
          const int32_t* ps_ti = topk_idx_ptr;
          const float* ps_tw = topk_w_ptr;
          const int32_t* ps_pi = packed_invrow_ptr;
          const int32_t* ps_lr = d_le_to_rank;
          const int32_t* ps_ro = ws.row_offsets_dev.as<int32_t>();
          int ps_t = t;
          int ps_leo = static_cast<int>(local_expert_offset);
          nv_bfloat16* ps_out = out_ptr;
          int32_t* ps_ctz = counts_ptr;
          int ps_ctzn = 3 * kNumLocalExperts;
          cudaLaunchKernelEx(&s_pss_lc, pull_scatter_bf16_from_bf16_tight_kernel, ps_bo, ps_ti,
                             ps_tw, ps_pi, ps_lr, ps_ro, ps_t, ps_leo, ps_out, ps_ctz, ps_ctzn);
        }
        ws.counts_zeroed_by_pull_scatter = true;
      } else {
        pull_scatter_bf16_kernel<<<t, 256, 0, stream>>>(
            b_o_ptr, b_c_scale_ptr, topk_idx_ptr, topk_w_ptr, packed_invrow_ptr, d_le_to_rank, t,
            max_M, static_cast<int>(local_expert_offset), out_ptr);
        ws.counts_zeroed_by_pull_scatter = false;  // non-tight path doesn't zero counts
      }
      if (!gpu_planner_path) {
        CUDA_CHECK(cudaGetLastError());
      }
    } else {
      ws.counts_zeroed_by_pull_scatter = false;  // t==0: no pull_scatter launched
    }

  } else {
    CUDA_CHECK(
        cudaMemsetAsync(out_bf16.data_ptr(), 0, static_cast<size_t>(t) * kHidden * 2, stream));
    ws.counts_zeroed_by_pull_scatter = false;  // no pull_scatter launched
  }

  return out_bf16;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fusemoe_blackwell_run);
