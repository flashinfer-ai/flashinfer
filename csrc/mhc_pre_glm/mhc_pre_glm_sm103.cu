// Three-candidate row dispatcher for GLM mHC PRE on GB300.
// Candidate bodies are unchanged except for removed includes/export,
// namespace isolation, and the one-time-configuration hardening below:
// every cudaFuncSetAttribute/occupancy cache is per-device and published
// only after the CUDA calls succeed, with the lock held through the calls.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

// Per-device, publish-after-success shared-memory attribute opt-in: the
// attribute belongs to (function, device), and a device is recorded as
// configured only after the call succeeds (the lock is held through the
// call, so a concurrent first launch on the same device cannot race it).
static bool mhc_attr_once(const void* func, int smem_bytes, int key_salt) {
  static std::mutex mu;
  static std::set<std::pair<int, const void*>> done;
  (void)key_salt;  // template instantiations already have distinct func pointers
  int dev = 0;
  cudaGetDevice(&dev);
  const std::pair<int, const void*> key(dev, func);
  std::lock_guard<std::mutex> lk(mu);
  if (done.count(key)) return true;
  const cudaError_t rc =
      cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (rc != cudaSuccess) {
    throw std::runtime_error(std::string("mhc_pre attribute opt-in failed: ") +
                             cudaGetErrorString(rc));
  }
  done.insert(key);
  return true;
}
#include <tvm/ffi/extra/c_env_api.h>

#include "compact_small_split.cuh"
#include "compact_t256.cuh"
#include "frontier_small_split.cuh"

// BEGIN_EXACT_BODY_MID 8aca060c6ef731b99e488d5a53c6c6506e26ce0d27ab078b534fb5e8749a156d
namespace mid_impl {
namespace ffi = tvm::ffi;

namespace {

constexpr int kHC = 4;
constexpr int kH = 4096;
constexpr int kK = kHC * kH;
constexpr int kNF = 24;
constexpr float kEps = 1.0e-6f;

__device__ __forceinline__ float rcp_fast(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Algebraically carry the twenty Sinkhorn half-iteration scale factors and
// rescale the matrix once at the end. This preserves every epsilon and the
// required column/row order while avoiding 16 elementwise scales per step.
__device__ __forceinline__ void sink_factored16(float (&m)[16]) {
  float R0 = 1.0f, R1 = 1.0f, R2 = 1.0f, R3 = 1.0f;
  float S0 = ((m[0] + m[4]) + (m[8] + m[12])) + kEps;
  float S1 = ((m[1] + m[5]) + (m[9] + m[13])) + kEps;
  float S2 = ((m[2] + m[6]) + (m[10] + m[14])) + kEps;
  float S3 = ((m[3] + m[7]) + (m[11] + m[15])) + kEps;
#pragma unroll 1
  for (int it = 0; it < 19; ++it) {
    const float e0 = kEps * R0;
    const float e1 = kEps * R1;
    const float e2 = kEps * R2;
    const float e3 = kEps * R3;
    const float T0 = rcp_fast(S0);
    const float T1 = rcp_fast(S1);
    const float T2 = rcp_fast(S2);
    const float T3 = rcp_fast(S3);
    R0 = fmaf(m[1], T1, fmaf(m[0], T0, e0)) + fmaf(m[3], T3, m[2] * T2);
    R1 = fmaf(m[5], T1, fmaf(m[4], T0, e1)) + fmaf(m[7], T3, m[6] * T2);
    R2 = fmaf(m[9], T1, fmaf(m[8], T0, e2)) + fmaf(m[11], T3, m[10] * T2);
    R3 = fmaf(m[13], T1, fmaf(m[12], T0, e3)) + fmaf(m[15], T3, m[14] * T2);
    const float f0 = kEps * S0;
    const float f1 = kEps * S1;
    const float f2 = kEps * S2;
    const float f3 = kEps * S3;
    const float U0 = rcp_fast(R0);
    const float U1 = rcp_fast(R1);
    const float U2 = rcp_fast(R2);
    const float U3 = rcp_fast(R3);
    S0 = fmaf(m[4], U1, fmaf(m[0], U0, f0)) + fmaf(m[12], U3, m[8] * U2);
    S1 = fmaf(m[5], U1, fmaf(m[1], U0, f1)) + fmaf(m[13], U3, m[9] * U2);
    S2 = fmaf(m[6], U1, fmaf(m[2], U0, f2)) + fmaf(m[14], U3, m[10] * U2);
    S3 = fmaf(m[7], U1, fmaf(m[3], U0, f3)) + fmaf(m[15], U3, m[11] * U2);
  }
  const float T0 = rcp_fast(S0);
  const float T1 = rcp_fast(S1);
  const float T2 = rcp_fast(S2);
  const float T3 = rcp_fast(S3);
  const float U0 = rcp_fast(R0);
  const float U1 = rcp_fast(R1);
  const float U2 = rcp_fast(R2);
  const float U3 = rcp_fast(R3);
  m[0] *= U0 * T0;
  m[1] *= U0 * T1;
  m[2] *= U0 * T2;
  m[3] *= U0 * T3;
  m[4] *= U1 * T0;
  m[5] *= U1 * T1;
  m[6] *= U1 * T2;
  m[7] *= U1 * T3;
  m[8] *= U2 * T0;
  m[9] *= U2 * T1;
  m[10] *= U2 * T2;
  m[11] *= U2 * T3;
  m[12] *= U3 * T0;
  m[13] *= U3 * T1;
  m[14] *= U3 * T2;
  m[15] *= U3 * T3;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float sigmoidf_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float cvt_tf32(float x) {
  unsigned bits;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x));
  return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t tc_smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t tc_desc_128b(const void* ptr) {
  const uint64_t addr = tc_smem_addr(ptr);
  return ((addr & 0x3ffffULL) >> 4) | (1ULL << 16)  // K-major leading byte offset = 16 B
         | (64ULL << 32)                            // 8 rows * 128 B, encoded in 16 B units
         | (1ULL << 46)                             // Gen5 descriptor version
         | (2ULL << 61);                            // 128 B swizzle
}

template <int kM>
__device__ __forceinline__ void tc_mma_n24_bf16(uint32_t tmem, uint64_t a_desc, uint64_t b_desc,
                                                int accumulate) {
  static_assert(kM == 64 || kM == 128);
  constexpr uint32_t kIDesc =
      (1u << 4) | (1u << 7) | (1u << 10) | (3u << 17) | (static_cast<uint32_t>(kM / 16) << 24);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %8, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 "
      "[%0], %1, %2, %3, {%4, %5, %6, %7}, p;\n\t"
      "}" ::"r"(tmem),
      "l"(a_desc), "l"(b_desc), "r"(kIDesc), "r"(0u), "r"(0u), "r"(0u), "r"(0u), "r"(accumulate)
      : "memory");
}

__device__ __forceinline__ void tc_wait_mbar(uint32_t addr, int phase) {
  uint32_t ready;
  constexpr uint32_t kTicks = 0x989680u;
  do {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "p, [%1], %2, %3;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(ready)
        : "r"(addr), "r"(phase), "r"(kTicks)
        : "memory");
  } while (!ready);
}

struct SharedState {
  float logits[kNF];
  float sqsum;
  float pre[kHC];
};

__device__ __forceinline__ void run_epilogue(const float* __restrict__ dot, float sqsum,
                                             const float* __restrict__ hc_scale,
                                             const float* __restrict__ hc_base,
                                             float* __restrict__ post_mix, float* __restrict__ comb,
                                             float* __restrict__ pre) {
  const float inv_rms = rsqrtf(sqsum * (1.0f / static_cast<float>(kK)) + kEps);
  float logits[kNF];
#pragma unroll
  for (int i = 0; i < kNF; ++i) logits[i] = dot[i] * inv_rms;

#pragma unroll
  for (int i = 0; i < kHC; ++i) {
    pre[i] = sigmoidf_fast(fmaf(logits[i], hc_scale[0], hc_base[i])) + kEps;
    post_mix[i] = sigmoidf_fast(fmaf(logits[kHC + i], hc_scale[1], hc_base[kHC + i]));
  }

  float c[kHC * kHC];
#pragma unroll
  for (int r = 0; r < kHC; ++r) {
    float z0 = fmaf(logits[8 + r * 4], hc_scale[2], hc_base[8 + r * 4]);
    float z1 = fmaf(logits[9 + r * 4], hc_scale[2], hc_base[9 + r * 4]);
    float z2 = fmaf(logits[10 + r * 4], hc_scale[2], hc_base[10 + r * 4]);
    float z3 = fmaf(logits[11 + r * 4], hc_scale[2], hc_base[11 + r * 4]);
    const float m = fmaxf(fmaxf(z0, z1), fmaxf(z2, z3));
    z0 = __expf(z0 - m);
    z1 = __expf(z1 - m);
    z2 = __expf(z2 - m);
    z3 = __expf(z3 - m);
    const float inv = 1.0f / (z0 + z1 + z2 + z3);
    c[r * 4] = z0 * inv + kEps;
    c[r * 4 + 1] = z1 * inv + kEps;
    c[r * 4 + 2] = z2 * inv + kEps;
    c[r * 4 + 3] = z3 * inv + kEps;
  }

  sink_factored16(c);

#pragma unroll
  for (int i = 0; i < 16; ++i) comb[i] = c[i];
}

__global__ __launch_bounds__(256, 2) void mhc_pre_token_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ post_mix, float* __restrict__ comb, __nv_bfloat16* __restrict__ layer_input,
    int tokens) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int q0 = warp * 3;
  const __nv_bfloat16* x = residual + static_cast<long long>(token) * kK;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float sq = 0.0f;

  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
  const float2* w0 = reinterpret_cast<const float2*>(fn + q0 * kK);
  const float2* w1 = reinterpret_cast<const float2*>(fn + (q0 + 1) * kK);
  const float2* w2 = reinterpret_cast<const float2*>(fn + (q0 + 2) * kK);

#pragma unroll 4
  for (int pair = lane; pair < kK / 2; pair += 32) {
    const float2 xv = __bfloat1622float2(x2[pair]);
    const float2 a = w0[pair];
    const float2 b = w1[pair];
    const float2 c = w2[pair];
    acc0 = fmaf(xv.x, a.x, acc0);
    acc0 = fmaf(xv.y, a.y, acc0);
    acc1 = fmaf(xv.x, b.x, acc1);
    acc1 = fmaf(xv.y, b.y, acc1);
    acc2 = fmaf(xv.x, c.x, acc2);
    acc2 = fmaf(xv.y, c.y, acc2);
    if (warp == 0) {
      sq = fmaf(xv.x, xv.x, sq);
      sq = fmaf(xv.y, xv.y, sq);
    }
  }

  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);
  acc2 = warp_sum(acc2);
  if (warp == 0) sq = warp_sum(sq);

  __shared__ SharedState shared;
  if (lane == 0) {
    shared.logits[q0] = acc0;
    shared.logits[q0 + 1] = acc1;
    shared.logits[q0 + 2] = acc2;
    if (warp == 0) shared.sqsum = sq;
  }
  __syncthreads();

  if (tid == 0) {
    run_epilogue(shared.logits, shared.sqsum, hc_scale, hc_base, post_mix + token * 4,
                 comb + token * 16, shared.pre);
  }
  __syncthreads();

  // Each thread writes two adjacent hidden elements at a time.
  __nv_bfloat162* y2 =
      reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(token) * kH);
#pragma unroll
  for (int pair = tid; pair < kH / 2; pair += 256) {
    const int h2 = pair * 2;
    const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + h2));
    const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + kH + h2));
    const float2 x2v =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + 2 * kH + h2));
    const float2 x3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + 3 * kH + h2));
    float2 out;
    out.x = fmaf(shared.pre[3], x3.x,
                 fmaf(shared.pre[2], x2v.x, fmaf(shared.pre[1], x1.x, shared.pre[0] * x0.x)));
    out.y = fmaf(shared.pre[3], x3.y,
                 fmaf(shared.pre[2], x2v.y, fmaf(shared.pre[1], x1.y, shared.pre[0] * x0.y)));
    y2[pair] = __float22bfloat162_rn(out);
  }
}

constexpr int kCoopSplit = 8;
constexpr int kCoopKSlice = kK / kCoopSplit;

struct CoopSplitShared {
  __align__(16) __nv_bfloat16 x[kCoopKSlice];
  float pre[kHC];
};

// Small-token single-launch path.  Eight ordinary CTAs split K, write
// disjoint projection/sqsum partials into the layer_input destination, then
// grid-sync.  Split 0 reduces those initialized partials and finishes every
// output, avoiding both a second launch and cluster/DSMEM setup.
__global__ __launch_bounds__(256, 1) void mhc_pre_coop_split8_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ post_mix, float* __restrict__ comb, __nv_bfloat16* __restrict__ layer_input,
    int tokens) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  const int block = static_cast<int>(blockIdx.x);
  const int token = block / kCoopSplit;
  const int split = block - token * kCoopSplit;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int k_begin = split * kCoopKSlice;
  const __nv_bfloat16* xg = residual + static_cast<long long>(token) * kK;
  __shared__ CoopSplitShared shared;

  auto* sx2 = reinterpret_cast<__nv_bfloat162*>(shared.x);
  const auto* gx2 = reinterpret_cast<const __nv_bfloat162*>(xg + k_begin);
#pragma unroll
  for (int pair = tid; pair < kCoopKSlice / 2; pair += 256) {
    sx2[pair] = gx2[pair];
  }
  __syncthreads();

  const int q0 = warp * 3;
  const auto* w0 = reinterpret_cast<const float2*>(fn + static_cast<long long>(q0) * kK + k_begin);
  const auto* w1 =
      reinterpret_cast<const float2*>(fn + static_cast<long long>(q0 + 1) * kK + k_begin);
  const auto* w2 =
      reinterpret_cast<const float2*>(fn + static_cast<long long>(q0 + 2) * kK + k_begin);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float sq = 0.0f;
#pragma unroll 4
  for (int pair = lane; pair < kCoopKSlice / 2; pair += 32) {
    const float2 xv = __bfloat1622float2(sx2[pair]);
    const float2 a = w0[pair];
    const float2 b = w1[pair];
    const float2 c = w2[pair];
    acc0 = fmaf(xv.x, a.x, acc0);
    acc0 = fmaf(xv.y, a.y, acc0);
    acc1 = fmaf(xv.x, b.x, acc1);
    acc1 = fmaf(xv.y, b.y, acc1);
    acc2 = fmaf(xv.x, c.x, acc2);
    acc2 = fmaf(xv.y, c.y, acc2);
    if (warp == 0) {
      sq = fmaf(xv.x, xv.x, sq);
      sq = fmaf(xv.y, xv.y, sq);
    }
  }
  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);
  acc2 = warp_sum(acc2);
  if (warp == 0) sq = warp_sum(sq);

  float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
  if (lane == 0) {
    scratch[split * (kNF + 1) + q0] = acc0;
    scratch[split * (kNF + 1) + q0 + 1] = acc1;
    scratch[split * (kNF + 1) + q0 + 2] = acc2;
    if (warp == 0) scratch[split * (kNF + 1) + kNF] = sq;
  }
  grid.sync();

  if (split != 0) return;
  if (tid == 0) {
    float dot[kNF];
#pragma unroll
    for (int n = 0; n < kNF; ++n) dot[n] = 0.0f;
    float sqsum = 0.0f;
#pragma unroll
    for (int s = 0; s < kCoopSplit; ++s) {
#pragma unroll
      for (int n = 0; n < kNF; ++n) {
        dot[n] += scratch[s * (kNF + 1) + n];
      }
      sqsum += scratch[s * (kNF + 1) + kNF];
    }
    run_epilogue(dot, sqsum, hc_scale, hc_base, post_mix + token * 4, comb + token * 16,
                 shared.pre);
  }
  __syncthreads();

  auto* y2 = reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(token) * kH);
#pragma unroll
  for (int pair = tid; pair < kH / 2; pair += 256) {
    const int h2 = pair * 2;
    const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xg + h2));
    const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xg + kH + h2));
    const float2 x2v =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xg + 2 * kH + h2));
    const float2 x3 =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xg + 3 * kH + h2));
    float2 out;
    out.x = fmaf(shared.pre[3], x3.x,
                 fmaf(shared.pre[2], x2v.x, fmaf(shared.pre[1], x1.x, shared.pre[0] * x0.x)));
    out.y = fmaf(shared.pre[3], x3.y,
                 fmaf(shared.pre[2], x2v.y, fmaf(shared.pre[1], x1.y, shared.pre[0] * x0.y)));
    y2[pair] = __float22bfloat162_rn(out);
  }
}

constexpr int kCluster = 8;
constexpr int kHSlice = kH / kCluster;

struct ClusterShared {
  __align__(16) __nv_bfloat16 x[kHC * kHSlice];
  float partial[kNF + 1];
  float pre[kHC];
};

__global__ __cluster_dims__(kCluster, 1, 1) __launch_bounds__(256, 1) void mhc_pre_cluster8_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ post_mix, float* __restrict__ comb, __nv_bfloat16* __restrict__ layer_input,
    int tokens) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int token = static_cast<int>(blockIdx.x) / kCluster;
  if (token >= tokens) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int h0 = rank * kHSlice;
  const __nv_bfloat16* xg = residual + static_cast<long long>(token) * kK;

  __shared__ ClusterShared shared;
  __nv_bfloat162* sx2 = reinterpret_cast<__nv_bfloat162*>(shared.x);
#pragma unroll
  for (int hc = 0; hc < kHC; ++hc) {
    const __nv_bfloat162* gx2 = reinterpret_cast<const __nv_bfloat162*>(xg + hc * kH + h0);
    sx2[hc * (kHSlice / 2) + tid] = gx2[tid];
  }
  __syncthreads();

  const int q0 = warp * 3;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float sq = 0.0f;
#pragma unroll
  for (int hc = 0; hc < kHC; ++hc) {
    const __nv_bfloat162* xs = sx2 + hc * (kHSlice / 2);
    const float2* w0 = reinterpret_cast<const float2*>(fn + q0 * kK + hc * kH + h0);
    const float2* w1 = reinterpret_cast<const float2*>(fn + (q0 + 1) * kK + hc * kH + h0);
    const float2* w2 = reinterpret_cast<const float2*>(fn + (q0 + 2) * kK + hc * kH + h0);
#pragma unroll
    for (int pair = lane; pair < kHSlice / 2; pair += 32) {
      const float2 xv = __bfloat1622float2(xs[pair]);
      const float2 a = w0[pair];
      const float2 b = w1[pair];
      const float2 c = w2[pair];
      acc0 = fmaf(xv.x, a.x, acc0);
      acc0 = fmaf(xv.y, a.y, acc0);
      acc1 = fmaf(xv.x, b.x, acc1);
      acc1 = fmaf(xv.y, b.y, acc1);
      acc2 = fmaf(xv.x, c.x, acc2);
      acc2 = fmaf(xv.y, c.y, acc2);
      if (warp == 0) {
        sq = fmaf(xv.x, xv.x, sq);
        sq = fmaf(xv.y, xv.y, sq);
      }
    }
  }

  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);
  acc2 = warp_sum(acc2);
  if (warp == 0) sq = warp_sum(sq);
  if (lane == 0) {
    shared.partial[q0] = acc0;
    shared.partial[q0 + 1] = acc1;
    shared.partial[q0 + 2] = acc2;
    if (warp == 0) shared.partial[kNF] = sq;
  }
  cluster.sync();

  if (rank == 0 && tid == 0) {
    float dot[kNF];
#pragma unroll
    for (int j = 0; j < kNF; ++j) dot[j] = 0.0f;
    float sqsum = 0.0f;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) {
      float* p = cluster.map_shared_rank(shared.partial, r);
#pragma unroll
      for (int j = 0; j < kNF; ++j) dot[j] += p[j];
      sqsum += p[kNF];
    }
    run_epilogue(dot, sqsum, hc_scale, hc_base, post_mix + token * 4, comb + token * 16,
                 shared.pre);
  }
  cluster.sync();

  const float* pre = cluster.map_shared_rank(shared.pre, 0);
  const float2 x0 = __bfloat1622float2(sx2[tid]);
  const float2 x1 = __bfloat1622float2(sx2[kHSlice / 2 + tid]);
  const float2 x2v = __bfloat1622float2(sx2[kHSlice + tid]);
  const float2 x3 = __bfloat1622float2(sx2[3 * kHSlice / 2 + tid]);
  float2 out;
  out.x = fmaf(pre[3], x3.x, fmaf(pre[2], x2v.x, fmaf(pre[1], x1.x, pre[0] * x0.x)));
  out.y = fmaf(pre[3], x3.y, fmaf(pre[2], x2v.y, fmaf(pre[1], x1.y, pre[0] * x0.y)));
  __nv_bfloat162* yg2 =
      reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(token) * kH + h0);
  yg2[tid] = __float22bfloat162_rn(out);
  // Rank 0 owns the DSMEM pre-gate buffer; keep it alive until every peer
  // has consumed the values and completed its disjoint layer-input store.
  cluster.sync();
}

// TF32 projection path for the token-parallel regime.  Each warp owns one
// 16x16 output tile.  The physical fn layout [24,K] is exactly the column-
// major representation of logical B [K,24].  We pad N to 32 only in shared
// memory and retain fp32 accumulators throughout.
template <int MTILE, int KTILE>
struct __align__(32) WmmaShared {
  float a[MTILE * KTILE];
  union {
    float b[32 * KTILE];
    float out[MTILE * 32];
  } workspace;
};

template <int MTILE>
__global__ __launch_bounds__(MTILE * 4, 1) void mhc_project_tf32_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  namespace wmma = nvcuda::wmma;
  constexpr int kKTile = 128;
  constexpr int kThreads = MTILE * 4;
  static_assert(MTILE == 16 || MTILE == 32 || MTILE == 64);
  static_assert(kK % kKTile == 0);

  __shared__ WmmaShared<MTILE, kKTile> shared;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int row = tid >> 2;
  const int subgroup_lane = tid & 3;
  const int token0 = static_cast<int>(blockIdx.x) * MTILE;
  const int token = token0 + row;
  const bool valid = token < tokens;

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  float sq = 0.0f;
#pragma unroll 1
  for (int k0 = 0; k0 < kK; k0 += kKTile) {
    // Four adjacent threads own one token row.  At each iteration they load
    // four adjacent bf16 pairs, while preserving a row-local sqsum partial.
#pragma unroll
    for (int pair = subgroup_lane; pair < kKTile / 2; pair += 4) {
      float2 xv;
      if (valid) {
        const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(
            residual + static_cast<long long>(token) * kK + k0);
        xv = __bfloat1622float2(x2[pair]);
        sq = fmaf(xv.x, xv.x, sq);
        sq = fmaf(xv.y, xv.y, sq);
      } else {
        xv = make_float2(0.0f, 0.0f);
      }
      const int kk = pair * 2;
      shared.a[row * kKTile + kk] = cvt_tf32(xv.x);
      shared.a[row * kKTile + kk + 1] = cvt_tf32(xv.y);
    }

    // Shared B is physically [N,K], which load_matrix_sync consumes as
    // logical column-major [K,N] with ldm=KTILE.
    for (int i = tid; i < 32 * kKTile; i += kThreads) {
      const int n = i / kKTile;
      const int kk = i - n * kKTile;
      const float v = n < kNF ? fn[n * kK + k0 + kk] : 0.0f;
      shared.workspace.b[i] = cvt_tf32(v);
    }
    __syncthreads();

    const int m_frag = warp >> 1;
    const int n_frag = warp & 1;
#pragma unroll
    for (int kk = 0; kk < kKTile; kk += 8) {
      wmma::load_matrix_sync(a_frag, shared.a + m_frag * 16 * kKTile + kk, kKTile);
      wmma::load_matrix_sync(b_frag, shared.workspace.b + n_frag * 16 * kKTile + kk, kKTile);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  // The B staging buffer is dead now, so reuse it for the row-major output.
  const int m_frag = warp >> 1;
  const int n_frag = warp & 1;
  wmma::store_matrix_sync(shared.workspace.out + m_frag * 16 * 32 + n_frag * 16, acc_frag, 32,
                          wmma::mem_row_major);
  __syncthreads();

  float sqsum = sq + __shfl_xor_sync(0xffffffffu, sq, 1);
  sqsum += __shfl_xor_sync(0xffffffffu, sqsum, 2);
  if (valid) {
    // The header lives inside this token's own output row.  The epilogue CTA
    // loads it before overwriting the row, so token CTAs cannot race.
    float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
#pragma unroll
    for (int n = subgroup_lane; n < kNF; n += 4) {
      scratch[n] = shared.workspace.out[row * 32 + n];
    }
    if (subgroup_lane == 0) scratch[kNF] = sqsum;
  }
}

struct __align__(32) WmmaWarp16Shared {
  float a[2][16 * 8];
  float b[2][16 * 8];
  float out[2][16 * 16];
};

template <int kSplitK>
__global__ __launch_bounds__(64, 2) void mhc_project_tf32_warp16_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  namespace wmma = nvcuda::wmma;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token0 = static_cast<int>(blockIdx.x) * 16;
  constexpr int kSplitSize = kK / kSplitK;
  static_assert(kK % kSplitK == 0);
  const int split = static_cast<int>(blockIdx.y);
  const int k_begin = split * kSplitSize;
  const int k_end = k_begin + kSplitSize;
  const int n0 = warp * 16;
  __shared__ WmmaWarp16Shared shared;

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Warp 0's two-lane groups accumulate the exact fp32 square sum for one
  // token each while both warps independently stage their MMA operands.
  const int row = lane >> 1;
  const int sub = lane & 1;
  const int token = token0 + row;
  const bool valid = token < tokens;
  float sq = 0.0f;

#pragma unroll 1
  for (int k0 = k_begin; k0 < k_end; k0 += 8) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = sub + 2 * j;
      float x = 0.0f;
      if (valid) {
        x = __bfloat162float(residual[static_cast<long long>(token) * kK + k0 + kk]);
        if (warp == 0) sq = fmaf(x, x, sq);
      }
      shared.a[warp][row * 8 + kk] = cvt_tf32(x);
    }
#pragma unroll
    for (int i = lane; i < 16 * 8; i += 32) {
      const int n = i >> 3;
      const int kk = i & 7;
      const float v = n0 + n < kNF ? fn[(n0 + n) * kK + k0 + kk] : 0.0f;
      shared.b[warp][i] = cvt_tf32(v);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_frag, shared.a[warp], 8);
    wmma::load_matrix_sync(b_frag, shared.b[warp], 8);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(shared.out[warp], acc_frag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 16 * 16; i += 32) {
    const int r = i >> 4;
    const int n = i & 15;
    const int t = token0 + r;
    if (t < tokens && n0 + n < kNF) {
      float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(t) * kH);
      scratch[split * (kNF + 1) + n0 + n] = shared.out[warp][i];
    }
  }
  if (warp == 0) {
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    if (sub == 0 && valid) {
      float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
      scratch[split * (kNF + 1) + kNF] = sq;
    }
  }
}

template <int kKTile, int kLd>
struct __align__(32) Bf16WmmaShared {
  __nv_bfloat16 a[64 * kLd];
  union {
    __nv_bfloat16 b[32 * kLd];
    float out[64 * 32];
  } workspace;
};

template <int kSplitK>
__global__ __launch_bounds__(256, 3) void mhc_project_bf16_m64_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  namespace wmma = nvcuda::wmma;
  constexpr int kKTile = 128;
  constexpr int kLd = 136;
  constexpr int kSplitSize = kK / kSplitK;
  static_assert(kK % kSplitK == 0 && kSplitSize % kKTile == 0);
  __shared__ Bf16WmmaShared<kKTile, kLd> shared;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token0 = static_cast<int>(blockIdx.x) * 64;
  const int split = static_cast<int>(blockIdx.y);
  const int k_begin = split * kSplitSize;
  const int k_end = k_begin + kSplitSize;

  wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 32, 16, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);
  float sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
  for (int k0 = k_begin; k0 < k_end; k0 += kKTile) {
    for (int i = lane; i < 8 * (kKTile / 2); i += 32) {
      const int r = i / (kKTile / 2);
      const int pair = i - r * (kKTile / 2);
      const int token = token0 + warp * 8 + r;
      __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
      if (token < tokens) {
        const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(
            residual + static_cast<long long>(token) * kK + k0);
        packed = x2[pair];
        const float2 xv = __bfloat1622float2(packed);
        sq[r] = fmaf(xv.x, xv.x, sq[r]);
        sq[r] = fmaf(xv.y, xv.y, sq[r]);
      }
      auto* sa2 = reinterpret_cast<__nv_bfloat162*>(shared.a);
      sa2[(warp * 8 + r) * (kLd / 2) + pair] = packed;
    }

    auto* sb2 = reinterpret_cast<__nv_bfloat162*>(shared.workspace.b);
    for (int pair = tid; pair < 32 * (kKTile / 2); pair += 256) {
      const int n = pair / (kKTile / 2);
      const int kp = pair - n * (kKTile / 2);
      __nv_bfloat162 packed = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
      if (n < kNF) {
        const float2 v = reinterpret_cast<const float2*>(fn + n * kK + k0)[kp];
        packed = __float22bfloat162_rn(v);
      }
      sb2[n * (kLd / 2) + kp] = packed;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kKTile; kk += 16) {
      wmma::load_matrix_sync(a_frag, shared.a + warp * 8 * kLd + kk, kLd);
      wmma::load_matrix_sync(b_frag, shared.workspace.b + kk, kLd);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(shared.workspace.out + warp * 8 * 32, acc_frag, 32, wmma::mem_row_major);
  __syncthreads();

  for (int i = lane; i < 8 * kNF; i += 32) {
    const int r = i / kNF;
    const int n = i - r * kNF;
    const int token = token0 + warp * 8 + r;
    if (token < tokens) {
      float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
      scratch[split * (kNF + 1) + n] = shared.workspace.out[(warp * 8 + r) * 32 + n];
    }
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const float total = warp_sum(sq[r]);
    const int token = token0 + warp * 8 + r;
    if (lane == 0 && token < tokens) {
      float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
      scratch[split * (kNF + 1) + kNF] = total;
    }
  }
}

// Native Blackwell tensor-core projection.  A CTA owns one M tile and one
// split-K slice.  Multiple K64 blocks share a stage so their asynchronous
// tcgen05 MMAs are covered by one completion barrier.
constexpr int kPackedWeightRowElems = 512;
constexpr int kPackedWeightStart = kH - kPackedWeightRowElems;

__device__ __forceinline__ float2 load_global_evict_last_float2(const float2* ptr,
                                                                uint64_t policy) {
  union {
    uint64_t packed;
    float2 value;
  } result;
  asm volatile("ld.global.L2::cache_hint.b64 %0, [%1], %2;"
               : "=l"(result.packed)
               : "l"(ptr), "l"(policy)
               : "memory");
  return result.value;
}

__device__ __forceinline__ float4 load_global_evict_last_float4(const float4* ptr,
                                                                uint64_t policy) {
  union {
    uint4 packed;
    float4 value;
  } result;
  asm volatile("ld.global.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(result.packed.x), "=r"(result.packed.y), "=r"(result.packed.z),
                 "=r"(result.packed.w)
               : "l"(ptr), "l"(policy)
               : "memory");
  return result.value;
}

__global__ __launch_bounds__(256,
                             2) void pack_fn_bf16_kernel(const float* __restrict__ fn,
                                                         __nv_bfloat16* __restrict__ layer_input) {
  constexpr int kPairs = kNF * kK / 2;
  const int global_tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = static_cast<int>(gridDim.x) * blockDim.x;
  const auto* fn2 = reinterpret_cast<const float2*>(fn);
  for (int pair = global_tid; pair < kPairs; pair += stride) {
    const int scalar = pair * 2;
    const int row = scalar / kPackedWeightRowElems;
    const int col = scalar - row * kPackedWeightRowElems;
    *reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(row) * kH +
                                       kPackedWeightStart + col) = __float22bfloat162_rn(fn2[pair]);
  }
}

__device__ __forceinline__ uint32_t pack_weight_bf16x2(float lo, float hi) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi), "f"(lo));
  return packed;
}

__device__ __forceinline__ void tc_ld_x8(uint32_t (&out)[8], uint32_t addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]),
        "=r"(out[6]), "=r"(out[7])
      : "r"(addr)
      : "memory");
}

__device__ __forceinline__ void tc_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <int kSplitK, int kScratchPitch>
__device__ __forceinline__ constexpr int scratch_offset(int split, int lane) {
  if constexpr (kScratchPitch == kNF) {
    return lane < kNF ? split * kNF + lane : kSplitK * kNF + split;
  } else {
    return split * kScratchPitch + lane;
  }
}

// Multi-stage pipelined projection.  The original tcgen05 projection waits on
// its own cp.async group every k stage, so the full HBM latency of each 16 KB
// A tile is exposed; NCU reported ~45% long-scoreboard for it.  Here kStages
// A buffers rotate so stage i's MMA overlaps stages i+1..i+kStages-1 in
// flight, and the FP32 weight slice for stage i+1 is already in registers.
inline bool encode_residual_map(CUtensorMap* map, const void* res, int tokens, unsigned box_rows) {
  const cuuint64_t gdim[2] = {static_cast<cuuint64_t>(kK), static_cast<cuuint64_t>(tokens)};
  const cuuint64_t gstride[1] = {static_cast<cuuint64_t>(kK) * 2u};
  const cuuint32_t bdim[2] = {64u, box_rows};
  const cuuint32_t estride[2] = {1u, 1u};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(res),
                                gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

__device__ __forceinline__ void sq_acc_f32x2(uint64_t& acc, uint32_t value) {
  uint64_t widened;
  asm("{\n\t.reg .b32 lo, hi;\n\t"
      "shl.b32 lo, %1, 16;\n\t"
      "and.b32 hi, %1, -65536;\n\t"
      "mov.b64 %0, {lo, hi};\n\t}"
      : "=l"(widened)
      : "r"(value));
  asm("fma.rn.f32x2 %0, %1, %1, %0;" : "+l"(acc) : "l"(widened));
}

template <int kSplitK, int kM, int kStages, int kThreads, int kScratchPitch = 32,
          bool kFloat4Weights = false, bool kProgrammaticTrigger = false>
__global__ __launch_bounds__(kThreads) void mhc_project_tma_kernel(
    const __grid_constant__ CUtensorMap res_map, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input, int split_base) {
  constexpr int kN = 24;
  static_assert(kM == 64 || kM == 128);
  static_assert(kThreads == 128 || kThreads == 256);
  constexpr int kWarps = kThreads / 32;
  constexpr int kRowBytes = 128;
  constexpr int kABlockBytes = kM * kRowBytes;
  constexpr int kBBlockBytes = kN * kRowBytes;
  constexpr int kSplitSize = kK / kSplitK;
  constexpr int kIters = kSplitSize / 64;
  static_assert(kIters >= kStages);
  constexpr int kTxBytes = kM * kRowBytes;
  constexpr int kWItems = kN * 32;
  static_assert(kWItems % kThreads == 0);
  constexpr int kWPerThread = kWItems / kThreads;
  constexpr int kWQuads = kN * 16;
  constexpr int kWQuadsPerThread = (kWQuads + kThreads - 1) / kThreads;

  extern __shared__ __align__(1024) unsigned char tma_smem[];
  unsigned char* const sA = tma_smem;
  unsigned char* const sB = tma_smem + kStages * kABlockBytes;
  __shared__ __align__(8) uint64_t tile_bar[kStages];
  __shared__ __align__(8) uint64_t mma_bar;
  __shared__ uint32_t tmem_slot;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token0 = static_cast<int>(blockIdx.x) * kM;
  const int split = static_cast<int>(blockIdx.y) + split_base;
  const int k_begin = split * kSplitSize;
  const uint32_t mma_bar_addr = tc_smem_addr(&mma_bar);
  const uint32_t tmem_slot_addr = tc_smem_addr(&tmem_slot);

  int w_n[kWPerThread];
  int w_pair[kWPerThread];
#pragma unroll
  for (int q = 0; q < kWPerThread; ++q) {
    const int item = tid + q * kThreads;
    w_n[q] = item >> 5;
    w_pair[q] = item & 31;
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tc_smem_addr(&tile_bar[s]))
                   : "memory");
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mma_bar_addr) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(tmem_slot_addr),
        "r"(32)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }

  auto issue_tile = [&](int stage, int k0) {
    if (tid == 0) {
      const uint32_t bar = tc_smem_addr(&tile_bar[stage]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                   "n"(kTxBytes)
                   : "memory");
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global"
          ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
              tc_smem_addr(sA + stage * kABlockBytes)),
          "l"(reinterpret_cast<uint64_t>(&res_map)), "r"(k0), "r"(token0), "r"(bar)
          : "memory");
    }
  };

#pragma unroll
  for (int s = 0; s < kStages; ++s) issue_tile(s, k_begin + s * 64);

  float2 wreg[kWPerThread];
  float4 wreg4[kWQuadsPerThread];
  if constexpr (kFloat4Weights) {
#pragma unroll
    for (int q = 0; q < kWQuadsPerThread; ++q) {
      const int item = tid + q * kThreads;
      if (item < kWQuads) {
        const int n = item >> 4;
        const int quad = item & 15;
        wreg4[q] =
            reinterpret_cast<const float4*>(fn + static_cast<long long>(n) * kK + k_begin)[quad];
      }
    }
  } else {
#pragma unroll
    for (int q = 0; q < kWPerThread; ++q) {
      wreg[q] = reinterpret_cast<const float2*>(fn + static_cast<long long>(w_n[q]) * kK +
                                                k_begin)[w_pair[q]];
    }
  }
  __syncthreads();
  const uint32_t tmem = tmem_slot;

  uint64_t sq_rows_packed[kM / (4 * kWarps)] = {};
  int mma_phase = 0;

#pragma unroll 1
  for (int it = 0; it < kIters; ++it) {
    const int stage = it % kStages;
    unsigned char* const a_stage = sA + stage * kABlockBytes;
    // MMA completion and the CTA barrier below serialize each weight slice,
    // so B needs one shared buffer even while A rotates across TMA stages.
    unsigned char* const b_stage = sB;
    if constexpr (kFloat4Weights) {
#pragma unroll
      for (int q = 0; q < kWQuadsPerThread; ++q) {
        const int item = tid + q * kThreads;
        if (item < kWQuads) {
          const int n = item >> 4;
          const int quad = item & 15;
          const float4 value = wreg4[q];
          const uint2 packed = make_uint2(pack_weight_bf16x2(value.x, value.y),
                                          pack_weight_bf16x2(value.z, value.w));
          const int byte = (quad * 8) ^ ((n & 7) << 4);
          *reinterpret_cast<uint2*>(b_stage + n * kRowBytes + byte) = packed;
        }
      }
    } else {
#pragma unroll
      for (int q = 0; q < kWPerThread; ++q) {
        const uint32_t packed = pack_weight_bf16x2(wreg[q].x, wreg[q].y);
        const int byte = (w_pair[q] * 4) ^ ((w_n[q] & 7) << 4);
        *reinterpret_cast<uint32_t*>(b_stage + w_n[q] * kRowBytes + byte) = packed;
      }
    }
    if (it + 1 < kIters) {
      const int k_next = k_begin + (it + 1) * 64;
      if constexpr (kFloat4Weights) {
#pragma unroll
        for (int q = 0; q < kWQuadsPerThread; ++q) {
          const int item = tid + q * kThreads;
          if (item < kWQuads) {
            const int n = item >> 4;
            const int quad = item & 15;
            wreg4[q] =
                reinterpret_cast<const float4*>(fn + static_cast<long long>(n) * kK + k_next)[quad];
          }
        }
      } else {
#pragma unroll
        for (int q = 0; q < kWPerThread; ++q) {
          wreg[q] = reinterpret_cast<const float2*>(fn + static_cast<long long>(w_n[q]) * kK +
                                                    k_next)[w_pair[q]];
        }
      }
    }
    tc_wait_mbar(tc_smem_addr(&tile_bar[stage]), (it / kStages) & 1);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int kk = 0; kk < 64; kk += 16) {
        tc_mma_n24_bf16<kM>(tmem, tc_desc_128b(a_stage + kk * 2), tc_desc_128b(b_stage + kk * 2),
                            it == 0 && kk == 0 ? 0 : 1);
      }
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
          "shared::cluster.b64 [%0];" ::"r"(mma_bar_addr)
          : "memory");
    }
#pragma unroll
    for (int group = 0; group < kM / (4 * kWarps); ++group) {
      const int r = group * 4 + (lane >> 3);
      const int row = warp + kWarps * r;
      const int segment = lane & 7;
      const int byte = (segment * 16) ^ ((row & 7) << 4);
      const uint4 value = *reinterpret_cast<const uint4*>(a_stage + row * kRowBytes + byte);
      sq_acc_f32x2(sq_rows_packed[group], value.x);
      sq_acc_f32x2(sq_rows_packed[group], value.y);
      sq_acc_f32x2(sq_rows_packed[group], value.z);
      sq_acc_f32x2(sq_rows_packed[group], value.w);
    }
    if (tid == 0) tc_wait_mbar(mma_bar_addr, mma_phase);
    mma_phase ^= 1;
    __syncthreads();
    if (it + kStages < kIters) issue_tile(stage, k_begin + (it + kStages) * 64);
  }

  if (warp < 4) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    constexpr int kRowsPerWarp = kM == 128 ? 32 : 16;
    const uint32_t row_base = tmem + (static_cast<uint32_t>(warp * 32) << 16);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      uint32_t out[8];
      tc_ld_x8(out, row_base + q * 8);
      tc_wait_ld();
      float* const transpose = reinterpret_cast<float*>(sA) + warp * 8 * 32;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        transpose[j * 32 + ((lane + 4 * j) & 31)] = __uint_as_float(out[j]);
      }
      __syncwarp();
#pragma unroll
      for (int group = 0; group < kRowsPerWarp / 4; ++group) {
        const int row_in_warp = 4 * group + (lane >> 3);
        const int j = lane & 7;
        const float value = transpose[j * 32 + ((row_in_warp + 4 * j) & 31)];
        const int row = warp * kRowsPerWarp + row_in_warp;
        float* scratch =
            reinterpret_cast<float*>(layer_input + static_cast<long long>(token0 + row) * kH);
        scratch[scratch_offset<kSplitK, kScratchPitch>(split, q * 8 + j)] = value;
      }
      if (q != 2) __syncwarp();
    }
  }
#pragma unroll
  for (int group = 0; group < kM / (4 * kWarps); ++group) {
    float total = __uint_as_float(static_cast<uint32_t>(sq_rows_packed[group])) +
                  __uint_as_float(static_cast<uint32_t>(sq_rows_packed[group] >> 32));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset, 8);
    }
    const int r = group * 4 + (lane >> 3);
    const int row = warp + kWarps * r;
    if ((lane & 7) == 0) {
      float* scratch =
          reinterpret_cast<float*>(layer_input + static_cast<long long>(token0 + row) * kH);
      scratch[scratch_offset<kSplitK, kScratchPitch>(split, kNF)] = total;
    }
  }
  __syncthreads();
  if constexpr (kProgrammaticTrigger) {
    if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(32)
                 : "memory");
  }
}

template <int kSplitK, int kM, int kStageK, int kThreads, int kStages, int kScratchPitch = 32,
          bool kWeightFirst = false, bool kPackedSq = false, bool kResidualEvictLast = false,
          bool kProgrammaticTrigger = false>
__global__ __launch_bounds__(kThreads) void mhc_project_pipe_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input) {
  constexpr int kN = 24;
  static_assert(kM == 64 || kM == 128);
  constexpr int kWarps = kThreads / 32;
  static_assert(kStageK % 64 == 0);
  constexpr int kK64Blocks = kStageK / 64;
  constexpr int kRows = kM;
  constexpr int kRowBytes = 128;
  constexpr int kABlockBytes = kRows * kRowBytes;
  constexpr int kBBlockBytes = kN * kRowBytes;
  constexpr int kABytes = kK64Blocks * kABlockBytes;
  constexpr int kSplitSize = kK / kSplitK;
  static_assert(kK % kSplitK == 0 && kSplitSize % kStageK == 0);
  constexpr int kNumStages = kSplitSize / kStageK;
  static_assert(kNumStages >= kStages);
  constexpr int kWItems = kK64Blocks * kN * 16;
  constexpr int kWPerThread = (kWItems + kThreads - 1) / kThreads;
  constexpr int kSqGroups = kRows / (4 * kWarps);

  extern __shared__ __align__(1024) unsigned char tc_smem[];
  unsigned char* const smem_a = tc_smem;
  unsigned char* const smem_b = tc_smem + kStages * kABytes;
  __shared__ uint64_t done_mbar;
  __shared__ uint32_t tmem_slot;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token0 = static_cast<int>(blockIdx.x) * kRows;
  const int split = static_cast<int>(blockIdx.y);
  const int k_begin = split * kSplitSize;
  const uint32_t mbar_addr = tc_smem_addr(&done_mbar);
  const uint32_t tmem_slot_addr = tc_smem_addr(&tmem_slot);

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_addr) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(tmem_slot_addr),
        "r"(32)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  __syncthreads();
  const uint32_t tmem = tmem_slot;
  uint64_t residual_evict_last_policy = 0;
  if constexpr (kResidualEvictLast) {
    asm volatile(
        "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 "
        "%0, 1.0;"
        : "=l"(residual_evict_last_policy));
  }

  auto issue_a = [&](int k0, int buf) {
    unsigned char* dst_base = smem_a + buf * kABytes;
#pragma unroll
    for (int copy = tid; copy < kK64Blocks * kRows * 8; copy += kThreads) {
      const int block = copy / (kRows * 8);
      const int in_block = copy - block * (kRows * 8);
      const int row = in_block >> 3;
      const int segment = in_block & 7;
      const __nv_bfloat16* src =
          residual + static_cast<long long>(token0 + row) * kK + k0 + block * 64 + segment * 8;
      const int byte = (segment * 16) ^ ((row & 7) << 4);
      const uint32_t dst = tc_smem_addr(dst_base + block * kABlockBytes + row * kRowBytes + byte);
      if constexpr (kResidualEvictLast) {
        asm volatile(
            "cp.async.cg.shared::cta.global.L2::cache_hint "
            "[%0], [%1], 16, %2;" ::"r"(dst),
            "l"(src), "l"(residual_evict_last_policy)
            : "memory");
      } else {
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
                     : "memory");
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  float4 w_cur[kWPerThread];
  float4 w_next[kWPerThread];
  auto load_w = [&](int k0, float4(&w)[kWPerThread]) {
#pragma unroll
    for (int q = 0; q < kWPerThread; ++q) {
      const int item = tid + q * kThreads;
      if constexpr (kWItems % kThreads != 0) {
        if (item >= kWItems) {
          w[q] = make_float4(0.f, 0.f, 0.f, 0.f);
          continue;
        }
      }
      const int block = item / (kN * 16);
      const int in_block = item - block * (kN * 16);
      const int n = in_block >> 4;
      const int quad = in_block & 15;
      const float4* weight_ptr =
          reinterpret_cast<const float4*>(fn + static_cast<long long>(n) * kK + k0 + block * 64) +
          quad;
      if constexpr (kResidualEvictLast)
        w[q] = load_global_evict_last_float4(weight_ptr, residual_evict_last_policy);
      else
        w[q] = *weight_ptr;
    }
  };
  auto store_w = [&](const float4(&w)[kWPerThread]) {
#pragma unroll
    for (int q = 0; q < kWPerThread; ++q) {
      const int item = tid + q * kThreads;
      if constexpr (kWItems % kThreads != 0) {
        if (item >= kWItems) continue;
      }
      const int block = item / (kN * 16);
      const int in_block = item - block * (kN * 16);
      const int n = in_block >> 4;
      const int quad = in_block & 15;
      const int byte = (quad * 8) ^ ((n & 7) << 4);
      *reinterpret_cast<uint2*>(smem_b + block * kBBlockBytes + n * kRowBytes + byte) =
          make_uint2(pack_weight_bf16x2(w[q].x, w[q].y), pack_weight_bf16x2(w[q].z, w[q].w));
    }
  };

  float sq_rows[kSqGroups] = {};
  uint64_t sq_rows_packed[kSqGroups] = {};
  int phase = 0;

  if constexpr (kWeightFirst) load_w(k_begin, w_cur);
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) issue_a(k_begin + s * kStageK, s);
  if constexpr (!kWeightFirst) load_w(k_begin, w_cur);

#pragma unroll 1
  for (int i = 0; i < kNumStages; ++i) {
    const int pre = i + kStages - 1;
    if (pre < kNumStages) {
      issue_a(k_begin + pre * kStageK, pre % kStages);
    } else {
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (i + 1 < kNumStages) load_w(k_begin + (i + 1) * kStageK, w_next);
    store_w(w_cur);
    asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 1) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    unsigned char* const a_buf = smem_a + (i % kStages) * kABytes;
    if (tid == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
          tc_mma_n24_bf16<kM>(tmem, tc_desc_128b(a_buf + block * kABlockBytes + kk * 2),
                              tc_desc_128b(smem_b + block * kBBlockBytes + kk * 2),
                              (i == 0 && block == 0 && kk == 0) ? 0 : 1);
        }
      }
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
          "shared::cluster.b64 [%0];" ::"r"(mbar_addr)
          : "memory");
    }
#pragma unroll
    for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
      for (int group = 0; group < kSqGroups; ++group) {
        const int r = group * 4 + (lane >> 3);
        const int row = warp + kWarps * r;
        const int segment = lane & 7;
        const int byte = (segment * 16) ^ ((row & 7) << 4);
        if constexpr (kPackedSq) {
          const uint4 value = *reinterpret_cast<const uint4*>(a_buf + block * kABlockBytes +
                                                              row * kRowBytes + byte);
          sq_acc_f32x2(sq_rows_packed[group], value.x);
          sq_acc_f32x2(sq_rows_packed[group], value.y);
          sq_acc_f32x2(sq_rows_packed[group], value.z);
          sq_acc_f32x2(sq_rows_packed[group], value.w);
        } else {
          union Bf16x8 {
            uint4 packed;
            __nv_bfloat162 pairs[4];
          } value;
          value.packed = *reinterpret_cast<const uint4*>(a_buf + block * kABlockBytes +
                                                         row * kRowBytes + byte);
#pragma unroll
          for (int half = 0; half < 4; ++half) {
            const float2 x = __bfloat1622float2(value.pairs[half]);
            sq_rows[group] = fmaf(x.x, x.x, sq_rows[group]);
            sq_rows[group] = fmaf(x.y, x.y, sq_rows[group]);
          }
        }
      }
    }
    if (tid == 0) tc_wait_mbar(mbar_addr, phase);
    phase ^= 1;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kWPerThread; ++q) w_cur[q] = w_next[q];
  }

  if (warp < 4) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const uint32_t row_base = tmem + (static_cast<uint32_t>(warp * 32) << 16);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      uint32_t out[8];
      tc_ld_x8(out, row_base + q * 8);
      tc_wait_ld();
      constexpr int kRowsPerWarp = kM == 128 ? 32 : 16;
      float* const transpose = reinterpret_cast<float*>(smem_a) + warp * 8 * 32;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        transpose[j * 32 + ((lane + 4 * j) & 31)] = __uint_as_float(out[j]);
      }
      __syncwarp();
#pragma unroll
      for (int group = 0; group < kRowsPerWarp / 4; ++group) {
        const int row_in_warp = 4 * group + (lane >> 3);
        const int j = lane & 7;
        const float value = transpose[j * 32 + ((row_in_warp + 4 * j) & 31)];
        const int row = warp * kRowsPerWarp + row_in_warp;
        float* scratch =
            reinterpret_cast<float*>(layer_input + static_cast<long long>(token0 + row) * kH);
        scratch[scratch_offset<kSplitK, kScratchPitch>(split, q * 8 + j)] = value;
      }
      if (q != 2) __syncwarp();
    }
  }

#pragma unroll
  for (int group = 0; group < kSqGroups; ++group) {
    float total = sq_rows[group];
    if constexpr (kPackedSq) {
      total = __uint_as_float(static_cast<uint32_t>(sq_rows_packed[group])) +
              __uint_as_float(static_cast<uint32_t>(sq_rows_packed[group] >> 32));
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset, 8);
    }
    const int r = group * 4 + (lane >> 3);
    const int row = warp + kWarps * r;
    if ((lane & 7) == 0) {
      float* scratch =
          reinterpret_cast<float*>(layer_input + static_cast<long long>(token0 + row) * kH);
      scratch[scratch_offset<kSplitK, kScratchPitch>(split, kNF)] = total;
    }
  }
  __syncthreads();
  if constexpr (kProgrammaticTrigger) {
    if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(32)
                 : "memory");
  }
}

template <int kSplitK, int kM, int kStageK, bool kPackedWeights = false, bool kFullM = true,
          int kThreads = 128, bool kWeightEvictLast = false, bool kPrefetchWeights256 = false,
          bool kResidualEvictLast = false, bool kPipelineTmemLoad = false,
          bool kPrefetchWeightLinesStart = false, int kScratchPitch = 32,
          bool kProgrammaticTrigger = false>
__global__ __launch_bounds__(kThreads) void mhc_project_tcgen_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  constexpr int kN = 24;
  static_assert(kM == 64 || kM == 128);
  static_assert(kThreads == 128 || kThreads == 256);
  static_assert(!kPrefetchWeights256 || (kThreads == 256 && !kPackedWeights));
  constexpr int kWarps = kThreads / 32;
  static_assert(kStageK % 64 == 0);
  constexpr int kK64Blocks = kStageK / 64;
  constexpr int kRows = kM;
  constexpr int kRowBytes = 128;
  constexpr int kABlockBytes = kRows * kRowBytes;
  constexpr int kBBlockBytes = kN * kRowBytes;
  constexpr int kABytes = kK64Blocks * kABlockBytes;
  constexpr int kBBytes = kK64Blocks * kBBlockBytes;
  constexpr int kSplitSize = kK / kSplitK;
  static_assert(kK % kSplitK == 0 && kSplitSize % kStageK == 0);

  extern __shared__ __align__(1024) unsigned char tc_smem[];
  unsigned char* const smem_a = tc_smem;
  unsigned char* const smem_b = tc_smem + kABytes;
  __shared__ uint64_t done_mbar;
  __shared__ uint32_t tmem_slot;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token0 = static_cast<int>(blockIdx.x) * kRows;
  const int split = static_cast<int>(blockIdx.y);
  const int k_begin = split * kSplitSize;
  const int k_end = k_begin + kSplitSize;
  const uint32_t mbar_addr = tc_smem_addr(&done_mbar);
  const uint32_t tmem_slot_addr = tc_smem_addr(&tmem_slot);
  uint64_t weight_evict_last_policy = 0;
  if constexpr (kWeightEvictLast) {
    asm volatile(
        "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 "
        "%0, 1.0;"
        : "=l"(weight_evict_last_policy));
  }
  uint64_t residual_evict_last_policy = 0;
  if constexpr (kResidualEvictLast) {
    asm volatile(
        "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 "
        "%0, 1.0;"
        : "=l"(residual_evict_last_policy));
  }

  if constexpr (kPrefetchWeightLinesStart) {
    constexpr int kLinesPerRow = kStageK / 32;
    constexpr int kWeightLines = kN * kLinesPerRow;
    for (int item = tid; item < kWeightLines; item += kThreads) {
      const int n = item / kLinesPerRow;
      const int line = item - n * kLinesPerRow;
      const float* ptr = fn + static_cast<long long>(n) * kK + k_begin + line * 32;
      asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
    }
  }

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_addr) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
        "[%0], %1;" ::"r"(tmem_slot_addr),
        "r"(32)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  __syncthreads();
  const uint32_t tmem = tmem_slot;

  float sq_rows[kRows / (4 * kWarps)] = {};
  int phase = 0;
#pragma unroll 1
  for (int k0 = k_begin; k0 < k_end; k0 += kStageK) {
    constexpr int kNPerWarp = kN / kWarps;
    float2 prefetched_weights[kK64Blocks][kNPerWarp];
    __nv_bfloat162 prefetched_packed_weights[kK64Blocks][kNPerWarp];
    if constexpr (kThreads == 128) {
// Issue the full weight slice before consuming any load.  This projection is
// under one wave at mid T and NCU reports long-scoreboard stalls as its main
// limiter; shared memory still caps residency after these extra registers.
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int slot = 0; slot < kNPerWarp; ++slot) {
          const int n = warp + slot * kWarps;
          if constexpr (kPackedWeights) {
            const int scalar = n * kK + k0 + block * 64 + lane * 2;
            const int packed_row = scalar / kPackedWeightRowElems;
            const int packed_col = scalar - packed_row * kPackedWeightRowElems;
            prefetched_packed_weights[block][slot] = *reinterpret_cast<const __nv_bfloat162*>(
                layer_input + static_cast<long long>(packed_row) * kH + kPackedWeightStart +
                packed_col);
          } else {
            const float2* weight_ptr = reinterpret_cast<const float2*>(
                                           fn + static_cast<long long>(n) * kK + k0 + block * 64) +
                                       lane;
            if constexpr (kWeightEvictLast) {
              prefetched_weights[block][slot] =
                  load_global_evict_last_float2(weight_ptr, weight_evict_last_policy);
            } else {
              prefetched_weights[block][slot] = *weight_ptr;
            }
          }
        }
      }
    }
    // Stage A with 16-byte asynchronous copies.  The copy group overlaps the
    // residual transfer with FP32-to-BF16 weight conversion below.
    if constexpr (kThreads == 256) {
#pragma unroll
      for (int copy = tid; copy < kK64Blocks * kRows * 8; copy += kThreads) {
        const int block = copy / (kRows * 8);
        const int in_block = copy - block * (kRows * 8);
        const int row = in_block >> 3;
        const int segment = in_block & 7;
        const int token = token0 + row;
        const int source_token = kFullM || token < tokens ? token : token0;
        const __nv_bfloat16* src =
            residual + static_cast<long long>(source_token) * kK + k0 + block * 64 + segment * 8;
        const int byte = (segment * 16) ^ ((row & 7) << 4);
        const uint32_t dst = tc_smem_addr(smem_a + block * kABlockBytes + row * kRowBytes + byte);
        if constexpr (kFullM) {
          if constexpr (kResidualEvictLast) {
            asm volatile(
                "cp.async.cg.shared::cta.global.L2::cache_hint "
                "[%0], [%1], 16, %2;" ::"r"(dst),
                "l"(src), "l"(residual_evict_last_policy)
                : "memory");
          } else {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
                         : "memory");
          }
        } else {
          const int source_bytes = token < tokens ? 16 : 0;
          asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
                       "r"(source_bytes)
                       : "memory");
        }
      }
    } else {
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int group = 0; group < kRows / 16; ++group) {
          const int r = group * 4 + (lane >> 3);
          const int row = warp + 4 * r;
          const int token = token0 + row;
          const int segment = lane & 7;
          const int source_token = kFullM || token < tokens ? token : token0;
          const __nv_bfloat16* src =
              residual + static_cast<long long>(source_token) * kK + k0 + block * 64 + segment * 8;
          const int byte = (segment * 16) ^ ((row & 7) << 4);
          const uint32_t dst = tc_smem_addr(smem_a + block * kABlockBytes + row * kRowBytes + byte);
          if constexpr (kFullM) {
            if constexpr (kResidualEvictLast) {
              asm volatile(
                  "cp.async.cg.shared::cta.global.L2::cache_hint "
                  "[%0], [%1], 16, %2;" ::"r"(dst),
                  "l"(src), "l"(residual_evict_last_policy)
                  : "memory");
            } else {
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
                           : "memory");
            }
          } else {
            const int source_bytes = token < tokens ? 16 : 0;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
                         "r"(source_bytes)
                         : "memory");
          }
        }
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    // Convert the FP32 projection weights to BF16 while staging the exact
    // 24 output columns (no N=32 padding in shared memory or TMEM).
    if constexpr (kThreads == 256) {
      if constexpr (kPrefetchWeights256) {
        constexpr int kItems = kK64Blocks * kN * 32;
        constexpr int kItemsPerThread = kItems / kThreads;
        static_assert(kItems % kThreads == 0);
        float2 values[kItemsPerThread];
#pragma unroll
        for (int q = 0; q < kItemsPerThread; ++q) {
          const int item = tid + q * kThreads;
          const int block = item / (kN * 32);
          const int in_block = item - block * (kN * 32);
          const int n = in_block >> 5;
          const int pair = in_block & 31;
          values[q] = reinterpret_cast<const float2*>(fn + static_cast<long long>(n) * kK + k0 +
                                                      block * 64)[pair];
        }
#pragma unroll
        for (int q = 0; q < kItemsPerThread; ++q) {
          const int item = tid + q * kThreads;
          const int block = item / (kN * 32);
          const int in_block = item - block * (kN * 32);
          const int n = in_block >> 5;
          const int pair = in_block & 31;
          const int byte = (pair * 4) ^ ((n & 7) << 4);
          *reinterpret_cast<__nv_bfloat162*>(smem_b + block * kBBlockBytes + n * kRowBytes + byte) =
              __float22bfloat162_rn(values[q]);
        }
      } else if constexpr (!kPackedWeights) {
#pragma unroll
        for (int item = tid; item < kK64Blocks * kN * 16; item += kThreads) {
          const int block = item / (kN * 16);
          const int in_block = item - block * (kN * 16);
          const int n = in_block >> 4;
          const int quad = in_block & 15;
          const float4 value = reinterpret_cast<const float4*>(fn + static_cast<long long>(n) * kK +
                                                               k0 + block * 64)[quad];
          const uint2 packed = make_uint2(pack_weight_bf16x2(value.x, value.y),
                                          pack_weight_bf16x2(value.z, value.w));
          const int byte = (quad * 8) ^ ((n & 7) << 4);
          *reinterpret_cast<uint2*>(smem_b + block * kBBlockBytes + n * kRowBytes + byte) = packed;
        }
      } else {
#pragma unroll
        for (int item = tid; item < kK64Blocks * kN * 32; item += kThreads) {
          const int block = item / (kN * 32);
          const int in_block = item - block * (kN * 32);
          const int n = in_block >> 5;
          const int pair = in_block & 31;
          __nv_bfloat162 packed;
          const int scalar = n * kK + k0 + block * 64 + pair * 2;
          const int packed_row = scalar / kPackedWeightRowElems;
          const int packed_col = scalar - packed_row * kPackedWeightRowElems;
          packed = *reinterpret_cast<const __nv_bfloat162*>(
              layer_input + static_cast<long long>(packed_row) * kH + kPackedWeightStart +
              packed_col);
          const int byte = (pair * 4) ^ ((n & 7) << 4);
          *reinterpret_cast<__nv_bfloat162*>(smem_b + block * kBBlockBytes + n * kRowBytes + byte) =
              packed;
        }
      }
    } else {
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int slot = 0; slot < kNPerWarp; ++slot) {
          const int n = warp + slot * kWarps;
          __nv_bfloat162 packed;
          if constexpr (kPackedWeights) {
            packed = prefetched_packed_weights[block][slot];
          } else {
            packed = __float22bfloat162_rn(prefetched_weights[block][slot]);
          }
          const int byte = (lane * 4) ^ ((n & 7) << 4);
          *reinterpret_cast<__nv_bfloat162*>(smem_b + block * kBBlockBytes + n * kRowBytes + byte) =
              packed;
        }
      }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
          const bool first = (k0 == k_begin && block == 0 && kk == 0);
          tc_mma_n24_bf16<kM>(tmem, tc_desc_128b(smem_a + block * kABlockBytes + kk * 2),
                              tc_desc_128b(smem_b + block * kBBlockBytes + kk * 2), first ? 0 : 1);
        }
      }
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
          "shared::cluster.b64 [%0];" ::"r"(mbar_addr)
          : "memory");
    }

    // The tcgen05 operations are asynchronous.  Re-read the staged A tile
    // while they execute and accumulate the exact FP32 square sum.
#pragma unroll
    for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
      for (int group = 0; group < kRows / (4 * kWarps); ++group) {
        const int r = group * 4 + (lane >> 3);
        const int row = warp + kWarps * r;
        const int segment = lane & 7;
        const int byte = (segment * 16) ^ ((row & 7) << 4);
        union Bf16x8 {
          uint4 packed;
          __nv_bfloat162 pairs[4];
        } value;
        value.packed =
            *reinterpret_cast<const uint4*>(smem_a + block * kABlockBytes + row * kRowBytes + byte);
#pragma unroll
        for (int half = 0; half < 4; ++half) {
          const float2 x = __bfloat1622float2(value.pairs[half]);
          sq_rows[group] = fmaf(x.x, x.x, sq_rows[group]);
          sq_rows[group] = fmaf(x.y, x.y, sq_rows[group]);
        }
      }
    }
    if (tid == 0) tc_wait_mbar(mbar_addr, phase);
    phase ^= 1;
    __syncthreads();

#if 0
    if constexpr (false) {
      // Reuse the converted B slab for a second M128 tile.  Split-K is
      // doubled at dispatch, so this preserves total CTA work and grid size
      // while halving weight loads and FP32-to-BF16 conversions.
      if constexpr (kThreads == 256) {
#pragma unroll
        for (int copy = tid; copy < kK64Blocks * kRows * 8;
             copy += kThreads) {
          const int block = copy / (kRows * 8);
          const int in_block = copy - block * (kRows * 8);
          const int row = in_block >> 3;
          const int segment = in_block & 7;
          const int token = token0 + kRows + row;
          const int source_token = kFullM || token < tokens ? token : token0;
          const __nv_bfloat16* src = residual
              + static_cast<long long>(source_token) * kK + k0 + block * 64
              + segment * 8;
          const int byte = (segment * 16) ^ ((row & 7) << 4);
          const uint32_t dst = tc_smem_addr(
              smem_a + block * kABlockBytes + row * kRowBytes + byte);
          if constexpr (kFullM) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                         :: "r"(dst), "l"(src) : "memory");
          } else {
            const int source_bytes = token < tokens ? 16 : 0;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                         :: "r"(dst), "l"(src), "r"(source_bytes) : "memory");
          }
        }
      } else {
#pragma unroll
        for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
          for (int group = 0; group < kRows / 16; ++group) {
            const int r = group * 4 + (lane >> 3);
            const int row = warp + 4 * r;
            const int token = token0 + kRows + row;
            const int segment = lane & 7;
            const int source_token = kFullM || token < tokens ? token : token0;
            const __nv_bfloat16* src = residual
                + static_cast<long long>(source_token) * kK + k0 + block * 64
                + segment * 8;
            const int byte = (segment * 16) ^ ((row & 7) << 4);
            const uint32_t dst = tc_smem_addr(
                smem_a + block * kABlockBytes + row * kRowBytes + byte);
            if constexpr (kFullM) {
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                           :: "r"(dst), "l"(src) : "memory");
            } else {
              const int source_bytes = token < tokens ? 16 : 0;
              asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                           :: "r"(dst), "l"(src), "r"(source_bytes) : "memory");
            }
          }
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
        for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
          for (int kk = 0; kk < 64; kk += 16) {
            const bool first = (k0 == k_begin && block == 0 && kk == 0);
            tc_mma_n24_bf16<kM>(
                tmem + 32,
                tc_desc_128b(smem_a + block * kABlockBytes + kk * 2),
                tc_desc_128b(smem_b + block * kBBlockBytes + kk * 2),
                first ? 0 : 1);
          }
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
            "shared::cluster.b64 [%0];"
            :: "r"(mbar_addr) : "memory");
      }
#pragma unroll
      for (int block = 0; block < kK64Blocks; ++block) {
#pragma unroll
        for (int group = 0; group < kRows / (4 * kWarps); ++group) {
          const int r = group * 4 + (lane >> 3);
          const int row = warp + kWarps * r;
          const int segment = lane & 7;
          const int byte = (segment * 16) ^ ((row & 7) << 4);
          union Bf16x8Second {
            uint4 packed;
            __nv_bfloat162 pairs[4];
          } value;
          value.packed = *reinterpret_cast<const uint4*>(
              smem_a + block * kABlockBytes + row * kRowBytes + byte);
#pragma unroll
          for (int half = 0; half < 4; ++half) {
            const float2 x = __bfloat1622float2(value.pairs[half]);
            sq_rows[1][group] = fmaf(x.x, x.x, sq_rows[1][group]);
            sq_rows[1][group] = fmaf(x.y, x.y, sq_rows[1][group]);
          }
        }
      }
      if (tid == 0) tc_wait_mbar(mbar_addr, phase);
      phase ^= 1;
      __syncthreads();
    }
#endif
  }

  if (warp < 4) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint32_t ring[2][8];
    const uint32_t row_base = tmem + (static_cast<uint32_t>(warp * 32) << 16);
    if constexpr (kPipelineTmemLoad) {
      tc_ld_x8(ring[0], row_base);
      tc_wait_ld();
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      uint32_t out[8];
      if constexpr (kPipelineTmemLoad) {
        if (q != 2) tc_ld_x8(ring[(q + 1) & 1], row_base + (q + 1) * 8);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = ring[q & 1][j];
      } else {
        tc_ld_x8(out, row_base + q * 8);
        tc_wait_ld();
      }
      constexpr int kRowsPerWarp = kM == 128 ? 32 : 16;
      float* const transpose = reinterpret_cast<float*>(smem_a) + warp * 8 * 32;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        transpose[j * 32 + ((lane + 4 * j) & 31)] = __uint_as_float(out[j]);
      }
      __syncwarp();
#pragma unroll
      for (int group = 0; group < kRowsPerWarp / 4; ++group) {
        const int row_in_warp = 4 * group + (lane >> 3);
        const int j = lane & 7;
        const float value = transpose[j * 32 + ((row_in_warp + 4 * j) & 31)];
        const int row = warp * kRowsPerWarp + row_in_warp;
        const int token = token0 + row;
        if constexpr (!kFullM) {
          if (token >= tokens) continue;
        }
        float* scratch = reinterpret_cast<float*>(layer_input + static_cast<long long>(token) * kH);
        scratch[scratch_offset<kSplitK, kScratchPitch>(split, q * 8 + j)] = value;
      }
      if (q != 2) {
        if constexpr (kPipelineTmemLoad) tc_wait_ld();
        __syncwarp();
      }
    }
  }

#pragma unroll
  for (int group = 0; group < kRows / (4 * kWarps); ++group) {
    float total = sq_rows[group];
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset, 8);
    }
    const int r = group * 4 + (lane >> 3);
    const int row = warp + kWarps * r;
    if ((lane & 7) == 0) {
      if constexpr (!kFullM) {
        if (token0 + row >= tokens) continue;
      }
      float* scratch =
          reinterpret_cast<float*>(layer_input + static_cast<long long>(token0 + row) * kH);
      scratch[scratch_offset<kSplitK, kScratchPitch>(split, kNF)] = total;
    }
  }
  __syncthreads();
  if constexpr (kProgrammaticTrigger) {
    if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(32)
                 : "memory");
  }
}

__device__ __forceinline__ uint64_t load_global_evict_first_u64(const void* ptr, uint64_t policy) {
  uint64_t value;
  asm volatile("ld.global.L2::cache_hint.b64 %0, [%1], %2;"
               : "=l"(value)
               : "l"(ptr), "l"(policy)
               : "memory");
  return value;
}

__device__ __forceinline__ uint4 load_global_evict_first_u128(const void* ptr, uint64_t policy) {
  uint4 value;
  asm volatile("ld.global.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr), "l"(policy)
               : "memory");
  return value;
}

__device__ __forceinline__ void store_global_evict_first_u128(void* ptr, uint4 value,
                                                              uint64_t policy) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(ptr), "r"(value.x),
               "r"(value.y), "r"(value.z), "r"(value.w), "l"(policy)
               : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t bits;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(hi), "f"(lo));
  return bits;
}

__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t result;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

__device__ __forceinline__ void sink_rownorm16(float (&m)[16]) {
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const float v = (m[4 * a] + m[4 * a + 1]) + (m[4 * a + 2] + m[4 * a + 3]) + kEps;
    const float inv = 1.0f / v;
#pragma unroll
    for (int b = 0; b < 4; ++b) m[4 * a + b] *= inv;
  }
}

__device__ __forceinline__ void sink_colnorm16(float (&m)[16]) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float v = (m[b] + m[4 + b]) + (m[8 + b] + m[12 + b]) + kEps;
    const float inv = 1.0f / v;
#pragma unroll
    for (int a = 0; a < 4; ++a) m[4 * a + b] *= inv;
  }
}

template <int kMaxThreads, int kSplitK, int kScratchPitch = kNF + 1, bool kEvictFirst = false,
          bool kWideEvict = false, bool kScalarSink = false, bool kProgrammaticWait = false>
__global__ __launch_bounds__(kMaxThreads, kMaxThreads == 288 ? 2 : 4) void mhc_tf32_epilogue_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base, float* __restrict__ post_mix, float* __restrict__ comb,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  static_assert(!kWideEvict || kEvictFirst);
  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const __nv_bfloat16* x = residual + static_cast<long long>(token) * kK;
  const float* scratch =
      reinterpret_cast<const float*>(layer_input + static_cast<long long>(token) * kH);
  // hc_scale / hc_base are tiny cold reads whose latency would otherwise be
  // exposed right after the split-K reduction.  Issue them in warp 0 here and
  // consume them from registers below.
  float sc_reg = 0.0f;
  float ba_reg = 0.0f;
  if (warp == 0) {
    if (lane < 3) sc_reg = hc_scale[lane];
    if (lane < kNF) ba_reg = hc_base[lane];
  }
  uint64_t evict_first_policy = 0;
  if constexpr (kEvictFirst) {
    asm volatile(
        "createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 "
        "%0, 1.0;"
        : "=l"(evict_first_policy));
  }

  __shared__ float pre[kHC];
  __shared__ uint32_t pre_packed[kHC];
  __shared__ __align__(16) float split_partials[5][32];
  // T=256/512/1024 run well under one wave, so the residual load latency that
  // follows the split-K reduction and the gate chain is otherwise fully
  // exposed: no co-resident CTA is available to cover it.  residual is a
  // kernel input, not the previous grid's output, so this demand load is
  // issued ahead of cudaGridDependencySynchronize() and overlaps the
  // projection kernel's own tail as well as the gate chain.
  uint4 wide_first_x0;
  uint4 wide_first_x1;
  uint4 wide_first_x2;
  uint4 wide_first_x3;
  if constexpr ((kMaxThreads == 128 || kMaxThreads == 160) && kSplitK >= 64 && !kEvictFirst) {
    if (warp > 0) {
      const int first_h2 = 8 * (tid - 32);
      wide_first_x0 = *reinterpret_cast<const uint4*>(x + first_h2);
      wide_first_x1 = *reinterpret_cast<const uint4*>(x + kH + first_h2);
      wide_first_x2 = *reinterpret_cast<const uint4*>(x + 2 * kH + first_h2);
      wide_first_x3 = *reinterpret_cast<const uint4*>(x + 3 * kH + first_h2);
    }
  }
  if constexpr (kProgrammaticWait) cudaGridDependencySynchronize();
  float reduced = 0.0f;
  if constexpr (kSplitK >= 16) {
    if ((kMaxThreads == 128 || warp < (kMaxThreads == 160 ? 5 : 4)) && lane <= kNF) {
      if constexpr (kSplitK == 64) {
        if constexpr (kMaxThreads == 160) {
          float reduced0 = 0.0f;
          float reduced1 = 0.0f;
#pragma unroll
          for (int split = warp; split < kSplitK; split += 10) {
            reduced0 += scratch[scratch_offset<kSplitK, kScratchPitch>(split, lane)];
            if (split + 5 < kSplitK)
              reduced1 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 5, lane)];
          }
          reduced = reduced0 + reduced1;
        } else if constexpr (kMaxThreads >= 256) {
          // The wider epilogue has register headroom to expose all scratch
          // loads.  The 128-thread T=1024 path instead keeps the lower-live-
          // range chains below; a 16-record array causes a register cliff.
          float record[16];
#pragma unroll
          for (int q = 0; q < 16; ++q)
            record[q] = scratch[scratch_offset<kSplitK, kScratchPitch>(warp + 4 * q, lane)];
          float reduced0 = 0.0f;
          float reduced1 = 0.0f;
#pragma unroll
          for (int q = 0; q < 16; q += 2) {
            reduced0 += record[q];
            reduced1 += record[q + 1];
          }
          reduced = reduced0 + reduced1;
        } else {
          float reduced0 = 0.0f;
          float reduced1 = 0.0f;
#pragma unroll
          for (int split = warp; split < kSplitK; split += 16) {
            reduced0 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 0, lane)];
            reduced1 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 4, lane)];
            reduced0 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 8, lane)];
            reduced1 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 12, lane)];
          }
          reduced = reduced0 + reduced1;
        }
      } else if constexpr (kSplitK == 32) {
        // Issue all records before consuming any result, then use a balanced
        // tree so scratch latency is not serialized by dependent additions.
        float record[8];
#pragma unroll
        for (int q = 0; q < 8; ++q)
          record[q] = scratch[scratch_offset<kSplitK, kScratchPitch>(warp + 4 * q, lane)];
        const float r01 = record[0] + record[1];
        const float r23 = record[2] + record[3];
        const float r45 = record[4] + record[5];
        const float r67 = record[6] + record[7];
        reduced = (r01 + r23) + (r45 + r67);
      } else if constexpr (kSplitK == 16) {
        float reduced0 = 0.0f;
        float reduced1 = 0.0f;
#pragma unroll
        for (int split = warp; split < kSplitK; split += 8) {
          reduced0 += scratch[scratch_offset<kSplitK, kScratchPitch>(split, lane)];
          reduced1 += scratch[scratch_offset<kSplitK, kScratchPitch>(split + 4, lane)];
        }
        reduced = reduced0 + reduced1;
      } else {
#pragma unroll
        for (int split = warp; split < kSplitK; split += 4) {
          reduced += scratch[scratch_offset<kSplitK, kScratchPitch>(split, lane)];
        }
      }
      split_partials[warp][lane] = reduced;
    }
    __syncthreads();
    if (warp == 0 && lane <= kNF) {
      reduced = split_partials[0][lane] + split_partials[1][lane] + split_partials[2][lane] +
                split_partials[3][lane];
      if constexpr (kMaxThreads == 160) reduced += split_partials[4][lane];
    }
  } else {
    if (warp == 0 && lane <= kNF) {
#pragma unroll
      for (int split = 0; split < kSplitK; ++split) {
        reduced += scratch[scratch_offset<kSplitK, kScratchPitch>(split, lane)];
      }
    }
  }

  // At T=2048, issue the first residual vector before warp 0 publishes the
  // gates so HBM latency overlaps the nonlinear setup and CTA barrier.
  uint2 t2048_first_x0;
  uint2 t2048_first_x1;
  uint2 t2048_first_x2;
  uint2 t2048_first_x3;
  if constexpr (kMaxThreads == 128 && kSplitK == 32 && !kEvictFirst) {
    if (warp > 0) {
      const int first_h2 = 4 * (tid - 32);
      t2048_first_x0 = *reinterpret_cast<const uint2*>(x + first_h2);
      t2048_first_x1 = *reinterpret_cast<const uint2*>(x + kH + first_h2);
      t2048_first_x2 = *reinterpret_cast<const uint2*>(x + 2 * kH + first_h2);
      t2048_first_x3 = *reinterpret_cast<const uint2*>(x + 3 * kH + first_h2);
    }
  }

  uint4 t4096_first_x0;
  uint4 t4096_first_x1;
  uint4 t4096_first_x2;
  uint4 t4096_first_x3;
  if constexpr (kMaxThreads == 128 && kSplitK == 16 && kEvictFirst && !kWideEvict) {
    if (warp > 0) {
      const int first_h2 = 8 * (tid - 32);
      t4096_first_x0 = load_global_evict_first_u128(x + first_h2, evict_first_policy);
      t4096_first_x1 = load_global_evict_first_u128(x + kH + first_h2, evict_first_policy);
      t4096_first_x2 = load_global_evict_first_u128(x + 2 * kH + first_h2, evict_first_policy);
      t4096_first_x3 = load_global_evict_first_u128(x + 3 * kH + first_h2, evict_first_policy);
    }
  }

  uint64_t twide_first_x0_lo, twide_first_x0_hi;
  uint64_t twide_first_x1_lo, twide_first_x1_hi;
  uint64_t twide_first_x2_lo, twide_first_x2_hi;
  uint64_t twide_first_x3_lo, twide_first_x3_hi;
  if constexpr (kMaxThreads == 128 && kEvictFirst && kWideEvict) {
    if (warp > 0) {
      const int first_h2 = 8 * (tid - 32);
      twide_first_x0_lo = load_global_evict_first_u64(x + first_h2, evict_first_policy);
      twide_first_x0_hi = load_global_evict_first_u64(x + first_h2 + 4, evict_first_policy);
      twide_first_x1_lo = load_global_evict_first_u64(x + kH + first_h2, evict_first_policy);
      twide_first_x1_hi = load_global_evict_first_u64(x + kH + first_h2 + 4, evict_first_policy);
      twide_first_x2_lo = load_global_evict_first_u64(x + 2 * kH + first_h2, evict_first_policy);
      twide_first_x2_hi =
          load_global_evict_first_u64(x + 2 * kH + first_h2 + 4, evict_first_policy);
      twide_first_x3_lo = load_global_evict_first_u64(x + 3 * kH + first_h2, evict_first_policy);
      twide_first_x3_hi =
          load_global_evict_first_u64(x + 3 * kH + first_h2 + 4, evict_first_policy);
    }
  }

  if constexpr (kMaxThreads >= 256) {
    // Use otherwise-idle worker warps to start the first residual cache lines
    // while warp 0 finishes the split-K reduction and nonlinear setup.
    if (warp > 0 && (lane & 15) == 0) {
      const int h2 = 4 * (tid - 32);
      asm volatile("prefetch.global.L1 [%0];" ::"l"(x + h2));
      asm volatile("prefetch.global.L1 [%0];" ::"l"(x + kH + h2));
      asm volatile("prefetch.global.L1 [%0];" ::"l"(x + 2 * kH + h2));
      asm volatile("prefetch.global.L1 [%0];" ::"l"(x + 3 * kH + h2));
    }
  }

  float value = 0.0f;
  if (warp == 0) {
    constexpr unsigned kMask = 0xffffffffu;
    const float sqsum = __shfl_sync(kMask, reduced, kNF);
    const float inv_rms = rsqrtf(sqsum * (1.0f / static_cast<float>(kK)) + kEps);
    const float logit = lane < kNF ? reduced * inv_rms : 0.0f;
    const float sc0 = __shfl_sync(kMask, sc_reg, 0);
    const float sc1 = __shfl_sync(kMask, sc_reg, 1);
    const float sc2 = __shfl_sync(kMask, sc_reg, 2);
    if (lane < 4) {
      const float gate = sigmoidf_fast(fmaf(logit, sc0, ba_reg)) + kEps;
      pre[lane] = gate;
      if constexpr (kMaxThreads == 128 || kMaxThreads == 160 || kMaxThreads == 256 ||
                    kMaxThreads == 288)
        pre_packed[lane] = pack_bf16x2(gate, gate);
    } else if (lane < 8) {
      post_mix[token * 4 + lane - 4] = sigmoidf_fast(fmaf(logit, sc1, ba_reg));
    }
    const int source_lane = lane < 16 ? lane + 8 : 0;
    const float comb_logit = __shfl_sync(kMask, logit, source_lane);
    const float comb_bias = __shfl_sync(kMask, ba_reg, source_lane);
    value = lane < 16 ? fmaf(comb_logit, sc2, comb_bias) : -1.0e30f;
  }
  __syncthreads();

  if (warp == 0) {
    constexpr unsigned kMask = 0xffffffffu;
    float row_max = fmaxf(value, __shfl_xor_sync(kMask, value, 1));
    row_max = fmaxf(row_max, __shfl_xor_sync(kMask, row_max, 2));
    value = lane < 16 ? __expf(value - row_max) : 0.0f;
    float denom = value + __shfl_xor_sync(kMask, value, 1);
    denom += __shfl_xor_sync(kMask, denom, 2);
    value = lane < 16 ? value / denom + kEps : 0.0f;

    if constexpr (kScalarSink) {
      if (lane < 16) split_partials[0][lane] = value;
      __syncwarp();
      if (lane == 0) {
        float m[16];
        float4* const m4 = reinterpret_cast<float4*>(m);
        const float4* const gathered = reinterpret_cast<const float4*>(split_partials[0]);
        m4[0] = gathered[0];
        m4[1] = gathered[1];
        m4[2] = gathered[2];
        m4[3] = gathered[3];
        sink_factored16(m);
        float4* out4 = reinterpret_cast<float4*>(comb + token * 16);
        out4[0] = make_float4(m[0], m[1], m[2], m[3]);
        out4[1] = make_float4(m[4], m[5], m[6], m[7]);
        out4[2] = make_float4(m[8], m[9], m[10], m[11]);
        out4[3] = make_float4(m[12], m[13], m[14], m[15]);
      }
    } else {
      denom = value + __shfl_xor_sync(kMask, value, 4);
      denom += __shfl_xor_sync(kMask, denom, 8);
      value = lane < 16 ? value / (denom + kEps) : 0.0f;
#pragma unroll
      for (int iter = 0; iter < 19; ++iter) {
        denom = value + __shfl_xor_sync(kMask, value, 1);
        denom += __shfl_xor_sync(kMask, denom, 2);
        value = lane < 16 ? value / (denom + kEps) : 0.0f;
        denom = value + __shfl_xor_sync(kMask, value, 4);
        denom += __shfl_xor_sync(kMask, denom, 8);
        value = lane < 16 ? value / (denom + kEps) : 0.0f;
      }
      if (lane < 16) comb[token * 16 + lane] = value;
    }
  } else {
    auto* y2 = reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(token) * kH);
    const int worker = tid - 32;
    constexpr int worker_threads = kMaxThreads - 32;
    if constexpr (kMaxThreads == 128 && kSplitK == 32 && !kEvictFirst) {
      union Bf16x4 {
        uint2 packed;
        __nv_bfloat162 pairs[2];
      };
      const int first_pair = 2 * worker;
      {
        Bf16x4 x0, x1, x2v, x3, out;
        x0.packed = t2048_first_x0;
        x1.packed = t2048_first_x1;
        x2v.packed = t2048_first_x2;
        x3.packed = t2048_first_x3;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float2 a = __bfloat1622float2(x0.pairs[half]);
          const float2 b = __bfloat1622float2(x1.pairs[half]);
          const float2 c = __bfloat1622float2(x2v.pairs[half]);
          const float2 d = __bfloat1622float2(x3.pairs[half]);
          float2 result;
          result.x = fmaf(pre[3], d.x, fmaf(pre[2], c.x, fmaf(pre[1], b.x, pre[0] * a.x)));
          result.y = fmaf(pre[3], d.y, fmaf(pre[2], c.y, fmaf(pre[1], b.y, pre[0] * a.y)));
          out.pairs[half] = __float22bfloat162_rn(result);
        }
        *reinterpret_cast<uint2*>(y2 + first_pair) = out.packed;
      }
#pragma unroll 2
      for (int pair = first_pair + 2 * worker_threads; pair < kH / 2; pair += 2 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x4 x0, x1, x2v, x3, out;
        x0.packed = *reinterpret_cast<const uint2*>(x + h2);
        x1.packed = *reinterpret_cast<const uint2*>(x + kH + h2);
        x2v.packed = *reinterpret_cast<const uint2*>(x + 2 * kH + h2);
        x3.packed = *reinterpret_cast<const uint2*>(x + 3 * kH + h2);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float2 a = __bfloat1622float2(x0.pairs[half]);
          const float2 b = __bfloat1622float2(x1.pairs[half]);
          const float2 c = __bfloat1622float2(x2v.pairs[half]);
          const float2 d = __bfloat1622float2(x3.pairs[half]);
          float2 result;
          result.x = fmaf(pre[3], d.x, fmaf(pre[2], c.x, fmaf(pre[1], b.x, pre[0] * a.x)));
          result.y = fmaf(pre[3], d.y, fmaf(pre[2], c.y, fmaf(pre[1], b.y, pre[0] * a.y)));
          out.pairs[half] = __float22bfloat162_rn(result);
        }
        *reinterpret_cast<uint2*>(y2 + pair) = out.packed;
      }
    } else if constexpr ((kMaxThreads == 128 || kMaxThreads == 160) && kSplitK >= 64 &&
                         !kEvictFirst) {
      union Bf16x8 {
        uint4 packed;
        uint32_t raw[4];
        __nv_bfloat162 pairs[4];
      };
      const uint32_t gate0 = pre_packed[0];
      const uint32_t gate1 = pre_packed[1];
      const uint32_t gate2 = pre_packed[2];
      const uint32_t gate3 = pre_packed[3];
      {
        Bf16x8 x0, x1, x2v, x3, out;
        x0.packed = wide_first_x0;
        x1.packed = wide_first_x1;
        x2v.packed = wide_first_x2;
        x3.packed = wide_first_x3;
#pragma unroll
        for (int half = 0; half < 4; ++half) {
          uint32_t result = fma_bf16x2(x0.raw[half], gate0, 0u);
          result = fma_bf16x2(x1.raw[half], gate1, result);
          result = fma_bf16x2(x2v.raw[half], gate2, result);
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, result);
        }
        *reinterpret_cast<uint4*>(y2 + 4 * worker) = out.packed;
      }
#pragma unroll 2
      for (int pair = 4 * worker + 4 * worker_threads; pair < kH / 2; pair += 4 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x8 x0, x1, x2v, x3;
        x0.packed = *reinterpret_cast<const uint4*>(x + h2);
        x1.packed = *reinterpret_cast<const uint4*>(x + kH + h2);
        x2v.packed = *reinterpret_cast<const uint4*>(x + 2 * kH + h2);
        x3.packed = *reinterpret_cast<const uint4*>(x + 3 * kH + h2);
        Bf16x8 out;
        if constexpr (kMaxThreads == 160) {
          uint32_t accum[4];
#pragma unroll
          for (int half = 0; half < 4; ++half) accum[half] = fma_bf16x2(x0.raw[half], gate0, 0u);
#pragma unroll
          for (int half = 0; half < 4; ++half)
            accum[half] = fma_bf16x2(x1.raw[half], gate1, accum[half]);
#pragma unroll
          for (int half = 0; half < 4; ++half)
            accum[half] = fma_bf16x2(x2v.raw[half], gate2, accum[half]);
#pragma unroll
          for (int half = 0; half < 4; ++half)
            out.raw[half] = fma_bf16x2(x3.raw[half], gate3, accum[half]);
        } else {
#pragma unroll
          for (int half = 0; half < 4; ++half) {
            uint32_t result = fma_bf16x2(x0.raw[half], gate0, 0u);
            result = fma_bf16x2(x1.raw[half], gate1, result);
            result = fma_bf16x2(x2v.raw[half], gate2, result);
            out.raw[half] = fma_bf16x2(x3.raw[half], gate3, result);
          }
        }
        *reinterpret_cast<uint4*>(y2 + pair) = out.packed;
      }
    } else if constexpr (kMaxThreads == 256 || kMaxThreads == 288) {
      union Bf16x8 {
        uint4 packed;
        uint32_t raw[4];
      };
      const uint32_t gate0 = pre_packed[0];
      const uint32_t gate1 = pre_packed[1];
      const uint32_t gate2 = pre_packed[2];
      const uint32_t gate3 = pre_packed[3];
#pragma unroll 2
      for (int pair = 4 * worker; pair < kH / 2; pair += 4 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x8 x0, x1, x2v, x3;
        x0.packed = *reinterpret_cast<const uint4*>(x + h2);
        x1.packed = *reinterpret_cast<const uint4*>(x + kH + h2);
        x2v.packed = *reinterpret_cast<const uint4*>(x + 2 * kH + h2);
        x3.packed = *reinterpret_cast<const uint4*>(x + 3 * kH + h2);
        Bf16x8 out;
#pragma unroll
        for (int half = 0; half < 4; ++half) {
          uint32_t result = fma_bf16x2(x0.raw[half], gate0, 0u);
          result = fma_bf16x2(x1.raw[half], gate1, result);
          result = fma_bf16x2(x2v.raw[half], gate2, result);
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, result);
        }
        *reinterpret_cast<uint4*>(y2 + pair) = out.packed;
      }
    } else if constexpr (kEvictFirst && !kWideEvict) {
      union Bf16x8 {
        uint4 packed;
        uint32_t raw[4];
      };
      const uint32_t gate0 = pre_packed[0];
      const uint32_t gate1 = pre_packed[1];
      const uint32_t gate2 = pre_packed[2];
      const uint32_t gate3 = pre_packed[3];
      const int first_pair = 4 * worker;
      {
        Bf16x8 x0, x1, x2v, x3, out;
        x0.packed = t4096_first_x0;
        x1.packed = t4096_first_x1;
        x2v.packed = t4096_first_x2;
        x3.packed = t4096_first_x3;
        uint32_t accum[4];
#pragma unroll
        for (int half = 0; half < 4; ++half) accum[half] = fma_bf16x2(x0.raw[half], gate0, 0u);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          accum[half] = fma_bf16x2(x1.raw[half], gate1, accum[half]);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          accum[half] = fma_bf16x2(x2v.raw[half], gate2, accum[half]);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, accum[half]);
        *reinterpret_cast<uint4*>(y2 + first_pair) = out.packed;
      }
#pragma unroll 2
      for (int pair = first_pair + 4 * worker_threads; pair < kH / 2; pair += 4 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x8 x0, x1, x2v, x3;
        x0.packed = load_global_evict_first_u128(x + h2, evict_first_policy);
        x1.packed = load_global_evict_first_u128(x + kH + h2, evict_first_policy);
        x2v.packed = load_global_evict_first_u128(x + 2 * kH + h2, evict_first_policy);
        x3.packed = load_global_evict_first_u128(x + 3 * kH + h2, evict_first_policy);
        Bf16x8 out;
        uint32_t accum[4];
#pragma unroll
        for (int half = 0; half < 4; ++half) accum[half] = fma_bf16x2(x0.raw[half], gate0, 0u);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          accum[half] = fma_bf16x2(x1.raw[half], gate1, accum[half]);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          accum[half] = fma_bf16x2(x2v.raw[half], gate2, accum[half]);
#pragma unroll
        for (int half = 0; half < 4; ++half)
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, accum[half]);
        *reinterpret_cast<uint4*>(y2 + pair) = out.packed;
      }
    } else if constexpr (kWideEvict) {
      union Bf16x8 {
        uint4 packed;
        uint64_t halves[2];
        uint32_t raw[4];
      };
      const uint32_t gate0 = pack_bf16x2(pre[0], pre[0]);
      const uint32_t gate1 = pack_bf16x2(pre[1], pre[1]);
      const uint32_t gate2 = pack_bf16x2(pre[2], pre[2]);
      const uint32_t gate3 = pack_bf16x2(pre[3], pre[3]);
      const int first_pair = 4 * worker;
      {
        Bf16x8 x0, x1, x2v, x3, out;
        x0.halves[0] = twide_first_x0_lo;
        x0.halves[1] = twide_first_x0_hi;
        x1.halves[0] = twide_first_x1_lo;
        x1.halves[1] = twide_first_x1_hi;
        x2v.halves[0] = twide_first_x2_lo;
        x2v.halves[1] = twide_first_x2_hi;
        x3.halves[0] = twide_first_x3_lo;
        x3.halves[1] = twide_first_x3_hi;
#pragma unroll
        for (int half = 0; half < 4; ++half) {
          uint32_t result = fma_bf16x2(x0.raw[half], gate0, 0u);
          result = fma_bf16x2(x1.raw[half], gate1, result);
          result = fma_bf16x2(x2v.raw[half], gate2, result);
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, result);
        }
        store_global_evict_first_u128(y2 + first_pair, out.packed, evict_first_policy);
      }
#pragma unroll 2
      for (int pair = first_pair + 4 * worker_threads; pair < kH / 2; pair += 4 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x8 x0, x1, x2v, x3;
        x0.halves[0] = load_global_evict_first_u64(x + h2, evict_first_policy);
        x0.halves[1] = load_global_evict_first_u64(x + h2 + 4, evict_first_policy);
        x1.halves[0] = load_global_evict_first_u64(x + kH + h2, evict_first_policy);
        x1.halves[1] = load_global_evict_first_u64(x + kH + h2 + 4, evict_first_policy);
        x2v.halves[0] = load_global_evict_first_u64(x + 2 * kH + h2, evict_first_policy);
        x2v.halves[1] = load_global_evict_first_u64(x + 2 * kH + h2 + 4, evict_first_policy);
        x3.halves[0] = load_global_evict_first_u64(x + 3 * kH + h2, evict_first_policy);
        x3.halves[1] = load_global_evict_first_u64(x + 3 * kH + h2 + 4, evict_first_policy);
        Bf16x8 out;
#pragma unroll
        for (int half = 0; half < 4; ++half) {
          uint32_t result = fma_bf16x2(x0.raw[half], gate0, 0u);
          result = fma_bf16x2(x1.raw[half], gate1, result);
          result = fma_bf16x2(x2v.raw[half], gate2, result);
          out.raw[half] = fma_bf16x2(x3.raw[half], gate3, result);
        }
        store_global_evict_first_u128(y2 + pair, out.packed, evict_first_policy);
      }
    } else {
      union Bf16x4 {
        uint2 packed;
        uint64_t packed64;
        __nv_bfloat162 pairs[2];
      };
#pragma unroll 2
      for (int pair = 2 * worker; pair < kH / 2; pair += 2 * worker_threads) {
        const int h2 = pair * 2;
        Bf16x4 x0, x1, x2v, x3;
        if constexpr (kEvictFirst) {
          x0.packed64 = load_global_evict_first_u64(x + h2, evict_first_policy);
          x1.packed64 = load_global_evict_first_u64(x + kH + h2, evict_first_policy);
          x2v.packed64 = load_global_evict_first_u64(x + 2 * kH + h2, evict_first_policy);
          x3.packed64 = load_global_evict_first_u64(x + 3 * kH + h2, evict_first_policy);
        } else {
          x0.packed = *reinterpret_cast<const uint2*>(x + h2);
          x1.packed = *reinterpret_cast<const uint2*>(x + kH + h2);
          x2v.packed = *reinterpret_cast<const uint2*>(x + 2 * kH + h2);
          x3.packed = *reinterpret_cast<const uint2*>(x + 3 * kH + h2);
        }
        Bf16x4 out;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float2 a = __bfloat1622float2(x0.pairs[half]);
          const float2 b = __bfloat1622float2(x1.pairs[half]);
          const float2 c = __bfloat1622float2(x2v.pairs[half]);
          const float2 d = __bfloat1622float2(x3.pairs[half]);
          float2 result;
          result.x = fmaf(pre[3], d.x, fmaf(pre[2], c.x, fmaf(pre[1], b.x, pre[0] * a.x)));
          result.y = fmaf(pre[3], d.y, fmaf(pre[2], c.y, fmaf(pre[1], b.y, pre[0] * a.y)));
          out.pairs[half] = __float22bfloat162_rn(result);
        }
        *reinterpret_cast<uint2*>(y2 + pair) = out.packed;
      }
    }
  }
}

constexpr int kBulkChunk = 1024;
constexpr int kBulkChunks = kH / kBulkChunk;

struct __align__(128) BulkEpilogueShared {
  float pre[kHC];
  float split_partials[4][32];
  __align__(128) __nv_bfloat16 xbuf[2][kHC * kBulkChunk];
  __align__(8) uint64_t full[2];
};

__device__ __forceinline__ void issue_bulk_residual_chunk(__nv_bfloat16* dst,
                                                          const __nv_bfloat16* src, uint64_t* bar) {
  const uint32_t bar_addr = tc_smem_addr(bar);
  constexpr int kBytes = kHC * kBulkChunk * sizeof(__nv_bfloat16);
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar_addr),
               "r"(kBytes)
               : "memory");
#pragma unroll
  for (int hc = 0; hc < kHC; ++hc) {
    const uint32_t dst_addr = tc_smem_addr(dst + hc * kBulkChunk);
    const __nv_bfloat16* src_hc = src + hc * kH;
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(dst_addr),
        "l"(src_hc), "r"(kBulkChunk * 2), "r"(bar_addr)
        : "memory");
  }
}

template <int kSplitK, int kScratchPitch = kNF + 1>
__global__ __launch_bounds__(128, 4) void mhc_tf32_epilogue_bulk_tiled_kernel(
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base, float* __restrict__ post_mix, float* __restrict__ comb,
    __nv_bfloat16* __restrict__ layer_input, int tokens) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const __nv_bfloat16* x = residual + static_cast<long long>(token) * kK;
  const float* scratch =
      reinterpret_cast<const float*>(layer_input + static_cast<long long>(token) * kH);
  __shared__ BulkEpilogueShared shared;

  if (tid < 2) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tc_smem_addr(&shared.full[tid]))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 32) {
    issue_bulk_residual_chunk(shared.xbuf[0], x, &shared.full[0]);
    issue_bulk_residual_chunk(shared.xbuf[1], x + kBulkChunk, &shared.full[1]);
  }

  float reduced = 0.0f;
  if (lane <= kNF) {
#pragma unroll
    for (int split = warp; split < kSplitK; split += 4) {
      reduced += scratch[scratch_offset<kSplitK, kScratchPitch>(split, lane)];
    }
    shared.split_partials[warp][lane] = reduced;
  }
  __syncthreads();
  if (warp == 0 && lane <= kNF) {
    reduced = shared.split_partials[0][lane] + shared.split_partials[1][lane] +
              shared.split_partials[2][lane] + shared.split_partials[3][lane];
  }

  float value = 0.0f;
  if (warp == 0) {
    constexpr unsigned kMask = 0xffffffffu;
    const float sqsum = __shfl_sync(kMask, reduced, kNF);
    const float inv_rms = rsqrtf(sqsum * (1.0f / static_cast<float>(kK)) + kEps);
    const float logit = lane < kNF ? reduced * inv_rms : 0.0f;
    if (lane < 4) {
      shared.pre[lane] = sigmoidf_fast(fmaf(logit, hc_scale[0], hc_base[lane])) + kEps;
    } else if (lane < 8) {
      post_mix[token * 4 + lane - 4] = sigmoidf_fast(fmaf(logit, hc_scale[1], hc_base[lane]));
    }
    const int source_lane = lane < 16 ? lane + 8 : 0;
    const float comb_logit = __shfl_sync(kMask, logit, source_lane);
    value = lane < 16 ? fmaf(comb_logit, hc_scale[2], hc_base[8 + lane]) : -1.0e30f;
  }
  __syncthreads();

  if (warp == 0) {
    constexpr unsigned kMask = 0xffffffffu;
    float row_max = fmaxf(value, __shfl_xor_sync(kMask, value, 1));
    row_max = fmaxf(row_max, __shfl_xor_sync(kMask, row_max, 2));
    value = lane < 16 ? __expf(value - row_max) : 0.0f;
    float denom = value + __shfl_xor_sync(kMask, value, 1);
    denom += __shfl_xor_sync(kMask, denom, 2);
    value = lane < 16 ? value / denom + kEps : 0.0f;
    denom = value + __shfl_xor_sync(kMask, value, 4);
    denom += __shfl_xor_sync(kMask, denom, 8);
    value = lane < 16 ? value / (denom + kEps) : 0.0f;
#pragma unroll
    for (int iter = 0; iter < 19; ++iter) {
      denom = value + __shfl_xor_sync(kMask, value, 1);
      denom += __shfl_xor_sync(kMask, denom, 2);
      value = lane < 16 ? value / (denom + kEps) : 0.0f;
      denom = value + __shfl_xor_sync(kMask, value, 4);
      denom += __shfl_xor_sync(kMask, denom, 8);
      value = lane < 16 ? value / (denom + kEps) : 0.0f;
    }
    if (lane < 16) comb[token * 16 + lane] = value;
  } else {
    auto* y2 = reinterpret_cast<__nv_bfloat162*>(layer_input + static_cast<long long>(token) * kH);
    const int worker = tid - 32;
    union Bf16x4 {
      uint2 packed;
      __nv_bfloat162 pairs[2];
    };
    int phase[2] = {0, 0};
#pragma unroll 1
    for (int chunk = 0; chunk < kBulkChunks; ++chunk) {
      const int stage = chunk & 1;
      if (lane == 0) {
        tc_wait_mbar(tc_smem_addr(&shared.full[stage]), phase[stage]);
      }
      __syncwarp();
      const __nv_bfloat16* sb = shared.xbuf[stage];
#pragma unroll 2
      for (int h = 4 * worker; h < kBulkChunk; h += 4 * 96) {
        Bf16x4 x0, x1, x2v, x3;
        x0.packed = *reinterpret_cast<const uint2*>(sb + h);
        x1.packed = *reinterpret_cast<const uint2*>(sb + kBulkChunk + h);
        x2v.packed = *reinterpret_cast<const uint2*>(sb + 2 * kBulkChunk + h);
        x3.packed = *reinterpret_cast<const uint2*>(sb + 3 * kBulkChunk + h);
        Bf16x4 out;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float2 a = __bfloat1622float2(x0.pairs[half]);
          const float2 b = __bfloat1622float2(x1.pairs[half]);
          const float2 c = __bfloat1622float2(x2v.pairs[half]);
          const float2 d = __bfloat1622float2(x3.pairs[half]);
          float2 result;
          result.x = fmaf(shared.pre[3], d.x,
                          fmaf(shared.pre[2], c.x, fmaf(shared.pre[1], b.x, shared.pre[0] * a.x)));
          result.y = fmaf(shared.pre[3], d.y,
                          fmaf(shared.pre[2], c.y, fmaf(shared.pre[1], b.y, shared.pre[0] * a.y)));
          out.pairs[half] = __float22bfloat162_rn(result);
        }
        const int pair = (chunk * kBulkChunk + h) / 2;
        *reinterpret_cast<uint2*>(y2 + pair) = out.packed;
      }
      asm volatile("barrier.cta.sync 1, 96;" ::: "memory");
      if (tid == 32 && chunk + 2 < kBulkChunks) {
        issue_bulk_residual_chunk(shared.xbuf[stage], x + (chunk + 2) * kBulkChunk,
                                  &shared.full[stage]);
      }
      phase[stage] ^= 1;
    }
  }
}

constexpr int kTokenTile = 8;

struct Cluster8x8Shared {
  __align__(16) __nv_bfloat16 x[kTokenTile * kHC * kHSlice];
  float partial[kTokenTile][kNF + 1];
  float pre_local[kHC];
};

__global__ __cluster_dims__(kCluster, 1, 1)
    __launch_bounds__(256, 1) void mhc_pre_cluster8x8_kernel(
        const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
        const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
        float* __restrict__ post_mix, float* __restrict__ comb,
        __nv_bfloat16* __restrict__ layer_input, int tokens) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id = static_cast<int>(blockIdx.x) / kCluster;
  const int token0 = cluster_id * kTokenTile;
  int valid = tokens - token0;
  if (valid > kTokenTile) valid = kTokenTile;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int h0 = rank * kHSlice;
  __shared__ Cluster8x8Shared shared;
  __nv_bfloat162* sx2 = reinterpret_cast<__nv_bfloat162*>(shared.x);

#pragma unroll
  for (int m = 0; m < kTokenTile; ++m) {
    if (m < valid) {
      const __nv_bfloat16* xg = residual + static_cast<long long>(token0 + m) * kK;
#pragma unroll
      for (int hc = 0; hc < kHC; ++hc) {
        const __nv_bfloat162* gx2 = reinterpret_cast<const __nv_bfloat162*>(xg + hc * kH + h0);
        sx2[(m * kHC + hc) * (kHSlice / 2) + tid] = gx2[tid];
      }
    }
  }
  __syncthreads();

  const int q0 = warp * 3;
  float acc0[kTokenTile];
  float acc1[kTokenTile];
  float acc2[kTokenTile];
  float sq[kTokenTile];
#pragma unroll
  for (int m = 0; m < kTokenTile; ++m) {
    acc0[m] = 0.0f;
    acc1[m] = 0.0f;
    acc2[m] = 0.0f;
    sq[m] = 0.0f;
  }

#pragma unroll
  for (int hc = 0; hc < kHC; ++hc) {
    const float2* w0 = reinterpret_cast<const float2*>(fn + q0 * kK + hc * kH + h0);
    const float2* w1 = reinterpret_cast<const float2*>(fn + (q0 + 1) * kK + hc * kH + h0);
    const float2* w2 = reinterpret_cast<const float2*>(fn + (q0 + 2) * kK + hc * kH + h0);
#pragma unroll 1
    for (int pair = lane; pair < kHSlice / 2; pair += 32) {
      const float2 a = w0[pair];
      const float2 b = w1[pair];
      const float2 c = w2[pair];
#pragma unroll
      for (int m = 0; m < kTokenTile; ++m) {
        if (m < valid) {
          const __nv_bfloat162* xs = sx2 + (m * kHC + hc) * (kHSlice / 2);
          const float2 xv = __bfloat1622float2(xs[pair]);
          acc0[m] = fmaf(xv.x, a.x, acc0[m]);
          acc0[m] = fmaf(xv.y, a.y, acc0[m]);
          acc1[m] = fmaf(xv.x, b.x, acc1[m]);
          acc1[m] = fmaf(xv.y, b.y, acc1[m]);
          acc2[m] = fmaf(xv.x, c.x, acc2[m]);
          acc2[m] = fmaf(xv.y, c.y, acc2[m]);
          if (warp == 0) {
            sq[m] = fmaf(xv.x, xv.x, sq[m]);
            sq[m] = fmaf(xv.y, xv.y, sq[m]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < kTokenTile; ++m) {
    acc0[m] = warp_sum(acc0[m]);
    acc1[m] = warp_sum(acc1[m]);
    acc2[m] = warp_sum(acc2[m]);
    if (warp == 0) sq[m] = warp_sum(sq[m]);
    if (lane == 0 && m < valid) {
      shared.partial[m][q0] = acc0[m];
      shared.partial[m][q0 + 1] = acc1[m];
      shared.partial[m][q0 + 2] = acc2[m];
      if (warp == 0) shared.partial[m][kNF] = sq[m];
    }
  }
  cluster.sync();

  // Each rank owns one token's nonlinear epilogue, keeping all eight
  // Sinkhorn computations concurrent instead of serializing them on rank 0.
  if (rank < valid && tid == 0) {
    float dot[kNF];
#pragma unroll
    for (int j = 0; j < kNF; ++j) dot[j] = 0.0f;
    float sqsum = 0.0f;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) {
      float* p = cluster.map_shared_rank(&shared.partial[rank][0], r);
#pragma unroll
      for (int j = 0; j < kNF; ++j) dot[j] += p[j];
      sqsum += p[kNF];
    }
    const int token = token0 + rank;
    run_epilogue(dot, sqsum, hc_scale, hc_base, post_mix + token * 4, comb + token * 16,
                 shared.pre_local);
  }
  cluster.sync();

#pragma unroll
  for (int m = 0; m < kTokenTile; ++m) {
    if (m < valid) {
      const float* pre = cluster.map_shared_rank(shared.pre_local, m);
      const __nv_bfloat162* xs = sx2 + m * kHC * (kHSlice / 2);
      const float2 x0 = __bfloat1622float2(xs[tid]);
      const float2 x1 = __bfloat1622float2(xs[kHSlice / 2 + tid]);
      const float2 x2v = __bfloat1622float2(xs[kHSlice + tid]);
      const float2 x3 = __bfloat1622float2(xs[3 * kHSlice / 2 + tid]);
      float2 out;
      out.x = fmaf(pre[3], x3.x, fmaf(pre[2], x2v.x, fmaf(pre[1], x1.x, pre[0] * x0.x)));
      out.y = fmaf(pre[3], x3.y, fmaf(pre[2], x2v.y, fmaf(pre[1], x1.y, pre[0] * x0.y)));
      __nv_bfloat162* yg2 = reinterpret_cast<__nv_bfloat162*>(
          layer_input + static_cast<long long>(token0 + m) * kH + h0);
      yg2[tid] = __float22bfloat162_rn(out);
    }
  }
  cluster.sync();
}

constexpr int kWmmaTokenTile = 16;
constexpr int kWmmaLd = 136;

struct ClusterWmma16Shared {
  __nv_bfloat16 a[kWmmaTokenTile * kWmmaLd];
  union {
    __nv_bfloat16 b[32 * kWmmaLd];
    float out[kWmmaTokenTile * 32];
  } workspace;
  float partial[kWmmaTokenTile][kNF + 1];
  float pre_local[2][kHC];
};

// One cluster projects sixteen tokens in a single launch.  The eight CTAs
// partition K, use BF16 tensor cores for their partial logits, reduce through
// DSMEM, and then finish gates, Sinkhorn, and layer_input before returning.
__global__ __cluster_dims__(kCluster, 1, 1)
    __launch_bounds__(64, 1) void mhc_pre_cluster_wmma16_kernel(
        const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
        const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
        float* __restrict__ post_mix, float* __restrict__ comb,
        __nv_bfloat16* __restrict__ layer_input, int tokens) {
  namespace cg = cooperative_groups;
  namespace wmma = nvcuda::wmma;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id = static_cast<int>(blockIdx.x) / kCluster;
  const int token0 = cluster_id * kWmmaTokenTile;
  int valid = tokens - token0;
  if (valid > kWmmaTokenTile) valid = kWmmaTokenTile;

  constexpr int kSplitSize = kK / kCluster;
  constexpr int kKTile = 128;
  const int k_begin = rank * kSplitSize;
  const int k_end = k_begin + kSplitSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  __shared__ ClusterWmma16Shared shared;

  wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 32, 16, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);
  float sq[8] = {0.0f};

#pragma unroll 1
  for (int k0 = k_begin; k0 < k_end; k0 += kKTile) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = warp * 8 + r;
      const int token = token0 + row;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int pair = lane + half * 32;
        __nv_bfloat162 value = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
        if (row < valid) {
          const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(
              residual + static_cast<long long>(token) * kK + k0);
          value = x2[pair];
          const float2 x = __bfloat1622float2(value);
          sq[r] = fmaf(x.x, x.x, sq[r]);
          sq[r] = fmaf(x.y, x.y, sq[r]);
        }
        reinterpret_cast<__nv_bfloat162*>(shared.a)[row * (kWmmaLd / 2) + pair] = value;
      }
    }

    auto* sb2 = reinterpret_cast<__nv_bfloat162*>(shared.workspace.b);
    for (int i = tid; i < 32 * (kKTile / 2); i += 64) {
      const int n = i / (kKTile / 2);
      const int pair = i - n * (kKTile / 2);
      __nv_bfloat162 value = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
      if (n < kNF) {
        const float2 w =
            reinterpret_cast<const float2*>(fn + static_cast<long long>(n) * kK + k0)[pair];
        value = __float22bfloat162_rn(w);
      }
      sb2[n * (kWmmaLd / 2) + pair] = value;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kKTile; kk += 16) {
      wmma::load_matrix_sync(a_frag, shared.a + warp * 8 * kWmmaLd + kk, kWmmaLd);
      wmma::load_matrix_sync(b_frag, shared.workspace.b + kk, kWmmaLd);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(shared.workspace.out + warp * 8 * 32, acc_frag, 32, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < kWmmaTokenTile * kNF; i += 64) {
    const int row = i / kNF;
    const int n = i - row * kNF;
    if (row < valid) {
      shared.partial[row][n] = shared.workspace.out[row * 32 + n];
    }
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const float total = warp_sum(sq[r]);
    const int row = warp * 8 + r;
    if (lane == 0 && row < valid) shared.partial[row][kNF] = total;
  }
  cluster.sync();

#pragma unroll
  for (int local = 0; local < 2; ++local) {
    const int row = rank * 2 + local;
    if (tid == 0 && row < valid) {
      float dot[kNF];
#pragma unroll
      for (int n = 0; n < kNF; ++n) dot[n] = 0.0f;
      float sqsum = 0.0f;
#pragma unroll
      for (int r = 0; r < kCluster; ++r) {
        const float* partial = cluster.map_shared_rank(&shared.partial[row][0], r);
#pragma unroll
        for (int n = 0; n < kNF; ++n) dot[n] += partial[n];
        sqsum += partial[kNF];
      }
      const int token = token0 + row;
      run_epilogue(dot, sqsum, hc_scale, hc_base, post_mix + token * 4, comb + token * 16,
                   shared.pre_local[local]);
    }
  }
  cluster.sync();

  const int h0 = rank * kHSlice;
#pragma unroll
  for (int row = 0; row < kWmmaTokenTile; ++row) {
    if (row < valid) {
      const int owner = row >> 1;
      const int local = row & 1;
      const float* pre = cluster.map_shared_rank(shared.pre_local[local], owner);
      const __nv_bfloat16* x = residual + static_cast<long long>(token0 + row) * kK + h0;
      __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(
          layer_input + static_cast<long long>(token0 + row) * kH + h0);
#pragma unroll
      for (int pair = tid; pair < kHSlice / 2; pair += 64) {
        const int h2 = pair * 2;
        const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + h2));
        const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + kH + h2));
        const float2 x2v =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + 2 * kH + h2));
        const float2 x3 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + 3 * kH + h2));
        float2 out;
        out.x = fmaf(pre[3], x3.x, fmaf(pre[2], x2v.x, fmaf(pre[1], x1.x, pre[0] * x0.x)));
        out.y = fmaf(pre[3], x3.y, fmaf(pre[2], x2v.y, fmaf(pre[1], x1.y, pre[0] * x0.y)));
        y2[pair] = __float22bfloat162_rn(out);
      }
    }
  }
  cluster.sync();
}

template <int kSplitK, int kM, int kStages, int kThreads, int kScratchPitch,
          bool kFloat4Weights = false, bool kProgrammaticTrigger = false>
static void launch_proj_tma(int tokens, const __nv_bfloat16* res, const float* fnp,
                            __nv_bfloat16* lip, cudaStream_t stream) {
  constexpr int kSmem = kStages * (kM + kNF) * 128;
  alignas(64) CUtensorMap res_map;
  if (encode_residual_map(&res_map, res, tokens, kM)) {
    mhc_project_tma_kernel<kSplitK, kM, kStages, kThreads, kScratchPitch, kFloat4Weights,
                           kProgrammaticTrigger>
        <<<dim3(tokens / kM, kSplitK), kThreads, kSmem, stream>>>(res_map, fnp, lip, 0);
  } else {
    mhc_project_pipe_kernel<kSplitK, kM, 64, kThreads, kStages, kScratchPitch, false, false, false,
                            kProgrammaticTrigger>
        <<<dim3(tokens / kM, kSplitK), kThreads, kStages * kM * 128 + kNF * 128, stream>>>(res, fnp,
                                                                                           lip);
  }
}

template <int kSplitK, int kM, int kStages, int kStageK = 64, int kThreads = 256,
          int kScratchPitch = 32, bool kWeightFirst = false, bool kPackedSq = false,
          bool kResidualEvictLast = false, bool kProgrammaticTrigger = false>
static void launch_proj_pipe(int tokens, const __nv_bfloat16* res, const float* fnp,
                             __nv_bfloat16* lip, cudaStream_t stream) {
  constexpr int kSmem = kStages * (kStageK / 64) * kM * 128 + (kStageK / 64) * kNF * 128;
  mhc_attr_once((const void*)mhc_project_pipe_kernel<kSplitK, kM, kStageK, kThreads, kStages,
                                                     kScratchPitch, kWeightFirst, kPackedSq,
                                                     kResidualEvictLast, kProgrammaticTrigger>,
                kSmem, 1);
  mhc_project_pipe_kernel<kSplitK, kM, kStageK, kThreads, kStages, kScratchPitch, kWeightFirst,
                          kPackedSq, kResidualEvictLast, kProgrammaticTrigger>
      <<<dim3(tokens / kM, kSplitK), kThreads, kSmem, stream>>>(res, fnp, lip);
}

template <int kMaxThreads, int kSplitK, int kScratchPitch = kNF + 1, bool kEvictFirst = false,
          bool kWideEvict = false, bool kScalarSink = false>
static void launch_epilogue_pdl(int tokens, const __nv_bfloat16* residual, const float* scale,
                                const float* base, float* post, float* comb, __nv_bfloat16* layer,
                                cudaStream_t stream) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(tokens, 1, 1);
  config.blockDim = dim3(kMaxThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute[1] = {};
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config,
                     mhc_tf32_epilogue_kernel<kMaxThreads, kSplitK, kScratchPitch, kEvictFirst,
                                              kWideEvict, kScalarSink, true>,
                     residual, scale, base, post, comb, layer, tokens);
}

static void launch_pipe(int tokens, const __nv_bfloat16* residual_p, const float* fn_p,
                        const float* scale_p, const float* base_p, float* post_p, float* comb_p,
                        __nv_bfloat16* layer_p, cudaStream_t stream) {
  if (tokens == 256) {
    constexpr int kTcgenSmemBytes = 4 * (64 + 24) * 128;
    mhc_attr_once((const void*)mhc_project_tcgen_kernel<64, 64, 256, false, true, 256, false, false,
                                                        false, true, true, 25, true>,
                  kTcgenSmemBytes, 2);
    mhc_project_tcgen_kernel<64, 64, 256, false, true, 256, false, false, false, true, true, 25,
                             true><<<dim3(tokens / 64, 64), 256, kTcgenSmemBytes, stream>>>(
        residual_p, fn_p, layer_p, tokens);
    launch_epilogue_pdl<160, 64, 25, false, false, true>(tokens, residual_p, scale_p, base_p,
                                                         post_p, comb_p, layer_p, stream);
  } else if (tokens == 512) {
    constexpr int kTcgenSmemBytes = 4 * (64 + 24) * 128;
    mhc_attr_once((const void*)mhc_project_tcgen_kernel<64, 64, 256, false, true, 256, false, false,
                                                        false, false, true, 24, true>,
                  kTcgenSmemBytes, 3);
    mhc_project_tcgen_kernel<64, 64, 256, false, true, 256, false, false, false, false, true, 24,
                             true><<<dim3(tokens / 64, 64), 256, kTcgenSmemBytes, stream>>>(
        residual_p, fn_p, layer_p, tokens);
    launch_epilogue_pdl<160, 64, 24, false, false, true>(tokens, residual_p, scale_p, base_p,
                                                         post_p, comb_p, layer_p, stream);
  } else if (tokens == 1024) {
    launch_proj_pipe<64, 128, 2, 64, 128, 24, true, true, true, true>(tokens, residual_p, fn_p,
                                                                      layer_p, stream);
    launch_epilogue_pdl<160, 64, 24, false, false, true>(tokens, residual_p, scale_p, base_p,
                                                         post_p, comb_p, layer_p, stream);
  } else if (tokens == 2048) {
    launch_proj_pipe<32, 128, 2, 64, 128, 24, true, true, true, true>(tokens, residual_p, fn_p,
                                                                      layer_p, stream);
    launch_epilogue_pdl<128, 32, 24>(tokens, residual_p, scale_p, base_p, post_p, comb_p, layer_p,
                                     stream);
  } else if (tokens == 4096) {
    launch_proj_tma<16, 128, 2, 128, 24, true, true>(tokens, residual_p, fn_p, layer_p, stream);
    launch_epilogue_pdl<128, 16, 24, true>(tokens, residual_p, scale_p, base_p, post_p, comb_p,
                                           layer_p, stream);
  } else if (tokens == 8192) {
    launch_proj_tma<8, 128, 2, 128, 24, true, true>(tokens, residual_p, fn_p, layer_p, stream);
    launch_epilogue_pdl<128, 8, 24, true, true>(tokens, residual_p, scale_p, base_p, post_p, comb_p,
                                                layer_p, stream);
  } else {
    launch_proj_tma<4, 128, 2, 256, 25, false, true>(tokens, residual_p, fn_p, layer_p, stream);
    launch_epilogue_pdl<128, 4, 25, true, true>(tokens, residual_p, scale_p, base_p, post_p, comb_p,
                                                layer_p, stream);
  }
}

}  // namespace

void Run(ffi::TensorView residual, ffi::TensorView fn, ffi::TensorView hc_scale,
         ffi::TensorView hc_base, ffi::TensorView post_mix, ffi::TensorView comb,
         ffi::TensorView layer_input) {
  const int tokens = static_cast<int>(residual.size(0));
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  auto* residual_p = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  auto* fn_p = static_cast<const float*>(fn.data_ptr());
  auto* scale_p = static_cast<const float*>(hc_scale.data_ptr());
  auto* base_p = static_cast<const float*>(hc_base.data_ptr());
  auto* post_p = static_cast<float*>(post_mix.data_ptr());
  auto* comb_p = static_cast<float*>(comb.data_ptr());
  auto* layer_p = static_cast<__nv_bfloat16*>(layer_input.data_ptr());
  if (tokens == 256 || tokens == 512 || tokens == 1024 || tokens == 2048 || tokens == 4096 ||
      tokens == 8192 || tokens == 16384) {
    launch_pipe(tokens, residual_p, fn_p, scale_p, base_p, post_p, comb_p, layer_p, stream);
  } else {
    if (tokens <= 256) {
      mhc_project_bf16_m64_kernel<64>
          <<<dim3((tokens + 63) / 64, 64), 256, 0, stream>>>(residual_p, fn_p, layer_p, tokens);
      mhc_tf32_epilogue_kernel<128, 64><<<tokens, 128, 0, stream>>>(
          residual_p, scale_p, base_p, post_p, comb_p, layer_p, tokens);
    } else if (tokens <= 512) {
      mhc_project_bf16_m64_kernel<32>
          <<<dim3((tokens + 63) / 64, 32), 256, 0, stream>>>(residual_p, fn_p, layer_p, tokens);
      mhc_tf32_epilogue_kernel<128, 32><<<tokens, 128, 0, stream>>>(
          residual_p, scale_p, base_p, post_p, comb_p, layer_p, tokens);
    } else if (tokens <= 1024) {
      mhc_project_bf16_m64_kernel<16>
          <<<dim3((tokens + 63) / 64, 16), 256, 0, stream>>>(residual_p, fn_p, layer_p, tokens);
      mhc_tf32_epilogue_kernel<128, 16><<<tokens, 128, 0, stream>>>(
          residual_p, scale_p, base_p, post_p, comb_p, layer_p, tokens);
    } else if (tokens <= 2048) {
      mhc_project_bf16_m64_kernel<8>
          <<<dim3((tokens + 63) / 64, 8), 256, 0, stream>>>(residual_p, fn_p, layer_p, tokens);
      mhc_tf32_epilogue_kernel<128, 8><<<tokens, 128, 0, stream>>>(residual_p, scale_p, base_p,
                                                                   post_p, comb_p, layer_p, tokens);
    } else {
      mhc_project_bf16_m64_kernel<4>
          <<<dim3((tokens + 63) / 64, 4), 256, 0, stream>>>(residual_p, fn_p, layer_p, tokens);
      mhc_tf32_epilogue_kernel<128, 4><<<tokens, 128, 0, stream>>>(residual_p, scale_p, base_p,
                                                                   post_p, comb_p, layer_p, tokens);
    }
  }
}
}  // namespace mid_impl
// END_EXACT_BODY_MID 8aca060c6ef731b99e488d5a53c6c6506e26ce0d27ab078b534fb5e8749a156d

namespace ffi = tvm::ffi;

void Run(ffi::TensorView residual, ffi::TensorView fn, ffi::TensorView hc_scale,
         ffi::TensorView hc_base, ffi::TensorView post_mix, ffi::TensorView comb,
         ffi::TensorView layer_input) {
  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  (void)cudaGetLastError();  // clear stale state; launch status checked below
  const int T = static_cast<int>(residual.shape()[0]);
  if (T == 256) {
    mid_impl::Run(residual, fn, hc_scale, hc_base, post_mix, comb, layer_input);
  } else if (T <= 64) {
    persistent_compact::Run(residual, fn, hc_scale, hc_base, post_mix, comb, layer_input);
  } else if (T == 128) {
    persistent_split::Run(residual, fn, hc_scale, hc_base, post_mix, comb, layer_input);
  } else {
    mid_impl::Run(residual, fn, hc_scale, hc_base, post_mix, comb, layer_input);
  }
  const cudaError_t launch_rc = cudaGetLastError();
  if (launch_rc != cudaSuccess) {
    throw std::runtime_error(std::string("mhc_pre_glm launch failed: ") +
                             cudaGetErrorString(launch_rc));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
